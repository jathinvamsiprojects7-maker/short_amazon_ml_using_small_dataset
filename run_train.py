"""
Stage 4: train the matcher, validate on the S1-grouped split, and optimise
the decision threshold for S1-level Macro F0.5.

Metric
------
Macro-averaged F0.5 per S1 (beta = 0.5, so precision counts twice as much as
recall). Each S1 gets its own F0.5, then the mean is taken. Two consequences
shape the design:

  * a false positive on a ZERO-match S1 scores 0 for that S1, so emitting
    nothing is often better than guessing;
  * an S1 with several true matches only gains from accepting MORE of them,
    since precision also enters the per-S1 F0.5.

The decision is a single global probability threshold (no final top-K),
chosen by direct search on the validation split.

Data
----
Features are computed for the SAMPLED pairs from run_training_pairs.py (all
positives + capped negatives). The full 4.4e9-pair candidate universe is never
featurised.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import FEATURE_NAMES, N_FEATURES
from src.models import DeterministicBaseline, LRModel, LGBMModel
from src.utils import get_config, guard, macro_f05, paths

CACHE = Path("D:/amazon_ml/cache")
SHIFT = np.int64(1) << np.int64(32)


def f05(precision: float, recall: float) -> float:
    denom = 0.25 * precision + recall
    return 0.0 if denom == 0 else 1.25 * precision * recall / denom


def macro_f05_from_arrays(s1_rows: np.ndarray, is_pos: np.ndarray,
                          accept: np.ndarray) -> dict:
    """
    S1-level Macro F0.5 from flat arrays.

    ``is_pos`` and ``accept`` are boolean masks aligned with ``s1_rows``; a pair
    is a true match iff is_pos, and is predicted iff accept.

    Every S1 that appears in ``s1_rows`` is scored, including those with no
    positive pair (they are the zero-match segment, where any prediction is
    punished).
    """
    order = np.argsort(s1_rows, kind="stable")
    sr = s1_rows[order]
    pos = is_pos[order]
    acc = accept[order]
    uniq, start = np.unique(sr, return_index=True)
    ends = np.append(start[1:], len(sr))

    tp = np.add.reduceat((pos & acc).astype(np.int64), start)
    npos = np.add.reduceat(pos.astype(np.int64), start)
    npred = np.add.reduceat(acc.astype(np.int64), start)

    fp = npred - tp
    fn = npos - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float),
                     where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float),
                    where=(tp + fn) > 0)
    denom = 0.25 * prec + rec
    f = np.divide(1.25 * prec * rec, denom, out=np.zeros_like(prec),
                  where=denom > 0)
    # zero-match S1: perfect if nothing predicted, else 0
    zero = npos == 0
    f[zero] = np.where(npred[zero] == 0, 1.0, 0.0)

    tpsum = int(tp.sum())
    return {
        "macro_f05": float(f.mean()),
        "micro_precision": float(tpsum / max(int(npred.sum()), 1)),
        "micro_recall": float(tpsum / max(int(npos.sum()), 1)),
        "n_s1": int(len(uniq)),
        "n_zero_match_s1": int(zero.sum()),
        "zero_match_accuracy": float((npred[zero] == 0).mean()) if zero.any() else None,
        "mean_pred_per_s1": float(npred.mean()),
        "mean_true_per_s1": float(npos.mean()),
    }


def main() -> None:
    g = guard()
    p = paths()
    cfg = get_config()
    mcfg = cfg["model"]
    tcfg = cfg["threshold"]

    d = np.load(CACHE / "train_pairs.npz")
    y, s1_row, split, src_offsets = (d["y"], d["s1_row"], d["split"],
                                     d["src_offsets"])
    del d
    print(f"[pairs] {len(y):,} sampled  pos {int(y.sum()):,}  "
          f"train {int((split == 0).sum()):,}  val {int((split == 1).sum()):,}",
          flush=True)

    meta = json.loads((CACHE / "train_features.meta.json")
                      .read_text(encoding="utf-8"))
    X_all = np.load(CACHE / "train_features.npy", mmap_mode="r")
    if X_all.shape[0] != meta["n_pairs"]:
        raise SystemExit(f"feature cache stale: {X_all.shape} vs {meta}")

    # 94M x 41 float32 is 15.4 GB - larger than the whole machine. Only a
    # subsample is materialised: every positive is kept (5.7M, the signal), and
    # negatives are thinned to a fixed multiple of the positives. The val split
    # keeps ALL of its positives and a capped negative set, so the threshold
    # search still sees the true positive density per S1.
    n_pos_total = int((y == 1).sum())
    neg_target = int(n_pos_total * float(cfg["training"].get(
        "train_neg_per_pos", 2)))
    rng = np.random.default_rng(int(mcfg["random_seed"]))

    def take(mask: np.ndarray, neg_cap: int) -> np.ndarray:
        """All positives of the mask plus at most ``neg_cap`` negatives."""
        rows = np.flatnonzero(mask)
        is_pos = y[rows] == 1
        pos_rows = rows[is_pos]
        neg_rows = rows[~is_pos]
        if len(neg_rows) > neg_cap:
            sel = rng.choice(len(neg_rows), neg_cap, replace=False)
            neg_rows = neg_rows[np.sort(sel)]
        return np.concatenate([pos_rows, neg_rows])

    tr_rows = take(split == 0, neg_target)
    va_rows = take(split == 1, neg_target)
    print(f"[subsample] train rows {len(tr_rows):,} "
          f"(pos {int((y[tr_rows] == 1).sum()):,}) | "
          f"val rows {len(va_rows):,} "
          f"(pos {int((y[va_rows] == 1).sum()):,})", flush=True)

    X_all = np.load(CACHE / "train_features.npy", mmap_mode="r")
    if X_all.shape[0] != meta["n_pairs"]:
        raise SystemExit(f"feature cache stale: {X_all.shape} vs {meta}")

    def gather(rows: np.ndarray) -> np.ndarray:
        out = np.empty((len(rows), N_FEATURES), dtype=np.float32)
        block = 4_000_000
        for s0 in range(0, len(rows), block):
            s1e = min(s0 + block, len(rows))
            out[s0:s1e] = X_all[rows[s0:s1e]]
        return out

    t0 = time.perf_counter()
    Xtr = gather(tr_rows)
    ytr = y[tr_rows].astype(np.int32)
    str_ = s1_row[tr_rows]
    print(f"[feat] train {Xtr.shape} ({Xtr.nbytes / 1e9:.1f} GB) in "
          f"{(time.perf_counter() - t0) / 60:.1f}m | {g.status()}", flush=True)
    g.gc()

    t0 = time.perf_counter()
    Xva = gather(va_rows)
    yva = y[va_rows].astype(np.int32)
    sva = s1_row[va_rows]
    del X_all, y, s1_row, split, tr_rows, va_rows
    g.gc()
    print(f"[feat] val {Xva.shape} ({Xva.nbytes / 1e9:.1f} GB) in "
          f"{(time.perf_counter() - t0) / 60:.1f}m | {g.status()}", flush=True)

    # ---- model ladder ----
    results: dict[str, dict] = {}
    scores_val: dict[str, np.ndarray] = {}

    def evaluate(name: str, s: np.ndarray) -> dict:
        is_pos = yva == 1
        best = (None, -1.0, None)
        for th in np.linspace(float(tcfg["search_min"]),
                              float(tcfg["search_max"]),
                              int(tcfg["search_steps"])):
            m = macro_f05_from_arrays(sva, is_pos, s >= th)
            if m["macro_f05"] > best[1]:
                best = (float(th), m["macro_f05"], m)
        th, f, m = best
        print(f"  {name:<12} threshold={th:.4f}  macroF0.5={f:.6f}  "
              f"P={m['micro_precision']:.4f} R={m['micro_recall']:.4f}  "
              f"zero_acc={m['zero_match_accuracy']}", flush=True)
        r = {"threshold": th, "macro_f05": round(f, 6), "metrics": m}
        results[name] = r
        scores_val[name] = s
        return r

    # L0 deterministic
    print("[L0] deterministic baseline", flush=True)
    l0 = DeterministicBaseline()
    evaluate("L0", l0.predict_proba(Xva)[:, 1])

    # L1 logistic regression
    print("[L1] logistic regression", flush=True)
    l1 = LRModel(C=float(mcfg["lr_C"]), max_iter=int(mcfg["lr_max_iter"]))
    t0 = time.perf_counter()
    l1.fit(Xtr, ytr)
    print(f"  fit {time.perf_counter() - t0:.0f}s | {g.status()}", flush=True)
    s_l1 = l1.predict_proba(Xva)[:, 1]
    evaluate("L1", s_l1)
    g.gc()

    # L2 LightGBM (uses the same Xtr, so it must run before it is freed)
    print("[L2] lightgbm", flush=True)
    import lightgbm as lgb
    n_pos = int(ytr.sum())
    n_neg = int((ytr == 0).sum())
    n_workers = max(1, min(int(cfg["resources"]["max_workers"]), g.max_workers))
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=int(mcfg["lgbm_n_estimators"]),
        learning_rate=float(mcfg["lgbm_learning_rate"]),
        num_leaves=int(mcfg["lgbm_num_leaves"]),
        min_child_samples=int(mcfg["lgbm_min_child_samples"]),
        subsample=float(mcfg["lgbm_subsample"]),
        colsample_bytree=float(mcfg["lgbm_colsample_bytree"]),
        reg_alpha=float(mcfg["lgbm_reg_alpha"]),
        reg_lambda=float(mcfg["lgbm_reg_lambda"]),
        scale_pos_weight=n_neg / max(n_pos, 1),
        random_state=int(mcfg["random_seed"]),
        n_jobs=n_workers,
        verbose=-1,
    )
    t0 = time.perf_counter()
    lgb_clf.fit(Xtr, ytr, feature_name=FEATURE_NAMES)
    print(f"  fit {time.perf_counter() - t0:.0f}s | {g.status()}", flush=True)
    l2 = LGBMModel.__new__(LGBMModel)
    l2._model = lgb_clf
    evaluate("L2", l2.predict_proba(Xva)[:, 1])
    del Xtr
    g.gc()

    imp = sorted(zip(FEATURE_NAMES, lgb_clf.feature_importances_),
                 key=lambda t: -t[1])
    print("  top features: " + ", ".join(f"{k}={v}" for k, v in imp[:10]),
          flush=True)

    # ---- pick the best ----
    best_name = max(results, key=lambda k: results[k]["macro_f05"])
    print(f"[best] {best_name}  F0.5={results[best_name]['macro_f05']:.6f}  "
          f"threshold={results[best_name]['threshold']:.4f}", flush=True)

    models_dir = p["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    l0.save(models_dir / "l0_deterministic.pkl")
    l1.save(models_dir / "l1_logistic.pkl")
    l2.save(models_dir / "l2_lgbm.pkl")
    (models_dir / "l2_lgbm.pkl").write_bytes(pickle.dumps(lgb_clf))
    with open(models_dir / "best_model.json", "w", encoding="utf-8") as f:
        json.dump({
            "name": best_name,
            "threshold": results[best_name]["threshold"],
            "val_macro_f05": results[best_name]["macro_f05"],
            "feature_names": FEATURE_NAMES,
            "n_features": N_FEATURES,
        }, f, indent=2)

    with open(p["reports_dir"] / "model_comparison.json", "w",
              encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # ---- per-segment diagnostics at the best threshold ----
    th = results[best_name]["threshold"]
    s = scores_val[best_name]
    is_pos = yva == 1
    acc = s >= th
    seg = {}
    npos_per_s1 = np.add.reduceat(
        is_pos[np.argsort(sva, kind="stable")].astype(np.int64),
        np.unique(sva, return_index=True)[1])
    m_all = macro_f05_from_arrays(sva, is_pos, acc)
    for label, mask in (("zero_match", npos_per_s1 == 0),
                        ("singleton", npos_per_s1 == 1),
                        ("multi", npos_per_s1 > 1)):
        if not mask.any():
            continue
        keep_s1 = np.unique(sva)[mask]
        sel = np.isin(sva, keep_s1)
        m = macro_f05_from_arrays(sva[sel], is_pos[sel], acc[sel])
        seg[label] = {"n_s1": m["n_s1"], "macro_f05": round(m["macro_f05"], 6),
                      "micro_precision": round(m["micro_precision"], 6),
                      "micro_recall": round(m["micro_recall"], 6)}
    print("[segments] " + json.dumps(seg), flush=True)

    with open(p["reports_dir"] / "val_summary.json", "w", encoding="utf-8") as f:
        json.dump({"best": best_name, "threshold": th,
                   "overall": m_all, "segments": seg,
                   "ladder": results}, f, indent=2, default=float)
    print("[saved] reports/val_summary.json", flush=True)


if __name__ == "__main__":
    main()
