"""
10-entity Adaptive Decision Engine demonstration.

THIS IS A SMOKE / WORKFLOW TEST, NOT A PERFORMANCE EVALUATION. Ten entities
cannot establish any overall Macro F0.5 change; that requires the full
S1-grouped validation run.

Every number printed here comes from the real pipeline:
  * candidates and retrieval-channel provenance: ``cache/train_cand_S{2,3}.parquet``
  * pair attributes: the fixed-width column stores under ``cache/store``
  * scores: the validated LightGBM model in ``models/l2_lgbm.pkl``
  * ground truth: ``cache/train_ground_truth.json``

The 10 S1 entities are the first 10 rows of the already-selected controlled
validation case set (``cache/control_cases.json``), which was chosen
deliberately to cover single / multi / variation / no-match / hard-negative
situations. Selection rationale is recorded in the report.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adaptive_decision import AdaptiveConfig, decide_candidates
from src.fixed_store import ColumnBundle
from src.utils import guard

CACHE = Path("D:/amazon_ml/cache")
N_DEMO = 10
BASELINE = 0.7136


def gt_row_index(entity_id: str) -> tuple[int, int] | None:
    want = pa.array([entity_id], type=pa.string())
    for sid in (2, 3):
        base = 0
        pf = pq.ParquetFile(str(CACHE / f"train_source{sid}_norm.parquet"))
        for rb in pf.iter_batches(batch_size=1_000_000, columns=["entity_id"]):
            col = rb.column("entity_id")
            m = pc.is_in(col, value_set=want)
            if pc.any(m).as_py():
                return sid, base + int(np.asarray(m).nonzero()[0][0])
            base += len(rb)
    return None


def main() -> None:
    g = guard()
    cases = json.loads((CACHE / "control_cases.json").read_text(encoding="utf-8"))
    gt = json.loads((CACHE / "train_ground_truth.json").read_text(encoding="utf-8"))
    sel = cases[:N_DEMO]
    s1_ids = pq.read_table(str(CACHE / "train_source1_norm.parquet"),
                           columns=["entity_id"]).column("entity_id").to_pylist()
    row_of = {e: i for i, e in enumerate(s1_ids)}
    targets = {row_of[c["s1"]]: c for c in sel if c["s1"] in row_of}
    print(f"[demo] {len(targets)} S1 entities: "
          f"{[c['kind'] for c in sel]}", flush=True)

    # --- candidates for these rows only (row-group pruning) ---
    per_s1: dict[int, dict[int, list]] = {r: {} for r in targets}
    for sid in (2, 3):
        pf = pq.ParquetFile(str(CACHE / f"train_cand_S{sid}.parquet"))
        m = pf.metadata
        n_rg = m.num_row_groups
        b = np.zeros(n_rg + 1, dtype=np.int64)
        for i in range(n_rg):
            b[i + 1] = b[i] + m.row_group(i).num_rows
        # Row-group statistics give each group's exact s1_row range, so the
        # groups holding the selected rows are found directly from the footer
        # with no scanning and no ratio guess (rows-per-S1 is not constant).
        lo_r, hi_r = int(min(targets)), int(max(targets))
        want_groups = []
        for gi in range(pf.metadata.num_row_groups):
            cs = pf.metadata.row_group(gi).column(0).statistics
            if cs is None or not hasattr(cs, "min"):
                continue
            if cs.min <= hi_r and cs.max >= lo_r:
                want_groups.append(gi)
        print(f"[demo]   S{sid}: {len(want_groups)} row groups overlap the "
              f"selected S1 rows (footer statistics)", flush=True)
        for gidx in want_groups:
            t = pf.read_row_group(gidx, columns=["s1_row", "cand_row",
                                                 "channels"])
            sr = t.column("s1_row").to_numpy()
            if len(sr) == 0 or sr.min() > hi_r or sr.max() < lo_r:
                del t
                continue
            t = pf.read_row_group(gidx, columns=["s1_row", "cand_row",
                                                 "channels"])
            sr = t.column("s1_row").to_numpy()
            sel_m = np.fromiter((int(x) in targets for x in sr.tolist()),
                                dtype=bool, count=len(sr))
            if not sel_m.any():
                del t
                continue
            idx = np.flatnonzero(sel_m)
            rows = sr[idx]
            crows = t.column("cand_row").to_numpy()[idx]
            cc = t.column("channels")
            for r, c, ch in zip(rows.tolist(), crows.tolist(),
                                [cc[int(i)].as_py() for i in idx.tolist()]):
                per_s1[r].setdefault(sid, []).append((int(c), ch))
            del t
        del pf
        g.gc()
    n_c = sum(len(v) for d in per_s1.values() for v in d.values())
    print(f"[demo] {n_c:,} candidate rows for the selected S1 | {g.status()}",
          flush=True)

    # --- attributes, features, scores ---
    store = CACHE / "store"
    s1b = ColumnBundle(CACHE / "train_source1_norm.parquet", store,
                       with_ids=False)
    srcb = {2: ColumnBundle(CACHE / "train_source2_norm.parquet", store,
                            with_ids=True),
            3: ColumnBundle(CACHE / "train_source3_norm.parquet", store,
                            with_ids=True)}
    model = pickle.loads((Path("D:/amazon_ml/models") / "l2_lgbm.pkl")
                         .read_bytes())
    import run_features
    from run_features import compute_batch
    run_features.WORKERS = 4

    nw = 9
    out_rows = []
    for r, case in sorted(targets.items(), key=lambda kv: kv[1]["kind"]):
        rows_s1, rows_c, chans, sids = [], [], [], []
        for sid in (2, 3):
            for c, ch in per_s1[r].get(sid, []):
                rows_s1.append(r)
                rows_c.append(c)
                chans.append(ch)
                sids.append(sid)
        if not rows_c:
            continue
        a_s1 = np.asarray(rows_s1, dtype=np.int64)
        a_c = np.asarray(rows_c, dtype=np.int64)
        sids_arr = np.asarray(sids)
        X = np.zeros((len(a_c), 41), dtype=np.float32)
        for sid in (2, 3):
            msk = sids_arr == sid
            if not msk.any():
                continue
            b1, b2 = s1b.take(a_s1[msk], 0), s1b.take(a_s1[msk], 1)
            b3, b4 = s1b.take(a_s1[msk], 2), s1b.take(a_s1[msk], 3)
            st = srcb[sid]
            X[msk] = compute_batch(
                b1, b2, b3, b4, st.take(a_c[msk], 0), st.take(a_c[msk], 1),
                st.take(a_c[msk], 2), st.take(a_c[msk], 3),
                np.fromiter((x.count("|") + 1 if x else 0 for x in
                             [chans[i] for i in np.flatnonzero(msk)]),
                            dtype=np.float32, count=int(msk.sum())),
                {k: np.fromiter((k in x for x in
                                 [chans[i] for i in np.flatnonzero(msk)]),
                                dtype=np.float32, count=int(msk.sum()))
                 for k in run_features.CHANNEL_KEYS})
        scores = model.predict_proba(X)[:, 1]
        g.gc(2)

        # deterministic override, measured precision 1.000
        ov = np.zeros(len(a_c), dtype=bool)
        nname = (X[:, 0] > 0)                       # name_exact
        naddr = (X[:, 9] > 0)                      # addr_exact
        ndig = (X[:, 18] > 0)                      # digit_exact
        for i in range(len(a_c)):
            same_name = nname[i]
            if not same_name:
                continue
            cnt_name = int((nname & (sids_arr == sids_arr[i])).sum())
            cnt_addr = int((naddr & (sids_arr == sids_arr[i])).sum())
            cnt_dig = int((ndig & (sids_arr == sids_arr[i])).sum())
            uniq_name = cnt_name == 1
            agree = (naddr[i] and cnt_addr == 1) or (ndig[i] and cnt_dig == 1)
            ov[i] = bool(uniq_name and agree)

        cfg = AdaptiveConfig()
        acc_b, ev, dec_b = decide_candidates(
            chans, scores, a_s1, X, run_features.FEATURE_NAMES,
            decision_mode="baseline", config=cfg, deterministic_override=ov)
        acc_a, ev_a, dec_a = decide_candidates(
            chans, scores, a_s1, X, run_features.FEATURE_NAMES,
            decision_mode="adaptive", config=cfg, deterministic_override=ov)

        # truth for this S1 as candidate rows
        truth: set = set()
        for mid in gt.get(s1_ids[r], []):
            res = gt_row_index(mid)
            if res:
                truth.add(res)
        best_i = int(np.argmax(scores))
        eid_best = srcb[int(sids_arr[best_i])].take(
            np.array([a_c[best_i]]), 4)[0]
        n_tp_b = int((acc_b & np.array([(sids_arr[i], a_c[i]) in truth
                                         for i in range(len(a_c))])).sum())
        n_tp_a = int((acc_a & np.array([(sids_arr[i], a_c[i]) in truth
                                         for i in range(len(a_c))])).sum())
        out_rows.append({
            "s1": s1_ids[r], "kind": case["kind"],
            "n_gt": len(gt.get(s1_ids[r], [])),
            "n_cand": len(a_c),
            "top_cand": eid_best,
            "top_score": float(scores[best_i]),
            "top_channels": chans[best_i],
            "margin_best": float(ev.margin[best_i]) if np.isfinite(
                ev.margin[best_i]) else None,
            "n_above_baseline": int(ev.n_above_normal[best_i]),
            "name_exact_top": bool(nname[best_i]),
            "addr_exact_top": bool(naddr[best_i]),
            "digit_exact_top": bool(ndig[best_i]),
            "baseline_accept": int(acc_b.sum()),
            "adaptive_accept": int(acc_a.sum()),
            "baseline_tp": n_tp_b,
            "adaptive_tp": n_tp_a,
            "changed": bool((acc_b != acc_a).any()),
            "reason_top": dec_a.reason[best_i],
        })
        del X, scores
        g.gc(2)

    (Path("D:/amazon_ml/reports") / "adaptive_demo_10.json").write_text(
        json.dumps(out_rows, indent=2), encoding="utf-8")
    print(f"[demo] {len(out_rows)} entities evaluated", flush=True)
    for o in out_rows:
        print(f"  {o['s1']} kind={o['kind']:<10} cand={o['n_cand']:>5} "
              f"top={o['top_score']:.4f} ch={o['top_channels'][:28]:<28} "
              f"base={o['baseline_accept']:>3} adapt={o['adaptive_accept']:>3} "
              f"changed={o['changed']}", flush=True)


if __name__ == "__main__":
    main()
