"""
Controlled end-to-end validation on the selected training S1 cases.

Runs the REAL pipeline on ~23 deliberately chosen S1 entities and compares the
prediction against the known training ground truth:

    retrieve -> features -> score -> threshold -> compare -> metric

Only the selected S1 rows are touched. Candidate rows for those S1 are read
from the already-built full candidate files by streaming with a row filter, so
the multi-GB files are scanned sequentially but nothing large is held in RAM.

Outputs
-------
* reports/control_results.json : per-case and aggregate metrics
* reports/control_failures.txt : human-readable failure inspection
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

import run_features
from run_features import RowReader, compute_batch
from src.features import N_FEATURES
from src.utils import get_config, guard, paths

CACHE = Path("D:/amazon_ml/cache")


def _f05(p: float, r: float) -> float:
    d = 0.25 * p + r
    return 0.0 if d == 0 else 1.25 * p * r / d


# entity_id -> row, resolved once per source by a single streaming pass that
# records ONLY the ids we need. entity_id values are unique and the file is in
# row order, so one pass with a membership test is the cheapest correct option
# (no index exists on disk and building one would cost more than it saves for
# ~100 lookups).
_NEEDED: dict[int, set] = {2: set(), 3: set()}
_ROW_OF: dict[int, dict[str, int]] = {2: {}, 3: {}}


def resolve_needed_ids(sources=(2, 3)) -> None:
    import pyarrow.compute as pc
    for sid in sources:
        if not _NEEDED[sid] or _ROW_OF[sid]:
            continue
        want = pa.array(sorted(_NEEDED[sid]), type=pa.string())
        base = 0
        pf = pq.ParquetFile(str(CACHE / f"train_source{sid}_norm.parquet"))
        for rb in pf.iter_batches(batch_size=1_000_000, columns=["entity_id"]):
            col = rb.column("entity_id")
            m = pc.is_in(col, value_set=want)
            if pc.any(m).as_py():
                sel = col.filter(m).to_pylist()
                idx = np.asarray(m).nonzero()[0]
                for e, j in zip(sel, idx.tolist()):
                    _ROW_OF[sid][e] = base + j
            base += len(rb)
            del rb
        del pf
        g_gc()


def g_gc() -> None:
    import gc
    gc.collect()


def _find_entity_row(path: Path, entity_id: str, sid: int) -> int:
    return _ROW_OF[sid].get(entity_id, -1)


def main() -> None:
    g = guard()
    p = paths()
    cfg = get_config()
    run_features.WORKERS = max(1, min(int(cfg["resources"]["max_workers"]),
                                       g.max_workers))

    cases = json.loads((CACHE / "control_cases.json").read_text(encoding="utf-8"))
    gt = json.loads((CACHE / "train_ground_truth.json").read_text(encoding="utf-8"))
    meta = json.loads((p["models_dir"] / "best_model.json")
                      .read_text(encoding="utf-8"))
    threshold = float(meta["threshold"])
    name = meta["name"]
    import lightgbm as lgb
    model = pickle.loads((p["models_dir"] / "l2_lgbm.pkl").read_bytes())
    print(f"[model] {name} threshold={threshold:.4f} | {g.status()}", flush=True)

    # ---- row indices of the selected S1 ----
    s1_ids = pq.read_table(str(CACHE / "train_source1_norm.parquet"),
                           columns=["entity_id"]).column("entity_id").to_pylist()
    row_of_s1 = {e: i for i, e in enumerate(s1_ids)}
    want_rows = {row_of_s1[c["s1"]]: c for c in cases if c["s1"] in row_of_s1}
    target = set(want_rows.keys())
    print(f"[cases] {len(target)} S1 rows | {g.status()}", flush=True)

    # ---- candidates for the selected S1, via row-group pruning ----
    #
    # The candidate files are written in ascending s1_row order, and every S1
    # contributes a near-constant number of rows, so one row group spans a
    # contiguous, bounded range of s1_row values. Reading the parquet footer
    # gives each group's row count, hence the S1 rows it can contain; only the
    # groups that actually overlap the selected S1 are read. This replaces a
    # 10 GB sequential scan with a few MB of targeted I/O.
    #
    # The mapping is derived, not assumed: each retrieval chunk wrote
    # ``retrieval_chunk_size`` S1 rows contiguously, so
    #     s1_rows_in_group ~ [group_start_row, group_end_row]
    # is recovered from the observed rows-per-S1 ratio. It is verified by
    # reading the group and checking the actual min/max s1_row.
    t0 = time.perf_counter()
    per_s1: dict[int, dict[int, list]] = {r: {} for r in target}
    for sid in (2, 3):
        path = CACHE / f"train_cand_S{sid}.parquet"
        if not path.exists():
            continue
        pf = pq.ParquetFile(str(path))
        meta_rg = pf.metadata
        n_rg = meta_rg.num_row_groups
        bounds = np.zeros(n_rg + 1, dtype=np.int64)
        for i in range(n_rg):
            bounds[i + 1] = bounds[i] + meta_rg.row_group(i).num_rows
        total_rows = int(bounds[-1])
        n_s1_rows = len(s1_ids)
        rows_per_s1 = total_rows / max(n_s1_rows, 1)

        lo = np.array(sorted(target), dtype=np.int64)
        # group index for each target, from the cumulative s1_row range
        approx = (lo * rows_per_s1).astype(np.int64)
        gi = np.searchsorted(bounds, approx, side="right") - 1
        gi = np.clip(gi, 0, n_rg - 1)
        # widen by +/-1 group because the ratio is an estimate
        cand_groups = set()
        for x in gi.tolist():
            for d in (-1, 0, 1):
                y = x + d
                if 0 <= y < n_rg:
                    cand_groups.add(y)
        print(f"[cand] {path.name}: {n_rg} groups, ~{rows_per_s1:.0f} cand/S1, "
              f"probing {len(cand_groups)} groups | {g.status()}", flush=True)

        found_groups = set()
        for gidx in sorted(cand_groups):
            tbl = pf.read_row_group(gidx, columns=["s1_row", "cand_row", "channels"])
            s1r_all = tbl.column("s1_row").to_numpy()
            if len(s1r_all) == 0:
                del tbl
                continue
            gmin, gmax = int(s1r_all.min()), int(s1r_all.max())
            hit_mask = (s1r_all >= lo.min()) & (s1r_all <= lo.max())
            sel = np.flatnonzero(hit_mask)
            if len(sel) == 0:
                del tbl
                continue
            # only keep rows whose s1_row is actually a target
            crows = tbl.column("cand_row").to_numpy()[sel]
            chans = tbl.column("channels")
            keep_local = np.fromiter(
                (int(x) in target for x in s1r_all[sel].tolist()),
                dtype=bool, count=len(sel))
            if not keep_local.any():
                del tbl
                continue
            sel2 = sel[keep_local]
            rows_out = s1r_all[sel2]
            crows = tbl.column("cand_row").to_numpy()[sel2]
            for r, c, ch in zip(rows_out.tolist(), crows.tolist(),
                                [chans[int(i)].as_py() for i in sel2.tolist()]):
                per_s1[r].setdefault(sid, []).append((c, ch))
            found_groups.add(gidx)
            del tbl, s1r_all
            g.gc()
        print(f"[cand] {path.name}: {len(found_groups)} groups contained "
              f"target S1 rows", flush=True)
        del pf
        g.gc()
    n_pairs = sum(len(v) for d in per_s1.values() for v in d.values())
    print(f"[cand] {n_pairs:,} candidate pairs for the selected S1 in "
          f"{time.perf_counter() - t0:.1f}s | {g.status()}", flush=True)

    # Targeted row readers: only the row groups holding the selected S1 and
    # their candidate rows are read, so RAM stays flat.
    COLS = ("name_norm", "address_norm", "address_dig", "country")
    s1_reader = RowReader(CACHE / "train_source1_norm.parquet", COLS)
    readers = {2: RowReader(CACHE / "train_source2_norm.parquet", COLS),
               3: RowReader(CACHE / "train_source3_norm.parquet", COLS)}
    # id lists are needed to translate rows back to entity ids for the report
    id_lists = {sid: None for sid in (2, 3)}
    print(f"[readers] ready | {g.status()}", flush=True)

    # ---- score ----
    n_workers = int(cfg["resources"]["max_workers"])
    per_s1_scores: dict[int, dict[int, list]] = {}
    block = 100_000
    for sid in (2, 3):
        rd = readers[sid]
        for r, d in per_s1.items():
            items = d.get(sid)
            if not items:
                continue
            out = []
            for s0 in range(0, len(items), block):
                part = items[s0:s0 + block]
                a_s1 = np.full(len(part), r, dtype=np.int64)
                a_c = np.asarray([c for c, _ in part], dtype=np.int64)
                ch = [x for _, x in part]
                nb = len(part)
                X = compute_batch(
                    s1_reader.take(a_s1, 0), s1_reader.take(a_s1, 1),
                    s1_reader.take(a_s1, 2), s1_reader.take(a_s1, 3),
                    rd.take(a_c, 0), rd.take(a_c, 1),
                    rd.take(a_c, 2), rd.take(a_c, 3),
                    np.fromiter((x.count("|") + 1 if x else 0 for x in ch),
                                dtype=np.float32, count=nb),
                    {k: np.fromiter((k in x for x in ch), dtype=np.float32,
                                    count=nb)
                     for k in run_features.CHANNEL_KEYS},
                )
                s = model.predict(X, num_threads=n_workers)
                out.extend(zip(a_c.tolist(), s.tolist(), ch))
                del X, s
                g.gc(1)
            per_s1_scores.setdefault(r, {})[sid] = out
    print(f"[score] done | {g.status()}", flush=True)

    # ---- compare with ground truth ----
    # Resolve the ground-truth entity ids to candidate rows in ONE streaming
    # pass per source, recording only the ~50 ids actually needed.
    for r in want_rows:
        for m in gt.get(s1_ids[r], []):
            _NEEDED[2 if m.startswith("S2-") else 3].add(m)
    resolve_needed_ids()
    print(f"[ids] resolved "
          f"{len(_ROW_OF[2])} S2 + {len(_ROW_OF[3])} S3 GT ids | {g.status()}",
          flush=True)
    id_readers = None
    row_of_id = _ROW_OF

    id_lists = {2: None, 3: None}

    results = []
    for r, case in sorted(want_rows.items(), key=lambda kv: kv[1]["kind"]):
        s1eid = s1_ids[r]
        truth: dict[int, set] = {2: set(), 3: set()}
        for m in gt.get(s1eid, []):
            sid = 2 if m.startswith("S2-") else 3
            rr = row_of_id[sid].get(m)
            if rr is not None:
                truth[sid].add(rr)
        pred: dict[int, set] = {2: set(), 3: set()}
        for sid, items in per_s1_scores.get(r, {}).items():
            for c, s, _ch in items:
                if s >= threshold:
                    pred[sid].add(c)
        n_truth = sum(len(v) for v in truth.values())
        n_pred = sum(len(v) for v in pred.values())
        tp = sum(len(pred[s] & truth[s]) for s in (2, 3))
        prec = tp / n_pred if n_pred else (1.0 if n_truth == 0 else 0.0)
        rec = tp / n_truth if n_truth else (1.0 if n_pred == 0 else 0.0)
        sc = _f05(prec, rec) if n_truth else (1.0 if n_pred == 0 else 0.0)
        results.append({
            "s1": s1eid, "kind": case["kind"],
            "n_truth": n_truth, "n_pred": n_pred, "tp": tp,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f05": round(sc, 4),
            "n_candidates": sum(len(v) for v in per_s1.get(r, {}).values()),
            "missed_rows": sorted(truth[2] - pred[2])[:5]
            + sorted(truth[3] - pred[3])[:5],
            "false_pos_rows": sorted(pred[2] - truth[2])[:5]
            + sorted(pred[3] - truth[3])[:5],
        })

    macro = float(np.mean([x["f05"] for x in results])) if results else 0.0
    ttp = sum(x["tp"] for x in results)
    tpred = sum(x["n_pred"] for x in results)
    tt = sum(x["n_truth"] for x in results)
    summary = {
        "model": name, "threshold": threshold,
        "n_cases": len(results),
        "macro_f05": round(macro, 4),
        "micro_precision": round(ttp / tpred, 4) if tpred else None,
        "micro_recall": round(ttp / tt, 4) if tt else None,
        "total_truth": tt, "total_pred": tpred, "total_tp": ttp,
        "by_kind": {},
    }
    for k in ("single", "multi", "variation", "nomatch", "hardneg"):
        sub = [x for x in results if x["kind"] == k]
        if sub:
            summary["by_kind"][k] = {
                "n": len(sub),
                "macro_f05": round(float(np.mean([x["f05"] for x in sub])), 4),
                "mean_truth": round(float(np.mean([x["n_truth"] for x in sub])), 2),
                "mean_pred": round(float(np.mean([x["n_pred"] for x in sub])), 2),
            }

    (p["reports_dir"] / "control_results.json").write_text(
        json.dumps({"summary": summary, "cases": results}, indent=2),
        encoding="utf-8")

    lines = [f"CONTROLLED VALIDATION  model={name}  threshold={threshold:.4f}",
             f"cases={summary['n_cases']}  macro_f05={summary['macro_f05']}",
             f"micro P={summary['micro_precision']}  R={summary['micro_recall']}",
             "", f"{'kind':<11}{'truth':>6}{'pred':>6}{'tp':>5}{'P':>7}"
             f"{'R':>7}{'F0.5':>7}", "-" * 52]
    for x in sorted(results, key=lambda r: r["kind"]):
        lines.append(f"{x['kind']:<11}{x['n_truth']:>6}{x['n_pred']:>6}"
                     f"{x['tp']:>5}{x['precision']:>7.3f}{x['recall']:>7.3f}"
                     f"{x['f05']:>7.3f}")
    fails = [x for x in results if x["f05"] < 1.0]
    lines += ["", f"FAILURES ({len(fails)}):"]
    for x in fails:
        lines.append(f"  {x['kind']:<10} {x['s1']}  truth={x['n_truth']} "
                     f"pred={x['n_pred']} tp={x['tp']} "
                     f"cands={x['n_candidates']}")
        if x["missed_rows"]:
            lines.append(f"      missed rows (retrieval failures): "
                         f"{x['missed_rows'][:3]}")
        if x["false_pos_rows"]:
            lines.append(f"      false-positive rows: {x['false_pos_rows'][:3]}")
    (p["reports_dir"] / "control_report.txt").write_text(
        "\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
