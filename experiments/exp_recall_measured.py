"""
Candidate recall measurement on the REAL full-scale candidate files.

This is the gate before any modelling: candidate recall is the ceiling on
final recall, so it must be measured on the actual artifact that the pipeline
produced, not on a sample estimate.

Reads train_cand_S2.parquet / train_cand_S3.parquet (int32 row ids), maps the
ground-truth match ids to their row index in the corresponding source, and
reports:

    candidate recall  = retrieved GT pairs / total GT pairs
    per-S1 distribution, coverage of S1 with zero matches
    S1->S2 and S1->S3 separately

Memory: the candidate file is streamed in row-group batches; only counters and
the per-S1 hit sets are kept (2.2M small ints).
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

from src.utils import guard, paths

CACHE = Path("D:/amazon_ml/cache")


def locate_rows(path: Path, wanted: set[str]) -> dict[str, int]:
    found: dict[str, int] = {}
    want = pa.array(sorted(wanted), type=pa.string())
    base = 0
    for b in pq.ParquetFile(str(path)).iter_batches(batch_size=500_000,
                                                    columns=["entity_id"]):
        col = b.column("entity_id")
        m = pc.is_in(col, value_set=want)
        if pc.any(m).as_py():
            sel = col.filter(m).to_pylist()
            idxs = np.asarray(m).nonzero()[0]
            for eid, j in zip(sel, idxs):
                found[eid] = base + int(j)
        base += len(b)
        del b, col, m
    return found


def main() -> None:
    g = guard()
    p = paths()
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    gt = json.loads((CACHE / "train_ground_truth.json").read_text(encoding="utf-8"))
    with open(CACHE / "_diag_sample.pkl", "rb") as f:
        sample = pickle.load(f)["sample"]
    if limit:
        sample = sample[:limit]
    sample_set = set(sample)
    print(f"[sample] {len(sample):,} S1 | {g.status()}", flush=True)

    s1_ids = pq.read_table(str(CACHE / "train_source1_norm.parquet"),
                           columns=["entity_id"]).column("entity_id")
    s1_ids = s1_ids.to_numpy(zero_copy_only=False)
    s1_index = {e: i for i, e in enumerate(s1_ids.tolist())}
    del s1_ids

    total_gt = 0
    zero_s1 = 0
    for s in sample:
        m = gt.get(s, [])
        total_gt += len(m)
        if not m:
            zero_s1 += 1

    out = {
        "n_sample_s1": len(sample),
        "total_gt_pairs": total_gt,
        "zero_match_s1": zero_s1,
        "sources": {},
    }

    for label, src_num in (("S2", 2), ("S3", 3)):
        cand_path = CACHE / f"train_cand_{label}.parquet"
        if not cand_path.exists():
            print(f"[skip] {cand_path.name} missing")
            continue
        print(f"\n=== {label} === | {g.status()}", flush=True)
        want = set()
        for s in sample:
            for mid in gt.get(s, []):
                if mid.startswith(label + "-"):
                    want.add(mid)
        t0 = time.perf_counter()
        row_of = locate_rows(CACHE / f"train_source{src_num}_norm.parquet", want)
        print(f"  located {len(row_of):,}/{len(want):,} GT rows "
              f"({time.perf_counter() - t0:.0f}s) | {g.status()}", flush=True)

        # per-S1 counts and hit counts
        counts = np.zeros(len(sample), dtype=np.int64)
        hits = np.zeros(len(sample), dtype=np.int64)
        sample_rows = np.fromiter((s1_index[s] for s in sample), dtype=np.int64,
                                 count=len(sample))
        row_pos = {r: i for i, r in enumerate(sample_rows.tolist())}

        # GT row -> (position, ) for membership tests
        gt_total = 0
        pairs = []
        for i, s in enumerate(sample):
            t = {row_of[m] for m in gt.get(s, []) if m in row_of}
            if t:
                gt_total += len(t)
                for r in t:
                    pairs.append(r)
        gt_arr = np.array(sorted(set(pairs)), dtype=np.int64)
        gt_set = set(gt_arr.tolist())
        print(f"  {gt_total:,} GT pairs resolved for {label}", flush=True)

        pf = pq.ParquetFile(str(cand_path))
        n = 0
        for batch in pf.iter_batches(batch_size=500_000,
                                      columns=["s1_row", "cand_row"]):
            s1r = batch.column("s1_row").to_numpy()
            cr = batch.column("cand_row").to_numpy()
            # count candidates per sampled S1
            lo = np.searchsorted(sample_rows, s1r, side="left")
            valid = (lo < len(sample_rows))
            valid &= sample_rows[np.minimum(lo, len(sample_rows) - 1)] == s1r
            idx = np.flatnonzero(valid)
            if len(idx):
                pos = lo[idx]
                np.add.at(counts, pos, 1)
                # hits: candidate row is a GT row for this S1
                # build a per-S1 GT membership check via sorted pair set
                hitmask = np.fromiter(
                    (int(c) in gt_set for c in cr[idx].tolist()),
                    dtype=bool, count=len(idx))
                if hitmask.any():
                    np.add.at(hits, pos[hitmask], 1)
            n += len(s1r)
            del batch, s1r, cr
            g.gc(1)
        print(f"  scanned {n:,} candidate rows | {g.status()}", flush=True)

        rec = hits.sum() / max(gt_total, 1)
        out["sources"][label] = {
            "candidate_rows": n,
            "gt_pairs": gt_total,
            "recalled": int(hits.sum()),
            "recall": round(rec, 4),
            "s1_with_any_candidate": int((counts > 0).sum()),
            "mean_candidates_per_s1": round(float(counts.mean()), 1),
            "p50": int(np.percentile(counts, 50)),
            "p95": int(np.percentile(counts, 95)),
            "p99": int(np.percentile(counts, 99)),
            "zero_candidate_s1": int((counts == 0).sum()),
        }
        print(json.dumps(out["sources"][label], indent=2), flush=True)

    tot_gt = sum(v["gt_pairs"] for v in out["sources"].values())
    tot_rec = sum(v["recalled"] for v in out["sources"].values())
    if tot_gt:
        out["overall_recall"] = round(tot_rec / tot_gt, 4)
    print("\n===== CANDIDATE RECALL (full-scale artifacts) =====")
    print(json.dumps(out, indent=2))
    (p["reports_dir"] / "candidate_recall_measured.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
