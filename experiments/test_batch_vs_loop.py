"""
Validation of the vectorised retrieval path on REAL ground truth.

Exact bit-for-bit parity with the per-row reference is not required: both
consume token buckets in ascending posting-length order and stop at the same
per-S1 budget, so ties inside one bucket may be broken differently. What must
hold is that the vectorised path does not lose recall.

This measures union recall and volume for both implementations on a real
sample and asserts the vectorised recall is not worse.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.indexing import PostingIndex
from src.retrieval import (
    RareTokenIndex, RetrievalBudget, retrieve_chunk, retrieve_chunk_batch,
)
from src.utils import guard

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
    return found


def main() -> None:
    g = guard()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    with open(CACHE / "_diag_sample.pkl", "rb") as f:
        d = pickle.load(f)
    s1_map, gt = d["s1"], d["gt"]
    sample = d["sample"][:n]
    total = sum(len(gt[s]) for s in sample)
    s1_df = pd.DataFrame([
        {"name_norm": s1_map[s][0] or "", "address_norm": s1_map[s][1] or "",
         "address_dig": s1_map[s][2] or ""} for s in sample])
    print(f"[sample] {len(sample):,} S1 / {total:,} GT pairs | {g.status()}",
          flush=True)

    ref_hits, bat_hits = set(), set()
    ref_n = bat_n = 0
    ref_t = bat_t = 0.0

    for label, path in [("S2", CACHE / "train_source2_norm.parquet"),
                        ("S3", CACHE / "train_source3_norm.parquet")]:
        ix_n = PostingIndex.from_column(path, "name_norm")
        ix_a = PostingIndex.from_column(path, "address_norm")
        ix_d = PostingIndex.from_column(path, "address_dig")
        ti = [RareTokenIndex.build(path, "name_norm", 1000),
              RareTokenIndex.build(path, "address_norm", 1000)]
        want = set()
        for s in sample:
            for mid in gt[s]:
                if mid.startswith(label + "-"):
                    want.add(mid)
        row_of = locate_rows(path, want)
        tgt = [{row_of[mid] for mid in gt[s] if mid in row_of} for s in sample]
        print(f"  {label}: located {len(row_of):,} GT rows | {g.status()}",
              flush=True)

        for name, fn, hits, acc_n, acc_t in [
            ("reference", retrieve_chunk, ref_hits, "n", "t"),
            ("vectorised", retrieve_chunk_batch, bat_hits, "n", "t"),
        ]:
            bud = RetrievalBudget(per_s1=1000, exact_cap=200, digit_cap=200,
                                  max_token_df=1000)
            t0 = time.perf_counter()
            o, c, _b = fn(s1_df, ix_n, ix_a, ix_d, ti, bud)
            dt = time.perf_counter() - t0
            if acc_n == "n":
                ref_n += len(o) if name == "reference" else 0
                bat_n += len(o) if name == "vectorised" else 0
                ref_t += dt if name == "reference" else 0.0
                bat_t += dt if name == "vectorised" else 0.0
            for i, r in zip(o.tolist(), c.tolist()):
                if r in tgt[i]:
                    hits.add((label, sample[i], r))
            print(f"    {name:>10}: {len(o):>10,} pairs in {dt:6.1f}s "
                  f"({len(o)/max(dt,1e-9):,.0f} pairs/s) | {g.status()}",
                  flush=True)
            del o, c, _b
            g.gc(1)

        del ix_n, ix_a, ix_d, ti, row_of
        g.gc()

    r_ref = len(ref_hits) / total
    r_bat = len(bat_hits) / total
    print("\n===== EQUIVALENCE ON REAL DATA =====")
    print(f"  reference recall : {r_ref:.4f}   pairs {ref_n:,}  time {ref_t:.1f}s")
    print(f"  vectorised recall: {r_bat:.4f}   pairs {bat_n:,}  time {bat_t:.1f}s")
    print(f"  speedup          : {ref_t/max(bat_t,1e-9):.1f}x")
    print(f"  volume ratio     : {bat_n/max(ref_n,1):.4f}")
    ok = r_bat >= r_ref - 0.002
    print(f"  ASSERT vectorised recall not worse: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
