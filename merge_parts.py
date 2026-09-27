"""
Merge per-chunk candidate parts into the final candidate parquet.

Separated from run_retrieval.py so an interrupted run can be finished without
recomputing retrieval: the parts are the durable checkpoint, and merging is a
pure streaming concatenation of parquet row groups.

Idempotent: if the output already exists it is left alone unless --force.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow as pa
import pyarrow.parquet as pq

from src.retrieval import CAND_SCHEMA
from src.utils import guard, paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--source", default="S3", choices=["S2", "S3"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-parts", action="store_true")
    args = ap.parse_args()

    g = guard()
    cache = paths()["cache_dir"]
    out = cache / f"{args.split}_cand_{args.source}.parquet"
    parts_dir = cache / f"{args.split}_cand_{args.source}_parts"

    if out.exists() and not args.force:
        n = pq.ParquetFile(str(out)).metadata.num_rows
        print(f"[skip] {out.name} exists ({out.stat().st_size / 1e6:.0f} MB, "
              f"{n:,} rows)")
        return

    parts = sorted(parts_dir.glob("part_*.parquet"))
    if not parts:
        raise SystemExit(f"no parts found in {parts_dir}")
    print(f"[merge] {len(parts)} parts from {parts_dir}", flush=True)

    t0 = time.perf_counter()
    total = 0
    w = pq.ParquetWriter(str(out), schema=CAND_SCHEMA, compression="snappy")
    for i, p in enumerate(parts):
        pf = pq.ParquetFile(str(p))
        for rb in pf.iter_batches(batch_size=500_000):
            w.write_table(pa.Table.from_batches([rb]))
            total += rb.num_rows
        del pf
        if i % 20 == 0 or i == len(parts) - 1:
            print(f"  {i + 1}/{len(parts)} parts  rows={total:,}  "
                  f"{time.perf_counter() - t0:.0f}s | {g.status()}", flush=True)
    w.close()
    print(f"[merge] wrote {total:,} rows to {out.name} "
          f"({out.stat().st_size / 1e6:.0f} MB) in "
          f"{(time.perf_counter() - t0) / 60:.1f}m", flush=True)

    if not args.keep_parts:
        freed = 0
        for p in parts:
            freed += p.stat().st_size
            p.unlink(missing_ok=True)
        try:
            parts_dir.rmdir()
        except OSError:
            pass
        print(f"[merge] removed parts, freed {freed / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
