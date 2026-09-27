"""
Stage 1: candidate generation for one source.

Design (decision D-003, see reports/RETRIEVAL_DECISION_D003.md):
  exact name + exact address + address digits + rare-token postings,
  consuming tokens rarest-first and stopping at a per-S1 budget.

Engineering:
  * one source is indexed at a time and fully released before the next;
  * indexes are cached on disk, so repeated runs (experiments, train, test)
    skip the ~50-70s/field build;
  * S1 is streamed from parquet in row chunks; candidates are written to
    parquet incrementally, so neither side is ever fully resident;
  * S1 chunks are processed in a process pool. The per-row retrieval loop is
    pure Python/C and holds the GIL-free portion of the work in numpy, so
    chunk-level parallelism is the way to use the other cores;
  * total-system RAM is polled by ResourceGuard and the chunk size is reduced
    automatically if pressure rises.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.indexing import PostingIndex
from src.retrieval import (
    CandidateWriter, RareTokenIndex, RetrievalBudget, retrieve_chunk,
)
from src.utils import guard, paths

# Globals populated once per process so the pool does not re-pickle indexes.
_G: dict = {}


def _init_worker(ix_n, ix_a, ix_d, token_idx, budget, part_dir, source_id):
    _G["ix_n"] = ix_n
    _G["ix_a"] = ix_a
    _G["ix_d"] = ix_d
    _G["ti"] = token_idx
    _G["budget"] = budget
    _G["part_dir"] = Path(part_dir)
    _G["source_id"] = source_id


def _work(item):
    """
    Retrieve one chunk and write it to its own parquet part.

    Returning the arrays to the parent costs ~160 MB of pickling per chunk
    (20k S1 x ~1k candidates x 2 int32 columns), i.e. ~17 GB of IPC traffic
    over a full run - which made the 9-worker pool *slower* than a single
    process. Writing the part in the worker and returning only a small summary
    keeps IPC negligible.
    """
    idx, base, block = item
    s1r, cr, bits = retrieve_chunk(block, _G["ix_n"], _G["ix_a"], _G["ix_d"],
                                  _G["ti"], _G["budget"])
    path = _G["part_dir"] / f"part_{idx:05d}.parquet"
    if len(s1r):
        s1r = s1r + np.int32(base)
        with CandidateWriter(path) as w:
            w.write(s1r, cr, bits, _G["source_id"])
    return idx, len(s1r), str(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--source", default="S2", choices=["S2", "S3"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--per-s1", type=int, default=None)
    ap.add_argument("--exact-cap", type=int, default=None)
    ap.add_argument("--digit-cap", type=int, default=None)
    ap.add_argument("--max-df", type=int, default=None)
    ap.add_argument("--bucket-cap", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    g = guard()
    cfgp = paths()
    cfg = __import__("src.utils", fromlist=["get_config"]).get_config()
    rcfg = cfg["retrieval"]
    res = cfg.get("resources", {})

    cache = cfgp["cache_dir"]
    idx_cache = cache / "index"
    idx_cache.mkdir(parents=True, exist_ok=True)
    src_num = args.source[-1]
    src_path = cache / f"{args.split}_source{src_num}_norm.parquet"
    s1_path = cache / f"{args.split}_source1_norm.parquet"
    out = Path(args.out) if args.out else \
        cache / f"{args.split}_cand_{args.source}.parquet"

    if out.exists() and not args.force:
        print(f"[skip] {out.name} exists "
              f"({out.stat().st_size / 1e6:.0f} MB); use --force to rebuild")
        return

    budget = RetrievalBudget(
        per_s1=args.per_s1 if args.per_s1 is not None else rcfg["per_s1"],
        exact_cap=args.exact_cap if args.exact_cap is not None
        else rcfg["exact_cap"],
        digit_cap=args.digit_cap if args.digit_cap is not None
        else rcfg["digit_cap"],
        max_token_df=args.max_df if args.max_df is not None
        else rcfg["max_token_df"],
        token_bucket_cap=args.bucket_cap if args.bucket_cap is not None
        else rcfg.get("token_bucket_cap", 5000),
    )
    workers = args.workers or int(res.get("max_workers", g.max_workers))
    workers = max(1, min(workers, g.max_workers))
    # Each worker holds a copy-on-write view of the indexes, but the parent also
    # keeps them alive, so peak RAM scales with the worker count. Cap workers by
    # measured headroom rather than trusting the configured maximum: 9 workers
    # with the S3 indexes pushed TOTAL system RAM to 82%, above the 80% ceiling.
    headroom = g.headroom_bytes()
    if workers > 1 and headroom > 0:
        # conservative: assume ~350 MB of per-worker overhead
        affordable = int(headroom / (350 * 1024 ** 2))
        if affordable < workers:
            print(f"[run] reducing workers {workers} -> {max(1, affordable)} "
                  f"(headroom {headroom / 1024 ** 3:.1f} GB)")
            workers = max(1, affordable)

    print(f"[source] {args.source} <- {src_path.name}")
    print(f"[budget] {budget.as_dict()}")
    print(f"[run]    workers={workers} chunk={args.chunk} | {g.status()}",
          flush=True)

    # ---- indexes (cached) ----
    t0 = time.perf_counter()
    ix_n = PostingIndex.from_column(src_path, "name_norm")
    ix_a = PostingIndex.from_column(src_path, "address_norm")
    ix_d = PostingIndex.from_column(src_path, "address_dig")
    token_idx = [
        RareTokenIndex.build_or_load(src_path, "name_norm",
                                    budget.max_token_df, idx_cache),
        RareTokenIndex.build_or_load(src_path, "address_norm",
                                    budget.max_token_df, idx_cache),
    ]
    print(f"[indexes] ready in {time.perf_counter() - t0:.0f}s | {g.status()}",
          flush=True)
    if g.system_ram_percent() > 78:
        # Do not start the pool while already at the ceiling.
        print(f"[run] RAM at {g.system_ram_percent():.1f}% - waiting for headroom",
              flush=True)
        for _ in range(60):
            if g.system_ram_percent() <= 75:
                break
            time.sleep(5)
        print(f"[run] {g.status()}", flush=True)
        workers = 1

    n_s1 = pq.ParquetFile(str(s1_path)).metadata.num_rows
    if args.limit:
        n_s1 = min(n_s1, args.limit)
    cols = ["name_norm", "address_norm", "address_dig"]

    total = 0
    t0 = time.perf_counter()
    parts_dir = out.parent / (out.stem + "_parts")
    parts_dir.mkdir(parents=True, exist_ok=True)

    it = _chunks(s1_path, cols, args.chunk, n_s1)
    if workers == 1:
        _init_worker(ix_n, ix_a, ix_d, token_idx, budget, parts_dir,
                     int(src_num))
        pool = None
        stream = (_work((i, b, d)) for i, (b, d) in enumerate(it))
    else:
        ctx = __import__("multiprocessing").get_context("spawn")
        pool = ProcessPoolExecutor(
            max_workers=workers, mp_context=ctx,
            initializer=_init_worker,
            initargs=(ix_n, ix_a, ix_d, token_idx, budget, str(parts_dir),
                      int(src_num)))
        stream = pool.map(_work, ((i, b, d) for i, (b, d) in enumerate(it)),
                          chunksize=1)

    done = 0
    part_paths: list[Path] = []
    for idx, n_pairs, path in stream:
        total += n_pairs
        done += 1
        if n_pairs:
            part_paths.append(Path(path))
        if done % 20 == 0 or total == 0:
            el = time.perf_counter() - t0
            rate = done * args.chunk / max(el, 1e-6)
            print(f"  {done * args.chunk:,}/{n_s1:,}  pairs={total:,}  "
                  f"{rate:,.0f} S1/s  elapsed {el / 60:.1f}m | {g.status()}",
                  flush=True)
        if g.pressure() > 0.95:
            g.pause_for_memory()
    if pool is not None:
        pool.shutdown()
        del pool

    # ---- concatenate the parts into the final file ----
    from src.retrieval import CAND_SCHEMA
    if part_paths:
        w = pq.ParquetWriter(str(out), schema=CAND_SCHEMA, compression="snappy")
        for p in part_paths:
            pf = pq.ParquetFile(str(p))
            for rb in pf.iter_batches(batch_size=500_000):
                w.write_table(pa.Table.from_batches([rb]))
            del pf
        w.close()
        for p in part_paths:
            p.unlink(missing_ok=True)
        try:
            parts_dir.rmdir()
        except OSError:
            pass
    elif not out.exists():
        pq.ParquetWriter(str(out), schema=CAND_SCHEMA,
                         compression="snappy").close()

    el = time.perf_counter() - t0
    summary = {
        "source": args.source,
        "split": args.split,
        "n_s1_processed": n_s1,
        "candidate_pairs": total,
        "pairs_per_s1": round(total / max(n_s1, 1), 2),
        "elapsed_sec": round(el, 1),
        "out_file_mb": round(out.stat().st_size / 1e6, 1),
        "workers": workers,
        "budget": budget.as_dict(),
    }
    print("\n[summary] " + json.dumps(summary, indent=2), flush=True)
    rep = cfgp["reports_dir"] / f"cand_{args.split}_{args.source}.json"
    rep.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[saved] {rep}")


def _chunks(path: Path, cols: list[str], size: int, limit: int):
    """Yield (base_row, block_dataframe) from a parquet file."""
    base = 0
    for batch in pq.ParquetFile(str(path)).iter_batches(
            batch_size=size, columns=cols):
        if base >= limit:
            return
        m = len(batch)
        if base + m > limit:
            batch = batch.slice(0, limit - base)
            m = limit - base
        yield base, batch.to_pandas()
        base += m


if __name__ == "__main__":
    main()
