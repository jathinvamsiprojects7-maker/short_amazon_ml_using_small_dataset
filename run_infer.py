"""
Full test inference, parallel and resumable.

Performance
-----------
Profiling a single-process run gave 8,416 pairs/s, of which 53% was spent in
per-row Python lookups. ``src.fixed_store.ColumnBundle`` replaces those with
memory-mapped fixed-width column reads (verified byte-exact against
``RowReader`` on 100k random rows) and removes that cost.

Row groups are independent (each covers a disjoint range of s1_row), so work is
split across processes. Each worker writes its own output shard, so nothing
large crosses a process boundary and the parent's memory stays flat.

Resumability
------------
A shard that already produced its ``.done`` marker is skipped, so an
interrupted run continues from the last completed shard.

Memory
------
Each worker memory-maps the column stores (no resident copy) and holds one row
group of features at a time (~500k x 41 float32 = 82 MB).
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_features
from run_features import compute_batch
from src.fixed_store import ColumnBundle
from src.utils import get_config, guard, paths

CACHE = Path("D:/amazon_ml/cache")
STORE = CACHE / "store"
CHANNEL_KEYS = ("exact_name", "exact_address", "address_digits",
                "rare_token", "ngram")

_G: dict = {}


def _init(threshold, n_workers, store_dir, s1_src, srcs, out_dir, limit_groups):
    _G["threshold"] = threshold
    _G["n_workers"] = n_workers
    _G["s1"] = ColumnBundle(s1_src, store_dir, with_ids=False)
    _G["src"] = {sid: ColumnBundle(p, store_dir, with_ids=True)
                 for sid, p in srcs.items()}
    _G["out"] = out_dir
    _G["limit"] = limit_groups
    _G["model"] = pickle.loads(
        (Path("D:/amazon_ml/models") / "l2_lgbm.pkl").read_bytes())
    run_features.WORKERS = 1     # processes already provide parallelism


def _work(job):
    sid, gi = job
    acc_path = _G["out"] / f"acc_S{sid}.g{gi:05d}.txt"
    cand_path = _G["out"] / f"cand_S{sid}.g{gi:05d}.txt"
    done = _G["out"] / f"done_S{sid}.g{gi:05d}.marker"
    if done.exists():
        return gi, 0, 0, True
    path = CACHE / f"test_cand_S{sid}.parquet"
    pf = pq.ParquetFile(str(path))
    if _G["limit"] and gi >= _G["limit"]:
        return gi, 0, 0, True
    tbl = pf.read_row_group(gi, columns=["s1_row", "cand_row", "channels"])
    s1_all = tbl.column("s1_row").to_numpy()
    if len(s1_all) == 0:
        done.write_text("1", encoding="utf-8")
        return gi, 0, 0, False
    a_s1 = s1_all.astype(np.int64)
    a_c = tbl.column("cand_row").to_numpy().astype(np.int64)
    ch = tbl.column("channels").to_pylist()
    n = len(a_c)
    del tbl

    s1b = _G["s1"]
    rb = _G["src"][sid]
    X = compute_batch(
        s1b.take(a_s1, 0), s1b.take(a_s1, 1), s1b.take(a_s1, 2), s1b.take(a_s1, 3),
        rb.take(a_c, 0), rb.take(a_c, 1), rb.take(a_c, 2), rb.take(a_c, 3),
        np.fromiter((x.count("|") + 1 if x else 0 for x in ch),
                    dtype=np.float32, count=n),
        {k: np.fromiter((k in x for x in ch), dtype=np.float32, count=n)
         for k in CHANNEL_KEYS},
    )
    s = _G["model"].predict_proba(X)[:, 1]
    keep = s >= _G["threshold"]
    del X, s

    order = np.argsort(a_s1, kind="stable")
    a_s1o = a_s1[order]
    c_o = a_c[order]
    k_o = keep[order]
    uniq, start = np.unique(a_s1o, return_index=True)
    ends = np.append(start[1:], len(a_s1o))
    n_acc = 0
    with open(acc_path, "w", encoding="utf-8") as fa, \
            open(cand_path, "w", encoding="utf-8") as fc:
        for u, a, b in zip(uniq.tolist(), start.tolist(), ends.tolist()):
            allrows = c_o[a:b]
            accrows = allrows[k_o[a:b]]
            fc.write(f"{u}\t{_join(rb, allrows)}\n")
            if len(accrows):
                fa.write(f"{u}\t{_join(rb, accrows)}\n")
                n_acc += len(accrows)
    done.write_text("1", encoding="utf-8")
    return gi, n, n_acc, False


def _join(rb, rows: np.ndarray) -> str:
    """Comma-join de-duplicated entity ids (validator rejects repeats)."""
    vals = rb.take(rows, 4)
    seen: set[str] = set()
    out: list[str] = []
    for v in vals:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return ",".join(out)


def main() -> None:
    g = guard()
    p = paths()
    cfg = get_config()
    n_workers = max(1, min(int(cfg["resources"]["max_workers"]),
                           g.max_workers))
    env_w = os.environ.get("WORKERS")
    if env_w:
        n_workers = max(1, int(env_w))
    meta = json.loads((p["models_dir"] / "best_model.json")
                      .read_text(encoding="utf-8"))
    threshold = float(meta["threshold"])
    print(f"[model] {meta['name']} threshold={threshold:.4f} "
          f"val_macro_f05={meta['val_macro_f05']:.6f} | {g.status()}", flush=True)

    out_dir = p["output_dir"] / "_parts"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Resumability: shards are the durable checkpoint and must NOT be wiped on
    # start. Only an explicit RESUME_FROM_SCRATCH=1 clears them, because
    # deleting them here previously destroyed 8 completed row groups every time
    # the script was launched.
    if os.environ.get("RESUME_FROM_SCRATCH") == "1":
        for pat in ("*.txt", "*.marker"):
            for old in out_dir.glob(pat):
                old.unlink()
        print("[clean] RESUME_FROM_SCRATCH=1 -> removed previous shards",
              flush=True)
    else:
        n_done = len(list(out_dir.glob("*.marker")))
        n_txt = len(list(out_dir.glob("*.txt")))
        print(f"[resume] {n_done} completed shards present "
              f"({n_txt} shard files) - keeping them", flush=True)

    limit = int(os.environ.get("LIMIT_GROUPS", "0"))
    srcs = {2: CACHE / "test_source2_norm.parquet",
            3: CACHE / "test_source3_norm.parquet"}
    # Build every store up front in the parent: a worker memmapping a file that
    # another worker is still writing fails with OSError EINVAL.
    print("[store] building column stores (one-off) | " + g.status(), flush=True)
    t_store = time.perf_counter()
    ColumnBundle(CACHE / "test_source1_norm.parquet", STORE, with_ids=False)
    for sid, src in srcs.items():
        ColumnBundle(src, STORE, with_ids=True)
    print(f"[store] ready in {(time.perf_counter() - t_store) / 60:.1f}m | "
          f"{g.status()}", flush=True)

    jobs = []
    for sid in (2, 3):
        n_rg = pq.ParquetFile(str(CACHE / f"test_cand_S{sid}.parquet")
                              ).metadata.num_row_groups
        for gi in range(n_rg):
            jobs.append((sid, gi))
    print(f"[plan] {len(jobs):,} row-group jobs on {n_workers} workers "
          f"(limit_groups={limit})", flush=True)

    t0 = time.perf_counter()
    n_scored = n_acc = 0
    n_skip = 0
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
            max_workers=n_workers, mp_context=ctx, initializer=_init,
            initargs=(threshold, n_workers, STORE,
                      CACHE / "test_source1_norm.parquet", srcs, out_dir,
                      limit)) as ex:
        for k, (gi, n, na, skipped) in enumerate(ex.map(_work, jobs,
                                                        chunksize=1), 1):
            n_scored += n
            n_acc += na
            if skipped:
                n_skip += 1
            if k % 100 == 0 or k == len(jobs):
                el = time.perf_counter() - t0
                rate = n_scored / max(el, 1e-9)
                left = (len(jobs) - k) / max(k / max(el, 1e-9), 1e-9)
                print(f"  {k}/{len(jobs)} groups  scored {n_scored:,}  "
                      f"accepted {n_acc:,}  {rate:,.0f} pairs/s  "
                      f"ETA {left / 60:.0f}m | {g.status()}", flush=True)
    el = time.perf_counter() - t0
    n_done = len(list(out_dir.glob("*.marker")))
    print(f"[score] this run: {n_scored:,} candidates, {n_acc:,} accepted, "
          f"{n_skip:,} shards reused, in {el / 60:.2f}m "
          f"({n_scored / max(el, 1e-9):,.0f} pairs/s)", flush=True)
    print(f"[score] total completed shards on disk: {n_done}", flush=True)


if __name__ == "__main__":
    main()
