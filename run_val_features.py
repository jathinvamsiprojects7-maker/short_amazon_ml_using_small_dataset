"""
Recompute pair features for the VALIDATION pairs only.

The 15.4 GB full feature matrix was released after training, but the labelled
pair indices, split assignment and ground-truth labels are all still in
``train_pairs.npz``. Only the 14.1M validation pairs are needed to evaluate a
decision layer, so features are recomputed for exactly those, which costs about
12 minutes instead of 76.

Ground truth here is the pair label stored in ``train_pairs.npz`` (y == 1 means
the candidate is a true match), not any model output.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_features
from run_features import compute_batch
from src.features import N_FEATURES
from src.utils import guard

CACHE = Path("D:/amazon_ml/cache")


def main() -> None:
    g = guard()
    run_features.WORKERS = 6
    d = np.load(CACHE / "train_pairs.npz")
    idx, y, s1_row, split, src_off = (d["idx"], d["y"], d["s1_row"],
                                      d["split"], d["src_offsets"])
    va = np.flatnonzero(split == 1)
    v_idx = idx[va]
    v_y = y[va].astype(np.int8)
    v_s1 = s1_row[va].astype(np.int32)
    n = len(v_idx)
    print(f"[val-feat] {n:,} validation pairs, {int(v_y.sum()):,} positive | "
          f"{g.status()}", flush=True)

    from src.fixed_store import ColumnBundle
    store = CACHE / "store"
    s1b = ColumnBundle(CACHE / "train_source1_norm.parquet", store,
                       with_ids=False)
    srcb = {2: ColumnBundle(CACHE / "train_source2_norm.parquet", store,
                            with_ids=False),
            3: ColumnBundle(CACHE / "train_source3_norm.parquet", store,
                            with_ids=False)}
    print(f"[val-feat] column stores ready | {g.status()}", flush=True)

    order = np.argsort(v_idx, kind="stable")
    v_idx = v_idx[order]
    v_y = v_y[order]
    v_s1 = v_s1[order]
    del order

    out = CACHE / "val_features.npy"
    fp = np.lib.format.open_memmap(str(out), mode="w+", dtype=np.float32,
                                   shape=(n, N_FEATURES))
    b2 = int(src_off[0][1])
    src_of = np.where(v_idx < b2, 2, 3)

    t0 = time.perf_counter()
    for sid, path, start_g in ((2, CACHE / "train_cand_S2.parquet", 0),
                               (3, CACHE / "train_cand_S3.parquet", b2)):
        want = np.flatnonzero(src_of == sid)
        if not len(want):
            continue
        lo, hi = int(want[0]), int(want[-1]) + 1
        print(f"[val-feat] S{sid}: positions {lo:,}..{hi:,} | {g.status()}",
              flush=True)
        pf = pq.ParquetFile(str(path))
        pos = 0                      # index into v_idx
        file_pos = 0
        need_hi = hi
        rb_all = {}
        for rb in pf.iter_batches(batch_size=4_000_000,
                                  columns=["s1_row", "cand_row", "channels"]):
            m = len(rb)
            nlo = max(lo - start_g, file_pos)
            nhi = min(need_hi - start_g, file_pos + m)
            if nhi > nlo:
                a = nlo - file_pos
                b = nhi - file_pos
                a_s1 = rb.column("s1_row").to_numpy()[a:b].astype(np.int64)
                a_c = rb.column("cand_row").to_numpy()[a:b].astype(np.int64)
                ch = rb.column("channels").to_pylist()[a:b]
                nb = len(a_s1)
                st = srcb[sid]
                X = compute_batch(
                    s1b.take(a_s1, 0), s1b.take(a_s1, 1), s1b.take(a_s1, 2),
                    s1b.take(a_s1, 3),
                    st.take(a_c, 0), st.take(a_c, 1), st.take(a_c, 2),
                    st.take(a_c, 3),
                    np.fromiter((x.count("|") + 1 if x else 0 for x in ch),
                                dtype=np.float32, count=nb),
                    {k: np.fromiter((k in x for x in ch), dtype=np.float32,
                                    count=nb)
                     for k in run_features.CHANNEL_KEYS},
                )
                gpos = start_g + nlo - start_g
                fp[pos:pos + nb] = X
                pos += nb
                del X
            file_pos += m
            del rb
            g.gc(1)
            if pos and pos % 2_000_000 < 4_000_000:
                el = time.perf_counter() - t0
                print(f"  {pos:,}/{n:,}  {pos / max(el, 1e-9):,.0f}/s | "
                      f"{g.status()}", flush=True)
            if file_pos >= need_hi - start_g:
                break
        del pf
        g.gc()
    fp.flush()
    del fp
    g.gc()
    np.savez(CACHE / "val_pairs.npz", idx=v_idx, y=v_y, s1_row=v_s1)
    (CACHE / "val_features.meta.json").write_text(
        json.dumps({"n_pairs": n, "n_features": N_FEATURES}),
        encoding="utf-8")
    print(f"[val-feat] wrote {n:,} x {N_FEATURES} in "
          f"{(time.perf_counter() - t0) / 60:.1f}m | {g.status()}", flush=True)


if __name__ == "__main__":
    main()
