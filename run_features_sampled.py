"""
Stage 2b driver: features for the SAMPLED candidate pairs.

The full candidate universe is 4.4e9 pairs and cannot be featurised in the
available time. The model trains and validates on the sampled set produced by
run_training_pairs.py (all positives + capped negatives), so only those rows
need features.

Rows are addressed by their global index into the concatenated candidate set
(S2 rows first, then S3). ``s1_row``, ``cand_row`` and the channel bitmask are
read straight from the candidate parquet, so no per-pair strings are built.
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
from run_features import RowStore, compute_batch
from src.features import N_FEATURES
from src.utils import guard, paths

CACHE = Path("D:/amazon_ml/cache")


def main() -> None:
    g = guard()
    p = paths()
    cfg = __import__("src.utils", fromlist=["get_config"]).get_config()
    run_features.WORKERS = max(1, min(int(cfg["resources"]["max_workers"]),
                                       g.max_workers))
    print(f"[cfg] rapidfuzz workers={run_features.WORKERS} | {g.status()}",
          flush=True)

    d = np.load(CACHE / "train_pairs.npz")
    idx = d["idx"]
    src_offsets = d["src_offsets"]
    del d

    order = np.argsort(idx, kind="stable")
    idx_sorted = idx[order]
    del idx, order
    n = len(idx_sorted)
    g.gc()
    print(f"[pairs] {n:,} rows to featurise | {g.status()}", flush=True)

    out = CACHE / "train_features.npy"
    meta_path = CACHE / "train_features.meta.json"
    if out.exists() and meta_path.exists():
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        if m.get("n_pairs") == n and m.get("n_features") == N_FEATURES:
            print(f"[skip] feature cache valid: {n:,} x {N_FEATURES}")
            return
        out.unlink(missing_ok=True)

    s1 = RowStore(CACHE / "train_source1_norm.parquet")
    stores = {2: RowStore(CACHE / "train_source2_norm.parquet"),
              3: RowStore(CACHE / "train_source3_norm.parquet")}
    print(f"[stores] loaded | {g.status()}", flush=True)

    b2 = int(src_offsets[0][1])
    src_of = np.where(idx_sorted < b2, 2, 3)
    del idx_sorted
    g.gc()

    fp = np.lib.format.open_memmap(str(out), mode="w+", dtype=np.float32,
                                   shape=(n, N_FEATURES))
    t0 = time.perf_counter()
    batch = 200_000
    for sid, path, start_g, end_g in (
            (2, CACHE / "train_cand_S2.parquet", 0, b2),
            (3, CACHE / "train_cand_S3.parquet", b2, n)):
        want = np.flatnonzero(src_of == sid)
        if not len(want):
            continue
        lo, hi = int(want[0]), int(want[-1]) + 1
        print(f"[src] S{sid}: global rows {lo:,}..{hi:,} | {g.status()}", flush=True)
        pf = pq.ParquetFile(str(path))
        file_pos = 0          # rows consumed from this candidate file
        for rb in pf.iter_batches(batch_size=2_000_000,
                                  columns=["s1_row", "cand_row", "channels"]):
            mlen = len(rb)
            # overlap of [file_pos, file_pos+mlen) with the needed range
            need_lo = max(lo - start_g, file_pos)
            need_hi = min(hi - start_g, file_pos + mlen)
            if need_hi > need_lo:
                a = need_lo - file_pos
                b = need_hi - file_pos
                s1r = rb.column("s1_row").to_numpy()[a:b]
                cr = rb.column("cand_row").to_numpy()[a:b]
                ch = rb.column("channels").to_pylist()[a:b]
                nb = len(s1r)
                st = stores[sid]
                X = compute_batch(
                    s1.take(s1r, 0), s1.take(s1r, 1), s1.take(s1r, 2),
                    s1.take(s1r, 3),
                    st.take(cr, 0), st.take(cr, 1), st.take(cr, 2),
                    st.take(cr, 3),
                    np.fromiter((c.count("|") + 1 if c else 0 for c in ch),
                                dtype=np.float32, count=nb),
                    {k: np.fromiter((k in c for c in ch),
                                    dtype=np.float32, count=nb)
                     for k in run_features.CHANNEL_KEYS},
                )
                g_lo = start_g + need_lo
                fp[g_lo:g_lo + nb] = X
                del X
            file_pos += mlen
            del rb
            g.gc(1)
            if file_pos % 20_000_000 < 2_000_000 or file_pos >= hi - start_g:
                el = time.perf_counter() - t0
                done = min(file_pos, hi - start_g)
                print(f"  S{sid} {done:,}/{hi - lo:,}  "
                      f"{max(done, 1) / max(el, 1e-9):,.0f} pairs/s | "
                      f"{g.status()}", flush=True)
            if file_pos >= hi - start_g:
                break
        del pf
        g.gc()

    fp.flush()
    del fp
    g.gc()
    meta_path.write_text(json.dumps({
        "n_pairs": n, "n_features": N_FEATURES,
        "feature_names": run_features.FEATURE_NAMES,
    }), encoding="utf-8")
    print(f"[done] {n:,} x {N_FEATURES} in "
          f"{(time.perf_counter() - t0) / 60:.1f}m -> {out.name} "
          f"({out.stat().st_size / 1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
