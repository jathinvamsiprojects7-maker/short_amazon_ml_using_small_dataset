"""
Stage 3: labelled train/validation pairs (S1-grouped).

Labelling
---------
A candidate pair is positive iff the candidate's entity id is in that S1's
ground-truth list. Only the training ground truth is used.

Split
-----
Grouped by S1 entity: every pair of an S1 goes entirely to train or entirely
to validation, stratified by match cardinality (0 / 1 / >1). The split is a
single lookup on the pair's ``s1_row`` against a precomputed byte array, so no
pair can leak across the split.

Efficiency
----------
The candidate universe is ~4.4e9 pairs and cannot be materialised:

  * positives are found by a sorted int64-key intersection. All positive
    ``(s1_row << 32 | cand_row)`` keys for a source are sorted once, then each
    candidate batch is resolved with ``np.searchsorted`` - O(n log n), no
    Python loop, no giant hash set.
  * negatives are capped per S1 (``max_neg_per_s1``), which bounds the sample
    while keeping every S1 represented.

Output
------
``train_pairs.npz``: idx (int64, row index into the flattened candidate set),
s1_row (int32), y (int8), split (uint8; 0=train, 1=val).
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import get_config, guard, paths

CACHE = Path("D:/amazon_ml/cache")
SHIFT = np.int64(1) << np.int64(32)


def s1_grouped_split(gt: dict, val_fraction: float, seed: int):
    """
    Split S1 entities into train/val, stratified by match cardinality.

    Stratifying keeps zero-match and singleton S1 represented in validation,
    which matters because a false positive on a zero-match S1 scores 0 for that
    S1 under Macro F0.5.
    """
    rng = np.random.default_rng(seed)
    buckets: list[list[str]] = [[], [], []]
    for s, v in gt.items():
        k = len(v)
        buckets[0 if k == 0 else 1 if k == 1 else 2].append(s)
    train: list[str] = []
    val: list[str] = []
    for group in buckets:
        arr = np.array(group, dtype=object)
        idx = rng.permutation(len(arr))
        n_val = int(round(len(arr) * val_fraction))
        vset = set(arr[idx[:n_val]].tolist())
        for s in group:
            (val if s in vset else train).append(s)
    return train, val


def main() -> None:
    g = guard()
    cfg = get_config()
    tcfg = cfg["training"]
    val_fraction = float(tcfg["val_fraction"])
    seed = int(tcfg["random_seed"])
    max_neg = int(tcfg.get("max_neg_per_s1", 40))

    gt = json.loads((CACHE / "train_ground_truth.json").read_text(encoding="utf-8"))
    n_gt_pairs = sum(len(v) for v in gt.values())
    print(f"[gt] {len(gt):,} S1, {n_gt_pairs:,} pairs", flush=True)

    # ---- S1 entity_id -> row, and the split lookup ----
    s1_ids = pq.read_table(str(CACHE / "train_source1_norm.parquet"),
                           columns=["entity_id"]).column("entity_id").to_pylist()
    s1_row_lookup = {e: i for i, e in enumerate(s1_ids)}
    n_s1 = len(s1_ids)
    del s1_ids
    g.gc()
    print(f"[ids] s1 rows {n_s1:,} | {g.status()}", flush=True)

    train_s1, val_s1 = s1_grouped_split(gt, val_fraction, seed)
    is_val = np.zeros(n_s1, dtype=np.uint8)
    for s in val_s1:
        r = s1_row_lookup.get(s)
        if r is not None:
            is_val[r] = 1
    print(f"[split] train {len(train_s1):,} | val {len(val_s1):,}", flush=True)

    # ---- positive keys per source ----
    pos_keys: dict[int, np.ndarray] = {}
    for sid in (2, 3):
        ids = pq.read_table(str(CACHE / f"train_source{sid}_norm.parquet"),
                            columns=["entity_id"]).column("entity_id").to_pylist()
        row_of = {e: i for i, e in enumerate(ids)}
        del ids
        g.gc()
        parts_k = []
        for s, matches in gt.items():
            sr = s1_row_lookup.get(s)
            if sr is None:
                continue
            rows = [row_of[m] for m in matches
                    if m.startswith(f"S{sid}-") and m in row_of]
            if rows:
                # key layout MUST match the scan: (s1_row << 32) | cand_row
                parts_k.append(np.asarray(rows, dtype=np.int64)
                               + np.int64(sr) * SHIFT)
        if parts_k:
            k = np.concatenate(parts_k)
            # k MUST be sorted: the scan resolves positives with
            # np.searchsorted, which is only correct on sorted data. Sorting
            # only `o` (as an earlier version did) left k unsorted and made
            # every positive lookup fail.
            k.sort(kind="stable")
            pos_keys[sid] = k
        else:
            pos_keys[sid] = np.empty(0, np.int64)
        del row_of, parts_k
        g.gc()
        print(f"[pos] S{sid}: {len(pos_keys[sid]):,} positive pairs "
              f"| {g.status()}", flush=True)

    # ---- scan candidates ----
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    offset = 0
    src_offsets: dict[int, tuple[int, int]] = {}
    out_idx: list[np.ndarray] = []
    out_s1: list[np.ndarray] = []
    out_y: list[np.ndarray] = []
    n_pos = n_neg = 0
    neg_taken: Counter = Counter()

    for sid in (2, 3):
        path = CACHE / f"train_cand_S{sid}.parquet"
        if not path.exists():
            print(f"[skip] {path.name} missing", flush=True)
            continue
        print(f"[scan] {path.name} | {g.status()}", flush=True)
        pk = pos_keys[sid]
        start = offset
        pf = pq.ParquetFile(str(path))
        seen = 0
        neg_budget = max_neg
        for rb in pf.iter_batches(batch_size=2_000_000,
                                  columns=["s1_row", "cand_row"]):
            s1r = rb.column("s1_row").to_numpy().astype(np.int64)
            cr = rb.column("cand_row").to_numpy().astype(np.int64)
            m = len(s1r)

            key = s1r * SHIFT + cr
            is_pos = np.zeros(m, dtype=bool)
            if len(pk):
                j = np.minimum(np.searchsorted(pk, key), len(pk) - 1)
                is_pos = pk[j] == key

            pos_here = np.flatnonzero(is_pos)
            if len(pos_here):
                out_idx.append(pos_here.astype(np.int64) + start)
                out_s1.append(s1r[pos_here].astype(np.int32))
                out_y.append(np.ones(len(pos_here), dtype=np.int8))
                n_pos += len(pos_here)

            neg_here = np.flatnonzero(~is_pos)
            if len(neg_here):
                rows_s1 = s1r[neg_here]
                # keep the first `max_neg` negatives of each S1 across the WHOLE
                # scan, not per batch, so the cap is exact and every S1 is
                # represented. neg_taken tracks the running count per S1 row.
                order = np.argsort(rows_s1, kind="stable")
                sorted_s1 = rows_s1[order]
                uniq, first_idx, counts = np.unique(
                    sorted_s1, return_index=True, return_counts=True)
                already = np.fromiter(
                    (neg_taken.get(int(u), 0) for u in uniq.tolist()),
                    dtype=np.int64, count=len(uniq))
                room = np.maximum(neg_budget - already, 0)
                rank = np.arange(len(sorted_s1), dtype=np.int64) \
                    - np.repeat(first_idx, counts)
                keep_sorted = rank < np.repeat(room, counts)
                if keep_sorted.any():
                    sel = neg_here[order[keep_sorted]]
                    out_idx.append(sel.astype(np.int64) + start)
                    out_s1.append(s1r[sel].astype(np.int32))
                    out_y.append(np.zeros(len(sel), dtype=np.int8))
                    n_neg += len(sel)
                    for u, c in zip(uniq.tolist(), counts.tolist()):
                        neg_taken[u] = neg_taken.get(u, 0) + min(c, neg_budget)

            seen += m
            offset += m
            del rb, s1r, cr, key, is_pos
            g.gc(1)
            if seen % 400_000_000 == 0:
                print(f"  {path.name}: {seen:,} scanned  pos={n_pos:,} "
                      f"neg={n_neg:,}  ({time.perf_counter() - t0:.0f}s) "
                      f"| {g.status()}", flush=True)
        del pf
        g.gc()
        src_offsets[sid] = (start, offset)
        print(f"[scan] {path.name} done: {seen:,} rows "
              f"({time.perf_counter() - t0:.0f}s) | {g.status()}", flush=True)

    idx = np.concatenate(out_idx)
    s1row = np.concatenate(out_s1)
    y = np.concatenate(out_y)
    del out_idx, out_s1, out_y
    g.gc()

    split = is_val[s1row]
    print(f"[pairs] total {len(idx):,}  pos {n_pos:,}  neg {n_neg:,}")
    print(f"[split] train pairs {int((split == 0).sum()):,}  "
          f"val pairs {int((split == 1).sum()):,}")

    out = CACHE / "train_pairs.npz"
    np.savez(out, idx=idx.astype(np.int64), s1_row=s1row.astype(np.int32),
             y=y.astype(np.int8), split=split.astype(np.uint8),
             src_offsets=np.array([src_offsets.get(2, (0, 0)),
                                   src_offsets.get(3, (0, 0))], dtype=np.int64))
    print(f"[saved] {out} ({out.stat().st_size / 1e6:.0f} MB)")
    print(f"[val positives] {int(y[split == 1].sum()):,}  "
          f"[val S1] {len(np.unique(s1row[split == 1])):,}")
    print(f"[val S1 w/ >=1 pos] "
          f"{len(np.unique(s1row[(split == 1) & (y == 1)])):,}")


if __name__ == "__main__":
    main()
