"""
Experiment: per-channel candidate recall on real ground truth.

Why
---
Exact channels (name / address / address-digits) were measured at 0.6903 union
recall on a 4000-S1 sample: 31% of true matches are invisible to exact
blocking. This experiment measures whether a rare-token channel and/or a
fuzzy channel close that gap, and at what candidate volume.

Method
------
For each source (S2, S3) independently:
  1. one streaming scan to locate the row index of every sampled GT match
     (so recall can be evaluated without a row->id map in RAM),
  2. build the array-based indexes from that source,
  3. query every sampled S1 through each channel and record which GT matches
     were retrieved,
  4. free the source index before touching the next source.

Only one source index exists at a time. All sizes are under ResourceGuard.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.indexing import PostingIndex, TokenIndex, hash_str_array
from src.utils import guard


CACHE = Path("D:/amazon_ml/cache")
SOURCES = [("S2", CACHE / "train_source2_norm.parquet"),
           ("S3", CACHE / "train_source3_norm.parquet")]
CHANNELS = ["exact_name", "exact_address", "address_digits",
            "name_token", "addr_token"]


def locate_rows(path: Path, wanted: set[str]) -> dict[str, int]:
    """Return {entity_id: row_index} for the given ids via one streaming scan."""
    found: dict[str, int] = {}
    want = pa.array(sorted(wanted), type=pa.string())
    base = 0
    for batch in pq.ParquetFile(str(path)).iter_batches(
        batch_size=500_000, columns=["entity_id"]
    ):
        col = batch.column("entity_id")
        mask = pc_in(col, want)
        if mask is not None:
            sel = col.filter(mask).to_pylist()
            for j, eid in enumerate(sel):
                found[eid] = base + int(np.asarray(mask).nonzero()[0][j])
        base += len(batch)
        del batch, col, mask
    return found


def pc_in(col, want):
    import pyarrow.compute as pc
    return pc.is_in(col, value_set=want)


def main() -> None:
    g = guard()
    print(f"[guard] {g.status()}", flush=True)

    n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    gt = json.loads((CACHE / "train_ground_truth.json").read_text(encoding="utf-8"))
    pos_keys = np.array([k for k, v in gt.items() if v], dtype=object)
    rng = np.random.default_rng(seed)
    sample = [str(x) for x in rng.choice(pos_keys, min(n_sample, len(pos_keys)),
                                        replace=False)]
    sample_set = set(sample)
    print(f"[sample] {len(sample):,} S1 entities with >=1 match", flush=True)

    # ---- S1 attributes for the sample ----
    s1_attr: dict[str, tuple[str, str, str]] = {}
    for batch in pq.ParquetFile(str(CACHE / "train_source1_norm.parquet")).iter_batches(
        batch_size=200_000,
        columns=["entity_id", "name_norm", "address_norm", "address_dig"],
    ):
        d = batch.to_pandas()
        sel = d[d["entity_id"].isin(sample_set)]
        for r in sel.itertuples(index=False):
            s1_attr[r.entity_id] = (r.name_norm, r.address_norm, r.address_dig)
        del d, batch
        if len(s1_attr) == len(sample_set):
            break
    print(f"[sample] loaded {len(s1_attr):,} S1 rows | {g.status()}", flush=True)

    # per-channel set of retrieved (s1, gt_match) pairs
    hit: dict[str, set] = {c: set() for c in CHANNELS}
    volume: dict[str, int] = {c: 0 for c in CHANNELS}
    n_docs: dict[str, int] = {}
    timings: dict[str, float] = {}

    for label, path in SOURCES:
        print(f"\n===== {label} =====  | {g.status()}", flush=True)

        # 1. locate GT match rows in this source
        wanted: set[str] = set()
        for s in sample:
            wanted.update(gt[s])
        t0 = time.perf_counter()
        row_of = locate_rows(path, wanted)
        print(f"  located {len(row_of):,}/{len(wanted):,} GT rows in {label} "
              f"({time.perf_counter()-t0:.0f}s) | {g.status()}", flush=True)
        n_docs[label] = pq.ParquetFile(str(path)).metadata.num_rows

        # per-S1 target row set restricted to this source
        targets: dict[str, set[int]] = {s: set() for s in sample}
        for s in sample:
            for m in gt[s]:
                r = row_of.get(m)
                if r is not None:
                    targets[s].add(r)

        # 2. build indexes
        t0 = time.perf_counter()
        ix_name = PostingIndex.from_column(path, "name_norm")
        ix_addr = PostingIndex.from_column(path, "address_norm")
        ix_dig = PostingIndex.from_column(path, "address_dig")
        timings[f"{label}_exact_index"] = round(time.perf_counter() - t0, 1)
        print(f"  exact indexes built in {timings[f'{label}_exact_index']}s "
              f"| name={len(ix_name.keys):,}k addr={len(ix_addr.keys):,}k "
              f"dig={len(ix_dig.keys):,}k | {g.status()}", flush=True)

        max_df = max(1, int(0.01 * n_docs[label]))
        t0 = time.perf_counter()
        tk_name = TokenIndex.from_column(path, "name_tokens_str", 3, max_df)
        timings[f"{label}_name_token_index"] = round(time.perf_counter() - t0, 1)
        print(f"  name-token index {tk_name.n_vocab:,} vocab / "
              f"{len(tk_name.rows):,} postings in "
              f"{timings[f'{label}_name_token_index']}s | {g.status()}", flush=True)
        t0 = time.perf_counter()
        tk_addr = TokenIndex.from_column(path, "address_tokens_str", 3, max_df)
        timings[f"{label}_addr_token_index"] = round(time.perf_counter() - t0, 1)
        print(f"  addr-token index {tk_addr.n_vocab:,} vocab / "
              f"{len(tk_addr.rows):,} postings in "
              f"{timings[f'{label}_addr_token_index']}s | {g.status()}", flush=True)

        # 3. query
        t0 = time.perf_counter()
        h_name = hash_str_array([s1_attr[s][0] for s in sample])
        h_addr = hash_str_array([s1_attr[s][1] for s in sample])
        h_dig = hash_str_array([s1_attr[s][2] for s in sample])

        for s, kn, ka, kd in zip(sample, h_name, h_addr, h_dig):
            tgt = targets[s]
            if not tgt:
                continue
            for key, ix, ch in ((kn, ix_name, "exact_name"),
                                (ka, ix_addr, "exact_address"),
                                (kd, ix_dig, "address_digits")):
                if key == 0:
                    continue
                rows = ix.get(int(key))
                volume[ch] += len(rows)
                for r in rows.tolist():
                    if r in tgt:
                        hit[ch].add((s, r))

            for toks, tk, ch in ((s1_attr[s][0], tk_name, "name_token"),
                                  (s1_attr[s][1], tk_addr, "addr_token")):
                seen: set[int] = set()
                for t in toks.split():
                    if len(t) < 3:
                        continue
                    rows = tk.get(t)
                    if len(rows):
                        seen.update(rows.tolist())
                volume[ch] += len(seen)
                for r in seen:
                    if r in tgt:
                        hit[ch].add((s, r))

        timings[f"{label}_query"] = round(time.perf_counter() - t0, 1)
        print(f"  queried {len(sample):,} S1 in {timings[f'{label}_query']}s "
              f"| {g.status()}", flush=True)

        del ix_name, ix_addr, ix_dig, tk_name, tk_addr, row_of, targets
        g.gc()
        print(f"  {label} freed | {g.status()}", flush=True)

    # ---- report ----
    total_gt = sum(len(gt[s]) for s in sample)
    report = {
        "n_sample_s1": len(sample),
        "total_gt_pairs": total_gt,
        "timings_sec": timings,
        "n_docs": n_docs,
        "channels": {},
    }
    union: set = set()
    for c in CHANNELS:
        union |= hit[c]
        report["channels"][c] = {
            "recall": round(len(hit[c]) / total_gt, 4),
            "hits": len(hit[c]),
            "candidate_rows_emitted": volume[c],
        }
    report["union_all_channels"] = {
        "recall": round(len(union) / total_gt, 4),
        "hits": len(union),
    }
    print("\n===== CHANNEL RECALL =====")
    print(json.dumps(report, indent=2), flush=True)
    out = Path("D:/amazon_ml/reports/channel_recall_experiment.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
