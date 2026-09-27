"""
Experiment: DEDUPLICATED candidate volume and recall per channel union.

Why
---
All previous volume numbers were *pre-deduplication* sums of posting lengths.
Multiple rare tokens of the same S1 hit the same document, so the true
candidate count is much smaller. Without the deduplicated number we cannot
choose a max_df operating point, so it must be measured.

Method
------
For a sample of S1 entities, run the full channel set with a configurable
rare-token max_df, deduplicate candidates per S1, and report:

    recall      = GT pairs retrieved / GT pairs
    rows/S1     = deduplicated candidate rows emitted per S1
    pctiles     of the per-S1 candidate count

Only one source index exists at a time; S2 is fully released before S3.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.indexing import PostingIndex, TokenIndex, hash_str_array
from src.utils import guard

CACHE = Path("D:/amazon_ml/cache")
SOURCES = [("S2", CACHE / "train_source2_norm.parquet"),
           ("S3", CACHE / "train_source3_norm.parquet")]
MIN_LEN = 3


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
    print(f"[guard] {g.status()}", flush=True)

    max_df_list = [100, 250, 500, 1000, 2500]
    n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 2000

    with open(CACHE / "_diag_sample.pkl", "rb") as f:
        d = pickle.load(f)
    s1, gt, found, sample = d["s1"], d["gt"], d["found"], d["sample"]
    sample = sample[:n_sample]
    total = sum(len(gt[s]) for s in sample)
    print(f"[sample] {len(sample):,} S1 / {total:,} GT pairs", flush=True)

    # accumulated per (max_df) -> (hit pairs, per-S1 counts)
    acc = {m: {"pairs": set(), "counts": Counter()} for m in max_df_list}
    exact_pairs: set = set()

    for label, path in SOURCES:
        print(f"\n===== {label} =====  | {g.status()}", flush=True)
        n_docs = pq.ParquetFile(str(path)).metadata.num_rows

        ix_name = PostingIndex.from_column(path, "name_norm")
        ix_addr = PostingIndex.from_column(path, "address_norm")
        ix_dig = PostingIndex.from_column(path, "address_dig")
        print(f"  exact indexes built | {g.status()}", flush=True)

        want = set()
        for s in sample:
            for mid in gt[s]:
                if mid.startswith(label + "-"):
                    want.add(mid)
        row_of = locate_rows(path, want)
        print(f"  located {len(row_of):,}/{len(want):,} GT rows", flush=True)

        tk_cache = {}
        for m in max_df_list:
            tk_name = TokenIndex.from_column(path, "name_tokens_str", MIN_LEN, m)
            tk_addr = TokenIndex.from_column(path, "address_tokens_str", MIN_LEN, m)
            tk_cache[m] = (tk_name, tk_addr)
            print(f"  max_df={m:>5} name-vocab={tk_name.n_vocab:,} "
                  f"addr-vocab={tk_addr.n_vocab:,} | {g.status()}", flush=True)

        hn = hash_str_array([s1[s][0] for s in sample])
        ha = hash_str_array([s1[s][1] for s in sample])
        hd = hash_str_array([s1[s][2] for s in sample])

        for si, (s, kn, ka, kd) in enumerate(zip(sample, hn, ha, hd)):
            tgt = {row_of[mid] for mid in gt[s] if mid in row_of}
            if not tgt:
                continue
            key0 = (label, s)

            # exact channels (counted once, not per max_df)
            bucket: set[int] = set()
            if kn:
                bucket.update(ix_name.get(int(kn)).tolist())
            if ka:
                bucket.update(ix_addr.get(int(ka)).tolist())
            if kd:
                bucket.update(ix_dig.get(int(kd)).tolist())
            for r in bucket:
                if r in tgt:
                    exact_pairs.add(key0 + (r,))

            for m in max_df_list:
                tk_name, tk_addr = tk_cache[m]
                cand: set[int] = set()
                for toks, tk in ((s1[s][0], tk_name), (s1[s][1], tk_addr)):
                    if not toks:
                        continue
                    rows = tk.rarest(toks, MIN_LEN)
                    if len(rows):
                        cand.update(rows.tolist())
                acc[m]["counts"][s] += len(cand)
                for r in cand:
                    if r in tgt:
                        acc[m]["pairs"].add(key0 + (r,))
            if si % 500 == 0:
                print(f"    {si}/{len(sample)} | {g.status()}", flush=True)

        del ix_name, ix_addr, ix_dig, tk_cache
        g.gc()
        print(f"  {label} freed | {g.status()}", flush=True)

    # ---- report ----
    print("\n===== DEDUPLICATED VOLUME / RECALL =====")
    out = {"n_sample_s1": len(sample), "total_gt_pairs": total, "rows": {}}

    e_rec = len(exact_pairs) / total
    print(f"  exact channels: recall {e_rec:.4f}")
    out["exact"] = {"recall": round(e_rec, 4)}

    for m in max_df_list:
        c = np.array([acc[m]["counts"].get(s, 0) for s in sample], dtype=np.float64)
        tok_rec = len(acc[m]["pairs"]) / total
        union = acc[m]["pairs"] | exact_pairs
        u_rec = len(union) / total
        # union volume = exact volume + token volume (upper bound, exact
        # overlaps are small because exact keys are mostly unique)
        out["rows"][m] = {
            "token_recall": round(tok_rec, 4),
            "union_recall": round(u_rec, 4),
            "token_rows_per_s1_mean": round(float(c.mean()), 1),
            "p50": int(np.percentile(c, 50)),
            "p95": int(np.percentile(c, 95)),
            "p99": int(np.percentile(c, 99)),
            "est_total_pairs_full_train": int(c.mean() * 2_206_821),
        }
        r = out["rows"][m]
        print(f"  max_df={m:>5}: union_recall {u_rec:.4f}  "
              f"rows/S1 mean {c.mean():8,.0f}  p50 {r['p50']:>7,}  "
              f"p95 {r['p95']:>8,}  p99 {r['p99']:>8,}  "
              f"full-train pairs ~{r['est_total_pairs_full_train']/1e6:,.0f}M")

    Path("D:/amazon_ml/reports/dedup_volume_experiment.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\n[saved] D:/amazon_ml/reports/dedup_volume_experiment.json")


if __name__ == "__main__":
    main()
