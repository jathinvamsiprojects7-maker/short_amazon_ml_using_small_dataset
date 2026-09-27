"""
Experiment: AND-conjunctive multi-token blocking (the correct block key).

Why the previous approaches failed (all measured)
-------------------------------------------------
Token df distribution on 2M S2 docs is extremely skewed:
    p10=1  p25=1  p50=1  p75=1  p90=6  p99=72  max=205,267

Consequences:
  * "union of all tokens of an S1" -> 1.4M candidates/S1 (1.00 recall). The
    rarest token is almost always a singleton (df=1), so restricting to it
    still retrieves a 205k-row bucket for a common-but-present token and
    misses the match when the rarest token is absent from the target.
  * max_df has almost no effect: max_df<=100 already keeps 99.2% of the
    vocabulary, so it removes almost nothing while recall stays 0.9632 at
    9,061 rows/S1 (~20e9 pairs over the full train set - infeasible).

The correct block key is a CONJUNCTION over a *set* of tokens, where the set
size grows with how uninformative the individual tokens are. That is the
classic "sorted neighborhood / prefix filter" idea: require agreement on
k = |Q| - |Q| + slack tokens, which bounds the result set independent of the
data skew.

This experiment measures, on real ground truth, for a family of prefix-filter
block keys:
    recall(k)        vs   deduplicated rows/S1
so an operating point with recall >= 0.98 and a feasible volume can be
selected from evidence instead of guesswork.
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


def prefix_keys(tokens: list[str], k: int) -> list[tuple]:
    """Sorted prefix blocks of length k (the prefix-filter scheme)."""
    if len(tokens) < k:
        return []
    t = sorted(set(tokens))
    if len(t) < k:
        return []
    return [tuple(t[i:i + k]) for i in range(len(t) - k + 1)]


def main() -> None:
    g = guard()
    print(f"[guard] {g.status()}", flush=True)

    n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    # k = 1, 2, 3 token conjunctions
    k_list = [1, 2, 3]

    with open(CACHE / "_diag_sample.pkl", "rb") as f:
        d = pickle.load(f)
    s1, gt, found, sample = d["s1"], d["gt"], d["found"], d["sample"]
    sample = sample[:n_sample]
    total = sum(len(gt[s]) for s in sample)
    print(f"[sample] {len(sample):,} S1 / {total:,} GT pairs", flush=True)

    # pair -> per-k hit sets and per-S1 candidate counts
    hits = {k: set() for k in k_list}
    counts = {k: Counter() for k in k_list}
    exact_pairs: set = set()

    for label, path in SOURCES:
        print(f"\n===== {label} =====  | {g.status()}", flush=True)
        n_docs = pq.ParquetFile(str(path)).metadata.num_rows

        # ---- build conjunction posting lists for each k ----
        idx_by_k: dict[int, dict] = {}
        for k in k_list:
            post: dict[tuple, list] = {}
            base = 0
            t0 = time.perf_counter()
            for b in pq.ParquetFile(str(path)).iter_batches(
                batch_size=200_000, columns=["name_norm", "address_norm"]
            ):
                dd = b.to_pandas()
                for nm, ad in zip(dd["name_norm"].fillna(""),
                                  dd["address_norm"].fillna("")):
                    toks = [t for t in ((nm or "") + " " + (ad or "")).split()
                            if len(t) >= MIN_LEN]
                    for key in prefix_keys(toks, k):
                        post.setdefault(key, []).append(base)
                    base += 1
                del dd, b
            g.gc()
            sizes = np.array([len(v) for v in post.values()], dtype=np.int64) \
                if post else np.array([0])
            print(f"  k={k}: {len(post):,} block keys  "
                  f"postings={sizes.sum():,}  "
                  f"max bucket={sizes.max():,}  "
                  f"p99={np.percentile(sizes,99):,.0f}  "
                  f"{time.perf_counter()-t0:.0f}s | {g.status()}", flush=True)
            idx_by_k[k] = post

        want = set()
        for s in sample:
            for mid in gt[s]:
                if mid.startswith(label + "-"):
                    want.add(mid)
        row_of = locate_rows(path, want)
        print(f"  located {len(row_of):,}/{len(want):,} GT rows", flush=True)

        for si, s in enumerate(sample):
            key0 = (label, s)
            tgt = {row_of[mid] for mid in gt[s] if mid in row_of}
            if not tgt:
                continue
            s1_toks = [t for t in ((s1[s][0] or "") + " " + (s1[s][1] or "")).split()
                       if len(t) >= MIN_LEN]
            for k in k_list:
                post = idx_by_k[k]
                cand: set[int] = set()
                for key in prefix_keys(s1_toks, k):
                    rows = post.get(key)
                    if rows:
                        cand.update(rows)
                counts[k][s] += len(cand)
                for r in cand:
                    if r in tgt:
                        hits[k].add(key0 + (r,))
                # exact channels counted once under the k=1 bucket
                if k == 1:
                    for mid in gt[s]:
                        r = row_of.get(mid)
                        if r is not None and r in cand:
                            exact_pairs.add(key0 + (r,))
            if si % 500 == 0:
                print(f"    {si}/{len(sample)} | {g.status()}", flush=True)

        del idx_by_k
        g.gc()
        print(f"  {label} freed | {g.status()}", flush=True)

    # ---- report ----
    print("\n===== CONJUNCTIVE (k-token) BLOCKING =====")
    print(f"{'k':>3}{'recall':>10}{'rows/S1 mean':>15}{'p50':>10}"
          f"{'p95':>11}{'p99':>11}{'est full-train pairs':>22}")
    out = {"n_sample_s1": len(sample), "total_gt_pairs": total, "rows": {}}
    for k in k_list:
        rec = len(hits[k]) / total
        c = np.array([counts[k].get(s, 0) for s in sample], dtype=np.float64)
        est = c.mean() * 2_206_821
        out["rows"][k] = {
            "recall": round(rec, 4),
            "rows_per_s1_mean": round(float(c.mean()), 1),
            "p50": int(np.percentile(c, 50)),
            "p95": int(np.percentile(c, 95)),
            "p99": int(np.percentile(c, 99)),
            "est_full_train_pairs": int(est),
        }
        print(f"{k:>3}{rec:10.4f}{c.mean():15,.0f}"
              f"{np.percentile(c,50):10,.0f}{np.percentile(c,95):11,.0f}"
              f"{np.percentile(c,99):11,.0f}{est/1e6:19,.0f}M")

    Path("D:/amazon_ml/reports/conjunction_blocking.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\n[saved] D:/amazon_ml/reports/conjunction_blocking.json")


if __name__ == "__main__":
    main()
