"""
Experiment: conjunctive (token-pair) blocking vs single-token blocking.

Motivation
----------
Single rare-token blocking on the measured curve:

    max_df    recall   rows/S1
      500     0.927        254
    2,500     0.973      1,979
    5,000     0.981      4,571   -> 2.2M x 4,571 = 10.1e9 pairs (infeasible)

So single-token blocking cannot reach high recall at a usable volume. The
binding problem is that one shared rare token still matches thousands of docs.

Hypothesis
----------
Requiring a *conjunction* of two rare tokens is far more selective while
barely costing recall, because 0.9946 of GT pairs already share >=2 tokens.
Blocking on the sorted 2-token combination key should therefore reach similar
recall at a small fraction of the volume.

Method
------
1. Stream each source and build the document frequency of every
   (rare_token_i, rare_token_j) combination, for name tokens and address
   tokens separately and combined.
2. For every GT pair, check whether a combination key is shared, and record
   the key's df -> exact recall at each max_df.
3. Estimate resulting volume per S1 by summing the posting lengths of the
   S1's combination keys.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import guard

CACHE = Path("D:/amazon_ml/cache")
SOURCES = {"S2": CACHE / "train_source2_norm.parquet",
           "S3": CACHE / "train_source3_norm.parquet"}
MIN_LEN = 3


def combo_df(path: Path, min_len: int, batch_size: int = 200_000):
    """Document frequency of sorted 2-token combination keys, per field."""
    import pyarrow.parquet as pq

    name_df: Counter = Counter()
    addr_df: Counter = Counter()
    n_docs = 0
    n_tok = []
    for batch in pq.ParquetFile(str(path)).iter_batches(
        batch_size=batch_size, columns=["name_norm", "address_norm"]
    ):
        d = batch.to_pandas()
        n_docs += len(d)
        for nm, ad in zip(d["name_norm"].fillna(""), d["address_norm"].fillna("")):
            nt = [t for t in nm.split() if len(t) >= min_len] if nm else []
            at = [t for t in ad.split() if len(t) >= min_len] if ad else []
            n_tok.append(len(nt) + len(at))
            if len(nt) >= 2:
                name_df.update(set(combinations(sorted(nt), 2)))
            if len(at) >= 2:
                addr_df.update(set(combinations(sorted(at), 2)))
        del d, batch
    return name_df, addr_df, n_docs, np.array(n_tok)


def main() -> None:
    g = guard()
    print(f"[guard] {g.status()}", flush=True)

    import pickle
    with open(CACHE / "_diag_sample.pkl", "rb") as f:
        d = pickle.load(f)
    s1, gt, found, sample = d["s1"], d["gt"], d["found"], d["sample"]
    total = sum(len(gt[s]) for s in sample)
    print(f"[sample] {len(sample):,} S1 / {total:,} GT pairs", flush=True)

    dfs = {}
    n_docs = {}
    for label, path in SOURCES.items():
        t0 = time.perf_counter()
        ndf, adf, nd, ntok = combo_df(path, MIN_LEN)
        dfs[label] = {"name": ndf, "addr": adf}
        n_docs[label] = nd
        print(f"  {label}: {nd:,} docs  name-combos={len(ndf):,}  "
              f"addr-combos={len(adf):,}  mean tokens/doc="
              f"{ntok.mean():.1f}  {time.perf_counter()-t0:.0f}s | {g.status()}",
              flush=True)

    def combos(tokens: str, min_len: int) -> set:
        t = [x for x in tokens.split() if len(x) >= min_len] if tokens else []
        return set(combinations(sorted(t), 2)) if len(t) >= 2 else set()

    # ---- recall per max_df for single vs conjunctive blocking ----
    min_name = np.full(total, 10**9, dtype=np.int64)
    min_addr = np.full(total, 10**9, dtype=np.int64)
    min_both = np.full(total, 10**9, dtype=np.int64)
    min_single = np.full(total, 10**9, dtype=np.int64)
    i = 0
    for s in sample:
        a_n, a_a = s1[s][0], s1[s][1]
        s1_nc, s1_ac = combos(a_n, MIN_LEN), combos(a_a, MIN_LEN)
        s1_nt = set(t for t in a_n.split() if len(t) >= MIN_LEN) if a_n else set()
        s1_at = set(t for t in a_a.split() if len(t) >= MIN_LEN) if a_a else set()
        for m in gt[s]:
            b = found.get(m)
            if b is None:
                i += 1
                continue
            m_in3 = m.startswith("S3-")
            srcs = ["S3"] if m_in3 else ["S2"]
            k = dfs[srcs[0]]["name"]
            ka = dfs[srcs[0]]["addr"]
            shared_n = s1_nc & combos(b[0], MIN_LEN)
            shared_a = s1_ac & combos(b[1], MIN_LEN)
            if shared_n:
                v = min(min(k.get(c, 10**9) for c in shared_n) for _ in [0])
                min_name[i] = v
                min_both[i] = min(min_both[i], v)
            if shared_a:
                v = min(ka.get(c, 10**9) for c in shared_a)
                min_addr[i] = v
                min_both[i] = min(min_both[i], v)
            sh_tok = (s1_nt & set(b[0].split())) | (s1_at & set(b[1].split()))
            sh_tok = {t for t in sh_tok if len(t) >= MIN_LEN}
            if sh_tok:
                src = dfs[srcs[0]]
                v = min([min(src["name"].get(("x", t), 10**9), 0) for t in []] or
                        [10**9])
                # single-token df approximated from combo table is not available;
                # use the measured rarity curve instead (see report)
            i += 1

    print(f"\n  conjunctive name-pair recall (any max_df): {np.mean(min_name<10**9):.4f}")
    print(f"  conjunctive addr-pair recall (any max_df): {np.mean(min_addr<10**9):.4f}")
    print(f"  conjunctive union recall     (any max_df): {np.mean(min_both<10**9):.4f}")

    print("\n  max_df   recall(name-pair)  recall(addr-pair)  recall(union)")
    curve = []
    for m in [10, 25, 50, 100, 250, 500, 1000, 2500, 5000]:
        rn = np.mean(min_name <= m)
        ra = np.mean(min_addr <= m)
        rb = np.mean(min_both <= m)
        curve.append({"max_df": m, "name": round(float(rn), 4),
                      "addr": round(float(ra), 4), "union": round(float(rb), 4)})
        print(f"  {m:>6,}   {rn:13.4f}  {ra:14.4f}  {rb:13.4f}", flush=True)

    # ---- expected volume per S1 ----
    print("\n  expected candidate rows/S1 (union of name+addr combo keys)")
    vol = []
    for m in [25, 50, 100, 250, 500, 1000, 2500, 5000]:
        per = []
        for s in sample[:1500]:
            tot = 0
            for label in ("S2", "S3"):
                for c in (combos(s1[s][0], MIN_LEN), combos(s1[s][1], MIN_LEN)):
                    for key in c:
                        v = min(dfs[label]["name"].get(key, 10**9),
                                dfs[label]["addr"].get(key, 10**9))
                        if v <= m:
                            tot += v
            per.append(tot)
        a = np.array(per, dtype=np.float64)
        vol.append({"max_df": m, "mean": round(float(a.mean()), 1),
                    "p50": int(np.percentile(a, 50)), "p95": int(np.percentile(a, 95)),
                    "p99": int(np.percentile(a, 99))})
        print(f"    max_df={m:>6,}  mean {a.mean():9,.0f}  p50 {np.percentile(a,50):8,.0f}"
              f"  p95 {np.percentile(a,95):9,.0f}  p99 {np.percentile(a,99):9,.0f}",
              flush=True)

    out = {"n_pairs": total, "n_docs": n_docs,
           "recall_name_pair_any": round(float(np.mean(min_name < 10**9)), 4),
           "recall_addr_pair_any": round(float(np.mean(min_addr < 10**9)), 4),
           "recall_union_any": round(float(np.mean(min_both < 10**9)), 4),
           "recall_curve": curve, "volume": vol}
    Path("D:/amazon_ml/reports/combo_blocking_curve.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\n[saved] D:/amazon_ml/reports/combo_blocking_curve.json")


if __name__ == "__main__":
    main()
