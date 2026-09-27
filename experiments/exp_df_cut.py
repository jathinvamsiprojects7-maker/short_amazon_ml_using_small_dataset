"""
The decisive recall/volume question for this dataset.

Facts established so far:
  * exact name/addr/digits union            -> recall 0.68, ~1.7k rows/S1
  * rare tokens, UNBOUNDED                   -> 753,203 rows/S1 mean
      (a handful of tokens dominate: 'limited' df=517k, 'private' 618k,
       'llc' 559k, 'ltd' 421k, 'inc' 411k on S3 name tokens)
  * budgeted at 300 rows/S1                  -> recall ~0.67

So the budget is the binding constraint and even an unbounded token union
cannot be enumerated. The question that decides the architecture is:

    if we EXCLUDE the dominant generic tokens (a small stop-list derived
    only from measured df, not hard-coded intuition), what recall/volume do
    the remaining rare tokens give?

This script measures recall and volume for a df-threshold sweep with the
generic tokens excluded, which is the only remaining lever that can raise
recall within the ~700M-pair budget.

It also measures the fallback problem: for S1 whose tokens are ALL generic,
how many candidates remain reachable at all (these are the "no candidate"
S1s that the model must be able to reject).
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import guard

CACHE = Path("D:/amazon_ml/cache")
SRC = [("S2", CACHE / "train_source2_norm.parquet"),
       ("S3", CACHE / "train_source3_norm.parquet")]
MIN_LEN = 3


def build_df(path: Path, min_len: int) -> Counter:
    df: Counter = Counter()
    for b in pq.ParquetFile(str(path)).iter_batches(
        batch_size=500_000, columns=["name_norm", "address_norm"]
    ):
        for nm, ad in zip(b.column("name_norm").to_pylist(),
                          b.column("address_norm").to_pylist()):
            toks = [t for t in ((nm or "") + " " + (ad or "")).split()
                    if len(t) >= min_len]
            if toks:
                df.update(set(toks))
        del b
    return df


def main() -> None:
    g = guard()
    n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    with open(CACHE / "_diag_sample.pkl", "rb") as f:
        d = pickle.load(f)
    s1_map, gt, found = d["s1"], d["gt"], d["found"]
    sample = d["sample"][:n_sample]
    total = sum(len(gt[s]) for s in sample)
    print(f"[sample] {len(sample):,} S1 / {total:,} GT pairs | {g.status()}",
          flush=True)

    dfs: dict[str, Counter] = {}
    for label, path in SRC:
        dfs[label] = build_df(path, MIN_LEN)
        print(f"  {label}: vocab {len(dfs[label]):,} | {g.status()}", flush=True)

    # minimum df across the two sources = a token is "generic" if it is common
    # in EITHER source (either one can produce a huge bucket)
    min_df: dict[str, int] = {}
    for t in dfs["S2"]:
        a = dfs["S2"][t]
        b = dfs["S3"].get(t, 0)
        min_df[t] = max(a, b)

    vals = np.array(sorted(min_df.values()))
    print(f"\n  combined df: p50={np.percentile(vals,50):.0f} "
          f"p90={np.percentile(vals,90):.0f} p99={np.percentile(vals,99):.0f} "
          f"max={vals.max()}")
    top = sorted(min_df.items(), key=lambda kv: -kv[1])[:15]
    print("  most generic tokens:", [(t, c) for t, c in top])

    # ---- recall / volume as a function of the df cut ----
    print("\n  max_df   recall(all pairs)  recall(|S1| tokens>=2 rare)  "
          "expected rows/S1")
    rows_out = []
    for mdf in [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 5000]:
        hit = 0
        hit2 = 0
        s1_with_rare = 0
        per_s1 = []
        for s in sample:
            a = [t for t in (s1_map[s][0] + " " + s1_map[s][1]).split()
                 if len(t) >= MIN_LEN]
            rare = [t for t in a if min_df.get(t, 0) <= mdf]
            if rare:
                s1_with_rare += 1
            # expected candidates = sum of df of the S1's rare tokens
            per_s1.append(sum(min_df.get(t, 0) for t in rare))
            for m in gt[s]:
                b = set(t for t in (found[m][0] + " " + found[m][1]).split()
                        if len(t) >= MIN_LEN)
                shared_rare = {t for t in a if t in b and min_df.get(t, 0) <= mdf}
                if shared_rare:
                    hit += 1
                if rare and shared_rare:
                    hit2 += 1
        a2 = np.array(per_s1)
        r = hit / total
        print(f"  {mdf:>6}   {r:15.4f}  {hit2/total:24.4f}  "
              f"{a2.mean():10,.0f} (p50 {np.percentile(a2,50):,.0f})",
              flush=True)
        rows_out.append({
            "max_df": mdf, "recall": round(r, 4),
            "pct_s1_with_rare_token": round(s1_with_rare / len(sample), 4),
            "expected_rows_per_s1_mean": round(float(a2.mean()), 1),
            "expected_rows_per_s1_p50": int(np.percentile(a2, 50)),
        })

    out = {
        "n_sample_s1": len(sample), "total_gt_pairs": total,
        "most_generic": [{"token": t, "max_df": c} for t, c in top],
        "sweep": rows_out,
    }
    Path("D:/amazon_ml/reports/df_cut_sweep.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\n[saved] D:/amazon_ml/reports/df_cut_sweep.json")


if __name__ == "__main__":
    main()
