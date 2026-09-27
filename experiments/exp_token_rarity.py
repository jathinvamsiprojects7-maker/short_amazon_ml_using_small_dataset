"""
Experiment: token-rarity trade-off curve for candidate blocking.

Evidence so far (3000 S1 sample, 10,906 GT pairs):

    channel          recall    rows/S1 (raw, no dedup)
    exact_name       0.2241            10
    exact_address    0.0833             0
    address_digits   0.5930         3,350
    name_token       0.8565       736,514
    addr_token       0.9520       770,220
    union            0.9996     ~1.5M

So token channels give the recall we need but an unusable volume. The only
lever that matters is *token rarity*: how rare must a shared token be for the
channel to fire?

Method
------
1. Build the document-frequency table for S2 and S3 tokens (name + address).
2. For every GT pair, find the rarest shared token and its df.
3. Sweep max_df: recall(max_df) = fraction of GT pairs that share at least
   one token with df <= max_df.
4. Report the required max_df for high recall, plus expected volume per S1
   from the observed token-length distribution.

This is a pure recall/volume analysis - no candidate generation - so it is
cheap and decisive.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import guard

CACHE = Path("D:/amazon_ml/cache")
SOURCES = ["train_source2_norm.parquet", "train_source3_norm.parquet"]


def token_df(path: Path, columns=("name_tokens_str", "address_tokens_str"),
             min_len: int = 3, batch_size: int = 200_000) -> tuple[Counter, int, list[int]]:
    """Document frequency of every token across the given columns."""
    import pyarrow.parquet as pq

    df: Counter = Counter()
    n_docs = 0
    tok_lens: list[int] = []
    for batch in pq.ParquetFile(str(path)).iter_batches(
        batch_size=batch_size, columns=list(columns)
    ):
        d = batch.to_pandas()
        for col in columns:
            for toks in d[col].fillna(""):
                if not toks:
                    continue
                ts = [t for t in toks.split() if len(t) >= min_len]
                tok_lens.append(len(ts))
                df.update(set(ts))
        n_docs += len(d)
        del d, batch
    return df, n_docs, tok_lens


def main() -> None:
    g = guard()
    print(f"[guard] {g.status()}", flush=True)

    d = pickle_load(CACHE / "_diag_sample.pkl")
    s1 = d["s1"]          # {s1_id: (name_norm, addr_norm, addr_dig, country)}
    gt = d["gt"]
    found = d["found"]    # {match_id: (name_norm, addr_norm, addr_dig)}
    sample = d["sample"]
    print(f"[sample] {len(sample):,} S1, {sum(len(gt[s]) for s in sample):,} GT pairs",
          flush=True)

    # ---- token df for both sources (name + address tokens together) ----
    dfs = {}
    n_docs = {}
    for name in SOURCES:
        t0 = time.perf_counter()
        df, nd, lens = token_df(CACHE / name)
        dfs[name] = df
        n_docs[name] = nd
        print(f"  {name}: {nd:,} docs, {len(df):,} distinct tokens, "
              f"mean {len(lens)/max(len(lens),1):.1f} tokens/doc, "
              f"{time.perf_counter()-t0:.0f}s | {g.status()}", flush=True)

    # ---- rarest shared token per GT pair ----
    pairs = []
    for s in sample:
        a_name, a_addr = set(s1[s][0].split()), set(s1[s][1].split())
        for m in gt[s]:
            b = found.get(m)
            if b is None:
                continue
            shared_n = a_name & set(b[0].split())
            shared_a = a_addr & set(b[1].split())
            shared = shared_n | shared_a
            best = min(
                (min(dfs[n].get(t, 0) for n in SOURCES) for t in shared),
                default=None,
            )
            pairs.append({
                "n_shared": len(shared),
                "n_shared_name": len(shared_n),
                "n_shared_addr": len(shared_a),
                "min_df": best,
                "src_s3": m.startswith("S3-"),
            })
    print(f"  analysed {len(pairs):,} pairs with attributes", flush=True)

    n = len(pairs)
    have_shared = sum(1 for p in pairs if p["n_shared"] > 0)
    print(f"  pairs sharing >=1 token: {have_shared/n:.4f}", flush=True)
    print(f"  pairs sharing >=2 tokens: "
          f"{sum(1 for p in pairs if p['n_shared']>=2)/n:.4f}", flush=True)

    min_df = np.array([p["min_df"] if p["min_df"] is not None else 10**9
                       for p in pairs], dtype=np.int64)
    n_shared = np.array([p["n_shared"] for p in pairs])

    print("\n  max_df   recall(all)  recall(>=2 shared tokens)")
    curve = []
    for m in [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000]:
        hit_all = np.mean((min_df <= m) & (n_shared >= 1))
        hit2 = np.mean((min_df <= m) & (n_shared >= 2))
        curve.append({"max_df": m, "recall": round(float(hit_all), 4),
                      "recall_ge2": round(float(hit2), 4)})
        print(f"  {m:>7,}   {hit_all:9.4f}   {hit2:14.4f}", flush=True)

    # ---- expected volume ----
    print("\n  expected candidate rows/S1 (sum of posting lengths, no dedup)")
    for m in [50, 100, 250, 500, 1000, 2500, 5000]:
        per_s1 = []
        for s in sample[:800]:
            tot = 0
            for txt in (s1[s][0], s1[s][1]):
                for t in set(txt.split()):
                    if len(t) < 3:
                        continue
                    c = min(dfs[n_].get(t, 0) for n_ in SOURCES)
                    if c <= m:
                        tot += c
            per_s1.append(tot)
        a = np.array(per_s1, dtype=np.float64)
        print(f"    max_df={m:>6,}  mean {a.mean():12,.0f}  "
              f"p50 {np.percentile(a,50):10,.0f}  "
              f"p95 {np.percentile(a,95):12,.0f}  "
              f"p99 {np.percentile(a,99):12,.0f}", flush=True)

    out = {
        "n_pairs": n,
        "share_ge1": round(have_shared / n, 4),
        "share_ge2": round(sum(1 for p in pairs if p["n_shared"] >= 2) / n, 4),
        "recall_curve": curve,
        "n_docs": n_docs,
    }
    Path("D:/amazon_ml/reports/token_rarity_curve.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\n[saved] D:/amazon_ml/reports/token_rarity_curve.json")


def pickle_load(path: Path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    main()
