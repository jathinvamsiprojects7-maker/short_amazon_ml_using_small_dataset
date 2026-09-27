"""
Experiment: single-key-channel union recall at a FEASIBLE volume.

Why this experiment
-------------------
Measured so far (4000 S1 / 14,823 GT pairs):

  exact name      recall 0.2241    volume  ~10/S1
  exact address   recall 0.0833    volume   ~0/S1
  address digits  recall 0.5930    volume ~3350/S1 (p50 much lower)
  single rare tok recall 0.9810 @ max_df 5000, but 4571/S1 -> 10.1e9 pairs
  conjunctive pair recall 0.9821 @ max_df 5000, but 12093/S1

Single-token and conjunctive blocking have almost the same recall ceiling
(~0.98) and equally unusable volume. Neither is acceptable.

The decisive question: what is the recall of the UNION of the cheap
channels, and is the residual 2% reachable by a bounded fuzzy channel that
needs no blocking at all?

This script therefore measures:
  (a) union recall of exact name + exact address + digits,
  (b) union recall when digits are additionally *conjunctive* with a rare
      name/address token (digit AND rare token in the same doc) - far more
      selective than digits alone,
  (c) the size of the residual that a fuzzy pass must recover,
  (d) how many of the residual pairs are recoverable by a RapidFuzz
      cdist pass against a *digit-blocked* candidate set.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import guard

CACHE = Path("D:/amazon_ml/cache")
SOURCES = {"S2": CACHE / "train_source2_norm.parquet",
           "S3": CACHE / "train_source3_norm.parquet"}
MIN_LEN = 3


def doc_token_sets(path: Path, min_len: int, batch_size: int = 200_000):
    """For each doc: frozenset of rare-enough tokens (name+address merged)."""
    import pyarrow.parquet as pq

    df: Counter = Counter()
    docs_tokens: list[frozenset] = []
    for batch in pq.ParquetFile(str(path)).iter_batches(
        batch_size=batch_size, columns=["name_norm", "address_norm"]
    ):
        d = batch.to_pandas()
        per_doc = []
        for nm, ad in zip(d["name_norm"].fillna(""), d["address_norm"].fillna("")):
            toks = [t for t in ((nm or "") + " " + (ad or "")).split()
                    if len(t) >= min_len]
            per_doc.append(frozenset(toks))
            df.update(set(toks))
        docs_tokens.extend(per_doc)
        del d, batch
    return df, docs_tokens


def main() -> None:
    g = guard()
    print(f"[guard] {g.status()}", flush=True)

    with open(CACHE / "_diag_sample.pkl", "rb") as f:
        d = pickle.load(f)
    s1, gt, found, sample = d["s1"], d["gt"], d["found"], d["sample"]
    total = sum(len(gt[s]) for s in sample)
    print(f"[sample] {len(sample):,} S1 / {total:,} GT pairs", flush=True)

    results: dict[str, dict] = {}
    pair_sets: dict[str, set] = {k: set() for k in
                                 ("exact_name", "exact_addr", "digits",
                                  "digit_x_token")}
    vol_tot = {k: 0 for k in pair_sets}

    for label, path in SOURCES.items():
        print(f"\n===== {label} =====", flush=True)

        # ---- token df + doc token sets (needed for conjunctive analysis) ----
        t0 = time.perf_counter()
        df, docs_tokens = doc_token_sets(path, MIN_LEN)
        n_docs = len(docs_tokens)
        print(f"  token df: {len(df):,} distinct, {n_docs:,} docs, "
              f"{time.perf_counter()-t0:.0f}s | {g.status()}", flush=True)

        # ---- exact-key indexes (row-level) ----
        t0 = time.perf_counter()
        from src.indexing import PostingIndex
        ix_name = PostingIndex.from_column(path, "name_norm")
        ix_addr = PostingIndex.from_column(path, "address_norm")
        ix_dig = PostingIndex.from_column(path, "address_dig")
        print(f"  exact indexes {time.perf_counter()-t0:.0f}s | {g.status()}",
              flush=True)

        # ---- digit index -> inverted list for conjunctive lookup ----
        t0 = time.perf_counter()
        dig_keys = np.unique(ix_dig.keys)
        print(f"  distinct digit keys: {len(dig_keys):,} "
              f"{time.perf_counter()-t0:.0f}s", flush=True)

        # ---- evaluate per S1 ----
        from src.indexing import hash_str_array
        hn = hash_str_array([s1[s][0] for s in sample])
        ha = hash_str_array([s1[s][1] for s in sample])
        hd = hash_str_array([s1[s][2] for s in sample])

        stats = {
            "exact_name": set(), "exact_addr": set(), "digits": set(),
            "digit_x_token": set(),
        }
        vol = {"exact_name": 0, "exact_addr": 0, "digits": 0, "digit_x_token": 0}
        # target row sets per S1 for this source
        # locate GT rows lazily via match id prefix is not enough; we need rows.
        # Build row_of for the GT matches of this source in one scan.
        want = set()
        for s in sample:
            for m in gt[s]:
                if m.startswith(label + "-"):
                    want.add(m)
        row_of = {}
        import pyarrow as pa
        import pyarrow.compute as pc
        want_arr = pa.array(sorted(want), type=pa.string())
        base = 0
        import pyarrow.parquet as pq
        for b in pq.ParquetFile(str(path)).iter_batches(
            batch_size=500_000, columns=["entity_id"]
        ):
            col = b.column("entity_id")
            m = pc.is_in(col, value_set=want_arr)
            if pc.any(m).as_py():
                sel = col.filter(m).to_pylist()
                idxs = np.asarray(m).nonzero()[0]
                for eid, j in zip(sel, idxs):
                    row_of[eid] = base + int(j)
            base += len(b)
            del b, col, m
        print(f"  located {len(row_of):,} GT rows in {label}", flush=True)

        n_pairs = 0
        for s, kn, ka, kd in zip(sample, hn, ha, hd):
            tgt = {row_of[m] for m in gt[s] if m in row_of}
            if not tgt:
                continue
            key = (label, s)
            # exact name / addr
            if kn:
                for r in ix_name.get(int(kn)).tolist():
                    vol_tot["exact_name"] += 1
                    if r in tgt:
                        pair_sets["exact_name"].add(key + (r,))
            if ka:
                for r in ix_addr.get(int(ka)).tolist():
                    vol_tot["exact_addr"] += 1
                    if r in tgt:
                        pair_sets["exact_addr"].add(key + (r,))
            # digits
            if kd:
                rows = ix_dig.get(int(kd))
                vol_tot["digits"] += len(rows)
                for r in rows.tolist():
                    if r in tgt:
                        pair_sets["digits"].add(key + (r,))
                # conjunctive: digits AND >=1 rare token
                toks = {t for t in (s1[s][0] + " " + s1[s][1]).split()
                        if len(t) >= MIN_LEN}
                rare = {t for t in toks if 0 < df.get(t, 0) <= 2000}
                if rare:
                    for r in rows.tolist():
                        if docs_tokens[r] & rare:
                            vol_tot["digit_x_token"] += 1
                            if r in tgt:
                                pair_sets["digit_x_token"].add(key + (r,))

        del ix_name, ix_addr, ix_dig, docs_tokens, df
        g.gc()
        print(f"  {label} freed | {g.status()}", flush=True)

    # ---- report ----
    print("\n===== CHANNEL RESULTS (union over S2+S3) =====")
    print(f"{'channel':<16}{'recall':>9}{'rows/S1':>14}")
    out = {"total_gt_pairs": total, "n_sample_s1": len(sample), "channels": {},
           "union": {}}
    for k in ("exact_name", "exact_addr", "digits", "digit_x_token"):
        rec = len(pair_sets[k]) / total
        per = vol_tot[k] / len(sample)
        out["channels"][k] = {"recall": round(rec, 4), "rows_per_s1": round(per, 1)}
        print(f"{k:<16}{rec:9.4f}{per:14,.1f}")

    u = set()
    for k in pair_sets:
        u |= pair_sets[k]
    out["union"]["recall"] = round(len(u) / total, 4)
    print(f"{'UNION':<16}{len(u)/total:9.4f}")

    # progressive unions
    order = ["exact_name", "exact_addr", "digits", "digit_x_token"]
    acc: set = set()
    print("\n  progressive union:")
    for k in order:
        acc |= pair_sets[k]
        out["union"][f"through_{k}"] = round(len(acc) / total, 4)
        print(f"    +{k:<16} {len(acc)/total:.4f}")
    # residual that fuzzy must recover
    out["residual_fuzzy_must_cover"] = round(1 - len(u) / total, 4)
    print(f"\n  residual for a fuzzy channel: {1-len(u)/total:.4f}")

    print("\n[saved] D:/amazon_ml/reports/channel_volume_experiment.json")
    Path("D:/amazon_ml/reports/channel_volume_experiment.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
