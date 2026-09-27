"""
Decisive measurement: UNION recall of the exact + rare-token channels.

Earlier sweeps measured each channel family separately. What matters is the
union, because retrieval keeps every channel. This measures, for a range of
per-S1 budgets, the recall of the union of:

    exact name + exact address + address digits  (capped)
    rare-token postings, rarest-first, up to the per-S1 budget

and the resulting on-disk volume, so an operating point can be chosen that
fits the disk budget (20% of D: must stay free -> 42.4 GB usable).

Recall is computed against ground-truth matches resolved to their row index in
the source, so the count is unambiguous.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.indexing import PostingIndex
from src.retrieval import RareTokenIndex, RetrievalBudget, retrieve_chunk
from src.utils import guard

CACHE = Path("D:/amazon_ml/cache")
SOURCES = [("S2", CACHE / "train_source2_norm.parquet", 2),
           ("S3", CACHE / "train_source3_norm.parquet", 3)]

# (per_s1, exact_cap, digit_cap, max_token_df)
BUDGETS = [
    (300, 100, 100, 500),
    (600, 150, 150, 1000),
    (1000, 200, 200, 1000),
    (1500, 250, 250, 2000),
    (2000, 300, 300, 2000),
    (3000, 400, 400, 5000),
]

N_TRAIN_S1 = 2_206_821
BYTES_PER_PAIR = 12  # int32 s1_row + int32 cand_row + uint8 bits + int8 source


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
    n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    print(f"[sample target] {n_sample:,} S1 | {g.status()}", flush=True)

    with open(CACHE / "_diag_sample.pkl", "rb") as f:
        d = pickle.load(f)
    s1_map, gt = d["s1"], d["gt"]
    sample = d["sample"][:n_sample]
    total = sum(len(gt[s]) for s in sample)
    print(f"[sample] {len(sample):,} S1 / {total:,} GT pairs", flush=True)

    s1_df = pd.DataFrame([
        {"name_norm": s1_map[s][0] or "", "address_norm": s1_map[s][1] or "",
         "address_dig": s1_map[s][2] or ""} for s in sample
    ])

    hits = {k: set() for k in BUDGETS}
    counts = {k: np.zeros(len(sample), dtype=np.int64) for k in BUDGETS}

    for label, path, sid in SOURCES:
        print(f"\n===== {label} ===== | {g.status()}", flush=True)
        ix_n = PostingIndex.from_column(path, "name_norm")
        ix_a = PostingIndex.from_column(path, "address_norm")
        ix_d = PostingIndex.from_column(path, "address_dig")
        print("  exact indexes ready", flush=True)

        want = set()
        for s in sample:
            for mid in gt[s]:
                if mid.startswith(label + "-"):
                    want.add(mid)
        row_of = locate_rows(path, want)
        print(f"  located {len(row_of):,}/{len(want):,} GT rows", flush=True)
        tgt = [{row_of[mid] for mid in gt[s] if mid in row_of} for s in sample]

        cache_ti: dict[int, list[RareTokenIndex]] = {}
        for per_s1, ecap, dcap, mdf in BUDGETS:
            if mdf not in cache_ti:
                cache_ti[mdf] = [
                    RareTokenIndex.build(path, "name_norm", mdf),
                    RareTokenIndex.build(path, "address_norm", mdf),
                ]
                print(f"  tokens max_df={mdf} ready | {g.status()}", flush=True)
            bud = RetrievalBudget(per_s1=per_s1, exact_cap=ecap, digit_cap=dcap,
                                  max_token_df=mdf)
            t0 = time.perf_counter()
            sr, cr, _b = retrieve_chunk(s1_df, ix_n, ix_a, ix_d,
                                        cache_ti[mdf], bud)
            dt = time.perf_counter() - t0
            cnt = (np.bincount(sr, minlength=len(sample)) if len(sr)
                   else np.zeros(len(sample), dtype=np.int64))
            k = (per_s1, ecap, dcap, mdf)
            counts[k] += cnt
            for i, r in zip(sr.tolist(), cr.tolist()):
                if r in tgt[i]:
                    hits[k].add((label, sample[i], r))
            print(f"  per_s1={per_s1:>5} ecap={ecap:>4} dcap={dcap:>4} "
                  f"maxdf={mdf:>5} -> {len(sr):>10,} pairs "
                  f"mean {cnt.mean():8.1f}/S1 max {cnt.max():>6} "
                  f"({dt:.0f}s) | {g.status()}", flush=True)
            del sr, cr, _b, cnt
            g.gc(1)

        del ix_n, ix_a, ix_d, cache_ti, row_of
        g.gc()
        print(f"  {label} freed | {g.status()}", flush=True)

    print("\n===== UNION RECALL vs BUDGET =====")
    print(f"{'per_s1':>7}{'maxdf':>7}{'recall':>9}{'mean':>10}{'p95':>9}"
          f"{'est pairs':>14}{'est GB':>9}{'fits 42GB':>10}")
    out = {"n_sample_s1": len(sample), "total_gt_pairs": total, "rows": {}}
    for per_s1, ecap, dcap, mdf in BUDGETS:
        c = counts[(per_s1, ecap, dcap, mdf)]
        rec = len(hits[(per_s1, ecap, dcap, mdf)]) / total
        est = c.mean() * N_TRAIN_S1
        gb = est * BYTES_PER_PAIR / 1024 ** 3
        key = f"per_s1={per_s1},exact={ecap},digit={dcap},max_df={mdf}"
        out["rows"][key] = {
            "recall": round(rec, 4), "mean": round(float(c.mean()), 1),
            "p95": int(np.percentile(c, 95)),
            "est_pairs": int(est), "est_gb": round(gb, 1),
        }
        print(f"{per_s1:>7}{mdf:>7}{rec:9.4f}{c.mean():10.1f}"
              f"{np.percentile(c,95):9.0f}{est/1e6:11,.0f}M{gb:8.1f}G"
              f"{('YES' if gb < 40 else 'no'):>10}")

    Path("D:/amazon_ml/reports/union_budget_recall.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\n[saved] D:/amazon_ml/reports/union_budget_recall.json")


if __name__ == "__main__":
    main()
