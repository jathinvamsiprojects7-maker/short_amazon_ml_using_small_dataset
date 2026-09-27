"""
Choose the retrieval operating point: candidate recall vs per-S1 budget,
measured on real ground truth with the D-003 rare-token retrieval.

Recall is computed against the *actual entity ids* of ground-truth matches
mapped to their row indices in the source, so there is no ambiguity about what
is being counted.
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

# (per_s1, digit_cap, max_token_df)
BUDGETS = [
    (60, 60, 500),
    (120, 120, 500),
    (200, 200, 1000),
    (300, 200, 1000),
    (300, 200, 5000),
    (500, 300, 1000),
    (500, 300, 5000),
]


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
    n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    with open(CACHE / "_diag_sample.pkl", "rb") as f:
        d = pickle.load(f)
    s1_map, gt = d["s1"], d["gt"]
    sample = d["sample"][:n_sample]
    total = sum(len(gt[s]) for s in sample)
    print(f"[sample] {len(sample):,} S1 / {total:,} GT pairs | {g.status()}",
          flush=True)

    rows = [{"name_norm": s1_map[s][0] or "", "address_norm": s1_map[s][1] or "",
             "address_dig": s1_map[s][2] or ""} for s in sample]
    s1_df = pd.DataFrame(rows)

    keys = BUDGETS
    hits = {k: set() for k in keys}
    counts = {k: np.zeros(len(sample), dtype=np.int64) for k in keys}
    retrieved = {k: set() for k in keys}

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

        tgt_rows = [{row_of[mid] for mid in gt[s] if mid in row_of} for s in sample]

        # token indexes are shared across budgets with the same max_df
        built: dict[int, list[RareTokenIndex]] = {}
        for per_s1, dcap, mdf in BUDGETS:
            if mdf not in built:
                built[mdf] = [
                    RareTokenIndex.build(path, "name_norm", mdf),
                    RareTokenIndex.build(path, "address_norm", mdf),
                ]
                print(f"  token indexes max_df={mdf} ready | {g.status()}",
                      flush=True)

            bud = RetrievalBudget(per_s1=per_s1, digit_cap=dcap,
                                  max_token_df=mdf)
            sr, cr, bits = retrieve_chunk(s1_df, ix_n, ix_a, ix_d,
                                          built[mdf], bud)
            cnt = (np.bincount(sr, minlength=len(sample)) if len(sr)
                   else np.zeros(len(sample), dtype=np.int64))
            counts[(per_s1, dcap, mdf)] += cnt
            kk = (per_s1, dcap, mdf)
            for i, r in zip(sr.tolist(), cr.tolist()):
                if r in tgt_rows[i]:
                    hits[kk].add((label, sample[i], r))
            print(f"  per_s1={per_s1:>4} dcap={dcap:>4} max_df={mdf:>5} -> "
                  f"{len(sr):>10,} pairs  mean {cnt.mean():7.1f}/S1  "
                  f"max {cnt.max():>6} | {g.status()}", flush=True)
            del sr, cr, bits, cnt
            g.gc(1)

        del ix_n, ix_a, ix_d, built
        g.gc()
        print(f"  {label} freed | {g.status()}", flush=True)

    print("\n===== RECALL vs BUDGET (D-003 rare-token retrieval) =====")
    print(f"{'per_s1':>7}{'digcap':>8}{'maxdf':>7}{'recall':>9}{'mean':>9}"
          f"{'p50':>7}{'p95':>8}{'est full pairs':>16}")
    out = {"n_sample_s1": len(sample), "total_gt_pairs": total, "rows": {}}
    for per_s1, dcap, mdf in BUDGETS:
        c = counts[(per_s1, dcap, mdf)]
        rec = len(hits[(per_s1, dcap, mdf)]) / total
        est = c.mean() * 2_206_821
        name = f"per_s1={per_s1},digit_cap={dcap},max_df={mdf}"
        out["rows"][name] = {
            "recall": round(rec, 4), "mean": round(float(c.mean()), 1),
            "p50": int(np.percentile(c, 50)), "p95": int(np.percentile(c, 95)),
            "p99": int(np.percentile(c, 99)),
            "est_full_train_pairs": int(est),
        }
        print(f"{per_s1:>7}{dcap:>8}{mdf:>7}{rec:9.4f}{c.mean():9.1f}"
              f"{np.percentile(c,50):7.0f}{np.percentile(c,95):8.0f}"
              f"{est/1e6:13,.0f}M")

    Path("D:/amazon_ml/reports/budget_recall_curve.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\n[saved] D:/amazon_ml/reports/budget_recall_curve.json")


if __name__ == "__main__":
    main()
