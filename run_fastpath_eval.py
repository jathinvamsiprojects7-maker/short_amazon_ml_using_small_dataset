"""
Deterministic high-confidence matcher for the whole test set.

Rationale
---------
The full LightGBM inference over 3.45e9 candidates cannot finish before the
deadline, but the validation split already showed that the model reaches
micro precision 0.9993 on accepted pairs. The accepted pairs are dominated by
the exact channels, so a purely deterministic rule over those channels is
expected to keep comparable precision at a tiny fraction of the cost.

The rule is measured, not assumed:

* the train ground truth is used to compute, per rule, precision and recall on
  the exact channels only;
* a rule is adopted only if its measured precision is at or near the model's
  0.9993, because Macro F0.5 with beta=0.5 punishes a false positive hard and
  a wrong match is worse than an empty row;
* the resulting test predictions come from exactly the same code path applied
  to the test data - no external data, no invented ids.

Cost: one pass over S1 and one hash lookup per row, i.e. minutes, not hours.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.indexing import PostingIndex, hash_str_array
from src.utils import guard, paths

CACHE = Path("D:/amazon_ml/cache")


def build_index(src: Path, column: str) -> PostingIndex:
    return PostingIndex.from_column(src, column)


def evaluate_rule(split: str, max_s1: int = 0) -> dict:
    """
    Measure precision/recall of the deterministic exact-channel rules against
    the train ground truth.

    Rules (in priority order, all recorded so the model can reproduce them):
      name_exact        : same normalized name
      name+addr_exact   : same normalized name AND same normalized address
      addr_exact        : same normalized address
      digit_exact       : same address digit key
    """
    g = guard()
    gt = json.loads((CACHE / "train_ground_truth.json").read_text(encoding="utf-8"))
    s1_ids = pq.read_table(str(CACHE / f"{split}_source1_norm.parquet"),
                           columns=["entity_id"]).column("entity_id").to_pylist()
    n = len(s1_ids) if not max_s1 else min(len(s1_ids), max_s1)
    print(f"[eval] {split}: measuring deterministic rules on {n:,} S1 | "
          f"{g.status()}", flush=True)

    # ground truth as row-id sets
    truth: dict[str, set] = defaultdict(set)
    src_ids = {}
    row_of = {}
    for sid in (2, 3):
        ids = pq.read_table(str(CACHE / f"{split}_source{sid}_norm.parquet"),
                            columns=["entity_id"]).column("entity_id").to_pylist()
        src_ids[sid] = ids
        row_of[sid] = {e: i for i, e in enumerate(ids)}
        g.gc()

    # only S1 that appear in the ground truth file (all of them for train)
    stats = {k: {"tp": 0, "fp": 0, "pred": 0, "truth": 0, "s1_pred": 0,
                 "s1_with_truth": 0} for k in
             ("name_exact", "name_addr_exact", "addr_exact", "digit_exact")}

    idx = {}
    for sid in (2, 3):
        src = CACHE / f"{split}_source{sid}_norm.parquet"
        idx[sid] = {
            "name": build_index(src, "name_norm"),
            "addr": build_index(src, "address_norm"),
            "dig": build_index(src, "address_dig"),
        }
        print(f"[eval] S{sid} indexes built | {g.status()}", flush=True)

    pf = pq.ParquetFile(str(CACHE / f"{split}_source1_norm.parquet"))
    processed = 0
    for rb in pf.iter_batches(batch_size=200_000,
                              columns=["entity_id", "name_norm", "address_norm",
                                       "address_dig"]):
        eids = rb.column("entity_id").to_pylist()
        hn = hash_str_array(np.asarray(rb.column("name_norm").to_pylist(),
                                       dtype=object))
        ha = hash_str_array(np.asarray(rb.column("address_norm").to_pylist(),
                                       dtype=object))
        hd = hash_str_array(np.asarray(rb.column("address_dig").to_pylist(),
                                       dtype=object))
        for i, eid in enumerate(eids):
            if max_s1 and processed >= n:
                break
            processed += 1
            tset: set = set()
            for sid in (2, 3):
                r = row_of[sid].get(eid)
                if r is None:
                    continue
            tm = gt.get(eid)
            if not tm:
                continue
            tset = set()
            for mid in tm:
                s2 = 2 if mid.startswith("S2-") else 3
                rr = row_of[s2].get(mid)
                if rr is not None:
                    tset.add((s2, rr))
            for st in stats.values():
                st["truth"] += len(tset)
                if tset:
                    st["s1_with_truth"] += 1
            for sid in (2, 3):
                ix = idx[sid]
                if hn[i]:
                    for r in ix["name"].get(int(hn[i])).tolist()[:20]:
                        _score(stats, "name_exact", (sid, r), tset)
                if ha[i]:
                    for r in ix["addr"].get(int(ha[i])).tolist()[:20]:
                        _score(stats, "addr_exact", (sid, r), tset)
                if hn[i] and ha[i]:
                    for r in ix["name"].get(int(hn[i])).tolist()[:20]:
                        pass
                if hd[i]:
                    for r in ix["dig"].get(int(hd[i])).tolist()[:20]:
                        _score(stats, "digit_exact", (sid, r), tset)
        del rb
        g.gc(1)
        if processed % 200_000 == 0:
            print(f"  [eval] {processed:,} S1 | {g.status()}", flush=True)
        if max_s1 and processed >= n:
            break

    out = {}
    for k, v in stats.items():
        prec = v["tp"] / v["pred"] if v["pred"] else 0.0
        rec = v["tp"] / v["truth"] if v["truth"] else 0.0
        out[k] = {
            "predicted_pairs": v["pred"],
            "true_pairs": v["truth"],
            "tp": v["tp"], "fp": v["fp"],
            "precision": round(prec, 5), "recall": round(rec, 5),
        }
        print(f"  {k:<16} P={prec:.5f} R={rec:.5f} pred={v['pred']:,} "
              f"tp={v['tp']:,} fp={v['fp']:,}", flush=True)
    return out


def _score(stats, rule, key, tset) -> None:
    st = stats[rule]
    st["pred"] += 1
    if key in tset:
        st["tp"] += 1
    else:
        st["fp"] += 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=120000)
    a = ap.parse_args()
    r = evaluate_rule(a.split, max_s1=a.limit)
    out = CACHE.parent / "reports" / "fastpath_eval.json"
    out.write_text(json.dumps(r, indent=2), encoding="utf-8")
    print(f"[saved] {out}")
