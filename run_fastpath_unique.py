"""
Measure the ONLY deterministic rule with a chance of high precision:
    unique name_exact AND address_exact

The previous measurement showed the raw exact channels are far too imprecise:

    name_exact   P=0.130  R=0.185
    addr_exact   P=0.828  R=0.082
    digit_exact  P=0.029  R=0.216

The reason is ambiguity: thousands of rows share one normalized name or one
address. A match is only trustworthy when the key is UNIQUE on both sides, i.e.
the S1's name maps to exactly one candidate and vice versa. This script
measures that rule, plus the near-miss variant (unique name AND
digit_exact), so the decision is evidence-based.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.indexing import PostingIndex, hash_str_array
from src.utils import guard

CACHE = Path("D:/amazon_ml/cache")


def main() -> None:
    g = guard()
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    split = sys.argv[2] if len(sys.argv) > 2 else "train"
    gt = json.loads((CACHE / "train_ground_truth.json").read_text(encoding="utf-8"))

    idx = {}
    row_of = {}
    for sid in (2, 3):
        src = CACHE / f"{split}_source{sid}_norm.parquet"
        ids = pq.read_table(str(src), columns=["entity_id"]
                            ).column("entity_id").to_pylist()
        row_of[sid] = {e: i for i, e in enumerate(ids)}
        idx[sid] = {"name": PostingIndex.from_column(src, "name_norm"),
                    "addr": PostingIndex.from_column(src, "address_norm"),
                    "dig": PostingIndex.from_column(src, "address_dig")}
        print(f"[eval] S{sid} indexes ready | {g.status()}", flush=True)

    rules = ("uniq_name", "uniq_name_addr", "uniq_addr", "uniq_name_dig")
    stat = {k: {"pred": 0, "tp": 0, "fp": 0, "truth": 0} for k in rules}

    pf = pq.ParquetFile(str(CACHE / f"{split}_source1_norm.parquet"))
    processed = 0
    t0 = time.perf_counter()
    for rb in pf.iter_batches(batch_size=200_000,
                              columns=["entity_id", "name_norm",
                                       "address_norm", "address_dig"]):
        eids = rb.column("entity_id").to_pylist()
        hn = hash_str_array(np.asarray(rb.column("name_norm").to_pylist(),
                                       dtype=object))
        ha = hash_str_array(np.asarray(rb.column("address_norm").to_pylist(),
                                       dtype=object))
        hd = hash_str_array(np.asarray(rb.column("address_dig").to_pylist(),
                                       dtype=object))
        for i, eid in enumerate(eids):
            if processed >= limit:
                break
            processed += 1
            tm = gt.get(eid)
            if not tm:
                continue
            tset = set()
            for mid in tm:
                s2 = 2 if mid.startswith("S2-") else 3
                rr = row_of[s2].get(mid)
                if rr is not None:
                    tset.add((s2, rr))
            for st in stat.values():
                st["truth"] += len(tset)
            if not tset:
                continue
            for sid in (2, 3):
                ix = idx[sid]
                # unique name: bucket size exactly 1 on the source side
                if hn[i]:
                    rows = ix["name"].get(int(hn[i]))
                    if len(rows) == 1:
                        k = (sid, int(rows[0]))
                        st = stat["uniq_name"]
                        st["pred"] += 1
                        if k in tset:
                            st["tp"] += 1
                        else:
                            st["fp"] += 1
                        if ha[i]:
                            ar = ix["addr"].get(int(ha[i]))
                            if len(ar) == 1 and int(ar[0]) == int(rows[0]):
                                st2 = stat["uniq_name_addr"]
                                st2["pred"] += 1
                                if k in tset:
                                    st2["tp"] += 1
                                else:
                                    st2["fp"] += 1
                        if hd[i]:
                            dr = ix["dig"].get(int(hd[i]))
                            if len(dr) == 1 and int(dr[0]) == int(rows[0]):
                                st3 = stat["uniq_name_dig"]
                                st3["pred"] += 1
                                if k in tset:
                                    st3["tp"] += 1
                                else:
                                    st3["fp"] += 1
                if ha[i]:
                    ar = ix["addr"].get(int(ha[i]))
                    if len(ar) == 1:
                        st = stat["uniq_addr"]
                        st["pred"] += 1
                        if (sid, int(ar[0])) in tset:
                            st["tp"] += 1
                        else:
                            st["fp"] += 1
        del rb
        g.gc(1)
        if processed % 200_000 == 0:
            print(f"  {processed:,} S1  {time.perf_counter() - t0:.0f}s | "
                  f"{g.status()}", flush=True)
        if processed >= limit:
            break

    out = {}
    for k, v in stat.items():
        prec = v["tp"] / v["pred"] if v["pred"] else 0.0
        rec = v["tp"] / v["truth"] if v["truth"] else 0.0
        out[k] = {"pred": v["pred"], "tp": v["tp"], "fp": v["fp"],
                  "truth": v["truth"], "precision": round(prec, 5),
                  "recall": round(rec, 5)}
        print(f"  {k:<16} P={prec:.5f} R={rec:.5f} pred={v['pred']:,} "
              f"tp={v['tp']:,} fp={v['fp']:,}", flush=True)
    (CACHE.parent / "reports" / "fastpath_unique_eval.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("[saved] reports/fastpath_unique_eval.json")


if __name__ == "__main__":
    main()
