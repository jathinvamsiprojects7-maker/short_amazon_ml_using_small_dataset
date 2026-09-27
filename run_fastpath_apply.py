"""
Deterministic high-confidence matcher for the whole test set.

Evidence (reports/fastpath_unique_eval.json, 100k train S1, ground truth):

    rule             precision   recall   predicted
    uniq_name          0.760     0.098      44,437
    uniq_name_addr     1.000     0.003       1,174
    uniq_addr          0.924     0.056      20,871
    uniq_name_dig      1.000     0.003       1,111

Only the two 1.000-precision rules are safe. Macro F0.5 with beta=0.5 weights
precision above recall, so a rule at precision 1.000 can only help, while a rule
at 0.76 (uniq_name) would inject ~10,600 false positives per 100k S1 and is
rejected on measurement.

The accepted rule is the UNION of the two perfect rules, i.e. accept
candidate c for S1 s when the normalized name is unique on the source side AND
either the normalized address or the digit key also matches c uniquely. It uses
only locally derived indexes, no external data, and the same code path is what
produces the train measurement above.

Output: per-S1 accepted id lists, written as shards so assembly can merge them
with the completed LightGBM shards.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.indexing import PostingIndex, hash_str_array
from src.utils import guard, paths

CACHE = Path("D:/amazon_ml/cache")


def main() -> None:
    g = guard()
    p = paths()
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    out_name = sys.argv[2] if len(sys.argv) > 2 else f"fast_{split}"

    idx = {}
    for sid in (2, 3):
        src = CACHE / f"{split}_source{sid}_norm.parquet"
        idx[sid] = {
            "name": PostingIndex.from_column(src, "name_norm"),
            "addr": PostingIndex.from_column(src, "address_norm"),
            "dig": PostingIndex.from_column(src, "address_dig"),
        }
        print(f"[fast] S{sid} indexes ready | {g.status()}", flush=True)

    out_dir = p["output_dir"] / "_parts"
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = open(out_dir / f"{out_name}.txt", "w", encoding="utf-8")

    pf = pq.ParquetFile(str(CACHE / f"{split}_source1_norm.parquet"))
    n_rg = pf.metadata.num_row_groups
    n_pred = 0
    t0 = time.perf_counter()
    base = 0
    for gi in range(n_rg):
        rb = pf.read_row_group(gi, columns=["s1_row", "cand_row"]) \
            if "s1_row" in pf.schema_arrow.names else \
            pf.read_row_group(gi, columns=["name_norm", "address_norm",
                                           "address_dig"])
        if "s1_row" in rb.schema.names:
            s1_rows = rb.column("s1_row").to_numpy()
        else:
            s1_rows = np.arange(base, base + len(rb), dtype=np.int64)
        hn = hash_str_array(np.asarray(rb.column("name_norm").to_pylist(),
                                       dtype=object))
        ha = hash_str_array(np.asarray(rb.column("address_norm").to_pylist(),
                                       dtype=object))
        hd = hash_str_array(np.asarray(rb.column("address_dig").to_pylist(),
                                       dtype=object))
        for i, r in enumerate(s1_rows.tolist()):
            if not hn[i]:
                continue
            for sid in (2, 3):
                ix = idx[sid]
                rows = ix["name"].get(int(hn[i]))
                if len(rows) != 1:
                    continue
                c = int(rows[0])
                ok = False
                if ha[i]:
                    ar = ix["addr"].get(int(ha[i]))
                    if len(ar) == 1 and int(ar[0]) == c:
                        ok = True
                if not ok and hd[i]:
                    dr = ix["dig"].get(int(hd[i]))
                    if len(dr) == 1 and int(dr[0]) == c:
                        ok = True
                if ok:
                    fp.write(f"{r}\t{sid}\t{c}\n")
                    n_pred += 1
        base += len(s1_rows)
        del rb
        g.gc(1)
        if gi % 200 == 0 or gi == n_rg - 1:
            el = time.perf_counter() - t0
            print(f"  group {gi + 1}/{n_rg}  s1 {base:,}  preds {n_pred:,}  "
                  f"{el / 60:.1f}m | {g.status()}", flush=True)
    fp.close()
    print(f"[fast] {n_pred:,} deterministic predictions over {base:,} S1 in "
          f"{(time.perf_counter() - t0) / 60:.1f}m", flush=True)
    print(f"[fast] wrote {out_dir / (out_name + '.txt')}", flush=True)


if __name__ == "__main__":
    main()
