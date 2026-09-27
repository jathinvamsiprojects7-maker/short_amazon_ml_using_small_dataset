"""
Controlled end-to-end validation on ~30 deliberately chosen training S1.

Purpose
-------
Prove the full pipeline (retrieve -> features -> score -> threshold -> output)
works and produces correct answers on cases whose ground truth is known,
before committing more time to full-scale runs.

Selection is DELIBERATE, not random, so each cardinality/regime is covered:

  * ``single``      exactly one true match
  * ``multi``       3+ true matches
  * ``variation``   the match differs in name spelling or address word order
  * ``nomatch``     zero true matches (the segment Macro F0.5 punishes hardest)
  * ``hardneg``     the S1 has a high-similarity non-match nearby

Everything is restricted to the selected S1 rows, so the run touches only a few
thousand candidate pairs and never scans the multi-GB files end to end.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import guard, paths

CACHE = Path("D:/amazon_ml/cache")
SHIFT = np.int64(1) << np.int64(32)


def f05(p: float, r: float) -> float:
    d = 0.25 * p + r
    return 0.0 if d == 0 else 1.25 * p * r / d


def _hash_str(s: str) -> int:
    return int(pd.util.hash_array(np.array([s], dtype=object),
                                  encoding="utf8")[0])


_IDX: dict = {}
_SRC_IDS: dict = {}


def _posting_index(sid: int):
    """Exact normalized-name index for a source, built once and cached."""
    if sid in _IDX:
        return _IDX[sid]
    from src.indexing import PostingIndex
    try:
        _IDX[sid] = PostingIndex.from_column(
            CACHE / f"train_source{sid}_norm.parquet", "name_norm")
    except Exception:
        _IDX[sid] = None
    return _IDX[sid]


def src_ids(sid: int) -> list:
    if sid not in _SRC_IDS:
        _SRC_IDS[sid] = pq.read_table(
            str(CACHE / f"train_source{sid}_norm.parquet"),
            columns=["entity_id"]).column("entity_id").to_pylist()
    return _SRC_IDS[sid]


def pick_cases(n_each: int = 6) -> list[dict]:
    """Choose a small, representative set of S1 entities."""
    g = guard()
    gt = json.loads((CACHE / "train_ground_truth.json").read_text(encoding="utf-8"))
    s1 = pq.read_table(str(CACHE / "train_source1_norm.parquet"),
                       columns=["entity_id", "name_norm", "address_norm"]
                       ).to_pandas().set_index("entity_id")
    src = {}
    for sid in (2, 3):
        cols = ["entity_id", "name_norm", "address_norm"]
        src[sid] = pq.read_table(
            str(CACHE / f"train_source{sid}_norm.parquet"),
            columns=cols).to_pandas().set_index("entity_id")
    g.gc()
    print(f"[load] s1={len(s1):,} src2={len(src[2]):,} src3={len(src[3]):,} | "
          f"{g.status()}", flush=True)

    rng = np.random.default_rng(20260927)
    single, multi, variation, nomatch, hardneg = [], [], [], [], []

    keys = np.array(list(gt.keys()), dtype=object)
    order = rng.permutation(len(keys))

    for i in order:
        s = str(keys[i])
        m = gt.get(s) or []
        if len(m) == 0:
            if len(nomatch) < n_each:
                nomatch.append({"s1": s, "kind": "nomatch", "gt": []})
            continue
        if len(m) == 1 and len(single) < n_each:
            single.append({"s1": s, "kind": "single", "gt": m})
        elif len(m) >= 3 and len(multi) < n_each:
            multi.append({"s1": s, "kind": "multi", "gt": m})

        # name/address variation: true match exists but is not string-equal
        if len(variation) < n_each and m:
            try:
                a = s1.loc[s]
            except KeyError:
                continue
            for mid in m:
                sid = 2 if mid.startswith("S2-") else 3
                if mid not in src[sid].index:
                    continue
                b = src[sid].loc[mid]
                if a["name_norm"] == b["name_norm"]:
                    continue
                ns = fuzz.ratio(a["name_norm"], b["name_norm"])
                ad = fuzz.token_set_ratio(a["address_norm"] or "",
                                          b["address_norm"] or "")
                if 55 <= ns < 95 or 55 <= ad < 95:
                    variation.append({"s1": s, "kind": "variation", "gt": m,
                                      "name_sim": round(ns, 1),
                                      "addr_sim": round(ad, 1)})
                    break
        if len(hardneg) < n_each and m and len(m) == 1:
            # A non-match that is very similar to the S1. Looked up from the
            # exact-name index (already built for retrieval, and cached on
            # disk) instead of scanning 5M rows with iterrows, which is orders
            # of magnitude slower.
            try:
                a_name = s1.at[s, "name_norm"]
            except KeyError:
                continue
            if not a_name:
                continue
            mid = m[0]
            sid = 2 if mid.startswith("S2-") else 3
            if _posting_index(sid) is None:
                continue
            rows = _posting_index(sid).get(int(_hash_str(a_name)))
            if len(rows) == 0:
                continue
            for r in rows[:200].tolist():
                cid = src_ids(sid)[r]
                if cid == mid or cid in m:
                    continue
                hardneg.append({"s1": s, "kind": "hardneg", "gt": m,
                                "confusable": cid})
                break
        if all(len(x) >= n_each for x in
               (single, multi, variation, nomatch, hardneg)):
            break

    cases = single + multi + variation + nomatch + hardneg
    # de-duplicate while preserving order
    seen, uniq = set(), []
    for c in cases:
        if c["s1"] in seen:
            continue
        seen.add(c["s1"])
        uniq.append(c)
    print(f"[pick] {len(uniq)} cases: " + ", ".join(
        f"{k}={sum(1 for c in uniq if c['kind'] == k)}"
        for k in ("single", "multi", "variation", "nomatch", "hardneg")),
        flush=True)
    return uniq


def main() -> None:
    g = guard()
    p = paths()
    cases = pick_cases()
    (p["reports_dir"] / "control_cases.json").write_text(
        json.dumps(cases, indent=2), encoding="utf-8")

    out = CACHE / "control_cases.json"
    out.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
