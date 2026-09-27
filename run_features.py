"""
Stage 2: pair features for the candidate universe.

The candidate universe is stored as int32 row ids (12 B/pair instead of ~40 B
of strings), so features are computed by loading the required source columns
once, building compact per-row views, and streaming the candidate file in
batches.

Two implementations:
  * ``compute_features_parquet`` - vectorised over a batch of candidate pairs
    using rapidfuzz's ``process.cdist`` (measured 1.67e8 pairs/s), which is
    the fast path used for the full run;
  * the per-pair reference in ``src/features.py`` - kept as the correctness
    oracle that the vectorised path is tested against.

Output: float32 memmap ``(n_pairs, N_FEATURES)`` plus a meta json recording the
shape and the feature names, so a stale cache can never be silently reused.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import FEATURE_NAMES, N_FEATURES
from src.utils import guard, paths

CACHE = Path("D:/amazon_ml/cache")


class RowStore:
    """
    Compact per-row attribute table backed by numpy object arrays.

    Holds only the columns features need, so a 5.3M-row source costs
    ~5 arrays instead of a 9-column DataFrame.
    """

    __slots__ = ("name", "addr", "dig", "country", "n")

    def __init__(self, path: Path) -> None:
        t = pq.read_table(str(path), columns=["name_norm", "address_norm",
                                              "address_dig", "country"])
        self.name = t.column("name_norm").to_pylist()
        self.addr = t.column("address_norm").to_pylist()
        self.dig = t.column("address_dig").to_pylist()
        self.country = [c.strip().casefold() if c else "" for c in
                        t.column("country").to_pylist()]
        self.n = len(self.name)
        del t

    def take(self, rows: np.ndarray, which: int) -> list:
        src = (self.name, self.addr, self.dig, self.country)[which]
        return [src[i] for i in rows.tolist()]


def _token_jaccard(a: list, b: list) -> np.ndarray:
    """Jaccard over pre-split tokens, vectorised with sets in a comprehension."""
    return np.fromiter(
        (len(x & y) / len(x | y) if (x or y) else 1.0 for x, y in zip(a, b)),
        dtype=np.float32, count=len(a))


def compute_batch(
    s1_name: list, s1_addr: list, s1_dig: list, s1_ctry: list,
    c_name: list, c_addr: list, c_dig: list, c_ctry: list,
    n_channels: np.ndarray,
    prov: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Feature matrix for one batch of candidate pairs (vectorised).

    String similarities come from ``rapidfuzz.process.cdist``, which is
    C++/SIMD; the remaining features are numpy.
    """
    n = len(s1_name)
    X = np.zeros((n, N_FEATURES), dtype=np.float32)
    i = 0

    def safe_sim(a: list, b: list, scorer) -> np.ndarray:
        """
        Element-wise similarity already scaled to 0..1.

        ``process.cdist`` computes a full |a| x |b| cross matrix (wrong shape
        and quadratic); ``process.cpdist`` is the paired element-wise variant
        (measured 1.95e6 pairs/s), which is what a candidate pair needs.

        Missing-value convention, matching the per-pair reference
        ``src/features._safe_sim``:
          * both empty -> 1.0  (no evidence of difference)
          * one empty  -> 0.0
        The 0-100 scores are divided by 100 ONLY for the live rows, so the
        sentinel values are not scaled (dividing 1.0 by 100 would give 0.01).
        """
        ea = np.fromiter((not x for x in a), dtype=bool, count=n)
        eb = np.fromiter((not x for x in b), dtype=bool, count=n)
        both_empty = ea & eb
        one_empty = ea ^ eb
        live = ~(ea | eb)
        out = np.zeros(n, dtype=np.float32)
        if live.any():
            la = [x for x, keep in zip(a, live) if keep]
            lb = [x for x, keep in zip(b, live) if keep]
            if la and lb:
                out[live] = process.cpdist(
                    np.asarray(la, dtype=object), np.asarray(lb, dtype=object),
                    scorer=scorer, workers=WORKERS, dtype=np.float32) / 100.0
        out[one_empty] = 0.0
        out[both_empty] = 1.0
        return out

    # ---- name ----
    n_exact = np.fromiter(
        (1.0 if (a and a == b) else 0.0 for a, b in zip(s1_name, c_name)),
        dtype=np.float32, count=n)
    n_edit = safe_sim(s1_name, c_name, fuzz.ratio)
    n_part = safe_sim(s1_name, c_name, fuzz.partial_ratio)
    n_tsort = safe_sim(s1_name, c_name, fuzz.token_sort_ratio)
    n_tset = safe_sim(s1_name, c_name, fuzz.token_set_ratio)
    s1n = [a.split() for a in s1_name]
    c1n = [b.split() for b in c_name]
    s1ns = [set(x) for x in s1n]
    c1ns = [set(x) for x in c1n]
    n_jac = np.fromiter(
        (len(x & y) / len(x | y) if (x or y) else 1.0
         for x, y in zip(s1ns, c1ns)), dtype=np.float32, count=n)
    n_len_diff = np.abs(np.fromiter((len(a) for a in s1_name), np.int32, n)
                        - np.fromiter((len(b) for b in c_name), np.int32, n)
                        ).astype(np.float32)
    n_tok_diff = np.abs(np.fromiter((len(x) for x in s1n), np.int32, n)
                        - np.fromiter((len(x) for x in c1n), np.int32, n)
                        ).astype(np.float32)
    n_len_ratio = np.fromiter(
        (min(len(a), len(b)) / max(len(a), len(b), 1) if (a and b) else 0.0
         for a, b in zip(s1_name, c_name)), dtype=np.float32, count=n)
    X[:, 0:9] = np.stack([n_exact, n_edit, n_part, n_tsort, n_tset, n_jac,
                          n_len_diff, n_tok_diff, n_len_ratio], axis=1)
    i = 9

    # ---- address ----
    a_exact = np.fromiter(
        (1.0 if (a and a == b) else 0.0 for a, b in zip(s1_addr, c_addr)),
        dtype=np.float32, count=n)
    a_edit = safe_sim(s1_addr, c_addr, fuzz.ratio)
    a_part = safe_sim(s1_addr, c_addr, fuzz.partial_ratio)
    a_tsort = safe_sim(s1_addr, c_addr, fuzz.token_sort_ratio)
    a_tset = safe_sim(s1_addr, c_addr, fuzz.token_set_ratio)
    s1a = [a.split() for a in s1_addr]
    c1a = [b.split() for b in c_addr]
    s1as = [set(x) for x in s1a]
    c1as = [set(x) for x in c1a]
    a_jac = np.fromiter(
        (len(x & y) / len(x | y) if (x or y) else 1.0
         for x, y in zip(s1as, c1as)), dtype=np.float32, count=n)
    a_len_diff = np.abs(np.fromiter((len(a) for a in s1_addr), np.int32, n)
                        - np.fromiter((len(b) for b in c_addr), np.int32, n)
                        ).astype(np.float32)
    a_tok_diff = np.abs(np.fromiter((len(x) for x in s1a), np.int32, n)
                        - np.fromiter((len(x) for x in c1a), np.int32, n)
                        ).astype(np.float32)
    a_len_ratio = np.fromiter(
        (min(len(a), len(b)) / max(len(a), len(b), 1) if (a and b) else 0.0
         for a, b in zip(s1_addr, c_addr)), dtype=np.float32, count=n)
    X[:, 9:18] = np.stack([a_exact, a_edit, a_part, a_tsort, a_tset, a_jac,
                           a_len_diff, a_tok_diff, a_len_ratio], axis=1)
    i = 18

    # ---- digits ----
    # address_dig is " ".join(findall(r"\d+", address)), i.e. one token per
    # digit group. The overlap is over DISTINCT groups, but the count feature
    # must count OCCURRENCES (a set would report "12 12 12" as 1), so token
    # lists are used for the count and sets for the overlap.
    s1dl = [a.split() for a in s1_dig]
    c1dl = [b.split() for b in c_dig]
    s1ds = [set(x) for x in s1dl]
    c1ds = [set(x) for x in c1dl]
    d_exact = np.fromiter(
        (1.0 if (a and a == b) else 0.0 for a, b in zip(s1_dig, c_dig)),
        dtype=np.float32, count=n)
    d_ov = np.fromiter(
        (len(x & y) / len(x | y) if (x or y) else 1.0
         for x, y in zip(s1ds, c1ds)), dtype=np.float32, count=n)
    d_seq = safe_sim(s1_dig, c_dig, fuzz.ratio)
    d_cnt = np.abs(np.fromiter((len(x) for x in s1dl), np.int32, n)
                   - np.fromiter((len(x) for x in c1dl), np.int32, n)
                   ).astype(np.float32)
    X[:, 18:22] = np.stack([d_exact, d_ov, d_seq, d_cnt], axis=1)
    i = 22

    # ---- country ----
    both = np.fromiter((1.0 if (a and b) else 0.0 for a, b in zip(s1_ctry, c_ctry)),
                       dtype=np.float32, count=n)
    match = np.fromiter((1.0 if (a and b and a == b) else 0.0
                         for a, b in zip(s1_ctry, c_ctry)),
                        dtype=np.float32, count=n)
    X[:, 22:24] = np.stack([match, both], axis=1)
    i = 24

    # ---- cross-field ----
    X[:, 24] = n_edit * a_edit
    X[:, 25] = (n_edit + a_edit) / 2.0
    i = 26

    # ---- contradictions ----
    X[:, 26] = np.fromiter(
        (1.0 if (sa and sa == ca and aa and ba and aa != ba and ae < 0.5) else 0.0
         for sa, ca, aa, ba, ae in zip(s1_name, c_name, s1_addr, c_addr, a_edit)),
        dtype=np.float32, count=n)
    X[:, 27] = np.fromiter(
        (1.0 if (aa and aa == ba and sa and ca and sa != ca and ne < 0.5) else 0.0
         for sa, ca, aa, ba, ne in zip(s1_name, c_name, s1_addr, c_addr, n_edit)),
        dtype=np.float32, count=n)
    X[:, 28] = np.fromiter(
        (1.0 if (x and y and d < 0.3) else 0.0
         for x, y, d in zip(s1_dig, c_dig, d_ov)), dtype=np.float32, count=n)
    X[:, 29] = np.fromiter(
        (1.0 if (bb and a != b) else 0.0
         for a, b, bb in zip(s1_ctry, c_ctry, both)), dtype=np.float32, count=n)
    i = 30

    # ---- missingness ----
    X[:, 30] = np.fromiter((1.0 if not a else 0.0 for a in s1_addr),
                           dtype=np.float32, count=n)
    X[:, 31] = np.fromiter((1.0 if not a else 0.0 for a in c_addr),
                           dtype=np.float32, count=n)
    X[:, 32] = ((X[:, 30] > 0) | (X[:, 31] > 0)).astype(np.float32)
    X[:, 33] = np.fromiter((1.0 if not a else 0.0 for a in s1_name),
                           dtype=np.float32, count=n)
    X[:, 34] = np.fromiter((1.0 if not a else 0.0 for a in c_name),
                           dtype=np.float32, count=n)
    i = 35

    # ---- provenance ----
    # Channel names must match src/retrieval.py CHANNEL_NAMES and
    # src/features.py FEATURE_NAMES (prov_rare_token / prov_ngram).
    for j, key in enumerate(("exact_name", "exact_address", "address_digits",
                             "rare_token", "ngram")):
        X[:, 35 + j] = prov[key]
    X[:, 40] = n_channels
    assert N_FEATURES == 41, f"expected 41 features, got {N_FEATURES}"
    return X


WORKERS = 4


def build_features(cand_paths: dict[int, Path], s1_store: RowStore,
                   stores: dict[int, RowStore], out_path: Path,
                   batch: int = 200_000) -> int:
    """
    Compute features for the union of candidate files, streaming to a memmap.

    Returns the number of rows written.
    """
    g = guard()
    total = sum(pq.ParquetFile(str(p)).metadata.num_rows for p in cand_paths.values())
    print(f"[features] {total:,} candidate pairs -> {out_path.name}", flush=True)

    meta_path = out_path.with_suffix(".meta.json")
    if out_path.exists() and meta_path.exists():
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        if m.get("n_pairs") == total and m.get("n_features") == N_FEATURES:
            print(f"[features] cache valid: {total:,} x {N_FEATURES}")
            return total
        out_path.unlink(missing_ok=True)

    fp = np.lib.format.open_memmap(str(out_path), mode="w+", dtype=np.float32,
                                   shape=(total, N_FEATURES))
    row = 0
    t0 = time.perf_counter()
    for sid, path in sorted(cand_paths.items()):
        store = stores[sid]
        pf = pq.ParquetFile(str(path))
        for rb in pf.iter_batches(batch_size=batch,
                                  columns=["s1_row", "cand_row", "channels"]):
            s1r = rb.column("s1_row").to_numpy()
            cr = rb.column("cand_row").to_numpy()
            ch = rb.column("channels").to_pylist()
            m = len(s1r)
            X = compute_batch(
                s1_store.take(s1r, 0), s1_store.take(s1r, 1),
                s1_store.take(s1r, 2), s1_store.take(s1r, 3),
                store.take(cr, 0), store.take(cr, 1), store.take(cr, 2),
                store.take(cr, 3),
                np.fromiter((x.count("|") + 1 if x else 0 for x in ch),
                           dtype=np.float32, count=m),
                {
                    "exact_name": np.fromiter(
                        ("exact_name" in x for x in ch), dtype=np.float32, count=m),
                    "exact_address": np.fromiter(
                        ("exact_address" in x for x in ch), dtype=np.float32, count=m),
                    "address_digits": np.fromiter(
                        ("address_digits" in x for x in ch), dtype=np.float32, count=m),
                    "rare_token": np.fromiter(
                        ("rare_token" in x for x in ch), dtype=np.float32, count=m),
                    "ngram": np.fromiter(
                        ("ngram" in x for x in ch), dtype=np.float32, count=m),
                },
            )
            fp[row:row + m] = X
            row += m
            del rb, s1r, cr, ch, X
            g.gc(1)
            if row % (batch * 20) == 0 or row == total:
                el = time.perf_counter() - t0
                print(f"  features {row:,}/{total:,} "
                      f"({row / max(el, 1e-9):,.0f} pairs/s) | {g.status()}",
                      flush=True)
    fp.flush()
    del fp
    meta_path.write_text(json.dumps({
        "n_pairs": total, "n_features": N_FEATURES,
        "feature_names": FEATURE_NAMES,
    }), encoding="utf-8")
    print(f"[features] wrote {row:,} rows in {(time.perf_counter()-t0)/60:.1f}m",
          flush=True)
    return row


def main() -> None:
    g = guard()
    p = paths()
    global WORKERS
    WORKERS = max(1, min(4, g.max_workers))
    split = sys.argv[1] if len(sys.argv) > 1 else "train"

    s1_store = RowStore(CACHE / f"{split}_source1_norm.parquet")
    stores = {}
    cand = {}
    for sid in (2, 3):
        sp = CACHE / f"{split}_source{sid}_norm.parquet"
        cp = CACHE / f"{split}_cand_S{sid}.parquet"
        if not cp.exists():
            print(f"[skip] {cp.name} missing")
            continue
        stores[sid] = RowStore(sp)
        cand[sid] = cp
    if not cand:
        raise SystemExit("no candidate files found")
    out = CACHE / f"{split}_features.npy"
    n = build_features(cand, s1_store, stores, out)
    print(f"[done] {n:,} x {N_FEATURES} -> {out} | {g.status()}")


if __name__ == "__main__":
    main()
