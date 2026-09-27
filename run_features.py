"""
Pair-feature computation (vectorised).

The per-pair reference implementation lives in ``src/features.py``
(``compute_features``) and is the correctness oracle;
``experiments/test_features.py`` asserts that this module reproduces it
feature-for-feature.

Design notes
------------
* String similarity uses ``rapidfuzz.process.cpdist``, the *paired* element-wise
  variant (measured 1.95e6 pairs/s). ``process.cdist`` is the wrong tool: it
  returns a full |a| x |b| cross matrix.
* Missing values follow the reference convention: both empty -> 1.0, exactly
  one empty -> 0.0. The 0-100 scores are divided by 100 ONLY for live rows, so
  the sentinel values are not scaled.
* ``address_dig`` is ``" ".join(findall(r"\\d+", raw_address))``, i.e. one token
  per digit group. The overlap uses DISTINCT groups while the count feature
  must count OCCURRENCES, so token lists (not sets) are used for the count.
* Channel names must stay in sync with ``src/retrieval.py`` CHANNEL_NAMES and
  the ``prov_*`` entries of ``src/features.py`` FEATURE_NAMES.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import FEATURE_NAMES, N_FEATURES
from src.utils import guard, paths

WORKERS = 4

CHANNEL_KEYS = ("exact_name", "exact_address", "address_digits",
                "rare_token", "ngram")


class RowStore:
    """
    Compact per-row attribute table backed by Python lists.

    Holds only the columns the features need, so a 5.3M-row source costs four
    lists instead of a 9-column DataFrame.
    """

    __slots__ = ("name", "addr", "dig", "country", "n")

    def __init__(self, path: Path) -> None:
        t = pq.read_table(str(path), columns=["name_norm", "address_norm",
                                              "address_dig", "country"])
        self.name = t.column("name_norm").to_pylist()
        self.addr = t.column("address_norm").to_pylist()
        self.dig = t.column("address_dig").to_pylist()
        self.country = [c.strip().casefold() if c else ""
                        for c in t.column("country").to_pylist()]
        self.n = len(self.name)
        del t

    def take(self, rows: np.ndarray, which: int) -> list:
        src = (self.name, self.addr, self.dig, self.country)[which]
        return [src[i] for i in rows.tolist()]


class RowReader:
    """
    Random access to a few source rows without loading the whole column.

    ``RowStore`` materialises every row of a 5.3M-row source as Python lists
    (~3 GB per source for the four feature columns). That is acceptable for a
    streaming pass over all pairs, but fatal when only a few thousand rows are
    needed: the controlled-validation run peaked at 99.7% total RAM because of
    it. This reads only the row groups that actually contain the requested rows
    and caches just those values.
    """

    __slots__ = ("_pf", "_cols", "_cache", "_bounds")

    def __init__(self, path: Path, columns: tuple[str, ...]) -> None:
        self._pf = pq.ParquetFile(str(path))
        self._cols = list(columns)
        n_rg = self._pf.metadata.num_row_groups
        bounds = np.zeros(n_rg + 1, dtype=np.int64)
        for i in range(n_rg):
            bounds[i + 1] = bounds[i] + self._pf.metadata.row_group(i).num_rows
        self._bounds = bounds
        self._cache: dict[tuple, str] = {}

    def take(self, rows: np.ndarray, which: int) -> list:
        """
        Return the requested column for the given global row ids.

        Only the row groups containing those ids are read, and the values are
        cached, so repeated calls for the same rows are free.
        """
        col = self._cols[which]
        want = rows.tolist()
        todo = np.unique(rows)
        groups = np.searchsorted(self._bounds, todo, side="right") - 1
        by_group: dict[int, list[int]] = {}
        for r, gi in zip(todo.tolist(), groups.tolist()):
            by_group.setdefault(int(gi), []).append(int(r))
        for gi, rs in by_group.items():
            missing = [r for r in rs if (r, which) not in self._cache]
            if not missing:
                continue
            start = int(self._bounds[gi])
            vals = self._pf.read_row_group(gi, columns=[col]).column(col).to_pylist()
            for r in missing:
                v = vals[r - start] or ""
                if which == 3:            # country: normalise like RowStore
                    v = v.strip().casefold()
                self._cache[(r, which)] = v
            del vals
        return [self._cache.get((r, which), "") for r in want]

    def take_country(self, rows: np.ndarray) -> list:
        return self.take(rows, 3)


def compute_batch(
    s1_name: list, s1_addr: list, s1_dig: list, s1_ctry: list,
    c_name: list, c_addr: list, c_dig: list, c_ctry: list,
    n_channels: np.ndarray,
    prov: dict[str, np.ndarray],
) -> np.ndarray:
    """Feature matrix for one batch of candidate pairs (float32)."""
    n = len(s1_name)
    X = np.zeros((n, N_FEATURES), dtype=np.float32)

    def safe_sim(a: list, b: list, scorer) -> np.ndarray:
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

    # ---- name (0-8) ----
    s1n = [a.split() for a in s1_name]
    c1n = [b.split() for b in c_name]
    s1ns = [set(x) for x in s1n]
    c1ns = [set(x) for x in c1n]
    X[:, 0:9] = np.stack([
        np.fromiter((1.0 if (a and a == b) else 0.0
                     for a, b in zip(s1_name, c_name)),
                    dtype=np.float32, count=n),
        safe_sim(s1_name, c_name, fuzz.ratio),
        safe_sim(s1_name, c_name, fuzz.partial_ratio),
        safe_sim(s1_name, c_name, fuzz.token_sort_ratio),
        safe_sim(s1_name, c_name, fuzz.token_set_ratio),
        np.fromiter((len(x & y) / len(x | y) if (x or y) else 1.0
                     for x, y in zip(s1ns, c1ns)), dtype=np.float32, count=n),
        np.abs(np.fromiter((len(a) for a in s1_name), np.int32, n)
               - np.fromiter((len(b) for b in c_name), np.int32, n)
               ).astype(np.float32),
        np.abs(np.fromiter((len(x) for x in s1n), np.int32, n)
               - np.fromiter((len(x) for x in c1n), np.int32, n)
               ).astype(np.float32),
        np.fromiter((min(len(a), len(b)) / max(len(a), len(b), 1)
                     if (a and b) else 0.0
                     for a, b in zip(s1_name, c_name)),
                    dtype=np.float32, count=n),
    ], axis=1)

    # ---- address (9-17) ----
    s1a = [a.split() for a in s1_addr]
    c1a = [b.split() for b in c_addr]
    s1as = [set(x) for x in s1a]
    c1as = [set(x) for x in c1a]
    a_edit = safe_sim(s1_addr, c_addr, fuzz.ratio)
    X[:, 9:18] = np.stack([
        np.fromiter((1.0 if (a and a == b) else 0.0
                     for a, b in zip(s1_addr, c_addr)),
                    dtype=np.float32, count=n),
        a_edit,
        safe_sim(s1_addr, c_addr, fuzz.partial_ratio),
        safe_sim(s1_addr, c_addr, fuzz.token_sort_ratio),
        safe_sim(s1_addr, c_addr, fuzz.token_set_ratio),
        np.fromiter((len(x & y) / len(x | y) if (x or y) else 1.0
                     for x, y in zip(s1as, c1as)), dtype=np.float32, count=n),
        np.abs(np.fromiter((len(a) for a in s1_addr), np.int32, n)
               - np.fromiter((len(b) for b in c_addr), np.int32, n)
               ).astype(np.float32),
        np.abs(np.fromiter((len(x) for x in s1a), np.int32, n)
               - np.fromiter((len(x) for x in c1a), np.int32, n)
               ).astype(np.float32),
        np.fromiter((min(len(a), len(b)) / max(len(a), len(b), 1)
                     if (a and b) else 0.0
                     for a, b in zip(s1_addr, c_addr)),
                    dtype=np.float32, count=n),
    ], axis=1)

    # ---- digits (18-21) ----
    s1dl = [a.split() for a in s1_dig]
    c1dl = [b.split() for b in c_dig]
    s1ds = [set(x) for x in s1dl]
    c1ds = [set(x) for x in c1dl]
    d_ov = np.fromiter((len(x & y) / len(x | y) if (x or y) else 1.0
                        for x, y in zip(s1ds, c1ds)),
                       dtype=np.float32, count=n)
    X[:, 18:22] = np.stack([
        np.fromiter((1.0 if (a and a == b) else 0.0
                     for a, b in zip(s1_dig, c_dig)),
                    dtype=np.float32, count=n),
        d_ov,
        safe_sim(s1_dig, c_dig, fuzz.ratio),
        np.abs(np.fromiter((len(x) for x in s1dl), np.int32, n)
               - np.fromiter((len(x) for x in c1dl), np.int32, n)
               ).astype(np.float32),
    ], axis=1)

    # ---- country (22-23) ----
    both = np.fromiter((1.0 if (a and b) else 0.0
                        for a, b in zip(s1_ctry, c_ctry)),
                       dtype=np.float32, count=n)
    X[:, 22] = np.fromiter((1.0 if (a and b and a == b) else 0.0
                            for a, b in zip(s1_ctry, c_ctry)),
                           dtype=np.float32, count=n)
    X[:, 23] = both

    # ---- cross-field (24-25) ----
    n_edit = X[:, 1]
    X[:, 24] = n_edit * a_edit
    X[:, 25] = (n_edit + a_edit) / 2.0

    # ---- contradictions (26-29) ----
    X[:, 26] = np.fromiter(
        (1.0 if (sa and sa == ca and aa and ba and aa != ba and a_edit[i] < 0.5)
         else 0.0 for i, (sa, ca, aa, ba) in
         enumerate(zip(s1_name, c_name, s1_addr, c_addr))),
        dtype=np.float32, count=n)
    X[:, 27] = np.fromiter(
        (1.0 if (aa and aa == ba and sa and ca and sa != ca and n_edit[i] < 0.5)
         else 0.0 for i, (sa, ca, aa, ba) in
         enumerate(zip(s1_name, c_name, s1_addr, c_addr))),
        dtype=np.float32, count=n)
    X[:, 28] = np.fromiter(
        (1.0 if (x and y and d < 0.3) else 0.0
         for x, y, d in zip(s1_dig, c_dig, d_ov)), dtype=np.float32, count=n)
    X[:, 29] = np.fromiter(
        (1.0 if (bb and a != b) else 0.0
         for a, b, bb in zip(s1_ctry, c_ctry, both)),
        dtype=np.float32, count=n)

    # ---- missingness (30-34) ----
    X[:, 30] = np.fromiter((1.0 if not a else 0.0 for a in s1_addr),
                           dtype=np.float32, count=n)
    X[:, 31] = np.fromiter((1.0 if not a else 0.0 for a in c_addr),
                           dtype=np.float32, count=n)
    X[:, 32] = ((X[:, 30] > 0) | (X[:, 31] > 0)).astype(np.float32)
    X[:, 33] = np.fromiter((1.0 if not a else 0.0 for a in s1_name),
                           dtype=np.float32, count=n)
    X[:, 34] = np.fromiter((1.0 if not a else 0.0 for a in c_name),
                           dtype=np.float32, count=n)

    # ---- provenance (35-40) ----
    for j, key in enumerate(CHANNEL_KEYS):
        X[:, 35 + j] = prov[key]
    X[:, 40] = n_channels
    return X
