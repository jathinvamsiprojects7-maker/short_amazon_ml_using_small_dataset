"""
Compact, array-based retrieval indexes.

Design goals (hard constraints from the architecture + resource policy):
  * NO dense S1 x S2/S3 similarity matrix, ever.
  * NO Python dict of lists for postings (millions of small lists = GBs of
    interpreter overhead).
  * NO holding a full source DataFrame in RAM.
  * Everything is a contiguous numpy array so it can be memory-mapped,
    streamed in chunks, and released deterministically.

Structures
----------
PostingIndex
    uint64 key -> sorted int32 row ids, via argsort + searchsorted.
    Replaces ``dict[str, list[str]]``.

TokenIndex
    uint64 token hash -> sorted int32 row ids, stored as a flat CSR-like
    (offsets, rows) pair.  Replaces ``dict[str, list[str]]`` for tokens.

PairBuffer
    Growable int32/int32/uint8 arrays for (s1_row, cand_row, channel_bits).
    Replaces ``dict[tuple[str, str], int]``.

Hashing
-------
Keys are stored as 64-bit hashes (pandas' ``hash_array``, siphash) rather than
strings.  For 5.3M keys the collision probability is ~7e-7, and a collision can
only ever introduce an extra *candidate* (never lose a true match), which the
downstream model then scores.  Collision risk is therefore benign for a
recall-then-rank architecture, and it removes all string-object overhead.

Empty strings are never indexed: they are assigned the reserved hash 0 and
excluded from every index, so a missing address can never match another
missing address.
"""

from __future__ import annotations

import gc
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Reserved hash for empty / missing strings. Never queried, never indexed.
EMPTY_HASH = np.uint64(0)

_EMPTY_I32 = np.empty(0, dtype=np.int32)


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def hash_str_array(values) -> np.ndarray:
    """Hash an array of Python strings to uint64. Empty strings -> 0.

    Accepts a numpy object array, a pyarrow StringArray, or any iterable.
    """
    if isinstance(values, pa.Array):
        arr = values.to_numpy(zero_copy_only=False)
    else:
        arr = np.asarray(values, dtype=object)

    out = np.zeros(len(arr), dtype=np.uint64)
    if len(arr) == 0:
        return out

    # pandas hash_array is a vectorised Cython loop (~1.5M strings/s).
    nonempty = arr != ""
    n_ne = int(nonempty.sum())
    if n_ne:
        out[nonempty] = pd.util.hash_array(arr[nonempty], encoding="utf8")
    return out


def hash_str_list(strings: list[str]) -> np.ndarray:
    """Hash a python list of strings to uint64 (empty -> 0)."""
    out = np.empty(len(strings), dtype=np.uint64)
    for i, s in enumerate(strings):
        out[i] = 0 if not s else pd.util.hash_array(np.array([s], dtype=object), encoding="utf8")[0]
    return out


# ---------------------------------------------------------------------------
# PostingIndex: uint64 key -> int32 row ids
# ---------------------------------------------------------------------------

class PostingIndex:
    """
    Immutable sorted inverted index over uint64 keys.

    Build cost: one argsort over ``n`` uint64 values.
    Memory: keys (8B/key) + offsets (8B/key) + rows (4B/entry).
    """

    __slots__ = ("keys", "offsets", "rows", "n_docs", "n_postings")

    def __init__(self, keys: np.ndarray, rows: np.ndarray) -> None:
        """
        keys: uint64[n]   -- EMPTY_HASH entries are dropped
        rows: int32[n]    -- global row ids, parallel to keys
        """
        keys = np.ascontiguousarray(keys, dtype=np.uint64)
        rows = np.ascontiguousarray(rows, dtype=np.int32)

        keep = keys != EMPTY_HASH
        n_keep = int(keep.sum())
        self.n_docs = len(rows)
        if n_keep == 0:
            self.keys = np.empty(0, dtype=np.uint64)
            self.offsets = np.zeros(1, dtype=np.int64)
            self.rows = _EMPTY_I32
            self.n_postings = 0
            return

        keys = keys[keep]
        rows = rows[keep]

        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        sorted_rows = rows[order]
        del keys, rows

        uniq, start = np.unique(sorted_keys, return_index=True)
        counts = np.diff(np.append(start, np.int64(len(sorted_keys))))

        offsets = np.zeros(len(uniq) + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])

        self.keys = uniq
        self.offsets = offsets
        self.rows = sorted_rows
        self.n_postings = len(sorted_rows)
        del order, sorted_keys, start, counts, uniq

    @classmethod
    def from_column(
        cls,
        parquet_path,
        column: str,
        batch_size: int = 500_000,
    ) -> "PostingIndex":
        """Build from one string column of a parquet file (streaming, low RAM)."""
        key_parts: list[np.ndarray] = []
        row_parts: list[np.ndarray] = []
        base = 0
        for batch in _iter_batches(parquet_path, [column], batch_size):
            h = hash_str_array(batch.column(column))
            n = len(h)
            key_parts.append(h)
            row_parts.append(np.arange(base, base + n, dtype=np.int32))
            base += n
        if not key_parts:
            return cls(np.empty(0, np.uint64), np.empty(0, np.int32))
        return cls(np.concatenate(key_parts), np.concatenate(row_parts))

    def get(self, key: int | np.uint64) -> np.ndarray:
        """Return int32 array of row ids for ``key`` (empty if absent)."""
        if len(self.keys) == 0:
            return _EMPTY_I32
        k = np.uint64(key)
        lo = int(np.searchsorted(self.keys, k, side="left"))
        if lo >= len(self.keys) or self.keys[lo] != k:
            return _EMPTY_I32
        a = int(self.offsets[lo])
        b = int(self.offsets[lo + 1])
        return self.rows[a:b]

    def get_many(self, keys: np.ndarray) -> list[np.ndarray]:
        """Vectorised lookup of many keys -> list of int32 arrays."""
        out: list[np.ndarray] = []
        if len(self.keys) == 0:
            return [_EMPTY_I32] * len(keys)
        lo = np.searchsorted(self.keys, keys, side="left")
        hi = np.searchsorted(self.keys, keys, side="right")
        offs = self.offsets
        for l, h in zip(lo, hi):
            if l >= h or l >= len(self.keys) or self.keys[l] != keys[l]:
                out.append(_EMPTY_I32)
            else:
                out.append(self.rows[int(offs[l]):int(offs[h])])
        return out

    def nbytes(self) -> int:
        return int(self.keys.nbytes + self.offsets.nbytes + self.rows.nbytes)


# ---------------------------------------------------------------------------
# TokenIndex: uint64 token hash -> int32 doc rows (CSR-style)
# ---------------------------------------------------------------------------

class TokenIndex:
    """
    Inverted index over hashed tokens, built from a token-string column.

    Tokens shorter than ``min_len`` are ignored.  Tokens whose document
    frequency exceeds ``max_df`` are dropped (they are non-informative block
    keys such as "the" / "street" / "inc").

    Stored as CSR: ``offsets`` indexes into ``rows``.
    """

    __slots__ = ("keys", "offsets", "rows", "n_docs", "min_len", "max_df", "n_vocab")

    def __init__(self, keys, offsets, rows, n_docs, min_len, max_df) -> None:
        self.keys = keys
        self.offsets = offsets
        self.rows = rows
        self.n_docs = n_docs
        self.min_len = min_len
        self.max_df = max_df
        self.n_vocab = len(keys)
        # NOTE: postings for a common token can cover a large share of the
        # source. Callers must select tokens explicitly (e.g. ``rarest``)
        # rather than unioning every token of a document; see ``rarest``.

    @classmethod
    def from_column(
        cls,
        parquet_path,
        column: str,
        min_len: int = 3,
        max_df: int = 10_000,
        batch_size: int = 500_000,
    ) -> "TokenIndex":
        """
        Two streaming passes over ``column``:
          pass 1 - document frequency of every token
          pass 2 - build postings for tokens with df <= max_df

        Both passes are fully vectorised per batch: tokens are flattened into
        one object array, hashed with a single ``pd.util.hash_array`` call
        (~0.05us/token), and de-duplicated within a document using a combined
        ``token * n_docs + doc`` integer key. Hashing per token (112us each)
        or accumulating df in a Python loop over uniques makes this ~100x
        slower and turns a 5M-row source into an hours-long build.
        """
        n_docs_total = 0

        # ---- pass 1: document frequency ----
        df_counts: dict[int, int] = {}
        for batch in _iter_batches(parquet_path, [column], batch_size):
            flat, spans, n_docs_total = _flatten_tokens(
                batch.column(column).to_pylist(), min_len, n_docs_total
            )
            if flat:
                toks, docs = _unique_within_doc(flat, spans, len(batch))
                for t in np.unique(toks).tolist():
                    df_counts[t] = df_counts.get(t, 0) + 1
            del batch
            gc.collect()

        allowed = {h for h, c in df_counts.items() if c <= max_df}
        del df_counts
        gc.collect()

        if not allowed:
            return cls(
                np.empty(0, np.uint64), np.zeros(1, np.int64),
                _EMPTY_I32, n_docs_total, min_len, max_df,
            )

        # ---- pass 2: postings ----
        tok_parts: list[np.ndarray] = []
        row_parts: list[np.ndarray] = []
        base = 0
        for batch in _iter_batches(parquet_path, [column], batch_size):
            flat, spans, _ = _flatten_tokens(
                batch.column(column).to_pylist(), min_len, 0
            )
            if flat:
                toks, docs = _unique_within_doc(flat, spans, len(batch))
                allowed_arr = np.fromiter((t in allowed for t in toks.tolist()),
                                          dtype=bool, count=len(toks))
                if allowed_arr.any():
                    tok_parts.append(toks[allowed_arr].astype(np.uint64))
                    row_parts.append((base + docs[allowed_arr]).astype(np.int32))
            base += len(batch)
            del batch
            gc.collect()

        if not tok_parts:
            return cls(
                np.empty(0, np.uint64), np.zeros(1, np.int64),
                _EMPTY_I32, n_docs_total, min_len, max_df,
            )

        tok_hash = np.concatenate(tok_parts)
        doc_row = np.concatenate(row_parts)
        del tok_parts, row_parts

        order = np.argsort(tok_hash, kind="stable")
        tok_hash = tok_hash[order]
        doc_row = doc_row[order]
        uniq, start = np.unique(tok_hash, return_index=True)
        counts = np.diff(np.append(start, np.int64(len(tok_hash))))
        offsets = np.zeros(len(uniq) + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])

        return cls(uniq, offsets, doc_row, n_docs_total, min_len, max_df)

    def get(self, token: str) -> np.ndarray:
        return self.get_hash(_hash_one(token))

    def get_hash(self, h: int) -> np.ndarray:
        if len(self.keys) == 0:
            return _EMPTY_I32
        k = np.uint64(h)
        lo = int(np.searchsorted(self.keys, k, side="left"))
        if lo >= len(self.keys) or self.keys[lo] != k:
            return _EMPTY_I32
        return self.rows[int(self.offsets[lo]):int(self.offsets[lo + 1])]

    def query_tokens_batched(
        self,
        token_lists: list[str],
        min_len: int = 3,
    ) -> list[np.ndarray]:
        """
        Vectorised multi-token lookup for a batch of documents.

        All tokens of all documents in the batch are hashed with a single
        vectorised call, then resolved against the sorted key array. This keeps
        the per-token cost at ~0.05us instead of ~112us.
        """
        out: list[np.ndarray] = []
        if len(self.keys) == 0:
            return [_EMPTY_I32] * len(token_lists)

        flat: list[str] = []
        spans: list[tuple[int, int]] = []
        for toks in token_lists:
            start = len(flat)
            if toks:
                flat.extend(t for t in toks.split() if len(t) >= min_len)
            spans.append((start, len(flat)))

        if not flat:
            return [_EMPTY_I32] * len(token_lists)

        h = _hash_token_list(flat)
        lo = np.searchsorted(self.keys, h, side="left")
        ok = (lo < len(self.keys)) & (self.keys[np.minimum(lo, len(self.keys) - 1)] == h)
        offs = self.offsets
        for a, b in spans:
            if b == a:
                out.append(_EMPTY_I32)
                continue
            sel = lo[a:b][ok[a:b]]
            if len(sel) == 0:
                out.append(_EMPTY_I32)
                continue
            starts = offs[sel]
            counts = offs[sel + 1] - starts
            out.append(self.rows[np.concatenate(
                [np.arange(s, s + c) for s, c in zip(starts, counts)]
            )] if len(sel) else _EMPTY_I32)
        return out

    def posting_len(self, h: int) -> int:
        """Number of documents carrying token hash ``h`` (0 if absent)."""
        if len(self.keys) == 0:
            return 0
        k = np.uint64(h)
        lo = int(np.searchsorted(self.keys, k, side="left"))
        if lo >= len(self.keys) or self.keys[lo] != k:
            return 0
        return int(self.offsets[lo + 1] - self.offsets[lo])

    def rarest(self, token_list: str, min_len: int = 3) -> np.ndarray:
        """
        Postings of the single rarest token in ``token_list``.

        Unioning postings from every token of a document is unusable: mean
        tokens/doc is ~9 and a single common token can carry millions of rows,
        so the union approaches the entire source (measured: 1.4M candidates
        per S1, i.e. essentially every document). Selecting the rarest token
        bounds fan-out to the smallest available bucket while still firing on
        the most discriminative evidence.

        Returns postings sorted by document row.
        """
        if len(self.keys) == 0:
            return _EMPTY_I32
        toks = [t for t in token_list.split() if len(t) >= min_len]
        if not toks:
            return _EMPTY_I32
        h = _hash_token_list(toks)
        lo = np.searchsorted(self.keys, h, side="left")
        safe = np.minimum(lo, len(self.keys) - 1)
        ok = (lo < len(self.keys)) & (self.keys[safe] == h)
        if not ok.any():
            return _EMPTY_I32
        cand = h[ok]
        left = np.searchsorted(self.keys, cand, side="left")
        right = np.searchsorted(self.keys, cand, side="right")
        lens = self.offsets[right] - self.offsets[left]
        best = int(cand[int(np.argmin(lens))])
        return self.get_hash(best)

    def nbytes(self) -> int:
        return int(self.keys.nbytes + self.offsets.nbytes + self.rows.nbytes)


def _hash_one(s: str) -> int:
    return int(pd.util.hash_array(np.array([s], dtype=object), encoding="utf8")[0])


def _hash_token_list(tokens: list[str]) -> np.ndarray:
    """
    Hash a python list of token strings in one vectorised call.

    Per-call ``pd.util.hash_array`` costs ~112us, so hashing token-by-token is
    ~2250x slower and is the difference between seconds and hours on a 5M-row
    source. Always hash in batches.
    """
    if not tokens:
        return np.empty(0, dtype=np.uint64)
    return pd.util.hash_array(np.asarray(tokens, dtype=object), encoding="utf8")


def _flatten_tokens(
    toks_list: list[str],
    min_len: int,
    n_docs_so_far: int,
) -> tuple[list[str], list[tuple[int, int]], int]:
    """
    Flatten a batch of token strings into one list plus per-document spans.

    Returns (flat_tokens, spans, total_docs_seen).
    """
    flat: list[str] = []
    spans: list[tuple[int, int]] = []
    n = n_docs_so_far
    for s in toks_list:
        start = len(flat)
        if s:
            flat.extend(t for t in s.split() if len(t) >= min_len)
        spans.append((start, len(flat)))
        n += 1
    return flat, spans, n


def _unique_within_doc(
    flat: list[str],
    spans: list[tuple[int, int]],
    n_docs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hash flattened tokens and drop repeats inside each document.

    Returns (token_hash uint64[n], doc_index int64[n]) with one entry per
    distinct (token, document) pair.
    """
    h = _hash_token_list(flat)
    counts = np.fromiter((b - a for a, b in spans), dtype=np.int64, count=len(spans))
    doc_of = np.repeat(np.arange(len(spans), dtype=np.int64), counts)
    # combined key that is unique per (token, document)
    stride = np.int64(len(spans) + 1)
    combined = h.astype(np.int64) * stride + doc_of
    _, first = np.unique(combined, return_index=True)
    first.sort()
    return h[first], doc_of[first]


# ---------------------------------------------------------------------------
# PairBuffer: growable candidate accumulator
# ---------------------------------------------------------------------------

class PairBuffer:
    """
    Growable parallel arrays of (s1_row, cand_row, channel_bits).

    Replaces a ``dict[tuple[str, str], int]``: no per-pair Python object,
    no hashing of tuples, deterministic memory, O(1) amortised append.

    Channels are OR-ed into ``bits`` so a pair found by several channels is
    stored once (deduplication) while keeping full provenance.
    """

    __slots__ = ("s1", "cand", "bits", "_n", "_cap", "dedup")

    def __init__(self, initial_capacity: int = 1 << 20) -> None:
        self._cap = int(initial_capacity)
        self.s1 = np.empty(self._cap, dtype=np.int32)
        self.cand = np.empty(self._cap, dtype=np.int32)
        self.bits = np.empty(self._cap, dtype=np.uint8)
        self._n = 0
        # dedup map: combined int64 key -> position, rebuilt lazily via sorted view
        self.dedup: dict[int, int] | None = None

    def __len__(self) -> int:
        return self._n

    def _grow(self) -> None:
        new_cap = self._cap * 2
        s1 = np.empty(new_cap, dtype=np.int32)
        cand = np.empty(new_cap, dtype=np.int32)
        bits = np.empty(new_cap, dtype=np.uint8)
        s1[:self._n] = self.s1[:self._n]
        cand[:self._n] = self.cand[:self._n]
        bits[:self._n] = self.bits[:self._n]
        self.s1, self.cand, self.bits = s1, cand, bits
        self._cap = new_cap

    def add_postings(self, s1_row: int, rows: np.ndarray, bit: int) -> None:
        """Add all ``rows`` as candidates for ``s1_row`` under channel ``bit``."""
        n = len(rows)
        if n == 0:
            return
        if self._n + n > self._cap:
            need = self._n + n
            while self._cap < need:
                self._grow()
        a = self._n
        b = a + n
        self.s1[a:b] = s1_row
        self.cand[a:b] = rows
        self.bits[a:b] = bit
        self._n = b
        if self.dedup is not None:
            self.dedup.clear()

    def consolidated(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return deduplicated (s1, cand, bits) sorted by (s1, cand).

        Deduplication is a single sort over a combined int64 key, which is
        O(n log n) with no Python objects.  Repeated for the same buffer
        instance this is recomputed; call once per flush.
        """
        n = self._n
        if n == 0:
            return _EMPTY_I32, _EMPTY_I32, np.empty(0, np.uint8)
        s1 = self.s1[:n]
        cand = self.cand[:n]
        bits = self.bits[:n]
        key = s1.astype(np.int64) * np.int64(1 << 32) + cand.astype(np.int64)
        order = np.argsort(key, kind="stable")
        key_s = key[order]
        bits_s = bits[order]
        # OR-reduce bits within each unique key group
        uniq, start = np.unique(key_s, return_index=True)
        counts = np.diff(np.append(start, np.int64(len(key_s))))
        out_bits = np.zeros(len(uniq), dtype=np.uint8)
        for b in np.unique(bits_s):
            if b == 0:
                continue
            sel = (bits_s == b)
            grp = np.cumsum(sel) - 1
            np.bitwise_or.at(out_bits, grp, np.uint8(b))
        out_s1 = (uniq >> np.int64(32)).astype(np.int32)
        out_cand = (uniq & np.int64(0xFFFFFFFF)).astype(np.int32)
        self.s1[:n] = out_s1
        self.cand[:n] = out_cand
        self.bits[:n] = out_bits
        return out_s1, out_cand, out_bits

    def reset(self) -> None:
        self._n = 0
        if self.dedup is not None:
            self.dedup.clear()

    def nbytes(self) -> int:
        return int(self.s1.nbytes + self.cand.nbytes + self.bits.nbytes)


# ---------------------------------------------------------------------------
# Parquet streaming helper
# ---------------------------------------------------------------------------

def _iter_batches(path, columns, batch_size) -> Iterator[pa.RecordBatch]:
    pf = pq.ParquetFile(str(path))
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        yield batch


# ---------------------------------------------------------------------------
# Fixed-width string store (memory-mappable)
# ---------------------------------------------------------------------------

class FixedWidthStore:
    """
    Stores a parquet string column as a flat uint8 matrix with fixed row width.

    Enables O(1) random access to any row without creating a Python string
    object, and can be backed by ``np.memmap`` so it lives on disk rather than
    RAM.  Row width is padded to a multiple of 8 bytes for alignment.
    """

    __slots__ = ("data", "offsets", "width", "n_rows")

    def __init__(self, data: np.ndarray, offsets: np.ndarray, width: int) -> None:
        self.data = data
        self.offsets = offsets
        self.width = width
        self.n_rows = len(offsets) - 1

    @classmethod
    def from_parquet(
        cls,
        path,
        column: str,
        out_path=None,
        batch_size: int = 500_000,
        align: int = 8,
    ) -> "FixedWidthStore":
        """
        Materialise ``column`` into a fixed-width uint8 store.
        If ``out_path`` is given the payload is memory-mapped from disk.
        """
        pf = pq.ParquetFile(str(path))
        total = pf.metadata.num_rows

        # first pass: max utf8 width
        max_len = 0
        for batch in _iter_batches(path, [column], batch_size):
            lens = pc.utf8_length(batch.column(column)).to_numpy()
            if len(lens):
                max_len = max(max_len, int(lens.max()))
        width = ((max_len + align - 1) // align) * align
        if width == 0:
            width = align

        offsets = np.zeros(total + 1, dtype=np.int64)
        pos = 0
        for batch in _iter_batches(path, [column], batch_size):
            n = len(batch)
            offsets[pos + 1: pos + 1 + n] = (
                np.arange(1, n + 1, dtype=np.int64) * width
            )
            pos += n
        offsets += offsets[0]

        if out_path is not None:
            mm = np.memmap(str(out_path), dtype=np.uint8, mode="w+",
                           shape=(total, width))
        else:
            mm = np.empty((total, width), dtype=np.uint8)

        pos = 0
        for batch in _iter_batches(path, [column], batch_size):
            col = batch.column(column)
            raw = col.to_pylist()
            n = len(raw)
            for i in range(n):
                s = raw[i]
                if s:
                    b = s.encode("utf-8")
                    L = len(b)
                    if L > width:
                        b = b[:width]
                        L = width
                    mm[pos + i, :L] = np.frombuffer(b, dtype=np.uint8)
            pos += n
            del batch, raw
            gc.collect()

        if out_path is not None:
            mm.flush()
            del mm
            mm = np.memmap(str(out_path), dtype=np.uint8, mode="r",
                           shape=(total, width))
        return cls(mm, offsets, width)

    def row(self, i: int) -> bytes:
        a = int(self.offsets[i])
        b = int(self.offsets[i + 1])
        return self.data[i, : b - a].tobytes()

    def row_str(self, i: int) -> str:
        return self.row(i).decode("utf-8", errors="ignore")

    def nbytes(self) -> int:
        return int(self.data.nbytes + self.offsets.nbytes)
