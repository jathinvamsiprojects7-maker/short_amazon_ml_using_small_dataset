"""
Rare-token candidate retrieval (decision D-003).

Rationale (all measured - see reports/RETRIEVAL_DECISION_D003.md):

  * exact name/address/digit channels cap at 0.68 recall on their own;
  * only 0.13% of true pairs share no token at all, so token overlap is a
    near-complete signal;
  * the rarest shared token is highly discriminative, but sorted multi-token
    conjunctions are brittle because addresses are reordered/transliterated.

So retrieval unions the exact channels with rare-token postings, consuming
tokens in ascending order of posting length and stopping at a per-S1 budget.
Because posting length is exactly the token's document frequency, "rarest
first" is the prefix-filter optimal order.

Everything is contiguous numpy; one source is indexed at a time and released
before the next. Candidates are streamed to parquet in int32 row-id space.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.indexing import PostingIndex, hash_str_array
from src.utils import guard

# channel bit flags. The string names are part of the contract with
# src/features.py (prov_rare_token / prov_ngram) and with the candidate
# parquet's "channels" column, so they must stay in sync.
CH_EXACT_NAME = 1
CH_EXACT_ADDR = 2
CH_ADDRESS_DIG = 4
CH_TOKEN = 8
CH_FALLBACK = 16

CHANNEL_NAMES = {
    CH_EXACT_NAME: "exact_name",
    CH_EXACT_ADDR: "exact_address",
    CH_ADDRESS_DIG: "address_digits",
    CH_TOKEN: "rare_token",
    CH_FALLBACK: "ngram",
}

MIN_TOKEN_LEN = 3
SEP = "\x1f"

CAND_SCHEMA = pa.schema([
    ("s1_row", pa.int32()),
    ("cand_row", pa.int32()),
    ("cand_source", pa.int8()),
    ("channels", pa.string()),
    ("n_channels", pa.int8()),
])


def _bits_to_str(bits: int) -> str:
    return "|".join(nm for bit, nm in CHANNEL_NAMES.items() if bits & bit)


def n_channels(bits: int) -> int:
    return int(bin(bits).count("1"))


# ---------------------------------------------------------------------------
# Rare-token index over a single field
# ---------------------------------------------------------------------------

class RareTokenIndex:
    """
    Inverted index over tokens, restricted to tokens with df <= max_df.

    Built with two streaming passes (df, then postings), fully vectorised per
    batch. 465M postings would be far too large, so ``max_df`` is the primary
    volume control: it removes the handful of tokens whose buckets alone can
    cover a large fraction of the source.
    """

    __slots__ = ("keys", "offsets", "rows", "max_df", "n_vocab", "n_postings")

    def __init__(self, keys, offsets, rows, max_df: int) -> None:
        self.keys = keys
        self.offsets = offsets
        self.rows = rows
        self.max_df = max_df
        self.n_vocab = len(keys)
        self.n_postings = len(rows)

    @classmethod
    def build(cls, path: Path, column: str, max_df: int,
              min_len: int = MIN_TOKEN_LEN, batch_size: int = 500_000) -> "RareTokenIndex":
        from src.indexing import _flatten_tokens, _hash_token_list, _unique_within_doc

        g = guard()
        started = time.perf_counter()

        # ---- pass 1: document frequency ----
        df_counts: dict[int, int] = {}
        for batch in _iter(path, column, batch_size):
            flat, spans, _ = _flatten_tokens(batch.column(column).to_pylist(),
                                             min_len, 0)
            if flat:
                toks, _docs = _unique_within_doc(flat, spans, len(batch))
                for t in np.unique(toks).tolist():
                    df_counts[t] = df_counts.get(t, 0) + 1
            del batch, flat, spans
            g.gc(1)

        allowed = {h for h, c in df_counts.items() if c <= max_df}
        del df_counts
        g.gc()

        if not allowed:
            return cls(np.empty(0, np.uint64), np.zeros(1, np.int64),
                       np.empty(0, np.int32), max_df)

        # ---- pass 2: postings ----
        tp: list[np.ndarray] = []
        rp: list[np.ndarray] = []
        base = 0
        for batch in _iter(path, column, batch_size):
            flat, spans, _ = _flatten_tokens(batch.column(column).to_pylist(),
                                             min_len, 0)
            if flat:
                toks, docs = _unique_within_doc(flat, spans, len(batch))
                arr = np.fromiter((int(t) in allowed for t in toks.tolist()),
                                  dtype=bool, count=len(toks))
                if arr.any():
                    tp.append(toks[arr].astype(np.uint64))
                    rp.append((base + docs[arr]).astype(np.int32))
            base += len(batch)
            del batch, flat, spans
            g.gc(1)

        if not tp:
            return cls(np.empty(0, np.uint64), np.zeros(1, np.int64),
                       np.empty(0, np.int32), max_df)

        keys = np.concatenate(tp)
        rows = np.concatenate(rp)
        del tp, rp
        g.gc()
        order = np.argsort(keys, kind="stable")
        keys = keys[order]
        rows = rows[order]
        del order
        g.gc()
        uniq, start = np.unique(keys, return_index=True)
        counts = np.diff(np.append(start, np.int64(len(keys))))
        max_bucket = int(counts.max()) if len(counts) else 0
        offsets = np.zeros(len(uniq) + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        out = cls(uniq, offsets, rows, max_df)
        del counts, uniq
        g.gc()
        print(f"    rare[{column}] max_df={max_df:,} -> vocab {out.n_vocab:,} "
              f"postings {out.n_postings:,} max bucket {max_bucket:,} "
              f"{time.perf_counter()-started:.0f}s | {g.status()}", flush=True)
        return out

    # -- query ----------------------------------------------------------
    def postings_sorted_cached(
        self,
        token_hashes: np.ndarray,
        token_texts: list[str],
    ) -> list[tuple[np.ndarray, int]]:
        """
        Postings for a document's tokens, ordered by ascending posting length.

        ``token_hashes`` must be the *deduplicated, order-preserving* hashes of
        the document's tokens (see :func:`hash_document_tokens`). Hashing is
        separated from lookup because profiling showed ~57% of retrieval time
        was re-hashing the same token strings once per token index.

        Ordering by posting length is what makes "rarest token first" the
        prefix-filter-optimal order: a fixed candidate budget is spent on the
        most discriminative evidence available.
        """
        if len(self.keys) == 0 or len(token_hashes) == 0:
            return []
        h = np.asarray(token_hashes, dtype=np.uint64)
        lo = np.searchsorted(self.keys, h, side="left")
        safe = np.minimum(lo, len(self.keys) - 1)
        ok = (lo < len(self.keys)) & (self.keys[safe] == h)
        if not ok.any():
            return []
        cand = h[ok]
        left = np.searchsorted(self.keys, cand, side="left")
        right = np.searchsorted(self.keys, cand, side="right")
        lens = self.offsets[right] - self.offsets[left]
        order = np.argsort(lens, kind="stable")
        offs = self.offsets
        return [(self.rows[int(offs[l]):int(offs[r])], int(n))
                for l, r, n in zip(left[order], right[order], lens[order])]

    def postings_sorted(self, token_list: str, min_len: int = MIN_TOKEN_LEN):
        """Convenience wrapper that hashes ``token_list`` first."""
        hashes, _texts = hash_document_tokens(token_list, min_len)
        return self.postings_sorted_cached(hashes, [])

    # -- vectorised batch query ------------------------------------------
    def postings_for_hashes(self, hashes: np.ndarray) -> list[tuple[np.ndarray, int]]:
        """Postings for many token hashes at once, in ascending length order."""
        if len(self.keys) == 0 or len(hashes) == 0:
            return []
        h = np.asarray(hashes, dtype=np.uint64)
        lo = np.searchsorted(self.keys, h, side="left")
        safe = np.minimum(lo, len(self.keys) - 1)
        ok = (lo < len(self.keys)) & (self.keys[safe] == h)
        if not ok.any():
            return []
        cand = h[ok]
        left = np.searchsorted(self.keys, cand, side="left")
        right = np.searchsorted(self.keys, cand, side="right")
        lens = self.offsets[right] - self.offsets[left]
        order = np.argsort(lens, kind="stable")
        offs = self.offsets
        return [(self.rows[int(offs[l]):int(offs[r])], int(n))
                for l, r, n in zip(left[order], right[order], lens[order])]

    def hash_tokens(self, text: str, min_len: int = MIN_TOKEN_LEN) -> np.ndarray:
        from src.indexing import _hash_token_list
        if not text:
            return np.empty(0, dtype=np.uint64)
        toks = [t for t in text.split() if len(t) >= min_len]
        if not toks:
            return np.empty(0, dtype=np.uint64)
        return _hash_token_list(toks)

    def nbytes(self) -> int:
        return int(self.keys.nbytes + self.offsets.nbytes + self.rows.nbytes)

    # -- persistence ----------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, keys=self.keys, offsets=self.offsets, rows=self.rows,
                 meta=np.array([self.max_df, self.n_vocab, self.n_postings],
                               dtype=np.int64))

    @classmethod
    def load(cls, path: Path, max_df: int) -> "RareTokenIndex | None":
        """Load a cached index, or return None if absent/incompatible."""
        p = Path(path)
        if not p.exists():
            return None
        try:
            z = np.load(p)
            meta = z["meta"]
            if int(meta[0]) != int(max_df):
                return None
            return cls(z["keys"], z["offsets"], z["rows"], int(meta[0]))
        except Exception:
            return None

    @classmethod
    def build_or_load(cls, path: Path, column: str, max_df: int,
                      cache_dir: Path | None = None,
                      min_len: int = MIN_TOKEN_LEN,
                      batch_size: int = 500_000) -> "RareTokenIndex":
        """
        Load a cached index when available, else build and cache it.

        Building a rare-token index costs ~50-70s per field per source, and the
        same index is reused by every experiment and by the train and test
        runs, so caching removes minutes of repeated work.
        """
        if cache_dir is not None:
            tag = f"{Path(path).stem}__{column}__df{max_df}.npz"
            cpath = Path(cache_dir) / tag
            got = cls.load(cpath, max_df)
            if got is not None:
                print(f"    rare[{column}] cache hit ({tag}) "
                      f"vocab={got.n_vocab:,} postings={got.n_postings:,}",
                      flush=True)
                return got
            built = cls.build(path, column, max_df, min_len, batch_size)
            try:
                built.save(cpath)
            except Exception as exc:      # cache is an optimisation only
                print(f"    (index cache write failed: {exc})", flush=True)
            return built
        return cls.build(path, column, max_df, min_len, batch_size)

    # -- vectorised batch lookup ---------------------------------------
    def lookup(self, hashes: np.ndarray):
        """
        Resolve many token hashes at once against the CSR postings.

        Returns ``(positions, lengths)`` where ``positions`` is the flat index
        of the first posting of each *present* hash and ``lengths`` the number
        of postings. Empty/absent hashes are dropped, so the caller never sees
        them. This is the vectorised equivalent of one dict lookup per token
        and is what keeps the retrieval stage CPU-bound instead of
        interpreter-bound.
        """
        if len(self.keys) == 0 or len(hashes) == 0:
            e = np.empty(0, np.int64)
            return e, e
        h = np.asarray(hashes, dtype=np.uint64)
        lo = np.searchsorted(self.keys, h, side="left")
        safe = np.minimum(lo, len(self.keys) - 1)
        ok = (lo < len(self.keys)) & (self.keys[safe] == h)
        if not ok.any():
            e = np.empty(0, np.int64)
            return e, e
        lo = lo[ok]
        starts = self.offsets[lo]
        lens = self.offsets[lo + 1] - starts
        keep = lens > 0
        return starts[keep], lens[keep]

    def rows_for(self, starts: np.ndarray, lens: np.ndarray) -> np.ndarray:
        """Gather the posting rows for the given (start, length) pairs."""
        total = int(lens.sum())
        if total == 0:
            return np.empty(0, np.int32)
        cum = np.cumsum(lens) - lens
        idx = (np.arange(total, dtype=np.int64)
               - np.repeat(cum, lens) + np.repeat(starts, lens))
        return self.rows[idx]


def _iter(path: Path, column: str, batch_size: int):
    pf = pq.ParquetFile(str(path))
    for batch in pf.iter_batches(batch_size=batch_size, columns=[column]):
        yield batch


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalBudget:
    """
    Per-S1 candidate budget, split by channel.

    The split matters: measured on real data, the exact/digit channels alone
    emit ~1,697 candidates per S1 (digit bucket p50=178, p90=7,879,
    p99=24,080), which would consume any shared budget before the
    discriminative rare-token evidence is ever used. The digit channel in
    particular is high-recall but very low-precision, so it gets a small
    dedicated share and the bulk of the budget goes to rare tokens.
    """
    per_s1: int = 200
    digit_cap: int = 100
    exact_cap: int = 100
    max_token_df: int = 1000
    #: A token whose posting list is longer than this is skipped by retrieval.
    #: It could never fit inside ``per_s1`` candidates, so expanding it is pure
    #: waste - and one such token can carry ~500k rows.
    token_bucket_cap: int = 5000

    def as_dict(self) -> dict:
        return {
            "per_s1": self.per_s1,
            "digit_cap": self.digit_cap,
            "exact_cap": self.exact_cap,
            "max_token_df": self.max_token_df,
            "token_bucket_cap": self.token_bucket_cap,
        }


# ---------------------------------------------------------------------------
# Chunk retrieval
# ---------------------------------------------------------------------------

def hash_document_tokens(text: str, min_len: int = MIN_TOKEN_LEN):
    """
    Hash a document's tokens once, deduplicated, order-preserving.

    Profiling the retrieval loop showed ~57% of its runtime was
    ``pd.util.hash_array`` being called once per (S1, token index) pair. The
    hash of a token does not depend on the index, so it is computed once here
    and reused across the name and address indexes.
    """
    if not text:
        return np.empty(0, dtype=np.uint64), []
    seen: set[str] = set()
    toks: list[str] = []
    for t in text.split():
        if len(t) >= min_len and t not in seen:
            seen.add(t)
            toks.append(t)
    if not toks:
        return np.empty(0, dtype=np.uint64), []
    from src.indexing import _hash_token_list
    return _hash_token_list(toks), toks


def hash_chunk_tokens(
    texts: list[str],
    min_len: int = MIN_TOKEN_LEN,
) -> tuple[list[np.ndarray], list[list[str]]]:
    """
    Hash every token of every document in a chunk with a single vectorised
    call, then split the result back per document.

    Profiling showed ``pd.util.hash_array`` dominated retrieval (~10s of 18s
    for a 20k chunk) purely because it was invoked once per document, each
    time on a list of a handful of strings. One call over the whole chunk
    amortises the pandas overhead and is ~40x faster.

    Returns ``(hashes_per_doc, tokens_per_doc)``, deduplicated per document and
    order-preserving.
    """
    from src.indexing import _hash_token_list

    n = len(texts)
    hashes: list[np.ndarray] = [np.empty(0, dtype=np.uint64)] * n
    toks_out: list[list[str]] = [[] for _ in range(n)]

    flat: list[str] = []
    counts = np.zeros(n, dtype=np.int64)
    per_doc: list[list[str]] = []
    for i, t in enumerate(texts):
        seen: set[str] = set()
        td: list[str] = []
        if t:
            for x in t.split():
                if len(x) >= min_len and x not in seen:
                    seen.add(x)
                    td.append(x)
        per_doc.append(td)
        flat.extend(td)
        counts[i] = len(td)
    if not flat:
        return hashes, toks_out

    all_h = _hash_token_list(flat)
    ends = np.cumsum(counts)
    starts = ends - counts
    for i in range(n):
        a, b = int(starts[i]), int(ends[i])
        if b > a:
            hashes[i] = all_h[a:b]
            toks_out[i] = per_doc[i]
    return hashes, toks_out


def retrieve_chunk(
    s1_block: pd.DataFrame,
    ix_name: PostingIndex,
    ix_addr: PostingIndex,
    ix_dig: PostingIndex,
    token_idx: list[RareTokenIndex],
    budget: RetrievalBudget,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retrieve candidates for a chunk of S1 rows. Returns int32 row ids."""
    hn = hash_str_array(_obj(s1_block["name_norm"]))
    ha = hash_str_array(_obj(s1_block["address_norm"]))
    hd = hash_str_array(_obj(s1_block["address_dig"]))
    names = s1_block["name_norm"].tolist()
    addrs = s1_block["address_norm"].tolist()
    n_rows = len(s1_block)
    cap = budget.per_s1

    # Hash every token in the chunk with one vectorised call, then reuse the
    # result across both token indexes.
    name_h, _ = hash_chunk_tokens(names)
    addr_h, _ = hash_chunk_tokens(addrs)
    tok_cache: list = [
        (name_h[i], addr_h[i]) if (len(name_h[i]) or len(addr_h[i])) else None
        for i in range(n_rows)
    ]

    out_s1: list[np.ndarray] = []
    out_cand: list[np.ndarray] = []
    out_bits: list[np.ndarray] = []

    for i in range(n_rows):
        cm: dict[int, int] = {}

        # --- high-precision exact channels, tightly capped ---
        exact_cap = budget.exact_cap
        if hn[i]:
            rows = ix_name.get(int(hn[i]))
            if len(rows) > exact_cap:
                rows = rows[:exact_cap]
            for r in rows.tolist():
                cm[r] = cm.get(r, 0) | CH_EXACT_NAME
        if ha[i]:
            rows = ix_addr.get(int(ha[i]))
            if len(rows) > exact_cap:
                rows = rows[:exact_cap]
            for r in rows.tolist():
                cm[r] = cm.get(r, 0) | CH_EXACT_ADDR

        # --- digits: high recall, low precision, own small cap ---
        if hd[i]:
            rows = ix_dig.get(int(hd[i]))
            if len(rows) > budget.digit_cap:
                rows = rows[:budget.digit_cap]
            for r in rows.tolist():
                cm[r] = cm.get(r, 0) | CH_ADDRESS_DIG

        # --- rare tokens get the remaining budget, rarest first ---
        for ti in token_idx:
            if len(cm) >= cap:
                break
            entry = tok_cache[i]
            if entry is None:
                break
            for hashes in entry:
                if hashes is None or len(hashes) == 0:
                    continue
                for rows, ln in ti.postings_sorted_cached(hashes, []):
                    if len(cm) >= cap:
                        break
                    if ln > budget.token_bucket_cap:
                        # too large to ever fit in the budget: skip it rather
                        # than expanding hundreds of thousands of rows
                        continue
                    for r in rows.tolist():
                        if len(cm) >= cap:
                            break
                        if r not in cm:
                            cm[r] = CH_TOKEN
                    if len(cm) >= cap:
                        break
                if len(cm) >= cap:
                    break

        if cm:
            m = len(cm)
            out_s1.append(np.full(m, i, dtype=np.int32))
            out_cand.append(np.fromiter(cm.keys(), dtype=np.int32, count=m))
            out_bits.append(np.fromiter(cm.values(), dtype=np.uint8, count=m))

    if not out_s1:
        return (np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.uint8))
    return (np.concatenate(out_s1), np.concatenate(out_cand), np.concatenate(out_bits))


def _obj(series) -> np.ndarray:
    """Plain object-dtype array of strings (pandas 3 returns Arrow arrays)."""
    return np.asarray(series.tolist(), dtype=object)


def _csr_gather(starts: np.ndarray, lens: np.ndarray, rows: np.ndarray,
                cap_each: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Expand (start, length) CSR spans into a flat row array plus a per-group id,
    truncating any span longer than ``cap_each``.
    """
    lens = np.asarray(lens, dtype=np.int64)
    if cap_each and len(lens) and int(lens.max()) > cap_each:
        lens = np.minimum(lens, cap_each)
    total = int(lens.sum())
    if total == 0:
        return np.empty(0, np.int32), np.empty(0, np.int32)
    cum = np.cumsum(lens) - lens
    idx = (np.arange(total, dtype=np.int64) - np.repeat(cum, lens)
           + np.repeat(starts, lens))
    return rows[idx], np.repeat(np.arange(len(lens), dtype=np.int32), lens)


def _rank_in_group(sorted_ids: np.ndarray) -> np.ndarray:
    """0-based position of each element inside its run of equal values."""
    n = len(sorted_ids)
    if n == 0:
        return np.empty(0, np.int64)
    change = np.empty(n, dtype=bool)
    change[0] = True
    np.not_equal(sorted_ids[1:], sorted_ids[:-1], out=change[1:])
    grp_start = np.flatnonzero(change)
    grp_id = np.cumsum(change) - 1
    return np.arange(n, dtype=np.int64) - grp_start[grp_id]


def _expand_keys(keys: np.ndarray, index: PostingIndex, cap_each: int,
                 bit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resolve a whole chunk of hashed keys against one PostingIndex at once.

    Returns ``(owner, cand, bit)`` where ``owner`` is the S1 row index within
    the chunk and ``bit`` is the channel flag repeated per candidate.
    """
    empty = np.empty(0, np.int32)
    if len(index.keys) == 0 or len(keys) == 0:
        return empty, empty, np.empty(0, np.uint8)
    nz = np.flatnonzero(keys != 0)
    if nz.size == 0:
        return empty, empty, np.empty(0, np.uint8)
    k = keys[nz]
    lo = np.searchsorted(index.keys, k, side="left")
    safe = np.minimum(lo, len(index.keys) - 1)
    ok = (lo < len(index.keys)) & (index.keys[safe] == k)
    if not ok.any():
        return empty, empty, np.empty(0, np.uint8)
    lo = lo[ok]
    owner_key = nz[ok]
    starts = index.offsets[lo]
    lens = index.offsets[lo + 1] - starts
    keep = lens > 0
    owner_key, starts, lens = owner_key[keep], starts[keep], lens[keep]
    if len(lens) == 0:
        return empty, empty, np.empty(0, np.uint8)
    cand, group_local = _csr_gather(starts, lens, index.rows, cap_each)
    return (owner_key[group_local].astype(np.int32), cand.astype(np.int32),
            np.full(len(cand), bit, dtype=np.uint8))


class CandidateWriter:
    """Streams candidate pairs (int32 row ids) to parquet."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._w: pq.ParquetWriter | None = None
        self.n_written = 0

    def write(self, s1_rows: np.ndarray, cand_rows: np.ndarray,
              bits: np.ndarray, source_id: int) -> None:
        n = len(s1_rows)
        if n == 0:
            return
        table = pa.table({
            "s1_row": pa.array(s1_rows, type=pa.int32()),
            "cand_row": pa.array(cand_rows, type=pa.int32()),
            "cand_source": pa.array(np.full(n, source_id, dtype=np.int8),
                                    type=pa.int8()),
            "channels": pa.array([_bits_to_str(int(b)) for b in bits],
                                 type=pa.string()),
            "n_channels": pa.array(
                np.fromiter((n_channels(int(b)) for b in bits),
                            dtype=np.int8, count=n), type=pa.int8()),
        }, schema=CAND_SCHEMA)
        if self._w is None:
            self._w = pq.ParquetWriter(str(self.path), schema=CAND_SCHEMA,
                                       compression="snappy")
        self._w.write_table(table)
        self.n_written += n

    def close(self) -> None:
        if self._w is not None:
            self._w.close()
            self._w = None

    def __enter__(self) -> "CandidateWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()
