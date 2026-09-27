"""
Vectorised random access to source columns via a fixed-width byte store.

Why
---
Profiling the test-inference loop on a 500k-pair row group gave:

    parquet read               0.02 s
    RowReader S1 (cached)      1.21 s
    RowReader candidate cols  21.90 s     <- 37% of wall clock
    RowReader entity_id        9.48 s     <- 16%
    compute_batch             25.55 s
    predict_proba              2.47 s
    ------------------------
    total                     59.41 s  ->  8,416 pairs/s

``RowReader.take`` returned values through a per-row Python dict lookup, so
every column access cost ~5.5 s per 500k rows. With 3.45e9 candidates that alone
is ~11 hours.

This module stores each column once as a flat uint8 matrix with a fixed row
width (padded to 8 bytes) plus an int64 offset array. A row read is then a
single numpy slice and one ``bytes.decode`` - no dict, no Python loop over rows -
and the whole column can be memory-mapped from disk instead of held in RAM.

Build cost is one streaming pass per column; the result is cached on disk and
reused by every later run, so the cost is paid once.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

ALIGN = 8


class FixedWidthStore:
    """
    Fixed-width uint8 matrix of a string column, plus int64 row offsets.

    Memory is ``n_rows * width`` bytes, so a 5.3M-row column of 128-byte names is
    ~680 MB on disk. Loaded with ``mmap_mode="r"`` it costs no resident RAM
    beyond the pages actually touched, and every access is a contiguous slice.
    """

    __slots__ = ("data", "offsets", "width", "n_rows", "path")

    def __init__(self, data: np.ndarray, offsets: np.ndarray, width: int,
                 path: Path | None = None) -> None:
        self.data = data
        self.offsets = offsets
        self.width = width
        self.n_rows = len(offsets) - 1
        self.path = path

    # -- construction ---------------------------------------------------
    @classmethod
    def build(cls, src: Path, column: str, out_dir: Path,
              batch_size: int = 500_000) -> "FixedWidthStore":
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{src.stem}__{column}.fws"
        out = out_dir / tag
        meta = out.with_suffix(".json")
        n_rows = pq.ParquetFile(str(src)).metadata.num_rows
        if out.exists() and meta.exists():
            try:
                import json
                m = json.loads(meta.read_text(encoding="utf-8"))
                if m.get("n_rows") == n_rows:
                    width = int(m["width"])
                    data = np.memmap(str(out), dtype=np.uint8, mode="r",
                                     shape=(n_rows, width))
                    return cls(data, np.load(meta.with_suffix(".offsets.npy")),
                               width, out)
            except Exception:
                pass

        t0 = time.perf_counter()
        # pass 1: max utf8 BYTE width.
        # pc.utf8_length counts CHARACTERS, not bytes, so it underestimates for
        # multi-byte scripts (Devanagari is 3 bytes/char): using it truncated
        # long non-Latin names. Byte length is ceil(len(chars) * 3) at worst for
        # the scripts present, so measure the encoded length directly.
        max_len = 0
        for rb in pq.ParquetFile(str(src)).iter_batches(
                batch_size=batch_size, columns=[column]):
            for v in rb.column(column).to_pylist():
                if v:
                    L = len(v.encode("utf-8"))
                    if L > max_len:
                        max_len = L
            del rb
        width = max(ALIGN, ((max_len + ALIGN - 1) // ALIGN) * ALIGN)

        mm = np.memmap(str(out), dtype=np.uint8, mode="w+",
                       shape=(n_rows, width))
        pos = 0
        for rb in pq.ParquetFile(str(src)).iter_batches(
                batch_size=batch_size, columns=[column]):
            vals = rb.column(column).to_pylist()
            n = len(vals)
            for i, v in enumerate(vals):
                if not v:
                    continue
                b = v.encode("utf-8")
                L = len(b)
                if L > width:
                    b = b[:width]
                    L = width
                if L:
                    mm[pos + i, :L] = np.frombuffer(b, dtype=np.uint8)
            pos += n
            del rb, vals
            gc.collect()
        mm.flush()
        del mm

        # offsets are implicit: every row occupies exactly `width` bytes, but
        # keeping a trailing offset makes the class usable for a variable-width
        # store later, and costs 8 bytes per row.
        offsets = np.arange(n_rows + 1, dtype=np.int64) * width
        np.save(meta.with_suffix(".offsets.npy"), offsets)
        meta.write_text(
            f'{{"n_rows": {n_rows}, "width": {width}}}', encoding="utf-8")
        data = np.memmap(str(out), dtype=np.uint8, mode="r",
                         shape=(n_rows, width))
        print(f"    store[{src.stem}.{column}] {n_rows:,} rows x {width}B "
              f"= {n_rows * width / 1e6:.0f}MB in {time.perf_counter() - t0:.0f}s",
              flush=True)
        return cls(data, offsets, width, out)

    # -- access ---------------------------------------------------------
    def get(self, rows: np.ndarray) -> list[str]:
        """
        Return the values for the given row ids (order preserved).

        Fixed-width rows are NUL-padded, so the decoded bytes are stripped of
        trailing NULs (and any interior NUL, which cannot occur in valid text).
        """
        if self.n_rows == 0 or len(rows) == 0:
            return [""] * len(rows)
        r = rows.astype(np.int64, copy=False)
        data = self.data
        w = self.width
        # row i occupies bytes [i*w, (i+1)*w) of the flat buffer, which is the
        # same as data[i, 0:w]. Indexing must use the ROW index, not the byte
        # offset: data[byte_offset, ...] returns a single byte and silently
        # truncates the value to its first character.
        idx = r
        out: list[str] = []
        for i in idx.tolist():
            if i < 0 or i >= self.n_rows:
                out.append("")
                continue
            raw = data[i, :w].tobytes()
            z = raw.find(b"\x00")
            if z >= 0:
                raw = raw[:z]
            out.append(raw.decode("utf-8", "ignore"))
        return out

    def get_matrix(self, rows: np.ndarray) -> np.ndarray:
        """
        Gather many rows as one (n, width) uint8 matrix.

        Fancy-indexing the memmap is a single vectorised copy, which is much
        faster than a Python loop and is what the batch feature code uses.
        """
        r = rows.astype(np.int64, copy=False)
        starts = self.offsets[r] // self.width
        ends = self.offsets[r + 1] // self.width
        if len(starts) == 0:
            return np.empty((0, self.width), dtype=np.uint8)
        # rows may repeat; build an index list per output row
        idx = np.concatenate([np.arange(a, b) for a, b in
                              zip(starts.tolist(), ends.tolist())]) \
            if len(starts) else np.empty(0, np.int64)
        return self.data[idx]


class ColumnBundle:
    """The four feature columns plus entity_id for one source, as memmaps."""

    COLS = ("name_norm", "address_norm", "address_dig", "country")

    def __init__(self, src: Path, store_dir: Path,
                 with_ids: bool = True) -> None:
        self.name = FixedWidthStore.build(src, "name_norm", store_dir)
        self.addr = FixedWidthStore.build(src, "address_norm", store_dir)
        self.dig = FixedWidthStore.build(src, "address_dig", store_dir)
        self.ctry = FixedWidthStore.build(src, "country", store_dir)
        self.ids = (FixedWidthStore.build(src, "entity_id", store_dir)
                    if with_ids else None)
        self._cache: dict[tuple, list] = {}

    def take(self, rows: np.ndarray, which: int) -> list:
        store = (self.name, self.addr, self.dig, self.ctry,
                 self.ids)[which]
        if store is None:
            return [""] * len(rows)
        key = (which, int(rows[0]) if len(rows) else -1, len(rows),
               int(rows[-1]) if len(rows) else -1)
        hit = self._cache.get(key)
        if hit is not None and len(hit) == len(rows):
            return hit
        vals = store.get(rows)
        if which == 3:
            vals = [v.strip().casefold() for v in vals]
        if len(self._cache) < 64:
            self._cache[key] = vals
        return vals

    def nbytes(self) -> int:
        tot = 0
        for s in (self.name, self.addr, self.dig, self.ctry, self.ids):
            if s is not None:
                tot += s.data.size
        return tot
