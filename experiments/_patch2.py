import io
import sys

path = "D:/amazon_ml/src/indexing.py"
with io.open(path, encoding="utf-8") as f:
    src = f.read()

old_sig = '''    def __init__(self, keys, offsets, rows, n_docs, min_len, max_df) -> None:
        self.keys = keys
        self.offsets = offsets
        self.rows = rows
        self.n_docs = n_docs
        self.min_len = min_len
        self.max_df = max_df
        self.n_vocab = len(keys)'''

new_sig = '''    def __init__(self, keys, offsets, rows, n_docs, min_len, max_df) -> None:
        self.keys = keys
        self.offsets = offsets
        self.rows = rows
        self.n_docs = n_docs
        self.min_len = min_len
        self.max_df = max_df
        self.n_vocab = len(keys)
        # Postings grouped by token, so one rare token can explode into a
        # huge candidate set. A caller that unions postings from every token of
        # an S1 therefore retrieves almost the whole source. Callers must
        # select tokens explicitly (e.g. the rarest ones) instead of unioning
        # all of them; ``posting_len`` supports that.'''

if old_sig not in src:
    print("SIG PATTERN NOT FOUND")
    sys.exit(1)
src = src.replace(old_sig, new_sig)

# add posting_len helper + rarest-token selector after get_hash
anchor = '''    def nbytes(self) -> int:
        return int(self.keys.nbytes + self.offsets.nbytes + self.rows.nbytes)


# ---------------------------------------------------------------------------
# PairBuffer'''

helper = '''    def posting_len(self, h: int) -> int:
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
        tokens/doc is ~9, and one common token can carry millions of rows, so
        the union approaches the whole source. Selecting the rarest token
        bounds the fan-out to the smallest available bucket while still
        firing on the most discriminative evidence.
        """
        if len(self.keys) == 0:
            return _EMPTY_I32
        toks = [t for t in token_list.split() if len(t) >= min_len]
        if not toks:
            return _EMPTY_I32
        h = _hash_token_list(toks)
        lo = np.searchsorted(self.keys, h, side="left")
        ok = (lo < len(self.keys)) & (
            self.keys[np.minimum(lo, len(self.keys) - 1)] == h)
        if not ok.any():
            return _EMPTY_I32
        cand = h[ok]
        lens = (self.offsets[
            np.searchsorted(self.keys, cand, side="right")]
            - self.offsets[
                np.searchsorted(self.keys, cand, side="left")])
        best = cand[int(np.argmin(lens))]
        return self.get_hash(int(best))

    def nbytes(self) -> int:
        return int(self.keys.nbytes + self.offsets.nbytes + self.rows.nbytes)


# ---------------------------------------------------------------------------
# PairBuffer'''

if anchor not in src:
    print("ANCHOR PATTERN NOT FOUND")
    sys.exit(1)
src = src.replace(anchor, helper)

with io.open(path, "w", encoding="utf-8") as f:
    f.write(src)
print("patched indexing.py")
