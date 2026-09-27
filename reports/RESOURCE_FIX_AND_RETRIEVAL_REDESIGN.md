# RESOURCE FIX + RETRIEVAL REDESIGN — Evidence & Decision Record

Date: 2026-09-26
Status: IMPLEMENTING

## 1. Incident

`run_pipeline.py` crashed during S2 candidate retrieval:

```
[timer] exact idx [S2]: 36.09s
[timer] ngram fit+transform (5,034,616 docs): 49.14s
  Ngram matrix: (5034616, 30000)  nnz=120,084,848
[after S2 index build] RAM 13.22GB/15.6GB (84.5%)
[timer] S2 retrieval: 146.73s
MemoryError  (candidate_gen.py:467, pair_bitmap)
```

Total system RAM reached ~84.5% (and ~90% per user observation), above the
70% hard ceiling. The run was terminated by the OS.

## 2. Root cause (measured, not assumed)

Primary cause — **dense all-pairs similarity matrix**, which the architecture
explicitly forbids ("Never construct a dense S1 x S2/S3 similarity matrix"):

`src/candidate_gen.py:159` (original)
```python
scores = (Q @ self._matrix.T).toarray()   # n_query x 5,034,616 float
```

With `ngram_batch_size=2000` and S2 = 5,034,616 rows this is
`2000 x 5,034,616 x 4B = 40.3 GB` **per batch**.

Measured on this machine (400,000-column reference):
| batch | densified chunk |
|-------|-----------------|
| 250   | 400 MB          |
| 500   | 800 MB          |
| 1000  | 1600 MB         |
| 2000  | 3200 MB (x 5.03M cols -> 40 GB) |

Contributing causes:

1. `NgramIndex.build()` loaded the full `name_norm` column as a Python list of
   5.03M strings (`df[text_col].fillna("").tolist()`) plus kept
   `self._entity_ids = list(df["entity_id"])` — ~2 GB of Python string objects
   duplicated from the DataFrame.
2. `InvertedIndex._index: dict[str, list[str]]` for name/address/digits.
   Measured key counts: name 4,030,166 / addr 4,286,083 / dig 836,355.
   ~9.2M Python dict entries, each holding a separate list object
   (~56-64 B list header + 8 B/element + 49 B per interned-ish string).
   Estimated 2.5-3.5 GB for S2 alone; S3 is larger still.
3. `MultiTokenIndex` built two more `dict[str, list[str]]` maps
   (rare name tokens, rare address tokens) on top.
4. `pair_bitmap: dict[tuple[str, str], int]` — a tuple key per candidate pair.
   A tuple is ~64 B plus two pointers; with 5000 S1/chunk and large
   rare-token buckets this reached hundreds of MB per chunk and then
   multiplied by 4 candidate columns in the final list comprehensions.
5. `merge_candidate_parquets()` did `pd.read_parquet` of both full candidate
   files, `pd.concat`, then `drop_duplicates` — three full copies.
6. The full S1 DataFrame (2.2M x 9 object columns) stayed resident (~2.5 GB).

Combined steady-state footprint exceeded the 15.6 GB machine.

## 3. Measured retrieval-signal evidence (4000 GT S1 sample, 14,823 GT pairs)

Exact-channel coverage of ground-truth positives:

| channel        | recall |
|----------------|--------|
| exact name     | 0.2201 |
| exact address  | 0.0853 |
| exact digits   | 0.5979 |
| **union of 3** | **0.6903** |
| residual       | 0.3097 |

**31% of true matches are invisible to exact blocking.** Any design that relies
on exact keys alone caps final recall at ~0.69 and cannot be competitive.

Residual pair characteristics (the 31% that exact channels miss):

```
name both present        : 1.000
address both present     : 0.900
name_ratio    p10/p25/p50/p75  : 16 / 66 / 82 / 90
name_tokset   p10/p25/p50/p75  : 17 / 80 / 94 / 100
addr_ratio    p10/p25/p50/p75  : 45 / 59 / 77 / 86
frac name_ratio >= 70/80/90    : 0.717 / 0.559 / 0.284
frac name_tokset >= 90/95      : 0.618 / 0.473
frac addr_ratio >= 60/80       : 0.745 / 0.445
```

Conclusion: the residual is **fuzzy but strongly bounded**. Most residual pairs
still have high name or address similarity. This means a bounded
*similarity-thresholded* channel (not a fixed top-K) is the correct recall
mechanism — and it must be cheap.

## 4. Fix design (evidence-driven)

### 4.1 Eliminate the dense matrix (the OOM)

Replace TF-IDF n-gram `toarray()` retrieval with a **sparse CSR top-k** that
never materialises a dense block. Verified correct against a dense reference:

```
sparse_topn bs=500  -> 0.75s, 40000 top-k entries
row-by-row comparison vs dense argsort: match True (all rows)
```

Memory for a chunk is then `O(nnz(C))`, not `O(batch x n_docs)`.

### 4.2 Eliminate all Python-dict indexes

New `src/indexing.py` replaces `dict[str, list[str]]` with contiguous numpy
arrays:

- `PostingIndex`   — `uint64 key -> sorted int32 rows` via
  `argsort` + `searchsorted`. For S2 name keys this is
  `4.03M*8B keys + 4.03M*8B offsets + 5.03M*4B rows = 84 MB`
  vs ~1.5 GB for the dict-of-lists.
- `TokenIndex`     — CSR `(offsets, rows)` over hashed tokens.
- `PairBuffer`     — growable `int32 s1 / int32 cand / uint8 bits`,
  deduplicated by one `argsort` on a combined `int64` key.
  `12 B/pair` vs ~150 B/pair for a tuple-keyed dict.

Keys are stored as **uint64 hashes** rather than strings. For 5.3M keys the
collision probability is ~7e-7, and a hash collision can only *add* a
candidate (never remove a true match), which the model then scores. This is
benign for a recall-then-rank architecture and removes all string-object
overhead. Empty strings map to reserved hash `0` and are never indexed, so a
missing address can never match another missing address.

### 4.3 Use RapidFuzz cdist for the fuzzy channel

Measured on this machine:
```
process.cdist 5,000 x 30,000, fuzz.ratio, workers=4 -> 0.90s
= 1.67e8 pairs/s
```
This is C++ SIMD, needs **no index at all**, and is faster per pair than
building a TF-IDF vocabulary over 5M documents. Since residual recall needs
similarity *scores* (to threshold), not just top-K, this is a better fit than
char-n-gram IDF retrieval.

### 4.4 Stream, never materialise

- Chunked parquet reads via `iter_batches` (no full column into RAM).
- S2 and S3 retrieved sequentially; S2 indexes freed before S3 is built.
- Candidate output streamed to parquet per chunk with a `ParquetWriter`.
- No `pd.concat` of full candidate files; merge is a streaming row-group pass.

## 5. Resource envelope after the fix (target)

| stage                | peak target |
|----------------------|-------------|
| S1 load              | < 2.5 GB    |
| one source index     | < 1.5 GB    |
| retrieval chunk      | < 300 MB    |
| **total system**     | **<= ~9.5 GB / 15.6 GB (61%)** |

Ceiling enforced by a runtime guard that shrinks chunk sizes and pauses
before total system RAM can exceed the configured fraction.

## 6. What is NOT changing

- `FINAL_ARCHITECTURE.txt` pipeline order is preserved.
- Multi-channel retrieval + union + provenance is preserved.
- No top-K final decision; the threshold + model decides.
- Cached preprocessing artefacts (`train_source*_norm.parquet`,
  `train_ground_truth.json`) are NOT regenerated.

## 7. New channel evidence requirement

Because exact channels cap at 0.69, the added similarity channel must be
measured for recall and volume before it is accepted. That measurement is the
next gate.
