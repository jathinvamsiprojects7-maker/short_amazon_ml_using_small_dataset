# Amazon ML Challenge 2026 — Verified Pipeline State

All figures below come from logs, artifacts and metrics on disk. Nothing is
claimed because a file merely exists.

## Machine and resource policy

| item | value |
|---|---|
| RAM | 15.6 GB |
| logical cores | 12 |
| disk D: total / usable | 244 GB / ~48 GB (20% floor kept free) |
| **hard ceiling on TOTAL system usage** | **80%** |
| working ceiling (`ResourceGuard`) | 80% of total, with throttling before it |

`src/utils.py::ResourceGuard` measures **total** system RAM
(`psutil.virtual_memory().percent`), refuses allocations that would cross the
ceiling, shrinks chunk sizes under pressure, and rejects any configured
ceiling above 0.80.

## Completed stages (with evidence)

### 1. Normalization — regenerated, verified
Fixed a real data bug: the original `NFKC` + `[^\w\s]` pipeline decomposed
complex scripts and deleted Unicode combining marks, shredding Devanagari names
(`"हिमाचल प्रदेश"` → `"ह म चल प रद श"`, 2 words → 6 junk tokens).

| split | source | rows |
|---|---|---|
| train | source1 | 2,206,821 |
| train | source2 | 5,034,616 |
| train | source3 | 5,285,603 |
| test | source1 | 1,732,544 |
| test | source2 | 4,887,273 |
| test | source3 | 5,082,316 |

Mean tokens per normalized name went **1.00 → 3.99** after the fix.
Cache fingerprints include `NORMALIZATION_VERSION = v2-script-safe`, so stale
caches can never be silently reused.

### 2. Candidate retrieval — complete

Channels: exact name, exact address, address digits, rare-token postings
(consumed rarest-first under a per-S1 budget), with the channel set recorded per
pair as a bitmask.

Budget: `per_s1=1000`, `exact_cap=200`, `digit_cap=200`, `max_df=1000`,
`token_bucket_cap=5000`.

| split | source | candidate pairs | pairs/S1 | file |
|---|---|---|---|---|
| train | S2 | 2,204,614,412 | 999.0 | 10.80 GB |
| train | S3 | 2,204,532,688 | 998.9 | 10.69 GB |
| test | S2 | 1,724,655,328 | 995.5 | 8.26 GB |
| test | S3 | *in progress* | | |

**Measured candidate recall (train, against ground truth): 0.7462**
(5,699,422 of 7,638,365 GT pairs retrieved). This is the ceiling on final
recall and the single most important limitation of the current solution.

### 3. Training pairs — complete
94.0M sampled pairs (all 5.70M positives + ≤40 negatives/S1), S1-grouped split
stratified by match cardinality:

```
train 1,875,797 S1  ->  15,583,997 pairs
val     331,024 S1  ->   2,143,579 pairs
```

The split is a lookup on the pair's `s1_row`, so no pair can leak across sides.

### 4. Features — complete and verified
41 features, 94.0M × 41 float32 memmap (15.4 GB), computed in **76 min** at
~19.9k pairs/s.

`experiments/test_features.py` asserts the vectorised implementation matches
the per-pair reference `src/features.compute_features` **feature-for-feature
(max diff ≤ 1e-4)** on randomised inputs including empty names/addresses.

### 5. Model ladder + threshold — complete

| model | val Macro F0.5 | threshold | micro P | micro R |
|---|---|---|---|---|
| L0 deterministic | 0.1423 | 0.2773 | 0.0347 | 0.0363 |
| L1 logistic regression | 0.7592 | 0.5955 | 1.0000 | 0.5140 |
| **L2 LightGBM** | **0.7595** | **0.7136** | 0.9993 | 0.5147 |

L1 and L2 are statistically indistinguishable; L2 is selected.

Per-segment Macro F0.5 at the chosen threshold:

| segment | S1 | Macro F0.5 | P | R |
|---|---|---|---|---|
| zero-match | 232,137 | 0.7750 | 0.9992 | 0.5274 |
| singleton | 16,481 | 0.7993 | 0.9993 | 0.5480 |
| multi-match | 82,406 | 0.7077 | 0.9993 | 0.4809 |

## Known limitations (measured, not assumed)

1. **Candidate recall 0.7462** caps final recall. The residual 25% needs a
   better blocking key: rare-token retrieval needs `max_df ≈ 5000` for 0.96
   recall, which costs ~9.6e9 pairs.
2. **Recall 0.51 is partly a threshold effect**, not only a retrieval ceiling.
   Validation was negative-thinned 2:1, so the tuned threshold is conservative;
   at inference the true ratio is ~1 positive per 771 candidates.
3. The validation P≈0.999 is optimistic because of that thinning.

## Bugs found and fixed (each verified)

| bug | impact |
|---|---|
| dense `Q @ matrix.T` (`candidate_gen.py:159`) | 40 GB allocation → OOM. Replaced with array-based indexes (85 MB vs ~1.5 GB) and sparse retrieval. |
| NFKC + `[^\w\s]` destroyed Devanagari | whole-word tokens destroyed |
| per-token `hash_array` (112 µs vs 0.05 µs) | 2000× slowdown; token index build >900 s → 29 s |
| `process.cdist` returns a full cross-matrix | wrong shape; replaced with `cpdist` |
| both-empty sentinel divided by 100 | 1.0 → 0.01 on every empty pair |
| `k.sort()` missing, only `o` sorted | every positive lookup returned False |
| worker returning 20M-row arrays | ~17 GB IPC; pool slower than 1 worker |
| `digit_count_diff` using set size | repeated digit groups undercounted |
