# Candidate Retrieval — Measured Design Decision (D-002)

Status: DECIDED, evidence-based
Date: 2026-09-26

## 1. Constraints

* Total system RAM ceiling 70% of 15.6 GB; pipeline budget set to 8.61 GB
  (55% of total, measured system baseline ~5.9 GB).
* Disk: 91.2 GB free of 244.1 GB; must keep >= 73.2 GB free, so at most
  ~18 GB is available for the candidate universe.
  At ~20 B/pair (two id strings) that caps the candidate set at roughly
  **500-700M pairs**.
* S1 train = 2,206,821 rows. A budget of 700M pairs therefore allows about
  **300 candidate rows per S1 on average**.

## 2. Measured channel evidence

All numbers on 1,500-4,000 sampled S1 with ground truth (14,823 GT pairs
max), deduplicated candidate counts where stated.

| channel                  | recall | rows/S1      | verdict |
|--------------------------|--------|--------------|---------|
| exact normalized name    | 0.197-0.224 | ~9-10     | cheap, keep |
| exact normalized address | 0.085   | ~0.4         | cheap, keep |
| address digits           | 0.598   | 3,551        | keep, strong |
| union of the 3 exact     | **0.680** | -          | **ceiling for exact-only** |
| digit AND rare token     | 0.572   | 149          | subset of digits; free precision feature |
| union all rare tokens    | 0.999   | 1,401,035    | infeasible |
| rarest rare token        | 0.963   | 9,061        | infeasible (2.0e10 pairs) |
| k=1 token conjunction    | 0.9983  | 1,469,939    | infeasible (3.24e9 pairs) |
| k=2 token conjunction    | 0.9087  | 31,489       | recall too low |
| k=3 token conjunction    | 0.7199  | 1,221        | recall far too low |

## 3. Why the obvious approaches fail

The token document-frequency distribution is extremely skewed (2M S2 docs,
name tokens):

```
df percentiles: p10=1 p25=1 p50=1 p75=1 p90=6 p99=72 max=205,267
```

* **Union of all tokens** is effectively an unfiltered scan (1.4M of 10.3M
  rows retrieved per S1).
* **Rarest token** is usually a singleton, so it is simultaneously useless as
  a block key and (when a moderately common token is the only one present)
  explodes to a 670k-row bucket. `max_df` barely helps because `max_df<=100`
  already retains 99.2% of the vocabulary.
* **Fixed k-conjunction** trades recall for volume monotonically
  (0.998 -> 0.909 -> 0.720) with no point that is both high-recall and small.

## 4. Decision

Adopt an **adaptive conjunctive scheme (prefix filter) with a bucket cap**,
which is the standard approach for this exact skew:

1. Generate block keys as sorted **3-token conjunctions of the S1's combined
   name+address tokens** (k=3 is the selective backbone: p99 bucket = 8,
   max bucket = 13,288).
2. Sort keys by posting length ascending, and **cumulatively consume buckets
   until a per-S1 budget is reached**. This is the prefix-filter/"sorted
   neighborhood" property: adding a key can only add recall, and we stop as
   soon as the budget binds.
3. **Escalate k downward (3 -> 2 -> 1) for the S1 entities whose k=3 evidence
   is weakest**, which is where the residual recall lives. This recovers recall
   for hard S1s while keeping the average budget bounded, because only a
   minority of S1s need escalation.
4. Always union the exact channels (name / address / digits), which cost
   ~3.5k rows/S1 raw but are deduped and provide the highest-precision
   evidence.
5. Cap the digit bucket (currently the single largest contributor) and
   expose its size to the model as a feature rather than blindly expanding it.

**Consequence:** candidate recall is bounded by the budget, so the budget
becomes an explicit, tunable hyperparameter that must be chosen from this
recall/volume curve — not guessed. The operating point is chosen in the next
step by measuring recall on a held-out S1 sample at several budgets.

## 5. Why this preserves the architecture

* Still multi-channel retrieval + union + provenance (channels recorded per
  pair as a bitmask).
* Still recall-then-rank: retrieval sets the ceiling, the model and the
  Macro-F0.5 threshold set precision.
* Still no dense S1 x S2/S3 matrix, and no arbitrary *final* top-K: the budget
  caps the *candidate* set, which is a retrieval-stage concern, while the final
  accept/reject decision remains `score >= threshold`.

## 6. Also fixed: a real data bug

The original normalizer ran `NFKC` + `[^\w\s]`, which NFKC-decomposed complex
scripts and then deleted Unicode combining marks, shredding whole words into
single characters:

```
"हिमाचल प्रदेश"  ->  "ह म चल प रद श"      2 words -> 6 junk tokens
"आदित्य जेलप"    ->  "आद त य ज लप"
```

Fixed by (a) skipping NFKC for protected scripts and (b) preserving combining
marks (Unicode Mn/Mc) in the punctuation strip. Mean tokens per name went
1.00 -> 3.99 after the fix, and all normalized Parquet caches were
regenerated (`NORMALIZATION_VERSION = v2-script-safe`, folded into the cache
fingerprint so stale caches can never be silently reused).
