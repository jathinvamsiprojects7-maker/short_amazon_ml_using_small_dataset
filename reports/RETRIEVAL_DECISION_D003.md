"""
Token-retrieval design — evidence-based decision record (D-003, supersedes D-002)

Date: 2026-09-26

## Summary of the evidence

Measured on 1,000-4,000 sampled S1 entities with ground truth.

### 1. Exact channels are a hard ceiling of 0.68

| channel          | recall |
|------------------|--------|
| exact name       | 0.197-0.224 |
| exact address    | 0.085 |
| address digits   | 0.598 |
| **union**        | **0.680** |

### 2. Conjunctive (multi-token) blocking has a low ceiling

Token-set overlap between an S1 and its true match:

```
identical token set : 0.034
match tokens subset of S1 : 0.177
mean Jaccard       : 0.605
frac Jaccard >= 0.5: 0.727
```

Addresses are frequently **reordered** (e.g. S1 "... bali rajasthan bali pali rajasthan"
vs match "... bali rajasthan bali rajasthan") and translated
(`rajasthan` vs `राजस्थान`). So:

* a sorted k-token window is brittle -> the match often falls outside every
  window. Verified directly: for one real S1/match pair, all 5 k=3 windows
  matched documents *other than* the true match.
* k=2 -> recall 0.909, k=3 -> 0.720 on an 800-entity sample.

### 3. Rare-token blocking is the right primitive, but must use the *rarest shared* token

Only **0.13%** of true pairs share no token at all, so token overlap is a
near-complete signal. The distribution of the rarest shared token's df:

```
p50=27  p75=93  p90=513  p95=2121  p99=22096
```

recall obtainable by accepting pairs whose rarest shared token has:

| df threshold | recall |
|--------------|--------|
| <=10    | 0.272 |
| <=50    | 0.658 |
| <=100   | 0.759 |
| <=500   | 0.899 |
| <=1000  | 0.922 |
| <=5000  | 0.967 |
| <=20000 | 0.988 |

Note this is *recall achievable if we could block on the rarest shared token*.
In retrieval we do not know which token is shared, so we must query the tokens
of the S1 and cap the total candidates. Combining this with a per-S1 budget
gives the operating point.

## Decision

Retrieval = union of:

1. **exact name**, **exact address**, **address digits** (cheap, precise),
2. **rare-token postings**, accumulated over the S1's tokens in ascending order
   of posting length, stopping when the per-S1 budget is exhausted, and
3. when the S1's own tokens are all too common, a **bounded fallback** so an
   S1 never ends up with zero candidates.

The digit channel is capped because it alone produced p95 = 10,251
candidates/S1 and dominated the candidate count (87% of all pairs were
`token_block`, but the digit channel was the reason the budget was consumed
before any token evidence was used).

## Consequence

The per-S1 budget is a *measured* hyperparameter. The chosen operating point
and the resulting recall/volume are recorded in
`reports/budget_recall_curve.json` and `reports/retrieval_final_config.json`.
