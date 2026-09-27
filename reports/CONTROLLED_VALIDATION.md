# Controlled Validation — 23 training S1 cases

## Purpose
Prove the full pipeline (retrieve → features → score → threshold → compare)
works on cases with known ground truth, and inspect failures.

## Resource behaviour (the point of the redesign)
The first attempt scanned the 10.8 GB `train_cand_S2.parquet` sequentially and
peaked at **99.7% total RAM**, breaching the 80% ceiling. Two causes:

1. `RowStore` materialised all four feature columns of every 5.3M-row source
   as Python lists (~3 GB per source) although only a few thousand rows were
   needed.
2. A full sequential scan to find 23 S1 out of 2.2M.

Fixes (both verified):
* `run_features.RowReader` reads only the row groups containing the requested
  rows and caches just those values.
* Candidates are fetched by **row-group pruning**: the candidate files are
  written in ascending `s1_row` order, so each row group spans a bounded,
  predictable range of S1 rows. 5 groups read per source instead of 4,414.

Result: **peak RAM 48%**, candidate extraction **3.9 s** (was a multi-minute
10 GB scan).

## Result

```
CONTROLLED VALIDATION  model=L2  threshold=0.7136
cases=23  macro_f05=0.2174
```

| kind | cases | F0.5 behaviour |
|---|---|---|
| nomatch | 6 | 5/6 score **1.0** (correctly predict nothing); 1 emits 5 false positives |
| single / multi / variation / hardneg | 17 | tp=0 — the true match is **not in the retrieved candidate set** |

## Root cause (traced on one exact pair)

Pair: `S1-46128417` (row 1370356) ↔ `S2-442468794` (row 1499339).

```
S1 name : 'ib data private limited'
S2 name : 'ib dáta private limited'          <- accent differs
S1 addr : 'clover hill plaza s no 27 1st flr shop no u67 nibm undri road ...'
S2 addr : '551 clover hill plaza s no 27 1st flr shop no u67 nibm ...'
S1 dig  : '27 1 67'
S2 dig  : '551 27 1 67'                      <- extra leading number
```

The candidate file for this S1 contains exactly 1000 candidates, and the true
match is **not among them**. Every channel misses it:

| channel | why it misses |
|---|---|
| exact name | `data` vs `dáta` (accent) |
| exact address | S2 has an extra `551` prefix |
| address digits | S2 digits `551 27 1 67` ≠ `27 1 67` |
| rare token | the only shared name tokens are `limited` (df=516,567) and `private` (df=510,817) — both far above `max_df=1000` and above `token_bucket_cap=5000` |

**This is a retrieval failure, not a scoring or metric failure.** The scoring
path was verified separately: `RowReader` returns correct values, the 41-feature
vector is produced, and `LGBMClassifier.predict_proba` is used correctly in
`run_train.py` and `run_control_eval.py`.

## Consequence

This is the same weakness that produced the measured **candidate recall of
0.7462** (5,699,422 of 7,638,365 ground-truth pairs retrieved). The blocker is
**address noise** — extra/missing tokens such as a house number — which breaks
exact-key blocking, combined with **accent variation** in names.

Both are fixable without changing the architecture:
* an accent-folding variant of `name_norm` as an additional exact channel;
* a digit-*subset* key (any 2+ shared digit groups) instead of exact digit
  sequence equality.

Neither is implemented here because the deadline priority is a valid complete
submission. They are recorded in `reports/RETRIEVAL_FIXES_BACKLOG.md`.

## Bottom line

The pipeline is verified end to end: retrieval produces candidates, features
compute, the model scores, the threshold applies, and zero-match S1 are
predicted empty (the behaviour Macro F0.5 rewards). The measured ceiling on
final recall is candidate recall 0.7462.
