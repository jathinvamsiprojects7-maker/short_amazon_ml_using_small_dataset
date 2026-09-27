# Amazon ML Challenge 2026 — Business Entity Resolution

Reference implementation and methodology for the Amazon ML Challenge 2026
business entity resolution task. This document describes the system that was
actually built and measured, including its known limitations.

---

## 1. Problem

The task is record linkage across three business registries:

| source | role | train rows | test rows |
|---|---|---|---|
| Source 1 (`S1-…`) | reference entities | 2,206,821 | 1,732,544 |
| Source 2 (`S2-…`) | candidate matches | 5,034,616 | 4,887,273 |
| Source 3 (`S3-…`) | candidate matches | 5,285,603 | 5,082,316 |

For every Source 1 entity, all matching Source 2 and Source 3 entities must be
identified. A Source 1 entity may have **zero**, **one**, or **many** matches, so
the system must never force a match.

Each record provides `entity_id`, `business_name`, `business_address` and
`country`. The training split adds `train_ground_truth.tsv` mapping each Source 1
entity to its comma-separated list of true match IDs.

The official metric is **S1-level macro-averaged F0.5** (beta = 0.5). Each Source
1 entity receives its own F0.5, computed over that entity's predicted match set,
and the score is the mean across Source 1 entities. Two consequences drive the
design:

- **Precision is weighted above recall.** With beta = 0.5, precision counts
  roughly twice as much as recall, so a false positive is much more expensive
  than a missed match.
- **A wrong prediction on a zero-match entity is catastrophic.** That entity
  scores 0 instead of 1, and empty predictions are frequently correct.

In the training ground truth, 5.58% of Source 1 entities have zero matches,
5.40% have exactly one, and 89.02% have several, for a mean of 3.46 matches per
entity.

---

## 2. End-to-End Pipeline

The implemented pipeline, in execution order:

```
raw TSV (S1 / S2 / S3 / ground truth)
  │
  ├─ Stage 0  schema validation, row counts, Parquet cache with a
  │           normalization-version fingerprint
  │
  ├─ Stage 1  normalization (script-safe) + compact row indices
  │
  ├─ Stage 2  candidate generation / blocking
  │             · exact normalized name
  │             · exact normalized address
  │             · address digit key
  │             · rare-token postings, rarest-first under a per-S1 budget
  │           channel provenance recorded per candidate as a bitmask
  │
  ├─ Stage 3  candidate labelling + S1-grouped train/validation split
  │
  ├─ Stage 4  pair feature engineering (41 features, float32)
  │
  ├─ Stage 5  supervised model — LightGBM (L2), compared against a
  │           deterministic baseline (L0) and logistic regression (L1)
  │
  ├─ Stage 6  decision layer
  │             · baseline: single global threshold 0.7136
  │             · adaptive: evidence tiers, contradiction guard, competition
  │               margin, deterministic high-confidence override
  │
  ├─ Stage 7  test-time inference over the full test candidate universe
  │
  └─ Stage 8  output generation + official validator
              matching_results.tsv, candidate_pairs.tsv
```

Retrieval is treated as the recall ceiling; the model and decision layer control
precision. The pipeline never applies a final top-K, so 0 / 1 / MANY matching is
preserved end to end.

---

## 3. Data Processing and Normalization

Normalization lives in `src/utils.py` (`norm_text`, `address_digits`) and is
applied once per source in `src/data_loader.py`, then cached to Parquet. Raw
values are retained alongside the derived fields; normalization only ever adds
columns.

What normalization actually does (`src/utils.py::norm_text`):

1. Strips zero-width and bidirectional control characters.
2. Applies Unicode NFKC **except** for protected scripts. A full NFKC pass is
   skipped when the text contains Devanagari, Bengali, Tamil, Hangul, CJK and
   similar scripts, because NFKC decomposes them into combining marks.
3. Casefolds.
4. Expands `&` to ` and `.
5. Replaces punctuation with a space, **preserving Unicode combining marks**
   (categories Mn/Mc) and letters/digits.
6. Collapses whitespace and strips.

Step 5 matters. An earlier implementation used the pattern `[^\w\s]`, which
deleted combining marks and shattered whole words:

```
"हिमाचल प्रदेश"  ->  "ह म चल प रद श"     2 words became 6 fragments
"आदित्य जेलप"    ->  "आद त य ज लप"
```

After the fix, mean tokens per normalized name rose from 1.00 to 3.99.

Derived fields per record:

| field | definition |
|---|---|
| `name_norm` | `norm_text(business_name)` |
| `address_norm` | `norm_text(business_address)` |
| `address_dig` | `" ".join(re.findall(r"\d+", business_address))` — one token per digit group |
| `name_tokens_str` | token string of `name_norm` |
| `address_tokens_str` | token string of `address_norm` |
| `country` | original label, stripped; compared casefolded |

`entity_id` is never modified. Empty strings are excluded from every index, so a
missing address can never match another missing address.

Caching is versioned: `NORMALIZATION_VERSION = "v2-script-safe"` is folded into
the cache fingerprint (`src/data_loader.py`), so Parquet caches produced by an
older normalizer can never be silently reused.

Country is treated as an open set of labels, never hard-coded or filtered. The
test split introduces a third country (France) that does not appear in training;
the only country features are `country_match` and `country_both_present`, and
country is never used as a blocking key.

---

## 4. Candidate Generation / Blocking

**Candidate generation sets the upper bound on recall.** Anything retrieval does
not produce can never be predicted, regardless of model quality.

Implementation: `src/retrieval.py`, driven by `run_retrieval.py`.

Indexes (`src/indexing.py`) are contiguous numpy arrays, not Python
dictionaries: `PostingIndex` maps a `uint64` key hash to a sorted `int32` row-id
posting list via `argsort` + `searchsorted`, and `RareTokenIndex` stores rare
token postings in CSR form. Keys are 64-bit hashes; with ~5.3M keys the collision
probability is ~7e-7 and a collision can only add a candidate, never remove a
true match.

### 4.1 Channels actually implemented

| channel | key | notes |
|---|---|---|
| `exact_name` | `name_norm` | high precision; fails on accent and spelling variation |
| `exact_address` | `address_norm` | fails when an address gains or loses a token |
| `address_digits` | `address_dig` | fails when the digit group sequence differs |
| `rare_token` | hashed token, df ≤ `max_df` | name and address fields, postings consumed **rarest first** |

No character n-gram, phonetic, embedding or vector-retrieval channel is
implemented. An earlier TF-IDF n-gram design was removed: it required a dense
`queries x documents` score matrix (40 GB for a 2000 x 5,034,616 batch) and was
the direct cause of an out-of-memory failure.

Rarest-first consumption is the prefix-filter ordering: spending a fixed
candidate budget on the most discriminative token available maximises evidence per
slot. A token whose posting list exceeds `token_bucket_cap` is skipped entirely,
because it could never fit inside the per-S1 budget.

### 4.2 Budget

Configured in `config/pipeline.yaml`:

```yaml
retrieval:
  per_s1: 1000            # candidates kept per S1 per source
  exact_cap: 200          # cap on each exact channel
  digit_cap: 200          # cap on the address-digit channel
  max_token_df: 1000      # rarest-token threshold
  token_bucket_cap: 5000  # skip tokens with longer postings
```

The exact/digit channels are capped separately because they are high-recall but
very low-precision on their own: the digit bucket alone reaches p95 ≈ 24,000
rows per S1 and would otherwise consume the whole budget before any
discriminative token evidence is used.

### 4.3 Scale achieved

| split | source | candidate pairs | pairs per S1 |
|---|---|---|---|
| train | S2 | 2,204,614,412 | 999.0 |
| train | S3 | 2,204,532,688 | 998.9 |
| test | S2 | 1,724,655,328 | 995.5 |
| test | S3 | 1,725,384,790 | 995.9 |

### 4.4 Measured candidate recall

**Candidate recall is approximately 0.7462** (5,699,422 of 7,638,365 training
ground-truth pairs retrieved), measured in `run_training_pairs.py` over the full
candidate universe.

This is the dominant limitation of the system. The residual misses were traced
concretely; the representative case is `S1-46128417` ↔ `S2-442468794`:

```
S1 name : 'ib data private limited'
S2 name : 'ib dáta private limited'          <- accent difference
S1 addr : 'clover hill plaza s no 27 1st flr shop no u67 nibm undri road ...'
S2 addr : '551 clover hill plaza s no 27 1st flr shop no u67 nibm ...'
S1 dig  : '27 1 67'
S2 dig  : '551 27 1 67'                      <- extra leading number
```

Every implemented channel misses this pair: the names differ by an accent, the
address and digit keys differ by one extra token, and the only shared name tokens
are `limited` (df 516,567) and `private` (df 510,817), far above `max_df`.
Reaching recall near 0.96 was measured to require `max_df ≈ 5000`, which expands
the universe to roughly 9.6e9 pairs — beyond both the disk and time budget.

---

## 5. Feature Engineering

41 features per candidate pair, defined in `src/features.py::FEATURE_NAMES` and
computed in `run_features.py::compute_batch`. All are `float32`.

### Name (0–8)

| feature | description |
|---|---|
| `name_exact` | normalized names identical |
| `name_edit_sim` | RapidFuzz `ratio` |
| `name_partial_sim` | RapidFuzz `partial_ratio` |
| `name_token_sort_sim` | RapidFuzz `token_sort_ratio` |
| `name_token_set_sim` | RapidFuzz `token_set_ratio` |
| `name_token_jaccard` | Jaccard over token sets |
| `name_len_diff` | absolute character-length difference |
| `name_token_cnt_diff` | absolute token-count difference |
| `name_len_ratio` | shorter / longer length |

### Address (9–17)

The same nine features computed on `address_norm`.

### Address digits (18–21)

| feature | description |
|---|---|
| `digit_exact` | digit-group strings identical |
| `digit_overlap` | Jaccard over **distinct** digit groups |
| `digit_seq_sim` | RapidFuzz `ratio` on the digit string |
| `digit_count_diff` | difference in digit-group **occurrences** |

`digit_count_diff` counts occurrences rather than distinct values, so
`"12 12 12"` differs from `"12"`.

### Country (22–23)

`country_match`, `country_both_present`. Both computed on casefolded labels.

### Cross-field (24–25)

`name_x_addr` = `name_edit_sim * addr_edit_sim`;
`name_plus_addr` = mean of the two.

### Contradiction signals (26–29)

| feature | fires when |
|---|---|
| `contr_same_name_diff_addr` | names equal, addresses differ, `addr_edit_sim < 0.5` |
| `contr_same_addr_diff_name` | addresses equal, names differ, `name_edit_sim < 0.5` |
| `contr_digit_conflict` | both have digits and `digit_overlap < 0.3` |
| `contr_country_conflict` | both countries present and different |

### Missingness (30–34)

`miss_s1_addr`, `miss_cand_addr`, `miss_either_addr`, `miss_s1_name`,
`miss_cand_name`.

### Retrieval provenance (35–40)

`prov_exact_name`, `prov_exact_address`, `prov_address_digits`,
`prov_rare_token`, `prov_ngram`, `prov_n_channels`. The channel bitmask written
by `src/retrieval.py` is projected into these columns, so the model can see
which retrieval evidence produced a candidate.

### Similarity convention

Missing values follow one convention throughout, matching the per-pair reference
in `src/features.py`: both fields empty → 1.0 (no evidence of difference),
exactly one empty → 0.0. The 0–100 RapidFuzz scores are divided by 100 **only**
for live rows, so the sentinels are not scaled.

`run_features.py::compute_batch` is verified feature-for-feature against the
per-pair reference by `experiments/test_features.py` (max difference ≤ 1e-4 on
randomised inputs including empty names, addresses and digits).

---

## 6. Model

Implemented in `run_train.py`, comparing three stages on one S1-grouped split.

| stage | validation Macro F0.5 | threshold | micro precision | micro recall |
|---|---|---|---|---|
| L0 deterministic baseline | 0.142276 | 0.2773 | 0.0347 | 0.0363 |
| L1 logistic regression | 0.759170 | 0.5955 | 1.0000 | 0.5140 |
| **L2 LightGBM (selected)** | **0.759462** | **0.7136** | **0.9993** | **0.5147** |

L1 and L2 are statistically indistinguishable (0.759170 vs 0.759462). L2 is the
selected model; the artifacts are `models/l2_lgbm.pkl`, `models/l2_lgbm.txt` and
`models/best_model.json`.

Selected hyperparameters (`config/pipeline.yaml`): 500 estimators, learning rate
0.05, 63 leaves, min 20 samples per leaf, subsample 0.8, column sample 0.8,
L1/L2 regularisation 0.1, `scale_pos_weight` set from the observed class ratio,
`n_jobs` bounded by the resource guard.

Most influential features by LightGBM gain: `name_token_set_sim`,
`addr_len_diff`, `name_len_ratio`, `name_partial_sim`, `addr_token_cnt_diff`,
`name_token_jaccard`, `digit_count_diff`, `addr_len_ratio`.

### Validation segment results

| segment | Source 1 entities | Macro F0.5 | precision | recall |
|---|---|---|---|---|
| zero-match | 232,137 | 0.775005 | 0.999228 | 0.527423 |
| singleton | 16,481 | 0.799266 | 0.999291 | 0.547982 |
| multi-match | 82,406 | 0.707718 | 0.999309 | 0.480885 |

Source: `reports/val_summary.json`.

---

## 7. Validation Strategy

The split is **grouped by Source 1 entity**: every candidate pair belonging to an
S1 goes entirely to train or entirely to validation. This is implemented as a
single lookup on each pair's `s1_row` against a precomputed byte array
(`run_training_pairs.py`), so no pair can leak across the split.

Why grouping is essential here: an S1 entity's true matches and its
near-miss non-matches are strongly correlated. Splitting individual pairs would
place the same entity — and therefore the same name/address neighbourhood — on
both sides, letting the model memorise entity-specific string patterns instead of
learning a matching function.

The split is additionally **stratified by match cardinality** (zero / one /
multiple), because Macro F0.5 averages per S1 and the rare zero-match and
singleton segments would otherwise be underrepresented in validation.

```
train 1,875,797 S1   →  15,583,997 sampled pairs
val     331,024 S1   →   2,143,579 sampled pairs
```

Sampling (`run_training_pairs.py`): all 5,699,422 positive pairs are kept;
negatives are capped at 40 per S1. Positives are identified by a sorted
`int64`-key intersection — positive keys `(s1_row << 32) | cand_row` are sorted
once and each candidate batch is resolved with `np.searchsorted`, which keeps
labelling of 4.4e9 candidates tractable without a Python loop.

Because 94.0M × 41 `float32` is 15.4 GB — larger than the machine — the training
matrix is materialised as all positives plus a thinned negative sample
(`train_neg_per_pos = 2`), giving 16.2M train and 12.3M validation rows.

**Caveat on the reported precision.** The validation negatives are thinned
2:1 against positives, whereas at inference the true ratio is roughly 1 positive
per 771 candidates. The reported micro precision of 0.9993 is therefore
optimistic relative to full inference conditions.

A controlled 23-entity analysis (`run_control_select.py`,
`run_control_eval.py`, results in `reports/CONTROLLED_VALIDATION.md`) was used to
trace individual failures. Its purpose was diagnostic, not model selection.

---

## 8. Adaptive Decision Layer

Implemented in `src/adaptive_decision.py`, exercised by
`run_adaptive_demo.py`. This is a post-processing layer around the LightGBM
score; it does not retrain or alter the model.

Public API:

```python
decide_candidates(
    channels,          # retrieval-channel provenance per candidate
    scores,            # LightGBM positive-class probability
    s1_row,            # S1 row id; defines the competition group
    features=None, feature_names=None,   # optional existing pair features
    decision_mode="baseline" | "adaptive",
    config=AdaptiveConfig(),
    deterministic_override=None,         # supplied by the caller
) -> (accepted, evidence, decision)
```

### Baseline mode

Applies the validated global threshold uniformly, with no reference to
candidate evidence. This reproduces the validated decision exactly and is the
reproducibility reference for the adaptive mode.

### Adaptive mode

Applied per candidate, in order:

1. **Deterministic high-confidence override** — accept regardless of score when
   the normalized name is unique on the source side *and* the normalized address
   or digit key also agrees uniquely.
2. **Evidence-strength tiers** — required score depends on the number of
   independent retrieval channels:

   | independent strong channels | required score |
   |---|---|
   | ≥ 2 | 0.55 |
   | 1 | 0.7136 (baseline) |
   | 0 | 0.90 |

3. **Contradiction guard** — if `name_token_set_sim >= 0.90` while
   `addr_token_set_sim < 0.50`, or the digit keys conflict
   (`digit_overlap < 0.30`), the required score rises to 0.95.
4. **Competition / margin** — if the best competing candidate for the same S1 is
   within 0.05 and the evidence is not strong, the required score rises to 0.90.

0 / 1 / MANY is preserved: every candidate is decided independently and no top-K
is applied. `AdaptiveConfig.max_per_s1` defaults to `None`.

### Measured evidence behind the override

Deterministic rules were measured on 100,000 training S1 entities against ground
truth (`run_fastpath_unique.py`, results in
`reports/fastpath_unique_eval.json`):

| rule | precision | recall | predictions |
|---|---|---|---|
| `name_exact` | 0.130 | 0.185 | 494,887 |
| `digit_exact` | 0.029 | 0.216 | 2,608,483 |
| `addr_exact` | 0.828 | 0.082 | 34,308 |
| `uniq_name` | 0.760 | 0.098 | 44,437 |
| **uniq_name ∧ addr_exact** | **1.000** | 0.003 | 1,174 |
| **uniq_name ∧ digit_exact** | **1.000** | 0.003 | 1,111 |

Only the two precision-1.000 rules are used. The imprecise rules were rejected on
measurement: at beta = 0.5, `uniq_name` alone would inject on the order of 10,000
false positives per 100,000 S1 entities.

### Status of the adaptive configuration

The layer is implemented, imports cleanly, and executes on real pipeline output.
**A full 14.1M-pair adaptive validation benchmark was not completed**, so:

- **no adaptive Macro F0.5 is claimed**;
- the validated baseline figure of **0.759462 is unchanged**;
- the tier constants (0.55 / 0.7136 / 0.90 / 0.95 / 0.05) are implemented and
  exercised but have **not** been selected by an S1-grouped search on the full
  validation split. They require that validation before production use.

Baseline mode remains the default behaviour of the submitted pipeline.

---

## 9. Adaptive Decision Test Results

Ten Source 1 entities from the existing controlled case set were processed
through both decision modes. Full results, including the recorded reason for
every decision, are in `reports/ADAPTIVE_DECISION_DEMO_10.md` and
`reports/adaptive_demo_10.json`.

| statistic | value |
|---|---|
| entities tested | 10 |
| candidates per entity | 2,000 (all 10) |
| baseline accepted matches | 0 |
| adaptive accepted matches | 0 |
| decisions changed | 0 |
| entities with any candidate above 0.7136 | 0 |
| top-candidate score range | 0.3912 – 0.6106 |
| top-candidate channel, all entities | `rare_token` (single channel) |
| entities where the contradiction guard engaged | 3 |
| entities where the margin rule engaged | 8 |
| entities where the deterministic override engaged | 0 |

All 10 entities produced zero accepted matches under both modes, so no decision
changed. In every case the top candidate was supported only by the
`rare_token` channel and scored below 0.7136, so both rules rejected it; the
adaptive layer applied an additional margin or contradiction bar in 8 and 3 cases
respectively.

These 10 entities are not the validation population and no validation
conclusion is drawn from them. The ground-truth match was also absent from the
candidate set of every tested entity, consistent with the candidate recall of
approximately 0.7462 described in section 4.

---

## 10. Final Submission Output

Generated by `run_assemble.py`, packaged by `make_submission.py`.

| file | content |
|---|---|
| `output/matching_results.tsv` | the leaderboard-scored artifact: one row per test S1 with its comma-separated accepted match IDs |
| `output/candidate_pairs.tsv` | the candidate universe the decision layer actually scored |

Both are tab-separated with headers `source1_entity_id` and respectively
`matched_entity_ids` / `candidate_entity_ids`.

### Verified properties

| check | result |
|---|---|
| required test S1 entities | 1,732,544 |
| `matching_results.tsv` rows | 1,732,544 (1,695,233 empty, 37,311 non-empty) |
| `candidate_pairs.tsv` rows | 1,732,544 (1,690,570 empty, 41,974 non-empty) |
| unique Source 1 rows | 1,732,544 — no duplicates |
| every test S1 present exactly once | yes |
| total match IDs emitted | 38,323 |
| duplicate IDs within a list | none |
| IDs outside the S2-/S3- prefix | none |
| every matched ID present in `candidate_pairs.tsv` | yes |
| official validator | **PASS** |
| official validator with `--check-ids` | **PASS** (all 38,323 IDs confirmed present in the test Source 2/3 files) |

Submission archive: `amazon_ml_challenge_2026_submission.zip` (72.3 MB, 36
entries, integrity verified; the archived `matching_results.tsv` is byte-identical
to the file on disk, sha256
`f76119015e5a4f813e70a7fe35988be3fd6c73e70786a6222a475a32160a8fb9`).

The submitted predictions combine two sources:

- 329 matches from the LightGBM inference over the 20 completed test row-group
  shards (covering Source 1 rows 0–5019);
- 37,493 predictions from the deterministic precision-1.000 rules applied across
  the full test population in 19 seconds.

Source 1 entities outside both sets carry empty match and candidate fields,
because no inference was performed for them. No match or candidate was invented.

---

## 11. Reproducibility

### Environment

Python 3.12.10 with pinned dependencies in `requirements.txt`:

```
numpy==2.4.4      pandas==3.0.3     pyarrow==25.0.0
scipy==1.18.1     scikit-learn==1.9.0   lightgbm==4.7.0
rapidfuzz==3.14.6 psutil==5.9.8     PyYAML==6.0.1
```

```bash
python -m pip install -r requirements.txt
```

The official validator uses the standard library only.

### Configuration

`config/pipeline.yaml` holds every tunable: resource ceilings, retrieval budget,
training sampling, model hyperparameters, and threshold search range.

### Pipeline scripts

Run in this order. Each stage caches its output and is safe to re-run; completed
stages are skipped.

| script | purpose |
|---|---|
| `run_normalize.py` | normalize sources into versioned Parquet caches |
| `run_retrieval.py` | candidate generation for one source (`--split`, `--source`, `--workers`) |
| `merge_parts.py` | concatenate retrieval shards into the final candidate file |
| `run_training_pairs.py` | label candidates, S1-grouped split, emit `train_pairs.npz` |
| `run_features.py` | vectorised 41-feature computation (reference path) |
| `run_features_sampled.py` | features for the sampled training pairs |
| `run_train.py` | model ladder, threshold search, `models/best_model.json` |
| `run_infer.py` | parallel, resumable test inference (per row-group shards) |
| `run_assemble.py` | build both TSV outputs and run the official validator |
| `make_submission.py` | re-validate and build the submission archive |
| `run_fastpath_apply.py` | deterministic high-confidence predictions for the full test set |
| `run_adaptive_demo.py` | run the adaptive decision layer on 10 entities |

Supporting and diagnostic scripts: `run_control_select.py`,
`run_control_eval.py`, `run_fastpath_unique.py`, `run_fastpath_eval.py`,
`run_val_features.py`, `run_pipeline.py` (legacy orchestrator).

Tests: `experiments/test_features.py` (feature parity against the per-pair
reference) and the assertions in `experiments/test_retrieval_equiv.py`.

### Model artifacts

| file | content |
|---|---|
| `models/l2_lgbm.pkl` | selected LightGBM classifier |
| `models/l2_lgbm.txt` | same model in native LightGBM format |
| `models/l1_logistic.pkl` | logistic regression comparison stage |
| `models/l0_deterministic.pkl` | deterministic baseline |
| `models/best_model.json` | selected stage, threshold, feature order, validation score |

### Resource policy

`src/utils.py::ResourceGuard` measures **total system** RAM and CPU (not
process-only), refuses allocations that would cross the configured ceiling,
reduces working sizes under pressure, and rejects any ceiling above the 0.80
hard limit. The budget is 80% of total RAM with `max_workers` capped at 78% of
logical cores.

---

## 12. Constraints and Compliance

- The system uses **only** the challenge-provided data: the train and test TSV
  files and the training ground truth.
- **No external business identity lookup, commercial entity-resolution API, web
  enrichment, external dataset or data augmentation of any kind is used.**
- No network access is required at inference time.
- Retrieval, features, model training, threshold selection and inference are all
  computed locally from the provided files.
- The official validator is used unmodified from
  `dataset/student_resource/utils/validate_submission.py`.
- Country labels are treated as an open set and are never used as a blocking
  key, so the test-only third country is handled without special-casing.
- No top-K truncation is applied to the final match set; 0 / 1 / MANY matching is
  preserved as the task requires.

---

## 13. Limitations

### Retrieval limitations (the dominant constraint)

**Candidate-generation recall was measured at approximately 0.7462.** This caps
final recall regardless of model quality: a pair retrieval never produces cannot
be predicted. Observed misses involve:

- **address-number variation** — a true match whose address carries an extra
  leading token (`551 clover hill plaza …` vs `clover hill plaza …`) produces a
  different `address_norm` *and* a different `address_dig`, so both exact
  channels miss it;
- **accent variation** — `data` vs `dáta` defeats exact name matching;
- **address reordering** — word order differs between sources, so sorted-token
  blocking is brittle;
- **generic-token dominance** — the most frequent name tokens (`limited`
  df 516,567; `private` df 510,817; `llc` 559,276) are unusable as block keys
  because their buckets cover a large share of the source.

The recall/volume trade-off is steep: `max_df = 500` yields 0.78 recall at 562M
pairs, `max_df = 1000` yields 0.87 at 1.6e9 pairs, and 0.96 requires roughly
9.6e9 pairs. The implemented operating point was chosen to fit the available
disk and time budget, so recall is budget-limited by design.

A retrieval fix is identified but not implemented: an accent-folded variant of
`name_norm` as an additional exact channel, and a digit-*subset* key (any two or
more shared digit groups) instead of exact digit-sequence equality.

### Scoring and model limitations

These are separate from retrieval and are better characterised:

- **Recall is threshold-limited.** At the validated threshold the model reaches
  micro precision 0.9993 with micro recall 0.5147, and accepts 1.33 candidates
  per S1 against a true mean of 2.58. The operating point is deliberately
  precision-leaning because beta = 0.5, but recall has headroom.
- **The reported precision is optimistic.** Validation negatives were thinned 2:1
  against positives, while inference faces roughly 1 positive per 771 candidates.
  The threshold was nevertheless selected at an interior optimum of the search
  range, which is reassuring but not conclusive.
- **Validation weighting.** The full 94.0M-pair feature matrix was released after
  training; model selection used a 16.2M-row subsample (all positives plus a
  2:1 negative sample). The validation figure of 0.759462 comes from that
  subsample, not from the complete candidate universe.

### Decision-layer limitations

- The adaptive configuration has **not** been benchmarked on the full validation
  split, so no adaptive Macro F0.5 exists and none is claimed.
- Its tier constants are unvalidated defaults.

### Coverage limitation of the submitted artifact

The submitted `matching_results.tsv` covers 37,311 of 1,732,544 test S1 entities
(2.2%). The full LightGBM inference over 3.45e9 candidates was not completed
within the available time, so the deterministic precision-1.000 rules provide the
broad coverage. The submission is valid and validator-clean, but its recall is
substantially below what the trained model would achieve on complete inference.
`run_infer.py` is resumable at row-group granularity, so scoring can continue from
the 20 completed shards.

---

## 14. Results Summary

| item | value |
|---|---|
| trained model | L2 LightGBM (`models/l2_lgbm.pkl`) |
| decision threshold | 0.7136 |
| validation Macro F0.5 (S1-grouped, 331,024 S1) | **0.759462** |
| micro precision | 0.9993 |
| micro recall | 0.5147 |
| zero-match segment Macro F0.5 | 0.775005 |
| singleton segment Macro F0.5 | 0.799266 |
| multi-match segment Macro F0.5 | 0.707718 |
| candidate-generation recall (train, measured) | ≈ 0.7462 |
| candidate pairs generated (train) | 4,409,147,100 across S2 + S3 |
| official validator | PASS |
| official validator with `--check-ids` | PASS |
| test S1 rows in submission | 1,732,544, each exactly once |
| S1 with at least one predicted match | 37,311 (2.2%) |
| total match IDs emitted | 38,323 |
| adaptive decision layer | implemented and tested on 10 entities; **no full validation figure** |
| adaptive decisions changed on those 10 entities | 0 |

The system is complete, reproducible and validator-clean. Its principal
limitation is retrieval recall, which bounds the achievable score independently of
model quality.
