# Final Architecture Decision

## Executive summary

Use a local, reproducible, lexical-first entity-resolution pipeline: conservative Unicode-aware normalization, union-based candidate retrieval, pairwise lexical features, supervised reranking, and S1-level macro F0.5 decision tuning. This is the architecture baseline, not a claim that one untested classifier or threshold is final.

## Dataset evidence

| Finding | Evidence | Architectural implication |
|---|---:|---|
| Training S1 records | 2,206,821 | Stream data and cache only reusable indexes/features. |
| Ground-truth links | 7,638,365 | Optimize candidate recall before matcher complexity. |
| Zero-match S1 records | 5.58% | Empty predictions are a first-class decision. |
| Multi-match S1 records | 89.02% | Do not use top-1 or one-to-one selection. |
| Inverse target sharing | 0 S2 and 0 S3 records shared across S1s | An S1-grouped split prevents label leakage; graph constraints are unnecessary. |
| Positive name exactness | 22.02% after current normalization, sample | Exact-name matching alone has poor recall. |
| Positive address exactness | 8.48% after current normalization, sample | Address needs fuzzy/token/digit representations. |
| Same-name hard negatives | 10,000 ground-truth-excluded candidates | Name equality cannot be an acceptance rule. |
| Same-country hard negatives | 99.11% | Country agreement is not a sufficient match signal or hard blocker. |
| Test-only France | 259,452 S1 records | Treat country as an open-set string; no US/India-specific logic. |

## Final pipeline

```text
TSV validation [FIXED]
  -> conservative normalization [FIXED]
  -> source-local indexes and multi-signal candidate union [EXPERIMENTALLY SELECTED]
  -> candidate_pairs.tsv [FIXED output contract]
  -> lexical pair features and blocker provenance [FIXED feature families]
  -> supervised pair scorer [EXPERIMENT REQUIRED]
  -> optional score calibration [EXPERIMENT REQUIRED]
  -> S1-level macro F0.5 thresholding [FIXED objective]
  -> zero/one/many accepted matches per S1 [FIXED]
  -> matching_results.tsv and official validator [FIXED]
```

## Architecture table

| Stage | Baseline component | Alternatives considered | Reason | Validation experiment | Status |
|---|---|---|---|---|---|
| Ingestion | Chunked pandas TSV reads | Polars, distributed frameworks | Existing runs prove pandas works at 200k-row chunks. | Measure throughput and peak RAM. | Frozen |
| Validation | Schema, ID-prefix, duplicate and output validation | None | Official format is strict. | Unit and official-validator checks. | Frozen |
| Normalization | Unicode NFKC, casefold, punctuation/whitespace canonicalization; accent-folded auxiliary key | Aggressive suffix removal, transliteration | The sampled positive agreement rises from 2,171 to 5,209 through these conservative variants; collision risk remains. | Ablation on held-out S1s. | Frozen baseline |
| Indexing | Per-source in-memory/disk-backed exact keys plus sparse lexical indexes | Vector database | Avoids Cartesian comparison and external services. | Memory and candidate-volume measurement. | Frozen |
| Blocking | Union of exact normalized name, exact normalized address, address-number/token keys, and character TF-IDF top-k fallback | Country hard block, one blocker only | Positives vary heavily; same-name negatives rule out name-only acceptance. | Candidate recall and reduction ratio. | Experiment required |
| Features | Name/address exact, token overlap, character similarity, edit similarity, digits, lengths, missingness, country agreement, source pair, blocker provenance | Embedding-only features | Directly supported by observed positive variability and hard negatives. | Held-out feature ablation. | Frozen families |
| Negatives | Positives plus random controls and retrieved same-name/same-country/high-lexical candidates | Random-only negatives | Random negatives do not model the documented false-merge risk. | Precision/F0.5 comparison. | Frozen strategy |
| Matcher | Regularized logistic regression baseline, tree-based reranker candidate | Rules only, LightGBM, semantic reranker | Need nonlinear comparison only after blocker recall and feature baseline exist. | Leakage-safe model comparison. | Experiment required |
| Calibration | None by default | Platt, isotonic | Thresholds are optimized for S1 macro F0.5, not probability accuracy. | Compare validation F0.5. | Experiment required |
| Decision | S1-level score threshold with zero/one/many outputs | Top-1, one-to-one | Ground truth is one-to-many and includes singletons. | Optimize macro F0.5. | Frozen objective |
| Output | Candidate and matching TSVs, official validator | None | Required by challenge. | Validator including ID check before submission. | Frozen |

## Blocking and normalization

Use country as a feature and optional retrieval partition only when present and non-conflicting. Never use it as a hard exclusion: all sampled positives agree by country, but 99.11% of same-name hard negatives also agree, and France is unseen during training.

Keep raw values. Produce a conservative normalized string for equality and character features, plus auxiliary address digit and token forms. Do not remove legal suffixes, transliterate, reorder tokens, or discard unit information until their recall/collision trade-off is measured on a held-out split.

The initial candidate union should include exact name, exact address, address-number/token, and character n-gram TF-IDF retrieval over name and address. Missing-address records rely on name retrieval plus source and ambiguity features. Final accepted matches must be a subset of this last candidate set.

## Matching, negatives, and validation

Train at pair level only on candidates generated without validation-label leakage. Use all true links for selected training S1s and sampled negatives in three groups: random candidates, same-name/same-country collisions, and high lexical retrieval candidates. The 10,000 hard-negative candidates are diagnostic examples; they are not assumed exhaustively verified beyond ground-truth exclusion.

Split by S1 entity with stratification for zero, one, and multi-match labels. Because no S2/S3 ID is linked to more than one S1 in ground truth, component splitting adds complexity without demonstrated benefit. Fit TF-IDF, any calibration model, and threshold only on the training side of each split. Score selection at S1 level with the official macro F0.5, including empty predictions.

The primary matcher is not frozen. Start with a logistic regression scorecard as the reproducible baseline, then compare a histogram gradient booster or LightGBM only if it materially improves held-out macro F0.5 and is reproducible under a compatible license. No one-to-one constraint, top-1 truncation, or global assignment is allowed in the baseline.

## France and modern methods

France must flow through the same normalization, lexical retrieval, and feature code as every other country. A US-versus-India leave-one-country-out experiment measures open-set robustness but does not estimate France leaderboard quality.

Embeddings are optional only. Current evidence supports lexical methods first: the data has useful names, addresses, digits, and string variation, but no evidence yet that lexical candidate recall fails chiefly on semantic or cross-script cases. Consider a compliant sub-8B Apache/MIT multilingual embedding model only after error analysis demonstrates a residual retrieval gap; it must be benchmarked against the lexical baseline and run locally without external lookup.

## Computational architecture and compliance

Use Python 3.12, pandas, numpy, RapidFuzz, scikit-learn, joblib, pytest, and PyYAML. Keep dependency versions pinned when implementation begins. Use chunked reads, cached sparse indexes/features, deterministic seeds, and local CPU execution. Do not use AWS, external APIs, geocoding, business lookups, hosted LLMs, distributed frameworks, or external entity data.

## High-value experiment roadmap

| ID | Hypothesis | Change | Metric | Cost | Decision unlocked |
|---|---|---|---|---|---|
| E0 | Evaluation is correct | Small S1-level metric and output tests | Macro F0.5 | Low | Correctness harness |
| E1 | Conservative lexical baseline is viable | Exact plus fuzzy score baseline | Macro F0.5 | Low | Baseline floor |
| E2 | Candidate union retains links efficiently | Measure each blocker and union on validation S1s | Candidate recall, volume | Medium | Blocking choice |
| E3 | Auxiliary normalization improves recall safely | Normalization ablation | Candidate recall, F0.5 | Medium | Normalization scope |
| E4 | Hard negatives improve precision | Compare random-only vs mixed negatives | Precision, macro F0.5 | Medium | Sampling mix |
| E5 | Nonlinear model earns complexity | Logistic vs tree-based reranker | Macro F0.5 | Medium | Matcher choice |
| E6 | S1-level decisions improve the metric | Threshold/margin sweep | Macro F0.5 | Low | Acceptance policy |
| E7 | Country can safely assist retrieval | Soft country partition ablation | Recall by country | Medium | Country usage |
| E8 | Semantic retrieval has residual value | Small local multilingual retrieval test | Incremental recall/F0.5 | High | Embedding go/no-go |

## Decisions

### Frozen now

- Local, chunked, reproducible processing with no external data or services.
- Open-set country handling and no country hard block.
- One-to-many S1 match selection with explicit singleton predictions; no top-1 or one-to-one constraint.
- Conservative normalization with raw-value retention.
- Candidate union, candidate-pair audit output, lexical feature families, hard-negative-aware training, and S1 macro F0.5 evaluation.

### Experiment required

- Individual blocker keys, TF-IDF top-k, candidate volume, and candidate recall.
- Legal-suffix removal, transliteration, token reordering, and other aggressive normalization.
- Logistic regression versus tree-based scorer, calibration, threshold/margin policy, and any segmentation.
- Whether soft country partitioning helps without reducing recall.

### Optional or stretch

- Multilingual embeddings and semantic reranking, only after residual error analysis.
- Polars migration, GPU, or cloud infrastructure; none is currently justified.

## Implementation order

1. Build the S1-level metric, data validator wrapper, and deterministic split harness.
2. Implement normalization and candidate-index interfaces with caches.
3. Run E1-E3, record candidate recall and volume, then freeze the candidate union.
4. Implement lexical features and mixed negative sampling; run E4-E5.
5. Optimize S1-level decision policy in E6 and run country robustness E7.
6. Run E8 only when lexical errors justify it.
7. Refit the chosen pipeline, generate both TSVs, validate, document, and package.
