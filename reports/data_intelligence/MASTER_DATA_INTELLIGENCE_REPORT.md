# Master Data Intelligence Report

- [EXACT] Training S1 rows: 2,206,821
- [EXACT] Ground-truth links: 7,638,365
- [EXACT] Zero-match rate: 5.58%
- [EXACT] Singleton rate: 5.40%
- [EXACT] Multi-match rate: 89.02%
- [EXACT] S2 inverse shared entities: 0
- [EXACT] S3 inverse shared entities: 0
- [EXACT] Ground-truth components: 2,206,821
- [SAMPLE] True pairs analysed: 20,000
- [EXACT] Ground-truth file rows: 2,206,821
- [EXACT] Ground-truth file rows: 2,206,821
- [SAMPLE][GROUND-TRUTH-EXCLUDED] Hard negatives: 10,000 same-normalized-name, different-address cross-source candidates. They are 99.11% same-country and have mean normalized-address similarity 0.2621, confirming that name and country agreement alone are unsafe acceptance rules.

- [EXACT] dataset\student_resource\dataset\train\train_source1.tsv: 2,206,821 rows, 210.07 MB
- [EXACT] dataset\student_resource\dataset\train\train_source2.tsv: 5,034,616 rows, 489.30 MB
- [EXACT] dataset\student_resource\dataset\train\train_source3.tsv: 5,285,603 rows, 503.71 MB
- [EXACT] dataset\student_resource\dataset\test\test_source1.tsv: 1,732,544 rows, 175.02 MB
- [EXACT] dataset\student_resource\dataset\test\test_source2.tsv: 4,887,273 rows, 509.46 MB
- [EXACT] dataset\student_resource\dataset\test\test_source3.tsv: 5,082,316 rows, 506.00 MB

## Open decisions

[INFERENCE] Final normalization, blocker, matcher, thresholds, and semantic-model usage remain open and must be decided by leakage-safe end-to-end experiments.

## Recommended experiment order

1. Deterministic exact/normalized baseline.
2. Measure blocking recall and candidate reduction separately.
3. Compare lexical feature families on hard negatives.
4. Test supervised pair classification only after candidate recall is measured.
5. Test semantic retrieval/reranking only for residual cross-script or transliteration cases.
6. Optimize thresholds against macro F0.5 including singleton S1s.

## Evidence artifacts

See `dataset_inventory.json`, `ground_truth_analysis.json`, and `master_data_intelligence.json` for full machine-readable values.
