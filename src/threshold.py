"""
Threshold optimization and S1-level decision making.
Primary metric: S1-level Macro F0.5.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import macro_f05, f05_score, Timer


def optimize_threshold(
    candidates_df: pd.DataFrame,
    scores: np.ndarray,
    gt: dict[str, list[str]],
    val_s1_ids: set[str],
    search_min: float = 0.05,
    search_max: float = 0.95,
    search_steps: int = 100,
) -> dict[str, Any]:
    """
    Search for the global threshold that maximizes S1-level Macro F0.5.
    candidates_df must have: s1_id, candidate_id
    scores: predicted match probability (positive class), same order as candidates_df
    gt: {s1_id: [matched_ids]}
    val_s1_ids: set of validation S1 IDs
    """
    gt_sets = {s1: set(m) for s1, m in gt.items() if s1 in val_s1_ids}

    # Include all val S1 ids even those with no candidates
    all_val_s1 = {s1: set(m) for s1, m in gt.items() if s1 in val_s1_ids}

    # Build score lookup per S1
    s1_candidates: dict[str, list[tuple[str, float]]] = {}
    for (s1_id, cand_id), score in zip(
        zip(candidates_df["s1_id"], candidates_df["candidate_id"]), scores
    ):
        if s1_id not in val_s1_ids:
            continue
        if s1_id not in s1_candidates:
            s1_candidates[s1_id] = []
        s1_candidates[s1_id].append((cand_id, float(score)))

    thresholds = np.linspace(search_min, search_max, search_steps)
    best_threshold = thresholds[0]
    best_f05 = -1.0
    results = []

    for thresh in thresholds:
        predictions = {}
        for s1_id in all_val_s1:
            cands = s1_candidates.get(s1_id, [])
            predictions[s1_id] = {cid for cid, sc in cands if sc >= thresh}

        metrics = macro_f05(predictions, all_val_s1)
        f05 = metrics["macro_f05"]
        results.append({"threshold": round(float(thresh), 4), "macro_f05": f05,
                        "micro_precision": metrics["micro_precision"],
                        "micro_recall": metrics["micro_recall"]})
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = float(thresh)

    # Final metrics at best threshold
    predictions = {}
    for s1_id in all_val_s1:
        cands = s1_candidates.get(s1_id, [])
        predictions[s1_id] = {cid for cid, sc in cands if sc >= best_threshold}

    final_metrics = macro_f05(predictions, all_val_s1)

    return {
        "best_threshold": round(best_threshold, 4),
        "best_macro_f05": round(best_f05, 6),
        "final_metrics": final_metrics,
        "search_results": results,
    }


def apply_threshold(
    candidates_df: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    all_s1_ids: list[str],
) -> dict[str, list[str]]:
    """
    Apply threshold to produce final predictions.
    Returns {s1_id: [matched_ids]} — every s1_id in all_s1_ids has an entry.
    """
    predictions: dict[str, list[str]] = {s1: [] for s1 in all_s1_ids}

    for (s1_id, cand_id), score in zip(
        zip(candidates_df["s1_id"], candidates_df["candidate_id"]), scores
    ):
        if s1_id in predictions and float(score) >= threshold:
            predictions[s1_id].append(cand_id)

    return predictions


def write_output_files(
    matching: dict[str, list[str]],
    candidates: dict[str, list[str]],
    output_dir: Path,
) -> None:
    """Write matching_results.tsv and candidate_pairs.tsv."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # matching_results.tsv
    matching_path = output_dir / "matching_results.tsv"
    with open(matching_path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in sorted(matching.keys()):
            ids = matching[s1_id]
            f.write(f"{s1_id}\t{','.join(ids)}\n")

    # candidate_pairs.tsv
    candidate_path = output_dir / "candidate_pairs.tsv"
    with open(candidate_path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in sorted(candidates.keys()):
            ids = candidates[s1_id]
            f.write(f"{s1_id}\t{','.join(ids)}\n")

    print(f"  Written {matching_path} ({sum(1 for v in matching.values() if v):,} non-empty S1s)")
    print(f"  Written {candidate_path} ({sum(1 for v in candidates.values() if v):,} non-empty S1s)")


def candidates_df_to_dict(candidates_df: pd.DataFrame, all_s1_ids: list[str]) -> dict[str, list[str]]:
    """Convert candidates DataFrame to dict for output."""
    result: dict[str, list[str]] = {s1: [] for s1 in all_s1_ids}
    for _, row in candidates_df.iterrows():
        s1_id = row["s1_id"]
        if s1_id in result:
            result[s1_id].append(row["candidate_id"])
    return result


def error_analysis(
    candidates_df: pd.DataFrame,
    scores: np.ndarray,
    gt: dict[str, list[str]],
    s1_ids: set[str],
    threshold: float,
    s1_lookup: dict[str, tuple],
    cand_lookup: dict[str, tuple],
    max_examples: int = 20,
) -> dict[str, Any]:
    """
    Classify false positives and false negatives.
    Returns analysis dict.
    """
    gt_sets = {s1: set(m) for s1, m in gt.items() if s1 in s1_ids}
    s1_candidates: dict[str, list[tuple[str, float, str]]] = {}

    for i, row in enumerate(candidates_df.itertuples(index=False)):
        s1_id = row.s1_id
        if s1_id not in s1_ids:
            continue
        if s1_id not in s1_candidates:
            s1_candidates[s1_id] = []
        s1_candidates[s1_id].append((row.candidate_id, float(scores[i]), row.channels))

    fp_examples, fn_examples = [], []
    fp_types: dict[str, int] = {}
    fn_types: dict[str, int] = {}

    for s1_id, true_set in gt_sets.items():
        cands = s1_candidates.get(s1_id, [])
        accepted = {cid for cid, sc, _ in cands if sc >= threshold}
        fps = accepted - true_set
        fns = true_set - accepted

        s1_data = s1_lookup.get(s1_id, ("", "", "", ""))

        for cid in fps:
            cand_data = cand_lookup.get(cid, ("", "", "", ""))
            # Classify FP
            if s1_data[0] and cand_data[0] and s1_data[0] == cand_data[0]:
                fp_types["same_name_diff_addr"] = fp_types.get("same_name_diff_addr", 0) + 1
            elif s1_data[1] and cand_data[1] and s1_data[1] == cand_data[1]:
                fp_types["same_addr_diff_name"] = fp_types.get("same_addr_diff_name", 0) + 1
            else:
                fp_types["high_similarity"] = fp_types.get("high_similarity", 0) + 1

            if len(fp_examples) < max_examples:
                sc = next((sc for c, sc, _ in cands if c == cid), 0.0)
                fp_examples.append({
                    "s1_id": s1_id, "cand_id": cid,
                    "score": round(sc, 4),
                    "s1_name": s1_data[0], "cand_name": cand_data[0],
                    "s1_addr": s1_data[1], "cand_addr": cand_data[1],
                })

        for cid in fns:
            # Check if it was retrieved
            retrieved = any(c == cid for c, _, _ in cands)
            cand_data = cand_lookup.get(cid, ("", "", "", ""))
            if not retrieved:
                fn_types["retrieval_failure"] = fn_types.get("retrieval_failure", 0) + 1
            else:
                sc = next((sc for c, sc, _ in cands if c == cid), 0.0)
                if sc < threshold:
                    fn_types["threshold_too_high"] = fn_types.get("threshold_too_high", 0) + 1

            if len(fn_examples) < max_examples:
                sc = next((sc for c, sc, _ in cands if c == cid), None)
                fn_examples.append({
                    "s1_id": s1_id, "cand_id": cid,
                    "retrieved": retrieved,
                    "score": round(sc, 4) if sc is not None else None,
                    "s1_name": s1_data[0], "cand_name": cand_data[0],
                    "s1_addr": s1_data[1], "cand_addr": cand_data[1],
                })

    return {
        "fp_count": sum(fp_types.values()),
        "fn_count": sum(fn_types.values()),
        "fp_types": fp_types,
        "fn_types": fn_types,
        "fp_examples": fp_examples[:max_examples],
        "fn_examples": fn_examples[:max_examples],
    }
