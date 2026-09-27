"""
Hard negative mining: after initial model, find high-scoring known negatives
and add them to training pool for retraining.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils import Timer


def mine_hard_negatives(
    train_candidates_df: pd.DataFrame,
    scores: np.ndarray,
    gt: dict[str, list[str]],
    train_s1_ids: set[str],
    min_score: float = 0.3,
    max_per_s1: int = 5,
    top_n: int = 50000,
) -> pd.DataFrame:
    """
    Find high-scoring false positives (known negatives with high predicted scores).
    Returns DataFrame of hard negatives to add to training.
    
    NEVER uses validation/test labels.
    """
    gt_sets = {s1: set(m) for s1, m in gt.items()}

    # Only training S1s
    mask = train_candidates_df["s1_id"].isin(train_s1_ids)
    df = train_candidates_df[mask].copy()
    sc = scores[:len(df)] if len(scores) >= len(df) else np.zeros(len(df))

    # This shouldn't happen but guard against index misalignment
    if len(sc) != len(df):
        # scores are for entire df, need to align
        sc = scores[mask.values] if len(scores) == len(train_candidates_df) else np.zeros(len(df))

    df = df.reset_index(drop=True)
    df["_score"] = sc

    # Mark as hard negative: high score but NOT a true positive
    hard_neg_rows = []
    for s1_id, group in df.groupby("s1_id"):
        true_set = gt_sets.get(s1_id, set())
        neg_group = group[~group["candidate_id"].isin(true_set)]
        if neg_group.empty:
            continue
        # Take top-scored negatives above min_score
        high = neg_group[neg_group["_score"] >= min_score].nlargest(max_per_s1, "_score")
        if not high.empty:
            hard_neg_rows.append(high)

    if not hard_neg_rows:
        return pd.DataFrame(columns=list(train_candidates_df.columns))

    hard_df = pd.concat(hard_neg_rows, ignore_index=True)
    hard_df = hard_df.drop(columns=["_score"])
    hard_df["label"] = 0

    # Limit total
    if len(hard_df) > top_n:
        hard_df = hard_df.sample(n=top_n, random_state=20260925)

    print(f"  Hard negatives mined: {len(hard_df):,}")
    return hard_df
