"""
Leakage-safe training pair construction.

Split strategy: S1-entity-level stratified split.
All pairs for one S1 stay in the same partition.

Negative categories:
  1. Random negatives (from candidate pool)
  2. Same-name / different-address hard negatives
  3. Same-address / different-name hard negatives
  4. Near-miss / high-similarity hard negatives
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import get_config, paths, Timer


def stratified_s1_split(
    gt: dict[str, list[str]],
    val_fraction: float = 0.15,
    seed: int = 20260925,
) -> tuple[list[str], list[str]]:
    """
    Split S1 IDs into train / val with stratification by match count.
    Returns (train_s1_ids, val_s1_ids).
    """
    rng = random.Random(seed)
    zero = [s1 for s1, m in gt.items() if len(m) == 0]
    single = [s1 for s1, m in gt.items() if len(m) == 1]
    multi = [s1 for s1, m in gt.items() if len(m) > 1]

    train_ids, val_ids = [], []
    for group in (zero, single, multi):
        rng.shuffle(group)
        n_val = max(1, int(len(group) * val_fraction))
        val_ids.extend(group[:n_val])
        train_ids.extend(group[n_val:])

    return train_ids, val_ids


def build_training_pairs(
    candidates_df: pd.DataFrame,
    gt: dict[str, list[str]],
    train_s1_ids: set[str],
    s1_lookup: dict[str, tuple],
    cand_lookup: dict[str, tuple],
    neg_ratio_random: int = 3,
    neg_ratio_hard: int = 3,
    max_hard_per_s1: int = 10,
    seed: int = 20260925,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build training pairs DataFrame with labels.
    Returns (pairs_df, labels) where pairs_df has:
      s1_id, candidate_id, channels, n_channels, label (0/1)
    
    Only uses training S1 IDs.
    """
    rng = random.Random(seed)

    train_s1_set = set(train_s1_ids)
    gt_sets = {s1: set(m) for s1, m in gt.items()}

    # Filter candidates to training S1s only
    train_cands = candidates_df[candidates_df["s1_id"].isin(train_s1_set)].copy()

    positives = []
    all_negatives = []  # (is_hard, row_dict)

    # Group candidates by S1
    for s1_id, group in train_cands.groupby("s1_id"):
        true_set = gt_sets.get(s1_id, set())
        cand_ids = list(group["candidate_id"])
        cand_row_map = {row["candidate_id"]: row for _, row in group.iterrows()}

        pos_ids = [c for c in cand_ids if c in true_set]
        neg_ids = [c for c in cand_ids if c not in true_set]

        for cid in pos_ids:
            row = cand_row_map[cid]
            positives.append({
                "s1_id": s1_id,
                "candidate_id": cid,
                "channels": row["channels"],
                "n_channels": row["n_channels"],
                "label": 1,
            })

        # Classify negatives
        s1_name = s1_lookup.get(s1_id, ("", "", "", ""))[0]
        s1_addr = s1_lookup.get(s1_id, ("", "", "", ""))[1]

        hard_neg_pool = []
        random_neg_pool = []

        for cid in neg_ids:
            row = cand_row_map[cid]
            cand_name = cand_lookup.get(cid, ("", "", "", ""))[0]
            cand_addr = cand_lookup.get(cid, ("", "", "", ""))[1]

            is_hard = (
                (s1_name and cand_name and s1_name == cand_name and s1_addr != cand_addr)
                or (s1_addr and cand_addr and s1_addr == cand_addr and s1_name != cand_name)
                or "exact_name" in (row["channels"] or "")
                or "exact_address" in (row["channels"] or "")
            )
            rec = {
                "s1_id": s1_id,
                "candidate_id": cid,
                "channels": row["channels"],
                "n_channels": row["n_channels"],
                "label": 0,
            }
            if is_hard:
                hard_neg_pool.append(rec)
            else:
                random_neg_pool.append(rec)

        n_pos = len(pos_ids)
        if n_pos == 0:
            # For zero-match S1s, sample a few random negatives
            n_rand = min(neg_ratio_random, len(random_neg_pool))
            all_negatives.extend(rng.sample(random_neg_pool, n_rand))
            continue

        # Sample hard negatives
        n_hard = min(n_pos * neg_ratio_hard, max_hard_per_s1, len(hard_neg_pool))
        if n_hard > 0:
            all_negatives.extend(rng.sample(hard_neg_pool, n_hard))

        # Sample random negatives
        n_rand = min(n_pos * neg_ratio_random, len(random_neg_pool))
        if n_rand > 0:
            all_negatives.extend(rng.sample(random_neg_pool, n_rand))

    neg_dicts = [rec for _, rec in enumerate(all_negatives)]

    all_pairs = positives + neg_dicts
    pairs_df = pd.DataFrame(all_pairs)
    labels = pairs_df["label"].values.astype(np.int32)

    print(f"  Training pairs: {len(pairs_df):,} total | {labels.sum():,} positives | {(labels==0).sum():,} negatives")
    return pairs_df, labels


def build_validation_pairs(
    candidates_df: pd.DataFrame,
    gt: dict[str, list[str]],
    val_s1_ids: set[str],
) -> pd.DataFrame:
    """
    Build validation pairs: all candidates for validation S1 IDs with true labels.
    """
    val_s1_set = set(val_s1_ids)
    gt_sets = {s1: set(m) for s1, m in gt.items()}

    val_cands = candidates_df[candidates_df["s1_id"].isin(val_s1_set)].copy()
    val_cands = val_cands.copy()
    val_cands["label"] = val_cands.apply(
        lambda r: int(r["candidate_id"] in gt_sets.get(r["s1_id"], set())),
        axis=1,
    )

    print(f"  Validation pairs: {len(val_cands):,} total | {val_cands['label'].sum():,} positives")
    return val_cands
