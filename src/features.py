"""
Pair feature engineering for S1–candidate pairs.

Feature families:
  - Name: exact, similarity (edit, partial, token_sort, token_set), token overlap, jaccard, lengths
  - Address: same families + digit overlap + digit sequence
  - Cross-field: name*address interaction
  - Contradiction: same-name/diff-address, same-addr/diff-name, digit conflict, country conflict
  - Missingness: name/address/digit/country missing flags
  - Retrieval provenance: per-channel flags + n_channels

Performance: vectorized batch computation using numpy arrays.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from src.utils import norm_text, address_digits, norm_tokens


FEATURE_NAMES: list[str] = [
    # Name features
    "name_exact",
    "name_edit_sim",
    "name_partial_sim",
    "name_token_sort_sim",
    "name_token_set_sim",
    "name_token_jaccard",
    "name_len_diff",
    "name_token_cnt_diff",
    "name_len_ratio",
    # Address features
    "addr_exact",
    "addr_edit_sim",
    "addr_partial_sim",
    "addr_token_sort_sim",
    "addr_token_set_sim",
    "addr_token_jaccard",
    "addr_len_diff",
    "addr_token_cnt_diff",
    "addr_len_ratio",
    # Digit features
    "digit_exact",
    "digit_overlap",
    "digit_seq_sim",
    "digit_count_diff",
    # Country
    "country_match",
    "country_both_present",
    # Cross-field
    "name_x_addr",
    "name_plus_addr",
    # Contradiction signals
    "contr_same_name_diff_addr",
    "contr_same_addr_diff_name",
    "contr_digit_conflict",
    "contr_country_conflict",
    # Missingness
    "miss_s1_addr",
    "miss_cand_addr",
    "miss_either_addr",
    "miss_s1_name",
    "miss_cand_name",
    # Retrieval provenance
    "prov_exact_name",
    "prov_exact_address",
    "prov_address_digits",
    "prov_rare_token",
    "prov_ngram",
    "prov_n_channels",
]

N_FEATURES = len(FEATURE_NAMES)

CHANNEL_LIST = ["exact_name", "exact_address", "address_digits", "rare_token", "ngram"]


def _jaccard(a: str, b: str) -> float:
    ta = set(a.split())
    tb = set(b.split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _digit_set_overlap(a: str, b: str) -> float:
    da = set(re.findall(r"\d+", a))
    db = set(re.findall(r"\d+", b))
    if not da and not db:
        return 1.0
    if not da or not db:
        return 0.0
    return len(da & db) / len(da | db)


def _safe_sim(func, a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return func(a, b) / 100.0


def compute_features(
    s1_name_norm: str,
    s1_addr_norm: str,
    s1_addr_dig: str,
    s1_country: str,
    cand_name_norm: str,
    cand_addr_norm: str,
    cand_addr_dig: str,
    cand_country: str,
    channels: str,
    n_channels: int,
) -> np.ndarray:
    """Compute all features for a single pair. Returns float32 array of shape (N_FEATURES,)."""
    vec = np.zeros(N_FEATURES, dtype=np.float32)
    i = 0

    # --- Name features ---
    ne = float(s1_name_norm == cand_name_norm and bool(s1_name_norm))
    n_edit = _safe_sim(fuzz.ratio, s1_name_norm, cand_name_norm)
    n_partial = _safe_sim(fuzz.partial_ratio, s1_name_norm, cand_name_norm)
    n_token_sort = _safe_sim(fuzz.token_sort_ratio, s1_name_norm, cand_name_norm)
    n_token_set = _safe_sim(fuzz.token_set_ratio, s1_name_norm, cand_name_norm)
    n_jaccard = _jaccard(s1_name_norm, cand_name_norm)
    n_len_diff = abs(len(s1_name_norm) - len(cand_name_norm))
    s1_ntoks = len(s1_name_norm.split())
    cand_ntoks = len(cand_name_norm.split())
    n_tok_diff = abs(s1_ntoks - cand_ntoks)
    n_len_ratio = (min(len(s1_name_norm), len(cand_name_norm)) /
                   max(len(s1_name_norm), len(cand_name_norm), 1))

    vec[i:i+9] = [ne, n_edit, n_partial, n_token_sort, n_token_set,
                  n_jaccard, n_len_diff, n_tok_diff, n_len_ratio]
    i += 9

    # --- Address features ---
    ae = float(s1_addr_norm == cand_addr_norm and bool(s1_addr_norm))
    a_edit = _safe_sim(fuzz.ratio, s1_addr_norm, cand_addr_norm)
    a_partial = _safe_sim(fuzz.partial_ratio, s1_addr_norm, cand_addr_norm)
    a_token_sort = _safe_sim(fuzz.token_sort_ratio, s1_addr_norm, cand_addr_norm)
    a_token_set = _safe_sim(fuzz.token_set_ratio, s1_addr_norm, cand_addr_norm)
    a_jaccard = _jaccard(s1_addr_norm, cand_addr_norm)
    a_len_diff = abs(len(s1_addr_norm) - len(cand_addr_norm))
    s1_atoks = len(s1_addr_norm.split())
    cand_atoks = len(cand_addr_norm.split())
    a_tok_diff = abs(s1_atoks - cand_atoks)
    a_len_ratio = (min(len(s1_addr_norm), len(cand_addr_norm)) /
                   max(len(s1_addr_norm), len(cand_addr_norm), 1))

    vec[i:i+9] = [ae, a_edit, a_partial, a_token_sort, a_token_set,
                  a_jaccard, a_len_diff, a_tok_diff, a_len_ratio]
    i += 9

    # --- Digit features ---
    dig_exact = float(bool(s1_addr_dig) and s1_addr_dig == cand_addr_dig)
    dig_overlap = _digit_set_overlap(s1_addr_dig, cand_addr_dig)
    dig_seq_sim = _safe_sim(fuzz.ratio, s1_addr_dig, cand_addr_dig)
    s1_ndig = len(re.findall(r"\d+", s1_addr_dig))
    cand_ndig = len(re.findall(r"\d+", cand_addr_dig))
    dig_cnt_diff = abs(s1_ndig - cand_ndig)

    vec[i:i+4] = [dig_exact, dig_overlap, dig_seq_sim, dig_cnt_diff]
    i += 4

    # --- Country ---
    s1_c = s1_country.strip().casefold()
    cand_c = cand_country.strip().casefold()
    both_present = float(bool(s1_c) and bool(cand_c))
    country_match = float(both_present and s1_c == cand_c)

    vec[i:i+2] = [country_match, both_present]
    i += 2

    # --- Cross-field ---
    name_x_addr = n_edit * a_edit
    name_plus_addr = (n_edit + a_edit) / 2.0

    vec[i:i+2] = [name_x_addr, name_plus_addr]
    i += 2

    # --- Contradictions ---
    contr_same_name_diff_addr = float(
        bool(s1_name_norm) and s1_name_norm == cand_name_norm
        and bool(s1_addr_norm) and bool(cand_addr_norm)
        and s1_addr_norm != cand_addr_norm
        and a_edit < 0.5
    )
    contr_same_addr_diff_name = float(
        bool(s1_addr_norm) and s1_addr_norm == cand_addr_norm
        and bool(s1_name_norm) and bool(cand_name_norm)
        and s1_name_norm != cand_name_norm
        and n_edit < 0.5
    )
    contr_digit_conflict = float(
        bool(s1_addr_dig) and bool(cand_addr_dig)
        and dig_overlap < 0.3
    )
    contr_country_conflict = float(
        both_present and s1_c != cand_c
    )

    vec[i:i+4] = [contr_same_name_diff_addr, contr_same_addr_diff_name,
                  contr_digit_conflict, contr_country_conflict]
    i += 4

    # --- Missingness ---
    miss_s1_addr = float(not bool(s1_addr_norm))
    miss_cand_addr = float(not bool(cand_addr_norm))
    miss_either_addr = float(not bool(s1_addr_norm) or not bool(cand_addr_norm))
    miss_s1_name = float(not bool(s1_name_norm))
    miss_cand_name = float(not bool(cand_name_norm))

    vec[i:i+5] = [miss_s1_addr, miss_cand_addr, miss_either_addr,
                  miss_s1_name, miss_cand_name]
    i += 5

    # --- Retrieval provenance ---
    chan_set = set(channels.split("|")) if channels else set()
    vec[i] = float("exact_name" in chan_set)
    vec[i+1] = float("exact_address" in chan_set)
    vec[i+2] = float("address_digits" in chan_set)
    vec[i+3] = float("rare_token" in chan_set)
    vec[i+4] = float("ngram" in chan_set)
    vec[i+5] = float(n_channels)
    i += 6

    assert i == N_FEATURES, f"Feature count mismatch: {i} vs {N_FEATURES}"
    return vec


def compute_features_batch(
    candidates_df: pd.DataFrame,
    s1_lookup: dict[str, tuple],
    cand_lookup: dict[str, tuple],
    chunk_size: int = 100000,
) -> np.ndarray:
    """
    Batch compute features for a DataFrame of candidate pairs.
    candidates_df must have: s1_id, candidate_id, channels, n_channels
    s1_lookup: {entity_id: (name_norm, addr_norm, addr_dig, country)}
    cand_lookup: {entity_id: (name_norm, addr_norm, addr_dig, country)}
    Returns float32 array of shape (n_pairs, N_FEATURES).

    Processes in chunks with progress reporting.
    """
    n = len(candidates_df)
    X = np.zeros((n, N_FEATURES), dtype=np.float32)
    empty = ("", "", "", "")

    # Pre-extract numpy arrays from the df for speed
    s1_ids = candidates_df["s1_id"].values
    cand_ids = candidates_df["candidate_id"].values
    channels_arr = candidates_df["channels"].values
    n_channels_arr = candidates_df["n_channels"].values

    for chunk_start in range(0, n, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n)
        for idx in range(chunk_start, chunk_end):
            s1_data = s1_lookup.get(s1_ids[idx], empty)
            cand_data = cand_lookup.get(cand_ids[idx], empty)
            X[idx] = compute_features(
                s1_name_norm=s1_data[0],
                s1_addr_norm=s1_data[1],
                s1_addr_dig=s1_data[2],
                s1_country=s1_data[3],
                cand_name_norm=cand_data[0],
                cand_addr_norm=cand_data[1],
                cand_addr_dig=cand_data[2],
                cand_country=cand_data[3],
                channels=channels_arr[idx],
                n_channels=int(n_channels_arr[idx]),
            )
        if chunk_end % 500000 < chunk_size:
            print(f"  Features: {chunk_end:,}/{n:,} pairs computed")

    return X


def make_lookup(df: pd.DataFrame) -> dict[str, tuple]:
    """Build {entity_id: (name_norm, addr_norm, addr_dig, country)} lookup.
    Vectorized — no iterrows.
    """
    return dict(zip(
        df["entity_id"],
        zip(df["name_norm"], df["address_norm"], df["address_dig"], df["country"])
    ))
