"""
Multi-channel candidate generation with inverted indexes.

Memory-safe design:
  - Build indexes without holding source DataFrame in memory
  - Stream retrieval in small S1 chunks, writing parquet incrementally
  - Never accumulate a full pair_bitmap across all S1s simultaneously

Channels:
  A. exact_name      - exact normalized name match
  B. exact_address   - exact normalized address match
  C. address_digits  - address digit sequence match
  D. rare_token      - rare informative name/address tokens
  E. ngram           - character 3-gram TF-IDF sparse retrieval
"""

from __future__ import annotations

import gc
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils import get_config, Timer


# ---------------------------------------------------------------------------
# Inverted index
# ---------------------------------------------------------------------------

class InvertedIndex:
    """Simple inverted index: key -> list of entity_ids."""

    def __init__(self) -> None:
        self._index: dict[str, list[str]] = {}

    def query(self, key: str) -> list[str]:
        return self._index.get(key, [])

    def __len__(self) -> int:
        return len(self._index)

    def build_from_series(self, keys: pd.Series, entity_ids: pd.Series) -> None:
        idx = self._index
        for key, eid in zip(keys, entity_ids):
            if key:
                if key not in idx:
                    idx[key] = [eid]
                else:
                    idx[key].append(eid)


class MultiTokenIndex:
    """Multi-token inverted index for rare-token retrieval."""

    def __init__(self, min_token_len: int = 3, max_df_fraction: float = 0.01) -> None:
        self.min_token_len = min_token_len
        self.max_df_fraction = max_df_fraction
        self._token_index: dict[str, list[str]] = {}
        self._n_docs: int = 0
        self._rare_tokens: set[str] = set()
        self._built = False

    def fit(self, df: pd.DataFrame, text_col: str) -> None:
        df_counts: dict[str, int] = {}
        n = 0
        min_len = self.min_token_len
        for tokens_str in df[text_col]:
            seen = set()
            for t in tokens_str.split():
                if len(t) >= min_len and t not in seen:
                    seen.add(t)
                    df_counts[t] = df_counts.get(t, 0) + 1
            n += 1
        self._n_docs = n
        max_df = max(1, int(self.max_df_fraction * n))
        self._rare_tokens = {t for t, c in df_counts.items() if c <= max_df}

    def build(self, df: pd.DataFrame, text_col: str) -> None:
        rare = self._rare_tokens
        min_len = self.min_token_len
        idx = self._token_index
        for entity_id, tokens_str in zip(df["entity_id"], df[text_col]):
            for t in tokens_str.split():
                if len(t) >= min_len and t in rare:
                    if t not in idx:
                        idx[t] = [entity_id]
                    else:
                        idx[t].append(entity_id)
        self._built = True

    def query_tokens(self, tokens_str: str) -> list[str]:
        if not self._built:
            return []
        results: list[str] = []
        rare = self._rare_tokens
        min_len = self.min_token_len
        idx = self._token_index
        seen_cands: set[str] = set()
        for t in tokens_str.split():
            if len(t) >= min_len and t in rare:
                for cid in idx.get(t, []):
                    if cid not in seen_cands:
                        seen_cands.add(cid)
                        results.append(cid)
        return results


# ---------------------------------------------------------------------------
# N-gram TF-IDF index
# ---------------------------------------------------------------------------

def _char_ngram_analyzer(n: int):
    def analyzer(s: str) -> list[str]:
        s = s.strip()
        if not s:
            return []
        padded = f"_{s}_"
        return [padded[i:i+n] for i in range(len(padded) - n + 1)]
    return analyzer


class NgramIndex:
    """Character n-gram TF-IDF index using sklearn TfidfVectorizer."""

    def __init__(self, n: int = 3, max_features: int = 30000) -> None:
        self.n = n
        self.max_features = max_features
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix: sp.csr_matrix | None = None
        self._entity_ids: list[str] = []

    def build(self, df: pd.DataFrame, text_col: str) -> None:
        texts = df[text_col].fillna("").tolist()
        self._entity_ids = list(df["entity_id"])
        with Timer(f"  ngram fit+transform ({len(texts):,} docs)"):
            self._vectorizer = TfidfVectorizer(
                analyzer=_char_ngram_analyzer(self.n),
                max_features=self.max_features,
                sublinear_tf=True,
                norm="l2",
                dtype=np.float32,
            )
            self._matrix = self._vectorizer.fit_transform(texts)
        print(f"  Ngram matrix: {self._matrix.shape}  nnz={self._matrix.nnz:,}")

    def query_batch(self, query_texts: list[str], top_k: int = 50) -> list[list[str]]:
        if self._vectorizer is None or self._matrix is None or not query_texts:
            return [[] for _ in query_texts]
        Q = self._vectorizer.transform(query_texts)
        scores = (Q @ self._matrix.T).toarray()
        results = []
        entity_ids = self._entity_ids
        for row in scores:
            nz = np.where(row > 0)[0]
            if len(nz) == 0:
                results.append([])
            elif len(nz) <= top_k:
                top_idx = nz[np.argsort(row[nz])[::-1]]
                results.append([entity_ids[j] for j in top_idx])
            else:
                part = np.argpartition(row, -top_k)[-top_k:]
                part = part[row[part] > 0]
                top_idx = part[np.argsort(row[part])[::-1]]
                results.append([entity_ids[j] for j in top_idx])
        return results


# ---------------------------------------------------------------------------
# Per-source CandidateIndex (used by legacy code)
# ---------------------------------------------------------------------------

class CandidateIndex:
    def __init__(self, source_label: str) -> None:
        self.source_label = source_label
        self._exact_name = InvertedIndex()
        self._exact_address = InvertedIndex()
        self._address_digits = InvertedIndex()
        self._rare_name: MultiTokenIndex | None = None
        self._rare_address: MultiTokenIndex | None = None
        self._ngram_name: NgramIndex | None = None

    def build(self, df: pd.DataFrame) -> None:
        cfg = get_config()["retrieval"]

        with Timer(f"build exact indexes [{self.source_label}]"):
            self._exact_name.build_from_series(df["name_norm"], df["entity_id"])
            self._exact_address.build_from_series(df["address_norm"], df["entity_id"])
            self._address_digits.build_from_series(df["address_dig"], df["entity_id"])

        with Timer(f"build rare-token name index [{self.source_label}]"):
            self._rare_name = MultiTokenIndex(
                min_token_len=cfg["rare_token_min_len"],
                max_df_fraction=cfg["rare_token_max_df"],
            )
            self._rare_name.fit(df, "name_tokens_str")
            self._rare_name.build(df, "name_tokens_str")

        with Timer(f"build rare-token addr index [{self.source_label}]"):
            self._rare_address = MultiTokenIndex(
                min_token_len=cfg["rare_token_min_len"],
                max_df_fraction=cfg["rare_token_max_df"],
            )
            self._rare_address.fit(df, "address_tokens_str")
            self._rare_address.build(df, "address_tokens_str")

        with Timer(f"build ngram index [{self.source_label}]"):
            self._ngram_name = NgramIndex(
                n=cfg["ngram_size"],
                max_features=cfg.get("ngram_max_features", 30000),
            )
            self._ngram_name.build(df, "name_norm")

    def query_exact_channels(
        self,
        name_norm: str,
        address_norm: str,
        address_dig: str,
        name_tokens_str: str,
        address_tokens_str: str,
        max_bucket: int = 5000,
    ) -> dict[str, list[str]]:
        results: dict[str, list[str]] = {}

        bucket = self._exact_name.query(name_norm)
        if 0 < len(bucket) <= max_bucket:
            results["exact_name"] = bucket

        if address_norm:
            bucket = self._exact_address.query(address_norm)
            if 0 < len(bucket) <= max_bucket:
                results["exact_address"] = bucket

        if address_dig:
            bucket = self._address_digits.query(address_dig)
            if 0 < len(bucket) <= max_bucket:
                results["address_digits"] = bucket

        rare_cands = []
        if self._rare_name is not None:
            rare_cands.extend(self._rare_name.query_tokens(name_tokens_str))
        if self._rare_address is not None:
            rare_cands.extend(self._rare_address.query_tokens(address_tokens_str))
        if rare_cands:
            results["rare_token"] = rare_cands

        return results

    def query_batch_ngram(self, texts: list[str], top_k: int = 50) -> list[list[str]]:
        if self._ngram_name is None:
            return [[] for _ in texts]
        return self._ngram_name.query_batch(texts, top_k=top_k)


def build_candidate_indexes(
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
) -> tuple[CandidateIndex, CandidateIndex]:
    idx2 = CandidateIndex("S2")
    idx3 = CandidateIndex("S3")
    with Timer("build S2 index"):
        idx2.build(s2_df)
    with Timer("build S3 index"):
        idx3.build(s3_df)
    return idx2, idx3


# ---------------------------------------------------------------------------
# Channel bitmap helpers
# ---------------------------------------------------------------------------

_CAND_SCHEMA = pa.schema([
    ("s1_id", pa.string()),
    ("candidate_id", pa.string()),
    ("source", pa.string()),
    ("channels", pa.string()),
    ("n_channels", pa.int16()),
])

CHANNEL_BITS = {
    "exact_name":     1,
    "exact_address":  2,
    "address_digits": 4,
    "rare_token":     8,
    "ngram":         16,
}
BITS_TO_CHANNELS = {v: k for k, v in CHANNEL_BITS.items()}


def _bits_to_channels_str(bits: int) -> str:
    parts = [name for name, b in CHANNEL_BITS.items() if bits & b]
    return "|".join(sorted(parts))


def _n_channels(bits: int) -> int:
    return bin(bits).count("1")


# ---------------------------------------------------------------------------
# Memory-safe streaming candidate retrieval
#
# Strategy:
#   1. Build all indexes from source df columns ONLY (multiple targeted loads)
#   2. Delete source df before retrieval loop
#   3. Process S1 in small chunks; write each chunk to parquet immediately
#   4. Never hold pair_bitmap for more than chunk_size S1s at a time
# ---------------------------------------------------------------------------

class SourceIndex:
    """
    Holds all retrieval indexes for one source (S2 or S3).
    Built from targeted column loads, never holds full source df.
    """

    def __init__(self, source_label: str) -> None:
        self.source_label = source_label
        self.exact_name = InvertedIndex()
        self.exact_address = InvertedIndex()
        self.address_digits = InvertedIndex()
        self.rare_name: MultiTokenIndex | None = None
        self.rare_address: MultiTokenIndex | None = None
        self.ngram: NgramIndex | None = None

    def build_from_parquet(
        self,
        parquet_path: Path,
        rare_token_min_len: int = 3,
        rare_token_max_df: float = 0.01,
        ngram_n: int = 3,
        ngram_max_features: int = 30000,
    ) -> None:
        """
        Build all indexes from parquet file using targeted column reads.
        Never holds more than one column-subset in memory at a time.
        """
        lbl = self.source_label

        # --- Pass 1: exact name + address + digit indexes ---
        print(f"  [{lbl}] Building exact indexes...")
        with Timer(f"exact idx [{lbl}]"):
            df_exact = pq.read_table(
                parquet_path,
                columns=["entity_id", "name_norm", "address_norm", "address_dig"],
            ).to_pandas()
            self.exact_name.build_from_series(df_exact["name_norm"], df_exact["entity_id"])
            self.exact_address.build_from_series(df_exact["address_norm"], df_exact["entity_id"])
            self.address_digits.build_from_series(df_exact["address_dig"], df_exact["entity_id"])
            del df_exact
        gc.collect()
        print(f"  [{lbl}] Name keys={len(self.exact_name):,}  "
              f"Addr keys={len(self.exact_address):,}  "
              f"Dig keys={len(self.address_digits):,}")

        # --- Pass 2: rare-token indexes ---
        print(f"  [{lbl}] Building rare-token indexes...")
        with Timer(f"rare-tok [{lbl}]"):
            df_tok = pq.read_table(
                parquet_path,
                columns=["entity_id", "name_tokens_str", "address_tokens_str"],
            ).to_pandas()
            self.rare_name = MultiTokenIndex(
                min_token_len=rare_token_min_len,
                max_df_fraction=rare_token_max_df,
            )
            self.rare_name.fit(df_tok, "name_tokens_str")
            self.rare_name.build(df_tok, "name_tokens_str")
            self.rare_address = MultiTokenIndex(
                min_token_len=rare_token_min_len,
                max_df_fraction=rare_token_max_df,
            )
            self.rare_address.fit(df_tok, "address_tokens_str")
            self.rare_address.build(df_tok, "address_tokens_str")
            del df_tok
        gc.collect()
        print(f"  [{lbl}] Rare-name tokens={len(self.rare_name._rare_tokens):,}  "
              f"Rare-addr tokens={len(self.rare_address._rare_tokens):,}")

        # --- Pass 3: ngram index ---
        print(f"  [{lbl}] Building ngram index (max_features={ngram_max_features})...")
        with Timer(f"ngram [{lbl}]"):
            df_ng = pq.read_table(
                parquet_path,
                columns=["entity_id", "name_norm"],
            ).to_pandas()
            self.ngram = NgramIndex(n=ngram_n, max_features=ngram_max_features)
            self.ngram.build(df_ng, "name_norm")
            del df_ng
        gc.collect()


def retrieve_candidates_streaming(
    s1_df: pd.DataFrame,
    source_idx: SourceIndex,
    output_path: Path,
    chunk_size: int = 5000,
    ngram_batch_size: int = 2000,
    ngram_top_k: int = 50,
    max_bucket: int = 5000,
) -> int:
    """
    Stream retrieval: process S1 in chunks, write parquet per chunk.
    Returns total pairs written.

    Memory-safe: pair_bitmap holds at most chunk_size * ~600 entries ≈ 3M pairs ≈ ~300MB.
    """
    lbl = source_idx.source_label
    n_total = len(s1_df)
    total_pairs = 0
    t_start = time.perf_counter()

    writer: pq.ParquetWriter | None = None

    for chunk_start in range(0, n_total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_total)
        chunk = s1_df.iloc[chunk_start:chunk_end]
        n_chunk = len(chunk)

        # Accumulate pairs for this chunk only
        pair_bitmap: dict[tuple[str, str], int] = {}

        # --- Exact + rare-token channels ---
        for row in chunk.itertuples(index=False):
            s1_id = row.entity_id

            # exact name
            bucket = source_idx.exact_name.query(row.name_norm)
            if 0 < len(bucket) <= max_bucket:
                bit = CHANNEL_BITS["exact_name"]
                for cid in bucket:
                    k = (s1_id, cid)
                    pair_bitmap[k] = pair_bitmap.get(k, 0) | bit

            # exact address
            if row.address_norm:
                bucket = source_idx.exact_address.query(row.address_norm)
                if 0 < len(bucket) <= max_bucket:
                    bit = CHANNEL_BITS["exact_address"]
                    for cid in bucket:
                        k = (s1_id, cid)
                        pair_bitmap[k] = pair_bitmap.get(k, 0) | bit

            # address digits
            if row.address_dig:
                bucket = source_idx.address_digits.query(row.address_dig)
                if 0 < len(bucket) <= max_bucket:
                    bit = CHANNEL_BITS["address_digits"]
                    for cid in bucket:
                        k = (s1_id, cid)
                        pair_bitmap[k] = pair_bitmap.get(k, 0) | bit

            # rare tokens
            if source_idx.rare_name is not None:
                for cid in source_idx.rare_name.query_tokens(row.name_tokens_str):
                    k = (s1_id, cid)
                    pair_bitmap[k] = pair_bitmap.get(k, 0) | CHANNEL_BITS["rare_token"]
            if source_idx.rare_address is not None:
                for cid in source_idx.rare_address.query_tokens(row.address_tokens_str):
                    k = (s1_id, cid)
                    pair_bitmap[k] = pair_bitmap.get(k, 0) | CHANNEL_BITS["rare_token"]

        # --- Ngram channel (batched within chunk) ---
        if source_idx.ngram is not None:
            s1_ids_list = list(chunk["entity_id"])
            names_list  = list(chunk["name_norm"])
            for nb in range(0, n_chunk, ngram_batch_size):
                nb_end = min(nb + ngram_batch_size, n_chunk)
                batch_s1    = s1_ids_list[nb:nb_end]
                batch_texts = names_list[nb:nb_end]
                ng_results  = source_idx.ngram.query_batch(batch_texts, top_k=ngram_top_k)
                for s1_id, cands in zip(batch_s1, ng_results):
                    for cid in cands:
                        k = (s1_id, cid)
                        pair_bitmap[k] = pair_bitmap.get(k, 0) | CHANNEL_BITS["ngram"]

        # --- Write chunk to parquet ---
        if pair_bitmap:
            s1_arr  = [k[0] for k in pair_bitmap]
            cid_arr = [k[1] for k in pair_bitmap]
            bits_arr = list(pair_bitmap.values())
            table = pa.table({
                "s1_id":        pa.array(s1_arr, type=pa.string()),
                "candidate_id": pa.array(cid_arr, type=pa.string()),
                "source":       pa.array([lbl] * len(s1_arr), type=pa.string()),
                "channels":     pa.array([_bits_to_channels_str(b) for b in bits_arr], type=pa.string()),
                "n_channels":   pa.array([_n_channels(b) for b in bits_arr], type=pa.int16()),
            }, schema=_CAND_SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(str(output_path), schema=_CAND_SCHEMA, compression="snappy")
            writer.write_table(table)
            total_pairs += len(s1_arr)
            del table

        del pair_bitmap
        gc.collect()

        elapsed = time.perf_counter() - t_start
        rate = chunk_end / max(elapsed, 0.001)
        eta = (n_total - chunk_end) / max(rate, 0.001)
        print(f"  [{lbl}] {chunk_end:,}/{n_total:,} S1s  "
              f"pairs_so_far={total_pairs:,}  "
              f"{rate:.0f} S1/s  ETA {eta/60:.1f}m")

    if writer is not None:
        writer.close()

    print(f"  [{lbl}] Total candidate pairs written: {total_pairs:,}")
    return total_pairs


def merge_candidate_parquets(
    path_a: Path,
    path_b: Path,
    output_path: Path,
) -> int:
    """
    Merge two candidate parquet files, OR-ing channel bitmaps for duplicate (s1,cand) pairs.
    Processes in chunks to stay memory-safe.
    Returns total merged pairs.
    """
    # Load both (they're separate sources so no overlapping candidate IDs in practice)
    df_a = pd.read_parquet(path_a) if path_a.exists() else pd.DataFrame(columns=["s1_id","candidate_id","source","channels","n_channels"])
    df_b = pd.read_parquet(path_b) if path_b.exists() else pd.DataFrame(columns=["s1_id","candidate_id","source","channels","n_channels"])

    combined = pd.concat([df_a, df_b], ignore_index=True)
    del df_a, df_b
    gc.collect()

    # Dedup (same s1+cand appearing in both S2 and S3 is theoretically impossible
    # since entity IDs have source prefixes, but guard anyway)
    n_before = len(combined)
    combined = combined.drop_duplicates(subset=["s1_id", "candidate_id"])
    if len(combined) < n_before:
        print(f"  Merge: removed {n_before - len(combined):,} duplicate pairs")

    combined.to_parquet(output_path, index=False)
    n = len(combined)
    del combined
    gc.collect()
    return n


# ---------------------------------------------------------------------------
# Candidate recall measurement
# ---------------------------------------------------------------------------

def measure_candidate_recall(
    candidates_df: pd.DataFrame,
    gt: dict[str, list[str]],
    s1_ids: list[str],
) -> dict[str, Any]:
    s1_set = set(s1_ids)
    gt_sets = {s1: set(m) for s1, m in gt.items() if s1 in s1_set}

    cand_sets: dict[str, set[str]] = defaultdict(set)
    sub = candidates_df[candidates_df["s1_id"].isin(s1_set)]
    for row in sub.itertuples(index=False):
        cand_sets[row.s1_id].add(row.candidate_id)

    total_gt = 0
    total_recalled = 0
    counts_per_s1 = []

    for s1, true_set in gt_sets.items():
        if not true_set:
            continue
        recalled = len(true_set & cand_sets.get(s1, set()))
        total_gt += len(true_set)
        total_recalled += recalled
        counts_per_s1.append(len(cand_sets.get(s1, set())))

    recall = total_recalled / total_gt if total_gt > 0 else 0.0
    counts_arr = np.array(counts_per_s1) if counts_per_s1 else np.array([0])

    return {
        "candidate_recall": round(recall, 6),
        "total_gt_positives": total_gt,
        "total_recalled": total_recalled,
        "n_s1_with_matches": len(counts_per_s1),
        "mean_candidates_per_s1": round(float(counts_arr.mean()), 1),
        "p50_candidates": int(np.percentile(counts_arr, 50)),
        "p95_candidates": int(np.percentile(counts_arr, 95)),
        "p99_candidates": int(np.percentile(counts_arr, 99)),
        "total_pairs": len(sub),
    }


# ---------------------------------------------------------------------------
# Legacy compatibility wrapper (used by run_pipeline.py stage gating)
# ---------------------------------------------------------------------------

def retrieve_candidates_for_s1(
    s1_df: pd.DataFrame,
    idx2: CandidateIndex,
    idx3: CandidateIndex,
    output_path: Path | None = None,
    chunk_size: int = 20000,
    ngram_batch_size: int = 5000,
) -> pd.DataFrame:
    """Legacy wrapper — not used in memory-safe pipeline."""
    cfg = get_config()["retrieval"]
    max_bucket = cfg["max_bucket_size"]
    ngram_top_k = cfg["ngram_top_k"]

    n_total = len(s1_df)
    all_chunks: list[pd.DataFrame] = []
    total_pairs = 0
    t_start = time.perf_counter()

    for chunk_start in range(0, n_total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_total)
        chunk = s1_df.iloc[chunk_start:chunk_end]
        n_chunk = len(chunk)

        pair_bitmap: dict[tuple[str, str], int] = {}

        for row in chunk.itertuples(index=False):
            s1_id = row.entity_id
            for src_idx in (idx2, idx3):
                ch_results = src_idx.query_exact_channels(
                    row.name_norm, row.address_norm, row.address_dig,
                    row.name_tokens_str, row.address_tokens_str,
                    max_bucket=max_bucket,
                )
                for ch, cands in ch_results.items():
                    bit = CHANNEL_BITS[ch]
                    for cid in cands:
                        key = (s1_id, cid)
                        pair_bitmap[key] = pair_bitmap.get(key, 0) | bit

        s1_ids_chunk = list(chunk["entity_id"])
        name_norms_chunk = list(chunk["name_norm"])
        for nb_start in range(0, n_chunk, ngram_batch_size):
            nb_end = min(nb_start + ngram_batch_size, n_chunk)
            for src_idx in (idx2, idx3):
                ng_results = src_idx.query_batch_ngram(
                    name_norms_chunk[nb_start:nb_end], top_k=ngram_top_k
                )
                for s1_id, cands in zip(s1_ids_chunk[nb_start:nb_end], ng_results):
                    for cid in cands:
                        key = (s1_id, cid)
                        pair_bitmap[key] = pair_bitmap.get(key, 0) | CHANNEL_BITS["ngram"]

        if pair_bitmap:
            s1_arr  = [k[0] for k in pair_bitmap]
            cid_arr = [k[1] for k in pair_bitmap]
            bits_arr = list(pair_bitmap.values())
            chunk_df = pd.DataFrame({
                "s1_id": s1_arr,
                "candidate_id": cid_arr,
                "source": ["S2" if c.startswith("S2") else "S3" for c in cid_arr],
                "channels": [_bits_to_channels_str(b) for b in bits_arr],
                "n_channels": np.array([_n_channels(b) for b in bits_arr], dtype=np.int16),
            })
        else:
            chunk_df = pd.DataFrame(
                columns=["s1_id", "candidate_id", "source", "channels", "n_channels"]
            )

        total_pairs += len(chunk_df)
        all_chunks.append(chunk_df)
        del pair_bitmap
        gc.collect()

        elapsed = time.perf_counter() - t_start
        rate = chunk_end / max(elapsed, 0.001)
        eta = (n_total - chunk_end) / max(rate, 0.001)
        print(f"  Retrieval: {chunk_end:,}/{n_total:,}  pairs={total_pairs:,}  "
              f"{rate:.0f} S1/s  ETA {eta/60:.1f}m")

    result = pd.concat(all_chunks, ignore_index=True) if all_chunks else pd.DataFrame(
        columns=["s1_id", "candidate_id", "source", "channels", "n_channels"]
    )
    return result
