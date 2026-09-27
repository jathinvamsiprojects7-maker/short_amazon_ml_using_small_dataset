"""Data loading with chunked streaming and Parquet caching."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Generator, Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils import (
    get_config, paths, norm_text, address_digits, norm_tokens, Timer, guard,
)


EXPECTED_COLS = ["entity_id", "business_name", "business_address", "country"]
GT_COLS = ["source1_entity_id", "matched_entity_ids"]

#: Bump whenever the normalization logic changes. It is folded into the cache
#: key so that stale parquet files (produced by an older normalizer) are
#: automatically regenerated instead of silently reused.
#:
#: v2: script-safe normalization. The previous NFKC + `[^\w\s]` pipeline
#:     decomposed complex scripts and deleted combining marks, which shattered
#:     Devanagari names into per-character tokens (e.g. "हिमाचल प्रदेश" became
#:     "ह म चल प रद श", 2 words -> 6 junk tokens).
NORMALIZATION_VERSION = "v2-script-safe"


def _cache_fingerprint(source_hash: str) -> str:
    """Hash covering both the input file and the normalization logic."""
    return hashlib.md5(
        f"{source_hash}:{NORMALIZATION_VERSION}".encode()
    ).hexdigest()[:12]


def _file_hash(path: Path) -> str:
    """Fast hash using file size + mtime (not content hash — too slow for GB files)."""
    stat = path.stat()
    return hashlib.md5(f"{path.name}:{stat.st_size}:{stat.st_mtime}".encode()).hexdigest()[:12]


def _cache_path(cache_dir: Path, name: str) -> Path:
    return cache_dir / f"{name}.parquet"


def _meta_path(cache_dir: Path, name: str) -> Path:
    return cache_dir / f"{name}.meta.json"


def _cache_valid(cache_dir: Path, name: str, source_hash: str) -> bool:
    meta = _meta_path(cache_dir, name)
    if not meta.exists():
        return False
    try:
        stored = json.loads(meta.read_text(encoding="utf-8"))
        return stored.get("source_hash") == source_hash
    except Exception:
        return False


def _write_meta(cache_dir: Path, name: str, source_hash: str, rows: int) -> None:
    meta = _meta_path(cache_dir, name)
    meta.write_text(
        json.dumps({"source_hash": source_hash, "rows": rows}),
        encoding="utf-8",
    )


def load_source_cached(split: str, source: str) -> pd.DataFrame:
    """
    Load a source TSV with normalization applied, using Parquet cache.
    split: 'train' or 'test'
    source: 'source1', 'source2', 'source3'
    Returns DataFrame with columns:
      entity_id, business_name, business_address, country,
      name_norm, address_norm, address_dig, name_tokens_str, address_tokens_str
    """
    cfg = get_config()
    p = paths()
    cache_dir = p["cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    tsv_path = p[f"{split}_dir"] / f"{split}_{source}.tsv"
    name = f"{split}_{source}_norm"
    src_hash = _file_hash(tsv_path)
    fingerprint = _cache_fingerprint(src_hash)

    if _cache_valid(cache_dir, name, fingerprint):
        with Timer(f"load cached {name}"):
            df = pq.read_table(_cache_path(cache_dir, name)).to_pandas()
        print(f"  [cache hit] {name}: {len(df):,} rows")
        return df

    print(f"  [cache miss] loading {tsv_path.name} ({tsv_path.stat().st_size / 1e6:.1f} MB) ...")
    chunk_size = cfg["data"]["chunk_size"]
    writer: pq.ParquetWriter | None = None
    n_rows = 0
    g = guard()

    schema = pa.schema([
        ("entity_id", pa.string()),
        ("business_name", pa.string()),
        ("business_address", pa.string()),
        ("country", pa.string()),
        ("name_norm", pa.string()),
        ("address_norm", pa.string()),
        ("address_dig", pa.string()),
        ("name_tokens_str", pa.string()),
        ("address_tokens_str", pa.string()),
    ])

    out_path = _cache_path(cache_dir, name)
    with Timer(f"read+normalize {name}"):
        reader = pd.read_csv(
            tsv_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=chunk_size,
            encoding="utf-8-sig",
        )
        for chunk in reader:
            chunk.columns = [c.strip() for c in chunk.columns]
            if list(chunk.columns) != EXPECTED_COLS:
                raise ValueError(f"Unexpected columns in {tsv_path}: {list(chunk.columns)}")
            for col in EXPECTED_COLS:
                chunk[col] = chunk[col].fillna("").str.strip()
            chunk["name_norm"] = chunk["business_name"].map(norm_text)
            chunk["address_norm"] = chunk["business_address"].map(norm_text)
            chunk["address_dig"] = chunk["business_address"].map(address_digits)
            chunk["name_tokens_str"] = chunk["name_norm"]
            chunk["address_tokens_str"] = chunk["address_norm"]
            table = pa.Table.from_pandas(
                chunk[list(schema.names)], schema=schema, preserve_index=False
            )
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), schema=schema,
                                          compression="snappy")
            writer.write_table(table)
            n_rows += len(chunk)
            del chunk, table
            g.gc(1)
            if n_rows % (chunk_size * 4) == 0:
                print(f"    {n_rows:,} rows normalized | {g.status()}", flush=True)

    if writer is None:
        raise RuntimeError(f"No rows read from {tsv_path}")
    writer.close()
    del writer
    g.gc()

    _write_meta(cache_dir, name, fingerprint, n_rows)
    size_mb = out_path.stat().st_size / 1e6
    print(f"  [cached] {name}: {n_rows:,} rows -> {out_path.name} ({size_mb:.1f} MB)")

    return pq.read_table(out_path).to_pandas()


def load_ground_truth(split: str = "train") -> dict[str, list[str]]:
    """Load ground truth TSV → {s1_id: [matched_ids]}."""
    p = paths()
    gt_path = p[f"{split}_dir"] / f"{split}_ground_truth.tsv"
    links: dict[str, list[str]] = {}
    with open(gt_path, encoding="utf-8-sig", newline="") as f:
        header = f.readline().strip().split("\t")
        if header != GT_COLS:
            raise ValueError(f"Unexpected GT header: {header}")
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            s1 = parts[0].strip()
            raw = parts[1].strip() if len(parts) > 1 else ""
            matched = [x.strip() for x in raw.split(",") if x.strip()] if raw else []
            links[s1] = matched
    return links


def load_ground_truth_cached(split: str = "train") -> dict[str, list[str]]:
    """Cached version using JSON."""
    p = paths()
    cache_dir = p["cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{split}_ground_truth.json"
    gt_path = p[f"{split}_dir"] / f"{split}_ground_truth.tsv"
    src_hash = _file_hash(gt_path)
    meta_file = cache_dir / f"{split}_ground_truth.meta.json"

    if meta_file.exists():
        stored = json.loads(meta_file.read_text(encoding="utf-8"))
        if stored.get("source_hash") == src_hash and cache_file.exists():
            print(f"  [cache hit] {split}_ground_truth")
            return json.loads(cache_file.read_text(encoding="utf-8"))

    print(f"  [cache miss] loading ground truth {gt_path.name} ...")
    with Timer("load ground truth"):
        links = load_ground_truth(split)

    cache_file.write_text(json.dumps(links), encoding="utf-8")
    meta_file.write_text(json.dumps({"source_hash": src_hash, "rows": len(links)}), encoding="utf-8")
    print(f"  [cached] {split}_ground_truth: {len(links):,} S1 entities")
    return links
