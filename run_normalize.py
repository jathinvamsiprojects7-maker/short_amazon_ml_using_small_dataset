"""
Stage 0b: regenerate normalized Parquet caches with the corrected,
script-safe normalizer.

The previous caches were built with a normalizer that NFKC-decomposed complex
scripts and dropped Unicode combining marks, which shredded Devanagari names
into per-character tokens. Those caches are therefore stale, not merely
old, and are regenerated here.

Runs one source at a time, streams to parquet, and prints the ResourceGuard
status periodically so total system RAM can be watched during the run.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_source_cached, NORMALIZATION_VERSION
from src.utils import guard


def main() -> None:
    g = guard()
    print(f"[guard] {g.status()}", flush=True)
    print(f"[normalization] {NORMALIZATION_VERSION}", flush=True)

    jobs: list[tuple[str, str]] = []
    if "--train" in sys.argv or not any(a in sys.argv for a in ("--test", "--all")):
        jobs = [("train", "source1"), ("train", "source2"), ("train", "source3")]
    if "--test" in sys.argv or "--all" in sys.argv:
        jobs += [("test", "source1"), ("test", "source2"), ("test", "source3")]

    for split, source in jobs:
        print(f"\n=== {split} {source} ===", flush=True)
        t0 = time.perf_counter()
        df = load_source_cached(split, source)
        rows = len(df)
        del df
        g.gc()
        print(f"  done {split}_{source}: {rows:,} rows in "
              f"{time.perf_counter()-t0:.0f}s | {g.status()}", flush=True)

    print("\n[all caches refreshed]", flush=True)


if __name__ == "__main__":
    main()
