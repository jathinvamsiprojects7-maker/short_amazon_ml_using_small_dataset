"""
Assemble the final TSV files from the per-source part files written by
run_infer.py, then run the official validator.

The part files hold one line per S1 that had at least one candidate:

    <s1_row>\t<comma-separated entity ids>

They are produced in ascending s1_row order within each source, so the
assembly is a linear merge of two sorted streams plus the S1 row axis. Every
test S1 entity must appear exactly once, including those with no candidates
(empty match list).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow.parquet as pq

from src.utils import guard, paths

CACHE = Path("D:/amazon_ml/cache")


def read_parts(path: Path) -> dict[int, str]:
    """
    Load one part file into {s1_row: ids_string}.

    Tolerates a directory being passed (the legacy layout used
    ``_parts/acc_S2/part.txt``), in which case the shards inside are read.
    """
    out: dict[int, str] = {}
    if path.is_dir():
        for f in sorted(path.glob("*.txt")):
            out.update(read_parts(f))
        return out
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            r, _, ids = line.partition("\t")
            try:
                out[int(r)] = ids
            except ValueError:
                continue
    return out


def _dedup_join(*fields: str) -> str:
    """
    Join id lists from several sources, dropping duplicates and keeping order.

    Sources overlap by construction: a candidate found by the LightGBM shards
    can also be found by the deterministic fast path, and the official
    validator rejects repeated ids inside a list.
    """
    seen: set[str] = set()
    out: list[str] = []
    for f in fields:
        if not f:
            continue
        for v in f.split(","):
            if v and v not in seen:
                seen.add(v)
                out.append(v)
    return ",".join(out)


def main() -> None:
    g = guard()
    p = paths()
    tmp = p["output_dir"] / "_parts"
    out_dir = p["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    s1_ids = pq.read_table(str(CACHE / "test_source1_norm.parquet"),
                           columns=["entity_id"]).column("entity_id").to_pylist()
    n_s1 = len(s1_ids)
    print(f"[assemble] S1 entities: {n_s1:,}", flush=True)

    t0 = time.perf_counter()
    # The parallel inference run writes shards at the top level of _parts as
    # acc_S<sid>.g<group>.txt / cand_S<sid>.g<group>.txt; the legacy single-file
    # layout used the acc_S2/ and cand_S2/ subdirectories. Read both.
    acc = {2: {}, 3: {}}
    cand = {2: {}, 3: {}}
    for sid in (2, 3):
        for f in sorted(tmp.glob(f"acc_S{sid}.g*.txt")):
            acc[sid].update(read_parts(f))
        for f in sorted(tmp.glob(f"cand_S{sid}.g*.txt")):
            cand[sid].update(read_parts(f))
        acc[sid].update(read_parts(tmp / f"acc_S{sid}" / "part.txt"))
        cand[sid].update(read_parts(tmp / f"cand_S{sid}" / "part.txt"))

    # Deterministic fast-path predictions (precision 1.000 measured on train,
    # reports/fastpath_unique_eval.json). File format: <s1_row>\t<sid>\t<row>
    # These cover the whole test S1 population, so they are merged in for rows
    # the LightGBM shards did not reach.
    fast_file = tmp / "fast_test.txt"
    n_fast = 0
    if fast_file.exists():
        src_ids = {
            2: pq.read_table(str(CACHE / "test_source2_norm.parquet"),
                             columns=["entity_id"]).column("entity_id").to_pylist(),
            3: pq.read_table(str(CACHE / "test_source3_norm.parquet"),
                             columns=["entity_id"]).column("entity_id").to_pylist(),
        }
        g.gc()
        with open(fast_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    continue
                r, sid, c = int(parts[0]), int(parts[1]), int(parts[2])
                eid = src_ids[sid][c]
                prev = acc[sid].get(r, "")
                acc[sid][r] = (prev + "," + eid) if prev else eid
                # a final match must also appear in candidate_pairs
                cprev = cand[sid].get(r, "")
                cand[sid][r] = (cprev + "," + eid) if cprev else eid
                n_fast += 1
        del src_ids
        g.gc()
    print(f"[assemble] deterministic fast-path predictions merged: {n_fast:,}",
          flush=True)
    print(f"[assemble] loaded parts in {time.perf_counter() - t0:.0f}s | "
          f"{g.status()}", flush=True)
    for k, v in acc.items():
        print(f"   S{k} accepted rows: {len(v):,}")
    for k, v in cand.items():
        print(f"   S{k} candidate rows: {len(v):,}")

    matching_path = out_dir / "matching_results.tsv"
    candidate_path = out_dir / "candidate_pairs.tsv"
    n_zero_match = n_zero_cand = 0
    n_match_ids = 0
    t0 = time.perf_counter()
    with open(matching_path, "w", encoding="utf-8", newline="") as fm, \
            open(candidate_path, "w", encoding="utf-8", newline="") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for r in range(n_s1):
            m2, m3 = acc[2].get(r, ""), acc[3].get(r, "")
            c2, c3 = cand[2].get(r, ""), cand[3].get(r, "")
            # The same id can legitimately arrive from the LightGBM shards and
            # again from the deterministic fast path; the official validator
            # rejects repeated ids inside a list, so both fields are
            # de-duplicated while preserving order.
            mids = _dedup_join(m2, m3)
            cids = _dedup_join(c2, c3)
            eid = s1_ids[r]
            fm.write(f"{eid}\t{mids}\n")
            fc.write(f"{eid}\t{cids}\n")
            if not mids:
                n_zero_match += 1
            else:
                n_match_ids += mids.count(",") + 1
            if not cids:
                n_zero_cand += 1
    print(f"[assemble] wrote {n_s1:,} rows in "
          f"{(time.perf_counter() - t0) / 60:.1f}m", flush=True)
    print(f"[assemble] matched S1: {n_s1 - n_zero_match:,} | "
          f"empty-match S1: {n_zero_match:,} | total match ids: {n_match_ids:,}")
    print(f"[assemble] S1 with no candidates: {n_zero_cand:,}")
    print(f"[assemble] {matching_path.name} "
          f"({matching_path.stat().st_size / 1e6:.0f} MB) | "
          f"{candidate_path.name} ({candidate_path.stat().st_size / 1e6:.0f} MB)",
          flush=True)

    print("[validate] official validator", flush=True)
    # The validator prints box-drawing characters; capture bytes and decode
    # leniently, otherwise the subprocess pipe can fail with UnicodeDecodeError
    # on a cp1252 console even though the validator itself ran fine.
    res = subprocess.run(
        [sys.executable, str(p["validator"]),
         "--matching", str(matching_path),
         "--candidate", str(candidate_path),
         "--test-dir", str(p["test_dir"])],
        capture_output=True)
    out = res.stdout.decode("utf-8", errors="replace")
    err = res.stderr.decode("utf-8", errors="replace")
    print(out, flush=True)
    if err.strip():
        print("STDERR:", err[:1500], flush=True)
    ok = res.returncode == 0
    print(f"[validate] {'PASS' if ok else 'FAIL'} (rc={res.returncode})",
          flush=True)

    with open(p["reports_dir"] / "final_summary.json", "w",
              encoding="utf-8") as f:
        json.dump({
            "n_test_s1": n_s1,
            "n_s1_with_match": n_s1 - n_zero_match,
            "n_s1_empty_match": n_zero_match,
            "n_total_match_ids": n_match_ids,
            "n_s1_no_candidates": n_zero_cand,
            "validator_pass": ok,
        }, f, indent=2)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
