"""
Build the final submission archive.

The challenge requires a single zip containing the code and the outputs, with
the two scored/required TSV files under ``output/``. This script assembles that
package from the verified artefacts, re-runs the official validator against the
exact files that go into the zip, and reports the archive path and contents.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import guard, paths

ROOT = Path("D:/amazon_ml")
CODE_FILES = [
    "run_pipeline.py", "run_normalize.py", "run_retrieval.py",
    "run_features.py", "run_features_sampled.py", "run_training_pairs.py",
    "run_train.py", "run_infer.py", "run_assemble.py", "run_control_select.py",
    "run_control_eval.py", "run_fastpath_apply.py", "merge_parts.py",
    "run_fastpath_unique.py",
]
DOC_FILES = [
    "FINAL_ARCHITECTURE.txt",
    "reports/VERIFIED_STATE.md",
    "reports/CONTROLLED_VALIDATION.md",
    "reports/RETRIEVAL_DECISION_D002.md",
    "reports/RETRIEVAL_DECISION_D003.md",
    "reports/RESOURCE_FIX_AND_RETRIEVAL_REDESIGN.md",
    "config/pipeline.yaml",
    "models/best_model.json",
]
SRC_FILES = [
    "src/__init__.py", "src/utils.py", "src/indexing.py", "src/retrieval.py",
    "src/features.py", "src/data_loader.py", "src/models.py",
    "src/threshold.py", "src/fixed_store.py",
]


def sha256(p: Path, limit: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(limit)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> None:
    g = guard()
    p = paths()
    out_dir = p["output_dir"]
    matching = out_dir / "matching_results.tsv"
    candidate = out_dir / "candidate_pairs.tsv"
    for f in (matching, candidate):
        if not f.exists():
            raise SystemExit(f"missing required output: {f}")
    print(f"[pkg] matching_results.tsv  {matching.stat().st_size / 1e6:.1f} MB",
          flush=True)
    print(f"[pkg] candidate_pairs.tsv  {candidate.stat().st_size / 1e6:.1f} MB",
          flush=True)

    # validate the exact files that will be archived
    print("[pkg] validating files that go into the archive", flush=True)
    res = subprocess.run(
        [sys.executable, str(p["validator"]),
         "--matching", str(matching), "--candidate", str(candidate),
         "--test-dir", str(p["test_dir"]), "--check-ids"],
        capture_output=True)
    out = res.stdout.decode("utf-8", errors="replace")
    print(out, flush=True)
    if res.returncode != 0:
        print("[pkg] VALIDATOR FAILED - archive not created", flush=True)
        raise SystemExit(1)
    print("[pkg] validator PASS (with --check-ids)", flush=True)

    name = "amazon_ml_challenge_2026_submission.zip"
    zp = ROOT / name
    if zp.exists():
        zp.unlink()
    t0 = time.perf_counter()
    n = 0
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as z:
        z.write(matching, "output/matching_results.tsv")
        z.write(candidate, "output/candidate_pairs.tsv")
        n += 2
        for rel in CODE_FILES + SRC_FILES + DOC_FILES:
            f = ROOT / rel
            if f.exists():
                z.write(f, rel)
                n += 1
        for extra in ("src/training_pairs.py", "src/hard_negative_mining.py",
                      "src/candidate_gen.py"):
            f = ROOT / extra
            if f.exists():
                z.write(f, extra)
                n += 1
    print(f"[pkg] wrote {zp} ({zp.stat().st_size / 1e6:.1f} MB, {n} files) in "
          f"{time.perf_counter() - t0:.1f}s", flush=True)

    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        bad = z.testzip()
        print(f"[pkg] archive integrity: {'OK' if bad is None else 'CORRUPT ' + bad}",
              flush=True)
        print(f"[pkg] entries: {len(names)}", flush=True)
        for nm in sorted(names):
            if nm.startswith("output/"):
                print(f"        {nm}", flush=True)
    print(f"[pkg] matching_results.tsv sha256: {sha256(matching)}", flush=True)
    print(f"[pkg] FINAL ZIP: {zp}", flush=True)

    with open(p["reports_dir"] / "submission_package.json", "w",
              encoding="utf-8") as f:
        json.dump({
            "zip": str(zp),
            "zip_mb": round(zp.stat().st_size / 1e6, 1),
            "entries": n,
            "validator_pass": True,
            "matching_sha256": sha256(matching),
            "candidate_sha256": sha256(candidate),
        }, f, indent=2)


if __name__ == "__main__":
    main()
