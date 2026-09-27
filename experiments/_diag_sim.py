import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import run_features
from src.features import compute_features
from rapidfuzz import fuzz, process

# Case: one side empty, other non-empty. Reference: _safe_sim -> 0.0
# Batch: live mask excludes it, out=0.0. Should agree.
for a, b in [("", "apollo"), ("apollo", ""), ("", ""), ("apollo", "apollo")]:
    r = compute_features(a, "x", "", "", b, "x", "", "", "token", 1)
    idx = 1  # name_edit_sim
    got = run_features.compute_batch(
        [a], ["x"], [""], [""], [b], ["x"], [""], [""],
        np.ones(1, np.float32),
        {k: np.zeros(1, np.float32) for k in
         ("exact_name", "exact_address", "address_digits", "token", "fallback")},
    )
    print(f"  a={a!r:10} b={b!r:10} ref_edit={r[idx]:.3f} batch_edit={got[0,idx]:.3f}"
          f"  ref_tset={r[4]:.3f} batch_tset={got[0,4]:.3f}")

print()
print("--- token_set_ratio on short/odd inputs ---")
pairs = [("apollo medical", "apollo medical center"),
         ("abc", "abd"), ("a b", "b a"), ("x", "xx"),
         ("apollo", "apollo apollo")]
for a, b in pairs:
    ref = fuzz.token_set_ratio(a, b)
    bat = process.cpdist(np.array([a], dtype=object),
                         np.array([b], dtype=object),
                         scorer=fuzz.token_set_ratio, dtype=np.float32)[0]
    print(f"  {a!r:24} {b!r:26} ref={ref:6.2f} batch={bat:6.2f} "
          f"{'OK' if abs(ref-bat) < 1e-3 else 'MISMATCH'}")
