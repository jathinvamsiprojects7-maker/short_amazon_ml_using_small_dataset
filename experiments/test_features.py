"""
Correctness test: vectorised compute_batch vs the per-pair reference.

src/features.py compute_features is the oracle (it was written first and is
simple to read); run_features.compute_batch is the fast path. They must agree
feature-by-feature on randomised inputs that include the awkward cases:
missing names, missing addresses, missing digits, and empty strings.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import run_features
from src.features import FEATURE_NAMES, compute_features

rng = np.random.default_rng(11)
# Name/address vocabulary (letters only, like the real normalized fields).
WORDS = ["apollo", "medical", "center", "st", "llc", "inc", "pvt", "ltd",
         "road", "na", "consulting", "services", "a", "b"]
# address_dig in the real pipeline is src.utils.address_digits() =
# " ".join(re.findall(r"\d+", raw_address)), i.e. it contains ONLY digit groups.
# The test must respect that invariant, otherwise the reference implementation
# (which re-extracts digits with a regex) and the vectorised one legitimately
# disagree.
DIGITS = ["12", "345", "7", "900", "42", "108"]


def rand_digits(maxlen=4):
    k = int(rng.integers(0, maxlen + 1))
    return " ".join(rng.choice(DIGITS, k, replace=True)) if k else ""


def rand_text(maxlen=5, allow_empty=True):
    n = int(rng.integers(0 if allow_empty else 1, maxlen + 1))
    return " ".join(rng.choice(WORDS, n, replace=True)) if n else ""


n = 600
s1_name = [rand_text(4) for _ in range(n)]
s1_addr = [rand_text(6) for _ in range(n)]
s1_dig = [rand_digits() for _ in range(n)]
s1_ctry = [str(rng.choice(["us", "india", "", "fr"])) for _ in range(n)]

# make candidates correlated with S1 so similarities are non-trivial
c_name, c_addr, c_dig, c_ctry = [], [], [], []
for i in range(n):
    if rng.random() < 0.5:
        c_name.append(s1_name[i] if rng.random() < 0.7 else rand_text(4))
    else:
        c_name.append(rand_text(4))
    if rng.random() < 0.5:
        c_addr.append(s1_addr[i] if rng.random() < 0.7 else rand_text(6))
    else:
        c_addr.append(rand_text(6))
    if rng.random() < 0.5:
        c_dig.append(s1_dig[i] if rng.random() < 0.7 else rand_digits())
    else:
        c_dig.append(rand_digits())
    c_ctry.append(s1_ctry[i] if rng.random() < 0.6 else str(rng.choice(["us", "de", ""])))

chans = []
for i in range(n):
    k = int(rng.integers(1, 4))
    parts = rng.choice(["exact_name", "exact_address", "address_digits",
                        "rare_token", "ngram"], size=k, replace=False)
    chans.append("|".join(sorted(parts)))
nch = np.array([c.count("|") + 1 for c in chans], dtype=np.float32)

prov = {
    "exact_name": np.array([1.0 if "exact_name" in c else 0.0 for c in chans],
                           dtype=np.float32),
    "exact_address": np.array([1.0 if "exact_address" in c else 0.0 for c in chans],
                              dtype=np.float32),
    "address_digits": np.array([1.0 if "address_digits" in c else 0.0 for c in chans],
                               dtype=np.float32),
    "rare_token": np.array([1.0 if "rare_token" in c else 0.0 for c in chans],
                           dtype=np.float32),
    "ngram": np.array([1.0 if "ngram" in c else 0.0 for c in chans],
                      dtype=np.float32),
}

got = run_features.compute_batch(s1_name, s1_addr, s1_dig, s1_ctry,
                                 c_name, c_addr, c_dig, c_ctry, nch, prov)

exp = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
for i in range(n):
    exp[i] = compute_features(
        s1_name[i], s1_addr[i], s1_dig[i], s1_ctry[i],
        c_name[i], c_addr[i], c_dig[i], c_ctry[i],
        chans[i], int(nch[i]))

bad = {}
for j, name in enumerate(FEATURE_NAMES):
    d = np.abs(got[:, j] - exp[:, j])
    if d.max() > 1e-4:
        bad[name] = (float(d.max()), int((d > 1e-4).sum()))

print(f"features compared: {len(FEATURE_NAMES)}, rows: {n}")
if bad:
    print("MISMATCHES:")
    for k, (mx, cnt) in bad.items():
        print(f"  {k:<24} max_diff={mx:.6f}  n_rows={cnt}")
    raise SystemExit(1)
print("ALL FEATURES MATCH the per-pair reference (max diff <= 1e-4)")
