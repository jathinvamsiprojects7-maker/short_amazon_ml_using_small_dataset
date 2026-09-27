import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import run_features
from src.features import FEATURE_NAMES, compute_features

rng = np.random.default_rng(11)
WORDS = ["apollo", "medical", "center", "st", "llc", "12", "345", "road"]


def rt(mx=4, empty_ok=True):
    n = int(rng.integers(0 if empty_ok else 1, mx + 1))
    return " ".join(rng.choice(WORDS, n, replace=True)) if n else ""


n = 300
s1n = [rt(3) for _ in range(n)]
s1a = [rt(5) for _ in range(n)]
s1d = [rt(2) for _ in range(n)]
s1c = [str(rng.choice(["us", "in", ""])) for _ in range(n)]
cn = [rt(3) for _ in range(n)]
ca = [rt(5) for _ in range(n)]
cd = [rt(2) for _ in range(n)]
cc = [str(rng.choice(["us", "de", ""])) for _ in range(n)]
chans = ["exact_name"] * n
nch = np.ones(n, dtype=np.float32)
prov = {
    "exact_name": np.ones(n, dtype=np.float32),
    "exact_address": np.zeros(n, dtype=np.float32),
    "address_digits": np.zeros(n, dtype=np.float32),
    "token": np.zeros(n, dtype=np.float32),
    "fallback": np.zeros(n, dtype=np.float32),
}

got = run_features.compute_batch(s1n, s1a, s1d, s1c, cn, ca, cd, cc, nch, prov)

for j, name in enumerate(FEATURE_NAMES):
    if name not in ("addr_edit_sim", "digit_overlap", "digit_count_diff"):
        continue
    for i in range(n):
        exp = compute_features(s1n[i], s1a[i], s1d[i], s1c[i],
                               cn[i], ca[i], cd[i], cc[i], chans[i], 1)
        if abs(float(got[i, j]) - float(exp[j])) > 1e-4:
            print(f"--- {name} row {i}: batch={got[i,j]:.4f} ref={exp[j]:.4f}")
            print(f"    s1_addr={s1a[i]!r}  c_addr={ca[i]!r}")
            print(f"    s1_dig ={s1d[i]!r}  c_dig ={cd[i]!r}")
            break
