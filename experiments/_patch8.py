import io

p = "D:/amazon_ml/experiments/test_features.py"
with io.open(p, encoding="utf-8") as f:
    s = f.read()

s = s.replace(
    '    parts = rng.choice(["exact_name", "exact_address", "address_digits", "token"],\n'
    '                       size=k, replace=False)',
    '    parts = rng.choice(["exact_name", "exact_address", "address_digits",\n'
    '                        "token", "fallback"], size=k, replace=False)',
)
s = s.replace(
    '    "token": np.array([1.0 if "token" in c else 0.0 for c in chans],\n'
    '                      dtype=np.float32),\n'
    '}',
    '    "token": np.array([1.0 if "token" in c else 0.0 for c in chans],\n'
    '                      dtype=np.float32),\n'
    '    "fallback": np.array([1.0 if "fallback" in c else 0.0 for c in chans],\n'
    '                        dtype=np.float32),\n'
    '}',
)
with io.open(p, "w", encoding="utf-8") as f:
    f.write(s)
print("patched test")
