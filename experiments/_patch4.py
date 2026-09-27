import io

path = "D:/amazon_ml/experiments/exp_budget_recall.py"
with io.open(path, encoding="utf-8") as f:
    s = f.read()

pairs = [
    ('hits = {b: set() for b, _ in BUDGETS}\n'
     '    counts = {b: np.zeros(len(sample), dtype=np.int64) for b, _ in BUDGETS}',
     'hits = {key: set() for key, _ in BUDGETS}\n'
     '    counts = {key: np.zeros(len(sample), dtype=np.int64)\n'
     '              for key, _ in BUDGETS}'),
    ('            n = np.bincount(sr, minlength=len(sample)) if len(sr) else \\\n'
     '                np.zeros(len(sample), dtype=np.int64)\n'
     '            counts[b] += n',
     '            n = np.bincount(sr, minlength=len(sample)) if len(sr) else \\\n'
     '                np.zeros(len(sample), dtype=np.int64)\n'
     '            counts[(per_s1, dcap)] += n'),
    ('                    hits[b].add((label, s, r))',
     '                    hits[(per_s1, dcap)].add((label, s, r))'),
]
for old, new in pairs:
    if old not in s:
        print("NOT FOUND:\n" + old)
        raise SystemExit(1)
    s = s.replace(old, new)

with io.open(path, "w", encoding="utf-8") as f:
    f.write(s)
print("patched")
