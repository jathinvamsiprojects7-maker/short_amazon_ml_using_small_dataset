import io

path = "D:/amazon_ml/experiments/exp_budget_recall.py"
with io.open(path, encoding="utf-8") as f:
    s = f.read()

old = """            for i, r, s in zip(sr.tolist(), cr.tolist(), sample):
                if r in tgt_rows[i]:
                    hits[(per_s1, dcap)].add((label, s, r))"""
new = """            idx_of = {sid_i: i for i, sid_i in enumerate(sample)}
            for i, r, s in zip(sr.tolist(), cr.tolist(), sample):
                local = idx_of[s]
                if r in tgt_rows[local]:
                    hits[(per_s1, dcap)].add((label, s, r))"""
if old not in s:
    print("NOT FOUND")
    raise SystemExit(1)
s = s.replace(old, new)
with io.open(path, "w", encoding="utf-8") as f:
    f.write(s)
print("patched recall indexing")
