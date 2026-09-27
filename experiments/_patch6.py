import io

p = "D:/amazon_ml/experiments/test_retrieval_equiv.py"
with io.open(p, encoding="utf-8") as f:
    s = f.read()

s = s.replace(
    "for budget in [50, 200, 1000]:",
    "for _b in [50, 200, 1000]:\n"
    "    budget = RetrievalBudget(per_s1=_b, exact_cap=_b, digit_cap=_b,\n"
    "                            max_token_df=4000)",
)
s = s.replace('print(f"per_s1={budget:>5}', 'print(f"per_s1={_b:>5}')

with io.open(p, "w", encoding="utf-8") as f:
    f.write(s)
print("patched")
