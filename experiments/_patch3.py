import io
import sys

path = "D:/amazon_ml/experiments/exp_dedup_volume.py"
with io.open(path, encoding="utf-8") as f:
    src = f.read()

old = '''                cand: set[int] = set()
                for toks, tk in ((s1[s][0], tk_name), (s1[s][1], tk_addr)):
                    if not toks:
                        continue
                    rows = tk.query_tokens_batched([toks], MIN_LEN)[0]
                    if len(rows):
                        cand.update(rows.tolist())'''

new = '''                cand: set[int] = set()
                for toks, tk in ((s1[s][0], tk_name), (s1[s][1], tk_addr)):
                    if not toks:
                        continue
                    rows = tk.rarest(toks, MIN_LEN)
                    if len(rows):
                        cand.update(rows.tolist())'''

if old not in src:
    print("PATTERN NOT FOUND")
    sys.exit(1)
src = src.replace(old, new)
with io.open(path, "w", encoding="utf-8") as f:
    f.write(src)
print("patched exp_dedup_volume.py")
