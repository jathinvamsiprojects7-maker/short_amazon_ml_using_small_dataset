import io

p = "D:/amazon_ml/experiments/test_features.py"
with io.open(p, encoding="utf-8") as f:
    s = f.read()

old_words = '''WORDS = ["apollo", "medical", "center", "st", "llc", "inc", "pvt", "ltd",
         "123", "45", "road", "na", "consulting", "services", "a", "b"]'''
new_words = '''# Name/address vocabulary (letters only, like the real normalized fields).
WORDS = ["apollo", "medical", "center", "st", "llc", "inc", "pvt", "ltd",
         "road", "na", "consulting", "services", "a", "b"]
# address_dig in the real pipeline is src.utils.address_digits() =
# " ".join(re.findall(r"\\d+", raw_address)), i.e. it contains ONLY digit groups.
# The test must respect that invariant, otherwise the reference implementation
# (which re-extracts digits with a regex) and the vectorised one legitimately
# disagree.
DIGITS = ["12", "345", "7", "900", "42", "108"]'''
assert old_words in s
s = s.replace(old_words, new_words)

s = s.replace('s1_dig = [rand_text(3) for _ in range(n)]',
              's1_dig = [rand_digits() for _ in range(n)]')
s = s.replace('c_dig = [rand_text(3) for _ in range(n)]',
              'c_dig = [rand_digits() for _ in range(n)]')
s = s.replace('''        c_dig.append(s1_dig[i] if rng.random() < 0.7 else rand_text(3))
    else:
        c_dig.append(rand_text(3))''',
              '''        c_dig.append(s1_dig[i] if rng.random() < 0.7 else rand_digits())
    else:
        c_dig.append(rand_digits())''')

s = s.replace('''def rand_text(maxlen=5, allow_empty=True):''',
              '''def rand_digits(maxlen=4):
    k = int(rng.integers(0, maxlen + 1))
    return " ".join(rng.choice(DIGITS, k, replace=True)) if k else ""


def rand_text(maxlen=5, allow_empty=True):''')

with io.open(p, "w", encoding="utf-8") as f:
    f.write(s)
print("patched test with realistic address_dig")
