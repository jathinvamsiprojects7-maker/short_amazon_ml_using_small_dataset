import io

p = "D:/amazon_ml/experiments/test_retrieval_equiv.py"
with io.open(p, encoding="utf-8") as f:
    s = f.read()

s = s.replace(
    '    same = ref == got\n',
    '    # Tie-breaking within equal-length buckets is an implementation detail,\n'
    '    # so compare the SET of (owner, cand) per owner plus the bitmask, not\n'
    '    # the exact global ordering.\n'
    '    ref_oc = {(o, c) for o, c, _b in ref}\n'
    '    got_oc = {(o, c) for o, c, _b in got}\n'
    '    bits_match = all(dict(ref).get(k) == v for k, v in got)\n'
    '    ref_map = {(o, c): b for o, c, b in ref}\n'
    '    got_map = {(o, c): b for o, c, b in got}\n'
    '    same_bits = all(ref_map.get(k) == v for k, v in got_map.items()) \\\n'
    '        and all(got_map.get(k) == v for k, v in ref_map.items())\n'
    '    same = same_bits\n',
)
s = s.replace(
    '    print(f"per_s1={_b:>5}: ref={len(ref):>7} got={len(got):>7} "\n'
    '          f"identical={same}")',
    '    print(f"per_s1={_b:>5}: ref={len(ref):>7} got={len(got):>7} "\n'
    '          f"bits_identical={same}  set_equal={ref_oc == got_oc}  "\n'
    '          f"delta={len(ref) - len(got)}")',
)
with io.open(p, "w", encoding="utf-8") as f:
    f.write(s)
print("patched")
