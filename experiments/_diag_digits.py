import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import re

from src.features import _digit_set_overlap

print("address_dig is produced by src.utils.address_digits = ' '.join(findall \\d+)")
for a, b in [("12 345", "345 12"), ("12 12 12", "12"), ("7", ""),
             ("12 12", "12 12"), ("", "")]:
    ref_ov = _digit_set_overlap(a, b)
    ref_cnt = abs(len(re.findall(r"\d+", a)) - len(re.findall(r"\d+", b)))
    sa = set(x for x in a.split() if x)
    sb = set(x for x in b.split() if x)
    mine_ov = len(sa & sb) / len(sa | sb) if (sa or sb) else 1.0
    mine_cnt = abs(len(sa) - len(sb))
    print(f"  a={a!r:16} b={b!r:14} ov ref={ref_ov:.3f} mine={mine_ov:.3f} | "
          f"cnt ref={ref_cnt} mine={mine_cnt}")
print()
print("=> digit_count_diff differs whenever a number REPEATS, because the")
print("   reference counts occurrences (findall) while a set counts distinct.")
print("   The regex-count version is the intended semantics: '12 12 12' has 3")
print("   digit groups, not 1.")
