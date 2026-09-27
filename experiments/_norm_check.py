import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import norm_text

tests = [
    "हिमाचल प्रदेश",
    "आदित्य जेलप",
    "Café Müller",
    "Müller & Sons",
    "Ромашка",
    "शिविर २०१",
    "O'Brien Ltd.",
    "洞海 貿易",
    "서울 상사",
    "D-4/12, Near Bus Stand",
    "アbangou",
    "Café Áphex",
    "ñandú",
]
for t in tests:
    n = norm_text(t)
    print("%-24r -> %-26r %s" % (t, n, n.split()[:6]))
