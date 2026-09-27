import io
import sys

path = "D:/amazon_ml/experiments/exp_channel_volume.py"
with io.open(path, encoding="utf-8") as f:
    src = f.read()

bad = (
    '        for nm, ad in zip(d["name_norm"].fillna(""), '
    'd["address_norm"].fillna(""))):\n'
    '            toks = [t for t in (nm + " " + ad).split() '
    'if len(t) >= min_len]\n'
)
good = (
    '        for nm, ad in zip(d["name_norm"].fillna(""), '
    'd["address_norm"].fillna("")):\n'
    '            toks = [t for t in ((nm or "") + " " + (ad or "")).split()\n'
    '                    if len(t) >= min_len]\n'
)

if bad in src:
    src = src.replace(bad, good)
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print("patched")
else:
    print("PATTERN NOT FOUND")
    sys.exit(1)
