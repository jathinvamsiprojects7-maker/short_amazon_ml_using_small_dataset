import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.indexing import PostingIndex
from src.retrieval import (
    RareTokenIndex, RetrievalBudget, retrieve_chunk,
)

rng = np.random.default_rng(0)
words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
docs = []
for _ in range(3000):
    n = int(rng.integers(1, 5))
    toks = list(rng.choice(words, n, replace=False))
    docs.append(" ".join(toks))

p = Path("C:/Users/Jathi/AppData/Local/Temp/opencode/_rt_test.parquet")
p.parent.mkdir(parents=True, exist_ok=True)
pq.write_table(pa.table({
    "entity_id": pa.array([f"S2-{i:07d}" for i in range(len(docs))]),
    "name_norm": pa.array(docs),
    "address_norm": pa.array(["" for _ in docs]),
    "address_dig": pa.array(["" for _ in docs]),
}), p)

# reference df
ref_df: dict[str, int] = {}
for d in docs:
    for t in set(d.split()):
        ref_df[t] = ref_df.get(t, 0) + 1
max_df = 2000
print("token dfs:", ref_df)

ti = RareTokenIndex.build(p, "name_norm", max_df)
print("vocab", ti.n_vocab, "postings", ti.n_postings)

# reference postings
ref_post: dict[str, set] = {}
for i, d in enumerate(docs):
    for t in set(d.split()):
        if ref_df[t] <= max_df:
            ref_post.setdefault(t, set()).add(i)

bad = 0
for i in range(0, len(docs), 91):
    got = []
    for rows, _n in ti.postings_sorted(docs[i]):
        got.extend(rows.tolist())
    exp = set()
    for t in set(docs[i].split()):
        exp |= ref_post.get(t, set())
    if set(got) != exp:
        bad += 1
print("mismatches:", bad, "of", len(range(0, len(docs), 91)))

# ordering: rarest first
sample_doc = "alpha beta gamma"
lens = [n for _r, n in ti.postings_sorted(sample_doc)]
print("posting lengths for", sample_doc, "->", lens, "(must be non-decreasing)")
assert lens == sorted(lens), "rarest-first ordering violated"

# end-to-end with budget
ix_n = PostingIndex.from_column(p, "name_norm")
ix_a = PostingIndex.from_column(p, "address_norm")
ix_d = PostingIndex.from_column(p, "address_dig")
df = pd.DataFrame({"name_norm": docs[:200], "address_norm": [""] * 200,
                   "address_dig": [""] * 200})
for cap in (10, 100, 1000):
    sr, cr, bits = retrieve_chunk(df, ix_n, ix_a, ix_d, [ti],
                                  RetrievalBudget(per_s1=cap, digit_cap=cap,
                                                  max_token_df=max_df))
    cnt = np.bincount(sr, minlength=200) if len(sr) else np.zeros(200, int)
    print(f"cap={cap:>5} pairs={len(sr):>7} max/S1={cnt.max():>5} "
          f"(<= cap+digit_cap: {cnt.max() <= max(cap, 0) + cap})")
print("OK")
