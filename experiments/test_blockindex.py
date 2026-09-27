import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.retrieval import BlockIndex, _s1_tokens, retrieve_chunk, RetrievalBudget
from src.indexing import PostingIndex

rng = np.random.default_rng(0)
vocab = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota"]

docs = []
for _ in range(4000):
    n = int(rng.integers(1, 7))
    toks = list(rng.choice(vocab, n, replace=False))
    names = " ".join(toks[: max(1, n // 2)])
    addrs = " ".join(toks[max(1, n // 2):]) or "x"
    docs.append((names, addrs))

# small parquet resembling the normalized layout
p = "C:/Users/Jathi/AppData/Local/Temp/opencode/_blk_test.parquet"
Path(p).parent.mkdir(parents=True, exist_ok=True)
pq.write_table(pa.table({
    "entity_id": pa.array([f"S2-{i:07d}" for i in range(len(docs))]),
    "name_norm": pa.array([d[0] for d in docs]),
    "address_norm": pa.array([d[1] for d in docs]),
    "address_dig": pa.array(["" for _ in docs]),
}), p)

k = 2
bi = BlockIndex.build(Path(p), k, columns=("name_norm", "address_norm"))
print("built blocks:", bi.n_blocks, "postings:", len(bi.rows), "max:", bi.max_bucket)

# reference
ref = {}
for i, (nm, ad) in enumerate(docs):
    toks = [t for t in (nm + " " + ad).split() if len(t) >= 3]
    toks = sorted(set(toks))
    if len(toks) < k:
        continue
    for j in range(len(toks) - k + 1):
        ref.setdefault(tuple(toks[j:j + k]), []).append(i)

# check that query for a doc returns exactly the docs sharing some window
bad = 0
for i in range(0, len(docs), 37):
    toks = _s1_tokens(docs[i][0], docs[i][1], 3)
    got = set()
    for rows, _s in bi.buckets_for(toks):
        got.update(rows.tolist())
    if len(toks) < k:
        exp = set()
    else:
        exp = set()
        for j in range(len(toks) - k + 1):
            exp.update(ref.get(tuple(toks[j:j + k]), []))
    if got != exp:
        bad += 1
        print("MISMATCH doc", i, "got", len(got), "exp", len(exp))
print("checked", len(range(0, len(docs), 37)), "docs, mismatches:", bad)

# end-to-end retrieve_chunk smoke test
df = pd.DataFrame({
    "name_norm": [d[0] for d in docs[:50]],
    "address_norm": [d[1] for d in docs[:50]],
    "address_dig": ["" for _ in docs[:50]],
})
ix_n = PostingIndex.from_column(Path(p), "name_norm")
ix_a = PostingIndex.from_column(Path(p), "address_norm")
ix_d = PostingIndex.from_column(Path(p), "address_dig")
s1r, cr, bits = retrieve_chunk(df, ix_n, ix_a, ix_d, {k: bi}, 2,
                                RetrievalBudget(per_s1=50, digit_cap=10))
print("retrieve_chunk ->", len(s1r), "pairs; max cands/S1:",
      int(np.bincount(s1r, minlength=50).max()))
assert (s1r >= 0).all() and (cr >= 0).all()
print("OK")
