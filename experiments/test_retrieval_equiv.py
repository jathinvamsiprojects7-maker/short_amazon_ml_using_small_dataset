"""
Equivalence test: vectorised retrieve_chunk_batch vs per-row retrieve_chunk.

The vectorised path is the fast one (it is what actually saturates the CPU),
so it must be proven to return exactly the same (owner, cand, bits) triples as
the simple reference implementation on random data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.indexing import PostingIndex
from src.retrieval import (
    RareTokenIndex, RetrievalBudget, retrieve_chunk, retrieve_chunk_batch,
)

rng = np.random.default_rng(7)
words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
         "iota", "kappa"]
n_src = 6000
src_names = []
src_addrs = []
for _ in range(n_src):
    k = int(rng.integers(1, 5))
    src_names.append(" ".join(rng.choice(words, k, replace=False)))
    k2 = int(rng.integers(1, 6))
    src_addrs.append(" ".join(rng.choice(words, k2, replace=False)))

p = Path("C:/Users/Jathi/AppData/Local/Temp/opencode/_equiv.parquet")
p.parent.mkdir(parents=True, exist_ok=True)
pq.write_table(pa.table({
    "entity_id": pa.array([f"S2-{i:07d}" for i in range(n_src)]),
    "name_norm": pa.array(src_names),
    "address_norm": pa.array(src_addrs),
    "address_dig": pa.array(["" for _ in range(n_src)]),
}), p)

ix_n = PostingIndex.from_column(p, "name_norm")
ix_a = PostingIndex.from_column(p, "address_norm")
ix_d = PostingIndex.from_column(p, "address_dig")
ti = [RareTokenIndex.build(p, "name_norm", 4000),
      RareTokenIndex.build(p, "address_norm", 4000)]

# S1 chunk with some empty fields and duplicates to exercise edge cases
n_s1 = 400
s1 = pd.DataFrame({
    "name_norm": [(src_names[i % n_src] if i % 7 else "") for i in range(n_s1)],
    "address_norm": [(src_addrs[(i * 3) % n_src] if i % 5 else "") for i in range(n_s1)],
    "address_dig": ["" for _ in range(n_s1)],
})

for _b in [50, 200, 1000]:
    budget = RetrievalBudget(per_s1=_b, exact_cap=_b, digit_cap=_b,
                            max_token_df=4000)
    ref_o, ref_c, ref_b = retrieve_chunk(s1, ix_n, ix_a, ix_d, ti, budget)
    got_o, got_c, got_b = retrieve_chunk_batch(s1, ix_n, ix_a, ix_d, ti, budget)

    ref = set(zip(ref_o.tolist(), ref_c.tolist(), ref_b.tolist()))
    got = set(zip(got_o.tolist(), got_c.tolist(), got_b.tolist()))
    # Tie-breaking within equal-length buckets is an implementation detail,
    # so compare the SET of (owner, cand) per owner plus the bitmask, not
    # the exact global ordering.
    ref_oc = {(o, c) for o, c, _b in ref}
    got_oc = {(o, c) for o, c, _b in got}
    ref_map = {(o, c): b for o, c, b in ref}
    got_map = {(o, c): b for o, c, b in got}
    same_bits = all(ref_map.get(k) == v for k, v in got_map.items()) \
        and all(got_map.get(k) == v for k, v in ref_map.items())
    same = same_bits
    print(f"per_s1={_b:>5}: ref={len(ref):>7} got={len(got):>7} "
          f"bits_identical={same}  set_equal={ref_oc == got_oc}  "
          f"delta={len(ref) - len(got)}")
    if not same:
        only_ref = ref - got
        only_got = got - ref
        print("  only in ref:", list(only_ref)[:5])
        print("  only in got:", list(only_got)[:5])
        # ignore ordering differences: compare as sets per owner
        by_owner_ref: dict[int, set] = {}
        for o, c, b in ref:
            by_owner_ref.setdefault(o, set()).add((c, b))
        by_owner_got: dict[int, set] = {}
        for o, c, b in got:
            by_owner_got.setdefault(o, set()).add((c, b))
        print("  per-owner sets equal:", by_owner_ref == by_owner_got)
        for o in sorted(set(by_owner_ref) | set(by_owner_got)):
            if by_owner_ref.get(o) != by_owner_got.get(o):
                print("   owner", o, "ref", len(by_owner_ref.get(o, ())),
                      "got", len(by_owner_got.get(o, ())))

print("done")
