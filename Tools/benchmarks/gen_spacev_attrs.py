#!/usr/bin/env python3
"""Build two attributes on spacev1b (1B x 100 int8), in the SAME ingestion format as
the sift1m multitenant benchmark.

Attribute 1 — ACL (4-level hierarchy): org/dept/team/project, a perfect 4-ary tree.
  cardinalities  [4, 16, 64, 256]   (branching factor 4 per level)
  global offsets [0, 4, 20, 84]     (tag ids are unique across levels)
  Each vector is assigned a uniform-random PROJECT leaf; the path up the tree fixes
  team/dept/org (so the 4 columns are mutually consistent, exactly like sift's tags.npy).

Attribute 2 — numerical: an integer "price" in [0, PRICE_MAX), used with a range
  predicate price < X (sweepable selectivity), exactly like sift num_bench.

Per-vector artifacts (single tenant 0 = all 1B vectors), mirroring sift layout:
  multitenant/tags.npy            (N,4) uint32   ACL tags [org,dept,team,project]
  multitenant/num_attr.npy        (N,)  int32    numerical price
  multitenant/tenant_ids.npy      (N,)  int32    all 0 (single tenant)
  multitenant/query/query_tags.npy        (Nq,4) uint32  per-query ACL path (non-empty filter)
  multitenant/query/query_vectors.npy     (Nq,dim) float32
  multitenant/query/query_tenant_ids.npy  (Nq,) int32  all 0
  multitenant/tenant_tag_scenario.json    describes both attributes + sweep grid
"""
import os, sys, json, time, struct
import numpy as np

# Dataset root holding spacev1b_base.i8bin + query.i8bin. Override with argv[1] or
# the SPACEV1B_ROOT env var; the path below is only the author's local default.
ROOT  = (sys.argv[1] if len(sys.argv) > 1
         else os.environ.get('SPACEV1B_ROOT',
                             '/home/v-mochengli/datasets/big-ann/MSSPACEV1B'))
MT    = f'{ROOT}/multitenant'
QDIR  = f'{MT}/query'
BASE  = f'{ROOT}/spacev1b_base.i8bin'
QBIN  = f'{ROOT}/query.i8bin'

SEED      = 20260615
PRICE_MAX = 100000
# ACL 4-ary tree
LEVELS        = ['org', 'dept', 'team', 'project']
CARD          = [4, 16, 64, 256]
OFFSETS       = [0, 4, 20, 84]
N_LEAF        = CARD[-1]                       # 256
# numeric selectivity sweep (price < X)
PRICE_SWEEP = {'pnum_6': 6200, 'pnum_12': 12400, 'pnum_25': 25000,
               'pnum_50': 50000, 'pnum_75': 75000, 'pnum_100': 100000}

CHUNK = 50_000_000


def read_header(path):
    n, d = np.fromfile(path, dtype=np.int32, count=2)
    return int(n), int(d)


def leaf_to_tags(leaf):
    """leaf: int array in [0,256) -> (k,4) uint32 [org,dept,team,project] global ids."""
    out = np.empty((leaf.shape[0], 4), dtype=np.uint32)
    out[:, 0] = OFFSETS[0] + (leaf // 64)     # org   (0..3)
    out[:, 1] = OFFSETS[1] + (leaf // 16)     # dept  (4..19)
    out[:, 2] = OFFSETS[2] + (leaf // 4)      # team  (20..83)
    out[:, 3] = OFFSETS[3] + leaf             # project (84..339)
    return out


def main():
    os.makedirs(QDIR, exist_ok=True)
    N, dim = read_header(BASE)
    Nq, qdim = read_header(QBIN)
    assert dim == qdim, (dim, qdim)
    print(f'spacev1b: N={N:,} dim={dim}  queries Nq={Nq:,}', flush=True)

    rng = np.random.default_rng(SEED)

    # ---- per-vector ACL tags + numeric price (chunked, memmapped to disk) ----
    tags = np.lib.format.open_memmap(f'{MT}/tags.npy', mode='w+',
                                     dtype=np.uint32, shape=(N, 4))
    price = np.lib.format.open_memmap(f'{MT}/num_attr.npy', mode='w+',
                                      dtype=np.int32, shape=(N,))
    t0 = time.perf_counter()
    for s in range(0, N, CHUNK):
        e = min(s + CHUNK, N)
        k = e - s
        leaf = rng.integers(0, N_LEAF, size=k, dtype=np.int64)
        tags[s:e] = leaf_to_tags(leaf)
        price[s:e] = rng.integers(0, PRICE_MAX, size=k, dtype=np.int64).astype(np.int32)
        print(f'  vectors {e:,}/{N:,}  ({time.perf_counter()-t0:.0f}s)', flush=True)
    tags.flush(); price.flush()
    del tags, price

    # ---- single tenant: all 0 ----
    ten = np.lib.format.open_memmap(f'{MT}/tenant_ids.npy', mode='w+',
                                    dtype=np.int32, shape=(N,))
    for s in range(0, N, CHUNK):
        e = min(s + CHUNK, N)
        ten[s:e] = 0
    ten.flush(); del ten

    # ---- query attributes ----
    qleaf = rng.integers(0, N_LEAF, size=Nq, dtype=np.int64)
    np.save(f'{QDIR}/query_tags.npy', leaf_to_tags(qleaf))
    qv = np.fromfile(QBIN, dtype=np.int8, offset=8, count=Nq * dim).reshape(Nq, dim)
    np.save(f'{QDIR}/query_vectors.npy', qv.astype(np.float32))
    np.save(f'{QDIR}/query_tenant_ids.npy', np.zeros(Nq, dtype=np.int32))

    # ---- scenario / meta ----
    scenario = {
        'schema_version': 1,
        'dataset': 'spacev1b',
        'data_file': BASE,
        'vector_count': N,
        'dimension': dim,
        'data_dtype': 'int8',
        'metric': 'l2',
        'seed': SEED,
        'num_tenants': 1,
        'tenant_file': f'{MT}/tenant_ids.npy',
        'tenant_counts': {'0': N},
        'attributes': {
            'acl': {
                'type': 'categorical_hierarchy',
                'file': f'{MT}/tags.npy',
                'shape': [N, 4],
                'dtype': 'uint32',
                'tag_levels': LEVELS,
                'tag_level_cardinalities': CARD,
                'tag_level_offsets': OFFSETS,
                'tag_path_assignment': 'uniform-random-project-leaf',
                'total_tags': sum(CARD),
            },
            'numeric': {
                'type': 'range',
                'file': f'{MT}/num_attr.npy',
                'shape': [N],
                'dtype': 'int32',
                'name': 'price',
                'range': [0, PRICE_MAX],
                'distribution': 'uniform',
                'predicate': 'price < X',
                'sweep': PRICE_SWEEP,
            },
        },
        'query': {
            'count': Nq,
            'vectors': f'{QDIR}/query_vectors.npy',
            'acl_tags': f'{QDIR}/query_tags.npy',
            'tenant_ids': f'{QDIR}/query_tenant_ids.npy',
            'acl_level_column': {'org': 0, 'dept': 1, 'team': 2, 'project': 3},
        },
    }
    with open(f'{MT}/tenant_tag_scenario.json', 'w') as f:
        json.dump(scenario, f, indent=2)
    print('done. artifacts in', MT, flush=True)


if __name__ == '__main__':
    main()
