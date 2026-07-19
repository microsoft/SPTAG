#!/usr/bin/env python3
"""Build SPACEV-style ACL + numeric attributes for BigANN/SIFT1B.

Expected input layout under SIFT1B_ROOT (default:
/mnt/nvme/baotonglu/mocheng/datasets/sift1b):
  raw/bigann_query.bvecs or raw/bigann_query.bvecs.gz
  sift1b_base.u8bin or raw/bigann_base.bvecs(.gz) for vector count metadata

Outputs mirror gen_spacev_attrs.py:
  multitenant/tags.npy                 (N,4) uint32
  multitenant/num_attr.npy             (N,) int32
  multitenant/tenant_ids.npy           (N,) int32, all 0
  multitenant/query/query_vectors.npy  (Nq,128) float32
  multitenant/query/query_tags.npy     (Nq,4) uint32
  multitenant/query/query_tenant_ids.npy
  multitenant/tenant_tag_scenario.json
"""
import gzip
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "SIFT1B_ROOT", "/mnt/nvme/baotonglu/mocheng/datasets/sift1b"))
RAW = ROOT / "raw"
MT = ROOT / "multitenant"
QDIR = MT / "query"
BASE_U8BIN = ROOT / "sift1b_base.u8bin"
QUERY_BVECS = RAW / "bigann_query.bvecs"
QUERY_BVECS_GZ = RAW / "bigann_query.bvecs.gz"
BASE_BVECS = RAW / "bigann_base.bvecs"
BASE_BVECS_GZ = RAW / "bigann_base.bvecs.gz"

SEED = 20260709
PRICE_MAX = 100000
LEVELS = ["org", "dept", "team", "project"]
CARD = [4, 16, 64, 256]
OFFSETS = [0, 4, 20, 84]
N_LEAF = CARD[-1]
PRICE_SWEEP = {
    "pnum_6": 6200,
    "pnum_12": 12400,
    "pnum_25": 25000,
    "pnum_50": 50000,
    "pnum_75": 75000,
    "pnum_100": 100000,
}
CHUNK = 50_000_000


def bvecs_shape(path: Path, gz: bool = False) -> tuple[int, int]:
    opener = gzip.open if gz else open
    with opener(path, "rb") as f:
        dim_bytes = f.read(4)
    if len(dim_bytes) != 4:
        raise RuntimeError(f"cannot read bvecs dimension from {path}")
    dim = struct.unpack("<i", dim_bytes)[0]
    if dim <= 0:
        raise RuntimeError(f"invalid bvecs dimension {dim} in {path}")
    if gz:
        raise RuntimeError("gzipped bvecs shape needs decompression or u8bin header for N")
    rec = 4 + dim
    size = path.stat().st_size
    if size % rec != 0:
        raise RuntimeError(f"{path} size {size} is not a multiple of record {rec}")
    return size // rec, dim


def u8bin_shape(path: Path) -> tuple[int, int]:
    n, d = np.fromfile(path, dtype=np.int32, count=2)
    return int(n), int(d)


def infer_base_shape() -> tuple[int, int, Path]:
    if BASE_U8BIN.exists():
        n, d = u8bin_shape(BASE_U8BIN)
        return n, d, BASE_U8BIN
    if BASE_BVECS.exists():
        n, d = bvecs_shape(BASE_BVECS)
        return n, d, BASE_BVECS
    if BASE_BVECS_GZ.exists():
        # BigANN/SIFT1B canonical size. Keep this path usable before the base is
        # decompressed; conversion still writes the exact u8bin later.
        return 1_000_000_000, 128, BASE_BVECS_GZ
    raise FileNotFoundError(f"missing {BASE_U8BIN} or {BASE_BVECS}(.gz)")


def load_bvecs(path: Path, gz: bool = False) -> np.ndarray:
    opener = gzip.open if gz else open
    with opener(path, "rb") as f:
        first = f.read(4)
        if len(first) != 4:
            raise RuntimeError(f"empty bvecs: {path}")
        dim = struct.unpack("<i", first)[0]
        rest = f.read()
    arr = np.frombuffer(first + rest, dtype=np.uint8)
    rec = 4 + dim
    if arr.size % rec != 0:
        raise RuntimeError(f"{path} byte count {arr.size} not divisible by {rec}")
    return np.ascontiguousarray(arr.reshape(arr.size // rec, rec)[:, 4:], dtype=np.uint8)


def leaf_to_tags(leaf: np.ndarray) -> np.ndarray:
    out = np.empty((leaf.shape[0], 4), dtype=np.uint32)
    out[:, 0] = OFFSETS[0] + (leaf // 64)
    out[:, 1] = OFFSETS[1] + (leaf // 16)
    out[:, 2] = OFFSETS[2] + (leaf // 4)
    out[:, 3] = OFFSETS[3] + leaf
    return out


def main() -> None:
    MT.mkdir(parents=True, exist_ok=True)
    QDIR.mkdir(parents=True, exist_ok=True)
    n, dim, base_file = infer_base_shape()
    if dim != 128:
        raise RuntimeError(f"expected SIFT dim=128, got {dim}")
    print(f"sift1b: N={n:,} dim={dim} base={base_file}", flush=True)

    rng = np.random.default_rng(SEED)
    tags = np.lib.format.open_memmap(MT / "tags.npy", mode="w+", dtype=np.uint32, shape=(n, 4))
    price = np.lib.format.open_memmap(MT / "num_attr.npy", mode="w+", dtype=np.int32, shape=(n,))
    t0 = time.perf_counter()
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        k = e - s
        leaf = rng.integers(0, N_LEAF, size=k, dtype=np.int64)
        tags[s:e] = leaf_to_tags(leaf)
        price[s:e] = rng.integers(0, PRICE_MAX, size=k, dtype=np.int64).astype(np.int32)
        print(f"  vectors {e:,}/{n:,} ({time.perf_counter() - t0:.0f}s)", flush=True)
    tags.flush()
    price.flush()
    del tags, price

    ten = np.lib.format.open_memmap(MT / "tenant_ids.npy", mode="w+", dtype=np.int32, shape=(n,))
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        ten[s:e] = 0
    ten.flush()
    del ten

    qpath = QUERY_BVECS if QUERY_BVECS.exists() else QUERY_BVECS_GZ
    if not qpath.exists():
        raise FileNotFoundError(f"missing query bvecs: {QUERY_BVECS}(.gz)")
    qv_u8 = load_bvecs(qpath, gz=qpath.suffix == ".gz")
    nq = qv_u8.shape[0]
    qleaf = rng.integers(0, N_LEAF, size=nq, dtype=np.int64)
    np.save(QDIR / "query_vectors.npy", qv_u8.astype(np.float32))
    np.save(QDIR / "query_tags.npy", leaf_to_tags(qleaf))
    np.save(QDIR / "query_tenant_ids.npy", np.zeros(nq, dtype=np.int32))

    scenario = {
        "schema_version": 1,
        "dataset": "sift1b",
        "data_file": str(BASE_U8BIN if BASE_U8BIN.exists() else base_file),
        "vector_count": n,
        "dimension": dim,
        "data_dtype": "uint8",
        "metric": "l2",
        "seed": SEED,
        "num_tenants": 1,
        "tenant_file": str(MT / "tenant_ids.npy"),
        "tenant_counts": {"0": n},
        "attributes": {
            "acl": {
                "type": "categorical_hierarchy",
                "file": str(MT / "tags.npy"),
                "shape": [n, 4],
                "dtype": "uint32",
                "tag_levels": LEVELS,
                "tag_level_cardinalities": CARD,
                "tag_level_offsets": OFFSETS,
                "tag_path_assignment": "uniform-random-project-leaf",
                "total_tags": sum(CARD),
            },
            "numeric": {
                "type": "range",
                "file": str(MT / "num_attr.npy"),
                "shape": [n],
                "dtype": "int32",
                "name": "price",
                "range": [0, PRICE_MAX],
                "distribution": "uniform",
                "predicate": "price < X",
                "sweep": PRICE_SWEEP,
            },
        },
        "query": {
            "count": nq,
            "vectors": str(QDIR / "query_vectors.npy"),
            "acl_tags": str(QDIR / "query_tags.npy"),
            "tenant_ids": str(QDIR / "query_tenant_ids.npy"),
            "acl_level_column": {"org": 0, "dept": 1, "team": 2, "project": 3},
        },
    }
    with open(MT / "tenant_tag_scenario.json", "w", encoding="utf-8") as f:
        json.dump(scenario, f, indent=2)
    print(f"done. artifacts in {MT}", flush=True)


if __name__ == "__main__":
    main()
