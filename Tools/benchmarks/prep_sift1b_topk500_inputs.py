#!/usr/bin/env python3
"""Prepare shared, exact SIFT1B TopK-500 inputs for PipeANN and SPANN."""

import argparse
import json
import shutil
import struct
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT = Path(
    "/mnt/nvme/baotonglu/mocheng/pipeann/audits/"
    "sift1b_topk500_scale_check_20260720/inputs"
)
QUERY_DIR = Path("/mnt/nvme/baotonglu/mocheng/datasets/sift1b/multitenant/query")
QUERY_BIN = Path("/mnt/nvme/baotonglu/mocheng/datasets/sift1b/query.u8bin")
GROUNDTRUTH = Path(
    "/mnt/nvme/baotonglu/mocheng/datasets/sift1b/raw/gnd/idx_1000M.ivecs"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--topk", type=int, default=500)
    parser.add_argument("--query-count", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 < args.topk <= 1000:
        parser.error("--topk must be in [1, 1000]")
    if args.query_count < 120:
        parser.error("--query-count must be at least 120 for 20 warmups and 100 measurements")
    return args


def load_ivecs(path: Path) -> np.ndarray:
    words = np.memmap(path, dtype="<i4", mode="r")
    if words.size % 1001:
        raise ValueError(f"{path} is not an integer number of 1000-neighbor ivecs rows")
    rows = words.reshape(-1, 1001)
    if not np.all(rows[:, 0] == 1000):
        raise ValueError(f"{path} does not contain 1000-neighbor ivecs rows")
    return rows[:, 1:]


def load_u8bin(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = handle.read(8)
        count, dim = struct.unpack("<ii", header)
        values = np.fromfile(handle, dtype=np.uint8, count=count * dim)
    if values.size != count * dim:
        raise ValueError(f"{path} is truncated")
    return values.reshape(count, dim)


def write_u8bin(path: Path, values: np.ndarray) -> None:
    values = np.ascontiguousarray(values, dtype=np.uint8)
    with path.open("wb") as handle:
        handle.write(struct.pack("<ii", *values.shape))
        values.tofile(handle)


def write_truthset(path: Path, ids: np.ndarray) -> None:
    ids = np.ascontiguousarray(ids, dtype=np.uint32)
    with path.open("wb") as handle:
        handle.write(struct.pack("<ii", *ids.shape))
        ids.tofile(handle)


def main() -> None:
    args = parse_args()
    output = args.output
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")
        shutil.rmtree(output)

    raw_ids = load_ivecs(GROUNDTRUTH)
    source_gt = np.load(QUERY_DIR / "groundtruth_unfilter_local_ids.npy", mmap_mode="r")
    if (
        source_gt.shape[0] < args.query_count
        or source_gt.shape[1] < 100
        or not np.array_equal(raw_ids[: args.query_count, :100], source_gt[: args.query_count, :100])
    ):
        raise RuntimeError("Official SIFT1B 1000-neighbor truth does not match the existing TopK-100 input")

    query_vectors = np.load(QUERY_DIR / "query_vectors.npy", mmap_mode="r")[: args.query_count]
    query_tags = np.load(QUERY_DIR / "query_tags.npy", mmap_mode="r")[: args.query_count]
    query_tenants = np.load(QUERY_DIR / "query_tenant_ids.npy", mmap_mode="r")[: args.query_count]
    native_queries = load_u8bin(QUERY_BIN)[: args.query_count]
    if not np.array_equal(native_queries, np.rint(query_vectors).astype(np.uint8)):
        raise RuntimeError("The native UInt8 queries disagree with query_vectors.npy")

    exact_ids = np.ascontiguousarray(raw_ids[: args.query_count, : args.topk], dtype=np.int64)
    if np.any(exact_ids < 0) or any(len(set(row)) != args.topk for row in exact_ids):
        raise RuntimeError("TopK ground truth contains invalid or duplicate IDs")

    spann_dir = output / "spann_query"
    pipeann_dir = output / "pipeann_query"
    spann_dir.mkdir(parents=True)
    pipeann_dir.mkdir(parents=True)
    np.save(spann_dir / "query_vectors.npy", np.ascontiguousarray(query_vectors))
    np.save(spann_dir / "query_tags.npy", np.ascontiguousarray(query_tags, dtype=np.uint32))
    np.save(spann_dir / "query_tenant_ids.npy", np.ascontiguousarray(query_tenants, dtype=np.int32))
    np.save(spann_dir / "groundtruth_unfilter_local_ids.npy", exact_ids)
    write_u8bin(pipeann_dir / "query.u8bin", native_queries)
    write_truthset(pipeann_dir / "gt_unfilter.ibin", exact_ids)

    manifest = {
        "dataset": "sift1b",
        "topk": args.topk,
        "query_count": args.query_count,
        "warmup_queries": 20,
        "measured_queries": 100,
        "groundtruth_source": str(GROUNDTRUTH),
        "top100_matches_existing_groundtruth": True,
    }
    (output / "metadata.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Prepared exact TopK-{args.topk} inputs under {output}")


if __name__ == "__main__":
    main()
