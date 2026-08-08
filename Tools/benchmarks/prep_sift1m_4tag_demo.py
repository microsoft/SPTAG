#!/usr/bin/env python3
"""Prepare a shared UInt8 SIFT1M + four-level ACL input for PipeANN and SPANN."""

import argparse
import json
import struct
from pathlib import Path

import h5py
import numpy as np


DEFAULT_ROOT = Path("/mnt/nvme/baotonglu/mocheng/datasets/sift1m_4tag_demo")
LEVEL_NAMES = ("org", "dept", "team", "project")
LEVEL_CARDINALITIES = (4, 16, 64, 256)
LEVEL_OFFSETS = (0, 4, 20, 84)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-hdf5",
        type=Path,
        default=DEFAULT_ROOT / "raw" / "sift-128-euclidean.hdf5",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--query-count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def rounded_uint8(values: np.ndarray, source: str) -> np.ndarray:
    rounded = np.rint(values)
    if (
        np.any(rounded < 0)
        or np.any(rounded > 255)
        or not np.array_equal(values, rounded)
    ):
        raise ValueError(f"{source} must contain integral values in [0, 255]")
    return np.ascontiguousarray(rounded, dtype=np.uint8)


def write_u8bin_from_hdf5(dataset: h5py.Dataset, output: Path) -> None:
    count, dim = dataset.shape
    with output.open("wb") as handle:
        handle.write(struct.pack("<ii", count, dim))
        for start in range(0, count, 65_536):
            stop = min(start + 65_536, count)
            handle.write(rounded_uint8(dataset[start:stop], dataset.name).tobytes())


def write_u8bin(values: np.ndarray, output: Path) -> None:
    vectors = rounded_uint8(values, str(output))
    with output.open("wb") as handle:
        handle.write(struct.pack("<ii", *vectors.shape))
        handle.write(vectors.tobytes())


def write_spmat(labels: np.ndarray, output: Path) -> None:
    labels = np.ascontiguousarray(labels, dtype=np.int32)
    rows, labels_per_row = labels.shape
    indptr = np.arange(rows + 1, dtype=np.int64) * labels_per_row
    data = np.ones(rows * labels_per_row, dtype=np.float32)
    with output.open("wb") as handle:
        np.asarray([rows, 340, labels.size], dtype=np.int64).tofile(handle)
        indptr.tofile(handle)
        labels.reshape(-1).tofile(handle)
        data.tofile(handle)


def ensure_outputs_available(outputs: list[Path], overwrite: bool) -> None:
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        joined = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite derived demo inputs; pass --overwrite:\n  {joined}"
        )


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    multitenant = output_root / "multitenant"
    query_dir = multitenant / "query"
    pipeann_dir = multitenant / "pipeann"
    build_dir = output_root / "sift1m_build"

    outputs = [
        output_root / "sift1m_base.u8bin",
        output_root / "sift1m_query_120.u8bin",
        multitenant / "tags.npy",
        multitenant / "num_attr.npy",
        query_dir / "query_vectors.npy",
        query_dir / "query_tags.npy",
        query_dir / "query_tenant_ids.npy",
        query_dir / "groundtruth_unfilter_hdf5_local_ids.npy",
        pipeann_dir / "base_acl.spmat",
        *(pipeann_dir / f"query_acl_{level}.spmat" for level in LEVEL_NAMES),
        output_root / "demo_metadata.json",
    ]
    ensure_outputs_available(outputs, args.overwrite)

    if not args.input_hdf5.is_file():
        raise FileNotFoundError(args.input_hdf5)

    multitenant.mkdir(parents=True, exist_ok=True)
    query_dir.mkdir(parents=True, exist_ok=True)
    pipeann_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input_hdf5, "r") as source:
        train = source["train"]
        test = source["test"]
        neighbors = source["neighbors"]
        if train.shape != (1_000_000, 128):
            raise ValueError(f"Expected SIFT1M train shape (1000000, 128), got {train.shape}")
        if test.ndim != 2 or test.shape[1] != 128:
            raise ValueError(f"Expected 128-D test vectors, got {test.shape}")
        if args.query_count <= 20 or args.query_count > test.shape[0]:
            raise ValueError(
                f"query-count must be in [21, {test.shape[0]}], got {args.query_count}"
            )
        if neighbors.shape[0] < args.query_count or neighbors.shape[1] < 10:
            raise ValueError(f"Need at least {args.query_count} x 10 HDF5 neighbors")

        write_u8bin_from_hdf5(train, output_root / "sift1m_base.u8bin")
        query_u8 = rounded_uint8(test[: args.query_count], "test")
        write_u8bin(query_u8, output_root / "sift1m_query_120.u8bin")
        hdf5_groundtruth = np.asarray(
            neighbors[: args.query_count, :10], dtype=np.int64
        )

    rng = np.random.default_rng(args.seed)
    leaves = rng.integers(0, LEVEL_CARDINALITIES[-1], size=1_000_000, dtype=np.uint32)
    tags = np.empty((1_000_000, 4), dtype=np.uint32)
    tags[:, 0] = leaves // 64 + LEVEL_OFFSETS[0]
    tags[:, 1] = leaves // 16 + LEVEL_OFFSETS[1]
    tags[:, 2] = leaves // 4 + LEVEL_OFFSETS[2]
    tags[:, 3] = leaves + LEVEL_OFFSETS[3]
    numeric = rng.integers(0, 100_000, size=1_000_000, dtype=np.int32)

    np.save(multitenant / "tags.npy", tags)
    np.save(multitenant / "num_attr.npy", numeric)
    np.save(query_dir / "query_vectors.npy", query_u8.astype(np.float32))
    query_tags = np.ascontiguousarray(tags[hdf5_groundtruth[:, 0]], dtype=np.uint32)
    np.save(query_dir / "query_tags.npy", query_tags)
    np.save(query_dir / "query_tenant_ids.npy", np.zeros(args.query_count, dtype=np.int32))
    np.save(query_dir / "groundtruth_unfilter_hdf5_local_ids.npy", hdf5_groundtruth)

    write_spmat(tags, pipeann_dir / "base_acl.spmat")
    for level, column in zip(LEVEL_NAMES, range(len(LEVEL_NAMES)), strict=True):
        write_spmat(query_tags[:, column : column + 1], pipeann_dir / f"query_acl_{level}.spmat")

    metadata = {
        "dataset": "SIFT1M",
        "source_hdf5": str(args.input_hdf5.resolve()),
        "base_count": 1_000_000,
        "dimension": 128,
        "query_count": args.query_count,
        "warmup_count": 20,
        "measured_count": args.query_count - 20,
        "metric": "l2",
        "dtype": "uint8",
        "seed": args.seed,
        "acl_levels": list(LEVEL_NAMES),
        "acl_cardinalities": list(LEVEL_CARDINALITIES),
        "acl_offsets": list(LEVEL_OFFSETS),
        "pipeann_label_columns": 340,
        "hdf5_groundtruth": str(query_dir / "groundtruth_unfilter_hdf5_local_ids.npy"),
    }
    (output_root / "demo_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared shared SIFT1M demo inputs under {output_root}")


if __name__ == "__main__":
    main()
