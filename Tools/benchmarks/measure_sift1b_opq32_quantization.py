#!/usr/bin/env python3
"""Compare SIFT1B PipePQ32 and OPQ32 reconstruction distortion at equal code width."""

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def read_pipepq_pivots(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        rows, columns = struct.unpack("=II", handle.read(8))
        if (rows, columns) != (5, 1):
            raise ValueError(f"Invalid PipePQ offsets header: {(rows, columns)}")
        offsets = struct.unpack("=5Q", handle.read(40))

    def matrix(offset: int, dtype: np.dtype) -> np.ndarray:
        with path.open("rb") as handle:
            handle.seek(offset)
            rows, columns = struct.unpack("=II", handle.read(8))
        return np.memmap(
            path,
            mode="r",
            dtype=dtype,
            offset=offset + 8,
            shape=(rows, columns),
        )

    tables = matrix(offsets[0], np.float32)
    centroid = matrix(offsets[1], np.float32).reshape(-1)
    chunk_offsets = matrix(offsets[3], np.uint32).reshape(-1)
    if tables.shape != (256, 128) or centroid.shape != (128,) or chunk_offsets.shape != (33,):
        raise ValueError(
            f"Unexpected PipePQ layout: tables={tables.shape}, centroid={centroid.shape}, "
            f"chunks={chunk_offsets.shape}"
        )
    return tables, centroid, chunk_offsets


def read_opq_quantizer(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = path.read_bytes()
    kind, value_type, m, k, subdim = struct.unpack_from("=BBiii", raw)
    if (kind, value_type, m, k, subdim) != (2, 3, 32, 256, 4):
        raise ValueError(
            "Expected Float OPQ32x256x4 quantizer, got "
            f"{(kind, value_type, m, k, subdim)}"
        )
    offset = struct.calcsize("=BBiii")
    codebook_values = m * k * subdim
    codebooks = np.frombuffer(
        raw, dtype=np.float32, count=codebook_values, offset=offset
    ).reshape(m, k, subdim)
    offset += codebook_values * np.dtype(np.float32).itemsize
    rotation = np.frombuffer(
        raw, dtype=np.float32, count=128 * 128, offset=offset
    ).reshape(128, 128)
    if offset + rotation.nbytes != len(raw):
        raise ValueError("Unexpected trailing bytes in OPQ quantizer")
    return codebooks, rotation


def reconstruction_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    error = original - reconstructed
    squared_l2 = np.einsum("ij,ij->i", error, error)
    return {
        "mse_per_dimension": float(np.mean(error * error)),
        "mean_squared_l2": float(np.mean(squared_l2)),
        "p50_squared_l2": float(np.quantile(squared_l2, 0.5)),
        "p95_squared_l2": float(np.quantile(squared_l2, 0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("/mnt/nvme/baotonglu/mocheng/datasets/sift1b/sift1b_base.u8bin"),
    )
    parser.add_argument(
        "--pipepq-codes",
        type=Path,
        default=Path(
            "/mnt/nvme/baotonglu/mocheng/datasets/sift1b/sift1b_build/"
            "sift1b_pq_compressed.bin"
        ),
    )
    parser.add_argument(
        "--pipepq-pivots",
        type=Path,
        default=Path(
            "/mnt/nvme/baotonglu/mocheng/datasets/sift1b/sift1b_build/"
            "sift1b_pq_pivots.bin"
        ),
    )
    parser.add_argument(
        "--opq-codes",
        type=Path,
        default=Path(
            "/mnt/nvme/baotonglu/mocheng/datasets/sift1b/sift1b_opq32_build/"
            "opq_codes_m32.bin"
        ),
    )
    parser.add_argument(
        "--opq-quantizer",
        type=Path,
        default=Path(
            "/mnt/nvme/baotonglu/mocheng/datasets/sift1b/sift1b_opq32_build/"
            "opq_quantizer.bin"
        ),
    )
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.samples <= 0 or args.samples > 1_000_000_000:
        raise ValueError("--samples must be in [1, 1_000_000_000]")
    with args.base.open("rb") as handle:
        base_n, base_dim = struct.unpack("=ii", handle.read(8))
    if (base_n, base_dim) != (1_000_000_000, 128):
        raise ValueError(f"Unexpected SIFT1B base header: {(base_n, base_dim)}")
    with args.pipepq_codes.open("rb") as handle:
        pipe_n, pipe_m = struct.unpack("=II", handle.read(8))
    if (pipe_n, pipe_m) != (base_n, 32):
        raise ValueError(f"Unexpected PipePQ code header: {(pipe_n, pipe_m)}")
    if args.opq_codes.stat().st_size != base_n * 32:
        raise ValueError("OPQ sidecar is not a 1B x 32-byte raw code matrix")

    vector_ids = np.linspace(0, base_n - 1, args.samples, dtype=np.int64)
    base = np.memmap(args.base, mode="r", dtype=np.uint8, offset=8, shape=(base_n, 128))
    original = np.asarray(base[vector_ids], dtype=np.float32)

    pipe_codes_all = np.memmap(
        args.pipepq_codes, mode="r", dtype=np.uint8, offset=8, shape=(base_n, 32)
    )
    pipe_codes = np.asarray(pipe_codes_all[vector_ids])
    pipe_tables, pipe_centroid, chunk_offsets = read_pipepq_pivots(args.pipepq_pivots)
    pipe_reconstructed = np.empty_like(original)
    for chunk in range(32):
        begin, end = int(chunk_offsets[chunk]), int(chunk_offsets[chunk + 1])
        pipe_reconstructed[:, begin:end] = (
            pipe_tables[pipe_codes[:, chunk], begin:end] + pipe_centroid[begin:end]
        )

    opq_codes_all = np.memmap(
        args.opq_codes, mode="r", dtype=np.uint8, shape=(base_n, 32)
    )
    opq_codes = np.asarray(opq_codes_all[vector_ids])
    opq_codebooks, opq_matrix = read_opq_quantizer(args.opq_quantizer)
    rotated_reconstructed = np.empty_like(original)
    for chunk in range(32):
        begin, end = chunk * 4, (chunk + 1) * 4
        rotated_reconstructed[:, begin:end] = opq_codebooks[chunk, opq_codes[:, chunk]]
    opq_reconstructed = rotated_reconstructed @ opq_matrix.T

    pipe_metrics = reconstruction_metrics(original, pipe_reconstructed)
    opq_metrics = reconstruction_metrics(original, opq_reconstructed)
    result = {
        "dataset": "sift1b",
        "sample_count": int(args.samples),
        "sample_strategy": "uniformly spaced vector IDs across all 1B base vectors",
        "code_bytes_per_vector": 32,
        "pipepq32": pipe_metrics,
        "opq32": opq_metrics,
        "opq_over_pipepq_mse_ratio": opq_metrics["mse_per_dimension"]
        / pipe_metrics["mse_per_dimension"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
