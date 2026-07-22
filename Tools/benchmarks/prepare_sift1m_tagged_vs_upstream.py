#!/usr/bin/env python3
"""Materialize identical tenant-0 SIFT1M inputs for tagged and vanilla SPANN."""

import argparse
import hashlib
import json
import struct
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DEFAULT_SCENARIO = "/home/v-mochengli/datasets/sift1m/multitenant/tenant_tag_scenario_1m.json"
DEFAULT_OUTPUT = "/datadisk/yfcc_fast/sptag_sift1m_tagged_vs_upstream"
DEFAULT_PIPEPQ_CODES = "/home/v-mochengli/datasets/sift1m/multitenant/pipeann/index/sift1m_cos_pq_compressed.bin"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fvecs_info(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        dimension = struct.unpack("<i", stream.read(4))[0]
    record_bytes = (dimension + 1) * np.dtype(np.float32).itemsize
    size = path.stat().st_size
    if size % record_bytes:
        raise ValueError(f"{path}: size is not a whole number of fvecs records")
    return dimension, size // record_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-file", default=DEFAULT_SCENARIO)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--tenant", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--benchmark-query-count", type=int, default=1000)
    parser.add_argument("--pipepq-reference-codes", default=DEFAULT_PIPEPQ_CODES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario_path = Path(args.scenario_file)
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    base_path = Path(scenario["data_file"])
    tenant_path = Path(scenario["tenant_file"])
    tags_path = Path(scenario["tag_file"])
    query_dir = scenario_path.parent / "query"
    query_meta = json.loads((query_dir / "query_meta.json").read_text(encoding="utf-8"))
    query_source = Path(query_meta["query_file"])

    dimension, vector_count = fvecs_info(base_path)
    if dimension != scenario["dimension"] or vector_count != scenario["vector_count"]:
        raise ValueError("scenario and base fvecs dimensions/count do not match")

    tenant_ids = np.loadtxt(tenant_path, dtype=np.int64).reshape(-1)
    tags = np.load(tags_path, mmap_mode="r", allow_pickle=False)
    if tenant_ids.size != vector_count or tags.shape != (vector_count, 4) or tags.dtype != np.uint32:
        raise ValueError("tenant IDs or tags are not aligned with the base vectors")

    selected = np.flatnonzero(tenant_ids == args.tenant)
    if selected.size == 0:
        raise ValueError(f"tenant {args.tenant} has no vectors")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"tenant{args.tenant}_base.f32"
    fbin_path = output_dir / f"tenant{args.tenant}_base.fbin"
    normalized_raw_path = output_dir / f"tenant{args.tenant}_base_normalized.f32"
    normalized_fbin_path = output_dir / f"tenant{args.tenant}_base_normalized.fbin"
    pipepq_codes_path = output_dir / f"tenant{args.tenant}_pipepq32_codes.bin"
    raw_tags_path = output_dir / f"tenant{args.tenant}_tags.u32"
    truth_path = output_dir / f"tenant{args.tenant}_unfilter_top10.truth.bin"
    query_path = output_dir / f"tenant{args.tenant}_query_{args.benchmark_query_count}.fvecs"
    query_truth_path = output_dir / f"tenant{args.tenant}_unfilter_top10_{args.benchmark_query_count}.truth.bin"

    records = np.memmap(base_path, dtype=np.float32, mode="r", shape=(vector_count, dimension + 1))
    with (
        raw_path.open("wb") as raw_stream,
        fbin_path.open("wb") as fbin_stream,
        normalized_raw_path.open("wb") as normalized_raw_stream,
        normalized_fbin_path.open("wb") as normalized_fbin_stream,
        raw_tags_path.open("wb") as tag_stream,
    ):
        fbin_stream.write(struct.pack("<ii", int(selected.size), dimension))
        normalized_fbin_stream.write(struct.pack("<ii", int(selected.size), dimension))
        for start in range(0, selected.size, args.chunk_size):
            rows = selected[start:start + args.chunk_size]
            vectors = np.ascontiguousarray(records[rows, 1:], dtype=np.float32)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            np.maximum(norms, 1e-12, out=norms)
            normalized_vectors = np.ascontiguousarray(vectors / norms, dtype=np.float32)
            local_tags = np.ascontiguousarray(tags[rows], dtype=np.uint32)
            vectors.tofile(raw_stream)
            vectors.tofile(fbin_stream)
            normalized_vectors.tofile(normalized_raw_stream)
            normalized_vectors.tofile(normalized_fbin_stream)
            local_tags.tofile(tag_stream)

    reference_codes_path = Path(args.pipepq_reference_codes)
    with reference_codes_path.open("rb") as stream:
        reference_count, reference_width = struct.unpack("<ii", stream.read(8))
    if reference_count != vector_count or reference_width != 32:
        raise ValueError("PipePQ reference codes do not match the scenario vector count or 32-byte code width")
    reference_codes = np.memmap(
        reference_codes_path,
        dtype=np.uint8,
        mode="r",
        offset=8,
        shape=(reference_count, reference_width),
    )
    np.ascontiguousarray(reference_codes[selected], dtype=np.uint8).tofile(pipepq_codes_path)

    truth_ids = np.load(query_dir / "groundtruth_unfilter_local_ids.npy", allow_pickle=False)
    truth_dists = np.load(query_dir / "groundtruth_unfilter_dists.npy", allow_pickle=False)
    if truth_ids.ndim != 2 or truth_ids.shape != truth_dists.shape or truth_ids.shape[1] != 10:
        raise ValueError("expected aligned top-10 unfiltered ground truth")
    if np.any(truth_ids < 0) or np.any(truth_ids >= selected.size):
        raise ValueError("ground-truth IDs are outside the tenant-local input range")
    with truth_path.open("wb") as stream:
        stream.write(struct.pack("<ii", truth_ids.shape[0], truth_ids.shape[1]))
        np.ascontiguousarray(truth_ids, dtype="<i4").tofile(stream)
        np.ascontiguousarray(truth_dists, dtype="<f4").tofile(stream)

    query_dimension, query_count = fvecs_info(query_source)
    if query_dimension != dimension or args.benchmark_query_count <= 0 or args.benchmark_query_count > query_count:
        raise ValueError("benchmark query count is not available from the source query file")
    query_bytes = (dimension + 1) * np.dtype(np.float32).itemsize * args.benchmark_query_count
    with query_source.open("rb") as source, query_path.open("wb") as destination:
        destination.write(source.read(query_bytes))
    with query_truth_path.open("wb") as stream:
        stream.write(struct.pack("<ii", args.benchmark_query_count, truth_ids.shape[1]))
        np.ascontiguousarray(truth_ids[:args.benchmark_query_count], dtype="<i4").tofile(stream)
        np.ascontiguousarray(truth_dists[:args.benchmark_query_count], dtype="<f4").tofile(stream)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scenario_file": str(scenario_path.resolve()),
        "source_base_file": str(base_path.resolve()),
        "tenant": args.tenant,
        "vector_count": int(selected.size),
        "dimension": dimension,
        "value_type": "Float",
        "distance": "Cosine",
        "tag_layout": ["org", "dept", "team", "project"],
        "benchmark_query_count": args.benchmark_query_count,
        "files": {
            "raw_vectors": {"path": str(raw_path), "sha256": sha256(raw_path)},
            "default_vectors": {"path": str(fbin_path), "sha256": sha256(fbin_path)},
            "normalized_raw_vectors": {"path": str(normalized_raw_path), "sha256": sha256(normalized_raw_path)},
            "normalized_default_vectors": {
                "path": str(normalized_fbin_path),
                "sha256": sha256(normalized_fbin_path),
            },
            "pipepq32_codes": {"path": str(pipepq_codes_path), "sha256": sha256(pipepq_codes_path)},
            "tags": {"path": str(raw_tags_path), "sha256": sha256(raw_tags_path)},
            "truth": {"path": str(truth_path), "sha256": sha256(truth_path)},
            "benchmark_query": {"path": str(query_path), "sha256": sha256(query_path)},
            "benchmark_truth": {"path": str(query_truth_path), "sha256": sha256(query_truth_path)},
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
