#!/usr/bin/env python3
"""Verify that the SIFT1B OPQ32 index is a topology-preserving PipePQ32 clone."""

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path


MUTABLE_FILES = {
    "indexloader.ini",
    "DeletedIDs.bin",
    "ssdmapping",
    "ssdmapping_postings",
    "ssdmapping_postings_blockpool",
    "ssdinfo",
    "checksum",
    "posting_pure_counts.bin",
}
SOURCE_ONLY_FILES = {"inpost_pipepq.bin"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_ini(path: Path) -> dict[str, dict[str, str]]:
    parsed: dict[str, dict[str, str]] = {}
    section = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            parsed.setdefault(section, {})
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            parsed.setdefault(section, {})[key] = value
    return parsed


def marker(path: Path) -> tuple[int, int]:
    raw = path.read_bytes()
    if len(raw) != 8:
        raise ValueError(f"Invalid marker size for {path}: {len(raw)}")
    return struct.unpack("=ii", raw)


def verify_immutable_hardlinks(source: Path, target: Path) -> int:
    linked_files = 0
    for source_entry in source.rglob("*"):
        relative = source_entry.relative_to(source)
        if relative.parts[0] in MUTABLE_FILES or relative.parts[0] in SOURCE_ONLY_FILES or source_entry.is_dir():
            continue
        target_entry = target / relative
        if not target_entry.is_file():
            raise ValueError(f"Missing immutable clone file: {target_entry}")
        source_stat = source_entry.stat()
        target_stat = target_entry.stat()
        if source_stat.st_ino != target_stat.st_ino:
            raise ValueError(f"Immutable file is not hard-linked: {relative}")
        linked_files += 1
    return linked_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "/mnt/nvme/baotonglu/mocheng/datasets/sift1b/"
            "sift1b_spann_pipepq32_r010_tail1"
        ),
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path(
            "/mnt/nvme/baotonglu/mocheng/datasets/sift1b/"
            "sift1b_spann_opq32_r010_tail1"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.source / "tenant_0"
    target = args.target / "tenant_0"
    if marker(source / "inpost_pipepq.bin") != (32, 57):
        raise ValueError("Source marker is not PipePQ32 with a 57-byte record")
    if (target / "inpost_pipepq.bin").exists():
        raise ValueError("Target still has the PipePQ source marker")
    if marker(target / "inpost_opq.bin") != (32, 57):
        raise ValueError("Target marker is not OPQ32 with a 57-byte record")

    source_ini = parse_ini(source / "indexloader.ini")
    target_ini = parse_ini(target / "indexloader.ini")
    target_ssd = target_ini["BuildSSDIndex"]
    if target_ini["Base"]["IndexDirectory"] != str(target):
        raise ValueError("Target INI does not point to the target tenant")
    expected_target = {
        "PostingQuantizer": "OPQ",
        "PostingQuantM": "32",
        "PostingQuantizerFile": "opq_codes_m32.bin",
        "RequantizeFromPipePQ": "true",
    }
    for key, value in expected_target.items():
        if target_ssd.get(key) != value:
            raise ValueError(f"Target INI {key}={target_ssd.get(key)!r}, expected {value!r}")
    if source_ini["BuildSSDIndex"].get("PostingQuantizer") != "PipePQ":
        raise ValueError("Source INI is not PipePQ")

    for name in MUTABLE_FILES:
        source_path = source / name
        target_path = target / name
        if not source_path.is_file() or not target_path.is_file():
            raise ValueError(f"Missing mutable file: {name}")
        if source_path.stat().st_ino == target_path.stat().st_ino:
            raise ValueError(f"Mutable file remains hard-linked: {name}")

    for name in ("DeletedIDs.bin", "posting_pure_counts.bin", "ssdinfo"):
        if sha256(source / name) != sha256(target / name):
            raise ValueError(f"Topology sidecar changed unexpectedly: {name}")

    code_path = target / "opq_codes_m32.bin"
    quantizer_path = target / "opq_quantizer.bin"
    if code_path.stat().st_size != 32_000_000_000:
        raise ValueError("Target OPQ sidecar has an unexpected size")
    if quantizer_path.stat().st_size == 0:
        raise ValueError("Target OPQ quantizer is empty")

    result = {
        "source": str(args.source),
        "target": str(args.target),
        "source_marker": list(marker(source / "inpost_pipepq.bin")),
        "target_marker": list(marker(target / "inpost_opq.bin")),
        "immutable_hardlinked_file_count": verify_immutable_hardlinks(source, target),
        "topology_sidecar_sha256": {
            name: sha256(target / name)
            for name in ("DeletedIDs.bin", "posting_pure_counts.bin", "ssdinfo")
        },
        "opq_code_bytes": code_path.stat().st_size,
        "opq_quantizer_bytes": quantizer_path.stat().st_size,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
