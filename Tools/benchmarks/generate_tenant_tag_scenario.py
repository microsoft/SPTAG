#!/usr/bin/env python3
"""Generate multi-tenant ids and hierarchical virtual tags for a SIFT-style dataset.

Outputs (consumed by the other Tools/benchmarks scripts via a scenario json):

* ``tenant_ids.txt`` : one int tenant internal-id per base vector (``np.loadtxt``).
* ``tags.npy``       : ``(vector_count, 4)`` uint32 matrix, columns ordered
                       ``[org, dept, team, project]`` with GLOBALLY UNIQUE tag ids
                       across the four levels (required by the groundtruth /
                       stress scripts which index every tag value in one dict).
* ``tenant_tag_scenario_<tag>.json`` : manifest referenced by the consumers.

Tenant sizes follow a Zipf law ``count_k ~ 1 / k**s``. The exponent ``s`` is
solved numerically so the largest tenant (tenant 0) matches a requested size.

Tags form a strict nested 4-ary tree shared globally across tenants:
``4 org -> 16 dept -> 64 team -> 256 project`` (4 children per node). Each base
vector is assigned a leaf ``project`` uniformly at random; the parent
team/dept/org are derived by nesting.
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


LEVEL_NAMES = ("org", "dept", "team", "project")
# Cardinality per level of the nested 4-ary tag tree.
LEVEL_CARDINALITIES = (4, 16, 64, 256)
# Global id offsets so tag ids never collide across levels:
#   org     -> [0, 4)
#   dept    -> [4, 20)
#   team    -> [20, 84)
#   project -> [84, 340)
LEVEL_OFFSETS = (0, 4, 20, 84)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_fvecs_count(path: str) -> tuple[int, int]:
    """Return (dimension, vector_count) of an .fvecs file without loading it."""
    with open(path, "rb") as handle:
        dimension = int(np.frombuffer(handle.read(4), dtype=np.int32)[0])
    record_bytes = (dimension + 1) * 4
    file_bytes = os.path.getsize(path)
    if file_bytes % record_bytes != 0:
        raise RuntimeError(f"{path}: size {file_bytes} not a multiple of record {record_bytes}")
    return dimension, file_bytes // record_bytes


def solve_zipf_exponent(
    num_tenants: int,
    total_count: int,
    target_top_count: int,
    tol: float = 1e-9,
    max_iter: int = 200,
) -> float:
    """Solve s so that round(total * (1/1^s) / sum_k 1/k^s) ~= target_top_count.

    The top-bucket fraction p1(s) = 1 / sum_{k=1..N} k**-s is monotonically
    increasing in s, so a simple bisection converges.
    """
    ranks = np.arange(1, num_tenants + 1, dtype=np.float64)
    target_fraction = target_top_count / total_count
    if not (1.0 / num_tenants < target_fraction < 1.0):
        raise RuntimeError(
            f"target fraction {target_fraction:.6f} unreachable with {num_tenants} tenants"
        )

    def top_fraction(s: float) -> float:
        return 1.0 / np.sum(ranks ** (-s))

    low, high = 0.0, 1.0
    while top_fraction(high) < target_fraction and high < 1e6:
        high *= 2.0
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        if top_fraction(mid) < target_fraction:
            low = mid
        else:
            high = mid
        if high - low < tol:
            break
    return 0.5 * (low + high)


def zipf_counts(num_tenants: int, total_count: int, exponent: float) -> np.ndarray:
    """Integer per-tenant counts ~ 1/k^s summing exactly to total_count.

    Uses largest-remainder rounding to preserve the total while keeping counts
    as close as possible to the ideal real-valued Zipf weights.
    """
    ranks = np.arange(1, num_tenants + 1, dtype=np.float64)
    weights = ranks ** (-exponent)
    ideal = weights / weights.sum() * total_count
    floor = np.floor(ideal).astype(np.int64)
    remainder = total_count - int(floor.sum())
    # Distribute the leftover to the largest fractional parts.
    fractional_order = np.argsort(-(ideal - floor))
    for idx in fractional_order[:remainder]:
        floor[idx] += 1
    return floor


def assign_tenants(vector_count: int, counts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomly assign each vector to a tenant respecting the exact counts."""
    tenant_ids = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    if tenant_ids.size != vector_count:
        raise RuntimeError(f"tenant counts sum {tenant_ids.size} != vector_count {vector_count}")
    rng.shuffle(tenant_ids)
    return tenant_ids


def assign_tags(vector_count: int, rng: np.random.Generator) -> np.ndarray:
    """Assign a nested-tree tag path to every vector via a uniform leaf draw.

    Returns a (vector_count, 4) uint32 matrix with globally unique tag ids in
    column order [org, dept, team, project].
    """
    num_projects = LEVEL_CARDINALITIES[3]
    project = rng.integers(0, num_projects, size=vector_count, dtype=np.int64)
    team = project // 4
    dept = team // 4
    org = dept // 4

    tags = np.empty((vector_count, 4), dtype=np.uint32)
    tags[:, 0] = org + LEVEL_OFFSETS[0]
    tags[:, 1] = dept + LEVEL_OFFSETS[1]
    tags[:, 2] = team + LEVEL_OFFSETS[2]
    tags[:, 3] = project + LEVEL_OFFSETS[3]
    return tags


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-file",
        default="/home/v-mochengli/datasets/sift1m/sift/sift_base.fvecs",
        help="Base vectors .fvecs file (used to derive vector_count).",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/v-mochengli/datasets/sift1m/multitenant",
        help="Directory to write tenant_ids.txt, tags.npy and the scenario json.",
    )
    parser.add_argument("--num-tenants", type=int, default=10)
    parser.add_argument(
        "--tenant0-count",
        type=int,
        default=404819,
        help="Target vector count for tenant 0; the Zipf exponent is solved to match it.",
    )
    parser.add_argument(
        "--zipf-exponent",
        type=float,
        default=None,
        help="Override the Zipf exponent instead of solving it from --tenant0-count.",
    )
    parser.add_argument("--seed", type=int, default=20260413)
    parser.add_argument(
        "--scenario-tag",
        default="1m",
        help="Suffix used in the scenario json filename (tenant_tag_scenario_<tag>.json).",
    )
    parser.add_argument(
        "--index-dir",
        default="",
        help="Path of the (to-be-)built tenant index; stored in the scenario json. "
        "Defaults to <output-dir>/index.",
    )
    parser.add_argument("--vector-count", type=int, default=0, help="Override vector count (0 = read from data file).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dimension, file_vector_count = read_fvecs_count(args.data_file)
    vector_count = args.vector_count if args.vector_count > 0 else file_vector_count
    if vector_count > file_vector_count:
        raise RuntimeError(f"requested {vector_count} vectors but data file only has {file_vector_count}")

    if args.zipf_exponent is not None:
        exponent = float(args.zipf_exponent)
    else:
        exponent = solve_zipf_exponent(args.num_tenants, vector_count, args.tenant0_count)

    counts = zipf_counts(args.num_tenants, vector_count, exponent)

    rng = np.random.default_rng(args.seed)
    tenant_ids = assign_tenants(vector_count, counts, rng)
    tags = assign_tags(vector_count, rng)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tenant_file = output_dir / "tenant_ids.txt"
    tag_file = output_dir / "tags.npy"
    scenario_file = output_dir / f"tenant_tag_scenario_{args.scenario_tag}.json"
    index_dir = args.index_dir or str(output_dir / "index")

    np.savetxt(tenant_file, tenant_ids, fmt="%d")
    np.save(tag_file, tags)

    tenant_mapping = {str(i): int(i) for i in range(args.num_tenants)}
    tenant_counts = {str(i): int(counts[i]) for i in range(args.num_tenants)}

    scenario = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "dataset": "sift1m",
        "data_file": str(Path(args.data_file).resolve()),
        "vector_count": int(vector_count),
        "dimension": int(dimension),
        "tenant_file": str(tenant_file.resolve()),
        "tag_file": str(tag_file.resolve()),
        "index_dir": index_dir,
        "num_tenants": int(args.num_tenants),
        "tenant_mapping": tenant_mapping,
        "tenant_counts": tenant_counts,
        "zipf_exponent": float(exponent),
        "tag_levels": list(LEVEL_NAMES),
        "tag_level_cardinalities": list(LEVEL_CARDINALITIES),
        "tag_level_offsets": list(LEVEL_OFFSETS),
        "tag_hierarchy": "nested-4ary",
        "tag_assignment": "uniform-random-leaf",
        "seed": int(args.seed),
    }
    scenario_file.write_text(json.dumps(scenario, indent=2) + "\n", encoding="utf-8")

    print(f"vector_count    : {vector_count}")
    print(f"zipf_exponent s : {exponent:.6f}")
    print("tenant_counts   :")
    for i in range(args.num_tenants):
        print(f"  tenant {i}: {int(counts[i]):>8d}  ({counts[i] / vector_count * 100:5.2f}%)")
    print(f"tenant_ids.txt  : {tenant_file}")
    print(f"tags.npy        : {tag_file}  shape=({vector_count}, 4) dtype=uint32")
    print(f"scenario json   : {scenario_file}")


if __name__ == "__main__":
    main()
