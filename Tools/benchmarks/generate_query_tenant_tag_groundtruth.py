#!/usr/bin/env python3
"""Generate query tenant/tag assignments and 5 groundtruths for SIFT queries.

All query vectors belong to tenant 0. Each query is given one nested tag path
(a uniformly random ``project`` leaf, from which ``team/dept/org`` are derived).
For every query we then compute five exact top-k groundtruths over the tenant-0
base vectors using the selected distance metric (``--metric l2`` or
``cosine``; use ``cosine`` to match the SPTAG index build):

* ``unfilter`` : all tenant-0 vectors
* ``org``      : tenant-0 vectors whose org     tag matches the query
* ``dept``     : tenant-0 vectors whose dept    tag matches the query
* ``team``     : tenant-0 vectors whose team    tag matches the query
* ``project``  : tenant-0 vectors whose project tag matches the query

Groundtruth neighbor ids are tenant-0 LOCAL ids (row index into the tenant-0
base subset), matching the existing ``groundtruth_local_ids`` convention.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


LEVEL_NAMES = ("org", "dept", "team", "project")
DEFAULT_SCENARIO = "/home/v-mochengli/datasets/sift1m/multitenant/tenant_tag_scenario_1m.json"
DEFAULT_QUERY_FILE = "/home/v-mochengli/datasets/sift1m/sift/sift_query.fvecs"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_fvecs(path: str, max_vectors: int = 0) -> np.ndarray:
    with open(path, "rb") as handle:
        dimension = int(np.frombuffer(handle.read(4), dtype=np.int32)[0])
    stride = dimension + 1
    raw = np.fromfile(path, dtype=np.float32, count=max_vectors * stride if max_vectors else -1)
    count = raw.size // stride
    return np.ascontiguousarray(raw.reshape(count, stride)[:, 1:], dtype=np.float32)


def topk_local_ids(
    base_sub: np.ndarray,
    base_sub_sqnorm: np.ndarray,
    candidate_local_ids: np.ndarray,
    query_batch: np.ndarray,
    query_sqnorm: np.ndarray,
    topk: int,
    metric: str = "l2",
) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k nearest over a candidate subset for a batch of queries.

    Ranking is always by squared L2. For ``metric == "cosine"`` the caller must
    pass L2-normalized vectors, in which case squared L2 = 2 - 2*cos is monotone
    in cosine distance, so the returned ids are the true cosine top-k and the
    stored distance is the cosine distance (1 - cos) = dist_sq / 2.

    Returns (ids[B, topk], dists[B, topk]) where ids are tenant-0 LOCAL ids.
    Missing slots are -1 / inf.
    """
    batch_size = query_batch.shape[0]
    out_ids = np.full((batch_size, topk), -1, dtype=np.int64)
    out_dists = np.full((batch_size, topk), np.inf, dtype=np.float32)
    if candidate_local_ids.size == 0:
        return out_ids, out_dists

    cand_vecs = base_sub[candidate_local_ids]
    cand_sqnorm = base_sub_sqnorm[candidate_local_ids]
    # squared L2 = |c|^2 - 2 c.q + |q|^2  -> (Ncand, B)
    dist_sq = cand_sqnorm[:, None] - 2.0 * (cand_vecs @ query_batch.T) + query_sqnorm[None, :]
    np.maximum(dist_sq, 0.0, out=dist_sq)

    k = min(topk, candidate_local_ids.size)
    for col in range(batch_size):
        column = dist_sq[:, col]
        if k < column.size:
            part = np.argpartition(column, k - 1)[:k]
            order = part[np.argsort(column[part])]
        else:
            order = np.argsort(column)
        out_ids[col, :k] = candidate_local_ids[order]
        if metric == "cosine":
            out_dists[col, :k] = column[order] * 0.5
        else:
            out_dists[col, :k] = np.sqrt(column[order])
    return out_ids, out_dists


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario-file", default=DEFAULT_SCENARIO)
    parser.add_argument("--query-file", default=DEFAULT_QUERY_FILE)
    parser.add_argument("--output-dir", default=None, help="Default: <scenario dir>/query")
    parser.add_argument("--tenant", type=int, default=0, help="Tenant all queries belong to.")
    parser.add_argument("--num-queries", type=int, default=0, help="0 = use all query vectors.")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--query-batch", type=int, default=256, help="Queries processed per matmul batch.")
    parser.add_argument(
        "--metric",
        choices=("l2", "cosine"),
        default="l2",
        help="Distance metric for groundtruth. Use 'cosine' to match the SPTAG index build.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = json.loads(Path(args.scenario_file).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.scenario_file).parent / "query"
    output_dir.mkdir(parents=True, exist_ok=True)

    tenant = args.tenant
    offsets = scenario.get("tag_level_offsets", [0, 4, 20, 84])
    cards = scenario.get("tag_level_cardinalities", [4, 16, 64, 256])
    num_projects = cards[3]

    # --- load base vectors + tenant/tag metadata, restrict to the tenant ---
    base_vectors = read_fvecs(scenario["data_file"], scenario["vector_count"])
    tenant_ids = np.loadtxt(scenario["tenant_file"], dtype=np.int64).reshape(-1)[: scenario["vector_count"]]
    tags = np.asarray(np.load(scenario["tag_file"], allow_pickle=False), dtype=np.uint32)

    tenant_mask = tenant_ids == tenant
    base_sub = np.ascontiguousarray(base_vectors[tenant_mask], dtype=np.float32)
    tags_sub = np.ascontiguousarray(tags[tenant_mask], dtype=np.uint32)
    tenant_vector_count = base_sub.shape[0]
    if tenant_vector_count == 0:
        raise RuntimeError(f"tenant {tenant} has no base vectors")

    if args.metric == "cosine":
        base_norm = np.linalg.norm(base_sub, axis=1, keepdims=True)
        np.maximum(base_norm, 1e-12, out=base_norm)
        base_sub = np.ascontiguousarray(base_sub / base_norm, dtype=np.float32)
    base_sub_sqnorm = np.einsum("ij,ij->i", base_sub, base_sub).astype(np.float32)

    # local-id candidate lists per (level, tag-value)
    level_value_ids: list[dict[int, np.ndarray]] = []
    for level in range(4):
        value_map: dict[int, np.ndarray] = {}
        for value in np.unique(tags_sub[:, level]).tolist():
            value_map[int(value)] = np.flatnonzero(tags_sub[:, level] == value).astype(np.int64)
        level_value_ids.append(value_map)

    # --- queries: all belong to tenant; assign a random nested project leaf ---
    query_vectors = read_fvecs(args.query_file, args.num_queries)
    num_queries = query_vectors.shape[0]
    query_vectors_raw = query_vectors
    if args.metric == "cosine":
        query_norm = np.linalg.norm(query_vectors, axis=1, keepdims=True)
        np.maximum(query_norm, 1e-12, out=query_norm)
        query_vectors = np.ascontiguousarray(query_vectors / query_norm, dtype=np.float32)
    query_sqnorm = np.einsum("ij,ij->i", query_vectors, query_vectors).astype(np.float32)

    rng = np.random.default_rng(args.seed)
    leaf = rng.integers(0, num_projects, size=num_queries, dtype=np.int64)
    org = leaf // 64
    dept = leaf // 16
    team = leaf // 4
    query_tags = np.empty((num_queries, 4), dtype=np.uint32)
    query_tags[:, 0] = org + offsets[0]
    query_tags[:, 1] = dept + offsets[1]
    query_tags[:, 2] = team + offsets[2]
    query_tags[:, 3] = leaf + offsets[3]
    query_tenant_ids = np.full(num_queries, tenant, dtype=np.int64)

    # --- 5 groundtruths ---
    topk = args.topk
    gt = {
        name: {
            "ids": np.full((num_queries, topk), -1, dtype=np.int64),
            "dists": np.full((num_queries, topk), np.inf, dtype=np.float32),
        }
        for name in ("unfilter",) + LEVEL_NAMES
    }
    all_local_ids = np.arange(tenant_vector_count, dtype=np.int64)

    # unfilter: batched matmul over all tenant-0 vectors
    for start in range(0, num_queries, args.query_batch):
        end = min(start + args.query_batch, num_queries)
        ids, dists = topk_local_ids(
            base_sub, base_sub_sqnorm, all_local_ids, query_vectors[start:end], query_sqnorm[start:end], topk,
            metric=args.metric,
        )
        gt["unfilter"]["ids"][start:end] = ids
        gt["unfilter"]["dists"][start:end] = dists
        print(f"  unfilter {end}/{num_queries}", flush=True)

    # filtered levels: group queries by shared tag value so each candidate set
    # is gathered once and all its queries are processed in batched matmuls.
    for level, name in enumerate(LEVEL_NAMES):
        ids_out = gt[name]["ids"]
        dists_out = gt[name]["dists"]
        query_value = query_tags[:, level].astype(np.int64)
        for tag_value in np.unique(query_value).tolist():
            group = np.flatnonzero(query_value == tag_value)
            candidates = level_value_ids[level].get(int(tag_value), np.empty(0, dtype=np.int64))
            for start in range(0, group.size, args.query_batch):
                rows = group[start : start + args.query_batch]
                ci, cd = topk_local_ids(
                    base_sub, base_sub_sqnorm, candidates, query_vectors[rows], query_sqnorm[rows], topk,
                    metric=args.metric,
                )
                ids_out[rows] = ci
                dists_out[rows] = cd
        print(f"  {name} done", flush=True)

    # --- write outputs ---
    np.save(output_dir / "query_vectors.npy", query_vectors_raw)
    np.save(output_dir / "query_tenant_ids.npy", query_tenant_ids)
    np.save(output_dir / "query_tags.npy", query_tags)
    for name in ("unfilter",) + LEVEL_NAMES:
        np.save(output_dir / f"groundtruth_{name}_local_ids.npy", gt[name]["ids"])
        np.save(output_dir / f"groundtruth_{name}_dists.npy", gt[name]["dists"])

    match_counts = {
        name: int(np.count_nonzero(level_value_ids[level].get(int(query_tags[0, level]), np.empty(0))))
        for level, name in enumerate(LEVEL_NAMES)
    }
    meta = {
        "created_at_utc": utc_timestamp(),
        "scenario_file": str(Path(args.scenario_file).resolve()),
        "query_file": str(Path(args.query_file).resolve()),
        "tenant": tenant,
        "num_queries": int(num_queries),
        "topk": topk,
        "metric": args.metric,
        "id_space": "tenant_local",
        "seed": int(args.seed),
        "tag_path_assignment": "uniform-random-project-leaf",
        "query_types": ["unfilter", *LEVEL_NAMES],
        "tenant_vector_count": int(tenant_vector_count),
        "groundtruth_files": {
            name: f"groundtruth_{name}_local_ids.npy" for name in ("unfilter", *LEVEL_NAMES)
        },
        "dist_files": {name: f"groundtruth_{name}_dists.npy" for name in ("unfilter", *LEVEL_NAMES)},
    }
    (output_dir / "query_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"num_queries     : {num_queries} (all tenant {tenant})")
    print(f"tenant0 vectors : {tenant_vector_count}")
    print(f"topk / metric   : {topk} / {args.metric}")
    print(f"output_dir      : {output_dir}")
    print("avg candidate count per query type:")
    for level, name in enumerate(LEVEL_NAMES):
        sizes = np.array([level_value_ids[level][int(query_tags[q, level])].size for q in range(num_queries)])
        print(f"  {name:7s}: mean {sizes.mean():.1f}  min {sizes.min()}  max {sizes.max()}")
    print(f"  unfilter: {tenant_vector_count}")


if __name__ == "__main__":
    main()
