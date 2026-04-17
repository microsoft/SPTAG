#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DEFAULT_SCENARIO_FILE = "/home/v-mochengli/test/tenant_tag_scenario_1m.json"
DEFAULT_QUERY_FILE = "/home/v-mochengli/dataset/sift/sift_query.fvecs"
DEFAULT_OUTPUT_ROOT = Path("/home/v-mochengli/test")
LEVEL_NAMES = ("org", "dept", "team", "project")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_output_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_ROOT / f"multitenant_tag_gt_{ts}"


def read_fvecs(path: str, max_vectors: int = 0) -> np.ndarray:
    with open(path, "rb") as input_file:
        dimension = int(np.frombuffer(input_file.read(4), dtype=np.int32)[0])
    stride = dimension + 1
    raw = np.fromfile(path, dtype=np.float32, count=max_vectors * stride if max_vectors else -1)
    vector_count = raw.size // stride
    return np.ascontiguousarray(raw.reshape(vector_count, stride)[:, 1:], dtype=np.float32)


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def parse_manifest(path: Path) -> tuple[dict[str, int], dict[str, int]]:
    tenant_mapping: dict[str, int] = {}
    tenant_heads: dict[str, int] = {}
    with path.open(encoding="utf-8") as manifest_file:
        for line in manifest_file:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "tenant_mapping":
                tenant_mapping[parts[2]] = int(parts[1])
            elif parts[0] == "tenant" and len(parts) >= 5:
                tenant_heads[parts[1]] = int(parts[4])
    return tenant_mapping, tenant_heads


def exact_topk_local_ids(
    tenant_vectors_norm: np.ndarray,
    query_norm: np.ndarray,
    candidate_local_ids: np.ndarray,
    topk: int,
) -> tuple[np.ndarray, np.ndarray]:
    if candidate_local_ids.size == 0 or topk <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    scores = tenant_vectors_norm[candidate_local_ids] @ query_norm
    k = min(topk, candidate_local_ids.size)
    if candidate_local_ids.size <= k:
        order = np.argsort(-scores)
    else:
        partial = np.argpartition(scores, -k)[-k:]
        order = partial[np.argsort(-scores[partial])]

    picked = order[:k]
    return (
        np.asarray(candidate_local_ids[picked], dtype=np.int64),
        np.asarray(scores[picked], dtype=np.float32),
    )


def make_tenant_infos(
    base_vectors: np.ndarray,
    tenant_ids: np.ndarray,
    tags: np.ndarray,
    tenant_counts: dict[str, int],
    tenant_mapping: dict[str, int],
    selected_tenants: list[str],
) -> dict[str, dict]:
    tenant_infos: dict[str, dict] = {}
    for tenant_id in selected_tenants:
        mask = tenant_ids == int(tenant_id)
        tenant_tags = np.ascontiguousarray(tags[mask], dtype=np.uint32)
        tenant_vectors_norm = normalize_rows(np.ascontiguousarray(base_vectors[mask], dtype=np.float32))
        if tenant_tags.size == 0:
            raise RuntimeError(f"tenant {tenant_id} has no tags")
        if tenant_vectors_norm.shape[0] != tenant_tags.shape[0]:
            raise RuntimeError(f"tenant {tenant_id} vector/tag size mismatch")

        tag_local_id_map: dict[int, np.ndarray] = {}
        for level in range(tenant_tags.shape[1]):
            values = np.unique(tenant_tags[:, level])
            for value in values.tolist():
                tag_local_id_map[int(value)] = np.flatnonzero(tenant_tags[:, level] == value).astype(np.int64)

        tenant_infos[tenant_id] = {
            "tenant_id": tenant_id,
            "internal_id": int(tenant_mapping[tenant_id]),
            "vector_count": int(tenant_counts[tenant_id]),
            "tags": tenant_tags,
            "vectors_norm": tenant_vectors_norm,
            "tag_local_id_map": tag_local_id_map,
        }
    return tenant_infos


def sample_query_tags(
    rng: np.random.Generator,
    tenant_info: dict,
    min_tags_per_query: int,
    max_tags_per_query: int,
) -> tuple[np.ndarray, np.ndarray]:
    tenant_tags = tenant_info["tags"]
    level_count = tenant_tags.shape[1]
    use_tag_count = int(rng.integers(min_tags_per_query, max_tags_per_query + 1))
    use_tag_count = max(1, min(use_tag_count, level_count))
    chosen_levels = np.asarray(rng.choice(level_count, size=use_tag_count, replace=False), dtype=np.int32)

    row = int(rng.integers(0, tenant_tags.shape[0]))
    picked_tags = np.asarray([int(tenant_tags[row, lv]) for lv in chosen_levels.tolist()], dtype=np.int64)
    return picked_tags, chosen_levels


def build_workload_items(
    query_vectors: np.ndarray,
    query_vectors_norm: np.ndarray,
    tenant_infos: dict[str, dict],
    tenant_ids: list[str],
    mode: str,
    batch_size: int,
    rng: np.random.Generator,
    min_tags_per_query: int,
    max_tags_per_query: int,
) -> list[dict]:
    items: list[dict] = []
    if mode not in {"random-mixed", "sequential-batches"}:
        raise RuntimeError(f"unsupported mode: {mode}")

    for query_index, (query, query_norm) in enumerate(zip(query_vectors, query_vectors_norm)):
        if mode == "random-mixed":
            tenant_id = str(rng.choice(tenant_ids))
        else:
            batch_id = query_index // batch_size
            tenant_id = tenant_ids[batch_id % len(tenant_ids)]

        tenant_info = tenant_infos[tenant_id]
        query_tags, levels = sample_query_tags(
            rng,
            tenant_info,
            min_tags_per_query=min_tags_per_query,
            max_tags_per_query=max_tags_per_query,
        )
        items.append(
            {
                "query_index": query_index,
                "query": query,
                "query_norm": query_norm,
                "tenant_id": tenant_id,
                "internal_id": tenant_info["internal_id"],
                "query_tags": query_tags,
                "query_levels": levels,
            }
        )
    return items


def build_candidate_local_ids(
    tenant_tags: np.ndarray,
    query_tags: np.ndarray,
    query_levels: np.ndarray,
    tag_local_id_map: dict[int, np.ndarray],
    filter_mode: str,
) -> np.ndarray:
    if query_tags.size == 0:
        return np.empty(0, dtype=np.int64)

    if filter_mode == "any":
        candidate_chunks: list[np.ndarray] = []
        for tag in query_tags.tolist():
            ids = tag_local_id_map.get(int(tag))
            if ids is not None and ids.size > 0:
                candidate_chunks.append(ids)
        if not candidate_chunks:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(candidate_chunks).astype(np.int64, copy=False))

    # Default exact mode: all (level, tag) constraints must be satisfied.
    mask = np.ones(tenant_tags.shape[0], dtype=bool)
    for tag, level in zip(query_tags.tolist(), query_levels.tolist()):
        mask &= tenant_tags[:, int(level)] == np.uint32(tag)
    return np.flatnonzero(mask).astype(np.int64)


def write_outputs(
    output_dir: Path,
    payload: dict,
    workload_rows: list[dict],
    query_vectors: np.ndarray,
    query_source_indices: np.ndarray,
    query_tenant_ids: np.ndarray,
    query_internal_ids: np.ndarray,
    query_tag_counts: np.ndarray,
    query_tags_matrix: np.ndarray,
    query_levels_matrix: np.ndarray,
    query_match_counts: np.ndarray,
    gt_local_ids: np.ndarray,
    gt_scores: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "meta.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    np.save(output_dir / "query_vectors.npy", query_vectors)
    np.save(output_dir / "query_source_indices.npy", query_source_indices)
    np.save(output_dir / "query_tenant_ids.npy", query_tenant_ids)
    np.save(output_dir / "query_internal_ids.npy", query_internal_ids)
    np.save(output_dir / "query_tag_counts.npy", query_tag_counts)
    np.save(output_dir / "query_tags.npy", query_tags_matrix)
    np.save(output_dir / "query_levels.npy", query_levels_matrix)
    np.save(output_dir / "query_match_counts.npy", query_match_counts)
    np.save(output_dir / "groundtruth_local_ids.npy", gt_local_ids)
    np.save(output_dir / "groundtruth_scores.npy", gt_scores)

    with (output_dir / "workload.jsonl").open("w", encoding="utf-8") as fp:
        for row in workload_rows:
            fp.write(json.dumps(row, ensure_ascii=True) + "\n")

    lines = [
        "# Multi-Tenant Multi-Tag Groundtruth",
        "",
        f"- Created at (UTC): {payload['created_at_utc']}",
        f"- Scenario file: {payload['scenario_file']}",
        f"- Index dir: {payload['index_dir']}",
        f"- Query file: {payload['query_file']}",
        f"- Num queries: {payload['num_queries']}",
        f"- TopK: {payload['topk']}",
        f"- Mode: {payload['mode']}",
        f"- Filter mode: {payload['filter_mode']}",
        f"- Seed: {payload['seed']}",
        f"- Tenants: {', '.join(payload['tenant_ids'])}",
        f"- Tags per query: [{payload['min_tags_per_query']}, {payload['max_tags_per_query']}]",
        "",
        "## Files",
        "",
        "- workload.jsonl",
        "- query_vectors.npy",
        "- query_source_indices.npy",
        "- query_tenant_ids.npy",
        "- query_internal_ids.npy",
        "- query_tag_counts.npy",
        "- query_tags.npy",
        "- query_levels.npy",
        "- query_match_counts.npy",
        "- groundtruth_local_ids.npy",
        "- groundtruth_scores.npy",
        "- meta.json",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate persisted groundtruth for multi-tenant multi-tag SIFT1M workload"
    )
    parser.add_argument("--scenario-file", default=DEFAULT_SCENARIO_FILE)
    parser.add_argument("--query-file", default=DEFAULT_QUERY_FILE)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-queries", type=int, default=1000)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--tenant-range", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--mode", choices=["random-mixed", "sequential-batches"], default="random-mixed")
    parser.add_argument("--seed", type=int, default=20260414)
    parser.add_argument("--min-tags-per-query", type=int, default=1)
    parser.add_argument("--max-tags-per-query", type=int, default=2)
    parser.add_argument(
        "--filter-mode",
        choices=["exact", "any"],
        default="exact",
        help="Tag filtering mode: exact=intersection, any=union",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_queries <= 0:
        raise RuntimeError("num-queries must be positive")
    if args.topk <= 0:
        raise RuntimeError("topk must be positive")
    if args.batch_size <= 0:
        raise RuntimeError("batch-size must be positive")
    if args.min_tags_per_query <= 0 or args.max_tags_per_query <= 0:
        raise RuntimeError("tags-per-query bounds must be positive")
    if args.min_tags_per_query > args.max_tags_per_query:
        raise RuntimeError("min-tags-per-query cannot be greater than max-tags-per-query")

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    scenario = json.loads(Path(args.scenario_file).read_text(encoding="utf-8"))

    index_dir = Path(scenario["index_dir"])
    tenant_mapping_manifest, _ = parse_manifest(index_dir / "manifest.txt")
    tenant_mapping = {
        str(key): int(value) for key, value in scenario.get("tenant_mapping", {}).items()
    } or tenant_mapping_manifest

    selected_tenants = [item.strip() for item in args.tenant_range.split(",") if item.strip()]
    if not selected_tenants:
        raise RuntimeError("tenant-range must not be empty")
    unknown_tenants = [tenant_id for tenant_id in selected_tenants if tenant_id not in tenant_mapping]
    if unknown_tenants:
        raise RuntimeError(f"tenant-range contains unknown tenant ids: {unknown_tenants}")

    query_vectors_all = read_fvecs(args.query_file)
    if len(query_vectors_all) < args.num_queries:
        raise RuntimeError(f"query file has {len(query_vectors_all)} vectors, need {args.num_queries}")

    base_vectors = read_fvecs(scenario["data_file"], scenario["vector_count"])
    if len(base_vectors) < scenario["vector_count"]:
        raise RuntimeError(f"data file has {len(base_vectors)} vectors, need {scenario['vector_count']}")

    tenant_ids = np.loadtxt(scenario["tenant_file"], dtype=np.int64).reshape(-1)[: scenario["vector_count"]]
    tags = np.asarray(np.load(scenario["tag_file"], allow_pickle=False), dtype=np.uint32)

    tenant_infos = make_tenant_infos(
        base_vectors=base_vectors,
        tenant_ids=tenant_ids,
        tags=tags,
        tenant_counts=scenario["tenant_counts"],
        tenant_mapping=tenant_mapping,
        selected_tenants=selected_tenants,
    )

    rng = np.random.default_rng(args.seed)
    if args.mode == "random-mixed":
        query_source_indices = np.asarray(rng.choice(len(query_vectors_all), size=args.num_queries, replace=False), dtype=np.int64)
    else:
        query_source_indices = np.arange(args.num_queries, dtype=np.int64)

    query_vectors = np.ascontiguousarray(query_vectors_all[query_source_indices], dtype=np.float32)
    query_vectors_norm = normalize_rows(query_vectors)

    items = build_workload_items(
        query_vectors=query_vectors,
        query_vectors_norm=query_vectors_norm,
        tenant_infos=tenant_infos,
        tenant_ids=selected_tenants,
        mode=args.mode,
        batch_size=args.batch_size,
        rng=rng,
        min_tags_per_query=args.min_tags_per_query,
        max_tags_per_query=args.max_tags_per_query,
    )

    max_tags_per_query = args.max_tags_per_query
    gt_local_ids = np.full((args.num_queries, args.topk), -1, dtype=np.int64)
    gt_scores = np.full((args.num_queries, args.topk), -1e30, dtype=np.float32)
    query_tenant_ids = np.empty(args.num_queries, dtype=np.int64)
    query_internal_ids = np.empty(args.num_queries, dtype=np.int64)
    query_tag_counts = np.empty(args.num_queries, dtype=np.int32)
    query_tags_matrix = np.full((args.num_queries, max_tags_per_query), -1, dtype=np.int64)
    query_levels_matrix = np.full((args.num_queries, max_tags_per_query), -1, dtype=np.int32)
    query_match_counts = np.zeros(args.num_queries, dtype=np.int64)
    workload_rows: list[dict] = []

    for item in items:
        query_index = int(item["query_index"])
        tenant_id = str(item["tenant_id"])
        tenant_info = tenant_infos[tenant_id]
        query_tags = np.asarray(item["query_tags"], dtype=np.int64)
        query_levels = np.asarray(item["query_levels"], dtype=np.int32)

        candidate_local_ids = build_candidate_local_ids(
            tenant_tags=np.asarray(tenant_info["tags"], dtype=np.uint32),
            query_tags=query_tags,
            query_levels=query_levels,
            tag_local_id_map=tenant_info["tag_local_id_map"],
            filter_mode=args.filter_mode,
        )
        picked_ids, picked_scores = exact_topk_local_ids(
            tenant_vectors_norm=tenant_info["vectors_norm"],
            query_norm=np.asarray(item["query_norm"], dtype=np.float32),
            candidate_local_ids=candidate_local_ids,
            topk=args.topk,
        )

        gt_local_ids[query_index, : len(picked_ids)] = picked_ids
        gt_scores[query_index, : len(picked_scores)] = picked_scores

        query_tenant_ids[query_index] = int(tenant_id)
        query_internal_ids[query_index] = int(item["internal_id"])
        query_tag_counts[query_index] = len(query_tags)
        query_tags_matrix[query_index, : len(query_tags)] = query_tags
        query_levels_matrix[query_index, : len(query_levels)] = query_levels
        query_match_counts[query_index] = candidate_local_ids.size

        workload_rows.append(
            {
                "query_index": query_index,
                "query_source_index": int(query_source_indices[query_index]),
                "tenant_id": int(tenant_id),
                "internal_id": int(item["internal_id"]),
                "query_tags": [int(x) for x in query_tags.tolist()],
                "query_levels": [LEVEL_NAMES[int(x)] for x in query_levels.tolist()],
                "filter_mode": args.filter_mode,
                "match_count": int(candidate_local_ids.size),
            }
        )

    payload = {
        "created_at_utc": utc_timestamp(),
        "scenario_file": args.scenario_file,
        "index_dir": str(index_dir),
        "query_file": args.query_file,
        "num_queries": args.num_queries,
        "topk": args.topk,
        "seed": args.seed,
        "mode": args.mode,
        "batch_size": args.batch_size,
        "filter_mode": args.filter_mode,
        "tenant_ids": selected_tenants,
        "min_tags_per_query": args.min_tags_per_query,
        "max_tags_per_query": args.max_tags_per_query,
        "files": {
            "workload": "workload.jsonl",
            "query_vectors": "query_vectors.npy",
            "query_source_indices": "query_source_indices.npy",
            "query_tenant_ids": "query_tenant_ids.npy",
            "query_internal_ids": "query_internal_ids.npy",
            "query_tag_counts": "query_tag_counts.npy",
            "query_tags": "query_tags.npy",
            "query_levels": "query_levels.npy",
            "query_match_counts": "query_match_counts.npy",
            "groundtruth_local_ids": "groundtruth_local_ids.npy",
            "groundtruth_scores": "groundtruth_scores.npy",
        },
    }

    write_outputs(
        output_dir=output_dir,
        payload=payload,
        workload_rows=workload_rows,
        query_vectors=query_vectors,
        query_source_indices=query_source_indices,
        query_tenant_ids=query_tenant_ids,
        query_internal_ids=query_internal_ids,
        query_tag_counts=query_tag_counts,
        query_tags_matrix=query_tags_matrix,
        query_levels_matrix=query_levels_matrix,
        query_match_counts=query_match_counts,
        gt_local_ids=gt_local_ids,
        gt_scores=gt_scores,
    )

    print(f"Saved groundtruth workload to {output_dir}")


if __name__ == "__main__":
    main()
