#!/usr/bin/env python3
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np


DEFAULT_SCENARIO_FILE = "/home/v-mochengli/test/tenant_tag_scenario_1m.json"
DEFAULT_QUERY_FILE = "/home/v-mochengli/dataset/sift/sift_query.fvecs"
DEFAULT_OUTPUT_ROOT = Path("/home/v-mochengli/test")
LEVEL_NAMES = ("org", "dept", "team", "project")


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
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


def parse_manifest(path: Path) -> dict[str, int]:
    tenant_mapping: dict[str, int] = {}
    with path.open(encoding="utf-8") as manifest_file:
        for line in manifest_file:
            parts = line.strip().split()
            if parts and parts[0] == "tenant_mapping":
                tenant_mapping[parts[2]] = int(parts[1])
    return tenant_mapping


def compute_recall(retrieved_ids: np.ndarray, gt_ids: np.ndarray) -> float:
    hits = 0
    total = 0
    for query_idx in range(gt_ids.shape[0]):
        gt_valid = {int(value) for value in gt_ids[query_idx] if value >= 0}
        if not gt_valid:
            continue
        retrieved_valid = {int(value) for value in retrieved_ids[query_idx] if value >= 0}
        hits += len(gt_valid & retrieved_valid)
        total += len(gt_valid)
    return hits / total if total > 0 else 1.0


def quantiles_ms(latencies_ms: np.ndarray) -> tuple[float, float, float, float]:
    if latencies_ms.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(np.mean(latencies_ms)),
        float(np.percentile(latencies_ms, 50)),
        float(np.percentile(latencies_ms, 95)),
        float(np.percentile(latencies_ms, 99)),
    )


def choose_level_tags(tenant_tags: np.ndarray, requested: list[int] | None) -> list[int]:
    chosen: list[int] = []
    if requested:
        for level, tag in enumerate(requested):
            if tag < 0:
                values, counts = np.unique(tenant_tags[:, level], return_counts=True)
                chosen.append(int(values[int(np.argmax(counts))]))
            else:
                chosen.append(int(tag))
        return chosen

    for level in range(tenant_tags.shape[1]):
        values, counts = np.unique(tenant_tags[:, level], return_counts=True)
        chosen.append(int(values[int(np.argmax(counts))]))
    return chosen


def build_gt_for_tag(
    tenant_vectors_norm: np.ndarray,
    tenant_tags: np.ndarray,
    level: int,
    tag_value: int,
    query_vectors_norm: np.ndarray,
    topk: int,
) -> tuple[np.ndarray, int]:
    mask = tenant_tags[:, level] == tag_value
    candidate_local_ids = np.flatnonzero(mask).astype(np.int64)
    match_count = int(candidate_local_ids.size)

    gt_ids = np.full((query_vectors_norm.shape[0], topk), -1, dtype=np.int64)
    if match_count == 0:
        return gt_ids, match_count

    subset = np.ascontiguousarray(tenant_vectors_norm[candidate_local_ids], dtype=np.float32)
    k = min(topk, match_count)
    index = faiss.IndexFlatIP(subset.shape[1])
    index.add(subset)
    _, local_top = index.search(query_vectors_norm, k)
    gt_ids[:, :k] = candidate_local_ids[local_top]
    return gt_ids, match_count


def write_markdown(output_path: Path, payload: dict, rows: list[dict]) -> None:
    lines = [
        "# Tenant 0 Selectivity Latency Report",
        "",
        f"- Created at: {payload['created_at']}",
        f"- Scenario: {payload['scenario_file']}",
        f"- Index: {payload['index_dir']}",
        f"- Query file: {payload['query_file']}",
        f"- Tenant: 0 (internal_id={payload['tenant_internal_id']})",
        f"- Num queries: {payload['num_queries']}",
        f"- TopK: {payload['topk']}",
        f"- ForceDenseTagSearch: {payload['force_dense_tag_search']}",
        f"- SearchInternalResultNum: {payload['search_internal_result_num']}",
        f"- DirectSparseMaxPostings: {payload['direct_sparse_max_postings']}",
        f"- FilteredSearchNprobeSafety: {payload['filtered_search_nprobe_safety']}",
        f"- FilteredSearchTargetRecall: {payload['filtered_search_target_recall']}",
        f"- FilteredSearchCoverageExponent: {payload['filtered_search_coverage_exponent']}",
        f"- EnableAdaptiveFilteredNprobe: {payload['enable_adaptive_filtered_nprobe']}",
        f"- WarmupQueries: {payload['warmup_queries']}",
        "",
        "| Level | Tag | Selectivity | Match Count | Recall | QPS | Avg Latency | P95 | P99 | Avg Nprobe(PostingRead) | Avg Valid | FP Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        lines.append(
            f"| {row['level_name']} | {row['tag']} | {row['selectivity_pct']:.4f}% | {row['match_count']} | "
            f"{row['recall']:.4f} | {row['qps']:.2f} | {row['avg_latency_ms']:.2f} ms | {row['p95_latency_ms']:.2f} ms | "
            f"{row['p99_latency_ms']:.2f} ms | {row['avg_posting_read']:.2f} | {row['avg_valid']:.2f} | {row['fp_rate_pct']:.2f}% |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark tenant-0 tag selectivity and report latency/recall/QPS in markdown"
    )
    parser.add_argument("--scenario-file", default=DEFAULT_SCENARIO_FILE)
    parser.add_argument("--query-file", default=DEFAULT_QUERY_FILE)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--tags",
        default=None,
        help="comma-separated org,dept,team,project tags; use -1 for auto by highest frequency",
    )
    parser.add_argument("--force-dense-tag-search", action="store_true")
    parser.add_argument(
        "--search-internal-result-num",
        type=int,
        default=None,
        help="Set SearchInternalResultNum before running queries; use 64 to force fixed nprobe=64",
    )
    parser.add_argument("--direct-sparse-max-postings", type=int, default=320)
    parser.add_argument("--filtered-search-nprobe-safety", type=float, default=1.0)
    parser.add_argument("--filtered-search-target-recall", type=float, default=1.0)
    parser.add_argument("--filtered-search-coverage-exponent", type=float, default=0.5)
    parser.add_argument(
        "--disable-adaptive-filtered-nprobe",
        action="store_true",
        help="Disable adaptive filtered nprobe growth and keep postingTarget at the base nprobe",
    )
    parser.add_argument(
        "--warmup-queries",
        type=int,
        default=0,
        help="Run this many queries per level before timing to warm caches and IO state",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_queries <= 0 or args.topk <= 0:
        raise RuntimeError("num-queries and topk must be positive")

    scenario = json.loads(Path(args.scenario_file).read_text(encoding="utf-8"))
    index_dir = Path(scenario["index_dir"])
    output_dir = Path(args.output_dir) if args.output_dir else (DEFAULT_OUTPUT_ROOT / f"tenant0_selectivity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    queries = read_fvecs(args.query_file)
    if len(queries) < args.num_queries:
        raise RuntimeError(f"query file has {len(queries)} vectors, need {args.num_queries}")
    query_vectors = np.ascontiguousarray(queries[: args.num_queries], dtype=np.float32)
    query_vectors_norm = normalize_rows(query_vectors)

    base_vectors = read_fvecs(scenario["data_file"], scenario["vector_count"])
    tenant_ids = np.loadtxt(scenario["tenant_file"], dtype=np.int64).reshape(-1)[: scenario["vector_count"]]
    tags = np.asarray(np.load(scenario["tag_file"], allow_pickle=False), dtype=np.uint32)

    manifest_mapping = parse_manifest(index_dir / "manifest.txt")
    scenario_mapping = {str(k): int(v) for k, v in scenario.get("tenant_mapping", {}).items()}
    tenant_mapping = scenario_mapping or manifest_mapping
    if "0" not in tenant_mapping:
        raise RuntimeError("tenant 0 not found in tenant mapping")
    tenant_internal_id = int(tenant_mapping["0"])

    mask0 = tenant_ids == 0
    tenant0_vectors_norm = normalize_rows(np.ascontiguousarray(base_vectors[mask0], dtype=np.float32))
    tenant0_tags = np.ascontiguousarray(tags[mask0], dtype=np.uint32)
    tenant0_size = tenant0_vectors_norm.shape[0]

    requested_tags: list[int] | None = None
    if args.tags is not None:
        parts = [x.strip() for x in args.tags.split(",") if x.strip()]
        if len(parts) != 4:
            raise RuntimeError("--tags must have exactly 4 entries: org,dept,team,project")
        requested_tags = [int(x) for x in parts]
    level_tags = choose_level_tags(tenant0_tags, requested_tags)

    from sptag import SPTAG

    manager = SPTAG.CreateTenantIndexManager(scenario["dimension"], "SPANN", "Float")
    if not manager.LoadAll(str(index_dir)):
        raise RuntimeError(f"LoadAll failed for {index_dir}")
    if manager.BuildSignatures(tenant_internal_id, tenant0_tags.tobytes(), tenant0_size, tenant0_tags.shape[1]) is False:
        raise RuntimeError("BuildSignatures failed for tenant 0")

    if args.force_dense_tag_search:
        manager.SetSearchParam("ForceDenseTagSearch", "true", "BuildSSDIndex")
    if args.search_internal_result_num is not None:
        manager.SetSearchParam("SearchInternalResultNum", str(args.search_internal_result_num), "BuildSSDIndex")
    manager.SetSearchParam("DirectSparseMaxPostings", str(args.direct_sparse_max_postings), "BuildSSDIndex")
    manager.SetSearchParam("FilteredSearchNprobeSafety", str(args.filtered_search_nprobe_safety), "BuildSSDIndex")
    manager.SetSearchParam("FilteredSearchTargetRecall", str(args.filtered_search_target_recall), "BuildSSDIndex")
    manager.SetSearchParam("FilteredSearchCoverageExponent", str(args.filtered_search_coverage_exponent), "BuildSSDIndex")
    manager.SetSearchParam(
        "EnableAdaptiveFilteredNprobe",
        "false" if args.disable_adaptive_filtered_nprobe else "true",
        "BuildSSDIndex",
    )

    rows: list[dict] = []
    query_workload = []

    for level, tag_value in enumerate(level_tags):
        gt_ids, match_count = build_gt_for_tag(
            tenant_vectors_norm=tenant0_vectors_norm,
            tenant_tags=tenant0_tags,
            level=level,
            tag_value=tag_value,
            query_vectors_norm=query_vectors_norm,
            topk=args.topk,
        )

        qtag = np.asarray([tag_value], dtype=np.uint32)
        retrieved = np.full((args.num_queries, args.topk), -1, dtype=np.int64)
        latencies_ms = np.zeros(args.num_queries, dtype=np.float64)
        posting_reads = np.zeros(args.num_queries, dtype=np.float64)
        posting_fps = np.zeros(args.num_queries, dtype=np.float64)
        valid_counts = np.zeros(args.num_queries, dtype=np.float64)

        warmup_queries = min(max(args.warmup_queries, 0), args.num_queries)
        for query in query_vectors[:warmup_queries]:
            manager.SearchWithACL(
                query.tobytes(),
                tenant_internal_id,
                args.topk,
                qtag.tobytes(),
                1,
            )

        start_total = time.perf_counter()
        for query_idx, query in enumerate(query_vectors):
            t0 = time.perf_counter()
            result = manager.SearchWithACL(
                query.tobytes(),
                tenant_internal_id,
                args.topk,
                qtag.tobytes(),
                1,
            )
            latencies_ms[query_idx] = (time.perf_counter() - t0) * 1000.0
            posting_reads[query_idx] = float(manager.GetLastPostingReadCount())
            posting_fps[query_idx] = float(manager.GetLastPostingFP())

            if result is None:
                continue
            ids = np.asarray(result[0], dtype=np.int64)
            dists = np.asarray(result[1], dtype=np.float32)
            valid = ids[(ids >= 0) & (dists < 1e30)][: args.topk]
            retrieved[query_idx, : len(valid)] = valid
            valid_counts[query_idx] = len(valid)
            query_workload.append(
                {
                    "query_id": query_idx,
                    "level": LEVEL_NAMES[level],
                    "tag": int(tag_value),
                    "tenant_id": 0,
                    "internal_id": tenant_internal_id,
                }
            )

        elapsed_s = time.perf_counter() - start_total
        recall = compute_recall(retrieved, gt_ids)
        avg_ms, p50_ms, p95_ms, p99_ms = quantiles_ms(latencies_ms)

        total_posting_read = np.sum(posting_reads)
        total_posting_fp = np.sum(posting_fps)
        fp_rate = (100.0 * total_posting_fp / total_posting_read) if total_posting_read > 0 else 0.0

        rows.append(
            {
                "level": level,
                "level_name": LEVEL_NAMES[level],
                "tag": int(tag_value),
                "match_count": int(match_count),
                "selectivity_pct": 100.0 * match_count / tenant0_size if tenant0_size > 0 else 0.0,
                "recall": float(recall),
                "qps": float(args.num_queries / elapsed_s) if elapsed_s > 0 else 0.0,
                "avg_latency_ms": float(avg_ms),
                "p50_latency_ms": float(p50_ms),
                "p95_latency_ms": float(p95_ms),
                "p99_latency_ms": float(p99_ms),
                "avg_posting_read": float(np.mean(posting_reads)),
                "avg_valid": float(np.mean(valid_counts)),
                "fp_rate_pct": float(fp_rate),
            }
        )

    payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scenario_file": args.scenario_file,
        "index_dir": str(index_dir),
        "query_file": args.query_file,
        "tenant_internal_id": tenant_internal_id,
        "num_queries": args.num_queries,
        "topk": args.topk,
        "force_dense_tag_search": bool(args.force_dense_tag_search),
        "search_internal_result_num": (
            int(args.search_internal_result_num) if args.search_internal_result_num is not None else None
        ),
        "direct_sparse_max_postings": int(args.direct_sparse_max_postings),
        "filtered_search_nprobe_safety": float(args.filtered_search_nprobe_safety),
        "filtered_search_target_recall": float(args.filtered_search_target_recall),
        "filtered_search_coverage_exponent": float(args.filtered_search_coverage_exponent),
        "enable_adaptive_filtered_nprobe": not bool(args.disable_adaptive_filtered_nprobe),
        "warmup_queries": int(args.warmup_queries),
        "rows": rows,
    }

    (output_dir / "query_workload.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in query_workload) + "\n",
        encoding="utf-8",
    )
    (output_dir / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(output_dir / "report.md", payload, rows)

    print(f"Report written: {output_dir / 'report.md'}")
    print(f"Result json:    {output_dir / 'result.json'}")
    print(f"Workload file:  {output_dir / 'query_workload.jsonl'}")


if __name__ == "__main__":
    main()