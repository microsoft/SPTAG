#!/usr/bin/env python3
import argparse
import csv
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIO_FILE = "/home/v-mochengli/test/tenant_tag_scenario_1m.json"
DEFAULT_QUERY_FILE = "/home/v-mochengli/dataset/sift/sift_query.fvecs"
DEFAULT_OUTPUT_ROOT = Path("/tmp")
MB = 1024 * 1024
LEVEL_NAMES = ("org", "dept", "team", "project")
DEFAULT_SEARCH_PARAMS = {
    "force_dense_tag_search": False,
    "direct_sparse_max_postings": 320,
    "filtered_search_nprobe_safety": 1.0,
    "filtered_search_target_recall": 1.0,
    "filtered_search_coverage_exponent": 0.5,
}


def import_sptag_module():
    def supports_required_api(module) -> bool:
        manager_cls = getattr(module, "TenantIndexManager", None)
        native_module = getattr(module, "_SPTAG", None)
        return (
            manager_cls is not None
            and native_module is not None
            and hasattr(manager_cls, "GetTenantHeadIndexSize")
            and hasattr(native_module, "TenantIndexManager_GetTenantHeadIndexSize")
            and hasattr(native_module, "TenantIndexManager_GetTagRoutingStatsBlob")
        )

    try:
        from sptag import SPTAG as sptag_module
        if supports_required_api(sptag_module):
            return sptag_module
    except ImportError:
        pass

    release_dir = REPO_ROOT / "Release"
    if release_dir.is_dir() and str(release_dir) not in sys.path:
        sys.path.insert(0, str(release_dir))

    import SPTAG as sptag_module
    if not supports_required_api(sptag_module):
        raise ImportError("SPTAG Python wrapper does not expose GetTenantHeadIndexSize")
    return sptag_module


SPTAG = import_sptag_module()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_git_command(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def detect_git_state() -> tuple[str | None, bool | None]:
    commit = run_git_command("rev-parse", "HEAD")
    status = run_git_command("status", "--porcelain")
    if status is None:
        return commit, None
    return commit, bool(status)


def default_output_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_ROOT / f"multitenant_tag_cache_stress_{ts}"


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
    tenant_mapping = {}
    tenant_heads = {}
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


def choose_cache_limit_bytes(head_sizes: list[int]) -> int:
    if not head_sizes:
        return 64 * MB
    total = sum(head_sizes)
    largest = max(head_sizes)
    candidate = max(largest * 2, total // 4)
    candidate_mb = max(1, (candidate + MB - 1) // MB)
    return candidate_mb * MB


def compute_quantiles(latencies_ms: list[float]) -> tuple[float, float, float, float]:
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return (
        float(np.mean(arr)) if arr.size else 0.0,
        float(np.percentile(arr, 50)) if arr.size else 0.0,
        float(np.percentile(arr, 95)) if arr.size else 0.0,
        float(np.percentile(arr, 99)) if arr.size else 0.0,
    )


def exact_topk_local_ids(
    tenant_vectors_norm: np.ndarray,
    query_norm: np.ndarray,
    candidate_local_ids: np.ndarray,
    topk: int,
) -> np.ndarray:
    if candidate_local_ids.size == 0 or topk <= 0:
        return np.empty(0, dtype=np.int64)

    scores = tenant_vectors_norm[candidate_local_ids] @ query_norm
    k = min(topk, candidate_local_ids.size)
    if candidate_local_ids.size <= k:
        order = np.argsort(-scores)
    else:
        partial = np.argpartition(scores, -k)[-k:]
        order = partial[np.argsort(-scores[partial])]
    return np.asarray(candidate_local_ids[order[:k]], dtype=np.int64)


def compute_recall_value(retrieved_ids: np.ndarray, gt_ids: np.ndarray) -> float:
    gt_valid = {int(value) for value in gt_ids if value >= 0}
    if not gt_valid:
        return 1.0
    retrieved_valid = {int(value) for value in retrieved_ids if value >= 0}
    return len(gt_valid & retrieved_valid) / len(gt_valid)


def make_tenant_infos(
    base_vectors: np.ndarray,
    tenant_ids: np.ndarray,
    tags: np.ndarray,
    tenant_counts: dict[str, int],
    tenant_mapping: dict[str, int],
    tenant_heads: dict[str, int],
    selected_tenants: list[str],
) -> dict[str, dict]:
    tenant_infos = {}
    for tenant_id in selected_tenants:
        mask = tenant_ids == int(tenant_id)
        tenant_tags = np.ascontiguousarray(tags[mask], dtype=np.uint32)
        tenant_vectors_norm = normalize_rows(np.ascontiguousarray(base_vectors[mask], dtype=np.float32))
        if tenant_tags.size == 0:
            raise RuntimeError(f"tenant {tenant_id} has no tags")
        if tenant_vectors_norm.shape[0] != tenant_tags.shape[0]:
            raise RuntimeError(f"tenant {tenant_id} vector/tag size mismatch")

        level_count_maps = []
        level_local_id_maps = []
        for level in range(tenant_tags.shape[1]):
            values, counts = np.unique(tenant_tags[:, level], return_counts=True)
            level_count_maps.append({int(value): int(count) for value, count in zip(values.tolist(), counts.tolist())})
            level_local_id_maps.append({
                int(value): np.flatnonzero(tenant_tags[:, level] == value).astype(np.int64)
                for value in values.tolist()
            })

        tenant_infos[tenant_id] = {
            "tenant_id": tenant_id,
            "internal_id": int(tenant_mapping[tenant_id]),
            "vector_count": int(tenant_counts[tenant_id]),
            "head_count": int(tenant_heads.get(tenant_id, 0)),
            "head_bytes": 0,
            "tags": tenant_tags,
            "vectors_norm": tenant_vectors_norm,
            "level_count_maps": level_count_maps,
            "level_local_id_maps": level_local_id_maps,
        }
    return tenant_infos


def sample_random_tag(rng: np.random.Generator, tenant_info: dict) -> tuple[np.ndarray, int, int]:
    tenant_tags = tenant_info["tags"]
    level = int(rng.integers(0, tenant_tags.shape[1]))
    row = int(rng.integers(0, tenant_tags.shape[0]))
    tag = int(tenant_tags[row, level])
    match_count = tenant_info["level_count_maps"][level][tag]
    return np.asarray([tag], dtype=np.uint32), level, match_count


def build_sequential_batches(
    queries: np.ndarray,
    queries_norm: np.ndarray,
    tenant_infos: dict[str, dict],
    tenant_ids: list[str],
    batch_size: int,
    rng: np.random.Generator,
) -> list[dict]:
    batches = []
    for batch_idx, tenant_id in enumerate(tenant_ids):
        start = batch_idx * batch_size
        end = start + batch_size
        batch_queries = queries[start:end]
        batch_queries_norm = queries_norm[start:end]
        items = []
        for query, query_norm in zip(batch_queries, batch_queries_norm):
            query_tags, level, match_count = sample_random_tag(rng, tenant_infos[tenant_id])
            items.append({
                "query": query,
                "query_norm": query_norm,
                "tenant_id": tenant_id,
                "internal_id": tenant_infos[tenant_id]["internal_id"],
                "query_tags": query_tags,
                "level": level,
                "match_count": match_count,
            })
        batches.append({
            "scenario": "sequential",
            "batch_index": batch_idx,
            "label": f"tenant_{tenant_id}",
            "items": items,
        })
    return batches


def build_random_batches(
    queries: np.ndarray,
    queries_norm: np.ndarray,
    tenant_infos: dict[str, dict],
    tenant_ids: list[str],
    batch_size: int,
    rng: np.random.Generator,
) -> list[dict]:
    batches = []
    for batch_idx in range(len(tenant_ids)):
        start = batch_idx * batch_size
        end = start + batch_size
        batch_queries = queries[start:end]
        batch_queries_norm = queries_norm[start:end]
        items = []
        chosen_tenants = rng.choice(tenant_ids, size=batch_size, replace=True)
        for query, query_norm, tenant_id in zip(batch_queries, batch_queries_norm, chosen_tenants.tolist()):
            query_tags, level, match_count = sample_random_tag(rng, tenant_infos[tenant_id])
            items.append({
                "query": query,
                "query_norm": query_norm,
                "tenant_id": tenant_id,
                "internal_id": tenant_infos[tenant_id]["internal_id"],
                "query_tags": query_tags,
                "level": level,
                "match_count": match_count,
            })
        batches.append({
            "scenario": "random",
            "batch_index": batch_idx,
            "label": f"batch_{batch_idx}",
            "items": items,
        })
    return batches


def build_signatures(manager, tenant_infos: dict[str, dict], ordered_tenants: list[str]) -> None:
    for tenant_id in ordered_tenants:
        tenant_info = tenant_infos[tenant_id]
        ok = manager.BuildSignatures(
            tenant_info["internal_id"],
            tenant_info["tags"].tobytes(),
            tenant_info["vector_count"],
            tenant_info["tags"].shape[1],
        )
        if ok is False:
            raise RuntimeError(f"BuildSignatures failed for tenant {tenant_id}")


def unload_all_tenants(manager, tenant_infos: dict[str, dict], ordered_tenants: list[str]) -> None:
    for tenant_id in ordered_tenants:
        manager.UnloadTenant(tenant_infos[tenant_id]["internal_id"])


def run_batches(manager, batches: list[dict], tenant_infos: dict[str, dict], topk: int) -> tuple[list[dict], dict]:
    batch_rows = []
    all_latencies = []
    total_queries = 0
    total_valid = 0
    total_expected_valid = 0
    total_shortfall_queries = 0
    total_posting_read = 0
    total_posting_match = 0
    total_posting_fp = 0
    total_recall = 0.0
    total_selectivity_pct = 0.0

    for batch in batches:
        latencies = []
        valid_counts = []
        expected_valid_counts = []
        posting_reads = []
        posting_matches = []
        posting_fps = []
        recall_values = []
        selectivity_values = []
        level_hist = {level_name: 0 for level_name in LEVEL_NAMES}
        tenant_sequence = [item["tenant_id"] for item in batch["items"]]
        unique_tenants = sorted(set(tenant_sequence), key=lambda value: int(value))
        tenant_switches = sum(
            1 for index in range(1, len(tenant_sequence)) if tenant_sequence[index] != tenant_sequence[index - 1]
        )

        for item in batch["items"]:
            tenant_info = tenant_infos[item["tenant_id"]]
            level_hist[LEVEL_NAMES[item["level"]]] += 1
            expected_valid = min(topk, item["match_count"])
            selectivity_values.append(
                100.0 * item["match_count"] / tenant_info["vector_count"] if tenant_info["vector_count"] > 0 else 0.0
            )

            start_time = time.perf_counter()
            result = manager.SearchWithACL(
                item["query"].tobytes(),
                item["internal_id"],
                topk,
                item["query_tags"].tobytes(),
                len(item["query_tags"]),
            )
            latencies.append((time.perf_counter() - start_time) * 1000.0)

            posting_read = int(manager.GetLastPostingReadCount())
            posting_match = int(manager.GetLastPostingMatchCount())
            posting_fp = int(manager.GetLastPostingFP())
            posting_reads.append(posting_read)
            posting_matches.append(posting_match)
            posting_fps.append(posting_fp)

            valid_count = 0
            valid_ids = np.empty(0, dtype=np.int64)
            if result is not None:
                ids = np.asarray(result[0], dtype=np.int64)
                dists = np.asarray(result[1], dtype=np.float32)
                valid_mask = (ids >= 0) & (dists < 1e30)
                valid_ids = np.asarray(ids[valid_mask][:topk], dtype=np.int64)
                valid_count = int(len(valid_ids))

            tag_value = int(item["query_tags"][0])
            candidate_local_ids = tenant_info["level_local_id_maps"][item["level"]][tag_value]
            gt_ids = exact_topk_local_ids(tenant_info["vectors_norm"], item["query_norm"], candidate_local_ids, topk)
            recall_values.append(compute_recall_value(valid_ids, gt_ids))

            valid_counts.append(valid_count)
            expected_valid_counts.append(expected_valid)

        avg_ms, p50_ms, p95_ms, p99_ms = compute_quantiles(latencies)
        total_time_s = float(np.sum(latencies) / 1000.0)
        qps = len(batch["items"]) / total_time_s if total_time_s > 0 else 0.0
        valid_arr = np.asarray(valid_counts, dtype=np.float64)
        expected_valid_arr = np.asarray(expected_valid_counts, dtype=np.float64)
        posting_read_sum = int(np.sum(posting_reads))
        posting_match_sum = int(np.sum(posting_matches))
        posting_fp_sum = int(np.sum(posting_fps))
        fp_rate = (100.0 * posting_fp_sum / posting_read_sum) if posting_read_sum > 0 else 0.0
        cache_usage_mb = manager.GetHeadIndexCacheUsage() / MB
        shortfall_queries = int(np.sum(valid_arr + 1e-9 < expected_valid_arr))

        batch_row = {
            "scenario": batch["scenario"],
            "batch_index": batch["batch_index"],
            "label": batch["label"],
            "unique_tenants": len(unique_tenants),
            "tenant_ids": ",".join(unique_tenants),
            "tenant_switches": tenant_switches,
            "avg_latency_ms": avg_ms,
            "p50_latency_ms": p50_ms,
            "p95_latency_ms": p95_ms,
            "p99_latency_ms": p99_ms,
            "qps": qps,
            "avg_recall": float(np.mean(recall_values)) if recall_values else 0.0,
            "avg_selectivity_pct": float(np.mean(selectivity_values)) if selectivity_values else 0.0,
            "avg_valid": float(np.mean(valid_arr)) if valid_arr.size else 0.0,
            "avg_expected_valid": float(np.mean(expected_valid_arr)) if expected_valid_arr.size else 0.0,
            "shortfall_queries": shortfall_queries,
            "avg_posting_read": float(np.mean(posting_reads)) if posting_reads else 0.0,
            "avg_posting_match": float(np.mean(posting_matches)) if posting_matches else 0.0,
            "fp_rate_pct": fp_rate,
            "cache_usage_mb": cache_usage_mb,
            "level_hist": level_hist,
        }
        batch_rows.append(batch_row)

        all_latencies.extend(latencies)
        total_queries += len(batch["items"])
        total_valid += int(np.sum(valid_arr))
        total_expected_valid += int(np.sum(expected_valid_arr))
        total_shortfall_queries += shortfall_queries
        total_posting_read += posting_read_sum
        total_posting_match += posting_match_sum
        total_posting_fp += posting_fp_sum
        total_recall += float(np.sum(recall_values))
        total_selectivity_pct += float(np.sum(selectivity_values))

    avg_ms, p50_ms, p95_ms, p99_ms = compute_quantiles(all_latencies)
    total_time_s = float(np.sum(all_latencies) / 1000.0)
    overall = {
        "scenario": batches[0]["scenario"] if batches else "unknown",
        "batch_count": len(batches),
        "query_count": total_queries,
        "avg_latency_ms": avg_ms,
        "p50_latency_ms": p50_ms,
        "p95_latency_ms": p95_ms,
        "p99_latency_ms": p99_ms,
        "qps": (total_queries / total_time_s) if total_time_s > 0 else 0.0,
        "avg_recall": (total_recall / total_queries) if total_queries > 0 else 0.0,
        "avg_selectivity_pct": (total_selectivity_pct / total_queries) if total_queries > 0 else 0.0,
        "avg_valid": (total_valid / total_queries) if total_queries > 0 else 0.0,
        "avg_expected_valid": (total_expected_valid / total_queries) if total_queries > 0 else 0.0,
        "shortfall_queries": total_shortfall_queries,
        "avg_posting_read": (total_posting_read / total_queries) if total_queries > 0 else 0.0,
        "avg_posting_match": (total_posting_match / total_queries) if total_queries > 0 else 0.0,
        "fp_rate_pct": (100.0 * total_posting_fp / total_posting_read) if total_posting_read > 0 else 0.0,
        "cache_usage_end_mb": manager.GetHeadIndexCacheUsage() / MB,
    }
    return batch_rows, overall


def apply_search_params(manager, search_params: dict[str, object]) -> None:
    if bool(search_params["force_dense_tag_search"]):
        manager.SetSearchParam("ForceDenseTagSearch", "true", "BuildSSDIndex")
    manager.SetSearchParam("DirectSparseMaxPostings", str(search_params["direct_sparse_max_postings"]), "BuildSSDIndex")
    manager.SetSearchParam("FilteredSearchNprobeSafety", str(search_params["filtered_search_nprobe_safety"]), "BuildSSDIndex")
    manager.SetSearchParam("FilteredSearchTargetRecall", str(search_params["filtered_search_target_recall"]), "BuildSSDIndex")
    manager.SetSearchParam(
        "FilteredSearchCoverageExponent",
        str(search_params["filtered_search_coverage_exponent"]),
        "BuildSSDIndex",
    )


def meta_to_text_lines(meta: dict) -> list[str]:
    lines = []
    for key in sorted(meta):
        value = meta[key]
        if isinstance(value, (dict, list)):
            value = json.dumps(value, sort_keys=True)
        lines.append(f"{key}={value}")
    return lines


def write_outputs(output_dir: Path, payload: dict, meta: dict) -> None:
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "batch_summary.csv"
    md_path = output_dir / "summary.md"
    meta_json_path = output_dir / "meta.json"
    meta_txt_path = output_dir / "meta.txt"

    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    meta_json_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    meta_txt_path.write_text("\n".join(meta_to_text_lines(meta)) + "\n", encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "scenario", "batch_index", "label", "unique_tenants", "tenant_ids", "tenant_switches",
            "avg_latency_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms", "qps",
            "avg_recall", "avg_selectivity_pct",
            "avg_valid", "avg_expected_valid", "shortfall_queries",
            "avg_posting_read", "avg_posting_match", "fp_rate_pct", "cache_usage_mb",
            "org_count", "dept_count", "team_count", "project_count",
        ])
        for row in payload["batch_rows"]:
            writer.writerow([
                row["scenario"], row["batch_index"], row["label"], row["unique_tenants"], row["tenant_ids"], row["tenant_switches"],
                row["avg_latency_ms"], row["p50_latency_ms"], row["p95_latency_ms"], row["p99_latency_ms"], row["qps"],
                row["avg_recall"], row["avg_selectivity_pct"],
                row["avg_valid"], row["avg_expected_valid"], row["shortfall_queries"],
                row["avg_posting_read"], row["avg_posting_match"], row["fp_rate_pct"], row["cache_usage_mb"],
                row["level_hist"]["org"], row["level_hist"]["dept"], row["level_hist"]["team"], row["level_hist"]["project"],
            ])

    lines = [
        "# Multi-Tenant Tag Cache Stress",
        "",
        f"- Created at (UTC): `{meta['created_at_utc']}`",
        f"- Script: `{meta['script_path']}`",
        f"- Repository root: `{meta['repo_root']}`",
        f"- Git commit: `{meta['git_commit']}`",
        f"- Git dirty: `{meta['git_dirty']}`",
        f"- Scenario file: `{payload['scenario_file']}`",
        f"- Index: `{payload['index_dir']}`",
        f"- Query file: `{payload['query_file']}`",
        f"- Tenants: {', '.join(payload['tenant_ids'])}",
        f"- Queries per workload: {payload['num_queries']}",
        f"- Batch size: {payload['batch_size']}",
        f"- TopK: {payload['topk']}",
        f"- Seed: {payload['seed']} (random workload seed `{payload['random_seed']}`)",
        f"- Cache limit: {payload['cache_limit_mb']:.1f} MB ({payload['cache_limit_source']})",
        f"- Cache policy: `{payload['cache_limit_policy']}`",
        f"- Drop page cache on evict: {payload['drop_page_cache_on_evict']}",
        "",
        "## Search Params",
        "",
        f"- ForceDenseTagSearch: {payload['search_params']['force_dense_tag_search']}",
        f"- DirectSparseMaxPostings: {payload['search_params']['direct_sparse_max_postings']}",
        f"- FilteredSearchNprobeSafety: {payload['search_params']['filtered_search_nprobe_safety']}",
        f"- FilteredSearchTargetRecall: {payload['search_params']['filtered_search_target_recall']}",
        f"- FilteredSearchCoverageExponent: {payload['search_params']['filtered_search_coverage_exponent']}",
        "",
        "## HeadIndex Sizes",
        "",
        "| Tenant | Internal ID | Vectors | Heads | HeadIndex MB |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["tenant_rows"]:
        lines.append(
            f"| {row['tenant_id']} | {row['internal_id']} | {row['vector_count']} | {row['head_count']} | {row['head_index_mb']:.2f} |"
        )

    lines.extend([
        "",
        "## Overall",
        "",
        "| Scenario | Avg Latency | P95 | P99 | QPS | Avg Recall | Avg Selectivity | Avg Valid | Avg Expected Valid | Shortfall Queries | Avg Posting Read | Avg Posting Match | FP Rate | Cache End |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in payload["overall_rows"]:
        lines.append(
            f"| {row['scenario']} | {row['avg_latency_ms']:.2f} ms | {row['p95_latency_ms']:.2f} ms | {row['p99_latency_ms']:.2f} ms | "
            f"{row['qps']:.2f} | {row['avg_recall']:.4f} | {row['avg_selectivity_pct']:.2f}% | {row['avg_valid']:.2f} | {row['avg_expected_valid']:.2f} | {row['shortfall_queries']} | "
            f"{row['avg_posting_read']:.2f} | {row['avg_posting_match']:.2f} | {row['fp_rate_pct']:.2f}% | {row['cache_usage_end_mb']:.2f} MB |"
        )

    for scenario_name in ("sequential", "random"):
        lines.extend([
            "",
            f"## {scenario_name.title()} Batches",
            "",
            "| Batch | Label | Unique Tenants | Switches | Avg Latency | P95 | P99 | QPS | Avg Recall | Avg Selectivity | Avg Valid | Avg Expected Valid | Shortfall Queries | Avg Posting Read | FP Rate | Cache Usage | Levels (org/dept/team/project) |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ])
        for row in payload["batch_rows"]:
            if row["scenario"] != scenario_name:
                continue
            level_hist = row["level_hist"]
            lines.append(
                f"| {row['batch_index']} | {row['label']} | {row['unique_tenants']} | {row['tenant_switches']} | {row['avg_latency_ms']:.2f} ms | "
                f"{row['p95_latency_ms']:.2f} ms | {row['p99_latency_ms']:.2f} ms | {row['qps']:.2f} | {row['avg_recall']:.4f} | {row['avg_selectivity_pct']:.2f}% | {row['avg_valid']:.2f} | "
                f"{row['avg_expected_valid']:.2f} | {row['shortfall_queries']} | {row['avg_posting_read']:.2f} | {row['fp_rate_pct']:.2f}% | "
                f"{row['cache_usage_mb']:.2f} MB | {level_hist['org']}/{level_hist['dept']}/{level_hist['team']}/{level_hist['project']} |"
            )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-tenant + tag-filtering cache stress benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scenario-file", default=DEFAULT_SCENARIO_FILE)
    parser.add_argument("--query-file", default=DEFAULT_QUERY_FILE)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-queries", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--tenant-range", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--seed", type=int, default=20260413)
    parser.add_argument("--cache-limit-mb", type=float, default=None)
    parser.add_argument("--drop-page-cache-on-evict", action="store_true")
    parser.add_argument("--force-dense-tag-search", action="store_true")
    parser.add_argument("--direct-sparse-max-postings", type=int, default=DEFAULT_SEARCH_PARAMS["direct_sparse_max_postings"])
    parser.add_argument("--filtered-search-nprobe-safety", type=float, default=DEFAULT_SEARCH_PARAMS["filtered_search_nprobe_safety"])
    parser.add_argument("--filtered-search-target-recall", type=float, default=DEFAULT_SEARCH_PARAMS["filtered_search_target_recall"])
    parser.add_argument(
        "--filtered-search-coverage-exponent",
        type=float,
        default=DEFAULT_SEARCH_PARAMS["filtered_search_coverage_exponent"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_queries <= 0 or args.batch_size <= 0:
        raise RuntimeError("num-queries and batch-size must be positive")
    if args.num_queries % args.batch_size != 0:
        raise RuntimeError("num-queries must be divisible by batch-size")

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario = json.loads(Path(args.scenario_file).read_text(encoding="utf-8"))
    index_dir = Path(scenario["index_dir"])
    query_vectors = read_fvecs(args.query_file, args.num_queries)
    query_vectors_norm = normalize_rows(query_vectors)
    if len(query_vectors) < args.num_queries:
        raise RuntimeError(f"query file only has {len(query_vectors)} vectors, need {args.num_queries}")

    base_vectors = read_fvecs(scenario["data_file"], scenario["vector_count"])
    if len(base_vectors) < scenario["vector_count"]:
        raise RuntimeError(f"data file only has {len(base_vectors)} vectors, need {scenario['vector_count']}")
    tenant_ids_data = np.loadtxt(scenario["tenant_file"], dtype=np.int64).reshape(-1)[:scenario["vector_count"]]
    tags = np.asarray(np.load(scenario["tag_file"], allow_pickle=False), dtype=np.uint32)

    manifest_mapping, tenant_heads = parse_manifest(index_dir / "manifest.txt")
    tenant_mapping = {str(key): int(value) for key, value in scenario.get("tenant_mapping", {}).items()} or manifest_mapping
    selected_tenants = [item.strip() for item in args.tenant_range.split(",") if item.strip()]
    if not selected_tenants:
        raise RuntimeError("tenant-range must not be empty")
    if len(selected_tenants) != args.num_queries // args.batch_size:
        raise RuntimeError("tenant-range length must match num-queries / batch-size for sequential batches")

    missing_tenants = [tenant_id for tenant_id in selected_tenants if tenant_id not in tenant_mapping]
    if missing_tenants:
        raise RuntimeError(f"tenant-range contains unknown tenant ids: {missing_tenants}")

    tenant_infos = make_tenant_infos(
        base_vectors=base_vectors,
        tenant_ids=tenant_ids_data,
        tags=tags,
        tenant_counts=scenario["tenant_counts"],
        tenant_mapping=tenant_mapping,
        tenant_heads=tenant_heads,
        selected_tenants=selected_tenants,
    )
    del base_vectors
    del tags

    search_params = {
        "force_dense_tag_search": bool(args.force_dense_tag_search),
        "direct_sparse_max_postings": int(args.direct_sparse_max_postings),
        "filtered_search_nprobe_safety": float(args.filtered_search_nprobe_safety),
        "filtered_search_target_recall": float(args.filtered_search_target_recall),
        "filtered_search_coverage_exponent": float(args.filtered_search_coverage_exponent),
    }

    print("=" * 90)
    print("  Multi-Tenant + Tag Filtering Cache Stress")
    print("=" * 90)
    print(f"Scenario file: {args.scenario_file}")
    print(f"Index: {index_dir}")
    print(f"Queries: {args.num_queries} (batch size {args.batch_size})")
    print(f"TopK: {args.topk}")
    print(f"Tenants: {', '.join(selected_tenants)}")

    print("\n[1] Loading manager and building tenant signatures ...")
    manager = SPTAG.CreateTenantIndexManager(scenario["dimension"], "SPANN", "Float")
    if not manager.LoadAll(str(index_dir)):
        raise RuntimeError(f"LoadAll failed for {index_dir}")

    for tenant_id in selected_tenants:
        tenant_info = tenant_infos[tenant_id]
        tenant_info["head_bytes"] = int(manager.GetTenantHeadIndexSize(tenant_info["internal_id"]))

    head_sizes = [tenant_infos[tenant_id]["head_bytes"] for tenant_id in selected_tenants]
    if args.cache_limit_mb is not None:
        cache_limit_bytes = int(args.cache_limit_mb * MB)
        cache_limit_source = "user"
    else:
        cache_limit_bytes = choose_cache_limit_bytes(head_sizes)
        cache_limit_source = "auto"
    cache_limit_policy = "max(2 * largest_head_index, total_head_index / 4), rounded up to MB" if cache_limit_source == "auto" else "user-specified"

    apply_search_params(manager, search_params)
    build_signatures(manager, tenant_infos, selected_tenants)

    tenant_rows = []
    print("\n[2] Tenant head-index sizes ...")
    for tenant_id in selected_tenants:
        info = tenant_infos[tenant_id]
        row = {
            "tenant_id": tenant_id,
            "internal_id": info["internal_id"],
            "vector_count": info["vector_count"],
            "head_count": info["head_count"],
            "head_index_mb": info["head_bytes"] / MB,
        }
        tenant_rows.append(row)
        print(
            f"  tenant {tenant_id}: iid={info['internal_id']} vectors={info['vector_count']} "
            f"heads={info['head_count']} head_index={info['head_bytes'] / MB:.2f} MB"
        )

    manager.SetHeadIndexCacheLimit(cache_limit_bytes)
    manager.SetDropPageCacheOnEvict(bool(args.drop_page_cache_on_evict))

    rng_sequential = np.random.default_rng(args.seed)
    rng_random = np.random.default_rng(args.seed + 1)
    sequential_batches = build_sequential_batches(
        query_vectors, query_vectors_norm, tenant_infos, selected_tenants, args.batch_size, rng_sequential
    )
    random_batches = build_random_batches(
        query_vectors, query_vectors_norm, tenant_infos, selected_tenants, args.batch_size, rng_random
    )

    print("\n[3] Running sequential workload (tenant 0 -> N, one tenant per batch) ...")
    unload_all_tenants(manager, tenant_infos, selected_tenants)
    sequential_rows, sequential_overall = run_batches(manager, sequential_batches, tenant_infos, args.topk)

    print("\n[4] Running random-tenant workload (tenants mixed within each batch) ...")
    unload_all_tenants(manager, tenant_infos, selected_tenants)
    random_rows, random_overall = run_batches(manager, random_batches, tenant_infos, args.topk)

    created_at = utc_timestamp()
    git_commit, git_dirty = detect_git_state()
    batch_rows = sequential_rows + random_rows
    overall_rows = [sequential_overall, random_overall]
    payload = {
        "created_at_utc": created_at,
        "script_path": str(Path(__file__).resolve()),
        "repo_root": str(REPO_ROOT),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "scenario_file": args.scenario_file,
        "index_dir": str(index_dir),
        "query_file": args.query_file,
        "data_file": scenario["data_file"],
        "tenant_file": scenario["tenant_file"],
        "tag_file": scenario["tag_file"],
        "num_queries": args.num_queries,
        "batch_size": args.batch_size,
        "topk": args.topk,
        "seed": args.seed,
        "sequential_seed": args.seed,
        "random_seed": args.seed + 1,
        "tenant_ids": selected_tenants,
        "cache_limit_mb": cache_limit_bytes / MB,
        "cache_limit_source": cache_limit_source,
        "cache_limit_policy": cache_limit_policy,
        "drop_page_cache_on_evict": bool(args.drop_page_cache_on_evict),
        "search_params": search_params,
        "tenant_rows": tenant_rows,
        "overall_rows": overall_rows,
        "batch_rows": batch_rows,
    }
    meta = {
        "created_at_utc": created_at,
        "script_path": str(Path(__file__).resolve()),
        "repo_root": str(REPO_ROOT),
        "output_dir": str(output_dir),
        "cwd": os.getcwd(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "ld_preload": os.environ.get("LD_PRELOAD"),
        "command_argv": sys.argv,
        "scenario_file": args.scenario_file,
        "query_file": args.query_file,
        "index_dir": str(index_dir),
        "seed": args.seed,
        "sequential_seed": args.seed,
        "random_seed": args.seed + 1,
        "num_queries": args.num_queries,
        "batch_size": args.batch_size,
        "topk": args.topk,
        "tenant_range": args.tenant_range,
        "cache_limit_mb": cache_limit_bytes / MB,
        "cache_limit_source": cache_limit_source,
        "cache_limit_policy": cache_limit_policy,
        "drop_page_cache_on_evict": bool(args.drop_page_cache_on_evict),
        "search_params": search_params,
    }
    write_outputs(output_dir, payload, meta)

    print("\n[5] Overall summary")
    for row in overall_rows:
        print(
            f"  {row['scenario']}: avg={row['avg_latency_ms']:.2f}ms p95={row['p95_latency_ms']:.2f}ms "
            f"p99={row['p99_latency_ms']:.2f}ms qps={row['qps']:.2f} recall={row['avg_recall']:.4f} "
            f"sel={row['avg_selectivity_pct']:.2f}% valid={row['avg_valid']:.2f}/{row['avg_expected_valid']:.2f} "
            f"fp={row['fp_rate_pct']:.2f}% cache_end={row['cache_usage_end_mb']:.2f}MB"
        )

    print(f"\nArtifacts written to {output_dir}")


if __name__ == "__main__":
    main()