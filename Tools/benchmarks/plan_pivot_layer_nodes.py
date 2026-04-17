#!/usr/bin/env python3
import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


DEFAULT_SCENARIO_FILE = "/home/v-mochengli/test/tenant_tag_scenario_1m.json"
DEFAULT_OUTPUT_ROOT = Path("/home/v-mochengli/test")
LEVEL_NAMES = ("org", "dept", "team", "project")


@dataclass
class LevelInfo:
    name: str
    base: int
    cardinality: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan pivot-layer node split with hierarchy mapping and multiplicative selectivity cost"
    )
    parser.add_argument("--scenario-file", default=DEFAULT_SCENARIO_FILE)
    parser.add_argument("--tenant-id", type=int, default=0)
    parser.add_argument("--pivot-level", type=int, default=1, help="0=org,1=dept,2=team,3=project")
    parser.add_argument("--max-nodes", type=int, default=5)
    parser.add_argument("--r-target", type=float, default=0.99)
    parser.add_argument("--lambda-recall", type=float, default=10.0)
    parser.add_argument("--estimated-recall", type=float, default=1.0)
    parser.add_argument("--query-level-weights", default="1,1,1,1")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def load_level_infos(scenario: dict) -> list[LevelInfo]:
    infos: list[LevelInfo] = []
    for row in scenario["tag_levels"]:
        infos.append(LevelInfo(name=str(row["name"]), base=int(row["base"]), cardinality=int(row["cardinality"])))
    return infos


def load_tenant_arrays(scenario: dict, tenant_id: int) -> tuple[np.ndarray, np.ndarray]:
    tenant_ids = np.loadtxt(scenario["tenant_file"], dtype=np.int64).reshape(-1)[: scenario["vector_count"]]
    tags = np.asarray(np.load(scenario["tag_file"], allow_pickle=False), dtype=np.uint32)
    mask = tenant_ids == tenant_id
    local_ids = np.flatnonzero(mask).astype(np.int64)
    return local_ids, np.ascontiguousarray(tags[mask], dtype=np.uint32)


def parse_weights(raw: str, level_count: int) -> np.ndarray:
    parts = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(parts) != level_count:
        raise RuntimeError(f"query-level-weights must have {level_count} values")
    arr = np.asarray(parts, dtype=np.float64)
    if np.any(arr < 0):
        raise RuntimeError("query-level-weights must be non-negative")
    s = float(np.sum(arr))
    if s <= 0:
        raise RuntimeError("query-level-weights sum must be positive")
    return arr / s


def tag_parent(child_tag: int, parent_level: int, level_infos: list[LevelInfo]) -> int:
    child_level = parent_level + 1
    parent = level_infos[parent_level]
    child = level_infos[child_level]
    if child.cardinality % parent.cardinality != 0:
        raise RuntimeError("cardinality ratio must be integer for static hierarchy mapping")
    fanout = child.cardinality // parent.cardinality
    child_idx = child_tag - child.base
    parent_idx = child_idx // fanout
    return parent.base + parent_idx


def ancestor_at_level(tag_value: int, from_level: int, target_level: int, level_infos: list[LevelInfo]) -> int:
    current = int(tag_value)
    level = int(from_level)
    while level > target_level:
        current = tag_parent(current, level - 1, level_infos)
        level -= 1
    return current


def build_pivot_groups(
    pivot_tags: np.ndarray,
    pivot_counts: np.ndarray,
    node_count: int,
    pivot_level: int,
    level_infos: list[LevelInfo],
) -> list[list[int]]:
    groups: list[list[int]] = [[] for _ in range(node_count)]
    if node_count <= 0:
        return groups

    if pivot_level == 0:
        ordered = sorted(zip(pivot_tags.tolist(), pivot_counts.tolist()), key=lambda x: x[0])
        for i, (tag, _) in enumerate(ordered):
            groups[i % node_count].append(int(tag))
        return groups

    parent_to_children: dict[int, list[tuple[int, int]]] = {}
    for tag, cnt in zip(pivot_tags.tolist(), pivot_counts.tolist()):
        parent = tag_parent(int(tag), pivot_level - 1, level_infos)
        parent_to_children.setdefault(parent, []).append((int(tag), int(cnt)))

    ordered_parents = sorted(
        parent_to_children.items(),
        key=lambda item: sum(cnt for _, cnt in item[1]),
        reverse=True,
    )

    group_load = [0 for _ in range(node_count)]
    for _, children in ordered_parents:
        children_sorted = sorted(children, key=lambda x: x[0])
        idx = int(np.argmin(group_load))
        for tag, cnt in children_sorted:
            groups[idx].append(tag)
            group_load[idx] += cnt

    return groups


def build_tag_to_node(groups: list[list[int]]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for node_id, tags in enumerate(groups):
        for tag in tags:
            mapping[int(tag)] = node_id
    return mapping


def estimate_cost(
    tenant_tags: np.ndarray,
    groups: list[list[int]],
    pivot_level: int,
    level_infos: list[LevelInfo],
    query_level_weights: np.ndarray,
    r_target: float,
    lambda_recall: float,
    estimated_recall: float,
) -> dict:
    node_count = len(groups)
    tag_to_node = build_tag_to_node(groups)
    pivot_col = tenant_tags[:, pivot_level].astype(np.int64)

    node_sizes = np.zeros(node_count, dtype=np.int64)
    for pivot_tag in pivot_col.tolist():
        node_sizes[tag_to_node[int(pivot_tag)]] += 1

    total_rows = float(tenant_tags.shape[0])
    if total_rows <= 0:
        raise RuntimeError("tenant has no vectors")

    level_components: list[dict] = []
    total_latency_proxy = 0.0

    for level in range(tenant_tags.shape[1]):
        values, counts = np.unique(tenant_tags[:, level], return_counts=True)
        level_weight = float(query_level_weights[level])
        level_cost = 0.0

        for tag, cnt in zip(values.tolist(), counts.tolist()):
            sel = max(float(cnt) / total_rows, 1e-9)

            if level < pivot_level:
                touched_nodes = set()
                for pivot_tag in build_tag_to_node(groups).keys():
                    anc = ancestor_at_level(int(pivot_tag), pivot_level, level, level_infos)
                    if anc == int(tag):
                        touched_nodes.add(tag_to_node[int(pivot_tag)])
            elif level == pivot_level:
                touched_nodes = {tag_to_node[int(tag)]} if int(tag) in tag_to_node else set()
            else:
                anc = ancestor_at_level(int(tag), level, pivot_level, level_infos)
                touched_nodes = {tag_to_node[int(anc)]} if int(anc) in tag_to_node else set()

            if not touched_nodes:
                continue

            touched_size = float(np.sum(node_sizes[list(touched_nodes)]))
            base_latency = math.log2(touched_size + 1.0)
            latency_proxy = base_latency / sel
            prob = float(cnt) / total_rows
            level_cost += prob * latency_proxy

        weighted = level_weight * level_cost
        total_latency_proxy += weighted
        level_components.append(
            {
                "level": level,
                "level_name": LEVEL_NAMES[level] if level < len(LEVEL_NAMES) else str(level),
                "weight": level_weight,
                "latency_proxy": level_cost,
                "weighted_latency_proxy": weighted,
            }
        )

    recall_penalty = float(lambda_recall) * max(0.0, float(r_target) - float(estimated_recall))
    total_cost = total_latency_proxy + recall_penalty

    return {
        "node_sizes": node_sizes.tolist(),
        "level_components": level_components,
        "latency_proxy": total_latency_proxy,
        "recall_penalty": recall_penalty,
        "estimated_recall": float(estimated_recall),
        "total_cost": total_cost,
    }


def main() -> None:
    args = parse_args()
    scenario = json.loads(Path(args.scenario_file).read_text(encoding="utf-8"))
    level_infos = load_level_infos(scenario)

    if args.pivot_level < 0 or args.pivot_level >= len(level_infos):
        raise RuntimeError(f"pivot-level must be in [0, {len(level_infos) - 1}]")
    if args.max_nodes < 2:
        raise RuntimeError("max-nodes must be >= 2")

    max_nodes = min(int(args.max_nodes), 5)
    query_weights = parse_weights(args.query_level_weights, len(level_infos))

    _, tenant_tags = load_tenant_arrays(scenario, args.tenant_id)
    pivot_values, pivot_counts = np.unique(tenant_tags[:, args.pivot_level], return_counts=True)

    plans: list[dict] = []
    for node_count in range(2, max_nodes + 1):
        groups = build_pivot_groups(
            pivot_tags=pivot_values,
            pivot_counts=pivot_counts,
            node_count=node_count,
            pivot_level=args.pivot_level,
            level_infos=level_infos,
        )
        cost = estimate_cost(
            tenant_tags=tenant_tags,
            groups=groups,
            pivot_level=args.pivot_level,
            level_infos=level_infos,
            query_level_weights=query_weights,
            r_target=args.r_target,
            lambda_recall=args.lambda_recall,
            estimated_recall=args.estimated_recall,
        )
        plans.append(
            {
                "node_count": node_count,
                "pivot_groups": groups,
                "cost": cost,
            }
        )

    best = min(plans, key=lambda x: x["cost"]["total_cost"])

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else DEFAULT_OUTPUT_ROOT / f"pivot_layer_plan_t{args.tenant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scenario_file": args.scenario_file,
        "tenant_id": int(args.tenant_id),
        "pivot_level": int(args.pivot_level),
        "pivot_level_name": level_infos[args.pivot_level].name,
        "max_nodes": int(max_nodes),
        "query_level_weights": query_weights.tolist(),
        "r_target": float(args.r_target),
        "lambda_recall": float(args.lambda_recall),
        "estimated_recall": float(args.estimated_recall),
        "plans": plans,
        "best_plan": best,
    }

    (output_dir / "plan.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Pivot-Layer Node Plan",
        "",
        f"- Scenario: {args.scenario_file}",
        f"- Tenant: {args.tenant_id}",
        f"- Pivot level: {args.pivot_level} ({level_infos[args.pivot_level].name})",
        f"- Max nodes: {max_nodes}",
        f"- Recall penalty: lambda * max(0, r_target - r_est)",
        f"- Cost latency term: base_latency / selectivity",
        "",
        "## Candidate Costs",
        "",
        "| Nodes | Total Cost | Latency Proxy | Recall Penalty | Node Sizes |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]

    for plan in plans:
        c = plan["cost"]
        lines.append(
            f"| {plan['node_count']} | {c['total_cost']:.4f} | {c['latency_proxy']:.4f} | {c['recall_penalty']:.4f} | {c['node_sizes']} |"
        )

    lines += [
        "",
        "## Best Plan",
        "",
        f"- Node count: {best['node_count']}",
        f"- Total cost: {best['cost']['total_cost']:.4f}",
        f"- Node sizes: {best['cost']['node_sizes']}",
        "- Pivot tag groups:",
    ]

    for node_id, tags in enumerate(best["pivot_groups"]):
        lines.append(f"  - Node {node_id}: {tags}")

    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Plan json:  {output_dir / 'plan.json'}")
    print(f"Plan report:{output_dir / 'report.md'}")
    print(f"Best node count: {best['node_count']}")


if __name__ == "__main__":
    main()
