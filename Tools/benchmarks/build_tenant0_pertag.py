#!/usr/bin/env python3
"""Build a tenant-0 SPANN dual-pool index (PerTagBKT head selection).

Reads the multi-tenant SIFT data produced by ``generate_tenant_tag_scenario.py``
(tenant ids + 4-level tags), restricts to one tenant, and builds a SPANN index
whose head selection uses ``PerTagBKT`` with ``--group-target`` bundle
subgraphs. Cross-edges between bundles are added afterwards by the
``augmentheadgraph`` binary (run separately).

Mode B (no U_extra)  : run this script as-is.
Mode C (with U_extra): additionally
    export SPTAG_DUAL_POOL_AUGMENT=1
    export SPTAG_DUAL_POOL_EXTRA_RATIO=0.10
before invoking it. Everything else is identical.

The index is built with the wrapper's hardcoded Cosine metric; generate query
groundtruth with ``--metric cosine`` to match.
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import SPTAG  # provided by Release/SPTAG.py (PYTHONPATH=Release)


DEFAULT_SCENARIO = "/home/v-mochengli/datasets/sift1m/multitenant/tenant_tag_scenario_1m.json"


def read_fvecs(path: str, max_vectors: int = 0) -> np.ndarray:
    with open(path, "rb") as handle:
        dimension = int(np.frombuffer(handle.read(4), dtype=np.int32)[0])
    stride = dimension + 1
    raw = np.fromfile(path, dtype=np.float32, count=max_vectors * stride if max_vectors else -1)
    count = raw.size // stride
    return np.ascontiguousarray(raw.reshape(count, stride)[:, 1:], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario-file", default=DEFAULT_SCENARIO)
    p.add_argument("--index-dir", required=True)
    p.add_argument("--group-tag-file", default="/tmp/tenant0_group_tags.txt",
                   help="Per-vector grouping signal consumed by PerTagBKT.")
    p.add_argument("--target-tenant", type=int, default=0)
    # PerTagBKT knobs
    p.add_argument("--final-ratio", type=float, default=0.139)
    p.add_argument("--group-target", type=int, default=4,
                   help="Number of bundle subgraphs (0/1 => single per-tag graph).")
    p.add_argument("--tail-replica", type=int, default=0,
                   help="K_replica for the unfilter-tail (independent of posting ReplicaCount). "
                        "Each base vector inserts into its top-K nearest heads' tail region "
                        "(tag-agnostic, scanned only by unfilter queries). 0 => disabled.")
    return p.parse_args()


def build_group_column(sub_tags: np.ndarray, group_target: int, offsets) -> tuple[np.ndarray, str, int]:
    """Greedy leaf-ancestor packing into ``group_target`` balanced buckets.

    Leaves (project tags) are lex-sorted by their (org, dept, team, project)
    ancestor path, then packed contiguously into N buckets of approximately
    equal cumulative vector count, respecting the ACL hierarchy. Returns the
    per-vector group id, a label, and the number of distinct groups used.
    """
    if not group_target or group_target <= 0:
        write_col = sub_tags[:, 2].astype(np.int64)  # team column as grouping signal
        return write_col, "team", int(np.unique(write_col).size)

    leaf_col = sub_tags[:, 3].astype(np.int64)  # project tag (global-unique id)
    order = np.argsort(leaf_col, kind="stable")
    sorted_leaf = leaf_col[order]
    sorted_paths = sub_tags[order]
    _, first_idx = np.unique(sorted_leaf, return_index=True)
    unique_leaves = sorted_leaf[first_idx]
    unique_paths = sorted_paths[first_idx]                 # (k, 4) org/dept/team/project
    leaf_counts = np.bincount(leaf_col)
    leaf_count_for_unique = leaf_counts[unique_leaves]
    lex_order = np.lexsort((unique_paths[:, 3], unique_paths[:, 2],
                            unique_paths[:, 1], unique_paths[:, 0]))
    ordered_leaves = unique_leaves[lex_order]
    ordered_counts = leaf_count_for_unique[lex_order]

    total = int(ordered_counts.sum())
    bucket_target = total // group_target
    leaf_to_group: dict[int, int] = {}
    gid = 0
    cur_size = 0
    for leaf_id, cnt in zip(ordered_leaves.tolist(), ordered_counts.tolist()):
        if cur_size >= bucket_target and gid < group_target - 1:
            gid += 1
            cur_size = 0
        leaf_to_group[int(leaf_id)] = gid
        cur_size += int(cnt)

    group_col = np.array([leaf_to_group[int(v)] for v in leaf_col.tolist()], dtype=np.int64)
    sizes = np.bincount(group_col, minlength=group_target)
    print(f"[3a] Greedy leaf-packing -> {group_target} groups, sizes={sizes.tolist()} (target~={bucket_target})")
    for g in range(group_target):
        ls = [int(l) for l in ordered_leaves.tolist() if leaf_to_group[int(l)] == g]
        if not ls:
            continue
        paths_in_g = unique_paths[np.isin(unique_leaves, ls)]
        org_set = sorted(set((paths_in_g[:, 0] - offsets[0]).tolist()))
        dept_set = sorted(set((paths_in_g[:, 1] - offsets[1]).tolist()))
        team_set = sorted(set((paths_in_g[:, 2] - offsets[2]).tolist()))
        print(f"     group {g}: {len(ls)} leaves | org={org_set} | depts={dept_set} | teams={len(team_set)}")
    return group_col, f"group(target={group_target})", group_target


def main() -> None:
    args = parse_args()
    scenario = json.loads(Path(args.scenario_file).read_text(encoding="utf-8"))
    offsets = scenario.get("tag_level_offsets", [0, 4, 20, 84])
    n_total = scenario["vector_count"]

    print(f"[1] Load vectors ({n_total}) {scenario['data_file']}")
    vectors = read_fvecs(scenario["data_file"], n_total)
    n, dim = vectors.shape
    print(f"    {n} x {dim}")

    print("[2] Load tenant ids + tags")
    tenant_ids = np.loadtxt(scenario["tenant_file"], dtype=np.int64).reshape(-1)[:n_total]
    tags = np.asarray(np.load(scenario["tag_file"], allow_pickle=False), dtype=np.uint32)
    assert tenant_ids.shape[0] == n
    assert tags.shape == (n, 4)

    mask = tenant_ids == args.target_tenant
    n_t0 = int(mask.sum())
    print(f"    tenant {args.target_tenant}: {n_t0} vectors")
    sub_vectors = np.ascontiguousarray(vectors[mask], dtype=np.float32)
    sub_tags = np.ascontiguousarray(tags[mask], dtype=np.uint32)

    group_col, col_label, n_groups = build_group_column(sub_tags, args.group_target, offsets)
    Path(args.group_tag_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.group_tag_file, "w") as fh:
        for v in group_col.tolist():
            fh.write(f"{int(v)}\n")
    print(f"[3] Wrote per-vector {col_label} grouping file: {args.group_tag_file} "
          f"({n_t0} entries, {n_groups} distinct values)")

    metadata = ("0\n" * n_t0).encode("utf-8")

    env_settings = {
        "SPTAG_HEAD_SELECT_DEBUG":    "1",
        "SPTAG_PER_VECTOR_TAGS_FILE": args.group_tag_file,
        "SPTAG_PERTAG_HEAD_RATIO":    str(args.final_ratio),
        "SPTAG_SELECT_TYPE_OVERRIDE": "PerTagBKT",
    }
    for k, v in env_settings.items():
        os.environ[k] = v
        print(f"    {k}={v}")
    for opt in ("SPTAG_DUAL_POOL_AUGMENT", "SPTAG_DUAL_POOL_EXTRA_RATIO"):
        if opt in os.environ:
            print(f"    {opt}={os.environ[opt]}  (U_extra augmentation ON)")

    if args.tail_replica > 0:
        os.environ["SPTAG_UNFILTER_TAIL_K_REPLICA"] = str(args.tail_replica)
        print(f"    SPTAG_UNFILTER_TAIL_K_REPLICA={args.tail_replica}  (unfilter-tail K_replica ON)")

    out = Path(args.index_dir)
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for stale in Path("/tmp").glob("sptag_pertag_tenant_*"):
        if stale.is_dir():
            shutil.rmtree(stale)

    print(f"[4] BuildFromDataWithTags (ratio={args.final_ratio}, group-target={args.group_target})")
    mgr = SPTAG.CreateTenantIndexManager(dim, "SPANN", "Float")
    storage_backend = os.environ.get("SPTAG_STORAGE_BACKEND", "FILEIO")
    if storage_backend != "FILEIO":
        mgr.SetStorageBackend(storage_backend)
        print(f"    StorageBackend={storage_backend}")
    t0 = time.perf_counter()
    ok = mgr.BuildFromDataWithTags(
        sub_vectors.tobytes(), metadata, n_t0,
        sub_tags.tobytes(), sub_tags.shape[1], True, False,
    )
    dt = time.perf_counter() - t0
    print(f"    ok={ok}, time={dt:.1f}s")
    if not ok:
        raise RuntimeError("BuildFromDataWithTags failed")
    if not mgr.SaveAll(str(out)):
        raise RuntimeError(f"SaveAll failed for {out}")
    print(f"[5] Saved -> {out}")
    print(f"    Next: Release/augmentheadgraph -d {out}/tenant_0/HeadIndex -k 15 -m 10 -t 16 -w true")


if __name__ == "__main__":
    main()
