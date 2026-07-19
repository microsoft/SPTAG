#!/usr/bin/env python3
"""Generate the 5 ACL groundtruths for SPACEV-1B (single tenant).

SPACEV-1B is 1e9 x 100 int8 vectors. The SPTAG index is built with
``Normalized=true`` + ``DistCalcMethod=L2``, i.e. cosine ranking, so the
groundtruth is computed on L2-normalized vectors by largest inner product
(= cosine nearest = normalized-L2 nearest).

Query vectors/tags are taken AS-IS from the existing stage-1 outputs
(``query_vectors.npy`` / ``query_tags.npy``); this script never regenerates
them. Base tag columns and query tag columns share the same offset encoding
(col0 in [0,4), col1 in [4,20), col2 in [20,84), col3 in [84,340)), so the
filtered candidate test is a direct ``base_tags[:,level] == query_tags[:,level]``.

Produces, under ``<scenario dir>/query/``:
  groundtruth_unfilter_local_ids.npy  - all 1e9 vectors
  groundtruth_org_local_ids.npy       - same org     as the query
  groundtruth_dept_local_ids.npy      - same org+dept
  groundtruth_team_local_ids.npy      - same ...+team
  groundtruth_project_local_ids.npy   - same ...+project
plus matching *_dists.npy (cosine distance = 1 - ip) and query_meta.json.

Neighbor ids are tenant-0 LOCAL ids = base row index (tenant is single, so
local id == global id 0..N-1).
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import faiss

LEVEL_NAMES = ("org", "dept", "team", "project")
DEFAULT_SCENARIO = "/mnt/nvme/baotonglu/mocheng/datasets/spacev1b/multitenant/tenant_tag_scenario.json"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_normalized_base(path: str, n: int, dim: int, chunk: int = 20_000_000) -> np.ndarray:
    """Memmap the i8bin (8-byte header) and build an in-RAM normalized float32 base."""
    raw = np.memmap(path, dtype=np.int8, mode="r", offset=8, shape=(n, dim))
    out = np.empty((n, dim), dtype=np.float32)
    t0 = time.time()
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        blk = np.asarray(raw[start:end], dtype=np.float32)
        norm = np.linalg.norm(blk, axis=1, keepdims=True)
        np.maximum(norm, 1e-12, out=norm)
        out[start:end] = blk / norm
        print(f"  base normalize {end}/{n}  ({time.time()-t0:.0f}s)", flush=True)
    del raw
    return out


def knn_ip(xq: np.ndarray, xb: np.ndarray, topk: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k by inner product. Returns (positions[B,k_pad], ip[B,k_pad]) padded with -1/-inf."""
    nq = xq.shape[0]
    out_pos = np.full((nq, topk), -1, dtype=np.int64)
    out_ip = np.full((nq, topk), -np.inf, dtype=np.float32)
    if xb.shape[0] == 0:
        return out_pos, out_ip
    k = min(topk, xb.shape[0])
    D, I = faiss.knn(np.ascontiguousarray(xq, dtype=np.float32),
                     np.ascontiguousarray(xb, dtype=np.float32),
                     k, metric=faiss.METRIC_INNER_PRODUCT)
    out_pos[:, :k] = I
    out_ip[:, :k] = D
    return out_pos, out_ip


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario-file", default=DEFAULT_SCENARIO)
    p.add_argument("--output-dir", default=None, help="Default: <scenario dir>/query")
    p.add_argument("--topk", type=int, default=100)
    p.add_argument("--threads", type=int, default=64)
    p.add_argument("--query-batch", type=int, default=4096)
    p.add_argument("--levels", default="unfilter,org,dept,team,project",
                   help="Comma list of GT types to (re)compute.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    faiss.omp_set_num_threads(args.threads)
    scenario = json.loads(Path(args.scenario_file).read_text(encoding="utf-8"))
    sdir = Path(args.scenario_file).parent
    output_dir = Path(args.output_dir) if args.output_dir else sdir / "query"
    output_dir.mkdir(parents=True, exist_ok=True)
    want = set(s.strip() for s in args.levels.split(",") if s.strip())

    n = int(scenario["vector_count"])
    dim = int(scenario["dimension"])
    data_file = scenario["data_file"]
    tag_file = scenario["attributes"]["acl"]["file"]

    # queries (AS-IS) -------------------------------------------------------
    qv = np.ascontiguousarray(np.load(output_dir / "query_vectors.npy"), dtype=np.float32)
    qtags = np.ascontiguousarray(np.load(output_dir / "query_tags.npy"), dtype=np.uint32)
    nq = qv.shape[0]
    qnorm = np.linalg.norm(qv, axis=1, keepdims=True)
    np.maximum(qnorm, 1e-12, out=qnorm)
    qn = np.ascontiguousarray(qv / qnorm, dtype=np.float32)
    print(f"queries: {nq} x {dim}  topk={args.topk}  threads={args.threads}", flush=True)

    # base ------------------------------------------------------------------
    print("loading + normalizing base ...", flush=True)
    base = load_normalized_base(data_file, n, dim)
    print("base ready.", flush=True)

    topk = args.topk

    def save(name: str, pos: np.ndarray, ip: np.ndarray) -> None:
        np.save(output_dir / f"groundtruth_{name}_local_ids.npy", pos)
        dist = np.where(np.isfinite(ip), 1.0 - ip, np.inf).astype(np.float32)
        np.save(output_dir / f"groundtruth_{name}_dists.npy", dist)
        print(f"  saved groundtruth_{name}_local_ids.npy", flush=True)

    # unfilter --------------------------------------------------------------
    if "unfilter" in want:
        t0 = time.time()
        pos = np.full((nq, topk), -1, dtype=np.int64)
        ip = np.full((nq, topk), -np.inf, dtype=np.float32)
        for s in range(0, nq, args.query_batch):
            e = min(s + args.query_batch, nq)
            pos[s:e], ip[s:e] = knn_ip(qn[s:e], base, topk)
            print(f"  unfilter {e}/{nq}  ({time.time()-t0:.0f}s)", flush=True)
        save("unfilter", pos, ip)

    # filtered levels -------------------------------------------------------
    base_tags = None
    for level, name in enumerate(LEVEL_NAMES):
        if name not in want:
            continue
        if base_tags is None:
            print("loading base tags ...", flush=True)
            base_tags = np.load(tag_file, mmap_mode="r")
        t0 = time.time()
        pos = np.full((nq, topk), -1, dtype=np.int64)
        ip = np.full((nq, topk), -np.inf, dtype=np.float32)
        col = np.ascontiguousarray(base_tags[:, level], dtype=np.uint32)  # 1e9 uint32 = 4GB
        qval = qtags[:, level].astype(np.uint32)
        for tag_value in np.unique(qval).tolist():
            group = np.flatnonzero(qval == tag_value)
            cand_ids = np.flatnonzero(col == np.uint32(tag_value)).astype(np.int64)
            sub = np.ascontiguousarray(base[cand_ids])  # gather candidate subset
            for s in range(0, group.size, args.query_batch):
                rows = group[s:s + args.query_batch]
                sp, si = knn_ip(qn[rows], sub, topk)
                valid = sp >= 0
                mapped = np.where(valid, cand_ids[np.where(valid, sp, 0)], -1)
                pos[rows] = mapped
                ip[rows] = si
            del sub
            print(f"  {name} tag={tag_value} cand={cand_ids.size} grp={group.size} ({time.time()-t0:.0f}s)", flush=True)
        del col
        save(name, pos, ip)

    # meta ------------------------------------------------------------------
    meta = {
        "created_at_utc": utc_timestamp(),
        "scenario_file": str(Path(args.scenario_file).resolve()),
        "dataset": scenario.get("dataset", "spacev1b"),
        "tenant": 0,
        "num_queries": int(nq),
        "topk": topk,
        "metric": "cosine(normalized-ip)",
        "id_space": "tenant_local(==global, single tenant)",
        "query_types": list(want),
        "vector_count": n,
        "groundtruth_files": {nm: f"groundtruth_{nm}_local_ids.npy" for nm in (("unfilter",) + LEVEL_NAMES) if nm in want},
    }
    (output_dir / "query_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"done. output_dir = {output_dir}", flush=True)


if __name__ == "__main__":
    main()
