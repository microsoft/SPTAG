#!/usr/bin/env python3
"""Select U_extra by RNG dominate-count (standalone, does NOT touch the build flow).

Idea (graph-native, distance-agnostic): a candidate base vector is valuable as a
new head iff inserting it would contribute many *non-redundant* edges to the head
graph. We emulate the "insert a new point" step: for each non-head candidate we
fetch its nearest heads, then apply the RNG (relative-neighbourhood) occlusion
rule and count how many of those neighbours get *dominated* (occluded by a closer
neighbour). Few dominated  -> neighbours point in diverse directions -> the point
strengthens connectivity / consolidates a fragmented region -> keep it.

We pick the ``ratio * |H1|`` candidates with the FEWEST dominated neighbours and
write their tenant-local VIDs to a binary file:  int32 count, then int32[count].
That file is consumed by the build via  ``SPTAG_UEXTRA_ID_FILE``  (a gated branch
in the DualPool augmentation; the default random path is unchanged).

RNG occlusion (lune form, alpha=1, matching the head graph's RNGFactor=1.0):
  neighbour n_i is DOMINATED iff  exists n_j with
        d(p, n_j) < d(p, n_i)  AND  d(n_j, n_i) < d(p, n_i)
All distances are cosine ``1 - dot`` on L2-normalised vectors (the head index uses
Cosine). The head index (graph search) is used ONLY to retrieve the candidate
neighbour set; every distance used in the rule is recomputed here for fidelity.
"""
import argparse
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/v-mochengli/SPTAG/Release")
import SPTAG  # noqa: E402


def read_fvecs(path: str, count: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32, count=-1)
    dim = int(raw[0])
    rec = dim + 1
    raw = raw.reshape(-1, rec)[:count]
    return np.ascontiguousarray(raw[:, 1:].view(np.float32), dtype=np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-file", default="/home/v-mochengli/datasets/sift1m/sift/sift_base.fvecs")
    p.add_argument("--tenant-file", default="/home/v-mochengli/datasets/sift1m/multitenant/tenant_ids.txt")
    p.add_argument("--vector-count", type=int, default=1000000)
    p.add_argument("--target-tenant", type=int, default=0)
    p.add_argument("--role-index", required=True,
                   help="An existing build's tenant_0 dir; H1 = heads with role==0 "
                        "(head_role.bin + SPTAGHeadVectorIDs.bin).")
    p.add_argument("--ratio", type=float, default=0.10, help="|U_extra| = round(ratio*|H1|)")
    p.add_argument("--knn", type=int, default=64,
                   help="candidate neighbour pool per insert (=2*degree; ef is already 1000)")
    p.add_argument("--threads", type=int, default=0, help="0 => os.cpu_count()")
    p.add_argument("--out", default="/tmp/uextra_rng_ids.bin")
    p.add_argument("--mode", choices=["dominate", "bridge"], default="bridge",
                   help="dominate=fewest-dominated (hub-ish); bridge=d1*spread(survivors)")
    p.add_argument("--limit", type=int, default=0, help="score only first N candidates (timing probe; 0=all)")
    return p.parse_args()


def load_h1_vids(role_dir: Path) -> np.ndarray:
    role = np.fromfile(role_dir / "head_role.bin", dtype=np.uint8)
    raw = np.fromfile(role_dir / "SPTAGHeadVectorIDs.bin", dtype=np.int32)
    n = int(raw[0])
    vids = np.frombuffer(raw[2:].tobytes(), dtype=np.int64)[:n].astype(np.int64)
    assert len(role) == n, f"role={len(role)} vs ids={n}"
    return vids[role == 0].astype(np.int64)


def main() -> None:
    args = parse_args()
    threads = args.threads or (os.cpu_count() or 8)

    print(f"[1] load base vectors {args.data_file}")
    vecs = read_fvecs(args.data_file, args.vector_count)
    tenant = np.loadtxt(args.tenant_file, dtype=np.int64).reshape(-1)[:args.vector_count]
    mask = tenant == args.target_tenant
    sub = np.ascontiguousarray(vecs[mask], dtype=np.float32)
    n, dim = sub.shape
    print(f"    tenant {args.target_tenant}: {n} x {dim}")

    norms = np.linalg.norm(sub, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    nrm = (sub / norms).astype(np.float32)

    print(f"[2] load H1 from {args.role_index}")
    h1 = load_h1_vids(Path(args.role_index))
    h1_set = np.zeros(n, dtype=bool)
    h1_set[h1] = True
    n_extras = int(round(args.ratio * len(h1)))
    print(f"    |H1|={len(h1)}  ratio={args.ratio}  -> |U_extra|={n_extras}")

    head_vecs = np.ascontiguousarray(nrm[h1], dtype=np.float32)  # ordinal -> H1 row

    print(f"[3] build head KDT index ({len(h1)} heads, Cosine)")
    idx = SPTAG.AnnIndex("KDT", "Float", dim)
    idx.SetBuildParam("NumberOfThreads", str(threads), "Index")
    idx.SetBuildParam("DistCalcMethod", "Cosine", "Index")
    idx.SetBuildParam("NeighborhoodSize", "32", "Index")
    idx.SetBuildParam("GraphNeighborhoodScale", "2", "Index")
    idx.SetBuildParam("CEF", "1000", "Index")
    idx.SetBuildParam("RNGFactor", "1", "Index")
    idx.SetBuildParam("MaxCheckForRefineGraph", "8192", "Index")
    t0 = time.perf_counter()
    if not idx.Build(head_vecs, head_vecs.shape[0], False):
        raise RuntimeError("head index Build failed")
    idx.SetSearchParam("MaxCheck", "8192", "Index")
    print(f"    built in {time.perf_counter()-t0:.1f}s")

    cand = np.where(~h1_set)[0]
    if args.limit and args.limit < len(cand):
        cand = cand[:args.limit]
        print(f"    [limit] scoring only first {len(cand)} candidates")
    K = args.knn
    print(f"[4] score {len(cand)} candidates (knn={K}, RNG lune alpha=1, mode={args.mode})")
    dominated = np.empty(len(cand), dtype=np.int32)
    survivors = np.empty(len(cand), dtype=np.int32)
    bscore = np.zeros(len(cand), dtype=np.float64)   # bridge score = d1 * mean_pairwise(survivors)
    d1arr = np.zeros(len(cand), dtype=np.float64)

    t0 = time.perf_counter()
    for ci, v in enumerate(cand):
        res = idx.Search(nrm[v], K)
        nb = np.asarray(res[0], dtype=np.int64)
        nb = nb[nb >= 0]
        m = len(nb)
        if m <= 1:
            dominated[ci] = 0
            survivors[ci] = m
            continue
        H = head_vecs[nb]                       # (m, dim) normalised
        dp = 1.0 - H @ nrm[v]                    # d(p, n_i)  (m,)
        order = np.argsort(dp, kind="stable")    # ascending by distance to p
        dp = dp[order]
        H = H[order]
        pair = 1.0 - H @ H.T                     # d(n_j, n_i) (m, m)
        # n_i dominated iff exists j<i (closer to p) with pair[j,i] < dp[i]
        closer = dp[:, None] < dp[None, :]       # closer[j,i] = dp[j] < dp[i]
        occ = (pair < dp[None, :]) & closer      # (j,i)
        dom = occ.any(axis=0)                    # over j, per i
        d = int(dom.sum())
        dominated[ci] = d
        survivors[ci] = m - d
        # bridge: gap (distance to nearest head) x spread of the surviving heads
        # (mean mutual distance among RNG survivors). Large only when p sits in a
        # gap AND its diverse heads span separated clusters (not a dense hub).
        surv = ~dom
        ns = int(surv.sum())
        d1 = float(dp[0])
        d1arr[ci] = d1
        if ns >= 2:
            sp = pair[np.ix_(surv, surv)]
            iu = np.triu_indices(ns, 1)
            bscore[ci] = d1 * float(sp[iu].mean())
        if (ci + 1) % 50000 == 0:
            el = time.perf_counter() - t0
            print(f"    {ci+1}/{len(cand)}  ({el:.0f}s, {(ci+1)/el:.0f}/s)")

    print(f"    scored in {time.perf_counter()-t0:.1f}s")
    print(f"    survivors: mean={survivors.mean():.2f} "
          f"p10={np.percentile(survivors,10):.0f} p50={np.percentile(survivors,50):.0f} "
          f"p90={np.percentile(survivors,90):.0f} max={survivors.max()}")
    print(f"    d1(nearest-head dist): mean={d1arr.mean():.4f} p50={np.percentile(d1arr,50):.4f} "
          f"p90={np.percentile(d1arr,90):.4f}")
    print(f"    bridge_score: mean={bscore.mean():.5f} p50={np.percentile(bscore,50):.5f} "
          f"p90={np.percentile(bscore,90):.5f} max={bscore.max():.5f}")

    take = min(n_extras, len(cand))
    if args.mode == "bridge":
        # highest bridge_score; tie-break by VID
        rank = np.lexsort((cand, -bscore))
        thr_i = rank[take - 1]
        print(f"[5] selected {take} U_extra by BRIDGE "
              f"(bridge_score>= {bscore[thr_i]:.5f}, d1>= ~, survivors~{survivors[rank[:take]].mean():.1f})")
    else:
        # fewest dominated == most survivors; tie-break by VID
        rank = np.lexsort((cand, dominated))
        print(f"[5] selected {take} U_extra by DOMINATE "
              f"(dominated<= {dominated[rank[take-1]]}, survivors>= {survivors[rank[take-1]]})")
    chosen = np.sort(cand[rank[:take]]).astype(np.int32)

    with open(args.out, "wb") as f:
        f.write(struct.pack("<i", len(chosen)))
        f.write(chosen.tobytes())
    print(f"[6] wrote {args.out}  (int32 count={len(chosen)} + int32 VIDs)")


if __name__ == "__main__":
    main()
