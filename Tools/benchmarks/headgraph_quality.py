#!/usr/bin/env python3
"""Assess head-graph approximation quality for unfilter search.

Question: at a given nprobe, how many of the heads the graph search actually
probes are NOT among the true nearest heads (brute force)? Those false-positive
heads are "far postings that should not be candidates" -> wasted IO/scan.

We rebuild the full head graph (all heads, role 0 and 1) with the SAME build
params as deployment (Cosine, deg32, CEF1000, RNG1) -- the deployed graph also
has augmentheadgraph cross-edges, so this is a conservative (lower-bound) proxy
for the real graph's reach. For 2000 SIFT queries we compare graph top-nprobe
vs brute-force top-nprobe and report recall@nprobe and the TRUE rank of the
wrongly-probed heads.
"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/v-mochengli/SPTAG/Release")
import SPTAG


def read_fvecs(path, count=-1):
    raw = np.fromfile(path, dtype=np.int32)
    dim = int(raw[0]); rec = dim + 1
    raw = raw.reshape(-1, rec)
    if count > 0:
        raw = raw[:count]
    return np.ascontiguousarray(raw[:, 1:].view(np.float32), dtype=np.float32)


def load_all_head_vids(role_dir):
    role = np.fromfile(role_dir / "head_role.bin", dtype=np.uint8)
    raw = np.fromfile(role_dir / "SPTAGHeadVectorIDs.bin", dtype=np.int32)
    n = int(raw[0])
    vids = np.frombuffer(raw[2:].tobytes(), dtype=np.int64)[:n].astype(np.int64)
    assert len(role) == n
    return vids, role


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-file", default="/home/v-mochengli/datasets/sift1m/sift/sift_base.fvecs")
    ap.add_argument("--query-file", default="/home/v-mochengli/datasets/sift1m/sift/sift_query.fvecs")
    ap.add_argument("--tenant-file", default="/home/v-mochengli/datasets/sift1m/multitenant/tenant_ids.txt")
    ap.add_argument("--vector-count", type=int, default=1000000)
    ap.add_argument("--target-tenant", type=int, default=0)
    ap.add_argument("--role-index", required=True)
    ap.add_argument("--num-queries", type=int, default=2000)
    ap.add_argument("--nprobes", default="8,16,32,44")
    ap.add_argument("--maxchecks", default="1024,2048,4096,8192")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    threads = args.threads or (os.cpu_count() or 8)

    print(f"[1] load base {args.data_file}")
    vecs = read_fvecs(args.data_file)
    tenant = np.loadtxt(args.tenant_file, dtype=np.int64).reshape(-1)[:args.vector_count]
    sub = np.ascontiguousarray(vecs[tenant == args.target_tenant], dtype=np.float32)
    n, dim = sub.shape
    nr = sub / np.clip(np.linalg.norm(sub, axis=1, keepdims=True), 1e-12, None)
    nr = nr.astype(np.float32)
    print(f"    tenant {args.target_tenant}: {n} x {dim}")

    vids, role = load_all_head_vids(Path(args.role_index))
    head_vecs = np.ascontiguousarray(nr[vids], dtype=np.float32)   # ordinal -> head row
    H = head_vecs.shape[0]
    print(f"[2] heads={H}  (H1={int((role==0).sum())}, U_extra={int((role==1).sum())})")

    print(f"[3] build full head KDT ({H} heads, Cosine, deg32 CEF1000 RNG1)")
    idx = SPTAG.AnnIndex("KDT", "Float", dim)
    for k, v in [("NumberOfThreads", str(threads)), ("DistCalcMethod", "Cosine"),
                 ("NeighborhoodSize", "32"), ("GraphNeighborhoodScale", "2"),
                 ("CEF", "1000"), ("RNGFactor", "1"), ("MaxCheckForRefineGraph", "8192")]:
        idx.SetBuildParam(k, v, "Index")
    t0 = time.perf_counter()
    if not idx.Build(head_vecs, H, False):
        raise RuntimeError("build failed")
    print(f"    built in {time.perf_counter()-t0:.1f}s")

    q = read_fvecs(args.query_file)[:args.num_queries]
    qn = (q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)).astype(np.float32)
    Q = qn.shape[0]
    print(f"[4] queries={Q}")

    nprobes = [int(x) for x in args.nprobes.split(",")]
    maxchecks = [int(x) for x in args.maxchecks.split(",")]
    maxnp = max(nprobes)

    # brute-force true ranking of heads per query (top by cosine = top by dot)
    print("[5] brute-force true nearest heads ...")
    sims = qn @ head_vecs.T                       # (Q, H) cosine sim
    true_order = np.argsort(-sims, axis=1)        # ascending distance
    true_topmax = true_order[:, :maxnp]           # (Q, maxnp)
    # true rank of every head per query (inverse permutation)
    true_rank = np.empty((Q, H), dtype=np.int32)
    rows = np.arange(Q)[:, None]
    true_rank[rows, true_order] = np.arange(H)[None, :]

    for mc in maxchecks:
        idx.SetSearchParam("MaxCheck", str(mc), "Index")
        # collect graph top-maxnp per query
        approx = np.full((Q, maxnp), -1, dtype=np.int64)
        t0 = time.perf_counter()
        for i in range(Q):
            r = idx.Search(qn[i], maxnp)
            a = np.asarray(r[0], dtype=np.int64)
            approx[i, :len(a)] = a[:maxnp]
        el = time.perf_counter() - t0
        line = [f"\n=== MaxCheck={mc}  ({Q/el:.0f} head-search/s) ==="]
        for npb in nprobes:
            tt = set
            hits = np.zeros(Q, dtype=np.int32)
            fp_ranks = []
            for i in range(Q):
                ap_set = set(int(x) for x in approx[i, :npb] if x >= 0)
                tr_set = set(int(x) for x in true_topmax[i, :npb])
                inter = ap_set & tr_set
                hits[i] = len(inter)
                for fp in (ap_set - tr_set):
                    fp_ranks.append(true_rank[i, fp])
            rec = hits.mean() / npb
            fp_ranks = np.array(fp_ranks) if fp_ranks else np.array([npb])
            line.append(
                f"  nprobe={npb:>3}: head-recall@{npb}={rec:.4f}  "
                f"wasted/q={npb*(1-rec):.2f}  "
                f"FP true-rank p50={np.percentile(fp_ranks,50):.0f} "
                f"p90={np.percentile(fp_ranks,90):.0f} max={fp_ranks.max():.0f}")
        print("\n".join(line))


if __name__ == "__main__":
    main()
