#!/usr/bin/env python3
"""Regenerate head_node_meta.bin (incl. per-posting hier masks) for a slim index
by calling BuildSignatures on the loaded tenant-0 manager."""
import argparse
from pathlib import Path
import numpy as np
import SPTAG

SCEN = {
    "tenant_file": "/home/v-mochengli/datasets/sift1m/multitenant/tenant_ids.txt",
    "tag_file": "/home/v-mochengli/datasets/sift1m/multitenant/tags.npy",
    "vector_count": 1000000,
    "dimension": 128,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--tenant", type=int, default=0)
    args = ap.parse_args()

    tenant_ids = np.loadtxt(SCEN["tenant_file"], dtype=np.int64).reshape(-1)[: SCEN["vector_count"]]
    tags = np.asarray(np.load(SCEN["tag_file"], allow_pickle=False), dtype=np.uint32)
    mask = tenant_ids == args.tenant
    t0_tags = np.ascontiguousarray(tags[mask], dtype=np.uint32)
    t0_size = int(t0_tags.shape[0])
    print(f"tenant {args.tenant}: {t0_size} vectors, {t0_tags.shape[1]} tag levels")

    mgr = SPTAG.CreateTenantIndexManager(SCEN["dimension"], "SPANN", "Float")
    if not mgr.LoadAll(str(args.index_dir)):
        raise RuntimeError("LoadAll failed")
    ok = mgr.BuildSignatures(args.tenant, t0_tags.tobytes(), t0_size, t0_tags.shape[1])
    if ok is False:
        raise RuntimeError("BuildSignatures failed")
    print("BuildSignatures OK -> head_node_meta.bin regenerated")


if __name__ == "__main__":
    main()
