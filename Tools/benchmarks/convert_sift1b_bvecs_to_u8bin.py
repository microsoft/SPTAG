#!/usr/bin/env python3
"""Convert BigANN/SIFT bvecs(.gz) to SPTAG-style u8bin.

bvecs stores every vector as [int32 dim][uint8 dim bytes]. SPTAG's billion-scale
builders expect one header [int32 n][int32 dim] followed by contiguous vector
bytes, matching SPACEV's i8bin layout.
"""
import argparse
import gzip
import os
import struct
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="bigann_base/query/learn.bvecs or .bvecs.gz")
    p.add_argument("--output", required=True, help="output .u8bin")
    p.add_argument("--n", type=int, default=0, help="optional expected vector count")
    p.add_argument("--chunk-vectors", type=int, default=1_000_000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.input)
    dst = Path(args.output)
    opener = gzip.open if src.suffix == ".gz" else open

    with opener(src, "rb") as f:
        first = f.read(4)
        if len(first) != 4:
            raise RuntimeError(f"cannot read dimension from {src}")
        dim = struct.unpack("<i", first)[0]
        if dim <= 0:
            raise RuntimeError(f"invalid dimension {dim}")

    if src.suffix == ".gz":
        if args.n <= 0:
            raise RuntimeError("--n is required for gzipped input")
        n = args.n
    else:
        rec = 4 + dim
        size = src.stat().st_size
        if size % rec != 0:
            raise RuntimeError(f"{src} size {size} not divisible by bvecs record {rec}")
        n = size // rec
        if args.n and args.n != n:
            raise RuntimeError(f"--n {args.n} does not match file-derived n {n}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    t0 = time.time()
    with opener(src, "rb") as f, open(tmp, "wb", buffering=1024 * 1024) as out:
        out.write(struct.pack("<ii", n, dim))
        rec_payload = bytearray()
        report_next = 0
        for i in range(n):
            db = f.read(4)
            if len(db) != 4:
                raise RuntimeError(f"short read dimension at vector {i}")
            d = struct.unpack("<i", db)[0]
            if d != dim:
                raise RuntimeError(f"dimension mismatch at vector {i}: {d} != {dim}")
            vec = f.read(dim)
            if len(vec) != dim:
                raise RuntimeError(f"short read payload at vector {i}")
            rec_payload.extend(vec)
            if (i + 1) % args.chunk_vectors == 0:
                out.write(rec_payload)
                rec_payload.clear()
            if i + 1 >= report_next:
                print(f"converted {i + 1:,}/{n:,} ({time.time() - t0:.0f}s)", flush=True)
                report_next += max(args.chunk_vectors * 10, 1)
        if rec_payload:
            out.write(rec_payload)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, dst)
    print(f"done: {dst} n={n} dim={dim} bytes={dst.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
