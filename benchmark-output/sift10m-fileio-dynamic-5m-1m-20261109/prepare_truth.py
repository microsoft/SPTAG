import pathlib
import struct
import time

import faiss
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parent
SOURCE = ROOT.parent / "sift10m-fileio-layer2-build80-20261109" / "input"
DESTINATION = ROOT / "perftest_batchtruth.L2.UInt8_5000000_128_1000_5_1000000_0_5"


def vectors(path):
    with path.open("rb") as stream:
        count, dimension = struct.unpack("<ii", stream.read(8))
    if dimension != 128 or path.stat().st_size != 8 + count * dimension:
        raise ValueError(f"Invalid UInt8 vector file: {path}")
    return np.memmap(path, dtype=np.uint8, mode="r", offset=8, shape=(count, dimension))


def main():
    data = vectors(SOURCE / "sift-first-10m.u8bin")
    queries = np.asarray(vectors(SOURCE / "query.10K.u8bin")[:1000], dtype=np.float32)
    if data.shape[0] < 10_000_000 or queries.shape != (1000, 128):
        raise ValueError("The experiment requires 10M vectors and 1000 queries")
    faiss.omp_set_num_threads(16)
    index = faiss.IndexFlatL2(128)
    ids, distances = [], []
    previous = 0
    for count in range(5_000_000, 10_000_001, 1_000_000):
        started = time.monotonic()
        for offset in range(previous, count, 250_000):
            index.add(np.asarray(data[offset:min(offset + 250_000, count)], dtype=np.float32))
        distance, selected = index.search(queries, 5)
        if selected.shape != (1000, 5) or np.any(selected < 0) or np.any(selected >= count):
            raise RuntimeError(f"Invalid exact neighbor IDs for prefix {count}")
        difference = queries[:, None, :] - np.asarray(data[selected], dtype=np.float32)
        actual = np.sum(difference * difference, axis=2)
        if not np.array_equal(actual, distance):
            raise RuntimeError(f"Faiss distances disagree with direct squared L2 for prefix {count}")
        ids.append(selected.astype("<i4"))
        distances.append(distance.astype("<f4"))
        previous = count
        print(f"Exact ground truth: {count} live vectors, 1000 queries, {time.monotonic() - started:.1f}s", flush=True)
    with DESTINATION.open("wb") as stream:
        stream.write(struct.pack("<ii", 12000, 5))
        stream.write(np.concatenate(ids).tobytes())
        stream.write(np.concatenate(distances).tobytes())
    if DESTINATION.stat().st_size != 240008:
        raise RuntimeError("Unexpected benchmark truth file size")
    print(f"Saved {DESTINATION}", flush=True)


if __name__ == "__main__":
    main()
