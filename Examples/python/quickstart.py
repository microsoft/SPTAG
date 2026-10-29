"""Build or reload a Float/L2 index using the public SPTAG Python binding."""

import argparse
import json
from pathlib import Path

import numpy as np
import SPTAG


def load_vectors(path):
    vectors = np.load(path, allow_pickle=False)
    if vectors.ndim != 2 or min(vectors.shape) <= 0:
        raise ValueError("Vectors must be a nonempty two-dimensional .npy array")
    if vectors.dtype != np.float32 or not np.isfinite(vectors).all():
        raise ValueError("Vectors must contain finite float32 values")
    return np.ascontiguousarray(vectors)


def load_ids(path, count):
    ids = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(ids, list) or len(ids) != count:
        raise ValueError("Business IDs must be a JSON list with one entry per vector")
    if any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != count:
        raise ValueError("Business IDs must be unique, nonempty strings")
    return ids


def configure_spann(index, output):
    settings = {
        "Base": {
            "IndexAlgoType": "BKT", "DistCalcMethod": "L2",
            "IndexDirectory": str(output / "index"),
        },
        "SelectHead": {"isExecute": "true", "NumberOfThreads": "1", "Ratio": "0.25"},
        "BuildHead": {"isExecute": "true", "NumberOfThreads": "1", "NeighborhoodSize": "16"},
        "BuildSSDIndex": {
            "isExecute": "true", "BuildSsdIndex": "true", "Storage": "STATIC",
            "NumberOfThreads": "1", "SearchThreadNum": "1", "IOThreadsPerHandler": "1",
            "InternalResultNum": "32", "SearchInternalResultNum": "32",
            "PostingPageLimit": "4", "SearchPostingPageLimit": "4",
            "ReplicaCount": "2", "ExcludeHead": "false", "Layers": "1", "TmpDir": str(output),
        },
    }
    for section, parameters in settings.items():
        for name, value in parameters.items():
            index.SetBuildParam(name, value, section)


def run(args):
    output = args.index.resolve()
    if args.k <= 0:
        raise ValueError("--k must be positive")
    if args.load and (args.vectors or args.ids):
        raise ValueError("--load cannot be combined with --vectors or --ids")
    if args.ids and not args.vectors:
        raise ValueError("--ids requires --vectors")
    demo = not args.load and args.vectors is None
    if args.load:
        if args.queries is None:
            raise ValueError("--load requires --queries")
        vectors = load_vectors(output / "vectors.npy")
        ids = load_ids(output / "ids.json", len(vectors))
        index = SPTAG.AnnIndex.Load(str(output / "index"))
        if not index.ReadyToServe():
            raise RuntimeError("Loading the saved index failed")
    else:
        if output.exists():
            raise FileExistsError("Output exists; choose a new --index directory or use --load")
        if demo:
            vectors = np.arange(128, dtype=np.float32)[:, None] + np.arange(8, dtype=np.float32)[None, :] / 8
        else:
            vectors = load_vectors(args.vectors)
        ids = load_ids(args.ids, len(vectors)) if args.ids else [
            "document-{}".format(row) for row in range(len(vectors))
        ]
        index = SPTAG.AnnIndex(args.algorithm, "Float", vectors.shape[1])
        if args.algorithm == "BKT":
            index.SetBuildParam("DistCalcMethod", "L2", "Index")
            index.SetBuildParam("NumberOfThreads", "1", "Index")
    if args.k > len(vectors):
        raise ValueError("--k must not exceed the number of indexed vectors")
    if args.queries:
        queries = load_vectors(args.queries)
    else:
        queries = np.ascontiguousarray(vectors[[0, 31, 63]] if demo else vectors[:1])
    if queries.shape[1] != vectors.shape[1]:
        raise ValueError("Query and index dimensions must match")
    if not args.load:
        if args.algorithm == "SPANN":
            output.mkdir(parents=True)
            configure_spann(index, output)
        if not index.Build(vectors, len(vectors), False):
            raise RuntimeError("Building the index failed")
        if args.algorithm == "BKT":
            output.mkdir(parents=True)
        if not index.Save(str(output / "index")):
            raise RuntimeError("Saving the index failed")
        np.save(output / "vectors.npy", vectors, allow_pickle=False)
        np.save(output / "queries.npy", queries, allow_pickle=False)
        (output / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
        index = SPTAG.AnnIndex.Load(str(output / "index"))
        if not index.ReadyToServe():
            raise RuntimeError("Reloading the saved index failed")
    for query_number, query in enumerate(queries):
        row_ids, distances, metadata, relaxed = index.Search(query, args.k)
        if not all(len(items) == args.k for items in (row_ids, distances, metadata, relaxed)):
            raise RuntimeError("Unexpected Search result shape")
        matches = []
        for row, distance in zip(row_ids, distances):
            if not 0 <= row < len(ids) or not np.isfinite(distance):
                raise RuntimeError("Search returned an invalid result")
            matches.append({"id": ids[row], "row_id": row, "distance": distance})
        if demo and args.queries is None:
            expected = (0, 31, 63)[query_number]
            if row_ids[0] != expected or distances[0] != 0:
                raise RuntimeError("Exact self-match check failed")
        print(json.dumps({"query": query_number, "neighbors": matches}))
    print("Python SDK example passed: {} queries; artifacts retained at {}".format(len(queries), output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="New artifact directory, or saved directory with --load")
    parser.add_argument("--vectors", type=Path, help="Your float32 (count, dimension) .npy array")
    parser.add_argument("--ids", type=Path, help="JSON list of business IDs, in vector row order")
    parser.add_argument("--queries", type=Path, help="Your float32 (query_count, dimension) .npy array")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--algorithm", choices=("BKT", "SPANN"), default="BKT",
                        help="Build an in-memory BKT or disk-backed STATIC SPANN index; --load uses saved settings")
    parser.add_argument("--load", action="store_true", help="Load without rebuilding; requires --queries")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
