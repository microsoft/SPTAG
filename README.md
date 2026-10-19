# SPTAG: Scalable Vector Search with Dynamic Indexing

[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

SPTAG (Space Partition Tree And Graph) is a C++ vector search engine with Python
bindings and command-line tools. It combines partition trees and neighborhood
graphs for approximate nearest-neighbor search, with in-memory and disk-backed
indexes, incremental index maintenance, vector compression, and scalable index
construction. It was originally released by Microsoft Research and Microsoft Bing.

**Start with:** [choose a capability](#choose-a-capability) |
[quick start](docs/QuickStart.md) |
[API and data contracts](docs/SDK.md) |
[platform and feature status](docs/SupportMatrix.md).

SPTAG indexes vectors you supply; it does not generate embeddings or provide a
managed vector database. The development line brings together SPANN disk
indexing, SPFresh updates, TiKV posting storage, RaBitQ quantization, and
distributed construction associated with SPIRE. These are different layers of
the engine, not interchangeable index types or equally validated configurations.

<p align="center">
  <img src="docs/img/sptag.png" alt="SPTAG tree and graph search architecture" width="500"/>
</p>

## Choose a capability

| Your workload | Engine capability | Start here | Validation status |
| --- | --- | --- | --- |
| Search vectors in your application's memory | BKT/KDT | [C++ and Python examples](docs/QuickStart.md) | Basic SDK paths tested |
| Search disk-backed postings | SPANN | [Small disk-index example](docs/QuickStart.md#spann-disk-backed-search) | Single-layer STATIC build/search/reload tested |
| Maintain a changing corpus | SPFresh incremental insert/delete and split/reassign | [Dynamic indexing](docs/Architecture.md#dynamic-index-maintenance) | Implemented; affected visibility paths have a known correctness blocker |
| Reduce vector storage cost | PQ/OPQ and RaBitQ; native RaBitQ bit-width tuning | [Quantization and tuning](docs/RaBitQ_Global_Quantizer.md) | Opt-in; accuracy and compatibility require separate validation |
| Put dynamic postings in a storage cluster | TiKV RawKV backend | [TiKV integration](docs/TiKV-Integration.md) | Implemented integration; cluster behavior not certified by baseline tests |
| Scale index construction across machines | Distributed partition/build workflows; SPIRE research lineage | [Construction architecture](docs/Architecture.md#construction-and-multi-layer-indexes) | Multi-host workflow requires dedicated validation |

Capability importance and readiness are separate: core development directions
can still require integration testing or correctness fixes. See the
[support matrix](docs/SupportMatrix.md) for the complete scope.

## Choose your entry point

| You want to | Entry point | Documentation |
| --- | --- | --- |
| Use NumPy vectors in your application | Python `SPTAG.AnnIndex` | [Python quick start](docs/QuickStart.md#python) |
| Embed search in a C++ application | `SPTAG::VectorIndex`, CMake target `SPTAG::SPTAG` | [C++ quick start](docs/QuickStart.md#c) |
| Build or query an index from files | `indexbuilder`, `indexsearcher` | [CLI quick start](docs/CLIQuickStart.md) |

## Install from source

This SDK workflow currently tracks the **`users/qiazh/doc` development branch**,
not a versioned SDK release. The checkout below selects that branch explicitly.
Do not assume a package named `sptag` on a package index contains these changes.
Record `git rev-parse HEAD` when integrating a snapshot.

The validated starting environment is Ubuntu 24.04 x86-64 with GCC 13,
Python 3.12, NumPy 1.26, and an **AVX2-capable CPU**; WSL2 can run this Linux
workflow. Native Windows has a separate [compatibility guide](docs/WindowsInstallation.md).
No external dataset, running service, or special SSD is needed.

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build git swig python3-dev python3-numpy \
  libboost-system-dev libboost-thread-dev libboost-serialization-dev \
  libboost-regex-dev libboost-filesystem-dev libboost-test-dev libtbb-dev libnuma-dev

git clone --branch users/qiazh/doc https://github.com/microsoft/SPTAG.git
cd SPTAG
git submodule update --init ThirdParty/zstd

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DFOLLY=OFF -DTBB=ON -DSPDK=OFF -DROCKSDB=OFF -DTIKV=OFF -DGPU=OFF -DRABITQ=OFF
cmake --build build --target _SPTAG --parallel 2
export PYTHONPATH="$PWD/Release${PYTHONPATH:+:$PYTHONPATH}"
python3 Examples/python/quickstart.py --index quickstart-output/python
```

The first example uses BKT to introduce the API with minimal setup. It checks
three exact self-matches and **keeps** its index, input vectors, queries, and
business-ID mapping. Reload it in another process:

```bash
python3 Examples/python/quickstart.py --index quickstart-output/python \
  --load --queries quickstart-output/python/queries.npy
```

Use a new output directory for another build. Next, follow the
[SDK quick start](docs/QuickStart.md) to use your own vectors and IDs or embed
the C++ library. To exercise disk-backed search with the same binding:

```bash
python3 Examples/python/quickstart.py --algorithm SPANN --index quickstart-output/spann
python3 Examples/python/quickstart.py --index quickstart-output/spann \
  --load --queries quickstart-output/spann/queries.npy
```

This small SPANN example needs no external storage service and does not enable
dynamic updates. Node.js is not required for either SDK path.

## Python API at a glance

After building and setting `PYTHONPATH` above:

```python
import numpy as np
import SPTAG

vectors = np.arange(1024, dtype=np.float32).reshape(128, 8)
index = SPTAG.AnnIndex("BKT", "Float", 8)
index.SetBuildParam("DistCalcMethod", "L2", "Index")
index.SetBuildParam("NumberOfThreads", "1", "Index")
if not index.Build(vectors, len(vectors), False):
    raise RuntimeError("Index build failed")
row_ids, squared_distances, metadata, relaxed = index.Search(vectors[0], 3)
print(row_ids, squared_distances)
```

Returned IDs are vector row numbers, not automatically your application IDs.
See [data, result, error, and persistence contracts](docs/SDK.md) before
integrating. ANN results are approximate; evaluate recall on your own data.

## Validation and compatibility

The developer baseline checks in-memory BKT/KDT and a small disk-backed SPANN
path. It is a starting validation profile, not the boundary of the engine.
The [CI workflow](.github/workflows/ci.yml) checks the public SDK paths as well
as selected core tests. See [recorded validation](docs/Validation.md) and the
[support matrix](docs/SupportMatrix.md) for exact environments and limits;
this development snapshot is not a blanket production-readiness guarantee.

Updates, quantization, iterators, and distributed integrations need their own
correctness, compatibility, and workload tests. In particular, the
[known dynamic-visibility regression](docs/SupportMatrix.md#known-validation-limits)
is not waived by a passing static SDK test.

## Documentation

- [Install and troubleshoot on Linux](docs/LinuxInstallation.md).
- [SDK quick start and your own data](docs/QuickStart.md).
- [SDK API and compatibility contracts](docs/SDK.md).
- [Parameters](docs/Parameters.md) and [build options](docs/BuildOptions.md).
- [Documentation index](docs/README.md), [changelog](CHANGELOG.md), and
  [contributing and tests](CONTRIBUTING.md).
- [Benchmark methodology](BENCHMARK_GUIDE.md) and [evaluation index](evaluation/README.md).

## Research and citation

SPTAG also hosts research associated with SPANN, SPFresh, and **SPIRE**.
These papers explain the underlying techniques; their experimental results
are not SDK release guarantees.

- [SPIRE: Scalable Distributed Vector Search via Accuracy Preserving Index Construction (arXiv, 2025)](https://arxiv.org/abs/2512.17264).
- [SPFresh: Incremental In-Place Update for Billion-Scale Vector Search (SOSP 2023)](https://dl.acm.org/doi/10.1145/3600006.3613166).
- [VBASE: Unifying Online Vector Similarity Search and Relational Queries via Relaxed Monotonicity (OSDI 2023)](https://www.usenix.org/system/files/osdi23-zhang-qianxi_1.pdf).
- [SPANN: Highly-efficient Billion-scale Approximate Nearest Neighbor Search (NeurIPS 2021)](https://proceedings.neurips.cc/paper_files/paper/2021/file/299dc35e747eb77177d9cea10a802da2-Paper.pdf).

Citation entries and earlier references are in [References](docs/References.md).
Experimental reports describe particular runs; they are not performance
guarantees for the current branch.

## Contributing and reporting issues

Read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting changes. Report
reproducible bugs and feature requests through
[GitHub issues](https://github.com/microsoft/SPTAG/issues).
Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md),
not through public issues.

## License

SPTAG is available under the [MIT license](LICENSE). Third-party dependencies
and datasets retain their own licenses; see [dependency notes](docs/BuildOptions.md).
