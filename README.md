# SPTAG: vector search SDK for C++ and Python

[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

SPTAG (Space Partition Tree And Graph) provides approximate nearest-neighbor
search inside your application. Build an index from your vectors, retrieve
Top-K neighbors, and save and reload the index using C++ or Python. The core
combines partition trees with neighborhood graphs and was originally released
by Microsoft Research and Microsoft Bing.

**Start with:** [SDK quick start](docs/QuickStart.md) |
[API and data contracts](docs/SDK.md) |
[platform and feature status](docs/SupportMatrix.md).

SPTAG indexes vectors you supply; it does not generate embeddings or provide a
managed vector database. Start with in-memory BKT or KDT. Disk-based, dynamic,
and distributed capabilities have separate integration and validation requirements.

<p align="center">
  <img src="docs/img/sptag.png" alt="SPTAG tree and graph search architecture" width="500"/>
</p>

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

The example checks three exact self-matches and **keeps** its index, input
vectors, queries, and business-ID mapping. Reload it in another process:

```bash
python3 Examples/python/quickstart.py --index quickstart-output/python \
  --load --queries quickstart-output/python/queries.npy
```

Use a new output directory for another build. Next, follow the
[SDK quick start](docs/QuickStart.md) to use your own vectors and IDs or embed
the C++ library. Node.js is not required for either SDK example.

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

## Capabilities and support

The developer baseline covers in-memory BKT/KDT build, search, and persistence.
The [CI workflow](.github/workflows/ci.yml) checks the public SDK paths as well
as selected core tests. See [recorded validation](docs/Validation.md) and the
[support matrix](docs/SupportMatrix.md) for exact environments and limits;
this development snapshot is not a blanket production-readiness guarantee.

| Additional capability | Entry point and status |
| --- | --- |
| SPANN disk-based search | [Advanced configuration](docs/GettingStart.md); separately validate storage and scale |
| SPFresh incremental updates | [Architecture](docs/Architecture.md); dynamic visibility has a [known correctness blocker](docs/SupportMatrix.md#known-validation-limits) |
| TiKV distributed postings | [Experimental integration](docs/TiKV-Integration.md) |
| RaBitQ quantization | [Opt-in integration](docs/RaBitQ_Global_Quantizer.md) |
| SPIRE distributed construction | [Multi-host workflow](evaluation/distributed/Readme.md), outside the SDK baseline |

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
