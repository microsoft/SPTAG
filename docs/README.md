# Documentation

Use SPTAG as a vector search engine through C++, Python, or command-line tools.
The tutorials start small; they are an entry point to disk indexing, dynamic
maintenance, quantization, and scalable construction, not a restriction to
in-memory ANN.

## Start here

1. [Install the developer baseline on Linux](LinuxInstallation.md)
   (Ubuntu 24.04, including WSL2); [native Windows status](WindowsInstallation.md).
2. [Python and C++ quick start, including your own vectors and business IDs](QuickStart.md).
3. [SDK API, data, error, concurrency, and compatibility contracts](SDK.md).
4. [Platform and feature status](SupportMatrix.md) and [build options](BuildOptions.md).

## Choose a workflow

| Goal | Guide | Current boundary |
| --- | --- | --- |
| Learn the API and use your own vectors/IDs | [SDK quick start](QuickStart.md) | Basic C++/Python in-memory path tested |
| Build and reload a disk index | [Small SPANN workflow](QuickStart.md#spann-disk-backed-search), then [SPANN configuration](GettingStart.md) | Single-layer STATIC example tested; large workloads require evaluation |
| Keep an index fresh as data changes | [Dynamic maintenance](Architecture.md#dynamic-index-maintenance), [workload harness](Benchmarking.md) | Affected visibility paths have a known correctness blocker |
| Compress vectors and select a bit width | [RaBitQ and native auto-tuning](RaBitQ_Global_Quantizer.md) | Opt-in pipeline; accuracy must be measured separately |
| Store postings in a cluster | [TiKV integration](TiKV-Integration.md) | Requires a separately validated PD/TiKV deployment |
| Partition and build on multiple hosts | [Construction architecture](Architecture.md#construction-and-multi-layer-indexes), [SPIRE workflow](../evaluation/distributed/Readme.md) | Multi-host validation required |

## User guides

- [CLI quick start](CLIQuickStart.md).
- [Advanced command-line and SPANN examples](GettingStart.md).
- [Parameter reference](Parameters.md).
- [TiKV posting storage](TiKV-Integration.md).
- [RaBitQ training, automatic bit selection, and search](RaBitQ_Global_Quantizer.md).
- Additional [Python tutorial](Tutorial.ipynb) and
  [notebook examples](examples/QuickstartGuide.ipynb), outside the SDK smoke-test contract.
- [OPQ tools](../Tools/OPQ/README.md).
- [Legacy NNI parameter-tuning experiment](../Tools/nni-auto-tune/README.md).

## Development

- [Contributing, testing, and local debugging](../CONTRIBUTING.md).
- [Recorded local validation and limits](Validation.md).
- [Architecture and storage boundaries](Architecture.md).
- [Local version tracking](development/LocalVersionMap.md).
- [RaBitQ implementation and model compatibility](development/RaBitQ.md).
- [TiKV distributed version-map design](TiKVDistributedVersionMapDesign.md)
  (design discussion, not a deployment or support guarantee).
- [Research references and citations](References.md).

## Research, benchmarks, and historical artifacts

- [SPIRE paper and citation](References.md#spire).
- [Benchmark methodology and reproduction](Benchmarking.md).
- [Latest SIFT1M validation: TiKV recall fix and partial update runs](../evaluation/2026-10-23-sift1m/README.md).
- [Evaluation and report index](../evaluation/README.md).
- [Historical 1M observations](../evaluation/historical/TiKV-1M.md).
- [Historical 10M observations](../evaluation/historical/TiKV-10M.md).
- [SPIRE distributed construction research workflow](../evaluation/distributed/Readme.md).
- [SPFresh paper artifact evaluation](../Script_AE/README.md).

Historical documents may reference old binaries, private host layouts, or
destructive storage preparation. They are retained as research records and
must not be interpreted as current, portable installation instructions.
