# Documentation

Embed vector search in your C++ or Python application. Start with the SDK
workflow below; optional backends and research experiments have separate
requirements and are not installation prerequisites.

## Start here

1. [Install the developer baseline on Linux](LinuxInstallation.md)
   (Ubuntu 24.04, including WSL2); [native Windows status](WindowsInstallation.md).
2. [Python and C++ quick start, including your own vectors and business IDs](QuickStart.md).
3. [SDK API, data, error, concurrency, and compatibility contracts](SDK.md).
4. [Platform and feature status](SupportMatrix.md) and [build options](BuildOptions.md).

## User guides

- [CLI quick start](CLIQuickStart.md).
- [Advanced command-line and SPANN examples](GettingStart.md).
- [Parameter reference](Parameters.md).
- [TiKV posting storage](TiKV-Integration.md).
- [RaBitQ training and search](RaBitQ_Global_Quantizer.md).
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
- [Benchmark methodology and reproduction](../BENCHMARK_GUIDE.md).
- [Evaluation and report index](../evaluation/README.md).
- [Historical 1M results](../BENCHMARK_RESULTS_1M.md).
- [Historical 10M results](../BENCHMARK_RESULTS_10M.md).
- [SPIRE distributed construction research workflow](../evaluation/distributed/Readme.md).
- [SPFresh paper artifact evaluation](../Script_AE/README.md).

Historical documents may reference old binaries, private host layouts, or
destructive storage preparation. They are retained as research records and
must not be interpreted as current, portable installation instructions.
