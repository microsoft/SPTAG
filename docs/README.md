# Documentation

This index separates supported entry points, optional integrations, developer
contracts, and historical experiments. Start with the baseline; do not use a
paper's machine-specific setup as an installation guide.

## Start here

1. [Platform and feature status](SupportMatrix.md).
2. [Linux installation](LinuxInstallation.md) or
   [Windows installation](WindowsInstallation.md).
3. [Small end-to-end example](QuickStart.md).
4. [Build options and dependencies](BuildOptions.md).

## User guides

- [Command-line and SPANN examples](GettingStart.md).
- [Parameter reference](Parameters.md).
- [TiKV posting storage](TiKV-Integration.md).
- [RaBitQ training and search](RaBitQ_Global_Quantizer.md).
- [Python tutorial](Tutorial.ipynb) and
  [notebook examples](examples/QuickstartGuide.ipynb).
- [OPQ tools](../Tools/OPQ/README.md).
- [NNI parameter tuning](../Tools/nni-auto-tune/README.md).

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
