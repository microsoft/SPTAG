# Architecture and storage boundaries

[Documentation index](README.md)

## Index families

- **BKT/KDT:** in-memory vector indexes combining partition trees with a
  neighborhood graph. Build and search through `VectorIndex` or the CLI.
- **SPANN:** a head index chooses posting lists whose data is read from
  configured storage. Head-index and posting settings are separate sections.
- **SPFresh:** incremental workflows around dynamic SPANN storage, including
  updates, splitting, reassigning vectors, and version tracking.

The [quick start](QuickStart.md) exercises only the BKT path. It does not certify
dynamic storage consistency or multi-machine behavior.

## Posting backends

```text
SPANN index
  head selection / head graph / local index metadata
    -> posting search and update
      -> local static postings
      -> dynamic KeyValueIO interface
          -> FILEIO
          -> RocksDB
          -> SPDK
          -> TiKV RawKV
```

Storage selection is a runtime index setting, but optional backends must also
be compiled in. Setting `Storage=TIKVIO` cannot enable a binary built with
`TIKV=OFF`.

TiKV stores postings and backend-specific state. The head index and other
index metadata still include local files. A replicated TiKV cluster alone
does not make the complete SPTAG index recoverable or highly available.
Back up and coordinate those local artifacts as well.

## Configuration and formats

SPANN configurations use `Base`, `SelectHead`, `BuildHead`, and `BuildSSDIndex`
sections. The benchmark's `Benchmark` section configures the harness, not a
general pass-through for posting-backend parameters. See
[the benchmark guide](../BENCHMARK_GUIDE.md) for the distinction.

Use matching ID widths (`LARGEVID`), value types, dimensions, distance
settings, quantizer models, and encoded data. Copying one configuration file
does not make incompatible persisted artifacts interchangeable.

## Developer contracts

- [Local version-map behavior](development/LocalVersionMap.md).
- [RaBitQ model and stored-code behavior](development/RaBitQ.md).
- [TiKV version-map design](TiKVDistributedVersionMapDesign.md), a design record
  whose individual proposals must be checked against the current implementation.

## Research versus serving

The repository's distributed index-construction research is associated with
**SPIRE**, [Scalable Distributed Vector Search via Accuracy Preserving Index Construction](https://arxiv.org/abs/2512.17264).
The paper studies partition granularity and accuracy-preserving recursive
multi-level construction. See the
[distributed workflow](../evaluation/distributed/Readme.md) and
[citation](References.md#spire); the baseline smoke test does not validate the
paper's multi-node evaluation.

Benchmark/test programs may create derived vector files, truth files, cloned
indexes, and JSON outputs. Run them in dedicated output directories. Historical
experiments often assume a particular machine and storage layout; they are not
deployment defaults. See the [evaluation index](../evaluation/README.md).
