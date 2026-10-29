# Architecture and storage boundaries

[Documentation index](README.md)

SPTAG is a vector search engine. C++, Python, and the CLI are access paths to
its components, not separate engines. The development direction combines
large-scale construction, disk-backed search, incremental maintenance, and
compressed vectors. A small tested SDK profile does not define the full
capability set or certify every component combination.

## Index families

- **BKT/KDT:** in-memory vector indexes combining partition trees with a
  neighborhood graph. Build and search through `VectorIndex` or the CLI.
- **SPANN:** a head index chooses posting lists whose data is read from
  configured storage. Head-index and posting settings are separate sections.
- **SPFresh:** incremental workflows around dynamic SPANN storage, including
  updates, splitting, reassigning vectors, and version tracking.

SPFresh describes maintenance around SPANN, not another `IndexAlgoType`.
Likewise, TiKV is a storage backend and RaBitQ is a quantizer, not replacements
for the index API. The [quick start](QuickStart.md) exercises BKT and a small
single-layer STATIC SPANN path; it does not certify dynamic consistency or
multi-machine behavior.

## Engine data flow

```text
Your embeddings and application IDs
  -> C++ VectorIndex / Python AnnIndex / command-line tools
     -> BKT or KDT in-memory tree and graph
     -> SPANN head routing and posting layers
        -> Top-K search or iterative results
        -> STATIC posting files or dynamic KeyValueIO storage

Construction: head selection / parallel tree and graph build / posting assignment
Maintenance: SPFresh insert/delete / split/merge/reassign / version checks
Compression: quantizer training / encoded vectors / matching distance computation
```

The head tree/graph provides routing, not the entire SPANN corpus. Posting
storage holds candidate vectors and associated state. Compression affects
both stored data and distance calculations, so it must be configured consistently
across construction and queries.

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

## Dynamic index maintenance

SPFresh maintains changing SPANN indexes through insertion/deletion and
background split, merge, and reassignment work. Version maps distinguish
current vector occurrences from stale copies in postings. Local version-map
conditional updates and TiKV version state are correctness mechanisms, not
just performance caches.

Start with a working static SPANN index, then select a dynamic backend and
validate the intended update/search workload with the
[benchmark harness](Benchmarking.md). Check recall during ingestion,
stale/deleted visibility, save/reload, and restart behavior; compilation alone
does not establish these contracts.
The [known visibility regression](SupportMatrix.md#known-validation-limits)
blocks stable-use claims for affected paths. Its existence does not make
incremental maintenance an incidental feature, and a passing static test does
not remove that blocker.

## Construction and multi-layer indexes

The construction path selects heads, builds their tree/graph, and assigns
vectors to postings. Parallel BKT/TPT work and lower-memory construction
options address build cost; they are distinct from distributing runtime storage.

SPANN also has `Layers` and `ResumeLayer` settings and multi-layer search/
iterator implementation. These require consistent local/global ID mappings,
per-layer artifacts, and version behavior. The one-layer tutorial does not
establish a multi-layer correctness or recovery guarantee.

The [distributed construction workflow](../evaluation/distributed/Readme.md)
adds data partitioning, MPI clustering, shuffle, parallel local builds, and
head merging. This work is associated with **SPIRE**,
[Scalable Distributed Vector Search via Accuracy Preserving Index Construction](https://arxiv.org/abs/2512.17264),
which studies accuracy-preserving recursive, multi-level construction.
See the [citation](References.md#spire).
SPIRE is a research/construction lineage here, not a newly added SDK algorithm enum.
Those machine-specific recipes need multi-host validation; they are not a
complete distributed-serving deployment.

## Quantization and query modes

PQ/OPQ and RaBitQ address the vector representation and distance-computation
cost, not the posting backend. Quantizers, encoded vectors, and index settings
must agree; build-time stored-code comparisons and query-time ADC are not
interchangeable. The [RaBitQ guide](RaBitQ_Global_Quantizer.md) covers manual
training and native automatic bit-width selection, with separate accuracy and
model-compatibility requirements.

Top-K search returns a requested set of candidates. An iterator requests
successive batches and exposes error/relaxed-monotonicity state; do not interpret
its flags as confidence scores or assume globally sorted/exhaustive results
without checking that algorithm's contract. See the
[SDK entry points](SDK.md#spann-quantizers-and-iterative-search).

## Configuration and formats

SPANN configurations use `Base`, `SelectHead`, `BuildHead`, and `BuildSSDIndex`
sections. The benchmark's `Benchmark` section configures the harness, not a
general pass-through for posting-backend parameters. See
[the benchmark guide](Benchmarking.md) for the distinction.

Use matching ID widths (`LARGEVID`), value types, dimensions, distance
settings, quantizer models, and encoded data. Copying one configuration file
does not make incompatible persisted artifacts interchangeable.

## Developer contracts

- [Local version-map behavior](development/LocalVersionMap.md).
- [RaBitQ model and stored-code behavior](development/RaBitQ.md).
- [TiKV version-map design](TiKVDistributedVersionMapDesign.md), a design record
  whose individual proposals must be checked against the current implementation.

## Research versus serving

Benchmark/test programs may create derived vector files, truth files, cloned
indexes, and JSON outputs. Run them in dedicated output directories. Historical
experiments often assume a particular machine and storage layout; they are not
deployment defaults. See the [evaluation index](../evaluation/README.md).
