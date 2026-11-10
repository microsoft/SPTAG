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

### Experimental dynamic top-layer growth

Unquantized FILEIO and TiKV indexes can opt into `EnableDynamicLayers=true` in
`BuildSSDIndex`. When the estimated head footprint reaches
`HeadMemoryHighWatermark`, a dedicated coordinator builds a smaller head and
a new posting layer containing the old head's live vectors. Existing posting
layers and global VIDs are unchanged. Only growth is implemented; automatic
shrinking and cooldown are not implemented.

| Setting | Default | Meaning |
| --- | --- | --- |
| `EnableDynamicLayers` | `false` | Enable automatic growth and the C++ `GrowTopLayer()` entry point; configure before build/load |
| `HeadMemoryHighWatermark` | `0` | Trigger threshold in estimated bytes |
| `HeadMemoryTarget` | `0` | Maximum estimated bytes for the replacement head; must be positive and below the trigger |
| `MaxLayers` | `16` | Maximum posting-layer count, including layer 0 |
| `LayerResizeTimeoutSeconds` | `300` | Timeout for write admission, maintenance draining and query/iterator barriers; not a cancellation deadline for head construction |
| `LayerStoragePrefixes` | empty | Persisted per-layer TiKV prefixes or FILEIO relative mapping-file stems, separated by `|`; generated by growth, not a tuning parameter |

Malformed new controls are rejected rather than silently using their defaults;
memory budgets are nonnegative decimal byte counts, without unit suffixes.
Reaching the layer/key limit above the high watermark reports a resize error.
An already-grown index can be loaded with automatic growth disabled; its saved
topology and query/write synchronization remain in use, and memory thresholds
are not required while growth is disabled.

`GetHeadMemoryUsage()` sums the head's reported buffers, ID mapping and an
estimated reverse-map overhead. This is not RSS or an exact allocator/capacity
measurement. Reserve memory for the old head, exported vectors and candidate
construction; a trigger at the process's hard memory limit is too late.
Candidate construction retries with fewer representatives until it meets the
target. `SelectHead.Ratio` guides the replacement representative fraction
(for example, 0.1 targets about 10% of the old live head); the memory budget
may require fewer representatives. It fails rather than truncating a routing vector when posting capacity
or the minimum head size cannot satisfy the configured budgets.
BKT selection thresholds are recomputed from each candidate's requested count,
rather than inherited from the old head; reducing the budget must also change
the selection granularity. Candidate retries check the resize deadline between
builds, but do not interrupt a build already in progress.
If all RNG-selected routing postings are full, construction searches progressively
more nearby heads for an available posting instead of discarding the routing
vector. Insufficient total capacity still fails without publishing the candidate.
An automatic resize failure is latched: further insertions do not repeatedly
rebuild the same failed candidate. After correcting the configuration, explicitly
retry `GrowTopLayer()`; a successful manual retry clears the error status.
FILEIO reserves workspace IDs and backend I/O contexts for `MaxLayers` when
dynamic growth is enabled. Growth does not issue new IDs beyond that reservation:
modulo-aliasing a new ID onto an old backend's active context can corrupt concurrent
I/O. Cached thread-local workspaces rebind their ID lease when changing indexes.
The [FILEIO SIFT experiment runner](../benchmark-output/sift10m-fileio-dynamic-5m-1m-20261109/run_experiment.py)
limits each attempt to one hour and preserves separate logs and measurements.
Its configuration uses `Rebuild=false` and `Resume=-1` to resume from the latest
completed checkpoint instead of discarding completed insertion batches.
The [recorded SIFT10M validation](../benchmark-output/sift10m-fileio-dynamic-5m-1m-20261109/validation.json)
includes the real growth event, per-prefix measurements, hourly checkpoint
recovery, and a separate reload/query check of the final 10M checkpoint.
The [real TiKV functional test runner](../benchmark-output/tikv-dynamic-growth-20261110/run_tests.py)
starts an isolated three-PD/three-TiKV cluster using the pinned reproduction
images. After building `SPTAGTest` with TiKV enabled, run it with
`python3 benchmark-output/tikv-dynamic-growth-20261110/run_tests.py`.
It explicitly enables `TiKVDynamicLayerIntegrationTest/RealTiKVGrowthAndReload`, which is
disabled in ordinary test runs. Both local and distributed version maps exercise
automatic growth, concurrent queries, subsequent insertion/deletion and reload.
The workload has a one-hour deadline; owned containers are stopped afterward,
while logs and data are retained. This small functional test requires 250 GiB
free and does not replace the 5M-to-10M experiment or its 500 GiB startup guard.

Writes pause while the coordinator drains maintenance and builds the new
top. Queries continue against the frozen old topology during construction;
search-triggered merges are suppressed. Publication briefly takes an exclusive
topology lock. A reader-admission gate lets waiting topology and write barriers
drain active readers even under continuous query/insert traffic.
Iterators pin the old topology until `Close()`; an active iterator
can cause a timed-out resize, but is not invalidated. Use an iterator on its
creating thread and close it promptly. Do not mutate the objects returned by
`GetMemoryIndex()`, `GetDiskIndex()` or `GetOptions()` concurrently with growth.

Each candidate uses a generation directory and either an isolated, never-reused
TiKV prefix or separate FILEIO mapping/posting/block-pool files. Shared lower
FILEIO storage is retained unchanged by the topology transition; new layers
have their own backend even when `ShareDB=true`. Mapping, posting and block-pool
files are checkpointed and synchronized before publication.
After writing and verifying postings and persisting the head,
mapping and version state, growth atomically replaces `indexloader.ini` with
the new topology configuration and publishes the in-memory state. The ordinary
folder loader reads this manifest; old indexes without per-layer prefixes
retain their original storage layout. Normal saves are serialized with growth.
Normal saves also hold the topology write lock, temporarily blocking queries;
this permits reentrant destination-directory updates when saving a full copy.
Do not copy only the head files: preserve the full local index directory and
all TiKV namespaces referenced by its configuration.

Failures before manifest publication leave the old topology active. Their local generation
directory is removed; partially written TiKV namespaces are logged and retained
for explicit administrative cleanup. A directory-sync error after publication
is reported without rolling back the already-published topology; inspect the
actual layer count as well as the error status. Inspect `GetLayerResizeStatus()` and
`[DYNAMIC-LAYER]` logs; `AllFinished()` indicates completion, not success.
The benchmark accepts these controls in `BuildSSDIndex` and adds
`postingLayerCount`, `headMemoryEstimateBytes` and `layerResizeErrorCode` to
insertion-batch output when growth is enabled. It also reports head counts and
continues background queries through insert draining/resizing when
`NumSearchDuringInsertThreads` is positive; query failures fail the run.

This first implementation requires Linux, FILEIO or compiled TiKV support, no quantizer,
no WAL/recovery mode, and no centering transform. Both local and distributed
version maps are supported. Posting keys must fit `SizeType`:
`MaxID * logicalLayer + globalVID`. Growth reports overflow explicitly; use
a suitable `MaxID` and VID width rather than assuming 16 layers always fit.
The coordinator is process-local: do not run another writer against the same
index/prefix during a resize. Atomic topology publication does not add
multi-process writer coordination or crash-safe ordinary ingestion.

Focused tests use an in-memory `KeyValueIO` implementation to exercise routing,
thresholds, concurrent queries, failure rollback and reload. They do not certify
real-cluster RPC behavior, crash/power-loss durability, or recall on large data.
Validate those separately before enabling this experimental feature in serving.
For TiKV validation builds, CMake's `KVPROTO_GEN_DIR` can point to externally
generated kvproto stubs compatible with the installed protobuf/gRPC versions;
the default remains the repository's generated-stub directory.

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
