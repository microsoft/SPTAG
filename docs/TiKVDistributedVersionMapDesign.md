# TiKV Distributed Version Map Design

This document describes the current TiKV-backed distributed version map used by SPFresh/SPANN when `Storage=TIKVIO` and `DistributedVersionMap=true`.

## Goals

- Keep delete/version metadata outside local process memory so multiple workers can observe a shared version map.
- Avoid per-vector TiKV reads during TiKV posting search by checking versions in a post-heap batch step.
- Support multi-layer SPANN search while allowing intermediate layers to skip version-map reads.
- Preserve local version-map behavior when TiKV distributed version map is disabled.

## Configuration

The relevant options are defined under the SSD/SPANN parameter set:

| Option | Default | Meaning |
| --- | ---: | --- |
| `DistributedVersionMap` | `true` | Use `TiKVVersionMap` when `Storage=TIKVIO`; otherwise use local in-memory `LocalVersionMap`. |
| `SearchCheckVersionMapOnlyLayer0` | `false` | When enabled with TiKV distributed VM, search checks the version map only on the current search target layer. For normal user search (`p_tolayer=0`), this means layer 0 only. |
| `VersionChunkSize` | `4096` | Number of VIDs stored in one TiKV version chunk. |
| `VersionCacheTTLMs` | `0` | Cache-entry freshness TTL in milliseconds. `0` means cached chunks do not expire. |
| `VersionCacheMaxChunks` | `10000` | Max chunks in the local LRU cache. `<= 0` disables cache. |
| `AsyncRpcMaxInflight` | `0` | TiKV async RPC in-flight limit. This is owned by TiKV IO, not the version map itself. |

Current benchmark config enables:

```ini
Storage=TIKVIO
DistributedVersionMap=true
SearchCheckVersionMapOnlyLayer0=true
VersionCacheMaxChunks=100000
VersionCacheTTLMs=0
AsyncRpcMaxInflight=512
```

## Key Schema

`TiKVVersionMap` stores one logical version map per SPANN layer. Keys are plain strings before the TiKV IO key-prefix wrapper is applied.

| Logical key | Value | Description |
| --- | --- | --- |
| `vc:{layer}` | `SizeType` bytes | Current vector count for this layer. |
| `v:{layer}:{chunkId}` | `uint8_t[VersionChunkSize]` | Version bytes for VIDs in this chunk. |

Chunk mapping:

```text
chunkId = VID / VersionChunkSize
offset  = VID % VersionChunkSize
```

Version byte meanings:

| Byte | Meaning |
| --- | --- |
| `0xfe` | Deleted or missing/unavailable. |
| `0xff` | Uninitialized live slot, matching local `VersionLabel` behavior. |
| `0x00..0x7f` | Active version values used by reassign/update logic. |

Missing chunks, short chunks, invalid VIDs, or failed reads are treated as `0xfe` during read paths. This is conservative: unknown version state is filtered as deleted.

## Object Wiring

Each `ExtraDynamicSearcher` owns an `IVersionMap`:

```text
if Storage == TIKVIO and DistributedVersionMap:
    m_versionMap = TiKVVersionMap(db, layer, VersionChunkSize, VersionCacheMaxChunks)
else:
    m_versionMap = LocalVersionMap()
```

For TiKV mode, the version map uses the same `KeyValueIO` instance as posting storage. This means the same TiKV key prefix configured by `TiKVKeyPrefix` is applied below the logical version-map keys.

## Initialization

`BuildIndex()` initializes the version map after head selection and posting-cut preparation:

```text
m_versionMap->Initialize(vectorSize, blockSize, capacity, &localToGlobal)
```

Layer initialization behavior:

- Layer 0 or no `globalIDs`: initialize all chunks as alive (`0x00`).
- Non-leaf layers with `globalIDs`: initialize all chunks as deleted (`0xfe`), then mark VIDs present in `globalIDs` as alive (`0x00`).

The count key `vc:{layer}` is saved after initialization. TiKV-backed `Save()` and `Load()` do not write/read local version-map files; the authoritative state is already in TiKV. `Load()` reads `vc:{layer}` and scans chunks to recompute the delete count.

## Write Paths

### Add Capacity

`AddBatch(num, deleted)` extends the logical vector count and creates any new chunks needed.

- `deleted=false`: new chunks are initialized with `0xff`.
- `deleted=true`: the new VID range is filled with `0xfe`, and `m_deleted` is incremented.

The count key is saved after the extension.

### Delete

`Delete(VID)` writes `0xfe` at the VID's chunk offset. It uses a striped per-chunk mutex to serialize read-modify-write operations in this process.

### Set Version

`SetVersion(VID, version)` writes the requested byte and updates `m_deleted` when the value crosses the deleted/non-deleted boundary.

### Increment Version

`IncVersion(VID, newVersion, expectedOld)` reads the chunk, validates the current byte, increments modulo `0x80`, and writes the chunk back.

When `expectedOld != 0xff`, the operation uses compare-style semantics:

- Compute `target = (expectedOld + 1) & 0x7f`.
- If the current byte is already `target`, return success and report `target`. This makes the operation idempotent when another path already performed the same version increment.
- If the current byte is `expectedOld`, write `target` and return success.
- If the current byte is neither `expectedOld` nor `target`, return failure. This prevents a stale caller from overwriting a different/newer version with its own target value.

When `expectedOld == 0xff`, the operation increments the current non-deleted value without a compare expectation. Current SPANN reassign paths pass the observed old version as `expectedOld`, so they use the compare-style path above.

Current limitation: this is a best-effort read-modify-write. It is protected by local process locks, but it is not a distributed CAS across multiple processes. The code notes this should use TiKV RawCAS if/when available through the dependency stack.

## Read Paths

### Single VID

`Deleted(VID)` and `GetVersion(VID)` read the owning chunk and return one byte. They support two read policies:

- `UseCache`: check/fill the local chunk LRU cache.
- `BypassCacheNoFill`: read TiKV directly and do not populate the cache.

Search-created workspaces currently use `BypassCacheNoFill` so query traffic does not pollute the cache with one-off chunks.

### Batch VID Lookup

`BatchGetVersions(vids, versions, policy)` groups VIDs by chunk, resolves cached chunks first, and issues one TiKV `MultiGet` for missing chunks.

Resolution phases:

1. Group VIDs by `chunkId`.
2. Read cache hits under a shared lock.
3. Batch fetch missing chunks from TiKV.
4. Optionally insert fetched chunks into the LRU cache.
5. Fill one output byte per input VID.

If a chunk is missing or too short, affected VIDs are returned as `0xfe`.

## Search Semantics

SPANN disk search walks layers from high to low:

```text
for layer = maxLayer down to p_tolayer:
    isTargetLayer = (layer == p_tolayer)
    searcher[layer].SearchIndex(..., p_checkVersionMap=isTargetLayer)
```

`p_tolayer` is the final target layer for the current search, not the current layer being expanded.

Examples:

| Search | Layers visited | Target layer | Version-map check with `SearchCheckVersionMapOnlyLayer0=true` |
| --- | --- | --- | --- |
| Normal user search, `p_tolayer=0` | layer 1, then layer 0 | layer 0 | Skip layer 1; check layer 0. |
| Head/maintenance search, `p_tolayer=1` | layer 1 only | layer 1 | Check layer 1. |

The option name says `OnlyLayer0` because the original target was normal user search. The implemented semantics are more precise: when enabled, TiKV distributed VM is checked only on the target layer of the current search. This avoids reading layer 1 VM for normal searches while preserving delete filtering for searches whose target layer is 1.

### Intermediate Layers

Intermediate layers (`layer > p_tolayer`) are used only to expand posting IDs for the next lower layer. They do not add final results and do not call `ContainSample()` for target-layer delete filtering.

### Target Layer

At the target layer:

- Existing candidate heads are filtered through `targetLayerContains(VID)` before being added to final results.
- `SearchIndex()` performs the TiKV post-heap `BatchGetVersions()` check when `p_checkVersionMap=true`.

### TiKV Posting Search

For TiKV posting search, per-entry version checks inside the posting scan are skipped. Search first reads postings and computes distances, then checks only the top candidate heap via `BatchGetVersions()`.

Simplified flow:

```text
read postings from TiKV
for each vector in postings:
    dedup
    compute distance
    add candidate to heap

if TiKV and p_checkVersionMap:
    collect current result buffer VIDs
    BatchGetVersions(candidateVIDs)
    mark 0xfe candidates invalid
    SortResult()
```

For non-TiKV storage, the local version map still performs per-vector `Deleted(VID)` filtering during posting scan. The TiKV-only search policy does not affect local version-map correctness.

## Coprocessor Search

When TiKV coprocessor search is enabled, posting distance computation is pushed into TiKV and returns top candidates. The same `p_checkVersionMap` policy is applied afterward:

```text
CoprocessorSearch(postingIDs, query)
dedup returned candidates
if p_checkVersionMap:
    BatchGetVersions(top candidates)
    filter deleted candidates
```

## Cache Design

The version map has a per-process LRU chunk cache:

- Front of the list is most recently inserted/updated.
- `VersionCacheMaxChunks <= 0` disables the cache.
- `VersionCacheTTLMs <= 0` keeps the current pure-LRU behavior: cached chunks do not expire by age.
- `VersionCacheTTLMs > 0` makes cache hits valid only while `now - refreshTime < VersionCacheTTLMs`.
- Fresh cache hits use shared locking and do not reorder entries.
- Cache misses and expired entries fetch from TiKV and insert under an exclusive lock.
- Writes update the cache after a successful TiKV `Put`.
- `BypassCacheNoFill` disables both lookup and fill for that read.

Single-chunk reads use a striped refresh mutex so multiple threads do not all refresh the same expired/missing chunk at once. Batched reads group miss/expired chunks and refresh them with one TiKV `MultiGet`.

This cache reduces repeated chunk reads but is not a distributed coherence mechanism. Other processes can update TiKV without invalidating this process's cached chunks before TTL expiry. Search paths can bypass cache when freshness is preferred.

## Failure And Missing-Key Behavior

Read-side behavior is fail-closed:

- Missing count key on load sets count to `0` and logs a warning.
- Missing chunk data returns `0xfe` for affected VIDs.
- Failed `MultiGet` logs a warning and unresolved chunks return `0xfe`.
- Short chunk values return `0xfe` for out-of-range offsets.

Write-side behavior:

- Failed chunk writes are logged and return failure to the caller.
- Delete/set/inc operations do not silently mark a write successful if TiKV write failed.

## Current Limitations

- `IncVersion()` is not a distributed atomic CAS. Concurrent writers from different processes can still race at chunk granularity.
- Delete count is maintained locally during writes and recomputed by scanning chunks on load; it is not stored as a separate authoritative TiKV counter.
- The LRU cache is local to one process and has no cross-process invalidation.
- `VersionCacheTTLMs` bounds cache staleness only for code paths that use `UseCache`; `BypassCacheNoFill` still reads TiKV directly.
- `SearchCheckVersionMapOnlyLayer0` is historically named; with the current implementation it means target-layer-only checking for TiKV distributed VM.
- Missing version chunks are treated as deleted. This favors correctness over recall when TiKV data is incomplete.

## Relevant Code

- `AnnService/inc/Core/Common/TiKVVersionMap.h`: TiKV-backed `IVersionMap` implementation.
- `AnnService/inc/Core/SPANN/ExtraDynamicSearcher.h`: chooses TiKV/local version map, owns add/delete/search version checks.
- `AnnService/src/Core/SPANN/SPANNIndex.cpp`: layer traversal and target-layer search policy.
- `AnnService/inc/Core/SPANN/ParameterDefinitionList.h`: public config definitions.
- `AnnService/inc/Core/SPANN/Options.h`: config storage.