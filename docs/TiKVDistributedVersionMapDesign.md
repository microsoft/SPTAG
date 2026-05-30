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
| `VersionChunkSize` | `4096` | Legacy no-op for TiKV distributed VM; versions are stored per VID. |
| `AsyncRpcMaxInflight` | `0` | TiKV async RPC in-flight limit. This is owned by TiKV IO, not the version map itself. |

Current benchmark config enables:

```ini
Storage=TIKVIO
DistributedVersionMap=true
SearchCheckVersionMapOnlyLayer0=true
AsyncRpcMaxInflight=512
```

## Key Schema

`TiKVVersionMap` stores one logical version map per SPANN layer. Keys are plain strings before the TiKV IO key-prefix wrapper is applied.

| Logical key | Value | Description |
| --- | --- | --- |
| `vc:{layer}` | `SizeType` bytes | Current vector count for this layer. |
| `v:{layer}:{vid}` | `uint8_t` | Version byte for one VID. `vid` is zero-padded to 10 digits so TiKV regions can be pre-split by VID range. |

Version byte meanings:

| Byte | Meaning |
| --- | --- |
| `0xfe` | Deleted or missing/unavailable. |
| `0xff` | Uninitialized live slot, matching local `VersionLabel` behavior. |
| `0x00..0x7f` | Active version values used by reassign/update logic. |

Missing per-VID keys use a code-defined layer default: layer 0 defaults to alive version `0x00`, while non-leaf layers default to deleted `0xfe`. Invalid VIDs or failed reads are treated as `0xfe` during read paths. This is conservative: unknown version state is filtered as deleted.

## Object Wiring

Each `ExtraDynamicSearcher` owns an `IVersionMap`:

```text
if Storage == TIKVIO and DistributedVersionMap:
    m_versionMap = TiKVVersionMap(db, layer)
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

- Layer 0 or no `globalIDs`: set the default missing-key version to alive (`0x00`).
- Non-leaf layers with `globalIDs`: set the default missing-key version to deleted (`0xfe`), then write one alive (`0x00`) key for each VID present in `globalIDs`.

The count metadata key is saved after initialization. TiKV-backed `Save()` and `Load()` do not write/read local version-map files; the authoritative state is already in TiKV. Different runs should use distinct `TiKVKeyPrefix` values or wipe TiKV before reuse. `GetDeleteCount()` returns 0 for the TiKV-backed version map to avoid expensive distributed scans.

## Write Paths

### Add Capacity

`AddBatch(num, deleted)` extends the logical vector count.

- `deleted=false`: new VIDs use the map default unless the default is deleted, in which case per-VID alive entries are written for the new range.
- `deleted=true`: new VIDs use the map default if it is deleted, otherwise per-VID deleted entries are written for the new range. `m_deleted` is incremented.

The count key is saved after the extension.

### Delete

`Delete(VID)` writes `0xfe` to that VID's key. It uses a striped per-VID mutex to serialize read-modify-write operations in this process.

### Set Version

`SetVersion(VID, version)` writes the requested byte and updates `m_deleted` when the value crosses the deleted/non-deleted boundary.

### Increment Version

`IncVersion(VID, newVersion, expectedOld)` reads that VID's key/default version, validates the current byte, increments modulo `0x80`, and writes the VID key back.

When `expectedOld != 0xff`, the operation uses compare-style semantics:

- Compute `target = (expectedOld + 1) & 0x7f`.
- If the current byte is already `target`, return success and report `target`. This makes the operation idempotent when another path already performed the same version increment.
- If the current byte is `expectedOld`, write `target` and return success.
- If the current byte is neither `expectedOld` nor `target`, return failure. This prevents a stale caller from overwriting a different/newer version with its own target value.

When `expectedOld == 0xff`, the operation increments the current non-deleted value without a compare expectation. Current SPANN reassign paths pass the observed old version as `expectedOld`, so they use the compare-style path above.

`IncVersion()` uses TiKV `RawCompareAndSwap` through `KeyValueIO::CompareAndSwap`, so the compare-and-update is atomic at the TiKV server for one VID key. On CAS conflict it evaluates the value returned by TiKV and preserves the idempotent `current == target` success case described above.

The per-VID key uses a zero-padded VID so deployment scripts can pre-split version-map regions with keys such as `spfresh_sift1b_v:0:0100000000` and `spfresh_sift1b_v:1:0100000000`.

## Read Paths

### Single VID

`Deleted(VID)` and `GetVersion(VID)` read that VID's key and return one byte. Version read policies are accepted for interface compatibility, but TiKV distributed VM no longer has a local version cache, so both policies read TiKV/default metadata directly.

### Batch VID Lookup

`BatchGetVersions(vids, versions, policy)` builds one key per valid VID and issues one TiKV `MultiGet`.

Resolution phases:

1. Filter invalid VIDs to `0xfe`.
2. Batch fetch per-VID keys from TiKV.
3. Fill one output byte per input VID, using the map default for missing keys.

If the batch read fails, unresolved valid VIDs use the map default; invalid VIDs remain `0xfe`.

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

TiKV distributed VM no longer keeps a local version cache. Each single or batch read goes to TiKV and falls back to the code-defined layer default for missing per-VID keys.

During `AddIndex`, newly allocated VIDs can use the TiKV version map's default alive version directly, avoiding a point `Get` for keys that are known not to be materialized yet.

## Failure And Missing-Key Behavior

Read-side behavior is fail-closed:

- Missing count key on load sets count to `0` and logs a warning.
- Missing per-VID keys use the code-defined layer default.
- Failed `MultiGet` logs a warning and unresolved valid VIDs use the code-defined layer default.

Write-side behavior:

- Failed per-VID writes are logged and return failure to the caller.
- Delete/set/inc operations do not silently mark a write successful if TiKV write failed.

## Current Limitations

- Count is stored as a metadata key and flushed periodically or on `Save()`; it is not transactionally coupled with each per-VID write.
- `SearchCheckVersionMapOnlyLayer0` is historically named; with the current implementation it means target-layer-only checking for TiKV distributed VM.
- Missing per-VID keys use the layer's default version. Failed reads are treated as deleted. This favors correctness over recall when TiKV data is unavailable.

## Relevant Code

- `AnnService/inc/Core/Common/TiKVVersionMap.h`: TiKV-backed `IVersionMap` implementation.
- `AnnService/inc/Core/SPANN/ExtraDynamicSearcher.h`: chooses TiKV/local version map, owns add/delete/search version checks.
- `AnnService/src/Core/SPANN/SPANNIndex.cpp`: layer traversal and target-layer search policy.
- `AnnService/inc/Core/SPANN/ParameterDefinitionList.h`: public config definitions.
- `AnnService/inc/Core/SPANN/Options.h`: config storage.