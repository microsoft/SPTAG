# Single-node multi-tenant SPANN

This feature provides tenant-isolated SPANN indexes managed by one process.
Tenants share a posting backend and node-owned RocksDB recovery/state databases.
Posting keys include the tenant and index generation; tenants retain separate
head indexes, version maps, update sequences and snapshots.

Start with the [runnable multi-tenant quick start](MultiTenantQuickStart.md)
for building, starting a server and creating/querying tenants from Python.

The APIs are for trusted callers. Tenant identifiers are namespaces, not
authentication or authorization. TiKV postings provide distributed storage,
not distributed tenant ownership, cross-node recovery or failover.

## Build and compatibility

`MULTITENANT` defaults to `OFF`. Enabling it automatically sets `ROCKSDB=ON`
because recovery and metadata need RocksDB even with FILEIO or TiKV postings.
This selects a build dependency; it does not install RocksDB or start a service.
If available, the RocksDB package's shared target is used so distribution
packages with non-PIC static archives can link into the SPTAG shared library.

```sh
git submodule update --init ThirdParty/zstd
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DFOLLY=OFF -DTBB=ON -DMULTITENANT=ON -DTIKV=OFF
cmake --build build --target tenanttool server _SPTAGClient SPTAGTenantTest
ctest --test-dir build -R 'MultiTenant|PostingBackends' --output-on-failure
python3 Test/tenant_smoke.py --bin-dir Release
```

Install the normal SPTAG build dependencies and a RocksDB CMake package first.
For TiKV support also enable `TIKV=ON` and provide the existing protobuf, gRPC
and Abseil dependencies. The managed FILEIO/TiKV backends are Linux-oriented;
Windows backend behavior has not been validated.

Disabling `MULTITENANT` removes the node manager and `tenanttool`; ordinary
SPTAG indexes remain available. `ROCKSDB=ON` can still be selected independently
for native RocksDB postings or the existing optional `EnableWAL` path.
`EnableWAL` does not automatically enable the build dependency.

The client protocol/API remains available in client builds without a local
manager; it can address a remote tenant-enabled server. A server built without
`MULTITENANT` rejects tenant operations rather than using a default tenant.
Without a configured `MultiTenant.Root`, the server uses its existing index path.

The target baseline's `ConcurrentHashMapTest` has three existing failures with
`FOLLY=OFF, TBB=ON`: overwrite return values, iterator snapshots after clear,
and concurrent conditional updates. They also reproduce when compiling only
the unmodified upstream header/tests, without the multi-tenant implementation.
This change does not fix or suppress them. Passing tenant/backend tests does
not mean the entire baseline suite is green.

## Server configuration

```ini
[MultiTenant]
Root=/dedicated/path/to/node
Storage=ROCKSDBIO
MaxTenants=64
MaxResidentTenants=8
CacheBytes=3221225472
AutoCheckpoint=true
CheckpointLogBytes=1073741824
CheckpointPressureBytes=4294967296
CheckpointMinIntervalSeconds=300
```

`Storage` selects `ROCKSDBIO`, `FILEIO` or `TIKVIO` postings only. Recovery logs,
tenant catalog, head snapshots and checkpoint metadata remain in local RocksDB.
TiKV additionally requires `TiKVPDAddresses` and an owned `TiKVKeyPrefix`.
An existing node must reopen with its recorded backend configuration; changing
the setting is not an online backend migration.

The node root must be dedicated and initially empty. RocksDB's recovery lock
excludes a second process owner. Do not run the offline CLI against a live server.

## Recovery and automatic checkpoints

The manager records acknowledged inserts/deletes in `recovery_log`. Managed
indexes disable their separate native WAL to avoid logging the same operation
twice. `AutoCheckpoint=false` disables scheduling/backpressure, not recovery
logging or manual/lifecycle checkpoints.

The default policy is node-wide:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `AutoCheckpoint` | true | Enable one background checkpoint scheduler |
| `CheckpointLogBytes` | 1 GiB | Soft trigger for retained recovery-record key/value bytes |
| `CheckpointMinIntervalSeconds` | 300 | Minimum interval since the last successful checkpoint |
| `CheckpointPressureBytes` | 4 GiB | Bypass cooldown and hold new insert/delete admission |

Both byte thresholds must be positive when enabled, and the pressure threshold
must be at least the soft threshold. An interval of zero removes the cooldown.
A timer alone never checkpoints an idle or below-threshold node.

The scheduler reuses the same node checkpoint protocol as manual checkpoints.
Scheduling is asynchronous, but execution takes the update barrier and can
pause writers. Resident searches retain their epoch-protected read path.
Only one checkpoint executes at a time, including manual and lifecycle work.
Pressure waiters hold no leases or locks needed by checkpoint.

After snapshots are complete, the new checkpoint/catalog is published before
covered recovery records are removed. Failure fences the node and wakes waiting
writers with an error; reopen to recover rather than retrying a possibly partial
mutation. Successful checkpoints reset the byte count and cooldown. Reopening
counts retained records again, including old covered records awaiting cleanup.
Closing stops/joins the scheduler and performs the normal final checkpoint.
Destruction without `Close` stops the scheduler without initiating a checkpoint.

The thresholds are **not physical disk quotas**. Already-admitted concurrent
updates and a single large batch can overshoot the pressure threshold.
RocksDB WAL/SST overhead, compression, tombstones, checkpoints and FILEIO/TiKV
snapshot storage are not included. Logical log deletion does not guarantee
immediate SST space reclamation. Operators still need disk-space monitoring and
headroom for snapshots/compaction.

`TenantIndexManager::NodeStats` exposes `recoveryLogBytes`, `autoCheckpoint`,
`checkpointInProgress` and process-local `autoCheckpoints`. Residency byte limits
account for serialized index components, not total process RSS.

## APIs and administration

The in-process API is declared in
[`TenantIndexManager.h`](../AnnService/inc/Core/SPANN/TenantIndexManager.h).
It provides create, insert, vector/tenant deletion, search, describe, list,
residency control, statistics, checkpoint and close.
Resident searches use read epochs; eviction withdraws published index pointers
and waits for old readers before releasing index memory.

`tenanttool` supports the same administration operations using a stopped node.
Run it without arguments for command syntax. Binary vector/query files contain
int32 count, int32 dimension and packed values; a query contains one vector.
The Python client and real socket/CLI reload paths are exercised by
[`tenant_smoke.py`](../Test/tenant_smoke.py).

CLI environment overrides:

```text
SPTAG_POSTING_STORAGE=ROCKSDBIO|FILEIO|TIKVIO
SPTAG_TIKV_PD=host:port
SPTAG_TIKV_PREFIX=owned-prefix
SPTAG_AUTO_CHECKPOINT=0|1
SPTAG_CHECKPOINT_LOG_BYTES=1073741824
SPTAG_CHECKPOINT_PRESSURE_BYTES=4294967296
SPTAG_CHECKPOINT_MIN_INTERVAL_SECONDS=300
```

Checkpoint numeric overrides must be unsigned integers. In-process callers set
the corresponding `TenantIndexManager::Limits` fields.

## TiKV repair and performance boundaries

The managed TiKV client preserves the native 50 ms **per-RPC** timeout semantics,
with a separate 10-second batch cap, four retry rounds and at most 16 in-flight
RPCs. Read retries pause 1/2/4 ms, retain valid routes on timeouts and reuse
metadata/connections. Request-owned buffers and cancellation/CQ draining remain
mandatory on failure. This is a correctness repair, not a promise of a 50 ms
whole-query deadline. Live integration tests require `SPTAG_TIKV_PD`; absent an
endpoint they skip rather than establish real-backend coverage.

Historical SIFT10M measurements showed roughly 10% lower single-tenant RocksDB
throughput than native SPANN. Investigation found matching logical postings but
different SST/blob/merge-record layouts. Materializing an isolated shared copy
improved QPS by 9.36% with the same executable; compaction also changed SST
boundaries, so this does not isolate the merge function alone.

No forced full compaction or diagnostic key-reuse optimization is included.
Unrelated worker/heap/buffer tuning and large raw experiment artifacts are kept
outside this feature change. Historical throughput is not a measurement of the
final PR binary or an exact multi-tenancy overhead percentage.
