# Head-index cache design

Current release provenance and validation on `users/qiazh/pre-merge-tikv-bugfix`
(rebased onto `62deacc681cc3218fd3eaa377a07d35340850374`; full SIFT1M backend
campaign on `059c99ab00b31e4e155f69dfc1c60d5006c3801e` plus SIFT10M
1,024-tenant RSS/lock scalability verification) are tracked in
[MultiTenantReleaseValidation.md](MultiTenantReleaseValidation.md).

The benchmark notes at the end of this document record earlier pre-curation
design iterations (`143/144` initial cache matrix on `196d1e5`), before the
final FILEIO merge, local version-map fast-path, and distributed TiKV
version-map/client fixes documented in
[MultiTenantReleaseValidation.md](MultiTenantReleaseValidation.md).

## Implemented allocation foundations

`Core/Common/ResidentMemory.h` provides a shared byte-capacity ledger, tenant
accounts, bounded reservation waiting, oversized-reservation rejection, and
move-only allocation ownership. Its `Stats` fields are `allocatedBytes`,
`reservedBytes`, `peakBytes`, `capacityBytes`, `waits`, `timeouts`, and
`oversized`. Global and account allocated/reserved counters change together under
one ledger mutex. Serialized snapshot size is not an input.

`ResidentAllocator<T>` is an optional accounting adapter over the standard
allocator, not an arena or slab. Managed allocations include the adapter's
ownership header and alignment padding. Unmanaged instances use the standard
allocator directly. Destruction frees the backing before releasing its charge,
including destruction on another thread.

Implemented ownership hooks cover:

- Dataset base allocations, growth blocks and growth-owner vector capacity.
  `ReserveRows` prepares physical growth without changing the published row
  count, and charges whole blocks rather than their populated rows.
- Optimized vector/graph datasets sharing one interleaved base allocation and
  growth ownership; either view can outlive the other without premature free.
- Fixed fine-grained lock arrays, lazy per-head mutex allocations and their
  hash-map backing, including allocation through a const accessor.
- BKT/KDT retained tree vectors, BKT sample-map allocations and tree mutexes.
  Replacement trees retain the originating allocation account.
- Version/deletion backing, deterministic head maps, configuration-map nodes and
  strings, index/searcher objects, and PQ/OPQ quantizer objects and arrays.
- Guarded standard-string replacement, including first adoption of an existing
  string, and standard-array replacement. Both charge replacement overlap and
  preserve the original target when admission fails.
- Index-owned workspace registration metadata: managed factories prepare 64
  live-worker slots. Exited workers are reclaimed on subsequent registration,
  returning cached workspace I/O IDs without growing resident metadata inside
  a read epoch. Beyond 64 live registrations, query scratch is uncached. This
  is not a global concurrent-query limit; the I/O-ID pool has its own capacity.

Managed RaBitQ is explicitly rejected rather than silently bypassing allocation
ownership. Native, unmanaged RaBitQ behavior is unchanged. Query scratch remains
outside this budget and is not separately byte-instrumented; process RSS must
not be treated as an exact scratch-memory measurement.

Prepared reservations distinguish reusable growth headroom from live backing.
Freeing backing always decrements allocated bytes and notifies all ledger
waiters. While a preparation is still owned, the released capacity returns to
that preparation's explicitly reported reserved headroom; it becomes globally
available when the last preparation owner releases it. Captured memory contexts
retain preparation ownership independently of an insertion's `UpdateContext`.
Both posting-maintenance jobs and head-tree rebuild jobs propagate this context.

The manager now installs tenant accounts during loading/recovery and admits
tracked allocation increments against `MaxResidentBytes`. Its local defaults
are 32 GiB, `MaxResidentTenants=0` (no count guard), and
`ResidentWaitMilliseconds=5000`. Snapshot bytes remain persistence metadata,
not the capacity charge. Node/tenant allocated and reserved bytes, loading and
retiring bytes, charged peak and pressure counters are exposed in manager
statistics and offline CLI output. Shared block-cache usage/capacity and local
state/recovery RocksDB memtables are reported separately, outside head capacity.
Remote TiKV server memory is also outside the client head budget.

Tracked load allocation can evict another eligible LRU tenant outside the
ledger mutex. First loads are serialized to prevent partially constructed,
unevictable entries from mutually exhausting the budget; this can reduce cold
miss concurrency and must be included in the performance evaluation.
Managed mutations are serialized per tenant, with growth reservation and
preparation before VID reservation or recovery recording. Background owners
retain preparation lifetime. Growth bounds use admitted vector IDs, unique head
promotion, prepared read locks, container replacement peaks and overlapping tree
rebuilds, not the configured maximum future tenant size. Rebuilds are serialized
and propagate failures. The bound is deliberately conservative: the existing
dataset growth quantum can make even a small write require substantial temporary
headroom. Such a write can be rejected despite low current allocated bytes.

**When is the tree term excess?** A head append can enqueue a rebuild only
if `end - sizePerTree() >= AddCountForRebuild`. The tree size is nonnegative,
and unique promotion bounds every `end` by the admitted VID count, even if a
split promotes old VIDs. If that count is strictly below the threshold, the
write needs no new rebuild credit. At equality or above, admission still
reserves both tree peaks; existing backing and outstanding job owners remain
charged independently. This test does not predict whether a posting will split.

For example, `ResidentGrowthBelowRebuildThresholdSurvivesReopen` inserts one
4D float after 512 seed vectors (one posting layer, BKT, 32-bit IDs). In a
measured load the tenant had 156,545,024 bytes free. The first possible
promotion beyond the loaded rows needs full growth blocks: 16 MiB samples,
128 MiB graph, 1 MiB deletion labels and 4 MiB local-to-global IDs. Those
149 MiB are not excess. Block owners, global-to-local nodes/rehashes, layer
versions and locks accounted for another 216,328 bytes of reservation. The
old request also included 205,712 bytes for two unreachable tree peaks,
so it requested 156,659,864 bytes and failed before VID assignment. The new
request of 156,454,152 fits. Metadata varies with the selected heads and ABI;
this is not evidence that smaller budgets can admit the same physical blocks.

Capacity waiting consumes one shared wait budget across cold-loader admission,
the optional count guard and update growth. Actual capacity/queue waiting counts
toward that budget; ordinary load I/O is not an overall operation deadline.

## Agreed direction

**Use a byte-capacity budget as the primary residency constraint, not a fixed
number of tenant head indexes.** An optional tenant-count limit may be a separate
operational guardrail, but it must not silently prevent the intended byte-cache
utilization.

A 4 GiB budget should accommodate one large entry or many small entries,
depending on their charged memory and access pattern. For example, 1,000 entries
charged at 4 MiB each occupy about 3.91 GiB. If the head payload alone occupies
4 GiB, associated resident metadata can make the complete entry too large.
The exact accounting scope must therefore be explicit.

Whole-tenant eviction is also the agreed replacement granularity. Do not add
page-level head eviction: its partial-residency, pinning and update machinery
would make head-index management substantially more complex.

Keep the existing default allocator. Do not introduce per-tenant arenas or a
custom shared head slab pool for now. Mature scalable allocators can be evaluated
later without making one a new dependency in this redesign.

### Confirmed implementation and test requirements

The following target behavior is implemented locally:

- Default resident capacity: **32 GiB**, configurable. By default no resident
  tenant-count limit applies; retain an explicit count limit as a separate
  optional guardrail.
- Scope: the entire loaded tenant index and auxiliary resident structures whose
  lifetime follows that index. Query scratch and backend RocksDB block
  caches/memtables are outside this capacity and must be reported separately.
- Reject an entry larger than the configured capacity, including when the cache
  is empty. Do not retain the current oversized-entry exception.
- Wait for capacity for at most **5 seconds**, configurable, then report a
  retryable pressure result. This is not an end-to-end RPC/load deadline.
- Reserve update growth before durable mutation; ordinary capacity rejection
  must not leave a recorded mutation that later takes effect during replay.
- Preserve default allocation and whole-tenant timestamp LRU. Do not add a
  per-tenant arena, custom shared slab or page-level eviction.
- Compare against the worker-restored pre-cache candidate, with the same SIFT1M
  32/128-tenant partitions. Test RocksDB, FILEIO and TiKV, both all-resident and
  capacity-pressure scenarios, for search-only and mixed read/write traffic.

The local implementation addresses these requirements; the published baseline
does not. Performance acceptance remains separate from the allocation tests.

## Baseline before this redesign

### Three different controls, not one head-memory budget

| Setting | Baseline default | Meaning |
| --- | ---: | --- |
| `MaxResidentTenants` | 8 | Maximum resident/loading tenant count |
| `MaxResidentBytes` | 0 | Disabled by default; otherwise a soft target charged using serialized snapshot bytes |
| `CacheBytes` | 3 GiB | Shared RocksDB block-cache capacity, not head-index capacity |

Both enabled residency constraints must pass when admitting a load. An empty
cache can admit one tenant larger than `MaxResidentBytes`, so the byte target is
not a hard bound. The current implementation requires a positive
`MaxResidentTenants`; zero does not disable that count constraint.
`MaxTenants`, separately defaulting to 64, limits the tenant catalog.
Supporting 1,000 tenants requires addressing that catalog setting too.

`snapshotBytes` sums serialized index/snapshot components, including per-layer
version/deletion data. It is not measured heap capacity or process RSS.
It is refreshed during snapshot save/load rather than on every live allocation.
Consequently, enabling the existing byte target is not the complete redesign.

### Replacement unit: one resident tenant index

The manager holds a loaded `VectorIndex` for each resident tenant. Eviction
releases that tenant's index ownership as a unit after safe reclamation.
It does not retain some head pages while evicting others.

Calling this a "head cache" is convenient but incomplete: loading/evicting the
managed index also involves auxiliary resident state, including version maps.
The accounting scope must match everything whose lifetime is controlled by this
cache, or explicitly budget the auxiliary state elsewhere.

### Current LRU implementation

This is timestamp-based, whole-tenant LRU, not an intrusive LRU linked list.

1. Resident searches enter a node read epoch before reading the published index
   pointer. Their completion updates the tenant's atomic `lastReadAccess`.
2. The lease-based path updates `lastAccess` when acquiring/releasing a tenant.
3. Recency is `max(lastAccess, lastReadAccess)`.
4. On an admission miss that does not fit, the manager scans tenants and selects
   the oldest eligible resident tenant: no active lease and no observed active
   epoch reader for that tenant.
5. It withdraws the published pointer, advances the epoch, waits for pre-existing
   readers, saves dirty state when needed, and releases the index. It then
   reevaluates admission and can evict another tenant.

If every candidate is busy, the miss waits instead of freeing memory still used
by requests. The epoch wait is node-domain based; it can also wait for older
readers of other tenants, not only the selected tenant.

The common resident-search path does not move a shared LRU-list node under the
cache mutex. Replacement scans the tenant catalog and consults reader records;
it is not constant-time replacement. With concurrent access, completion-based
timestamps and eligibility checks give approximate LRU ordering, not a globally
serialized exact access order. Scalability at thousands of tenants needs
measurement before selecting a replacement-policy data structure.

## Is there paging?

Three different mechanisms must not be conflated:

| Mechanism | Current head-index cache | Purpose |
| --- | --- | --- |
| OS virtual-memory paging | Normal heap allocations are backed through the OS virtual-memory system | Map virtual addresses to physical memory |
| Application page/slab allocation | No dedicated head-cache page/slab allocator | Manage allocation/reuse in regular-size units |
| Application page-level residency/eviction | Not implemented | Load/evict individual head pages instead of whole tenant indexes |

The implementation has posting I/O `PageBuffer` objects and RocksDB block
caching, but neither makes the in-memory head index a demand-paged application
cache. Worker size-class pools reuse query scratch; they do not allocate or
page the long-lived head tree/graph/vector data.

Existing head data structures use ordinary aligned allocations, vectors and
other heap-owned structures. For example, `Dataset` owns aligned base/growth
blocks, while an optimized load can share one backing allocation between vector
and graph views. These are not a unified head-cache pager.

## Does no application paging imply fragmentation?

**Fragmentation is possible, but application paging is neither necessary for
byte-based LRU nor a guarantee that fragmentation disappears.**

Distinguish:

- **Allocator/external fragmentation:** differently sized objects are allocated
  and freed over time, leaving reusable space that may not fit a later request.
- **Internal fragmentation:** alignment, size-class rounding and partially used
  pages/chunks retain capacity beyond the logical payload.
- **Allocator retention:** freed memory can remain in allocator arenas instead
  of immediately reducing process RSS. This is not a live cache entry.
- **Physical-memory layout:** a virtually contiguous array does not generally
  require physically contiguous RAM. OS paging already provides that indirection.

Whole-tenant eviction can release all owned head allocations, but cannot promise
an immediate matching RSS reduction. Shared references, scratch pools, RocksDB,
allocator retention and unrelated live allocations must be separated when
diagnosing the result.

A fixed-page/slab allocator can improve reuse and capacity accounting, but adds
rounding/tail waste and metadata. Large arrays may still need contiguous virtual
address ranges or data-structure changes to use multiple chunks. Adding a page
allocator alone does not make arbitrary C++ objects independently evictable.

## Capacity-first model

This section records the design rationale. The local implementation follows the
allocation-increment approach above; the published PR still has baseline behavior.

### Define the cache entry and charge

Use the loaded tenant's owned index memory, rather than snapshot file length,
as the accounting basis. Inventory head vectors, tree/graph buffers, allocation
capacity and auxiliary structures before choosing a public name such as
`HeadCacheBytes` versus `ResidentIndexBytes`.

Shared backing allocations must be counted once. Reserved allocation capacity,
not only populated element counts, matters. Worker scratch, RocksDB block cache
and memtables, checkpoint/build staging, and catalog/recovery state need
separate budgets or explicitly documented inclusion.

A useful admission invariant is:

```text
resident + loading + retiring + growth reservations + retained cache-owned pools
    <= configured cache capacity
```

These are disjoint ledger categories; transfer reservations into actual
allocations rather than double-counting them. Retiring memory stays charged
until its readers/owners release it. Resident growth needs admission before
allocation, not only a later checkpoint measurement. Temporary replacement
allocations can overlap old and new storage and must be covered too.

This is an allocation-capacity contract, not an exact process-RSS cap.
Allocator overhead and non-cache memory require separate headroom and
observability.

### Admission and replacement

Estimate or bound the upcoming load's allocation requirement before loading.
Reserve under the capacity ledger so concurrent misses cannot independently
spend the same free space. If the bound proves insufficient, acquire additional
capacity safely or fail/unwind the load; do not silently exceed the limit.

Evict eligible LRU entries until the reservation fits. Keep the existing safety
rule: withdraw pointers before epoch reclamation, and do not hold read epochs
while waiting for capacity/loading/lifecycle work. Dirty eviction must preserve
the current save/recovery guarantees.

Whole-tenant replacement and byte-based capacity are compatible. There is no
requirement to introduce page-level eviction just to replace the count limit.

### Implemented policy boundaries

| Topic | Local implementation |
| --- | --- |
| Oversized entry | Reject explicitly under the configured budget |
| All entries pinned | Capacity wait defaults to 5 seconds, configurable; return retryable pressure on timeout |
| Replacement policy | Keep timestamp scanning; evaluate alternatives only after measurement |
| Allocation integration | Existing default allocator with allocation ownership and pre-allocation reservations; no new arena/slab |
| Page-level head eviction | Out of scope by agreement; retain whole-tenant replacement |
| Dynamic growth | Reserve before durable mutation using admitted IDs, unique promotion and overlapping replacement/rebuild bounds |

Per-tenant arenas and a custom shared head slab pool are deferred by agreement.
Capacity accounting and safe reclamation are still needed with the existing
allocator; neither requires implementing a new allocator.

## Recommended optimization boundary

The recommendation is **capacity accounting and safe admission first; allocator
and replacement-policy changes only after measurement**. Accurate charging is
necessary to deliver the capacity contract, not an optional performance tweak.
It does not require routing all allocations through a new general-purpose
allocator or adding a shared lock to every search.

| Mechanism | Recommendation | Benefit versus cost |
| --- | --- | --- |
| Capacity ledger and pre-allocation reservations | Required for the byte-budget redesign | Prevent concurrent loads/growth from overspending; requires complete allocation-path coverage |
| Allocation ownership/capacity reporting | Required | Charge actual reserved backing once, including slack; reporting alone cannot prevent an allocation from exceeding the budget |
| Growth admission and explicit pressure errors | Required | Avoid silent overshoot or unbounded waits; must integrate with asynchronous updates and recovery semantics |
| Memory and miss-latency metrics | Add with the redesign | Distinguish accounting errors, fragmentation and I/O delay before selecting a remedy |
| Existing aligned/block allocations | Keep initially | Preserve contiguous vector/graph access and low integration risk; allocator retention can remain |
| Per-tenant arena | Do not add for now, by agreement | Bulk ownership can be useful, but allocator integration adds complexity; monotonic variants also retain replaced buffers |
| Shared head size classes/slabs | Do not add for now, by agreement | Potentially better cross-tenant reuse, but rounding, retained pools, synchronization and allocator plumbing add cost |
| Explicit OS mappings for large backing blocks | Conditional experiment | Unmapping can return those mappings promptly; adds mapping/fault overhead and platform-specific ownership paths |
| Timestamp-scan LRU | Keep initially; instrument | Preserves current resident-search behavior; miss-side scans and reader-record checks may become expensive |
| Exact linked-list LRU | Do not add by default | Faster victim lookup does not justify shared-list updates on every hit without evidence |
| CLOCK/segmented admission or cost-aware replacement | Defer until workload evidence | Can reduce scan cost or pollution, but changes hit rate, large-tenant behavior and fairness |
| Background eviction/preflush or prefetch | Defer | Can hide miss latency, but adds I/O, capacity overlap, races and failure handling |

### Why keep the existing allocator?

The existing `Dataset` implementation already allocates growth in blocks, and
the optimized vector/graph loader can share an aligned backing allocation.
This is not a cache page allocator, but it means large head payloads are not
necessarily millions of unrelated small heap allocations.

First expose the owned base bytes, allocated growth-block bytes, pointer-array
capacity and other resident structures. A partially filled growth block still
costs its full allocation. Do not charge a configured maximum row count as
though all future blocks were already allocated, and do not use serialized
`BufferSize()` as an allocated-capacity measurement. Combined vector/graph views
must charge their shared backing only once.

A monotonic arena fits immutable objects destroyed together. A mutable head can
replace buffers while remaining resident for a long time: keeping old buffers
until eviction can make the arena worse than the existing allocator. A
reclaimable arena or shared slab allocator avoids some of this waste but needs
allocator-aware ownership through the index structures. Custom shared slabs
also need size classes, free lists, alignment and concurrency handling. Neither
is a small manager-only change.

jemalloc, TCMalloc and mimalloc are examples of mature allocator alternatives
with scalable allocation designs. They are candidates for a later controlled
benchmark, not a reason to implement a custom head allocator now. Integration
must preserve aligned allocation/free pairing and platform support. No allocator
automatically enforces the head-cache budget or guarantees an immediate RSS
drop after tenant eviction.

If churn testing shows unacceptable allocator-retained memory, first isolate
which allocation sizes and owners retain it. Consider an alternative allocator
as a controlled deployment experiment, or explicit mappings for a small number
of large owned blocks. Mappings do not address small-object fragmentation,
shared ownership, outstanding readers or separately retained scratch. Avoid
global heap trimming on every eviction: it is not a reliable per-entry
reclamation contract and can hurt unrelated allocations.

### Growth pressure is a correctness decision

The baseline insertion path reserves vector IDs and records recovery data before
`InsertAt`, and waits for asynchronous update completion afterward. A new
capacity rejection introduced after durable recording cannot simply be treated
as "nothing happened; retry": replay may still apply the recorded operation.

Prefer reserving a defensible allocation bound before durable mutation and
carrying that reservation through all associated asynchronous work. Release it
only when ownership/completion permits. Where that bound cannot be established,
define rollback or existing fenced-recovery behavior explicitly before calling
the budget strict. A post-insert memory survey is useful telemetry, not admission
control. Budget exhaustion should normally be rejected before mutation rather
than turning ordinary pressure into a node-wide failure.

Never solve growth pressure by waiting while holding an epoch or an
update/allocator lock needed by eviction. Do not evict the very entry being
updated. Several pinned writers can collectively prevent progress even if each
individual head fits, so bounded waiting, cancellation and explicit pressure
errors are needed rather than indefinite mutual waiting.

For oversized entries, the approved policy is an explicit error under the normal
capacity contract. For temporarily pinned entries, the approved policy is a
configurable capacity wait (5 seconds by default), followed by a retryable
pressure result. The local manager integrates these before VID/log mutation; a separately
bounded oversized execution mode is not part of this implementation.

### On-demand allocation versus pre-durable admission

The discussion must distinguish a **node-wide cache capacity** from a
**preconfigured maximum resident size for each tenant**. The latter is not
required for byte-based eviction. A tenant can request additional memory as its
head grows, and the manager can evict other eligible tenants to make room.

The approved on-demand capacity flow is:

1. Before allocating, request the actual additional backing capacity.
2. Atomically reserve free node capacity, or reclaim eligible LRU entries and
   retry. Do not select the requesting tenant or free pinned memory.
3. Allocate with the existing allocator. On success transfer the reservation
   into owned allocated bytes; on failure release the unused reservation.
4. Keep old backing charged during replacement/rebuild overlap, and release its
   charge only when that backing is actually reclaimed.

#### Reclamation must return capacity to every tenant

After an old backing allocation is safely freed, update the owning tenant's
memory-consumption metadata and the node-wide capacity ledger in one
synchronized accounting transition. Then notify capacity waiters so other
tenants can retry their allocation requests against the newly available space.
For an allocation whose preparation has already closed, the released capacity
is globally available immediately. An active pre-durable preparation can instead
recycle the freed charge into its separately reported reserved headroom until
that preparation's last owner finishes. Allocated bytes still fall immediately,
and waiters are notified; allocated and reserved bytes must not be conflated.
This prevents later work from losing capacity already admitted before its
durable record. Closing that preparation returns unused headroom globally.

For example, replacing a 100 MiB buffer with a 150 MiB buffer temporarily charges
250 MiB. Once the old buffer is actually reclaimed, the entry's allocated bytes
become 150 MiB. An active preparation can still hold the freed 100 MiB as reserved,
so total charge need not fall yet. The node has 100 MiB more globally available
when that preparation closes; without an active preparation, release is immediate.

Publish the replacement pointer safely before reclaiming old backing. Reader
epochs and other shared owners can delay the free; merely replacing a pointer,
scheduling destruction or completing an insertion must not return capacity
early. Release each owned backing allocation's charge exactly once, including
failure/unwind paths and shared vector/graph backing.

Waiters must recheck the capacity predicate and atomically reserve their own
increment; a notification is not a capacity grant. Synchronize the predicate
and waiting protocol to avoid lost wakeups. Also notify on unused reservation
release after allocation failure. This returns logical allocation capacity to
the cache budget; the default allocator need not immediately lower process RSS.
The "buffer pool" here is a capacity coordinator, not a new arena/slab allocator.

This is reservation for a particular allocation, not advance reservation of a
tenant's entire future capacity. Allocating first and updating the size afterward
does not enforce a strict shared budget under concurrency. Allocation callbacks
also cannot wait while holding epochs or locks required for reclamation.

The mutation issue is narrower: on-demand allocation admission alone does
not guarantee that an entire mutation is rejected **before** its durable recovery
record. A later split/rebuild can require memory after the record exists. Such
a failure cannot be reported as a clean, unapplied, retryable write rejection
without additional preparation or defined transaction/recovery semantics.
The capacity algorithm and this mutation guarantee are separate concerns.

The following conservative bounds are implementation alternatives for that
mutation guarantee, not requirements to preconfigure per-tenant cache sizes:

Source inspection establishes a finite but potentially pessimistic bound:
dataset capacities limit retained heads, and splits promote existing vector IDs.
However, a cascade can promote pre-existing vectors unrelated to the size of
the current insert. Therefore, a small reservation proportional only to the
incoming vector count is not yet a justified bound.

One implementation can reserve remaining growth up to the tenant's configured
capacity plus overlapping tree rebuild allocations before recording an update.
This is conservative, but can reject small writes or evict useful tenants while
actual allocated memory remains well below the node capacity.

A tighter alternative bounds growth by admitted vector IDs and explicitly
coordinates unique-head promotion, container capacities and rebuild ownership.
It requires more changes and proof, but avoids treating the entire configured
future tenant capacity as an immediate update reservation.

The local implementation takes this tighter alternative. The broader
configured-maximum reservation above is retained here only as a rejected
alternative, not the implemented admission rule.

BKT background rebuilds are not covered by the insertion's `UpdateContext`.
Whichever strategy is selected, rebuild reservations must remain owned and
charged through rebuild/reclamation completion, not end at the insert response.
Lazy persistent read-side structures also need standing reservations or eager
preparation; a resident query cannot wait for capacity inside its epoch.

This is a choice about admission pessimism and implementation scope. It does
not require adding a new maintenance scheduler, and an unbounded transient
job queue alone does not establish unbounded retained index memory.

### Keep replacement simple without ignoring its limitations

An exact LRU list would add synchronization or deferred-maintenance complexity
to the resident hit path. Keep timestamps initially, but measure candidate
scans, reader-record checks, mutex time and eviction latency separately. The
current scan cost includes the tenant catalog and reader inspection, not just
resident entries; a resident-only candidate collection is a narrower possible
follow-up than replacing the entire policy.

Ordinary LRU is not inherently optimal for unequal entry sizes: one cold large
head can displace many frequently reused small heads. Conversely, always
preferring small entries can starve an expensive large tenant. Start with
byte-bounded LRU, then compare policies using mixed-size request traces, query
hit rate, backend load, tail latency and per-tenant service quality. A byte hit
rate alone is not enough.

Keep the existing per-tenant loading state that coalesces concurrent misses.
Byte reservations bound head memory but do not by themselves bound aggregate
load I/O or staging memory. Add a load-concurrency limit only if burst tests show
that separate bottleneck. Similarly, do not introduce a fixed eviction
low-watermark now: extra eviction may reduce stalls but also unnecessarily
discard useful heads. Reserve for the actual incoming allocation first.

### Suggested delivery order

1. Inventory owned allocations and temporary peaks, add metrics, and specify
   admission/error semantics. Until coverage is complete, label estimates as
   estimates rather than advertise a hard allocation bound.
2. Implement the ledger, load/growth reservations and safe reclamation; make
   tenant count an explicitly separate guardrail. Keep whole-tenant timestamp
   LRU and the existing allocator. This is the first bounded-cache target.
3. Run mixed-size churn, concurrent-load/update and failure tests. Change
   allocator strategy, replacement policy or background work only for a
   demonstrated bottleneck, one mechanism at a time.

## Validation and measured results

### Completed checks on the local candidate

| Check | Result |
| --- | --- |
| Final multi-tenant build, RocksDB + TiKV enabled | Passed |
| Local tenant/backend tests without a live TiKV endpoint | 93 cases, 6,370 assertions passed |
| Same suite against the owned live TiKV cluster | 93 cases, 9,313 assertions passed |
| Production CLI/Python socket smoke, concurrent CRUD and crash recovery | Passed |
| Python SDK regression | 8 tests passed |
| Isolated `MULTITENANT=OFF`, `TIKV=OFF` build and C++ SDK save/reload | Passed |
| Selected native regression suites | Not green: six cases passed and five failed on both candidate and worker-restored pre-cache control |
| Candidate benchmark coverage | All 72 planned candidate runs passed |
| Full matched timing matrix | Incomplete: 143/144 accepted runs, 23/24 complete matched cases; final baseline TiKV API failure is fatal |
| Accepted-run audit | Passed capacity, data/restart, accuracy-digest and frozen-binary checks for all 143 accepted runs |

The five native failures are the two legacy SPANN algorithm cases, the
visibility-interleaving deduplication case, SPANN reconstruction and the storage
workspace-layout case. The same cases fail in a separately built pre-cache
control; approximate-result assertion counts differ slightly. These are retained
as baseline limitations, not silently reported as passing or fixed in this
cache change.

Focused tests include heterogeneous tenant allocations, concurrent misses,
oversized rejection, growth rejection before VID/log changes, shared backing,
replacement overlap, cross-account wakeups, failed loads, dirty-eviction
failure/recovery and background preparation lifetime. These do not replace a
long-duration, thousands-of-tenants mixed-size fragmentation study.

### Matched workload and interpretation

The control is the published candidate plus the restored worker optimizations,
before the cache redesign, not an unrelated upstream build. Frozen runtime and
instrumented-helper hashes were checked again after measurement.

SIFT1M was partitioned equally into 32 or 128 tenants, using UInt8 128D vectors,
20% heads, two layers, internal result count 64, posting pages 4 and reassign K64.
Runs used the production socket handlers, 16 closed-loop client workers,
16 service workers, two socket workers and a 3 GiB backend block-cache setting.
Each run had a fresh private node clone, preload, five-second search warmup and
eight-second measured window. Three repeats used AB / BA / AB ordering.
Mixed traffic attempted 90% search and 10% single-vector insert. Its deterministic
stride visits all tenant slots overall, but inserts target only even-numbered
slots; it is not uniform write coverage across every tenant. Every tenant still
receives a post-restart VID probe.

All-resident runs used 32 GiB. Pressure runs used half the candidate's calibrated
all-resident allocated footprint, with the same numeric byte setting for both
versions. **These are not equal true-RAM policies:** the baseline charges small
serialized snapshots, while the candidate charges allocation capacity and
growth reservations. Pressure throughput is a comparison of different capacity
semantics, not a pure overhead regression measurement.

Only insert `CapacityExceeded`/`CapacityTimeout` responses count as expected
pressure rejections. They are excluded from successful throughput, but included
in aggregate operation-latency percentiles. All acknowledged inserts, including
those finishing after the timed window, are verified across two restarts.
Verification uses 32 GiB outside timing so even pressure-rejected tenants can
receive VID probes. Warmup, load/eviction I/O and verification are not an overall
five-second capacity deadline.

The owned TiKV server used an eight-CPU quota, 12 GiB container limit and 3 GiB
block cache. Its remote memory is outside the client head budget. Cluster
generations were replaced at a planned tenant-count boundary and after failed
attempts; immutable standalone fixtures and completed results were preserved.
The VM was shared, without global cache dropping or host tuning. Short-run
medians below are descriptive, not statistical proof of an improvement.

### All-resident throughput

Values are baseline / candidate medians of three accepted repeats.
Mixed throughput counts successful operations, not just searches.

| Backend | Tenants | Search QPS | Change | Mixed ops/s | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| RocksDB | 32 | 4,764.9 / 5,109.5 | +7.2% | 4,612.5 / 4,831.9 | +4.8% |
| RocksDB | 128 | 6,095.8 / 6,203.4 | +1.8% | 5,717.9 / 5,756.6 | +0.7% |
| FILEIO | 32 | 2,215.1 / 2,231.0 | +0.7% | 1,902.1 / 1,775.5 | -6.7% |
| FILEIO | 128 | 2,342.2 / 2,348.4 | +0.3% | 2,042.6 / 2,049.1 | +0.3% |
| TiKV | 32 | 2,616.5 / 2,728.2 | +4.3% | 1,859.1 / 2,142.5 | +15.2% |
| TiKV | 128 | 3,031.4 / 3,030.0 | -0.05% | 2,366.4 / 2,086.3 | -11.8% |

| Backend | Tenants | Search p99 ms, baseline / candidate | Mixed p99 ms, baseline / candidate |
| --- | ---: | ---: | ---: |
| RocksDB | 32 | 5.59 / 5.23 | 5.87 / 5.58 |
| RocksDB | 128 | 4.47 / 4.36 | 4.82 / 4.74 |
| FILEIO | 32 | 13.47 / 13.14 | 53.17 / 57.82 |
| FILEIO | 128 | 8.29 / 8.22 | 49.05 / 49.25 |
| TiKV | 32 | 8.93 / 8.42 | 30.76 / 24.10 |
| TiKV | 128 | 7.55 / 7.60 | 19.33 / 25.70 |

The FILEIO32 and TiKV128 mixed regressions are real observations in this matrix,
not omitted outliers. Their causes have not been isolated by profiling; do not
claim that the stricter admission/accounting design is free of performance cost.

### Capacity pressure

All 36 candidate pressure runs exercised eviction. Every candidate sample
satisfied allocated + reserved <= capacity, and the ledger's cumulative exact
charged peak stayed within the integer-byte limit, including replacement overlap.
The maximum charged peak reached the configured limit in each backend/tenant
group; the rounded budgets below are not the precision used by the audit.

| Backend | Tenants | Budget MiB | Search QPS, baseline / candidate | Mixed ops/s, baseline / candidate | Candidate mixed rejections/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| RocksDB | 32 | 201.0 | 4,768.9 / 225.4 | 4,559.9 / 225.5 | 24.9 |
| RocksDB | 128 | 598.0 | 6,105.4 / 360.3 | 5,677.0 / 519.8 | 0 |
| FILEIO | 32 | 201.0 | 2,205.4 / 239.0 | 1,875.5 / 273.1 | 30.3 |
| FILEIO | 128 | 598.0 | 2,346.4 / 357.9 | 2,038.6 / 512.9 | 0 |
| TiKV | 32 | 201.0 | 2,649.4 / 218.5 | 2,220.1 / 199.9 | 22.4 |
| TiKV | 128 | 598.1 | 2,945.0 / 318.4 | Baseline incomplete / 442.1 | 0 |

The final TiKV mixed baseline has only two accepted repeats; no complete
three-repeat baseline median or matched speedup is claimed for that row.
The candidate's three repeats passed, including acknowledged insert, deletion,
repeated-deletion and contiguous-VID checks across restarts.

All three 32-tenant pressure groups admitted no timed inserts: conservative
growth preparation can require more than the remaining budget even when current
live head bytes look small. The 128-tenant pressure groups admitted writes.
This is a material admission tradeoff, not a reason to count rejected writes
as successful throughput.

#### Follow-up interpretation of RocksDB/FILEIO pressure results

A 2026-11-17 read-only analysis aligned the first/last 500 ms telemetry samples
inside each measured window and took the median of three per-run counter rates.
These are approximate rates, not exact boundary counters:

| Backend | Tenants | Baseline search loads/evictions per second | Candidate search loads/evictions per second |
| --- | ---: | ---: | ---: |
| RocksDB | 32 | 0 / 0 | 56.8 / 56.8 |
| RocksDB | 128 | 0 / 0 | 109.1 / 109.1 |
| FILEIO | 32 | 0 / 0 | 59.8 / 59.8 |
| FILEIO | 128 | 0 / 0 | 110.3 / 110.5 |

Baseline retained all 32/128 tenants and charged only about 16-17 MiB of
serialized snapshots. Candidate all-resident allocated footprints were about
402/1,196 MiB, against pressure budgets of about 201/598 MiB. The old snapshot
charge is **not** a measurement of its actual head RAM. Candidate direct-hit
rates were only about 0-3.5 requests/s in these search runs; almost every request
entered a miss/wait path. Misses can coalesce into one load, so requests and
loads are not one-to-one. The trace is a deterministic full-tenant cycle per
worker, not independent random-uniform sampling; LRU can thrash under such
scans even when a substantial subset fits.

There are concrete implementation opportunities beyond that semantic mismatch:

- Each `FineGrainedRWLock` eagerly constructs 32,768 map buckets. On the tested
  x86-64 ABI, a candidate bucket is 112 bytes versus 96 bytes in baseline.
  *(Update: subsequently resolved on this branch — `FineGrainedRWLock` now
  lazily allocates `4,096` `alignas(64)` buckets on first lock access rather
  than `32,768` buckets on construction, `FineGrainedLock` lazily allocates
  `4,096` mutexes on first write, `PrepareResidentReads` no longer precreates
  write-only posting locks for read-only tenants, and background `ThreadPool` /
  `SPDKThreadPool` workers spawn lazily on first enqueued job.)*
- The global `m_loadingMutex` serializes complete index loads to prevent
  partially constructed, unevictable entries from exhausting the budget
  together. Loads reconstruct index metadata and update thread pools; eviction
  retires readers and destroys the index. Safe parallel loading requires
  upfront complete-load reservations and bounded concurrency, not simply
  removing this guard. Phase-level timing is still needed before attributing
  a percentage of the slowdown to queueing, thread creation, allocation or I/O.
- Managed mutation preparation conservatively covers possible head promotion,
  dataset/graph growth and tree overlap. The inherited block quantum is
  1,048,576 rows: a 128D UInt8 vector block alone is 128 MiB. Together with graph
  growth, preparation can exceed a 201 MiB cache for a single tenant. This
  explains why eviction alone cannot make those 32-tenant writes admissible.
  Smaller managed growth quanta and a tighter proven promotion bound are
  candidates; removing reservation/overlap accounting is not.

The FILEIO32 all-resident mixed regression (-6.7%) is a separate unresolved
observation, not explained by pressure eviction. No new performance runs or
phase profiles were performed in this analysis, and no speedup is promised.
`pressure-diagnosis.json` preserves the 48 pressure-run inputs summarized into
16 backend/count/workload/version rows. Follow-up comparisons should hold
effective residency comparable and retain the cyclic stress trace while also
measuring independent random and hot-tenant traces; do not silently replace
the original workload or report the old snapshot budget as equal true RAM.

The lock buckets are a directory from posting/head IDs to per-posting read/write
mutexes, not LRU replacement locks. A bucket has synchronization for its map,
allocator/account handles, table pointers and size/capacity bookkeeping; the
per-ID mutex nodes are additional allocations. The same 32,768-bucket design
exists in pre-tenant SPANN commit `539f6a60711c490d72902176cb0aad608bcb60e4`.
The candidate increases this fixed two-layer directory from 6 to 7 MiB/tenant;
it did not introduce the entire directory cost. Managed `PrepareResidentReads`
also precreates lock nodes for known heads; the old map creates nodes on first
use. Those per-ID nodes are additional to the fixed 7 MiB directory, so the
fixed-header increase is not the entirety of the new memory overhead.

The reported **75% means 128 x 7 MiB / approximately 1,196 MiB** of candidate
all-resident tracked head/index allocations for the equal-partition 128-tenant
case. Its denominator is neither server RSS nor the 598 MiB pressure budget.
The ratio depends on index size and layer count, and is not universal.

Likewise, block-at-a-time `Dataset::AddBatch` allocation predates tenancy.
What the cache implementation adds is pre-durable admission for a conservative
possible-growth bound, including head promotion, graph/tree growth and old/new
overlap. Reserved credit is not an allocation and an ordinary posting append
does not necessarily create a new head. The inherited large block size and the
new pessimistic preparation are separate costs; the latter can reject a write
that would not actually trigger those large allocations.

### Head capacity is not process RSS

The following server high-water RSS values are medians, sampled after the timed
client and before restart verification. They include preload/warmup and are not
an isolated eight-second RSS peak. Units are MiB, baseline / candidate.

| Backend | Tenants | All-resident search RSS | Pressure search RSS |
| --- | ---: | ---: | ---: |
| RocksDB | 32 | 2,036 / 2,173 | 2,036 / 1,907 |
| RocksDB | 128 | 3,774 / 4,082 | 3,774 / 4,515 |
| FILEIO | 32 | 875 / 1,006 | 875 / 1,192 |
| FILEIO | 128 | 2,650 / 2,940 | 2,649 / 3,541 |
| TiKV | 32 | 915 / 1,052 | 915 / 1,348 |
| TiKV | 128 | 2,700 / 3,009 | 2,699 / 3,399 |

**A smaller charged head cache did not consistently reduce RSS.** Candidate
all-resident RSS was higher, and churn increased pressure RSS in five of these
six groups relative to baseline. The data do not isolate allocator retention,
fragmentation, scratch, backend memory or other loading overhead. Scratch is
excluded and not separately byte-instrumented; RSS minus ledger bytes is not a
valid scratch measurement. Allocator-retained memory was not measured separately.
Do not advertise this allocation budget as a process/physical-DRAM ceiling or
claim that fragmentation has been eliminated.

### Fatal and retained failures

The full matrix remains incomplete, rather than retrying away an API failure.
In baseline `TIKVIO-128-pressure-mixed-r2-baseline`, `RawBatchGet` exhausted four
region-error attempts in about 10-11 ms, posting maintenance failed, and the
manager fenced the node. Follow-up diagnosis identified a split/leader-readiness
transition combined with prematurely exhausted client retries, as detailed below. The client
reported **94,642 request failures** and exited 1. Its 1,198 acknowledged inserts
were not restart-verified and are excluded from verified totals. This is not a
permitted capacity-pressure rejection, and the failed baseline run was not
retried or accepted. The last candidate repeat was run independently so candidate
coverage could finish without suppressing the baseline failure.

Follow-up diagnosis on 2026-11-17 established the following failure chain:

- TiKV initiated 24 load-based region splits at `04:05:46.086-088`. Its logs
  contain a CAS `EpochNotMatch` at `.118`, followed by `stale command` responses
  on new regions through `.372`. Region 252 became leader at `.146`, but its
  log still reported that the leader had not committed in its term at `.325`.
- `ManagedRawClient::MultiGet` permits only four attempts, with 1/2/4 ms retry
  pauses; the three logged read failures exhausted these attempts in
  10.5-11.0 ms, with zero transport failures and four region errors each.
  The separate managed-view batch deadline is up to ten seconds, not 11 ms.
  Scalar `Call`, used by CAS, also has a four-attempt ceiling, with
  25/50/100 ms pauses. These ceilings can terminate recovery while the region
  transition is still in progress.
- A posting `MultiMerge` failure propagates through the insert update status
  into `TenantIndexManager::Poison`, fencing the entire node. The client
  recorded 94,639 `Fail` and three `DiskIOFail` responses. Most are the resulting
  fail-fast cascade, not 94,642 independent TiKV RPC failures.
- This baseline loaded all 128 tenants, recorded zero cache evictions and a
  maximum old-style resident charge of 17,569,574 bytes against its
  627,152,951-byte budget. This is not a head-cache-capacity rejection.

An isolated extension of the existing TiKV client tests used the unchanged
frozen baseline implementation. All ten cases / 167 assertions passed,
including three diagnostic probes: four transient `stale_command` replies
caused a read to fail in 12.1 ms and a CAS to fail in 181.5 ms; a subsequent
call succeeded within the *same* original deadline. Another probe showed that
the client ignores `EpochNotMatch.current_regions` hints and depends on fresh
PD lookup results; PD lag is a possible amplifier, not an established cause of
this incident. The original read log omits region IDs and error subtypes, so
those individual read failures cannot be assigned a specific region-error
subtype retrospectively.

At measurement time, baseline and candidate TiKV client sources were
byte-identical: candidate completion did not demonstrate immunity. A repair should distinguish safe
region-transition retries from uncertain mutation transport outcomes, consume
a bounded operation recovery budget rather than four rapid attempts, and
preserve node fencing when a mutation really fails. No production code,
measurement binary or accepted-run result was changed by this investigation;
the original failed run remains excluded. The subsequent candidate-only repair
is documented separately below.

Five other failed attempts are preserved and excluded from accepted-run medians:

| Attempt | Observation |
| --- | --- |
| RocksDB128 pressure mixed r2 baseline | Boost socket exception during shutdown after timing |
| FILEIO128 pressure mixed r2 baseline | Native SDK destructor abort after deletion/restart assertions passed |
| TiKV32 resident mixed r2 candidate | Final verification-server nonzero exit after all 1,353 acknowledged-ID and 32 VID-probe checks passed; original exit status was not recorded |
| TiKV128 pressure mixed r0 baseline | First server shutdown SIGSEGV (-11), after client success with 1,949 ACKs; restart checks not reached |
| TiKV128 pressure mixed r1 baseline | First server shutdown SIGSEGV (-11), after client success with 2,076 ACKs; restart checks not reached |

The candidate final-exit anomaly remains unresolved. An exact-state clone passed
the same checks and exited 0 under strace, and a complete rerun passed, but that
does not establish a fix or prove the cause was baseline transport behavior.
A separate baseline diagnostic clone with a SIGSEGV backtrace preload also
exited 0, capturing no crash stack. It does not explain the two baseline faults
or establish that they share the candidate anomaly's cause.

Outside timing, the harness increased accepted-socket drain from 0.5 to two
seconds. After a baseline SDK destructor abort, later client helpers flushed
results and exited at OS level only after all responses/assertions completed.
Production SDK/runtime source and frozen binaries were not changed to bypass
these lifecycle issues. Thus these measurements do not certify SDK teardown
reliability.

### Post-matrix TiKV recovery repair

The isolated candidate now uses a shared, deadline-bounded region recovery
policy for batch reads, scalar RPCs and scans: exponential equal-jitter pauses
start at 5-10 ms and cap at 50-100 ms, with a 256-attempt safety ceiling.
Per-RPC deadlines and the separate operation deadline remain distinct.
Transport failures retain their former four-round ceiling; an uncertain
mutation transport failure is never replayed. Malformed responses and
unsupported region errors fail without a retry loop.

`EpochNotMatch.current_regions` updates split ranges and regroups pending keys;
the previous leader's store is only a candidate peer for each child.
`NotLeader` updates a known member's leader/store connection without waiting
for PD. Other routing failures fall back to PD. Route revisions prevent late
RPC errors from invalidating newer routes, and epoch comparisons prevent stale
PD responses or older epoch hints from replacing newer metadata. A new route
must pass a linearizable read readiness check before a mutation is submitted;
successful reads can satisfy that check for the same route revision.

**Important safety correction:** `StaleCommand` is safe to retry for reads,
but a mutation may already have committed on an old leader. The old diagnostic
CAS probe demonstrated early exhaustion, not that replay was safe. The repair
does not replay a mutation after `StaleCommand`, unknown region errors or a
transport failure; it marks an uncertain watched-namespace outcome and leaves
the manager's existing fencing/recovery protection intact. A concurrent
leadership change can still produce a legitimate uncertain-outcome failure;
transparent exactly-once mutation recovery is not claimed.

Validation of the repaired candidate:

| Check | Result |
| --- | --- |
| TiKV client regression suite | 19 cases / 220 assertions passed, including delayed readiness, split/leader hints, concurrent stale responses, deadlines and uncertain-write non-replay |
| Controlled live TiKV split test | Region count increased from 5 to 55 during concurrent reads/CAS; 31,857 successful batch reads and 36,460 committed CAS operations, zero API/data errors, final per-writer values verified; 527 assertions passed |
| Live tenant/backend suites | 93 cases / 9,313 assertions passed across RocksDB, FILEIO and TiKV |
| Changed C++ files | clang-format and scoped whitespace checks passed |

These are separate correctness checks, not replacement formal benchmark runs
or a post-fix performance claim. The original matrix remains 143/144 and the
failed baseline stays excluded. Frozen baseline/candidate measurement server
and shared-library hashes were checked unchanged. Production source changes
are confined to the isolated candidate, with no commit, push or PR update.
The dedicated validation containers and their owned data children were removed.
`tikv-repair-tests.log`, `tikv-repair-live-splits.log`,
`tikv-repair-live-backends.log` and `tikv-repair-live-result.json` retain the
new evidence alongside the original benchmark artifacts.

### Workspace lifetime correction found by single-tenant preflight

The original bounded workspace factory retained cached objects for exited
workers until index eviction. Those objects also retained finite SPANN I/O
workspace IDs. Repeated 16-thread batches could exhaust the I/O-ID pool before
or alongside the 64 lifetime-registration limit, even though no more than
16 query threads were active together. The initial candidate smoke completed
its query APIs but emitted workspace-exhaustion errors and was rejected.

The candidate now keeps a per-worker liveness marker in each preallocated slot.
TLS publishes exit with a release store; reclamation uses an acquire load
before accessing the returned workspace. Reference-count expiry alone is not
used as a cross-thread publication barrier. The shared marker contains no
index/factory reference. New registrations reclaim exited workers' cached
objects, returning I/O IDs and reusing slots for new unique worker identities.
TLS never dereferences a retired factory or keeps index metadata alive after
eviction. The 64 live-registration bound and
the independent I/O-ID capacity remain unchanged; neither error logging nor
real concurrent-capacity checks were bypassed.

Two regressions failed before the repair and passed afterward: registering a
65th sequential worker and six waves of 16 workers borrowing a finite
16-workspace resource pool. A third case confirms the 64-live-worker bound and
complete tracked reclamation when the factory retires while workers remain
alive. A fourth transfers returned workspaces through 96 retiring workers
without joining earlier workers before the next registration; the explicit
exit marker supplies publication, not `join` or reference-count expiry.
The local tenant/backend suites passed 96 cases / 6,388 assertions;
the live TiKV suites passed 96 / 9,331, and the 19-case TiKV client suite still
passed. The subsequent native/pre-cache/current comparison retains the same
fresh-thread batch protocol rather than concealing the bug with persistent
benchmark workers. The original smoke failure is preserved separately.

### Single-tenant original/pre-cache/current comparison

All figures below use the final explicit worker-exit release/acquire revision.
The earlier weak-liveness experiment remains under `runs-fixed` and
`weak-liveness-generation.zip`; its timings are not mixed with `runs-final`.

After the workspace fix, 27 independent in-process query processes completed:
three backends x original native/pre-cache manager/current manager x three
interleaved repetitions. Each process ran full10K warmup/measured/warmup/measured
passes using 16 query workers, TopK5, SearchInternalResultNum64, MaxCheck4096,
posting/search pages4 and the same frozen checkpoint16 two-layer UInt8/128D
SIFT1M index. No ANN rebuild, cache dropping or host tuning was performed.
The managed head budget was 32 GiB. RocksDB block caches and the TiKV server
block cache were 3 GiB; FILEIO had no posting read cache. Managed recovery/state
RocksDBs used their shared 3 GiB cache separately from FILEIO posting reads.
This is a read-only, all-resident hot-path comparison, not a pressure,
load/eviction or insertion benchmark.

The original native executable is the archived pre-tenant
`539f6a60711c490d72902176cb0aad608bcb60e4` build with its previously documented
compatibility patch, not a shared-adapter bypass mislabeled as original SPANN.
The two manager controls compile the identical harness against the frozen
pre-cache and repaired current static libraries. Current includes the
allocation/container changes, worker optimizations, TiKV recovery repair and
workspace lifetime correction; the version delta is not pure dispatch cost.
Each TiKV repetition used a new owned cluster under the same eight-CPU,
12 GiB container and 3 GiB block-cache limits.

QPS values are medians of six measured full10K passes:

| Backend | Original native | Pre-cache manager | Current manager | Current vs pre-cache | Current measured QPS range |
| --- | ---: | ---: | ---: | ---: | ---: |
| RocksDB | 2,399.5 | 2,268.1 | 3,486.9 | +53.7% | 3,472.6-3,543.7 |
| FILEIO | 1,898.9 | 2,142.2 | 2,207.8 | +3.1% | 2,206.9-2,211.2 |
| TiKV | 842.0 | 1,110.7 | 1,233.2 | +11.0% | 1,050.2-1,423.3 |

| Backend | Native / pre-cache / current median p95 ms | Native / pre-cache / current process RSS high-water MiB |
| --- | ---: | ---: |
| RocksDB | 10.89 / 11.22 / 6.38 | 237.5 / 1,483.0 / 1,561.7 |
| FILEIO | 12.02 / 9.94 / 8.60 | 271.3 / 278.1 / 362.6 |
| TiKV | 23.22 / 19.03 / 17.07 | 277.8 / 373.5 / 452.4 |

All 1,080,000 query statuses and 5,400,000 ordered answer rows, including
excluded warmups, passed independent verification. Ordered VIDs and distance
bits match exactly across every process/pass/backend. Tie-aware Recall@5 is
97.802% (strict-ID 97.800%). All 18 manager processes retained one loaded
tenant, one load and zero evictions. Current tracked head allocation stayed
about 126.947 MiB throughout each process; the same 7 MiB fixed posting-lock
directory is only about 5.5% here, not the 75% of the small-partition
128-tenant case.

**This is not a universal or pressure-improvement claim.** Final FILEIO passes
were tightly grouped, but two passes in the superseded weak-liveness experiment
reached only 1,476.3 and 1,235.0 QPS. Their client CPU times were
33.4-33.7 seconds versus 34.4-35.3 seconds in the fast passes, and every pass
read the same 8.984 GiB with zero measured client writes. This is consistent
with added non-CPU wait/I/O/scheduling time rather than doubled CPU work, but
there is no device-queue or off-CPU profile establishing its cause. Both slow
passes remain in the superseded raw data and summaries, not discarded or
substituted within a generation. The exit-publication hardening is not
established as their performance fix. The earlier FILEIO32 mixed regression and
pressure behavior are not resolved by these read-only results.

Native-path distinctions prevent treating native/current differences as
manager-only overhead:

- The same 240,479 logical postings / 1,204,667,982 bytes have different RocksDB
  representations. Read-only inspection of new owned copies found native
  three-SST/six-blob materialized values with zero merge operands; shared
  postings have 65 SSTs, zero blob files and 240,479 merge operands. Both
  manager versions use the identical frozen shared input. Matching cache
  settings and logical hashes does not make the native layout equivalent.
  RSS high-water includes startup/backend caches and is not head allocation.
- Native FILEIO uses its original packed mapping/storage format; managed
  FILEIO uses the shared append-only posting image. Both manager controls use
  byte-identical frozen inputs, but native/managed are not identical I/O paths.
- Correct original native TiKV uses its intended distributed version map;
  managed TiKV uses local version maps. That additional remote work and the
  repaired client path are disclosed, not attributed to tenant dispatch.

There were no query API failures or answer mismatches, but this was not a
warning-free native campaign. Native FILEIO emitted 871 non-aborting
`ReadBlocks ... timeout, continue for next batch` warnings before its first
warmup completed; the source drains the batches rather than aborting them.
Native TiKV emitted 262 recovered `RawGet` transport-retry warnings before the
first warmup completed. Its original internal transport-retry behavior was
not modified; the benchmark process still has an external deadline.
An initially overbroad native warning gate stopped after the first native
TiKV process completed. Its original failed log-gate record remains intact,
with a separate full-output/source audit. That timing was not rerun or
replaced. Recovery time is never trimmed from its original phase; startup/warmup
recovery remains in the raw startup/warmup records, and measured QPS uses only
the predeclared measured passes. Only proven recovery/budget
warnings are classified separately; terminal/unknown warnings, API failures
and data mismatches remain fatal.

Evidence is in the session's `files/single-tenant-20261117` directory:
`summary.json`, `summary.csv`, `analysis.log`, `validation.json`,
`workspace-validation.json`, `rocksdb-layout-{native,shared}.csv` and complete
per-process raw outputs under `runs-final`. The initial failed workspace smoke
is retained under `runs`; the superseded generation and its two earlier
log-gate audits remain under `runs-fixed`. The final generation's native-read
audit remains beside its unchanged original record. Immutable native graphs/maps/version files/INI,
logical posting manifests and original benchmark server/shared-library hashes
were checked unchanged. Successful node clones and all owned test containers
were retired. These new measurements do not fill or alter the original
143/144 matrix, and they make no claim of statistical significance or a
capacity-pressure fix.

### Audit artifacts and remaining acceptance

The accepted-run audit verified **143 runs: 72 candidate and 71 baseline**,
including 71 mixed runs, 135,446 acknowledged inserts across restarts, 224,920
timed-sample distance checks, and 192,000 accuracy queries with 960,000 returned
distance checks. All six resident-search baseline/candidate result digests
matched. Runtime/helper binary hashes stayed frozen. All accepted private node
clones were retired, and the owned cluster containers/data children were cleaned
up; failed-run artifacts and immutable fixtures were retained.

Session artifacts are under
`/home/baotonglu/.copilot/session-state/bdc176fc-dad0-4047-b20e-44617f35d16f/files/head-cache-performance`.
`partial-results.json`, `partial-results.csv` and `partial-audit.json` preserve
the incomplete final group explicitly. `results.json` still says
`complete=false`; the strict full-matrix `audit-results.py` rejects it.
`native-control.json`, `method.json`, failed-run logs and per-run verification
JSON retain the provenance and limitations.

Release acceptance remains blocked by the TiKV baseline API failure and
unresolved lifecycle observations. The observed mixed-workload regressions and
RSS behavior also need explicit acceptance or further profiling. Thousands-of-
tenants, long-running mixed-size churn, allocator-retention attribution and
replacement scan/lock profiling were not established by this short equal-size
matrix. No arena/slab/page eviction or allocator replacement was added to hide
those limits.

## Source references

- [`TenantIndexManager.h`](../AnnService/inc/Core/SPANN/TenantIndexManager.h):
  limits, tenant/index ownership, residency statistics and leases.
- [`TenantIndexManager.cpp`](../AnnService/src/Core/SPANN/TenantIndexManager.cpp):
  `LastAccess`, `ReadLease`, `Acquire`, `Evict`, `Save`, `Load` and `Search`.
- [`ReadEpoch.h`](../AnnService/inc/Core/SPANN/ReadEpoch.h):
  published-pointer lifetime protection and safe reclamation.
- [`Dataset.h`](../AnnService/inc/Core/Common/Dataset.h):
  aligned backing storage, growth blocks and optimized shared allocations.
- [`MultiTenantTest.cpp`](../Test/src/MultiTenantTest.cpp):
  allocation ownership, replacement overlap, capacity/LRU/concurrent-miss tests,
  oversized rejection and pre-durable update rejection.
