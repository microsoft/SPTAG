# Off-CPU AddIndex Mutex Analysis

Date: 2026-05-08

Run under analysis:

- Benchmark log: `evaluation/2026-04-23/benchmark_rpc512_20260508_001943.log`
- Output JSON: `evaluation/2026-04-23/output_rpc512_20260508_001943.json`
- Off-CPU hot flamegraph: `evaluation/2026-04-23/spf_rpc512_offcpu_active_20260508_032116.hot.svg`
- Off-CPU hot summary: `evaluation/2026-04-23/spf_rpc512_offcpu_active_20260508_032116.hot.summary.txt`

## 1. What The Off-CPU Profile Says

After filtering out idle gRPC executor threads and idle SPDK worker threads, the hot off-CPU summary is:

```text
total_s 832.85

770.38s  92.50%  AddIndex mutex wait
 35.17s   4.22%  AddIndexAsyncSingleKey posting wait
 26.99s   3.24%  SearchDiskIndex MultiGet wait
  0.12s   0.01%  VersionMap SaveCount RawPut
  0.07s   0.01%  VersionMap Delete ReadChunk RawGet
  0.07s   0.01%  VersionMap Delete WriteChunk RawPut
```

The dominant off-CPU stack is:

```text
SPTAGTest
  __GI___clone3
  start_thread
  SPFreshTest::InsertVectors<unsigned char>(...)::{lambda()#1}::operator()() const
  SPTAG::SPANN::Index<unsigned char>::AddIndex(...)
  __lll_lock_wait
  futex_wait
```

This tells us that most blocked time is not directly inside a TiKV RPC wait stack. Most insert worker threads are waiting on a pthread mutex while trying to enter or proceed through `SPANN::Index<T>::AddIndex`.

## 2. Candidate Locks In `Index`

The SPANN index has three relevant top-level locks:

```cpp
// AnnService/inc/Core/SPANN/Index.h

std::mutex m_dataAddLock;
std::shared_timed_mutex m_dataDeleteLock;
std::shared_timed_mutex m_checkPointLock;
```

The off-CPU stack reports `__lll_lock_wait`, which is the glibc mutex slow path for a pthread mutex. Among the top-level locks in `Index`, the direct `std::mutex` is `m_dataAddLock`.

`m_checkPointLock` is acquired as a shared lock in the insert path, so multiple insert threads should be able to hold it concurrently. It protects against checkpoint/save/refine style operations, not normal insert-to-insert serialization.

`m_dataDeleteLock` is not acquired in this insert code path. It appears in delete/refine operations, not in the off-CPU stack we captured.

So the source-level candidate that matches the stack is:

```cpp
std::mutex m_dataAddLock;
```

## 3. Where `m_dataAddLock` Is Acquired

`Index<T>::AddIndex` acquires `m_dataAddLock` after taking the checkpoint shared lock:

```cpp
// AnnService/src/Core/SPANN/SPANNIndex.cpp

ErrorCode Index<T>::AddIndex(const void *p_data,
                             SizeType p_vectorNum,
                             DimensionType p_dimension,
                             std::shared_ptr<MetadataSet> p_metadataSet,
                             bool p_withMetaIndex,
                             bool p_normalized,
                             SizeType* VID)
{
    if ((m_options.m_storage == Storage::STATIC) || m_extraSearchers.size() == 0)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Only Support KV Extra Update\n");
        return ErrorCode::Fail;
    }

    if (p_data == nullptr || p_vectorNum == 0 || p_dimension == 0)
        return ErrorCode::EmptyData;
    if (p_dimension != GetFeatureDim())
        return ErrorCode::DimensionSizeMismatch;

    std::shared_lock<std::shared_timed_mutex> lock(m_checkPointLock);

    SizeType begin, end;
    {
        std::lock_guard<std::mutex> lock(m_dataAddLock);

        begin = m_options.m_vectorSize;
        end = begin + p_vectorNum;

        if (begin == 0)
        {
            return ErrorCode::EmptyIndex;
        }

        for (int layer = 0; layer < m_extraSearchers.size(); layer++)
        {
            if (m_extraSearchers[layer]->AddIDCapacity(
                    p_vectorNum,
                    layer == 0 ? false : true) != ErrorCode::Success)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "MemoryOverFlow for layer %d: add VID: %d\n",
                    layer,
                    begin);
                return ErrorCode::MemoryOverFlow;
            }
        }

        if (m_pMetadata != nullptr)
        {
            if (p_metadataSet != nullptr)
            {
                m_pMetadata->AddBatch(*p_metadataSet);
                if (HasMetaMapping())
                {
                    for (SizeType i = begin; i < end; i++)
                    {
                        ByteArray meta = m_pMetadata->GetMetadata(i);
                        std::string metastr((char *)meta.Data(), meta.Length());
                        UpdateMetaMapping(metastr, i);
                    }
                }
            }
            else
            {
                for (SizeType i = begin; i < end; i++)
                    m_pMetadata->Add(ByteArray::c_empty);
            }
        }
        m_options.m_vectorSize = end;
    }

    ...

    return m_extraSearchers[0]->AddIndex(workSpace.get(), vectorSet, begin);
}
```

Important detail: `m_dataAddLock` is released before this final call:

```cpp
return m_extraSearchers[0]->AddIndex(workSpace.get(), vectorSet, begin);
```

Therefore, the hot off-CPU stack `Index::AddIndex -> __lll_lock_wait` is not waiting on posting-list locks inside `ExtraDynamicSearcher::AddIndex`. It is waiting before the code reaches that call, at the top-level `m_dataAddLock` region.

## 4. Why The Lock Is Held Too Long

At first glance, `m_dataAddLock` seems to protect only ID allocation:

```cpp
begin = m_options.m_vectorSize;
end = begin + p_vectorNum;
m_options.m_vectorSize = end;
```

That part should be very cheap. The problem is that the same critical section also calls `AddIDCapacity` for every layer:

```cpp
for (int layer = 0; layer < m_extraSearchers.size(); layer++)
{
    m_extraSearchers[layer]->AddIDCapacity(
        p_vectorNum,
        layer == 0 ? false : true);
}
```

For this run, the active config has:

```text
Layers=2
DistributedVersionMap=true
Storage=TIKVIO
```

So `AddIDCapacity` goes through the TiKV-backed distributed version map.

## 5. `AddIDCapacity` Calls Version Map Growth And Delete

The dynamic searcher implementation is:

```cpp
// AnnService/inc/Core/SPANN/ExtraDynamicSearcher.h

virtual ErrorCode AddIDCapacity(SizeType capa, bool deleted) override
{
    SizeType begin = m_versionMap->Count();
    auto ret = m_versionMap->AddBatch(capa);
    if (ret == ErrorCode::Success && deleted) {
        for (SizeType i = begin; i < begin + capa; i++) {
            m_versionMap->Delete(i);
        }
    }
    return ret;
}
```

For layer 0, the call is:

```cpp
AddIDCapacity(p_vectorNum, false)
```

For layer 1 and above, the call is:

```cpp
AddIDCapacity(p_vectorNum, true)
```

That means every outer `Index::AddIndex` call extends the version map for both layers while holding `m_dataAddLock`. For non-leaf layers, it also marks the new IDs as deleted one by one.

In this benchmark each insert call appears to add a small batch, not one full million-vector batch, so this path is executed very frequently by the insert worker threads.

### What The `deleted` Argument Means

`AddIDCapacity(capa, deleted)` does two things:

```cpp
SizeType begin = m_versionMap->Count();
auto ret = m_versionMap->AddBatch(capa);
if (ret == ErrorCode::Success && deleted) {
        for (SizeType i = begin; i < begin + capa; i++) {
                m_versionMap->Delete(i);
        }
}
```

So `deleted` means: after extending this layer's version-map count, should the newly reserved VID range be immediately marked as deleted in this layer?

The version-map deleted marker is `0xfe`:

```cpp
return ReadVersionByte(key, policy) == 0xfe;
```

Therefore:

```text
deleted=false  -> reserve VID range, keep it alive / visible in this layer
deleted=true   -> reserve VID range, but mark it deleted / invisible in this layer
```

The reason layer 0 uses `false` is that layer 0 is the data/posting layer receiving the actual newly inserted vectors. Those new VIDs should be visible immediately for normal search and posting-list insertion.

The reason layer 1 and above use `true` is that a newly inserted vector is not automatically a head in upper layers. Upper layers contain only selected head vectors / posting-list representatives. The code still extends the version-map count for those layers so VID space stays aligned, but it marks the new VIDs deleted until a later split promotes one of them into an upper-layer head.

When a deleted VID is later promoted into a layer, `ExtraDynamicSearcher::AddIndex` can resurrect it:

```cpp
if (m_versionMap->Deleted(VID)) m_versionMap->SetVersion(VID, -1);
```

That is the lifecycle:

```text
new data vector inserted
    layer 0: alive immediately
    layer 1+: reserved but deleted / invisible

later split chooses VID as a new head
    AddHeadIndex(..., tolayer=layer+1)
        ExtraDynamicSearcher::AddIndex(..., begin=VID)
            if Deleted(VID), SetVersion(VID, -1) -> alive in that upper layer
```

## 6. TiKV Version Map `AddBatch` Does Synchronous TiKV Writes

With `DistributedVersionMap=true`, `m_versionMap` is `TiKVVersionMap`.

Its `AddBatch` implementation is:

```cpp
// AnnService/inc/Core/Common/TiKVVersionMap.h

ErrorCode AddBatch(SizeType num) override
{
    SizeType oldCount = m_count.load();
    SizeType newCount = oldCount + num;

    // Create any new chunks needed (init to 0xff = uninitialized, matching VersionLabel)
    SizeType oldLastChunk = (oldCount > 0) ? ChunkId(oldCount - 1) : -1;
    SizeType newLastChunk = ChunkId(newCount - 1);

    for (SizeType c = oldLastChunk + 1; c <= newLastChunk; c++) {
        std::string newChunk(m_chunkSize, static_cast<char>(0xff));
        WriteChunk(c, newChunk);
    }

    m_count = newCount;
    SaveCount();
    return ErrorCode::Success;
}
```

`WriteChunk` is synchronous TiKV `Put`:

```cpp
ErrorCode WriteChunk(SizeType chunkId, const std::string& data)
{
    auto ret = m_db->Put(ChunkKey(chunkId), data, MaxTimeout, nullptr);
    if (ret == ErrorCode::Success) {
        std::unique_lock<std::shared_mutex> lock(m_cacheMutex);
        CachePut(chunkId, data);
    }
    return ret;
}
```

`SaveCount` is also synchronous TiKV `Put`:

```cpp
void SaveCount()
{
    SizeType count = m_count.load();
    std::string val(reinterpret_cast<const char*>(&count), sizeof(SizeType));
    m_db->Put(CountKey(), val, MaxTimeout, nullptr);
}
```

So while `m_dataAddLock` is held, the thread may synchronously write TiKV version-map chunks and the TiKV version-map count.

Even if chunk creation is not frequent, `SaveCount()` is called on every `AddBatch`. That is enough to serialize insert workers behind synchronous TiKV `Put` latency.

## 7. Non-Leaf Layer `deleted=true` Adds More Synchronous TiKV RMW

For layer 1, `AddIDCapacity(..., true)` runs:

```cpp
for (SizeType i = begin; i < begin + capa; i++) {
    m_versionMap->Delete(i);
}
```

`TiKVVersionMap::Delete` calls `WriteVersionByte`:

```cpp
bool Delete(const SizeType& key) override
{
    if (key < 0 || key >= m_count.load()) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
            "TiKVVersionMap::Delete: invalid key %d (max %d)\n",
            key,
            m_count.load());
        return false;
    }
    uint8_t oldVal;
    if (!WriteVersionByte(key, 0xfe, oldVal)) {
        return false;
    }
    if (oldVal == 0xfe) {
        ...
        return false;
    }
    m_deleted++;
    return true;
}
```

`WriteVersionByte` serializes by chunk stripe and performs a synchronous TiKV read-modify-write:

```cpp
bool WriteVersionByte(SizeType vid, uint8_t newVal, uint8_t& oldVal)
{
    SizeType cid = ChunkId(vid);
    int offset = ChunkOffset(vid);
    std::lock_guard<std::mutex> lock(ChunkMutex(cid));
    std::string chunk = ReadChunk(cid);
    if (chunk.empty()) {
        chunk.assign(m_chunkSize, static_cast<char>(0xff));
    }
    oldVal = static_cast<uint8_t>(chunk[offset]);
    chunk[offset] = static_cast<char>(newVal);
    auto ret = WriteChunk(cid, chunk);
    if (ret != ErrorCode::Success) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
            "TiKVVersionMap::WriteVersionByte: WriteChunk failed vid=%d chunk=%d layer=%d\n",
            vid,
            cid,
            m_layer);
        return false;
    }
    return true;
}
```

`ReadChunk` is synchronous TiKV `Get`:

```cpp
std::string ReadChunk(SizeType chunkId) const
{
    std::string value;
    auto ret = m_db->Get(ChunkKey(chunkId), &value, MaxTimeout, nullptr);
    if (ret != ErrorCode::Success || value.empty()) {
        return std::string();
    }
    return value;
}
```

So for layer 1, the lock-held work can include:

```text
m_dataAddLock
  AddIDCapacity(layer 1, deleted=true)
    TiKVVersionMap::AddBatch
      SaveCount -> TiKV Put
    Delete(new VID)
      WriteVersionByte
        ReadChunk -> TiKV Get
        WriteChunk -> TiKV Put
```

This is exactly the kind of path that creates large `__lll_lock_wait` time in other insert threads.

## 8. Why The Off-CPU Profile Looks Like Mutex Wait, Not TiKV Wait

Only the thread that currently owns `m_dataAddLock` performs the synchronous version-map TiKV operations inside the critical section.

The other insert threads do not show TiKV stacks because they never get that far. They block here:

```cpp
std::lock_guard<std::mutex> lock(m_dataAddLock);
```

That is why the hot off-CPU profile is dominated by:

```text
Index::AddIndex
  __lll_lock_wait
```

while the direct TiKV wait categories are much smaller in the filtered summary.

In other words, TiKV latency still matters, but it is amplified by a global insert mutex. A small synchronous TiKV call inside `m_dataAddLock` can force all other insert workers to wait.

## 9. Why This Matters For Throughput

The benchmark uses multiple insert threads:

```text
NumInsertThreads=16
```

But this section effectively serializes the ID-capacity phase:

```text
thread 1 enters Index::AddIndex
  holds m_dataAddLock
  performs AddIDCapacity for each layer
  may perform synchronous TiKV Get/Put through TiKVVersionMap

threads 2..16
  block in __lll_lock_wait on m_dataAddLock
```

Therefore, increasing async RPC concurrency or append workers cannot fully solve this bottleneck. The insert path has a global serialized section before it reaches the more parallel dynamic posting path.

## 10. Most Likely Root Cause

The mutex identified by off-CPU is:

```cpp
Index<T>::m_dataAddLock
```

The problematic lock-held work is:

```cpp
m_extraSearchers[layer]->AddIDCapacity(...)
```

When using `TiKVVersionMap`, that call can synchronously touch TiKV:

```text
TiKVVersionMap::AddBatch
  WriteChunk -> TiKV Put
  SaveCount  -> TiKV Put

TiKVVersionMap::Delete
  WriteVersionByte
    ReadChunk  -> TiKV Get
    WriteChunk -> TiKV Put
```

So the issue is not just that `m_dataAddLock` exists. The issue is that it encloses distributed version-map growth and initialization work.

## 11. Suggested Fix Direction

The first low-risk improvement is to remove the per-VID `Delete` loop from `AddIDCapacity(..., deleted=true)`.

Current shape before optimization:

```cpp
SizeType begin = m_versionMap->Count();
auto ret = m_versionMap->AddBatch(capa);
if (ret == ErrorCode::Success && deleted) {
    for (SizeType i = begin; i < begin + capa; i++) {
        m_versionMap->Delete(i);
    }
}
```

For TiKV version map, that loop turns into many synchronous chunk read-modify-write operations. The implemented improvement is to add a batched version-map API:

```cpp
virtual ErrorCode AddBatch(SizeType num, bool deleted);
```

Then `AddIDCapacity` becomes:

```cpp
return m_versionMap->AddBatch(capa, deleted);
```

For local version map, the default implementation preserves the old behavior. For TiKV version map, `AddBatch(num, true)` initializes the newly added VID range by chunk, so each affected chunk is read/written at most once instead of once per VID.

This does not remove `m_dataAddLock`, but it should reduce the lock hold time sharply for layer 1+ capacity initialization.

The larger follow-up improvement is to shrink the `m_dataAddLock` critical section.

Current shape:

```cpp
{
    std::lock_guard<std::mutex> lock(m_dataAddLock);

    begin = m_options.m_vectorSize;
    end = begin + p_vectorNum;

    for (int layer = 0; layer < m_extraSearchers.size(); layer++) {
        m_extraSearchers[layer]->AddIDCapacity(...);
    }

    ... metadata update ...

    m_options.m_vectorSize = end;
}
```

Better shape:

```cpp
{
    std::lock_guard<std::mutex> lock(m_dataAddLock);

    begin = m_options.m_vectorSize;
    end = begin + p_vectorNum;
    m_options.m_vectorSize = end;

    ... only metadata/ID structures that truly require this global lock ...
}

for (int layer = 0; layer < m_extraSearchers.size(); layer++) {
    m_extraSearchers[layer]->AddIDCapacityForRange(begin, end, layer == 0 ? false : true);
}
```

However, this needs care: `AddIDCapacity` currently computes its own `begin` from `m_versionMap->Count()`. Moving it outside the global lock is only correct if version-map reservation is made range-aware or atomic.

A safer design is to split version-map growth into an explicit reservation API:

```cpp
// Pseudocode
SizeType reservedBegin = m_versionMap->Reserve(num);
assert(reservedBegin == begin);
m_versionMap->InitializeRange(begin, end, deleted);
```

For TiKV-backed version maps, `Reserve` should avoid per-insert synchronous `SaveCount` if possible. It could reserve larger chunks/ranges and flush count less frequently or batch count updates.

## 12. Low-Risk Confirmation Instrumentation

Before changing locking semantics, add timing around the suspected lock:

```cpp
auto waitBegin = std::chrono::high_resolution_clock::now();
std::lock_guard<std::mutex> lock(m_dataAddLock);
auto waitEnd = std::chrono::high_resolution_clock::now();

// record wait_us = waitEnd - waitBegin

auto holdBegin = waitEnd;
...
auto holdEnd = std::chrono::high_resolution_clock::now();

// record hold_us = holdEnd - holdBegin
```

Then separately time:

```cpp
AddIDCapacity(layer 0, false)
AddIDCapacity(layer 1, true)
metadata AddBatch / UpdateMetaMapping
```

Expected result if this analysis is correct:

```text
IndexAddLockWait: very high under 16 insert threads
IndexAddLockHold: includes AddIDCapacity latency
AddIDCapacity layer 1: noticeably more expensive because deleted=true
TiKVVersionMap SaveCount/WriteVersionByte: visible inside hold time
```

## 13. Practical Conclusion

The off-CPU stack alone shows `Index::AddIndex -> __lll_lock_wait`. The source code narrows that wait to `m_dataAddLock` because it is the only plain `std::mutex` acquired in that part of `Index::AddIndex`.

The deeper cause is that this mutex encloses version-map capacity growth. In distributed TiKV mode, version-map growth includes synchronous TiKV `Get` and `Put` operations. That turns a small global ID-allocation lock into a distributed-storage latency amplifier.

For this benchmark, the next meaningful optimization should be around `m_dataAddLock` and `AddIDCapacity`, not only around async posting RPC concurrency.