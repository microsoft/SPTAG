# SPFresh TiKV Async RPC Optimization Analysis, Batch 1-3

Date: 2026-05-08

This note summarizes the recent async RPC optimization run against the previous local version-map baseline. The active asyncopt run is:

- Output: `evaluation/2026-04-23/output_asyncopt_20260508_095357.json`
- Log: `evaluation/2026-04-23/benchmark_asyncopt_20260508_095357.log`
- Baseline output: `evaluation/2026-04-23/output_localvm_20260508_065348.json`
- Baseline log: `evaluation/2026-04-23/benchmark_localvm_20260508_065348.log`

The benchmark configuration is `evaluation/2026-04-23/benchmark_spfresh_sift1b_v10_multichunk.ini` with local version map enabled by `DistributedVersionMap=false`, single-key postings (`UseMultiChunkPosting=false`), `AppendThreadNum=96`, `NumInsertThreads=16`, `ReassignK=64`, and `AsyncRpcMaxInflight=512`.

## Code Changes Under Test

The asyncopt binary includes three relevant changes.

### Region-Batched AddIndex Posting RMW

`AddIndexAsyncSingleKey` no longer issues one async RawGet and one async RawPut per target head. It now batches posting reads and writes by TiKV region:

```cpp
// AnnService/inc/Core/SPANN/ExtraDynamicSearcher.h
tikv->MultiGetWithStatus(keys, &getValues, &getOk, MaxTimeout,
                         &(p_exWorkSpace->m_diskRequests));

tikv->MultiPutWithStatus(putKeys, putValues, &putOk, MaxTimeout,
                         &(p_exWorkSpace->m_diskRequests));
```

The helper methods live in `AnnService/inc/Core/SPANN/ExtraTiKVController.h`:

```cpp
ErrorCode MultiGetWithStatus(const std::vector<std::string>& keys,
                             std::vector<std::string>* values,
                             std::vector<uint8_t>* okFlags,
                             const std::chrono::microseconds& timeout,
                             std::vector<Helper::AsyncReadRequest>* reqs);

ErrorCode MultiPutWithStatus(const std::vector<std::string>& keys,
                             const std::vector<std::string>& values,
                             std::vector<uint8_t>* okFlags,
                             const std::chrono::microseconds& timeout,
                             std::vector<Helper::AsyncReadRequest>* reqs);
```

This preserves per-key fallback behavior while reducing RPC fanout for AddIndex posting RMW.

### Atomic Async RPC Permit Fast Path

`AcquireAsyncRpcPermit` now uses an atomic CAS fast path and only enters the mutex/condition-variable path when inflight RPCs hit the configured limit:

```cpp
while (current < limit) {
    if (m_asyncRpcInflight.compare_exchange_weak(
            current, current + 1,
            std::memory_order_acq_rel,
            std::memory_order_relaxed)) {
        acquiredPermit = true;
        ObserveAsyncRpcInflight(current + 1);
        return true;
    }
}
```

This removes a mutex operation from the normal async RPC submission path.

### Multiple CompletionQueues

The async TiKV client now uses four CompletionQueues and four pump threads:

```cpp
static constexpr int kAsyncCompletionQueueCount = 4;
std::vector<std::unique_ptr<grpc::CompletionQueue>> m_asyncCqs;
std::vector<std::thread> m_asyncPumpThreads;
std::atomic<uint64_t> m_asyncCqNext{0};
```

Startup confirmed the new path:

```text
TiKVIO: Async RPC completion queues=4
```

## Batch-Level Results

### Insert Throughput

| Batch | localvm time (s) | localvm throughput | asyncopt time (s) | asyncopt throughput | Throughput delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 1063.6912 | 940.1225 vec/s | 876.2823 | 1141.1847 vec/s | +21.4% |
| 2 | 1247.5818 | 801.5507 vec/s | 1041.3273 | 960.3129 vec/s | +19.8% |
| 3 | 2277.9395 | 438.9932 vec/s | 1857.6390 | 538.3177 vec/s | +22.6% |

Batch 3 is the key stress point. The previous localvm run collapsed to `438.9932 vec/s`; asyncopt stays at `538.3177 vec/s`, a 22.6% improvement. However, asyncopt still drops from batch 2 to batch 3:

```text
asyncopt batch2 throughput = 960.3129 vec/s
asyncopt batch3 throughput = 538.3177 vec/s
relative drop = -43.9%
```

So the async RPC changes clearly help, but they do not remove the later-batch split/reassign bottleneck.

### Search After Insert

| Batch | localvm post-search mean | asyncopt post-search mean | localvm QPS | asyncopt QPS | Recall delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 7.1652 ms | 5.8343 ms | 557.6965 | 680.6356 | -0.001 |
| 2 | 7.4192 ms | 7.1046 ms | 534.9768 | 560.0735 | -0.006 |
| 3 | 7.9942 ms | 6.5874 ms | 500.0488 | 605.8275 | -0.006 |

Batch 3 post-insert search also improves materially: latency drops by 17.6% and QPS rises by 21.2% versus localvm.

## Diagnostic Comparison

### Batch 2

Batch 2 has similar split/reassign scale between the two runs, so it is a useful controlled comparison.

| Metric | localvm batch2 | asyncopt batch2 | Change |
|---|---:|---:|---:|
| split | 9032 | 9113 | comparable |
| reassign | 705522 | 719342 | comparable |
| triggeredSplit | 9063 | 9136 | comparable |
| AppendLockWait avg | 625.34 us | 294.15 us | -53.0% |
| ReassignJobUs avg | 69143.00 us | 45726.50 us | -33.9% |
| MultiGetPageBuffer avg | 7448.01 us | 5416.20 us | -27.3% |
| AddIndexSingleKeyGet avg | 3689.26 us | 3427.68 us | -7.1% |
| AddIndexSingleKeyPut avg | 2999.43 us | 3100.02 us | +3.4% |
| RpcThrottle wait samples | 80419074 | 6670728 | -91.7% |

Batch 2 shows the cleanest improvement profile: posting MultiGet wait falls, append lock wait falls, reassign job time falls, and throttle sample volume drops by an order of magnitude.

### Batch 3

Batch 3 is where both runs enter the high split/reassign pressure region.

| Metric | localvm batch3 | asyncopt batch3 | Change |
|---|---:|---:|---:|
| split | 85411 | 85895 | comparable |
| reassign | 4160324 | 4200625 | comparable |
| triggeredSplit | 85921 | 86205 | comparable |
| split latency avg | 317.5 ms | 242.6 ms | -23.6% |
| split latency max | 35826.5 ms | 16186.8 ms | -54.8% |
| AppendLockWait avg | 18435.48 us | 5895.96 us | -68.0% |
| SplitLockWait avg | 166.34 us | 133.81 us | -19.6% |
| ReassignJobUs avg | 176626.88 us | 108588.32 us | -38.5% |
| MultiGetPageBuffer avg | 9620.95 us | 8353.33 us | -13.2% |
| AddIndexSingleKeyGet avg | 7773.93 us | 7457.01 us | -4.1% |
| AddIndexSingleKeyPut avg | 8155.84 us | 7824.81 us | -4.1% |
| RpcThrottle wait samples | 142950298 | 28573756 | -80.0% |

The asyncopt run does not eliminate batch 3 pressure, but it reduces the worst effects. The most important gains are lower append lock wait, lower reassign job time, lower split latency, and fewer throttle waits.

## On-CPU Profile Summary

A short on-CPU `perf record` was collected from the active asyncopt run:

```text
evaluation/2026-04-23/profiles/asyncopt_oncpu_20260508_105040.data
```

The dominant child call chain was:

```text
SPDKThreadPool
  -> SplitAsyncJob::exec
  -> ExtraDynamicSearcher::Split
  -> CollectReAssign
  -> RNGSelection
  -> Index::SearchHeadIndex
  -> Index::SearchDiskIndex
  -> ExtraDynamicSearcher::SearchIndex
  -> TiKVIO::MultiGet
  -> AsyncRawBatchGetPrefixed
  -> grpc_call_start_batch / protobuf / tcp_send
```

Top child-attributed frames:

| Frame | CPU share |
|---|---:|
| `SplitAsyncJob -> Split` | 36.97% |
| `CollectReAssign -> RNGSelection` | 33.45% |
| `SearchHeadIndex` | 33.27% |
| `SearchDiskIndex` | 20.11% |
| `ExtraDynamicSearcher::SearchIndex` | 19.52% |
| `TiKVIO::MultiGet` | 14.39% |
| `AsyncRawBatchGetPrefixed` | 11.54% |
| `BKT::Index::SearchIndex` | 9.53% |

Top self-time leaf symbols:

| Symbol | Self CPU |
|---|---:|
| `__memset_evex_unaligned_erms` | 9.14% |
| `ComputeL2Distance_AVX512` | 7.40% |
| `_raw_spin_unlock_irqrestore` / futex wake | 4.68% |
| `_int_malloc` | 3.78% |

Interpretation: CPU is not simply waiting on TiKV. It is also busy in split/reassign search, distance computation, workspace/buffer initialization, protobuf parsing, memory allocation, and completion wakeups.

## Off-CPU / Wait Analysis

True `sched:sched_switch` off-CPU profiling was blocked by tracefs permissions:

```text
No permissions to read /sys/kernel/tracing/events/sched/sched_switch
```

The fallback thread snapshot showed:

```text
threads = 346
335 threads in futex_wait_queue
process CPU ~= 770% - 833%
```

This matches the async wait diagnostics: most waiting threads are blocked on condition variables while async TiKV RPC groups complete.

The largest business wait is `MultiGetPageBuffer`:

```text
batch2: waits=1273826 avg=5416.20us avgBatch=50.65
batch3: waits=2420316 avg=8353.33us avgBatch=50.66
```

`MultiGetPageBuffer` means `SearchIndex` or reassign code is reading a batch of posting lists into page buffers. The TiKV path groups keys by region/leader, fires async RawBatchGet RPCs, then waits for all region groups:

```cpp
batch->Add(static_cast<int>(groups.size()));

for (size_t i = 0; i < groups.size(); i++) {
    AsyncRawBatchGetPrefixed(groups[i].leaderAddr, context, groups[i].keys,
                             &tmpValues, batch, &okFlags[i], timeout);
}

auto waitBegin = std::chrono::high_resolution_clock::now();
batch->Wait();
RecordAsyncWait(AsyncWaitKind::MultiGetPageBuffer, groups.size(), elapsedUs);
```

The average batch fanout is about 50 region groups. Therefore the wait is dominated by a fanout barrier: each caller waits for the slowest region group RPC in that MultiGet.

## Current Bottleneck

The original distributed version-map bottleneck is solved. In batch 3:

```text
lockWaitAvg = 0.56us
lockHoldAvg = 2.42us
addCapL0Avg = 0.13us
addCapUpperAvg = 0.11us
```

`m_dataAddLock` and `AddIDCapacity` are not limiting throughput now.

The current bottleneck is:

```text
Split/reassign growth
  -> CollectReAssign/RNGSelection repeated head and posting searches
  -> SearchIndex posting MultiGet fanout
  -> TiKV RawBatchGet tail latency and client completion waits
  -> AddIndex posting RMW Get/Put latency rises under the same load
```

This is visible in batch 3:

```text
split ~= 86k
reassign ~= 4.2M
MultiGetPageBuffer avg = 8.35ms
AddIndexSingleKeyGet avg = 7.46ms
AddIndexSingleKeyPut avg = 7.82ms
```

## Recommended Next Steps

1. Add per-store/per-region diagnostics for `MultiGetPageBuffer`: group count, key count, returned bytes, max group latency, failed group count, fallback count, and p50/p95/p99 wait.
2. Reduce repeated `RNGSelection -> SearchHeadIndex/SearchDiskIndex -> MultiGet` calls inside `CollectReAssign` by caching or reusing posting reads within a split/reassign job.
3. Consider limiting or staggering split/reassign posting reads so they do not contend directly with AddIndex posting RMW traffic.
4. Optimize client-side allocation/copy in `AsyncBatchGetTag::OnComplete`, especially the per-response `unordered_map` and string moves/copies.
5. Keep `AsyncRpcMaxInflight=512` as the current tested throttle point; the remaining bottleneck is no longer the permit fast path itself.

## Bottom Line

Asyncopt is a real improvement: batch 1-3 insertion throughput improves by roughly 20-23%, and batch 3 avoids the previous localvm collapse. The main remaining bottleneck has shifted to split/reassign read amplification and TiKV posting MultiGet tail latency.
