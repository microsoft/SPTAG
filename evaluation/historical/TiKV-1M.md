# Historical TiKV observations: 1M scale

> Historical observations, not current-branch acceptance thresholds.
> For current evidence, read the [complete 2026-10-28 SIFT1M results](../2026-10-28-sift1m/README.md).
> See the [evaluation index](../README.md) and
> [reporting requirements](../../docs/Benchmarking.md).

The [developer investigation](../2026-10-23-sift1m/development/RecallRegression.md)
identified and fixed a TiKV result-heap ordering bug.
The revisions used below are unknown, so its impact on these historical runs
cannot be established. These recall numbers are not evidence of an inherent
TiKV accuracy limit, and have not been replaced with measurements from a
different index or workload.

## Provenance and limitations

| Field | Recorded status |
| --- | --- |
| SPTAG / TiKV commits and dirty state | Not recorded |
| PD/TiKV immutable image digests | Not recorded |
| Exact toolchain / hardware / replication factor | Not fully recorded |
| Input / truth / configuration checksums | Not recorded |
| Raw per-run logs / JSON | No artifact location recorded here |
| Topology | Three PD and three TiKV processes on one machine, not three independent hosts |

Do not guess these values retroactively. Preserve the observations below as
historical data; rerun a pinned configuration before making a current performance
claim. The scale summary below is ambiguous: `InsertVectorCount` in the current
harness is a total across all batches, so a configuration file is needed to
resolve whether "10K per batch" or "10K total" was intended.

## Configuration

- **Scale**: 1M vectors (base: 990K, insert: 10K per batch x 10)
- **Dimensions**: 128 (UInt8)
- **Distance**: L2
- **Index**: SPANN with BKT, 2 layers
- **TopK**: 5
- **Query count**: 200
- **Search threads**: 16
- **PostingPageLimit**: 12, BufferLength: 8
- **LatencyLimit**: 50ms
- **SearchInternalResultNum**: 64

## TiKV Cluster

- 3 PD nodes + 3 TiKV nodes (Docker, host network, single machine)
- region-max-size: 512MB, region-split-size: 384MB
- grpc-concurrency: 8

## Optimization: RawBatchGet

Replaced per-key `RawGet` calls with `RawBatchGet` that groups keys by TiKV region
and dispatches parallel requests per region group. Includes region cache with automatic
invalidation. Fallback to individual Get on failure.

At 100K scale: **7-10x improvement** (56ms -> 7.6ms mean latency)

## 1M Results - Three Independent Runs

### Latency Comparison (mean, ms)

| Batch | Run1 | Run2 | Run3 |
|-------|------|------|------|
| Pre | 65.8 | 61.5 | 64.0 |
| B1 | 60.9 | 61.1 | 63.2 |
| B2 | 70.5 | 60.0 | 66.3 |
| B3 | 62.7 | 67.6 | 67.2 |
| B4 | 62.8 | 66.2 | 55.0 |
| B5 | 43.7 | 64.0 | 56.3 |
| B6 | 44.6 | 63.5 | 56.2 |
| B7 | 45.6 | 41.6 | 57.1 |
| B8 | 41.8 | 41.3 | 57.5 |
| B9 | 41.2 | 41.6 | 58.1 |
| B10 | 41.6 | 45.3 | 55.6 |
| **B1-10 Avg** | **51.5** | **55.2** | **59.2** |

### Recall@5 Comparison

| Batch | Run1 | Run2 | Run3 |
|-------|------|------|------|
| Pre | 0.796 | 0.800 | 0.787 |
| B1 | 0.791 | 0.797 | 0.782 |
| B2 | 0.788 | 0.729 | 0.780 |
| B3 | 0.783 | 0.790 | 0.775 |
| B4 | 0.779 | 0.785 | 0.770 |
| B5 | 0.777 | 0.784 | 0.769 |
| B6 | 0.775 | 0.780 | 0.766 |
| B7 | 0.773 | 0.778 | 0.764 |
| B8 | 0.769 | 0.772 | 0.758 |
| B9 | 0.766 | 0.769 | 0.755 |
| B10 | 0.763 | 0.767 | 0.754 |

### Run3 Latency Percentiles (ms)

| Batch | Mean | P50 | P95 | P99 | Recall@5 |
|-------|------|-----|-----|-----|----------|
| Pre | 64.0 | 65.1 | 69.4 | 70.2 | 0.787 |
| B1 | 63.2 | 63.8 | 68.0 | 68.7 | 0.782 |
| B2 | 66.3 | 67.8 | 70.4 | 71.7 | 0.780 |
| B3 | 67.2 | 68.1 | 71.5 | 72.7 | 0.775 |
| B4 | 55.0 | 55.4 | 62.6 | 65.8 | 0.770 |
| B5 | 56.3 | 56.5 | 63.4 | 65.2 | 0.769 |
| B6 | 56.2 | 57.1 | 62.4 | 64.4 | 0.766 |
| B7 | 57.1 | 57.4 | 64.7 | 68.0 | 0.764 |
| B8 | 57.5 | 58.0 | 65.2 | 67.5 | 0.758 |
| B9 | 58.1 | 58.2 | 65.5 | 70.1 | 0.755 |
| B10 | 55.6 | 56.1 | 63.1 | 65.3 | 0.754 |

### Build Time

- Run1 (original): 986s
- Run2 (reverted): 982s
- Run3 (fresh): 991s

## Summary

- **Steady-state latency (B4-B10)**: ~55-58ms mean
- **P95 tail latency**: 62-72ms
- **Recall@5**: 0.754-0.787 (degrades slightly with inserts)
- **Build time**: ~985-991s (~16.5 min)

### Original interpretations (not revalidated)

1. Each search query requires 2 serial gRPC round trips (Layer 1 head-of-heads + Layer 0 main postings)
2. Each round trip fetches up to 64 keys via RawBatchGet grouped by region
3. Latency is dominated by network round-trip time (~25-30ms per RPC)
4. Block cache warmup effects visible in first 3-4 batches
5. Channel Pool (8 channels/store) was tested and found to ADD latency (~+7ms), reverted
6. Region balance after data load is suboptimal (PD scheduler too slow for bulk writes)
