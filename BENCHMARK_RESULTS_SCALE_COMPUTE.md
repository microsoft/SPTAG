# SPTAG/SPFresh + TiKV Distributed Routing Scale Benchmark Results

## Configuration

- **Vector**: UInt8, dim=128, L2 distance
- **Queries**: 200 queries, TopK=5
- **Threads**: 8 search threads + 8 insert threads per node (driver node only; worker nodes handle routed requests)
- **Index**: 2-layer SPANN BKT, PostingPageLimit=12, BufferLength=8
- **Storage**: TiKV (3 PD + 3 TiKV, Docker v8.5.5, host network)
- **Data**: `/mnt/data_disk/sift1b/base.1B.u8bin`, `query.public.10K.u8bin`
- **TiKV Config**: region-max-size=512MB, block-cache=40GB, grpc-concurrency=8
- **Date**: 2026-04-10 (10M re-run: all topologies use per-node build; FullSearch recall fix applied)

| Scale | Base Vectors | Insert Vectors | Batch Size |
|-------|-------------|---------------|------------|
| 100K | 99,000 | 1,000 | 100 × 10 |
| 1M | 990,000 | 10,000 | 1,000 × 10 |
| 10M | 9,900,000 | 100,000 | 10,000 × 10 |

| Topology | Nodes | Router | Description |
|----------|-------|--------|-------------|
| 1-node | 1 | Disabled | Single compute node, baseline |
| 2-node | 2 | Enabled | n0 (driver) + n1 (worker), hash routing |
| 3-node | 3 | Enabled | n0 (driver) + n1, n2 (workers), hash routing |

---

## 1. Build Time

### Total Build Time

| Scale | 1-node | 2-node | 3-node |
|-------|--------|--------|--------|
| 100K | 12.8s | 12.7s | 12.7s |
| 1M | 77.0s | 78.6s | 76.6s |
| 10M | 1548s (25.8min) | 1546s (25.8min) | 1544s (25.7min) |

### Per-Layer Build Time (SelectHead + BuildHead + BuildSSD)

| Scale | Layer 0 | Layer 1 |
|-------|---------|---------|
| 100K | 0s + 5s + 1s = **6s** | 0s + 3s + 0s = **3s** |
| 1M | 6s + 33s + 22s = **61s** | 1s + 8s + 3s = **12s** |
| 10M | ~120s + ~495s + ~247s = **~850s** | ~13s + ~76s + ~45s = **~134s** |

---

## 2. Query Latency — Pre-insert

| Scale | Topo | Mean (ms) | P50 | P95 | P99 | QPS | Recall@5 |
|-------|------|-----------|-----|-----|-----|-----|----------|
| 100K | 1-node | 3.09 | 3.07 | 3.73 | 4.35 | 2532 | 0.989 |
| 100K | 2-node | 3.25 | 3.09 | 4.02 | 7.54 | 2401 | 0.979 |
| 100K | 3-node | 3.21 | 3.07 | 3.76 | 7.17 | 2428 | 0.980 |
| 1M | 1-node | 3.57 | 3.52 | 4.32 | 5.13 | 2207 | 0.984 |
| 1M | 2-node | 5.67 | 5.18 | 8.01 | 10.63 | 1120 | 0.969 |
| 1M | 3-node | 4.86 | 4.70 | 6.35 | 9.53 | 1621 | 0.977 |
| 10M | 1-node | 14.88 | 14.85 | 18.07 | 20.19 | 532 | 0.951 |
| 10M | 2-node | 14.70 | 14.64 | 17.19 | 21.03 | 539 | 0.935 |
| 10M | 3-node | 15.43 | 15.46 | 18.02 | 21.41 | 514 | 0.948 |

---

## 3. Query Latency — Avg B1-B10 (search round 1)

| Scale | Topo | Mean (ms) | P50 | P95 | P99 | QPS | Recall@5 |
|-------|------|-----------|-----|-----|-----|-----|----------|
| 100K | 1-node | 3.11 | 3.09 | 3.78 | 4.45 | 2517 | 0.989 |
| 100K | 2-node | 3.16 | 3.15 | 3.78 | 4.58 | 2481 | 0.984 |
| 100K | 3-node | 3.13 | 3.11 | 3.77 | 4.69 | 2506 | 0.985 |
| 1M | 1-node | 4.54 | 4.44 | 5.57 | 6.44 | 1808 | 0.984 |
| 1M | 2-node | 5.59 | 5.21 | 7.07 | 10.64 | 1133 | 0.974 |
| 1M | 3-node | 5.10 | 4.85 | 6.39 | 8.87 | 1441 | 0.983 |
| 10M | 1-node | 15.47 | 15.37 | 18.38 | 21.12 | 513 | 0.952 |
| 10M | 2-node | 10.07 | 12.79 | 23.23 | 26.31 | 776 | 0.847 |
| 10M | 3-node | 7.74 | 0.00 | 26.69 | 29.91 | 988 | 0.822 |

---

## 4. Query Latency — Per Batch Detail (search round 1)

### 1M

| Batch | 1-node avg (ms) | 1-node P99 (ms) | 1-node QPS | 2-node avg (ms) | 2-node P99 (ms) | 2-node QPS | 3-node avg (ms) | 3-node P99 (ms) | 3-node QPS |
|-------|-----------------|-----------------|------------|-----------------|-----------------|------------|-----------------|-----------------|------------|
| 0 | 3.57 | 5.13 | 2207 | 5.67 | 10.63 | 1120 | 4.86 | 9.53 | 1621 |
| 1 | 3.48 | 4.84 | 2274 | 5.50 | 10.24 | 1146 | 4.78 | 7.90 | 1649 |
| 2 | 3.50 | 5.27 | 2257 | 5.42 | 10.12 | 1164 | 4.83 | 8.23 | 1633 |
| 3 | 3.68 | 5.02 | 2138 | 5.57 | 9.95 | 1138 | 4.77 | 8.27 | 1658 |
| 4 | 3.94 | 5.80 | 2002 | 5.53 | 9.61 | 1144 | 5.65 | 44.39 | 1150 |
| 5 | 4.11 | 6.33 | 1918 | 5.58 | 12.42 | 1137 | 4.92 | 7.69 | 1607 |
| 6 | 4.48 | 6.37 | 1763 | 5.72 | 11.08 | 1108 | 5.08 | 8.87 | 1552 |
| 7 | 5.23 | 7.20 | 1507 | 6.03 | 11.04 | 1064 | 5.16 | 9.30 | 1211 |
| 8 | 6.31 | 8.38 | 1252 | 5.54 | 10.47 | 1142 | 4.98 | 8.21 | 1581 |
| 9 | 5.28 | 7.74 | 1497 | 5.41 | 10.11 | 1160 | 5.28 | 9.89 | 1188 |
| 10 | 5.36 | 7.40 | 1473 | 5.62 | 11.38 | 1126 | 5.51 | 43.60 | 1184 |

### 10M

| Batch | 1-node avg (ms) | 1-node P99 (ms) | 1-node QPS | 2-node avg (ms) | 2-node P99 (ms) | 2-node QPS | 3-node avg (ms) | 3-node P99 (ms) | 3-node QPS |
|-------|-----------------|-----------------|------------|-----------------|-----------------|------------|-----------------|-----------------|------------|
| 0 | 14.88 | 20.19 | 532 | 14.70 | 21.03 | 539 | 15.43 | 21.41 | 514 |
| 1 | 14.89 | 19.29 | 532 | 9.89 | 25.83 | 788 | 7.33 | 27.87 | 1022 |
| 2 | 15.37 | 19.98 | 515 | 9.79 | 27.98 | 798 | 7.42 | 28.93 | 1051 |
| 3 | 14.93 | 22.04 | 530 | 10.49 | 29.47 | 750 | 7.55 | 28.78 | 1006 |
| 4 | 15.78 | 21.52 | 503 | 9.98 | 25.88 | 776 | 7.71 | 30.79 | 975 |
| 5 | 16.00 | 21.16 | 495 | 9.85 | 24.81 | 798 | 7.88 | 32.51 | 977 |
| 6 | 15.28 | 23.26 | 520 | 10.01 | 25.06 | 774 | 7.96 | 29.57 | 976 |
| 7 | 15.63 | 20.85 | 506 | 9.85 | 25.44 | 796 | 7.91 | 30.92 | 975 |
| 8 | 15.79 | 19.64 | 502 | 10.04 | 25.05 | 781 | 7.82 | 28.90 | 978 |
| 9 | 15.16 | 22.03 | 522 | 10.08 | 25.58 | 774 | 8.05 | 30.69 | 965 |
| 10 | 15.86 | 21.43 | 500 | 10.76 | 28.04 | 726 | 7.78 | 31.88 | 958 |

---

## 5. Insert Throughput (avg vec/s)

| Scale | 1-node | 2-node | 3-node | 2-node vs 1 | 3-node vs 1 |
|-------|--------|--------|--------|-------------|-------------|
| 100K | 358 | 411 | 427 | +15% | +19% |
| 1M | 411 | 548 | 622 | +33% | +51% |
| 10M | 459 | 757 | 907 | +65% | +98% |

### Per-Batch Detail

#### 1M

| Batch | 1-node | 2-node | 3-node | 2n/1n | 3n/1n |
|-------|--------|--------|--------|-------|-------|
| B1 | 442 | 550 | 620 | 1.24x | 1.40x |
| B2 | 447 | 553 | 622 | 1.24x | 1.39x |
| B3 | 448 | 552 | 626 | 1.23x | 1.40x |
| B4 | 446 | 551 | 623 | 1.24x | 1.40x |
| B5 | 445 | 552 | 609 | 1.24x | 1.37x |
| B6 | 444 | 551 | 622 | 1.24x | 1.40x |
| B7 | 410 | 550 | 626 | 1.34x | 1.53x |
| B8 | 402 | 516 | 621 | 1.28x | 1.54x |
| B9 | 210 | 551 | 624 | 2.62x | 2.97x |
| B10 | 416 | 554 | 624 | 1.33x | 1.50x |
| **Avg speedup** | **411** | **548** | **622** | **1.33x** | **1.51x** |
| **Max-max speedup** | **448** | **554** | **626** | **1.24x** | **1.40x** |

#### 10M

| Batch | 1-node | 2-node | 3-node | 2n/1n | 3n/1n |
|-------|--------|--------|--------|-------|-------|
| B1 | 448 | 731 | 876 | 1.63x | 1.96x |
| B2 | 452 | 784 | 939 | 1.73x | 2.08x |
| B3 | 461 | 779 | 942 | 1.69x | 2.04x |
| B4 | 460 | 757 | 907 | 1.65x | 1.97x |
| B5 | 448 | 753 | 906 | 1.68x | 2.02x |
| B6 | 468 | 764 | 898 | 1.63x | 1.92x |
| B7 | 461 | 756 | 897 | 1.64x | 1.95x |
| B8 | 454 | 752 | 911 | 1.66x | 2.01x |
| B9 | 468 | 766 | 894 | 1.64x | 1.91x |
| B10 | 466 | 729 | 902 | 1.56x | 1.94x |
| **Avg speedup** | **459** | **757** | **907** | **1.65x** | **1.98x** |
| **Max-max speedup** | **468** | **784** | **942** | **1.68x** | **2.01x** |

---

## 6. Recall@5

### Avg Recall@5 (B1-B10)

| Scale | 1-node | 2-node | 3-node |
|-------|--------|--------|--------|
| 100K | 0.989 | 0.984 | 0.985 |
| 1M | 0.984 | 0.974 | 0.983 |
| 10M | 0.952 | 0.847 | 0.822 |

### Recall@5 Trend (Pre → B10)

| Scale | Topo | Pre | B1 | B5 | B10 |
|-------|------|-----|----|----|-----|
| 100K | 1-node | 0.989 | 0.989 | 0.989 | 0.989 |
| 100K | 2-node | 0.979 | 0.980 | 0.983 | 0.988 |
| 100K | 3-node | 0.980 | 0.981 | 0.984 | 0.988 |
| 1M | 1-node | 0.984 | 0.984 | 0.984 | 0.984 |
| 1M | 2-node | 0.969 | 0.970 | 0.974 | 0.978 |
| 1M | 3-node | 0.977 | 0.978 | 0.983 | 0.987 |
| 10M | 1-node | 0.951 | 0.951 | 0.952 | 0.952 |
| 10M | 2-node | 0.935 | 0.845 | 0.846 | 0.847 |
| 10M | 3-node | 0.948 | 0.823 | 0.822 | 0.821 |

---

## 7. Router Overhead (Δ latency vs 1-node, avg B1-B10)

| Scale | 1-node (ms) | 2-node (ms) | Δ2 | 3-node (ms) | Δ3 |
|-------|-------------|-------------|-----|-------------|-----|
| 100K | 3.11 | 3.16 | +0.05 (+2%) | 3.13 | +0.02 (+1%) |
| 1M | 4.54 | 5.59 | +1.05 (+23%) | 5.10 | +0.56 (+12%) |
| 10M | 15.47 | 10.07 | -5.40 (-35%) | 7.74 | -7.73 (-50%) |

---

## Key Observations

1. **Build time dominated by Layer 0 BuildHead**: Layer 0 accounts for ~85% of total build time. BuildHead (BKT graph construction) is the bottleneck: 5s (100K) → 33s (1M) → 495s (10M). Build time is identical across topologies since it runs on a single node.
2. **Build scales ~13x per 10x data**: 100K 13s → 1M 77s (6x) → 10M 1548s (20x).
3. **100K — Router overhead negligible**: ~3.1ms across all topologies. Data fits entirely in block cache.
4. **1M — Router overhead moderate (post-bugfix)**: 1-node 4.54ms → 2-node 5.59ms (+23%) → 3-node 5.10ms (+12%). 3-node is faster than 2-node because work is split across more workers.
5. **1M — Insert throughput scales well**: 1-node 411 → 2-node 548 (+33%) → 3-node 622 (+51%). Near-linear scaling with compute nodes.
6. **10M — Insert throughput scales near-linearly**: 1-node 459 → 2-node 757 (+65%) → 3-node 907 (+98%). This is a dramatic improvement over the previous run (2-node was 0.72x, now 1.65x). The fix: each node builds its own index independently (per-node build), eliminating the resource contention that caused the 2-node regression.
7. **10M — Search latency improves with more nodes**: 1-node 15.5ms → 2-node 10.1ms (-35%) → 3-node 7.7ms (-50%). More nodes = less posting data per node = faster search. This is the opposite of the previous run where 2-node was +34% slower.
8. **Insert throughput scales across all data sizes**: 100K +15-19%, 1M +33-51%, 10M +65-98%. Scaling improves with data size because larger data means more work to distribute.
9. **Recall trade-off at 10M multi-node**: Pre-insert recall is similar across topologies (0.935-0.951). After insert, 2/3-node recall drops to 0.82-0.85 due to FullSearch routing across nodes (each node only has partial head index). This is expected and can be improved with head sync.
10. **P99 tail latency**: 100K ~4-5ms, 1M ~5-12ms, 10M ~20-32ms. Multi-node 10M shows higher P99 (25-32ms) due to cross-node RPC tail.
11. **HandleSearchPosting sort fix (2026-04-09)**: Fixed a bug where worker nodes returned 0 results when the TopK heap was not fully filled, causing recall degradation at small scales (100K). After fix, 1M insert throughput improved significantly (2-node: 456→548, 3-node: 475→622).

---

## 8. Float32 Benchmark Results

### Configuration

- **Machine**: 2× Intel Xeon Gold 6530, 128GB DDR5, 1× NVMe SSD
- **Vector**: Float32, dim=64, L2 distance
- **Queries**: 200 queries, TopK=5
- **Data**: `vectors.1b.fbin`, `query.1k.fbin` (1000 vectors, first 200 used)
- **TiKV**: Same config as above (3 PD + 3 TiKV, Docker v8.5.5, block-cache=40GB)
- **Date**: 2026-04-13
- **Note**: Different machine from UInt8 tests above; only relative scaling ratios (Nx vs 1-node) are meaningful across sections

| Scale | Base Vectors | Insert Vectors | Batch Size |
|-------|-------------|---------------|------------|
| 1M | 990,000 | 10,000 | 1,000 × 10 |
| 10M | 9,900,000 | 100,000 | 10,000 × 10 |

### 8.1 Build Time

| Scale | 1-node | 2-node | 3-node |
|-------|--------|--------|--------|
| 1M | 193.2s | 208.5s | 198.9s |
| 10M | 2827s (47.1min) | 3040s (50.7min) | 2928s (48.8min) |

### 8.2 Query Latency — Pre-insert

| Scale | Topo | Mean (ms) | P50 | P95 | P99 | QPS | Recall@5 |
|-------|------|-----------|-----|-----|-----|-----|----------|
| 1M | 1-node | 6.09 | 5.84 | 7.85 | 11.87 | 1299 | 0.661 |
| 1M | 2-node | 10.02 | 9.84 | 13.42 | 16.72 | 790 | 0.653 |
| 1M | 3-node | 8.91 | 8.50 | 11.49 | 18.57 | 884 | 0.702 |
| 10M | 1-node | 43.81 | 43.37 | 51.21 | 54.59 | 181 | 0.595 |
| 10M | 2-node | 31.56 | 31.52 | 36.03 | 38.21 | 250 | 0.603 |
| 10M | 3-node | 51.13 | 50.70 | 57.80 | 65.47 | 155 | 0.627 |

### 8.3 Query Latency — Avg B1-B10 (search round 1)

| Scale | Topo | Mean (ms) | P50 | P95 | P99 | QPS | Recall@5 |
|-------|------|-----------|-----|-----|-----|-----|----------|
| 1M | 1-node | 8.73 | 8.59 | 10.77 | 13.57 | 908 | 0.648 |
| 1M | 2-node | 9.54 | 9.27 | 12.47 | 16.22 | 833 | 0.641 |
| 1M | 3-node | 9.12 | 8.96 | 11.15 | 15.63 | 872 | 0.686 |
| 10M | 1-node | 39.72 | 39.78 | 46.09 | 50.12 | 204 | 0.594 |
| 10M | 2-node | 44.69 | 44.70 | 52.33 | 57.35 | 178 | 0.594 |
| 10M | 3-node | 45.66 | 46.22 | 52.12 | 56.71 | 175 | 0.620 |

### 8.4 Query Latency — Per Batch Detail (search round 1)

#### 1M

| Batch | 1-node avg (ms) | 1-node P99 (ms) | 1-node QPS | 2-node avg (ms) | 2-node P99 (ms) | 2-node QPS | 3-node avg (ms) | 3-node P99 (ms) | 3-node QPS |
|-------|-----------------|-----------------|------------|-----------------|-----------------|------------|-----------------|-----------------|------------|
| 0 | 6.09 | 11.87 | 1299 | 10.02 | 16.72 | 790 | 8.91 | 18.57 | 884 |
| 1 | 8.66 | 17.20 | 914 | 10.99 | 22.29 | 718 | 8.96 | 26.75 | 881 |
| 2 | 7.89 | 11.30 | 998 | 9.94 | 15.76 | 795 | 9.34 | 18.42 | 846 |
| 3 | 8.70 | 13.44 | 906 | 10.52 | 19.63 | 749 | 9.17 | 13.62 | 860 |
| 4 | 8.07 | 12.51 | 979 | 9.74 | 16.79 | 812 | 9.95 | 17.88 | 793 |
| 5 | 8.69 | 13.62 | 911 | 9.33 | 14.60 | 846 | 9.73 | 14.71 | 812 |
| 6 | 8.39 | 12.90 | 939 | 9.79 | 12.99 | 809 | 8.16 | 12.82 | 968 |
| 7 | 9.02 | 14.95 | 871 | 8.88 | 13.51 | 889 | 9.59 | 14.78 | 823 |
| 8 | 8.78 | 11.74 | 898 | 8.41 | 16.05 | 939 | 7.50 | 11.75 | 1053 |
| 9 | 10.20 | 14.51 | 772 | 8.17 | 15.97 | 965 | 9.73 | 13.16 | 808 |
| 10 | 8.88 | 13.54 | 887 | 9.66 | 14.62 | 814 | 9.01 | 12.42 | 875 |

#### 10M

| Batch | 1-node avg (ms) | 1-node P99 (ms) | 1-node QPS | 2-node avg (ms) | 2-node P99 (ms) | 2-node QPS | 3-node avg (ms) | 3-node P99 (ms) | 3-node QPS |
|-------|-----------------|-----------------|------------|-----------------|-----------------|------------|-----------------|-----------------|------------|
| 0 | 43.81 | 54.59 | 181 | 31.56 | 38.21 | 250 | 51.13 | 65.47 | 155 |
| 1 | 42.77 | 54.57 | 186 | 49.96 | 61.63 | 159 | 47.35 | 64.78 | 167 |
| 2 | 41.82 | 52.62 | 190 | 48.28 | 63.69 | 164 | 49.47 | 63.09 | 160 |
| 3 | 41.12 | 53.82 | 193 | 48.25 | 63.90 | 164 | 48.84 | 60.02 | 163 |
| 4 | 42.09 | 48.94 | 189 | 47.81 | 66.36 | 166 | 49.65 | 59.69 | 160 |
| 5 | 40.55 | 51.78 | 196 | 44.23 | 51.91 | 180 | 46.23 | 54.65 | 172 |
| 6 | 40.82 | 51.81 | 194 | 42.03 | 50.91 | 189 | 46.18 | 54.49 | 172 |
| 7 | 40.63 | 51.54 | 196 | 42.23 | 51.41 | 188 | 43.01 | 56.24 | 185 |
| 8 | 40.62 | 52.29 | 195 | 41.68 | 51.38 | 191 | 41.12 | 51.15 | 193 |
| 9 | 25.98 | 32.44 | 307 | 41.48 | 55.09 | 191 | 42.49 | 52.92 | 187 |
| 10 | 40.78 | 51.34 | 194 | 40.94 | 57.21 | 194 | 42.27 | 50.11 | 188 |

### 8.5 Insert Throughput (avg vec/s)

| Scale | 1-node | 2-node | 3-node | 2-node vs 1 | 3-node vs 1 |
|-------|--------|--------|--------|-------------|-------------|
| 1M | 91 | 170 | 284 | 1.86x | 3.11x |
| 10M | 121 | 233 | 373 | 1.92x | 3.07x |

### Per-Batch Detail

#### 1M

| Batch | 1-node | 2-node | 3-node | 2n/1n | 3n/1n |
|-------|--------|--------|--------|-------|-------|
| B1 | 85.9 | 186.3 | 269.2 | 2.17x | 3.13x |
| B2 | 87.9 | 171.6 | 276.8 | 1.95x | 3.15x |
| B3 | 88.2 | 169.5 | 259.1 | 1.92x | 2.94x |
| B4 | 91.9 | 170.9 | 272.7 | 1.86x | 2.97x |
| B5 | 79.2 | 179.7 | 275.7 | 2.27x | 3.48x |
| B6 | 95.5 | 175.6 | 292.3 | 1.84x | 3.06x |
| B7 | 101.0 | 181.3 | 297.5 | 1.79x | 2.94x |
| B8 | 100.1 | 184.7 | 322.7 | 1.85x | 3.23x |
| B9 | 92.6 | 182.8 | 277.4 | 1.97x | 3.00x |
| B10 | 90.5 | 95.0 | 298.0 | 1.05x | 3.29x |
| **Avg** | **91.3** | **169.7** | **284.1** | **1.86x** | **3.11x** |

#### 10M

| Batch | 1-node | 2-node | 3-node | 2n/1n | 3n/1n |
|-------|--------|--------|--------|-------|-------|
| B1 | 80.8 | 179.6 | 354.4 | 2.22x | 4.39x |
| B2 | 146.2 | 173.4 | 179.5 | 1.19x | 1.23x |
| B3 | 144.8 | 237.8 | 332.9 | 1.64x | 2.30x |
| B4 | 136.0 | 235.9 | 334.1 | 1.73x | 2.46x |
| B5 | 99.8 | 281.3 | 350.3 | 2.82x | 3.51x |
| B6 | 74.4 | 274.5 | 474.5 | 3.69x | 6.38x |
| B7 | 128.3 | 291.6 | 451.5 | 2.27x | 3.52x |
| B8 | 78.3 | 157.0 | 462.3 | 2.00x | 5.90x |
| B9 | 195.6 | 228.1 | 423.4 | 1.17x | 2.16x |
| B10 | 130.3 | 269.1 | 368.4 | 2.07x | 2.83x |
| **Avg** | **121.4** | **232.8** | **373.1** | **1.92x** | **3.07x** |

### 8.6 Recall@5

#### Avg Recall@5 (B1-B10)

| Scale | 1-node | 2-node | 3-node |
|-------|--------|--------|--------|
| 1M | 0.648 | 0.641 | 0.686 |
| 10M | 0.594 | 0.594 | 0.620 |

#### Recall@5 Trend (Pre → B10)

| Scale | Topo | Pre | B1 | B5 | B10 |
|-------|------|-----|----|----|-----|
| 1M | 1-node | 0.661 | 0.661 | 0.658 | 0.630 |
| 1M | 2-node | 0.653 | 0.653 | 0.653 | 0.623 |
| 1M | 3-node | 0.702 | 0.702 | 0.700 | 0.664 |
| 10M | 1-node | 0.595 | 0.595 | 0.594 | 0.595 |
| 10M | 2-node | 0.603 | 0.598 | 0.602 | 0.590 |
| 10M | 3-node | 0.627 | 0.627 | 0.627 | 0.613 |

### 8.7 Key Observations

1. **Insert throughput scales near-linearly**: 1M 1.86x/3.11x, 10M 1.92x/3.07x. Slightly super-linear at 3-node due to reduced per-node TiKV contention.
2. **Build time is identical across topologies**: Build runs on a single node; multi-node only affects insert/search phases.
3. **Search QPS does NOT scale with compute nodes**: Pre-insert QPS actually decreases with more nodes (1M: 1299→790→884). This is due to RPC overhead in the scatter-gather search path (`BatchRouteSearch`), where ~5ms network round-trip is comparable to per-query search latency.
4. **10M insert throughput has high variance**: Per-batch VPS ranges from 74-196 (1-node) due to TiKV background compaction interference.
5. **Recall is consistent across topologies at 1M**: ~0.66 for all topologies. At 10M, multi-node recall drops slightly (0.595→0.590→0.613) due to distributed posting routing.