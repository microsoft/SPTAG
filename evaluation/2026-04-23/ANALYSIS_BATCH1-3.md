# SPFresh + TiKV SIFT1B Insert — Batch 1–3 性能分析

**日期**: 2026-04-24
**配置**: `benchmark_spfresh_sift1b_v10_multichunk.ini` (`UseMultiChunkPosting=false`,
`NumInsertThreads=16`, `AppendThreadNum=48`, `ReassignK=64`, single-chunk RMW 路径)
**集群**: 3 PD + 3 TiKV (host network), tikv.toml = nightly default
**二进制**: 含 commit 227425e (CollectReAssign nearby snapshot fix) + DIAG 直方图埋点

---

## 1. 客户端 DIAG 增量 (从累计直方图做差)

DIAG 直方图是**累计的**，必须做差才能看单 batch 真实情况。

### Layer 0 RMW

| 指标                | Batch 1   | Batch 2 Δ  | Batch 3 Δ   | B2/B1 | B3/B2 |
| ------------------- | --------- | ---------- | ----------- | ----- | ----- |
| Append count        | 9.14M     | 4.97M      | **2.76M**   | 0.54× | 0.55× |
| Append/insert       | 9.14      | 4.97       | **2.76**    | ↓     | ↓     |
| AppendGetUs avg     | 474 µs    | 2,599 µs   | **7,102 µs**| 5.5×  | 2.7×  |
| AppendPutUs avg     | 957 µs    | 3,286 µs   | **8,385 µs**| 3.4×  | 2.6×  |
| AppendPostBytes avg | 8.83 KB   | 18.1 KB    | **16.3 KB** | 2.0×  | 0.9× ↓|
| 单次 RMW = G+P      | 1.43 ms   | 5.89 ms    | **15.5 ms** | 4.1×  | 2.6×  |
| Split Δ             | 77k       | 424k       | **296k**    | 5.4×  | 0.7×  |
| AppendLockWait avg  | 1.68 µs   | 2.20 µs    | 5.38 µs     | trivial | trivial |
| SplitLockWait avg   | 3.90 µs   | 2.48 µs    | 4.89 µs     | trivial | trivial |

### Layer 1 (B3 首次活跃)

| Layer 1 batch 3 | 数值 |
| --- | --- |
| Append count | **1.10M** |
| Get avg | 2,254 µs |
| Put avg | 2,986 µs |
| PostBytes avg | 10.97 KB |
| Split Δ | 24,297 |
| AppendLockWait avg | **58.3 µs** ← 首次离开 trivial 区 |
| SplitLockWait avg | **59.0 µs** ← 同上 |

### 三个关键现象

1. **Bytes 已稳定，但 RMW 仍翻倍变慢**。Batch 3 layer-0 PostBytes 反而下降到 16.3KB，
   但 Get/Put 各自又 2.6×。说明字节膨胀**已不再是主因**；下一层瓶颈在 TiKV 路径本身。
2. **Layer 1 在 Batch 3 爆发**。Layer 0 split 累计 ~80 万 → 触发父层 BKTree 重构，
   产生整套 Layer 1 RMW；layer-1 head 数少，单 head 锁更热（lock avg 58µs）。
3. **Append/insert 比值持续下降** (9.14 → 4.97 → 2.76)。reassign 链路在收敛，
   但单次 RMW 成本爆涨 11× 抵消了次数减少。

### 单 batch 时间估算 (16 线程并行)

| Batch | layer-0 串行 | layer-1 串行 | 16 线程并行预估 |
| ----- | ------------ | ------------ | --------------- |
| 1     | 9.14M × 1.43ms = 13,070s | ~0 | **13.6 min** |
| 2     | 4.97M × 5.89ms = 29,300s | ~0 | **30.5 min** |
| 3     | 2.76M × 15.5ms = 42,800s | 1.10M × 5.24ms = 5,760s | **50.6 min** |

按这个趋势 batch 4-10 单 batch 时间会突破 1 小时。

---

## 2. TiKV 服务端状态 (与客户端 DIAG 同时刻采样)

### 2.1 集群 / region 健康

- **78 个 region**, 27/27/24 leader 分布 (PD 完全健康)
- 每节点 5.46 TiB available
- 0 pending compaction (nodes 1/3); node 2 仅 738 MB (trivial)
- LSM 状态: L0-L4 全空, L5 ~43 SST, L6 ~46 SST → 写路径无积压
- **0 write stall** 跨所有节点和 cf
- 0 raft apply 异常 (`apply_log_duration_count`=49.5M, sum=4905s → avg 0.099 ms)

### 2.2 服务端单 op 平均延迟 (cumulative)

| 节点    | raw_get avg | raw_put avg | raw_batch_get avg |
| ------- | ----------- | ----------- | ----------------- |
| :20181  | 0.19 ms     | 0.67 ms     | 0.23 ms           |
| :20182  | 0.38 ms     | 0.90 ms     | 0.45 ms           |
| :20183  | 0.18 ms     | 0.71 ms     | 0.21 ms           |

服务端处理一次 raw_get/raw_put **不到 1 ms**, 完全健康。

### 2.3 流量分布

| 操作 | 累计请求数 (3 节点合计) | 占比 |
| --- | --- | --- |
| raw_batch_get | 538M | **90%** |
| raw_put       | 58M  | 10%  |
| raw_get       | 52M  | 9%   |
| raw_delete    | 4.8k | -    |

**raw_batch_get 占绝对主导** —— 来自 search 路径 + CollectReAssign nearby scan
的 MultiGet, 不是 RMW 流量本身。它们与 RMW 的 single-key get/put 在同一 client 通道
上排队。

### 2.4 服务端 CPU 利用率

apply 线程: 4 thread × ~1420 s / 进程 etime 16,040 s = **~8.8 % per thread**,
4 个 apply 线程合计仅 **~35 % 1 核**.

进程 ps `%cpu` 显示 ~350 % (3.5 cores) per tikv-server, 主要被 grpc_server_*
线程消耗。**服务端整体远未饱和**。

---

## 3. 客户端 vs 服务端对比 — 90% 延迟在 client side

Batch 3 末尾对比:

| 操作 | DIAG 客户端 avg | 服务端 avg | **差值** |
| --- | --- | --- | --- |
| AppendGetUs (single Get) | **7.10 ms** | 0.18 - 0.38 ms | **18 - 37×** |
| AppendPutUs (single Put) | **8.39 ms** | 0.67 - 0.90 ms | **9 - 12×**  |

对比 batch 1: 客户端 Get=474µs, 服务端 ~200µs, 差距仅 ~2×。
**3 个 batch 间客户端开销膨胀了 ~15×, 服务端几乎不变**。

---

## 4. 三大假设最终裁决

| 假设 | 结论 | 依据 |
| --- | --- | --- |
| **A. 代码内锁 (m_rwLocks) 竞争** | ❌ 排除 | Append/Split lock avg 全程 < 5 µs (layer 0); layer 1 在 B3 出现 58 µs 但非主导 |
| **B. RMW posting 字节膨胀** | ❌ 不再是主因 | Bytes 已稳定 12-18 KB; 但 latency 仍线性上涨 |
| **C. TiKV client 侧 overhead** | ✅ **实锤** | 客户端 vs 服务端有 10-37× 差距, 且差距随时间扩大 |

可能的 client 侧瓶颈机制:
- gRPC connection pool 不足 (16 insert threads + 1 search loop 抢 client 内部 mutex)
- region cache lookup / batch scheduling 在高并发下排队
- batch_get 占 90% 流量, 与 single-key RMW 在同一 channel 上互相阻塞

---

## 5. 后续行动建议

1. **Profile TiKV C++ client** 内部锁与 grpc completion queue (perf top / off-cpu)
2. **检查 `client/tikv-client` 的 connection pool 大小**:
   - `AnnService/inc/Helper/TikvClient*.h` / `ExtraTiKVController.h` 里的 init 参数
   - 默认通常每 store 单连接, 16 thread 必然在 client mutex 上排队
3. **快速验证实验**: 把 `NumInsertThreads` 从 16 → 4
   - 若 single-RMW latency 回到 ~1-2 ms ⇒ 是 client 并发 (强证据)
   - 若仍然 7-8 ms ⇒ 问题更深 (可能 search loop 也在抢)
4. **降低 batch_get 路径压力**:
   - `CollectReAssign` nearby scan 已 snapshot, 但每次 RMW reassign 都会触发一次
     MultiGet, 总量巨大
   - 考虑加 client-side LRU cache 或 coalesce 重复 keys

---

## 6. 数据来源

- 客户端 DIAG: `evaluation/2026-04-23/benchmark_multichunk_20260424_023309.log`
  (5 个 ALL DONE 块: build layer1, build layer0, batch1 layer0, batch2 layer0, batch3 layer0+layer1)
- 服务端 metrics: `curl http://127.0.0.1:{20181,20182,20183}/metrics` (B3 结束后即时)
- PD region/store: `curl http://127.0.0.1:23791/pd/api/v1/{regions/count,stores}`
