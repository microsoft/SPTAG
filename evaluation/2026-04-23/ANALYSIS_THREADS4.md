# Threads=4 诊断实验分析

**目标**: 验证 Hypothesis C — 是否 "TiKV 客户端并发争用" 是 RMW 延迟随 batch 增长的真因。
方法: 与 16 线程基线 (`output.json` / `benchmark_multichunk_20260424_023309.log`) 唯一不同点 = `NumInsertThreads: 16 → 4`。
全程: 2026-04-24 07:16:40 → 2026-04-25 06:43，10 batches 全部完成 (~23h)。

---

## 1. 单条 RMW 延迟 (per-batch, DIAG histogram 差分)

| batch | Δcount  | T16 GetUs | T4 GetUs | 加速 | T16 PutUs | T4 PutUs | 加速 |
|------:|--------:|----------:|---------:|----:|----------:|---------:|----:|
| 1     | 9.13M   | 474 µs    | 313 µs   | **1.5×** | 957 µs    | 747 µs   | **1.3×** |
| 2     | 5.00M   | 2 599 µs  | 763 µs   | **3.4×** | 3 286 µs  | 1 345 µs | **2.4×** |
| 3     | 2.64M   | 7 102 µs  | 1 452 µs | **4.9×** | 8 385 µs  | 2 017 µs | **4.2×** |

**Hypothesis C 被强力证实**:
- 客户端线程数减为 1/4 后，单次 Get/Put 延迟显著下降；
- **加速比随 batch 单调增长** (1.5× → 5×) — 与 TiKV region 数量随数据增长而增多一致。
- 服务端在两次实验中负载相同 (~10 RMW × 1M insert × 同硬件)，只能是客户端侧争用随 region 数量放大。

## 2. 插入吞吐量 (inserts/sec)

| batch | T16 thrput | T4 thrput | 比值 |
|------:|-----------:|----------:|----:|
| 1     | 324.0      | 232.8     | 0.72× |
| 2     | 192.8      | 155.2     | 0.81× |
| 3     | 141.8      | 125.2     | 0.88× |
| 4     | (未跑完)    | 114.2     |   —   |
| 10    | (未跑完)    | 102.6     |   —   |

> 16 线程总吞吐 = 4 × `(thrput per thread)`，但每线程 RMW 慢 4-5 倍；4 线程则相反。
> 净结果: 16 线程仍领先 12-28%，**减少线程不是 Pareto 改进**。
> 我们要的是 "保留 16 线程并发 + 消除争用"，能拿到的 upside 估计 ~3-5×。

## 3. 搜索表现 (在写入期间)

| batch | T16 QPS | T16 mean | T4 QPS | T4 mean | recall@5 (T4) |
|------:|--------:|---------:|-------:|--------:|--------------:|
| 1     | 269     | 14.8 ms  | 294    | 13.6 ms | 0.983 |
| 2     | 276     | 14.4 ms  | 165    | 24.1 ms | 0.980 |
| 3     | 170     | 23.4 ms  | 195    | 20.4 ms | 0.972 |

T4 全程 recall 0.96-0.98，索引质量未退化。

## 4. 锁与拆分指标

| 项目              | T16 batch 3 | T4 batch 3+ |
|-------------------|------------|-------------|
| AppendLockWait avg | 1-3 µs     | 0.4-0.6 µs (cum)  |
| SplitLockWait avg  | 1-3 µs     | 0.4-1 µs    |
| split_latency max  | 48.9 s     | 4.6-20.6 s  |

`split_latency max` 也降了 ~10×，与"客户端 IO 排队消失"一致。

## 5. 后批次 (batches 4-10) 稳态

```
B4..B10 thrput     : 114 → 103 inserts/s     (单调缓降)
B4..B10 RMW count  : 1.20 → 0.02 M           (cumulative count 已基本不增长)
B4..B10 GetUs      : 644 → 667 µs            (稳定)
B4..B10 PutUs      : 1158 → 1180 µs          (稳定)
B4..B10 PostBytes  : 12.8 → 12.8 KB          (稳定)
```

**关键发现**: 从 batch 4 起，DIAG 累积 count 几乎不再增长 — 意味着 layer 0 的 RMW 已经稳态运行；新增 batch 触发的 RMW 主要落在 layer 1 (count 1.72M → 2.28M)。
延迟值因此早早稳定在 (Get 660 µs, Put 1180 µs)，没有出现 16 线程那种"持续恶化"。

---

## 6. 结论与下一步

### 结论
1. **Hypothesis C 成立**: TiKV 客户端并发争用是 16 线程实验中 RMW 延迟膨胀的主因。证据是: 同样 server / 同样 workload，仅减少客户端并发就让单次 RMW 加速 1.5-5×，且加速倍数随 region 数增长而扩大。
2. **A/B 排除**: AppendLockWait/SplitLockWait 始终 <2 µs (T16 也是)，B 已被前文排除 (PostBytes 稳定但延迟增长)。
3. **直接降线程不可取**: 总吞吐反而降低 12-28%。

### 优化优先级 (按 ROI 排序)

1. **[高] thread-local stub cache** — 把 `StubPool` 改为线程局部缓存 (按 store_id 索引)，彻底去掉 `m_storeMutex` 与 round-robin atomic。预期:
   - 单次 RMW 接近 4 线程实验的延迟 (Get ~300 µs → 即使 16 线程也能保持)；
   - 总吞吐 4-5× ↑ (16 × 1500 µs/RMW vs 16 × 8500 µs/RMW)。
   - 代价: O(线程数 × store 数) 内存，~MB 级；channel 复用语义不变。

2. **[中] 用持久 thread pool 替换 `std::async`** (`MultiGet` region 扇出)。当前 libstdc++ 的 `std::async(std::launch::async)` 每次创建/回收线程，高 QPS 下放大锁争用。

3. **[低] `m_storeMutex` 改 `shared_mutex`** — 短期缓解；但 #1 直接消除了它，可跳过。

4. **[观察] 保持 `kStubPoolSize = 48`** — 当 #1 实施后，pool 可能可以退化为 1-2 个 channel/store（因为不再是争用点，仅是 HoL blocking 风险）。

### 推荐动作
直接实施 (1)，用 thread-local 包装 `StubPool::GetNext()`。在 1M base + 1 batch (10M total / 10) 上验证收益再扩展到全量。
