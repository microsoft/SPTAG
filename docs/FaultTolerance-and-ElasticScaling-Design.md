# SPTAG/SPANN 容错与弹性伸缩设计文档

## 目录

1. [系统架构概览](#1-系统架构概览)
2. [Client 层设计与吞吐量分析](#2-client-层设计与吞吐量分析)
3. [设计原则](#3-设计原则)
4. [查询流容错：完整路径图与故障处理表](#4-查询流容错完整路径图与故障处理表)
5. [写入流容错：完整路径图与故障处理表](#5-写入流容错完整路径图与故障处理表)
6. [维护流容错：Split/Merge/HeadSync](#6-维护流容错splitmergehead-sync)
7. [节点 Crash 故障处理总表](#7-节点-crash-故障处理总表)
8. [故障检测：Gossip 成员管理 (SWIM)](#8-故障检测gossip-成员管理-swim)
9. [弹性伸缩设计](#9-弹性伸缩设计)
10. [实现路线图：PR 计划](#10-实现路线图pr-计划)
11. [附录](#附录)

---

## 1. 系统架构概览

### 1.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                          Clients                                │
└────────────┬───────────────────────────────────┬────────────────┘
             │ Query                             │ Insert
             ▼                                   ▼
┌────────────────────────┐          ┌────────────────────────────┐
│   Aggregator × M       │          │   Any Compute Node         │
│   (无状态, LB 后面)     │          │   (接受写入的入口)          │
└────────────┬───────────┘          └─────────────┬──────────────┘
             │                                    │
             │ Fan-out Query                      │ Route via
             │ to Compute Nodes                   │ ConsistentHash
             ▼                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Compute Node × 2000                            │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  Head Index (SPTAG/BKT) ← 每节点全量副本               │     │
│  │  ExtraDynamicSearcher ← Posting 读写 + Split/Merge     │     │
│  │  PostingRouter ← 一致性哈希 headID→Node 路由            │     │
│  │  VersionMap ← TiKV-backed 版本管理                     │     │
│  │  SWIM Agent ← Gossip 成员检测（去中心化）               │     │
│  └────────────────────────────────────────────────────────┘     │
└─────────────────────────────┬───────────────────────────────────┘
                              │ gRPC: RawGet/RawPut/RawBatchGet
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   TiKV 集群 (3 副本 Raft)                       │
│                                                                 │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  PD × 5 (Placement Driver, 复用为 Cluster Controller) │       │
│  │  - TiKV Region 调度 / Store 路由                      │       │
│  │  - Compute Node 注册 / 成员列表 / Ring 版本管理        │       │
│  └─────────────────────────────────────────────────────┘        │
│                                                                 │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐         ┌──────┐            │
│  │TiKV │ │TiKV │ │TiKV │ │TiKV │  ...    │TiKV  │            │
│  │  0  │ │  1  │ │  2  │ │  3  │         │ 1999 │            │
│  └─────┘ └─────┘ └─────┘ └─────┘         └──────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 核心组件职责

| 组件 | 实例数 | 状态 | 职责 |
|------|--------|------|------|
| **Client** | 任意 | 无状态 | 发起查询/写入请求，重试，负载均衡选择入口 |
| **Aggregator** | M (≥3) | 无状态 | 查询入口，扇出到 Compute Node，合并 Top-K 结果 |
| **Compute Node** | 2000 | 有状态* | Head Index 搜索，Posting 读写，Split/Merge |
| **PostingRouter** | (内嵌) | 可重建 | 一致性哈希 headID→Node 路由 |
| **TiKV Store** | 2000 | 持久化 | Posting/VersionMap/WAL/Intent 存储 (Raft 3 副本) |
| **PD** | 5 | 持久化 | Region 调度 + Compute Node 集群管理 |

> *Compute Node "有状态"是指 Head Index 副本和内存缓存。核心数据（Posting）全部在 TiKV，Compute Node 本质上是**可重建**的。

---

## 2. Client 层设计与吞吐量分析

### 2.1 当前 Benchmark 架构 vs 生产架构

当前 benchmark 只有 **1 个 Client**（即 Driver 进程 n0），它同时承担三个角色：

```
┌──────────────────────────────────────────────────────────────────┐
│              当前 Benchmark 架构 (1 Driver = 3 合 1)               │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  Driver (Node n0) = Client + Aggregator + Compute Node   │    │
│  │                                                          │    │
│  │  ┌────────────┐  ┌──────────────┐  ┌─────────────────┐  │    │
│  │  │ 发起 Query │  │ Head Search  │  │ 本地 Posting    │  │    │
│  │  │ / Insert   │→│ (本地 BKT)   │→│ Read (如果是    │  │    │
│  │  │ 请求       │  │              │  │ 自己 Owner 的)  │  │    │
│  │  └────────────┘  └──────────────┘  └────────┬────────┘  │    │
│  │                                              │           │    │
│  │                    ┌─────────────────────────┘           │    │
│  │                    │ PostingRouter: 非本地的                │    │
│  │                    │ → 路由到 Worker 节点                  │    │
│  │                    ▼                                      │    │
│  └──────────────────────────────────────────────────────────┘    │
│                       │                                          │
│         ┌─────────────┴─────────────┐                           │
│         ▼                           ▼                           │
│  ┌────────────┐             ┌────────────┐                     │
│  │ Worker n1  │             │ Worker n2  │                     │
│  │ (WorkerNode)│             │ (WorkerNode)│                     │
│  └────────────┘             └────────────┘                     │
└──────────────────────────────────────────────────────────────────┘

代码对应关系:
  Driver:  SPFreshTest/BenchmarkFromConfig  (run_scale_benchmarks.sh:673)
  Worker:  SPFreshTest/WorkerNode           (run_scale_benchmarks.sh:650)
```

生产环境应分离为独立层：

```
┌──────────────────────────────────────────────────────────────────┐
│                      生产架构 (完全分离)                           │
│                                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐        ┌──────────┐    │
│  │ Client 1 │ │ Client 2 │ │ Client 3 │  ...   │ Client P │    │
│  └────┬─────┘ └─────┬────┘ └─────┬────┘        └────┬─────┘    │
│       │             │             │                   │          │
│       └──────┬──────┴──────┬──────┴───────────┬──────┘          │
│              │ DNS / L4 LB │                  │                 │
│              ▼             ▼                  ▼                 │
│        ┌───────────┐ ┌───────────┐     ┌───────────┐           │
│        │Aggregator │ │Aggregator │ ... │Aggregator │           │
│        │    1      │ │    2      │     │    M      │           │
│        └─────┬─────┘ └─────┬─────┘     └─────┬─────┘           │
│              │             │                  │                 │
│              └──────┬──────┴──────┬───────────┘                 │
│                     │ Fan-out     │                             │
│                     ▼             ▼                             │
│        ┌─────────────────────────────────────┐                 │
│        │    Compute Nodes × 2000             │                 │
│        │    (纯后端，不直接接受外部请求)       │                 │
│        └─────────────────────────────────────┘                 │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 吞吐量瓶颈分析

一个查询的端到端处理有 3 个关键阶段，每个阶段都可能成为瓶颈：

```
 ┌─────────────┐        ┌──────────────────┐       ┌─────────────────┐
 │  阶段 A     │        │  阶段 B          │       │  阶段 C         │
 │  Client/    │  ───▶  │  Head Search     │  ───▶ │  Posting Read   │
 │  Aggregator │        │  (CPU 密集)       │       │  (I/O 密集)     │
 │  接入+合并  │        │  BKT 图搜索       │       │  TiKV 网络往返   │
 └─────────────┘        └──────────────────┘       └─────────────────┘
```

### 2.3 三种扩展模式对比

```
┌─────────────────────────────────────────────────────────────────────────┐
│  模式 1: 1 Client → N Compute Nodes（当前 benchmark）                    │
│                                                                         │
│    ┌────────┐                                                           │
│    │Client 1│  ────────┬──────────┬─────────── ...                     │
│    └────────┘          │          │                                     │
│                   ┌────┴──┐  ┌───┴───┐  ┌──────┐                      │
│                   │Node 0 │  │Node 1 │  │Node N│                      │
│                   └───────┘  └───────┘  └──────┘                      │
│                                                                         │
│  ✅ Posting I/O 分摊到 N 个节点 → 阶段 C 吞吐量线性增长                  │
│  ❌ Head Search 只在 1 个节点 → 阶段 B 是瓶颈                            │
│  ❌ 扇出/合并都在 1 个进程 → 阶段 A 是瓶颈 (网络带宽/CPU)               │
│  ❌ 该 Client 宕机 → 全部不可用                                          │
│                                                                         │
│  瓶颈: Client CPU 和 网络                                                │
│  适用: 测试/小规模                                                       │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  模式 2: M Clients → 各自独立 (不分布)                                   │
│                                                                         │
│    ┌────────┐  ┌────────┐  ┌────────┐                                  │
│    │Client 1│  │Client 2│  │Client 3│  ...                             │
│    └───┬────┘  └───┬────┘  └───┬────┘                                  │
│        │           │           │                                        │
│    ┌───┴───┐  ┌───┴───┐  ┌───┴───┐                                    │
│    │Node 0 │  │Node 1 │  │Node 2 │     (每个 Client 绑定 1 个节点)     │
│    └───────┘  └───────┘  └───────┘                                     │
│                                                                         │
│  ✅ 阶段 A/B 随 Client 数量线性扩展                                      │
│  ❌ 每个节点只查自己的 Posting → 阶段 C 没有利用分布式                     │
│  ❌ 实际上退化为 M 个独立单机实例                                         │
│  ❌ 无法利用跨节点 Posting 分布优势                                      │
│                                                                         │
│  瓶颈: 单节点 I/O (每节点承担 1/1 的 Posting)                            │
│  适用: 不适合 SPANN (Posting 必须分布)                                   │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  模式 3: M Aggregators → N Compute Nodes（生产推荐）                     │
│                                                                         │
│    ┌────────┐  ┌────────┐  ┌────────┐                                  │
│    │Client 1│  │Client 2│  │Client P│  ...                             │
│    └───┬────┘  └───┬────┘  └───┬────┘                                  │
│        │           │           │   ← DNS/L4 LB                         │
│    ┌───┴───┐  ┌───┴───┐  ┌───┴───┐                                    │
│    │Agg  1 │  │Agg  2 │  │Agg  M │                                    │
│    └───┬───┘  └───┬───┘  └───┬───┘                                    │
│        │  fan-out  │  fan-out │                                        │
│    ┌───┴───────────┴─────────┴───┐                                     │
│    │    Compute Nodes × 2000     │                                     │
│    └─────────────┬───────────────┘                                     │
│                  │                                                      │
│    ┌─────────────┴───────────────┐                                     │
│    │      TiKV × 2000            │                                     │
│    └─────────────────────────────┘                                     │
│                                                                         │
│  ✅ 阶段 A: M 个 Aggregator 并行接入/合并 → Client 层不是瓶颈           │
│  ✅ 阶段 B: N 个 Compute Node 并行 Head Search → CPU 线性扩展           │
│  ✅ 阶段 C: Posting 分布在 N 节点 → I/O 线性扩展                        │
│  ✅ 任何一层故障都可独立恢复，无单点                                      │
│                                                                         │
│  瓶颈: 理论上无单一瓶颈，各层独立扩展                                     │
│  适用: 生产环境 (2000 节点)                                              │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.4 吞吐量公式

```
                  单查询延迟分解
  ┌───────────────────────────────────────────────────┐
  │                                                   │
  │  Latency = T_network + T_head + T_posting + T_merge│
  │                                                   │
  │  T_network : Client↔Aggregator + Aggregator↔Node  │
  │            ≈ 0.1ms (同机房)                         │
  │  T_head    : BKT 图搜索 (CPU)                      │
  │            ≈ 0.5-2ms                               │
  │  T_posting : TiKV RawBatchGet (I/O)                │
  │            ≈ 1-5ms (取决于 posting 大小和数量)       │
  │  T_merge   : Top-K 合并 (CPU)                      │
  │            ≈ 0.1ms                                 │
  │                                                   │
  │  单节点 QPS ≈ NumThreads / Latency                 │
  │            ≈ 32 / 3ms ≈ 10,000 QPS (单 Compute)    │
  └───────────────────────────────────────────────────┘

           三种模式吞吐量对比 (假设 N=2000 Compute Nodes)

  ┌───────────────┬──────────────────┬──────────────────┬────────────────────┐
  │               │ 模式 1           │ 模式 2           │ 模式 3 (推荐)      │
  │               │ 1 Client, N Node │ M Client, 独立   │ M Agg, N Node      │
  ├───────────────┼──────────────────┼──────────────────┼────────────────────┤
  │ 阶段 A 容量    │ 1 进程           │ M 进程           │ M Aggregator       │
  │ (接入/合并)    │ ~50K QPS 上限    │ M×10K QPS        │ M×50K QPS          │
  ├───────────────┼──────────────────┼──────────────────┼────────────────────┤
  │ 阶段 B 容量    │ 1 节点           │ M 节点           │ N 节点             │
  │ (Head Search) │ ~10K QPS         │ M×10K QPS        │ N×10K = 20M QPS    │
  ├───────────────┼──────────────────┼──────────────────┼────────────────────┤
  │ 阶段 C 容量    │ N 节点           │ 1 节点           │ N 节点             │
  │ (Posting I/O) │ 理论 N×10K       │ 1×10K QPS        │ N×10K = 20M QPS    │
  ├───────────────┼──────────────────┼──────────────────┼────────────────────┤
  │ 实际总吞吐量   │ min(50K,10K,20M) │ min(M×10K, M×10K,│ min(M×50K, 20M,    │
  │               │ = **~10K QPS**   │  M×10K)          │  20M)              │
  │               │ (瓶颈: B)        │ = **M×10K QPS**  │ = **M×50K QPS**    │
  │               │                  │ (瓶颈: 均匀)     │ (瓶颈: Aggregator) │
  ├───────────────┼──────────────────┼──────────────────┼────────────────────┤
  │ 回答你的问题   │                  │ M Client 不分布  │                    │
  │               │ 1 Client 分布    │ 也增加吞吐量     │ 两者结合最优        │
  │               │ → 增加阶段C吞吐  │ → 增加阶段A/B    │                    │
  └───────────────┴──────────────────┴──────────────────┴────────────────────┘

  结论: 两种方式都增加吞吐量，但瓶颈不同
  ─────────────────────────────────────────
  • 1 Client 多 Node (分布): 解决 I/O 瓶颈 (阶段 C)，但 Client 成为新瓶颈
  • 多 Client 不分布:        解决 Client 瓶颈 (阶段 A/B)，但单节点 I/O 成为新瓶颈
  • 多 Aggregator + 多 Node: 所有阶段独立扩展，生产唯一正确选择
```

### 2.5 Client SDK 设计

```
┌─────────────────────────────────────────────────────────────────┐
│                     Client SDK 架构                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Application                                                    │
│      │                                                         │
│      ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  SPANNClient                                            │   │
│  │                                                         │   │
│  │  ┌─────────────────┐  ┌──────────────────────────────┐  │   │
│  │  │ ConnectionPool  │  │ RetryPolicy                  │  │   │
│  │  │                 │  │                              │  │   │
│  │  │ • Aggregator    │  │ • MaxRetries: 3             │  │   │
│  │  │   endpoints[]   │  │ • Backoff: exp(100ms,2s)    │  │   │
│  │  │ • HealthCheck   │  │ • Idempotent:               │  │   │
│  │  │   (TCP ping 1s) │  │   Query → 总是安全重试      │  │   │
│  │  │ • RoundRobin    │  │   Write → VID+Ver 去重      │  │   │
│  │  │   / WeightedLB  │  │ • Non-retryable:            │  │   │
│  │  └─────────────────┘  │   400 Bad Request           │  │   │
│  │                       └──────────────────────────────┘  │   │
│  │  ┌─────────────────┐  ┌──────────────────────────────┐  │   │
│  │  │ Timeout Config  │  │ CircuitBreaker               │  │   │
│  │  │                 │  │                              │  │   │
│  │  │ • Connect: 1s   │  │ • Per-Aggregator 独立        │  │   │
│  │  │ • Query: 5s     │  │ • 5 failures in 30s → OPEN  │  │   │
│  │  │ • Write: 10s    │  │ • OPEN 30s → HALF_OPEN      │  │   │
│  │  │ • Overall: 30s  │  │ • 1 success → CLOSED        │  │   │
│  │  └─────────────────┘  └──────────────────────────────┘  │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  查询接口:                                                      │
│    result = client.Search(query_vector, top_k=10)               │
│    result.degraded   // true if partial nodes responded         │
│    result.latency_ms // end-to-end latency                     │
│                                                                 │
│  写入接口:                                                      │
│    ack = client.Insert(vid, vector, metadata)                   │
│    ack.persistent   // true = data in TiKV Raft                │
│    ack = client.Delete(vid)                                     │
│                                                                 │
│  批量接口:                                                      │
│    results = client.BatchSearch(query_vectors[], top_k=10)      │
│    acks = client.BatchInsert(vids[], vectors[], metas[])        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.6 Client 容错路径

```
  Client                        Load Balancer              Aggregator Pool
    │                               │                           │
    │  ① 发送请求                    │                           │
    │──────────────────────────────▶│                           │
    │                               │  ② 选择 Agg A            │
    │                               │──────────────────────────▶│ Agg A
    │                               │                           │
    │                 ┌─────────── 正常路径 ──────────────┐      │
    │                 │             │                     │      │
    │                 │             │  ③ Agg A 处理成功    │      │
    │                 │             │◀──────────────────── │      │
    │  ④ 返回结果     │             │                           │
    │◀──────────────────────────────│                           │
    │                                                           │
    │                 ┌─────────── 故障路径 ──────────────┐      │
    │                 │                                   │      │
    │  ⑤ Agg A 超时/失败                                  │      │
    │  (CircuitBreaker                                    │      │
    │   记录失败)                                         │      │
    │                               │                           │
    │  ⑥ 自动重试                    │                           │
    │──────────────────────────────▶│                           │
    │                               │  ⑦ LB 跳过 Agg A         │
    │                               │  选择 Agg B              │
    │                               │──────────────────────────▶│ Agg B
    │                               │                           │
    │  ⑧ Agg B 成功返回              │                           │
    │◀──────────────────────────────│◀──────────────────────────│
    │                                                           │

  故障处理表:
  ┌──────────────────────┬─────────────────┬───────────────────────────────┐
  │ 故障点               │ 检测方式         │ Client 行为                    │
  ├──────────────────────┼─────────────────┼───────────────────────────────┤
  │ DNS 解析失败          │ getaddrinfo err │ 使用缓存的 IP；fallback DNS    │
  │ LB 不可达            │ connect timeout │ 直连已知 Aggregator IP         │
  │ Aggregator 超时       │ read timeout    │ 重试到其他 Agg (max 3 次)     │
  │ Aggregator 返回错误   │ HTTP 5xx / gRPC │ CircuitBreaker 标记；重试      │
  │ 响应格式错误          │ decode error    │ 丢弃，重试到其他 Agg           │
  │ 部分结果 (degraded)   │ response flag   │ 返回给应用层，标记 degraded    │
  │ 全部 Agg 不可用       │ all retries fail│ 返回错误，应用层决定策略       │
  └──────────────────────┴─────────────────┴───────────────────────────────┘
```

### 2.7 写入路径的 Client 视角

```
  Client SDK                    Any Compute Node
    │                               │
    │  Insert(vid, vec, meta)       │
    │──────────────────────────────▶│
    │                               │
    │  写入有两种入口路径:            │
    │                               │
    │  路径 A: 通过 Aggregator       │
    │  (查询和写入统一入口)           │
    │  Client → Agg → Compute Node  │
    │                               │
    │  路径 B: 直连 Compute Node     │
    │  (写入密集型场景，减少一跳)     │
    │  Client → 任意 Compute Node   │
    │  → PostingRouter 路由到 Owner  │
    │                               │
    │  选择依据:                     │
    │  ┌──────────────────────────┐  │
    │  │ 查询为主 → 路径 A        │  │
    │  │   (统一入口，简单)        │  │
    │  │                          │  │
    │  │ 写入为主 → 路径 B        │  │
    │  │   (少一跳，低延迟)        │  │
    │  │   Client 需知道 Compute  │  │
    │  │   Node 地址列表           │  │
    │  └──────────────────────────┘  │
    │                               │
    │  ACK: {persistent: true}      │
    │◀──────────────────────────────│
    │                               │
    │  如果超时/失败:                 │
    │  → 安全重试 (VID+Version 去重) │
    │  → 可以发到任意 Compute Node   │
    │    (PostingRouter 会路由到     │
    │     正确的 Owner)             │
```

---

## 3. 设计原则

### 3.1 核心语义

```
┌─────────────────────────────────────────────────────────────┐
│                     五大设计原则                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ① Synchronous Write Semantics                              │
│     ┌─────────┐     ┌──────────────────────────────────┐    │
│     │ Success │ ──▶ │ 数据 100% 持久化到 TiKV (Raft)   │    │
│     └─────────┘     └──────────────────────────────────┘    │
│     ┌─────────┐     ┌──────────────────────────────────┐    │
│     │ Fail /  │ ──▶ │ 调用方明确知道失败，可决定重试    │    │
│     │ Timeout │     └──────────────────────────────────┘    │
│     └─────────┘                                             │
│                                                             │
│  ② Idempotent Retry                                         │
│     所有写操作（Append/Split/Merge）均可安全重试              │
│     VID + Version 去重，Intent 状态机防止重复执行             │
│                                                             │
│  ③ Consistent Hashing                                       │
│     节点增删时最小化 key 重映射（~1/N）                       │
│     Owner Blacklist 快速跳过故障节点                          │
│                                                             │
│  ④ Gossip Membership (SWIM)                                 │
│     去中心化故障检测，无单点依赖                              │
│     每个节点独立判定 alive/suspect/dead                       │
│                                                             │
│  ⑤ Fast Failure Detection + Owner Blacklist                 │
│     故障节点立即加入 blacklist，后续请求不再等待               │
│     避免长时间阻塞在不可达节点上                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 目标规模

| 参数 | 值 | 说明 |
|------|-----|------|
| TiKV Store | 2000 | Raft 3 副本 |
| Compute Node | 2000 | 1:1 对应 TiKV Store |
| PD | 5 | 容忍 2 同时故障 |
| 故障频率 | ~1-2 节点/周 | 2% 年故障率 × 2000 节点 |
| 设计容忍 | ≥3 同时不可用 | 多机架同时故障场景 |

---

## 4. 查询流容错：完整路径图与故障处理表

### 4.1 查询流完整路径图

```
  Client                Aggregator           Compute Node           TiKV Cluster
    │                      │                      │                      │
    │  ① Search Request    │                      │                      │
    │─────────────────────▶│                      │                      │
    │                      │                      │                      │
    │                      │  ② Fan-out Query     │                      │
    │                      │  (选 K 个健康节点)    │                      │
    │                      │─────────────────────▶│                      │
    │                      │─────────────────────▶│(Node B)              │
    │                      │─────────────────────▶│(Node C)              │
    │                      │                      │                      │
    │                      │                      │  ③ Head Index Search │
    │                      │                      │  (本地内存, 无网络)   │
    │                      │                      │                      │
    │                      │                      │  ④ Posting Read      │
    │                      │                      │─────────────────────▶│
    │                      │                      │  RawBatchGet         │
    │                      │                      │◀─────────────────────│
    │                      │                      │  Posting Data        │
    │                      │                      │                      │
    │                      │                      │  ⑤ Distance Compute  │
    │                      │                      │  + Top-K Selection   │
    │                      │                      │                      │
    │                      │  ⑥ Results Return    │                      │
    │                      │◀─────────────────────│                      │
    │                      │                      │                      │
    │                      │  ⑦ Merge K Results   │                      │
    │                      │  (取全局 Top-K)       │                      │
    │                      │                      │                      │
    │  ⑧ Final Results     │                      │                      │
    │◀─────────────────────│                      │                      │
    │                      │                      │                      │
```

### 4.2 查询流每步故障处理表

| 步骤 | 断开位置 | 故障模式 | 检测方式 | 处理策略 | 用户感知 |
|------|----------|----------|----------|----------|----------|
| **①** Client→Aggregator | 网络不通 / Aggregator 宕机 | TCP 连接失败 / 超时 | Client 侧 connect timeout (1s) | LB 自动切换到其他 Aggregator 实例；Client 重试 (max 3次, exp backoff) | 延迟增加 ~1s |
| **②** Aggregator→Compute | Compute Node 宕机 / 网络分区 | TCP RST / 超时 | Aggregator 连接池心跳 + 发送超时 (500ms) | 该节点加入 **Owner Blacklist**；请求自动路由到其他健康 Compute Node；Head Index 是全量副本，任何节点都能完成 | 延迟增加 ~500ms |
| **③** Head Index Search | 内存损坏 / OOM | 进程内异常 / SIGSEGV | Compute Node 内部 try-catch + watchdog | 节点自杀重启；Aggregator 超时后切到其他节点 | 延迟增加 ~1s |
| **④** Compute→TiKV Read | TiKV Leader 迁移 / Store 宕机 | gRPC UNAVAILABLE / Region Error | gRPC status code + region_error 字段 | **分级容错链**：<br>1. Retry Leader (cache invalidated, 400ms)<br>2. Follower Read (stale, 200ms)<br>3. Partial Result 降级 | Leader迁移: ~400ms<br>降级: 精度轻微下降 |
| **⑤** Distance Compute | CPU 异常 / 进程崩溃 | 同 ③ | 同 ③ | 同 ③ | 同 ③ |
| **⑥** Compute→Aggregator | 响应丢失 / Compute 中途宕机 | 超时 | Aggregator 等待超时 (2s) | Aggregator 仅需 K 个结果中的 **任意 1 个** 即可返回；其余超时节点结果丢弃 | 若仅部分返回，结果标记 `degraded=true` |
| **⑦** Merge Results | Aggregator 自身 OOM | 进程崩溃 | LB 健康检查 | Client 重试到另一 Aggregator | 延迟增加 ~1s |
| **⑧** Aggregator→Client | 网络断开 | TCP RST / 超时 | Client 侧超时 | Client 重试（查询天然幂等） | 延迟增加 ~1-2s |

### 4.3 查询流容错详图（含 Blacklist）

```
                    ② Aggregator 发送查询到 Compute Node
                    
      ┌──────────────────────────────────────────────────────┐
      │ Aggregator                                           │
      │                                                      │
      │  healthy_nodes = AllNodes - Blacklist                 │
      │  candidates = Select(healthy_nodes, K=3)             │
      │                                                      │
      │     ┌───────────┐  ┌───────────┐  ┌───────────┐     │
      │     │  Node A   │  │  Node B   │  │  Node C   │     │
      │     │  (500ms)  │  │  (500ms)  │  │  (500ms)  │     │
      │     └─────┬─────┘  └─────┬─────┘  └─────┬─────┘     │
      │           │              │              │             │
      │           ▼              ▼              ▼             │
      │     ┌─────────┐  ┌───────────┐  ┌──────────┐        │
      │     │   OK    │  │  Timeout  │  │   OK     │        │
      │     │ results │  │ (500ms)   │  │ results  │        │
      │     └─────┬───┘  └─────┬─────┘  └────┬─────┘        │
      │           │            │              │              │
      │           │     ┌──────▼──────┐       │              │
      │           │     │ Add Node B  │       │              │
      │           │     │ to Blacklist│       │              │
      │           │     │ (TTL=30s)   │       │              │
      │           │     └─────────────┘       │              │
      │           │                           │              │
      │           └──────────┬────────────────┘              │
      │                      ▼                               │
      │              Merge(A, C) → Top-K                     │
      │              Return to Client                        │
      └──────────────────────────────────────────────────────┘
```

### 4.4 TiKV 读取分级容错图

```
  Compute Node                                TiKV Cluster
      │                                           │
      │  ④-a: RawBatchGet (Leader, timeout=200ms)  │
      │───────────────────────────────────────────▶│
      │                                           │
      │  ┌─ Success? ─────────────────────────────┤
      │  │  YES → Return data                     │
      │  │  NO  ↓                                 │
      │  │                                        │
      │  │  ④-b: Invalidate region cache          │
      │  │       Retry Leader (timeout=400ms)      │
      │  │───────────────────────────────────────▶│
      │  │                                        │
      │  │  ┌─ Success? ─────────────────────────┤
      │  │  │  YES → Return data                 │
      │  │  │  NO  ↓                             │
      │  │  │                                    │
      │  │  │  ④-c: Follower Read (stale ok)     │
      │  │  │       (timeout=200ms)               │
      │  │  │───────────────────────────────────▶│
      │  │  │                                    │
      │  │  │  ┌─ Success? ─────────────────────┤
      │  │  │  │  YES → Return data (stale)     │
      │  │  │  │  NO  ↓                         │
      │  │  │  │                                │
      │  │  │  │  ④-d: Return partial result    │
      │  │  │  │  (head-search-only, degraded)  │
      │  │  │  │                                │
      └──┴──┴──┴────────────────────────────────┘
```

---

## 5. 写入流容错：完整路径图与故障处理表

### 5.1 写入流完整路径图

```
  Client          Compute Node A        PostingRouter        Owner Node B          TiKV
    │                  │                     │                     │                  │
    │  ① Insert(VID,   │                     │                     │                  │
    │     vector)      │                     │                     │                  │
    │─────────────────▶│                     │                     │                  │
    │                  │                     │                     │                  │
    │                  │  ② Write WAL        │                     │                  │
    │                  │  to TiKV            │                     │                  │
    │                  │─────────────────────┼─────────────────────┼─────────────────▶│
    │                  │◀────────────────────┼─────────────────────┼──────────────────│
    │                  │  WAL persisted      │                     │                  │
    │                  │                     │                     │                  │
    │                  │  ③ Head Index Search │                     │                  │
    │                  │  (local, in-memory)  │                     │                  │
    │                  │  → RNG Selection     │                     │                  │
    │                  │  → headID = H        │                     │                  │
    │                  │                     │                     │                  │
    │                  │  ④ GetOwner(H)      │                     │                  │
    │                  │────────────────────▶│                     │                  │
    │                  │  owner = Node B     │                     │                  │
    │                  │◀───────────────────│                     │                  │
    │                  │                     │                     │                  │
    │                  │  ⑤ Is Node B in Blacklist?                │                  │
    │                  │  NO → Send RemoteAppend ────────────────▶│                  │
    │                  │                     │                     │                  │
    │                  │                     │           ⑥ Node B: │                  │
    │                  │                     │           Acquire   │                  │
    │                  │                     │           headID    │                  │
    │                  │                     │           lock      │                  │
    │                  │                     │                     │                  │
    │                  │                     │           ⑦ RawPut  │                  │
    │                  │                     │           (append   │                  │
    │                  │                     │            posting) │                  │
    │                  │                     │                     │─────────────────▶│
    │                  │                     │                     │  Raft Commit     │
    │                  │                     │                     │◀─────────────────│
    │                  │                     │                     │                  │
    │                  │                     │           ⑧ VersionMap                 │
    │                  │                     │           .IncVersion(VID)              │
    │                  │                     │                     │─────────────────▶│
    │                  │                     │                     │◀─────────────────│
    │                  │                     │                     │                  │
    │                  │                     │           ⑨ Release │                  │
    │                  │                     │           lock      │                  │
    │                  │                     │                     │                  │
    │                  │  ⑩ AppendResponse(OK) ◀──────────────────│                  │
    │                  │                     │                     │                  │
    │                  │  ⑪ Clear WAL entry  │                     │                  │
    │                  │─────────────────────┼─────────────────────┼─────────────────▶│
    │                  │                     │                     │                  │
    │  ⑫ Insert OK     │                     │                     │                  │
    │◀─────────────────│                     │                     │                  │
    │                  │                     │                     │                  │
```

### 5.2 写入流每步故障处理表

| 步骤 | 断开位置 | 故障模式 | 处理策略 | 幂等保证 |
|------|----------|----------|----------|----------|
| **①** Client→Node A | 网络断开 | Client 重试到同一或其他 Compute Node | VID+Version 去重，安全重试 |
| **②** WAL Write→TiKV | TiKV 写入超时 | 10 次重试 + Region Cache 失效；WAL 写入是 Append，幂等 | WAL key = `wal:{nodeId}:{seqNo}`，唯一 |
| **③** Head Index Search | Node A 宕机 | WAL 已持久化 → 恢复后重放；或 Client 重试到其他节点 | VID+Version 去重 |
| **④** GetOwner(H) | Hash Ring 不一致 | Epoch 检查；若 ring 过期则从 PD 拉取最新成员列表 | 确定性哈希，相同输入相同输出 |
| **⑤** Node B 在 Blacklist | Node B 已知故障 | **跳过 Node B**，直接写 TiKV（bypass owner）+ 标记 headID 需后续 Merge 修复 | 直接写 TiKV 是幂等的 |
| **⑥** RemoteAppend→Node B | Node B 宕机 / 超时 | 2 次重试 + 重连；失败则 **加入 Blacklist** + Persist to pending queue in TiKV；后台 daemon 每 5s 扫描重试 | 幂等 Append (VID 去重) |
| **⑦** Node B→TiKV Write | TiKV 写入超时 / Region 迁移 | 10 次重试 + Region Cache 失效；成功 = Raft majority commit = 100% 持久化 | Synchronous Write Semantics |
| **⑧** VersionMap Inc | TiKV 写入失败 | 重试 3 次；失败则 Append 整体返回失败，由步骤⑥重试 | CAS 保证原子性 |
| **⑨** Release Lock | Node B 宕机（未释放锁） | **Lock TTL = 30s**，超时自动释放 | TTL 机制 |
| **⑩** Response→Node A | 网络断开（响应丢失） | Node A 超时 (30s) → 将请求放入 pending queue 重试；实际 TiKV 已写入成功 | 幂等重试安全 |
| **⑪** Clear WAL | TiKV 删除失败 | 非关键路径，后台重试；WAL 重放时会发现已完成（VID 版本匹配），跳过 | 重放幂等 |
| **⑫** Response→Client | 网络断开 | Client 超时重试；数据已持久化，重试时 VersionMap 去重 | 幂等 |

### 5.3 写入流 Owner 故障时的 Bypass 路径

```
  Node A 发现 Owner (Node B) 不可达
  
  正常路径:                         Bypass 路径 (Node B in Blacklist):
  
  Node A ──▶ Node B ──▶ TiKV       Node A ──────────────────▶ TiKV
              │  │                         │
              │  └─ lock headID            └─ 直接 RawPut
              │                               (无 per-headID 锁)
              └─── RawPut                     
                                          标记 headID 需要后续 Merge 修复
                                          (写入 repair_needed:{headID} key)
```

**Bypass 的权衡**：
- ✅ 保证数据不丢失（数据已写入 TiKV）
- ⚠️ 短暂窗口内同一 headID 可能有并发写入（无锁保护）
- ✅ 后续 Merge 操作会自动修复 posting 一致性

---

## 6. 维护流容错：Split/Merge/Head Sync

### 6.1 Split 操作完整路径图

```
  Owner Node                     TiKV                         All Nodes
      │                           │                              │
      │  ① Detect posting > limit │                              │
      │                           │                              │
      │  ② Write Split Intent     │                              │
      │  status=PREPARED          │                              │
      │──────────────────────────▶│                              │
      │                           │                              │
      │  ③ Read full posting      │                              │
      │──────────────────────────▶│                              │
      │◀──────────────────────────│                              │
      │                           │                              │
      │  ④ K-means clustering     │                              │
      │  → newHead1, newHead2     │                              │
      │                           │                              │
      │  ⑤ Write new postings     │                              │
      │  Update Intent=EXECUTING  │                              │
      │──────────────────────────▶│                              │
      │                           │                              │
      │  ⑥ Update Head Index      │                              │
      │  (add new, delete old)    │                              │
      │  Update Intent=HEAD_UPDATED                              │
      │──────────────────────────▶│                              │
      │                           │                              │
      │  ⑦ Broadcast HeadSync     │                              │
      │──────────────────────────▶│  (persist HeadSync Log)      │
      │────────────────────────────────────────────────────────▶│
      │                           │  (push to all nodes)         │
      │                           │                              │
      │  ⑧ Delete old posting     │                              │
      │  Update Intent=COMMITTED  │                              │
      │──────────────────────────▶│                              │
      │                           │                              │
      │  ⑨ Delete Intent          │                              │
      │──────────────────────────▶│                              │
      │                           │                              │
```

### 6.2 Split/Merge 每步故障处理表

| 步骤 | Crash 时的状态 | 恢复策略 | 谁来恢复 |
|------|---------------|----------|----------|
| **②后** Intent=PREPARED | 操作未开始 | 新 owner 检测到 PREPARED intent → **重新执行** Split 或 **删除** Intent (回滚) | 新 owner (consistent hash) |
| **⑤后** Intent=EXECUTING | 新 posting 已写入 TiKV，Head Index 未更新 | 新 owner 检测到 EXECUTING → **继续执行** ⑥⑦⑧⑨（数据已在 TiKV，可重读） | 新 owner |
| **⑥后** Intent=HEAD_UPDATED | Head Index 已更新，旧 posting 未删除 | 新 owner 检测到 HEAD_UPDATED → **继续** ⑦⑧⑨（补发 HeadSync + 清理） | 新 owner |
| **⑧后** Intent=COMMITTED | 操作完成，Intent 未清理 | 新 owner 检测到 COMMITTED → **直接删除** Intent | 新 owner |
| **HeadSync 部分节点未收到** | — | 未收到的节点通过 **拉取 HeadSync Log** 追赶（TiKV 持久化的有序日志） | 各节点自行拉取 |

### 6.3 Intent 状态机

```
                    ┌──────────┐
                    │ PREPARED │ ← 写入 Intent，操作未开始
                    └────┬─────┘
                         │ 开始执行
                         ▼
                    ┌──────────┐
              ┌─────│EXECUTING │ ← 新 posting 已写入
              │     └────┬─────┘
              │          │ Head Index 更新完成
    Crash恢复:│          ▼
    可从此继续 │    ┌────────────┐
              │    │HEAD_UPDATED│ ← Head Index 已更新
              │    └────┬───────┘
              │         │ 旧 posting 清理 + HeadSync
              │         ▼
              │    ┌──────────┐
              └───▶│COMMITTED │ ← 操作完成
                   └────┬─────┘
                        │ 清理 Intent
                        ▼
                   ┌──────────┐
                   │ (deleted)│
                   └──────────┘

    回滚路径 (任何阶段):
    ┌──────────┐
    │ROLLED_BACK│ → 清理已写入的新 posting → 删除 Intent
    └──────────┘
```

### 6.4 HeadSync 推送+拉取混合模式

```
  Node A (Split 执行者)             TiKV HeadSync Log            Node B (接收者)
      │                                  │                           │
      │  Write HeadSync entry            │                           │
      │  key: headsync:{epoch}:{seq}     │                           │
      │─────────────────────────────────▶│                           │
      │                                  │                           │
      │  Push HeadSyncEntry to Node B    │                           │
      │──────────────────────────────────┼──────────────────────────▶│
      │                 (Socket, best-effort)                        │  Apply locally
      │                                  │                           │
      │                                  │   如果 Push 失败或有 gap:  │
      │                                  │                           │
      │                                  │  Pull: 从 cursor 到 latest│
      │                                  │◀──────────────────────────│
      │                                  │─────────────────────────▶│
      │                                  │  (批量返回缺失的 entries)  │  Apply all
      │                                  │                           │
      │                                  │  Update cursor            │
      │                                  │◀──────────────────────────│
      │                                  │                           │
```

---

## 7. 节点 Crash 故障处理总表

### 7.1 TiKV Store Crash

| 场景 | 影响范围 | 检测时间 | 恢复时间 | 数据影响 | 自动恢复动作 |
|------|----------|----------|----------|----------|-------------|
| **1 Store 宕机** | 该 Store 上 Leader Region 暂不可写 | ~10s (PD 心跳) | ~30-60s (Leader 选举) | **无丢失** (2 副本存活) | PD 自动选举新 Leader → 30m 后补副本 |
| **1 Store 短暂重启** | 同上，持续时间更短 | ~10s | ~10-20s (Store 重连) | **无丢失** | PD 发现 Store 恢复 → 取消补副本 |
| **同机架多 Store 宕机** (交换机故障) | 该机架所有 Store 的 Leader Region | ~10s | ~30-60s | **无丢失** (机架感知保证副本在不同机架) | PD 批量选举新 Leader → 补副本 |
| **3 Store 同机架宕机** (极端) | 若某 Region 3 副本恰好在同机架 | ~10s | **不可恢复** | **可能丢失该 Region 数据** | 告警 → 人工介入；机架感知配置防止此场景 |

**TiKV 建议调参** (2000 节点)：

```toml
# tikv.toml
[raftstore]
raft-base-tick-interval = "1s"
raft-heartbeat-ticks = 2              # 2s 心跳
raft-election-timeout-ticks = 10      # 10s 选举超时

# pd.toml
[schedule]
max-store-down-time = "30m"           # 30min 后补副本（防抖动）
region-schedule-limit = 2048          # 2000 节点需要高并发调度
replica-schedule-limit = 64

[replication]
location-labels = ["zone", "rack", "host"]   # 机架感知
max-replicas = 3
```

### 7.2 Compute Node Crash

| 场景 | 影响范围 | 检测时间 | 恢复时间 | 数据影响 | 自动恢复动作 |
|------|----------|----------|----------|----------|-------------|
| **查询中 Node 宕机** | 该节点正在处理的查询丢失 | <1s (Aggregator 超时) | <1s (切换到其他节点) | **无** (查询无状态) | Aggregator Blacklist → 路由到其他节点 |
| **写入中 Node 宕机** (作为路由发送方) | 未完成的 Append | SWIM ~5-15s | 自动 (WAL重放) | **无** (WAL 已在 TiKV) | 节点恢复后重放 WAL；或 Client 重试 |
| **写入中 Node 宕机** (作为 Owner) | Owner 的 headID 暂无写入锁保护 | SWIM ~5-15s | ~15-30s (Ring 更新) | **无** (数据在 TiKV) | SWIM 检测 → RemoveNode → headID 重分配 → Pending Append 重路由 |
| **Split 进行中 Node 宕机** | Split 中断 | SWIM ~5-15s | ~30-60s (Intent 恢复) | **无** (Intent 在 TiKV) | 新 owner 检测 Intent → Resume 或 Rollback |
| **Merge 进行中 Node 宕机** | Merge 中断，Lock 未释放 | SWIM ~5-15s | ~30s (Lock TTL 过期) | **无** (Intent 在 TiKV) | Lock TTL 过期 → 新 owner 恢复 Intent |
| **HeadSync 广播中 Node 宕机** | 部分节点未收到更新 | SWIM ~5-15s | ~10-60s (拉取追赶) | **无** (Log 在 TiKV) | 其他节点检测 gap → 从 TiKV 拉取 HeadSync Log |
| **Node 恢复后** | — | 自动注册 | ~30-120s (加载 Head Index) | — | 1.注册到 PD 2.加载 Head Index 3.追赶 HeadSync 4.AddNode 回到 Ring |

### 7.3 Compute Node Crash 恢复完整流程图

```
  Crashed Node M          SWIM Network           PD (Cluster Mgr)      Other Nodes
      │                       │                       │                     │
      ✕ CRASH                 │                       │                     │
                              │                       │                     │
                   ┌──────────▼──────────┐            │                     │
                   │ SWIM: Node M 未响应  │            │                     │
                   │ ping (0-5s)         │            │                     │
                   │ ping-req via proxy  │            │                     │
                   │ (5-10s)             │            │                     │
                   │ → 判定 M = SUSPECT   │            │                     │
                   └──────────┬──────────┘            │                     │
                              │                       │                     │
                              │  Gossip: M is SUSPECT │                     │
                              │──────────────────────▶│                     │
                              │───────────────────────┼────────────────────▶│
                              │                       │                     │
                   ┌──────────▼──────────┐            │                     │
                   │ 确认: M is DOWN      │            │                     │
                   │ (15s total)          │            │                     │
                   └──────────┬──────────┘            │                     │
                              │                       │                     │
                              │  Broadcast: M is DOWN │                     │
                              │──────────────────────▶│  Update node list   │
                              │───────────────────────┼────────────────────▶│
                              │                       │                     │
                              │                       │                     ▼
                              │                       │          ┌─────────────────┐
                              │                       │          │ RemoveNode(M)   │
                              │                       │          │ from Hash Ring  │
                              │                       │          │ Add M to        │
                              │                       │          │ Blacklist       │
                              │                       │          │ Recover M's     │
                              │                       │          │ Split/Merge     │
                              │                       │          │ Intents         │
                              │                       │          └─────────────────┘
                              │                       │                     │
  ┌───────────────┐           │                       │                     │
  │ Node M Restart│           │                       │                     │
  └───────┬───────┘           │                       │                     │
          │                   │                       │                     │
          │  Register to PD   │                       │                     │
          │──────────────────────────────────────────▶│                     │
          │                   │                       │                     │
          │  Load Head Index (from shared storage)    │                     │
          │  Pull HeadSync Log (catch up)             │                     │
          │                   │                       │                     │
          │  Ready → AddNode(M)                       │                     │
          │──────────────────────────────────────────▶│  Broadcast          │
          │                   │                       │────────────────────▶│
          │                   │                       │                     │
          │  Remove from Blacklist                    │                     │
          │                   │                       │                     ▼
          │                   │                       │          ┌─────────────────┐
          │                   │                       │          │ AddNode(M)      │
          │                   │                       │          │ to Hash Ring    │
          │                   │                       │          │ Remove from     │
          │                   │                       │          │ Blacklist       │
          │  ◀── Start serving queries & writes       │          └─────────────────┘
          │                   │                       │                     │
```

### 7.4 PD Crash

| 场景 | 影响 | 恢复 |
|------|------|------|
| 1 PD 宕机 (5 节点) | **无影响**，Raft 多数派存活 | 自动 Leader 选举 |
| 2 PD 同时宕机 | **无影响**，3/5 多数派存活 | 自动 Leader 选举 |
| 3 PD 同时宕机 | **PD 不可用** → TiKV Region 调度暂停 + Compute 注册暂停；已有路由缓存仍可读写 | 恢复任意 1 个 PD 即可恢复服务 |

### 7.5 Aggregator Crash

| 场景 | 影响 | 恢复 |
|------|------|------|
| Aggregator 宕机 | 该实例正在处理的查询丢失 | LB <1s 切换到其他实例；完全无状态，无需恢复 |

### 7.6 Client Crash

| 场景 | 影响 | 恢复 |
|------|------|------|
| Client 进程崩溃 | 该 Client 正在等待的请求丢失 | 应用层重启 Client；已提交写入不受影响 (Synchronous Write) |
| Client 网络断开 | 同上 | Client 重连后可安全重试（查询幂等、写入 VID+Ver 去重）|
| Client CircuitBreaker OPEN | 该 Aggregator 暂时不可用 | 自动切换到其他 Aggregator；30s 后 HALF_OPEN 探测恢复 |

---

## 8. 故障检测：Gossip 成员管理 (SWIM)

### 8.1 为什么用 SWIM 而不是集中式心跳

| 方面 | 集中式心跳 | SWIM Gossip |
|------|-----------|-------------|
| 单点依赖 | 依赖中心节点 | **去中心化**，无单点 |
| 扩展性 | O(N) 连接集中在一个节点 | **O(1)** 每节点固定探测数 |
| 检测速度 | 取决于心跳间隔 | **亚秒级**（并行探测） |
| 2000 节点 | 中心节点压力大 | **天然适合** |
| 网络开销 | N 个心跳/周期 | **O(N)** 总 gossip 消息，均匀分布 |

### 8.2 SWIM 协议流程

```
  Node A                  Node B (被探测)              Node C (代理)
    │                         │                           │
    │  ① Ping (每 T=1s 随机   │                           │
    │     选一个节点)          │                           │
    │────────────────────────▶│                           │
    │                         │                           │
    │  ┌─ 收到 Ack? ─────────┤                           │
    │  │  YES → B is ALIVE    │                           │
    │  │  NO (timeout=500ms)  │                           │
    │  │       ↓              │                           │
    │  │                      │                           │
    │  │  ② Ping-Req (通过 K  │                           │
    │  │     个随机代理节点)   │                           │
    │  │─────────────────────────────────────────────────▶│
    │  │                      │                    ┌──────┤
    │  │                      │                    │ Ping │
    │  │                      │◀───────────────────┘      │
    │  │                      │                           │
    │  │                      │  ┌─ Ack? ────────────────┤
    │  │                      │  │  YES → 代理转发 Ack    │
    │  │◀────────────────────────┤                       │
    │  │  B is ALIVE           │  │  NO (timeout=500ms)  │
    │  │                      │  │       ↓               │
    │  │                      │  │  向 A 报告无响应       │
    │  │◀────────────────────────┤                       │
    │  │                      │                           │
    │  │  ③ K 个代理都无响应 → B is SUSPECT               │
    │  │     Gossip 传播 SUSPECT 状态                      │
    │  │                      │                           │
    │  │  ④ SUSPECT 超时(10s) → B is DEAD                 │
    │  │     Gossip 传播 DEAD → 触发 RemoveNode(B)        │
    │  │                      │                           │
    └──┘                      │                           │
```

### 8.3 SWIM 参数 (2000 节点)

| 参数 | 值 | 说明 |
|------|-----|------|
| Ping 间隔 T | 1s | 每秒每节点探测 1 个随机节点 |
| Ping 超时 | 500ms | 直接 ping 超时 |
| Ping-Req 代理数 K | 3 | 通过 3 个代理确认 |
| Ping-Req 超时 | 500ms | 代理 ping 超时 |
| SUSPECT 到 DEAD 超时 | 10s | 给 SUSPECT 节点恢复的机会 |
| 全集群传播时间 | O(log N) ≈ 11 轮 gossip | 2000 节点约 11s 全传播 |
| **总检测时间** | **~5-15s** | 从故障到全集群知晓 |

### 8.4 Owner Blacklist 机制

```
┌─────────────────────────────────────────────────────────┐
│                    Owner Blacklist                        │
│                                                         │
│  当 Node B 被判定为 SUSPECT 或 DEAD:                      │
│                                                         │
│  1. 所有节点将 Node B 加入本地 Blacklist (TTL=60s)        │
│                                                         │
│  2. 后续请求跳过 Node B:                                  │
│                                                         │
│     GetOwner(headID) → Node B (in Blacklist!)            │
│           │                                              │
│           ▼                                              │
│     Bypass: 直接写 TiKV / 路由到 Ring 中下一个节点         │
│                                                         │
│  3. 避免长时间阻塞:                                       │
│     ┌──────────────────────────────┐                     │
│     │ 无 Blacklist: 每次请求等 30s  │                     │
│     │              超时才发现故障   │                     │
│     ├──────────────────────────────┤                     │
│     │ 有 Blacklist: 立即跳过        │ ← 关键优化          │
│     │              零等待时间       │                     │
│     └──────────────────────────────┘                     │
│                                                         │
│  4. Node B 恢复后:                                        │
│     SWIM ALIVE gossip → 从 Blacklist 移除 → AddNode(B)   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 9. 弹性伸缩设计

### 9.1 Compute Node 扩容流程图

```
  New Node N          PD (Cluster Mgr)         Existing Nodes
      │                     │                       │
      │  ① Register         │                       │
      │────────────────────▶│                       │
      │                     │                       │
      │  ② Get cluster info │                       │
      │  (member list,      │                       │
      │   Head Index loc)   │                       │
      │◀────────────────────│                       │
      │                     │                       │
      │  ③ Load Head Index  │                       │
      │  (from shared       │                       │
      │   storage / peer)   │                       │
      │                     │                       │
      │  ④ Pull HeadSync    │                       │
      │  Log (catch up)     │                       │
      │                     │                       │
      │  ⑤ Join SWIM        │                       │
      │  gossip network     │                       │
      │                     │                       │
      │  ⑥ Ready signal     │                       │
      │────────────────────▶│                       │
      │                     │                       │
      │                     │  ⑦ Broadcast:         │
      │                     │  AddNode(N)           │
      │                     │──────────────────────▶│
      │                     │                       │
      │                     │                       ▼
      │                     │              ┌────────────────┐
      │                     │              │ Update Hash    │
      │                     │              │ Ring: ~1/N     │
      │                     │              │ keys remap     │
      │                     │              │ to Node N      │
      │                     │              │                │
      │                     │              │ Lazy Migration:│
      │                     │              │ old owner still│
      │                     │              │ serves until   │
      │                     │              │ first access   │
      │                     │              └────────────────┘
      │                     │                       │
      │  ⑧ Start serving    │                       │
      │  queries & writes   │                       │
      │                     │                       │
```

### 9.2 扩容/缩容对比表

| 方面 | 扩容 (Add Node) | 优雅缩容 (Remove Node) | 非优雅缩容 (Crash) |
|------|-----------------|----------------------|-------------------|
| **触发** | 手动/自动扩容 | 运维主动下线 | 节点宕机 |
| **数据迁移** | Lazy (按需) | 无需 (数据在 TiKV) | 无需 |
| **Hash Ring 影响** | ~1/N keys remap | ~1/N keys remap | ~1/N keys remap |
| **服务中断** | 无 | 无 (排空后下线) | 查询 <1s, 写入 <15s |
| **操作排空** | N/A | 等待 Split/Merge 完成 | Lock TTL 自动释放 |
| **头索引同步** | 新节点加载 + 追赶 | N/A | N/A |
| **Blacklist** | 无 | 移除后加入 Blacklist | SWIM 检测后加入 |
| **恢复时间** | ~30-120s (加载) | 即时 | ~15-30s (检测+Ring更新) |

### 9.3 TiKV 扩容/缩容

TiKV 原生支持弹性伸缩，无需 SPTAG 侧额外操作：

```
扩容: 启动新 TiKV Store → 注册到 PD → PD 自动 Rebalance Region
缩容: tikv-ctl store remove → PD 迁移 Region → 安全下线
```

Compute Node 通过 PD 自动发现 TiKV 拓扑变化，**无需配置变更**。

---

## 10. 实现路线图：PR 计划

### 10.1 PR 依赖关系图

```
  PR1 ─────────────────────────────────────┐
  (TiKV WAL)                               │
  🛡️ 写入容错                               │
       │                                   │
  PR2 ─┤                                   │
  (Lock TTL +                              │
   Intent SM)                              │
  🛡️ 写入容错                               │
       │                                   │
  PR3 ─┴─── PR4 ─── PR6                   │
  (SWIM)    (Blacklist   (Write Bypass)    │
  🔍 故障     Read Path)  🛡️ 写入容错        │
  检测      🛡️ 查询容错        │            │
                              │            │
  PR5 ───────────────────┬────┘            │
  (PD Controller         │                 │
   + Epoch)              │                 │
  🛡️ 集群容错             │                 │
       │            PR7 ─┤                 │
       │            (Aggregator)           │
       │            🛡️ 查询容错             │
       │                 │                 │
       │            PR8 ─┘                 │
       │            (Client SDK)           │
       │            🛡️ 端到端容错            │
       │                                   │
  PR9 ─┤                                   │
  (HeadSync Log)                           │
  🛡️ 数据容错                               │
       │                                   │
  PR10 ┤                                   │
  (Follower Read                           │
   + Head Checkpoint)                      │
  🛡️ 读容错+恢复                             │
       │                                   │
  PR11 ┘                                   │
  (Elastic Scaling)                        │
  📈 弹性伸缩                               │
                                           │
  PR12 ────────────────────────────────────┘
  (Metrics + CAS +                         
   GC + Chaos Test)                        
  🛡️ 容错验证                               
```

### 10.2 PR 总览

| PR | 标题 | 容错类别 | 解决什么故障场景 | 改动范围 | 大小 | 依赖 |
|----|------|---------|----------------|---------|------|------|
| **PR1** | TiKV WAL | 🛡️ **写入容错** | Node crash 后未完成的 Insert/Delete 能重放恢复 | `ExtraDynamicSearcher.h`, `ExtraTiKVController.h` | M | 无 |
| **PR2** | Remote Lock TTL + Split/Merge Intent 状态机 | 🛡️ **写入容错** | Crash 死锁自动释放；Split/Merge 中途 crash 能 resume/rollback | `PostingRouter.h`, 新增 `IntentManager` | M | PR1 |
| **PR3** | SWIM Gossip Agent | 🔍 **故障检测** | 2000 节点中任意节点宕机，亚秒级去中心化发现 | 新增 `SwimAgent.h/.cpp` | L | 无 |
| **PR4** | Owner Blacklist (仅读/查询路径) | 🛡️ **查询容错** | 查询不再等待已死节点超时，立即跳过 | `PostingRouter.h` | S | PR3 |
| **PR5** | PD 复用为 Cluster Controller + Ring Epoch | 🛡️ **集群容错** | 防止脑裂/stale ring；节点成员权威管理 | `PostingRouter.h`, PD 交互层 | M | 无 |
| **PR6** | Write Failover (Owner 不可达时 Bypass) | 🛡️ **写入容错** | Owner 宕机时写入不阻塞，降级直写 TiKV | `PostingRouter.h` | S-M | PR1,2,4 |
| **PR7** | Aggregator Service + 部分降级 | 🛡️ **查询容错** | 查询入口多实例无单点；部分节点超时仍返回降级结果 | 新增 `Aggregator/` 目录 | L | PR3,4,5 |
| **PR8** | Client SDK (Retry/CircuitBreaker/LB) | 🛡️ **端到端容错** | Client 侧重试/熔断/LB，Aggregator 挂了自动切换 | 新增 `Client/` 目录 | M | PR7 |
| **PR9** | HeadSync Durable Log + Pull 模式 | 🛡️ **数据容错** | 节点重启后不丢 HeadSync 消息，拉取补齐 | `PostingRouter.h`, TiKV log 存储 | M | 无 |
| **PR10** | TiKV Follower Read + Head Index 共享 Checkpoint | 🛡️ **读容错+恢复** | TiKV Leader 不可达降级读；节点快速恢复 | `ExtraTiKVController.h`, Checkpoint 逻辑 | M | PR9 |
| **PR11** | Elastic Scaling (手动 Add/Remove + Graceful Drain) | 📈 **弹性伸缩** | 新节点加入 + 旧节点优雅退出 + 批量操作 | `PostingRouter.h`, PD 交互 | L | PR5,9,10 |
| **PR12** | 生产加固 (Metrics, Merge CAS, State GC, Chaos Test) | 🛡️ **容错验证** | 证明以上容错机制真的有效；状态清理防泄露 | 全局 | L | 全部 |

> **12 个 PR 里 11 个是容错相关，1 个 (PR11) 是弹性伸缩。**

### 10.3 PR 详细说明

#### PR1: TiKV WAL — 🛡️ 写入容错

**问题**: 当前 WAL 仅支持 RocksDB 后端 (`ExtraDynamicSearcher.h:2644`)。使用 TiKV 后端时 Node crash = 未持久化的写入丢失。

**改动**:
- Insert/Delete 前先写 WAL entry 到 TiKV (key: `wal/{nodeId}/{seqNo}`)
- WAL entry 格式: `{op, vid, version, headID, vector, metadata}`
- Recovery 模式启动时从 TiKV 读取未完成 WAL 条目并重放
- Checkpoint 时清理已完成的 WAL 条目

**核心代码路径**:
```
ExtraDynamicSearcher::AddIndex()
  现在: RocksDB WAL → AddIndex → Checkpoint 清 WAL
  改后: TiKV WAL → AddIndex → Checkpoint 清 WAL
```

**测试**: 写入 100 条 → kill 进程 → 重启 → 验证 100 条全部恢复

---

#### PR2: Remote Lock TTL + Split/Merge Intent 状态机 — 🛡️ 写入容错

**问题 A**: Remote Lock (`PostingRouter.h:1780`) 无超时。Owner crash 后 Lock 永不释放 → 死锁。

**改动 A**: Lock 加 TTL (30s)，TiKV 存储 `{lockHolder, expireTime}`，获取 Lock 时检查过期。

**问题 B**: Split/Merge 是多步操作（读 Posting → 修改 → 写回 → 更新 Head）。中途 crash → 半完成状态。

**改动 B**:
```
Intent 状态机 (持久化到 TiKV):

  ┌──────────┐    ┌───────────┐    ┌──────────────┐    ┌───────────┐
  │ PREPARED │───▶│ EXECUTING │───▶│ HEAD_UPDATED │───▶│ COMMITTED │
  └──────────┘    └───────────┘    └──────────────┘    └───────────┘
       │               │                  │
       └── Crash ──────┴─── Crash ────────┘
            ↓               ↓                  
         Rollback      Resume from          
                       last state           
```

**测试**: Split 执行到 EXECUTING 阶段 → kill → 重启 → 验证自动 resume

---

#### PR3: SWIM Gossip Agent — 🔍 故障检测

**问题**: 2000 节点集群无故障检测机制。节点宕机只能靠请求超时被动发现 (2-5s 延迟)。

**改动**: 每个 Compute Node 内嵌 SWIM Agent:
- 每秒随机选 1 个节点 Ping
- Ping 超时 → 通过 K=3 个代理节点 Ping-Req
- Ping-Req 也超时 → 标记 SUSPECT
- SUSPECT 超过 10s → 标记 DEAD
- 状态变更 piggyback 到正常 Ping 消息中传播

**代码**: 新增 `AnnService/inc/Core/SPANN/SwimAgent.h`，独立线程运行，通过回调通知 PostingRouter。

**测试**: 3 节点集群 → kill 1 个 → 验证其他 2 个在 15s 内标记为 DEAD

---

#### PR4: Owner Blacklist (仅读/查询路径) — 🛡️ 查询容错

**问题**: 查询 fan-out 到 Compute Node 时，如果某节点已 DEAD，Aggregator 仍然发送请求并等待超时 (500ms-2s)。

**改动**:
- SWIM DEAD 事件 → 加入 Blacklist (TTL 60s)
- `PostingRouter::GetOwner()` 跳过 Blacklist 中的节点
- Aggregator fan-out 跳过 Blacklist 节点
- **仅改查询/读路径**，不改写入路径（写入 bypass 需要更多安全保证，在 PR6 做）

**为什么拆分读/写**: 读路径跳过故障节点是安全的（Head Index 是全量副本，任何节点都能搜索）。写路径跳过 Owner 涉及一致性风险，需要 WAL + Intent 先就绪。

**测试**: 3 节点 → kill Owner → 查询自动跳过 → latency 无明显增加

---

#### PR5: PD 复用为 Cluster Controller + Ring Epoch — 🛡️ 集群容错

**问题 A**: Compute Node 成员列表硬编码在配置中 (`Options.h:218` RouterNodeStores)。节点变更需要改配置重启。

**问题 B**: 无 Ring 版本号 → Ring 更新后旧节点拿着 stale ring 路由 → 写到错误 Owner → 数据不一致。

**改动**:
- Compute Node 启动时向 PD 注册 (`PUT /compute-nodes/{nodeId}`)
- PD 维护权威成员列表 + Ring 版本号 (Epoch)
- 每次 Ring 变更 (Add/Remove Node) → Epoch+1
- RPC 请求携带 Epoch，接收方校验：Epoch 不匹配 → 拒绝并返回最新 Ring

**测试**: 2 节点用旧 Epoch 发请求 → 被拒绝 → 自动拉取新 Ring → 重试成功

---

#### PR6: Write Failover (Owner 不可达时 Bypass) — 🛡️ 写入容错

**问题**: Owner 宕机时写入请求阻塞 → 超时 → 失败。

**前提**: PR1 (WAL 持久化) + PR2 (Intent + Lock TTL) + PR4 (Blacklist) 已就绪。

**改动**:
```
  正常写入路径:
    Client → Node A → PostingRouter → Owner Node B → TiKV
  
  Owner 不可达 (Blacklist 命中):
    Client → Node A → PostingRouter → Owner 在 Blacklist
                                    → 降级: Node A 直接写 TiKV
                                    → WAL 记录 bypass 标记
                                    → Owner 恢复后 reconcile
```

**为什么必须等 PR1/2/4**: bypass 写入跳过了 Owner 的本地缓存更新和 Lock 保护。必须有 WAL 保证持久化 + Intent 保证 Split/Merge 不冲突 + Blacklist 保证只在确认 DEAD 时才 bypass。

**测试**: 写入过程中 kill Owner → 写入自动 bypass → Owner 恢复后数据一致

---

#### PR7: Aggregator Service + 部分降级 — 🛡️ 查询容错

**问题**: 当前 Driver 节点 = Client + Aggregator + Compute Node 三合一 → 单点故障 → 全部不可用。

**改动**:
- 新增独立 Aggregator 进程 (`AnnService/Aggregator/`)
- 无状态，可部署 M ≥ 3 个实例
- 接收查询 → 选 K 个健康 Compute Node fan-out → 合并 Top-K 结果
- 部分节点超时 → 返回结果 + `degraded=true` 标记
- 集成 Blacklist: 不向 Blacklist 中的节点发送查询

```
  Client → LB → Aggregator A (挂了) → LB → Aggregator B (接管)
                                            │
                    ┌───────────────────────┤─────────────────┐
                    ▼                       ▼                 ▼
              Compute Node 1         Compute Node 2    Compute Node K
              (正常返回)             (超时, 跳过)       (正常返回)
                    │                                         │
                    └─────────────┬───────────────────────────┘
                                  ▼
                        Merge Top-K (degraded=true)
                                  │
                                  ▼
                              Client 收到结果
```

**测试**: 5 Compute Node + 2 Aggregator → kill 1 Aggregator + 1 Compute → 查询仍成功 (degraded)

---

#### PR8: Client SDK — 🛡️ 端到端容错

**问题**: 没有标准化的 Client 库，应用直连 Aggregator 需自己处理重试/超时/故障切换。

**改动**: 新增 `SPANNClient` SDK:
- **ConnectionPool**: 维护到多个 Aggregator 的连接，健康检查
- **RetryPolicy**: 查询 max 3 次 exp backoff；写入带 VID+Version 去重安全重试
- **CircuitBreaker**: 每 Aggregator 独立，5 次失败/30s → OPEN → 30s → HALF_OPEN → 1 成功 → CLOSED
- **LB**: Round-Robin / Weighted，自动跳过 OPEN 状态的 Aggregator
- **Response**: `result.degraded` 标记、`result.latency_ms`

**接口**:
```cpp
SPANNClient client({"agg1:8080", "agg2:8080", "agg3:8080"});
auto result = client.Search(query_vec, top_k);  // 自动重试/熔断/LB
auto ack = client.Insert(vid, vec, meta);        // ack.persistent = true
```

**测试**: kill Aggregator → Client 自动切换 → 延迟增加 <1s → CircuitBreaker 标记 → 恢复后自动重连

---

#### PR9: HeadSync Durable Log + Pull 模式 — 🛡️ 数据容错

**问题**: HeadSync 当前是纯推送广播 (`PostingRouter.h`)。节点重启 → 错过期间所有 Head Index 变更 → Head Index 和其他节点不一致。

**改动**:
- HeadSync 事件持久化到 TiKV (`headsync/{seqNo}` → `{op, headID, vector}`)
- 推送: 和现在一样广播，附带 seqNo
- 拉取: 节点启动后用本地 lastSeqNo 向 TiKV 拉取 `[lastSeqNo+1, latest]`
- 日志保留: 保留最近 10 万条 (或 7 天)，更早的靠 Head Index Checkpoint

**测试**: 节点停机 10 分钟 → 期间有 1000 次 HeadSync → 重启后拉取补齐 → Head Index 一致

---

#### PR10: TiKV Follower Read + Head Index 共享 Checkpoint — 🛡️ 读容错+恢复

**改动 A — Follower Read**:
- `ExtraTiKVController.h` 查询路径: Leader 超时 → fallback Follower Read
- SPANN 是近似搜索，stale data = 精度微降，不影响正确性
- 配置: `FollowerReadEnabled=true` (默认开启)

**改动 B — Head Index Checkpoint**:
- 定期 (每 10 min) 将 Head Index checkpoint 写入共享存储 (TiKV 或 NFS)
- 新节点/恢复节点: 加载最近 checkpoint + 拉取 HeadSync Log 补齐 (来自 PR9)
- 避免每次都从零重建 Head Index (当前需要全量数据扫描)

**测试 A**: kill TiKV Leader → 查询自动 Follower Read → latency 增加 <5ms
**测试 B**: 新节点加入 → 加载 checkpoint + pull HeadSync → 30s 内就绪

---

#### PR11: Elastic Scaling — 📈 弹性伸缩

**这是唯一一个非容错 PR。**

**改动**:
- **扩容**: 新 Compute Node → PD 注册 → 加载 Head Checkpoint → 加入 Ring → Lazy Migration
- **缩容**: 标记 Draining → 停止接受新写入 → 排空 in-flight → RemoveNode → PD 注销
- **批量**: 多个节点同时 Add/Remove → 原子 Ring 更新 (单次 Epoch bump)

```
  扩容:  New Node ──▶ PD 注册 ──▶ 加载 Head ──▶ 加入 Ring ──▶ Lazy Migration
  缩容:  Old Node ──▶ 标记 Drain ──▶ 排空请求 ──▶ 移出 Ring ──▶ PD 注销
```

**依赖**: PR5 (PD), PR9 (HeadSync Log), PR10 (Head Checkpoint)

**测试**: 3 节点 → 扩到 5 → 缩到 4 → 全程查询/写入不中断

---

#### PR12: 生产加固 — 🛡️ 容错验证

**改动**:
- **Prometheus Metrics**: SWIM 状态变更、Blacklist 命中率、WAL 重放次数、查询 degraded 比例、Ring Epoch
- **Merge CAS**: `TiKV RawCAS` 替代当前非原子 read-modify-write (`ExtraTiKVController.h:315`)
- **State GC**: WAL 条目清理 (checkpoint 后)、Intent 超时清理 (5min)、HeadSync Log 截断 (10 万条)
- **Chaos Test**: 故障注入框架 — 随机 kill 节点/断网/延迟注入 → 验证所有容错机制

**测试**: 10 节点集群 → 持续查询写入 → 随机 kill 2 节点/轮 → 运行 1 小时 → 零数据丢失 + 零中断

### 10.4 PR 规模与节奏

```
  大小说明:  S = ~300 行   M = ~500-1500 行   L = ~2000+ 行

  ┌─────┬──────────────────────────────────────┬──────┬─────────┐
  │ PR  │ 标题                                 │ 大小 │ 建议周期 │
  ├─────┼──────────────────────────────────────┼──────┼─────────┤
  │ PR1 │ TiKV WAL                             │  M   │  1-2 周  │
  │ PR2 │ Lock TTL + Intent SM                 │  M   │  1-2 周  │
  │ PR3 │ SWIM Gossip Agent                    │  L   │  2-3 周  │
  │ PR4 │ Owner Blacklist (读路径)              │  S   │  <1 周   │
  │ PR5 │ PD Controller + Epoch                │  M   │  1-2 周  │
  │ PR6 │ Write Bypass                         │ S-M  │  1 周    │
  │ PR7 │ Aggregator Service                   │  L   │  2-3 周  │
  │ PR8 │ Client SDK                           │  M   │  1-2 周  │
  │ PR9 │ HeadSync Log                         │  M   │  1-2 周  │
  │ PR10│ Follower Read + Head Checkpoint      │  M   │  1-2 周  │
  │ PR11│ Elastic Scaling                      │  L   │  2-3 周  │
  │ PR12│ 生产加固                              │  L   │  2-3 周  │
  ├─────┼──────────────────────────────────────┼──────┼─────────┤
  │     │ 总计                                 │      │ ~18-26 周│
  └─────┴──────────────────────────────────────┴──────┴─────────┘

  可并行的 PR (无依赖关系):
  ─────────────────────────────
  • PR1 ∥ PR3 ∥ PR5 ∥ PR9   (四个可同时开始)
  • PR2 ∥ PR4                (PR1/PR3 各自完成后)
  • PR7 ∥ PR10               (依赖满足后)
  
  利用并行，关键路径可压缩到 ~12-16 周。
```

### 10.5 关键拆分决策说明

**为什么 Blacklist 拆成读 (PR4) 和写 (PR6)**:

```
  读路径跳过故障 Owner:
  ┌─────────────────────────────────────────────────┐
  │  Head Index 是全量副本 → 任何节点都能搜索        │
  │  跳过 DEAD 节点 = 结果不变，只是少一个 source   │
  │  风险: 极低                                     │
  └─────────────────────────────────────────────────┘

  写路径 bypass 故障 Owner:
  ┌─────────────────────────────────────────────────┐
  │  Owner 维护 Posting 的本地缓存和 Lock 保护       │
  │  bypass Owner → 跳过 Lock → 可能和 Split/Merge  │
  │  冲突 → 数据不一致                               │
  │  必须等 WAL + Intent + Lock TTL 先就绪           │
  │  风险: 高，需要更多安全保证                       │
  └─────────────────────────────────────────────────┘
```

**为什么 Aggregator (PR7) 和 Client SDK (PR8) 分开**:
- Aggregator 是内部组件，改的是集群内部通信协议
- Client SDK 是外部 API 契约（`degraded` 标记、重试语义），需要独立评审
- 两者接口变更影响不同的调用方

---

## 附录

### A. 2000 节点容量估算

| 参数 | 值 |
|------|-----|
| 总向量数 | 10 亿 (假设) |
| 向量维度 | 128, UInt8 |
| 每向量存储 | ~140 bytes (含元数据) |
| 总数据量 (原始) | ~130 GB |
| TiKV 3 副本 | ~390 GB |
| 每 Store | ~195 MB |
| Head Index | ~1-5 GB (每 Compute Node) |

### B. 关键配置参数

```ini
[FaultTolerance]
# SWIM Gossip
SwimPingIntervalMs=1000               # Ping 间隔
SwimPingTimeoutMs=500                 # Ping 超时
SwimPingReqProxyCount=3               # Ping-Req 代理数
SwimSuspectTimeoutMs=10000            # SUSPECT→DEAD 超时
BlacklistTTLMs=60000                  # Blacklist 过期时间

# Write Safety
RemoteLockTTLMs=30000                 # Remote Lock 超时
SplitMergeIntentTimeoutMs=300000      # Intent 超时 (5min)
AppendRetryCount=3                    # Append 重试次数

# Query Safety
QueryTimeoutMs=2000                   # 查询总超时
QueryRetryCount=2                     # 查询重试次数
FollowerReadEnabled=true              # 允许 Follower Read

[ElasticScaling]
ConsistentHashVnodes=150              # 虚拟节点数
LazyMigrationTransitionMs=30000       # 惰性迁移过渡期
BatchAddNodeMaxConcurrent=100         # 批量扩容最大并发
HeadSyncCheckIntervalMs=600000        # 全量校验间隔 (10min)
```

### C. 当前代码已有的容错能力

| 机制 | 文件 | 状态 |
|------|------|------|
| TiKV Region 重试 (10次) | `ExtraTiKVController.h` | ✅ 已实现 |
| Region Cache 失效+重路由 | `ExtraTiKVController.h` | ✅ 已实现 |
| RawBatchGet Fallback | `ExtraTiKVController.h` | ✅ 已实现 |
| PostingRouter 连接重试 | `PostingRouter.h` | ✅ 已实现 |
| 一致性哈希 Add/RemoveNode | `PostingRouter.h` | ✅ 已实现 |
| ComputeMigration | `PostingRouter.h` | ✅ 已实现 |
| WAL (RocksDB only) | `ExtraDynamicSearcher.h` | ⚠️ 仅 RocksDB |
| Checkpoint / Recovery | `ExtraDynamicSearcher.h` | ✅ 已实现 |
| VersionMap TiKV Cache | `TiKVVersionMap.h` | ✅ 已实现 |
| Remote Lock | `PostingRouter.h` | ✅ 已实现 |
| HeadSync 广播 | `PostingRouter.h` | ✅ 已实现 |
| SWIM Gossip | — | ❌ 未实现 |
| Owner Blacklist | — | ❌ 未实现 |
| TiKV WAL | — | ❌ 未实现 |
| Split/Merge Intent | — | ❌ 未实现 |
| Remote Lock TTL | — | ❌ 未实现 |
| HeadSync Log 持久化 | — | ❌ 未实现 |
| Follower Read | — | ❌ 未实现 |
| Epoch 机制 | — | ❌ 未实现 |
