# SPTAG/SPANN 容错与弹性伸缩设计文档

> 本文档合并了两份独立设计：容错/弹性伸缩方案与对等模式架构设计，取各自最优部分。

## 目录

1. [现状分析与升级瓶颈](#1-现状分析与升级瓶颈)
2. [系统架构概览](#2-系统架构概览)
3. [设计原则](#3-设计原则)
4. [基础机制](#4-基础机制)
5. [Client 层设计与 API 契约](#5-client-层设计与-api-契约)
6. [查询流容错](#6-查询流容错)
7. [写入流容错](#7-写入流容错)
8. [维护流容错：Split/Merge/HeadSync](#8-维护流容错splitmergehead-sync)
9. [Job Routing 容错](#9-job-routing-容错)
10. [节点 Crash 故障处理总表](#10-节点-crash-故障处理总表)
11. [弹性伸缩设计](#11-弹性伸缩设计)
12. [四条链路容错全景](#12-四条链路容错全景)
13. [实现路线图：PR 计划](#13-实现路线图pr-计划)
14. [附录](#附录)

---

## 1. 现状分析与升级瓶颈

当前 `PostingRouter` **路由层已经是对称的**——每个节点同时运行 Server 和 Client，一致性哈希决定 headID owner，`AddNode()`/`RemoveNode()` 支持环变更。"Driver vs Worker" 仅是测试入口的区别。

### 不可直接扩到 2000 节点的瓶颈

| 瓶颈 | 根因 | 量级（2000 节点） |
|------|------|--------------------|
| **Full-mesh TCP** | `Start()` 中每个 node 连接所有 peer | 2000 x 1999 = ~4M 半连接 |
| **VID 单调递增** | `m_iCurAssignedVID` 在发起 insert 的节点上自增 | 只有发起者分配 VID，需全局锁或分段 |
| **Head index 全量广播** | `BroadcastHeadSync()` fire-and-forget 到 N-1 个 peer | 一次 split/merge 产生 1999 条 TCP 广播 |
| **静态成员配置** | `RouterNodeAddrs` INI 写死 | 加节点需改配置重启全集群 |
| **无故障检测** | 只能靠请求超时被动发现节点故障 | 每次超时等 30s |
| **无 WAL/Intent** | TiKV 后端 crash = 未持久化写入丢失 | 数据不一致 |

---

## 2. 系统架构概览

### 2.1 整体架构

```
+------------------------------------------------------------------+
|              Clients (内嵌 SDK, Client-side LB)                   |
+------------------------------------------------------------------+
             | Query / Insert
             | (SDK 从 PD 获取健康节点列表, round-robin 选节点)
             v
+------------------------------------------------------------------+
|                  Compute Node x 2000                              |
|  +------------------------------------------------------------+  |
|  |  Head Index (SPTAG/BKT)  <- 每节点全量副本                  |  |
|  |  ExtraDynamicSearcher    <- Posting 读写 + Split/Merge      |  |
|  |  PostingRouter           <- 一致性哈希 headID->Node 路由    |  |
|  |  VersionMap              <- TiKV-backed 版本管理            |  |
|  |  SWIM Agent              <- Gossip 成员检测（去中心化）      |  |
|  |  ConnectionPool          <- 按需连接 + LRU 淘汰             |  |
|  |  VIDAllocator            <- 分段预分配，本地 atomic++        |  |
|  +------------------------------------------------------------+  |
+---------------------------------+--------------------------------+
                              | gRPC: RawGet/RawPut/RawBatchGet
                              v
+------------------------------------------------------------------+
|                   TiKV 集群 (3 副本 Raft)                         |
|  +-----------------------------------------------------------+   |
|  |  PD x 5 (Placement Driver, 复用为 Cluster Controller)      |   |
|  |  - TiKV Region 调度 / Store 路由                           |   |
|  |  - Compute Node 注册 / 成员列表 / Ring Epoch 管理          |   |
|  +-----------------------------------------------------------+   |
|  +------+ +------+ +------+ +------+         +-------+          |
|  |TiKV 0| |TiKV 1| |TiKV 2| |TiKV 3|  ...   |TiKV   |          |
|  |      | |      | |      | |      |         | 1999  |          |
|  +------+ +------+ +------+ +------+         +-------+          |
+------------------------------------------------------------------+
```

**关键架构决策**：
- **查询路径**：Client SDK 直连任意 Compute Node（Node 本地完成全部查询：Head Search → RawBatchGet posting → Top-K）
- **写入路径**：Client SDK 直连任意 Compute Node（PostingRouter 路由到 Owner 写入）
- **无 Aggregator 层**：Client 数量有限（后端服务 pod），SDK 内置 LB 即可
- **共享存储**：所有 Posting 在 TiKV (Raft 3 副本)，"Owner" 仅用于写入串行化，查询不经过 Owner

### 2.2 核心组件职责

| 组件 | 实例数 | 状态 | 职责 |
|------|--------|------|------|
| **Client SDK** | 任意 | 无状态 | 内嵌在业务服务中；从 PD 获取节点列表；Client-side LB；重试；CircuitBreaker |
| **Compute Node** | 2000 | 有状态* | Head Index 搜索，Posting 读写，Split/Merge，查询+写入入口 |
| **PostingRouter** | (内嵌) | 可重建 | 一致性哈希 headID->Node 路由（仅写入路径使用） |
| **TiKV Store** | 2000 | 持久化 | Posting/VersionMap/WAL/Intent/HeadSyncLog (Raft 3 副本) |
| **PD** | 5 | 持久化 | Region 调度 + Compute Node 集群管理 + Ring Epoch |

> *Compute Node "有状态" 是指 Head Index 副本和内存缓存。核心数据全部在 TiKV，Compute Node 本质上是**可重建**的。

### 2.3 当前 Benchmark 架构 vs 生产架构

当前 benchmark 只有 **1 个 Client**（即 Driver 进程 n0），它同时承担两个角色：

```
+----------------------------------------------------------------+
|              当前 Benchmark 架构 (1 Driver = 2 合 1)              |
|                                                                  |
|  +----------------------------------------------------------+   |
|  |  Driver (Node n0) = Client + Compute Node                 |   |
|  +----------------------------------------------------------+   |
|                       |                                          |
|         +-------------+-------------+                            |
|         v                           v                            |
|  +------------+             +------------+                       |
|  | Worker n1  |             | Worker n2  |                       |
|  +------------+             +------------+                       |
+----------------------------------------------------------------+

代码对应:
  Driver:  SPFreshTest/BenchmarkFromConfig  (run_scale_benchmarks.sh:673)
  Worker:  SPFreshTest/WorkerNode           (run_scale_benchmarks.sh:650)
```

生产环境分离为独立层（见 2.1 架构图）。

### 2.4 目标规模

| 参数 | 值 | 说明 |
|------|-----|------|
| TiKV Store | 2000 | Raft 3 副本 |
| Compute Node | 2000 | 1:1 对应 TiKV Store |
| PD | 5 | 容忍 2 同时故障 |
| 故障频率 | ~1-2 节点/周 | 2% 年故障率 x 2000 节点 |
| 设计容忍 | >=3 同时不可用 | 多机架同时故障场景 |

---

## 3. 设计原则

```
+--------------------------------------------------------------------------+
|                                                                          |
|  P1  Synchronous Write Semantics -- 诚实、不骗 client                    |
|      SUCCESS = 数据 100% 持久化在 TiKV (Raft 3 副本)                     |
|      FAIL / TIMEOUT = client 明确知道失败, 自行 retry                    |
|      绝不出现: client 收到 SUCCESS 但数据没写进去                         |
|                                                                          |
|  P2  Idempotent Retry -- retry 永远安全                                  |
|      所有写操作（Append/Split/Merge）均可安全重试                         |
|      VID + Version 去重，Intent 状态机防止重复执行                        |
|      同 Ingress + idempotency_key -> 缓存命中, 不重复执行                |
|                                                                          |
|  P3  Consistent Hashing -- 增删节点最小影响                               |
|      FNV-1a + 150 vnodes/node -> 增删 1 节点只影响 ~1/N 的 headID       |
|      RCU lock-free ring 更新 -> 读路径无锁                               |
|      TiKV 共享存储 -> ring 变更期间读写仍然正确                           |
|                                                                          |
|  P4  Gossip Membership (SWIM) -- 去中心化故障发现                         |
|      每秒 Ping 1 个 peer, 怀疑时 PingReq 间接确认                       |
|      O(log N) 轮收敛 (~11s @ 2000 nodes)                                 |
|      自动 AddNode/RemoveNode -> ring 自愈                                |
|                                                                          |
|  P5  Fast Failure Detection + Owner Blacklist                            |
|      Owner Blacklist (TTL): 第一次超时后标记, 后续请求 ms 级快速失败      |
|      与 Gossip SWIM 互补: blacklist 桥接 gossip 检测空窗期               |
|                                                                          |
|  P6  Local-First -- 热路径零网络                                          |
|      VID 分配: 静态分段 + 本地 atomic++ (零网络)                          |
|      路由决策: 本地 hash ring 查 GetOwner() (零网络)                      |
|      Head search: 本地 BKT index (零网络)                                |
|      唯一需要网络的热路径: remote Append 到 Owner                         |
|                                                                          |
|  P7  Graceful Degradation -- Search 永远可用                              |
|      Owner 不可达 -> TiKV 直读 fallback (任何 node 都能读)               |
|      TiKV 也失败 -> 跳过该 headID (recall 降低, 不报错)                  |
|      标注 degraded=true -> client 知道结果可能不完整                      |
|                                                                          |
|  P8  Crash Recovery -- 重启即恢复, 无手动干预                             |
|      VID 计数器: 扫 TiKV 恢复 (防冲突)                                   |
|      孤儿 posting: 无害 (数据正确, 上层过滤/GC 清理)                      |
|      WAL 重放: TiKV-backed WAL -> 未完成写入自动恢复                      |
|      Intent 状态机: Split/Merge 中途 crash -> resume 或 rollback          |
|                                                                          |
+--------------------------------------------------------------------------+
```

---

## 4. 基础机制

本节描述容错设计依赖的底层机制。后续章节的查询/写入/维护流容错均建立在这些机制之上。

### 4.1 SWIM Gossip 故障检测

#### 为什么用 SWIM 而不是集中式心跳

| 方面 | 集中式心跳 | SWIM Gossip |
|------|-----------|-------------|
| 单点依赖 | 依赖中心节点 | **去中心化**，无单点 |
| 扩展性 | O(N) 连接集中在一个节点 | **O(1)** 每节点固定探测数 |
| 检测速度 | 取决于心跳间隔 | **亚秒级**（并行探测） |
| 2000 节点 | 中心节点压力大 | **天然适合** |
| 网络开销 | N 个心跳/周期 | **O(N)** 总 gossip 消息，均匀分布 |

#### SWIM 协议流程

```
  Node A                  Node B (被探测)              Node C (代理)
    |                         |                           |
    |  (1) Ping (每 T=1s      |                           |
    |     随机选一个节点)      |                           |
    |------------------------>|                           |
    |                         |                           |
    |  +- 收到 Ack? ----------|                           |
    |  |  YES -> B is ALIVE   |                           |
    |  |  NO (timeout=500ms)  |                           |
    |  |       |              |                           |
    |  |                      |                           |
    |  |  (2) Ping-Req (通过  |                           |
    |  |     K=3 个代理节点)  |                           |
    |  |--------------------------------------------->|
    |  |                      |                    +------+
    |  |                      |                    | Ping |
    |  |                      |<-------------------+      |
    |  |                      |                           |
    |  |                      |  +- Ack? -----------------+
    |  |                      |  |  YES -> 代理转发 Ack   |
    |  |<------------------------+                        |
    |  |  B is ALIVE          |  |  NO -> 向 A 报告无响应 |
    |  |                      |  |       |                |
    |  |<------------------------+                        |
    |  |                      |                           |
    |  |  (3) K 个代理都无响应 -> B is SUSPECT            |
    |  |     Gossip 传播 SUSPECT 状态                     |
    |  |                      |                           |
    |  |  (4) SUSPECT 超时(10s) -> B is DEAD              |
    |  |     Gossip 传播 DEAD -> 触发 RemoveNode(B)       |
    +--+                      |                           |
```

#### SWIM 参数 (2000 节点)

| 参数 | 值 | 说明 |
|------|-----|------|
| Ping 间隔 T | 1s | 每秒每节点探测 1 个随机节点 |
| Ping 超时 | 500ms | 直接 ping 超时 |
| Ping-Req 代理数 K | 3 | 通过 3 个代理确认 |
| Ping-Req 超时 | 500ms | 代理 ping 超时 |
| SUSPECT 到 DEAD 超时 | 10s | 给 SUSPECT 节点恢复的机会 |
| 全集群传播时间 | O(log N) ~ 11 轮 gossip | 2000 节点约 11s 全传播 |
| **总检测时间** | **~5-15s** | 从故障到全集群知晓 |

#### Gossip 数据结构

```cpp
// 新增 PacketType
MembershipPing    = 0x10  // 心跳探测
MembershipPingReq = 0x11  // 间接探测
MembershipAck     = 0x12  // 心跳回复
MembershipUpdate  = 0x13  // 成员变更广播 (piggyback on any message)

struct MembershipEntry {
    int nodeIndex;
    std::string addr, port;
    uint64_t epoch;           // 全局递增 epoch (from PD)
    MemberState state;        // Alive | Suspect | Dead | Left
    uint32_t incarnation;     // 递增计数, 用于 refute suspect
};
```

### 4.2 Owner Blacklist

```
+-----------------------------------------------------------+
|                    Owner Blacklist                          |
|                                                            |
|  当 Node B 被判定为 SUSPECT 或 DEAD:                        |
|                                                            |
|  1. 所有节点将 Node B 加入本地 Blacklist (TTL=60s)          |
|                                                            |
|  2. 后续请求跳过 Node B:                                    |
|     GetOwner(headID) -> Node B (in Blacklist!)             |
|           |                                                |
|           v                                                |
|     查询路径: 路由到 Ring 中下一个节点 / TiKV 直读           |
|     写入路径: 需满足安全条件才可 bypass (见 S7.3)           |
|                                                            |
|  3. 避免长时间阻塞:                                         |
|     +------------------------------+                       |
|     | 无 Blacklist: 每次请求等 30s  |                       |
|     |              超时才发现故障   |                       |
|     +------------------------------+                       |
|     | 有 Blacklist: 立即跳过        | <- 关键优化           |
|     |              零等待时间       |                       |
|     +------------------------------+                       |
|                                                            |
|  4. Node B 恢复后:                                          |
|     SWIM ALIVE gossip -> 从 Blacklist 移除 -> AddNode(B)   |
|                                                            |
|  * 安全分级 (Blacklist 不是万能的):                          |
|     查询路径: 跳过安全 (Head Index 全量副本)                |
|     写入路径: 需 WAL+Intent+LockTTL 前置 (见 PR4/PR6)     |
|     Split/Merge: 跳过 partner，选下一个 neighbor            |
|     Reassign: 放弃，向量留旧 head，后台重试                |
|                                                            |
+-----------------------------------------------------------+
```

### 4.3 Ring Epoch 与路由一致性规则

Ring 在 gossip 传播的 ~11s 内，不同 node 看到的 ring 版本不同。通过 Epoch 机制保证最终正确性：

```
+-------------------------------------------------------------------+
|  核心规则: 所有节点间 RPC 均携带 Ring Epoch                         |
|                                                                    |
|  发送方: 请求中附带 sender_epoch                                    |
|  接收方:                                                           |
|    if sender_epoch < my_epoch:                                     |
|      -> 拒绝请求                                                   |
|      -> 返回 EPOCH_MISMATCH + 最新成员列表                         |
|      -> 发送方收到后刷新本地 Ring，重试                             |
|                                                                    |
|    if sender_epoch > my_epoch:                                     |
|      -> 接受请求 (发送方有更新信息)                                 |
|      -> 从发送方或 PD 拉取最新成员列表                              |
|                                                                    |
|    if sender_epoch == my_epoch:                                    |
|      -> 正常处理                                                   |
|                                                                    |
|  Ring Epoch 变更时机:                                               |
|    每次 AddNode/RemoveNode -> PD 发布 epoch+1                      |
|    epoch 通过 gossip piggyback 传播                                 |
|                                                                    |
|  * Ring 不一致期间数据正确性保证:                                   |
|    TiKV 是共享存储，posting 按 headID key 存储                      |
|    无论 Append 从旧 owner 还是新 owner 发起 -> 写入同一个 TiKV key  |
|    TiKV Raft 保证串行化 -> 数据永远正确                             |
|    唯一代价: 旧 owner 多做了不必要的工作                             |
+-------------------------------------------------------------------+
```

### 4.4 按需连接池 (ConnectionPool LRU)

替代 Full-Mesh，解决 2000 节点 4M 连接问题：

```cpp
class ConnectionPool {
    static constexpr int MAX_ACTIVE_CONNECTIONS = 200;  // 每 node 最大活跃连接

    struct PeerConn {
        Socket::ConnectionID connID;
        std::chrono::steady_clock::time_point lastUsed;
    };

    std::unordered_map<int, PeerConn> m_activeConns;  // LRU map: nodeIndex -> conn
    std::mutex m_connMutex;

    ConnectionID GetOrConnect(int nodeIndex) {
        std::lock_guard<std::mutex> lock(m_connMutex);
        auto it = m_activeConns.find(nodeIndex);
        if (it != m_activeConns.end()) {
            it->second.lastUsed = now();
            return it->second.connID;
        }
        if (m_activeConns.size() >= MAX_ACTIVE_CONNECTIONS)
            EvictLRU();  // 淘汰最久未使用的连接
        auto connID = m_client->ConnectToServer(
            m_nodeAddrs[nodeIndex].first, m_nodeAddrs[nodeIndex].second);
        m_activeConns[nodeIndex] = {connID, now()};
        return connID;
    }
};
```

**连接数估算**：
- 每 node steady state 活跃连接 ~ 50-200（取决于 headID 分布热度）
- 全集群 TCP 连接 ~ 2000 x 100 = 200K（vs full-mesh 的 4M）
- LRU 淘汰: 60s idle timeout -> 冷节点连接自动释放

### 4.5 VID 分段预分配

VID 唯一要求：全局唯一。不需要有序，不需要连续，只要不重叠。

```
VID 空间: [0, 2^31)  ~ 2.1 billion

静态分段 (启动时本地计算, 零网络):
  BLOCK_SIZE = 2^31 / MAX_NODES        // MAX_NODES = 4096 (预留扩容)
            = 524,288

  Node 0:     [0,        524,288)
  Node 1:     [524,288,  1,048,576)
  ...
  Node 1999:  [1,047,527,424,  1,048,051,712)

运行时:
  VID = m_rangeStart + m_localCounter++    // 本地 atomic, 零网络

VID 回收 (GC):
  Delete 操作标记 VID 为 deleted (versionMap)
  GC 将 deleted VID 放回空闲列表, 新 insert 优先复用

Block 用完 (安全兜底, 极低概率):
  从 TiKV 原子 CAS 抢下一个空闲 block
  key = "__vid_overflow__"
  走 TiKV Raft leader (不经过 PD)
```

```cpp
class DistributedVIDAllocator {
    SizeType m_rangeStart, m_rangeEnd;
    std::atomic<SizeType> m_next;
    static constexpr int MAX_NODES = 4096;
    static constexpr SizeType BLOCK_SIZE = (1u << 31) / MAX_NODES;  // 524,288

    // Phase 1: 计算 range 边界 (纯本地, 零网络)
    void InitBlock(int nodeIndex) {
        m_rangeStart = static_cast<SizeType>(nodeIndex) * BLOCK_SIZE;
        m_rangeEnd = m_rangeStart + BLOCK_SIZE;
        m_next.store(m_rangeStart);
    }

    // Phase 2: 扫 TiKV 恢复真实计数器 (防 crash 后 VID 冲突)
    void RecoverVIDCounter() {
        SizeType maxUsed = m_tikvClient->RangeScanMaxKey(
            "__vid__", m_rangeStart, m_rangeEnd);
        if (maxUsed >= m_rangeStart) m_next.store(maxUsed + 1);
    }

    // 热路径: 本地 atomic, 零网络
    SizeType AllocateVID() {
        if (!m_freeList.empty()) return m_freeList.pop();  // 优先回收 VID
        SizeType vid = m_next.fetch_add(1);
        if (vid < m_rangeEnd) [[likely]] return vid;
        return AllocateOverflowVID();  // 冷路径: TiKV CAS 抢新 block
    }

    // GC 回收: Delete 后把 VID 放回空闲列表
    void RecycleVID(SizeType vid) { m_freeList.push(vid); }
};
```

---

## 5. Client 层设计与 API 契约

### 5.1 三种扩展模式对比

```
模式 1: 1 Client -> N Compute Nodes（当前 benchmark）
  +  Posting I/O 分摊到 N 个节点 -> 阶段 C 吞吐量线性增长
  x  Head Search 只在 1 个节点 -> 阶段 B 是瓶颈
  x  Client 宕机 -> 全部不可用

模式 2: M Clients -> 各自独立 (不分布)
  +  阶段 A/B 随 Client 数量线性扩展
  x  每节点只查自己的 Posting -> 退化为 M 个独立单机实例

模式 3: M Client SDK -> N Compute Nodes（生产推荐）
  +  M 个 Client SDK 各自选不同 Compute Node 发送查询（Client-side LB）
  +  N 个 Compute Node 并行处理不同查询（跨查询并行）
  +  每个查询在 1 个 Compute Node 内完成（不做单查询跨节点并行，避免网络开销）
  +  任何 Compute Node 故障 -> Client SDK 自动切到其他节点
  +  无 Aggregator 层，架构更简单
```

### 5.2 吞吐量公式

```
  单查询延迟分解 (单 Compute Node 内完成):
    Latency = T_head + T_routing + T_posting + T_distance
    T_head    ~ 0.5-2ms (BKT 图搜索, CPU, 本地内存)
    T_routing ~ 0.1ms (hash ring lookup + RPC to owner)
    T_posting ~ 1-5ms (Owner 从 TiKV RawBatchGet)
    T_distance~ 0.1ms (距离计算 + Top-K 选择)

  单节点 QPS ~ NumThreads / Latency ~ 32 / 3ms ~ 10,000 QPS (单 Compute)

  三种模式吞吐量对比 (N=2000 Compute Nodes):

  Client SDK 内置 LB, 直连 Compute Node。每个查询由 1 个 Node 完成。
  QPS 靠跨查询并行增长: 不同查询分到不同 Node。

  +---------------+-----------------+-----------------+-------------------+
  |               | 模式 1          | 模式 2          | 模式 3 (推荐)     |
  |               | 1 Client, N Node| M Client, 独立  | M SDK, N Node     |
  +---------------+-----------------+-----------------+-------------------+
  | 单查询延迟    | ~3ms            | ~3ms            | ~3ms              |
  | 总 QPS        | ~10K (1 节点)   | M x 10K         | N x 10K = 20M*   |
  +---------------+-----------------+-----------------+-------------------+

  * 每个查询只用 1 个 Compute Node，但 N 个 Node 并行处理不同查询
  * 总吞吐 = N x 单节点 QPS = 2000 x 10K = 20M QPS (理论上限)
  * 实际瓶颈通常在 TiKV I/O
```

### 5.3 三态写入语义 (API 契约)

```
+-------------------------------------------------------------------+
|                    写入语义契约 (三态)                               |
|                                                                    |
|  Client                  Compute Node            TiKV              |
|    |                       |                       |               |
|    |---Insert(batch)------>|                       |               |
|    |                       |--local append-------->|               |
|    |                       |--remote append via--->| (Owner)       |
|    |                       |<------- ALL ACK ------|               |
|    |<---- SUCCESS ---------| data 100% in TiKV    |               |
|    |                       |                       |               |
|    |      === OR ===       |                       |               |
|    |                       |--remote append------->|               |
|    |                       |         x timeout/fail|               |
|    |<---- FAIL ------------|                       |               |
|    |                       |                       |               |
|    |      === OR ===       |                       |               |
|    |                       |--local append-------->|               |
|    |                       #### CRASH ####         |               |
|    |<---- TIMEOUT ---------x  (TCP RST/超时)       |               |
|                                                                    |
|  三态语义:                                                          |
|    SUCCESS -> 数据 100% 在 TiKV (Raft 3 副本)                      |
|    FAIL    -> Compute Node 明确告知失败, 可能部分写入               |
|    TIMEOUT -> Compute Node 可能 crash, 数据可能部分写入             |
|    --> 后两者对 Client 处理方式一样: retry                          |
|    --> retry 安全性由幂等保证 (VID+Version 去重)                    |
+-------------------------------------------------------------------+
```

### 5.4 Client SDK 设计

```
+-----------------------------------------------------------------+
|                     Client SDK 架构                                |
+-----------------------------------------------------------------+
|  Application                                                      |
|      |                                                            |
|      v                                                            |
|  +-------------------------------------------------------------+ |
|  |  SPANNClient                                                 | |
|  |                                                              | |
|  |  +-----------------+  +------------------------------+       | |
|  |  | ConnectionPool  |  | RetryPolicy                  |       | |
|  |  | * ComputeNode   |  | * MaxRetries: 3              |       | |
|  |  |   endpoints[]   |  | * Backoff: exp(100ms,2s)     |       | |
|  |  | * From PD       |  | * Query -> 总是安全重试       |       | |
|  |  | * HealthCheck   |  | * Write -> VID+Ver 去重      |       | |
|  |  | * RoundRobin LB |  | * 400 Bad Request -> 不重试   |       | |
|  |  +-----------------+  +------------------------------+       | |
|  |  +-----------------+  +------------------------------+       | |
|  |  | Timeout Config  |  | CircuitBreaker               |       | |
|  |  | * Connect: 1s   |  | * Per-Node 独立              |       | |
|  |  | * Query: 5s     |  | * 5 failures in 30s -> OPEN  |       | |
|  |  | * Write: 10s    |  | * OPEN 30s -> HALF_OPEN      |       | |
|  |  | * Overall: 30s  |  | * 1 success -> CLOSED        |       | |
|  |  +-----------------+  +------------------------------+       | |
|  +-------------------------------------------------------------+ |
|                                                                   |
|  查询接口:                                                         |
|    result = client.Search(query_vector, top_k=10)                 |
|    result.degraded   // true if partial nodes responded           |
|                                                                   |
|  写入接口:                                                         |
|    ack = client.Insert(vid, vector, metadata)                     |
|    ack.persistent   // true = data in TiKV Raft                  |
|                                                                   |
|  批量接口 (原子语义: 全成功或全失败):                                |
|    results = client.BatchSearch(query_vectors[], top_k=10)        |
|    acks = client.BatchInsert(vids[], vectors[], metas[])          |
+-----------------------------------------------------------------+
```

### 5.5 Client 容错路径

```
  Client SDK (内嵌在业务服务中)              Compute Node Pool
    |                                           |
    |  (1) 从 PD 获取健康节点列表               |
    |  (2) RoundRobin 选 Node A                |
    |  (3) 发送查询/写入请求                     |
    |------------------------------------------>| Node A
    |                                           |
    |                 +----------- 正常路径 ----+
    |                 |                         |
    |                 |  (4) Node A 处理成功     |
    |  (5) 返回结果   |                         |
    |<------------------------------------------|
    |                                           |
    |                 +----------- 故障路径 ----+
    |  (6) Node A 超时/失败                     |
    |      CircuitBreaker 记录失败              |
    |                                           |
    |  (7) 自动重试: 选 Node B                  |
    |------------------------------------------>| Node B
    |                                           |
    |  (8) Node B 成功返回                      |
    |<------------------------------------------|

  故障处理表:
  +----------------------+-----------------+-------------------------------+
  | 故障点               | 检测方式         | Client SDK 行为                |
  +----------------------+-----------------+-------------------------------+
  | PD 不可达            | connect timeout | 使用缓存的节点列表              |
  | Compute Node 超时    | read timeout    | CircuitBreaker 标记; 重试到    |
  |                      |                 | 其他 Node (max 3 次)           |
  | Compute Node 返回错误| gRPC error      | CircuitBreaker 标记; 重试      |
  | 响应格式错误          | decode error    | 丢弃，重试到其他 Node          |
  | 全部 Node 不可用      | all retries fail| 返回错误，业务层决定策略       |
  +----------------------+-----------------+-------------------------------+
```

### 5.6 写入路径的 Client 视角

```
  Client SDK                    Any Compute Node
    |                               |
    |  Insert(vid, vec, meta)       |
    |------------------------------>|
    |                               |
    |  写入路径: Client SDK -> 任意 Compute Node
    |  -> PostingRouter 路由到 Owner
    |                               |
    |  ACK: {persistent: true}      |
    |<------------------------------|
    |                               |
    |  如果超时/失败:                 |
    |  -> 安全重试 (VID+Version 去重) |
    |  -> 可以发到任意 Compute Node   |
    |    (PostingRouter 会路由到      |
    |     正确的 Owner)              |
```

---

## 6. 查询流容错

### 6.1 查询流完整路径图

**单查询不做跨节点并行**——网络开销太大。Client SDK 直连 1 个 Compute Node，
该 Node 独立完成全部工作：Head Search → 直接从 TiKV 读所有 posting → 距离计算 → Top-K。
QPS 靠**跨查询并行**增长：N 个 Compute Node 同时处理不同 Client 的查询。

```
  Client SDK              Compute Node A                       TiKV Cluster
    |                          |                                    |
    |  (1) Search Request      |                                    |
    |  (SDK round-robin 选 A)  |                                    |
    |------------------------->|                                    |
    |                          |                                    |
    |                          |  (2) Head Index Search (本地内存)   |
    |                          |  -> K headIDs                      |
    |                          |                                    |
    |                          |  (3) RawBatchGet                   |
    |                          |  所有 K 个 headID 的 posting       |
    |                          |----------------------------------->|
    |                          |<-----------------------------------|
    |                          |                                    |
    |                          |  (4) Distance + Top-K Selection    |
    |                          |                                    |
    |  (5) Results             |                                    |
    |<-------------------------|                                    |
```

**关键点**：
- 查询路径**零跨节点通信**，只有 Compute Node <-> TiKV 的存储 I/O
- Owner / PostingRouter 在查询路径中**不参与**，仅用于写入路径
- QPS = N × 单节点 QPS（2000 × 10K = 20M QPS 理论上限）

### 6.2 查询流每步故障处理表

| 步骤 | 断开位置 | 故障模式 | 检测方式 | 处理策略 | 用户感知 |
|------|----------|----------|----------|----------|----------|
| **(1)** Client->Compute | Node 宕机 / 网络不通 | connect timeout (1s) | Client SDK | CircuitBreaker 标记; 重试到其他 Node (max 3) | 延迟 ~1s |
| **(2)** Head Index Search | 内存损坏 / OOM | SIGSEGV | 进程崩溃 | Client SDK 超时后重试到其他节点 | 延迟 ~1s |
| **(3)** Compute->TiKV Read | Leader 迁移 / Store 宕机 | gRPC UNAVAILABLE | Region Error | **分级容错** (见 6.4): 刷新 Region Cache → 重试 Leader → Follower Read | Leader迁移 ~400ms |
| **(4)** Distance + Top-K | CPU 异常 | SIGSEGV | 同 (2) | 同 (2) | 同 (2) |
| **(5)** Compute->Client | 响应丢失 / 中途宕机 | TCP RST / 超时 | Client SDK | 重试到另一 Compute Node（查询天然幂等） | 延迟 ~1-2s |

查询容错极其简单：**任何一步失败 → Client SDK 重试到另一个 Compute Node**
（因为每个 Node 都有全量 Head Index + 能直接读 TiKV）。

### 6.3 Client SDK 选节点与故障切换

```
  Client SDK 维护 Compute Node 健康列表 (从 PD 定期拉取):

  healthy_nodes = AllNodes - CircuitBreaker.OpenNodes
  selected = RoundRobin(healthy_nodes)

  +------------------------------------------------------+
  |  Client SDK                                          |
  |                                                      |
  |  选 Node A -> 直连发送查询                             |
  |       |                                              |
  |       +-- 成功 -> 返回给业务层                         |
  |       |                                              |
  |       +-- 超时/失败 (1s)                              |
  |              |                                       |
  |              +-- CircuitBreaker 记录                  |
  |              |   (连续 5 次失败 in 30s -> OPEN)        |
  |              |                                       |
  |              +-- 重试: 选 Node B -> 直连               |
  |                    |                                 |
  |                    +-- 成功 -> 返回                    |
  |                    +-- 失败 -> 选 Node C (max 3 次)   |
  +------------------------------------------------------+
```

### 6.4 TiKV 读取分级容错

```
  Compute Node                                TiKV Cluster
      |                                           |
      |  (4-a): RawBatchGet (Leader, timeout=200ms)|
      |------------------------------------------>|
      |                                           |
      |  +- Success? ----------------------------|
      |  |  YES -> Return data                    |
      |  |  NO  |                                 |
      |  |                                        |
      |  |  (4-b): Invalidate region cache        |
      |  |       Retry Leader (timeout=400ms)      |
      |  |---------------------------------------->|
      |  |                                        |
      |  |  +- Success? -------------------------|
      |  |  |  YES -> Return data                |
      |  |  |  NO  |                             |
      |  |  |                                    |
      |  |  |  (4-c): Follower Read (stale ok)   |
      |  |  |       (timeout=200ms)               |
      |  |  |------------------------------------->|
      |  |  |                                    |
      |  |  |  +- Success? ---------------------|
      |  |  |  |  YES -> Return data (stale)     |
      |  |  |  |  NO  |                         |
      |  |  |  |                                |
      |  |  |  |  (4-d): Return partial result   |
      |  |  |  |  (degraded)                     |
      +--+--+--+--------------------------------+
```

### 6.5 TiKV 读取部分失败

```
  查询路径中唯一可能的"部分失败"是 TiKV RawBatchGet:
  K 个 headID 的 posting 分布在多个 TiKV Region,
  部分 Region 可能暂时不可用。

  处理策略:
    - 成功读到的 posting -> 正常计算距离
    - 读取失败的 headID -> 跳过
    - 只要有 >=1 个 posting 成功 -> 返回结果 (recall 略降)
    - 全部失败 -> 返回错误, Client SDK 重试到其他 Compute Node

  对 Client 透明: response 中标记 degraded=true 即可
```

---

## 7. 写入流容错

### 7.1 写入流完整路径图

```
  Client       Compute Node A     PostingRouter     Owner Node B       TiKV
    |                  |                |                |                |
    |  (1) Insert(VID, |                |                |                |
    |     vector)      |                |                |                |
    |----------------->|                |                |                |
    |                  |                |                |                |
    |                  |  (2) Write WAL |                |                |
    |                  |  to TiKV       |                |                |
    |                  |----------------+----------------+--------------->|
    |                  |<---------------+----------------+----------------|
    |                  |  WAL persisted |                |                |
    |                  |                |                |                |
    |                  |  (3) Head Index Search (local)                   |
    |                  |  -> RNG Selection -> headID = H                  |
    |                  |                |                |                |
    |                  |  (4) GetOwner(H)               |                |
    |                  |--------------->|                |                |
    |                  |  owner = Node B|                |                |
    |                  |<---------------|                |                |
    |                  |                |                |                |
    |                  |  (5) Node B in Blacklist?       |                |
    |                  |  NO -> RemoteAppend ----------->|                |
    |                  |                |                |                |
    |                  |                |      (6) Lock  |                |
    |                  |                |      headID    |                |
    |                  |                |                |                |
    |                  |                |      (7) RawPut|                |
    |                  |                |      (append)  |                |
    |                  |                |                |--------------->|
    |                  |                |                |  Raft Commit   |
    |                  |                |                |<---------------|
    |                  |                |                |                |
    |                  |                |      (8) VersionMap.IncVer      |
    |                  |                |                |--------------->|
    |                  |                |                |<---------------|
    |                  |                |                |                |
    |                  |                |      (9) Release lock           |
    |                  |                |                |                |
    |                  |  (10) AppendResponse(OK) <------|                |
    |                  |                |                |                |
    |                  |  (11) Clear WAL entry           |                |
    |                  |----------------+----------------+--------------->|
    |                  |                |                |                |
    |  (12) Insert OK  |                |                |                |
    |<-----------------|                |                |                |
```

### 7.2 写入流每步故障处理表

| 步骤 | 断开位置 | 故障模式 | 处理策略 | 幂等保证 |
|------|----------|----------|----------|----------|
| **(1)** Client->Node A | 网络断开 | Client 重试到同一或其他 Compute Node | VID+Version 去重 |
| **(2)** WAL Write->TiKV | TiKV 写超时 | 10 次重试 + Region Cache 失效; WAL 是 Append 幂等 | WAL key = `wal:{nodeId}:{seqNo}` 唯一 |
| **(3)** Head Index Search | Node A 宕机 | WAL 已持久化 -> 恢复后重放; 或 Client 重试 | VID+Version 去重 |
| **(4)** GetOwner(H) | Hash Ring 不一致 | Epoch 检查; 过期则从 PD 拉最新 | 确定性哈希 |
| **(5)** Node B 在 Blacklist | Node B 已知故障 | **跳过 Node B**, 直接写 TiKV (bypass) + 标记需修复 | 直接写 TiKV 幂等 |
| **(6)** RemoteAppend->Node B | Node B 宕机/超时 | 2 次重试; 失败则加入 Blacklist + pending queue | VID 去重 |
| **(7)** Node B->TiKV Write | TiKV 写超时 | 10 次重试; 成功 = Raft majority commit = 100% 持久 | Synchronous Write |
| **(8)** VersionMap Inc | TiKV 写失败 | 重试 3 次; 失败则 Append 整体返回失败 | CAS 原子性 |
| **(9)** Release Lock | Node B 宕机 | **Lock TTL = 30s**, 超时自动释放 | TTL 机制 |
| **(10)** Response->Node A | 网络断开(响应丢失) | Node A 超时 -> pending queue 重试; 实际已写成功 | 幂等重试 |
| **(11)** Clear WAL | TiKV 删除失败 | 非关键路径，后台重试; 重放时发现已完成则跳过 | 重放幂等 |
| **(12)** Response->Client | 网络断开 | Client 超时重试; 数据已持久化 | VID+Ver 去重 |

### 7.3 Compute Node Crash 时刻分析

Compute Node A (接收写入的入口节点) 在写入过程中任意时刻 crash：

```
  Crash 时刻         TiKV 中已写入       孤儿数据         恢复方式
 ------------------------------------------------------------------
  C1  (1)->(2)       无                 无               Client 超时 -> retry
  C2  (2)->(3)       WAL                无               节点恢复后重放 WAL
  C3  (3)->(4)       WAL                无               WAL 重放
  C4  (4a) 本地写中   WAL + 部分 local   local 孤儿       Client retry -> 新 VID
  C5  (4a) 完成       WAL + 全部 local   local 孤儿       Client retry -> 新 VID
      (4b) 未发       remote 无                          旧 local 变孤儿(无害)
  C6  (4b) 发送中     local + Owner 部分 可能有孤儿        Client retry
  C7  (5) Owner 写中  local + TiKV 写中  部分孤儿          Client retry
  C8  (6) Response    全部已写入!        无!               Client 不知道成功
      在路上                            数据完整           -> retry 产生重复
  C9  (7)->(8)       全部已写入!        无!               同 C8

  所有 Crash 点, Client 统一行为:
    TCP RST / read timeout -> 没收到 SUCCESS -> retry 到其他 Compute

  关键观察:
  +------------------------------------------------------------+
  |  C1-C7: 数据不完整 -> retry 写入新 VID -> 旧的变孤儿 (无害) |
  |  C8-C9: 数据已完整 -> retry 写入新 VID -> 旧的也完整 (重复) |
  |                                                              |
  |  两种情况 retry 都安全:                                      |
  |    * 孤儿: 搜索可命中但 external ID 不存在 -> 上层过滤       |
  |    * 重复: 同一向量两组 VID -> 搜索结果去重                   |
  |    * 后台 GC 可选清理 (crash 稀有, 量极小)                   |
  +------------------------------------------------------------+
```

### 7.4 Owner Node Crash 时刻分析

Owner (Node B) 在步骤 (5)-(10) 之间 crash：

```
  Crash 时刻         Owner 侧 TiKV      Node A 感知       后续
 ------------------------------------------------------------------
  O1  (5) 到达前     无                  超时 -> FAIL      Client retry
  O2  (5) 解析中     无                  超时 -> FAIL      Client retry
  O3  (7) 写入中     部分 posting        超时 -> FAIL      Client retry
                     已写入 TiKV                          已写入的: 孤儿
  O4  (7) 完成       全部 posting        超时 -> FAIL      Client retry
      (10) 未发       已写入 TiKV                          数据实际完整
  O5  (10) 发送中    全部已写入          可能收到/可能断连  收到: SUCCESS

  Owner crash 对 Node A 的统一表现:
    步骤(5) 发出后超时 (30s) 没收到 (10) -> Node A 认为失败

  关键区别:
    Ingress crash -> Client 收到 TIMEOUT (TCP 断连)
    Owner crash   -> Client 收到明确的 FAIL (Node A 还活着, 主动返回)
    两者对 Client 处理方式一样: retry
```

### 7.5 故障点 x 容错矩阵

```
  故障点          位置             后果                   容错机制
 -----------------------------------------------------------------------
  F1  Client->   (1) 网络断      请求未到达              Client timeout -> retry
      Node A                                            (LB 切换到其他 node)

  F2  VID 分配   (2) 本地 block   本地 block 余量~52万    极低概率用完
                 用完            PD 不可用也不影响

  F3  RNG        (3) head index  选到稍旧的 head         最终一致, 不影响正确性
      Selection  过时

  F4  Local      (4a) TiKV 写失败 posting 未持久化       -> return FAIL, client retry
      Append

  F5  Remote     (4b) TCP 断连   append 未送达 Owner     -> 超时 -> return FAIL
      发送失败   或 Owner 不可达                          client retry

  F6  Owner      (5) TiKV 写失败 Owner 端 Append 失败    Owner 返回 FAIL
      写入失败                                           -> Node A return FAIL

  F7  Owner      (5) 进程 crash  部分 item 写入 TiKV     -> Node A 超时 -> FAIL
      Crash      部分完成                                client retry

  F8  Response   (6) TCP 断连    Node A 不知道结果        -> 超时 -> return FAIL
      丢失       (Owner 成功了)                           client retry (幂等 skip)

  F9  Node A     任意时刻        部分 posting 可能已写     Client TCP 超时 -> retry
      Crash                     VID 计数器丢失            VID 恢复: 重启时扫 TiKV

  F10 Ring       (4b) owner 变了 发给旧 owner             旧 owner 仍可写 TiKV
      变更中                     (共享存储)               -> 数据正确

  F11 Partial    (4a) OK         local 成了, remote 失败  -> return FAIL
      失败       (4b) FAIL                               client retry 整个 batch
```

### 7.6 Owner 故障时的 Bypass 路径

```
  Node A 发现 Owner (Node B) 不可达

  正常路径:                         Bypass 路径 (Node B in Blacklist):

  Node A --> Node B --> TiKV        Node A ----------------------> TiKV
              |  |                         |
              |  +- lock headID            +- 直接 RawPut
              |                               (无 per-headID 锁)
              +--- RawPut                     
                                          标记 headID 需要后续 Merge 修复
                                          (写入 repair_needed:{headID} key)
```

**Bypass 的前提条件** (必须全部满足):
- WAL 已实现 (PR1) -> 保证持久化
- Intent 状态机已实现 (PR2) -> 保证 Split/Merge 不冲突
- Lock TTL 已实现 (PR2) -> 保证不死锁
- Blacklist 已实现 (PR4) -> 保证只在确认 DEAD 时才 bypass

**Bypass 的权衡**：
- OK 保证数据不丢失（数据已写入 TiKV）
- !! 短暂窗口内同一 headID 可能有并发写入（无锁保护）
- OK 后续 Merge 操作会自动修复 posting 一致性

### 7.7 Batch 内 Partial Failure 处理

```
一个 batch (10 个向量) 的 RNGSelection 可能散到 3 个 owner:

  local(K):  headIDs [99, 200, 301]      -> (4a) 全部 OK
  Node J:    headIDs [42, 55]             -> (4b) OK
  Node M:    headIDs [1001, 1002, 1003]   -> (4b) FAIL (Node M 超时)

处理策略: 全部失败 (方案 A)

  理由:
  1. 简单 -- client 不需要追踪哪些成功了哪些没有
  2. 幂等 -- 全部 retry 不会重复写入
  3. 故障是稀有路径 -- 不值得为之复杂化 API
  4. batch 通常只涉及 2-3 个 owner -- 单个失败概率低
```

### 7.8 孤儿 Posting 分析

```
  场景:
    t=0  Client -> Node A: Insert(vec_A, vec_B)
    t=1  Node A: AllocateVID -> VID=100(vec_A), VID=101(vec_B)
    t=2  RNGSelection -> headID=42(owner=J), headID=99(owner=K,local)
    t=3  (4a): Append(99, VID=100) -> TiKV OK (成功写入)
    t=4  #### Node A CRASH ####  ((4b) 还没发出去)

  结果:
    - TiKV 中: headID=99 的 posting list 包含 VID=100 <- 孤儿!
    - headID=42 没有收到 VID=101 <- 丢失但无害
    - Client: TCP 超时, retry -> 新 VID=500(vec_A), VID=501(vec_B)

  孤儿的影响:
    - 搜索可命中孤儿 VID -> 返回正确的向量数据 (数据是对的)
    - 但它在 client 的 external ID mapping 中不存在
      -> client 认为 "unknown" -> 上层过滤即可
    - 浪费少量 TiKV 存储空间

  孤儿清理 (后台 GC, 非关键路径):
    定期扫描: posting 中的 VID 是否仍 "活跃"
    "活跃" = versionMap 中存在且未被 delete
    不活跃 -> 从 posting 中移除
    频率: 每小时或每天, crash 是稀有事件, 孤儿量极小

  +--------------------------------------------------------------+
  |  Crash != 灾难:                                               |
  |    Client: TCP 超时 -> retry -> 最终成功                       |
  |    节点恢复: 扫 TiKV 恢复 VID -> 不会冲突                      |
  |    孤儿: 无害, 后台 GC 可选清理                                |
  |    代价: 1 次 retry (~30s) + 几百条孤儿 posting               |
  +--------------------------------------------------------------+
```

---

## 8. 维护流容错：Split/Merge/HeadSync

### 8.1 Split 完整流程与 Intent 状态机

```
  Owner Node                     TiKV                         All Nodes
      |                           |                              |
      |  (1) Detect posting > limit                              |
      |                           |                              |
      |  (2) Write Split Intent   |                              |
      |  status=PREPARED          |                              |
      |-------------------------->|                              |
      |                           |                              |
      |  (3) Read full posting    |                              |
      |-------------------------->|                              |
      |<--------------------------|                              |
      |                           |                              |
      |  (4) K-means clustering   |                              |
      |  -> newHead1, newHead2    |                              |
      |                           |                              |
      |  (5) Write new postings   |                              |
      |  Update Intent=EXECUTING  |                              |
      |-------------------------->|                              |
      |                           |                              |
      |  (6) Update Head Index    |                              |
      |  (add new, delete old)    |                              |
      |  Update Intent=HEAD_UPDATED                              |
      |-------------------------->|                              |
      |                           |                              |
      |  (7) Broadcast HeadSync   |                              |
      |-------------------------->|  (persist HeadSync Log)      |
      |------------------------------------------------------------>|
      |                           |  (push to all nodes)         |
      |                           |                              |
      |  (8) Delete old posting   |                              |
      |  Update Intent=COMMITTED  |                              |
      |-------------------------->|                              |
      |                           |                              |
      |  (9) Delete Intent        |                              |
      |-------------------------->|                              |
```

#### Intent 状态机

```
                    +----------+
                    | PREPARED | <- 写入 Intent，操作未开始
                    +----+-----+
                         | 开始执行
                         v
                    +----------+
              +-----|EXECUTING | <- 新 posting 已写入
              |     +----+-----+
              |          | Head Index 更新完成
    Crash恢复:|          v
    可从此继续|    +------------+
              |    |HEAD_UPDATED| <- Head Index 已更新
              |    +----+-------+
              |         | 旧 posting 清理 + HeadSync
              |         v
              |    +----------+
              +--->|COMMITTED | <- 操作完成
                   +----+-----+
                        | 清理 Intent
                        v
                   +----------+
                   | (deleted)|
                   +----------+

    回滚路径 (任何阶段):
    +------------+
    |ROLLED_BACK | -> 清理已写入的新 posting -> 删除 Intent
    +------------+
```

### 8.2 Split/Merge Crash 时刻分析

```
  === Split Crash ===

  Crash 时刻         已完成的 TiKV 写       后果                恢复
 -----------------------------------------------------------------------
  S1  (2)后(4)前     无 TiKV 写              posting 未变        无影响
      (读完还没写)                           splitList 丢失     -> 下次 Append 重新触发

  S2  (5)后(6)前     headID 写了 cluster_A   cluster_B 的向量    向量暂时"消失"
      (原 posting     但 cluster_B 没写       不在任何 posting    -> versionMap 仍有记录
       已覆写)                                                   -> RefineIndex 修复

  S3  (6)后(7)前     两个 posting 都写了      数据完整!           head index 已更新
      (TiKV 写完,     headIndex 本地更新      但其他节点不知道    -> HeadSync 没广播
       HeadSync 没发)                         新 head            -> gossip 最终传播

  S4  步骤 6a 递归    cluster_B 合并到已有    可能触发递归 split   同 S2 分析
      split 中 crash  head, 但递归写到一半

  === Merge Crash ===

  Crash 时刻         已完成的 TiKV 写       后果                恢复
 -----------------------------------------------------------------------
  M1  Lock后Write前  无 TiKV 写              无影响             远程锁 30s 自动过期
                                             partner 暂时锁住   -> 不会死锁

  M2  Put完成        winner 有合并数据        两个 head 都存在    搜索可能返回重复
      Delete未完成   loser 没删              -> loser 数据重叠   -> 上层去重
                                                                -> GC stale 清理

  M3  headIndex      loser TiKV posting      posting 孤立        不被搜索命中
      Delete完成     没删                    -> 空间浪费         -> key GC 清理
      db->Delete没

  M4  HeadSync       本地操作全部完成         其他节点不知道       gossip 最终传播
      没广播                                 loser 被删了

  M5  Unlock crash   远程锁没释放             lease 30s 过期      -> 不会死锁
```

### 8.3 Reassign 流程 (先写后 bump)

```
  Split/Merge 后, 部分向量可能不再"属于"最近的 head
  Reassign 把它们移到更合适的 head

  关键设计决策: 先写后 bump version

  错误顺序 (先 bump 后写):
    (1) version++  -> 旧 entry 变 stale
    (2) Append(newHead, VID)  -> 如果失败?
    -> VID 在搜索中"消失"了! (旧 entry stale, 新 entry 不存在)

  正确顺序 (先写后 bump):
    (1) Append(newHead, {VID, newVersion})  -> 写入新 posting
    (2) version++  -> 旧 entry 变 stale
    -> 失败时: version 没变, 旧 entry 仍有效, 新 entry 不存在 -> 无影响
    -> 成功时: 新 entry 在, 旧 entry 变 stale -> 正确
    -> 代价: 短暂窗口内新旧 entry 同时有效 -> 搜索返回重复 -> 去重即可

  Reassign 失败不需要 rollback:
    Owner 不可达 -> 向量留旧 head -> 搜索仍能找到
    后续 RefineIndex 会重新尝试
```

### 8.4 HeadSync 推送+拉取混合模式

**Source of truth: TiKV HeadSync Log**（gossip 仅为快速路径）

```
  Node A (Split 执行者)          TiKV HeadSync Log         Node B (接收者)
      |                                  |                      |
      |  Write HeadSync entry            |                      |
      |  key: headsync:{epoch}:{seq}     |                      |
      |--------------------------------->|                      |
      |                                  |                      |
      |  Push HeadSyncEntry to Node B    |                      |
      |----------------------------------+--------------------->|
      |                 (Gossip piggyback, best-effort)         |  Apply locally
      |                                  |                      |
      |                                  |   如果 Push 失败:    |
      |                                  |                      |
      |                                  |  Pull: 从 cursor 到  |
      |                                  |  latest              |
      |                                  |<---------------------|
      |                                  |--------------------->|
      |                                  |  (批量返回缺失)      |  Apply all
      |                                  |                      |
      |                                  |  Update cursor       |
      |                                  |<---------------------|

  传播速度:
    Push (gossip): O(log N) ~ 11s 传遍 2000 节点
    Pull (catch-up): 节点启动时或检测到 gap 时 -> TiKV range scan

  日志保留: 最近 10 万条 (或 7 天), 更早的靠 Head Index Checkpoint
```

### 8.5 Split/Merge 与 Client 的关系

```
  +--------------------------------------------------------------------+
  |                                                                    |
  |  Client 完全不感知 Split/Merge!                                    |
  |                                                                    |
  |  Client 的 Insert 请求:                                            |
  |    (1) -> (12) 全部完成 -> return SUCCESS                          |
  |    这里的 "完成" = Append 写入 TiKV + 所有 Owner ACK              |
  |    Split/Merge 是 Append 之后的异步操作, 不在 Client 等待路径上    |
  |                                                                    |
  |  唯一例外: posting 爆满 (overflow), 同步 Split 阻塞 Append         |
  |    -> Append 等 Split 完成 -> 重新 Append -> 然后才返回 Client     |
  |    -> 对 Client 表现为: 这一次 Insert 延迟较高                     |
  |    -> 但仍然是 SUCCESS/FAIL 语义, 不需要 client 特殊处理           |
  |                                                                    |
  |  Split/Merge 失败不影响 Client 已收到的 SUCCESS:                   |
  |    Client 收到 SUCCESS = Append 已经写入 TiKV                     |
  |    Split 失败 = posting 暂时过长 -> 搜索变慢, 但数据不丢           |
  |    Merge 失败 = posting 暂时过短 -> recall 略低, 但可用            |
  |    -> 都是性能问题, 不是正确性问题                                  |
  |                                                                    |
  +--------------------------------------------------------------------+
```

---

## 9. Job Routing 容错

### 9.1 Job 类型与安全等级

系统中的后台 Job 对 Blacklist 节点有不同的安全策略：

```
  +-------------------+-------+-------------------------------------------+
  | Job 类型          | 安全? | Blacklist 节点处理                         |
  +-------------------+-------+-------------------------------------------+
  | Query (search)    |  OK   | 跳过 Blacklist 节点, route 到其他健康节点  |
  |                   |       | -> degradation ladder 处理                 |
  +-------------------+-------+-------------------------------------------+
  | Append (write)    | 条件  | Owner 在 Blacklist -> Bypass 直写 TiKV     |
  |                   |       | 需 WAL + Intent + LockTTL 支撑             |
  +-------------------+-------+-------------------------------------------+
  | Split             |  OK   | Partner 在 Blacklist -> 跳过本次 Split     |
  |                   |       | posting 暂时过长但不丢数据                  |
  +-------------------+-------+-------------------------------------------+
  | Merge             |  OK   | Partner 在 Blacklist -> 跳过本次 Merge     |
  |                   |       | posting 暂时过短但 recall 可接受            |
  +-------------------+-------+-------------------------------------------+
  | Reassign          |  OK   | 目标 Owner 在 Blacklist -> 放弃本次        |
  |                   |       | 向量留旧 head -> 仍可搜索                  |
  +-------------------+-------+-------------------------------------------+
  | HeadSync          | N/A   | Push 失败 -> 目标节点后续 Pull from TiKV   |
  +-------------------+-------+-------------------------------------------+
```

### 9.2 Job Routing 详细流程

```
  任意 Compute Node            PostingRouter / HashRing
      |                               |
      |  JobRequest(headID, type)      |
      |------------------------------>|
      |                               |
      |        GetOwner(headID)       |
      |<------------------------------|
      |        owner = Node X         |
      |                               |
      |  Is Node X in Blacklist?      |
      |        +                      |
      |        |                      |
      |        +-- NO  -> 正常发送给 Node X
      |        |
      |        +-- YES -> 根据 Job 类型分级处理:
      |              |
      |              +-- Query   -> Route to Next(hashRing)
      |              +-- Append  -> Bypass: 直写 TiKV
      |              +-- Split   -> 跳过, 等 Owner 恢复
      |              +-- Merge   -> 跳过, 等 Owner 恢复
      |              +-- Reassign-> 放弃, 向量留原处
      |              +-- HeadSync-> Push 失败记录, 后续 Pull
```

### 9.3 跨 Job 的 Epoch 一致性

```
  所有 Job 的 RPC 请求都携带 epoch:

    +------------------------------------------+
    | RPC Header                               |
    |  ring_epoch: uint64                      |
    |  sender_node_id: uint32                  |
    |  job_type: enum                          |
    +------------------------------------------+

  接收方检查:
    if request.ring_epoch < local_ring_epoch:
      return ERR_STALE_EPOCH
      (发送方需要更新 ring topology)

    if request.ring_epoch > local_ring_epoch:
      pull latest ring from PD
      retry locally with new ring

    if request.ring_epoch == local_ring_epoch:
      process normally
```

---

## 10. 节点 Crash 故障处理总表

> 无 Aggregator 层，Client SDK 直连 Compute Node，故障表只涉及 Compute Node 和 TiKV。

### 10.1 Compute Node Crash

```
  +----+--------------------+-------------------------------------------+
  | #  | 故障场景           | 处理方式                                   |
  +----+--------------------+-------------------------------------------+
  | N1 | Compute 进程崩溃   | SWIM 检测 (1s suspect -> 3s dead)          |
  |    |                    | -> Ring 更新, 邻居接管 owned headIDs       |
  |    |                    | -> 因 TiKV 共享存储, 邻居可立即服务         |
  +----+--------------------+-------------------------------------------+
  | N2 | Compute 主机宕机   | 同 N1; 该 node 持有的锁 30s TTL 自动过期   |
  |    |                    | -> 不会死锁                                 |
  +----+--------------------+-------------------------------------------+
  | N3 | Compute 恢复启动   | (1) 从 PD 获取当前 ring                     |
  |    |                    | (2) 从 TiKV 拉 HeadSync Log 追赶            |
  |    |                    | (3) 重建 Head Index (增量或全量)             |
  |    |                    | (4) 扫 WAL 前缀, 重放未完成事务              |
  |    |                    | (5) 加入 SWIM gossip                         |
  |    |                    | (6) PD 更新 ring -> 接管 headIDs            |
  +----+--------------------+-------------------------------------------+
  | N4 | 多个 Compute 同时  | Ring 自动 rebalance (O(K/N) 变动)          |
  |    | 故障 (批量宕机)    | -> 查询: Client SDK 重试到存活节点           |
  |    |                    | -> 写入: Bypass 直写 TiKV                   |
  |    |                    | -> 前提: TiKV Raft 仍有 majority            |
  +----+--------------------+-------------------------------------------+
  | N5 | 网络分区           | SWIM suspect -> 如果 PD 能联通              |
  |    | (node 活着但不可达) | -> PD 仲裁: 可达侧继续服务                  |
  |    |                    | -> 不可达侧不能写入 TiKV (TiKV raft 保证)   |
  +----+--------------------+-------------------------------------------+
```

### 10.2 TiKV 节点 Crash

```
  +----+--------------------+-------------------------------------------+
  | #  | 故障场景           | 处理方式                                   |
  +----+--------------------+-------------------------------------------+
  | T1 | 单个 TiKV Store 宕 | Raft Group 3 副本: majority 2/3 仍可写    |
  |    |                    | -> Leader Transfer (~400ms)                |
  |    |                    | -> Compute 端 Region Cache 失效 -> 重定向  |
  |    |                    | -> PD 15min 后调度新 Replica 补全          |
  +----+--------------------+-------------------------------------------+
  | T2 | 2 个 TiKV Store 宕 | 受影响 Region: 只剩 1/3 -> 不可写         |
  |    | (同一 Region)      | -> 未受影响 Region 仍正常                   |
  |    |                    | -> 读: 可能降级 (stale read)               |
  |    |                    | -> 写: 排队等恢复 (WAL 保证不丢)           |
  +----+--------------------+-------------------------------------------+
  | T3 | PD Leader 宕       | PD 3 节点 Raft -> 1s 内选举新 Leader      |
  |    |                    | -> 不影响已有的 Region 读写                 |
  |    |                    | -> 短暂影响: 调度新 Replica, Ring 变更      |
  +----+--------------------+-------------------------------------------+
  | T4 | TiKV 全集群不可用  | Compute Node 降级为只读模式               |
  |    | (灾难场景)         | -> 用本地 Head Index 缓存做近似搜索         |
  |    |                    | -> 所有写入失败, 等集群恢复                 |
  +----+--------------------+-------------------------------------------+
```

---

## 11. 弹性伸缩设计

### 11.1 Compute Node 扩缩容

**核心优势：TiKV 共享存储**

传统分布式系统（如 FAISS Distributed）扩缩容要迁移数据。SPANN 的 posting 全在 TiKV，扩缩容只需更新 hash ring 的 ownership 映射。

```
  扩容 (Add Node):

  +-------+-------+-------+
  |  N1   |  N2   |  N3   |       (before: 3 nodes, ring 三等分)
  +---+---+---+---+---+---+
      |       |       |
      v       v       v
  +---+---+---+---+---+---+---+---+
  | N1 | N4 | N2 | N4 | N3 | N4 |  (after: N4 插入, consistent hashing)
  +---+---+---+---+---+---+---+---+

  影响范围: O(K/N) 个 headID 从 N1/N2/N3 迁移到 N4
  "迁移" 只是更改 ownership (TiKV 数据不动!)
  
  扩容步骤:
    (1) N4 加入 SWIM gossip 集群
    (2) N4 从 TiKV 拉取 Head Index (或增量 HeadSync)
    (3) PD 更新 hash ring -> 增加 epoch
    (4) 新 ring 广播到所有 node (gossip + HeadSync)
    (5) N4 开始接管分配给它的 headIDs
    (6) 切换完成, 旧节点自动减少负载
    
  整个过程: ~10s (Head Index 恢复时间)
  数据迁移: 0 bytes (TiKV 共享存储!)


  缩容 (Remove Node):

  (1) PD 标记 N4 为 draining
  (2) N4 的 headIDs 在 ring 上自动 fallback 到邻居
  (3) 邻居节点已有 Head Index 全量副本 -> 可立即服务
  (4) 等待 N4 上所有进行中请求完成 (grace period 30s)
  (5) PD 从 ring 删除 N4 -> bump epoch
  (6) N4 安全退出

  grace period 内: 新请求路由到邻居, 旧请求在 N4 完成
  不丢任何请求
```

### 11.2 自动扩缩容触发条件

```
  +-----------------+-------------------+----------------------------+
  | 指标             | 阈值             | 动作                        |
  +-----------------+-------------------+----------------------------+
  | Compute CPU     | > 80% avg (5min) | 触发扩容 (+10% nodes)       |
  | Compute CPU     | < 30% avg (15min)| 触发缩容 (-10% nodes)       |
  | Query Latency   | p99 > 50ms       | 触发扩容 (+20% nodes)       |
  | Query Latency   | p99 > 200ms      | 紧急扩容 (+50% nodes)       |
  | TiKV Read Lat   | p99 > 20ms       | 不扩 Compute, 扩 TiKV Store |
  | Posting Size    | avg > 2x target  | 触发 Split (不是扩容)       |
  | HeadID Imbalance| max/min > 2x     | 触发 Ring Rebalance         |
  +-----------------+-------------------+----------------------------+
```

### 11.3 TiKV 扩缩容

TiKV 的扩缩容由 PD 自动调度，对 Compute Node 完全透明:

```
  PD 调度                       TiKV Cluster
    |                               |
    |  (1) 新 TiKV Store 加入       |
    |  (2) PD 检测到容量增加        |
    |  (3) PD 调度 Region Replica   |
    |      到新 Store               |
    |------------------------------>|
    |                               |  Raft Learner 追赶
    |                               |  Raft Learner -> Voter
    |                               |  旧 Replica 可能移除
    |                               |
    |  Compute Node 无感知:         |
    |  Region Cache -> 自动发现新 Leader
```

---

## 12. 四条链路容错全景

### 12.1 链路分类

系统中的数据流可以归纳为 **4 条链路**，每条链路有独立的容错机制:

```
  +-----+------------------------------------------+------------------------+
  | 链路 | 描述                                      | 容错机制               |
  +-----+------------------------------------------+------------------------+
  | L1   | Client SDK <-> Compute Node             | Client-side LB +       |
  |      | (查询 + 写入接入)                         | Retry + CircuitBreaker |
  +-----+------------------------------------------+------------------------+
  | L2   | Compute Node <-> TiKV                   | Region Cache + Leader  |
  |      | (存储 I/O: posting 读写)                  | Transfer + Raft 3x    |
  +-----+------------------------------------------+------------------------+
  | L3   | Compute Node <-> Compute Node           | SWIM + Ring Epoch +    |
  |      | (节点间: Append/Split/Merge/HeadSync)     | Intent + LockTTL      |
  +-----+------------------------------------------+------------------------+
  | L4   | Client SDK -> Compute Node -> Owner     | WAL + VID Idempotent + |
  |      | (写入路径, 含 Owner 路由)                  | Bypass + Retry         |
  +-----+------------------------------------------+------------------------+
```

### 12.2 四条链路故障 x 处理矩阵

```
  +-------+-------------------+--------------------+--------------------+
  |       | 超时/网络断       | 进程崩溃           | 部分失败           |
  +-------+-------------------+--------------------+--------------------+
  | L1    | Client SDK retry  | CircuitBreaker ->  | N/A (单查询单节点)  |
  |       | to other Node     | 切到其他 Node      |                    |
  +-------+-------------------+--------------------+--------------------+
  | L2    | Region Cache      | Raft majority ->   | RawBatchGet 部分   |
  |       | invalidate +      | Leader transfer    | Region 失败 ->     |
  |       | retry new leader  | (~400ms)           | 跳过 + degraded    |
  +-------+-------------------+--------------------+--------------------+
  | L3    | Blacklist +       | Intent 状态机 ->   | Split/Merge 部分   |
  |       | Bypass / Skip     | 恢复或回滚         | 完成 -> 数据正确   |
  |       |                   | Lock TTL 防死锁    | 但需 GC 清理       |
  +-------+-------------------+--------------------+--------------------+
  | L4    | Client retry +    | WAL 恢复 +         | Batch 全失败策略   |
  |       | VID 幂等          | 节点重启重放       | -> Client retry    |
  +-------+-------------------+--------------------+--------------------+
```

---

## 13. 实现路线图：PR 计划

### 13.1 PR 总览

共 12 个 PR，分为 4 个阶段:

```
  阶段 1: 基础容错 (PR1-PR4)
    PR1: WAL + Crash Recovery
    PR2: Intent 状态机 + Lock TTL
    PR3: Ring Epoch Enforcement
    PR4: SWIM + Blacklist + Owner Detection

  阶段 2: 路径容错 (PR5-PR8)
    PR5: 查询容错 (Client SDK Retry + TiKV 分级容错)
    PR6: 写入 Bypass + Pending Queue
    PR7: HeadSync TiKV Log + Pull Recovery
    PR8: Client SDK + Three-State Write + Client-side LB

  阶段 3: 高可用增强 (PR9-PR10)
    PR9:  VID Segmented Pre-allocation
    PR10: ConnectionPool LRU + HealthCheck

  阶段 4: 弹性伸缩 (PR11-PR12)
    PR11: 自动扩缩容 (PD 触发)
    PR12: 监控 + 可观测性
```

### 13.2 依赖关系图

```
  PR1 (WAL)  -----------+--------> PR6 (Write Bypass)
                         |
  PR2 (Intent/Lock) -----+--------> PR6
                         |
  PR3 (Ring Epoch) ------+--------> PR5 (Query 容错)
                         |         PR6
                         |
  PR4 (SWIM/Blacklist) --+--------> PR5
                                    PR6

  PR5 (Query) --(standalone after Phase 1)

  PR7 (HeadSync) --(standalone after Phase 1)

  PR8 (Client SDK) --(builds on PR5/PR6 API)

  PR9  (VID Pre-alloc) --(standalone)
  PR10 (ConnectionPool) --(standalone)

  PR11 (Auto Scale) --(需要 PR4)
  PR12 (Monitoring)  --(standalone, 可随时做)
```

### 13.3 PR 详细说明

#### PR1: WAL + Crash Recovery
- **范围**: `ExtraTiKVController.h`, `PostingRouter.h`
- **核心**: 写入前先写 WAL 到 TiKV (`wal:{nodeId}:{seqNo}`); 启动时扫描前缀重放
- **关键接口**: `WriteWAL()`, `ReplayWAL()`, `ClearWAL()`
- **行数预估**: ~800 行

#### PR2: Intent 状态机 + Lock TTL
- **范围**: `ExtraTiKVController.h`, 新文件 `IntentStateMachine.h`
- **核心**: Split/Merge 操作写 Intent (PREPARED->EXECUTING->COMMITTED); Lock 带 TTL 30s
- **关键接口**: `WriteIntent()`, `AdvanceIntent()`, `AcquireLock()`, `ReleaseLock()`
- **行数预估**: ~1000 行

#### PR3: Ring Epoch Enforcement
- **范围**: `PostingRouter.h`, RPC layer
- **核心**: 所有 inter-node RPC 携带 epoch; 接收方检查, stale 则 reject
- **关键变更**: RPC header 增加 `ring_epoch` 字段
- **行数预估**: ~400 行

#### PR4: SWIM + Blacklist + Owner Detection
- **范围**: 新文件 `SwimGossip.h`, `Blacklist.h`
- **核心**: SWIM ping/indirect-ping/suspect/dead; Blacklist TTL 30s; 触发 ring 更新
- **关键接口**: `SuspectNode()`, `ConfirmDead()`, `AddToBlacklist()`, `IsBlacklisted()`
- **行数预估**: ~1500 行 (最大 PR)

#### PR5: 查询容错 (Client SDK Retry + TiKV 分级容错)
- **范围**: `ExtraDynamicSearcher.h`, Client SDK
- **核心**: TiKV 读分级容错 (Leader retry → Follower Read → 跳过); Client SDK 自动重试到其他 Node
- **行数预估**: ~400 行

#### PR6: 写入 Bypass + Pending Queue
- **范围**: `PostingRouter.h`, `ExtraTiKVController.h`
- **核心**: Owner 不可达时直接写 TiKV; pending queue 异步重试
- **行数预估**: ~800 行

#### PR7: HeadSync TiKV Log + Pull Recovery
- **范围**: `ExtraTiKVController.h`, 新文件 `HeadSyncManager.h`
- **核心**: HeadSync 写入 TiKV durable log; 节点启动时 Pull 追赶; cursor 管理
- **行数预估**: ~600 行

#### PR8: Client SDK + Three-State Write + Client-side LB
- **范围**: 新目录 `Client/`
- **核心**: gRPC client; 三态写入语义; CircuitBreaker; Client-side LB (从 PD 拉节点列表); 批量接口
- **行数预估**: ~1200 行

#### PR9: VID Segmented Pre-allocation
- **范围**: `ExtraTiKVController.h`
- **核心**: PD 分配 VID 段 (e.g., [0, 52万)); 本地消耗, 不联网; PD 故障不影响分配
- **行数预估**: ~300 行

#### PR10: ConnectionPool LRU + HealthCheck
- **范围**: 新文件 `ConnectionPool.h`
- **核心**: Per-peer TCP 连接池; LRU 淘汰; 后台健康检查; 2000 节点 fan-out 支撑
- **行数预估**: ~500 行

#### PR11: 自动扩缩容
- **范围**: PD integration
- **核心**: CPU/latency 触发扩缩容; grace period; ring 更新
- **行数预估**: ~600 行

#### PR12: 监控 + 可观测性
- **范围**: 全局
- **核心**: metrics (Prometheus); 每条链路 latency/error rate; Dashboard
- **行数预估**: ~800 行

### 13.4 PR 阶段与优先级

```
  时间线:

  Phase 1: PR1 + PR2 + PR3 + PR4 (可并行)     -> "一切容错的基础"
  Phase 2: PR5 + PR6 + PR7 + PR8 (依赖 Phase 1) -> "四条链路全覆盖"
  Phase 3: PR9 + PR10                            -> "高可用增强"
  Phase 4: PR11 + PR12                           -> "弹性伸缩 + 可观测性"

  总代码量预估: ~8,200 行 (12 PRs)
```

---

## 14. 附录

### 14.1 Compute Node 写入 Crash 详细矩阵 (C1-C9)

| ID | Crash 时刻 | TiKV 已写数据 | 孤儿数据 | Client 行为 | 恢复方式 |
|----|-----------|---------------|---------|-------------|---------|
| C1 | Client->Node A 网络断 (步骤1前) | 无 | 无 | 超时->retry | 无需恢复 |
| C2 | WAL 写入后, Head Search 前 (步骤2-3) | WAL | 无 | 超时->retry | Node A 重启后 WAL 重放 |
| C3 | Head Search 后, 路由前 (步骤3-4) | WAL | 无 | 超时->retry | WAL 重放 |
| C4 | 本地 Append 进行中 (步骤4a) | WAL + 部分 local | Local 孤儿 | 超时->retry | Client retry 新 VID; 旧孤儿 GC |
| C5 | 本地 Append 完成, Remote 未发 (步骤4a->4b) | WAL + 全部 local | Local 孤儿 | 超时->retry | Client retry 新 VID; 旧孤儿 GC |
| C6 | Remote Append 发送中 (步骤4b) | Local + Owner 部分 | 可能有 | 超时->retry | Client retry |
| C7 | Owner 写入中 (步骤5) | Local + TiKV 部分 | 部分孤儿 | 超时->retry | Client retry |
| C8 | Owner 已完成, Response 在路上 (步骤6) | 全部已写入! | 无! | 超时->retry (不知道成功) | 数据完整, retry 产生重复, VID 去重 |
| C9 | ACK 返回 Client 前 (步骤7-8) | 全部已写入! | 无! | 超时->retry | 同 C8 |

### 14.2 Owner Node Crash 详细矩阵 (O1-O5)

| ID | Crash 时刻 | Owner TiKV 数据 | Node A 感知 | 后续 |
|----|-----------|----------------|------------|------|
| O1 | Request 到达 Owner 前 | 无 | 超时->FAIL | Client retry |
| O2 | Owner 解析请求中 | 无 | 超时->FAIL | Client retry |
| O3 | Owner TiKV 写入中 | 部分 posting | 超时->FAIL | Client retry; 已写部分为孤儿 |
| O4 | Owner 写完, Response 未发 | 全部 posting | 超时->FAIL | 数据完整! Client retry 产生重复 |
| O5 | Response 发送中 | 全部已写入 | 可能收到/可能断连 | 收到: SUCCESS; 断连: retry |

### 14.3 Split Crash 详细矩阵 (S1-S4)

| ID | Crash 时刻 | TiKV 已写 | 后果 | 恢复 |
|----|-----------|----------|------|------|
| S1 | Read posting 后, Write 前 | 无 | Posting 未变 | 下次 Append 重新触发 Split |
| S2 | 新 posting 写入后, Head Index 更新前 | Cluster A/B | 向量可能"消失" | versionMap 仍有, RefineIndex 修复 |
| S3 | Head Index 更新后, HeadSync 前 | 全部完整 | 其他节点不知新 head | Gossip 最终传播; 或 Pull from TiKV |
| S4 | 递归 Split 中 | 部分完整 | 同 S2 | 同 S2 |

### 14.4 Merge Crash 详细矩阵 (M1-M5)

| ID | Crash 时刻 | TiKV 已写 | 后果 | 恢复 |
|----|-----------|----------|------|------|
| M1 | Lock 后, Write 前 | 无 | Partner 暂锁住 | Lock TTL 30s 自动过期 |
| M2 | Put 完成, Delete 未完成 | Winner 有合并数据 | Loser 数据仍在, 重叠 | 上层去重; GC 清理 |
| M3 | Head Index Delete 完成, TiKV Delete 未完成 | Loser posting 孤立 | 空间浪费 | Key GC 清理 |
| M4 | HeadSync 未广播 | 本地操作完成 | 其他节点不知 | Gossip 传播; Pull |
| M5 | Unlock crash | 远程锁未释放 | Lease 锁住 | TTL 30s 过期 |

### 14.5 故障点完整编号表 (F1-F11)

| ID | 故障点 | 位置 | 后果 | 容错机制 |
|----|-------|------|------|---------|
| F1 | Client->Node A 网络断 | 步骤(1) | 请求未到达 | Client timeout + retry |
| F2 | VID 本地 block 用完 | 步骤(2) | 本地 block 余量~52万 | PD 不可用也不影响分配 |
| F3 | RNG Selection head 过时 | 步骤(3) | 选到旧 head | 最终一致, 不影响正确性 |
| F4 | Local Append TiKV 写失败 | 步骤(4a) | Posting 未持久化 | Return FAIL, client retry |
| F5 | Remote 发送失败 | 步骤(4b) | Append 未送达 Owner | 超时->FAIL, client retry |
| F6 | Owner TiKV 写失败 | 步骤(5) | Owner 端 Append 失败 | Owner 返回 FAIL |
| F7 | Owner Crash (部分完成) | 步骤(5) | 部分 item 写入 TiKV | Node A 超时->FAIL, retry |
| F8 | Response 丢失 | 步骤(6) | Node A 不知道结果 | 超时->FAIL, retry (幂等) |
| F9 | Node A Crash | 任意 | 部分 posting 可能已写 | Client retry; VID 扫 TiKV 恢复 |
| F10 | Ring 变更中 | 步骤(4b) | 发给旧 owner | 旧 owner 仍可写 TiKV |
| F11 | Partial Failure | 步骤(4a)+(4b) | Local 成 Remote 失 | 全部失败, client retry |

### 14.6 术语表

| 术语 | 含义 |
|------|------|
| **Client SDK** | 内嵌在业务服务中的客户端库，提供 Client-side LB、重试、CircuitBreaker |
| **Compute Node** | 计算节点，持有 Head Index 全量副本，执行 Head Search + Posting I/O，查询+写入入口 |
| **Owner** | 某个 headID 在 hash ring 上的主节点，负责该 headID 的 Append/Split/Merge（仅写入路径） |
| **Posting** | 一个 headID 下挂载的向量列表，存储在 TiKV 中 |
| **Head Index** | BKT/KDT 图索引，用于 Head Search，每个 Compute Node 都有全量副本 |
| **SWIM** | Scalable Weakly-consistent Infection-style Membership protocol |
| **Blacklist** | 被检测到故障的节点列表，TTL 过期自动移除（用于写入路径 Owner 检测） |
| **Ring Epoch** | Hash Ring 的版本号，每次 topology 变更递增 |
| **WAL** | Write-Ahead Log，写入前先持久化到 TiKV 的日志 |
| **Intent** | Split/Merge 操作的事务意图记录，存储在 TiKV 中 |
| **Lock TTL** | 基于时间的锁，超时自动释放，防止死锁 |
| **HeadSync** | 头索引同步机制，通知所有节点 head 变更 |
| **VID** | Vector ID，向量的唯一标识，分段预分配 |
| **Bypass** | Owner 不可达时绕过 Owner 直接写 TiKV 的备选路径 |
| **PD** | Placement Driver (TiKV 的调度组件)，复用为 Cluster Controller |
| **Raft** | TiKV 底层一致性协议，3 副本 majority commit |

---

*文档版本: v2.0 (Merged)*
*合并自: 容错与弹性伸缩设计 v1 + 对等模式架构设计*
