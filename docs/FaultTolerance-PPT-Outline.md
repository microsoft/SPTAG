# 容错设计 PPT 讲稿大纲

## 建议演讲结构

**总时长**: 40-50 分钟 (30 分钟讲 + 10-20 分钟讨论)

**核心思路**: 不要按模块讲，而是**沿着一个请求的生命周期走一遍**，在每个步骤上停下来问"这里挂了怎么办"。听众跟着数据流走，自然就理解了整个容错设计。

---

## Slide 1: 标题页

```
SPTAG/SPANN 容错设计
—— 面向 2000 节点集群的故障容忍方案

[你的名字]
[日期]
```

---

## Slide 2: 我们要解决什么问题 (2 分钟)

**开场直接给冲击力**：

```
2000 个节点的集群

  • 每台机器年故障率 ~2%
  • 2000 × 2% = 每年 ~40 次节点故障
  • 平均每周都有一台机器挂掉
  • 高峰期可能同时有 3+ 台不可用

问题: 任意节点随时可能挂，用户不能感知到。
```

> 讲法: "我们不是在讨论会不会挂，而是挂了以后怎么办。"

---

## Slide 3: 系统架构一张图 (3 分钟)

画出完整架构图:

```
Client × P
    ↓ LB
Aggregator × M (无状态)
    ↓ Fan-out
Compute Node × 2000 (Head Index + PostingRouter)
    ↓ gRPC
TiKV × 2000 (3 副本 Raft) + PD × 5
```

> 讲法: "我们从上往下讲，每一层挂了会怎样。"

**顺便解释**: 当前 benchmark 是 1 个 Driver 三合一（Client+Aggregator+Compute），生产要拆开。

---

## Slide 4: 五大设计原则 (3 分钟)

一张图列出五个原则，**每个只用一句话**:

```
① Synchronous Write    写入成功 = 数据 100% 在 TiKV Raft
② Idempotent Retry     所有操作可安全重试
③ Consistent Hashing   节点增删只影响 ~1/N 的数据
④ SWIM Gossip          去中心化故障检测，无单点
⑤ Owner Blacklist      故障节点立即跳过，不等超时
```

> 讲法: "记住这五个原则，后面每个设计决策都能回溯到其中之一。"

---

## Slide 5-8: 跟着一个查询走 (8 分钟)

**这是最核心的部分。画一个时序图，逐步走。**

### Slide 5: 查询正常路径

```
Client → Aggregator → Compute Node → Head Search → TiKV Get → 返回
  ①         ②            ③             ④           ⑤⑥       ⑦⑧
```

> 讲法: "正常情况下一个查询是这么走的，3ms 左右返回。现在我们逐个断开看看。"

### Slide 6: 查询故障 — Client/Aggregator 层

```
① Client → Aggregator 断开
   → LB 自动切到另一个 Aggregator (< 1s)
   → Client 无感知

⑦ Aggregator 自己挂了
   → 完全无状态，LB 切换，零恢复成本
```

> 讲法: "Aggregator 是最容易处理的，因为它无状态。挂了就换一个。"

### Slide 7: 查询故障 — Compute Node 层

```
② Aggregator → Compute Node 断开
   → Blacklist 标记这个节点
   → 请求路由到其他健康节点
   → Head Index 是全量副本，任何节点都能搜

③ Head Search 进程内崩溃
   → 节点自杀重启
   → Aggregator 超时后切到其他节点
```

> 讲法: "关键洞察：Head Index 每个节点都有完整副本，所以跳过任何一个节点，查询结果不受影响。"

### Slide 8: 查询故障 — TiKV 层

```
④⑤ TiKV Get 失败
   → Region Cache 失效 + 重路由 (已实现，10 次重试)
   → Leader 不可达 → Follower Read 降级
   → SPANN 是近似搜索，stale data 只影响精度不影响正确性

⑥ 部分 Compute Node 超时
   → Aggregator 只要收到 >= 1 个结果就返回
   → 标记 degraded=true
```

> 讲法: "查询容错的终极保底：近似搜索天然容忍不完美的数据。"

---

## Slide 9-12: 跟着一个写入走 (8 分钟)

### Slide 9: 写入正常路径

```
Client → Node A → WAL → PostingRouter → Owner Node B → Lock → TiKV Put → Unlock → ACK
  ①        ②      ③        ④              ⑤           ⑥⑦      ⑧⑨       ⑩⑪      ⑫
```

> 讲法: "写入比查询复杂得多，因为它涉及路由、Lock、持久化三个阶段。"

### Slide 10: 写入故障 — 路由层

```
④ PostingRouter → Owner 不可达
   情况 A: Owner 在 Blacklist
   → 降级: 跳过 Owner，直接写 TiKV
   → WAL 记录 bypass 标记
   → Owner 恢复后 reconcile

   情况 B: Owner 超时但不在 Blacklist
   → 重试 3 次 + exponential backoff
   → 超时后加入 Blacklist
```

> 讲法: "写入 bypass 是最复杂的设计决策。为什么不一开始就允许 bypass？因为需要 WAL + Intent + Lock TTL 三个前置保障。"

### Slide 11: 写入故障 — 持久化层

```
⑧ TiKV Put 失败
   → 重试 (TiKV Raft 保证：成功 = 3 副本写入)
   → 失败 → 返回错误给 Client
   → Client 可安全重试 (VID + Version 去重)

为什么安全？Synchronous Write 语义：
  成功 = 数据一定在
  失败 = 数据一定不在
  不存在"不知道有没有写进去"的状态
```

### Slide 12: 写入故障 — Crash Recovery

```
Node A 写到一半 crash:
  → WAL 已在 TiKV → 重启后重放
  → VID+Version 去重防止重复写入

Split/Merge 执行到一半 crash:
  → Intent 状态机 (PREPARED → EXECUTING → COMMITTED)
  → 重启后检查 Intent → Resume 或 Rollback
  → 不会留下半完成状态

Remote Lock 持有者 crash:
  → Lock 带 TTL (30s) → 自动过期释放
  → 不会死锁
```

> 讲法: "三个 crash 场景，三个不同机制，但核心都是同一个原则：Idempotent Retry。"

---

## Slide 13: 节点 Crash 总表 (3 分钟)

**一张大表，快速过**:

```
┌─────────────────┬──────────────┬────────────┬──────────────────┐
│ 组件挂了         │ 影响          │ 检测时间    │ 恢复时间          │
├─────────────────┼──────────────┼────────────┼──────────────────┤
│ Client          │ 该用户请求丢失 │ 即时        │ 重启即恢复        │
│ Aggregator      │ 在途查询丢失   │ LB < 1s    │ 无需恢复 (无状态) │
│ Compute Node    │ 该节点查询丢失 │ SWIM ~5-15s│ 30s (加载 Head)  │
│ TiKV Store      │ 无 (Raft)     │ Raft < 10s │ 自动 Leader 切换  │
│ PD (1/5)        │ 无            │ Raft       │ 自动选举          │
│ PD (3/5)        │ 调度暂停      │ —          │ 恢复 1 个即可     │
└─────────────────┴──────────────┴────────────┴──────────────────┘
```

> 讲法: "注意看恢复时间那一列——最慢的是 Compute Node 的 30 秒，但用户查询在 SWIM 检测到故障后立即切走，所以用户感知到的中断是 0。"

---

## Slide 14: SWIM Gossip — 怎么发现故障 (3 分钟)

```
每个节点每秒:
  1. 随机选 1 个节点 Ping
  2. 没回？→ 找 3 个代理 Ping-Req
  3. 还没回？→ 标记 SUSPECT
  4. 10 秒仍不回 → 标记 DEAD
  5. 状态变更 piggyback 在所有消息中传播

为什么用 SWIM 不用集中式心跳？
  集中式: 1 个监控节点要维护 2000 条连接 → 单点
  SWIM: 每节点只维护 O(1) 连接 → 天然适合 2000 节点
```

> 讲法: 可以用一个动画效果逐步展示 Ping → Ping-Req → SUSPECT → DEAD 的过程。

---

## Slide 15: 从 Benchmark 到生产的差距 (2 分钟)

```
当前 Benchmark:                    生产目标:
  1 Driver (三合一)                 P Clients
  2-3 Worker Nodes                  M Aggregators (≥ 3)
  无故障检测                         2000 Compute Nodes
  无 Aggregator 层                   SWIM + Blacklist
  WAL 仅 RocksDB                    TiKV WAL
  硬编码节点列表                     PD 动态管理
```

> 讲法: "这就是为什么需要 12 个 PR。"

---

## Slide 16: PR 计划 (3 分钟)

画一个甘特图/依赖图:

```
Week:  1    3    5    7    9    11   13   15
       ├────┤
       PR1 (TiKV WAL) ──→ PR2 (Lock+Intent) ──→ PR6 (Write Bypass)
       ├────────┤
       PR3 (SWIM) ──→ PR4 (Blacklist) ──┐
       ├────┤                            ├──→ PR7 (Aggregator) → PR8 (SDK)
       PR5 (PD+Epoch) ──────────────────┘
       ├────┤
       PR9 (HeadSync) ──→ PR10 (Follower+Ckpt) ──→ PR11 (Scaling)
                                                          │
                                                     PR12 (加固)

  🛡️ 容错: PR1-PR10, PR12 (11 个)
  📈 伸缩: PR11 (1 个)
```

> 讲法: "PR1/3/5/9 可以并行，关键路径 ~12-16 周。"

---

## Slide 17: 吞吐量为什么能线性扩展 (2 分钟)

```
一个查询 = 3 个阶段:

  阶段 A: Client/Aggregator 接入    → M 个 Aggregator 并行
  阶段 B: Head Search (CPU)         → N 个 Compute Node 并行
  阶段 C: Posting Read (I/O)        → N 个 TiKV 并行

  每一层独立扩展，无单一瓶颈。
  2000 Compute + 40 Aggregator → 理论 2000 万 QPS
```

> 讲法: "容错设计的副产品就是天然的水平扩展能力。没有单点 = 没有瓶颈。"

---

## Slide 18: 总结 — 记住这三点 (1 分钟)

```
① 每个写入要么完全成功，要么完全失败，可以安全重试
   (Synchronous Write + Idempotent Retry)

② 任何节点挂了，秒级检测，用户零感知
   (SWIM + Blacklist + 全量 Head 副本)

③ 所有状态都在 TiKV (3 副本 Raft)，Compute 可随时重建
   (WAL/Intent/HeadSync Log 全部持久化到 TiKV)
```

---

## Slide 19: Q&A

准备好回答这些可能的问题:

- "SWIM 的误判率怎么控制？" → SUSPECT 阶段 10s 缓冲，Ping-Req 3 代理交叉验证
- "Write Bypass 怎么保证不和 Split/Merge 冲突？" → Intent 状态机 + Lock TTL
- "Follower Read 的 stale 程度？" → TiKV Raft 通常 <100ms 延迟，对近似搜索可忽略
- "为什么不用 ZooKeeper 做故障检测？" → 2000 节点 ZK 压力大 + 单点依赖
- "12 个 PR 能不能合并一些？" → 可以，但 Blacklist 读/写必须分开 (安全考量)

---

## 演讲技巧建议

1. **不要按模块讲，要按数据流讲**
   - ❌ "先讲 SWIM，再讲 Blacklist，再讲 WAL..."
   - ✅ "一个查询从 Client 发出去，到底经历了什么，每一步挂了怎么办"

2. **每讲完一个故障场景，回扣设计原则**
   - "这里能安全重试，是因为 Idempotent Retry 原则"
   - "这里秒级切走，是因为 SWIM + Blacklist"

3. **Slide 5-12 是核心**，占一半时间
   - 查询流 + 写入流覆盖了 80% 的容错设计
   - 其他 Slide 是辅助

4. **用颜色区分**
   - 🟢 绿色 = 正常路径
   - 🔴 红色 = 故障点
   - 🔵 蓝色 = 恢复动作

5. **准备一个 demo 场景**（如果有时间）
   - 3 节点集群跑 benchmark → kill 1 个 worker → 观察查询自动切走
   - 这比任何 Slide 都有说服力
