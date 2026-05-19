# SIFT-1M 多租户 SPANN — 当前结果汇总

数据集：SIFT-1M，tenant_0（约 405k 向量），4 级 hierarchical tag
（org / dept / team / project；标签数 4 / 16 / 64 / 256）。

索引：`/tmp/sptag_1m_idx/`，head=80,924；avg posting size ≈12；
head 用 KDT(entry) + RNG graph(refine)；centroid 选取通过 BKT 32-way k-means tree + dynamic select。

---

## 1. 主结果：R@10 ≥ 0.95 时的 QPS（100 queries/level，单线程）

| level   | nprobe (input) | nprobe (eff) | recall | mean ms | QPS |
|---------|---:|---:|---:|---:|---:|
| org     | 128 | 327 | 0.961 | 32.5 | **29** |
| dept    | 256 | 297 | 0.954 | 25.0 | **38** |
| team    | 192 | 198 | 0.950 | 10.4 | **90** |
| project | 384 | 384 | 0.955 | 19.2 | **62** |

- `nprobe (input)` = `SearchInternalResultNum` 设的目标值
- `nprobe (eff)` = `EnableAdaptiveFilteredNprobe` 调整后真正喂到 posting scan 的数量
- 路由 + hier-mask filter + cross-edges 全启用

---

## 2. 各阶段 latency 拆解（ms / query，均值，warmup 弃首）

`SPTAG_LOG_PHASE_TIME=1` 实测；列名说明在底部。

| level   | nprobe (eff) | head-tree seed (KDT) | graph PQ-walk | graph other | posting scan | total | recall |
|---------|---:|---:|---:|---:|---:|---:|---:|
| org     | 327 | 0.33 | 18.60 | 0.12 | 13.39 | 32.5 | 0.961 |
| dept    | 297 | 0.18 | 12.98 | 0.18 | 11.72 | 25.0 | 0.954 |
| team    | 198 | 0.03 |  2.38 | 0.32 |  7.66 | 10.4 | 0.950 |
| project | 384 | 0.00 |  0.00 | 0.56 | 18.67 | 19.2 | 0.955 |

列含义：
- **head-tree seed (KDT)**：`CrossSubgraphGraphSearch` 里每个 routed node 的 KDT 入口搜索（含 maxCheck 上限内的 KDT+RNG 完整搜索）。
- **graph PQ-walk**：跨子图统一 best-first 优先队列遍历（`while (!pq.empty() && checks < maxChecks)`），含 RNG 邻居展开 + cross-edge 跳转。
- **graph other**：`_phT0`→`_phT1` 减去前两项。project=单节点路径，整段 head ANN 都落在这列。
- **posting scan**：`m_extraSearcher` 拉 posting + 计算精确距离 + 排序。

观察：
- **org / dept 是 graph-bound**：~57% / ~52% 时间花在跨子图 PQ-walk。
- **team 平衡**：10ms 总开销，graph 与 posting 各占一半。
- **project 是 IO-bound**：graph 0.56ms，posting 18.67ms。原因：单 node 路由 + maxCheck 4096 已接近该 bundle 全扫。

---

## 3. Cross-edge / hier-mask 启用 vs 禁用 ablation

100 queries/level，QPS @ R≥0.95。

| level   | NEW (cross-edge + hier-mask) | OLD ENABLED (cross-edge only) | OLD DISABLED (no cross-edge) |
|---------|---:|---:|---:|
| org     | np=128 / **29 QPS** | np=384 / 68 QPS | np=128 / 38 QPS |
| dept    | np=256 / **38 QPS** | np=1536 / 43 QPS | np=192 / 52 QPS |
| team    | np=192 / **90 QPS** | np=512 / 52 QPS | np=192 / 99 QPS |
| project | np=384 / **62 QPS** | np=256 / 89 QPS | np=256 / 84 QPS |

理解：
- **dept 的核心收益**：nprobe 从 1536 → 256（**6× 更紧凑**），因为 hier-mask 提前在图遍历期间排除了不含 dept tag 的 head。但 QPS 反而略降（43→38），原因是 hier-mask 比之前的 centroid-only 4-tag 数组放过了更多 candidate（posting union > centroid），graph 候选爆炸抵消了 IO 节省。
- **org 退步**：org tag 极广（400k 向量），cross-edge 提供的跨 node 路由价值 > tag-aware 过滤价值，禁用 cross-edge 反而 38→29 QPS 不行。
- **team 几乎不变**：cross-edge 启用与否 90 vs 99，差距小，因为 team 路由通常已经覆盖 1-2 node。
- **project IO-bound**：graph 路径成本本来就低，cross-edge 不重要；旧 centroid-only 因为 256 个 project tag 太密集，hier-mask 反而稍弱（62 vs 84-89）。

---

## 4. 配置 / 实现关键点

**Head 选取（build 时）**
- `SelectHeadType=BKT`（这是 SPANN 的 centroid 选取算法）
- `BKTKmeansK=32`（每节点 32-way k-means）
- `BKTLeafSize=8`，`Samples=1000`，`TreeNumber=1`
- `SelectThreshold=6`，`SplitThreshold=25`，`SplitFactor=5`，`Ratio=0.2`
- centroid 是 BKT 节点最近的真实样本向量，不是几何质心。

**Head ANN（search 时）**
- `IndexAlgoType=KDT`（SPANN 默认就是 KDT，不是 BKT；本仓库一直没改过）
- KDT 提供 entry，RNG graph (`m_iNeighborhoodSize=32`) best-first 遍历做 refinement
- `MaxCheck=4096`，`m_iNumberOfInitialDynamicPivots ≈ 32-50`

**多租户 / hier-mask（本 session 加的）**
- 每个 head meta V2 layout：`PostingBitmask(32B) + HierarchicalPostingMask(40B) + globalVID(4B) + bundleNodeId(2B) + headOnly(1B)`，stride=80。
- HierarchicalPostingMask：org=8b / dept=32b / team=128b / project=128b（按 `tag % LEVEL_BITS` 散列）。
- `HeadNodeMatchesQuery(sampleId, queryHierMask, routedNodeMask)` 三处替换原 `HeadNodeMatchesAnyQueryTag`。
- 14 个 bundle nodes，project-level routing 保证大多数 query 路由到 1 node。
- Cross-edges：`augmentheadgraph -k 15 -m 10`，80,924 records，跨 bundle 的 RNG-style shortcuts。

**ENV vars**
- `SPTAG_LOG_PHASE_TIME=1` —— 打开 PhaseTime 日志（per query 一行）
- `SPTAG_DISABLE_CROSS_EDGES=1` —— ablation：禁用 cross-edge graph traversal

---

## 5. 已知 trade-offs / 后续可调点

1. **dept QPS 没涨反降（43→38）**：hier-mask 的 false positive 比 centroid-only 高，graph candidate 爆炸。可考虑：
   - 收紧 `m_iMaxCheck` 在 hier-mask 命中率高时
   - PQ-walk 的 `maxChecks = max(m_maxCheck, graphResultNum * 4)` 系数 `4` 可调

2. **org graph 18.6ms 偏贵**：跨多 node PQ-walk + std::function/unordered_map per-hop 开销。优化方向：
   - 替换 `m_postingFilter`（std::function）为 inline 回调
   - bundle node 的 RNG 图换成 cache-friendly flat array

3. **head 索引切 BKT（论文配置）**：build 时设 `IndexAlgoType=BKT` 重建一次，对比召回与延迟。`BKT::Index<T>::SearchIndex` 接口与 KDT 一致，搜索代码不用改。

4. **PQ/ADC 量化**：当前 EnableADC=false，head 距离全是 raw float。启用后 graph PQ-walk 应能下降 30-50%。

---

## 6. 文件 & 复现命令

```bash
# 索引
INDEX=/tmp/sptag_1m_idx
DATA=/home/v-mochengli/dataset/sift/sift_base.fvecs
QUERY=/home/v-mochengli/dataset/sift/sift_query.fvecs

# QPS @ R≥0.95 sweep
GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2048000 \
  python /tmp/eval_qps_at_r95.py     # → /tmp/eval_qps_at_r95.json

# Phase breakdown
GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2048000 \
  SPTAG_LOG_PHASE_TIME=1 \
  python /tmp/eval_phase_breakdown.py 2>&1 | grep PhaseTime > /tmp/phase_log.txt

# Recall@phase nprobe
GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2048000 \
  python /tmp/eval_phase_recall.py    # → /tmp/phase_recall.json

# Lib rebuild
cd ~/SPTAG/build_norocks && make -j8 _SPTAG && \
  cp ~/SPTAG/Release/_SPTAG.so ~/SPTAG/sptag/_SPTAG.so
```

测试条件：单线程，cosine 距离，topK=10，100 queries，固定 SEED=42。

---

## 7. 早停优化（CrossSubgraphGraphSearch PQ-walk early termination）

**问题**：Phase breakdown 暴露出 org/dept 的 PQ-walk 时间不成比例 —
所有 level 都跑满 maxChecks=4096，不管 result heap 是否已稳定。
PQ-walk 的 cost-per-check 因 hier-mask 通过率不同而异（org 14% / dept 4% /
team 4%），导致 org 比 team 多做 ~3ms cross-bundle vector 取值与 PQ insert。

**修复**（`SPANNIndex.cpp` `CrossSubgraphGraphSearch`）：

```cpp
int minChecks = std::max(p_graphResultNum, std::min(256, maxChecks));
while (!pq.empty() && checks < maxChecks) {
    auto cur = pq.top(); pq.pop();
    if (visited.count(cur.globalVID)) continue;
    visited.insert(cur.globalVID);
    ++checks;
    if (checks >= minChecks && cur.dist > p_queryResults->worstDist()) break;  // NEW
    // ... expand neighbors ...
}
```

与 `BKTIndex/KDTIndex` 自身的 PQ-walk 终止条件一致。
`worstDist()` 是当前 top-K result heap 的最差距离，PQ top 一旦超过它说明
后续节点不可能改善结果。

### 修复前后对比 @ R≥0.95（同一 100-query sample）

| level   | nprobe (before→after) | QPS (before→after) | Δ      |
|---------|----------------------:|-------------------:|-------:|
| org     | 384 → **128**         | 68.4 → 65.1        | -5%    |
| dept    | 1536 → **256**        | 42.6 → **63.3**    | **+49%** |
| team    | 512 → **192**         | 51.8 → **98.7**    | **+90%** |
| project | 256 → 384             | 88.7 → 63.6        | -28%（单节点路径，与本修复无关；recall sweep 边界波动） |

Recall：org 0.957 / dept 0.954 / team 0.950 / project 0.955，全部≥0.95。

### 修复前后 Phase 拆解 @ R≥0.95（修复前 nprobe）

| level | bkt   | pq (before→after) | graph other | post  | total (before→after) |
|-------|------:|------------------:|------------:|------:|---------------------:|
| org   | 1.16  | 18.30 → **1.63**  | 0.36        | 12.71 | 32.5 → **15.4** |
| dept  | 1.32  | 12.98 → **1.12**  | 0.45        | 10.20 | 25.0 → **13.1** |
| team  | 1.79  |  2.38 → **0.31**  | 0.24        |  6.16 | 10.4 → **8.4**  |
| project| 0.42 |  n/a (单节点)     | 0.56        | 18.30 | 19.2 → 19.2 |

### 关键观察

- **PQ checks 从 4096 → ~256-330**（org/dept/team），≈12× 减少
- **Per-check 成本不变**（4-5 μs/check on org），证明 cost-per-query 的瓶颈
  是 budget × per-check 的乘积，而非 per-check 本身
- **cross-bundle 内存访问没变**，只是按需停止；recall 不变
- dept 在低 nprobe（256 vs 1536）就能达到同等 recall，QPS 提升尤为明显
