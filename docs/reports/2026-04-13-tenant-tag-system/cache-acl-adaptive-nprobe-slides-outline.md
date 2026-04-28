# Slides Outline: Cache Scheduling, ACL Design, And Adaptive Nprobe

Date: 2026-04-13

## Slide 1: Title

- Multi-Tenant Filtered Search In SPANN
- Cache scheduling, ACL design, adaptive dense search
- Branch snapshot and benchmark status as of 2026-04-13

## Slide 2: Problem Statement

- Multi-tenant ANN serving needs high recall, bounded tail latency, and strong tenant isolation
- Tag filtering complicates both routing and posting scan efficiency
- HeadIndex memory is limited, so tenant interleaving directly affects cache behavior

## Slide 3: System Layout

- `TenantIndexManager` owns multi-tenant load/save/search lifecycle
- SPANN postings stay on SSD, HeadIndex stays in memory on demand
- ACL/tag filtering is handled in `BuildSignatures` and `SearchWithACL`
- Dense-path search budget is decided in `SPANNIndex::SearchIndex`

## Slide 4: Cache Scheduling Optimization

- tenants are loaded lazily through `EnsureTenantLoaded`
- HeadIndex cache has an explicit byte budget
- LRU-like eviction removes cold tenants unless they are pinned by active searches
- benchmark can optionally drop OS page cache on eviction for stress testing
- new getter `GetTenantHeadIndexSize` exposes exact per-tenant HeadIndex bytes to the wrapper layer

## Slide 5: ACL Design

- `BuildSignatures` reconstructs posting-to-vector assignments from SSD data
- exact per-tag vector counts and posting counts are materialized
- sparse side index stores direct posting unions only for bounded posting fanout
- head-only vectors get tag metadata so valid head hits are not lost
- new getter `GetTagRoutingStatsBlob` exposes exact routing stats to Python tools

## Slide 6: Sparse vs Dense Routing

- sparse route triggers only when all query tags have direct posting lists and the exact posting union is small enough
- `DirectSparseMaxPostings` controls the sparse fast-path boundary
- `ForceDenseTagSearch` forces all filtered queries onto the dense path
- current design avoids tenant-size-scaled heuristics and uses exact posting fanout instead

## Slide 7: Adaptive Dense Search / Nprobe

- dense path computes selectivity from tag routing stats and head metadata
- `nprobeBase = max(SearchInternalResultNum, topk)`
- if `m_filterSelectivity < 1.0`:
  - `sel = max(m_filterSelectivity, 1e-6)`
  - `tenantSize = (vectorSize > 0 ? vectorSize : numSamples)`
  - `postingCount = max(1, numSamples)`
  - `avgPosting = max(1.0, tenantSize / postingCount)`
  - `recallTarget = clamp(FilteredSearchTargetRecall, 0.01, 1.0)`
  - `coverageExponent = clamp(FilteredSearchCoverageExponent, 0.0, 2.0)`
  - `filteredTopK = (topk > 0 ? topk : 10)`
  - `expectedMatchesPerPosting = max(1e-6, avgPosting * sel)`
  - `postingsForRecall = ceil(filteredTopK * recallTarget / expectedMatchesPerPosting)`
  - `coverageDenominator = max(1e-6, sel ^ coverageExponent)`
  - `postingsForCoverage = ceil(nprobeBase / coverageDenominator)`
  - `postingTarget = min(numSamples, max(nprobeBase, postingsForRecall, postingsForCoverage))`
- else `postingTarget = nprobeBase`
- graph phase requests `graphResultNum = postingTarget`
- runtime knobs:
  - `FilteredSearchNprobeSafety`
  - `FilteredSearchTargetRecall`
  - `FilteredSearchCoverageExponent`

## Slide 8: Current Multi-Tenant Cache Results

- sequential tenant batches: `31.87 ms` avg, `74.85 ms` p95, `31.38` qps, `0.9953` recall
- random mixed tenants: `47.27 ms` avg, `122.46 ms` p95, `21.15` qps, `0.9945` recall
- random tenant mixing increases avg latency by `48.32%` and p99 by `240.44%`
- recall remains near 1.0, so locality is the main differentiator

## Slide 9: Current Top-k Findings

- broad filters are relatively stable as `k` increases
- narrow filters such as `1_team` and `1_project` are not yet `topk`-general
- latency rises sharply for narrow filters at `k=50/100`
- next optimization should focus on dense coverage and high-`k` filtered search

## Slide 10: Recent Branch Work

- exact posting-union sparse routing
- lazy-load-safe search param persistence
- filtered `topk > 64` result copy-back fix
- topk-aware dense base budget
- wrapper getters for HeadIndex size and tag routing stats
- formal repo benchmark runner under `Tools/benchmarks/`

## Slide 11: Risks And Open Questions

- dense path currently disables posting-level PS gating
- cache admission still uses a rough size estimate in the manager hot path
- `BuildSignatures` remains expensive because it rebuilds from SSD artifacts
- wrapper/package binary drift can confuse debugging after rebuilds

## Slide 12: Next Steps

- cache-budget sweep under the formal benchmark runner
- revisit dense posting pre-filter design
- high-`k` filtered-search tuning for team/project filters
- reduce signature rebuild cost and improve metadata reuse