# Tenant-Tag System Summary For Next Optimization Chat

Date: 2026-04-13

This document is a compact handoff for the next Copilot chat. It summarizes the current system, the main code paths, the recent branch work, the validated benchmark workflow, and the most likely optimization directions.

## 1. System Goal

The system serves multi-tenant ANN search over SPANN indices with ACL-style tag filtering.

Current goals:

- support many tenants through a single `TenantIndexManager`
- keep large SPANN postings on SSD while caching HeadIndex structures in memory
- support tag-filtered search with a sparse fast path and a dense adaptive path
- provide reproducible benchmark workflows for cache pressure and filtered-search quality

## 2. Main Functional Capabilities

### Multi-tenant serving

Core API lives in:

- `Wrappers/inc/CoreInterface.h`
- `Wrappers/src/CoreInterface.cpp`

Key manager capabilities:

- `LoadAll` and `SaveAll` for unified multi-tenant storage
- lazy tenant loading through `EnsureTenantLoaded`
- HeadIndex cache budget via `SetHeadIndexCacheLimit`
- cache accounting via `GetHeadIndexCacheUsage`
- tenant eviction via `UnloadTenant`
- new getter APIs:
  - `GetTenantHeadIndexSize`
  - `GetTagRoutingStatsBlob`

### ACL / tag-filtered search

Filtered search entry point:

- `TenantIndexManager::SearchWithACL` in `Wrappers/src/CoreInterface.cpp`

Build-time tag metadata path:

- `TenantIndexManager::BuildSignatures` in `Wrappers/src/CoreInterface.cpp`
- sparse side index in `AnnService/inc/Core/Cache/PostingSignature.h`

Adaptive dense search path:

- `SPANN::Index<T>::SearchIndex` in `AnnService/src/Core/SPANN/SPANNIndex.cpp`

### Benchmark and reproducibility tooling

Formal repo benchmark entry points:

- `Tools/benchmarks/multitenant_tag_cache_stress.py`
- `Tools/benchmarks/run_multitenant_tag_cache_stress.sh`
- `Tools/benchmarks/README.md`

Benchmark workflow records:

- seed
- search params
- git commit / dirty state
- summary tables in md/json/csv
- human-readable meta files

## 3. Current Query-Time Logic

### Step 1: tenant loading and cache scheduling

`EnsureTenantLoaded` does the following:

- hot path under shared lock checks if the tenant is already loaded
- slow path takes exclusive lock
- estimates tenant HeadIndex size for cache accounting
- evicts least-recently-used tenants when the cache budget would be exceeded
- skips eviction for pinned tenants still in use by another search thread
- loads tenant index from disk and reapplies pending search params

Important limitation:

- cache admission still uses a rough size estimate derived from tenant vector count, not the exact on-disk HeadIndex size

### Step 2: ACL route choice in `SearchWithACL`

Current routing behavior:

- sparse path is used only when all query tags have direct posting lists and the exact union of posting IDs is at most `DirectSparseMaxPostings`
- otherwise the query takes the dense path
- `ForceDenseTagSearch=true` can disable the sparse route at runtime

### Step 3: sparse route

Sparse route behavior:

- uses the direct posting union from the sparse side index
- bypasses normal graph routing
- still uses exact inline tag checking during posting scan

This route is selective and intentionally narrow. It now depends on exact posting fanout rather than tenant-size-scaled heuristics.

### Step 4: dense route

Dense route behavior in the current branch:

- exact inline tag filtering remains active through `m_queryTags`
- tag-routing stats and head metadata are used to estimate selectivity
- head-node tag metadata helps preserve valid head hits
- posting-level PS gating is currently disabled in the dense path by setting `m_postingFilter = nullptr`

This means the dense branch currently behaves like a graph + posting-scan upper-bound path with exact inline tag enforcement, not a fully PS-gated posting selector.

### Step 5: adaptive nprobe / posting target

Current formula lives in `AnnService/src/Core/SPANN/SPANNIndex.cpp`.

Core idea:

- base budget is `max(SearchInternalResultNum, topk)`
- if selectivity is below 1.0, compute:
  - `postings_for_recall = target_recall * topk / expected_matches_per_posting`
  - `postings_for_coverage = base / selectivity^coverage_exponent`
- final `postingTarget = max(base, postings_for_recall, postings_for_coverage)` capped by head count

Current validated defaults:

- `DirectSparseMaxPostings=320`
- `FilteredSearchNprobeSafety=1.0`
- `FilteredSearchTargetRecall=1.0`
- `FilteredSearchCoverageExponent=0.5`
- `ForceDenseTagSearch=false`

## 4. Current Build-Time Tag Metadata Logic

`BuildSignatures` currently does all of the following:

- reads real posting-to-vector assignments back from SPANN SSD files
- rebuilds posting-level tag lists
- builds tenant posting signatures / bitmasks
- computes exact per-tag vector counts from the original tenant tag matrix
- computes exact per-tag posting counts from posting assignments
- writes exact routing stats into `m_tenantTagRoutingStats`
- materializes sparse direct-posting lists only for tags with bounded posting fanout
- builds head-node tag metadata for vectors not found in postings

Recent important correctness change:

- sparse materialization now uses exact posting fanout, not vector-count thresholding tied to a fixed `topk=10` assumption

## 5. Recent Branch Work

### Search quality and routing

- replaced vector-count-based sparse classification with exact posting-union gating
- added runtime knobs for sparse-vs-dense routing and adaptive dense search
- made `SetSearchParam` persistent across lazy tenant loads via `m_pendingSearchParams`
- fixed filtered result copy-back so `topk > SearchInternalResultNum` is safe
- made dense base budget `topk`-aware in SPANN search
- added manager/wrapper getters for:
  - per-tenant HeadIndex size
  - exact tag routing stats

### Benchmarking and reproducibility

- built formal repo benchmark runner under `Tools/benchmarks/`
- added exact recall and avg selectivity to the cache-stress output tables
- standardized artifact output under md/json/csv/meta files
- smoke-tested the repo runner with a small `20` query run after formalization

### Stability and rebuild lessons

- after wrapper or `VectorIndex` layout changes, stale wrapper binaries can cause misleading crashes or missing symbols
- a full clean rebuild plus wrapper sync is sometimes required before debugging logic-level issues

## 6. Current Performance State

### Multi-tenant cache stress

Validated 10-tenant, 1000-query run:

- sequential batches: `31.87 ms` avg, `74.85 ms` p95, `31.38` qps, `0.9953` recall
- random mixed tenants: `47.27 ms` avg, `122.46 ms` p95, `21.15` qps, `0.9945` recall

Takeaway:

- cache locality is the dominant difference between sequential and random tenant scheduling
- current cache pressure hurts latency much more than recall

### Top-k generality

Takeaway from the 1M sweep:

- broad filters stay stable as `k` grows
- narrow filters are not yet stable across `k`
- `topk=100` still exposes quality/latency issues on team/project-scale filters

## 7. Known Gaps And Cautions

- dense path currently disables posting-level PS gating; benchmark labels may still say `PS-filter`, so read code semantics, not only labels
- cache budgeting in `EnsureTenantLoaded` still relies on a rough estimate instead of exact per-tenant HeadIndex bytes
- `BuildSignatures` is heavy because it reconstructs posting assignments from SSD data
- benchmark exact recall computes GT from base vectors in Python and is therefore an evaluation tool, not a serving-time path
- package wrapper binaries and `Release/` binaries can drift after rebuilds if not kept in sync

## 8. Recommended Next Optimization Directions

### High-value targets

1. Improve multi-tenant cache admission and eviction using exact HeadIndex sizes instead of the current estimate.
2. Revisit dense-path posting selection; current dense branch likely leaves performance on the table because posting-level PS gating is disabled.
3. Re-evaluate adaptive dense coverage for `topk=50/100`, especially on `team` and `project` filters.
4. Reduce `BuildSignatures` overhead or cache more of its outputs when repeatedly rebuilding signatures for the same tenant artifacts.

### Good next prompts for another Copilot chat

1. "Read `Wrappers/src/CoreInterface.cpp` and propose how to replace the cache-size estimate in `EnsureTenantLoaded` with exact per-tenant HeadIndex bytes while keeping lock contention low."
2. "Analyze the current dense path in `SearchWithACL` and determine whether re-enabling a safe posting-level PS filter can improve latency without hurting recall."
3. "Use `Tools/benchmarks/multitenant_tag_cache_stress.py` to design a cache-budget sweep and summarize where recall or tail latency changes sharply."
4. "Investigate why narrow filters are unstable at `topk=100` by tracing selectivity estimation, posting target computation, and post-graph coverage."

## 9. Source Files Worth Reading First

- `Wrappers/inc/CoreInterface.h`
- `Wrappers/src/CoreInterface.cpp`
- `AnnService/src/Core/SPANN/SPANNIndex.cpp`
- `AnnService/inc/Core/Cache/PostingSignature.h`
- `Tools/benchmarks/multitenant_tag_cache_stress.py`
- `Tools/benchmarks/run_multitenant_tag_cache_stress.sh`
- `docs/reports/2026-04-13-tenant-tag-system/current-performance-comparison.md`