# Tenant 0 Head-Bundle V5 Greedy Local-Selectivity Checkpoint

Date: 2026-04-16

This note records the current 1M tenant-0 head-bundle checkpoint after two changes landed together:

- build-time routing nodes switched to greedy leaf packing with minimum local selectivity `>= 5%`
- runtime adaptive filtered nprobe was corrected to rescale tenant-level filter selectivity into routed-node local selectivity before computing posting target

The result is the current `v5_greedy_localsel` checkpoint.

## Inputs

- Current artifact scenario: `/home/v-mochengli/test/tenant_tag_scenario_1m_headbundle_20260416_v5_greedy.json`
- Current artifact index: `/home/v-mochengli/test/tenant_index_tags_1m_headbundle_20260416_v5_greedy`
- Current benchmark result: `/home/v-mochengli/test/tenant0_selectivity_latency_20260416_headbundle_v5_greedy_localsel_1000_nowarm/result.json`
- Pre-fix greedy benchmark result: `/home/v-mochengli/test/tenant0_selectivity_latency_20260416_headbundle_v5_greedy_1000_nowarm/result.json`
- Tree baseline result used for comparison: `/home/v-mochengli/test/tenant0_selectivity_latency_20260415_headbundle_v4_tree_1000_nowarm/result.json`
- Query file: `/home/v-mochengli/dataset/sift/sift_query.fvecs`
- Num queries: `1000`
- TopK: `10`

## Current Implementation Logic

### Build-Time Routing Plan

- `Wrappers/src/CoreInterface.cpp` now builds the routing plan from the deepest tag level.
- `BuildPivotEstimatorComputation(...)` greedily packs leaf tags into routing nodes while maintaining minimum node-local leaf selectivity `>= 5%`.
- For the current 1M tenant-0 artifact, this produced:
  - `pivot_level = 3`
  - `node_count = 14`
  - `assignment_ratio = 1.0`
  - `route_mismatch_count = 0`

### Query Routing Path

- Dense tag queries still go through the routed head-bundle path.
- `Wrappers/src/CoreInterface.cpp` computes routed nodes for the query tag set and stores them in thread-local search context as `m_searchHeadBundleNodes`.
- `AnnService/src/Core/SPANN/SPANNIndex.cpp` loads those routed nodes into `candidateNodes`, runs graph search on each selected bundle node, and merges the head results into one global heap before posting scan.

### Adaptive Nprobe Logic

- Before the 2026-04-16 runtime fix, adaptive nprobe still used tenant-global filter selectivity even when the query had already been routed to a much smaller set of bundle nodes.
- `AnnService/src/Core/SPANN/SPANNIndex.cpp` now rescales `filterSelectivity` from tenant scope to candidate-node scope using routed-node `assignmentCount`.
- This changes posting budget materially for low-level tags without changing the graph-search path.

### What Is Not Implemented

- No special org/dept safeguard is enabled.
- No fallback broad-query branch to tenant-global head graph is enabled.
- No double-buffer or pipelined posting read is implemented.

## Per-Level Performance

Reference comparison below uses `v4_tree` as the previous tree-structured checkpoint and `v5_greedy_localsel` as the current checkpoint.

| Level | Recall v4_tree | Recall Current | Recall Delta | QPS v4_tree | QPS Current | QPS x | Avg Lat v4_tree | Avg Lat Current | Avg Posting Read v4_tree | Avg Posting Read Current |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| org | 0.9436 | 0.8204 | -0.1232 | 135.71 | 20.67 | 0.15 | 7.34 ms | 48.36 ms | 127.87 | 78.00 |
| dept | 0.9890 | 0.8981 | -0.0909 | 75.73 | 148.10 | 1.96 | 13.18 ms | 6.73 ms | 254.79 | 98.98 |
| team | 0.9882 | 0.9467 | -0.0415 | 38.72 | 173.99 | 4.49 | 25.80 ms | 5.72 ms | 504.81 | 137.96 |
| project | 0.9879 | 0.9396 | -0.0483 | 20.24 | 92.49 | 4.57 | 49.38 ms | 10.78 ms | 992.17 | 271.92 |

## Why Low-Level Tags Improved

- Team/project node-local selectivity improved substantially under greedy leaf packing.
- For the benchmark tags used in the 1000-query run:
  - team `3053`: `8.56% -> 21.53%`
  - project `4163`: `2.22% -> 5.57%`
- After the runtime nprobe fix, that local selectivity now actually reduces posting fanout instead of only improving graph locality.
- Posting read dropped from roughly `505 -> 138` on team and `992 -> 272` on project, which is the main reason low-level QPS recovered.

## Workload Assumption

The intended workload assumption for this checkpoint is:

- query popularity follows tag popularity
- most users are ordinary users and query low-level tags
- only a minority of users have access to high-level org/dept tags

Under the synthetic hierarchy in the current 1M scenario, tag cardinalities are:

- org: `4`
- dept: `16`
- team: `64`
- project: `256`

If query probability is weighted by those tag counts, the level weights are:

- org: `1.18%`
- dept: `4.71%`
- team: `18.82%`
- project: `75.29%`

## Weighted Aggregate Under The Assumed Query Mix

| Metric | v4_tree | Current |
| --- | ---: | ---: |
| Weighted Avg Latency | 42.74 ms | 10.08 ms |
| Effective QPS | 23.40 | 99.19 |
| Weighted Recall | 0.9875 | 0.9376 |
| Weighted Avg Posting Read | 855.56 | 236.29 |

Relative to `v4_tree` under this workload model:

- weighted latency improves by `4.24x`
- effective QPS improves by `4.24x`
- weighted average posting read improves by `3.62x`
- weighted recall drops by about `0.0499`

## Current Readout

- The current checkpoint is intentionally optimized for low-level tag dominated traffic.
- Under that assumption, the current `v5_greedy_localsel` tradeoff is favorable overall even though org recall/QPS regressed.
- The runtime nprobe correction was necessary: before it, greedy leaf packing improved local selectivity but did not improve team/project posting fanout.
- The current implementation does not attempt to protect broad org/dept queries with a separate global graph path.
- No further runtime complexity is required for the current checkpoint if about `20 QPS` on org is considered acceptable for the minority management workload.

## Current Conclusion

- Keep the current checkpoint as the active implementation baseline for low-level-tag dominated workloads.
- Revisit org/dept protection only if future workloads show materially higher high-level query share.
- Revisit posting-read pipelining only if tail-latency reduction becomes a priority beyond the current acceptable performance envelope.