# Multi-Tenant SIFT-1M — Huffman Partition Size Ablation (post bug-fix)

Dataset: SIFT-1M, tenant_0 (404 819 vectors), 4-level ACL tree
(org/dept/team/project, cardinalities **4 / 16 / 64 / 256**).
Query: 100 SIFT queries, top-10, recall target **R ≥ 0.95**.

Build params shared across all partition sizes:
`final_ratio=0.05`, `oversample=3.0`, `merge_group=5`,
`ReplicaCount=8`, `PostingVectorLimit=118`,
`SelectHeadType=PerTagBKTMerge`, `NumTagsPerVec=4`.

The partition size **N** is the number of Huffman tree-aware subsets
(`SPTAG_PIVOT_FORCE_NODE_COUNT`); each subset has its own BKT over the
heads assigned to it.

## Two build/query bugs fixed in this iteration

### Bug 1 — `RNGSelection` global-search-then-filter (build-time vector drop)

In `ExtraDynamicSearcher.h::RNGSelection`, build-time posting selection
used to search the **global** head BKT for ~32 candidates and then
filter by `p_allowedHeads`. At N = 256 each subset owns only
~0.3 % of heads, so ≈ 88 % of vectors had **all** 32 candidates
rejected and were silently dropped from the index (zero replicas).
Fix: when `p_allowedHeads != nullptr`, do a brute-force scan over only
the allowed heads (≤ 430 heads per subset), then apply RNG dedup.
Per-vector cost increases by ~50 k FLOPs — negligible against SSD I/O.

### Bug 2 — Sparse-path duplicate when head VID coincides with GT

`SPANNIndex::SearchIndex` sparse-tag fast path scans postings (with
deduper) and *additionally* iterates over head samples that match the
query mask, calling `AddPoint(globalVID, distance)` for each match.
The head-iteration phase did **not** consult the deduper, so a GT
vector that happens to be a head node was inserted twice (once from
its replica posting, once from the head scan), pushing one real GT
out of the top-K. The 0.938 recall ceiling we observed on N=64 team
was 100 % explained by this — confirmed by single-query trace:

```
GT     = [359930, 389729, 331537, ..., 270610]
SPTAG  = [359930, 389729, 389729, 331537, ..., 317752]   ← 389729 is a head
```

Fix: add `if (workSpace->m_deduper.CheckAndSet(globalVID)) continue;`
before `AddPoint` in the head-iteration loop.

### Bug 3 — Routed head filter used own-tag mask instead of posting-union mask (V3 dual-mask regression)

The `feat(PerTag): V3 dual-mask HeadNodeMeta` refactor split the head
metadata into two masks:

* `HierMask` — head centroid's **own** tags only (used by
  `HeadNodeMatchesQuery` to gate top-K return of head-only ghost
  vectors).
* `PostingHierMask` — union of all member-vector tags in the head's
  posting (a safe, no-false-negative pre-filter for posting selection).

V3 updated the two new consumers it added (`HeadNodeMatchesQuery`,
`m_postingFilter`) to read the right mask, but **three older
consumers in the routed-head-bundle search were left calling
`HeadHierMaskMayIntersect`**, which still reads the now-narrowed
own-tag mask:

* `SPANNIndex.cpp::CrossSubgraphGraphSearch` — "in-filter" gate on which
  heads commit to the result heap (and thus drive posting I/O).
* `SPANNIndex.cpp` cross-edge expansion in the cross-subgraph walk.
* `SPANNIndex.cpp` routed head-bundle per-node fanout (the
  `tagAwareEnabled` filter on the `tagAwareQueryMask`).

Because real (non-ghost) heads in PerTagBKTMerge carry only **one**
own-tag, this silently turned the joint "is this head's posting
relevant to the query?" filter into "is the head centroid's own tag
equal to the query tag?". For narrow filters that drops almost every
otherwise-relevant head, and recall collapses.

Symptom (N = 4 v3, sparse off):

| Filter | Before fix       | After fix           |
|--------|------------------|---------------------|
| dept   | ❌ max R 0.887   | **R 0.998 @ np 200** |
| team   | ❌ max R 0.444   | **R 0.999 @ np 160** |

Fix: introduce `VectorIndex::HeadPostingHierMaskMayIntersect`
(reads the V3 `PostingHierMask`, fail-open for legacy / V2 indexes
that don't have it) and re-point all three SPANN routed-search call
sites to it. `HeadHierMaskMayIntersect` is kept for its current
own-tag semantics and its comment updated to spell out the contract.

## Sparse path vs dense path

`sparse_tags.bin` materializes tag → posting-id lists for tags whose
posting fanout ≤ `SPTAG_SPARSE_MAX_POSTINGS` (default 1024). At query
time those tags brute-force-scan all listed postings, **ignoring nprobe**.

For N = 64, team tags fall under the cap (fanout ≈ 430), so they take
the sparse path. Even with Bug 2 fixed, the sparse path is ~26× slower
than dense BKT (84 ms vs 3 ms at np = 8) and recall is no higher than
dense. We run the ablation below with **`SPTAG_DISABLE_SPARSE_PATH=1`**
so dense BKT handles team filters; project filters remain on the
separate `tagpure_meta.bin` fast path (which has its own dedup).

## Per-filter best operating point at R ≥ 0.95 (post-fix, dense path)

| Filter   | N = 1 (plain)¹           | N = 4 (v3)              | N = 64                  |
|----------|---------------------------|--------------------------|--------------------------|
| unfilter | R 0.961 @ np 32 → **168 QPS** | R 0.953 @ np 52 → 79 QPS  | ❌ max 0.78          |
| org      | R 0.964 @ np 40 → **133 QPS** | R 0.955 @ np 16 → **262 QPS** | R 0.973 @ np 100 → 49 QPS |
| dept     | R 0.965 @ np 64 → 86 QPS  | R 0.973 @ np 24 → **172 QPS** | R 0.952 @ np 24 → **209 QPS** |
| team     | R 0.977 @ np 128 → 56 QPS | R 0.963 @ np 32 → **126 QPS** | R 0.979 @ np 8  → **519 QPS** |
| project  | R 1.000 @ np 16 → 331 QPS | R 1.000 @ np 16 → 344 QPS | R 1.000 @ np 16 → 330 QPS |

¹ N = 1 "plain" = single subset (no partitioning), `final_ratio=0.1`,
`oversample=1.0`, `merge_group=1` — i.e. **no oversample, no merge**.
Closest setup we have to a baseline non-partitioned SPANN under the same
build code, used as a sanity floor.

Project remains R = 1.000 ≈ 330 QPS at all sizes because it routes
through `tagpure_meta.bin` (KV-backed exhaustive top-K) — independent
of partition count.

### Interpretation

Partition size **N** controls which filter level is best served:

* **N = 1 (plain)** behaves like a single-subset baseline: every level
  is reachable at R ≥ 0.95 (including team, at 56 QPS / np = 128), but
  no level is fast. It is the "balanced fallback" — useful as a sanity
  floor and as the QPS curve a partitioned build must beat.
* **N = 4** aligns with the org partition (each of the 4 subsets =
  one org, and is internally tree-aligned down to the leaf, so dept
  / team / project queries all route into exactly one subset). After
  Bug 3 is fixed, **all four ACL levels** reach R ≥ 0.95 under
  N = 4 — org is the sweet spot (262 QPS), dept and team are also
  strong (172 and 126 QPS). Unfilter (np = 52 → 79 QPS) is **slower**
  than N = 1 plain (168 QPS) — the 4-way fanout dilutes the nprobe
  budget across subsets, even though it's good for filtered queries.
* **N = 64** aligns with the team partition. dept and especially team
  become very fast (519 QPS at np = 8 for team) because each team
  query routes into ~1 subset of ~430 heads. unfilter degrades because
  the search has to cover 64 subsets in series — nprobe budget is
  spread thin, recall caps at 0.78 regardless of nprobe.

So the trade-off is monotonic in N: increasing N concentrates effort
at narrow filters at the cost of wide ones, and the inflection points
match the filter-level cardinalities (4 / 64 here).

The expected behaviour matches the design: **pick N close to the
cardinality of the filter level you care about most**. N = 4 for an
org-centric workload, N = 64 for a team-centric workload.

### Asymmetry: narrow filters can be rescued, unfilter cannot

The choice of N is **not** symmetric between "too small" and "too large":

* **N too small** — partition under-covers narrow filters. With the
  Bug 3 fix (joint posting-mask head filter), N = 4 actually does NOT
  exhibit this failure on this dataset — dept and team both clear
  R ≥ 0.95 because every level routes into exactly one subset and the
  joint filter keeps the right heads. Even if it did, every narrow
  query carries a tag and can be rescued by a tag-aware sidecar that
  bypasses partition entirely:
  - `tagpure_meta.bin` (KV-backed flat scan over all vectors with that
    tag) — independent of N. This is what keeps project at R = 1.000
    / ~330 QPS across all N.
  - `sparse_tags.bin` (per-tag posting list) — same idea, currently
    Bug 2-fixed but still inefficient.
  - **Conclusion: pick N small, and tag sidecars exist as a backstop
    if a narrow filter ever does miss.**

* **N too large (e.g. N = 64 for unfilter)** — there is **no
  sidecar to fall back to**. An unfilter query carries no tag, so
  neither `tagpure_meta.bin` nor `sparse_tags.bin` can match anything.
  The only options are:
  - accept the loss (we measured max R = 0.78 at N = 64),
  - or maintain a **second, non-partitioned global head BKT**
    alongside the partitioned one — i.e. roughly double the index
    storage and build time, equivalent to running global SPANN
    side-by-side.

* Engineering recommendation: **bias N toward the unfilter / wide-filter
  needs** (small N like 4 or 16), relying on the partition's tree
  alignment to also cover narrower filters, and use tag-aware sidecars
  to lift whichever narrow filter levels still miss. The reverse —
  picking large N for narrow filters and hoping to patch unfilter —
  has no patch available within this architecture.

## Single-query verification (n64, team = 43, 6 373 vectors)

Used to validate Bug 1 and Bug 2 fixes by direct measurement rather
than aggregate metrics (`diag_n64_team*.py`):

| Config | np | R | latency |
|--------|----|----|---------|
| Bug 2 unfixed, sparse path on  | any | 0.938 | 83 ms |
| Bug 2 fixed,   sparse path on  | any | **1.000** | 84 ms |
| Bug 2 fixed,   sparse path off | 8   | **0.979** | **3.2 ms** |
| Bug 2 fixed,   sparse path off | 32  | 0.996 | 6.4 ms |
| Bug 2 fixed,   sparse path off | 128 | 0.999 | 25 ms  |

Open question: the sparse path is still ~26× slower than dense even
after Bug 2 — it scans 430 postings exhaustively where dense BKT
reaches the same recall by probing 8 heads. Either drop the cap
(`SPTAG_SPARSE_MAX_POSTINGS=0`) at build time, gate the path by a
cost model, or remove it entirely; that decision is left for the
next iteration.

## Cross-edge ablation on unfilter (N=4, May 19)

Generated `head_cross_edges.bin` on the existing N=4 Huffman index
via `augmentheadgraph -m 10` (10 extra cross-subgraph edges per head,
27 338 heads). To test whether cross-edge can supply the unfilter
"capacity bonus" predicted by the model `unfilter_QPS ≈ narrow_QPS / N +
cross-edge bonus`, the `useCrossSubgraph` gate was relaxed under
env flag `SPTAG_CROSSEDGE_UNFILTER=1` to also fire for tag-less
queries (the in-loop tag filters inside `CrossSubgraphGraphSearch`
already self-guard when `numQueryTags == 0`). For unfilter queries
all bundle nodes are injected as candidates; for filtered queries
the existing routing applies (which under N=4 tree-aligned Huffman
still resolves to a single node, so cross-edge does not engage and
filtered numbers are unchanged by construction).

| Level | nprobe | Recall | QPS (baseline) | QPS (+cross-edge unfilter) | Δ |
|-------|-------:|-------:|---------------:|----------------------------:|---:|
| unfilter | 52 | 0.95 | **79.4** | **78.4** | ≈0 |
| org      | 16 | 0.96 | 262.1 | 252.8 | noise |
| dept     | 24 | 0.97 | 172.4 | 173.3 | noise |
| team     | 32 | 0.96 | 126.2 | 126.8 | noise |
| project  | 16 | 1.00 | 344.4 | 337.1 | noise |

**Result: cross-edge does not help unfilter at N=4.** The proposed
capacity model `unfilter ≈ narrow/N + bonus` does not hold here:
unfilter (79 QPS) is already roughly **2.5× of team/4** (31.5 QPS),
i.e. unfilter is not bottlenecked by per-node QPS multiplied by N,
it is bottlenecked by the total posting-scan budget across all four
nodes. Cross-edges change the *order* in which heads are visited
(global best-first walk vs. per-node serial fanout) but they do not
reduce the *total* posting target the SSD has to fetch — and at
N=4 the per-node BKTs are big enough that the serial fanout already
finds high-quality heads in each. The shortcut hops just substitute
for one or two BKT probes per node, which is in the same noise band.

**Implications:**

* For N=4 tree-aligned Huffman, cross-edge is not a useful lever.
  The unfilter ceiling is set by total `searchInternalResultNum`
  posting target and SSD bandwidth, not by graph topology.
* The clean capacity model is more applicable for *large* N where
  per-node BKT becomes too small to find good heads alone (cross
  edges then provide the missing global-neighbourhood signal).
  N=64 / N=256 cross-edge ablations would be a more honest test
  of the model.
* For N=4, the lever for narrow-filter QPS remains posting purity
  (small posting + no merge), which trades head count and replica
  budget for tag purity. That ablation requires a rebuild and is
  the next natural step.

Code changes (commit pending):

* `AnnService/src/Core/SPANN/SPANNIndex.cpp` — `candidateNodes`
  populated for unfilter under `SPTAG_CROSSEDGE_UNFILTER=1`;
  `useCrossSubgraph` gate no longer requires `numQueryTags > 0`.

Files:

* Sweep with cross-edge enabled: `huffman_sweeps/sweep_tenant_index_huffman_v3_rebuild.csv`
* Baseline preserved at: `huffman_sweeps/sweep_tenant_index_huffman_v3_rebuild_baseline.csv`
* Cross-edge file: `tenant_index_huffman_v3_rebuild/tenant_0/HeadIndex/head_cross_edges.bin` (HECH v1, 27 338 × 10)

## Tag-pure no-merge ablation (N=4, r=0.1, May 20)

Setting from prior section: `PerTagBKTMerge` still oversamples 3× per
tag and merges across tags. Hypothesis: that merge step costs both
tag purity (mixed tags inside a single posting) and *narrow-filter
QPS*, because narrow queries scan postings whose vectors are mostly
not for the queried tag. Direct test: rebuild with `oversample=1.0`,
`mergeAlpha=0.0`, target `finalRatio=0.1`. With α=0 no candidate
satisfies `d <= 0` in Phase 2, every Phase 3 group is a singleton,
the final head set is exactly the per-tag SelectHead output. Posting
size: ~3.24 M / 39 746 ≈ 82 vecs/posting (well below `PostingVectorLimit=118`).

Index: `tenant_index_huffman_pure_r10/` (39 746 heads, 4 Huffman
nodes, identical group-target=4 leaf-packing partition as `_v3_rebuild`).

| Level | Baseline (merge on, 27 338 heads) | No-merge r=0.1 (39 746 heads) | Δ |
|---|---:|---:|---:|
| unfilter | 79 | **129** | **+63 %** |
| org      | 262 | **408** | **+56 %** |
| dept     | 172 | **208** | **+21 %** |
| team     | 126 | **166** | **+32 %** |
| project  | 344 | 344 | ≈ noise |

All numbers @ R ≥ 0.95, post-warmup, best-of-3 trials.

Plus cross-edge unfilter on top (`SPTAG_CROSSEDGE_UNFILTER=1`,
augmentheadgraph -m 10): unfilter 129 → 130, org 408 → 439 (likely
run-to-run noise in the org column; team/dept/project stable). The
cross-edge bonus is in the same noise band as before — confirming
that at N=4 tree-aligned, cross-edges aren't the bottleneck. The
bottleneck was merge.

**Why merge hurts** (mechanism):

1. Phase 2 of `PerTagBKTMerge` greedily pairs heads from different
   tag groups whose representatives are within `α · meanNN1`. Each
   resulting merged head has a posting that draws vectors from all
   tags whose original heads were folded into it. Even with PerTag
   posting metadata (single own-tag), the posting *content* still
   leaks across tags via the replica-assignment step.
2. For narrow filters (team / project), the SSD scan reads all
   replicas of a candidate head, but only ~1/64 of them match the
   query tag. Most I/O is wasted. Filtered nprobe must compensate
   by sweeping more heads, costing latency.
3. With merge off, each head was selected from a single tag's
   sub-BKT — its replicas (8 nearest base vectors) are
   overwhelmingly same-tag. Narrow filters now hit useful data
   on the first scan, so a much smaller `nprobe` suffices.

**Capacity model revisited**: at r=0.1 / no-merge, unfilter is
129 QPS and team is 166 QPS. Predicted `unfilter ≈ narrow/N`
would give 166/4 = 41 QPS; actual is 129. So unfilter is **3×
better than the model** — meaning the system is not bottlenecked
by N-fold per-node fanout but by total useful posting reads. With
tag-pure postings the same posting budget covers far more relevant
vectors per query.

**What still costs**: project (cardinality 256) stays at 344 QPS
(same as baseline). At that selectivity the per-query work is
dominated by routing + posting fetch overhead, not by posting
content, so no-merge doesn't help. Acceptable: 344 QPS is already
near hardware ceiling on this box.

**Build cost**: ~24 min wall (vs ~21 min for v3_rebuild). Slight
increase from omitting merge dedup (more heads → bigger graph
build). Storage: similar.

### Reproduction

```bash
LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so.2 \
PYTHONPATH=/home/v-mochengli/SPTAG \
python3 /home/v-mochengli/test/build_tenant0_pertag.py \
    --index-dir /home/v-mochengli/test/tenant_index_huffman_pure_r10 \
    --team-tag-file /tmp/tenant0_grp4_tags.txt \
    --oversample 1.0 --merge-alpha 0.0 \
    --final-ratio 0.1 --group-target 4

/home/v-mochengli/SPTAG/Release/augmentheadgraph \
    -d /home/v-mochengli/test/tenant_index_huffman_pure_r10/tenant_0/HeadIndex \
    -m 10 -t 16 -w true

LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so.2 SPTAG_DISABLE_SPARSE_PATH=1 \
PYTHONPATH=/home/v-mochengli/SPTAG python3 \
    /home/v-mochengli/test/huffman_sweeps/strong_sweep_fine.py \
    /home/v-mochengli/test/tenant_index_huffman_pure_r10
```

### Files

* Sweep CSVs:
  `huffman_sweeps/sweep_tenant_index_huffman_pure_r10_nocross.csv` (no cross-edge),
  `huffman_sweeps/sweep_tenant_index_huffman_pure_r10.csv` (+cross-edge unfilter)
* Index: `tenant_index_huffman_pure_r10/` (39 746 heads, 4 Huffman nodes,
  `head_cross_edges.bin` HECH v1, 39 746 × 10)

## Files

* Indices: `tenant_index_huffman_v3` (N=4), `tenant_index_huffman_n64`
* Sweep CSVs (post-fix):
  `huffman_sweeps/sweep_tenant_index_huffman_n1_plain.csv` (N=1, no oversample, no merge),
  `huffman_sweeps/sweep_tenant_index_huffman_v3.csv` (N=4),
  `huffman_sweeps/sweep_noproj_tenant_index_huffman_n64.csv` (N=64, sparse off)
* Diagnostics:
  `huffman_sweeps/diag_one_query.py` (build-drop verification),
  `huffman_sweeps/diag_n64_team*.py` (sparse path + head-dedup verification)
* Source fixes:
  `AnnService/inc/Core/SPANN/ExtraDynamicSearcher.h` (RNGSelection per-subset scan),
  `AnnService/src/Core/SPANN/SPANNIndex.cpp` (sparse-path head-iteration deduper check)
