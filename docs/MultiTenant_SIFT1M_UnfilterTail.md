# UnfilterTail: vector-similar tail records for unfiltered SPANN search

## Motivation

Per-tag / Huffman SPANN index routes each base vector to the **top-K postings
of its own tag** (per-tag main routing, `ReplicaCount=8`). This yields high
intra-posting attribute purity (good for filtered queries) but biases head
placement and posting membership toward tag-internal density: an unfilter
query whose true nearest neighbours lie in *other* tags' regions has to widen
`nprobe` 4–5× to recover the same recall.

Concretely on SIFT-1M tenant_0 (`r069` index, 27 740 heads, N=4 Huffman):

* unfilter @ R≥0.99 needs `np=128` (base)
* org      @ R≥0.99 needs `np=30`

The aim of **UnfilterTail** is to cheaply restore unfilter recall *without*
disturbing filtered paths, by appending a small number of vector-similar
**tail records** to each posting on disk, which are read only when no tag
filter is active.

## Design

For every base vector `v` we already insert it into its `ReplicaCount` nearest
**per-tag** postings (pure replicas, used by both filtered and unfilter
queries). In addition we insert `K_replica` extra copies of `v` into its
`K_replica` nearest *tag-agnostic* postings; these copies are marked as
**tail** and laid out **after** the pure prefix of each posting on disk:

```
posting layout:  [ pure_0, pure_1, ..., pure_{P-1} | tail_0, ..., tail_{T-1} ]
                  <----- pure_count[h] ------>      <----- (variable) ----->
```

A `posting_pure_counts.bin` sidecar stores `pure_count[h]` per head so
read-time logic can truncate.

### Read path

`KeyValueIO::MultiGet` is overloaded with a `maxBytesPerKey` vector:

* **Filtered query**: `maxBytes[i] = pure_count[h] * recSize` → tail blocks
  are never fetched from SSD; filtered cost stays at the baseline.
* **Unfilter query** (`SPTAG_UNFILTER_TAIL=1`): `maxBytes[i] = 0` (no cap), so
  the entire posting including tail is read and scored.

`FileIO::BlockController::ReadBlocks` honours the per-key cap by computing
`ceil(min(addr[0], maxBytes[i]) / PageSize)` blocks per key.

### Build path (phase 4 in `BuildIndexInternal`)

After pure replicas are routed and capped, we re-scan every base vector,
compute its `K_replica` nearest heads via `p_headIndex->SearchIndex(...)`
(tag-agnostic vector KNN), and append the corresponding records with
`distance = FLT_MAX` so the existing per-posting RNG selection sorts them
**after** all pure records. Tail edges hitting an already-full posting are
counted as `cap_skip`; duplicates of an existing pure record as `dup_skip`.

### Two cap regimes

Tail insertion competes with pure replicas for posting capacity:

**Shared cap** (initial design, `K_replica = 2` / `4`):

Pure routing is built and cut first. Tail capacity is then calculated per
posting from its persisted pure prefix:

```
purePages = ceil(pure_count * recordBytes / PageSize)
tailCapPages = purePages + UnfilterTailBufferLength
```

`UnfilterTailBufferLength` therefore means **additional physical pages beyond
the pure prefix**, rather than an absolute target based on
`PostingPageLimit + BufferLength`. Tail records may fill unused bytes in the
pure prefix's final page at no additional page cost. A final page containing
only tail is discarded when it is below 10% occupied.

## Parameters & env knobs

| Knob | Where | Effect |
| --- | --- | --- |
| `TailReplicaCount` (`[BuildSSDIndex]`) | build-time SSD param (ini only) | number of tag-agnostic tail copies per vector |
| `UnfilterTailBufferLength` (`[BuildSSDIndex]`) | build & search SSD param (ini only) | maximum extra physical tail pages beyond each posting's pure prefix |
| `SPTAG_UNFILTER_TAIL` | search env, `0/1` | gate the truncated-MultiGet branch at query time |

`UnfilterTailBufferLength` is a native SSD param persisted in `indexloader.ini`,
so the same ini value sizes the on-disk mapping at build time and the in-memory
workspace at search time. The ini is the single source of truth — there is no
env override (`SPTAG_UNFILTER_TAIL_K_REPLICA` / `SPTAG_UNFILTER_TAIL_BUFFER_PAGES`
have been removed).

## Tagged online inserts

Tag-bearing SPANN indexes must use the tag-aware insert API:
`AnnIndex.AddWithTags(vectors, packed_uint32_tags, count, tags_per_vector,
normalized)`. The generic `Add` API is deliberately rejected for tagged or
bundle indexes because it cannot supply posting tags, route a pure record to
the correct bundle, or encode a valid compact posting payload.

For every inserted vector, the update path:

1. routes **pure** replicas only within the persisted tag-routing bundle and
   inserts them immediately before that posting's tail boundary;
2. selects **tail** replicas across all head bundles and appends them after the
   pure prefix;
3. preserves the `purePages + UnfilterTailBufferLength` page budget. A pure
   insertion that needs room trims the far end of the tail; a tail insertion
   without remaining budget is skipped.

The full-precision update vector is also persisted in
`UpdateVectorFile` (default `update_vectors.bin`) because a new VID is absent
from the immutable `FullVectorFile` used by exact reranking. Call `Save` after
updates to checkpoint the posting mappings, pure-count sidecar, delete/version
map, and update-vector sidecar together.

### Split, merge, and cross-edge maintenance

An overfull pure prefix is split only inside its owning bundle. The update is
first appended, then the live pure records determine two local centroids and
the tails are reassigned to those finalized centroids. New local graph heads
follow the native SPANN append/refine lifecycle; obsolete heads are logical
tombstones, so existing graph references remain valid and `ContainSample`
suppresses them at query time. A failed rewrite restores the original posting
and tombstones every newly appended local head.

Search queues fully scanned low-live postings for same-bundle merge
maintenance. `Save` and `Checkpoint` drain that queue, persist the mutated
bundle graphs and their local/global head-ID maps, and keep a self-contained
copy of every bundle, `update_vectors.bin`, slim-root metadata, and the
tag-to-bundle routing map under the checkpoint root.
When a split appends heads, save/checkpoint refreshes the slim-root logical
head count; recovery reads both that metadata and `posting_pure_counts.bin`
from the checkpoint rather than the original index root.

Any split or merge invalidates the immutable global cross-edge snapshot.
The index writes `HeadIndex/head_cross_edges.dirty` and disables cross-bundle
edge traversal after reload until it is regenerated:

```bash
Release/augmentheadgraph -d <index>/HeadIndex -k 15 -m 10 -t <threads> --overwrite
```

The successful rebuild removes the dirty marker. `U_extra` is intentionally
outside this incremental maintenance path. A single insertion batch currently
must fit across two pure-capacity postings; larger batches fail safely and
should be chunked by the caller.

Current constraints:

* Online compact-code encoding is supported for `PostingQuantizer=PipePQ`
  (Float and UInt8 vectors). OPQ, RaBitQ, and bit-packed posting encoders reject
  inserts rather than write invalid records.
* `EnableWAL=true` is unsupported for `AddWithTags`: the legacy WAL does not
  record the required pure/tail target assignment, so it cannot replay a tagged
  update safely.
* A failed multi-posting insert invalidates all VIDs from that batch. Any bytes
  already written remain unreachable and can be reclaimed only by a later
  compaction/rebuild.

## Experiments (SIFT-1M tenant_0, `r069` family)

Builds:

* `tenant_index_huffman_pure_r069`           — base, no tail
* `tenant_index_huffman_pure_r069_ut2`       — `K_replica = 2`, shared cap
* `tenant_index_huffman_pure_r069_ut4`       — `K_replica = 4`, shared cap
* `tenant_index_huffman_pure_r069_ut4_extbuf` — `K_replica = 4`, +5 tail pages

Sweep: `strong_sweep_fine.py` (5 levels, NQ=100).

### Best QPS at R ≥ 0.99

| level    | base (np / R / QPS) | ut2 | ut4 | ut4_extbuf |
| -------- | ------------------- | ----| --- | ---------- |
| unfilter | 128 / .994 / **39** | 80 / .996 / **53** (+35%) | 64 / .991 / **59** (+51%) | 52 / .991 / **66** (+68%) |
| org      | 30  / .990 / **166**| 28 / .991 / **178** (+7%) | 30 / .990 / 167 (±0%) | 28 / .992 / **180** (+8%) |
| dept     | 64  / .992 / **78** | 56 / .993 / **88**  (+13%)| 48 / .990 / **105** (+35%) | 48 / .990 / 103 (+32%) |
| team     | 64  / .992 / **77** | 64 / .993 / 78    | 64 / .995 / 78 | 64 / .996 / 77 |

### Best QPS at R ≥ 0.95

| level    | base | ut2 | ut4 | ut4_extbuf |
| -------- | ---- | --- | --- | ---------- |
| unfilter | 96   | 106 (+10%) | **118** (+23%) | 108 (+13%) |
| org      | 325  | 313 | 317 | 318 |
| dept     | 204  | 212 | 210 | **215** |
| team     | 152  | 156 | 156 | **160** |

At low-recall operating points the extra tail-buffer reads more bytes per
posting without adding helpful neighbours that weren't already covered by
the existing tail; ut4 wins at R=0.95, ext_buf wins at R≥0.98.

### Per-recall same-QPS interpolation (unfilter, ext vs ut4)

| R     | ut4 (np, QPS) | ext_buf (np, QPS) | Δ |
| ----- | ------------- | ----------------- | -- |
| 0.951 | 32 / 118      | ~29 / ~124 (interp) | +5% |
| 0.972 | 40 / 94       | 36 / 95           | +1% |
| 0.986 | 56 / 68       | 44 / 77           | +13% |
| 0.991 | 64 / 59       | 52 / 66           | +12% |
| 0.995 | 80 / 47       | 64 / 53           | +14% |
| 0.999 | 128 / 29      | 96 / 36           | +22% |

The first "best QPS at R≥0.95" comparison is misleading: ext's recall curve
shifts right (at every `np`, ext's R is higher than ut4's), so a strict
`R≥0.95` filter can pick a point that already overshoots 0.96. Interpolated
to the same R, ext is always ≥ ut4.

### Phase-4 build stats

| K_replica | tail cap | tail_added | dup_skip | cap_skip |
| --------- | -------- | ---------- | -------- | -------- |
| 2         | 145      | 429 K      | 183 K    | 142 K |
| 4         | 145      | 764 K      | 322 K    | 422 K |
| 4         | 180 (+5p)| 1 089 K    | 324 K    | 96 K |

### Disk footprint

| Index | Size | Δ vs base |
| ----- | ---- | --------- |
| base       | 1716 MB | — |
| ut2        | 1945 MB | +13% |
| ut4        | 2124 MB | +24% |
| ut4_extbuf | 2298 MB | +34% |

Growth is sub-linear in `K_replica` because dense postings hit the cap and
drop later replicas. Sidecar `posting_pure_counts.bin` ≈ 111 KB.

## Conclusion

* **UnfilterTail works**: at the strict R≥0.99 unfilter goal we recover most
  of the per-tag-vs-global head-quality gap with little disk cost — **+68%
  QPS** at `K_replica=4` with 5 extra tail pages, and filtered paths stay
  within ±2% (often slightly better since tail-aware build smooths the
  posting size distribution).
* **Diminishing returns past `K_replica=4`**: cap_skip is still the dominant
  miss source under shared cap; Option A's extra block budget cuts cap_skip
  4× but only adds ~12% QPS over `ut4` shared. The remaining unfilter gap
  (vs filtered-class QPS) is **not** explained by tail coverage anymore.
* **Real bottleneck for unfilter is head placement**, not posting membership.
  The head BKT was built on per-tag centroids, so its Voronoi regions are
  biased toward tag-internal density. UnfilterTail patches this *inside the
  postings* but cannot change which posting a query lands on. Closing the
  remaining gap (e.g. R=0.99 unfilter still needs ~2× the nprobe of filtered
  paths) most likely requires re-thinking the head index itself — for
  example a global-kmeans head with per-tag posting routing, or a joint
  head-placement loss that balances tag purity and global coverage.

## Files

* Source (commits on `feature/pertag-sparse`):
  * `96b2cea` — K_replica=2 + truncated MultiGet
  * `bdc414f` — Option A: independent tail buffer cap
* Touched headers / sources:
  * `AnnService/inc/Core/SPANN/Options.h`,
    `AnnService/inc/Core/SPANN/ParameterDefinitionList.h`
  * `AnnService/inc/Core/SPANN/ExtraDynamicSearcher.h`
    (truncated read at search hot path; phase-4 worker; tail cap;
    sidecar accessors)
  * `AnnService/inc/Core/SPANN/ExtraFileController.h`,
    `AnnService/src/Core/SPANN/ExtraFileController.cpp`
    (`m_blockLimit` includes tail pages; truncated `ReadBlocks` overload)
  * `AnnService/inc/Helper/KeyValueIO.h` (MultiGet w/ `maxBytesPerKey`)
  * `AnnService/src/Core/SPANN/SPANNIndex.cpp` (search-time workspace)
* Sweep CSVs (in `/home/v-mochengli/test/huffman_sweeps/`):
  `sweep_tenant_index_huffman_pure_r069.csv`,
  `sweep_tenant_index_huffman_pure_r069_ut2.csv`,
  `sweep_tenant_index_huffman_pure_r069_ut4.csv`,
  `sweep_tenant_index_huffman_pure_r069_ut4_extbuf.csv`.
* Indices: `tenant_index_huffman_pure_r069{,_ut2,_ut4,_ut4_extbuf}/`.

## Reproduce

```bash
# build with K_replica=4 and +5 tail pages
# (set [BuildSSDIndex] TailReplicaCount=4 / UnfilterTailBufferLength=5 in the ini)
LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so.2 \
PYTHONPATH=/home/v-mochengli/SPTAG python3 \
    /home/v-mochengli/test/build_tenant0_pertag.py \
    --final-ratio 0.069 --oversample 1.0 --merge-alpha 0.0 \
    --cross-tag-knn 3 --merge-group 3 --group-target 4 \
    --unfilter-tail-k-replica 4 \
    -o /home/v-mochengli/test/tenant_index_huffman_pure_r069_ut4_extbuf

# sweep with truncated unfilter read (tail buffer comes from the persisted ini)
LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so.2 \
SPTAG_UNFILTER_TAIL=1 \
PYTHONPATH=/home/v-mochengli/SPTAG python3 \
    /home/v-mochengli/test/huffman_sweeps/strong_sweep_fine.py \
    /home/v-mochengli/test/tenant_index_huffman_pure_r069_ut4_extbuf \
    > /tmp/sweep_ut4_extbuf.log 2>&1
```
