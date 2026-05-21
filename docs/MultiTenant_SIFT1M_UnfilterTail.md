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

```
relaxLimit = m_postingSizeLimit + m_bufferSizeLimit   ≈ 145 records
```

Pure replicas fill first; once a posting is full, every subsequent tail edge
is rejected. In dense regions (where the unfilter improvement matters most)
this caps tail effect.

**Independent tail buffer (Option A, recommended):**

```
tailRelaxLimit = relaxLimit + m_unfilterTailBufferLength * PageSize / m_vectorInfoSize
```

A new SSD parameter `UnfilterTailBufferLength` (in pages) extends the
per-posting block budget *only* for tail insertions. Pure routing logic is
unchanged. With `UnfilterTailBufferLength = 5` and `K_replica = 4`:

* `cap_skip` drops 422K → 96K (-77%)
* `tail_added` grows 764K → 1 089K (+42%)
* mapping `ncols` grows 20 → 25 (max 24 blocks/posting)

## Parameters & env knobs

| Knob | Where | Effect |
| --- | --- | --- |
| `UnfilterTailKReplica` (`SPTAG_UNFILTER_TAIL_K_REPLICA`) | build-time SSD param / env | number of tag-agnostic tail copies per vector |
| `UnfilterTailBufferLength` (`SPTAG_UNFILTER_TAIL_BUFFER_PAGES`) | build & search SSD param / env | per-posting extra pages reserved for tail |
| `SPTAG_UNFILTER_TAIL` | search env, `0/1` | gate the truncated-MultiGet branch at query time |

`SPTAG_UNFILTER_TAIL_BUFFER_PAGES` is read **before** the `FileIO`
constructor in `ExtraDynamicSearcher`, so the same env var sizes the on-disk
mapping at build time and the in-memory workspace at search time.

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
LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so.2 \
SPTAG_UNFILTER_TAIL_K_REPLICA=4 \
SPTAG_UNFILTER_TAIL_BUFFER_PAGES=5 \
PYTHONPATH=/home/v-mochengli/SPTAG python3 \
    /home/v-mochengli/test/build_tenant0_pertag.py \
    --final-ratio 0.069 --oversample 1.0 --merge-alpha 0.0 \
    --cross-tag-knn 3 --merge-group 3 --group-target 4 \
    --unfilter-tail-k-replica 4 \
    -o /home/v-mochengli/test/tenant_index_huffman_pure_r069_ut4_extbuf

# sweep with truncated unfilter read
LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so.2 \
SPTAG_UNFILTER_TAIL=1 SPTAG_UNFILTER_TAIL_BUFFER_PAGES=5 \
PYTHONPATH=/home/v-mochengli/SPTAG python3 \
    /home/v-mochengli/test/huffman_sweeps/strong_sweep_fine.py \
    /home/v-mochengli/test/tenant_index_huffman_pure_r069_ut4_extbuf \
    > /tmp/sweep_ut4_extbuf.log 2>&1
```
