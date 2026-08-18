# Benchmark Scripts

## STM1 Static Metadata Demo

The native INI is the source of truth for build and search settings. The
committed Float SIFT-1M fixture demonstrates node-pure STM1 postings, unbounded
unfilter tails, and the exact member-OR posting prefilter:

```bash
cmake --build build --target spannbuilder spannaclbench -j

CFG=Tools/benchmarks/build_spann_attr_sift1m_tagged_4node_static_fullfloat_tail_unbounded_ordered_page.ini
Tools/benchmarks/run_spann_attr_build.sh "$CFG"

IDX=/datadisk/yfcc_fast/sptag_sift1m_tagged_vs_upstream/index_tagged_4node_static_fullfloat_tail_unbounded_ordered_page
QDIR=/home/v-mochengli/datasets/sift1m/multitenant/query

Release/spannaclbench \
  --index "$IDX" \
  --queries "$QDIR/query_vectors.npy" \
  --truth "$QDIR/groundtruth_project_local_ids.npy" \
  --query-tags "$QDIR/query_tags.npy" \
  --tag-column 3 \
  --warmup 200 --max-queries 1000
```

`[SearchSSDIndex]` in the INI controls the persisted search behavior:
`InternalResultNum`, `MaxCheck`, `EnableUnfilterTail`, and
`EnableHierPostingFilter`. Do not override these with `SPTAG_*` environment
variables. The JSON output includes recall/QPS and loaded-posting contribution
metrics when `CollectPostingContributionStats=true` is enabled in a diagnostic
search overlay.

For a reload-only sweep, pass a separate native runtime overlay instead of
modifying the persisted index or using environment variables:

```bash
Release/spannaclbench ... \
  --search-ini Tools/benchmarks/search_turbopuffer_sift1m_tenant0_n20.ini
```

## BKT-Head Hybrid Distance Routing

`build_spann_attr_sift1m_global_static_hybrid_distance.ini` is the hybrid-on
experiment; `build_spann_attr_sift1m_global_static_bkt_control.ini` is its
matched hybrid-off control. Both retain the canonical SIFT1M BKT head
selection and degree-32 vector graph. Hybrid mode appends degree-16
hybrid-distance edges through the standard `head_cross_edges.bin` runtime
suffix; its marked version-2 extension binds the suffix to both the build
generation and deterministic hybrid content, including the ordered serialized
edge body. It does not create a second graph store or attribute subset. Its
sole STM1 posting is:

```text
H | (O \ H)
```

Here `H` is the hybrid-distance pure prefix and `O` is the complete original
vector-distance pure+tail posting. The suffix preserves every record in `O`
that is absent from `H`, sorted by vector distance:

```text
D_hybrid = w_v D_vector
         + sum_i w_cat,i [query_i != head_i]
         + sum_j w_num,j |query_j - head_j|
```

Unfiltered queries always navigate the original degree-32 graph and scan the
full contiguous pure+tail posting. Before filtered graph search, the router
computes pure-vector and predicate distances to a deterministic sample of head
vectors. It chooses hybrid navigation plus the pure prefix only when the
predicate is selective and its attribute penalty is large relative to the
near-sample vector-distance span:

```text
Phi(q, P) = RMS_i(D_attribute(P, h_i))
            / (Q90(D_vector(q, h_near)) - Q10(D_vector(q, h_near)) + epsilon)

Hybrid iff selectivity(P) <= HybridRouteSelectivityThreshold
           and Phi(q, P) >= HybridRouteDeformationThreshold
```

`HybridRouteSampleCount` defaults to 64 and performs no graph traversal.
The selected graph runs exactly once, and nprobe changes only the operating
point on that route. No second posting, overlap bitmap, sparse exact-set
reconstruction, or single-attribute partition exists. Exact record-level flat
or DNF filtering remains authoritative on either route. All routing thresholds
and distance weights are native INI parameters.

## Limited-Tag Static Postings

`build_spann_attr_sift1m_zipf200_limited_tag.ini` builds the single-attribute
limited-tag experiment. It keeps one canonical BKT head graph and does not
create attribute subsets, hybrid edges, or cross edges. Native
`[BuildSSDIndex] LimitedTagSlotsPerHead` accepts `2` (the default) or `4`.
Each head persists exactly that many support values in generation-bound
`limited_tag_support.bin`:

```text
support[head][0] = source vector attribute
support[head][1..N-1] = top-(N-1) external attributes, N in {2, 4}
```

The canonical INI sets `LimitedTagSlotsPerHead=2` explicitly. Use
`build_spann_attr_sift1m_zipf200_limited_tag4.ini` for the isolated four-slot
experiment.

Non-head vectors are assigned only to heads supporting their attribute, using
the normal BKT candidate search and RNG pruning for up to eight replicas. The
single STM1 posting remains `H | (O \ H)`: filtered queries scan `H`, while
unfiltered queries navigate the original graph and scan the complete union.
At load, the validated support sidecar builds only a compact tag-to-head lookup.
If a predicate matches no more heads than native `MaxCheck`, search computes
those head distances exactly; broader predicates use result admission on the
global BKT. This lookup is not a second graph or per-tag ANN index.

Generate the reproducible Zipf-200 attribute and build with:

```bash
python3 Tools/benchmarks/generate_sift1m_zipf_attribute.py \
  --output-dir /datadisk/yfcc_fast/sptag_sift1m_zipf200
Tools/benchmarks/run_spann_attr_build.sh \
  Tools/benchmarks/build_spann_attr_sift1m_zipf200_limited_tag.ini
```

## SIFT1B Raw STM1 Recommendation

The SIFT1B recommendation uses raw UInt8 vectors in postings, four ACL
bundles, cross edges, distance-ordered pure/tail segments, and U_extra
disabled. Pure postings have a six-page budget and tail replicas have the
same six-page budget beyond the pure prefix. Attribute tuple reordering is
not enabled. It retains SIFT1B's documented BKT construction and search
defaults rather than copying SIFT1M-scale build or search budgets.

```bash
CFG=Tools/benchmarks/build_spann_attr_sift1b_raw_static_tail_capped_distance_order.ini
Tools/benchmarks/run_spann_attr_build.sh "$CFG"
```

It consumes `sift1b_tags5.u32` and the group-tag routing file, but no quantizer
codes, quantizer pivots, `FullVectorFile`, or rerank source. Its 152-byte STM1
records still require multi-terabyte final storage.

`build_spann_attr_sift1b_raw_static_tail_unbounded_distance_order.ini` is
retained only to reproduce the existing unbounded-tail diagnostic artifact.
`build_spann_attr_sift1b_pipepq32_r010_tail_unbounded.ini` remains available
only as a quantized PipePQ32 comparison control; it is not the raw-vector
recommendation.

## Ordered ACL Page Starts for Static STM1

`EnableOrderedPageStart=true` together with `OrderedPageStartAttrs` enables
sparse static reads for ordered hierarchy filters. It sorts each STM1 pure
posting prefix by the hierarchy tuple and persists `ordered_page_starts.bin`:
one `int32` page-start signature ordinal per configured attribute per posting
page.

```ini
[BuildSSDIndex]
Storage=STATIC
EnableOrderedPageStart=true
OrderedPageStartAttrs=2,3
```

For the SIFT hierarchy, columns `2,3` are team and project. The directory is
used only for a single-clause DNF AND query containing a categorical equality
on team or project; project takes precedence when both are present. Unfilter,
flat ACL queries, multi-clause DNF, and unordered facets retain the normal
full-posting path. The configured attributes must remain globally monotonic
after ACL tuple sorting; the builder rejects an incompatible schema rather than
allowing a range lookup to drop matches.

For the distance-order path, set `EnableOrderedPageStart=false`. The builder
does not apply the attribute tuple sort: pure records retain the selection
order `(head distance, VID)`, while tail records retain their separate
`(head distance, VID)` order. It removes any stale
`ordered_page_starts.bin`, and the query path cannot perform ordered page
pruning. This is the canonical SIFT1B recommendation; ordered page starts
remain an optional sparse-filter experiment.

`UnfilterPureDistanceScanPercent` can benchmark computation reduction on this
distance-ordered layout. Values below `100` retain the nearest pure prefix and
the complete tail suffix. The runtime rejects this setting on attribute-ordered
snapshots and when bounded-tail page controls are active.

`build_spann_attr_sift1m_tagged_4node_static_fullfloat_tail_unbounded.ini`
is the matching SIFT1M no-order control; it explicitly sets this parameter to
`false`.

The native benchmark can issue this DNF form directly:

```bash
Release/spannaclbench \
  --index /path/to/index \
  --queries "$QDIR/query_vectors.npy" \
  --truth "$QDIR/groundtruth_project_local_ids.npy" \
  --query-tags "$QDIR/query_tags.npy" \
  --dnf-and-cols 2,3 \
  --warmup 200 --max-queries 1000
```

## Multi-Tenant Tag Cache Stress

Files:

- `multitenant_tag_cache_stress.py`: benchmark logic, exact recall computation, result summarization.
- `run_multitenant_tag_cache_stress.sh`: reproducible runner with fixed defaults and env-based overrides.

Default workload:

- `1000` queries split into `10` batches of `100`
- sequential workload: tenants `0 -> 9`, one tenant per batch
- random workload: tenants mixed within each batch
- single-tag filter per query, sampled from the tenant's true tag distribution
- `topk=10`
- `seed=20260413`
- cache limit policy: `max(2 * largest HeadIndex, total HeadIndex / 4)` rounded up to MB

Default search params:

- `ForceDenseTagSearch=false`
- `DirectSparseMaxPostings=320`
- `FilteredSearchNprobeSafety=1.0`
- `FilteredSearchTargetRecall=1.0`
- `FilteredSearchCoverageExponent=0.5`

Run with defaults:

```bash
bash Tools/benchmarks/run_multitenant_tag_cache_stress.sh
```

Run a small smoke test:

```bash
SPTAG_STRESS_NUM_QUERIES=20 \
SPTAG_STRESS_BATCH_SIZE=10 \
SPTAG_STRESS_TENANT_RANGE=0,1 \
bash Tools/benchmarks/run_multitenant_tag_cache_stress.sh
```

Run a small RSS high-water sweep relative to the benchmark process baseline RSS:

```bash
SPTAG_STRESS_NUM_QUERIES=20 \
SPTAG_STRESS_BATCH_SIZE=10 \
SPTAG_STRESS_TENANT_RANGE=0,1 \
SPTAG_STRESS_RSS_HIGH_WATER_SWEEP_MB=off,+64,+128 \
bash Tools/benchmarks/run_multitenant_tag_cache_stress.sh
```

Run with an absolute RSS high-water cap:

```bash
python Tools/benchmarks/multitenant_tag_cache_stress.py \
	--rss-high-water-mb 2048
```

Useful environment overrides for the runner:

- `SPTAG_STRESS_SCENARIO_FILE`
- `SPTAG_STRESS_QUERY_FILE`
- `SPTAG_STRESS_OUTPUT_ROOT`
- `SPTAG_STRESS_NUM_QUERIES`
- `SPTAG_STRESS_BATCH_SIZE`
- `SPTAG_STRESS_TOPK`
- `SPTAG_STRESS_TENANT_RANGE`
- `SPTAG_STRESS_SEED`
- `SPTAG_STRESS_CACHE_LIMIT_MB`
- `SPTAG_STRESS_RSS_HIGH_WATER_MB`
- `SPTAG_STRESS_RSS_HIGH_WATER_SWEEP_MB`
- `SPTAG_STRESS_DROP_PAGE_CACHE_ON_EVICT`
- `SPTAG_STRESS_FORCE_DENSE_TAG_SEARCH`
- `SPTAG_STRESS_DIRECT_SPARSE_MAX_POSTINGS`
- `SPTAG_STRESS_FILTERED_SEARCH_NPROBE_SAFETY`
- `SPTAG_STRESS_FILTERED_SEARCH_TARGET_RECALL`
- `SPTAG_STRESS_FILTERED_SEARCH_COVERAGE_EXPONENT`
- `SPTAG_STRESS_PYTHON`
- `SPTAG_STRESS_LD_PRELOAD`

Artifacts written per run:

- `benchmark.log`: full stdout/stderr
- `status.txt`: `running`, `success`, or `failed`
- `meta.txt`: human-readable run configuration
- `meta.json`: structured run metadata
- `summary.json`: machine-readable result summary
- `summary.md`: human-readable summary table
- `batch_summary.csv`: batch-level metrics

Artifacts written for RSS sweep mode:

- root `summary.json` / `summary.md`: aggregated per-budget summary
- root `budget_summary.csv`: one row per `(rss_budget, scenario)`
- one child directory per RSS budget, each containing the normal per-run artifacts above

Notes:

- The benchmark uses exact recall computed from the base vectors referenced by the scenario file.
- The runner records seed, search parameters, git commit, and runtime environment so the workload is reproducible.
- Latency is statistically reproducible, not bitwise identical, because OS scheduling and file cache state can vary.
- `--rss-high-water-mb` accepts `off`, an absolute MB value like `1024`, or a relative headroom like `+128` measured above the benchmark process RSS right before workloads start.
- `--rss-high-water-sweep-mb` accepts a comma-separated list in the same format and runs each budget in a fresh child process so process-level RSS measurements do not drift across sweep points.