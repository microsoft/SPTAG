# Multi-Tenant Dual-Pool SPANN — Usage Modes

This branch (`feature/filtered-head-unfiltered-tail`) extends SPANN with a
multi-tenant, per-tag **dual-pool** head index. **All modes share one binary**
(`_SPTAG.so`); behavior is selected entirely through environment variables and
build-script arguments. Nothing below requires a different build.

The three independent switches, from outermost to innermost:

| Layer | Switch | Vanilla | Dual-pool |
| ----- | ------ | ------- | --------- |
| ① Head-selection algorithm | `SPTAG_SELECT_TYPE_OVERRIDE` | `BKT` (global, ratio-based) | `PerTagBKT` (per-tag grouped) |
| ② Subgraph count (bundles) | `--group-target N` | none (single global `m_index`) | `N` bundles + cross-edges |
| ③ U_extra augmentation | `SPTAG_DUAL_POOL_AUGMENT` | off | on (`SPTAG_DUAL_POOL_EXTRA_RATIO`) |

> **Important:** Setting only `--group-target 1` and disabling U_extra does **not**
> reproduce vanilla. The head-selection layer (①) is still `PerTagBKT`, which
> picks a different head set than vanilla's global BKT-by-ratio. True vanilla
> additionally requires `selectType = BKT` (which `build_tenant0_baseline.py` sets
> by popping `SPTAG_SELECT_TYPE_OVERRIDE`).

Runtime dispatch is fully gated, so disabling a layer is a genuine no-op:
- `m_pendingNodeHeadSelections` non-empty ⇔ bundles + cross-edges are built
  (driven by selectType + `--group-target`).
- `m_pendingHeadRoles` / `m_pendingNodeUExtraSelections` non-empty ⇔ U_extra is
  built (driven by `SPTAG_DUAL_POOL_AUGMENT`).

Common environment for every command below:

```bash
export PYTHONPATH=/home/v-mochengli/SPTAG/Release   # build; use .../SPTAG for query/sweep
export LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so.2
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2048000
PY=/home/v-mochengli/anaconda3/envs/py310/bin/python
```

---

## Mode A — Vanilla baseline (single global graph)

Default BKT head selection by ratio; no bundles, no cross-edges, no U_extra.

```bash
$PY test/build_tenant0_baseline.py \
    --index-dir test/tenant_index_t0_vanilla_r14 \
    --ratio 0.139
```

Internally: pops `SPTAG_SELECT_TYPE_OVERRIDE` (→ default `BKT`), sets
`SPTAG_RATIO_OVERRIDE=0.139`. No augment binary step is needed.

---

## Mode B — Dual-pool: per-tag bundles + cross-edges (no U_extra)

Per-tag head selection, `N` bundle subgraphs stitched by cross-edges. Used for
filter (per-tag) and unfilter (global) search.

```bash
# (do NOT set SPTAG_DUAL_POOL_AUGMENT)
$PY test/build_tenant0_pertag.py \
    --index-dir test/tenant_index_t0_dualpool_r14_nouextra \
    --final-ratio 0.139 --group-target 4 --merge-group 1

# Build cross-subgraph edges (required whenever group-target > 1):
Release/augmentheadgraph \
    -d test/tenant_index_t0_dualpool_r14_nouextra/tenant_0/HeadIndex \
    -k 15 -m 10 -t 16 -w true
```

Key knobs:
- `--group-target N` — number of bundle subgraphs (N=1 collapses to a single
  per-tag graph; cross-edges then degenerate to empty).
- `--merge-group` — greedy merge group-size cap during head selection.
- `--final-ratio` — target head ratio after merge.

---

## Mode C — Dual-pool + U_extra (unfilter augmentation, asymmetric edges)

Same as Mode B, plus a random `extra-ratio` fraction of augmentation heads
(**U_extra**) injected into the bundle subgraphs to help **unfilter** search.

```bash
export SPTAG_DUAL_POOL_AUGMENT=1
export SPTAG_DUAL_POOL_EXTRA_RATIO=0.10

$PY test/build_tenant0_pertag.py \
    --index-dir test/tenant_index_t0_dualpool_r14_approach1 \
    --final-ratio 0.139 --group-target 4 --merge-group 1

# augmentheadgraph now ALSO emits reverse H1->U_extra cross-edges automatically:
Release/augmentheadgraph \
    -d test/tenant_index_t0_dualpool_r14_approach1/tenant_0/HeadIndex \
    -k 15 -m 10 -t 16 -w true
```

### U_extra asymmetric-edge design

- **H1 → U_extra**: only via cross-edge (U_extra never enters any H1 node's RNG
  neighbor list, and never appears in the BKT tree → never a query seed).
- **U_extra → all subgraphs**: via normal edges (within host bundle) + cross-edges.

This is realized by:
- `AddIndexIdxNoBackEdge(begin,end)` — builds the new node's own out-edges via
  `RefineNode(updateNeighbors=false)`, inserting **no** back-edges into existing
  neighbors (BKT/KDT/VectorIndex).
- `buildHeadIndexFromFile(..., n_h1_split)` — builds the bundle BKT tree + RNG
  over H1 only, then appends U_extra graph-only with `AddIndexId` +
  `AddIndexIdxNoBackEdge`. Per-bundle head files are laid out `[H1...][U_extra...]`;
  manifest `headCount` stays = H1 count.
- `AugmentHeadGraph` — after the symmetric cross-edge pass, a reverse pass adds
  explicit `H1 → U_extra` cross-edges so H1 nodes in other bundles can reach
  U_extra during unfilter (filter mode never follows cross-edges → U_extra is
  unfilter-only).

### Result (SIFT-1M tenant-0, r14, 4 bundles, extra-ratio 0.10)

Bundles hold 56261 H1 + 5626 U_extra heads; 56260 reverse cross-edges added.
Unfilter sweep overlaps the no-U_extra baseline within ±0.3% recall / ±2% QPS:

| nprobe | Mode C (U_extra) R/QPS | Mode B (no U_extra) R/QPS |
| ------ | ---------------------- | ------------------------- |
| 56  | 0.9430 / 148.8 | 0.9460 / 145.0 |
| 64  | 0.9580 / 133.0 | 0.9580 / 131.9 |
| 80  | 0.9710 / 111.8 | 0.9760 / 106.5 |
| 96  | 0.9810 /  93.6 | 0.9790 /  88.6 |
| 128 | 0.9900 /  70.8 | 0.9870 /  67.1 |

**Conclusion:** the U_extra reachability mechanism is correct (structural
dead-end fixed), but **random** 10% U_extra heads carry no useful signal for the
unfilter critical path. An informed selection strategy would be needed for gains.

---

## Optional slim head store

`test/slim_head_store.py <index>/tenant_0` is an orthogonal storage optimization
that strips redundant root head vectors/graph/tree (root `vectors+graph+tree`
40.5M → 3.7M; HeadIndex dir 114M → 79M) with identical recall/QPS. Apply it after
build+augment. It touches nothing else (bundles / cross_edges / meta / manifest).

---

## Relevant environment variables (SPANNIndex)

| Variable | Effect |
| -------- | ------ |
| `SPTAG_SELECT_TYPE_OVERRIDE` | `BKT` (vanilla) / `PerTagBKT` / `DualGlobal` |
| `SPTAG_RATIO_OVERRIDE` | global head ratio (vanilla) |
| `SPTAG_PERTAG_HEAD_RATIO` | per-tag final head ratio |
| `SPTAG_DUAL_POOL_AUGMENT` | enable U_extra augmentation pool |
| `SPTAG_DUAL_POOL_EXTRA_RATIO` | U_extra fraction (e.g. 0.10) |
| `SPTAG_DISABLE_CROSS_EDGES`, `SPTAG_CROSSEDGE_UNFILTER`, `SPTAG_FILTER_KEEP_CROSS` | cross-edge search toggles |
| `SPTAG_FIXED_NPROBE`, `SPTAG_UNIFIED_NPROBE_BUDGET`, `SPTAG_MULTI_NODE_BUDGET_KEEP_RATIO` | nprobe control |
