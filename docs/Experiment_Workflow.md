# Standard Experiment Workflow — Attribute-aware SPANN (multi-tenant)

End-to-end reference for running a filtered/unfiltered ANN experiment on the
attribute-aware (multi-tenant) SPANN index. The worked example is **SPACEV-1B**
(1 000 000 000 × 100 int8); the same five stages apply to SIFT-1M / YFCC-10M by
swapping the dataset paths.

```
(1) generate attributes  ->  (2) derived builder inputs  ->  (3) groundtruth
                                                                    |
(5) query / benchmark  <-  (4) build index  <------------------------+
```

All scripts below are committed under `Tools/benchmarks/`. Every billion-scale
build knob lives in the native `.ini` (see **AGENTS.md → "Build Config — Native
`.ini`"**); the launchers carry only process-loader env + the post-build
cross-graph step.

Conventions used in the commands:

```bash
DS=/path/to/MSSPACEV1B                 # dataset root: spacev1b_base.i8bin + query.i8bin
OUT=/datadisk/yfcc_fast/spacev1b_build # derived builder inputs (tags5, opq codes)
IDX=/datadisk/yfcc_fast/spacev1b_opq25 # IndexDirectory from the .ini ([Base] IndexDirectory)
REL=$PWD/Release                       # built binaries + SPTAG.py python binding
```

---

## (1) Generate attributes  —  `gen_spacev_attrs.py`

Synthesizes the per-vector attributes in the same layout the SPANN build/search
trust (single tenant 0 = all vectors):

| File (under `$DS/multitenant/`) | Shape / dtype | Meaning |
| --- | --- | --- |
| `tags.npy` | `(N,4)` uint32 | ACL 4-level hierarchy `[org,dept,team,project]`, globally-unique ids, perfect 4-ary tree (card `[4,16,64,256]`) |
| `num_attr.npy` | `(N,)` int32 | numeric **price** in `[0,100000)`, range predicate `price < X` |
| `tenant_ids.npy` | `(N,)` int32 | all 0 (single tenant) |
| `query/query_tags.npy`, `query/query_vectors.npy`, `query/query_tenant_ids.npy` | per-query | one random ACL path + the query vectors |
| `tenant_tag_scenario.json` | — | describes both attributes + the numeric selectivity sweep grid |

```bash
SPACEV1B_ROOT=$DS python3 Tools/benchmarks/gen_spacev_attrs.py
# (ROOT is also accepted as argv[1]: python3 gen_spacev_attrs.py $DS)
```

Deterministic (`SEED=20260615`). The ACL leaf is drawn uniformly per vector and
the team/dept/org columns are derived by nesting, so the four columns are always
mutually consistent.

---

## (2) Derived builder inputs  —  `prep_spacev1b_inputs.sh`

Pure-C++ (no Python, no generic quantizer) prep of the three sidecars the build
consumes, via `spannbuilder` subcommands that mirror `Quantizer/main.cpp` so the
artifacts are byte-exact with the in-posting convention:

| Output (under `$OUT`) | Built by | Meaning |
| --- | --- | --- |
| `spacev1b_tags5.u32` `(N,5)` uint32 | `--merge-tags5` | `[org,dept,team,project \| price]` — interleaves `tags.npy` + `num_attr.npy` |
| `spacev1b_group_tags.txt` | `--merge-tags5` | ACL col 0 (org), one int/line — the PerTagBKT routing key |
| `opq_codes_m25.bin` `(N,25)` uint8 | `--gen-opq-codes` | raw OPQ codes (raw-widen, ADC=false, header-less) — **not** the normalizing `Release/quantizer` |
| `opq_quantizer.bin` | copied | the OPQ codebook (search-time ADC) |

```bash
Tools/benchmarks/prep_spacev1b_inputs.sh        # full 1B
Tools/benchmarks/prep_spacev1b_inputs.sh 2000000  # smoke subset (first N vectors)
```

> The OPQ codebook is trained once on a small subset (3M) and reused; RaBitQ code
> sidecars are produced instead by `Release/rabitq2_encode_stream` (value-type
> aware, scales to 1B). Pick OPQ **or** RaBitQ in the `.ini`'s `[BuildSSDIndex]`.

---

## (3) Build groundtruth  —  `generate_query_tenant_tag_groundtruth.py`

Computes the **exact** top-k for every query, five ways (matmul-batched on the
tenant-0 base), and writes them next to the query vectors:

* `groundtruth_unfilter_local_ids.npy` — all tenant-0 vectors
* `groundtruth_{org,dept,team,project}_local_ids.npy` — vectors whose ACL tag at
  that level matches the query (the filtered cases)

Neighbor ids are tenant-0 **local** row indices (`groundtruth_local_ids`
convention). `--metric` MUST match the index build (`l2` for SPACEV/SIFT int8).

```bash
python3 Tools/benchmarks/generate_query_tenant_tag_groundtruth.py \
  --scenario-file $DS/multitenant/tenant_tag_scenario.json \
  --query-file    $DS/query.i8bin \
  --output-dir    $DS/multitenant/query \
  --topk 10 --metric l2
```

> **Scale caveat:** exact GT is `O(Nq × N)`. For 1B base this is GPU/large-RAM
> territory; run it on a subset of the base (or a GPU brute-force) when full-scale
> exact GT is infeasible, and report recall against that. The ACL-level GTs are
> cheap (they filter the base first). The numeric `price < X` predicate is
> described in `tenant_tag_scenario.json` (`sweep` grid) for selectivity studies.

---

## (4) Build index  —  `run_spann_attr_build.sh`

Thin launcher over the native `.ini`. It derives all paths FROM the `.ini` via
`sed`, runs `spannbuilder -c <config>`, then (gated by `[MultiTenant] CrossEdges`)
runs the post-build `augmentheadgraph` cross-graph step and copies the OPQ
codebook into `tenant_0/`.

```bash
Tools/benchmarks/run_spann_attr_build.sh Tools/benchmarks/build_spann_attr_spacev1b_opq25.ini
#   internally: Release/spannbuilder -c <ini>
#             + Release/augmentheadgraph -d $IDX/tenant_0/HeadIndex -k 15 -m N -t T -w true
```

Build phases in the log: `PerTagBKT` head selection → `DualPoolAugment` (U_extra)
→ `Begin Build Head` (BKT + RNG graph over the heads) → `BuildSSDIndex` (slim
in-posting postings) → in-place `SaveAll`. For the three **unfilter-enhancement
layers** (cross-graph / U_extra / unfilter-tail) — which must be enabled together
or unfilter degrades to a per-node fan-out — see **AGENTS.md → "Unfilter
Enhancement Pipeline"**. Billion-scale knobs (resume checkpoint, pinned BKT
balance factor, in-place build, slim SSD block-pool sizing) are documented in
**AGENTS.md → "Billion-scale build options"**.

The 3M-scale sibling config is `Script_AE/iniFile/build_spann_attr_spacev_opq25.ini`.

---

## (5) Query / benchmark  —  `compare_uextra_unfilter.py` / `compare_uextra_filtered.py`

Load the tenant-0 index through the Python binding and measure **recall@k + QPS**
against the stage-(3) groundtruth. `nprobe` is fixed per process via the
`SPTAG_FIXED_NPROBE` env var (the SPANN core reads it once), so sweep it by
re-invoking. Each run prints one `RESULT {…}` JSON line.

```bash
export PYTHONPATH=$REL        # exposes SPTAG.py
QDIR=$DS/multitenant/query

# unfilter (zero-tag SearchWithACL), recall@10 vs groundtruth_unfilter:
SPTAG_FIXED_NPROBE=80 python3 Tools/benchmarks/compare_uextra_unfilter.py \
  --index-dir $IDX/tenant_0 --query-dir $QDIR --topk 10 --warmup 200

# filtered (per-query tag at one ACL level) vs groundtruth_<level>:
SPTAG_FIXED_NPROBE=180 python3 Tools/benchmarks/compare_uextra_filtered.py \
  --index-dir $IDX/tenant_0 --query-dir $QDIR --level org --topk 10
#   --level unfilter|org|dept|team|project
```

Search-time feature toggles (in-posting RaBitQ/OPQ + deep-queue libaio rerank,
unfilter-tail, cross-edge switches) are env-gated — see **AGENTS.md →
"In-posting Quantization + Deep-queue Rerank"** and the env reference there.
A typical nprobe sweep at iso-recall (e.g. recall@10 = 0.95) is the standard
operating point for QPS comparisons.

---

### Quick checklist

| Stage | Script | Key output |
| --- | --- | --- |
| 1. attributes | `gen_spacev_attrs.py` | `tags.npy`, `num_attr.npy`, `query/` |
| 2. builder inputs | `prep_spacev1b_inputs.sh` | `*_tags5.u32`, `*_group_tags.txt`, `opq_codes_m25.bin` |
| 3. groundtruth | `generate_query_tenant_tag_groundtruth.py` | `groundtruth_{unfilter,org,dept,team,project}_local_ids.npy` |
| 4. build | `run_spann_attr_build.sh <ini>` | `$IDX/tenant_0/` (HeadIndex + SSD postings + cross-edges) |
| 5. query | `compare_uextra_{unfilter,filtered}.py` | `RESULT {recall@k, qps, latency}` |
