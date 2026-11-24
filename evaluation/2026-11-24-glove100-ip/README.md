# GloVe-100: Float32, RaBitQ4 and OPQ50 InnerProduct

[Measured results](Report.md) | [Runner](reproduce.py)

The Float32 and RaBitQ4 experiments use the native code at
`c1c8f3970188a8725c08521dc776b8ca96f6bdbb`. Rollback commit
`60c0331faddb86ec119732a690e4b251aaadd0ec` has exactly the same Git tree,
without rewriting branch history. The OPQ50 rerun uses the same base with
one line changed in `PQQuantizer.h`: non-ADC stored-code assignment uses
`DistCalcMethod::L2` instead of `m_distCalcMethod`. ADC/SDC search scoring
remains InnerProduct. Later centroid normalization and split/head
maintenance changes are absent from all three cases.

## Matched measurement

| Item | Setting |
| --- | --- |
| Dataset | Official GloVe-100 Angular: 1,183,514 vectors, 100 dimensions, 10,000 queries |
| Metric | InnerProduct on FP64-normalized, Float32-stored inputs; normalized Angular, not raw MIPS |
| Cases | Unquantized Float32, global RaBitQ4, OPQ50 with 8-bit subcodes |
| Index | BKT heads, head ratio 0.2, one dynamic FILEIO layer used statically |
| Search | MaxCheck 4000, 64 internal results, posting/search/buffer page limits 4/4/4 |
| Returns | Separate native Search10 and Search40; no reranking |
| Recall10@K | Intersection of returned K IDs with official first ten IDs, divided by 100,000 across all queries |
| Repeats | Five full-query passes for each return count, then fresh-process reload of both |
| Performance | Fastest pass's QPS and its latency distribution, separately for return10 and return40 |
| Recall selection | Last pass; verify Search40's first ten IDs equal Search10 |
| Threads | 32 for index construction, eight for training/encoding, one query thread pinned to logical CPU 1 |
| Other settings | 50ms read latency limit, no concurrent inserts or search-time merging |
| FILEIO growth | 1GiB initial/increment, 64GiB ceiling per case |

Both reported recalls use **strict IDs**, not ANN-Benchmarks' distance
tolerance. The pinned upstream `run_individual_query` times normalization,
native search and the Python binding. Its returned-candidate distance
evaluation occurs outside the search timer. No ground truth is used to
choose candidates.

| Representation | Vector/code bytes | Posting record bytes | Base / hard posting capacity |
| --- | ---: | ---: | ---: |
| Float32 | 400 | 405 | 40 / 80 |
| RaBitQ4 | 88 | 93 | 176 / 352 |
| OPQ50 | 50 | 55 | 297 / 595 |

Equal page budgets do not mean identical indexes or equal vector-comparison
counts. All three indexes are built independently. RaBitQ pads internally
to 128 dimensions; Float32 and OPQ use 100.

The FAISS factory is `OPQ50,PQ50x8` with `METRIC_INNER_PRODUCT`; standard
OPQ/PQ learning still minimizes L2 reconstruction error. The OPQ50 rerun
uses L2-nearest stored-code assignment, matching FAISS encoding, while
retaining IP query scoring. Its model, codes and index are regenerated.
The Float32 and RaBitQ4 measurements are retained from the original run.

## Reproduce on Linux

Use a new workspace under `/data` for each case. Each single-case run has
a 72GiB allocation limit and keeps a 250GiB free-space floor plus 50GiB
contingency; start with at least 372GiB free. Do not overlap timed runs
with builds or other benchmarks. The example reruns only OPQ50.

```bash
set -euo pipefail
REPO=/data/path/to/SPTAG
TOOLS="$REPO/evaluation/2026-11-24-glove100-ip/reproduce.py"
CASE=opq50-ip
WORK=/data/your-owned-directory/new-glove100-ip-"$CASE"
test ! -e "$WORK"
mkdir -p "$WORK"/{tmp,cache,logs,input}
export TMPDIR="$WORK/tmp" TMP="$WORK/tmp" TEMP="$WORK/tmp"
export XDG_CACHE_HOME="$WORK/cache" PIP_CACHE_DIR="$WORK/cache/pip"
export PYTHONDONTWRITEBYTECODE=1

git clone --quiet --no-hardlinks "$REPO" "$WORK/source"
git -C "$WORK/source" checkout --quiet --detach c1c8f3970188a8725c08521dc776b8ca96f6bdbb
if [ "$CASE" = opq50-ip ]; then
git -C "$WORK/source" apply <<'PATCH'
diff --git a/AnnService/inc/Core/Common/PQQuantizer.h b/AnnService/inc/Core/Common/PQQuantizer.h
--- a/AnnService/inc/Core/Common/PQQuantizer.h
+++ b/AnnService/inc/Core/Common/PQQuantizer.h
@@ -221,4 +221,4 @@
             {
-                auto distCalc = COMMON::DistanceCalcSelector<T>(m_distCalcMethod);
+                auto distCalc = COMMON::DistanceCalcSelector<T>(DistCalcMethod::L2);
                 T* subvec = (T*)vec;
                 T* subcodebooks = m_codebooks.get();
PATCH
fi
git -C "$WORK/source" submodule update --init ThirdParty/zstd ThirdParty/RaBitQOfficial
git init --quiet "$WORK/ann-benchmarks"
git -C "$WORK/ann-benchmarks" remote add origin https://github.com/erikbern/ann-benchmarks.git
git -C "$WORK/ann-benchmarks" fetch --quiet --depth 1 origin \
  2e081ad32c1eccab72dcb739ad886c310b90f715
git -C "$WORK/ann-benchmarks" checkout --quiet --detach FETCH_HEAD

python3 -m venv "$WORK/venv"
PY="$WORK/venv/bin/python"
"$PY" -m pip install \
  numpy==1.26.4 faiss-cpu==1.10.0 h5py==3.12.1 psutil==6.1.1 \
  ansicolors==1.1.8 docker==7.1.0 PyYAML==6.0.2
cp "$TOOLS" "$WORK/reproduce.py"
TOOLS="$WORK/reproduce.py"
"$PY" "$TOOLS" build --root "$WORK"
curl --fail --location --show-error --max-time 1800 --max-filesize 500000000 \
  --output "$WORK/input/glove-100-angular.hdf5" \
  https://ann-benchmarks.com/glove-100-angular.hdf5
"$PY" "$TOOLS" prepare --root "$WORK"
"$PY" "$TOOLS" run --root "$WORK" --case "$CASE" --allow-missing-ids
```

The build requires the repository's existing Linux C++ dependencies,
including CMake, GCC, Python development headers, Boost, TBB, NUMA and SWIG.
Native options are Release, RABITQ/TBB ON and
FOLLY/SPDK/ROCKSDB/TIKV/GPU OFF. Docker's Python package is an upstream
import dependency; no Docker daemon or containers are used.

Set `CASE=float32-ip` or `CASE=rabitq4-ip` to reproduce either original
case; those selections skip the native patch. The runner accepts only the
two reviewed source fingerprints and records which one was used. Models
and encoded data are produced afresh. `--max-checks` and
`--search-internal-results` are explicit overrides; the published
comparison uses their defaults.

## Coverage and failure policy

`--allow-missing-ids` explicitly accepts the existing c1 posting-truncation
policy for this measurement. The read-only audit still records every
missing ID, current-record coverage, duplicate records and empty postings.
Missing vectors remain in the recall denominator; they are never excluded
from ground truth. Without this flag, incomplete coverage stops the run.

Malformed postings, invalid IDs, native execution errors and storage
exhaustion still fail the run. Warnings are retained. The known
hash-table-growth message is recorded separately because the native search
grows its workspace and retries. No split/capacity repair is applied.

Each case retains configuration, training/encoding metadata, build timing,
coverage/missing-ID reports, per-pass HDF5 IDs/distances/times, query/reload
summaries and log reviews. Runtime/source hashes and the frozen runner
hash identify the measurement. Models, datasets, checkpoints and raw
results are kept outside Git.
