# GloVe-100 IP: c1 with corrected OPQ encoding

[Reproduction and protocol](README.md)

The native base is `c1c8f3970188a8725c08521dc776b8ca96f6bdbb`.
Rollback commit `60c0331faddb86ec119732a690e4b251aaadd0ec` has the same
tracked tree. Float32 and RaBitQ4 retain their original c1 measurements.
Only OPQ50 is rerun with one native line changed: non-ADC stored-code
assignment in `PQQuantizer.h` uses L2. Its ADC/SDC search scoring remains
InnerProduct. No centroid-normalization or split/head-maintenance changes
are included.

The OPQ50 rows below replace the original c1 OPQ50 results. Its model,
encoded data and index are generated afresh in a separate run directory.
The other two cases have not been rerun.

## Dataset and configuration

| Item | Value |
| --- | --- |
| Dataset | GloVe-100 Angular |
| Base vectors / dimensions | 1,183,514 / 100 |
| Queries / ground-truth neighbors | 10,000 / official first 10 |
| Input normalization / native distance | Unit-length Float32 vectors / InnerProduct |
| Index / head ratio | SPANN, BKT heads, one FILEIO layer / 0.2 |
| MaxCheck / internal results | 4000 / 64 |
| Posting / search / buffer pages | 4 / 4 / 4 |
| Returned neighbors | Separate Search10 and Search40 |
| Reranking / concurrent insertion | None / none |
| Build / quantizer / query threads | 32 / 8 / 1 |
| Query CPU affinity | Logical CPU 1 |
| Primary passes / reload passes | 5 / 1 for each case and return count |

Recall10@K is the number of official top-ten IDs found among the returned
K IDs, summed over all queries and divided by 100,000. Both recalls below
use **strict IDs**, with no distance-tolerance allowance. Missing indexed
IDs remain in the denominator.

## Recall and query throughput

Recall comes from primary pass 5. QPS is from the fastest of the five
primary passes, selected independently for Search10 and Search40.

| Case | Recall10@10 (%) | Recall10@40 (%) | Search10 QPS | Search40 QPS |
| --- | ---: | ---: | ---: | ---: |
| Float32 IP | 83.389 | 83.390 | 319.57 | 318.78 |
| RaBitQ4 IP | 75.780 | 83.061 | 346.32 | 343.53 |
| OPQ50 IP | 72.843 | 82.967 | 353.45 | 327.56 |

## Query latency

Each row uses the same primary pass selected for that case/return count's
QPS above. All latency values are milliseconds.

| Case | Returned | Selected pass | Mean ms | p50 ms | p95 ms | p99 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Float32 IP | 10 | 5 | 3.129 | 3.094 | 3.539 | 3.669 |
| Float32 IP | 40 | 1 | 3.137 | 3.103 | 3.546 | 3.678 |
| RaBitQ4 IP | 10 | 5 | 2.888 | 2.835 | 3.788 | 4.213 |
| RaBitQ4 IP | 40 | 1 | 2.911 | 2.858 | 3.818 | 4.210 |
| OPQ50 IP | 10 | 4 | 2.829 | 2.808 | 3.516 | 3.912 |
| OPQ50 IP | 40 | 1 | 3.053 | 3.023 | 3.870 | 4.323 |

## Construction

Times are seconds. Quantizer preparation includes training and encoding;
native index construction and explicit save are measured separately.
Dataset normalization and native compilation are excluded.

| Case | Quantizer preparation s | Native index build s | Explicit save s |
| --- | ---: | ---: | ---: |
| Float32 IP | 0 | 1041.792 | 1.794 |
| RaBitQ4 IP | 3.566 | 1144.394 | 0.341 |
| OPQ50 IP | 94.361 | 381.477 | 0.328 |

## Index coverage

The read-only audit covers all 1,183,514 source IDs. Counts below describe
the saved index before query measurements; no missing ID is removed from
the dataset or ground truth.

| Case | Live heads | Current records | Reachable source IDs | Missing source IDs | Duplicate current records within postings |
| --- | ---: | ---: | ---: | ---: | ---: |
| Float32 IP | 435,791 | 7,044,496 | 1,183,504 | 10 | 17,358 |
| RaBitQ4 IP | 239,349 | 8,840,673 | 1,183,514 | 0 | 0 |
| OPQ50 IP | 239,543 | 9,050,782 | 1,183,514 | 0 | 0 |

## Fresh-process reload

Each case is loaded in a new process and runs one complete pass for each
return count. Reload QPS is reported separately, not included in primary
pass selection.

| Case | Recall10@10 (%) | Recall10@40 (%) | Search10 QPS | Search40 QPS |
| --- | ---: | ---: | ---: | ---: |
| Float32 IP | 83.389 | 83.390 | 316.22 | 316.32 |
| RaBitQ4 IP | 75.780 | 83.061 | 344.08 | 341.61 |
| OPQ50 IP | 72.843 | 82.967 | 344.09 | 347.37 |

## Run completion and logs

Completed pass counts are combined across Search10 and Search40: the
expected totals per case are ten primary passes and two reload passes.
Errors and warnings are counts of matching lines in the retained log
reviews. Handled workspace hash-table growth messages are counted
separately.

| Case | Primary passes | Reload passes | Build errors | Build warnings | Query/reload errors | Query/reload warnings | Handled hash-table growth messages |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Float32 IP | 10 | 2 | 0 | 105 | 0 | 0 | 23 |
| RaBitQ4 IP | 10 | 2 | 0 | 0 | 0 | 0 | 32 |
| OPQ50 IP | 10 | 2 | 0 | 0 | 0 | 0 | 32 |

## Result-file mapping

Paths are relative to each case's run directory, listed below. `CASE` is
`float32-ip`, `rabitq4-ip` or `opq50-ip`.

| Table | Artifact and fields |
| --- | --- |
| Recall and throughput | `runs/CASE/queries/summary.json`: `points`, selected by `returned_neighbors`; `strict_recall_10_at_k * 100`, `qps` |
| Latency | Same summary: `best_round` and the matching entry in `passes`; `mean_ms`, `p50_ms`, `p95_ms`, `p99_ms` |
| Construction | `runs/CASE/build.json`: `quantization_preparation_seconds`, `native_build_seconds`, `save_seconds` |
| Coverage | `runs/CASE/physical-audit/coverage.json`: `live_top_heads`, `current_records`, `unique_current_reachable_ids`, `missing_source_count`, `duplicate_current_records` |
| Reload | `runs/CASE/reload/summary.json`: same point selection as primary results |
| Completed passes | Sum lengths of `points[].passes` in primary/reload summaries |
| Build logs | `logs/CASE-index.review.json`: lengths of `errors` and `warnings` |
| Query/reload logs | Sum matching counts from `logs/CASE-queries.review.json` and `logs/CASE-reload.review.json` |
| Handled growth messages | Sum `handled_hash_table_resizes` lengths across build, queries and reload reviews |

## Measurement provenance

Float32 and RaBitQ4 used the existing fingerprint-verified c1 Release
executable and libraries. OPQ50 uses a fresh Release build of c1 plus the
one-line stored-encoding correction. Each run freshly normalizes inputs
from the hash-verified official HDF5 and generates its models, encoded data
and indexes. The corrected OPQ50 run reuses no previous model, index or
query result.

Original Float32/RaBitQ4 runtime and shared input hashes:

| Artifact | SHA256 |
| --- | --- |
| Frozen runner | `81b19bf3ec85e577086bfd485af196483d8f656f8568b719972393f04cc1fb00` |
| Native 204-file source fingerprint | `41c85b549de9b83d539f6994c07d3ed5cc8ed530b0d3a7289f0fb9fd3d11009c` |
| Native library | `9215c647b2da082ab822a9bcbddbc1edda445bd9889f5207eff7868de8330229` |
| Python native binding | `4903b37f65f33ba244d886fea73dcc66b1bf77e16afccafed3b2fde82ca7852d` |
| Quantizer executable | `040358edfdf99e5fa57c9deba12dde3f7ae34139950bad49cc5637343dd20a38` |
| Official GloVe HDF5 | `544af1d5e84e112cd4749571dcfd8ca109818a572f850af75a3a09e093a953c4` |
| Normalized base | `54a36bc1d25aaa52f72c740335130a874028363ef6ab8caea6ea033ea0f38297` |
| Normalized queries | `1e25c68dd4aaef0654acbdd7f246de14fc73a555bf8b514e4bc85c748be620c2` |

OPQ50-only rerun:

| Artifact | SHA256 |
| --- | --- |
| Frozen runner | `ad26cf454eb6f9477e15434fc68b4b7d20a230e2a558bafa260b74ccc850c3ea` |
| Native 204-file source fingerprint | `8fd72392938519fecf5cab33a9836367d1d33246b97c63dda4497496a48619a5` |
| One-line native patch | `37850d5546aced248c5204cd88bed0f7ab52caa5f5409ca0405c77e33ff0cee6` |
| Native library | `4f7c72256ba7965419bff6bef9955fb2bea888ba57eb18e75775ce6b4b58c14e` |
| Python native binding | `553dd5f49cffa911d450bd46ddd28a29636a5eb576d71ab3323f4f6597071459` |
| Quantizer executable | `3563b0cfa0868525021b74119b10f168f912b03a894ecaab2181ccc10c3d7d6e` |
| OPQ50 model | `b4e474d3c9452fc9a6190be4e8e19a542c7a7049806c157ea34f9de5cf718334` |
| OPQ50 encoded base | `3baee8c6c0101471df9ecfebe880f5e3970b9eb40ae287e47e711c74137df662` |

The shared host uses an Intel Xeon E5-2698 v4, 40 logical CPUs and about
251GiB RAM. Software is Ubuntu 24.04, GCC 13.3, Python 3.12 and faiss-cpu
1.10.0. No GPU is used. Query processes are pinned to logical CPU 1.
These are shared-host measurements, not directly comparable to website
QPS or the earlier multithreaded SIFT measurements.

Runtime/source provenance, frozen recipes and private setup drivers,
models, configuration, audits, logs and raw query outputs are retained
under these run directories:

| Reported cases | Run directory |
| --- | --- |
| Float32 IP and RaBitQ4 IP | `/data/zhangt/glove100-ip-c1-20261124` |
| Corrected OPQ50 IP | `/data/zhangt/glove100-ip-opq-l2-20261124` |

Both runs explicitly enable `--allow-missing-ids`; missing IDs are reported
above and remain in the recall denominator.
