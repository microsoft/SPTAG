# Benchmark report template

Copy this structure for a new report. Replace every `Not recorded` value with
evidence before calling the run reproducible; do not invent historical values.

## Provenance

| Field | Value |
| --- | --- |
| Run date and timezone | Not recorded |
| Run status: complete, partial, or query-only validation | Not recorded |
| SPTAG commit and dirty-state diff | Not recorded |
| Executable and compared static-library checksums | Not recorded |
| Submodule commit IDs | Not recorded |
| TiKV commit / build identity | Not recorded |
| PD/TiKV image tags and immutable digests | Not recorded |
| Compiler / CMake / Boost / TBB / Protobuf / gRPC | Not recorded |
| Configure command and build type | Not recorded |
| Config file and checksum | Not recorded |
| Input and ground-truth checksums | Not recorded |
| Raw log / JSON artifact location | Not recorded |
| Artifact availability: committed, publicly downloadable, or local-only | Not recorded |

## Environment and topology

Record CPU, cores, SIMD features, RAM, disks, filesystem, OS/kernel, resource
limits, NUMA placement, network bandwidth/RTT, and process affinity.
State whether processes share one host or use independent hosts. Record
PD/TiKV node count, replication factor, cache sizes, API version, and storage
compression. Identify client-versus-server resource accounting.

## Workload

Record dataset/version, value type, dimension, metric, ID width, query count,
TopK, base size, insert/delete totals, per-batch counts, batch count, index
parameters, thread counts, background work, and index/model formats.
Record training split, quantizer settings, raw reranking, and random seed (or
state that a seed cannot be fixed by the current harness).

## Measurement

- Warmup policy and cold/warm cache definition:
- Number of independent repetitions and run order:
- Latency units, timed region, and percentile calculation:
- Throughput formula and concurrency:
- Recall ground-truth corpus/phase, tie handling, and denominator:
- Successful operations, retries, timeouts, and failed operations:
- Initial build time versus update time:
- Memory and disk accounting:

## Results and limitations

Include per-run results, variation, raw outputs, and relevant recall together
with speed. Identify missing data and confounders. Do not compare across
different binary revisions, replication settings, query sets, or cache states
without explicitly controlling or disclosing the difference.

Separate planned work from completed work. Mark log-extracted rows and
incomplete raw outputs, and do not count query repetitions on one saved index
as independent builds. A query-only fix validation does not replace a complete
update run or results at another scale. Keep historical measurements unchanged
and link to a new dated report when a correction changes their interpretation.
