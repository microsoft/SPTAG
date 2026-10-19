# Platform and feature status

[Documentation index](README.md)

This is a scope and validation matrix, not a production-readiness guarantee.
The baseline workflow is Ubuntu 24.04 x86-64, GCC 13, CMake 3.28, Boost 1.83,
oneTBB, AVX2, and the optional storage/quantization integrations disabled.
Check the actual workflow result for the commit you use.

The baseline is a reproducible validation profile, not the product's capability
boundary. Disk indexing, incremental maintenance, compression, and distributed
construction are important engine directions; implementation maturity and
test coverage must be stated independently.

| Feature / configuration | Linux x86-64 | Native Windows | Validation boundary |
| --- | --- | --- | --- |
| C++ BKT/KDT and persistence | Developer baseline | Manual validation required | Selected core tests; standalone CMake consumer checks BKT and independent-process reload |
| Python BKT/KDT and persistence | Developer baseline: Python 3.12, NumPy 1.26 | Manual validation required | Public binding tests cover results, metadata, buffers, and save/load; example uses independent processes |
| Python SPANN STATIC and persistence | Small developer example | Manual validation required | Single layer; nonempty disk postings and independent-process Top-K queries checked |
| CLI, save/load | Baseline | Compatibility path | The file-based example checks BKT save/load/search |
| Core helpers and non-Folly map tests | Baseline | Manual validation required | Selected Boost.Test suites, not all tests |
| SPANN / SPFresh FILEIO | Available | Manual validation required | Built in the baseline; large update experiments are not covered by the smoke test |
| TBB queues / non-Folly map | Enabled in Linux baseline | Manual validation required | Map operations are locked; returned read iterators own snapshots |
| Linux without TBB | Known build limitations | Not applicable | Existing queue adapter lacks operations used by FILEIO; not a supported baseline |
| Java / .NET wrappers | Optional | Optional | Build discovery exists; runtime compatibility is not certified by SDK checks |
| TiKV RawKV | Experimental | Not validated | Requires gRPC/Protobuf and a separately versioned TiKV/PD deployment |
| RaBitQ and native auto-tuning | Implemented, opt-in | Not validated | Requires official submodule, AVX2/FMA or AVX512; tuning calibration is not end-to-end SPANN recall certification |
| RocksDB | Optional integration | Not validated | External matching development library required |
| SPDK / io_uring | Linux-specific integration | Not supported by these recipes | Dedicated storage setup; never required for the quick start |
| Folly | Optional developer configuration | Not validated | CMake currently enables it by default; baseline explicitly disables it |
| MPI distributed construction | Experimental | Not validated | Multi-host topology, shared formats, and `LARGEVID` require separate verification |
| CUDA / GPU | Legacy research path | Not validated | Outside baseline CI; legacy Docker recipe is not a support claim |
| macOS / ARM / Clang | Not in baseline | Not applicable | Current CMake compiler/SIMD assumptions require separate work |

## Version and data compatibility

- C++20 is used by the current GNU/MSVC build. Historical minimum-version
  checks and old paper toolchains are not tested compiler guarantees.
- The default ID type is 32-bit; `LARGEVID=ON` changes it to 64-bit. Use
  matching settings on every participant and do not assume index-file
  interchangeability.
- RaBitQ models and stored codes must match. See the
  [model compatibility contract](development/RaBitQ.md).
- TiKV persists postings, not the entire local head index and metadata.
  See [storage boundaries](Architecture.md).
- Paper and dated evaluation results apply to their recorded revisions and
  configurations, not automatically to this branch.

## Known validation limits

- The SDK developer baseline covers static in-memory indexes and a small
  single-layer STATIC SPANN disk index. It does not
  establish a versioned binary distribution, installed CMake package, stable
  ABI, cross-version index compatibility, or concurrent mutation guarantees.
  See the [SDK contracts](SDK.md).
- `DedupCandidateTest/SearchPathsMatchDeleteFirstAcrossVisibilityInterleavings`
  currently fails. The existing search loops mark candidates in the deduper
  before checking local deletion, so a deleted occurrence can suppress a later
  live occurrence. The fixture also checks TiKV/iterative visibility variants.
  This is outside the documentation/concurrent-map build repair; CI reports
  it separately as non-blocking and retains its log. A passing baseline must
  not be presented as a clean full test suite.
  This correctness issue blocks promotion of affected dynamic update/delete
  search paths into the stable SDK surface; a non-blocking diagnostic is not
  a release approval for those capabilities.
- The legacy Azure pipeline includes old hosted-agent/toolset assumptions and
  organization-specific tasks. YAML parsing is not remote execution validation.
- Docker, native Windows, TiKV cluster behavior, RaBitQ accuracy, SPDK, and GPU
  require their own environment and are not certified by a WSL baseline run.

## Expanding support

Before promoting a configuration, record the toolchain, build and test commands,
hardware constraints, and output. Add a reproducible check where practical.
Do not mark an integration supported based only on compilation or on another
platform's results.
