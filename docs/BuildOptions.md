# Build options and dependencies

[Documentation index](README.md) | [Linux](LinuxInstallation.md) |
[Windows](WindowsInstallation.md)

The project currently uses C++20. The documented baseline is GCC 13 on Ubuntu
24.04; older compiler checks in CMake are historical, not certification.
SWIG 4.0.2+ is the documented wrapper baseline.

## CMake options

Defaults below describe the current root CMake file. The installation recipes
override defaults deliberately; existing developer build behavior is unchanged.

| Option | Default | Dependency / consequence |
| --- | --- | --- |
| `TBB` | `ON` | TBB development package; required by the current Linux FILEIO queue path |
| `FOLLY` | `ON` | Folly built under `ThirdParty/folly/_build`, glog; baseline uses the locked non-Folly map instead |
| `SPDK` | `OFF` | SPDK, DPDK, ISA-L, ISA-L Crypto and Linux-specific libraries |
| `ROCKSDB` | `OFF` | An installed RocksDB CMake package and its compression dependencies |
| `TIKV` | `OFF` | gRPC, Protobuf, generated kvproto sources; runtime PD/TiKV cluster |
| `URING` | `OFF` | Linux liburing development library |
| `RABITQ` | `OFF` | Official RaBitQ submodule plus local portability overlays; suitable SIMD CPU |
| `GPU` | `OFF` | Compatible CUDA toolchain; outside baseline validation |
| `LARGEVID` | `OFF` | 64-bit vector IDs; coordinate index/data formats across machines |
| `LIBRARYONLY` | `OFF` | Skips some CLI/test targets; not a global switch disabling every executable or wrapper |
| `USE_ASAN` | `OFF` | GNU AddressSanitizer; use a separate Debug build |

Wrapper targets are discovered from installed Python, SWIG, JNI, and .NET
tools. They are not the same as enabling `GPU` or `LIBRARYONLY`.
Build `_SPTAG` explicitly for the Python SDK. The current developer baseline
uses Python 3.12 and NumPy 1.26; see the [SDK quick start](QuickStart.md).
For a C++ source consumer, link `SPTAG::SPTAG` after adding the repository as
a subdirectory. This target is not yet an installed `find_package` export.

## Dependency checkout

Baseline:

```bash
git submodule update --init ThirdParty/zstd
```

RaBitQ:

```bash
git submodule update --init ThirdParty/RaBitQOfficial
```

SPDK and optional dependencies should be initialized on Linux:

```bash
git submodule update --init --recursive ThirdParty/spdk ThirdParty/isal-l_crypto
```

Do not manually patch submodule worktrees. RaBitQ overlays are already in
`ThirdParty/RaBitQOfficialPatches`. Commit intentional dependency version bumps
separately from documentation changes.

## Optional integrations

- **TiKV:** follow [the integration guide](TiKV-Integration.md); generate gRPC
  stubs with the same Protobuf toolchain used for compilation.
- **RaBitQ:** initialize the official submodule, add `-DRABITQ=ON` to a
  separate build directory, and follow [training and search](RaBitQ_Global_Quantizer.md).
- **RocksDB:** configure from `ThirdParty/RocksDB`, not whichever directory
  a previous example left active. Use a current C++20-capable compiler for
  SPTAG; do not copy the old GCC 7 command from paper-era instructions.
  Build/install the pinned RocksDB revision and set `RocksDB_DIR` to its
  exported CMake package before enabling `ROCKSDB`.
- **Folly:** initialize `ThirdParty/folly`, install its revision-specific
  dependencies, and build shared PIC libraries under `_build`. Root CMake
  searches that location and glog. This is not needed for the baseline.
- **SPDK:** its dependency install/build is distinct from binding or formatting
  a device. See the [dedicated-device procedure](../Script_AE/StoragePreparation.md).
  Do not run storage preparation during ordinary installation.

These are integration prerequisites, not claims that optional combinations
were tested by the baseline workflow.

## Licenses

The top-level [MIT license](../LICENSE) covers SPTAG's own code. Submodules,
vendored kvproto definitions, and dataset artifacts have their own license
terms. Review each dependency's license before redistribution; do not relabel
all third-party contents as MIT.
