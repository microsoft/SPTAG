# Linux installation

[Documentation index](README.md) | [Build options](BuildOptions.md)

## Baseline environment

Use Ubuntu 24.04 x86-64, GCC 13, CMake 3.28, Boost 1.83, and oneTBB 2021.
This is the baseline selected for the automated workflow, not a promise that
all older compiler versions or optional backends work. GCC builds compile
distance kernels with AVX2; use a CPU exposing AVX2.

Install dependencies:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build git swig python3-dev \
  libboost-system-dev libboost-thread-dev libboost-serialization-dev \
  libboost-regex-dev libboost-filesystem-dev libboost-test-dev libtbb-dev libnuma-dev
```

SWIG and Python development headers allow CMake to build the Python wrapper.
NumPy is additionally needed to run the Python tutorials. The baseline CLI
example does not depend on Python packages. Node.js 20+ is only required for
the self-checking example and documentation checker.

## Clone and configure

```bash
git clone https://github.com/microsoft/SPTAG.git
cd SPTAG
git submodule update --init ThirdParty/zstd

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DFOLLY=OFF -DTBB=ON -DSPDK=OFF -DROCKSDB=OFF -DTIKV=OFF -DGPU=OFF -DRABITQ=OFF
cmake --build build --target indexbuilder indexsearcher SPTAGTest --parallel 2
```

Run every command above from the repository root. `--parallel 2` is a
conservative starting point for memory-heavy C++ template compilation.
Increase it only when sufficient memory is available.

Only zstd is required as a submodule for this baseline. To work with all
integrations, initialize their dependencies explicitly or run
`git submodule update --init --recursive` on Linux. RaBitQ needs
`ThirdParty/RaBitQOfficial`; its local portability overlays live in
`ThirdParty/RaBitQOfficialPatches`, not in a modified submodule checkout.

## Verify

```bash
node Tools/quickstart.mjs Release
Release/SPTAGTest --run_test=CommonHelperTest:IniReaderTest:ConcurrentHashMapTest:AlgoTest/BKTTest:AlgoTest/KDTTest --log_level=test_suite
```

The quick start asserts search results after saving and reloading the index.
The selected unit suites need no TiKV cluster or external dataset. They are
not the full performance, storage, or integration test suite.

Linux output directories are relative to the **source root**:

| Build type | Executables and libraries |
| --- | --- |
| Release | `Release` |
| Debug | `Debug` |

For an optional wrapper session, set the module path using shell syntax:

```bash
export PYTHONPATH="$PWD/Release${PYTHONPATH:+:$PYTHONPATH}"
```

## Docker

The root [Dockerfile](../Dockerfile) builds the same CPU baseline on Ubuntu
24.04. Initialize zstd before building the image:

```bash
docker build -t sptag:local .
docker run --rm sptag:local ./Release/SPTAGTest \
  --run_test=CommonHelperTest:IniReaderTest:ConcurrentHashMapTest
```

The image builds CLI/test targets, not Python wrapper modules. It does not
deploy TiKV, initialize SPDK devices, or contain benchmark datasets.
The legacy [CUDA Dockerfile](../Dockerfile.cuda) is a separate,
unvalidated research recipe, not the baseline image.

## Troubleshooting

- **Folly missing:** use `-DFOLLY=OFF` for the baseline. Default CMake options
  still enable Folly for existing developer configurations.
- **oneTBB API errors on older revisions:** this development line repairs the
  non-Folly map adapter independently of TBB's container API. Use a matching
  source revision; disabling TBB is not a supported Linux baseline workaround.
- **Boost missing:** install all listed components, including the test library.
  Do not mix headers and libraries from different Boost installations.
- **Compiler errors with an older GCC:** use the C++20 baseline above; the
  historical GCC 5 configure-time check is not a support statement.
- **Illegal instruction:** check CPU/VM AVX2 availability.
- **Out of memory while compiling:** lower `--parallel`; do not omit failed
  targets and call the build successful.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for debugging and validation scope.
