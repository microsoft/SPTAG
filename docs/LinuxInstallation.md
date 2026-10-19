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
sudo apt-get install -y build-essential cmake ninja-build git swig python3-dev python3-numpy \
  libboost-system-dev libboost-thread-dev libboost-serialization-dev \
  libboost-regex-dev libboost-filesystem-dev libboost-test-dev libtbb-dev libnuma-dev
```

SWIG and Python development headers build the Python SDK; the example uses
NumPy. The current validated Python runtime is Ubuntu's Python 3.12 with NumPy
1.26. Check `python3 --version` and
`python3 -c "import numpy; print(numpy.__version__)"`.
Node.js is not required for either SDK example. Optional CLI smoke tests and
documentation checks require Node.js 20+; use the
[official installer](https://nodejs.org/en/download) and check `node --version`
rather than assuming Ubuntu's package meets that requirement.

## Clone and configure

```bash
git clone --branch users/qiazh/doc https://github.com/microsoft/SPTAG.git
cd SPTAG
git submodule update --init ThirdParty/zstd

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DFOLLY=OFF -DTBB=ON -DSPDK=OFF -DROCKSDB=OFF -DTIKV=OFF -DGPU=OFF -DRABITQ=OFF
cmake --build build --target _SPTAG indexbuilder indexsearcher SPTAGTest --parallel 2
```

These SDK instructions track `users/qiazh/doc` until merged into the default
branch. Record `git rev-parse HEAD`; do not substitute the default branch or a
package-index release and assume it contains this workflow.

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
export PYTHONPATH="$PWD/Release${PYTHONPATH:+:$PYTHONPATH}"
python3 Examples/python/quickstart.py --index quickstart-output/python
python3 Examples/python/quickstart.py --index quickstart-output/python \
  --load --queries quickstart-output/python/queries.npy
python3 -m unittest discover -s Test/python -v
(cd Release && ./SPTAGTest --run_test=CommonHelperTest:IniReaderTest:ConcurrentHashMapTest:AlgoTest/BKTTest:AlgoTest/KDTTest --log_level=test_suite)
```

The [SDK quick start](QuickStart.md) also covers a standalone C++ consumer and
your own vectors/business IDs. The Python example keeps its output; choose a
new directory to repeat a build. The optional [CLI smoke test](CLIQuickStart.md)
uses a separate temporary dataset.
The selected unit suites need no TiKV cluster or external dataset. They are
not the full performance, storage, or integration test suite.
Run them from the output directory as shown so their temporary INI and
index fixtures stay inside the ignored build output.

Linux output directories are relative to the **source root**:

| Build type | Executables and libraries |
| --- | --- |
| Release | `Release` |
| Debug | `Debug` |

`PYTHONPATH` makes the locally built `SPTAG.py` and `_SPTAG.so` importable:

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
- **`import SPTAG` fails:** build `_SPTAG`, set the source-root `Release` module
  path, and use the Python interpreter matching the CMake-discovered headers.
  Check `python3 -c "import SPTAG; print(SPTAG.__file__)"` for a stale installation.
- **Out of memory while compiling:** lower `--parallel`; do not omit failed
  targets and call the build successful.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for debugging and validation scope.
