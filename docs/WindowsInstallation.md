# Windows installation

[Documentation index](README.md) | [Support matrix](SupportMatrix.md)

Native Windows is a compatibility path, not the Linux CI baseline. TiKV, SPDK,
Folly, io_uring, and distributed experiment recipes are not validated on native
Windows. For the complete Linux workflow, use Ubuntu 24.04 under WSL2.

## Toolchain

- Visual Studio 2022 with Desktop development with C++, the Windows SDK,
  and CMake tools; use a Developer PowerShell.
- A C++20-capable MSVC toolchain.
- CMake 3.28+ and SWIG 4.0.2+ on `PATH`.
- Boost with system, thread, serialization, wserialization, regex, filesystem,
  and unit_test_framework libraries built for the same MSVC toolset and x64.
  The Linux baseline uses Boost 1.83; native Windows combinations require
  separate validation.
- Python development files only if building the Python wrapper.
- Node.js 20+ to run the self-checking CLI example and documentation checker.

Do not mix old precompiled Boost MSVC 14.1 binaries with a newer toolset.
Use `BOOST_ROOT` to identify your own matching Boost installation if CMake
does not discover it automatically.

## Clone only the required dependency

```powershell
git clone https://github.com/microsoft/SPTAG.git
Set-Location SPTAG
git submodule update --init ThirdParty/zstd

cmake -S . -B build -A x64 -DCMAKE_BUILD_TYPE=Release `
  -DFOLLY=OFF -DTBB=OFF -DSPDK=OFF -DROCKSDB=OFF -DTIKV=OFF -DGPU=OFF -DRABITQ=OFF
cmake --build build --config Release --target indexbuilder indexsearcher SPTAGTest --parallel 2
```

The baseline needs no SPDK submodules. Avoid a blanket recursive clone on
Windows: the pinned nested xNVMe dependency contains directories called `aux`,
a reserved Windows device name. If an existing recursive checkout fails
there, use Linux/WSL to work with SPDK rather than disabling NTFS protections
globally. Such a failure does not mean the baseline needs that dependency.

## Verify

Run from the repository root:

```powershell
node Tools\quickstart.mjs build\Release
.\build\Release\SPTAGTest.exe --run_test=CommonHelperTest:IniReaderTest:ConcurrentHashMapTest --log_level=test_suite
```

Native Windows CMake artifacts are under `build\Release`, unlike Linux's
source-root `Release`. An optional Python session can use:

```powershell
$env:PYTHONPATH = "$PWD\build\Release;$env:PYTHONPATH"
```

These commands document the intended native configuration; a successful Linux
or WSL run is **not** native Windows certification. Report the exact MSVC,
Boost, SWIG, and CMake versions when reporting a Windows build problem.
