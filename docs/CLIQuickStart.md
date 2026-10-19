# CLI quick start: build, save, reload, and search

[Documentation index](README.md)

For embedding SPTAG in your program, start with the
[Python and C++ SDK quick start](QuickStart.md). This page tests the file-based
command-line tools.

First build `indexbuilder` and `indexsearcher` using the
[Linux](LinuxInstallation.md) or [Windows](WindowsInstallation.md) guide.
This example uses BKT, Float vectors, and L2 distance. It needs no dataset
download, storage daemon, Python package, or special SSD.

## Run the self-checking example

Install Node.js 20+ if it is not already available. Node is only the portable
example driver, not a dependency of the C++ library.
Use the [official Node.js installation instructions](https://nodejs.org/en/download)
and check `node --version`; the Ubuntu distribution package may be too old.

From the repository root on Linux:

```bash
node Tools/quickstart.mjs Release
```

On native Windows:

```powershell
node Tools\quickstart.mjs build\Release
```

The [driver](../Tools/quickstart.mjs):

1. Creates 1,024 deterministic, eight-dimensional Float vectors and three queries.
2. Writes SPTAG DEFAULT-format input: little-endian 32-bit count and dimension,
   followed by row-major Float32 coordinates.
3. Runs the real `indexbuilder` CLI and saves a BKT index.
4. Runs `indexsearcher` in a new process, loading that saved index.
5. Checks all three returned IDs and zero self-match distances.
6. Reloads and queries again, checks the same results, and removes its own
   temporary directory.

The final line must be:

```text
Quick start passed: 3/3 nearest neighbors matched after save/load and repeat search.
```

Any missing binary, nonzero process exit, missing output, wrong result count,
wrong ID, or incorrect distance fails the command. Intermediate library logs
may vary. This is a correctness smoke test, **not** a recall or speed benchmark.

## Equivalent command-line operations

The driver invokes these operations using its generated absolute file paths:

```text
indexbuilder -d 8 -v Float -f DEFAULT -i vectors.bin -o index -a BKT -t 1 Index.DistCalcMethod=L2 Index.NumberOfThreads=1
indexsearcher -d 8 -v Float -f DEFAULT -i queries.bin -x index -o results.txt -k 1 -t 1
```

Result rows have the form `queryID:distance@vectorID|`. Expected vector IDs
are `0`, `127`, and `511`, respectively, each with distance zero. Library IDs
are zero-based row IDs in this example.

For your own data, keep the index directory and query it in a later process
with `indexsearcher -x`. Use a fresh output directory for a new index.
See the [command-line reference](GettingStart.md) and
[parameters](Parameters.md) for larger configurations and SPANN.
