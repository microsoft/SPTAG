# Legacy global RaBitQ quantizer

This reconstruction-style global adapter exists for ABI experimentation only.
It quantizes both BKT heads and postings into symmetric scalar codes, which
does not use the official RaBitQ asymmetric estimator or SIMD kernels. Use
[`OfficialRaBitQ_Postings.md`](OfficialRaBitQ_Postings.md) for the supported
performance path.

`RaBitQQuantizer` is a global `IQuantizer` implementation. It uses the same
quantized head and posting path as PQ/OPQ: BKT and KDT build over quantized
codes, and classic SPANN scans those codes directly. It does not require a
full-vector file, reranking, tags, multi-tenancy, tail replicas, or posting
sidecars.

## Encode vectors

Train the model and encode the base vectors in one command:

```bash
Release/quantizer \
  -d 128 -v UInt8 -f DEFAULT \
  -i sift1b/base.1B.u8bin \
  -o sift1b/base.1B.rabitq4.u8bin \
  -oq sift1b/rabitq4_global.bin \
  -qt RaBitQQuantizer -qd 4 -ts 1000000
```

`-qd` is the number of RaBitQ scalar bits per padded input dimension, from 1
through 8. It is not the output-vector dimension. The encoded dimension is:

```text
ceil(next_power_of_two(max(64, raw_dimension)) * bits / 8) + 8
```

The final eight bytes retain RaBitQ's per-vector reconstruction parameters. For
128-dimensional vectors with four bits, the encoded base-vector dimension is
72 bytes. Configure the encoded vector file as `ValueType=UInt8` and `Dim=72`.

## Build and search

`Script_AE/iniFile/build_SPANN_sift1b_rabitq_global.ini` is a native,
build-only INI example for the four-bit, 128-dimensional case. Its
`VectorPath`, `ValueType`, and `Dim` describe the encoded UInt8 code vectors.
After loading `QuantizerFilePath`, SSDServing derives the raw query type and
dimension from the model before quantizing each query in memory. It explicitly
uses `Rerank=0`, `EnableADC=false`, and `[SearchSSDIndex] isExecute=false`.

Run the standard executable with that config:

```bash
Release/ssdserving Script_AE/iniFile/build_SPANN_sift1b_rabitq_global.ini
```

For a standard SSDServing search config, keep the query paths as raw vectors.
`QueryResultSet` quantizes each raw query with the loaded model before search.
The core API also supports raw-query ADC by calling `SetQuantizerADC(true)`
before creating a `QueryResultSet`; this is covered by `RaBitQQuantizerTest`.
