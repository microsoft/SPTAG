# RaBitQ training and search

[Documentation index](README.md) |
[Implementation and compatibility](development/RaBitQ.md)

RaBitQ is an optional `IQuantizer` integration, not a replacement posting-storage
backend. This workflow is separate from the baseline CLI smoke test.

## Prerequisites

- A Linux baseline build environment and an AVX2/FMA or AVX512 CPU.
- The official `ThirdParty/RaBitQOfficial` submodule, with parent-repository
  portability overlays left in `ThirdParty/RaBitQOfficialPatches`.
- Float training/base vectors, raw Float queries, and matching ground truth.
- Separate output paths for each new model and encoded corpus.

```bash
git submodule update --init ThirdParty/RaBitQOfficial
cmake -S . -B build-rabitq -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DFOLLY=OFF -DTBB=ON -DSPDK=OFF -DROCKSDB=OFF -DTIKV=OFF -DGPU=OFF -DRABITQ=ON
cmake --build build-rabitq --target quantizer indexbuilder indexsearcher SPTAGTest --parallel 2
```

Native Windows is not certified by this recipe. Do not infer RaBitQ accuracy
or portability from a successful baseline build.

## Train and encode

For 128-dimensional SIFT Float XVEC data:

```bash
Release/quantizer \
  -d 128 -v Float -f XVEC \
  -i sift1m/sift_base.fvecs \
  -o sift1m/sift_base.rabitq3.u8bin \
  -oq sift1m/official_rabitq3_global.bin \
  -qt RaBitQQuantizer -qd 3 -ts 1000000
```

Training data and dataset downloads are not supplied by this guide. Use only
base/training vectors for training, not held-out query vectors.

## Build and query

Use the [canonical SIFT1M configuration](../Script_AE/iniFile/build_SPANN_sift1m_rabitq3_global.ini)
as an advanced example and adjust all input/output paths to your dataset.
It uses STATIC postings and a global 3-bit model.

- Encoded vectors use the `UInt8` container.
- For this 128-dimensional 3-bit example, `Dim=68`: 48 packed bytes plus
  five Float factors. This is not the original Float dimension.
- Set `QuantizerFilePath` to the matching model.
- Queries remain raw Float vectors; enable ADC for query evaluation.

Consult [the SPANN command examples](GettingStart.md) for build/search invocation.
The encoded dimension depends on bit width, padding, and model mode; do not
reuse `Dim=68` for another configuration.

## Compatibility and evaluation

Never replace only the model beneath an existing encoded corpus or index.
Rotation and optional local-centroid models require matching codes; changing
them requires re-encoding and a separate index build. The
[developer contract](development/RaBitQ.md) records model versions, ADC/SDC
distance requirements, local residual quantization, and rebuild conditions.

Run the RaBitQ tests with `--list_content` first to confirm the test names in
your binary, then select the RaBitQ suite. Evaluate recall on the complete
query set and report candidate/search settings, total model/code memory, and
latency. A small nominal bit width alone is not a total-index-size claim.
