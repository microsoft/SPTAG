# RaBitQ compression, automatic tuning, and search

[Documentation index](README.md) |
[Implementation and compatibility](development/RaBitQ.md)

RaBitQ compresses vector representations and supplies matching distance
estimation through `IQuantizer`; it is not a posting-storage backend.
Use it to investigate the storage/accuracy tradeoff, including native bit-width
selection before building SPANN. It is an opt-in engine capability, separate
from the unquantized BKT/STATIC SPANN tutorial.

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

## Automatic bit-width selection

The current `indexbuilder` has an INI-driven C++ tuning path. It evaluates
RaBitQ bit widths in the supported range 1..8, chooses a qualifying width,
encodes the input corpus, and continues index construction with that model.
This is not the legacy NNI/Python tuning example.

The recipe below is checked against the current source, **not a recorded
RABITQ-enabled end-to-end run**. The default developer build has `RABITQ=OFF`
and explicitly rejects this request. Follow the enabled build above first.

In a **new, empty run directory**, place a writable copy of your 128-dimensional
Float DEFAULT-format base vectors at `base.fbin`. Create an empty `tmp`
directory and save this as `build.ini`:

```ini
[Base]
ValueType=Float
Dim=128
DistCalcMethod=L2
IndexAlgoType=BKT
VectorPath=base.fbin
VectorType=DEFAULT
IndexDirectory=index

[RaBitQAutoTune]
isExecute=true
TrainSamples=10000
TestQueries=100
ResultCount=10
TargetRecall=0.9
BatchSize=1000000

[SelectHead]
isExecute=true
Ratio=0.12

[BuildHead]
isExecute=true

[BuildSSDIndex]
isExecute=true
BuildSsdIndex=true
Storage=STATIC
EnableADC=true
TmpDir=tmp
```

Set `SPTAG_ROOT` to the absolute source checkout. From that run directory:

```bash
"$SPTAG_ROOT/Release/indexbuilder" -c build.ini -a SPANN \
  -v Float -d 128 -f DEFAULT -o index -t 2
```

Do not pass `-pq` for automatic selection: an explicit quantizer file bypasses
the auto-tuning branch. This example reads its corpus from `Base.VectorPath`,
not the CLI `-i` option, and uses L2 plus `EnableADC=true` explicitly.

| Setting | Current default | Meaning / constraint |
| --- | --- | --- |
| `TrainSamples` | 10000 | Leading input rows used for tuning; choose a positive sample count with enough rows for K |
| `TestQueries` | 100 | Leading rows of that tuning sample, capped at the sample count; not an external held-out query set |
| `ResultCount` | 10 | Calibration K; positive and no larger than the tuning corpus |
| `TargetRecall` | 0.9 | Calibration target in [0, 1]; not the final SPANN query recall guarantee |
| `BatchSize` | 1000000 | Positive number of corpus vectors encoded per batch |
| CLI `-t` | Reader default unless supplied | Tuning worker count; this example sets 2 and supplies missing build-section thread settings |

Expect logs for candidate bits, `Recall@K`, and the selected width.
The current implementation writes `rabitq_auto_quantizer.bin` in the working
directory and appends `.quan` to the input vector path for the encoded corpus.
For this recipe that is `base.fbin.quan`; do not run it against a protected
or shared dataset location or reuse a directory containing published artifacts.
The builder changes the encoded value type/dimension internally; do not guess
the resulting code width or reuse a manually chosen `Dim=68`.
The requested saved index is under `index/`.

Failure to meet the target in the supported bit range, encoding/build errors,
or a disabled RaBitQ build must fail the command. A model or `.quan` file left
behind is not proof that the complete index was built and saved.
Reopen the saved index with matching raw Float queries and measure recall
against an **independent** query/truth set. The calibration uses training rows
and quantized distance estimates, not a full end-to-end SPANN recall evaluation.

## Train and encode

Use this manual path when you want a specific bit width or model configuration
instead of automatic selection.

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
