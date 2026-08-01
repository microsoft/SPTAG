# Global RaBitQ quantizer

Global RaBitQ is an `IQuantizer` implementation backed by the official scalar
APIs: `quantize_scalar` encodes vectors and `reconstruct_vec` reconstructs
them for distance evaluation. It follows the normal global quantizer workflow,
not the SPANN posting-quantizer workflow.

This is the global scalar API exposed by the official submodule. It does not
use the separate split-code raw-query posting estimator.

Train the model and encode Float base vectors with `Release/quantizer`:

```bash
Release/quantizer \
  -d 128 -v Float -f XVEC \
  -i sift/sift_base.fvecs \
  -o sift/sift_base.rabitq3.u8bin \
  -oq sift/official_rabitq3_global.bin \
  -qt RaBitQQuantizer -qd 3 -ts 1000000
```

The encoded vectors are `UInt8`. The official scalar layout stores one byte per
input dimension plus two Float reconstruction factors, so 128-dimensional SIFT
vectors use `Dim=136`. Configure the encoded base-vector file and
`QuantizerFilePath` in the regular SPANN build configuration. Queries remain
raw Float vectors and are quantized through the loaded global `IQuantizer`.

`Script_AE/iniFile/build_SPANN_sift1m_rabitq3_global.ini` is the canonical
SIFT1M example. It uses normal raw postings; do not set `PostingQuantizer` for
this workflow.
