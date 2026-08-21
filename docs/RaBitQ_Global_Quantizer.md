# Global RaBitQ quantizer

Global RaBitQ is an `IQuantizer` implementation backed by the official
full-code quantizer, compact-code packer, and asymmetric distance estimator.
Base vectors use `quantize_full_single` and `packing_rabitqplus_code`; ADC
queries are evaluated with `full_est_dist` and the official SIMD inner-product
kernel selected for the configured bit width. Query-only distance factors are
computed once when the ADC query buffer is prepared, not once per candidate.

The SDC compatibility path reconstructs its query code with the official
`reconstruct_vec` API before invoking the same official asymmetric estimator.
Online search with `EnableADC=true` does not reconstruct base vectors.

Train the model and encode Float base vectors with `Release/quantizer`:

```bash
Release/quantizer \
  -d 128 -v Float -f XVEC \
  -i sift/sift_base.fvecs \
  -o sift/sift_base.rabitq3.u8bin \
  -oq sift/official_rabitq3_global.bin \
  -qt RaBitQQuantizer -qd 3 -ts 1000000
```

The encoded vectors are stored in a `UInt8` container, but their code payload is
packed at the configured bit width. Five Float factors follow each packed code,
so 128-dimensional 3-bit SIFT vectors use `Dim=68`
(`128 * 3 / 8 + 5 * sizeof(float)`). Configure the encoded base-vector file and
`QuantizerFilePath` in the regular SPANN build configuration. Queries remain
raw Float vectors and are prepared through the loaded global `IQuantizer`.

The official compact kernels require AVX2/FMA or AVX512 and pad dimensions to a
multiple of 64. The current adapter supports the official L2 estimator; cosine
distance is intentionally unsupported.

`Script_AE/iniFile/build_SPANN_sift1m_rabitq3_global.ini` is the canonical
SIFT1M example. It uses STATIC postings containing the global RaBitQ codes;
keep `PostingQuantizer=None` because RaBitQ is already the global quantizer.
