#include "rabitqlib/simd/dispatch.hpp"

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

#include "rabitqlib/simd/space_dispatch.hpp"
#include "rabitqlib/simd/fastscan_dispatch.hpp"
#include "rabitqlib/simd/pack_excode_dispatch.hpp"
#include "rabitqlib/simd/rotator_dispatch.hpp"
#include "rabitqlib/simd/warmup_dispatch.hpp"
#include "rabitqlib/utils/cpu_features.hpp"

namespace rabitqlib::simd {

[[noreturn]] static void missing_feature(const char* feature_name) {
    throw std::runtime_error(std::string(feature_name) + " requires AVX2/FMA or AVX512 support");
}

static float unsupported_excode_ip(const float*, const uint8_t*, size_t) {
    missing_feature("excode ip functions");
}

static void unsupported_flip_sign(const uint8_t*, float*, size_t) {
    missing_feature("sign flip");
}

static void unsupported_kacs_walk(float*, size_t) {
    missing_feature("FhtKacRotator");
}

static void unsupported_scalar_quantize_uint8(uint8_t*, const float*, size_t, float, float) {
    missing_feature("uint8 quantize");
}

static void unsupported_scalar_quantize_uint16(uint16_t*, const float*, size_t, float, float) {
    missing_feature("uint16 quantize");
}

static void unsupported_pack_excode(const uint8_t*, uint8_t*, size_t) {
    missing_feature("excode packing");
}

static void unsupported_new_transpose_bin(const uint16_t*, uint64_t*, size_t, size_t) {
    missing_feature("new transpose bin");
}

static void unsupported_new_transpose_bin_512(const uint8_t*, uint64_t*, size_t, size_t) {
    missing_feature("new_transpose_bin_512");
}

static float unsupported_mask_ip_x0_q(const float*, const uint64_t*, size_t) {
    missing_feature("mask ip x0 q");
}

static void unsupported_accumulate(const uint8_t*, const uint8_t*, uint16_t*, size_t) {
    missing_feature("fastscan accumulate");
}

static void unsupported_transfer_lut_hacc(const uint16_t*, size_t, uint8_t*) {
    missing_feature("fastscan high-accuracy LUT transfer");
}

static void unsupported_accumulate_hacc(const uint8_t*, const uint8_t*, int32_t*, size_t) {
    missing_feature("fastscan high-accuracy accumulate");
}

static float unsupported_warmup_ip_x0_q_512(
    const uint64_t*, const uint64_t*, float, float, size_t, size_t
) {
    missing_feature("warmup_ip_x0_q_512");
}

ExcodeIpTable resolve_excode_ip_table() {
#if !defined(RABITQ_NO_AVX512)
    if (cpu::has_avx512_core()) {
        return {
            excode_ipimpl::ip16_fxu1_avx512,
            excode_ipimpl::ip16_fxu1_avx512,
            excode_ipimpl::ip64_fxu2_avx512,
            excode_ipimpl::ip64_fxu3_avx512,
            excode_ipimpl::ip16_fxu4_avx512,
            excode_ipimpl::ip64_fxu5_avx512,
            excode_ipimpl::ip64_fxu6_avx512,
            excode_ipimpl::ip64_fxu7_avx512,
            excode_ipimpl::ip16_fxu8_avx512,
        };
    }
#endif
    if (cpu::has_avx2()) {
        return {
            excode_ipimpl::ip16_fxu1_avx2,
            excode_ipimpl::ip16_fxu1_avx2,
            excode_ipimpl::ip64_fxu2_avx2,
            excode_ipimpl::ip64_fxu3_avx2,
            excode_ipimpl::ip16_fxu4_avx2,
            excode_ipimpl::ip64_fxu5_avx2,
            excode_ipimpl::ip64_fxu6_avx2,
            excode_ipimpl::ip64_fxu7_avx2,
            excode_ipimpl::ip16_fxu8_avx2,
        };
    } else {
        return {
            unsupported_excode_ip,
            unsupported_excode_ip,
            unsupported_excode_ip,
            unsupported_excode_ip,
            unsupported_excode_ip,
            unsupported_excode_ip,
            unsupported_excode_ip,
            unsupported_excode_ip,
            unsupported_excode_ip,
        };
    }
}

using FlipSignFn = void (*)(const uint8_t*, float*, size_t);
const FlipSignFn kFlipSignFn = [] {
#if !defined(RABITQ_NO_AVX512)
    if (cpu::has_avx512_core()) {
        return flip_sign_avx512;
    }
#endif
    if (cpu::has_avx2()) {
        return flip_sign_avx2;
    } else {
        return unsupported_flip_sign;
    }
}();

using KacsWalkFn = void (*)(float*, size_t);
const KacsWalkFn kKacsWalkFn = [] {
#if !defined(RABITQ_NO_AVX512)
    if (cpu::has_avx512_core()) {
        return kacs_walk_avx512;
    }
#endif
    if (cpu::has_avx2()) {
        return kacs_walk_avx2;
    } else {
        return unsupported_kacs_walk;
    }
}();

using ScalarQuantizeUint8Fn = void (*)(uint8_t*, const float*, size_t, float, float);
const ScalarQuantizeUint8Fn kScalarQuantizeUint8Fn = [] {
#if !defined(RABITQ_NO_AVX512)
    if (cpu::has_avx512_core()) {
        return scalar_quantize_uint8_avx512;
    }
#endif
    if (cpu::has_avx2()) {
        return scalar_quantize_uint8_avx2;
    } else {
        return unsupported_scalar_quantize_uint8;
    }
}();

using ScalarQuantizeUint16Fn = void (*)(uint16_t*, const float*, size_t, float, float);
const ScalarQuantizeUint16Fn kScalarQuantizeUint16Fn = [] {
#if !defined(RABITQ_NO_AVX512)
    if (cpu::has_avx512_core()) {
        return scalar_quantize_uint16_avx512;
    }
#endif
    if (cpu::has_avx2()) {
        return scalar_quantize_uint16_avx2;
    } else {
        return unsupported_scalar_quantize_uint16;
    }
}();

using PackExcodeFn = void (*)(const uint8_t*, uint8_t*, size_t);

static PackExcodeFn resolve_pack_excode_fn(PackExcodeFn avx512_fn, PackExcodeFn avx2_fn) {
#if !defined(RABITQ_NO_AVX512)
    if (cpu::has_avx512_core()) {
        return avx512_fn;
    }
#endif
    if (cpu::has_avx2()) {
        return avx2_fn;
    } else {
        return unsupported_pack_excode;
    }
}

#if defined(RABITQ_NO_AVX512)
const PackExcodeFn kPacking2BitExcodeFn =
    resolve_pack_excode_fn(nullptr, packing_2bit_excode_avx2);
const PackExcodeFn kPacking3BitExcodeFn =
    resolve_pack_excode_fn(nullptr, packing_3bit_excode_avx2);
const PackExcodeFn kPacking4BitExcodeFn =
    resolve_pack_excode_fn(nullptr, packing_4bit_excode_avx2);
const PackExcodeFn kPacking5BitExcodeFn =
    resolve_pack_excode_fn(nullptr, packing_5bit_excode_avx2);
const PackExcodeFn kPacking6BitExcodeFn =
    resolve_pack_excode_fn(nullptr, packing_6bit_excode_avx2);
const PackExcodeFn kPacking7BitExcodeFn =
    resolve_pack_excode_fn(nullptr, packing_7bit_excode_avx2);
#else
const PackExcodeFn kPacking2BitExcodeFn =
    resolve_pack_excode_fn(packing_2bit_excode_avx512, packing_2bit_excode_avx2);
const PackExcodeFn kPacking3BitExcodeFn =
    resolve_pack_excode_fn(packing_3bit_excode_avx512, packing_3bit_excode_avx2);
const PackExcodeFn kPacking4BitExcodeFn =
    resolve_pack_excode_fn(packing_4bit_excode_avx512, packing_4bit_excode_avx2);
const PackExcodeFn kPacking5BitExcodeFn =
    resolve_pack_excode_fn(packing_5bit_excode_avx512, packing_5bit_excode_avx2);
const PackExcodeFn kPacking6BitExcodeFn =
    resolve_pack_excode_fn(packing_6bit_excode_avx512, packing_6bit_excode_avx2);
const PackExcodeFn kPacking7BitExcodeFn =
    resolve_pack_excode_fn(packing_7bit_excode_avx512, packing_7bit_excode_avx2);
#endif

void flip_sign(const uint8_t* flip, float* data, size_t dim) {
    kFlipSignFn(flip, data, dim);
}

void kacs_walk(float* data, size_t len) {
    kKacsWalkFn(data, len);
}

void scalar_quantize_uint8(
    uint8_t* result, const float* vec0, size_t dim, float lo, float delta
) {
    kScalarQuantizeUint8Fn(result, vec0, dim, lo, delta);
}

void scalar_quantize_uint16(
    uint16_t* result, const float* vec0, size_t dim, float lo, float delta
) {
    kScalarQuantizeUint16Fn(result, vec0, dim, lo, delta);
}

void packing_2bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    kPacking2BitExcodeFn(o_raw, o_compact, dim);
}

void packing_3bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    kPacking3BitExcodeFn(o_raw, o_compact, dim);
}

void packing_4bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    kPacking4BitExcodeFn(o_raw, o_compact, dim);
}

void packing_5bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    kPacking5BitExcodeFn(o_raw, o_compact, dim);
}

void packing_6bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    kPacking6BitExcodeFn(o_raw, o_compact, dim);
}

void packing_7bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    kPacking7BitExcodeFn(o_raw, o_compact, dim);
}

}  // namespace rabitqlib::simd

namespace rabitqlib {

const simd::ExcodeIpTable kExcodeIpTable = simd::resolve_excode_ip_table();

const ex_ipfunc kIp16Fxu1AvxFn = kExcodeIpTable[1];
const ex_ipfunc kIp64Fxu2AvxFn = kExcodeIpTable[2];
const ex_ipfunc kIp64Fxu3AvxFn = kExcodeIpTable[3];
const ex_ipfunc kIp16Fxu4AvxFn = kExcodeIpTable[4];
const ex_ipfunc kIp64Fxu5AvxFn = kExcodeIpTable[5];
const ex_ipfunc kIp64Fxu6AvxFn = kExcodeIpTable[6];
const ex_ipfunc kIp64Fxu7AvxFn = kExcodeIpTable[7];

using NewTransposeBinFn = void (*)(const uint16_t*, uint64_t*, size_t, size_t);
const NewTransposeBinFn kNewTransposeBinFn = [] {
#if !defined(RABITQ_NO_AVX512)
    if (cpu::has_avx512_core()) {
        return simd::new_transpose_bin_avx512;
    }
#endif
    if (cpu::has_avx2()) {
        return simd::new_transpose_bin_avx2;
    } else {
        return simd::unsupported_new_transpose_bin;
    }
}();

using NewTransposeBin512Fn = void (*)(const uint8_t*, uint64_t*, size_t, size_t);
const NewTransposeBin512Fn kNewTransposeBin512Fn = [] {
#if !defined(RABITQ_NO_AVX512)
    if (cpu::has_avx512_core()) {
        return simd::new_transpose_bin_512_avx512;
    }
#endif
    if (cpu::has_avx2()) {
        return simd::new_transpose_bin_512_avx2;
    } else {
        return simd::unsupported_new_transpose_bin_512;
    }
}();

using MaskIpX0QFn = float (*)(const float*, const uint64_t*, size_t);
const MaskIpX0QFn kMaskIpX0QFn = [] {
#if !defined(RABITQ_NO_AVX512)
    if (cpu::has_avx512_core()) {
        return simd::mask_ip_x0_q_avx512;
    }
#endif
    if (cpu::has_avx2()) {
        return simd::mask_ip_x0_q_avx2;
    } else {
        return simd::unsupported_mask_ip_x0_q;
    }
}();

ex_ipfunc select_excode_ipfunc(size_t ex_bits) {
    if (ex_bits <= 8) {
        return kExcodeIpTable[ex_bits];
    }

    throw std::invalid_argument("Bad IP function for IVF");
}

float excode_ipimpl::ip16_fxu1_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return kIp16Fxu1AvxFn(query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu2_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return kIp64Fxu2AvxFn(query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu3_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return kIp64Fxu3AvxFn(query, compact_code, dim);
}

float excode_ipimpl::ip16_fxu4_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return kIp16Fxu4AvxFn(query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu5_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return kIp64Fxu5AvxFn(query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu6_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return kIp64Fxu6AvxFn(query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu7_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return kIp64Fxu7AvxFn(query, compact_code, dim);
}

void new_transpose_bin(
    const uint16_t* q, uint64_t* tq, size_t padded_dim, size_t b_query
) {
    kNewTransposeBinFn(q, tq, padded_dim, b_query);
}

void new_transpose_bin_512(
    const uint8_t* q, uint64_t* tq, size_t padded_dim, size_t b_query
) {
    kNewTransposeBin512Fn(q, tq, padded_dim, b_query);
}

float mask_ip_x0_q(const float* query, const uint64_t* data, size_t padded_dim) {
    return kMaskIpX0QFn(query, data, padded_dim);
}

}  // namespace rabitqlib

namespace rabitqlib::fastscan {

using AccumulateFn = void (*)(const uint8_t*, const uint8_t*, uint16_t*, size_t);
const AccumulateFn kAccumulateFn = [] {
#if !defined(RABITQ_NO_AVX512)
    if (cpu::has_avx512_core()) {
        return simd::accumulate_avx512;
    }
#endif
    if (cpu::has_avx2()) {
        return simd::accumulate_avx2;
    } else {
        return rabitqlib::simd::unsupported_accumulate;
    }
}();

using TransferLutHaccFn = void (*)(const uint16_t*, size_t, uint8_t*);
const TransferLutHaccFn kTransferLutHaccFn = [] {
#if !defined(RABITQ_NO_AVX512)
    if (cpu::has_avx512_core()) {
        return simd::transfer_lut_hacc_avx512;
    }
#endif
    if (cpu::has_avx2()) {
        return simd::transfer_lut_hacc_avx2;
    } else {
        return rabitqlib::simd::unsupported_transfer_lut_hacc;
    }
}();

using AccumulateHaccFn = void (*)(const uint8_t*, const uint8_t*, int32_t*, size_t);
const AccumulateHaccFn kAccumulateHaccFn = [] {
#if !defined(RABITQ_NO_AVX512)
    if (cpu::has_avx512_core()) {
        return simd::accumulate_hacc_avx512;
    }
#endif
    if (cpu::has_avx2()) {
        return simd::accumulate_hacc_avx2;
    } else {
        return rabitqlib::simd::unsupported_accumulate_hacc;
    }
}();

void accumulate(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lp_table,
    uint16_t* __restrict__ result,
    size_t dim
) {
    kAccumulateFn(codes, lp_table, result, dim);
}

void transfer_lut_hacc(const uint16_t* lut, size_t dim, uint8_t* hc_lut) {
    kTransferLutHaccFn(lut, dim, hc_lut);
}

void accumulate_hacc(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ hc_lut,
    int32_t* accu_res,
    size_t dim
) {
    kAccumulateHaccFn(codes, hc_lut, accu_res, dim);
}

}  // namespace rabitqlib::fastscan

namespace rabitqlib {

using WarmupIpX0Q512Fn = float (*)(const uint64_t*, const uint64_t*, float, float, size_t, size_t);
const WarmupIpX0Q512Fn kWarmupIpX0Q512Fn = [] {
#if !defined(RABITQ_NO_AVX512)
    if (rabitqlib::cpu::has_avx512_popcnt()) {
        return rabitqlib::simd::warmup_ip_x0_q_512_avx512;
    }
#endif
    if (rabitqlib::cpu::has_avx2()) {
        return rabitqlib::simd::warmup_ip_x0_q_512_avx2;
    } else {
        return rabitqlib::simd::unsupported_warmup_ip_x0_q_512;
    }
}();

float warmup_ip_x0_q_512(
    const uint64_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
) {
    return kWarmupIpX0Q512Fn(data, query, delta, vl, padded_dim, b_query);
}

}  // namespace rabitqlib
