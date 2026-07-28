#pragma once

#include <cstddef>
#include <cstdint>

#include "rabitqlib/utils/compiler_compat.hpp"

namespace rabitqlib::fastscan::simd {

void accumulate_avx2(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lp_table,
    uint16_t* __restrict__ result,
    size_t dim
);
void transfer_lut_hacc_avx2(const uint16_t* lut, size_t dim, uint8_t* hc_lut);
void accumulate_hacc_avx2(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ hc_lut,
    int32_t* accu_res,
    size_t dim
);

void accumulate_avx512(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lp_table,
    uint16_t* __restrict__ result,
    size_t dim
);
void transfer_lut_hacc_avx512(const uint16_t* lut, size_t dim, uint8_t* hc_lut);
void accumulate_hacc_avx512(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ hc_lut,
    int32_t* accu_res,
    size_t dim
);

}  // namespace rabitqlib::fastscan::simd
