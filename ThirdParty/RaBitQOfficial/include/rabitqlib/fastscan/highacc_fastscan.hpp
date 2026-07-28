#pragma once

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "rabitqlib/utils/compiler_compat.hpp"

namespace rabitqlib::fastscan {
/**
 * @brief Change u16 lookup table to u8. Since we use more bits (higher accuracy)
 * to quantize data vector by rabitq+, we also needs to increase the accuracy of data in
 * lut.
 * We split the higher & lower 8 bits of a u16 into two sub luts.
 **/
void transfer_lut_hacc(const uint16_t* lut, size_t dim, uint8_t* hc_lut);

void accumulate_hacc(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ hc_lut,
    int32_t* accu_res,
    size_t dim
);
}  // namespace rabitqlib::fastscan
