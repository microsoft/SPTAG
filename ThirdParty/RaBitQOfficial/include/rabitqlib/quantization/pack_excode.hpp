#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "rabitqlib/simd/pack_excode_dispatch.hpp"

namespace rabitqlib::quant::rabitq_impl::ex_bits {
inline void packing_1bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    // ! require dim % 16 == 0
    for (size_t j = 0; j < dim; j += 16) {
        uint16_t code = 0;
        for (size_t i = 0; i < 16; ++i) {
            code |= static_cast<uint16_t>(o_raw[i]) << i;
        }
        std::memcpy(o_compact, &code, sizeof(uint16_t));

        o_raw += 16;
        o_compact += 2;
    }
}

inline void packing_2bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    ::rabitqlib::simd::packing_2bit_excode(o_raw, o_compact, dim);
}

inline void packing_3bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    ::rabitqlib::simd::packing_3bit_excode(o_raw, o_compact, dim);
}

inline void packing_4bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    ::rabitqlib::simd::packing_4bit_excode(o_raw, o_compact, dim);
}

inline void packing_5bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    ::rabitqlib::simd::packing_5bit_excode(o_raw, o_compact, dim);
}

inline void packing_6bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    ::rabitqlib::simd::packing_6bit_excode(o_raw, o_compact, dim);
}

inline void packing_7bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    ::rabitqlib::simd::packing_7bit_excode(o_raw, o_compact, dim);
}

inline void packing_8bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    std::memcpy(o_compact, o_raw, sizeof(uint8_t) * dim);
}

/**
 * @brief Packing ex_bits code to save space. For example, two 4-bit code will be
 * stored as 1 uint8. To compute inner product with the support of SIMD, the
 * packed codes need to be stored in different patterns. For details, please check the
 * code and comments for certain number of bits.
 *
 * @param o_raw unpacked code, code for each dim is represented by uint8
 * @param o_compact compact format of code
 * @param dim   dimension of code, NOTICE: different num of bits requried different
 *               dimension padding, dim should obey the corresponding requirement
 * @param ex_bits number of bits used for code
 */
inline void packing_rabitqplus_code(
    const uint8_t* o_raw, uint8_t* o_compact, size_t dim, size_t ex_bits
) {
    if (ex_bits == 1) {
        packing_1bit_excode(o_raw, o_compact, dim);
    } else if (ex_bits == 2) {
        packing_2bit_excode(o_raw, o_compact, dim);
    } else if (ex_bits == 3) {
        packing_3bit_excode(o_raw, o_compact, dim);
    } else if (ex_bits == 4) {
        packing_4bit_excode(o_raw, o_compact, dim);
    } else if (ex_bits == 5) {
        packing_5bit_excode(o_raw, o_compact, dim);
    } else if (ex_bits == 6) {
        packing_6bit_excode(o_raw, o_compact, dim);
    } else if (ex_bits == 7) {
        packing_7bit_excode(o_raw, o_compact, dim);
    } else if (ex_bits == 8) {
        packing_8bit_excode(o_raw, o_compact, dim);
    } else {
        std::cerr << "Bad value for ex_bits in packing_rabitqplus_code()\b";
        exit(1);
    }
}
}  // namespace rabitqlib::quant::rabitq_impl::ex_bits
