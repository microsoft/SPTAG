#pragma once

#include <immintrin.h>
#include <omp.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>

namespace rabitqlib::quant::rabitq_impl::ex_bits {
inline void packing_1bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
#if defined(__AVX512F__) || defined(__AVX2__)
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
#else
    std::cerr << "Current only support AVX512F and AVX2 for packing excode\n" << std::flush;
    exit(1);
#endif
}

inline void packing_2bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
#if defined(__AVX512F__) || defined(__AVX2__)
    // ! require dim % 16 == 0
    for (size_t j = 0; j < dim; j += 64) {
        // pack 64 2-bit codes into 128 bits (16 bytes)
        // the lower 2 bits of each byte represent vec00 to vec04...

        __m128i vec_00_to_15 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw));
        __m128i vec_16_to_31 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw + 16));
        __m128i vec_32_to_47 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw + 32));
        __m128i vec_48_to_63 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw + 48));

        __m128i compact = _mm_or_si128(
            _mm_or_si128(vec_00_to_15, _mm_slli_epi16(vec_16_to_31, 2)),
            _mm_or_si128(_mm_slli_epi16(vec_32_to_47, 4), _mm_slli_epi16(vec_48_to_63, 6))
        );

        _mm_storeu_si128(reinterpret_cast<__m128i*>(o_compact), compact);
        
        o_raw += 64;
        o_compact += 16;
    }
#else
    std::cerr << "Current only support AVX512F and AVX2 for packing excode\n" << std::flush;
    exit(1);
#endif
}

inline void packing_3bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
#if defined(__AVX512F__) || defined(__AVX2__)
    // ! require dim % 64 == 0
    const __m128i mask = _mm_set1_epi8(0b11);
    for (size_t d = 0; d < dim; d += 64) {
        // split 3-bit codes into 2 bits and 1 bit
        // for 2-bit part, compact it like 2-bit code
        // for 1-bit part, compact 64 1-bit code into a int64
        __m128i vec_00_to_15 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw));
        __m128i vec_16_to_31 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw + 16));
        __m128i vec_32_to_47 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw + 32));
        __m128i vec_48_to_63 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw + 48));

        vec_00_to_15 = _mm_and_si128(vec_00_to_15, mask);
        vec_16_to_31 = _mm_slli_epi16(_mm_and_si128(vec_16_to_31, mask), 2);
        vec_32_to_47 = _mm_slli_epi16(_mm_and_si128(vec_32_to_47, mask), 4);
        vec_48_to_63 = _mm_slli_epi16(_mm_and_si128(vec_48_to_63, mask), 6);

        __m128i compact2 = _mm_or_si128(
            _mm_or_si128(vec_00_to_15, vec_16_to_31),
            _mm_or_si128(vec_32_to_47, vec_48_to_63)
        );

        _mm_storeu_si128(reinterpret_cast<__m128i*>(o_compact), compact2);
        o_compact += 16;

        // from lower to upper, each bit in each byte represents vec00 to vec07,
        // ..., vec56 to vec63
        int64_t top_bit = 0;
        constexpr int64_t kMask1 = 0x0101010101010101;
        for (size_t i = 0; i < 64; i += 8) {
            int64_t cur_codes = *reinterpret_cast<const int64_t*>(o_raw + i);
            top_bit |= ((cur_codes >> 2) & kMask1) << (i / 8);
        }
        std::memcpy(o_compact, &top_bit, sizeof(int64_t));

        o_raw += 64;
        o_compact += 8;
    }
#else
    std::cerr << "Current only support AVX512F and AVX2 for packing excode\n" << std::flush;
    exit(1);
#endif
}

inline void packing_4bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
// although this part only requries SSE, computing inner product for this orgnization
// requires AVX512F, similar for remaining functions
#if defined(__AVX512F__) || defined(__AVX2__)
    // ! require dim % 16 == 0
    for (size_t j = 0; j < dim; j += 16) {
        // pack 16 4-bit codes into uint64
        // the lower 4 bits represent vec00 to vec07
        int64_t code0 = *reinterpret_cast<const int64_t*>(o_raw);
        // the upper 4 bits represent vec08 to vec15
        int64_t code1 = *reinterpret_cast<const int64_t*>(o_raw + 8);

        int64_t compact = (code1 << 4) | code0;

        *reinterpret_cast<int64_t*>(o_compact) = compact;

        o_raw += 16;
        o_compact += 8;
    }
#else
    std::cerr << "Current only support AVX512F and AVX2 for packing excode\n" << std::flush;
    exit(1);
#endif
}

inline void packing_5bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
#if defined(__AVX512F__) || defined(__AVX2__)
    // ! require dim % 64 == 0
    const __m128i mask = _mm_set1_epi8(0b1111);
    for (size_t j = 0; j < dim; j += 64) {
        __m128i vec_00_to_15 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw));
        __m128i vec_16_to_31 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw + 16));
        __m128i vec_32_to_47 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw + 32));
        __m128i vec_48_to_63 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw + 48));

        vec_00_to_15 = _mm_and_si128(vec_00_to_15, mask);
        vec_16_to_31 = _mm_slli_epi16(_mm_and_si128(vec_16_to_31, mask), 4);
        vec_32_to_47 = _mm_and_si128(vec_32_to_47, mask);
        vec_48_to_63 = _mm_slli_epi16(_mm_and_si128(vec_48_to_63, mask), 4);

        __m128i compact4_1 = _mm_or_si128(vec_00_to_15, vec_16_to_31);
        __m128i compact4_2 = _mm_or_si128(vec_32_to_47, vec_48_to_63);

        _mm_storeu_si128(reinterpret_cast<__m128i*>(o_compact), compact4_1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(o_compact + 16), compact4_2);

        o_compact += 32;

        // from lower to upper, each bit in each byte represents vec00 to vec07,
        // ..., vec56 to vec63
        int64_t top_bit = 0;
        constexpr int64_t kMask1 = 0x0101010101010101;
        for (size_t i = 0; i < 64; i += 8) {
            int64_t cur_codes = *reinterpret_cast<const int64_t*>(o_raw + i);
            top_bit |= ((cur_codes >> 4) & kMask1) << (i / 8);
        }
        std::memcpy(o_compact, &top_bit, sizeof(int64_t));

        o_raw += 64;
        o_compact += 8;
    }
#else
    std::cerr << "Current only support AVX512F and AVX2 for packing excode\n" << std::flush;
    exit(1);
#endif
}

inline void packing_6bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
#if defined(__AVX512F__) || defined(__AVX2__)
    // for vec00 to vec47, split code into 6
    // for vec48 to vec63, split code into 2 + 2 + 2
    const __m128i mask2 = _mm_set1_epi8(static_cast<char>(0b11000000));
    const __m128i mask6 = _mm_set1_epi8(0b00111111);
    for (size_t d = 0; d < dim; d += 64) {
        __m128i vec_00_to_15 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw));
        __m128i vec_16_to_31 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw + 16));
        __m128i vec_32_to_47 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw + 32));
        __m128i vec_48_to_63 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw + 48));

        __m128i compact = _mm_or_si128(
            _mm_and_si128(vec_00_to_15, mask6),
            _mm_and_si128(_mm_slli_epi16(vec_48_to_63, 6), mask2)
        );
        _mm_storeu_si128(reinterpret_cast<__m128i*>(o_compact), compact);

        compact = _mm_or_si128(
            _mm_and_si128(vec_16_to_31, mask6),
            _mm_and_si128(_mm_slli_epi16(vec_48_to_63, 4), mask2)
        );
        _mm_storeu_si128(reinterpret_cast<__m128i*>(o_compact + 16), compact);

        compact = _mm_or_si128(
            _mm_and_si128(vec_32_to_47, mask6),
            _mm_and_si128(_mm_slli_epi16(vec_48_to_63, 2), mask2)
        );
        _mm_storeu_si128(reinterpret_cast<__m128i*>(o_compact + 32), compact);
        o_compact += 48;
        o_raw += 64;
    }
#else
    std::cerr << "Current only support AVX512F and AVX2 for packing excode\n" << std::flush;
    exit(1);
#endif
}

inline void packing_7bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
#if defined(__AVX512F__) || defined(__AVX2__)
    // for vec00 to vec47, split code into 6 + 1
    // for vec48 to vec63, split code into 2 + 2 + 2 + 1
    const __m128i mask2 = _mm_set1_epi8(static_cast<char>(0b11000000));
    const __m128i mask6 = _mm_set1_epi8(0b00111111);
    for (size_t d = 0; d < dim; d += 64) {
        __m128i vec_00_to_15 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw));
        __m128i vec_16_to_31 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw + 16));
        __m128i vec_32_to_47 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw + 32));
        __m128i vec_48_to_63 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw + 48));

        __m128i compact = _mm_or_si128(
            _mm_and_si128(vec_00_to_15, mask6),
            _mm_and_si128(_mm_slli_epi16(vec_48_to_63, 6), mask2)
        );
        _mm_storeu_si128(reinterpret_cast<__m128i*>(o_compact), compact);

        compact = _mm_or_si128(
            _mm_and_si128(vec_16_to_31, mask6),
            _mm_and_si128(_mm_slli_epi16(vec_48_to_63, 4), mask2)
        );
        _mm_storeu_si128(reinterpret_cast<__m128i*>(o_compact + 16), compact);

        compact = _mm_or_si128(
            _mm_and_si128(vec_32_to_47, mask6),
            _mm_and_si128(_mm_slli_epi16(vec_48_to_63, 2), mask2)
        );
        _mm_storeu_si128(reinterpret_cast<__m128i*>(o_compact + 32), compact);
        o_compact += 48;

        int64_t top_bit = 0;
        int64_t top_mask = 0x0101010101010101;
        for (size_t i = 0; i < 64; i += 8) {
            int64_t cur_codes = *reinterpret_cast<const int64_t*>(o_raw + i);
            top_bit |= ((cur_codes >> 6) & top_mask) << (i / 8);
        }
        std::memcpy(o_compact, &top_bit, sizeof(int64_t));

        o_compact += 8;
        o_raw += 64;
    }
#else
    std::cerr << "Current only support AVX512F and AVX2 for packing excode\n" << std::flush;
    exit(1);
#endif
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