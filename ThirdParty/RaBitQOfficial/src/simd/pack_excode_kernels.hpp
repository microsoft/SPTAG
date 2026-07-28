#pragma once

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace rabitqlib::simd::detail {

inline void packing_2bit_excode_intrinsics(
    const uint8_t* o_raw, uint8_t* o_compact, size_t dim
) {
    // ! require dim % 64 == 0
    for (size_t j = 0; j < dim; j += 64) {
        // pack 64 2-bit codes into 128 bits (16 bytes)
        // the lower 2 bits of each byte represent vec00 to vec04...
        __m128i vec_00_to_15 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw));
        __m128i vec_16_to_31 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw + 16));
        __m128i vec_32_to_47 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw + 32));
        __m128i vec_48_to_63 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(o_raw + 48));

        __m128i compact = _mm_or_si128(
            _mm_or_si128(vec_00_to_15, _mm_slli_epi16(vec_16_to_31, 2)),
            _mm_or_si128(_mm_slli_epi16(vec_32_to_47, 4), _mm_slli_epi16(vec_48_to_63, 6))
        );

        _mm_storeu_si128(reinterpret_cast<__m128i*>(o_compact), compact);

        o_raw += 64;
        o_compact += 16;
    }
}

inline void packing_3bit_excode_intrinsics(
    const uint8_t* o_raw, uint8_t* o_compact, size_t dim
) {
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
        uint64_t top_bit = 0;
        constexpr uint64_t kMask1 = 0x0101010101010101ULL;
        for (size_t i = 0; i < 64; i += 8) {
            uint64_t cur_codes;
            std::memcpy(&cur_codes, o_raw + i, sizeof(cur_codes));
            top_bit |= ((cur_codes >> 2) & kMask1) << (i / 8);
        }
        std::memcpy(o_compact, &top_bit, sizeof(uint64_t));

        o_raw += 64;
        o_compact += 8;
    }
}

inline void packing_4bit_excode_intrinsics(
    const uint8_t* o_raw, uint8_t* o_compact, size_t dim
) {
    // although this part only requries SSE, computing inner product for this orgnization
    // requires AVX512F, similar for remaining functions
    // ! require dim % 16 == 0
    for (size_t j = 0; j < dim; j += 16) {
        // pack 16 4-bit codes into uint64
        // the lower 4 bits represent vec00 to vec07
        uint64_t code0;
        std::memcpy(&code0, o_raw, sizeof(code0));
        // the upper 4 bits represent vec08 to vec15
        uint64_t code1;
        std::memcpy(&code1, o_raw + 8, sizeof(code1));

        uint64_t compact = (code1 << 4) | code0;
        std::memcpy(o_compact, &compact, sizeof(compact));

        o_raw += 16;
        o_compact += 8;
    }
}

inline void packing_5bit_excode_intrinsics(
    const uint8_t* o_raw, uint8_t* o_compact, size_t dim
) {
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
        uint64_t top_bit = 0;
        constexpr uint64_t kMask1 = 0x0101010101010101ULL;
        for (size_t i = 0; i < 64; i += 8) {
            uint64_t cur_codes;
            std::memcpy(&cur_codes, o_raw + i, sizeof(cur_codes));
            top_bit |= ((cur_codes >> 4) & kMask1) << (i / 8);
        }
        std::memcpy(o_compact, &top_bit, sizeof(uint64_t));

        o_raw += 64;
        o_compact += 8;
    }
}

inline void packing_6bit_excode_intrinsics(
    const uint8_t* o_raw, uint8_t* o_compact, size_t dim
) {
    // for vec00 to vec47, split code into 6
    // for vec48 to vec63, split code into 2 + 2 + 2
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
        o_raw += 64;
    }
}

inline void packing_7bit_excode_intrinsics(
    const uint8_t* o_raw, uint8_t* o_compact, size_t dim
) {
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

        uint64_t top_bit = 0;
        constexpr uint64_t top_mask = 0x0101010101010101ULL;
        for (size_t i = 0; i < 64; i += 8) {
            uint64_t cur_codes;
            std::memcpy(&cur_codes, o_raw + i, sizeof(cur_codes));
            top_bit |= ((cur_codes >> 6) & top_mask) << (i / 8);
        }
        std::memcpy(o_compact, &top_bit, sizeof(uint64_t));

        o_compact += 8;
        o_raw += 64;
    }
}

}  // namespace rabitqlib::simd::detail
