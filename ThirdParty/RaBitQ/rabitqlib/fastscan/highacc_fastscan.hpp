#pragma once

#include <immintrin.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace rabitqlib::fastscan {
/**
 * @brief Change u16 lookup table to u8. Since we use more bits (higher accuracy)
 * to quantize data vector by rabitq+, we also needs to increase the accuracy of data in
 * lut.
 * We split the higher & lower 8 bits of a u16 into two sub luts.
 **/
inline void transfer_lut_hacc(const uint16_t* lut, size_t dim, uint8_t* hc_lut) {
    size_t num_codebook = dim >> 2;

    for (size_t i = 0; i < num_codebook; i++) {
        // avx2 - 256, avx512 - 512
#if defined(__AVX512BW__)
        constexpr size_t kRegBits = 512;
#elif defined(__AVX2__)
        constexpr size_t kRegBits = 256;
#else
        static_assert(false, "At least requried AVX2 for using fastscan\n");
        exit(1);
#endif
        constexpr size_t kLaneBits = 128;
        constexpr size_t kByteBits = 8;

        constexpr size_t kLutPerIter = kRegBits / kLaneBits;
        constexpr size_t kCodePerIter = 2 * kRegBits / kByteBits;
        constexpr size_t kCodePerLine = kLaneBits / kByteBits;

        uint8_t* fill_lo =
            hc_lut + (i / kLutPerIter * kCodePerIter) + ((i % kLutPerIter) * kCodePerLine);
        uint8_t* fill_hi = fill_lo + (kRegBits / kByteBits);

#if defined(__AVX512BW__)
        __m512i tmp = _mm512_cvtepi16_epi32(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lut))
        );
        __m128i lo = _mm512_cvtepi32_epi8(tmp);
        __m128i hi = _mm512_cvtepi32_epi8(_mm512_srli_epi32(tmp, 8));
        _mm_store_si128(reinterpret_cast<__m128i*>(fill_lo), lo);
        _mm_store_si128(reinterpret_cast<__m128i*>(fill_hi), hi);
#else
        for (size_t j = 0; j < 16; ++j) {
            int tmp = lut[j];
            uint8_t lo = static_cast<uint8_t>(tmp);
            uint8_t hi = static_cast<uint8_t>(tmp >> 8);
            fill_lo[j] = lo;
            fill_hi[j] = hi;
        }
#endif
        lut += 16;
    }
}

inline void accumulate_hacc(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ hc_lut,
    int32_t* accu_res,
    size_t dim
) {
#if defined(__AVX512BW__)
    __m512i low_mask = _mm512_set1_epi8(0xf);
    __m512i accu[2][4];

    for (auto& a : accu) {
        for (auto& reg : a) {
            reg = _mm512_setzero_si512();
        }
    }

    size_t num_codebook = dim >> 2;

    // std::cerr << "FastScan YES!" << std::endl;
    for (size_t m = 0; m < num_codebook; m += 4) {
        __m512i c = _mm512_loadu_si512(codes);
        __m512i lo = _mm512_and_si512(c, low_mask);
        __m512i hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), low_mask);

        // accumulate lower & upper results respectively
        // accu[0][0-3] for lower 8-bit result
        // accu[1][0-3] for upper 8-bit result
        for (auto& i : accu) {
            __m512i lut = _mm512_loadu_si512(hc_lut);

            __m512i res_lo = _mm512_shuffle_epi8(lut, lo);
            __m512i res_hi = _mm512_shuffle_epi8(lut, hi);

            i[0] = _mm512_add_epi16(i[0], res_lo);
            i[1] = _mm512_add_epi16(i[1], _mm512_srli_epi16(res_lo, 8));

            i[2] = _mm512_add_epi16(i[2], res_hi);
            i[3] = _mm512_add_epi16(i[3], _mm512_srli_epi16(res_hi, 8));

            hc_lut += 64;
        }
        codes += 64;
    }

    // std::cerr << "FastScan YES!" << std::endl;

    __m512i res[2];
    __m512i dis0[2];
    __m512i dis1[2];

    for (size_t i = 0; i < 2; ++i) {
        __m256i tmp0 = _mm256_add_epi16(
            _mm512_castsi512_si256(accu[i][0]), _mm512_extracti64x4_epi64(accu[i][0], 1)
        );
        __m256i tmp1 = _mm256_add_epi16(
            _mm512_castsi512_si256(accu[i][1]), _mm512_extracti64x4_epi64(accu[i][1], 1)
        );
        tmp0 = _mm256_sub_epi16(tmp0, _mm256_slli_epi16(tmp1, 8));

        dis0[i] = _mm512_add_epi32(
            _mm512_cvtepu16_epi32(_mm256_permute2f128_si256(tmp0, tmp1, 0x21)),
            _mm512_cvtepu16_epi32(_mm256_blend_epi32(tmp0, tmp1, 0xF0))
        );

        __m256i tmp2 = _mm256_add_epi16(
            _mm512_castsi512_si256(accu[i][2]), _mm512_extracti64x4_epi64(accu[i][2], 1)
        );
        __m256i tmp3 = _mm256_add_epi16(
            _mm512_castsi512_si256(accu[i][3]), _mm512_extracti64x4_epi64(accu[i][3], 1)
        );
        tmp2 = _mm256_sub_epi16(tmp2, _mm256_slli_epi16(tmp3, 8));

        dis1[i] = _mm512_add_epi32(
            _mm512_cvtepu16_epi32(_mm256_permute2f128_si256(tmp2, tmp3, 0x21)),
            _mm512_cvtepu16_epi32(_mm256_blend_epi32(tmp2, tmp3, 0xF0))
        );
    }
    // shift res of high, add res of low
    res[0] =
        _mm512_add_epi32(dis0[0], _mm512_slli_epi32(dis0[1], 8));  // res for vec 0 to 15
    res[1] =
        _mm512_add_epi32(dis1[0], _mm512_slli_epi32(dis1[1], 8));  // res for vec 16 to 31

    _mm512_storeu_epi32(accu_res, res[0]);
    _mm512_storeu_epi32(accu_res + 16, res[1]);
#elif defined(__AVX2__)
    __m256i low_mask = _mm256_set1_epi8(0xf);
    __m256i accu[2][4];

    for (auto& a : accu) {
        for (auto& reg : a) {
            reg = _mm256_setzero_si256();
        }
    }

    size_t num_codebook = dim >> 2;

    for (size_t m = 0; m < num_codebook; m += 2) {
        __m256i c = _mm256_loadu_si256((__m256i*)codes);
        codes += 32;

        __m256i lo = _mm256_and_si256(c, low_mask);
        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);

        for (int q = 0; q < 2; ++q) {
            __m256i lut = _mm256_loadu_si256((__m256i*)hc_lut);
            hc_lut += 32;

            __m256i res_lo = _mm256_shuffle_epi8(lut, lo);
            __m256i res_hi = _mm256_shuffle_epi8(lut, hi);

            accu[q][0] = _mm256_add_epi16(accu[q][0], res_lo);
            accu[q][1] = _mm256_add_epi16(accu[q][1], _mm256_srli_epi16(res_lo, 8));
            accu[q][2] = _mm256_add_epi16(accu[q][2], res_hi);
            accu[q][3] = _mm256_add_epi16(accu[q][3], _mm256_srli_epi16(res_hi, 8));
        }
    }

    __m256i res[4];
    __m256i dis0[2];
    __m256i dis1[2];

    // lamda function to horizontal sum and combine low/high bytes
    auto combine2x2 = [&](__m256i a, __m256i b) -> __m256i {
        __m256i a1b0 = _mm256_permute2f128_si256(a, b, 0x21);
        __m256i a0b1 = _mm256_blend_epi32(a, b, 0xF0);
        return _mm256_add_epi16(a1b0, a0b1);
    };

    for (size_t i = 0; i < 2; i++) {
        accu[i][0] = _mm256_sub_epi16(accu[i][0], _mm256_slli_epi16(accu[i][1], 8));
        dis0[i] = combine2x2(accu[i][0], accu[i][1]);

        accu[i][2] = _mm256_sub_epi16(accu[i][2], _mm256_slli_epi16(accu[i][3], 8));
        dis1[i] = combine2x2(accu[i][2], accu[i][3]);
    }

    auto add_shiftl8 = [&](__m256i a, __m256i b, __m256i& r0, __m256i& r1) {
        __m256i a0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(a));
        __m256i a1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(a, 1));
        __m256i b0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(b));
        __m256i b1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(b, 1));
        r0 = _mm256_add_epi32(a0, _mm256_slli_epi32(b0, 8));
        r1 = _mm256_add_epi32(a1, _mm256_slli_epi32(b1, 8));
    };

    // shift res of high, add res of low
    add_shiftl8(dis0[0], dis0[1], res[0], res[1]);  // res for vec 0 to 15
    add_shiftl8(dis1[0], dis1[1], res[2], res[3]);  // res for vec 16 to 31

    _mm256_storeu_si256((__m256i*)(accu_res), res[0]);
    _mm256_storeu_si256((__m256i*)(accu_res + 8), res[1]);
    _mm256_storeu_si256((__m256i*)(accu_res + 16), res[2]);
    _mm256_storeu_si256((__m256i*)(accu_res + 24), res[3]);
#else
    std::cerr << "AVX512 or AVX2 required for using fastscan\n";
    exit(1);
#endif
}
}  // namespace rabitqlib::fastscan