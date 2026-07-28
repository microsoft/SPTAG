#include <immintrin.h>

#include <cstdint>

#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/fastscan/highacc_fastscan.hpp"

namespace rabitqlib::fastscan::simd {

void accumulate_avx2(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lp_table,
    uint16_t* __restrict__ result,
    size_t dim
) {
    size_t code_length = dim << 2;
    __m256i c, lo, hi, lut, res_lo, res_hi;

    __m256i low_mask = _mm256_set1_epi8(0xf);
    __m256i accu0 = _mm256_setzero_si256();
    __m256i accu1 = _mm256_setzero_si256();
    __m256i accu2 = _mm256_setzero_si256();
    __m256i accu3 = _mm256_setzero_si256();

    for (size_t i = 0; i < code_length; i += 64) {
        c = _mm256_loadu_si256((__m256i*)&codes[i]);
        lut = _mm256_loadu_si256((__m256i*)&lp_table[i]);
        lo = _mm256_and_si256(c, low_mask);
        hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);

        res_lo = _mm256_shuffle_epi8(lut, lo);
        res_hi = _mm256_shuffle_epi8(lut, hi);

        accu0 = _mm256_add_epi16(accu0, res_lo);
        accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
        accu2 = _mm256_add_epi16(accu2, res_hi);
        accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));

        c = _mm256_loadu_si256((__m256i*)&codes[i + 32]);
        lut = _mm256_loadu_si256((__m256i*)&lp_table[i + 32]);
        lo = _mm256_and_si256(c, low_mask);
        hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);

        res_lo = _mm256_shuffle_epi8(lut, lo);
        res_hi = _mm256_shuffle_epi8(lut, hi);

        accu0 = _mm256_add_epi16(accu0, res_lo);
        accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
        accu2 = _mm256_add_epi16(accu2, res_hi);
        accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));
    }

    accu0 = _mm256_sub_epi16(accu0, _mm256_slli_epi16(accu1, 8));
    __m256i dis0 = _mm256_add_epi16(
        _mm256_permute2f128_si256(accu0, accu1, 0x21),
        _mm256_blend_epi32(accu0, accu1, 0xF0)
    );
    _mm256_storeu_si256((__m256i*)result, dis0);

    accu2 = _mm256_sub_epi16(accu2, _mm256_slli_epi16(accu3, 8));
    __m256i dis1 = _mm256_add_epi16(
        _mm256_permute2f128_si256(accu2, accu3, 0x21),
        _mm256_blend_epi32(accu2, accu3, 0xF0)
    );
    _mm256_storeu_si256((__m256i*)&result[16], dis1);
}

void transfer_lut_hacc_avx2(const uint16_t* lut, size_t dim, uint8_t* hc_lut) {
    size_t num_codebook = dim >> 2;

    for (size_t i = 0; i < num_codebook; i++) {
        constexpr size_t kRegBits = 256;
        constexpr size_t kLaneBits = 128;
        constexpr size_t kByteBits = 8;

        constexpr size_t kLutPerIter = kRegBits / kLaneBits;
        constexpr size_t kCodePerIter = 2 * kRegBits / kByteBits;
        constexpr size_t kCodePerLine = kLaneBits / kByteBits;

        uint8_t* fill_lo =
            hc_lut + (i / kLutPerIter * kCodePerIter) + ((i % kLutPerIter) * kCodePerLine);
        uint8_t* fill_hi = fill_lo + (kRegBits / kByteBits);

        for (size_t j = 0; j < 16; ++j) {
            int tmp = lut[j];
            uint8_t lo = static_cast<uint8_t>(tmp);
            uint8_t hi = static_cast<uint8_t>(tmp >> 8);
            fill_lo[j] = lo;
            fill_hi[j] = hi;
        }
        lut += 16;
    }
}

void accumulate_hacc_avx2(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ hc_lut,
    int32_t* accu_res,
    size_t dim
) {
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
}

}  // namespace rabitqlib::fastscan::simd
