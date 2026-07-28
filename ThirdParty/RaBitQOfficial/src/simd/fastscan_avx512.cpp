#include <immintrin.h>

#include <cstdint>

#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/fastscan/highacc_fastscan.hpp"

namespace rabitqlib::fastscan::simd {

void accumulate_avx512(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lp_table,
    uint16_t* __restrict__ result,
    size_t dim
) {
    size_t code_length = dim << 2;
    __m512i c;
    __m512i lo;
    __m512i hi;
    __m512i lut;
    __m512i res_lo;
    __m512i res_hi;

    const __m512i lo_mask = _mm512_set1_epi8(0x0f);
    __m512i accu0 = _mm512_setzero_si512();
    __m512i accu1 = _mm512_setzero_si512();
    __m512i accu2 = _mm512_setzero_si512();
    __m512i accu3 = _mm512_setzero_si512();

    // ! here, we assume the code_length is a multiple of 64, thus the dim must be a
    // ! multiple of 16
    for (size_t i = 0; i < code_length; i += 64) {
        c = _mm512_loadu_si512(&codes[i]);
        lut = _mm512_loadu_si512(&lp_table[i]);
        lo = _mm512_and_si512(c, lo_mask);                        // code of vector 0 to 15
        hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), lo_mask);  // code of vector 16 to 31

        res_lo = _mm512_shuffle_epi8(lut, lo);  // get the target value in lookup table
        res_hi = _mm512_shuffle_epi8(lut, hi);

        // since values in lookup table are represented as i8, we add them as i16 to avoid
        // overflow. Since the data order is 0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14,
        // 7, 15, accu0 accumulates for vec 8 to 15 (the upper 8 bits need to be updated
        // since they stored useless info of vec 0 to 7) accu1 accumulates for vec 0 to 7
        // similar for accu2 and accu3
        accu0 = _mm512_add_epi16(accu0, res_lo);
        accu1 = _mm512_add_epi16(accu1, _mm512_srli_epi16(res_lo, 8));
        accu2 = _mm512_add_epi16(accu2, res_hi);
        accu3 = _mm512_add_epi16(accu3, _mm512_srli_epi16(res_hi, 8));
    }
    // remove the influence of upper 8 bits for accu0 and accu2
    accu0 = _mm512_sub_epi16(accu0, _mm512_slli_epi16(accu1, 8));
    accu2 = _mm512_sub_epi16(accu2, _mm512_slli_epi16(accu3, 8));

    // At this point, we already have the correct accumulating result (accu0: 8-15, accu1:
    // 0-7, accu2: 16-23, accu3: 24-31), but we still need to write them back to RAM. Also,
    // each accu contains 4 lines of __m128i and we need to sum them together to get the
    // final results. 512/16=32, so we can use one __m512i to contain all results. The
    // following codes are designed for this purpose. For detailed information, please check
    // the SIMD documentation.
    __m512i ret1 = _mm512_add_epi16(
        _mm512_mask_blend_epi64(0b11110000, accu0, accu1),
        _mm512_shuffle_i64x2(accu0, accu1, 0b01001110)
    );
    __m512i ret2 = _mm512_add_epi16(
        _mm512_mask_blend_epi64(0b11110000, accu2, accu3),
        _mm512_shuffle_i64x2(accu2, accu3, 0b01001110)
    );
    __m512i ret = _mm512_setzero_si512();

    ret = _mm512_add_epi16(ret, _mm512_shuffle_i64x2(ret1, ret2, 0b10001000));
    ret = _mm512_add_epi16(ret, _mm512_shuffle_i64x2(ret1, ret2, 0b11011101));

    _mm512_storeu_si512(result, ret);
}

void transfer_lut_hacc_avx512(const uint16_t* lut, size_t dim, uint8_t* hc_lut) {
    size_t num_codebook = dim >> 2;

    for (size_t i = 0; i < num_codebook; i++) {
        constexpr size_t kRegBits = 512;
        constexpr size_t kLaneBits = 128;
        constexpr size_t kByteBits = 8;

        constexpr size_t kLutPerIter = kRegBits / kLaneBits;
        constexpr size_t kCodePerIter = 2 * kRegBits / kByteBits;
        constexpr size_t kCodePerLine = kLaneBits / kByteBits;

        uint8_t* fill_lo =
            hc_lut + (i / kLutPerIter * kCodePerIter) + ((i % kLutPerIter) * kCodePerLine);
        uint8_t* fill_hi = fill_lo + (kRegBits / kByteBits);

        __m512i tmp = _mm512_cvtepi16_epi32(
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lut))
        );
        __m128i lo = _mm512_cvtepi32_epi8(tmp);
        __m128i hi = _mm512_cvtepi32_epi8(_mm512_srli_epi32(tmp, 8));
        _mm_store_si128(reinterpret_cast<__m128i*>(fill_lo), lo);
        _mm_store_si128(reinterpret_cast<__m128i*>(fill_hi), hi);
        lut += 16;
    }
}

void accumulate_hacc_avx512(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ hc_lut,
    int32_t* accu_res,
    size_t dim
) {
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
}

}  // namespace rabitqlib::fastscan::simd
