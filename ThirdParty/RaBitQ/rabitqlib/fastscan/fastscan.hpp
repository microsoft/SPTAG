// The implementation is largely based on the implementation of Faiss.
// https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)

#pragma once

#include <immintrin.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "rabitqlib/defines.hpp"

namespace rabitqlib::fastscan {

constexpr static size_t kBatchSize = 32;  // number of vectors in each batch

constexpr static std::array<int, 16> kPos = {
    3 /*0000*/,
    3 /*0001*/,
    2 /*0010*/,
    3 /*0011*/,
    1 /*0100*/,
    3 /*0101*/,
    2 /*0110*/,
    3 /*0111*/,
    0 /*1000*/,
    3 /*1001*/,
    2 /*1010*/,
    3 /*1011*/,
    1 /*1100*/,
    3 /*1101*/,
    2 /*1110*/,
    3 /*1111*/,
};  // all possible combination for a 4 bit string

constexpr static std::array<int, 16> kPerm0 = {
    0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15
};  // data order of packed quantization code, please refer to code and the link offered
    // above for detailed information

template <typename T, class TA>
static inline void get_column(
    const T* src, size_t rows, size_t cols, size_t row, size_t col, TA& dest
) {
    size_t k = 0;
    size_t max_k = std::min(rows - row, dest.size());
    for (; k < max_k; ++k) {
        dest[k] = src[((k + row) * cols) + col];
    }
    if (k < dest.size()) {
        std::fill(dest.begin() + k, dest.end(), 0);
    }
}

/**
 * @brief Pack quantization codes, store in blocks, the data orgnization is illustrated in
 * the link and kPerm0. Since we pack codes as 32-sized groups, if the num is not a multiple
 * of 32, we have to use some space for these absent data
 *
 * @param padded_dim dimension of quantized data (i.e., quantization code)
 * @param quantization_code quantizaiton code, stored as uint8
 * @param num   number of quantization code
 * @param blocks packed quantization code
 */
inline void pack_codes(
    size_t padded_dim, const uint8_t* quantization_code, size_t num, uint8_t* blocks
) {
    size_t num_rd = (num + 31) & ~31;  // round up num of vecs to multiple of batch size(32)

    // consider codes is a matrix
    // rows = number of vectors
    // cols = number of uint8_t of one vector's code
    size_t cols = padded_dim / 8;

    std::array<uint8_t, 32> col;    // column of a batch of code, 8 bits
    std::array<uint8_t, 32> col_0;  // upper 4 bits
    std::array<uint8_t, 32> col_1;  // lower 4 bits

    // pack codes batch by batch
    // each batch contain codes for 32 vectors
    for (size_t row = 0; row < num_rd; row += kBatchSize) {
        // get quantization codes for each column for each batch
        // i.e., we get the codes for 8 dims of 32 vectors and re-orgnize the data layout
        // based on the shuffle SIMD instruction used during quering
        for (size_t i = 0; i < cols; ++i) {
            get_column(quantization_code, num, cols, row, i, col);
            for (size_t j = 0; j < 32; ++j) {
                col_0[j] = col[j] >> 4;
                col_1[j] = col[j] & 15;
            }
            for (size_t j = 0; j < 16; ++j) {
                // the lower 4 bits represent vector 0 to 15
                // the upper 4 bits represent vector 16 to 31
                uint8_t val0 = col_0[kPerm0[j]] | (col_0[kPerm0[j] + 16] << 4);
                uint8_t val1 = col_1[kPerm0[j]] | (col_1[kPerm0[j] + 16] << 4);
                blocks[j] = val0;
                blocks[j + 16] = val1;
            }
            blocks += 32;
        }
    }
}

// use fast scan to accumulate one block, dim % 16 == 0
inline void accumulate(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lp_table,
    uint16_t* __restrict__ result,
    size_t dim
) {
    size_t code_length = dim << 2;
#if defined(__AVX512BW__)
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

#elif defined(__AVX2__)
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
#else
    std::cerr << "no avx simd supported!\n";
    exit(1);
#endif
}

// pack lookup table for fastscan, for each 4 dim, we have 16 (2^4) different results
// ! dim % 4 == 0
template <typename T>
inline void pack_lut(size_t dim, const T* __restrict__ query, T* __restrict__ lut) {
    size_t num_codebook = dim >> 2;
    for (size_t i = 0; i < num_codebook; ++i) {
        lut[0] = 0;
        for (size_t j = 1; j < 16; ++j) {
            lut[j] = lut[j - LOWBIT(j)] + query[kPos[j]];
        }
        lut += 16;
        query += 4;
    }
}
}  // namespace rabitqlib::fastscan