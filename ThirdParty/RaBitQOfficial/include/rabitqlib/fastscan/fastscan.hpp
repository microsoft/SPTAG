// The implementation is largely based on the implementation of Faiss.
// https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)

#pragma once

#include <algorithm>
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
void accumulate(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lp_table,
    uint16_t* __restrict__ result,
    size_t dim
);

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
