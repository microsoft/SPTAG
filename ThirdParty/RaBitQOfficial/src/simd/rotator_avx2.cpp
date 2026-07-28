#include "rabitqlib/simd/rotator_dispatch.hpp"

#include <immintrin.h>

#include <cstring>

namespace rabitqlib::simd {

void flip_sign_avx2(const uint8_t* flip, float* data, size_t dim) {
    // Process 32 floats (4 AVX2 registers) per iteration
    constexpr size_t kFloatsPerChunk = 32;

    const __m256i bit_select = _mm256_setr_epi32(
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80
    );
    const __m256 sign_flip = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

    // Utility lambda to create a mask for flipping signs
    auto create_mask = [&](uint8_t byte_mask) -> __m256 {
        __m256i mask_bits = _mm256_set1_epi32(byte_mask);
        __m256i test = _mm256_and_si256(mask_bits, bit_select);
        __m256i cmp = _mm256_cmpeq_epi32(test, bit_select);
        return _mm256_and_ps(_mm256_castsi256_ps(cmp), sign_flip);
    };

    for (size_t i = 0; i < dim; i += kFloatsPerChunk) {
        uint32_t mask_bits;
        std::memcpy(&mask_bits, &flip[i / 8], sizeof(mask_bits));

        for (int b = 0; b < 4; ++b) {
            __m256 xor_mask = create_mask((mask_bits >> (b * 8)) & 0xFF);
            __m256 vec = _mm256_loadu_ps(&data[i + b * 8]);
            vec = _mm256_xor_ps(vec, xor_mask);
            _mm256_storeu_ps(&data[i + b * 8], vec);
        }
    }
}

void kacs_walk_avx2(float* data, size_t len) {
    // ! len % 16 == 0;
    for (size_t i = 0; i < len / 2; i += 8) {
        __m256 x = _mm256_loadu_ps(&data[i]);
        __m256 y = _mm256_loadu_ps(&data[i + (len / 2)]);

        __m256 new_x = _mm256_add_ps(x, y);
        __m256 new_y = _mm256_sub_ps(x, y);

        _mm256_storeu_ps(&data[i], new_x);
        _mm256_storeu_ps(&data[i + (len / 2)], new_y);
    }
}

}  // namespace rabitqlib::simd
