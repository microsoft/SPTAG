#include "rabitqlib/simd/rotator_dispatch.hpp"

#include <immintrin.h>

#include <cstring>

namespace rabitqlib::simd {

void flip_sign_avx512(const uint8_t* flip, float* data, size_t dim) {
    constexpr size_t kFloatsPerChunk = 64;  // Process 64 floats per iteration

    static_assert(
        kFloatsPerChunk % 16 == 0,
        "floats_per_chunk must be divisible by AVX512 register width"
    );

    for (size_t i = 0; i < dim; i += kFloatsPerChunk) {
        // Load 64 bits (8 bytes) from the bit sequence
        uint64_t mask_bits;
        std::memcpy(&mask_bits, &flip[i / 8], sizeof(mask_bits));

        // Split into four 16-bit mask segments
        const __mmask16 mask0 = _cvtu32_mask16(static_cast<uint32_t>(mask_bits & 0xFFFF));
        const __mmask16 mask1 =
            _cvtu32_mask16(static_cast<uint32_t>((mask_bits >> 16) & 0xFFFF));
        const __mmask16 mask2 =
            _cvtu32_mask16(static_cast<uint32_t>((mask_bits >> 32) & 0xFFFF));
        const __mmask16 mask3 =
            _cvtu32_mask16(static_cast<uint32_t>((mask_bits >> 48) & 0xFFFF));

        // Prepare sign-flip constant
        const __m512 sign_flip = _mm512_castsi512_ps(_mm512_set1_epi32(0x80000000));

        // Process 16 floats at a time with each mask segment
        __m512 vec0 = _mm512_loadu_ps(&data[i]);
        vec0 = _mm512_mask_xor_ps(vec0, mask0, vec0, sign_flip);
        _mm512_storeu_ps(&data[i], vec0);

        __m512 vec1 = _mm512_loadu_ps(&data[i + 16]);
        vec1 = _mm512_mask_xor_ps(vec1, mask1, vec1, sign_flip);
        _mm512_storeu_ps(&data[i + 16], vec1);

        __m512 vec2 = _mm512_loadu_ps(&data[i + 32]);
        vec2 = _mm512_mask_xor_ps(vec2, mask2, vec2, sign_flip);
        _mm512_storeu_ps(&data[i + 32], vec2);

        __m512 vec3 = _mm512_loadu_ps(&data[i + 48]);
        vec3 = _mm512_mask_xor_ps(vec3, mask3, vec3, sign_flip);
        _mm512_storeu_ps(&data[i + 48], vec3);
    }
}

void kacs_walk_avx512(float* data, size_t len) {
    // ! len % 32 == 0;
    for (size_t i = 0; i < len / 2; i += 16) {
        __m512 x = _mm512_loadu_ps(&data[i]);
        __m512 y = _mm512_loadu_ps(&data[i + (len / 2)]);

        __m512 new_x = _mm512_add_ps(x, y);
        __m512 new_y = _mm512_sub_ps(x, y);

        _mm512_storeu_ps(&data[i], new_x);
        _mm512_storeu_ps(&data[i + (len / 2)], new_y);
    }
}

}  // namespace rabitqlib::simd
