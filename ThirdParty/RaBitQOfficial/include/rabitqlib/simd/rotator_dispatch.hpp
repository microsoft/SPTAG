#pragma once

#include <cstddef>
#include <cstdint>

namespace rabitqlib::simd {

void flip_sign_avx2(const uint8_t* flip, float* data, size_t dim);
void flip_sign_avx512(const uint8_t* flip, float* data, size_t dim);
void kacs_walk_avx2(float* data, size_t len);
void kacs_walk_avx512(float* data, size_t len);

void flip_sign(const uint8_t* flip, float* data, size_t dim);
void kacs_walk(float* data, size_t len);

}  // namespace rabitqlib::simd
