#pragma once

#include <cstddef>
#include <cstdint>

#include "rabitqlib/utils/compiler_compat.hpp"

namespace rabitqlib::simd {
namespace excode_ipimpl {

float ip16_fxu1_avx2(const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim);
float ip64_fxu2_avx2(const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim);
float ip64_fxu3_avx2(const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim);
float ip16_fxu4_avx2(const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim);
float ip64_fxu5_avx2(const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim);
float ip64_fxu6_avx2(const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim);
float ip64_fxu7_avx2(const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim);
float ip16_fxu8_avx2(const float* __restrict__ query, const uint8_t* __restrict__ code, size_t dim);

float ip16_fxu1_avx512(const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim);
float ip64_fxu2_avx512(const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim);
float ip64_fxu3_avx512(const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim);
float ip16_fxu4_avx512(const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim);
float ip64_fxu5_avx512(const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim);
float ip64_fxu6_avx512(const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim);
float ip64_fxu7_avx512(const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim);
float ip16_fxu8_avx512(const float* __restrict__ query, const uint8_t* __restrict__ code, size_t dim);

}  // namespace excode_ipimpl

void new_transpose_bin_avx2(const uint16_t* q, uint64_t* tq, size_t padded_dim, size_t b_query);
void new_transpose_bin_512_avx2(const uint8_t* q, uint64_t* tq, size_t padded_dim, size_t b_query);
float mask_ip_x0_q_avx2(const float* query, const uint64_t* data, size_t padded_dim);
void scalar_quantize_uint8_avx2(uint8_t* result, const float* vec0, size_t dim, float lo, float delta);
void scalar_quantize_uint16_avx2(uint16_t* result, const float* vec0, size_t dim, float lo, float delta);

void new_transpose_bin_avx512(const uint16_t* q, uint64_t* tq, size_t padded_dim, size_t b_query);
void new_transpose_bin_512_avx512(const uint8_t* q, uint64_t* tq, size_t padded_dim, size_t b_query);
float mask_ip_x0_q_avx512(const float* query, const uint64_t* data, size_t padded_dim);
void scalar_quantize_uint8_avx512(uint8_t* result, const float* vec0, size_t dim, float lo, float delta);
void scalar_quantize_uint16_avx512(uint16_t* result, const float* vec0, size_t dim, float lo, float delta);

void scalar_quantize_uint8(uint8_t* result, const float* vec0, size_t dim, float lo, float delta);
void scalar_quantize_uint16(uint16_t* result, const float* vec0, size_t dim, float lo, float delta);

}  // namespace rabitqlib::simd
