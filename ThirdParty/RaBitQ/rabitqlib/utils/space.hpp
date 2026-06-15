#pragma once

#include <emmintrin.h>
#include <immintrin.h>
#include <omp.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <type_traits>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/tools.hpp"

namespace rabitqlib {
namespace scalar_impl {
template <typename T>
void scalar_quantize_normal(
    T* __restrict__ result,
    const float* __restrict__ vec0,
    size_t dim,
    float lo,
    float delta
) {
    float one_over_delta = 1.0F / delta;

    ConstRowMajorArrayMap<float> v0(vec0, 1, static_cast<long>(dim));
    RowMajorArrayMap<T> res(result, 1, dim);

    // round to nearest integer, then cast to integer
    res = ((v0 - lo) * one_over_delta).round().template cast<T>();
}

template <typename T>
void scalar_quantize_optimized(
    T* __restrict__ result,
    const float* __restrict__ vec0,
    size_t dim,
    float lo,
    float delta
) {
    scalar_quantize_normal(result, vec0, dim, lo, delta);
}

template <>
inline void scalar_quantize_optimized<uint8_t>(
    uint8_t* __restrict__ result,
    const float* __restrict__ vec0,
    size_t dim,
    float lo,
    float delta
) {
#if defined(__AVX512F__)
    size_t mul16 = dim - (dim & 0b1111);
    size_t i = 0;
    float one_over_delta = 1 / delta;
    auto lo512 = _mm512_set1_ps(lo);
    auto od512 = _mm512_set1_ps(one_over_delta);
    for (; i < mul16; i += 16) {
        auto cur = _mm512_loadu_ps(&vec0[i]);
        cur = _mm512_mul_ps(_mm512_sub_ps(cur, lo512), od512);
        auto i8 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(cur));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&result[i]), i8);
    }
    for (; i < dim; ++i) {
        result[i] = static_cast<uint8_t>(std::round((vec0[i] - lo) * one_over_delta));
    }
#else
    scalar_quantize_normal(result, vec0, dim, lo, delta);
#endif
}

template <>
inline void scalar_quantize_optimized<uint16_t>(
    uint16_t* __restrict__ result,
    const float* __restrict__ vec0,
    size_t dim,
    float lo,
    float delta
) {
#if defined(__AVX512F__)
    size_t mul16 = dim - (dim & 0b1111);
    size_t i = 0;
    float one_over_delta = 1 / delta;
    auto lo512 = _mm512_set1_ps(lo);
    auto ow512 = _mm512_set1_ps(one_over_delta);
    for (; i < mul16; i += 16) {
        auto cur = _mm512_loadu_ps(&vec0[i]);
        cur = _mm512_mul_ps(_mm512_sub_ps(cur, lo512), ow512);
        auto i16 = _mm512_cvtepi32_epi16(_mm512_cvtps_epi32(cur));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&result[i]), i16);
    }
    for (; i < dim; ++i) {
        result[i] = static_cast<uint16_t>(std::round((vec0[i] - lo) * one_over_delta));
    }
#elif defined(__AVX2__)
    size_t mul8 = dim - (dim & 0b111);
    size_t i = 0;
    float one_over_delta = 1 / delta;
    auto lo256 = _mm256_set1_ps(lo);
    auto ow256 = _mm256_set1_ps(one_over_delta);
    for (; i < mul8; i += 8) {
        auto cur = _mm256_loadu_ps(&vec0[i]);
        cur = _mm256_mul_ps(_mm256_sub_ps(cur, lo256), ow256);
        auto i32 = _mm256_cvtps_epi32(cur);
        __m128i lo32 = _mm256_castsi256_si128(i32);
        __m128i hi32 = _mm256_extracti128_si256(i32, 1);
        __m128i i16 = _mm_packus_epi32(lo32, hi32);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(result + i), i16);
    }
    for (; i < dim; ++i) {
        result[i] = static_cast<uint16_t>(std::round((vec0[i] - lo) * one_over_delta));
    }
#else
    scalar_quantize_normal(result, vec0, dim, lo, delta);
#endif
}
}  // namespace scalar_impl

template <typename T>
inline void vec_rescale(T* data, size_t dim, T val) {
    RowMajorArrayMap<T> data_arr(data, 1, dim);
    data_arr *= val;
}

template <typename T>
inline T euclidean_sqr(const T* __restrict__ vec0, const T* __restrict__ vec1, size_t dim) {
    ConstVectorMap<T> v0(vec0, dim);
    ConstVectorMap<T> v1(vec1, dim);
    return (v0 - v1).dot(v0 - v1);
}

template <typename T>
inline T dot_product_dis(
    const T* __restrict__ vec0, const T* __restrict__ vec1, size_t dim
) {
    ConstVectorMap<T> v0(vec0, dim);
    ConstVectorMap<T> v1(vec1, dim);
    return 1 - v0.dot(v1);
}

template <typename T>
inline T l2norm_sqr(const T* __restrict__ vec0, size_t dim) {
    ConstVectorMap<T> v0(vec0, dim);
    return v0.dot(v0);
}

template <typename T>
inline T dot_product(const T* __restrict__ vec0, const T* __restrict__ vec1, size_t dim) {
    ConstVectorMap<T> v0(vec0, dim);
    ConstVectorMap<T> v1(vec1, dim);
    return v0.dot(v1);
}

template <typename T>
inline T normalize_vec(
    const T* __restrict__ vec, const T* __restrict__ centroid, T* res, T dist2c, size_t dim
) {
    RowMajorArrayMap<T> r(res, 1, dim);
    if (dist2c > 1e-5) {
        ConstRowMajorArrayMap<T> v(vec, 1, dim);
        ConstRowMajorArrayMap<T> c(centroid, 1, dim);
        r = (v - c) * (1 / dist2c);
        return r.sum();
    }
    T value = 1.0 / std::sqrt(static_cast<T>(dim));
    r = value;
    return static_cast<T>(dim) * value;
}

// pack 0/1 data to usigned integer
template <typename T>
inline void pack_binary(
    const int* __restrict__ binary_code, T* __restrict__ compact_code, size_t length
) {
    constexpr size_t kTypeBits = sizeof(T) * 8;

    for (size_t i = 0; i < length; i += kTypeBits) {
        T cur = 0;
        for (size_t j = 0; j < kTypeBits; ++j) {
            cur |= (static_cast<T>(binary_code[i + j]) << (kTypeBits - 1 - j));
        }
        *compact_code = cur;
        ++compact_code;
    }
}

template <typename T>
inline void data_range(const T* __restrict__ vec0, size_t dim, T& lo, T& hi) {
    ConstRowMajorArrayMap<T> v0(vec0, 1, dim);
    lo = v0.minCoeff();
    hi = v0.maxCoeff();
}

template <typename T, typename TD>
void scalar_quantize(
    T* __restrict__ result, const TD* __restrict__ vec0, size_t dim, TD lo, TD delta
) {
    assert_integral<T>();
    scalar_impl::scalar_quantize_optimized(result, vec0, dim, lo, delta);
}

template <typename T>
inline std::vector<T> compute_centroid(
    const T* data, size_t num_points, size_t dim, size_t num_threads
) {
    omp_set_num_threads(static_cast<int>(num_threads));
    std::vector<std::vector<T>> all_results(num_threads, std::vector<T>(dim, 0));

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points; ++i) {
        auto tid = omp_get_thread_num();
        std::vector<T>& cur_results = all_results[tid];
        const T* cur_data = data + (dim * i);
        for (size_t k = 0; k < dim; ++k) {
            cur_results[k] += cur_data[k];
        }
    }

    std::vector<T> centroid(dim, 0);
    for (auto& one_res : all_results) {
        for (size_t i = 0; i < dim; ++i) {
            centroid[i] += one_res[i];
        }
    }
    T inv_num_points = 1 / static_cast<T>(num_points);

    for (size_t i = 0; i < dim; ++i) {
        centroid[i] = centroid[i] * inv_num_points;
    }

    return centroid;
}

template <typename T>
inline PID exact_nn(
    const T* data,
    const T* query,
    size_t num_points,
    size_t dim,
    size_t num_threads,
    T (*dist_func)(const T*, const T*, size_t)
) {
    std::vector<AnnCandidate<T, PID>> best_entries(num_threads);

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_points; ++i) {
        auto tid = omp_get_thread_num();
        AnnCandidate<T, PID>& cur_entry = best_entries[tid];
        const T* cur_data = data + (dim * i);

        T distance = dist_func(cur_data, query, dim);
        if (distance < cur_entry.distance) {
            cur_entry.id = static_cast<PID>(i);
            cur_entry.distance = distance;
        }
    }

    PID nearest_neighbor = 0;
    T min_dist = std::numeric_limits<T>::max();
    for (auto& candi : best_entries) {
        if (candi.distance < min_dist) {
            nearest_neighbor = candi.id;
            min_dist = candi.distance;
        }
    }
    return nearest_neighbor;
}

namespace excode_ipimpl {

#if defined(__AVX2__)
// helper function for AVX2 inner product
inline void contribute_ip(__m128i vec, const float* __restrict__ query, __m256& sum) {
    /* // Equivalent AVX512 code:
        __m512 q = _mm512_loadu_ps(&query[i]);
        __m512 cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(q, cf, sum);
    */
    __m256 q = _mm256_loadu_ps(query);
    __m256 cf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(vec));
    sum = _mm256_fmadd_ps(q, cf, sum);

    q = _mm256_loadu_ps(query + 8);
    cf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(vec, 8)));
    sum = _mm256_fmadd_ps(q, cf, sum);
};

inline void contribute_ip_signed(
    __m128i vec, const float* __restrict__ query, __m256& sum
) {
    /* // Equivalent AVX512 code:
        __m512 q = _mm512_loadu_ps(&query[i]);
        __m512 cf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(c8));
        sum = _mm512_fmadd_ps(cf, q, sum);
    */
    __m256 q = _mm256_loadu_ps(query);
    __m256 cf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vec));
    sum = _mm256_fmadd_ps(cf, q, sum);

    q = _mm256_loadu_ps(query + 8);
    cf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(vec, 8)));
    sum = _mm256_fmadd_ps(cf, q, sum);
};

inline float mm256_reduce_add_ps(__m256 v) {
    std::array<float, 8> accumulator{};
    _mm256_storeu_ps(accumulator.data(), v);
    float result = 0.0F;
    for (const auto& i : accumulator) {
        result += i;
    }
    return result;
}
#endif

// ip16: this function is used to compute inner product of
// vectors padded to multiple of 16
// fxu1: the inner product is computed between float and 1-bit unsigned int (lay out can be
// found rabitq_impl.hpp)
// avx512: only applicable for avx512
inline float ip16_fxu1_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    float result = 0;
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();

    for (size_t i = 0; i < dim; i += 16) {
        __mmask16 mask = *reinterpret_cast<const __mmask16*>(compact_code);
        __m512 q = _mm512_loadu_ps(query);

        sum = _mm512_add_ps(_mm512_maskz_mov_ps(mask, q), sum);

        compact_code += 2;
        query += 16;
    }
    result = _mm512_reduce_add_ps(sum);
#elif defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();

    const __m256i bitmask = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);

    for (size_t i = 0; i < dim; i += 8) {
        __m256 q = _mm256_loadu_ps(query);

        __m256i byte_v = _mm256_set1_epi32(*compact_code);
        __m256i isolated = _mm256_and_si256(byte_v, bitmask);
        __m256i mask = _mm256_cmpeq_epi32(isolated, bitmask);
        __m256 masked = _mm256_and_ps(q, _mm256_castsi256_ps(mask));

        sum = _mm256_add_ps(sum, masked);
        query += 8;
        ++compact_code;
    }
    result = mm256_reduce_add_ps(sum);
#endif
    return result;
}

inline float ip64_fxu2_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
#elif defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
#else
    std::cerr << "AVX2 or AVX512 is required for excode ip functions\n";
    exit(1);
#endif
    float result = 0;
    const __m128i mask = _mm_set1_epi8(0b00000011);

    for (size_t i = 0; i < dim; i += 64) {
        __m128i compact = _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code));

        __m128i vec_00_to_15 = _mm_and_si128(compact, mask);
        __m128i vec_16_to_31 = _mm_and_si128(_mm_srli_epi16(compact, 2), mask);
        __m128i vec_32_to_47 = _mm_and_si128(_mm_srli_epi16(compact, 4), mask);
        __m128i vec_48_to_63 = _mm_and_si128(_mm_srli_epi16(compact, 6), mask);
#if defined(__AVX512F__)
        __m512 q;
        __m512 cf;

        q = _mm512_loadu_ps(&query[i]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 16]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 32]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 48]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_fmadd_ps(q, cf, sum);
#elif defined(__AVX2__)
        contribute_ip(vec_00_to_15, &query[i], sum);
        contribute_ip(vec_16_to_31, &query[i + 16], sum);
        contribute_ip(vec_32_to_47, &query[i + 32], sum);
        contribute_ip(vec_48_to_63, &query[i + 48], sum);
#endif
        compact_code += 16;
    }

#if defined(__AVX512F__)
    result = _mm512_reduce_add_ps(sum);
#elif defined(__AVX2__)
    result = mm256_reduce_add_ps(sum);
#endif
    return result;
}

inline float ip64_fxu3_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
#elif defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
#else
    std::cerr << "AVX2 or AVX512 is required for excode ip functions\n";
    exit(1);
#endif
    float result = 0;
    const __m128i mask = _mm_set1_epi8(0b11);
    const __m128i top_mask = _mm_set1_epi8(0b100);

    for (size_t i = 0; i < dim; i += 64) {
        __m128i compact2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code));
        compact_code += 16;

        int64_t top_bit = *reinterpret_cast<const int64_t*>(compact_code);
        compact_code += 8;

        __m128i vec_00_to_15 = _mm_and_si128(compact2, mask);
        __m128i vec_16_to_31 = _mm_and_si128(_mm_srli_epi16(compact2, 2), mask);
        __m128i vec_32_to_47 = _mm_and_si128(_mm_srli_epi16(compact2, 4), mask);
        __m128i vec_48_to_63 = _mm_and_si128(_mm_srli_epi16(compact2, 6), mask);

        __m128i top_00_to_15 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 1, top_bit << 2), top_mask);
        __m128i top_16_to_31 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 1, top_bit >> 0), top_mask);
        __m128i top_32_to_47 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 3, top_bit >> 2), top_mask);
        __m128i top_48_to_63 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 5, top_bit >> 4), top_mask);

        vec_00_to_15 = _mm_or_si128(top_00_to_15, vec_00_to_15);
        vec_16_to_31 = _mm_or_si128(top_16_to_31, vec_16_to_31);
        vec_32_to_47 = _mm_or_si128(top_32_to_47, vec_32_to_47);
        vec_48_to_63 = _mm_or_si128(top_48_to_63, vec_48_to_63);
#if defined(__AVX512F__)
        __m512 q;
        __m512 cf;

        q = _mm512_loadu_ps(&query[i]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 16]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 32]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 48]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_fmadd_ps(q, cf, sum);
#elif defined(__AVX2__)
        contribute_ip(vec_00_to_15, &query[i], sum);
        contribute_ip(vec_16_to_31, &query[i + 16], sum);
        contribute_ip(vec_32_to_47, &query[i + 32], sum);
        contribute_ip(vec_48_to_63, &query[i + 48], sum);
#endif
    }

#if defined(__AVX512F__)
    result = _mm512_reduce_add_ps(sum);
#elif defined(__AVX2__)
    result = mm256_reduce_add_ps(sum);
#endif
    return result;
}

inline float ip16_fxu4_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
#elif defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
#else
    std::cerr << "AVX2 or AVX512 is required for excode ip functions\n";
    exit(1);
#endif
    float result = 0.0F;
    constexpr int64_t kMask = 0x0f0f0f0f0f0f0f0f;
    for (size_t i = 0; i < dim; i += 16) {
        int64_t compact = *reinterpret_cast<const int64_t*>(compact_code);
        int64_t code0 = compact & kMask;
        int64_t code1 = (compact >> 4) & kMask;

        __m128i c8 = _mm_set_epi64x(code1, code0);
#if defined(__AVX512F__)
        __m512 q = _mm512_loadu_ps(&query[i]);
        __m512 cf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(c8));
        sum = _mm512_fmadd_ps(cf, q, sum);
#elif defined(__AVX2__)
        contribute_ip_signed(c8, &query[i], sum);
#endif
        compact_code += 8;
    }
#if defined(__AVX512F__)
    result = _mm512_reduce_add_ps(sum);
#elif defined(__AVX2__)
    result = mm256_reduce_add_ps(sum);
#endif
    return result;
}

inline float ip64_fxu5_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
#elif defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
#else
    std::cerr << "AVX2 or AVX512 is required for excode ip functions\n";
    exit(1);
#endif

    float result = 0.0F;
    const __m128i mask = _mm_set1_epi8(0b1111);
    const __m128i top_mask = _mm_set1_epi8(0b10000);

    for (size_t i = 0; i < dim; i += 64) {
        __m128i compact4_1 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code));
        __m128i compact4_2 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code + 16));
        compact_code += 32;

        int64_t top_bit = *reinterpret_cast<const int64_t*>(compact_code);
        compact_code += 8;

        __m128i vec_00_to_15 = _mm_and_si128(compact4_1, mask);
        __m128i vec_16_to_31 = _mm_and_si128(_mm_srli_epi16(compact4_1, 4), mask);
        __m128i vec_32_to_47 = _mm_and_si128(compact4_2, mask);
        __m128i vec_48_to_63 = _mm_and_si128(_mm_srli_epi16(compact4_2, 4), mask);

        __m128i top_00_to_15 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 3, top_bit << 4), top_mask);
        __m128i top_16_to_31 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 1, top_bit << 2), top_mask);
        __m128i top_32_to_47 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 1, top_bit >> 0), top_mask);
        __m128i top_48_to_63 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 3, top_bit >> 2), top_mask);

        vec_00_to_15 = _mm_or_si128(top_00_to_15, vec_00_to_15);
        vec_16_to_31 = _mm_or_si128(top_16_to_31, vec_16_to_31);
        vec_32_to_47 = _mm_or_si128(top_32_to_47, vec_32_to_47);
        vec_48_to_63 = _mm_or_si128(top_48_to_63, vec_48_to_63);

#if defined(__AVX512F__)
        __m512 q;
        __m512 cf;

        q = _mm512_loadu_ps(&query[i]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 16]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 32]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 48]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_fmadd_ps(q, cf, sum);
#elif defined(__AVX2__)
        contribute_ip(vec_00_to_15, &query[i], sum);
        contribute_ip(vec_16_to_31, &query[i + 16], sum);
        contribute_ip(vec_32_to_47, &query[i + 32], sum);
        contribute_ip(vec_48_to_63, &query[i + 48], sum);
#endif
    }
#if defined(__AVX512F__)
    result = _mm512_reduce_add_ps(sum);
#elif defined(__AVX2__)
    result = mm256_reduce_add_ps(sum);
#endif
    return result;
}

inline float ip64_fxu6_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
#elif defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
#else
    std::cerr << "AVX2 or AVX512 is required for excode ip functions\n";
    exit(1);
#endif
    float result = 0.0F;
    const __m128i mask6 = _mm_set1_epi8(0b00111111);
    const __m128i mask2 = _mm_set1_epi8(static_cast<char>(0b11000000));

    for (size_t i = 0; i < dim; i += 64) {
        __m128i cpt1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code));
        __m128i cpt2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code + 16));
        __m128i cpt3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code + 32));

        compact_code += 48;

        __m128i vec_00_to_15 = _mm_and_si128(cpt1, mask6);
        __m128i vec_16_to_31 = _mm_and_si128(cpt2, mask6);
        __m128i vec_32_to_47 = _mm_and_si128(cpt3, mask6);
        __m128i vec_48_to_63 = _mm_or_si128(
            _mm_or_si128(
                _mm_srli_epi16(_mm_and_si128(cpt1, mask2), 6),
                _mm_srli_epi16(_mm_and_si128(cpt2, mask2), 4)
            ),
            _mm_srli_epi16(_mm_and_si128(cpt3, mask2), 2)
        );

#if defined(__AVX512F__)
        __m512 q;
        __m512 cf;

        q = _mm512_loadu_ps(&query[i]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 16]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 32]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 48]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_fmadd_ps(q, cf, sum);
#elif defined(__AVX2__)
        contribute_ip(vec_00_to_15, &query[i], sum);
        contribute_ip(vec_16_to_31, &query[i + 16], sum);
        contribute_ip(vec_32_to_47, &query[i + 32], sum);
        contribute_ip(vec_48_to_63, &query[i + 48], sum);
#endif
    }
#if defined(__AVX512F__)
    result = _mm512_reduce_add_ps(sum);
#elif defined(__AVX2__)
    result = mm256_reduce_add_ps(sum);
#endif
    return result;
}

inline float ip64_fxu7_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
#elif defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
#else
    std::cerr << "AVX2 or AVX512 is required for excode ip functions\n";
    exit(1);
#endif

    float result = 0.0F;
    const __m128i mask6 = _mm_set1_epi8(0b00111111);
    const __m128i mask2 = _mm_set1_epi8(static_cast<char>(0b11000000));
    const __m128i top_mask = _mm_set1_epi8(0b1000000);

    for (size_t i = 0; i < dim; i += 64) {
        __m128i cpt1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code));
        __m128i cpt2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code + 16));
        __m128i cpt3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(compact_code + 32));
        compact_code += 48;

        __m128i vec_00_to_15 = _mm_and_si128(cpt1, mask6);
        __m128i vec_16_to_31 = _mm_and_si128(cpt2, mask6);
        __m128i vec_32_to_47 = _mm_and_si128(cpt3, mask6);
        __m128i vec_48_to_63 = _mm_or_si128(
            _mm_or_si128(
                _mm_srli_epi16(_mm_and_si128(cpt1, mask2), 6),
                _mm_srli_epi16(_mm_and_si128(cpt2, mask2), 4)
            ),
            _mm_srli_epi16(_mm_and_si128(cpt3, mask2), 2)
        );

        int64_t top_bit = *reinterpret_cast<const int64_t*>(compact_code);
        compact_code += 8;

        __m128i top_00_to_15 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 5, top_bit << 6), top_mask);
        __m128i top_16_to_31 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 3, top_bit << 4), top_mask);
        __m128i top_32_to_47 =
            _mm_and_si128(_mm_set_epi64x(top_bit << 1, top_bit << 2), top_mask);
        __m128i top_48_to_63 =
            _mm_and_si128(_mm_set_epi64x(top_bit >> 1, top_bit << 0), top_mask);

        vec_00_to_15 = _mm_or_si128(top_00_to_15, vec_00_to_15);
        vec_16_to_31 = _mm_or_si128(top_16_to_31, vec_16_to_31);
        vec_32_to_47 = _mm_or_si128(top_32_to_47, vec_32_to_47);
        vec_48_to_63 = _mm_or_si128(top_48_to_63, vec_48_to_63);

#if defined(__AVX512F__)
        __m512 q;
        __m512 cf;

        q = _mm512_loadu_ps(&query[i]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_00_to_15));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 16]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_16_to_31));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 32]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_32_to_47));
        sum = _mm512_fmadd_ps(q, cf, sum);

        q = _mm512_loadu_ps(&query[i + 48]);
        cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(vec_48_to_63));
        sum = _mm512_fmadd_ps(q, cf, sum);
#elif defined(__AVX2__)
        contribute_ip(vec_00_to_15, &query[i], sum);
        contribute_ip(vec_16_to_31, &query[i + 16], sum);
        contribute_ip(vec_32_to_47, &query[i + 32], sum);
        contribute_ip(vec_48_to_63, &query[i + 48], sum);
#endif
    }

#if defined(__AVX512F__)
    result = _mm512_reduce_add_ps(sum);
#elif defined(__AVX2__)
    result = mm256_reduce_add_ps(sum);
#endif
    return result;
}

// inner product between float type and int type vectors
template <typename TF, typename TI>
inline TF ip_fxi(const TF* __restrict__ vec0, const TI* __restrict__ vec1, size_t dim) {
    static_assert(std::is_floating_point_v<TF>, "TF must be an floating type");
    static_assert(std::is_integral_v<TI>, "TI must be an integeral type");

    ConstVectorMap<TF> v0(vec0, dim);
    ConstVectorMap<TI> v1(vec1, dim);
    return v0.dot(v1.template cast<TF>());
}
}  // namespace excode_ipimpl

using ex_ipfunc = float (*)(const float*, const uint8_t*, size_t);

inline ex_ipfunc select_excode_ipfunc(size_t ex_bits) {
    if (ex_bits <= 1) {
        // when ex_bits = 0, we do not use it
        return excode_ipimpl::ip16_fxu1_avx;
    }
    if (ex_bits == 2) {
        return excode_ipimpl::ip64_fxu2_avx;
    }
    if (ex_bits == 3) {
        return excode_ipimpl::ip64_fxu3_avx;
    }
    if (ex_bits == 4) {
        return excode_ipimpl::ip16_fxu4_avx;
    }
    if (ex_bits == 5) {
        return excode_ipimpl::ip64_fxu5_avx;
    }
    if (ex_bits == 6) {
        return excode_ipimpl::ip64_fxu6_avx;
    }
    if (ex_bits == 7) {
        return excode_ipimpl::ip64_fxu7_avx;
    }
    if (ex_bits == 8) {
        return excode_ipimpl::ip_fxi;
    }

    std::cerr << "Bad IP function for IVF\n";
    exit(1);
}

static inline uint32_t reverse_bits(uint32_t n) {
    n = ((n >> 1) & 0x55555555) | ((n << 1) & 0xaaaaaaaa);
    n = ((n >> 2) & 0x33333333) | ((n << 2) & 0xcccccccc);
    n = ((n >> 4) & 0x0f0f0f0f) | ((n << 4) & 0xf0f0f0f0);
    n = ((n >> 8) & 0x00ff00ff) | ((n << 8) & 0xff00ff00);
    n = ((n >> 16) & 0x0000ffff) | ((n << 16) & 0xffff0000);
    return n;
}

static inline uint64_t reverse_bits_u64(uint64_t n) {
    n = ((n >> 1) & 0x5555555555555555) | ((n << 1) & 0xaaaaaaaaaaaaaaaa);
    n = ((n >> 2) & 0x3333333333333333) | ((n << 2) & 0xcccccccccccccccc);
    n = ((n >> 4) & 0x0f0f0f0f0f0f0f0f) | ((n << 4) & 0xf0f0f0f0f0f0f0f0);
    n = ((n >> 8) & 0x00ff00ff00ff00ff) | ((n << 8) & 0xff00ff00ff00ff00);
    n = ((n >> 16) & 0x0000ffff0000ffff) | ((n << 16) & 0xffff0000ffff0000);
    n = ((n >> 32) & 0x00000000ffffffff) | ((n << 32) & 0xffffffff00000000);
    return n;
}

static inline void new_transpose_bin(
    const uint16_t* q, uint64_t* tq, size_t padded_dim, size_t b_query
) {
    // Easy
#if defined(__AVX512BW__)
    // 512 / 16 = 32
    for (size_t i = 0; i < padded_dim; i += 64) {
        __m512i vec_00_to_31 = _mm512_loadu_si512(q);
        __m512i vec_32_to_63 = _mm512_loadu_si512(q + 32);

        // the first (16 - b_query) bits are empty
        vec_00_to_31 = _mm512_slli_epi32(vec_00_to_31, (16 - b_query));
        vec_32_to_63 = _mm512_slli_epi32(vec_32_to_63, (16 - b_query));

        for (size_t j = 0; j < b_query; ++j) {
            uint32_t v0 = _mm512_movepi16_mask(vec_00_to_31);  // get most significant bit
            uint32_t v1 = _mm512_movepi16_mask(vec_32_to_63);  // get most significant bit
            // [TODO: remove all reverse_bits]
            v0 = reverse_bits(v0);
            v1 = reverse_bits(v1);
            uint64_t v = (static_cast<uint64_t>(v0) << 32) + v1;

            tq[b_query - j - 1] = v;

            vec_00_to_31 = _mm512_slli_epi16(vec_00_to_31, 1);
            vec_32_to_63 = _mm512_slli_epi16(vec_32_to_63, 1);
        }
        tq += b_query;
        q += 64;
    }
#elif defined(__AVX2__)
    for (size_t i = 0; i < padded_dim; i += 64) {
        __m256i vec_00_to_15 = _mm256_loadu_si256((__m256i const*)(q));
        __m256i vec_16_to_31 = _mm256_loadu_si256((__m256i const*)(q + 16));
        __m256i vec_32_to_47 = _mm256_loadu_si256((__m256i const*)(q + 32));
        __m256i vec_48_to_63 = _mm256_loadu_si256((__m256i const*)(q + 48));

        // the first (16 - b_query) bits are empty
        vec_00_to_15 = _mm256_slli_epi32(vec_00_to_15, (16 - b_query));
        vec_16_to_31 = _mm256_slli_epi32(vec_16_to_31, (16 - b_query));
        vec_32_to_47 = _mm256_slli_epi32(vec_32_to_47, (16 - b_query));
        vec_48_to_63 = _mm256_slli_epi32(vec_48_to_63, (16 - b_query));

        for (size_t j = 0; j < b_query; ++j) {
            // pack two 16-bit vectors to 8-bit interleaved vectors
            __m256i p0 = _mm256_packs_epi16(vec_00_to_15, vec_16_to_31);
            __m256i p1 = _mm256_packs_epi16(vec_32_to_47, vec_48_to_63);

            uint32_t m0 = _mm256_movemask_epi8(p0);
            uint32_t m1 = _mm256_movemask_epi8(p1);

            // Fix AVX2 Lane Ordering of the interleaved mask
            auto fix_avx2_mask = [](uint32_t m) {
                return (m & 0xFF0000FF) | ((m & 0x00FF0000) >> 8) | ((m & 0x0000FF00) << 8);
            };

            m0 = fix_avx2_mask(m0);
            m1 = fix_avx2_mask(m1);

            m0 = reverse_bits(m0);
            m1 = reverse_bits(m1);

            uint64_t v = (static_cast<uint64_t>(m0) << 32) | m1;

            tq[b_query - j - 1] = v;

            vec_00_to_15 = _mm256_slli_epi16(vec_00_to_15, 1);
            vec_16_to_31 = _mm256_slli_epi16(vec_16_to_31, 1);
            vec_32_to_47 = _mm256_slli_epi16(vec_32_to_47, 1);
            vec_48_to_63 = _mm256_slli_epi16(vec_48_to_63, 1);
        }
        tq += b_query;
        q += 64;
    }
#else
    std::cerr << "AVX512 or AVX2 is required for new transpose bin\n";
    exit(1);
#endif
}

static inline void new_transpose_bin_512(
    const uint8_t* q, uint64_t* tq, size_t padded_dim, size_t b_query
) {
#if defined(__AVX512BW__)
    // Keep full 512-dim blocks as 8 chunks, but store the tail as compact
    // [b_query x num_chunks] so runtime can use maskz loads without query padding.
    for (size_t i = 0; i < padded_dim;) {
        size_t block_size = 512;
        if (i + 512 > padded_dim) {
            block_size = padded_dim - i;
        }
        size_t num_chunks = block_size / 64;

        for (size_t k = 0; k < num_chunks; ++k) {
            const uint8_t* current_q = q + i + k * 64;
            __m512i vec = _mm512_loadu_si512(current_q);

            for (size_t j = 0; j < b_query; ++j) {
                int bit_idx = b_query - 1 - j;
                __mmask64 m = _mm512_test_epi8_mask(vec, _mm512_set1_epi8(1 << bit_idx));
                tq[(b_query - j - 1) * num_chunks + k] = reverse_bits_u64(static_cast<uint64_t>(m));
            }
        }

        i += block_size;
        tq += num_chunks * b_query;
    }
#elif defined(__AVX2__)
    for (size_t i = 0; i < padded_dim;) {
        size_t block_size = 512;
        if (i + 512 > padded_dim) {
            block_size = padded_dim - i;
        }
        // Each chunk represents 64 bytes (512 bits) of dimensions
        size_t num_chunks = block_size / 64;

        for (size_t k = 0; k < num_chunks; ++k) {
            // Load 64 bytes using two sequential 32-byte AVX2 registers
            const uint8_t* current_q_lo = q + i + k * 64;
            const uint8_t* current_q_hi = q + i + k * 64 + 32;

            __m256i vec_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(current_q_lo));
            __m256i vec_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(current_q_hi));

            for (size_t j = 0; j < b_query; ++j) {
                int bit_idx = b_query - 1 - j;
                __m256i mask_vec = _mm256_set1_epi8(static_cast<char>(1 << bit_idx));

                // Process lower 32 bytes
                __m256i res_lo = _mm256_and_si256(vec_lo, mask_vec);
                __m256i eq_lo = _mm256_cmpeq_epi8(res_lo, _mm256_setzero_si256());
                uint32_t m_lo = ~static_cast<uint32_t>(_mm256_movemask_epi8(eq_lo));

                // Process upper 32 bytes
                __m256i res_hi = _mm256_and_si256(vec_hi, mask_vec);
                __m256i eq_hi = _mm256_cmpeq_epi8(res_hi, _mm256_setzero_si256());
                uint32_t m_hi = ~static_cast<uint32_t>(_mm256_movemask_epi8(eq_hi));

                // Combine both 32-bit masks into a single 64-bit mask
                uint64_t m = (static_cast<uint64_t>(m_hi) << 32) | m_lo;

                // Write into the 64-bit structured macro-layout
                tq[(b_query - j - 1) * num_chunks + k] = reverse_bits_u64(m);
            }
        }

        i += block_size;
        tq += num_chunks * b_query;
    }
#else
    std::cerr << "AVX512BW or AVX2 is required for new_transpose_bin_512\n";
    exit(1);
#endif
}

inline float mask_ip_x0_q_old(const float* query, const uint64_t* data, size_t padded_dim) {
    auto num_blk = padded_dim / 64;
    const auto* it_data = data;
    const auto* it_query = query;

    __m512 sum = _mm512_setzero_ps();
    for (size_t i = 0; i < num_blk; ++i) {
        uint64_t bits = reverse_bits_u64(*it_data);
        auto mask0 = static_cast<__mmask16>(bits >> 00);  // for q[0..15]
        auto mask1 = static_cast<__mmask16>(bits >> 16);  // for q[16..31]
        auto mask2 = static_cast<__mmask16>(bits >> 32);  // for q[32..47]
        auto mask3 = static_cast<__mmask16>(bits >> 48);  // for q[48..63]

        __m512 q0 = _mm512_loadu_ps(it_query);
        __m512 q1 = _mm512_loadu_ps(it_query + 16);
        __m512 q2 = _mm512_loadu_ps(it_query + 32);
        __m512 q3 = _mm512_loadu_ps(it_query + 48);

        __m512 masked0 = _mm512_maskz_mov_ps(mask0, q0);
        __m512 masked1 = _mm512_maskz_mov_ps(mask1, q1);
        __m512 masked2 = _mm512_maskz_mov_ps(mask2, q2);
        __m512 masked3 = _mm512_maskz_mov_ps(mask3, q3);

        sum = _mm512_add_ps(sum, masked0);
        sum = _mm512_add_ps(sum, masked1);
        sum = _mm512_add_ps(sum, masked2);
        sum = _mm512_add_ps(sum, masked3);

        it_data++;
        it_query += 64;
    }
    return _mm512_reduce_add_ps(sum);
}

inline float mask_ip_x0_q(const float* query, const uint64_t* data, size_t padded_dim) {
    const size_t num_blk = padded_dim / 64;
    const uint64_t* it_data = data;
    const float* it_query = query;
// Easier
#if defined(__AVX512F__)

    //    __m512 sum0 = _mm512_setzero_ps();
    //    __m512 sum1 = _mm512_setzero_ps();
    //    __m512 sum2 = _mm512_setzero_ps();
    //    __m512 sum3 = _mm512_setzero_ps();

    __m512 sum = _mm512_setzero_ps();
    for (size_t i = 0; i < num_blk; ++i) {
        uint64_t bits = reverse_bits_u64(*it_data);

        auto mask0 = static_cast<__mmask16>(bits);
        auto mask1 = static_cast<__mmask16>(bits >> 16);
        auto mask2 = static_cast<__mmask16>(bits >> 32);
        auto mask3 = static_cast<__mmask16>(bits >> 48);

        __m512 masked0 = _mm512_maskz_loadu_ps(mask0, it_query);
        __m512 masked1 = _mm512_maskz_loadu_ps(mask1, it_query + 16);
        __m512 masked2 = _mm512_maskz_loadu_ps(mask2, it_query + 32);
        __m512 masked3 = _mm512_maskz_loadu_ps(mask3, it_query + 48);

        sum = _mm512_add_ps(sum, masked0);
        sum = _mm512_add_ps(sum, masked1);
        sum = _mm512_add_ps(sum, masked2);
        sum = _mm512_add_ps(sum, masked3);

        //         _mm_prefetch(reinterpret_cast<const char*>(it_query + 128), _MM_HINT_T1);

        ++it_data;
        it_query += 64;
    }

    //    __m512 sum = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
    return _mm512_reduce_add_ps(sum);
#elif defined(__AVX2__)

    __m256 sum = _mm256_setzero_ps();

    __m256i bit_checker = _mm256_set_epi32(0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01);

    for (size_t i = 0; i < num_blk; ++i) {
        uint64_t bits = reverse_bits_u64(*it_data);

        // 64 bits / 8 floats = 8 iterations
        for (int j = 0; j < 8; ++j) {
            uint8_t current_byte = static_cast<uint8_t>(bits >> (j * 8));
            __m256i v_byte = _mm256_set1_epi32(current_byte);
            __m256i masked_bits = _mm256_and_si256(v_byte, bit_checker);
            __m256i mask = _mm256_cmpgt_epi32(masked_bits, _mm256_setzero_si256());

            __m256 q_vals = _mm256_loadu_ps(it_query);
            __m256 masked = _mm256_and_ps(q_vals, _mm256_castsi256_ps(mask));

            sum = _mm256_add_ps(sum, masked);

            it_query += 8;
        }
        ++it_data;
    }

    float result = 0.0f;
    for (int i = 0; i < 8; ++i) {
        result += reinterpret_cast<float*>(&sum)[i];
    }
    return result;
#else
    std::cerr << "AVX512 or AVX2 is required for mask ip x0 q\n";
    exit(1);
#endif
    return 0.0F;
}

inline float ip_x0_q(
    const uint64_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
) {
    auto num_blk = padded_dim / 64;
    const auto* it_data = data;
    const auto* it_query = query;

    size_t ip = 0;
    size_t ppc = 0;

    for (size_t i = 0; i < num_blk; ++i) {
        uint64_t x = *static_cast<const uint64_t*>(it_data);
        ppc += __builtin_popcountll(x);

        for (size_t j = 0; j < b_query; ++j) {
            uint64_t y = *static_cast<const uint64_t*>(it_query);
            ip += (__builtin_popcountll(x & y) << j);
            it_query++;
        }
        it_data++;
    }

    return (delta * static_cast<float>(ip)) + (vl * static_cast<float>(ppc));
}

static inline uint32_t ip_bin_bin(const uint64_t* q, const uint64_t* d, size_t padded_dim) {
    uint64_t ret = 0;
    size_t iter = padded_dim / 64;
    for (size_t i = 0; i < iter; ++i) {
        ret += __builtin_popcountll((*d) & (*q));
        q++;
        d++;
    }
    return ret;
}

inline uint32_t ip_byte_bin(
    const uint64_t* q, const uint64_t* d, size_t padded_dim, size_t b_query
) {
    uint32_t ret = 0;
    size_t offset = (padded_dim / 64);
    for (size_t i = 0; i < b_query; i++) {
        ret += (ip_bin_bin(q, d, padded_dim) << i);
        q += offset;
    }
    return ret;
}

inline size_t popcount(const uint64_t* __restrict__ d, size_t length) {
    size_t ret = 0;
    for (size_t i = 0; i < length / 64; ++i) {
        ret += __builtin_popcountll((*d));
        ++d;
    }
    return ret;
}

template <typename T>
RowMajorMatrix<T> random_gaussian_matrix(
    size_t rows, size_t cols, std::optional<unsigned> seed = std::nullopt
) {
    RowMajorMatrix<T> rand(rows, cols);
    std::mt19937 gen(seed.has_value() ? *seed : std::random_device{}());
    std::normal_distribution<T> dist(0, 1);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            rand(i, j) = dist(gen);
        }
    }

    return rand;
}
}  // namespace rabitqlib
