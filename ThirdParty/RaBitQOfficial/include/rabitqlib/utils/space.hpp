#pragma once

#include <omp.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <type_traits>

#include "rabitqlib/utils/compiler_compat.hpp"
#include "rabitqlib/defines.hpp"
#include "rabitqlib/simd/space_dispatch.hpp"
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
    simd::scalar_quantize_uint8(result, vec0, dim, lo, delta);
}

template <>
inline void scalar_quantize_optimized<uint16_t>(
    uint16_t* __restrict__ result,
    const float* __restrict__ vec0,
    size_t dim,
    float lo,
    float delta
) {
    simd::scalar_quantize_uint16(result, vec0, dim, lo, delta);
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

float ip16_fxu1_avx(const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim);
float ip64_fxu2_avx(const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim);
float ip64_fxu3_avx(const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim);
float ip16_fxu4_avx(const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim);
float ip64_fxu5_avx(const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim);
float ip64_fxu6_avx(const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim);
float ip64_fxu7_avx(const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim);

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

ex_ipfunc select_excode_ipfunc(size_t ex_bits);

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

void new_transpose_bin(
    const uint16_t* q, uint64_t* tq, size_t padded_dim, size_t b_query
);

void new_transpose_bin_512(
    const uint8_t* q, uint64_t* tq, size_t padded_dim, size_t b_query
);

float mask_ip_x0_q(const float* query, const uint64_t* data, size_t padded_dim);

inline float mask_ip_x0_q_old(const float* query, const uint64_t* data, size_t padded_dim) {
    return mask_ip_x0_q(query, data, padded_dim);
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
