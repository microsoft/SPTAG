#pragma once

#include <omp.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/quantization/rabitq_impl.hpp"

namespace rabitqlib::quant {

struct RabitqConfig {
    double t_const = -1;
    explicit RabitqConfig() = default;
    RabitqConfig(RabitqConfig const&) = default;
    RabitqConfig(RabitqConfig&&) = default;
    RabitqConfig& operator=(const RabitqConfig&) = default;
    RabitqConfig& operator=(RabitqConfig&&) = default;
};

// init config for faster quantization rabitq, for total_bits > 0, store const rescaling
// factor in RabitqConfig
inline RabitqConfig faster_config(size_t dim, size_t total_bits) {
    RabitqConfig config;
    if (total_bits > 1) {
        config.t_const =
            rabitq_impl::ex_bits::get_const_scaling_factors(dim, total_bits - 1);
    }
    return config;
}

template <typename T, bool Parallel = false>
inline void quantize_one_batch(
    const T* data,
    const T* centroid,
    size_t num,
    size_t padded_dim,
    char* batch_data,
    MetricType metric_type = METRIC_L2
) {
    BatchDataMap<T> this_batch(batch_data, padded_dim);

    rabitq_impl::one_bit::one_bit_batch_code(
        data,
        centroid,
        num,
        padded_dim,
        this_batch.bin_code(),
        this_batch.f_add(),
        this_batch.f_rescale(),
        this_batch.f_error(),
        metric_type
    );
}

template <typename T, bool Parallel = false>
inline void quantize_one_batch(
    const T* data,
    size_t num,
    size_t padded_dim,
    char* batch_data,
    MetricType metric_type = METRIC_L2
) {
    std::vector<T> centroid(padded_dim, 0);
    BatchDataMap<T> this_batch(batch_data, padded_dim);

    rabitq_impl::one_bit::one_bit_batch_code(
        data,
        centroid.data(),
        num,
        padded_dim,
        this_batch.bin_code(),
        this_batch.f_add(),
        this_batch.f_rescale(),
        this_batch.f_error(),
        metric_type
    );
}

template <typename T>
static inline void quantize_qg_batch(
    const T* data,
    const T* centroid,
    size_t num,
    size_t padded_dim,
    char* batch_data,
    MetricType metric_type = METRIC_L2
) {
    std::vector<T> f_error(fastscan::kBatchSize);  // we dont need this factor for qg
    QGBatchDataMap<T> cur_batch(batch_data, padded_dim);

    rabitq_impl::one_bit::one_bit_batch_code<T>(
        data,
        centroid,
        num,
        padded_dim,
        cur_batch.bin_code(),
        cur_batch.f_add(),
        cur_batch.f_rescale(),
        f_error.data(),
        metric_type
    );
}

template <typename T>
static inline void quantize_qg_batch(
    const T* data,
    size_t num,
    size_t padded_dim,
    char* batch_data,
    MetricType metric_type = METRIC_L2
) {
    std::vector<T> centroid(padded_dim, 0);
    std::vector<T> f_error(fastscan::kBatchSize);  // we dont need this factor for qg
    QGBatchDataMap<T> cur_batch(batch_data, padded_dim);

    rabitq_impl::one_bit::one_bit_batch_code<T>(
        data,
        centroid.data(),
        num,
        padded_dim,
        cur_batch.bin_code(),
        cur_batch.f_add(),
        cur_batch.f_rescale(),
        f_error.data(),
        metric_type
    );
}

template <typename T>
inline void quantize_compact_one_bit(
    const T* data,
    const T* centroid,
    size_t padded_dim,
    char* bin_data,
    MetricType metric_type = METRIC_L2
) {
    BinDataMap<T> cur_bin_data(bin_data, padded_dim);

    rabitq_impl::one_bit::one_bit_compact_code(
        data,
        centroid,
        padded_dim,
        cur_bin_data.bin_code(),
        cur_bin_data.f_add(),
        cur_bin_data.f_rescale(),
        cur_bin_data.f_error(),
        metric_type
    );
}

template <typename T, typename TC>
inline void quantize_compact_one_bit(
    const T* data,
    const T* centroid,
    size_t padded_dim,
    TC compact_code,
    T& f_add,
    T& f_rescale,
    T& ferror,
    MetricType metric_type = METRIC_L2
) {
    rabitq_impl::one_bit::one_bit_compact_code(
        data, centroid, padded_dim, compact_code, f_add, f_rescale, ferror, metric_type
    );
}

template <typename T, typename TC>
inline void quantize_compact_one_bit(
    const T* data,
    size_t padded_dim,
    TC compact_code,
    T& f_add,
    T& f_rescale,
    T& ferror,
    MetricType metric_type = METRIC_L2
) {
    std::vector<T> centroid(padded_dim, 0);
    rabitq_impl::one_bit::one_bit_compact_code(
        data,
        centroid.data(),
        padded_dim,
        compact_code,
        f_add,
        f_rescale,
        ferror,
        metric_type
    );
}

template <typename T>
inline void quantize_compact_ex_bits(
    const T* data,
    const T* centroid,
    size_t padded_dim,
    size_t ex_bits,
    char* ex_data,
    MetricType metric_type = METRIC_L2,
    RabitqConfig config = RabitqConfig()
) {
    ExDataMap<T> cur_ex_data(ex_data, padded_dim, ex_bits);

    // we do not use this error factor here
    T ex_error;

    rabitq_impl::ex_bits::ex_bits_compact_code(
        data,
        centroid,
        padded_dim,
        ex_bits,
        cur_ex_data.ex_code(),
        cur_ex_data.f_add_ex(),
        cur_ex_data.f_rescale_ex(),
        ex_error,
        metric_type,
        config.t_const
    );
}

inline void quantize_split_batch(
    const float* data,
    const float* centroid,
    size_t num_points,
    size_t padded_dim,
    size_t ex_bits,
    char* batch_data,
    char* ex_data,
    MetricType metric_type = METRIC_L2,
    RabitqConfig config = RabitqConfig()
) {
    quantize_one_batch(data, centroid, num_points, padded_dim, batch_data, metric_type);

    if (ex_bits > 0) {
        for (size_t i = 0; i < num_points; ++i) {
            quantize_compact_ex_bits(
                data, centroid, padded_dim, ex_bits, ex_data, metric_type, config
            );

            ex_data += ExDataMap<float>::data_bytes(padded_dim, ex_bits);
            data += padded_dim;
        }
    }
}

inline void quantize_split_single(
    const float* data,
    const float* centroid,
    size_t padded_dim,
    size_t ex_bits,
    char* bin_data,
    char* ex_data,
    MetricType metric_type = METRIC_L2,
    RabitqConfig config = RabitqConfig()
) {
    quantize_compact_one_bit(data, centroid, padded_dim, bin_data, metric_type);

    if (ex_bits > 0) {
        quantize_compact_ex_bits(
            data, centroid, padded_dim, ex_bits, ex_data, metric_type, config
        );
    }
}

template <typename T, typename TP>
inline void quantize_full_single(
    const T* data,
    size_t dim,
    size_t total_bits,
    TP* total_code,
    T& f_add,
    T& f_rescale,
    T& f_error,
    MetricType metric_type = METRIC_L2,
    RabitqConfig config = RabitqConfig()
) {
    std::vector<T> centroid(dim, 0);
    rabitq_impl::total_bits::rabitq_full_impl<T, TP>(
        data,
        centroid.data(),
        dim,
        total_bits,
        total_code,
        f_add,
        f_rescale,
        f_error,
        metric_type,
        config.t_const
    );
}

template <typename T, typename TP>
inline void quantize_full_single(
    const T* data,
    const T* centroid,
    size_t dim,
    size_t total_bits,
    TP* total_code,
    T& f_add,
    T& f_rescale,
    T& f_error,
    MetricType metric_type = METRIC_L2,
    RabitqConfig config = RabitqConfig()
) {
    rabitq_impl::total_bits::rabitq_full_impl<T, TP>(
        data,
        centroid,
        dim,
        total_bits,
        total_code,
        f_add,
        f_rescale,
        f_error,
        metric_type,
        config.t_const
    );
}

template <typename T, typename TP>
inline void quantize_scalar(
    const T* data,
    size_t dim,
    size_t total_bits,
    TP* total_code,
    T& delta,
    T& vl,
    RabitqConfig config = RabitqConfig(),
    ScalarQuantizerType scalar_quantizer_type = ScalarQuantizerType::RECONSTRUCTION
) {
    std::vector<T> centroid(dim, 0);
    rabitq_impl::total_bits::rabitq_scalar_impl<T, TP>(
        data,
        centroid.data(),
        dim,
        total_bits,
        total_code,
        delta,
        vl,
        config.t_const,
        scalar_quantizer_type
    );
}

template <typename T, typename TP>
inline void quantize_scalar(
    const T* data,
    const T* centroid,
    size_t dim,
    size_t total_bits,
    TP* total_code,
    T& delta,
    T& vl,
    RabitqConfig config = RabitqConfig()
) {
    rabitq_impl::total_bits::rabitq_scalar_impl<T, TP>(
        data, centroid, dim, total_bits, total_code, delta, vl, config.t_const
    );
}

template <typename T, typename TP>
inline void reconstruct_vec(
    const TP* quantized_vec, T delta, T vl, size_t dim, T* results
) {
    RowMajorArrayMap<T> result_arr(results, 1, dim);
    result_arr =
        (ConstRowMajorArrayMap<TP>(quantized_vec, 1, dim).template cast<T>() * delta) + vl;
}

template <typename TF, typename TI>
inline TF full_est_dist(
    const TI* quantized_vec,
    const TF* query,
    TF (*ip_func_)(const TF*, const TI*, size_t),
    size_t dim,
    size_t bits,
    TF f_add,
    TF f_rescale,
    TF g_add,
    TF k1xsumq
) {
    TF est_dist = f_add + g_add +
                  (f_rescale * (ip_func_(query, quantized_vec, dim) +
                                k1xsumq * static_cast<float>((1 << bits) - 1)));

    return est_dist;
}
}  // namespace rabitqlib::quant