#pragma once

#include <limits>

#include "rabitqlib/third/Eigen/Dense"

#define BIT_ID(x) (__builtin_popcount((x) - 1))
#define LOWBIT(x) ((x) & (-(x)))

namespace rabitqlib {

using PID = uint32_t;

constexpr uint32_t kPidMax = 0xFFFFFFFF;

template <typename T>
using RowMajorMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using RowMajorMatrixMap = Eigen::Map<RowMajorMatrix<T>>;

template <typename T>
using ConstRowMajorMatrixMap = Eigen::Map<const RowMajorMatrix<T>>;

template <typename T>
using RowMajorArray = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T>
using RowMajorArrayMap = Eigen::Map<RowMajorArray<T>>;

template <typename T>
using ConstRowMajorArrayMap = Eigen::Map<const RowMajorArray<T>>;

template <typename T>
using VectorMap = Eigen::Map<Vector<T>>;

template <typename T>
using ConstVectorMap = Eigen::Map<const Vector<T>>;

template <typename T, typename TP = PID>
struct AnnCandidate {
    TP id = 0;
    T distance = std::numeric_limits<T>::max();

    AnnCandidate() = default;
    explicit AnnCandidate(TP vec_id, T dis) : id(vec_id), distance(dis) {}

    friend bool operator<(const AnnCandidate& first, const AnnCandidate& second) {
        return first.distance < second.distance;
    }
    friend bool operator>(const AnnCandidate& first, const AnnCandidate& second) {
        return first.distance > second.distance;
    }
    friend bool operator>=(const AnnCandidate& first, const AnnCandidate& second) {
        return first.distance >= second.distance;
    }
    friend bool operator<=(const AnnCandidate& first, const AnnCandidate& second) {
        return first.distance <= second.distance;
    }
};

enum MetricType : std::uint8_t { METRIC_L2, METRIC_IP };
enum ScalarQuantizerType : std::uint8_t { RECONSTRUCTION, UNBIASED_ESTIMATION, PLAIN };
}  // namespace rabitqlib