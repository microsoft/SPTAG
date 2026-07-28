// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

namespace rabitqlib
{
namespace quant
{
namespace global_scalar
{

inline float TightStart(std::size_t p_extraBits)
{
    static constexpr std::array<float, 9> values = {
        0.0F, 0.15F, 0.20F, 0.52F, 0.59F, 0.71F, 0.75F, 0.77F, 0.81F
    };
    return values[p_extraBits];
}

template <typename T>
inline double BestRescaleFactor(const T* p_absoluteValues,
                                std::size_t p_dim,
                                std::size_t p_extraBits)
{
    constexpr double kEpsilon = 1e-5;
    constexpr int kEnumerationSlack = 10;
    const double maxValue = static_cast<double>(
        *std::max_element(p_absoluteValues, p_absoluteValues + p_dim));
    if (maxValue <= 0.0 || !std::isfinite(maxValue)) {
        return 0.0;
    }

    const int maxCode = (1 << p_extraBits) - 1;
    const double end = static_cast<double>(maxCode + kEnumerationSlack) / maxValue;
    const double start = end * TightStart(p_extraBits);

    std::vector<int> codes(p_dim);
    double squaredDenominator = static_cast<double>(p_dim) * 0.25;
    double numerator = 0.0;
    for (std::size_t i = 0; i < p_dim; ++i) {
        const int code = static_cast<int>(start * p_absoluteValues[i] + kEpsilon);
        codes[i] = code;
        squaredDenominator += code * code + code;
        numerator += (code + 0.5) * p_absoluteValues[i];
    }

    std::priority_queue<
        std::pair<double, std::size_t>,
        std::vector<std::pair<double, std::size_t>>,
        std::greater<>>
        next;
    for (std::size_t i = 0; i < p_dim; ++i) {
        if (p_absoluteValues[i] > 0) {
            next.emplace(static_cast<double>(codes[i] + 1) / p_absoluteValues[i], i);
        }
    }

    double maxInnerProduct = 0.0;
    double scale = 0.0;
    while (!next.empty()) {
        const double current = next.top().first;
        const std::size_t index = next.top().second;
        next.pop();

        const int code = ++codes[index];
        squaredDenominator += 2 * code;
        numerator += p_absoluteValues[index];

        const double innerProduct = numerator / std::sqrt(squaredDenominator);
        if (innerProduct > maxInnerProduct) {
            maxInnerProduct = innerProduct;
            scale = current;
        }

        if (code < maxCode) {
            const double following = static_cast<double>(code + 1) / p_absoluteValues[index];
            if (following < end) {
                next.emplace(following, index);
            }
        }
    }
    return scale;
}

template <typename T, typename CodeType>
inline void QuantizeScalar(const T* p_data,
                           const T* p_centroid,
                           std::size_t p_dim,
                           std::size_t p_totalBits,
                           CodeType* p_codes,
                           T& p_delta,
                           T& p_lowerValue)
{
    const std::size_t extraBits = p_totalBits - 1;
    const int extraMask = (1 << extraBits) - 1;
    std::vector<T> residual(p_dim);
    std::vector<T> absoluteValues(p_dim);
    T residualSquaredNorm = 0;
    for (std::size_t i = 0; i < p_dim; ++i) {
        residual[i] = p_data[i] - p_centroid[i];
        residualSquaredNorm += residual[i] * residual[i];
    }

    const T residualNorm = std::sqrt(residualSquaredNorm);
    if (residualNorm <= std::numeric_limits<T>::epsilon()) {
        std::fill(p_codes, p_codes + p_dim, CodeType(0));
        p_delta = 0;
        p_lowerValue = 0;
        return;
    }

    if (extraBits > 0) {
        for (std::size_t i = 0; i < p_dim; ++i) {
            absoluteValues[i] = std::abs(residual[i]) / residualNorm;
        }
        const double scale = BestRescaleFactor(absoluteValues.data(), p_dim, extraBits);
        for (std::size_t i = 0; i < p_dim; ++i) {
            int code = static_cast<int>(scale * absoluteValues[i] + 1e-5);
            if (code >= (1 << extraBits)) {
                code = extraMask;
            }
            if (residual[i] < 0) {
                code = (~code) & extraMask;
            }
            p_codes[i] = static_cast<CodeType>(code);
        }
    } else {
        std::fill(p_codes, p_codes + p_dim, CodeType(0));
    }

    const T center = -(static_cast<T>(1 << extraBits) - T(0.5));
    T codeSquaredNorm = 0;
    T residualCodeInnerProduct = 0;
    for (std::size_t i = 0; i < p_dim; ++i) {
        p_codes[i] = static_cast<CodeType>(
            p_codes[i] + (static_cast<CodeType>(residual[i] > 0) << extraBits));
        const T centeredCode = static_cast<T>(p_codes[i]) + center;
        codeSquaredNorm += centeredCode * centeredCode;
        residualCodeInnerProduct += residual[i] * centeredCode;
    }

    if (codeSquaredNorm <= std::numeric_limits<T>::epsilon()) {
        p_delta = 0;
    } else {
        const T codeNorm = std::sqrt(codeSquaredNorm);
        const T cosineSimilarity =
            residualCodeInnerProduct / (residualNorm * codeNorm);
        p_delta = residualNorm / codeNorm * cosineSimilarity;
    }
    p_lowerValue = p_delta * center;
}

template <typename T, typename CodeType>
inline void Reconstruct(const CodeType* p_codes,
                        T p_delta,
                        T p_lowerValue,
                        std::size_t p_dim,
                        T* p_result)
{
    for (std::size_t i = 0; i < p_dim; ++i) {
        p_result[i] = static_cast<T>(p_codes[i]) * p_delta + p_lowerValue;
    }
}

} // namespace global_scalar
} // namespace quant
} // namespace rabitqlib
