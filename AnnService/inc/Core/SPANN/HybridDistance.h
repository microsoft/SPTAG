// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_HYBRIDDISTANCE_H_
#define _SPTAG_SPANN_HYBRIDDISTANCE_H_

#include "inc/Core/Common.h"
#include "inc/Core/Common/CommonUtils.h"
#include "inc/Core/Cache/PostingSignature.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace SPTAG
{
namespace SPANN
{

struct HybridWeightedColumn
{
    int m_column = -1;
    float m_weight = 0.0f;
};

struct HybridDistanceConfig
{
    bool m_enabled = false;
    float m_vectorWeight = 1.0f;
    std::vector<HybridWeightedColumn> m_categorical;
    std::vector<HybridWeightedColumn> m_numeric;

    static bool Parse(const std::string& p_categoricalColumns,
                      const std::string& p_categoricalWeights,
                      const std::string& p_numericColumns,
                      const std::string& p_numericWeights,
                      int p_numColumns,
                      float p_vectorWeight,
                      HybridDistanceConfig& p_config,
                      std::string& p_error)
    {
        p_config = HybridDistanceConfig();
        p_error.clear();
        if (!std::isfinite(p_vectorWeight) || p_vectorWeight < 0.0f) {
            p_error = "HybridVectorWeight must be finite and non-negative";
            return false;
        }
        p_config.m_vectorWeight = p_vectorWeight;

        std::vector<int> categoricalColumns;
        std::vector<float> categoricalWeights;
        std::vector<int> numericColumns;
        std::vector<float> numericWeights;
        if (!ParseCSV(p_categoricalColumns, categoricalColumns) ||
            !ParseCSV(p_categoricalWeights, categoricalWeights) ||
            !ParseCSV(p_numericColumns, numericColumns) ||
            !ParseCSV(p_numericWeights, numericWeights)) {
            p_error = "hybrid column and weight lists must be comma-separated numbers";
            return false;
        }
        if (categoricalColumns.size() != categoricalWeights.size()) {
            p_error = "HybridCategoricalCols and HybridCategoricalWeights must have equal lengths";
            return false;
        }
        if (numericColumns.size() != numericWeights.size()) {
            p_error = "HybridNumericCols and HybridNumericWeights must have equal lengths";
            return false;
        }

        std::unordered_set<int> seen;
        auto append = [&](const std::vector<int>& p_columns,
                          const std::vector<float>& p_weights,
                          std::vector<HybridWeightedColumn>& p_output,
                          const char* p_kind) {
            for (size_t i = 0; i < p_columns.size(); ++i) {
                const int column = p_columns[i];
                const float weight = p_weights[i];
                if (column < 0 || column >= p_numColumns) {
                    p_error = std::string("invalid ") + p_kind + " hybrid column " +
                        std::to_string(column);
                    return false;
                }
                if (!std::isfinite(weight) || weight < 0.0f) {
                    p_error = std::string(p_kind) +
                        " hybrid weights must be finite and non-negative";
                    return false;
                }
                if (!seen.insert(column).second) {
                    p_error = "a hybrid attribute column may be configured only once";
                    return false;
                }
                if (weight > 0.0f) p_output.push_back({column, weight});
            }
            return true;
        };

        if (!append(categoricalColumns, categoricalWeights,
                    p_config.m_categorical, "categorical") ||
            !append(numericColumns, numericWeights,
                    p_config.m_numeric, "numeric")) {
            return false;
        }

        p_config.m_enabled = !p_config.m_categorical.empty() ||
            !p_config.m_numeric.empty();
        if (!p_config.m_enabled) {
            p_error = "hybrid distance requires at least one positive attribute weight";
            return false;
        }
        return true;
    }

    double AttributeDistance(const std::uint32_t* p_left,
                             const std::uint32_t* p_right,
                             int p_numColumns) const
    {
        if (p_left == nullptr || p_right == nullptr || p_numColumns <= 0) {
            return 0.0;
        }
        double distance = 0.0;
        for (const auto& column : m_categorical) {
            if (column.m_column >= p_numColumns) continue;
            distance += static_cast<double>(column.m_weight) *
                (p_left[column.m_column] == p_right[column.m_column] ? 0.0 : 1.0);
        }
        for (const auto& column : m_numeric) {
            if (column.m_column >= p_numColumns) continue;
            distance += static_cast<double>(column.m_weight) *
                UnsignedDifference(
                    p_left[column.m_column], p_right[column.m_column]);
        }
        return distance;
    }

    float Combine(float p_vectorDistance, double p_attributeDistance) const
    {
        const double distance =
            static_cast<double>(m_vectorWeight) * p_vectorDistance +
            p_attributeDistance;
        if (std::isnan(distance) ||
            distance >= static_cast<double>((std::numeric_limits<float>::max)())) {
            return (std::numeric_limits<float>::max)();
        }
        if (distance <=
            static_cast<double>((std::numeric_limits<float>::lowest)())) {
            return (std::numeric_limits<float>::lowest)();
        }
        return static_cast<float>(distance);
    }

    float PairDistance(float p_vectorDistance,
                       const std::uint32_t* p_left,
                       const std::uint32_t* p_right,
                       int p_numColumns) const
    {
        return Combine(
            p_vectorDistance,
            AttributeDistance(p_left, p_right, p_numColumns));
    }

    double PredicateDistance(const std::uint32_t* p_values,
                             int p_numColumns,
                             const Cache::DNFPredicate* p_dnf,
                             const std::vector<std::pair<int, std::uint32_t>>&
                                 p_flatCategoricalValues) const
    {
        if (p_values == nullptr || p_numColumns <= 0) return 0.0;
        if (p_dnf != nullptr && !p_dnf->Empty()) {
            double best = (std::numeric_limits<double>::infinity)();
            for (const auto& clause : p_dnf->clauses) {
                if (clause.lits.empty()) continue;
                best = (std::min)(
                    best, ClauseDistance(p_values, p_numColumns, clause));
            }
            return std::isfinite(best) ? best : 0.0;
        }

        double best =
            (std::numeric_limits<double>::infinity)();
        for (const auto& value :
             p_flatCategoricalValues) {
            const auto column = std::find_if(
                m_categorical.begin(),
                m_categorical.end(),
                [&](const HybridWeightedColumn& candidate) {
                    return candidate.m_column ==
                        value.first;
                });
            if (column == m_categorical.end()) {
                return 0.0;
            }
            if (column->m_column < 0 ||
                column->m_column >= p_numColumns) {
                continue;
            }
            best = (std::min)(
                best,
                p_values[column->m_column] ==
                        value.second
                    ? 0.0
                    : static_cast<double>(
                          column->m_weight));
        }
        return std::isfinite(best) ? best : 0.0;
    }

private:
    template <typename ValueType>
    static bool ParseCSV(const std::string& p_csv,
                         std::vector<ValueType>& p_values)
    {
        p_values.clear();
        if (p_csv.empty()) return true;
        std::stringstream stream(p_csv);
        std::string token;
        while (std::getline(stream, token, ',')) {
            const size_t first = token.find_first_not_of(" \t");
            const size_t last = token.find_last_not_of(" \t");
            if (first == std::string::npos) return false;
            token = token.substr(first, last - first + 1);
            char* end = nullptr;
            if constexpr (std::is_same<ValueType, int>::value) {
                const long value = std::strtol(token.c_str(), &end, 10);
                if (end == token.c_str() || *end != '\0' ||
                    value < (std::numeric_limits<int>::min)() ||
                    value > (std::numeric_limits<int>::max)()) {
                    return false;
                }
                p_values.push_back(static_cast<int>(value));
            }
            else {
                const float value = std::strtof(token.c_str(), &end);
                if (end == token.c_str() || *end != '\0') return false;
                p_values.push_back(value);
            }
        }
        return true;
    }

    static double UnsignedDifference(std::uint32_t p_left,
                                     std::uint32_t p_right)
    {
        return p_left >= p_right
            ? static_cast<double>(p_left - p_right)
            : static_cast<double>(p_right - p_left);
    }

    const HybridWeightedColumn* CategoricalColumn(int p_column) const
    {
        const auto it = std::find_if(
            m_categorical.begin(), m_categorical.end(),
            [p_column](const HybridWeightedColumn& p_entry) {
                return p_entry.m_column == p_column;
            });
        return it == m_categorical.end() ? nullptr : &*it;
    }

    const HybridWeightedColumn* NumericColumn(int p_column) const
    {
        const auto it = std::find_if(
            m_numeric.begin(), m_numeric.end(),
            [p_column](const HybridWeightedColumn& p_entry) {
                return p_entry.m_column == p_column;
            });
        return it == m_numeric.end() ? nullptr : &*it;
    }

    double ClauseDistance(const std::uint32_t* p_values,
                          int p_numColumns,
                          const Cache::DNFClause& p_clause) const
    {
        double distance = 0.0;
        std::unordered_set<int> handledCategorical;
        std::unordered_set<int> handledNumeric;
        for (const auto& literal : p_clause.lits) {
            const int column = static_cast<int>(literal.col);
            if (column < 0 || column >= p_numColumns) continue;
            if (literal.kind == 0) {
                const auto* configured = CategoricalColumn(column);
                if (configured == nullptr ||
                    !handledCategorical.insert(column).second) {
                    continue;
                }
                bool haveExpected = false;
                std::uint32_t expected = 0;
                for (const auto& candidate : p_clause.lits) {
                    if (candidate.kind == 0 &&
                        static_cast<int>(candidate.col) == column &&
                        candidate.op == Cache::DNF_EQ) {
                        if (haveExpected && expected != candidate.val) {
                            return (std::numeric_limits<double>::infinity)();
                        }
                        expected = candidate.val;
                        haveExpected = true;
                    }
                }
                if (haveExpected && p_values[column] != expected) {
                    distance += configured->m_weight;
                }
                continue;
            }

            const auto* configured = NumericColumn(column);
            if (configured == nullptr ||
                !handledNumeric.insert(column).second) {
                continue;
            }
            std::uint64_t lower = 0;
            std::uint64_t upper =
                (std::numeric_limits<std::uint32_t>::max)();
            bool constrained = false;
            bool impossible = false;
            for (const auto& candidate : p_clause.lits) {
                if (candidate.kind == 0 ||
                    static_cast<int>(candidate.col) != column) {
                    continue;
                }
                constrained = true;
                switch (candidate.op) {
                case Cache::DNF_EQ:
                    lower = (std::max)(
                        lower, static_cast<std::uint64_t>(candidate.val));
                    upper = (std::min)(
                        upper, static_cast<std::uint64_t>(candidate.val));
                    break;
                case Cache::DNF_LT:
                    if (candidate.val == 0) {
                        impossible = true;
                    }
                    else {
                        upper = (std::min)(
                            upper,
                            static_cast<std::uint64_t>(candidate.val - 1));
                    }
                    break;
                case Cache::DNF_LE:
                    upper = (std::min)(
                        upper, static_cast<std::uint64_t>(candidate.val));
                    break;
                case Cache::DNF_GT:
                    if (candidate.val ==
                        (std::numeric_limits<std::uint32_t>::max)()) {
                        impossible = true;
                    }
                    else {
                        lower = (std::max)(
                            lower,
                            static_cast<std::uint64_t>(candidate.val) + 1);
                    }
                    break;
                case Cache::DNF_GE:
                    lower = (std::max)(
                        lower, static_cast<std::uint64_t>(candidate.val));
                    break;
                default:
                    break;
                }
            }
            if (!constrained) continue;
            if (impossible || lower > upper) {
                return (std::numeric_limits<double>::infinity)();
            }
            const std::uint64_t value = p_values[column];
            double delta = 0.0;
            if (value < lower) {
                delta = static_cast<double>(lower - value);
            }
            else if (value > upper) {
                delta = static_cast<double>(value - upper);
            }
            distance += static_cast<double>(configured->m_weight) * delta;
        }
        return distance;
    }
};

struct HybridQueryDistanceTransform
{
    double m_scale = 1.0;
    double m_offset = 0.0;

    float Apply(float p_distance) const
    {
        const double transformed =
            m_offset + m_scale * static_cast<double>(p_distance);
        if (std::isnan(transformed) ||
            transformed >=
                static_cast<double>((std::numeric_limits<float>::max)())) {
            return (std::numeric_limits<float>::max)();
        }
        if (transformed <=
            static_cast<double>((std::numeric_limits<float>::lowest)())) {
            return (std::numeric_limits<float>::lowest)();
        }
        return static_cast<float>(transformed);
    }

    template <typename ValueType>
    static HybridQueryDistanceTransform ForCosine(
        const ValueType* p_query,
        DimensionType p_dimension)
    {
        HybridQueryDistanceTransform transform;
        if (p_query == nullptr || p_dimension <= 0) {
            return transform;
        }
        double squaredNorm = 0.0;
        for (DimensionType dimension = 0;
             dimension < p_dimension; ++dimension) {
            const double value =
                static_cast<double>(p_query[dimension]);
            squaredNorm += value * value;
        }
        const double base = static_cast<double>(
            COMMON::Utils::GetBase<ValueType>());
        const double baseSquared = base * base;
        if (squaredNorm < 1e-12) {
            transform.m_scale = 0.0;
            transform.m_offset = baseSquared;
            return transform;
        }
        transform.m_scale = base / std::sqrt(squaredNorm);
        transform.m_offset =
            baseSquared * (1.0 - transform.m_scale);
        return transform;
    }
};

class HybridGenerationFingerprint
{
public:
    HybridGenerationFingerprint(const HybridDistanceConfig& p_distance,
                                int p_numTagColumns,
                                int p_degree,
                                int p_candidateCount)
    {
        AddConfig(static_cast<std::uint64_t>(p_numTagColumns));
        AddConfig(static_cast<std::uint64_t>(p_degree));
        AddConfig(static_cast<std::uint64_t>(p_candidateCount));
        AddConfig(FloatBits(p_distance.m_vectorWeight));
        AddConfig(p_distance.m_categorical.size());
        for (const auto& column : p_distance.m_categorical) {
            AddConfig(static_cast<std::uint64_t>(column.m_column));
            AddConfig(FloatBits(column.m_weight));
        }
        AddConfig(p_distance.m_numeric.size());
        for (const auto& column : p_distance.m_numeric) {
            AddConfig(static_cast<std::uint64_t>(column.m_column));
            AddConfig(FloatBits(column.m_weight));
        }
        m_numTagColumns = p_numTagColumns;
    }

    void AddHead(SizeType p_vectorID,
                 const std::uint32_t* p_attributes)
    {
        std::uint64_t item = Mix(
            static_cast<std::uint64_t>(p_vectorID));
        for (int column = 0;
             column < m_numTagColumns; ++column) {
            item = Mix(item ^
                static_cast<std::uint64_t>(
                    p_attributes[column]));
        }
        m_headXor ^= Mix(item);
        m_headSum += item;
        ++m_headCount;
    }

    std::uint64_t Value() const
    {
        std::uint64_t value = Mix(
            m_config ^ m_headXor ^
            Mix(m_headSum) ^ Mix(m_headCount));
        return value == 0
            ? 0x9e3779b97f4a7c15ULL
            : value;
    }

private:
    static std::uint64_t Mix(std::uint64_t p_value)
    {
        p_value += 0x9e3779b97f4a7c15ULL;
        p_value =
            (p_value ^ (p_value >> 30)) *
            0xbf58476d1ce4e5b9ULL;
        p_value =
            (p_value ^ (p_value >> 27)) *
            0x94d049bb133111ebULL;
        return p_value ^ (p_value >> 31);
    }

    static std::uint32_t FloatBits(float p_value)
    {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &p_value, sizeof(bits));
        return bits;
    }

    void AddConfig(std::uint64_t p_value)
    {
        m_config = Mix(m_config ^ p_value);
    }

    int m_numTagColumns = 0;
    std::uint64_t m_config =
        0x484252494447454eULL;
    std::uint64_t m_headXor = 0;
    std::uint64_t m_headSum = 0;
    std::uint64_t m_headCount = 0;
};

struct HybridPostingLayoutStats
{
    double m_averageRecords = 0.0;
    double m_averagePages = 0.0;
    double m_averageBytes = 0.0;
    double m_uniqueRatio = 1.0;
    double m_enrichment = 1.0;
    double m_headFixedCostUS = 0.0;
    double m_headPerPostingCostUS = 0.0;
};

struct HybridRouteCostConfig
{
    double m_resultSafety = 2.0;
    double m_ioFixedUS = 8.0;
    double m_pageUS = 4.0;
    double m_vectorUS = 0.04;
    double m_bytesPerUS = 4000.0;
};

struct HybridRouteEstimate
{
    int m_postings = 0;
    double m_expectedMatchesPerPosting = 0.0;
    double m_costUS = (std::numeric_limits<double>::infinity)();
};

inline HybridRouteEstimate EstimateHybridRouteCost(
    int p_resultCount,
    int p_basePostings,
    int p_maxPostings,
    double p_selectivity,
    const HybridPostingLayoutStats& p_layout,
    const HybridRouteCostConfig& p_cost)
{
    HybridRouteEstimate estimate;
    const double selectivity = (std::max)(
        1e-9, (std::min)(1.0, p_selectivity));
    const double enrichment = (std::max)(1e-9, p_layout.m_enrichment);
    const double records = (std::max)(1e-9, p_layout.m_averageRecords);
    const double uniqueRatio = (std::max)(
        1e-9, (std::min)(1.0, p_layout.m_uniqueRatio));
    estimate.m_expectedMatchesPerPosting =
        (std::min)(records, selectivity * enrichment * records * uniqueRatio);

    const double safety = (std::max)(1.0, p_cost.m_resultSafety);
    const long double requiredPostings = std::ceil(
        static_cast<long double>(safety) *
        static_cast<long double>(
            (std::max)(1, p_resultCount)) /
        static_cast<long double>(
            estimate.m_expectedMatchesPerPosting));
    std::uint64_t coveragePostings =
        requiredPostings >=
                static_cast<long double>(
                    (std::numeric_limits<
                        std::uint64_t>::max)())
            ? (std::numeric_limits<
                  std::uint64_t>::max)()
            : static_cast<std::uint64_t>(
                  requiredPostings);
    std::uint64_t target = (std::max)(
        static_cast<std::uint64_t>(
            (std::max)(1, p_basePostings)),
        coveragePostings);
    if (p_maxPostings > 0) {
        target = (std::min)(
            target,
            static_cast<std::uint64_t>(
                p_maxPostings));
    }
    target = (std::min)(
        target,
        static_cast<std::uint64_t>(
            (std::numeric_limits<int>::max)()));
    estimate.m_postings =
        static_cast<int>(target);

    const double perPosting =
        (std::max)(0.0, p_cost.m_ioFixedUS) +
        (std::max)(0.0, p_layout.m_averagePages) *
            (std::max)(0.0, p_cost.m_pageUS) +
        (std::max)(0.0, p_layout.m_averageBytes) /
            (std::max)(1e-9, p_cost.m_bytesPerUS) +
        records * (std::max)(0.0, p_cost.m_vectorUS);
    estimate.m_costUS =
        (std::max)(0.0, p_layout.m_headFixedCostUS) +
        static_cast<double>(estimate.m_postings) * perPosting;
    estimate.m_costUS += static_cast<double>(estimate.m_postings) *
        (std::max)(0.0, p_layout.m_headPerPostingCostUS);
    return estimate;
}

} // namespace SPANN
} // namespace SPTAG

#endif
