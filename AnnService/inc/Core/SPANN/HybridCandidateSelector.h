// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_HYBRIDCANDIDATESELECTOR_H_
#define _SPTAG_SPANN_HYBRIDCANDIDATESELECTOR_H_

#include "inc/Core/SPANN/HybridDistance.h"
#include "inc/Core/Common/QueryResultSet.h"
#include "inc/Core/VectorIndex.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace SPTAG
{
namespace SPANN
{

struct HybridScoredCandidate
{
    SizeType m_head = -1;
    float m_distance = MaxDist;
};

template <typename ValueType>
class HybridCandidateSelector
{
public:
    HybridCandidateSelector(
        VectorIndex* p_index,
        const std::uint32_t* p_headAttributes,
        SizeType p_headCount,
        int p_numTagColumns,
        const HybridDistanceConfig* p_distance)
        : m_index(p_index),
          m_headAttributes(p_headAttributes),
          m_headCount(p_headCount),
          m_numTagColumns(p_numTagColumns),
          m_distance(p_distance)
    {
    }

    bool Build(std::string& p_error)
    {
        p_error.clear();
        if (m_index == nullptr || m_index->m_pQuantizer != nullptr ||
            m_headAttributes == nullptr || m_headCount <= 0 ||
            m_index->GetNumSamples() != m_headCount ||
            m_numTagColumns <= 0 || m_distance == nullptr ||
            !m_distance->m_enabled) {
            p_error =
                "hybrid candidates require matching raw heads and attributes";
            return false;
        }
        m_categorical = m_distance->m_categorical;
        std::sort(
            m_categorical.begin(), m_categorical.end(),
            [](const HybridWeightedColumn& p_left,
               const HybridWeightedColumn& p_right) {
                if (p_left.m_weight != p_right.m_weight) {
                    return p_left.m_weight > p_right.m_weight;
                }
                return p_left.m_column < p_right.m_column;
            });
        for (const auto& column : m_categorical) {
            m_categoricalColumns.push_back(column.m_column);
        }
        m_categoricalBuckets.resize(m_categoricalColumns.size());
        for (SizeType head = 0; head < m_headCount; ++head) {
            const auto* attributes = Attributes(head);
            for (size_t prefix = 1;
                 prefix <= m_categoricalColumns.size(); ++prefix) {
                m_categoricalBuckets[prefix - 1]
                    [AttributeKey(
                        attributes, m_categoricalColumns, prefix)]
                        .push_back(head);
            }
        }

        m_numericOrders.resize(m_distance->m_numeric.size());
        for (size_t numeric = 0;
             numeric < m_distance->m_numeric.size(); ++numeric) {
            auto& order = m_numericOrders[numeric];
            order.reserve(static_cast<size_t>(m_headCount));
            const int column =
                m_distance->m_numeric[numeric].m_column;
            for (SizeType head = 0; head < m_headCount; ++head) {
                order.emplace_back(Attributes(head)[column], head);
            }
            std::sort(order.begin(), order.end());
        }
        return true;
    }

    bool Select(const ValueType* p_queryVector,
                const std::uint32_t* p_queryAttributes,
                SizeType p_queryIdentity,
                SizeType p_selfHead,
                int p_candidateCount,
                int p_resultCount,
                std::vector<HybridScoredCandidate>& p_results,
                std::uint64_t* p_checkedLeaves,
                std::string& p_error) const
    {
        p_results.clear();
        p_error.clear();
        if (p_queryVector == nullptr || p_queryAttributes == nullptr ||
            p_candidateCount <= 0 || p_resultCount <= 0 ||
            m_index == nullptr || m_headCount <= 0) {
            p_error = "invalid hybrid candidate query";
            return false;
        }

        std::unordered_set<SizeType> seen;
        seen.reserve(
            static_cast<size_t>(p_candidateCount) * 4 + 1);
        std::vector<SizeType> candidates;
        candidates.reserve(
            static_cast<size_t>(p_candidateCount) * 3);
        COMMON::QueryResultSet<ValueType> query(
            p_queryVector,
            (std::min)(
                static_cast<int>(m_headCount),
                p_candidateCount + 1));
        if (m_index->SearchIndex(query) != ErrorCode::Success) {
            p_error =
                "pure-distance search failed during hybrid candidate selection";
            return false;
        }
        if (p_checkedLeaves != nullptr) {
            *p_checkedLeaves += static_cast<std::uint64_t>(
                (std::max)(0, query.GetScanned()));
        }
        for (int result = 0; result < query.GetResultNum(); ++result) {
            const SizeType candidate =
                query.GetResult(result)->VID;
            if (candidate >= 0 && candidate != p_selfHead &&
                seen.insert(candidate).second) {
                candidates.push_back(candidate);
            }
        }

        for (size_t prefix = m_categoricalColumns.size();
             prefix > 0 &&
             candidates.size() <
                 static_cast<size_t>(p_candidateCount) * 2;
             --prefix) {
            const auto key = AttributeKey(
                p_queryAttributes,
                m_categoricalColumns, prefix);
            const auto bucket =
                m_categoricalBuckets[prefix - 1].find(key);
            if (bucket !=
                m_categoricalBuckets[prefix - 1].end()) {
                AddBucketCandidates(
                    bucket->second, p_queryIdentity,
                    p_selfHead, p_candidateCount,
                    seen, candidates);
            }
        }

        const int numericWindow = (std::max)(
            1, p_candidateCount /
                   (std::max)(
                       1, static_cast<int>(
                              m_distance->m_numeric.size())));
        for (size_t numeric = 0;
             numeric < m_numericOrders.size(); ++numeric) {
            const auto& order = m_numericOrders[numeric];
            const int column =
                m_distance->m_numeric[numeric].m_column;
            const auto lower = std::lower_bound(
                order.begin(), order.end(),
                std::make_pair(
                    p_queryAttributes[column],
                    (std::numeric_limits<SizeType>::min)()));
            const size_t position = static_cast<size_t>(
                lower - order.begin());
            for (int delta = 0; delta < numericWindow; ++delta) {
                if (position >=
                    static_cast<size_t>(delta + 1)) {
                    AddCandidate(
                        order[position -
                              static_cast<size_t>(
                                  delta + 1)]
                            .second,
                        p_selfHead, seen, candidates);
                }
                if (position + static_cast<size_t>(delta) <
                    order.size()) {
                    AddCandidate(
                        order[position +
                              static_cast<size_t>(delta)]
                            .second,
                        p_selfHead, seen, candidates);
                }
            }
        }

        std::vector<HybridScoredCandidate> scored;
        scored.reserve(candidates.size());
        for (SizeType candidate : candidates) {
            const float vectorDistance =
                m_index->ComputeDistance(
                    p_queryVector,
                    m_index->GetSample(candidate));
            scored.push_back({
                candidate,
                m_distance->PairDistance(
                    vectorDistance, p_queryAttributes,
                    Attributes(candidate), m_numTagColumns)});
        }
        std::sort(
            scored.begin(), scored.end(),
            [](const HybridScoredCandidate& p_left,
               const HybridScoredCandidate& p_right) {
                if (p_left.m_distance != p_right.m_distance) {
                    return p_left.m_distance <
                        p_right.m_distance;
                }
                return p_left.m_head < p_right.m_head;
            });

        p_results.reserve(
            static_cast<size_t>(p_resultCount));
        for (const auto& candidate : scored) {
            bool keep = true;
            for (const auto& selected : p_results) {
                const float betweenVector =
                    m_index->ComputeDistance(
                        m_index->GetSample(selected.m_head),
                        m_index->GetSample(candidate.m_head));
                const float betweenHybrid =
                    m_distance->PairDistance(
                        betweenVector,
                        Attributes(selected.m_head),
                        Attributes(candidate.m_head),
                        m_numTagColumns);
                if (betweenHybrid < candidate.m_distance) {
                    keep = false;
                    break;
                }
            }
            if (keep) {
                p_results.push_back(candidate);
                if (static_cast<int>(p_results.size()) ==
                    p_resultCount) {
                    break;
                }
            }
        }
        return true;
    }

private:
    using BucketMap =
        std::unordered_map<std::string, std::vector<SizeType>>;

    const std::uint32_t* Attributes(SizeType p_head) const
    {
        return m_headAttributes +
            static_cast<size_t>(p_head) *
                static_cast<size_t>(m_numTagColumns);
    }

    static std::string AttributeKey(
        const std::uint32_t* p_attributes,
        const std::vector<int>& p_columns,
        size_t p_columnCount)
    {
        std::string key;
        key.resize(
            p_columnCount * sizeof(std::uint32_t));
        for (size_t i = 0; i < p_columnCount; ++i) {
            std::memcpy(
                &key[i * sizeof(std::uint32_t)],
                p_attributes + p_columns[i],
                sizeof(std::uint32_t));
        }
        return key;
    }

    static void AddCandidate(
        SizeType p_candidate,
        SizeType p_selfHead,
        std::unordered_set<SizeType>& p_seen,
        std::vector<SizeType>& p_candidates)
    {
        if (p_candidate != p_selfHead &&
            p_seen.insert(p_candidate).second) {
            p_candidates.push_back(p_candidate);
        }
    }

    static void AddBucketCandidates(
        const std::vector<SizeType>& p_bucket,
        SizeType p_queryIdentity,
        SizeType p_selfHead,
        int p_limit,
        std::unordered_set<SizeType>& p_seen,
        std::vector<SizeType>& p_candidates)
    {
        if (p_limit <= 0 || p_bucket.empty()) return;
        const size_t count = (std::min)(
            p_bucket.size(), static_cast<size_t>(p_limit));
        const size_t stride = (std::max)(
            static_cast<size_t>(1),
            p_bucket.size() / count);
        const size_t start =
            (static_cast<size_t>(p_queryIdentity) *
             11400714819323198485ULL) %
            p_bucket.size();
        for (size_t i = 0; i < count; ++i) {
            AddCandidate(
                p_bucket[(start + i * stride) %
                         p_bucket.size()],
                p_selfHead, p_seen, p_candidates);
        }
    }

    VectorIndex* m_index = nullptr;
    const std::uint32_t* m_headAttributes = nullptr;
    SizeType m_headCount = 0;
    int m_numTagColumns = 0;
    const HybridDistanceConfig* m_distance = nullptr;
    std::vector<HybridWeightedColumn> m_categorical;
    std::vector<int> m_categoricalColumns;
    std::vector<BucketMap> m_categoricalBuckets;
    std::vector<
        std::vector<std::pair<std::uint32_t, SizeType>>>
        m_numericOrders;
};

} // namespace SPANN
} // namespace SPTAG

#endif
