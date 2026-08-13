// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_HYBRIDHEADGRAPH_H_
#define _SPTAG_SPANN_HYBRIDHEADGRAPH_H_

#include "inc/Core/SPANN/HybridCandidateSelector.h"
#include "inc/Core/SPANN/HybridDistance.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Helper/AtomicFile.h"
#include "inc/Helper/HeadCrossEdges.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
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

struct HybridHeadGraphNode
{
    int m_nodeID = -1;
    SizeType m_headCount = 0;
    std::vector<std::uint32_t> m_attributes;
    std::vector<SizeType> m_neighbors;

    const std::uint32_t* Attributes(SizeType p_head, int p_numColumns) const
    {
        if (p_head < 0 || p_head >= m_headCount || p_numColumns <= 0 ||
            m_attributes.size() !=
                static_cast<size_t>(m_headCount) *
                    static_cast<size_t>(p_numColumns)) {
            return nullptr;
        }
        return m_attributes.data() +
            static_cast<size_t>(p_head) * static_cast<size_t>(p_numColumns);
    }

    const SizeType* Neighbors(SizeType p_head, int p_degree) const
    {
        if (p_head < 0 || p_head >= m_headCount || p_degree <= 0 ||
            m_neighbors.size() !=
                static_cast<size_t>(m_headCount) *
                    static_cast<size_t>(p_degree)) {
            return nullptr;
        }
        return m_neighbors.data() +
            static_cast<size_t>(p_head) * static_cast<size_t>(p_degree);
    }
};

class HybridHeadGraph
{
public:
    int m_numTagColumns = 0;
    int m_degree = 0;
    std::uint64_t m_generationFingerprint = 0;
    std::uint64_t m_contentFingerprint = 0;
    std::uint64_t m_edgeBodyFingerprint = 0;
    std::vector<HybridHeadGraphNode> m_nodes;

    void Clear()
    {
        m_numTagColumns = 0;
        m_degree = 0;
        m_generationFingerprint = 0;
        m_contentFingerprint = 0;
        m_edgeBodyFingerprint = 0;
        m_nodes.clear();
    }

    bool SaveCrossEdges(
        const std::string& p_path,
        const std::vector<std::vector<SizeType>>& p_headVectorIDs,
        int p_searchTopK,
        std::string& p_error) const
    {
        p_error.clear();
        if (m_nodes.size() != 1 ||
            p_headVectorIDs.size() != 1 ||
            m_nodes[0].m_headCount <= 0 ||
            p_headVectorIDs[0].size() !=
                static_cast<size_t>(m_nodes[0].m_headCount) ||
            m_degree <= 0 || p_searchTopK <= 0 ||
            m_generationFingerprint == 0 ||
            m_contentFingerprint == 0 ||
            m_edgeBodyFingerprint == 0 ||
            m_nodes[0].m_neighbors.size() !=
                static_cast<size_t>(m_nodes[0].m_headCount) *
                    static_cast<size_t>(m_degree)) {
            p_error = "invalid hybrid cross-edge graph";
            return false;
        }
        std::uint64_t bodyFingerprint = 0;
        if (!ComputeEdgeBodyFingerprint(
                p_headVectorIDs,
                bodyFingerprint, p_error)) {
            return false;
        }
        if (bodyFingerprint != m_edgeBodyFingerprint) {
            p_error =
                "hybrid cross-edge body changed after fingerprinting";
            return false;
        }

        const std::string temporary = p_path + ".tmp";
        FILE* file = std::fopen(temporary.c_str(), "wb");
        if (file == nullptr) {
            p_error = "cannot create " + temporary;
            return false;
        }

        Helper::HeadCrossEdgesHeader header{};
        header.magic = Helper::kHeadCrossEdgesMagic;
        header.version =
            Helper::kHybridHeadCrossEdgesVersion;
        header.totalHeads =
            static_cast<std::int32_t>(m_nodes[0].m_headCount);
        header.maxEdgesPerHead = m_degree;
        header.searchTopK = p_searchTopK;
        header.reserved = Helper::kHybridHeadCrossEdgesMarker;
        const Helper::HybridHeadCrossEdgesExtension extension = {
            m_generationFingerprint,
            m_contentFingerprint};
        bool ok =
            std::fwrite(&header, sizeof(header), 1, file) == 1 &&
            std::fwrite(
                &extension, sizeof(extension), 1, file) == 1;
        for (SizeType source = 0;
             ok && source < m_nodes[0].m_headCount; ++source) {
            const SizeType sourceVID =
                p_headVectorIDs[0][static_cast<size_t>(source)];
            if (sourceVID < 0 ||
                sourceVID >
                    (std::numeric_limits<std::int32_t>::max)()) {
                ok = false;
                break;
            }
            const SizeType* neighbors =
                m_nodes[0].Neighbors(source, m_degree);
            std::int32_t edgeCount = 0;
            while (edgeCount < m_degree &&
                   neighbors[edgeCount] >= 0) {
                ++edgeCount;
            }
            const std::int32_t encodedSource =
                static_cast<std::int32_t>(sourceVID);
            ok =
                std::fwrite(
                    &encodedSource, sizeof(encodedSource), 1, file) == 1 &&
                std::fwrite(
                    &edgeCount, sizeof(edgeCount), 1, file) == 1;
            for (std::int32_t edge = 0;
                 ok && edge < edgeCount; ++edge) {
                const SizeType target = neighbors[edge];
                if (target < 0 ||
                    target >= m_nodes[0].m_headCount) {
                    ok = false;
                    break;
                }
                const SizeType targetVID =
                    p_headVectorIDs[0][static_cast<size_t>(target)];
                if (targetVID < 0 ||
                    targetVID >
                        (std::numeric_limits<std::int32_t>::max)()) {
                    ok = false;
                    break;
                }
                const Helper::HeadCrossEdgeEntry entry = {
                    static_cast<std::int32_t>(targetVID), 0.0f};
                ok = std::fwrite(
                         &entry, sizeof(entry), 1, file) == 1;
            }
        }
        if (std::fclose(file) != 0) ok = false;
        if (!ok) {
            std::remove(temporary.c_str());
            p_error = "failed to write hybrid cross edges " + p_path;
            return false;
        }
        if (!Helper::AtomicReplaceFile(temporary, p_path)) {
            std::remove(temporary.c_str());
            p_error = "cannot publish hybrid cross edges " + p_path;
            return false;
        }
        return true;
    }

    template <typename ValueType>
    bool Build(
        const std::vector<std::shared_ptr<VectorIndex>>& p_indexes,
        const std::vector<std::vector<SizeType>>& p_headVectorIDs,
        const std::vector<std::uint32_t>& p_vectorTags,
        int p_numTagColumns,
        const HybridDistanceConfig& p_distance,
        int p_degree,
        int p_candidateCount,
        std::string& p_error)
    {
        Clear();
        p_error.clear();
        if (p_indexes.size() != p_headVectorIDs.size() ||
            p_numTagColumns <= 0 || p_degree <= 0 || p_candidateCount <= 0 ||
            p_vectorTags.size() % static_cast<size_t>(p_numTagColumns) != 0) {
            p_error = "invalid hybrid head graph build inputs";
            return false;
        }
        const SizeType vectorCount = static_cast<SizeType>(
            p_vectorTags.size() / static_cast<size_t>(p_numTagColumns));
        m_numTagColumns = p_numTagColumns;
        m_degree = p_degree;
        m_nodes.resize(p_indexes.size());
        HybridGenerationFingerprint fingerprint(
            p_distance, p_numTagColumns,
            p_degree, p_candidateCount);

        for (size_t nodeIndex = 0; nodeIndex < p_indexes.size(); ++nodeIndex) {
            const auto& index = p_indexes[nodeIndex];
            const auto& headVectorIDs = p_headVectorIDs[nodeIndex];
            if (index == nullptr || index->m_pQuantizer != nullptr ||
                index->GetNumSamples() !=
                    static_cast<SizeType>(headVectorIDs.size())) {
                p_error = "hybrid graph requires matching unquantized head sets";
                Clear();
                return false;
            }
            auto& node = m_nodes[nodeIndex];
            node.m_nodeID = static_cast<int>(nodeIndex);
            node.m_headCount = static_cast<SizeType>(headVectorIDs.size());
            node.m_attributes.resize(
                static_cast<size_t>(node.m_headCount) *
                static_cast<size_t>(p_numTagColumns));
            node.m_neighbors.assign(
                static_cast<size_t>(node.m_headCount) *
                    static_cast<size_t>(p_degree),
                -1);
            for (SizeType local = 0; local < node.m_headCount; ++local) {
                const SizeType vectorID =
                    headVectorIDs[static_cast<size_t>(local)];
                if (vectorID < 0 || vectorID >= vectorCount) {
                    p_error = "head vector id is outside the attribute table";
                    Clear();
                    return false;
                }
                std::memcpy(
                    node.m_attributes.data() +
                        static_cast<size_t>(local) *
                            static_cast<size_t>(p_numTagColumns),
                    p_vectorTags.data() +
                        static_cast<size_t>(vectorID) *
                            static_cast<size_t>(p_numTagColumns),
                    static_cast<size_t>(p_numTagColumns) *
                        sizeof(std::uint32_t));
                fingerprint.AddHead(
                    vectorID,
                    node.m_attributes.data() +
                        static_cast<size_t>(local) *
                            static_cast<size_t>(
                                p_numTagColumns));
            }
            if (!BuildNode<ValueType>(
                    index.get(), p_distance, p_candidateCount, node, p_error)) {
                Clear();
                return false;
            }
        }
        if (!ComputeEdgeBodyFingerprint(
                p_headVectorIDs,
                m_edgeBodyFingerprint, p_error)) {
            Clear();
            return false;
        }
        fingerprint.AddEdgeBody(
            m_edgeBodyFingerprint);
        m_contentFingerprint = fingerprint.Value();
        m_generationFingerprint = m_contentFingerprint;
        return true;
    }

private:
    bool ComputeEdgeBodyFingerprint(
        const std::vector<std::vector<SizeType>>&
            p_headVectorIDs,
        std::uint64_t& p_fingerprint,
        std::string& p_error) const
    {
        if (p_headVectorIDs.size() != m_nodes.size()) {
            p_error =
                "hybrid head IDs do not match graph nodes";
            return false;
        }
        Helper::HeadCrossEdgesBodyFingerprint fingerprint;
        for (size_t nodeIndex = 0;
             nodeIndex < m_nodes.size();
             ++nodeIndex) {
            const auto& node = m_nodes[nodeIndex];
            const auto& headIDs =
                p_headVectorIDs[nodeIndex];
            if (node.m_headCount !=
                    static_cast<SizeType>(headIDs.size()) ||
                node.m_neighbors.size() !=
                    static_cast<size_t>(node.m_headCount) *
                        static_cast<size_t>(m_degree)) {
                p_error =
                    "hybrid head graph dimensions are invalid";
                return false;
            }
            for (SizeType source = 0;
                 source < node.m_headCount;
                 ++source) {
                const SizeType sourceVID =
                    headIDs[static_cast<size_t>(source)];
                if (sourceVID < 0 ||
                    sourceVID >
                        (std::numeric_limits<
                            std::int32_t>::max)()) {
                    p_error =
                        "hybrid source head ID exceeds "
                        "cross-edge format";
                    return false;
                }
                const SizeType* neighbors =
                    node.Neighbors(source, m_degree);
                if (neighbors == nullptr) {
                    p_error =
                        "hybrid head graph row is unavailable";
                    return false;
                }
                std::int32_t edgeCount = 0;
                while (edgeCount < m_degree &&
                       neighbors[edgeCount] >= 0) {
                    ++edgeCount;
                }
                fingerprint.AddRecord(
                    static_cast<std::int32_t>(
                        sourceVID),
                    edgeCount);
                for (std::int32_t edge = 0;
                     edge < edgeCount; ++edge) {
                    const SizeType target =
                        neighbors[edge];
                    if (target < 0 ||
                        target >= node.m_headCount) {
                        p_error =
                            "hybrid target head is outside "
                            "the graph";
                        return false;
                    }
                    const SizeType targetVID =
                        headIDs[
                            static_cast<size_t>(
                                target)];
                    if (targetVID < 0 ||
                        targetVID >
                            (std::numeric_limits<
                                std::int32_t>::max)()) {
                        p_error =
                            "hybrid target head ID exceeds "
                            "cross-edge format";
                        return false;
                    }
                    const Helper::HeadCrossEdgeEntry entry = {
                        static_cast<std::int32_t>(
                            targetVID),
                        0.0f};
                    fingerprint.AddEntry(entry);
                }
            }
        }
        p_fingerprint = fingerprint.Value();
        return true;
    }

    template <typename ValueType>
    bool BuildNode(VectorIndex* p_index,
                   const HybridDistanceConfig& p_distance,
                   int p_candidateCount,
                   HybridHeadGraphNode& p_node,
                   std::string& p_error)
    {
        HybridCandidateSelector<ValueType> selector(
            p_index,
            p_node.m_attributes.data(),
            p_node.m_headCount,
            m_numTagColumns,
            &p_distance);
        if (!selector.Build(p_error)) {
            return false;
        }

        for (SizeType source = 0; source < p_node.m_headCount; ++source) {
            std::vector<HybridScoredCandidate> selected;
            if (!selector.Select(
                static_cast<const ValueType*>(p_index->GetSample(source)),
                p_node.Attributes(source, m_numTagColumns),
                source,
                source,
                p_candidateCount,
                m_degree,
                selected,
                nullptr,
                p_error)) {
                return false;
            }
            SizeType* output =
                p_node.m_neighbors.data() +
                static_cast<size_t>(source) *
                    static_cast<size_t>(m_degree);
            for (size_t edge = 0; edge < selected.size(); ++edge) {
                output[edge] = selected[edge].m_head;
            }
        }
        return true;
    }
};

} // namespace SPANN
} // namespace SPTAG

#endif
