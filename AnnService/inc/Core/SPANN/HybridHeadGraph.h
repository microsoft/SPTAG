// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_HYBRIDHEADGRAPH_H_
#define _SPTAG_SPANN_HYBRIDHEADGRAPH_H_

#include "inc/Core/SPANN/HybridCandidateSelector.h"
#include "inc/Core/SPANN/HybridDistance.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Helper/AtomicFile.h"

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

constexpr std::uint32_t kHybridHeadGraphMagic = 0x47425948U; // HYBG
constexpr std::uint32_t kHybridHeadGraphVersion = 2;

struct HybridHeadGraphHeader
{
    std::uint32_t m_magic = kHybridHeadGraphMagic;
    std::uint32_t m_version = kHybridHeadGraphVersion;
    std::int32_t m_nodeCount = 0;
    std::int32_t m_numTagColumns = 0;
    std::int32_t m_degree = 0;
    std::int32_t m_totalHeads = 0;
    std::uint64_t m_generationFingerprint = 0;
};

struct HybridHeadGraphNodeHeader
{
    std::int32_t m_nodeID = -1;
    std::int32_t m_headCount = 0;
};

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
    std::vector<HybridHeadGraphNode> m_nodes;

    void Clear()
    {
        m_numTagColumns = 0;
        m_degree = 0;
        m_generationFingerprint = 0;
        m_nodes.clear();
    }

    SizeType TotalHeads() const
    {
        SizeType count = 0;
        for (const auto& node : m_nodes) count += node.m_headCount;
        return count;
    }

    size_t TotalEdges() const
    {
        size_t count = 0;
        for (const auto& node : m_nodes) {
            count += static_cast<size_t>(std::count_if(
                node.m_neighbors.begin(), node.m_neighbors.end(),
                [](SizeType p_neighbor) { return p_neighbor >= 0; }));
        }
        return count;
    }

    bool Save(const std::string& p_path, std::string& p_error) const
    {
        p_error.clear();
        if (m_numTagColumns <= 0 || m_degree <= 0 ||
            m_generationFingerprint == 0 ||
            m_nodes.size() >
                static_cast<size_t>((std::numeric_limits<std::int32_t>::max)()) ||
            TotalHeads() >
                (std::numeric_limits<std::int32_t>::max)()) {
            p_error = "invalid hybrid head graph dimensions";
            return false;
        }
        const std::string temporary = p_path + ".tmp";
        FILE* file = std::fopen(temporary.c_str(), "wb");
        if (file == nullptr) {
            p_error = "cannot create " + temporary;
            return false;
        }
        HybridHeadGraphHeader header;
        header.m_nodeCount = static_cast<std::int32_t>(m_nodes.size());
        header.m_numTagColumns = m_numTagColumns;
        header.m_degree = m_degree;
        header.m_totalHeads = static_cast<std::int32_t>(TotalHeads());
        header.m_generationFingerprint =
            m_generationFingerprint;
        bool ok = std::fwrite(&header, sizeof(header), 1, file) == 1;
        for (const auto& node : m_nodes) {
            const size_t attributeCount =
                static_cast<size_t>(node.m_headCount) *
                static_cast<size_t>(m_numTagColumns);
            const size_t neighborCount =
                static_cast<size_t>(node.m_headCount) *
                static_cast<size_t>(m_degree);
            if (node.m_headCount < 0 ||
                node.m_headCount >
                    (std::numeric_limits<std::int32_t>::max)() ||
                node.m_attributes.size() != attributeCount ||
                node.m_neighbors.size() != neighborCount) {
                ok = false;
                break;
            }
            HybridHeadGraphNodeHeader nodeHeader;
            nodeHeader.m_nodeID = node.m_nodeID;
            nodeHeader.m_headCount =
                static_cast<std::int32_t>(node.m_headCount);
            ok = std::fwrite(&nodeHeader, sizeof(nodeHeader), 1, file) == 1 &&
                (attributeCount == 0 ||
                 std::fwrite(node.m_attributes.data(), sizeof(std::uint32_t),
                             attributeCount, file) == attributeCount) &&
                (neighborCount == 0 ||
                 std::fwrite(node.m_neighbors.data(), sizeof(SizeType),
                             neighborCount, file) == neighborCount);
            if (!ok) break;
        }
        if (std::fclose(file) != 0) ok = false;
        if (!ok) {
            std::remove(temporary.c_str());
            p_error = "failed to write complete hybrid head graph " + p_path;
            return false;
        }
        if (!Helper::AtomicReplaceFile(
                temporary, p_path)) {
            std::remove(temporary.c_str());
            p_error = "cannot publish hybrid head graph " + p_path;
            return false;
        }
        return true;
    }

    bool Load(const std::string& p_path,
              const std::vector<SizeType>& p_expectedHeadCounts,
              int p_expectedTagColumns,
              int p_expectedDegree,
              std::string& p_error)
    {
        Clear();
        p_error.clear();
        FILE* file = std::fopen(p_path.c_str(), "rb");
        if (file == nullptr) {
            p_error = "cannot open " + p_path;
            return false;
        }
        HybridHeadGraphHeader header;
        bool ok = std::fread(&header, sizeof(header), 1, file) == 1 &&
            header.m_magic == kHybridHeadGraphMagic &&
            header.m_version == kHybridHeadGraphVersion &&
            header.m_nodeCount ==
                static_cast<std::int32_t>(p_expectedHeadCounts.size()) &&
            header.m_numTagColumns == p_expectedTagColumns &&
            header.m_degree == p_expectedDegree &&
            header.m_nodeCount >= 0 &&
            header.m_numTagColumns > 0 &&
            header.m_degree > 0 &&
            header.m_totalHeads >= 0 &&
            header.m_generationFingerprint != 0;
        if (ok) {
            m_numTagColumns = header.m_numTagColumns;
            m_degree = header.m_degree;
            m_generationFingerprint =
                header.m_generationFingerprint;
            m_nodes.resize(static_cast<size_t>(header.m_nodeCount));
            SizeType totalHeads = 0;
            for (int nodeIndex = 0; nodeIndex < header.m_nodeCount; ++nodeIndex) {
                HybridHeadGraphNodeHeader nodeHeader;
                if (std::fread(&nodeHeader, sizeof(nodeHeader), 1, file) != 1 ||
                    nodeHeader.m_nodeID != nodeIndex ||
                    nodeHeader.m_headCount < 0 ||
                    static_cast<SizeType>(nodeHeader.m_headCount) !=
                        p_expectedHeadCounts[static_cast<size_t>(nodeIndex)]) {
                    ok = false;
                    break;
                }
                auto& node = m_nodes[static_cast<size_t>(nodeIndex)];
                node.m_nodeID = nodeHeader.m_nodeID;
                node.m_headCount = nodeHeader.m_headCount;
                const size_t attributeCount =
                    static_cast<size_t>(node.m_headCount) *
                    static_cast<size_t>(m_numTagColumns);
                const size_t neighborCount =
                    static_cast<size_t>(node.m_headCount) *
                    static_cast<size_t>(m_degree);
                node.m_attributes.resize(attributeCount);
                node.m_neighbors.resize(neighborCount);
                if ((attributeCount > 0 &&
                     std::fread(node.m_attributes.data(),
                                sizeof(std::uint32_t), attributeCount, file) !=
                         attributeCount) ||
                    (neighborCount > 0 &&
                     std::fread(node.m_neighbors.data(), sizeof(SizeType),
                                neighborCount, file) != neighborCount)) {
                    ok = false;
                    break;
                }
                for (SizeType head = 0; head < node.m_headCount && ok; ++head) {
                    bool terminated = false;
                    const SizeType* neighbors =
                        node.Neighbors(head, m_degree);
                    for (int edge = 0; edge < m_degree; ++edge) {
                        const SizeType neighbor = neighbors[edge];
                        if (neighbor < 0) {
                            terminated = true;
                            continue;
                        }
                        if (terminated || neighbor >= node.m_headCount ||
                            neighbor == head) {
                            ok = false;
                            break;
                        }
                    }
                }
                totalHeads += node.m_headCount;
            }
            if (totalHeads != header.m_totalHeads) ok = false;
            if (ok && std::fgetc(file) != EOF) ok = false;
        }
        std::fclose(file);
        if (!ok) {
            Clear();
            p_error = "hybrid head graph format or topology mismatch at " +
                p_path;
        }
        return ok;
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
                p_error = "hybrid graph requires matching unquantized bundle heads";
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
        m_generationFingerprint =
            fingerprint.Value();
        return true;
    }

private:
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
