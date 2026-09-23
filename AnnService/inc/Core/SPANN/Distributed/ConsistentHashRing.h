// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "inc/Core/Common.h"
#include <cstdint>
#include <map>
#include <set>

namespace SPTAG::SPANN {

    /// Consistent hash ring for distributing headIDs across compute nodes.
    /// Uses virtual nodes (vnodes) for balanced distribution.
    /// When nodes are added/removed, only ~1/N of keys are remapped.
    class ConsistentHashRing {
    public:
        explicit ConsistentHashRing(int vnodeCount = 150)
            : m_vnodeCount(vnodeCount) {}

        /// Add a physical node to the ring with its virtual nodes.
        void AddNode(int nodeIndex) {
            for (int i = 0; i < m_vnodeCount; i++) {
                uint32_t h = HashVNode(nodeIndex, i);
                m_ring[h] = nodeIndex;
            }
            m_nodes.insert(nodeIndex);
        }

        /// Remove a physical node and all its virtual nodes from the ring.
        void RemoveNode(int nodeIndex) {
            for (int i = 0; i < m_vnodeCount; i++) {
                uint32_t h = HashVNode(nodeIndex, i);
                m_ring.erase(h);
            }
            m_nodes.erase(nodeIndex);
        }

        /// Find the owner node for a given key (headID).
        /// Returns -1 if the ring is empty.
        int GetOwner(SizeType headID) const {
            if (m_ring.empty()) return -1;
            uint32_t h = HashKey(headID);
            auto it = m_ring.lower_bound(h);
            if (it == m_ring.end()) it = m_ring.begin();
            return it->second;
        }

        bool Empty() const { return m_ring.empty(); }
        size_t NodeCount() const { return m_nodes.size(); }
        bool HasNode(int nodeIndex) const { return m_nodes.count(nodeIndex) > 0; }
        const std::set<int>& GetNodes() const { return m_nodes; }
        int GetVNodeCount() const { return m_vnodeCount; }

    private:
        static uint32_t HashKey(SizeType headID) {
            uint32_t hash = 2166136261u; // FNV-1a offset basis
            uint32_t val = static_cast<uint32_t>(headID);
            for (int i = 0; i < 4; i++) {
                hash ^= (val >> (i * 8)) & 0xFF;
                hash *= 16777619u; // FNV prime
            }
            return hash;
        }

        static uint32_t HashVNode(int nodeIndex, int vnodeIdx) {
            // Raw FNV-1a on tiny nodeIndex (1, 2, 3) produces a
            // pathologically biased ring (71.9% vs 28.1% for nodes 1/2 with
            // 150 vnodes). Pre-mix nodeIndex through Knuth's golden-ratio
            // multiplier so small node IDs become full-spectrum uint32 values
            // before they hit FNV's accumulator. Validated to give ≈50/50
            // for K=2 and stay within ±15% of even split for K up to 8.
            uint32_t saltedVnode =
                static_cast<uint32_t>(vnodeIdx) ^
                (static_cast<uint32_t>(nodeIndex) * 2654435761u);
            uint32_t hash = 2166136261u;
            auto mix = [&](uint32_t v) {
                for (int i = 0; i < 4; i++) {
                    hash ^= (v >> (i * 8)) & 0xFF;
                    hash *= 16777619u;
                }
            };
            mix(saltedVnode);
            mix(static_cast<uint32_t>(nodeIndex));
            return hash;
        }

        int m_vnodeCount;
        std::map<uint32_t, int> m_ring;  // hash position → nodeIndex
        std::set<int> m_nodes;           // active physical node indices
    };

} // namespace SPTAG::SPANN
