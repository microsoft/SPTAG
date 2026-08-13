// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// On-disk format for cross-subgraph edge augmentation produced by the
// augmentheadgraph tool. The file lives beside the head index and is loaded
// into each graph row's runtime suffix for cross-bundle or hybrid traversal.

#ifndef _SPTAG_HELPER_HEADCROSSEDGES_H_
#define _SPTAG_HELPER_HEADCROSSEDGES_H_

#include <cstdint>
#include <cstring>

namespace SPTAG
{
namespace Helper
{

constexpr std::uint32_t kHeadCrossEdgesMagic = 0x48434548U; // 'HCEH'
constexpr std::int32_t  kHeadCrossEdgesVersion = 1;
constexpr std::int32_t  kHybridHeadCrossEdgesVersion = 2;
constexpr std::int32_t  kHybridHeadCrossEdgesMarker = 0x48594252; // 'HYBR'
constexpr const char*   kHeadCrossEdgesFileName = "head_cross_edges.bin";
constexpr const char*   kHeadCrossEdgesDirtyFileName = "head_cross_edges.dirty";

#pragma pack(push, 1)
struct HeadCrossEdgesHeader
{
    std::uint32_t magic;            // kHeadCrossEdgesMagic
    std::int32_t  version;          // kHeadCrossEdgesVersion
    std::int32_t  totalHeads;       // number of records that follow
    std::int32_t  maxEdgesPerHead;  // M
    std::int32_t  searchTopK;       // per-subgraph BKT search top
    std::int32_t  reserved;         // kHybridHeadCrossEdgesMarker for hybrid edges
};

struct HybridHeadCrossEdgesExtension
{
    std::uint64_t generationFingerprint;
    std::uint64_t contentFingerprint;
};

struct HeadCrossEdgeEntry
{
    std::int32_t neighborGlobalVID;
    float        dist;
};

// Per-record layout in the file:
//   HybridHeadCrossEdgesExtension extension  // hybrid version 2 only
//   int32 globalHeadVID
//   int32 edgeCount  // <= maxEdgesPerHead
//   HeadCrossEdgeEntry edges[edgeCount]
#pragma pack(pop)

class HeadCrossEdgesBodyFingerprint
{
public:
    void AddRecord(
        std::int32_t p_sourceGlobalVID,
        std::int32_t p_edgeCount)
    {
        AddUint32(0x524f5731U);
        AddUint32(
            static_cast<std::uint32_t>(
                p_sourceGlobalVID));
        AddUint32(
            static_cast<std::uint32_t>(
                p_edgeCount));
    }

    void AddEntry(const HeadCrossEdgeEntry& p_entry)
    {
        std::uint32_t distanceBits = 0;
        std::memcpy(
            &distanceBits, &p_entry.dist,
            sizeof(distanceBits));
        AddUint32(0x45444731U);
        AddUint32(
            static_cast<std::uint32_t>(
                p_entry.neighborGlobalVID));
        AddUint32(distanceBits);
    }

    std::uint64_t Value() const
    {
        return m_value == 0
            ? 0xcbf29ce484222325ULL
            : m_value;
    }

private:
    void AddUint32(std::uint32_t p_value)
    {
        for (int shift = 0; shift < 32; shift += 8) {
            m_value ^=
                static_cast<std::uint8_t>(
                    p_value >> shift);
            m_value *= 0x100000001b3ULL;
        }
    }

    std::uint64_t m_value =
        0xcbf29ce484222325ULL;
};

} // namespace Helper
} // namespace SPTAG

#endif // _SPTAG_HELPER_HEADCROSSEDGES_H_
