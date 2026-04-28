// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// On-disk format for cross-subgraph edge augmentation produced by the
// augmentheadgraph tool. The file lives next to head_bundle_manifest.bin
// and is consumed (in a future search-path integration step) by SPANN
// to traverse cross-subgraph shortcut edges in the head graph.

#ifndef _SPTAG_HELPER_HEADCROSSEDGES_H_
#define _SPTAG_HELPER_HEADCROSSEDGES_H_

#include <cstdint>

namespace SPTAG
{
namespace Helper
{

constexpr std::uint32_t kHeadCrossEdgesMagic = 0x48434548U; // 'HCEH'
constexpr std::int32_t  kHeadCrossEdgesVersion = 1;
constexpr const char*   kHeadCrossEdgesFileName = "head_cross_edges.bin";

#pragma pack(push, 1)
struct HeadCrossEdgesHeader
{
    std::uint32_t magic;            // kHeadCrossEdgesMagic
    std::int32_t  version;          // kHeadCrossEdgesVersion
    std::int32_t  totalHeads;       // number of records that follow
    std::int32_t  maxEdgesPerHead;  // M
    std::int32_t  searchTopK;       // per-subgraph BKT search top
    std::int32_t  reserved;
};

struct HeadCrossEdgeEntry
{
    std::int32_t neighborGlobalVID;
    float        dist;
};

// Per-record layout in the file:
//   int32 globalHeadVID
//   int32 edgeCount  // <= maxEdgesPerHead
//   HeadCrossEdgeEntry edges[edgeCount]
#pragma pack(pop)

} // namespace Helper
} // namespace SPTAG

#endif // _SPTAG_HELPER_HEADCROSSEDGES_H_
