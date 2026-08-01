// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_HEADCROSSEDGEBUILDER_H_
#define _SPTAG_SPANN_HEADCROSSEDGEBUILDER_H_

#include "inc/Core/Common.h"
#include "inc/Core/Common/Dataset.h"
#include "inc/Core/VectorIndex.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace SPTAG
{
namespace SPANN
{
    // A loaded bundle graph plus its local-HID map. The map may resolve directly
    // to global vector IDs (stand-alone tool) or to dense head IDs; the latter is
    // translated through denseHeadIDs before serializing the sidecar.
    struct HeadCrossEdgeBuildNode
    {
        int nodeId = -1;
        SizeType h1HeadCount = 0;
        std::shared_ptr<VectorIndex> index;
        const std::vector<SizeType>* localHidToHeadIDs = nullptr;
        const COMMON::Dataset<std::uint64_t>* denseHeadIDs = nullptr;
    };

    struct HeadCrossEdgeBuildOptions
    {
        int searchTopK = 15;
        int extraEdges = 10;
        int threads = 1;
        bool overwrite = false;
    };

    // Builds a cross-subgraph sidecar atomically. The supplied bundle indexes must
    // remain loaded and immutable for the entire call.
    bool BuildHeadCrossEdges(
        const std::vector<HeadCrossEdgeBuildNode>& p_nodes,
        const std::string& p_outputPath,
        const std::string& p_dirtyPath,
        const HeadCrossEdgeBuildOptions& p_options);
}
}

#endif // _SPTAG_SPANN_HEADCROSSEDGEBUILDER_H_
