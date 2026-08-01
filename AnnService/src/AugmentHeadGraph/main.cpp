// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// augmentheadgraph
//
// Stand-alone post-processing entrypoint for the same cross-edge builder used
// by STATIC BuildSSD. It remains useful for existing indexes and explicit
// regeneration, while new bundle STATIC builds generate the sidecar before
// tail construction.

#include "inc/Core/Common.h"
#include "inc/Core/Common/Dataset.h"
#include "inc/Core/SPANN/HeadCrossEdgeBuilder.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Helper/ArgumentsParser.h"
#include "inc/Helper/HeadCrossEdges.h"
#include "inc/Helper/Logging.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace SPTAG;

namespace
{
constexpr std::uint32_t kHeadBundleManifestMagic = 0x48424D46U;
constexpr std::int32_t kHeadBundleManifestVersion = 2;

#pragma pack(push, 1)
struct HeadBundleManifestHeader
{
    std::uint32_t magic;
    std::int32_t version;
    std::int32_t nodeCount;
    std::int32_t reserved;
};

struct HeadBundleManifestNodeRecordV2
{
    std::int32_t nodeId;
    std::int32_t pathLength;
    std::int64_t headOffset;
    std::int64_t headCount;
    std::int64_t postingOffset;
    std::int64_t postingCount;
    std::int64_t assignmentCount;
};
#pragma pack(pop)

struct BundleNode
{
    int nodeId = 0;
    std::string relativePath;
    std::string absolutePath;
    std::int64_t headCount = 0;
    std::shared_ptr<VectorIndex> index;
    std::vector<SizeType> localHidToGlobalVID;
};

std::string JoinPath(const std::string& p_base, const std::string& p_relative)
{
    if (p_relative.empty()) return p_base;
    std::string result = p_base;
    if (!result.empty() && result.back() != FolderSep) result += FolderSep;
    result += p_relative;
    return result;
}

bool LoadManifest(const std::string& p_headIndexDir, std::vector<BundleNode>& p_nodes)
{
    const std::string manifestPath = JoinPath(p_headIndexDir, "head_bundle_manifest.bin");
    FILE* file = std::fopen(manifestPath.c_str(), "rb");
    if (file == nullptr) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot open manifest: %s\n", manifestPath.c_str());
        return false;
    }

    HeadBundleManifestHeader header{};
    if (std::fread(&header, sizeof(header), 1, file) != 1 ||
        header.magic != kHeadBundleManifestMagic ||
        header.version != kHeadBundleManifestVersion) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Invalid bundle manifest: %s\n", manifestPath.c_str());
        std::fclose(file);
        return false;
    }

    std::string baseDir = p_headIndexDir;
    if (!baseDir.empty() && baseDir.back() == FolderSep) baseDir.pop_back();
    const size_t separator = baseDir.find_last_of(FolderSep);
    baseDir = separator == std::string::npos ? "." : baseDir.substr(0, separator);

    p_nodes.clear();
    p_nodes.reserve(static_cast<size_t>(header.nodeCount));
    for (std::int32_t nodeIndex = 0; nodeIndex < header.nodeCount; ++nodeIndex) {
        HeadBundleManifestNodeRecordV2 record{};
        if (std::fread(&record, sizeof(record), 1, file) != 1 || record.pathLength < 0 ||
            record.headCount < 0) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Invalid manifest node %d.\n", nodeIndex);
            std::fclose(file);
            return false;
        }
        std::string path(static_cast<size_t>(record.pathLength), '\0');
        if (record.pathLength > 0 &&
            std::fread(&path[0], 1, static_cast<size_t>(record.pathLength), file) !=
                static_cast<size_t>(record.pathLength)) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read manifest path %d.\n", nodeIndex);
            std::fclose(file);
            return false;
        }
        BundleNode node;
        node.nodeId = record.nodeId;
        node.relativePath = std::move(path);
        node.absolutePath = JoinPath(baseDir, node.relativePath);
        node.headCount = record.headCount;
        p_nodes.emplace_back(std::move(node));
    }

    std::fclose(file);
    return true;
}

bool LoadAllSubgraphs(
    std::vector<BundleNode>& p_nodes,
    const std::string& p_headIDFile,
    int p_searchTopK)
{
    constexpr int kBlockSize = 1024 * 1024;
    constexpr int kCapacity = 1024 * 1024;
    for (auto& node : p_nodes) {
        if (node.headCount == 0) continue;

        std::shared_ptr<VectorIndex> index;
        if (VectorIndex::LoadIndex(node.absolutePath, index) != ErrorCode::Success || index == nullptr) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Failed to load subgraph node %d from %s\n",
                         node.nodeId, node.absolutePath.c_str());
            return false;
        }
        index->SetParameter("MaxCheck", std::to_string((std::max)(8192, p_searchTopK * 64)));
        if (index->UpdateIndex() != ErrorCode::Success) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Failed to initialize subgraph node %d.\n", node.nodeId);
            return false;
        }
        index->SetReady(true);

        COMMON::Dataset<std::uint64_t> headIDs;
        headIDs.SetName("HeadBundleNodeIDs");
        const std::string idsPath = JoinPath(node.absolutePath, p_headIDFile);
        if (headIDs.Load(idsPath, kBlockSize, kCapacity) != ErrorCode::Success ||
            headIDs.R() != index->GetNumSamples()) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Invalid head ID map for subgraph node %d: %s\n",
                         node.nodeId, idsPath.c_str());
            return false;
        }
        node.localHidToGlobalVID.resize(static_cast<size_t>(headIDs.R()));
        for (SizeType localHid = 0; localHid < headIDs.R(); ++localHid) {
            node.localHidToGlobalVID[static_cast<size_t>(localHid)] =
                static_cast<SizeType>(*(headIDs[localHid]));
        }
        node.index = std::move(index);
    }
    return true;
}

class AugmentOptions : public Helper::ArgumentsParser
{
public:
    AugmentOptions()
    {
        AddRequiredOption(m_headIndexDir, "-d", "--head_index_dir",
                          "Directory containing head_bundle_manifest.bin.");
        AddOptionalOption(m_searchTopK, "-k", "--search_topk",
                          "Per-subgraph BKT search top (default 15).");
        AddOptionalOption(m_extraEdges, "-m", "--extra_edges",
                          "Cross-subgraph edges to keep per head (default 10).");
        AddOptionalOption(m_threads, "-t", "--threads",
                          "Worker threads (default hardware_concurrency).");
        AddOptionalOption(m_overwrite, "-w", "--overwrite",
                          "Overwrite existing head_cross_edges.bin (default false).");
        AddOptionalOption(m_headIDFile, "-i", "--head_id_file",
                          "HeadVectorIDs file name inside each subgraph.");
    }

    std::string m_headIndexDir;
    int m_searchTopK = 15;
    int m_extraEdges = 10;
    int m_threads = static_cast<int>(std::thread::hardware_concurrency());
    bool m_overwrite = false;
    std::string m_headIDFile = "SPTAGHeadVectorIDs.bin";
};
}

int main(int argc, char** argv)
{
    auto options = std::make_shared<AugmentOptions>();
    if (!options->Parse(argc - 1, argv + 1)) return 1;

    const int threadCount = (std::max)(1, options->m_threads);
    const int searchTopK = (std::max)(1, options->m_searchTopK);
    const int extraEdges = (std::max)(1, options->m_extraEdges);
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                 "augmentheadgraph: dir=%s topk=%d M=%d threads=%d overwrite=%d\n",
                 options->m_headIndexDir.c_str(), searchTopK, extraEdges, threadCount,
                 static_cast<int>(options->m_overwrite));

    std::vector<BundleNode> loadedNodes;
    if (!LoadManifest(options->m_headIndexDir, loadedNodes) ||
        !LoadAllSubgraphs(loadedNodes, options->m_headIDFile, searchTopK)) {
        return 2;
    }

    std::vector<SPANN::HeadCrossEdgeBuildNode> nodes;
    nodes.reserve(loadedNodes.size());
    for (const auto& loaded : loadedNodes) {
        nodes.push_back(
            {loaded.nodeId, static_cast<SizeType>(loaded.headCount), loaded.index,
             &loaded.localHidToGlobalVID, nullptr});
    }

    const std::string outputPath =
        JoinPath(options->m_headIndexDir, Helper::kHeadCrossEdgesFileName);
    const std::string dirtyPath =
        JoinPath(options->m_headIndexDir, Helper::kHeadCrossEdgesDirtyFileName);
    const SPANN::HeadCrossEdgeBuildOptions buildOptions{
        searchTopK, extraEdges, threadCount, options->m_overwrite};
    return SPANN::BuildHeadCrossEdges(nodes, outputPath, dirtyPath, buildOptions) ? 0 : 3;
}
