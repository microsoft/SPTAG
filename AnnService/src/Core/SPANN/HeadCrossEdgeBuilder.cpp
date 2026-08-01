// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/SPANN/HeadCrossEdgeBuilder.h"

#include "inc/Core/SearchQuery.h"
#include "inc/Helper/HeadCrossEdges.h"
#include "inc/Helper/Logging.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace SPTAG
{
namespace SPANN
{
namespace
{
    struct CrossEdgeWorkItem
    {
        size_t nodeSlot;
        SizeType localHid;
    };

    struct CrossEdgeRecord
    {
        SizeType globalVID = -1;
        std::vector<Helper::HeadCrossEdgeEntry> edges;
    };

    bool PathExists(const std::string& p_path)
    {
        FILE* file = std::fopen(p_path.c_str(), "rb");
        if (file == nullptr) return false;
        std::fclose(file);
        return true;
    }

    bool ResolveGlobalVID(
        const HeadCrossEdgeBuildNode& p_node,
        SizeType p_localHid,
        SizeType& p_globalVID)
    {
        if (p_node.localHidToHeadIDs == nullptr || p_localHid < 0 ||
            p_localHid >= static_cast<SizeType>(p_node.localHidToHeadIDs->size())) {
            return false;
        }
        const SizeType headID =
            (*p_node.localHidToHeadIDs)[static_cast<size_t>(p_localHid)];
        if (p_node.denseHeadIDs == nullptr) {
            p_globalVID = headID;
            return p_globalVID >= 0;
        }
        if (headID < 0 || headID >= p_node.denseHeadIDs->R()) {
            return false;
        }
        p_globalVID = static_cast<SizeType>(*( (*p_node.denseHeadIDs)[headID] ));
        return p_globalVID != MaxSize && p_globalVID >= 0;
    }
}

bool BuildHeadCrossEdges(
    const std::vector<HeadCrossEdgeBuildNode>& p_nodes,
    const std::string& p_outputPath,
    const std::string& p_dirtyPath,
    const HeadCrossEdgeBuildOptions& p_options)
{
    if (p_nodes.empty()) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot build cross edges without bundle nodes.\n");
        return false;
    }

    const int searchTopK = (std::max)(1, p_options.searchTopK);
    const int extraEdges = (std::max)(1, p_options.extraEdges);
    const int threadCount = (std::max)(1, p_options.threads);
    if (!p_options.overwrite && PathExists(p_outputPath)) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "%s already exists. Refusing to overwrite the cross-edge sidecar.\n",
                     p_outputPath.c_str());
        return false;
    }

    VectorValueType valueType = VectorValueType::Undefined;
    DimensionType dimension = -1;
    size_t totalHeads = 0;
    bool hasUExtra = false;
    for (const auto& node : p_nodes) {
        if (node.nodeId < 0 || node.index == nullptr || node.localHidToHeadIDs == nullptr ||
            node.h1HeadCount < 0 ||
            node.index->GetNumSamples() !=
                static_cast<SizeType>(node.localHidToHeadIDs->size()) ||
            node.h1HeadCount > node.index->GetNumSamples()) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Invalid bundle node %d while building cross edges.\n", node.nodeId);
            return false;
        }

        if (valueType == VectorValueType::Undefined) {
            valueType = node.index->GetVectorValueType();
            dimension = node.index->GetFeatureDim();
        } else if (valueType != node.index->GetVectorValueType() ||
                   dimension != node.index->GetFeatureDim()) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Cross-edge bundle node %d has incompatible vector type or dimension.\n",
                         node.nodeId);
            return false;
        }

        totalHeads += static_cast<size_t>(node.index->GetNumSamples());
        hasUExtra = hasUExtra ||
            node.h1HeadCount < node.index->GetNumSamples();
    }
    if (totalHeads > static_cast<size_t>((std::numeric_limits<std::int32_t>::max)())) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "Cross-edge sidecar cannot represent %zu heads.\n", totalHeads);
        return false;
    }

    if (p_nodes.size() == 1) {
        const std::string temporaryPath = p_outputPath + ".tmp";
        FILE* file = std::fopen(temporaryPath.c_str(), "wb");
        if (file == nullptr) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Cannot open %s for write.\n", temporaryPath.c_str());
            return false;
        }
        Helper::HeadCrossEdgesHeader header{};
        header.magic = Helper::kHeadCrossEdgesMagic;
        header.version = Helper::kHeadCrossEdgesVersion;
        header.totalHeads = 0;
        header.maxEdgesPerHead = extraEdges;
        header.searchTopK = searchTopK;
        const bool wrote = std::fwrite(&header, sizeof(header), 1, file) == 1;
        const bool closed = std::fclose(file) == 0;
        if (!wrote || !closed || std::rename(temporaryPath.c_str(), p_outputPath.c_str()) != 0) {
            std::remove(temporaryPath.c_str());
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Failed to atomically write empty cross-edge sidecar %s.\n",
                         p_outputPath.c_str());
            return false;
        }
        if (std::remove(p_dirtyPath.c_str()) != 0 && errno != ENOENT) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                         "Wrote cross edges but could not clear dirty marker %s.\n",
                         p_dirtyPath.c_str());
        }
        return true;
    }

    std::vector<CrossEdgeWorkItem> work;
    work.reserve(totalHeads);
    for (size_t nodeSlot = 0; nodeSlot < p_nodes.size(); ++nodeSlot) {
        const SizeType count = p_nodes[nodeSlot].index->GetNumSamples();
        for (SizeType localHid = 0; localHid < count; ++localHid) {
            work.push_back({nodeSlot, localHid});
        }
    }

    std::vector<CrossEdgeRecord> records(work.size());
    std::atomic<size_t> nextWork{0};
    std::atomic<size_t> completed{0};
    std::atomic<size_t> nonEmpty{0};
    std::atomic<size_t> fullyFilled{0};
    std::atomic<bool> failed{false};

    auto worker = [&]() {
        std::vector<BasicResult> buffer(static_cast<size_t>(searchTopK));
        std::vector<Helper::HeadCrossEdgeEntry> merged;
        merged.reserve(static_cast<size_t>(searchTopK) * (p_nodes.size() - 1));

        while (!failed.load(std::memory_order_acquire)) {
            const size_t workIndex = nextWork.fetch_add(1);
            if (workIndex >= work.size()) return;

            const CrossEdgeWorkItem item = work[workIndex];
            const auto& source = p_nodes[item.nodeSlot];
            const void* query = source.index->GetSample(item.localHid);
            if (query == nullptr) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                             "Cross-edge source sample is missing for node %d, local HID %d.\n",
                             source.nodeId, static_cast<int>(item.localHid));
                failed.store(true, std::memory_order_release);
                return;
            }

            merged.clear();
            for (size_t targetSlot = 0; targetSlot < p_nodes.size(); ++targetSlot) {
                if (targetSlot == item.nodeSlot) continue;
                const auto& target = p_nodes[targetSlot];
                const SizeType targetCount = target.index->GetNumSamples();
                if (targetCount <= 0) continue;

                std::fill(buffer.begin(), buffer.end(), BasicResult());
                const int resultCount =
                    (std::min)(searchTopK, static_cast<int>(targetCount));
                QueryResult result(query, resultCount, false, buffer.data());
                if (target.index->SearchIndex(result) != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "Cross-edge search failed from bundle node %d to node %d.\n",
                                 source.nodeId, target.nodeId);
                    failed.store(true, std::memory_order_release);
                    return;
                }

                for (int rank = 0; rank < resultCount; ++rank) {
                    const BasicResult& candidate = buffer[static_cast<size_t>(rank)];
                    SizeType globalVID = MaxSize;
                    if (candidate.VID < 0 || candidate.Dist >= MaxDist ||
                        !ResolveGlobalVID(target, candidate.VID, globalVID)) {
                        continue;
                    }
                    if (globalVID < 0 ||
                        globalVID > static_cast<SizeType>(
                            (std::numeric_limits<std::int32_t>::max)())) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                     "Cross-edge global VID %d cannot be encoded.\n",
                                     static_cast<int>(globalVID));
                        failed.store(true, std::memory_order_release);
                        return;
                    }
                    merged.push_back(
                        {static_cast<std::int32_t>(globalVID), candidate.Dist});
                }
            }

            std::sort(
                merged.begin(), merged.end(),
                [](const Helper::HeadCrossEdgeEntry& p_left,
                   const Helper::HeadCrossEdgeEntry& p_right) {
                    return p_left.dist < p_right.dist;
                });
            const int keep = (std::min)(extraEdges, static_cast<int>(merged.size()));
            CrossEdgeRecord& record = records[workIndex];
            if (!ResolveGlobalVID(source, item.localHid, record.globalVID)) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                             "Cannot resolve the global VID for bundle node %d, local HID %d.\n",
                             source.nodeId, static_cast<int>(item.localHid));
                failed.store(true, std::memory_order_release);
                return;
            }
            record.edges.assign(merged.begin(), merged.begin() + keep);
            if (keep > 0) nonEmpty.fetch_add(1);
            if (keep >= extraEdges) fullyFilled.fetch_add(1);

            const size_t done = completed.fetch_add(1) + 1;
            if (done % 5000 == 0) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                             "Cross-edge progress: %zu/%zu heads processed.\n",
                             done, work.size());
            }
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(threadCount));
    for (int workerId = 0; workerId < threadCount; ++workerId) {
        workers.emplace_back(worker);
    }
    for (auto& workerThread : workers) workerThread.join();
    if (failed.load(std::memory_order_acquire)) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "Cross-edge generation failed; no sidecar was committed.\n");
        return false;
    }

    if (hasUExtra) {
        std::unordered_map<SizeType, size_t> recordByVID;
        recordByVID.reserve(records.size() * 2);
        std::unordered_set<SizeType> uExtraVIDs;
        for (size_t workIndex = 0; workIndex < work.size(); ++workIndex) {
            const CrossEdgeRecord& record = records[workIndex];
            if (record.globalVID >= 0) recordByVID[record.globalVID] = workIndex;
            const CrossEdgeWorkItem& item = work[workIndex];
            if (item.localHid >= p_nodes[item.nodeSlot].h1HeadCount &&
                record.globalVID >= 0) {
                uExtraVIDs.insert(record.globalVID);
            }
        }

        size_t reverseAdded = 0;
        for (size_t workIndex = 0; workIndex < work.size(); ++workIndex) {
            const CrossEdgeWorkItem& item = work[workIndex];
            if (item.localHid < p_nodes[item.nodeSlot].h1HeadCount) continue;

            const SizeType uExtraVID = records[workIndex].globalVID;
            if (uExtraVID < 0) continue;
            for (const auto& edge : records[workIndex].edges) {
                const SizeType neighborVID = static_cast<SizeType>(edge.neighborGlobalVID);
                if (uExtraVIDs.count(neighborVID) != 0) continue;
                const auto recordIt = recordByVID.find(neighborVID);
                if (recordIt == recordByVID.end()) continue;

                CrossEdgeRecord& h1Record = records[recordIt->second];
                const bool exists = std::any_of(
                    h1Record.edges.begin(), h1Record.edges.end(),
                    [uExtraVID](const Helper::HeadCrossEdgeEntry& p_existing) {
                        return p_existing.neighborGlobalVID ==
                            static_cast<std::int32_t>(uExtraVID);
                    });
                if (!exists) {
                    h1Record.edges.push_back(
                        {static_cast<std::int32_t>(uExtraVID), edge.dist});
                    ++reverseAdded;
                }
            }
        }
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                     "Cross-edge generation added %zu reverse H1-to-U_extra edges.\n",
                     reverseAdded);
    }

    int actualMaxEdges = extraEdges;
    for (const auto& record : records) {
        actualMaxEdges =
            (std::max)(actualMaxEdges, static_cast<int>(record.edges.size()));
    }

    const std::string temporaryPath = p_outputPath + ".tmp";
    FILE* output = std::fopen(temporaryPath.c_str(), "wb");
    if (output == nullptr) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "Cannot open %s for write.\n", temporaryPath.c_str());
        return false;
    }

    Helper::HeadCrossEdgesHeader header{};
    header.magic = Helper::kHeadCrossEdgesMagic;
    header.version = Helper::kHeadCrossEdgesVersion;
    header.totalHeads = static_cast<std::int32_t>(records.size());
    header.maxEdgesPerHead = actualMaxEdges;
    header.searchTopK = searchTopK;
    bool wrote = std::fwrite(&header, sizeof(header), 1, output) == 1;
    for (const auto& record : records) {
        const std::int32_t globalVID = static_cast<std::int32_t>(record.globalVID);
        const std::int32_t edgeCount = static_cast<std::int32_t>(record.edges.size());
        wrote = wrote &&
            std::fwrite(&globalVID, sizeof(globalVID), 1, output) == 1 &&
            std::fwrite(&edgeCount, sizeof(edgeCount), 1, output) == 1;
        if (wrote && edgeCount > 0) {
            wrote = std::fwrite(
                record.edges.data(), sizeof(Helper::HeadCrossEdgeEntry),
                static_cast<size_t>(edgeCount), output) ==
                static_cast<size_t>(edgeCount);
        }
        if (!wrote) break;
    }

    const bool closed = std::fclose(output) == 0;
    if (!wrote || !closed || std::rename(temporaryPath.c_str(), p_outputPath.c_str()) != 0) {
        std::remove(temporaryPath.c_str());
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "Failed to atomically write cross-edge sidecar %s.\n",
                     p_outputPath.c_str());
        return false;
    }
    if (std::remove(p_dirtyPath.c_str()) != 0 && errno != ENOENT) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                     "Wrote cross edges but could not clear dirty marker %s.\n",
                     p_dirtyPath.c_str());
    }

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                 "Cross-edge generation complete: %zu heads, %zu with edges, %zu fully filled; wrote %s.\n",
                 records.size(), nonEmpty.load(), fullyFilled.load(), p_outputPath.c_str());
    return true;
}
}
}
