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
#include <filesystem>
#include <limits>
#include <queue>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace SPTAG
{
namespace SPANN
{
namespace
{
    // Both limits are independent of the number of heads. The reverse-sort
    // buffer is deliberately small enough to leave room for BKT search state.
    const size_t kWorkChunkHeads = 4096;
    const size_t kReverseSortRecords = 1024 * 1024;
    const size_t kReverseShardCount = 64;
    const size_t kMergeFanIn = 32;

    struct CrossEdgeCandidate
    {
        Helper::HeadCrossEdgeEntry edge;
        std::int32_t reverseTargetOrdinal = -1;
    };

    struct CrossEdgeRecord
    {
        std::int32_t globalVID = -1;
        std::vector<CrossEdgeCandidate> edges;
    };

    struct ReverseCandidate
    {
        std::int32_t targetOrdinal;
        std::int32_t extraVID;
        float dist;
    };

    struct BuildStats
    {
        std::atomic<size_t> completed{0};
        std::atomic<size_t> nonEmpty{0};
        std::atomic<size_t> fullyFilled{0};
    };

    class TemporaryArtifacts
    {
    public:
        void Add(const std::string& p_path)
        {
            m_paths.push_back(p_path);
        }

        ~TemporaryArtifacts()
        {
            for (const auto& path : m_paths) std::remove(path.c_str());
        }

    private:
        std::vector<std::string> m_paths;
    };

    bool PathExists(const std::string& p_path)
    {
        FILE* file = std::fopen(p_path.c_str(), "rb");
        if (file == nullptr) return false;
        std::fclose(file);
        return true;
    }

    bool RemoveStaleCrossBuildArtifacts(const std::string& p_outputPath)
    {
        namespace fs = std::filesystem;
        const fs::path outputPath(p_outputPath);
        const fs::path parent =
            outputPath.has_parent_path() ? outputPath.parent_path() : fs::path(".");
        const std::string prefix = outputPath.filename().string() + ".crossbuild";
        std::error_code error;
        fs::directory_iterator entries(parent, error);
        if (error) {
            SPTAGLIB_LOG(
                Helper::LogLevel::LL_Error,
                "Cannot inspect cross-edge temporary directory %s: %s\n",
                parent.string().c_str(),
                error.message().c_str());
            return false;
        }
        for (const auto& entry : entries) {
            if (entry.path().filename().string().compare(0, prefix.size(), prefix) != 0) {
                continue;
            }
            fs::remove_all(entry.path(), error);
            if (error) {
                SPTAGLIB_LOG(
                    Helper::LogLevel::LL_Error,
                    "Cannot remove stale cross-edge temporary artifact %s: %s\n",
                    entry.path().string().c_str(),
                    error.message().c_str());
                return false;
            }
        }
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

    bool EntryLess(const Helper::HeadCrossEdgeEntry& p_left,
                   const Helper::HeadCrossEdgeEntry& p_right)
    {
        return p_left.dist < p_right.dist ||
            (p_left.dist == p_right.dist &&
             p_left.neighborGlobalVID < p_right.neighborGlobalVID);
    }

    void KeepNearest(std::vector<CrossEdgeCandidate>& p_edges,
                     const CrossEdgeCandidate& p_candidate,
                     size_t p_limit)
    {
        const auto it = std::upper_bound(
            p_edges.begin(), p_edges.end(), p_candidate,
            [](const CrossEdgeCandidate& p_left,
               const CrossEdgeCandidate& p_right) {
                return EntryLess(p_left.edge, p_right.edge);
            });
        if (p_edges.size() == p_limit && it == p_edges.end()) return;
        p_edges.insert(it, p_candidate);
        if (p_edges.size() > p_limit) p_edges.pop_back();
    }

    size_t NodeForOrdinal(const std::vector<size_t>& p_prefixes, size_t p_ordinal)
    {
        return static_cast<size_t>(
            std::upper_bound(p_prefixes.begin(), p_prefixes.end(), p_ordinal) -
            p_prefixes.begin());
    }

    size_t NodeStart(const std::vector<size_t>& p_prefixes, size_t p_nodeSlot)
    {
        return p_nodeSlot == 0 ? 0 : p_prefixes[p_nodeSlot - 1];
    }

    bool BuildRecordRange(
        const std::vector<HeadCrossEdgeBuildNode>& p_nodes,
        const std::vector<size_t>& p_prefixes,
        size_t p_startOrdinal,
        size_t p_count,
        int p_searchTopK,
        size_t p_edgeLimit,
        int p_threadCount,
        size_t p_totalHeads,
        BuildStats& p_stats,
        std::vector<CrossEdgeRecord>& p_records)
    {
        p_records.assign(p_count, CrossEdgeRecord());
        std::atomic<size_t> next{0};
        std::atomic<bool> failed{false};

        const int workerCount = static_cast<int>((std::min)(
            static_cast<size_t>(p_threadCount), p_count));
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(workerCount));
        for (int workerId = 0; workerId < workerCount; ++workerId) {
            workers.emplace_back([&]() {
                std::vector<BasicResult> buffer(static_cast<size_t>(p_searchTopK));
                while (!failed.load(std::memory_order_acquire)) {
                    const size_t recordIndex = next.fetch_add(1);
                    if (recordIndex >= p_count) return;

                    const size_t ordinal = p_startOrdinal + recordIndex;
                    const size_t sourceSlot = NodeForOrdinal(p_prefixes, ordinal);
                    const auto& source = p_nodes[sourceSlot];
                    const SizeType localHid = static_cast<SizeType>(
                        ordinal - NodeStart(p_prefixes, sourceSlot));
                    CrossEdgeRecord& record = p_records[recordIndex];

                    SizeType sourceVID = MaxSize;
                    if (!ResolveGlobalVID(source, localHid, sourceVID) ||
                        sourceVID > static_cast<SizeType>(
                            (std::numeric_limits<std::int32_t>::max)())) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                     "Cannot resolve an encodable global VID for bundle node %d, local HID %d.\n",
                                     source.nodeId, static_cast<int>(localHid));
                        failed.store(true, std::memory_order_release);
                        return;
                    }
                    record.globalVID = static_cast<std::int32_t>(sourceVID);
                    const size_t candidateLimit = p_nodes.size() > 1
                        ? (p_nodes.size() - 1) * static_cast<size_t>(p_searchTopK)
                        : 0;
                    record.edges.reserve((std::min)(p_edgeLimit, candidateLimit));

                    bool hasTarget = false;
                    for (size_t targetSlot = 0; targetSlot < p_nodes.size(); ++targetSlot) {
                        if (targetSlot != sourceSlot &&
                            p_nodes[targetSlot].index != nullptr &&
                            p_nodes[targetSlot].index->GetNumSamples() > 0) {
                            hasTarget = true;
                            break;
                        }
                    }
                    if (hasTarget) {
                        const void* query = source.index->GetSample(localHid);
                        if (query == nullptr) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                         "Cross-edge source sample is missing for node %d, local HID %d.\n",
                                         source.nodeId, static_cast<int>(localHid));
                            failed.store(true, std::memory_order_release);
                            return;
                        }

                        for (size_t targetSlot = 0; targetSlot < p_nodes.size(); ++targetSlot) {
                            if (targetSlot == sourceSlot) continue;
                            const auto& target = p_nodes[targetSlot];
                            if (target.index == nullptr) continue;
                            const SizeType targetCount = target.index->GetNumSamples();
                            if (targetCount <= 0) continue;

                            std::fill(buffer.begin(), buffer.end(), BasicResult());
                            const int resultCount =
                                (std::min)(p_searchTopK, static_cast<int>(targetCount));
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
                                if (globalVID > static_cast<SizeType>(
                                                    (std::numeric_limits<std::int32_t>::max)())) {
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                                 "Cross-edge global VID %d cannot be encoded.\n",
                                                 static_cast<int>(globalVID));
                                    failed.store(true, std::memory_order_release);
                                    return;
                                }

                                CrossEdgeCandidate edge{};
                                edge.edge = {static_cast<std::int32_t>(globalVID), candidate.Dist};
                                if (candidate.VID < target.h1HeadCount) {
                                    const size_t targetOrdinal =
                                        NodeStart(p_prefixes, targetSlot) +
                                        static_cast<size_t>(candidate.VID);
                                    edge.reverseTargetOrdinal =
                                        static_cast<std::int32_t>(targetOrdinal);
                                }
                                KeepNearest(record.edges, edge, p_edgeLimit);
                            }
                        }
                    }

                    if (!record.edges.empty()) p_stats.nonEmpty.fetch_add(1);
                    if (record.edges.size() >= p_edgeLimit) p_stats.fullyFilled.fetch_add(1);
                    const size_t done = p_stats.completed.fetch_add(1) + 1;
                    if (done % 5000 == 0) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                     "Cross-edge progress: %zu/%zu heads processed.\n",
                                     done, p_totalHeads);
                    }
                }
            });
        }
        for (auto& worker : workers) worker.join();
        return !failed.load(std::memory_order_acquire);
    }

    bool WriteRecord(FILE* p_file, const CrossEdgeRecord& p_record, size_t p_limit)
    {
        if (p_record.edges.size() > p_limit ||
            p_record.edges.size() > static_cast<size_t>((std::numeric_limits<std::uint16_t>::max)())) {
            return false;
        }
        const std::int32_t edgeCount = static_cast<std::int32_t>(p_record.edges.size());
        if (std::fwrite(&p_record.globalVID, sizeof(p_record.globalVID), 1, p_file) != 1 ||
            std::fwrite(&edgeCount, sizeof(edgeCount), 1, p_file) != 1) {
            return false;
        }
        for (const auto& edge : p_record.edges) {
            if (std::fwrite(&edge.edge, sizeof(edge.edge), 1, p_file) != 1) return false;
        }
        return true;
    }

    bool ReadRecord(FILE* p_file, CrossEdgeRecord& p_record, size_t p_limit)
    {
        std::int32_t edgeCount = 0;
        if (std::fread(&p_record.globalVID, sizeof(p_record.globalVID), 1, p_file) != 1 ||
            std::fread(&edgeCount, sizeof(edgeCount), 1, p_file) != 1 ||
            edgeCount < 0 || static_cast<size_t>(edgeCount) > p_limit ||
            static_cast<size_t>(edgeCount) >
                static_cast<size_t>((std::numeric_limits<std::uint16_t>::max)())) {
            return false;
        }
        p_record.edges.resize(static_cast<size_t>(edgeCount));
        for (auto& edge : p_record.edges) {
            if (std::fread(&edge.edge, sizeof(edge.edge), 1, p_file) != 1) return false;
            edge.reverseTargetOrdinal = -1;
        }
        return true;
    }

    bool WriteReverseCandidate(FILE* p_file, const ReverseCandidate& p_candidate)
    {
        return std::fwrite(&p_candidate.targetOrdinal, sizeof(p_candidate.targetOrdinal), 1, p_file) == 1 &&
            std::fwrite(&p_candidate.extraVID, sizeof(p_candidate.extraVID), 1, p_file) == 1 &&
            std::fwrite(&p_candidate.dist, sizeof(p_candidate.dist), 1, p_file) == 1;
    }

    bool ReadReverseCandidate(FILE* p_file, ReverseCandidate& p_candidate, bool& p_eof)
    {
        p_eof = false;
        if (std::fread(&p_candidate.targetOrdinal, sizeof(p_candidate.targetOrdinal), 1, p_file) != 1) {
            p_eof = std::feof(p_file) != 0;
            return false;
        }
        if (std::fread(&p_candidate.extraVID, sizeof(p_candidate.extraVID), 1, p_file) != 1 ||
            std::fread(&p_candidate.dist, sizeof(p_candidate.dist), 1, p_file) != 1) {
            return false;
        }
        return true;
    }

    bool ReverseLess(const ReverseCandidate& p_left, const ReverseCandidate& p_right)
    {
        return p_left.targetOrdinal < p_right.targetOrdinal ||
            (p_left.targetOrdinal == p_right.targetOrdinal &&
             (p_left.dist < p_right.dist ||
              (p_left.dist == p_right.dist && p_left.extraVID < p_right.extraVID)));
    }

    bool MergeReverseRuns(const std::vector<std::string>& p_runs, const std::string& p_output)
    {
        struct HeapEntry
        {
            ReverseCandidate candidate;
            size_t run;
        };
        struct HeapCompare
        {
            bool operator()(const HeapEntry& p_left, const HeapEntry& p_right) const
            {
                return ReverseLess(p_right.candidate, p_left.candidate);
            }
        };

        std::vector<FILE*> inputs(p_runs.size(), nullptr);
        FILE* output = nullptr;
        bool ok = true;
        std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapCompare> heap;
        for (size_t i = 0; i < p_runs.size() && ok; ++i) {
            inputs[i] = std::fopen(p_runs[i].c_str(), "rb");
            if (inputs[i] == nullptr) {
                ok = false;
                break;
            }
            ReverseCandidate candidate{};
            bool eof = false;
            if (ReadReverseCandidate(inputs[i], candidate, eof)) {
                heap.push({candidate, i});
            } else if (!eof) {
                ok = false;
            }
        }
        if (ok) output = std::fopen(p_output.c_str(), "wb");
        if (output == nullptr) ok = false;

        while (ok && !heap.empty()) {
            const HeapEntry next = heap.top();
            heap.pop();
            if (!WriteReverseCandidate(output, next.candidate)) {
                ok = false;
                break;
            }
            ReverseCandidate candidate{};
            bool eof = false;
            if (ReadReverseCandidate(inputs[next.run], candidate, eof)) {
                heap.push({candidate, next.run});
            } else if (!eof) {
                ok = false;
            }
        }
        if (output != nullptr && std::fclose(output) != 0) ok = false;
        for (FILE* input : inputs) {
            if (input != nullptr && std::fclose(input) != 0) ok = false;
        }
        if (!ok) std::remove(p_output.c_str());
        return ok;
    }

    bool SortReverseShard(const std::string& p_input,
                          const std::string& p_sorted,
                          TemporaryArtifacts& p_artifacts)
    {
        FILE* input = std::fopen(p_input.c_str(), "rb");
        if (input == nullptr) return false;

        std::vector<std::string> runs;
        std::vector<ReverseCandidate> buffer;
        buffer.reserve(kReverseSortRecords);
        bool ok = true;
        size_t runNumber = 0;
        while (ok) {
            buffer.clear();
            bool eof = false;
            while (buffer.size() < kReverseSortRecords) {
                ReverseCandidate candidate{};
                if (!ReadReverseCandidate(input, candidate, eof)) {
                    if (!eof) ok = false;
                    break;
                }
                buffer.push_back(candidate);
            }
            if (!ok || buffer.empty()) break;
            std::sort(buffer.begin(), buffer.end(), ReverseLess);
            const std::string runPath = p_input + ".run." + std::to_string(runNumber++);
            p_artifacts.Add(runPath);
            FILE* run = std::fopen(runPath.c_str(), "wb");
            if (run == nullptr) {
                ok = false;
                break;
            }
            for (const auto& candidate : buffer) {
                if (!WriteReverseCandidate(run, candidate)) {
                    ok = false;
                    break;
                }
            }
            if (std::fclose(run) != 0) ok = false;
            if (ok) runs.push_back(runPath);
            if (eof) break;
        }
        if (std::fclose(input) != 0) ok = false;
        if (!ok) return false;

        if (runs.empty()) {
            FILE* sorted = std::fopen(p_sorted.c_str(), "wb");
            if (sorted == nullptr) return false;
            return std::fclose(sorted) == 0;
        }

        size_t pass = 0;
        while (runs.size() > 1 && ok) {
            std::vector<std::string> merged;
            for (size_t start = 0; start < runs.size(); start += kMergeFanIn) {
                const size_t end = (std::min)(runs.size(), start + kMergeFanIn);
                const std::string mergedPath = p_input + ".merge." +
                    std::to_string(pass) + "." + std::to_string(merged.size());
                p_artifacts.Add(mergedPath);
                std::vector<std::string> group(runs.begin() + start, runs.begin() + end);
                if (!MergeReverseRuns(group, mergedPath)) {
                    ok = false;
                    break;
                }
                for (const auto& path : group) std::remove(path.c_str());
                merged.push_back(mergedPath);
            }
            runs.swap(merged);
            ++pass;
        }
        if (!ok || std::rename(runs.front().c_str(), p_sorted.c_str()) != 0) return false;
        return true;
    }

    size_t ReverseShard(size_t p_ordinal, size_t p_totalHeads)
    {
        return static_cast<size_t>(
            (static_cast<std::uint64_t>(p_ordinal) * kReverseShardCount) / p_totalHeads);
    }

    size_t ShardStart(size_t p_shard, size_t p_totalHeads)
    {
        return (p_shard * p_totalHeads + kReverseShardCount - 1) / kReverseShardCount;
    }

    bool CommitOutput(const std::string& p_temporary,
                      const std::string& p_output,
                      bool p_overwrite)
    {
        if (p_overwrite) {
            return std::rename(p_temporary.c_str(), p_output.c_str()) == 0;
        }
#if defined(_WIN32)
        if (PathExists(p_output)) return false;
        return std::rename(p_temporary.c_str(), p_output.c_str()) == 0;
#else
        // link() supplies no-replace atomicity for the non-overwrite contract.
        if (::link(p_temporary.c_str(), p_output.c_str()) != 0) return false;
        std::remove(p_temporary.c_str());
        return true;
#endif
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
    const size_t requestedEdges = static_cast<size_t>((std::max)(1, p_options.extraEdges));
    const int threadCount = (std::max)(1, p_options.threads);
    if (!p_options.overwrite && PathExists(p_outputPath)) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "%s already exists. Refusing to overwrite the cross-edge sidecar.\n",
                     p_outputPath.c_str());
        return false;
    }
    if (!RemoveStaleCrossBuildArtifacts(p_outputPath)) {
        return false;
    }

    VectorValueType valueType = VectorValueType::Undefined;
    DimensionType dimension = -1;
    size_t totalHeads = 0;
    bool hasUExtra = false;
    std::vector<size_t> prefixes;
    prefixes.reserve(p_nodes.size());
    for (const auto& node : p_nodes) {
        if (node.nodeId < 0 || node.localHidToHeadIDs == nullptr ||
            node.h1HeadCount < 0) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Invalid bundle node %d while building cross edges.\n", node.nodeId);
            return false;
        }
        if (node.index == nullptr) {
            if (node.h1HeadCount != 0 || !node.localHidToHeadIDs->empty()) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                             "Non-empty bundle node %d has no loaded index.\n", node.nodeId);
                return false;
            }
            prefixes.push_back(totalHeads);
            continue;
        }
        if (node.index->GetNumSamples() !=
                static_cast<SizeType>(node.localHidToHeadIDs->size()) ||
            node.h1HeadCount > node.index->GetNumSamples()) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Invalid bundle node %d while building cross edges.\n", node.nodeId);
            return false;
        }
        const SizeType count = node.index->GetNumSamples();
        if (count > 0) {
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
            totalHeads += static_cast<size_t>(count);
        }
        prefixes.push_back(totalHeads);
        hasUExtra = hasUExtra || node.h1HeadCount < count;
    }
    if (totalHeads > static_cast<size_t>((std::numeric_limits<std::int32_t>::max)())) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "Cross-edge sidecar cannot represent %zu heads.\n", totalHeads);
        return false;
    }

    const size_t maxOutgoingEdges = hasUExtra
        ? static_cast<size_t>((std::numeric_limits<std::uint16_t>::max)() / 2)
        : static_cast<size_t>((std::numeric_limits<std::uint16_t>::max)());
    const size_t edgeLimit = (std::min)(requestedEdges, maxOutgoingEdges);
    if (edgeLimit != requestedEdges) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                     "Cross-edge count %zu was capped at %zu so each record fits uint16.\n",
                     requestedEdges, edgeLimit);
    }
    const size_t maxEdgesPerHead = hasUExtra ? edgeLimit * 2 : edgeLimit;
    const std::string temporaryPath = p_outputPath + ".tmp";
    const std::string basePath = p_outputPath + ".crossbuild.base";
    TemporaryArtifacts artifacts;
    artifacts.Add(temporaryPath);
    artifacts.Add(basePath);
    std::remove(temporaryPath.c_str());
    std::remove(basePath.c_str());

    std::vector<std::string> reversePaths;
    std::vector<std::string> sortedPaths;
    if (hasUExtra) {
        reversePaths.reserve(kReverseShardCount);
        sortedPaths.reserve(kReverseShardCount);
        for (size_t shard = 0; shard < kReverseShardCount; ++shard) {
            const std::string reversePath = p_outputPath + ".crossbuild.reverse." +
                std::to_string(shard);
            const std::string sortedPath = reversePath + ".sorted";
            std::remove(reversePath.c_str());
            std::remove(sortedPath.c_str());
            artifacts.Add(reversePath);
            artifacts.Add(sortedPath);
            reversePaths.push_back(reversePath);
            sortedPaths.push_back(sortedPath);
        }
    }

    FILE* output = nullptr;
    FILE* base = nullptr;
    std::vector<FILE*> reverseFiles;
    if (hasUExtra) {
        base = std::fopen(basePath.c_str(), "wb");
        reverseFiles.assign(kReverseShardCount, nullptr);
        for (size_t shard = 0; shard < kReverseShardCount && base != nullptr; ++shard) {
            reverseFiles[shard] = std::fopen(reversePaths[shard].c_str(), "wb");
            if (reverseFiles[shard] == nullptr) {
                std::fclose(base);
                base = nullptr;
            }
        }
        if (base == nullptr) {
            for (FILE* file : reverseFiles) if (file != nullptr) std::fclose(file);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Cannot create bounded temporary files for cross-edge generation.\n");
            return false;
        }
    } else {
        output = std::fopen(temporaryPath.c_str(), "wb");
        if (output == nullptr) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Cannot open %s for write.\n", temporaryPath.c_str());
            return false;
        }
        Helper::HeadCrossEdgesHeader header{};
        header.magic = Helper::kHeadCrossEdgesMagic;
        header.version = Helper::kHeadCrossEdgesVersion;
        header.totalHeads = static_cast<std::int32_t>(totalHeads);
        header.maxEdgesPerHead = static_cast<std::int32_t>(maxEdgesPerHead);
        header.searchTopK = searchTopK;
        if (std::fwrite(&header, sizeof(header), 1, output) != 1) {
            std::fclose(output);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Cannot write cross-edge header to %s.\n", temporaryPath.c_str());
            return false;
        }
    }

    BuildStats stats;
    bool ok = true;
    for (size_t start = 0; start < totalHeads && ok; start += kWorkChunkHeads) {
        const size_t count = (std::min)(kWorkChunkHeads, totalHeads - start);
        std::vector<CrossEdgeRecord> records;
        if (!BuildRecordRange(p_nodes, prefixes, start, count, searchTopK, edgeLimit,
                              threadCount, totalHeads, stats, records)) {
            ok = false;
            break;
        }
        for (size_t offset = 0; offset < records.size(); ++offset) {
            CrossEdgeRecord& record = records[offset];
            FILE* destination = hasUExtra ? base : output;
            if (!WriteRecord(destination, record, edgeLimit)) {
                ok = false;
                break;
            }
            if (!hasUExtra) continue;

            const size_t ordinal = start + offset;
            const size_t sourceSlot = NodeForOrdinal(prefixes, ordinal);
            const SizeType localHid = static_cast<SizeType>(
                ordinal - NodeStart(prefixes, sourceSlot));
            if (localHid < p_nodes[sourceSlot].h1HeadCount) continue;
            for (const auto& edge : record.edges) {
                if (edge.reverseTargetOrdinal < 0) continue;
                const size_t shard = ReverseShard(
                    static_cast<size_t>(edge.reverseTargetOrdinal), totalHeads);
                const ReverseCandidate reverse{
                    edge.reverseTargetOrdinal, record.globalVID, edge.edge.dist};
                if (!WriteReverseCandidate(reverseFiles[shard], reverse)) {
                    ok = false;
                    break;
                }
            }
            if (!ok) break;
        }
    }

    if (hasUExtra) {
        if (base != nullptr && std::fclose(base) != 0) ok = false;
        for (FILE* file : reverseFiles) {
            if (file != nullptr && std::fclose(file) != 0) ok = false;
        }
    } else if (output != nullptr && std::fclose(output) != 0) {
        ok = false;
    }
    if (!ok) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "Cross-edge generation failed; no sidecar was committed.\n");
        return false;
    }

    if (hasUExtra) {
        for (size_t shard = 0; shard < kReverseShardCount; ++shard) {
            if (!SortReverseShard(reversePaths[shard], sortedPaths[shard], artifacts)) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                             "Cannot sort reverse cross-edge candidates for shard %zu.\n", shard);
                return false;
            }
        }

        base = std::fopen(basePath.c_str(), "rb");
        output = std::fopen(temporaryPath.c_str(), "wb");
        if (base == nullptr || output == nullptr) {
            if (base != nullptr) std::fclose(base);
            if (output != nullptr) std::fclose(output);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Cannot assemble cross-edge sidecar %s.\n", temporaryPath.c_str());
            return false;
        }
        Helper::HeadCrossEdgesHeader header{};
        header.magic = Helper::kHeadCrossEdgesMagic;
        header.version = Helper::kHeadCrossEdgesVersion;
        header.totalHeads = static_cast<std::int32_t>(totalHeads);
        header.maxEdgesPerHead = static_cast<std::int32_t>(maxEdgesPerHead);
        header.searchTopK = searchTopK;
        ok = std::fwrite(&header, sizeof(header), 1, output) == 1;

        size_t reverseAdded = 0;
        for (size_t shard = 0; shard < kReverseShardCount && ok; ++shard) {
            FILE* sorted = std::fopen(sortedPaths[shard].c_str(), "rb");
            if (sorted == nullptr) {
                ok = false;
                break;
            }
            ReverseCandidate current{};
            bool eof = false;
            bool hasCurrent = ReadReverseCandidate(sorted, current, eof);
            if (!hasCurrent && !eof) ok = false;

            const size_t start = ShardStart(shard, totalHeads);
            const size_t end = ShardStart(shard + 1, totalHeads);
            for (size_t ordinal = start; ordinal < end && ok; ++ordinal) {
                CrossEdgeRecord record;
                if (!ReadRecord(base, record, edgeLimit)) {
                    ok = false;
                    break;
                }
                const size_t sourceSlot = NodeForOrdinal(prefixes, ordinal);
                const SizeType localHid = static_cast<SizeType>(
                    ordinal - NodeStart(prefixes, sourceSlot));
                if (localHid < p_nodes[sourceSlot].h1HeadCount) {
                    std::vector<std::int32_t> seen;
                    seen.reserve(record.edges.size() + edgeLimit);
                    for (const auto& edge : record.edges) seen.push_back(edge.edge.neighborGlobalVID);
                    size_t accepted = 0;
                    while (hasCurrent && current.targetOrdinal == static_cast<std::int32_t>(ordinal)) {
                        if (std::find(seen.begin(), seen.end(), current.extraVID) == seen.end() &&
                            accepted < edgeLimit) {
                            record.edges.push_back({{current.extraVID, current.dist}, -1});
                            seen.push_back(current.extraVID);
                            ++accepted;
                            ++reverseAdded;
                        }
                        hasCurrent = ReadReverseCandidate(sorted, current, eof);
                        if (!hasCurrent && !eof) {
                            ok = false;
                            break;
                        }
                    }
                    std::sort(record.edges.begin(), record.edges.end(),
                              [](const CrossEdgeCandidate& p_left,
                                 const CrossEdgeCandidate& p_right) {
                                  return EntryLess(p_left.edge, p_right.edge);
                              });
                }
                if (hasCurrent && current.targetOrdinal < static_cast<std::int32_t>(ordinal)) {
                    ok = false;
                    break;
                }
                if (!WriteRecord(output, record, maxEdgesPerHead)) {
                    ok = false;
                    break;
                }
            }
            if (hasCurrent) ok = false;
            if (std::fclose(sorted) != 0) ok = false;
        }
        if (std::fgetc(base) != EOF || std::ferror(base) != 0) ok = false;
        if (std::fclose(base) != 0) ok = false;
        if (std::fclose(output) != 0) ok = false;
        if (!ok) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Cannot assemble valid bounded cross-edge sidecar.\n");
            return false;
        }
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                     "Cross-edge generation added %zu capped reverse H1-to-U_extra edges.\n",
                     reverseAdded);
    }

    if (!CommitOutput(temporaryPath, p_outputPath, p_options.overwrite)) {
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
                 totalHeads, stats.nonEmpty.load(), stats.fullyFilled.load(), p_outputPath.c_str());
    return true;
}
}
}
