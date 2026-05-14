// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/SPANN/Index.h"
#include "inc/Core/BKT/Index.h"
#include "inc/Core/KDT/Index.h"
#include "inc/Helper/VectorSetReaders/MemoryReader.h"
#include "inc/Core/SPANN/ExtraDynamicSearcher.h"
#include "inc/Core/SPANN/ExtraStaticSearcher.h"
#include "inc/Helper/HeadCrossEdges.h"
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <functional>
#include <map>
#include <queue>
#include <random>
#include <shared_mutex>
#include <unordered_set>

#include "inc/Core/ResultIterator.h"
#include "inc/Core/SPANN/SPANNResultIterator.h"

#pragma warning(disable : 4242) // '=' : conversion from 'int' to 'short', possible loss of data
#pragma warning(disable : 4244) // '=' : conversion from 'int' to 'short', possible loss of data
#pragma warning(disable : 4127) // conditional expression is constant

namespace SPTAG
{
template <typename T> thread_local std::unique_ptr<T> COMMON::ThreadLocalWorkSpaceFactory<T>::m_workspace;
namespace SPANN
{
EdgeCompare Selection::g_edgeComparer;

// Per-thread phase timings (set by CrossSubgraphGraphSearch, read by SearchIndex)
thread_local double g_bktSeedMs = 0.0;
thread_local double g_pqGraphMs = 0.0;

namespace
{
constexpr std::uint32_t kHeadBundleManifestMagic = 0x48424D46U;
constexpr std::int32_t kHeadBundleManifestVersion = 2;

// Helper to determine tag level from tag ID based on range
// Level 0 (org): 1000-1999, Level 1 (dept): 2000-2999,
// Level 2 (team): 3000-3999, Level 3 (project): 4000-4999
static inline int TagLevelFromId(uint32_t tag) {
    if (tag < 2000) return 0;
    if (tag < 3000) return 1;
    if (tag < 4000) return 2;
    return 3;
}

double GetMultiNodeBudgetKeepRatio()
{
    static const double kDefaultKeepRatio = 0.60;
    static const double ratio = []() {
        const char* value = std::getenv("SPTAG_MULTI_NODE_BUDGET_KEEP_RATIO");
        if (value == nullptr || *value == '\0') {
            return kDefaultKeepRatio;
        }

        char* end = nullptr;
        double parsed = std::strtod(value, &end);
        if (end == value || !std::isfinite(parsed) || parsed <= 0.0 || parsed > 1.0) {
            return kDefaultKeepRatio;
        }
        return parsed;
    }();
    return ratio;
}

// SPTAG_UNIFIED_NPROBE_BUDGET (default 1): use a single aggregate budget for
// all tag queries (including multi-subindex routed queries) instead of
// summing per-subindex budgets. Set to 0 to fall back to the legacy
// summed-child-budget × keepRatio formula.
bool UseUnifiedNprobeBudget()
{
    static const bool enabled = []() {
        const char* value = std::getenv("SPTAG_UNIFIED_NPROBE_BUDGET");
        if (value == nullptr || *value == '\0') return true;
        return !(value[0] == '0' && value[1] == '\0');
    }();
    return enabled;
}

// SPTAG_FIXED_NPROBE=N : if set to a positive integer, override adaptive
// nprobe estimation entirely and always use N as graphResultNum (still
// clamped to candidate posting count). Useful when adaptive estimation
// is uncertain and we want to fix the IO budget.
int GetFixedNprobeOverride()
{
    static const int v = []() {
        const char* value = std::getenv("SPTAG_FIXED_NPROBE");
        if (value == nullptr || *value == '\0') return 0;
        char* end = nullptr;
        long parsed = std::strtol(value, &end, 10);
        if (end == value || parsed <= 0 || parsed > 100000) return 0;
        return static_cast<int>(parsed);
    }();
    return v;
}

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

std::string HeadBundleManifestPath(const Options& p_options, const std::string& p_baseDir)
{
    std::string root = p_baseDir;
    if (!root.empty() && *(root.rbegin()) != FolderSep) {
        root += FolderSep;
    }
    return root + p_options.m_headIndexFolder + FolderSep + "head_bundle_manifest.bin";
}

std::string HeadBundleNodeRelativePath(const Options& p_options, int nodeId)
{
    return p_options.m_headIndexFolder + FolderSep + (std::string("node_") + std::to_string(nodeId));
}

std::string HeadBundleNodeAbsolutePath(const Options& p_options, const std::string& p_baseDir, int nodeId)
{
    std::string root = p_baseDir;
    if (!root.empty() && *(root.rbegin()) != FolderSep) {
        root += FolderSep;
    }
    return root + HeadBundleNodeRelativePath(p_options, nodeId);
}

bool EnsureDirectory(const std::string& path)
{
    if (path.empty()) return false;
    if (direxists(path.c_str())) return true;
    mkdir(path.c_str());
    return direxists(path.c_str());
}

std::string JoinPath(const std::string& baseDir, const std::string& relativePath)
{
    if (relativePath.empty()) return baseDir;

    std::string root = baseDir;
    if (!root.empty() && *(root.rbegin()) != FolderSep) {
        root += FolderSep;
    }
    return root + relativePath;
}

template <typename InternalDataType>
bool WriteSelectedHeadFiles(const COMMON::Dataset<InternalDataType>& data,
                            const std::vector<SizeType>& selected,
                            const std::string& vectorFilePath,
                            const std::string& idFilePath)
{
    std::shared_ptr<Helper::DiskIO> output = SPTAG::f_createIO(), outputIDs = SPTAG::f_createIO();
    if (output == nullptr || outputIDs == nullptr ||
        !output->Initialize(vectorFilePath.c_str(), std::ios::binary | std::ios::out) ||
        !outputIDs->Initialize(idFilePath.c_str(), std::ios::binary | std::ios::out))
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to create head selection files: %s %s\n",
                     vectorFilePath.c_str(), idFilePath.c_str());
        return false;
    }

    SizeType val = static_cast<SizeType>(selected.size());
    if (output->WriteBinary(sizeof(val), reinterpret_cast<char*>(&val)) != sizeof(val) ||
        outputIDs->WriteBinary(sizeof(val), reinterpret_cast<char*>(&val)) != sizeof(val))
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write head selection count.\n");
        return false;
    }

    DimensionType dims = data.C();
    if (output->WriteBinary(sizeof(dims), reinterpret_cast<char*>(&dims)) != sizeof(dims))
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write head vector dimensions.\n");
        return false;
    }

    DimensionType idDims = 1;
    if (outputIDs->WriteBinary(sizeof(idDims), reinterpret_cast<char*>(&idDims)) != sizeof(idDims))
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write head id dimensions.\n");
        return false;
    }

    for (SizeType vid : selected)
    {
        uint64_t storedVid = static_cast<uint64_t>(vid);
        if (outputIDs->WriteBinary(sizeof(storedVid), reinterpret_cast<char*>(&storedVid)) != sizeof(storedVid) ||
            output->WriteBinary(sizeof(InternalDataType) * data.C(), reinterpret_cast<const char*>(data[vid])) !=
                sizeof(InternalDataType) * data.C())
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write selected head vector %d.\n", vid);
            return false;
        }
    }

    return true;
}
}

std::function<std::shared_ptr<Helper::DiskIO>(void)> f_createAsyncIO = []() -> std::shared_ptr<Helper::DiskIO> {
    return std::shared_ptr<Helper::DiskIO>(new Helper::AsyncFileIO());
};

template <typename T> void Index<T>::InitializeDefaultHeadBundle()
{
    m_headBundleNodes.clear();

    HeadBundleNodeInfo nodeInfo;
    nodeInfo.nodeId = 0;
    nodeInfo.headIndexRelativePath = m_options.m_headIndexFolder;
    if (m_index != nullptr) {
        const SizeType sampleCount = m_index->GetNumSamples();
        nodeInfo.headCount = sampleCount;
        nodeInfo.postingCount = sampleCount;
        nodeInfo.assignmentCount = sampleCount;
    }
    m_headBundleNodes.emplace_back(std::move(nodeInfo));
}

template <typename T> ErrorCode Index<T>::SaveHeadBundleManifest(const std::string& p_baseDir) const
{
    if (p_baseDir.empty()) {
        return ErrorCode::Success;
    }

    std::vector<HeadBundleNodeInfo> nodes = m_headBundleNodes;
    if (nodes.empty()) {
        HeadBundleNodeInfo nodeInfo;
        nodeInfo.nodeId = 0;
        nodeInfo.headIndexRelativePath = m_options.m_headIndexFolder;
        if (m_index != nullptr) {
            const SizeType sampleCount = m_index->GetNumSamples();
            nodeInfo.headCount = sampleCount;
            nodeInfo.postingCount = sampleCount;
            nodeInfo.assignmentCount = sampleCount;
        }
        nodes.emplace_back(std::move(nodeInfo));
    }

    std::string headDir = p_baseDir;
    if (!headDir.empty() && *(headDir.rbegin()) != FolderSep) {
        headDir += FolderSep;
    }
    headDir += m_options.m_headIndexFolder;
    if (!direxists(headDir.c_str())) {
        mkdir(headDir.c_str());
    }

    const std::string manifestPath = HeadBundleManifestPath(m_options, p_baseDir);
    FILE* manifestFile = fopen(manifestPath.c_str(), "wb");
    if (manifestFile == nullptr) {
        return ErrorCode::FailedCreateFile;
    }

    HeadBundleManifestHeader header{};
    header.magic = kHeadBundleManifestMagic;
    header.version = kHeadBundleManifestVersion;
    header.nodeCount = static_cast<std::int32_t>(nodes.size());

    bool success = fwrite(&header, sizeof(header), 1, manifestFile) == 1;
    for (const auto& nodeInfo : nodes)
    {
        if (!success) break;

        const std::string& relativePath = nodeInfo.headIndexRelativePath;
        if (relativePath.size() > static_cast<size_t>(std::numeric_limits<std::int32_t>::max())) {
            success = false;
            break;
        }

        HeadBundleManifestNodeRecordV2 record{};
        record.nodeId = static_cast<std::int32_t>(nodeInfo.nodeId);
        record.pathLength = static_cast<std::int32_t>(relativePath.size());
        record.headOffset = static_cast<std::int64_t>(nodeInfo.headOffset);
        record.headCount = static_cast<std::int64_t>(nodeInfo.headCount);
        record.postingOffset = static_cast<std::int64_t>(nodeInfo.postingOffset);
        record.postingCount = static_cast<std::int64_t>(nodeInfo.postingCount);
        record.assignmentCount = static_cast<std::int64_t>(nodeInfo.assignmentCount);

        success = fwrite(&record, sizeof(record), 1, manifestFile) == 1;
        if (success && record.pathLength > 0) {
            success = fwrite(relativePath.data(), 1, static_cast<size_t>(record.pathLength), manifestFile) ==
                      static_cast<size_t>(record.pathLength);
        }
    }

    fclose(manifestFile);
    return success ? ErrorCode::Success : ErrorCode::Fail;
}

template <typename T> ErrorCode Index<T>::LoadHeadBundleManifest(const std::string& p_baseDir)
{
    m_headBundleNodes.clear();
    if (p_baseDir.empty()) {
        InitializeDefaultHeadBundle();
        return ErrorCode::Success;
    }

    const std::string manifestPath = HeadBundleManifestPath(m_options, p_baseDir);
    FILE* manifestFile = fopen(manifestPath.c_str(), "rb");
    if (manifestFile == nullptr) {
        InitializeDefaultHeadBundle();
        return ErrorCode::Success;
    }

    HeadBundleManifestHeader header{};
    bool success = fread(&header, sizeof(header), 1, manifestFile) == 1;
    success = success && header.magic == kHeadBundleManifestMagic &&
              header.version == kHeadBundleManifestVersion &&
              header.nodeCount >= 0;

    std::vector<HeadBundleNodeInfo> nodes;
    if (success) {
        nodes.reserve(static_cast<size_t>(header.nodeCount));
    }

    for (std::int32_t nodeIndex = 0; success && nodeIndex < header.nodeCount; ++nodeIndex)
    {
        HeadBundleManifestNodeRecordV2 record{};
        success = fread(&record, sizeof(record), 1, manifestFile) == 1;
        success = success && record.pathLength >= 0 && record.headOffset >= 0 && record.headCount >= 0 &&
                  record.postingOffset >= 0 && record.postingCount >= 0 && record.assignmentCount >= 0;
        if (!success) break;

        HeadBundleNodeInfo nodeInfo;
        nodeInfo.nodeId = static_cast<int>(record.nodeId);
        nodeInfo.headOffset = static_cast<SizeType>(record.headOffset);
        nodeInfo.headCount = static_cast<SizeType>(record.headCount);
        nodeInfo.postingOffset = static_cast<SizeType>(record.postingOffset);
        nodeInfo.postingCount = static_cast<SizeType>(record.postingCount);
        nodeInfo.assignmentCount = static_cast<SizeType>(record.assignmentCount);

        std::string relativePath(static_cast<size_t>(record.pathLength), '\0');
        if (record.pathLength > 0) {
            success = fread(&relativePath[0], 1, static_cast<size_t>(record.pathLength), manifestFile) ==
                      static_cast<size_t>(record.pathLength);
            if (!success) break;
        }

        nodeInfo.headIndexRelativePath = std::move(relativePath);
        nodes.emplace_back(std::move(nodeInfo));
    }

    fclose(manifestFile);
    if (!success || nodes.empty()) {
        InitializeDefaultHeadBundle();
        return ErrorCode::Success;
    }

    m_headBundleNodes = std::move(nodes);
    return ErrorCode::Success;
}

template <typename T> ErrorCode Index<T>::InitializeHeadBundleRuntime(const std::string& p_baseDir)
{
    std::lock_guard<std::mutex> lock(m_headBundleLoadLock);

    m_headBundleBaseDir = p_baseDir;
    m_loadedHeadBundleIndexes.clear();
    m_headBundleLocalToGlobalHIDs.clear();
    m_globalHeadVIDToLocalHID.clear();

    if (!m_headBundleNodes.empty())
    {
        m_loadedHeadBundleIndexes.resize(m_headBundleNodes.size());
        m_headBundleLocalToGlobalHIDs.resize(m_headBundleNodes.size());
    }

    return ErrorCode::Success;
}

template <typename T> ErrorCode Index<T>::EnsureHeadBundleNodeLoaded(int p_nodeId) const
{
    if (p_nodeId < 0 || p_nodeId >= static_cast<int>(m_headBundleNodes.size())) {
        return ErrorCode::Fail;
    }
    if (m_index == nullptr) {
        return ErrorCode::Fail;
    }

    const auto& nodeInfo = m_headBundleNodes[static_cast<size_t>(p_nodeId)];
    if (nodeInfo.headCount == 0) {
        return ErrorCode::Success;
    }

    std::lock_guard<std::mutex> lock(m_headBundleLoadLock);
    auto& localToGlobalHIDs = m_headBundleLocalToGlobalHIDs[static_cast<size_t>(p_nodeId)];

    if (m_globalHeadVIDToLocalHID.empty())
    {
        if (m_index->HasHeadNodeMeta())
        {
            const SizeType sampleCount = m_index->GetHeadNodeMetaSampleCount();
            m_globalHeadVIDToLocalHID.reserve(static_cast<size_t>(sampleCount) * 2 + 1);
            for (SizeType localHid = 0; localHid < sampleCount; ++localHid)
            {
                SizeType globalVID = m_index->GetHeadNodeGlobalVID(localHid);
                if (globalVID != MaxSize) {
                    m_globalHeadVIDToLocalHID[globalVID] = localHid;
                }
            }
        }
        else if (m_vectorTranslateMap.R() > 0)
        {
            m_globalHeadVIDToLocalHID.reserve(static_cast<size_t>(m_vectorTranslateMap.R()) * 2 + 1);
            for (SizeType localHid = 0; localHid < m_vectorTranslateMap.R(); ++localHid)
            {
                SizeType globalVID = static_cast<SizeType>(*(m_vectorTranslateMap[localHid]));
                if (globalVID != MaxSize) {
                    m_globalHeadVIDToLocalHID[globalVID] = localHid;
                }
            }
        }
    }

    const std::string nodeDir = JoinPath(m_headBundleBaseDir, nodeInfo.headIndexRelativePath);
    if (localToGlobalHIDs.empty())
    {
        COMMON::Dataset<std::uint64_t> nodeHeadIDs;
        nodeHeadIDs.SetName("HeadBundleNodeIDs");
        if (nodeHeadIDs.Load(nodeDir + FolderSep + m_options.m_headIDFile,
                             m_index->m_iDataBlockSize,
                             m_index->m_iDataCapacity) != ErrorCode::Success)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                         "Failed to load head bundle IDs for node %d from %s\n",
                         p_nodeId,
                         nodeDir.c_str());
            return ErrorCode::Fail;
        }

        localToGlobalHIDs.reserve(nodeHeadIDs.R());
        std::vector<std::pair<SizeType, SizeType>> reverseMapEntries;
        reverseMapEntries.reserve(nodeHeadIDs.R());
        for (SizeType localHid = 0; localHid < nodeHeadIDs.R(); ++localHid)
        {
            SizeType globalVID = static_cast<SizeType>(*(nodeHeadIDs[localHid]));
            auto globalHidIt = m_globalHeadVIDToLocalHID.find(globalVID);
            if (globalHidIt == m_globalHeadVIDToLocalHID.end())
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                             "Head bundle node %d references global VID %d with no global HID mapping.\n",
                             p_nodeId,
                             globalVID);
                localToGlobalHIDs.clear();
                return ErrorCode::Fail;
            }
            localToGlobalHIDs.push_back(globalHidIt->second);
            reverseMapEntries.emplace_back(globalVID, localHid);
        }

        // Publish the (globalVID -> (bundleNodeId, localHidWithinBundle)) reverse map
        // so cross-subgraph traversal can hop to a foreign node's BKT directly.
        {
            std::lock_guard<std::mutex> mapLock(m_globalVIDToBundleLocMutex);
            for (auto& entry : reverseMapEntries) {
                m_globalVIDToBundleLoc[entry.first] = std::make_pair(p_nodeId, entry.second);
            }
        }
    }

    if (m_loadedHeadBundleIndexes[static_cast<size_t>(p_nodeId)] == nullptr)
    {
        std::shared_ptr<VectorIndex> nodeIndex;
        if (VectorIndex::LoadIndex(nodeDir, nodeIndex) != ErrorCode::Success || nodeIndex == nullptr)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                         "Failed to load head bundle node index %d from %s\n",
                         p_nodeId,
                         nodeDir.c_str());
            return ErrorCode::Fail;
        }

        nodeIndex->SetParameter("NumberOfThreads", std::to_string(m_options.m_iSSDNumberOfThreads));
        nodeIndex->SetParameter("MaxCheck", std::to_string(m_options.m_maxCheck));
        if (nodeIndex->UpdateIndex() != ErrorCode::Success)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                         "Failed to update loaded head bundle node index %d from %s\n",
                         p_nodeId,
                         nodeDir.c_str());
            return ErrorCode::Fail;
        }
        nodeIndex->SetReady(true);
        m_loadedHeadBundleIndexes[static_cast<size_t>(p_nodeId)] = std::move(nodeIndex);
    }

    if (m_loadedHeadBundleIndexes[static_cast<size_t>(p_nodeId)] == nullptr ||
        m_loadedHeadBundleIndexes[static_cast<size_t>(p_nodeId)]->GetNumSamples() != localToGlobalHIDs.size())
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                     "Head bundle node %d sample count mismatch: index=%d mapping=%d\n",
                     p_nodeId,
                     m_loadedHeadBundleIndexes[static_cast<size_t>(p_nodeId)] == nullptr
                         ? -1
                         : static_cast<int>(m_loadedHeadBundleIndexes[static_cast<size_t>(p_nodeId)]->GetNumSamples()),
                     static_cast<int>(localToGlobalHIDs.size()));
        return ErrorCode::Fail;
    }

    return ErrorCode::Success;
}

template <typename T> ErrorCode Index<T>::LoadHeadCrossEdges() const
{
    if (m_headCrossEdgesLoaded.load(std::memory_order_acquire)) {
        return ErrorCode::Success;
    }
    std::lock_guard<std::mutex> lock(m_headCrossEdgesMutex);
    if (m_headCrossEdgesLoaded.load(std::memory_order_relaxed)) {
        return ErrorCode::Success;
    }

    std::string baseDir = m_headBundleBaseDir;
    if (baseDir.empty()) baseDir = m_options.m_indexDirectory;
    if (baseDir.empty()) {
        m_headCrossEdgesLoaded.store(true, std::memory_order_release);
        return ErrorCode::Success;
    }

    std::string path = baseDir;
    if (!path.empty() && path.back() != FolderSep) path += FolderSep;
    path += m_options.m_headIndexFolder;
    if (!path.empty() && path.back() != FolderSep) path += FolderSep;
    path += Helper::kHeadCrossEdgesFileName;

    FILE* fp = std::fopen(path.c_str(), "rb");
    if (fp == nullptr) {
        m_headCrossEdgesLoaded.store(true, std::memory_order_release);
        return ErrorCode::Success;
    }

    Helper::HeadCrossEdgesHeader header{};
    if (std::fread(&header, sizeof(header), 1, fp) != 1 ||
        header.magic != Helper::kHeadCrossEdgesMagic ||
        header.version != Helper::kHeadCrossEdgesVersion) {
        std::fclose(fp);
        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                     "head_cross_edges.bin format mismatch at %s — ignoring.\n", path.c_str());
        m_headCrossEdgesLoaded.store(true, std::memory_order_release);
        return ErrorCode::Success;
    }

    m_headCrossEdges.clear();
    m_headCrossEdges.reserve(static_cast<size_t>(header.totalHeads));
    bool ok = true;
    for (std::int32_t i = 0; i < header.totalHeads && ok; ++i) {
        std::int32_t globalVID = 0;
        std::int32_t edgeCount = 0;
        if (std::fread(&globalVID, sizeof(std::int32_t), 1, fp) != 1 ||
            std::fread(&edgeCount, sizeof(std::int32_t), 1, fp) != 1) { ok = false; break; }
        if (edgeCount < 0 || edgeCount > header.maxEdgesPerHead) { ok = false; break; }
        std::vector<Helper::HeadCrossEdgeEntry> entries(static_cast<size_t>(edgeCount));
        if (edgeCount > 0 &&
            std::fread(entries.data(), sizeof(Helper::HeadCrossEdgeEntry), entries.size(), fp) != entries.size()) {
            ok = false; break;
        }
        auto& neighbors = m_headCrossEdges[static_cast<SizeType>(globalVID)];
        neighbors.reserve(entries.size());
        for (const auto& e : entries) {
            neighbors.push_back(static_cast<SizeType>(e.neighborGlobalVID));
        }
    }
    std::fclose(fp);

    if (!ok) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                     "head_cross_edges.bin truncated at %s — partial load (%zu entries).\n",
                     path.c_str(), m_headCrossEdges.size());
    } else {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                     "Loaded head_cross_edges.bin: %d records, M=%d, K=%d.\n",
                     header.totalHeads, header.maxEdgesPerHead, header.searchTopK);
    }
    m_headCrossEdgesLoaded.store(true, std::memory_order_release);
    return ErrorCode::Success;
}

// Helper: access RNG graph + samples uniformly for either KDT or BKT head-bundle index.
template <typename T>
struct HeadBundleAccess {
    const COMMON::RelativeNeighborhoodGraph* graph = nullptr;
    DimensionType nbrSize = 0;
    std::function<const T*(SizeType)> getSample;
    SizeType numSamples = 0;
    bool valid = false;
};

template <typename T>
static HeadBundleAccess<T> AccessHeadBundleIndex(VectorIndex* p_idx) {
    HeadBundleAccess<T> a;
    if (p_idx == nullptr) return a;
    if (auto* b = dynamic_cast<BKT::Index<T>*>(p_idx)) {
        a.graph = &b->GetGraph();
        a.nbrSize = b->GetNeighborhoodSize();
        a.getSample = [b](SizeType i) { return static_cast<const T*>(b->GetSample(i)); };
        a.numSamples = b->GetNumSamples();
        a.valid = true;
    } else if (auto* k = dynamic_cast<KDT::Index<T>*>(p_idx)) {
        a.graph = &k->GetGraph();
        a.nbrSize = k->GetNeighborhoodSize();
        a.getSample = [k](SizeType i) { return static_cast<const T*>(k->GetSample(i)); };
        a.numSamples = k->GetNumSamples();
        a.valid = true;
    }
    return a;
}

template <typename T>
ErrorCode Index<T>::CrossSubgraphGraphSearch(
    QueryResult& p_query,
    COMMON::QueryResultSet<T>* p_queryResults,
    const std::vector<int>& p_candidateNodes,
    const std::uint32_t* p_queryTags,
    int p_numQueryTags,
    int p_graphResultNum,
    int& p_scannedOut) const
{
    if (p_candidateNodes.empty() || p_queryResults == nullptr || m_index == nullptr) {
        return ErrorCode::Fail;
    }
    if (LoadHeadCrossEdges() != ErrorCode::Success || m_headCrossEdges.empty()) {
        return ErrorCode::Fail;
    }

    // Force-load all bundle nodes once so the globalVID -> (nodeId, localHid)
    // reverse map is fully populated. With ~14-16 bundle nodes per tenant and
    // a few MB each, this is cheap and saves us mid-search lazy-load cost.
    for (const auto& bundleNodeInfo : m_headBundleNodes) {
        if (EnsureHeadBundleNodeLoaded(bundleNodeInfo.nodeId) != ErrorCode::Success) {
            return ErrorCode::Fail;
        }
    }

    int entryNode = p_candidateNodes.front();
    if (entryNode < 0 || entryNode >= static_cast<int>(m_loadedHeadBundleIndexes.size())) {
        return ErrorCode::Fail;
    }
    const auto& entryIdx = m_loadedHeadBundleIndexes[static_cast<size_t>(entryNode)];
    const auto& entryL2G = m_headBundleLocalToGlobalHIDs[static_cast<size_t>(entryNode)];
    if (entryIdx == nullptr || entryL2G.empty()) {
        return ErrorCode::Fail;
    }

    auto entryAcc = AccessHeadBundleIndex<T>(entryIdx.get());
    if (!entryAcc.valid) {
        return ErrorCode::Fail;
    }

    const T* qTarget = static_cast<const T*>(p_query.GetTarget());
    DimensionType dim = static_cast<DimensionType>(GetFeatureDim());

    // Build hierarchical query mask from query tags
    SPTAG::Cache::HierarchicalPostingMask queryHierMask;
    queryHierMask.Clear();
    if (p_queryTags != nullptr && p_numQueryTags > 0) {
        for (int i = 0; i < p_numQueryTags; ++i) {
            queryHierMask.Insert(TagLevelFromId(p_queryTags[i]), p_queryTags[i]);
        }
    }

    // Build routed node mask from candidateNodeSet
    uint32_t routedNodeMask = 0;
    for (int nid : p_candidateNodes) {
        if (nid >= 0 && nid < 32) {
            routedNodeMask |= (1u << nid);
        }
    }

    // Open priority queue (min-heap by distance) and visited set on globalVID.
    struct Cand {
        float dist;
        int nodeId;
        SizeType bundleLocal;
        SizeType globalA;     // tenant data global VID
        SizeType m_idx_B;     // m_index local hid (for AddPoint + PS lookup)
    };
    auto cmp = [](const Cand& a, const Cand& b) { return a.dist > b.dist; };
    std::priority_queue<Cand, std::vector<Cand>, decltype(cmp)> pq(cmp);
    std::unordered_set<SizeType> visitedA;
    visitedA.reserve(static_cast<size_t>(p_graphResultNum) * 16);

    auto pushCandidate = [&](int nodeId, SizeType bundleLocal, SizeType globalA,
                             SizeType m_idx_B, float dist) {
        if (visitedA.count(globalA)) return;
        pq.push({dist, nodeId, bundleLocal, globalA, m_idx_B});
    };

    // Phase 1: seed from EVERY routed node. Cross-edges alone are too sparse
    // a bridge when candidateNodes is small (e.g. 2-node dept queries) — the
    // entry-only seed leaves the other half of the routed scope unreachable.
    // We split the seed budget across all routed nodes and let each node's
    // BKT contribute its local nearest heads to the priority queue.
    //
    // Critical: per-node head indexes are configured with MaxCheck=4096 (the
    // global SPANN maxCheck). Calling their default SearchIndex() therefore
    // burns 4096 distance ops *per routed node* just to seed K~16 heads —
    // for a 4-node org query that's 16k ops before the unified PQ even
    // starts. We override with a tight budget here; the unified RNG +
    // cross-edge expansion below does the heavy lifting.
    int totalSeedK = std::max(16, std::min(p_graphResultNum, 64));
    int perNodeSeed = std::max(4, totalSeedK / static_cast<int>(p_candidateNodes.size()));
    int seedMaxCheck = std::max(64, perNodeSeed * 8);
    int scanned = 0;
    int totalSeeded = 0;
    int seedDroppedByTag = 0;

    auto _bktT0 = std::chrono::high_resolution_clock::now();

    for (int nodeId : p_candidateNodes) {
        if (nodeId < 0 || nodeId >= static_cast<int>(m_loadedHeadBundleIndexes.size())) continue;
        const auto& nodeIdx = m_loadedHeadBundleIndexes[static_cast<size_t>(nodeId)];
        const auto& nodeL2G = m_headBundleLocalToGlobalHIDs[static_cast<size_t>(nodeId)];
        if (nodeIdx == nullptr || nodeL2G.empty()) continue;
        auto nodeAcc = AccessHeadBundleIndex<T>(nodeIdx.get());
        if (!nodeAcc.valid) continue;

        int seedK = std::min(perNodeSeed, static_cast<int>(nodeL2G.size()));
        if (seedK <= 0) continue;

        COMMON::QueryResultSet<T> seedResults(qTarget, seedK);
        if (nodeIdx->SearchIndex(seedResults) != ErrorCode::Success) continue;
        scanned += seedResults.GetScanned();

        for (int i = 0; i < seedResults.GetResultNum(); ++i) {
            auto* r = seedResults.GetResult(i);
            if (r == nullptr || r->VID < 0) continue;
            if (static_cast<size_t>(r->VID) >= nodeL2G.size()) continue;
            SizeType bundleLocal = r->VID;
            SizeType m_idx_B = nodeL2G[static_cast<size_t>(bundleLocal)];
            SizeType globalA = m_index->GetHeadNodeGlobalVID(m_idx_B);
            if (globalA == MaxSize) continue;
            pushCandidate(nodeId, bundleLocal, globalA, m_idx_B, r->Dist);
            ++totalSeeded;
        }
    }

    if (totalSeeded == 0) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "CrossSubgraph DBG: totalSeeded=0 nodes=%zu seedMaxCheck=%d perNodeSeed=%d\n",
                     p_candidateNodes.size(), seedMaxCheck, perNodeSeed);
        return ErrorCode::Fail;
    }

    auto _bktT1 = std::chrono::high_resolution_clock::now();
    g_bktSeedMs = std::chrono::duration<double, std::milli>(_bktT1 - _bktT0).count();

    int maxChecks = std::max(m_options.m_maxCheck, p_graphResultNum * 4);
    int checks = 0;
    int crossHopCount = 0;
    int crossEdgesSeen = 0;
    int crossDroppedByTag = 0;

    // Early termination floor: ensure result heap is filled and PQ has settled
    // before allowing the worstDist gate to kick in. Without this, we would
    // exit before AddPoint has accumulated graphResultNum candidates (worstDist
    // is +inf until then, so it wouldn't trigger anyway, but pq could also be
    // tiny right after seeding).
    int minChecks = std::max(p_graphResultNum, std::min(256, maxChecks));

    while (!pq.empty() && checks < maxChecks) {
        Cand cur = pq.top();
        pq.pop();

        if (visitedA.count(cur.globalA)) continue;
        visitedA.insert(cur.globalA);
        ++checks;

        // Early termination: once we've done minChecks and the closest remaining
        // PQ candidate is already worse than the worst kept top-graphResultNum
        // distance, no future candidate can improve the result heap. This is the
        // same idea KDT uses (gnode.distance > p_query.worstDist()) but applied
        // to the cross-subgraph unified PQ.
        if (checks >= minChecks && cur.dist > p_queryResults->worstDist()) {
            break;
        }

        // Push to result heap. AddPoint uses m_idx_B (m_index local hid space) to
        // match the existing post-graph code path that translates B -> tenant
        // data globalVID downstream via translateHeadVID.
        // Tag-aware in-filter: only commit head as a result (and thus its posting
        // for I/O) if its hier_mask intersects the query mask. Continue walking
        // (RNG / cross-edge expansion) regardless so navigation isn't cut.
        bool keepHead = true;
        if (p_queryTags != nullptr && p_numQueryTags > 0 && m_index->HasHeadNodeMeta()) {
            keepHead = m_index->HeadHierMaskMayIntersect(cur.m_idx_B, queryHierMask);
        }
        if (keepHead) {
            p_queryResults->AddPoint(cur.m_idx_B, cur.dist);
        }

        // Expand RNG neighbors within the home node's head index (BKT or KDT).
        if (cur.nodeId >= 0 && cur.nodeId < static_cast<int>(m_loadedHeadBundleIndexes.size())) {
            const auto& homeIdx = m_loadedHeadBundleIndexes[static_cast<size_t>(cur.nodeId)];
            const auto& homeL2G = m_headBundleLocalToGlobalHIDs[static_cast<size_t>(cur.nodeId)];
            auto homeAcc = AccessHeadBundleIndex<T>(homeIdx.get());
            if (homeAcc.valid && cur.bundleLocal >= 0 &&
                cur.bundleLocal < homeAcc.numSamples) {
                const SizeType* nbrs = (*homeAcc.graph)[cur.bundleLocal];
                for (DimensionType i = 0; i < homeAcc.nbrSize; ++i) {
                    SizeType nbrLocal = nbrs[i];
                    if (nbrLocal < 0) break;
                    if (nbrLocal >= homeAcc.numSamples) continue;
                    if (static_cast<size_t>(nbrLocal) >= homeL2G.size()) continue;
                    SizeType nbr_B = homeL2G[static_cast<size_t>(nbrLocal)];
                    SizeType nbr_A = m_index->GetHeadNodeGlobalVID(nbr_B);
                    if (nbr_A == MaxSize) continue;
                    if (visitedA.count(nbr_A)) continue;
                    const T* nbrVec = homeAcc.getSample(nbrLocal);
                    if (nbrVec == nullptr) continue;
                    float d = m_fComputeDistance(qTarget, nbrVec, dim);
                    pushCandidate(cur.nodeId, nbrLocal, nbr_A, nbr_B, d);
                }
            }
        }

        // Expand cross-subgraph edges. Cross-edges store space A (tenant data
        // global VID). Translate the source key from B to A via m_index meta.
        static const bool s_disableCross = (std::getenv("SPTAG_DISABLE_CROSS_EDGES") != nullptr);
        auto crossIt = s_disableCross ? m_headCrossEdges.end() : m_headCrossEdges.find(cur.globalA);
        if (crossIt != m_headCrossEdges.end()) {
            for (SizeType nbrA : crossIt->second) {
                if (nbrA < 0) continue;
                ++crossEdgesSeen;
                if (visitedA.count(nbrA)) continue;

                // Hierarchical mask filter: skip neighbors that don't match query
                // This replaces the old HeadNodeMatchesAnyQueryTag check and integrates
                // bundle node routing.
                auto bIt = m_globalHeadVIDToLocalHID.find(nbrA);
                if (bIt == m_globalHeadVIDToLocalHID.end()) continue;
                SizeType nbr_B = bIt->second;
                if (p_queryTags != nullptr && p_numQueryTags > 0 && m_index->HasHeadNodeMeta()) {
                    // Bundle node routing check (separate from tag content)
                    if (routedNodeMask != 0) {
                        int16_t bundleNodeId = m_index->GetHeadNodeBundleNodeId(nbr_B);
                        if (bundleNodeId >= 0 &&
                            ((routedNodeMask >> bundleNodeId) & 1u) == 0) {
                            ++crossDroppedByTag;
                            continue;
                        }
                    }
                    // Tag-content check via posting hier_mask (no IsHeadNodeHeadOnly gate)
                    if (!m_index->HeadHierMaskMayIntersect(nbr_B, queryHierMask)) {
                        ++crossDroppedByTag;
                        continue;
                    }
                }

                // Locate which bundle node hosts this head and compute query
                // distance against its sample vector.
                std::pair<int, SizeType> loc;
                {
                    std::lock_guard<std::mutex> lk(m_globalVIDToBundleLocMutex);
                    auto locIt = m_globalVIDToBundleLoc.find(nbrA);
                    if (locIt == m_globalVIDToBundleLoc.end()) continue;
                    loc = locIt->second;
                }
                if (loc.first < 0 || loc.first >= static_cast<int>(m_loadedHeadBundleIndexes.size())) continue;
                const auto& othIdx = m_loadedHeadBundleIndexes[static_cast<size_t>(loc.first)];
                auto othAcc = AccessHeadBundleIndex<T>(othIdx.get());
                if (!othAcc.valid) continue;
                if (loc.second < 0 || loc.second >= othAcc.numSamples) continue;
                const T* nbrVec = othAcc.getSample(loc.second);
                if (nbrVec == nullptr) continue;

                float d = m_fComputeDistance(qTarget, nbrVec, dim);
                pushCandidate(loc.first, loc.second, nbrA, nbr_B, d);
                ++crossHopCount;
            }
        }
    }

    p_queryResults->SortResult();
    p_scannedOut = scanned + checks;

    if (m_options.m_logAdaptiveNprobe) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
            "CrossSubgraph: nodes=%zu totalSeeded=%d checks=%d crossHops=%d nodesVisited=%zu "
            "crossEdgesSeen=%d crossDroppedByTag=%d (tagPass=%.3f)\n",
            p_candidateNodes.size(), totalSeeded, checks, crossHopCount, visitedA.size(),
            crossEdgesSeen, crossDroppedByTag,
            crossEdgesSeen > 0 ? 1.0 - (double)crossDroppedByTag / crossEdgesSeen : 0.0);
    }
    {
        auto _pqT1 = std::chrono::high_resolution_clock::now();
        g_pqGraphMs = std::chrono::duration<double, std::milli>(_pqT1 - _bktT1).count();
    }
    static const bool s_logCS = (std::getenv("SPTAG_LOG_CROSS_STATS") != nullptr);
    if (s_logCS) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
            "CSStats: nodes=%zu seeded=%d seedDropTag=%d checks=%d cross=%d crossDropTag=%d visited=%zu maxC=%d\n",
            p_candidateNodes.size(), totalSeeded, seedDroppedByTag, checks, crossEdgesSeen, crossDroppedByTag,
            visitedA.size(), maxChecks);
    }
    return ErrorCode::Success;
}

template <typename T> bool Index<T>::CheckHeadIndexType()
{
    SPTAG::VectorValueType v1 = m_index->GetVectorValueType(), v2 = GetEnumValueType<T>();
    if (v1 != v2)
    {
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Error, "Head index and vectors don't have the same value types, which are %s %s\n",
            SPTAG::Helper::Convert::ConvertToString(v1).c_str(), SPTAG::Helper::Convert::ConvertToString(v2).c_str());
        if (!m_pQuantizer)
            return false;
    }
    return true;
}

template <typename T> void Index<T>::SetQuantizer(std::shared_ptr<SPTAG::COMMON::IQuantizer> quantizer)
{
    m_pQuantizer = quantizer;
    if (m_pQuantizer)
    {
        m_fComputeDistance = m_pQuantizer->DistanceCalcSelector<T>(m_options.m_distCalcMethod);
        m_iBaseSquare = (m_options.m_distCalcMethod == DistCalcMethod::Cosine)
                            ? m_pQuantizer->GetBase() * m_pQuantizer->GetBase()
                            : 1;
    }
    else
    {
        m_fComputeDistance = COMMON::DistanceCalcSelector<T>(m_options.m_distCalcMethod);
        m_iBaseSquare = (m_options.m_distCalcMethod == DistCalcMethod::Cosine)
                            ? COMMON::Utils::GetBase<std::uint8_t>() * COMMON::Utils::GetBase<std::uint8_t>()
                            : 1;
    }
    if (m_index)
    {
        m_index->SetQuantizer(quantizer);
    }
}

template <typename T> ErrorCode Index<T>::LoadConfig(Helper::IniReader &p_reader)
{
    IndexAlgoType algoType = p_reader.GetParameter("Base", "IndexAlgoType", IndexAlgoType::Undefined);
    VectorValueType valueType = p_reader.GetParameter("Base", "ValueType", VectorValueType::Undefined);
    if ((m_index = CreateInstance(algoType, valueType)) == nullptr)
        return ErrorCode::FailedParseValue;

    std::string sections[] = {"Base", "SelectHead", "BuildHead", "BuildSSDIndex"};
    for (int i = 0; i < 4; i++)
    {
        auto parameters = p_reader.GetParameters(sections[i].c_str());
        for (auto iter = parameters.begin(); iter != parameters.end(); iter++)
        {
            SetParameter(iter->first.c_str(), iter->second.c_str(), sections[i].c_str());
        }
    }

    if (m_pQuantizer)
    {
        m_pQuantizer->SetEnableADC(m_options.m_enableADC);
    }

    return ErrorCode::Success;
}

template <typename T> ErrorCode Index<T>::LoadIndexDataFromMemory(const std::vector<ByteArray> &p_indexBlobs)
{
    /** Need to modify **/
    m_index->SetQuantizer(m_pQuantizer);
    if (!m_options.m_persistentBufferPath.empty() && !direxists(m_options.m_persistentBufferPath.c_str()))
        mkdir(m_options.m_persistentBufferPath.c_str());

    if (m_index->LoadIndexDataFromMemory(p_indexBlobs) != ErrorCode::Success)
        return ErrorCode::Fail;

    m_index->SetParameter("NumberOfThreads", std::to_string(m_options.m_iSSDNumberOfThreads));
    // m_index->SetParameter("MaxCheck", std::to_string(m_options.m_maxCheck));
    // m_index->SetParameter("HashTableExponent", std::to_string(m_options.m_hashExp));
    m_index->UpdateIndex();
    m_index->SetReady(true);

    if (m_options.m_storage == Storage::STATIC)
    {
        if (m_pQuantizer)
            m_extraSearcher.reset(new ExtraStaticSearcher<std::uint8_t>());
        else
            m_extraSearcher.reset(new ExtraStaticSearcher<T>());
    }
    else
    {
        if (m_pQuantizer)
            m_extraSearcher.reset(new ExtraDynamicSearcher<std::uint8_t>(m_options));
        else
            m_extraSearcher.reset(new ExtraDynamicSearcher<T>(m_options));
    }

    if (!m_extraSearcher->LoadIndex(m_options, m_versionMap, m_vectorTranslateMap, m_index))
        return ErrorCode::Fail;

    m_vectorTranslateMap.Initialize(m_index->GetNumSamples(), 1, m_index->m_iDataBlockSize, m_index->m_iDataCapacity,
                                    p_indexBlobs.back().Data(), false);
    InitializeDefaultHeadBundle();
    InitializeHeadBundleRuntime(std::string());

    return ErrorCode::Success;
}

template <typename T>
ErrorCode Index<T>::LoadIndexData(const std::vector<std::shared_ptr<Helper::DiskIO>> &p_indexStreams)
{
    m_index->SetQuantizer(m_pQuantizer);
    if (!m_options.m_persistentBufferPath.empty() && !direxists(m_options.m_persistentBufferPath.c_str()))
        mkdir(m_options.m_persistentBufferPath.c_str());

    auto headfiles = m_index->GetIndexFiles();
    if (m_options.m_recovery)
    {
        std::shared_ptr<std::vector<std::string>> files(new std::vector<std::string>);
        auto headfiles = m_index->GetIndexFiles();
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Recovery: Loading another in-memory index\n");
        std::string filename = m_options.m_persistentBufferPath + FolderSep + m_options.m_headIndexFolder;
        for (auto file : *headfiles)
        {
            files->push_back(filename + FolderSep + file);
        }
        std::vector<std::shared_ptr<Helper::DiskIO>> handles;
        for (std::string &f : *files)
        {
            auto ptr = SPTAG::f_createIO();
            if (ptr == nullptr || !ptr->Initialize(f.c_str(), std::ios::binary | std::ios::in))
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot open file %s!\n", f.c_str());
                ptr = nullptr;
            }
            handles.push_back(std::move(ptr));
        }
        if (m_index->LoadIndexData(handles) != ErrorCode::Success)
            return ErrorCode::Fail;
    }
    else if (m_index->LoadIndexData(p_indexStreams) != ErrorCode::Success)
        return ErrorCode::Fail;

    m_index->SetParameter("NumberOfThreads", std::to_string(m_options.m_iSSDNumberOfThreads));
    m_index->SetParameter("MaxCheck", std::to_string(m_options.m_maxCheck));
    m_index->SetParameter("HashTableExponent", std::to_string(m_options.m_hashExp));
    m_index->UpdateIndex();
    m_index->SetReady(true);

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loading headID map\n");
    m_vectorTranslateMap.Load(p_indexStreams[m_index->GetIndexFiles()->size()], m_index->m_iDataBlockSize,
                              m_index->m_iDataCapacity);

    std::string kvpath = m_options.m_indexDirectory + FolderSep + m_options.m_KVFile;
    std::string ssdmappingpath = m_options.m_indexDirectory + FolderSep + m_options.m_ssdMappingFile;
    if (m_options.m_recovery)
    {
        kvpath = m_options.m_persistentBufferPath + FolderSep + m_options.m_KVFile;
        ssdmappingpath = m_options.m_persistentBufferPath + FolderSep + m_options.m_ssdMappingFile;
    }

    if (m_options.m_storage == Storage::STATIC)
    {
        if (m_pQuantizer)
            m_extraSearcher.reset(new ExtraStaticSearcher<std::uint8_t>());
        else
            m_extraSearcher.reset(new ExtraStaticSearcher<T>());
    }
    else
    {
        if (m_pQuantizer)
            m_extraSearcher.reset(new ExtraDynamicSearcher<std::uint8_t>(m_options));
        else
            m_extraSearcher.reset(new ExtraDynamicSearcher<T>(m_options));
    }

    if (m_options.m_storage != Storage::STATIC && !m_extraSearcher->Available())
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Extrasearcher is not available and failed to initialize.\n");
        return ErrorCode::Fail;
    }

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loading storage\n");
    if (!m_extraSearcher->LoadIndex(m_options, m_versionMap, m_vectorTranslateMap, m_index))
        return ErrorCode::Fail;

    if ((m_options.m_storage != Storage::STATIC) && m_options.m_preReassign)
    {
        if (m_extraSearcher->RefineIndex(m_index) != ErrorCode::Success)
            return ErrorCode::Fail;
    }

    const std::string bundleBaseDir = m_options.m_recovery ? m_options.m_persistentBufferPath : m_options.m_indexDirectory;
    if (LoadHeadBundleManifest(bundleBaseDir) != ErrorCode::Success)
        return ErrorCode::Fail;
    if (InitializeHeadBundleRuntime(bundleBaseDir) != ErrorCode::Success)
        return ErrorCode::Fail;

    return ErrorCode::Success;
}

template <typename T> ErrorCode Index<T>::SaveConfig(std::shared_ptr<Helper::DiskIO> p_configOut)
{
    IOSTRING(p_configOut, WriteString, "[Base]\n");
#define DefineBasicParameter(VarName, VarType, DefaultValue, RepresentStr)                                             \
    IOSTRING(p_configOut, WriteString,                                                                                 \
             (RepresentStr + std::string("=") + SPTAG::Helper::Convert::ConvertToString(m_options.VarName) +           \
              std::string("\n"))                                                                                       \
                 .c_str());

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineBasicParameter

    IOSTRING(p_configOut, WriteString, "[SelectHead]\n");
#define DefineSelectHeadParameter(VarName, VarType, DefaultValue, RepresentStr)                                        \
    IOSTRING(p_configOut, WriteString,                                                                                 \
             (RepresentStr + std::string("=") + SPTAG::Helper::Convert::ConvertToString(m_options.VarName) +           \
              std::string("\n"))                                                                                       \
                 .c_str());

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineSelectHeadParameter

    IOSTRING(p_configOut, WriteString, "[BuildHead]\n");
#define DefineBuildHeadParameter(VarName, VarType, DefaultValue, RepresentStr)                                         \
    IOSTRING(p_configOut, WriteString,                                                                                 \
             (RepresentStr + std::string("=") + SPTAG::Helper::Convert::ConvertToString(m_options.VarName) +           \
              std::string("\n"))                                                                                       \
                 .c_str());

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineBuildHeadParameter

    m_index->SaveConfig(p_configOut);

    Helper::Convert::ConvertStringTo<int>(m_index->GetParameter("HashTableExponent").c_str(), m_options.m_hashExp);
    IOSTRING(p_configOut, WriteString, "[BuildSSDIndex]\n");
#define DefineSSDParameter(VarName, VarType, DefaultValue, RepresentStr)                                               \
    IOSTRING(p_configOut, WriteString,                                                                                 \
             (RepresentStr + std::string("=") + SPTAG::Helper::Convert::ConvertToString(m_options.VarName) +           \
              std::string("\n"))                                                                                       \
                 .c_str());

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineSSDParameter

    IOSTRING(p_configOut, WriteString, "\n");
    return ErrorCode::Success;
}

template <typename T>
ErrorCode Index<T>::SaveIndexData(const std::vector<std::shared_ptr<Helper::DiskIO>> &p_indexStreams)
{
    if (m_index == nullptr || m_versionMap.Count() == 0)
        return ErrorCode::EmptyIndex;

    while (!AllFinished())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    ErrorCode ret;
    if ((ret = m_index->SaveIndexData(p_indexStreams)) != ErrorCode::Success)
        return ret;

    if ((ret = m_vectorTranslateMap.Save(p_indexStreams[m_index->GetIndexFiles()->size()])) != ErrorCode::Success)
        return ret;

    if ((ret = m_extraSearcher->Checkpoint(m_options.m_indexDirectory)) != ErrorCode::Success)
        return ret;

    if ((ret = SaveHeadBundleManifest(m_options.m_indexDirectory)) != ErrorCode::Success)
        return ret;
    return ErrorCode::Success;
}

#pragma region K - NN search

template <typename T> ErrorCode Index<T>::SearchIndex(QueryResult &p_query, bool p_searchDeleted) const
{
    if (!m_bReady)
        return ErrorCode::EmptyIndex;

    SPTAG::VectorIndex::ResetThreadLocalPostingScanStats();

    const auto* threadLocalSearchContext = SPTAG::VectorIndex::GetThreadLocalSearchContext();
    static const std::vector<SizeType> kEmptyDirectPostingIDs;
    static const std::function<bool(int)> kEmptyPostingFilter;
    const std::vector<SizeType>& directPostingIDs = threadLocalSearchContext != nullptr
        ? threadLocalSearchContext->m_directPostingIDs
        : kEmptyDirectPostingIDs;
    const uint32_t* queryTags = threadLocalSearchContext != nullptr
        ? threadLocalSearchContext->QueryTags()
        : nullptr;
    const int numQueryTags = threadLocalSearchContext != nullptr
        ? threadLocalSearchContext->NumQueryTags()
        : 0;
    const float filterSelectivity = threadLocalSearchContext != nullptr
        ? threadLocalSearchContext->m_filterSelectivity
        : 1.0f;
    const std::function<bool(int)>& postingFilter = threadLocalSearchContext != nullptr
        ? threadLocalSearchContext->m_postingFilter
        : kEmptyPostingFilter;
    static const std::vector<int> kEmptySearchHeadBundleNodes;
    const std::vector<int>& searchHeadBundleNodes = threadLocalSearchContext != nullptr
        ? threadLocalSearchContext->m_searchHeadBundleNodes
        : kEmptySearchHeadBundleNodes;

    // ═══ Sparse tag fast path: skip graph search, read postings directly ═══
    if (!directPostingIDs.empty() && m_extraSearcher != nullptr)
    {
        auto workSpace = m_workSpaceFactory->GetWorkSpace();
        if (!workSpace) {
            workSpace.reset(new ExtraWorkSpace());
            m_extraSearcher->InitWorkSpace(workSpace.get(), false);
        } else {
            m_extraSearcher->InitWorkSpace(workSpace.get(), true);
        }
        workSpace->m_queryTags = queryTags;
        workSpace->m_numQueryTags = numQueryTags;
        workSpace->m_deduper.clear();
        workSpace->m_postingIDs.clear();
        workSpace->m_postingFilter = nullptr;  // no PS needed, we know exact postings
        workSpace->m_postingProbeStats.Reset();

        const int directPostingCount = static_cast<int>(directPostingIDs.size());
        if (directPostingCount > m_options.m_searchInternalResultNum) {
            int maxPages = (std::max(m_options.m_postingPageLimit, m_options.m_searchPostingPageLimit)
                           + m_options.m_bufferLength) << PageSizeEx;
            workSpace->Clear(directPostingCount, maxPages, true, m_options.m_enableDataCompression);
        }

        // Directly inject all target posting IDs for sparse brute-force.
        int maxPostings = directPostingCount;
        for (SizeType pid : directPostingIDs) {
            if ((int)workSpace->m_postingIDs.size() >= maxPostings) break;
            if (m_extraSearcher->CheckValidPosting(pid)) {
                workSpace->m_postingIDs.emplace_back(pid);
            }
        }

        // Initialize result set — head vectors stay empty (no graph search)
        COMMON::QueryResultSet<T> *p_queryResults;
        if (p_query.GetResultNum() >= m_options.m_searchInternalResultNum)
            p_queryResults = (COMMON::QueryResultSet<T> *)&p_query;
        else
            p_queryResults = new COMMON::QueryResultSet<T>((const T *)p_query.GetTarget(),
                                                           m_options.m_searchInternalResultNum);

        // Read postings and scan with inline tag filter
        ErrorCode ret = m_extraSearcher->SearchIndex(workSpace.get(), *p_queryResults,
                                                     m_index, nullptr, nullptr, nullptr);
        SPTAG::VectorIndex::SetThreadLocalPostingScanStats(
            workSpace->m_postingProbeStats.m_readPostings,
            workSpace->m_postingProbeStats.m_matchedPostings);

        if (ret == ErrorCode::Success &&
            queryTags != nullptr &&
            numQueryTags > 0 &&
            m_index != nullptr)
        {
            // Build hierarchical query mask for head-only vectors
            SPTAG::Cache::HierarchicalPostingMask queryHierMask;
            queryHierMask.Clear();
            for (int i = 0; i < numQueryTags; ++i) {
                queryHierMask.Insert(TagLevelFromId(queryTags[i]), queryTags[i]);
            }

            const SizeType sampleCount = m_index->GetHeadNodeMetaSampleCount();
            for (SizeType sampleId = 0; sampleId < sampleCount; ++sampleId) {
                // Pass routedNodeMask=0 to skip node check (we're scanning all heads)
                if (!m_index->HeadNodeMatchesQuery(sampleId, queryHierMask, 0)) {
                    continue;
                }

                const void* headSample = m_index->GetSample(sampleId);
                if (headSample == nullptr) {
                    continue;
                }

                SizeType globalVID = m_index->GetHeadNodeGlobalVID(sampleId);
                if (globalVID == MaxSize) {
                    continue;
                }

                auto distance = m_index->ComputeDistance(p_queryResults->GetQuantizedTarget(), headSample);
                p_queryResults->AddPoint(globalVID, distance);
            }
        }

        p_queryResults->SortResult();
        if (p_queryResults != (COMMON::QueryResultSet<T>*)&p_query) {
            // Copy results back
            for (int i = 0; i < p_query.GetResultNum(); ++i) {
                auto* src = p_queryResults->GetResult(i);
                auto* dst = p_query.GetResult(i);
                dst->VID = src->VID;
                dst->Dist = src->Dist;
            }
            delete p_queryResults;
        }
        return ret;
    }

    // ═══ Normal path: graph search + post-graph PS + inline filter ═══
    // Adaptive nprobe: when tag filter is active, choose enough postings to
    // satisfy both (1) expected filtered top-k coverage and (2) graph-routing
    // coverage under the current selectivity.
    // expected_matches_per_posting ~= avg_posting_size * selectivity
    // postings_for_recall ~= target_recall * topk / expected_matches_per_posting
    // postings_for_coverage ~= nprobe_base / selectivity^coverage_exponent
    // final postingTarget = max(base, postings_for_recall, postings_for_coverage)
    // filterSelectivity is supplied by the thread-local ACL context at tenant
    // scope; for routed head-bundle queries we rescale it to candidate-node
    // scope before deriving postingTarget.
    int nprobeBase = std::max(m_options.m_searchInternalResultNum, p_query.GetResultNum());
    int postingTarget = nprobeBase;
    const bool adaptiveFilteredNprobeEnabled = m_options.m_enableAdaptiveFilteredNprobe;

    std::vector<int> candidateNodes;
    if (!searchHeadBundleNodes.empty() &&
        m_headBundleNodes.size() > 1 &&
        searchHeadBundleNodes.size() < m_headBundleNodes.size())
    {
        candidateNodes.reserve(searchHeadBundleNodes.size());
        for (int nodeId : searchHeadBundleNodes)
        {
            if (nodeId < 0 || nodeId >= static_cast<int>(m_headBundleNodes.size())) {
                continue;
            }
            if (m_headBundleNodes[static_cast<size_t>(nodeId)].headCount == 0 ||
                m_headBundleNodes[static_cast<size_t>(nodeId)].postingCount == 0) {
                continue;
            }
            candidateNodes.push_back(nodeId);
        }
    }

    if (adaptiveFilteredNprobeEnabled && filterSelectivity < 1.0f) {
        const double globalTenantSize = static_cast<double>(m_options.m_vectorSize > 0 ? m_options.m_vectorSize : m_index->GetNumSamples());
        const SizeType globalPostingCount = std::max<SizeType>(1, m_index->GetNumSamples());
        const double globalAvgPosting = std::max(1.0, globalTenantSize / static_cast<double>(globalPostingCount));

        float recallTarget = m_options.m_filteredSearchTargetRecall;
        if (recallTarget < 0.01f) recallTarget = 0.01f;
        if (recallTarget > 1.0f) recallTarget = 1.0f;

        float coverageExponent = m_options.m_filteredSearchCoverageExponent;
        if (coverageExponent < 0.0f) coverageExponent = 0.0f;
        if (coverageExponent > 2.0f) coverageExponent = 2.0f;

        int filteredTopK = p_query.GetResultNum();
        if (filteredTopK <= 0) filteredTopK = 10;

        auto computeAdaptivePostingTargetForScope = [&](double scopeTenantSize,
                                                        SizeType scopePostingCount,
                                                        double scopeSelectivity) -> int {
            if (scopeTenantSize <= 0.0 || scopePostingCount == 0) {
                return nprobeBase;
            }

            double sel = scopeSelectivity;
            if (sel < 1e-6) sel = 1e-6;
            if (sel > 1.0) sel = 1.0;

            double postingCount = static_cast<double>(scopePostingCount);
            double avgPosting = scopeTenantSize / postingCount;
            if (avgPosting < 1.0) avgPosting = 1.0;

            double expectedMatchesPerPosting = avgPosting * sel;
            if (expectedMatchesPerPosting < 1e-6) expectedMatchesPerPosting = 1e-6;

            int postingsForRecall = static_cast<int>(std::ceil(
                (static_cast<double>(filteredTopK) * static_cast<double>(recallTarget)) /
                expectedMatchesPerPosting));

            int target = std::max(nprobeBase, postingsForRecall);

            // Optional coverage term: only when explicitly enabled (exponent > 0).
            // The 1/sel^exp scaling has no theoretical basis and tends to dominate
            // postingsForRecall for low-sel queries, so it is opt-in.
            if (coverageExponent > 1e-6f) {
                double coverageDenominator = std::pow(sel, static_cast<double>(coverageExponent));
                if (coverageDenominator < 1e-6) coverageDenominator = 1e-6;
                int postingsForCoverage = static_cast<int>(std::ceil(
                    static_cast<double>(nprobeBase) / coverageDenominator));
                target = std::max(target, postingsForCoverage);
            }

            return std::min(static_cast<int>(scopePostingCount), target);
        };

        SizeType postingCountCap = globalPostingCount;
        double candidateTenantSize = globalTenantSize;
        double aggregateSelectivity = static_cast<double>(filterSelectivity);
        if (!candidateNodes.empty()) {
            SizeType candidatePostingCount = 0;
            double candidateAssignmentCount = 0.0;
            for (int nodeId : candidateNodes)
            {
                const auto& nodeInfo = m_headBundleNodes[static_cast<size_t>(nodeId)];
                candidatePostingCount += nodeInfo.postingCount;
                candidateAssignmentCount += static_cast<double>(nodeInfo.assignmentCount);
            }

            if (candidatePostingCount > 0) {
                postingCountCap = candidatePostingCount;
                candidateTenantSize = (candidateAssignmentCount > 0.0)
                    ? candidateAssignmentCount
                    : globalAvgPosting * static_cast<double>(candidatePostingCount);

                if (candidateTenantSize > 0.0 && globalTenantSize > 0.0) {
                    aggregateSelectivity *= (globalTenantSize / candidateTenantSize);
                }
            }
        }

        int aggregatePostingTarget = computeAdaptivePostingTargetForScope(
            candidateTenantSize,
            postingCountCap,
            aggregateSelectivity);

        postingTarget = aggregatePostingTarget;

        // For broad routed queries that span multiple bundle nodes, derive the
        // total budget from the sum of child-node budgets instead of relying
        // only on one aggregate local-selectivity estimate, then keep a
        // configurable fraction after merge by trimming the tail budget.
        // Disabled by default: SPTAG_UNIFIED_NPROBE_BUDGET=1 makes all routed
        // queries (including multi-subindex) trust the single aggregate budget,
        // which matches the unified cross-subgraph PQ search and avoids
        // amplifying nprobe by the number of routed subindexes.
        if (candidateNodes.size() > 1 && !UseUnifiedNprobeBudget()) {
            const double multiNodeBudgetKeepRatio = GetMultiNodeBudgetKeepRatio();
            long long summedChildPostingTarget = 0;
            for (int nodeId : candidateNodes)
            {
                const auto& nodeInfo = m_headBundleNodes[static_cast<size_t>(nodeId)];
                if (nodeInfo.postingCount == 0) {
                    continue;
                }

                double nodeTenantSize = (nodeInfo.assignmentCount > 0)
                    ? static_cast<double>(nodeInfo.assignmentCount)
                    : globalAvgPosting * static_cast<double>(nodeInfo.postingCount);
                double nodeSelectivity = static_cast<double>(filterSelectivity);
                if (nodeTenantSize > 0.0 && globalTenantSize > 0.0) {
                    nodeSelectivity *= (globalTenantSize / nodeTenantSize);
                }

                summedChildPostingTarget += static_cast<long long>(computeAdaptivePostingTargetForScope(
                    nodeTenantSize,
                    nodeInfo.postingCount,
                    nodeSelectivity));
            }

            if (summedChildPostingTarget > 0) {
                int trimmedChildPostingTarget = static_cast<int>(std::ceil(
                    static_cast<double>(summedChildPostingTarget) * multiNodeBudgetKeepRatio));
                trimmedChildPostingTarget = std::max(nprobeBase, trimmedChildPostingTarget);
                postingTarget = std::min(static_cast<int>(postingCountCap),
                                         std::max(aggregatePostingTarget, trimmedChildPostingTarget));
            }
        }
    }

    // SPTAG_FIXED_NPROBE override: if set, force a fixed posting budget for
    // all tag queries regardless of adaptive estimation. Still clamped to
    // available posting count.
    {
        int fixedNprobe = GetFixedNprobeOverride();
        if (fixedNprobe > 0) {
            SizeType cap = m_index ? m_index->GetNumSamples() : fixedNprobe;
            if (!candidateNodes.empty()) {
                SizeType candidateCap = 0;
                for (int nodeId : candidateNodes) {
                    candidateCap += m_headBundleNodes[static_cast<size_t>(nodeId)].postingCount;
                }
                if (candidateCap > 0) cap = candidateCap;
            }
            postingTarget = std::min(fixedNprobe, static_cast<int>(cap));
        }
    }

    if (m_options.m_logAdaptiveNprobe && adaptiveFilteredNprobeEnabled) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
            "AdaptiveNprobe: sel=%.4g topK=%d recallTarget=%.3g coverageExp=%.3g "
            "nprobeBase=%d nodes=%zu cap=%d -> postingTarget=%d\n",
            static_cast<double>(filterSelectivity),
            p_query.GetResultNum(),
            static_cast<double>(m_options.m_filteredSearchTargetRecall),
            static_cast<double>(m_options.m_filteredSearchCoverageExponent),
            nprobeBase,
            candidateNodes.size(),
            static_cast<int>(m_index->GetNumSamples()),
            postingTarget);
    }

    // Graph search must return at least postingTarget candidates
    // (PS will filter some out, so request more)
    int graphResultNum = postingTarget;

    COMMON::QueryResultSet<T> *p_queryResults;
    if (p_query.GetResultNum() >= graphResultNum)
        p_queryResults = (COMMON::QueryResultSet<T> *)&p_query;
    else
        p_queryResults =
            new COMMON::QueryResultSet<T>((const T *)p_query.GetTarget(), graphResultNum);

    ErrorCode ret;
    bool usedHeadBundleGraphSearch = false;
    static const bool s_phaseTime = (std::getenv("SPTAG_LOG_PHASE_TIME") != nullptr);
    g_bktSeedMs = 0.0;
    g_pqGraphMs = 0.0;
    auto _phT0 = s_phaseTime ? std::chrono::high_resolution_clock::now()
                             : std::chrono::high_resolution_clock::time_point{};
    if (!candidateNodes.empty())
    {
        bool canUseHeadBundle = true;
        for (int nodeId : candidateNodes)
        {
            if (EnsureHeadBundleNodeLoaded(nodeId) != ErrorCode::Success)
            {
                canUseHeadBundle = false;
                break;
            }
        }

        if (canUseHeadBundle)
        {
            p_queryResults->Reset();
            int scanned = 0;

            // Cross-subgraph unified traversal: when (a) cross-edges are
            // available, (b) the query tag scope spans more than one routing
            // node, and (c) we have query tags for PS in-filtering, run a
            // single best-first search across all bundle nodes' BKTs joined
            // by cross-edges instead of the serial per-node fanout. This
            // turns nprobe amplification (K_nodes * nprobe) into a single
            // budget walk while still covering query's true k-NN via cross
            // shortcut edges.
            bool useCrossSubgraph = (candidateNodes.size() > 1)
                && (queryTags != nullptr && numQueryTags > 0)
                && (LoadHeadCrossEdges() == ErrorCode::Success)
                && (!m_headCrossEdges.empty());

            static const bool s_logPathStats = (std::getenv("SPTAG_LOG_PATH_STATS") != nullptr);
            if (s_logPathStats) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                    "PathStats: nodes=%d cross=%d\n",
                    static_cast<int>(candidateNodes.size()), useCrossSubgraph ? 1 : 0);
            }

            if (useCrossSubgraph)
            {
                ret = CrossSubgraphGraphSearch(p_query, p_queryResults, candidateNodes,
                                               queryTags, numQueryTags, graphResultNum, scanned);
                if (ret != ErrorCode::Success) {
                    canUseHeadBundle = false;
                    p_queryResults->Reset();
                }
                else if (m_options.m_logAdaptiveNprobe)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                 "Using cross-subgraph unified search across %d candidate nodes (entry=%d).\n",
                                 static_cast<int>(candidateNodes.size()), candidateNodes.front());
                }
            }
            else
            {
                // Tag-aware head pre-filter: when a hierarchical query mask is
                // present, expand KDT search by alpha and filter returned heads
                // by hier_mask intersection so the downstream posting scan
                // budget is spent on tag-relevant heads (in-filter).
                static const int s_tagAwareExpansion = []() {
                    const char* e = std::getenv("SPTAG_TAG_AWARE_HEAD_EXPANSION");
                    int v = (e != nullptr) ? std::atoi(e) : 4;
                    if (v < 1) v = 1;
                    if (v > 32) v = 32;
                    return v;
                }();
                const bool hasTagFilterLocal = (queryTags != nullptr && numQueryTags > 0);
                SPTAG::Cache::HierarchicalPostingMask tagAwareQueryMask;
                const bool tagAwareEnabled = hasTagFilterLocal && s_tagAwareExpansion > 1
                    && m_index != nullptr && m_index->HasHeadNodeMeta();
                if (tagAwareEnabled) {
                    tagAwareQueryMask.Clear();
                    for (int qi = 0; qi < numQueryTags; ++qi) {
                        tagAwareQueryMask.Insert(TagLevelFromId(queryTags[qi]), queryTags[qi]);
                    }
                }

                for (int nodeId : candidateNodes)
                {
                const auto& nodeIndex = m_loadedHeadBundleIndexes[static_cast<size_t>(nodeId)];
                const auto& localToGlobalHIDs = m_headBundleLocalToGlobalHIDs[static_cast<size_t>(nodeId)];
                if (nodeIndex == nullptr || localToGlobalHIDs.empty())
                {
                    canUseHeadBundle = false;
                    break;
                }

                const int nodeGraphResultNum = std::min<int>(graphResultNum,
                                                             static_cast<int>(localToGlobalHIDs.size()));
                if (nodeGraphResultNum <= 0) {
                    continue;
                }

                const int searchNum = tagAwareEnabled
                    ? std::min<int>(nodeGraphResultNum * s_tagAwareExpansion,
                                    static_cast<int>(localToGlobalHIDs.size()))
                    : nodeGraphResultNum;

                COMMON::QueryResultSet<T> nodeResults((const T*)p_query.GetTarget(), searchNum);
                if ((ret = nodeIndex->SearchIndex(nodeResults)) != ErrorCode::Success)
                {
                    canUseHeadBundle = false;
                    break;
                }

                scanned += nodeResults.GetScanned();
                int kept = 0;
                for (int resultId = 0; resultId < nodeResults.GetResultNum(); ++resultId)
                {
                    if (kept >= nodeGraphResultNum) break;
                    auto* nodeResult = nodeResults.GetResult(resultId);
                    if (nodeResult == nullptr || nodeResult->VID == -1) {
                        continue;
                    }

                    if (nodeResult->VID < 0 || nodeResult->VID >= static_cast<SizeType>(localToGlobalHIDs.size())) {
                        continue;
                    }

                    SizeType globalHID = localToGlobalHIDs[static_cast<size_t>(nodeResult->VID)];

                    if (tagAwareEnabled
                        && !m_index->HeadHierMaskMayIntersect(globalHID, tagAwareQueryMask)) {
                        continue;
                    }

                    p_queryResults->AddPoint(globalHID, nodeResult->Dist);
                    ++kept;
                }
                }  // end for(nodeId : candidateNodes)
            }  // end else (per-node fanout branch)

            if (canUseHeadBundle)
            {
                if (!useCrossSubgraph) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                 "Using routed head bundle graph search across %d nodes.\n",
                                 static_cast<int>(candidateNodes.size()));
                    p_queryResults->SortResult();
                }
                p_queryResults->SetScanned(scanned);
                usedHeadBundleGraphSearch = true;
            }
            else
            {
                p_queryResults->Reset();
            }
        }
    }

    if (!usedHeadBundleGraphSearch)
    {
        // No graph-level filter — pure distance-based greedy navigation
        if ((ret = m_index->SearchIndex(*p_queryResults)) != ErrorCode::Success)
            return ret;
    }

    auto _phT1 = s_phaseTime ? std::chrono::high_resolution_clock::now()
                             : std::chrono::high_resolution_clock::time_point{};

    if (m_extraSearcher != nullptr)
    {
        auto workSpace = m_workSpaceFactory->GetWorkSpace();
        if (!workSpace)
        {
            workSpace.reset(new ExtraWorkSpace());
            m_extraSearcher->InitWorkSpace(workSpace.get(), false);
        }
        else
        {
            m_extraSearcher->InitWorkSpace(workSpace.get(), true);
        }

        // If adaptive nprobe > base, expand workspace buffers
        if (postingTarget > nprobeBase) {
            int maxPages = (std::max(m_options.m_postingPageLimit, m_options.m_searchPostingPageLimit)
                           + m_options.m_bufferLength) << PageSizeEx;
            workSpace->Clear(postingTarget, maxPages, true, m_options.m_enableDataCompression);
        }

        // Propagate posting-level PS pre-filter (applied in ExtraDynamicSearcher before MultiGet)
        workSpace->m_postingFilter = postingFilter;
        // Propagate inline tag filter (for per-vector exact tag check in posting scan)
        workSpace->m_queryTags = queryTags;
        workSpace->m_numQueryTags = numQueryTags;
        workSpace->m_deduper.clear();
        workSpace->m_postingIDs.clear();
        workSpace->m_postingProbeStats.Reset();

        const bool hasTagFilter = queryTags != nullptr && numQueryTags > 0;
        // Build hierarchical query mask once
        SPTAG::Cache::HierarchicalPostingMask queryHierMask;
        if (hasTagFilter) {
            queryHierMask.Clear();
            for (int i = 0; i < numQueryTags; ++i) {
                queryHierMask.Insert(TagLevelFromId(queryTags[i]), queryTags[i]);
            }
        }
        auto translateHeadVID = [&](SizeType localHid) -> SizeType {
            if (m_index != nullptr && m_index->HasHeadNodeMeta()) {
                SizeType metaVID = m_index->GetHeadNodeGlobalVID(localHid);
                if (metaVID != MaxSize) return metaVID;
            }
            if (m_vectorTranslateMap.R() != 0)
                return static_cast<SizeType>(*(m_vectorTranslateMap[localHid]));
            return MaxSize;
        };
        auto shouldKeepHeadResult = [&](SizeType localHid) -> bool {
            if (!hasTagFilter) return true;
            // Intentionally uses HeadNodeMatchesQuery (with IsHeadNodeHeadOnly
            // gate) so that only ghost head-only vectors can be returned as
            // top-K results. For real heads (centroids of postings) the head
            // VID's own tag is NOT guaranteed to match the query even when its
            // posting members do; rejecting them here ensures top-K is sourced
            // only from posting scans (and the rare head-only ghost vectors).
            return m_index != nullptr &&
                   m_index->HasHeadNodeMeta() &&
                   m_index->HeadNodeMatchesQuery(localHid, queryHierMask, 0);
        };

        float limitDist = p_queryResults->GetResult(0)->Dist * m_options.m_maxDistRatio;
        int i = 0;
        for (; i < graphResultNum; ++i)
        {
            if ((int)workSpace->m_postingIDs.size() >= postingTarget) break;
            auto res = p_queryResults->GetResult(i);
            if (res->VID == -1 || (limitDist > 0.1 && res->Dist > limitDist))
                break;
            SizeType localHid = res->VID;
            if (m_extraSearcher->CheckValidPosting(localHid))
            {
                // Post-graph PS filter: only add postings that pass PS check
                if (!workSpace->m_postingFilter || workSpace->m_postingFilter(localHid)) {
                    workSpace->m_postingIDs.emplace_back(localHid);
                }
            }
            SizeType globalVID = translateHeadVID(localHid);
            if (!shouldKeepHeadResult(localHid) || globalVID == MaxSize)
            {
                res->VID = -1;
                res->Dist = MaxDist;
            } else {
                res->VID = globalVID;
            }
        }

        for (; i < p_queryResults->GetResultNum(); ++i)
        {
            auto res = p_queryResults->GetResult(i);
            if (res->VID == -1)
                break;

            SizeType localHid = res->VID;
            SizeType globalVID = translateHeadVID(localHid);
            if (!shouldKeepHeadResult(localHid) || globalVID == MaxSize)
            {
                res->VID = -1;
                res->Dist = MaxDist;
            } else {
                res->VID = globalVID;
            }
        }

        int head = 0;
        for (int j = 0; j < p_queryResults->GetResultNum(); ++j)
        {
            SPTAG::BasicResult* ri = p_queryResults->GetResult(j);
            bool keep = false;
            if (ri->VID != -1 && !m_versionMap.Deleted(ri->VID) && !workSpace->m_deduper.CheckAndSet(ri->VID))
            {
                keep = true;
            }

            if (keep)
            {
                if (head != j)
                {
                    SPTAG::BasicResult* rhead = p_queryResults->GetResult(head);
                    *rhead = *ri;
                    ri->VID = -1;
                    ri->Dist = MaxDist;
                }

                ++head;
            }
            else
            {
                ri->VID = -1;
                ri->Dist = MaxDist;
            }
        }

        p_queryResults->Reverse();
        {
            static const bool s_dumpHeads = (std::getenv("SPTAG_DUMP_HEADS") != nullptr);
            if (s_dumpHeads) {
                std::string s = "HEADDUMP:";
                s.reserve(workSpace->m_postingIDs.size() * 8 + 16);
                for (auto h : workSpace->m_postingIDs) {
                    s.push_back(' ');
                    s += std::to_string(static_cast<long long>(h));
                }
                fprintf(stderr, "%s\n", s.c_str());
                fflush(stderr);
            }
        }
        ret = m_extraSearcher->SearchIndex(workSpace.get(), *p_queryResults, m_index, nullptr);
        SPTAG::VectorIndex::SetThreadLocalPostingScanStats(
            workSpace->m_postingProbeStats.m_readPostings,
            workSpace->m_postingProbeStats.m_matchedPostings);
        if (ret != ErrorCode::Success)
            return ret;
        m_workSpaceFactory->ReturnWorkSpace(std::move(workSpace));
        p_queryResults->SortResult();
    }

    if (s_phaseTime) {
        auto _phT2 = std::chrono::high_resolution_clock::now();
        double graphTotalMs = std::chrono::duration<double, std::milli>(_phT1 - _phT0).count();
        double postMs  = std::chrono::duration<double, std::milli>(_phT2 - _phT1).count();
        uint32_t firstTag = (queryTags != nullptr && numQueryTags > 0) ? queryTags[0] : 0u;
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
            "PhaseTime: tag=%u nprobe=%d bkt=%.3f pq=%.3f graphOther=%.3f post=%.3f total=%.3f\n",
            firstTag, postingTarget, g_bktSeedMs, g_pqGraphMs,
            graphTotalMs - g_bktSeedMs - g_pqGraphMs, postMs, graphTotalMs + postMs);
    }

    if (p_queryResults != (COMMON::QueryResultSet<T> *)&p_query)
    {
        std::copy(p_queryResults->GetResults(), p_queryResults->GetResults() + p_query.GetResultNum(),
                  p_query.GetResults());
        p_query.SetScanned(p_queryResults->GetScanned());
        delete p_queryResults;
    }

    if (p_query.WithMeta() && nullptr != m_pMetadata)
    {
        for (int i = 0; i < p_query.GetResultNum(); ++i)
        {
            SizeType result = p_query.GetResult(i)->VID;
            // if (result > m_pMetadata->Count()) SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "vid return is beyond the
            // metadata set:(%d > (%d, %d))\n", result, GetNumSamples(), m_pMetadata->Count());
            p_query.SetMetadata(i, (result < 0) ? ByteArray::c_empty : m_pMetadata->GetMetadataCopy(result));
        }
    }
    return ErrorCode::Success;
}

template <typename T>
ErrorCode Index<T>::SearchIndexIterative(QueryResult &p_headQuery, QueryResult &p_query,
                                         COMMON::WorkSpace *p_indexWorkspace, ExtraWorkSpace *p_extraWorkspace,
                                         int p_batch, int &resultCount, bool first) const
{
    if (!m_bReady)
        return ErrorCode::EmptyIndex;

    COMMON::QueryResultSet<T> *p_headQueryResults = (COMMON::QueryResultSet<T> *)&p_headQuery;
    COMMON::QueryResultSet<T> *p_queryResults = (COMMON::QueryResultSet<T> *)&p_query;

    if (first)
    {
        p_headQueryResults->SetResultNum(m_options.m_searchInternalResultNum);
        p_headQueryResults->Reset();
        m_index->SearchIndexIterativeFromNeareast(*p_headQueryResults, p_indexWorkspace, true);
        p_extraWorkspace->m_loadPosting = true;
    }

    bool continueSearch = true;
    resultCount = 0;
    while (continueSearch && resultCount < p_batch)
    {
        bool oldRelaxedMono = p_extraWorkspace->m_relaxedMono;
        ErrorCode ret = SearchDiskIndexIterative(p_headQuery, p_query, p_extraWorkspace);
        bool found = (ret == ErrorCode::Success);
        if (!found && ret != ErrorCode::VectorNotFound) return ret;
        p_extraWorkspace->m_loadPosting = false;
        if (!found)
        {
            p_headQueryResults->SetResultNum(m_options.m_headBatch);
            p_headQueryResults->Reset();
            continueSearch = m_index->SearchIndexIterativeFromNeareast(*p_headQueryResults, p_indexWorkspace, false);
            p_extraWorkspace->m_loadPosting = true;

            if (!oldRelaxedMono && p_extraWorkspace->m_relaxedMono)
                continueSearch = false;
        }
        else
            resultCount++;
    }
    p_queryResults->SortResult();

    if (p_query.WithMeta() && nullptr != m_pMetadata)
    {
        for (int i = 0; i < resultCount; ++i)
        {
            SizeType result = p_query.GetResult(i)->VID;
            p_query.SetMetadata(i, (result < 0) ? ByteArray::c_empty : m_pMetadata->GetMetadataCopy(result));
        }
    }
    return ErrorCode::Success;
}

template <typename T>
std::shared_ptr<ResultIterator> Index<T>::GetIterator(const void *p_target, bool p_searchDeleted, std::function<bool(const ByteArray&)> p_filterFunc, int p_maxCheck) const
{
    if (!m_bReady)
        return nullptr;
    auto extraWorkspace = m_workSpaceFactory->GetWorkSpace();
    if (!extraWorkspace)
    {
        extraWorkspace.reset(new ExtraWorkSpace());
        m_extraSearcher->InitWorkSpace(extraWorkspace.get(), false);
    }
    else
    {
        m_extraSearcher->InitWorkSpace(extraWorkspace.get(), true);
    }
    extraWorkspace->m_filterFunc = p_filterFunc;
    extraWorkspace->m_relaxedMono = false;
    extraWorkspace->m_loadedPostingNum = 0;
    extraWorkspace->m_deduper.clear();
    extraWorkspace->m_postingIDs.clear();
    std::shared_ptr<ResultIterator> resultIterator = std::make_shared<SPANNResultIterator<T>>(
        this, m_index.get(), p_target, std::move(extraWorkspace),
        max(m_options.m_headBatch, m_options.m_searchInternalResultNum), p_maxCheck);
    return resultIterator;
}

template <typename T>
ErrorCode Index<T>::SearchIndexIterativeNext(QueryResult &p_query, COMMON::WorkSpace *workSpace, int p_batch,
                                             int &resultCount, bool p_isFirst, bool p_searchDeleted) const
{
    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "ITERATIVE NOT SUPPORT FOR SPANN");
    return ErrorCode::Undefined;
}

template <typename T> ErrorCode Index<T>::SearchIndexIterativeEnd(std::unique_ptr<COMMON::WorkSpace> space) const
{
    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SearchIndexIterativeEnd NOT SUPPORT FOR SPANN");
    return ErrorCode::Fail;
}

template <typename T>
ErrorCode Index<T>::SearchIndexIterativeEnd(std::unique_ptr<SPANN::ExtraWorkSpace> extraWorkspace) const
{
    if (!m_bReady)
        return ErrorCode::EmptyIndex;

    if (extraWorkspace != nullptr)
        m_workSpaceFactory->ReturnWorkSpace(std::move(extraWorkspace));

    return ErrorCode::Success;
}

template <typename T>
bool Index<T>::SearchIndexIterativeFromNeareast(QueryResult &p_query, COMMON::WorkSpace *p_space, bool p_isFirst,
                                                bool p_searchDeleted) const
{
    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SearchIndexIterativeFromNeareast NOT SUPPORT FOR SPANN");
    return false;
}

template <typename T>
ErrorCode Index<T>::SearchIndexWithFilter(QueryResult &p_query, std::function<bool(const ByteArray &)> filterFunc,
                                          int maxCheck, bool p_searchDeleted) const
{
    if (nullptr == m_extraSearcher)
        return ErrorCode::EmptyIndex;

    // Result set for posting search: sized to topk only
    int topk = p_query.GetResultNum();
    // HeadIndex search needs more candidates
    int headSearchNum = max(topk, m_options.m_searchInternalResultNum);
    COMMON::QueryResultSet<T> headResults((const T*)p_query.GetQuantizedTarget(), headSearchNum);

    // Step 1: Search HeadIndex for posting routing
    m_index->SearchIndex(headResults);

    auto workSpace = m_workSpaceFactory->GetWorkSpace();
    if (!workSpace)
    {
        workSpace.reset(new ExtraWorkSpace());
        m_extraSearcher->InitWorkSpace(workSpace.get(), false);
    }
    else
    {
        m_extraSearcher->InitWorkSpace(workSpace.get(), true);
    }

    workSpace->m_deduper.clear();
    workSpace->m_postingIDs.clear();
    workSpace->m_filterFunc = filterFunc;
    workSpace->m_pFilterSource = this;

    // Collect posting IDs from head search results
    float limitDist = headResults.GetResult(0)->Dist * m_options.m_maxDistRatio;
    const int postingOffset = m_options.m_postingOffset;
    for (int i = 0; i < m_options.m_searchInternalResultNum; ++i)
    {
        auto res = headResults.GetResult(i);
        if (res->VID == -1 || (limitDist > 0.1 && res->Dist > limitDist))
            break;
        if (m_extraSearcher->CheckValidPosting(res->VID + postingOffset))
        {
            workSpace->m_postingIDs.emplace_back(res->VID + postingOffset);
        }
    }

    // Reuse headResults but clear it for posting search
    for (int j = 0; j < headResults.GetResultNum(); ++j)
    {
        headResults.SetResult(j, -1, MaxDist);
    }

    // Search postings with filter
    m_extraSearcher->SearchIndex(workSpace.get(), headResults, m_index, nullptr);
    m_workSpaceFactory->ReturnWorkSpace(std::move(workSpace));
    headResults.SortResult();

    // Copy results back to original query
    for (int j = 0; j < p_query.GetResultNum(); ++j)
    {
        if (j < headResults.GetResultNum())
        {
            auto src = headResults.GetResult(j);
            p_query.SetResult(j, src->VID, src->Dist);
        }
        else
        {
            p_query.SetResult(j, -1, MaxDist);
        }
    }
    return ErrorCode::Success;
}

template <typename T> ErrorCode Index<T>::SearchDiskIndex(QueryResult &p_query, SearchStats *p_stats) const
{
    if (nullptr == m_extraSearcher)
        return ErrorCode::EmptyIndex;

    COMMON::QueryResultSet<T> *p_queryResults = (COMMON::QueryResultSet<T> *)&p_query;

    auto workSpace = m_workSpaceFactory->GetWorkSpace();
    if (!workSpace)
    {
        workSpace.reset(new ExtraWorkSpace());
        m_extraSearcher->InitWorkSpace(workSpace.get(), false);
    }
    else
    {
        m_extraSearcher->InitWorkSpace(workSpace.get(), true);
    }

    workSpace->m_deduper.clear();
    workSpace->m_postingIDs.clear();

    float limitDist = p_queryResults->GetResult(0)->Dist * m_options.m_maxDistRatio;
    const int postingOffset = m_options.m_postingOffset;
    int i = 0;
    for (; i < m_options.m_searchInternalResultNum; ++i)
    {
        auto res = p_queryResults->GetResult(i);
        if (res->VID == -1 || (limitDist > 0.1 && res->Dist > limitDist))
            break;
        if (m_extraSearcher->CheckValidPosting(res->VID + postingOffset))
        {
            workSpace->m_postingIDs.emplace_back(res->VID + postingOffset);
        }

        if (m_vectorTranslateMap.R() != 0)
            res->VID = static_cast<SizeType>(*(m_vectorTranslateMap[res->VID]));
        else
        {
            res->VID = -1;
            res->Dist = MaxDist;
        }
        if (res->VID == MaxSize)
        {
            res->VID = -1;
            res->Dist = MaxDist;
        }
    }

    for (; i < p_queryResults->GetResultNum(); ++i)
    {
        auto res = p_queryResults->GetResult(i);
        if (res->VID == -1)
            break;
        if (m_vectorTranslateMap.R() != 0)
            res->VID = static_cast<SizeType>(*(m_vectorTranslateMap[res->VID]));
        else
        {
            res->VID = -1;
            res->Dist = MaxDist;
        }
        if (res->VID == MaxSize)
        {
            res->VID = -1;
            res->Dist = MaxDist;
        }
    }

    p_queryResults->Reverse();
    m_extraSearcher->SearchIndex(workSpace.get(), *p_queryResults, m_index, p_stats);
    m_workSpaceFactory->ReturnWorkSpace(std::move(workSpace));
    p_queryResults->SortResult();
    return ErrorCode::Success;
}

template <typename T>
ErrorCode Index<T>::SearchDiskIndexIterative(QueryResult &p_headQuery, QueryResult &p_query,
                                        ExtraWorkSpace *extraWorkspace) const
{
    if (extraWorkspace->m_loadPosting)
    {
        COMMON::QueryResultSet<T> *p__headQueryResults = (COMMON::QueryResultSet<T> *)&p_headQuery;
        // std::shared_ptr<ExtraWorkSpace> workSpace = m_workSpacePool->Rent();
        // workSpace->m_deduper.clear();
        extraWorkspace->m_postingIDs.clear();

        // float limitDist = p_queryResults->GetResult(0)->Dist * m_options.m_maxDistRatio;
        const int postingOffset = m_options.m_postingOffset;

        for (int i = 0; i < p__headQueryResults->GetResultNum(); ++i)
        {
            auto res = p__headQueryResults->GetResult(i);
            // break or continue
            if (res->VID == -1)
                break;
            if (m_extraSearcher->CheckValidPosting(res->VID + postingOffset))
            {
                extraWorkspace->m_postingIDs.emplace_back(res->VID + postingOffset);
            }

            if (m_vectorTranslateMap.R() != 0)
                res->VID = static_cast<SizeType>(*(m_vectorTranslateMap[res->VID]));
            else
            {
                res->VID = -1;
                res->Dist = MaxDist;
            }
            if (res->VID == MaxSize)
            {
                res->VID = -1;
                res->Dist = MaxDist;
            }
        }
        extraWorkspace->m_loadedPostingNum += (int)(extraWorkspace->m_postingIDs.size());
    }

    ErrorCode ret = m_extraSearcher->SearchIterativeNext(extraWorkspace, p_headQuery, p_query, m_index, this);
    if (ret == ErrorCode::VectorNotFound && extraWorkspace->m_loadedPostingNum >= m_options.m_searchInternalResultNum)
        extraWorkspace->m_relaxedMono = true;
    return ret;
}

template <typename T> std::unique_ptr<COMMON::WorkSpace> Index<T>::RentWorkSpace(int batch, std::function<bool(const ByteArray&)> p_filterFunc, int p_maxCheck) const
{
    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "RentWorkSpace NOT SUPPORT FOR SPANN");
    return nullptr;
}

template <typename T>
ErrorCode Index<T>::DebugSearchDiskIndex(QueryResult &p_query, int p_subInternalResultNum, int p_internalResultNum,
                                         SearchStats *p_stats, std::set<int> *truth,
                                         std::map<int, std::set<int>> *found) const
{
    if (nullptr == m_extraSearcher)
        return ErrorCode::EmptyIndex;

    COMMON::QueryResultSet<T> newResults(*((COMMON::QueryResultSet<T> *)&p_query));
    for (int i = 0; i < newResults.GetResultNum(); ++i)
    {
        auto res = newResults.GetResult(i);
        if (res->VID == -1)
            break;

        auto global_VID = -1;
        if (m_vectorTranslateMap.R() != 0)
            global_VID = static_cast<SizeType>(*(m_vectorTranslateMap[res->VID]));
        if (truth && truth->count(global_VID))
            (*found)[res->VID].insert(global_VID);
        res->VID = global_VID;
    }
    newResults.Reverse();

    auto workSpace = m_workSpaceFactory->GetWorkSpace();
    if (!workSpace)
    {
        workSpace.reset(new ExtraWorkSpace());
        m_extraSearcher->InitWorkSpace(workSpace.get(), false);
    }
    else
    {
        m_extraSearcher->InitWorkSpace(workSpace.get(), true);
    }
    workSpace->m_deduper.clear();

    int partitions = (p_internalResultNum + p_subInternalResultNum - 1) / p_subInternalResultNum;
    float limitDist = p_query.GetResult(0)->Dist * m_options.m_maxDistRatio;
    for (SizeType p = 0; p < partitions; p++)
    {
        int subInternalResultNum = min(p_subInternalResultNum, p_internalResultNum - p_subInternalResultNum * p);

        workSpace->m_postingIDs.clear();

        for (int i = p * p_subInternalResultNum; i < p * p_subInternalResultNum + subInternalResultNum; i++)
        {
            auto res = p_query.GetResult(i);
            if (res->VID == -1 || (limitDist > 0.1 && res->Dist > limitDist))
                break;
            if (!m_extraSearcher->CheckValidPosting(res->VID))
                continue;
            workSpace->m_postingIDs.emplace_back(res->VID);
        }

        m_extraSearcher->SearchIndex(workSpace.get(), newResults, m_index, p_stats, truth, found);
    }
    m_workSpaceFactory->ReturnWorkSpace(std::move(workSpace));
    newResults.SortResult();
    std::copy(newResults.GetResults(), newResults.GetResults() + newResults.GetResultNum(), p_query.GetResults());
    return ErrorCode::Success;
}
#pragma endregion

template <typename T>
ErrorCode Index<T>::GetPostingDebug(SizeType vid, std::vector<SizeType> &VIDs, std::shared_ptr<VectorSet> &vecs)
{
    VIDs.clear();
    if (!m_extraSearcher)
        return ErrorCode::EmptyIndex;
    if (!m_extraSearcher->CheckValidPosting(vid))
        return ErrorCode::Fail;

    auto workSpace = m_workSpaceFactory->GetWorkSpace();
    if (!workSpace)
    {
        workSpace.reset(new ExtraWorkSpace());
        m_extraSearcher->InitWorkSpace(workSpace.get(), false);
    }
    else
    {
        m_extraSearcher->InitWorkSpace(workSpace.get(), true);
    }
    workSpace->m_deduper.clear();

    auto out = m_extraSearcher->GetPostingDebug(workSpace.get(), m_index, vid, VIDs, vecs);
    m_workSpaceFactory->ReturnWorkSpace(std::move(workSpace));
    return out;
}

template <typename T> void Index<T>::SelectHeadAdjustOptions(int p_vectorCount)
{
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Begin Adjust Parameters...\n");

    if (m_options.m_headVectorCount != 0)
        m_options.m_ratio = m_options.m_headVectorCount * 1.0 / p_vectorCount;
    int headCnt = static_cast<int>(std::round(m_options.m_ratio * p_vectorCount));
    if (headCnt == 0)
    {
        for (double minCnt = 1; headCnt == 0; minCnt += 0.2)
        {
            m_options.m_ratio = minCnt / p_vectorCount;
            headCnt = static_cast<int>(std::round(m_options.m_ratio * p_vectorCount));
        }

        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                     "Setting requires to select none vectors as head, adjusted it to %d vectors\n", headCnt);
    }

    if (m_options.m_iBKTKmeansK > headCnt)
    {
        m_options.m_iBKTKmeansK = headCnt;
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Setting of cluster number is less than head count, adjust it to %d\n",
                     headCnt);
    }

    if (m_options.m_selectThreshold == 0)
    {
        m_options.m_selectThreshold = min(p_vectorCount - 1, static_cast<int>(1 / m_options.m_ratio));
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Set SelectThreshold to %d\n", m_options.m_selectThreshold);
    }

    if (m_options.m_splitThreshold == 0)
    {
        m_options.m_splitThreshold = min(p_vectorCount - 1, static_cast<int>(m_options.m_selectThreshold * 2));
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Set SplitThreshold to %d\n", m_options.m_splitThreshold);
    }

    if (m_options.m_splitFactor == 0)
    {
        m_options.m_splitFactor = min(p_vectorCount - 1, static_cast<int>(std::round(1 / m_options.m_ratio) + 0.5));
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Set SplitFactor to %d\n", m_options.m_splitFactor);
    }
}

template <typename T>
int Index<T>::SelectHeadDynamicallyInternal(const std::shared_ptr<COMMON::BKTree> p_tree, int p_nodeID,
                                            const Options &p_opts, std::vector<int> &p_selected)
{
    typedef std::pair<int, int> CSPair;
    std::vector<CSPair> children;
    int childrenSize = 1;
    const auto &node = (*p_tree)[p_nodeID];
    if (node.childStart >= 0)
    {
        children.reserve(node.childEnd - node.childStart);
        for (int i = node.childStart; i < node.childEnd; ++i)
        {
            int cs = SelectHeadDynamicallyInternal(p_tree, i, p_opts, p_selected);
            if (cs > 0)
            {
                children.emplace_back(i, cs);
                childrenSize += cs;
            }
        }
    }

    if (childrenSize >= p_opts.m_selectThreshold)
    {
        if (node.centerid < (*p_tree)[0].centerid)
        {
            p_selected.push_back(node.centerid);
        }

        if (childrenSize > p_opts.m_splitThreshold)
        {
            std::sort(children.begin(), children.end(),
                      [](const CSPair &a, const CSPair &b) { return a.second > b.second; });

            size_t selectCnt = static_cast<size_t>(std::ceil(childrenSize * 1.0 / p_opts.m_splitFactor) + 0.5);
            // if (selectCnt > 1) selectCnt -= 1;
            for (size_t i = 0; i < selectCnt && i < children.size(); ++i)
            {
                p_selected.push_back((*p_tree)[children[i].first].centerid);
            }
        }

        return 0;
    }

    return childrenSize;
}

template <typename T>
void Index<T>::SelectHeadDynamically(const std::shared_ptr<COMMON::BKTree> p_tree, int p_vectorCount,
                                     std::vector<int> &p_selected)
{
    p_selected.clear();
    p_selected.reserve(p_vectorCount);

    if (static_cast<int>(std::round(m_options.m_ratio * p_vectorCount)) >= p_vectorCount)
    {
        for (int i = 0; i < p_vectorCount; ++i)
        {
            p_selected.push_back(i);
        }

        return;
    }
    Options opts = m_options;

    int selectThreshold = m_options.m_selectThreshold;
    int splitThreshold = m_options.m_splitThreshold;

    double minDiff = 100;
    for (int select = 2; select <= m_options.m_selectThreshold; ++select)
    {
        opts.m_selectThreshold = select;
        opts.m_splitThreshold = m_options.m_splitThreshold;

        int l = m_options.m_splitFactor;
        int r = m_options.m_splitThreshold;

        while (l < r - 1)
        {
            opts.m_splitThreshold = (l + r) / 2;
            p_selected.clear();

            SelectHeadDynamicallyInternal(p_tree, 0, opts, p_selected);
            std::sort(p_selected.begin(), p_selected.end());
            p_selected.erase(std::unique(p_selected.begin(), p_selected.end()), p_selected.end());

            double diff = static_cast<double>(p_selected.size()) / p_vectorCount - m_options.m_ratio;

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Select Threshold: %d, Split Threshold: %d, diff: %.2lf%%.\n",
                         opts.m_selectThreshold, opts.m_splitThreshold, diff * 100.0);

            if (minDiff > fabs(diff))
            {
                minDiff = fabs(diff);

                selectThreshold = opts.m_selectThreshold;
                splitThreshold = opts.m_splitThreshold;
            }

            if (diff > 0)
            {
                l = (l + r) / 2;
            }
            else
            {
                r = (l + r) / 2;
            }
        }
    }

    opts.m_selectThreshold = selectThreshold;
    opts.m_splitThreshold = splitThreshold;

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Final Select Threshold: %d, Split Threshold: %d.\n",
                 opts.m_selectThreshold, opts.m_splitThreshold);

    p_selected.clear();
    SelectHeadDynamicallyInternal(p_tree, 0, opts, p_selected);
    std::sort(p_selected.begin(), p_selected.end());
    p_selected.erase(std::unique(p_selected.begin(), p_selected.end()), p_selected.end());
}

template <typename T>
template <typename InternalDataType>
bool Index<T>::SelectHeadInternal(std::shared_ptr<Helper::VectorSetReader> &p_reader)
{
    std::shared_ptr<VectorSet> vectorset = p_reader->GetVectorSet();
    if (m_options.m_distCalcMethod == DistCalcMethod::Cosine && !p_reader->IsNormalized())
        vectorset->Normalize(m_options.m_iSelectHeadNumberOfThreads);
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Begin initial data (%d,%d)...\n", vectorset->Count(),
                 vectorset->Dimension());

    COMMON::Dataset<InternalDataType> data(vectorset->Count(), vectorset->Dimension(), vectorset->Count(),
                                           vectorset->Count() + 1, (InternalDataType *)vectorset->GetData());

    // Allow runtime override of m_selectType via env var (used by experimental
    // selectType variants like PerTagBKTMerge that aren't surfaced in config yet).
    if (const char* selOverride = std::getenv("SPTAG_SELECT_TYPE_OVERRIDE"))
    {
        if (selOverride[0] != '\0' &&
            !Helper::StrUtils::StrEqualIgnoreCase(m_options.m_selectType.c_str(), selOverride))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                         "SPTAG_SELECT_TYPE_OVERRIDE: overriding selectType '%s' -> '%s'\n",
                         m_options.m_selectType.c_str(), selOverride);
            m_options.m_selectType = selOverride;
        }
    }

    if (const char* ratioOverride = std::getenv("SPTAG_RATIO_OVERRIDE"))
    {
        if (ratioOverride[0] != '\0')
        {
            try {
                double r = std::stod(ratioOverride);
                if (r > 0.0 && r < 1.0) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                 "SPTAG_RATIO_OVERRIDE: overriding m_ratio %.6f -> %.6f\n",
                                 m_options.m_ratio, r);
                    m_options.m_ratio = r;
                }
            } catch (...) {}
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    SelectHeadAdjustOptions(data.R());
    std::vector<int> selected;
    if (data.R() == 1)
    {
        selected.push_back(0);
    }
    else if (Helper::StrUtils::StrEqualIgnoreCase(m_options.m_selectType.c_str(), "Random"))
    {
        std::mt19937 rg;
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start generating Random head.\n");
        selected.resize(data.R());
        for (int i = 0; i < data.R(); i++)
            selected[i] = i;
        std::shuffle(selected.begin(), selected.end(), rg);
        int headCnt = static_cast<int>(std::round(m_options.m_ratio * data.R()));
        selected.resize(headCnt);
    }
    else if (Helper::StrUtils::StrEqualIgnoreCase(m_options.m_selectType.c_str(), "BKT"))
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start generating BKT.\n");
        std::shared_ptr<COMMON::BKTree> bkt = std::make_shared<COMMON::BKTree>();
        bkt->m_iBKTKmeansK = m_options.m_iBKTKmeansK;
        bkt->m_iBKTLeafSize = m_options.m_iBKTLeafSize;
        bkt->m_iSamples = m_options.m_iSamples;
        bkt->m_iTreeNumber = m_options.m_iTreeNumber;
        bkt->m_fBalanceFactor = m_options.m_fBalanceFactor;
        bkt->m_pQuantizer = m_pQuantizer;
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start invoking BuildTrees.\n");
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Info,
            "BKTKmeansK: %d, BKTLeafSize: %d, Samples: %d, BKTLambdaFactor:%f TreeNumber: %d, ThreadNum: %d.\n",
            bkt->m_iBKTKmeansK, bkt->m_iBKTLeafSize, bkt->m_iSamples, bkt->m_fBalanceFactor, bkt->m_iTreeNumber,
            m_options.m_iSelectHeadNumberOfThreads);

        bkt->BuildTrees<InternalDataType>(data, m_options.m_distCalcMethod, m_options.m_iSelectHeadNumberOfThreads,
                                          nullptr, nullptr, true);
        auto t2 = std::chrono::high_resolution_clock::now();
        double elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "End invoking BuildTrees.\n");
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Invoking BuildTrees used time: %.2lf minutes (about %.2lf hours).\n",
                     elapsedSeconds / 60.0, elapsedSeconds / 3600.0);

        if (m_options.m_saveBKT)
        {
            std::stringstream bktFileNameBuilder;
            bktFileNameBuilder << m_options.m_vectorPath << ".bkt." << m_options.m_iBKTKmeansK << "_"
                               << m_options.m_iBKTLeafSize << "_" << m_options.m_iTreeNumber << "_"
                               << m_options.m_iSamples << "_" << static_cast<int>(m_options.m_distCalcMethod) << ".bin";
            bkt->SaveTrees(bktFileNameBuilder.str());
        }
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Finish generating BKT.\n");

        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start selecting nodes...Select Head Dynamically...\n");
        SelectHeadDynamically(bkt, data.R(), selected);

        if (selected.empty())
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Can't select any vector as head with current settings\n");
            return false;
        }
    }
    else if (Helper::StrUtils::StrEqualIgnoreCase(m_options.m_selectType.c_str(), "PerTagBKTMerge"))
    {
        // Boss-proposed scheme:
        //   Phase 1 — partition vectors by tag value (e.g. team-level, 64 groups).
        //             For each group: build a BKTree on that subset and run
        //             SelectHeadDynamically with the head ratio bumped by an
        //             oversample factor (default 3x). Concatenate all per-tag
        //             heads into a single candidate set.
        //   Phase 2 — greedy 3-way merge: each candidate looks for its k nearest
        //             unmerged neighbours within (same tag) ∪ (top-K cross-tag
        //             neighbours by tag centroid). Merge if pairwise distances
        //             are within alpha * mean(1-NN spacing) and the implied
        //             posting size cap is respected.
        //   Phase 3 — for each merged group, elect the candidate closest to the
        //             group geometric mean as the representative head. Singletons
        //             survive as-is.
        // Inputs:
        //   env SPTAG_PER_VECTOR_TAGS_FILE       : text file, one int per line
        //                                          (length must equal data.R())
        //   env SPTAG_PARTITION_OVERSAMPLE       : default 3.0
        //   env SPTAG_MERGE_ALPHA                : default 0.2
        //   env SPTAG_CROSS_TAG_NEIGHBORS        : default 3
        //   env SPTAG_MERGE_GROUP_SIZE           : default 3
        //
        const char* tagsFile = std::getenv("SPTAG_PER_VECTOR_TAGS_FILE");
        if (tagsFile == nullptr)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "PerTagBKTMerge requires env SPTAG_PER_VECTOR_TAGS_FILE\n");
            return false;
        }
        float oversample = 3.0f;
        if (const char* e = std::getenv("SPTAG_PARTITION_OVERSAMPLE"))
            oversample = std::max(1.0f, static_cast<float>(std::atof(e)));
        float mergeAlpha = 1.0f;
        if (const char* e = std::getenv("SPTAG_MERGE_ALPHA"))
            mergeAlpha = std::max(0.0f, static_cast<float>(std::atof(e)));
        int crossTagNeighbors = 3;
        if (const char* e = std::getenv("SPTAG_CROSS_TAG_NEIGHBORS"))
            crossTagNeighbors = std::max(0, std::atoi(e));
        int mergeGroupSize = 3;
        if (const char* e = std::getenv("SPTAG_MERGE_GROUP_SIZE"))
            mergeGroupSize = std::max(2, std::atoi(e));
        // Target FINAL head ratio (after merge). Per-tag SelectHead aims at
        // `oversample × finalRatio`. Defaults to 0.016 ≈ SIFT-1M tenant-0
        // baseline. Override per dataset.
        double finalRatio = 0.016;
        if (const char* e = std::getenv("SPTAG_PERTAG_HEAD_RATIO"))
            finalRatio = std::max(1e-5, std::atof(e));
        // Cap on the sum of per-head catchments inside a merge group, expressed
        // as a multiplier of the mean catchment. Default 1.2 ⇒ no group's
        // resulting posting may exceed 1.2 × mergeGroupSize × meanCatchment
        // (i.e. ~1.2× of the uniform "fair share" if the group fills up).
        // Set to 0 (or a huge number) to disable the cap.
        double sizeCapMultiplier = 1.2;
        if (const char* e = std::getenv("SPTAG_PERTAG_SIZE_CAP_MULT"))
            sizeCapMultiplier = std::max(0.0, std::atof(e));

        // ---- Read per-vector tag column (one int per line) ----
        std::vector<int> perVecTag(data.R(), -1);
        {
            std::ifstream fin(tagsFile);
            if (!fin.good())
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                             "PerTagBKTMerge failed to open %s\n", tagsFile);
                return false;
            }
            int v; int idx = 0;
            while (idx < data.R() && (fin >> v))
                perVecTag[idx++] = v;
            if (idx != data.R())
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                             "PerTagBKTMerge tag file has %d entries, expected %d\n",
                             idx, data.R());
                return false;
            }
        }

        // ---- Group vector IDs by tag value ----
        std::map<int, std::vector<SizeType>> tagGroups;
        for (int i = 0; i < data.R(); ++i)
            if (perVecTag[i] >= 0)
                tagGroups[perVecTag[i]].push_back(static_cast<SizeType>(i));
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                     "PerTagBKTMerge: %zu distinct tag values "
                     "(oversample=%.2f, alpha=%.3f, cross-tag-knn=%d, group=%d)\n",
                     tagGroups.size(), oversample, mergeAlpha, crossTagNeighbors, mergeGroupSize);

        // ---- Phase 1: per-tag BKT + SelectHeadDynamically (oversampled) ----
        // Target ratio for SelectHeadDynamically per tag = oversample × finalRatio
        // so that after 3-way merge we land roughly at finalRatio.
        const double perTagTarget = std::min(0.9, oversample * finalRatio);
        const auto savedRatio    = m_options.m_ratio;
        const auto savedSelTh    = m_options.m_selectThreshold;
        const auto savedSplTh    = m_options.m_splitThreshold;
        const auto savedSplFa    = m_options.m_splitFactor;
        const auto savedKmeansK  = m_options.m_iBKTKmeansK;
        const auto savedSamples  = m_options.m_iSamples;

        std::vector<int> initialHeads;     // global VIDs
        std::vector<int> initialHeadTag;   // tag value of each initial head

        for (auto& kv : tagGroups)
        {
            int tagVal = kv.first;
            std::vector<SizeType>& subIdx = kv.second;
            int subSize = static_cast<int>(subIdx.size());
            if (subSize <= 1)
            {
                if (subSize == 1) {
                    initialHeads.push_back(static_cast<int>(subIdx[0]));
                    initialHeadTag.push_back(tagVal);
                }
                continue;
            }

            // Reset & adjust options for this subset.
            // Set target = oversample × finalRatio. Reset thresholds to 0 so
            // SelectHeadAdjustOptions derives them from the new ratio.
            m_options.m_ratio = perTagTarget;
            m_options.m_selectThreshold = 0;
            m_options.m_splitThreshold = 0;
            m_options.m_splitFactor = 0;
            m_options.m_iBKTKmeansK = savedKmeansK;
            m_options.m_iSamples = std::min(savedSamples, subSize);
            SelectHeadAdjustOptions(subSize);

            std::shared_ptr<COMMON::BKTree> bkt = std::make_shared<COMMON::BKTree>();
            bkt->m_iBKTKmeansK   = m_options.m_iBKTKmeansK;
            bkt->m_iBKTLeafSize  = m_options.m_iBKTLeafSize;
            bkt->m_iSamples      = std::min(m_options.m_iSamples, subSize);
            bkt->m_iTreeNumber   = m_options.m_iTreeNumber;
            bkt->m_fBalanceFactor = m_options.m_fBalanceFactor;
            bkt->m_pQuantizer    = m_pQuantizer;
            bkt->BuildTrees<InternalDataType>(data, m_options.m_distCalcMethod,
                                              m_options.m_iSelectHeadNumberOfThreads,
                                              &subIdx, nullptr, true);

            std::vector<int> subSelected;
            SelectHeadDynamically(bkt, subSize, subSelected);
            if (subSelected.empty())
                subSelected.push_back(static_cast<int>(subIdx[0]));

            for (int h : subSelected)
            {
                initialHeads.push_back(h);
                initialHeadTag.push_back(tagVal);
            }
        }

        // restore options for downstream Build phases
        m_options.m_ratio = savedRatio;
        m_options.m_selectThreshold = savedSelTh;
        m_options.m_splitThreshold = savedSplTh;
        m_options.m_splitFactor = savedSplFa;
        m_options.m_iBKTKmeansK = savedKmeansK;
        m_options.m_iSamples = savedSamples;

        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                     "PerTagBKTMerge Phase 1: %zu initial heads "
                     "(finalRatio=%.3f%%, perTagTarget=%.3f%%, achieved=%.3f%%)\n",
                     initialHeads.size(),
                     100.0 * finalRatio,
                     100.0 * perTagTarget,
                     100.0 * initialHeads.size() / data.R());

        // ---- Phase 2 prep: tag centroids + per-tag head lists ----
        const int dim = data.C();
        std::map<int, std::vector<int>> tagToHeadIdx;  // tag -> indices into initialHeads[]
        for (size_t i = 0; i < initialHeads.size(); ++i)
            tagToHeadIdx[initialHeadTag[i]].push_back(static_cast<int>(i));

        // Per-tag centroid in float (float regardless of InternalDataType)
        std::map<int, std::vector<float>> tagCentroid;
        for (auto& kv : tagGroups)
        {
            std::vector<float> c(dim, 0.0f);
            for (SizeType vid : kv.second)
            {
                const InternalDataType* p = data[vid];
                for (int d = 0; d < dim; ++d) c[d] += static_cast<float>(p[d]);
            }
            float inv = 1.0f / std::max<size_t>(1, kv.second.size());
            for (int d = 0; d < dim; ++d) c[d] *= inv;
            tagCentroid[kv.first] = std::move(c);
        }
        // For each tag, list its top-K nearest other tags by centroid L2
        std::map<int, std::vector<int>> tagNearest;
        {
            std::vector<int> tagList;
            tagList.reserve(tagCentroid.size());
            for (auto& kv : tagCentroid) tagList.push_back(kv.first);
            for (int t : tagList)
            {
                std::vector<std::pair<float, int>> dlist;
                dlist.reserve(tagList.size());
                const auto& ct = tagCentroid[t];
                for (int u : tagList)
                {
                    if (u == t) continue;
                    const auto& cu = tagCentroid[u];
                    float d = 0;
                    for (int k = 0; k < dim; ++k) {
                        float diff = ct[k] - cu[k];
                        d += diff * diff;
                    }
                    dlist.emplace_back(d, u);
                }
                int k = std::min<int>(crossTagNeighbors, static_cast<int>(dlist.size()));
                std::partial_sort(dlist.begin(), dlist.begin() + k, dlist.end());
                std::vector<int> nb;
                nb.reserve(k);
                for (int i = 0; i < k; ++i) nb.push_back(dlist[i].second);
                tagNearest[t] = std::move(nb);
            }
        }

        auto headDist = [&](int hi, int hj) -> float {
            const InternalDataType* a = data[initialHeads[hi]];
            const InternalDataType* b = data[initialHeads[hj]];
            float d = 0;
            for (int k = 0; k < dim; ++k) {
                float diff = static_cast<float>(a[k]) - static_cast<float>(b[k]);
                d += diff * diff;
            }
            return d;  // squared L2, fine for comparisons
        };

        // ---- Compute mean 1-NN distance among initialHeads (sampled) ----
        // For each sample head, scan its candidate set (same tag + nearest tags)
        // and find the min distance to any other head. Mean of these = baseline.
        float meanNN1 = 0.0f;
        {
            std::mt19937 rg(12345);
            int n = static_cast<int>(initialHeads.size());
            int sampleCount = std::min(n, 4096);
            std::vector<int> samp(n);
            for (int i = 0; i < n; ++i) samp[i] = i;
            std::shuffle(samp.begin(), samp.end(), rg);
            samp.resize(sampleCount);
            double sum = 0;
            int cnt = 0;
            for (int hi : samp)
            {
                int t = initialHeadTag[hi];
                std::vector<int> candTags = { t };
                for (int u : tagNearest[t]) candTags.push_back(u);
                float best = std::numeric_limits<float>::infinity();
                for (int ct : candTags) {
                    for (int hj : tagToHeadIdx[ct]) {
                        if (hj == hi) continue;
                        float d = headDist(hi, hj);
                        if (d < best) best = d;
                    }
                }
                if (std::isfinite(best)) {
                    sum += std::sqrt(best);
                    ++cnt;
                }
            }
            meanNN1 = (cnt > 0) ? static_cast<float>(sum / cnt) : 1.0f;
        }
        const float mergeThresholdSq = (mergeAlpha * meanNN1) * (mergeAlpha * meanNN1);
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                     "PerTagBKTMerge Phase 2: meanNN1=%.4f, threshold=%.4f (alpha=%.3f, sq=%.4f)\n",
                     meanNN1, mergeAlpha * meanNN1, mergeAlpha, mergeThresholdSq);

        // ---- Estimate per-head catchment (# base vecs whose nearest head is i) ----
        // Brute force within each tag partition: vecs in tag T snap to nearest
        // head in {heads(T) ∪ heads(tagNearest[T])}. This is identical to the
        // candidate set used during query routing and is what determines the
        // post-merge posting size.
        std::vector<int> catchment(initialHeads.size(), 0);
        {
            auto vecHeadDistSq = [&](SizeType vid, int hidx) -> float {
                const InternalDataType* a = data[vid];
                const InternalDataType* b = data[initialHeads[hidx]];
                float d = 0;
                for (int k = 0; k < dim; ++k) {
                    float diff = static_cast<float>(a[k]) - static_cast<float>(b[k]);
                    d += diff * diff;
                }
                return d;
            };
            for (auto& kv : tagGroups)
            {
                int tagVal = kv.first;
                std::vector<int> candHeads;
                auto it = tagToHeadIdx.find(tagVal);
                if (it != tagToHeadIdx.end())
                    candHeads.insert(candHeads.end(), it->second.begin(), it->second.end());
                auto itN = tagNearest.find(tagVal);
                if (itN != tagNearest.end()) {
                    for (int u : itN->second) {
                        auto itu = tagToHeadIdx.find(u);
                        if (itu != tagToHeadIdx.end())
                            candHeads.insert(candHeads.end(), itu->second.begin(), itu->second.end());
                    }
                }
                if (candHeads.empty()) continue;
                for (SizeType vid : kv.second) {
                    float best = std::numeric_limits<float>::infinity();
                    int bestH = -1;
                    for (int hidx : candHeads) {
                        float d = vecHeadDistSq(vid, hidx);
                        if (d < best) { best = d; bestH = hidx; }
                    }
                    if (bestH >= 0) {
                        catchment[bestH]++;
                    }
                }
            }
            double sum = 0; int nonzero = 0; int mx = 0;
            for (int c : catchment) { sum += c; if (c > 0) ++nonzero; if (c > mx) mx = c; }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                         "PerTagBKTMerge catchment: total=%.0f, heads=%zu, mean=%.2f, nonzero=%d, max=%d\n",
                         sum, initialHeads.size(),
                         sum / std::max<size_t>(1, initialHeads.size()), nonzero, mx);
        }
        const double meanCatchment =
            static_cast<double>(data.R()) / std::max<size_t>(1, initialHeads.size());
        // Per-group catchment-sum cap. If sizeCapMultiplier == 0, treat as disabled (∞).
        const double groupCatchmentCap = (sizeCapMultiplier > 0)
            ? sizeCapMultiplier * mergeGroupSize * meanCatchment
            : std::numeric_limits<double>::infinity();
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                     "PerTagBKTMerge merge cap: meanCatchment=%.2f, groupCap=%.2f (mult=%.2f × groupSize=%d)\n",
                     meanCatchment, groupCatchmentCap, sizeCapMultiplier, mergeGroupSize);

        // ---- Greedy K-way merge ----
        int n = static_cast<int>(initialHeads.size());
        std::vector<int> mergedGroup(n, -1);  // -1 = unmerged, else group ID
        std::vector<std::vector<int>> groups; // each = head indices in initialHeads[]
        std::vector<int> order(n);
        for (int i = 0; i < n; ++i) order[i] = i;
        std::mt19937 rg(54321);
        std::shuffle(order.begin(), order.end(), rg);

        size_t mergedHeadCount = 0;
        size_t capBlocked = 0;
        for (int hi : order)
        {
            if (mergedGroup[hi] != -1) continue;
            int t = initialHeadTag[hi];

            // Build candidate set: heads in same tag or nearest tags, within
            // mergeThresholdSq distance, not already merged.
            std::vector<int> candTags = { t };
            for (int u : tagNearest[t]) candTags.push_back(u);
            std::vector<std::pair<float, int>> cands;
            for (int ct : candTags) {
                for (int hj : tagToHeadIdx[ct]) {
                    if (hj == hi || mergedGroup[hj] != -1) continue;
                    float d = headDist(hi, hj);
                    if (d <= mergeThresholdSq)
                        cands.emplace_back(d, hj);
                }
            }
            // Sort by distance ascending; pick nearest, but skip any that would
            // push the group's catchment sum beyond groupCatchmentCap.
            std::sort(cands.begin(), cands.end());
            std::vector<int> picked = { hi };
            double pickedCatchment = static_cast<double>(catchment[hi]);
            for (auto& c : cands) {
                if (static_cast<int>(picked.size()) >= mergeGroupSize) break;
                double newSum = pickedCatchment + static_cast<double>(catchment[c.second]);
                if (newSum > groupCatchmentCap) { ++capBlocked; continue; }
                picked.push_back(c.second);
                pickedCatchment = newSum;
            }
            int gid = static_cast<int>(groups.size());
            for (int idx : picked) mergedGroup[idx] = gid;
            groups.push_back(picked);
            if (picked.size() > 1) mergedHeadCount += picked.size();
        }
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                     "PerTagBKTMerge merge done: %zu groups, %zu heads merged into groups, %zu candidates blocked by cap\n",
                     groups.size(), mergedHeadCount, capBlocked);

        // ---- Phase 3: re-elect representative head per group ----
        std::vector<int> finalHeads;
        finalHeads.reserve(groups.size());
        for (auto& g : groups)
        {
            if (g.size() == 1) {
                finalHeads.push_back(initialHeads[g[0]]);
                continue;
            }
            // geometric mean (float) of group members
            std::vector<float> mean(dim, 0.0f);
            for (int idx : g) {
                const InternalDataType* p = data[initialHeads[idx]];
                for (int d = 0; d < dim; ++d) mean[d] += static_cast<float>(p[d]);
            }
            float inv = 1.0f / static_cast<float>(g.size());
            for (int d = 0; d < dim; ++d) mean[d] *= inv;
            // pick member nearest to mean
            int best = g[0];
            float bestD = std::numeric_limits<float>::infinity();
            for (int idx : g) {
                const InternalDataType* p = data[initialHeads[idx]];
                float d = 0;
                for (int k = 0; k < dim; ++k) {
                    float diff = mean[k] - static_cast<float>(p[k]);
                    d += diff * diff;
                }
                if (d < bestD) { bestD = d; best = idx; }
            }
            finalHeads.push_back(initialHeads[best]);
        }

        // dedup + sort
        std::sort(finalHeads.begin(), finalHeads.end());
        finalHeads.erase(std::unique(finalHeads.begin(), finalHeads.end()), finalHeads.end());
        selected.swap(finalHeads);

        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                     "PerTagBKTMerge Phase 3: %zu groups -> %zu final heads "
                     "(%.3f%% of data, merged_in_groups=%zu)\n",
                     groups.size(), selected.size(),
                     100.0 * selected.size() / data.R(), mergedHeadCount);

        if (selected.empty()) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "PerTagBKTMerge produced no heads\n");
            return false;
        }
    }

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Seleted Nodes: %u, about %.2lf%% of total.\n",
                 static_cast<unsigned int>(selected.size()), selected.size() * 100.0 / data.R());

    m_pendingNodeHeadSelections.clear();
    m_pendingHeadVectorOwners.clear();
    if (!m_pendingPrimaryNodeVectorAssignments.empty())
    {
        std::vector<int> primaryOwner(data.R(), -1);
        m_pendingNodeHeadSelections.assign(m_pendingPrimaryNodeVectorAssignments.size(), std::vector<SizeType>());
        for (size_t nodeId = 0; nodeId < m_pendingPrimaryNodeVectorAssignments.size(); ++nodeId)
        {
            for (SizeType vectorId : m_pendingPrimaryNodeVectorAssignments[nodeId])
            {
                if (vectorId >= 0 && vectorId < data.R()) {
                    primaryOwner[vectorId] = static_cast<int>(nodeId);
                }
            }
        }

        std::unordered_set<SizeType> selectedSet;
        selectedSet.reserve(selected.size() * 2 + 1);
        for (int vectorId : selected)
        {
            if (vectorId < 0 || vectorId >= data.R()) {
                continue;
            }

            selectedSet.insert(static_cast<SizeType>(vectorId));
            int ownerNode = primaryOwner[vectorId];
            if (ownerNode >= 0 && ownerNode < static_cast<int>(m_pendingNodeHeadSelections.size()))
            {
                m_pendingNodeHeadSelections[static_cast<size_t>(ownerNode)].push_back(static_cast<SizeType>(vectorId));
                m_pendingHeadVectorOwners[static_cast<SizeType>(vectorId)] = ownerNode;
            }
        }

        for (size_t nodeId = 0; nodeId < m_pendingPrimaryNodeVectorAssignments.size(); ++nodeId)
        {
            if (!m_pendingNodeHeadSelections[nodeId].empty() || m_pendingPrimaryNodeVectorAssignments[nodeId].empty()) {
                continue;
            }

            SizeType chosenVector = m_pendingPrimaryNodeVectorAssignments[nodeId].front();
            for (SizeType candidateVector : m_pendingPrimaryNodeVectorAssignments[nodeId])
            {
                if (selectedSet.count(candidateVector) == 0)
                {
                    chosenVector = candidateVector;
                    break;
                }
            }

            if (selectedSet.insert(chosenVector).second) {
                selected.push_back(static_cast<int>(chosenVector));
            }
            m_pendingNodeHeadSelections[nodeId].push_back(chosenVector);
            m_pendingHeadVectorOwners[chosenVector] = static_cast<int>(nodeId);
        }

        std::sort(selected.begin(), selected.end());
        selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
        for (auto& nodeHeads : m_pendingNodeHeadSelections)
        {
            std::sort(nodeHeads.begin(), nodeHeads.end());
            nodeHeads.erase(std::unique(nodeHeads.begin(), nodeHeads.end()), nodeHeads.end());
        }
    }

    if (!m_options.m_noOutput)
    {
        std::sort(selected.begin(), selected.end());
        if (!WriteSelectedHeadFiles(data,
                                    std::vector<SizeType>(selected.begin(), selected.end()),
                                    m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile,
                                    m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile))
        {
            return false;
        }

        if (!m_pendingNodeHeadSelections.empty())
        {
            const std::string headRoot = m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder;
            if (!EnsureDirectory(headRoot)) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to create head bundle root directory %s\n", headRoot.c_str());
                return false;
            }

            for (size_t nodeId = 0; nodeId < m_pendingNodeHeadSelections.size(); ++nodeId)
            {
                const std::string nodeDir = HeadBundleNodeAbsolutePath(m_options, m_options.m_indexDirectory, static_cast<int>(nodeId));
                if (!EnsureDirectory(nodeDir)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to create head bundle node directory %s\n", nodeDir.c_str());
                    return false;
                }

                if (!WriteSelectedHeadFiles(data,
                                            m_pendingNodeHeadSelections[nodeId],
                                            nodeDir + FolderSep + m_options.m_headVectorFile,
                                            nodeDir + FolderSep + m_options.m_headIDFile))
                {
                    return false;
                }
            }
        }
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(t3 - t1).count();
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Total used time: %.2lf minutes (about %.2lf hours).\n",
                 elapsedSeconds / 60.0, elapsedSeconds / 3600.0);
    return true;
}

template <typename T> ErrorCode Index<T>::BuildIndexInternal(std::shared_ptr<Helper::VectorSetReader> &p_reader)
{
    if (!(m_options.m_indexDirectory.empty()) && !(direxists(m_options.m_indexDirectory.c_str())))
    {
        mkdir(m_options.m_indexDirectory.c_str());
    }
    if (!(m_options.m_persistentBufferPath.empty()) && !(direxists(m_options.m_persistentBufferPath.c_str())))
    {
        mkdir(m_options.m_persistentBufferPath.c_str());
    }
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Begin Select Head...\n");
    auto t1 = std::chrono::high_resolution_clock::now();
    if (m_options.m_selectHead)
    {
        bool success = false;
        if (m_pQuantizer)
        {
            success = SelectHeadInternal<std::uint8_t>(p_reader);
        }
        else
        {
            success = SelectHeadInternal<T>(p_reader);
        }
        if (!success)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SelectHead Failed!\n");
            return ErrorCode::Fail;
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double selectHeadTime = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "select head time: %.2lfs\n", selectHeadTime);

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Begin Build Head...\n");
    if (m_options.m_buildHead)
    {
        auto valueType = m_pQuantizer ? SPTAG::VectorValueType::UInt8 : m_options.m_valueType;
        auto dims = m_pQuantizer ? m_pQuantizer->GetNumSubvectors() : m_options.m_dim;
        auto buildHeadIndexFromFile = [&](const std::string& vectorFilePath, const std::string& saveDir) -> bool {
            std::shared_ptr<Helper::ReaderOptions> localVectorOptions(
                new Helper::ReaderOptions(valueType, dims, VectorFileType::DEFAULT));
            auto localVectorReader = Helper::VectorSetReader::CreateInstance(localVectorOptions);
            if (ErrorCode::Success != localVectorReader->LoadFile(vectorFilePath)) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head vector file %s.\n", vectorFilePath.c_str());
                return false;
            }

            auto localHeadIndex = SPTAG::VectorIndex::CreateInstance(m_options.m_indexAlgoType, valueType);
            if (localHeadIndex == nullptr) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to create node head index instance.\n");
                return false;
            }

            localHeadIndex->SetParameter("DistCalcMethod", SPTAG::Helper::Convert::ConvertToString(m_options.m_distCalcMethod));
            localHeadIndex->SetQuantizer(m_pQuantizer);
            for (const auto& iter : m_headParameters)
            {
                localHeadIndex->SetParameter(iter.first.c_str(), iter.second.c_str());
            }

            auto localHeadVectorSet = localVectorReader->GetVectorSet();
            if (localHeadIndex->BuildIndex(localHeadVectorSet, nullptr, false, true, true) != ErrorCode::Success)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to build node head index for %s.\n", saveDir.c_str());
                return false;
            }
            if (!m_options.m_quantizerFilePath.empty()) {
                localHeadIndex->SetQuantizerFileName(
                    m_options.m_quantizerFilePath.substr(m_options.m_quantizerFilePath.find_last_of("/\\") + 1));
            }
            if (localHeadIndex->SaveIndex(saveDir) != ErrorCode::Success)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to save node head index to %s.\n", saveDir.c_str());
                return false;
            }
            return true;
        };

        m_index = SPTAG::VectorIndex::CreateInstance(m_options.m_indexAlgoType, valueType);
        m_index->SetParameter("DistCalcMethod", SPTAG::Helper::Convert::ConvertToString(m_options.m_distCalcMethod));
        m_index->SetQuantizer(m_pQuantizer);
        for (const auto &iter : m_headParameters)
        {
            m_index->SetParameter(iter.first.c_str(), iter.second.c_str());
        }

        std::shared_ptr<Helper::ReaderOptions> vectorOptions(
            new Helper::ReaderOptions(valueType, dims, VectorFileType::DEFAULT));
        auto vectorReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
        if (ErrorCode::Success !=
            vectorReader->LoadFile(m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head vector file.\n");
            return ErrorCode::Fail;
        }
        {
            auto headvectorset = vectorReader->GetVectorSet();
            if (m_index->BuildIndex(headvectorset, nullptr, false, true, true) != ErrorCode::Success)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to build head index.\n");
                return ErrorCode::Fail;
            }
            if (!m_options.m_quantizerFilePath.empty())
                m_index->SetQuantizerFileName(
                    m_options.m_quantizerFilePath.substr(m_options.m_quantizerFilePath.find_last_of("/\\") + 1));
            if (m_index->SaveIndex(m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder) !=
                ErrorCode::Success)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to save head index.\n");
                return ErrorCode::Fail;
            }
        }
        m_index.reset();
        if (LoadIndex(m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder, m_index) !=
            ErrorCode::Success)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot load head index from %s!\n",
                         (m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder).c_str());
        }

        if (!m_pendingNodeHeadSelections.empty())
        {
            m_headBundleNodes.clear();
            SizeType headOffset = 0;
            SizeType postingOffset = 0;
            for (size_t nodeId = 0; nodeId < m_pendingNodeHeadSelections.size(); ++nodeId)
            {
                HeadBundleNodeInfo nodeInfo;
                nodeInfo.nodeId = static_cast<int>(nodeId);
                nodeInfo.headIndexRelativePath = HeadBundleNodeRelativePath(m_options, static_cast<int>(nodeId));
                nodeInfo.headOffset = headOffset;
                nodeInfo.postingOffset = postingOffset;
                nodeInfo.headCount = static_cast<SizeType>(m_pendingNodeHeadSelections[nodeId].size());
                nodeInfo.postingCount = nodeInfo.headCount;
                nodeInfo.assignmentCount = (nodeId < m_pendingNodeVectorAssignments.size())
                    ? static_cast<SizeType>(m_pendingNodeVectorAssignments[nodeId].size())
                    : nodeInfo.postingCount;

                if (nodeInfo.headCount > 0)
                {
                    const std::string nodeDir = HeadBundleNodeAbsolutePath(m_options, m_options.m_indexDirectory, static_cast<int>(nodeId));
                    const std::string nodeVectorFile = nodeDir + FolderSep + m_options.m_headVectorFile;
                    if (!buildHeadIndexFromFile(nodeVectorFile, nodeDir)) {
                        return ErrorCode::Fail;
                    }
                }

                m_headBundleNodes.emplace_back(nodeInfo);
                headOffset += nodeInfo.headCount;
                postingOffset += nodeInfo.postingCount;
            }
        }
        else
        {
            InitializeDefaultHeadBundle();
        }
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double buildHeadTime = std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count();
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "select head time: %.2lfs build head time: %.2lfs\n", selectHeadTime,
                 buildHeadTime);

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Begin Build SSDIndex...\n");
    if (m_options.m_enableSSD)
    {
        if (m_index == nullptr && LoadIndex(m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder,
                                            m_index) != ErrorCode::Success)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot load head index from %s!\n",
                         (m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder).c_str());
            return ErrorCode::Fail;
        }
        m_index->SetQuantizer(m_pQuantizer);
        if (!CheckHeadIndexType())
            return ErrorCode::Fail;

        m_index->SetParameter("NumberOfThreads", std::to_string(m_options.m_iSSDNumberOfThreads));
        m_index->SetParameter("MaxCheck", std::to_string(m_options.m_maxCheck));
        m_index->SetParameter("HashTableExponent", std::to_string(m_options.m_hashExp));

        m_index->UpdateIndex();

        if (m_options.m_storage == Storage::STATIC)
        {
            if (m_pQuantizer)
                m_extraSearcher.reset(new ExtraStaticSearcher<std::uint8_t>());
            else
                m_extraSearcher.reset(new ExtraStaticSearcher<T>());
        }
        else
        {
            if (m_pQuantizer)
                m_extraSearcher.reset(new ExtraDynamicSearcher<std::uint8_t>(m_options));
            else
                m_extraSearcher.reset(new ExtraDynamicSearcher<T>(m_options));
        }

        // Pass pending vector tags to ExtraDynamicSearcher for embedding in postings
        if (!m_pendingVectorTags.empty() && m_pendingNumTagsPerVec > 0) {
            auto* eds = dynamic_cast<ExtraDynamicSearcher<T>*>(m_extraSearcher.get());
            if (eds) {
                int numVecs = (int)(m_pendingVectorTags.size() / m_pendingNumTagsPerVec);
                eds->SetVectorTags(m_pendingVectorTags.data(), numVecs, m_pendingNumTagsPerVec);
                if (!m_pendingNodeVectorAssignments.empty()) {
                    eds->SetNodeVectorAssignments(m_pendingNodeVectorAssignments);
                }
                if (!m_pendingPrimaryNodeVectorAssignments.empty()) {
                    eds->SetPrimaryNodeVectorAssignments(m_pendingPrimaryNodeVectorAssignments);
                }
                if (!m_pendingHeadVectorOwners.empty()) {
                    eds->SetHeadVectorOwners(m_pendingHeadVectorOwners);
                }
            }
        }

       {
            std::shared_ptr<Helper::DiskIO> ptr = SPTAG::f_createIO();
            if (ptr == nullptr ||
                !ptr->Initialize((m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str(),
                                 std::ios::binary | std::ios::in))
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to open headIDFile file:%s\n",
                             (m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str());
                return ErrorCode::Fail;
            }
            m_vectorTranslateMap.Load(ptr, m_index->m_iDataBlockSize, m_index->m_iDataCapacity);
        }

        if (m_options.m_buildSsdIndex)
        {
            if (m_options.m_storage != Storage::STATIC && !m_extraSearcher->Available())
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Extrasearcher is not available and failed to initialize.\n");
                return ErrorCode::Fail;
            }
            if (!m_extraSearcher->BuildIndex(p_reader, m_index, m_options, m_versionMap, m_vectorTranslateMap))
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "BuildSSDIndex Failed!\n");
                return ErrorCode::Fail;
            }

            if (!m_options.m_excludehead)
            {
                std::uint64_t vid = (std::uint64_t)MaxSize;
                for (int i = 0; i < m_vectorTranslateMap.R(); i++) {
                    *(m_vectorTranslateMap[i]) = vid;
                }
                m_vectorTranslateMap.Save(m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Include all vectors into SSD index...\n");
            }
        }

        if (!m_extraSearcher->LoadIndex(m_options, m_versionMap, m_vectorTranslateMap, m_index))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot Load SSDIndex!\n");
            if (m_options.m_buildSsdIndex)
            {
                return ErrorCode::Fail;
            }
            else
            {
                m_extraSearcher.reset();
            }
        }

        if (m_extraSearcher != nullptr)
        {
            if ((m_options.m_storage != Storage::STATIC) && m_options.m_preReassign)
            {
                if (m_extraSearcher->RefineIndex(m_index) != ErrorCode::Success)
                    return ErrorCode::Fail;
            }
        }
    }

    auto t4 = std::chrono::high_resolution_clock::now();
    double buildSSDTime = std::chrono::duration_cast<std::chrono::seconds>(t4 - t3).count();
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "select head time: %.2lfs build head time: %.2lfs build ssd time: %.2lfs\n",
                 selectHeadTime, buildHeadTime, buildSSDTime);

    if (m_options.m_deleteHeadVectors)
    {
        if (fileexists((m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile).c_str()) &&
            remove((m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile).c_str()) != 0)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "Head vector file can't be removed.\n");
        }
    }

    if (m_headBundleNodes.empty())
    {
        InitializeDefaultHeadBundle();
    }
    if (SaveHeadBundleManifest(m_options.m_indexDirectory) != ErrorCode::Success)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to save head bundle manifest.\n");
        return ErrorCode::Fail;
    }
    if (InitializeHeadBundleRuntime(m_options.m_indexDirectory) != ErrorCode::Success)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to initialize head bundle runtime.\n");
        return ErrorCode::Fail;
    }

    m_bReady = true;
    return ErrorCode::Success;
}
template <typename T> ErrorCode Index<T>::BuildIndex(bool p_normalized)
{
    SPTAG::VectorValueType valueType = m_pQuantizer ? SPTAG::VectorValueType::UInt8 : m_options.m_valueType;
    SizeType dim = m_pQuantizer ? m_pQuantizer->GetNumSubvectors() : m_options.m_dim;
    std::shared_ptr<Helper::ReaderOptions> vectorOptions(
        new Helper::ReaderOptions(valueType, dim, m_options.m_vectorType, m_options.m_vectorDelimiter,
                                  m_options.m_iSSDNumberOfThreads, p_normalized));
    auto vectorReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
    if (m_options.m_vectorPath.empty())
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Vector file is empty. Skipping loading.\n");
    }
    else
    {
        if (ErrorCode::Success != vectorReader->LoadFile(m_options.m_vectorPath))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read vector file.\n");
            return ErrorCode::Fail;
        }
        m_options.m_vectorSize = vectorReader->GetVectorSet()->Count();
    }
    return BuildIndexInternal(vectorReader);
}

template <typename T>
ErrorCode Index<T>::BuildIndex(const void *p_data, SizeType p_vectorNum, DimensionType p_dimension, bool p_normalized,
                               bool p_shareOwnership)
{
    if (p_data == nullptr || p_vectorNum == 0 || p_dimension == 0)
        return ErrorCode::EmptyData;

    std::shared_ptr<VectorSet> vectorSet;
    if (p_shareOwnership)
    {
        vectorSet.reset(
            new BasicVectorSet(ByteArray((std::uint8_t *)p_data, sizeof(T) * p_vectorNum * p_dimension, false),
                               GetEnumValueType<T>(), p_dimension, p_vectorNum));
    }
    else
    {
        ByteArray arr = ByteArray::Alloc(sizeof(T) * p_vectorNum * p_dimension);
        memcpy(arr.Data(), p_data, sizeof(T) * p_vectorNum * p_dimension);
        vectorSet.reset(new BasicVectorSet(arr, GetEnumValueType<T>(), p_dimension, p_vectorNum));
    }

    if (m_options.m_distCalcMethod == DistCalcMethod::Cosine && !p_normalized)
    {
        vectorSet->Normalize(m_options.m_iSSDNumberOfThreads);
    }
    SPTAG::VectorValueType valueType = m_pQuantizer ? SPTAG::VectorValueType::UInt8 : m_options.m_valueType;
    std::shared_ptr<Helper::VectorSetReader> vectorReader(new Helper::MemoryVectorReader(
        std::make_shared<Helper::ReaderOptions>(valueType, p_dimension, VectorFileType::DEFAULT,
                                                m_options.m_vectorDelimiter, m_options.m_iSSDNumberOfThreads, true),
        vectorSet));

    m_options.m_valueType = GetEnumValueType<T>();
    m_options.m_dim = p_dimension;
    m_options.m_vectorSize = p_vectorNum;
    return BuildIndexInternal(vectorReader);
}

template <typename T> ErrorCode Index<T>::UpdateIndex()
{
    m_index->SetParameter("NumberOfThreads", std::to_string(m_options.m_iSSDNumberOfThreads));
    // m_index->SetParameter("MaxCheck", std::to_string(m_options.m_maxCheck));
    // m_index->SetParameter("HashTableExponent", std::to_string(m_options.m_hashExp));
    m_index->UpdateIndex();
    return ErrorCode::Success;
}

template <typename T>
ErrorCode Index<T>::RefineIndex(const std::vector<std::shared_ptr<Helper::DiskIO>> &p_indexStreams,
                                IAbortOperation *p_abort, std::vector<SizeType> *p_mapping)
{
    if (m_index == nullptr || m_versionMap.Count() == 0)
        return ErrorCode::EmptyIndex;

    while (!AllFinished())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    std::lock_guard<std::mutex> lock(m_dataAddLock);
    std::unique_lock<std::shared_timed_mutex> uniquelock(m_dataDeleteLock);

    std::vector<SizeType> headOldtoNew;
    ErrorCode ret;

    if ((ret = m_index->RefineIndex(p_indexStreams, nullptr, &headOldtoNew)) != ErrorCode::Success)
        return ret;

    std::vector<SizeType> OldtoNew;
    std::vector<SizeType> NewtoOld;
    SizeType newR = m_versionMap.Count();
    if (p_mapping == nullptr) p_mapping = &OldtoNew;
    p_mapping->resize(newR);
    
    for (SizeType i = 0; i < newR; i++)
    {
        if (!m_versionMap.Deleted(i))
        {
            NewtoOld.push_back(i);
            (*p_mapping)[i] = i;
        }
        else
        {
            while (m_versionMap.Deleted(newR - 1) && newR > i)
                newR--;
            if (newR == i)
                break;
            NewtoOld.push_back(newR - 1);
            (*p_mapping)[newR - 1] = i;
            newR--;
        }
    }

    COMMON::Dataset<std::uint64_t> new_vectorTranslateMap(m_index->GetNumSamples() - m_index->GetNumDeleted(), 1,
                                                          m_index->m_iDataBlockSize, m_index->m_iDataCapacity);
    for (int i = 0; i < m_vectorTranslateMap.R(); i++)
    {
        if (m_index->ContainSample(i))
        {
            auto oldID = *(m_vectorTranslateMap[i]);
            if (oldID == MaxSize)
            {
                // Special case: including head vectors in postings means map all IDs to MaxSize
                *(new_vectorTranslateMap[headOldtoNew[i]]) = oldID;
            }
            else if (oldID >= p_mapping->size())
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SPANNIndex::RefineIndex: Vector %d with old ID %llu cannot be mapped! p_mapping size: %llu.\n", i, oldID, p_mapping->size());
            }
            else {
                if (m_versionMap.Deleted(oldID))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SPANNIndex::RefineIndex: Vector %d with old ID %llu is deleted in disk index, but still in head index!\n", i, oldID);
                }
                *(new_vectorTranslateMap[headOldtoNew[i]]) = (*p_mapping)[oldID];
            }
        }
    }
    new_vectorTranslateMap.Save(p_indexStreams[m_index->GetIndexFiles()->size()]);

    COMMON::VersionLabel new_versionMap;
    new_versionMap.Initialize(newR, m_index->m_iDataBlockSize, m_index->m_iDataCapacity);
    for (SizeType i = 0; i < newR; i++)
        new_versionMap.SetVersion(i, m_versionMap.GetVersion(NewtoOld[i]));
    new_versionMap.Save(m_options.m_indexDirectory + FolderSep + m_options.m_deleteIDFile);

    if (nullptr != m_pMetadata)
    {
        if (p_indexStreams.size() < GetIndexFiles()->size() + 2)
            return ErrorCode::LackOfInputs;
        if ((ret = m_pMetadata->RefineMetadata(NewtoOld, p_indexStreams[GetIndexFiles()->size()],
                                               p_indexStreams[GetIndexFiles()->size() + 1])) != ErrorCode::Success)
            return ret;
    }
    for (int i = 0; i < p_indexStreams.size(); i++)
    {
        p_indexStreams[i]->ShutDown();
    }

    if ((ret = m_extraSearcher->RefineIndex(m_index, false, &headOldtoNew, p_mapping)) != ErrorCode::Success)
        return ret;

    return ret;
}

template <typename T> ErrorCode Index<T>::SetParameter(const char *p_param, const char *p_value, const char *p_section)
{
    if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_section, "BuildHead") &&
        !SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "isExecute"))
    {
        if (m_index != nullptr)
            return m_index->SetParameter(p_param, p_value);
        else
            m_headParameters[p_param] = p_value;
    }
    else
    {
        m_options.SetParameter(p_section, p_param, p_value);
    }
    if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "DistCalcMethod"))
    {
        if (m_pQuantizer)
        {
            m_fComputeDistance = m_pQuantizer->DistanceCalcSelector<T>(m_options.m_distCalcMethod);
            m_iBaseSquare = (m_options.m_distCalcMethod == DistCalcMethod::Cosine)
                                ? m_pQuantizer->GetBase() * m_pQuantizer->GetBase()
                                : 1;
        }
        else
        {
            m_fComputeDistance = COMMON::DistanceCalcSelector<T>(m_options.m_distCalcMethod);
            m_iBaseSquare = (m_options.m_distCalcMethod == DistCalcMethod::Cosine)
                                ? COMMON::Utils::GetBase<T>() * COMMON::Utils::GetBase<T>()
                                : 1;
        }
    }
    return ErrorCode::Success;
}

template <typename T> std::string Index<T>::GetParameter(const char *p_param, const char *p_section) const
{
    if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_section, "BuildHead") &&
        !SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "isExecute"))
    {
        if (m_index != nullptr)
            return m_index->GetParameter(p_param);
        else
        {
            auto iter = m_headParameters.find(p_param);
            if (iter != m_headParameters.end())
                return iter->second;
            return "Undefined!";
        }
    }
    else
    {
        return m_options.GetParameter(p_section, p_param);
    }
}

// Add insert entry to persistent buffer
template <typename T>
ErrorCode Index<T>::AddIndex(const void *p_data, SizeType p_vectorNum, DimensionType p_dimension,
                             std::shared_ptr<MetadataSet> p_metadataSet, bool p_withMetaIndex, bool p_normalized)
{
    if ((m_options.m_storage == Storage::STATIC) || m_extraSearcher == nullptr)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Only Support KV Extra Update\n");
        return ErrorCode::Fail;
    }

    if (p_data == nullptr || p_vectorNum == 0 || p_dimension == 0)
        return ErrorCode::EmptyData;
    if (p_dimension != GetFeatureDim())
        return ErrorCode::DimensionSizeMismatch;

    SizeType begin, end;
    {
        std::lock_guard<std::mutex> lock(m_dataAddLock);

        begin = m_versionMap.GetVectorNum();
        end = begin + p_vectorNum;

        if (begin == 0)
        {
            return ErrorCode::EmptyIndex;
        }

        if (m_versionMap.AddBatch(p_vectorNum) != ErrorCode::Success)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "MemoryOverFlow: VID: %d, Map Size:%d\n", begin,
                         m_versionMap.BufferSize());
            return ErrorCode::MemoryOverFlow;
        }

        if (m_pMetadata != nullptr)
        {
            if (p_metadataSet != nullptr)
            {
                m_pMetadata->AddBatch(*p_metadataSet);
                if (HasMetaMapping())
                {
                    for (SizeType i = begin; i < end; i++)
                    {
                        ByteArray meta = m_pMetadata->GetMetadata(i);
                        std::string metastr((char *)meta.Data(), meta.Length());
                        UpdateMetaMapping(metastr, i);
                    }
                }
            }
            else
            {
                for (SizeType i = begin; i < end; i++)
                    m_pMetadata->Add(ByteArray::c_empty);
            }
        }
    }

    std::shared_ptr<VectorSet> vectorSet;
    if (m_options.m_distCalcMethod == DistCalcMethod::Cosine && !p_normalized)
    {
        ByteArray arr = ByteArray::Alloc(sizeof(T) * p_vectorNum * p_dimension);
        memcpy(arr.Data(), p_data, sizeof(T) * p_vectorNum * p_dimension);
        vectorSet.reset(new BasicVectorSet(arr, GetEnumValueType<T>(), p_dimension, p_vectorNum));
        int base = COMMON::Utils::GetBase<T>();
        for (SizeType i = 0; i < p_vectorNum; i++)
        {
            COMMON::Utils::Normalize((T *)(vectorSet->GetVector(i)), p_dimension, base);
        }
    }
    else
    {
        vectorSet.reset(
            new BasicVectorSet(ByteArray((std::uint8_t *)p_data, sizeof(T) * p_vectorNum * p_dimension, false),
                               GetEnumValueType<T>(), p_dimension, p_vectorNum));
    }

    auto workSpace = m_workSpaceFactory->GetWorkSpace();
    if (!workSpace)
    {
        workSpace.reset(new ExtraWorkSpace());
        m_extraSearcher->InitWorkSpace(workSpace.get(), false);
    }
    else
    {
        m_extraSearcher->InitWorkSpace(workSpace.get(), true);
    }
    workSpace->m_deduper.clear();
    workSpace->m_postingIDs.clear();
    return m_extraSearcher->AddIndex(workSpace.get(), vectorSet, m_index, begin);
}

template <typename T>
ErrorCode Index<T>::Check()
{
    std::atomic<ErrorCode> ret = ErrorCode::Success;
    while (!AllFinished())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    //if ((ret = m_index->Check()) != ErrorCode::Success)
    //    return ret;

    std::vector<std::thread> mythreads;
    mythreads.reserve(m_options.m_iSSDNumberOfThreads);
    std::atomic_size_t sent(0);
    std::vector<std::uint8_t> checked(m_extraSearcher->GetNumBlocks(), false);
    for (int tid = 0; tid < m_options.m_iSSDNumberOfThreads; tid++)
    {
        mythreads.emplace_back([&, tid]() {
            auto workSpace = m_workSpaceFactory->GetWorkSpace();
            if (!workSpace)
            {
                workSpace.reset(new ExtraWorkSpace());
                m_extraSearcher->InitWorkSpace(workSpace.get(), false);
            }
            else
            {
                m_extraSearcher->InitWorkSpace(workSpace.get(), true);
            }
            size_t i = 0;
            while (true)
            {
                i = sent.fetch_add(1);
                if (i < m_index->GetNumSamples())
                {
                    if (m_index->ContainSample(i))
                    {
                        if (m_extraSearcher->CheckPosting(i, &checked, workSpace.get()) != ErrorCode::Success)
                        {
                            ret = ErrorCode::Fail;
                            return;
                        }

                        auto translatedID = *(m_vectorTranslateMap[i]);
                        if (translatedID >= m_versionMap.Count() && translatedID != MaxSize)
                        {
                            ret = ErrorCode::Fail;
                            return;
                        }
                    }
                }
                else
                {
                    return;
                }
            }
            m_workSpaceFactory->ReturnWorkSpace(std::move(workSpace));
        });
    }
    for (auto &t : mythreads)
    {
        t.join();
    }
    mythreads.clear();
    return ret.load();
}

template <typename T> ErrorCode Index<T>::DeleteIndex(const SizeType &p_id)
{
    std::shared_lock<std::shared_timed_mutex> sharedlock(m_dataDeleteLock);
    // if (m_versionMap.Delete(p_id)) return ErrorCode::Success;
    // return ErrorCode::VectorNotFound;
    return m_extraSearcher->DeleteIndex(p_id);
}

template <typename T> ErrorCode Index<T>::DeleteIndex(const void *p_vectors, SizeType p_vectorNum)
{
    // TODO: Support batch delete
    DimensionType p_dimension = GetFeatureDim();
    std::shared_ptr<VectorSet> vectorSet;
    if (m_options.m_distCalcMethod == DistCalcMethod::Cosine)
    {
        ByteArray arr = ByteArray::Alloc(sizeof(T) * p_vectorNum * p_dimension);
        memcpy(arr.Data(), p_vectors, sizeof(T) * p_vectorNum * p_dimension);
        vectorSet.reset(new BasicVectorSet(arr, GetEnumValueType<T>(), p_dimension, p_vectorNum));
        int base = COMMON::Utils::GetBase<T>();
        for (SizeType i = 0; i < p_vectorNum; i++)
        {
            COMMON::Utils::Normalize((T *)(vectorSet->GetVector(i)), p_dimension, base);
        }
    }
    else
    {
        vectorSet.reset(new BasicVectorSet(ByteArray((std::uint8_t *)p_vectors, sizeof(T) * p_vectorNum * p_dimension, false),
                                           GetEnumValueType<T>(), p_dimension, p_vectorNum));
    }

    auto workSpace = m_workSpaceFactory->GetWorkSpace();
    if (!workSpace)
    {
        workSpace.reset(new ExtraWorkSpace());
        m_extraSearcher->InitWorkSpace(workSpace.get(), false);
    }
    else
    {
        m_extraSearcher->InitWorkSpace(workSpace.get(), true);
    }
    workSpace->m_deduper.clear();
    workSpace->m_postingIDs.clear();

    SizeType p_id = m_extraSearcher->SearchVector(workSpace.get(), vectorSet, m_index);
    if (p_id == -1)
        return ErrorCode::VectorNotFound;
    return DeleteIndex(p_id);
}
} // namespace SPANN
} // namespace SPTAG

#define DefineVectorValueType(Name, Type) template class SPTAG::SPANN::Index<Type>;

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType
