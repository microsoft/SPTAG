// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/SPANN/Index.h"
#include "inc/Core/BKT/Index.h"
#include "inc/Core/KDT/Index.h"
#include "inc/Helper/VectorSetReaders/MemoryReader.h"
#include "inc/Core/SPANN/ExtraDynamicSearcher.h"
#include "inc/Core/SPANN/ExtraStaticSearcher.h"
#include "inc/Core/SPANN/HeadCrossEdgeBuilder.h"
#include "inc/Core/SPANN/PrimaryHeadCSR.h"
#include "inc/Helper/HeadCrossEdges.h"
#include <algorithm>
#include <chrono>
#include <array>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <atomic>
#include <fstream>
#include <functional>
#include <map>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <shared_mutex>
#include <thread>
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

constexpr std::int32_t kHeadNodeRoutingIndexVersion = 1;
struct HeadNodeRoutingIndexFileHeader {
    std::int32_t version;
    std::int32_t pivotLevel;
    std::int32_t nodeCount;
    std::int32_t numHeadSamples;
    std::int32_t numTagMappings;
};

static bool LoadUpdateRoutingNodes(const std::string& indexDir, const std::string& headIndexFolder,
                                   int& pivotLevel,
                                   std::unordered_map<std::uint32_t, std::vector<int>>& tagToNodes)
{
    pivotLevel = -1;
    tagToNodes.clear();
    FILE* file = fopen((indexDir + FolderSep + headIndexFolder + FolderSep + "tag_node_index.bin").c_str(), "rb");
    if (file == nullptr) return false;

    HeadNodeRoutingIndexFileHeader header {};
    bool ok = fread(&header, sizeof(header), 1, file) == 1 &&
              header.version == kHeadNodeRoutingIndexVersion &&
              header.pivotLevel >= 0 && header.nodeCount >= 0 &&
              header.numHeadSamples >= 0 && header.numTagMappings >= 0;
    for (int node = 0; ok && node < header.nodeCount; ++node) {
        std::int32_t count = 0;
        ok = fread(&count, sizeof(count), 1, file) == 1 && count >= 0;
        if (ok && count > 0) {
            std::vector<std::uint32_t> ignored(static_cast<size_t>(count));
            ok = fread(ignored.data(), sizeof(std::uint32_t), ignored.size(), file) == ignored.size();
        }
    }
    for (int mapping = 0; ok && mapping < header.numTagMappings; ++mapping) {
        std::uint32_t tag = 0;
        std::int32_t count = 0;
        ok = fread(&tag, sizeof(tag), 1, file) == 1 &&
             fread(&count, sizeof(count), 1, file) == 1 && count >= 0;
        if (!ok) break;
        std::vector<int> nodes(static_cast<size_t>(count));
        if (count > 0) {
            ok = fread(nodes.data(), sizeof(std::int32_t), nodes.size(), file) == nodes.size();
        }
        if (ok) tagToNodes.emplace(tag, std::move(nodes));
    }
    fclose(file);
    if (!ok) {
        tagToNodes.clear();
        return false;
    }
    pivotLevel = header.pivotLevel;
    return true;
}

// SelectHead checkpoint ('HSST'): persists the PerTagBKT-derived in-memory state
// (node head selections, per-bundle U_extra, node/primary vector assignments, head
// vector owners, head roles) so a failed BuildHead/BuildSSDIndex can be restarted
// WITHOUT re-running the expensive head-selection BKT k-means. Enabled by
// SPTAG_PERSIST_SELECTHEAD=1; resumed by additionally setting SPTAG_RESUME_BUILD=1.
constexpr std::uint32_t kHeadSelectStateMagic = 0x54535348U; // 'HSST'
constexpr std::int32_t  kHeadSelectStateVersion = 1;

struct HeadSelectStateHeader {
    std::uint32_t magic;
    std::int32_t  version;
    std::int64_t  nodeHeadSelOuter;
    std::int64_t  nodeUExtraOuter;
    std::int64_t  nodeVecAssignOuter;
    std::int64_t  primaryVecAssignOuter;
    std::int64_t  headOwnersCount;
    std::int64_t  headRolesCount;
};

static inline bool SpannEnvFlagOn(const char* name) {
    const char* v = std::getenv(name);
    return v && (v[0] == '1' || v[0] == 't' || v[0] == 'T' || v[0] == 'y' || v[0] == 'Y');
}

static bool WriteNestedSizeVec(FILE* f, const std::vector<std::vector<SizeType>>& v) {
    for (const auto& inner : v) {
        std::int64_t len = static_cast<std::int64_t>(inner.size());
        if (fwrite(&len, sizeof(len), 1, f) != 1) return false;
        if (len > 0 &&
            fwrite(inner.data(), sizeof(SizeType), static_cast<size_t>(len), f) != static_cast<size_t>(len))
            return false;
    }
    return true;
}

static bool ReadNestedSizeVec(FILE* f, std::vector<std::vector<SizeType>>& v, std::int64_t outer) {
    v.assign(static_cast<size_t>(outer), std::vector<SizeType>());
    for (std::int64_t i = 0; i < outer; ++i) {
        std::int64_t len = 0;
        if (fread(&len, sizeof(len), 1, f) != 1 || len < 0) return false;
        v[static_cast<size_t>(i)].resize(static_cast<size_t>(len));
        if (len > 0 &&
            fread(v[static_cast<size_t>(i)].data(), sizeof(SizeType), static_cast<size_t>(len), f) !=
                static_cast<size_t>(len))
            return false;
    }
    return true;
}

// Helper to determine tag level from tag ID based on range
// NOTE: Legacy thresholds below are a fallback ONLY. The real tag→level mapping
// is data-driven: the per-tenant tag_level_offsets.bin (ascending per-level minimum
// tag values, e.g. [0,4,20,84]) is plumbed through ThreadLocalSearchContext and
// consulted first. The legacy fixed scheme (1000/2000/3000/4000) does NOT match
// the offset-based tag encoding and silently collapsed every dept/team tag to
// level 0, breaking HierarchicalPostingMask intersection for mid-selectivity
// filtered queries.
// Level 0 (org): 1000-1999, Level 1 (dept): 2000-2999,
// Level 2 (team): 3000-3999, Level 3 (project): 4000-4999
static inline int TagLevelFromId(uint32_t tag) {
    const auto* ctx = SPTAG::VectorIndex::GetThreadLocalSearchContext();
    if (ctx != nullptr && !ctx->m_tagLevelOffsets.empty()) {
        const auto& off = ctx->m_tagLevelOffsets;
        int level = 0;
        for (int l = 0; l < static_cast<int>(off.size()); ++l) {
            if (tag >= off[l]) level = l;
            else break;
        }
        return level;
    }
    if (tag < 2000) return 0;
    if (tag < 3000) return 1;
    if (tag < 4000) return 2;
    return 3;
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

std::string HeadSelectStatePath(const Options& /*p_options*/, const std::string& p_baseDir)
{
    std::string root = p_baseDir;
    if (!root.empty() && *(root.rbegin()) != FolderSep) {
        root += FolderSep;
    }
    return root + "head_select_state.bin";
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

bool CopyFileAtomically(const std::string& sourcePath, const std::string& targetPath)
{
    if (sourcePath == targetPath) return true;
    std::ifstream source(sourcePath, std::ios::binary);
    const std::string temporaryPath = targetPath + ".tmp";
    std::ofstream target(temporaryPath, std::ios::binary | std::ios::trunc);
    if (!source || !target) {
        std::remove(temporaryPath.c_str());
        return false;
    }

    std::vector<char> buffer(1 << 20);
    while (source) {
        source.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize bytes = source.gcount();
        if (bytes > 0) target.write(buffer.data(), bytes);
    }
    target.close();
    if (!source.eof() || !target) {
        std::remove(temporaryPath.c_str());
        return false;
    }
    if (std::rename(temporaryPath.c_str(), targetPath.c_str()) != 0) {
        std::remove(temporaryPath.c_str());
        return false;
    }
    return true;
}

bool CopyMetadataOnlyHeadStore(const std::string& sourcePath, const std::string& targetPath,
                               SizeType totalHeads)
{
    std::ifstream source(sourcePath, std::ios::binary);
    std::uint32_t magic = 0;
    std::int32_t version = 0;
    std::int64_t ignoredTotal = 0;
    std::int64_t ignoredH1Split = 0;
    std::int32_t dimension = 0;
    if (!source.read(reinterpret_cast<char*>(&magic), sizeof(magic)) ||
        !source.read(reinterpret_cast<char*>(&version), sizeof(version)) ||
        !source.read(reinterpret_cast<char*>(&ignoredTotal), sizeof(ignoredTotal)) ||
        !source.read(reinterpret_cast<char*>(&ignoredH1Split), sizeof(ignoredH1Split)) ||
        !source.read(reinterpret_cast<char*>(&dimension), sizeof(dimension)) ||
        magic != 0x484D4F31u || version != 1 || totalHeads < 0) {
        return false;
    }
    source.close();

    const std::string temporaryPath = targetPath + ".tmp";
    std::ofstream target(temporaryPath, std::ios::binary | std::ios::trunc);
    const std::int64_t persistedHeadCount = static_cast<std::int64_t>(totalHeads);
    const bool written = target &&
        static_cast<bool>(target.write(reinterpret_cast<const char*>(&magic), sizeof(magic))) &&
        static_cast<bool>(target.write(reinterpret_cast<const char*>(&version), sizeof(version))) &&
        static_cast<bool>(target.write(reinterpret_cast<const char*>(&persistedHeadCount),
                                       sizeof(persistedHeadCount))) &&
        static_cast<bool>(target.write(reinterpret_cast<const char*>(&persistedHeadCount),
                                       sizeof(persistedHeadCount))) &&
        static_cast<bool>(target.write(reinterpret_cast<const char*>(&dimension), sizeof(dimension)));
    target.close();
    if (!written || !target || std::rename(temporaryPath.c_str(), targetPath.c_str()) != 0) {
        std::remove(temporaryPath.c_str());
        return false;
    }
    return true;
}

bool WriteMetadataOnlyHeadStore(const std::string& targetPath,
                                SizeType totalHeads,
                                DimensionType dimension)
{
    if (totalHeads < 0) return false;

    const std::string temporaryPath = targetPath + ".tmp";
    std::ofstream target(temporaryPath, std::ios::binary | std::ios::trunc);
    const std::uint32_t magic = 0x484D4F31u; // 'HMO1'
    const std::int32_t version = 1;
    const std::int64_t persistedHeadCount = static_cast<std::int64_t>(totalHeads);
    const std::int32_t persistedDimension = static_cast<std::int32_t>(dimension);
    const bool written = target &&
        static_cast<bool>(target.write(reinterpret_cast<const char*>(&magic), sizeof(magic))) &&
        static_cast<bool>(target.write(reinterpret_cast<const char*>(&version), sizeof(version))) &&
        static_cast<bool>(target.write(reinterpret_cast<const char*>(&persistedHeadCount),
                                       sizeof(persistedHeadCount))) &&
        static_cast<bool>(target.write(reinterpret_cast<const char*>(&persistedHeadCount),
                                       sizeof(persistedHeadCount))) &&
        static_cast<bool>(target.write(reinterpret_cast<const char*>(&persistedDimension),
                                       sizeof(persistedDimension)));
    target.close();
    if (!written || !target || std::rename(temporaryPath.c_str(), targetPath.c_str()) != 0) {
        std::remove(temporaryPath.c_str());
        return false;
    }
    return true;
}

bool WriteSelectedHeadIDs(const std::vector<SizeType>& selected,
                          const std::string& idFilePath)
{
    std::shared_ptr<Helper::DiskIO> outputIDs = SPTAG::f_createIO();
    if (outputIDs == nullptr ||
        !outputIDs->Initialize(idFilePath.c_str(), std::ios::binary | std::ios::out)) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "Failed to create head selection ID file: %s\n",
                     idFilePath.c_str());
        return false;
    }

    const SizeType count = static_cast<SizeType>(selected.size());
    const DimensionType dimensions = 1;
    if (outputIDs->WriteBinary(sizeof(count), reinterpret_cast<const char*>(&count)) != sizeof(count) ||
        outputIDs->WriteBinary(sizeof(dimensions), reinterpret_cast<const char*>(&dimensions)) !=
            sizeof(dimensions)) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write head selection ID header.\n");
        return false;
    }

    for (SizeType vid : selected) {
        const std::uint64_t storedVid = static_cast<std::uint64_t>(vid);
        if (outputIDs->WriteBinary(sizeof(storedVid), reinterpret_cast<const char*>(&storedVid)) !=
            sizeof(storedVid)) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Failed to write selected head ID %d.\n", vid);
            return false;
        }
    }
    return true;
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
    const std::string temporaryPath = manifestPath + ".tmp";
    FILE* manifestFile = fopen(temporaryPath.c_str(), "wb");
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

    success = success && fflush(manifestFile) == 0;
    fclose(manifestFile);
    if (!success) {
        remove(temporaryPath.c_str());
        return ErrorCode::Fail;
    }
    if (rename(temporaryPath.c_str(), manifestPath.c_str()) != 0) {
        remove(temporaryPath.c_str());
        return ErrorCode::FailedCreateFile;
    }
    return ErrorCode::Success;
}

template <typename T> ErrorCode Index<T>::SaveLoadedHeadBundles(const std::string& p_baseDir)
{
    const std::string baseDir = p_baseDir.empty() ? m_options.m_indexDirectory : p_baseDir;
    if (m_index == nullptr || baseDir.empty()) {
        return ErrorCode::Fail;
    }

    std::unique_lock<std::shared_timed_mutex> topologyLock(m_headTopologyLock);
    ErrorCode ret = m_vectorTranslateMap.Save(
        baseDir + FolderSep + m_options.m_headIDFile);
    if (ret != ErrorCode::Success) return ret;

    const std::string sourceBaseDir =
        m_headBundleBaseDir.empty() ? m_options.m_indexDirectory : m_headBundleBaseDir;
    const std::string sourceHeadDir = JoinPath(sourceBaseDir, m_options.m_headIndexFolder);
    const std::string targetHeadDir = JoinPath(baseDir, m_options.m_headIndexFolder);
    const bool sourceCrossEdgesDirty = fileexists(
        (sourceHeadDir + FolderSep + Helper::kHeadCrossEdgesDirtyFileName).c_str());
    if (!EnsureDirectory(targetHeadDir)) return ErrorCode::FailedCreateFile;
    const std::string sourceMetaOnly = sourceHeadDir + FolderSep + "head_metaonly.bin";
    {
        std::ifstream probe(sourceMetaOnly, std::ios::binary);
        if (probe && !CopyMetadataOnlyHeadStore(
                         sourceMetaOnly, targetHeadDir + FolderSep + "head_metaonly.bin",
                         m_vectorTranslateMap.R())) {
            return ErrorCode::FailedCreateFile;
        }
    }

    const std::string bundleBaseDir = baseDir;
    const size_t bundleCount = m_headBundleNodes.size();
    if (bundleCount > 1) {
        const std::string sourceRouting = sourceHeadDir + FolderSep + "tag_node_index.bin";
        const std::string targetRouting = targetHeadDir + FolderSep + "tag_node_index.bin";
        if (!CopyFileAtomically(sourceRouting, targetRouting)) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "[TaggedUpdate] failed to checkpoint routing map %s.\n",
                         sourceRouting.c_str());
            return ErrorCode::FailedCreateFile;
        }
    }
    const bool rootOnlyBundle = !m_metadataOnlyHeadStore && bundleCount == 1 &&
        m_headBundleNodes.front().headIndexRelativePath == m_options.m_headIndexFolder;
    if (rootOnlyBundle) {
        if ((ret = SaveHeadBundleManifest(baseDir)) != ErrorCode::Success) return ret;
        if (sourceCrossEdgesDirty || m_headCrossEdgesDirty.load(std::memory_order_acquire)) {
            if ((ret = MarkCrossEdgesDirty(baseDir)) != ErrorCode::Success) return ret;
        }
        return ErrorCode::Success;
    }

    if (m_loadedHeadBundleIndexes.size() < bundleCount ||
        m_headBundleLocalToGlobalHIDs.size() < bundleCount) {
        return ErrorCode::Fail;
    }

    // A recovery checkpoint is self-contained: unchanged bundles must be copied
    // as well as the bundle that received the tagged topology mutation.
    for (size_t slot = 0; slot < bundleCount; ++slot) {
        if (EnsureHeadBundleNodeLoaded(static_cast<int>(slot)) != ErrorCode::Success) {
            return ErrorCode::Fail;
        }
    }

    for (size_t slot = 0; slot < bundleCount; ++slot) {
        const auto& bundleIndex = m_loadedHeadBundleIndexes[slot];
        if (bundleIndex == nullptr) continue;
        const std::string nodeDir = JoinPath(
            bundleBaseDir, m_headBundleNodes[slot].headIndexRelativePath);
        if (!EnsureDirectory(nodeDir)) return ErrorCode::FailedCreateFile;
        if ((ret = bundleIndex->SaveIndex(nodeDir)) != ErrorCode::Success) return ret;

        const auto& localToGlobal = m_headBundleLocalToGlobalHIDs[slot];
        if (static_cast<SizeType>(localToGlobal.size()) != bundleIndex->GetNumSamples()) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "[TaggedUpdate] bundle %d has %zu ID mappings for %d local heads.\n",
                         m_headBundleNodes[slot].nodeId, localToGlobal.size(),
                         bundleIndex->GetNumSamples());
            return ErrorCode::Fail;
        }
        const std::string idPath = nodeDir + FolderSep + m_options.m_headIDFile;
        const std::string tmpPath = idPath + ".tmp";
        std::ofstream output(tmpPath, std::ios::binary | std::ios::trunc);
        if (!output) return ErrorCode::FailedCreateFile;
        const SizeType count = static_cast<SizeType>(localToGlobal.size());
        const DimensionType dimension = 1;
        output.write(reinterpret_cast<const char*>(&count), sizeof(count));
        output.write(reinterpret_cast<const char*>(&dimension), sizeof(dimension));
        for (SizeType globalHeadID : localToGlobal) {
            if (globalHeadID < 0 || globalHeadID >= m_vectorTranslateMap.R()) {
                output.close();
                std::remove(tmpPath.c_str());
                return ErrorCode::Key_OverFlow;
            }
            const std::uint64_t globalVID = *(m_vectorTranslateMap[globalHeadID]);
            output.write(reinterpret_cast<const char*>(&globalVID), sizeof(globalVID));
        }
        output.close();
        if (!output) {
            std::remove(tmpPath.c_str());
            return ErrorCode::FailedCreateFile;
        }
        if (std::rename(tmpPath.c_str(), idPath.c_str()) != 0) {
            std::remove(tmpPath.c_str());
            return ErrorCode::FailedCreateFile;
        }
    }
    if ((ret = SaveHeadBundleManifest(baseDir)) != ErrorCode::Success) return ret;
    if (sourceCrossEdgesDirty || m_headCrossEdgesDirty.load(std::memory_order_acquire)) {
        if ((ret = MarkCrossEdgesDirty(baseDir)) != ErrorCode::Success) return ret;
    }
    return ErrorCode::Success;
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

template <typename T> ErrorCode Index<T>::SaveHeadSelectState(const std::string& p_path) const
{
    const std::string tmpPath = p_path + ".tmp";
    FILE* f = fopen(tmpPath.c_str(), "wb");
    if (f == nullptr) {
        return ErrorCode::FailedCreateFile;
    }

    HeadSelectStateHeader header{};
    header.magic = kHeadSelectStateMagic;
    header.version = kHeadSelectStateVersion;
    header.nodeHeadSelOuter = static_cast<std::int64_t>(m_pendingNodeHeadSelections.size());
    header.nodeUExtraOuter = static_cast<std::int64_t>(m_pendingNodeUExtraSelections.size());
    header.nodeVecAssignOuter = static_cast<std::int64_t>(m_pendingNodeVectorAssignments.size());
    header.primaryVecAssignOuter = static_cast<std::int64_t>(m_pendingPrimaryNodeVectorAssignments.size());
    header.headOwnersCount = static_cast<std::int64_t>(m_pendingHeadVectorOwners.size());
    header.headRolesCount = static_cast<std::int64_t>(m_pendingHeadRoles.size());

    bool ok = fwrite(&header, sizeof(header), 1, f) == 1;
    ok = ok && WriteNestedSizeVec(f, m_pendingNodeHeadSelections);
    ok = ok && WriteNestedSizeVec(f, m_pendingNodeUExtraSelections);
    ok = ok && WriteNestedSizeVec(f, m_pendingNodeVectorAssignments);
    ok = ok && WriteNestedSizeVec(f, m_pendingPrimaryNodeVectorAssignments);
    if (ok) {
        for (const auto& kv : m_pendingHeadVectorOwners) {
            SizeType key = kv.first;
            std::int32_t val = static_cast<std::int32_t>(kv.second);
            if (fwrite(&key, sizeof(key), 1, f) != 1 || fwrite(&val, sizeof(val), 1, f) != 1) {
                ok = false;
                break;
            }
        }
    }
    if (ok && header.headRolesCount > 0) {
        ok = fwrite(m_pendingHeadRoles.data(), 1, static_cast<size_t>(header.headRolesCount), f) ==
             static_cast<size_t>(header.headRolesCount);
    }

    if (ok) ok = (fflush(f) == 0);
    fclose(f);
    if (!ok) {
        remove(tmpPath.c_str());
        return ErrorCode::Fail;
    }
    // Atomic publish so a crash mid-write never leaves a truncated checkpoint.
    if (rename(tmpPath.c_str(), p_path.c_str()) != 0) {
        remove(tmpPath.c_str());
        return ErrorCode::Fail;
    }
    return ErrorCode::Success;
}

template <typename T> ErrorCode Index<T>::LoadHeadSelectState(const std::string& p_path)
{
    FILE* f = fopen(p_path.c_str(), "rb");
    if (f == nullptr) {
        return ErrorCode::Fail;
    }

    HeadSelectStateHeader header{};
    bool ok = fread(&header, sizeof(header), 1, f) == 1 &&
              header.magic == kHeadSelectStateMagic &&
              header.version == kHeadSelectStateVersion &&
              header.nodeHeadSelOuter >= 0 && header.nodeUExtraOuter >= 0 &&
              header.nodeVecAssignOuter >= 0 && header.primaryVecAssignOuter >= 0 &&
              header.headOwnersCount >= 0 && header.headRolesCount >= 0;

    if (ok) ok = ReadNestedSizeVec(f, m_pendingNodeHeadSelections, header.nodeHeadSelOuter);
    if (ok) ok = ReadNestedSizeVec(f, m_pendingNodeUExtraSelections, header.nodeUExtraOuter);
    if (ok) ok = ReadNestedSizeVec(f, m_pendingNodeVectorAssignments, header.nodeVecAssignOuter);
    if (ok) ok = ReadNestedSizeVec(f, m_pendingPrimaryNodeVectorAssignments, header.primaryVecAssignOuter);

    if (ok) {
        m_pendingHeadVectorOwners.clear();
        m_pendingHeadVectorOwners.reserve(static_cast<size_t>(header.headOwnersCount));
        for (std::int64_t i = 0; ok && i < header.headOwnersCount; ++i) {
            SizeType key = 0;
            std::int32_t val = 0;
            if (fread(&key, sizeof(key), 1, f) != 1 || fread(&val, sizeof(val), 1, f) != 1) {
                ok = false;
                break;
            }
            m_pendingHeadVectorOwners[key] = static_cast<int>(val);
        }
    }
    if (ok) {
        m_pendingHeadRoles.resize(static_cast<size_t>(header.headRolesCount));
        if (header.headRolesCount > 0) {
            ok = fread(m_pendingHeadRoles.data(), 1, static_cast<size_t>(header.headRolesCount), f) ==
                 static_cast<size_t>(header.headRolesCount);
        }
    }

    fclose(f);
    return ok ? ErrorCode::Success : ErrorCode::Fail;
}

template <typename T> ErrorCode Index<T>::InitializeHeadBundleNodesFromSelections()
{
    if (m_pendingNodeHeadSelections.empty()) {
        return ErrorCode::Fail;
    }

    m_headBundleNodes.clear();
    SizeType headOffset = 0;
    SizeType postingOffset = 0;
    for (size_t nodeId = 0; nodeId < m_pendingNodeHeadSelections.size(); ++nodeId)
    {
        HeadBundleNodeInfo nodeInfo;
        nodeInfo.nodeId = static_cast<int>(nodeId);
        nodeInfo.headIndexRelativePath =
            HeadBundleNodeRelativePath(m_options, static_cast<int>(nodeId));
        nodeInfo.headOffset = headOffset;
        nodeInfo.headCount =
            static_cast<SizeType>(m_pendingNodeHeadSelections[nodeId].size());
        nodeInfo.postingOffset = postingOffset;
        nodeInfo.postingCount = nodeInfo.headCount;
        nodeInfo.assignmentCount = nodeId < m_pendingNodeVectorAssignments.size()
            ? static_cast<SizeType>(m_pendingNodeVectorAssignments[nodeId].size())
            : nodeInfo.postingCount;
        m_headBundleNodes.emplace_back(std::move(nodeInfo));
        headOffset += m_headBundleNodes.back().headCount;
        postingOffset += m_headBundleNodes.back().postingCount;
    }
    return ErrorCode::Success;
}

template <typename T> ErrorCode Index<T>::LoadTopLevelHeadIDMap(SizeType p_expectedHeadCount)
{
    if (p_expectedHeadCount <= 0) {
        return ErrorCode::Fail;
    }

    if (m_vectorTranslateMap.R() == 0)
    {
        const std::string headIDPath =
            m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile;
        std::shared_ptr<Helper::DiskIO> input = SPTAG::f_createIO();
        if (input == nullptr ||
            !input->Initialize(headIDPath.c_str(), std::ios::binary | std::ios::in) ||
            m_vectorTranslateMap.Load(
                input, m_options.m_datasetRowsInBlock, m_options.m_datasetCapacity) !=
                ErrorCode::Success)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Failed to load top-level head IDs from %s.\n",
                         headIDPath.c_str());
            return ErrorCode::Fail;
        }
    }

    if (m_vectorTranslateMap.R() != p_expectedHeadCount)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "Top-level head ID count mismatch: expected=%d actual=%d.\n",
                     static_cast<int>(p_expectedHeadCount),
                     static_cast<int>(m_vectorTranslateMap.R()));
        return ErrorCode::Fail;
    }
    return ErrorCode::Success;
}

template <typename T> ErrorCode Index<T>::ActivateMetadataOnlyBundleRoot()
{
    const SizeType totalHeads = TotalHeadSampleCount();
    if (totalHeads <= 0 ||
        LoadTopLevelHeadIDMap(totalHeads) != ErrorCode::Success)
    {
        return ErrorCode::Fail;
    }

    auto metadataRoot = SPTAG::VectorIndex::CreateInstance(
        SPTAG::IndexAlgoType::KDT, m_options.m_valueType);
    if (metadataRoot == nullptr) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "Failed to create KDT metadata root for bundled STATIC build.\n");
        return ErrorCode::Fail;
    }
    metadataRoot->SetParameter(
        "DistCalcMethod",
        SPTAG::Helper::Convert::ConvertToString(m_options.m_distCalcMethod));
    metadataRoot->SetQuantizer(nullptr);

    constexpr SizeType physicalCount = 1;
    ByteArray rootBytes = ByteArray::Alloc(
        static_cast<size_t>(physicalCount) * static_cast<size_t>(m_options.m_dim) * sizeof(T));
    std::memset(rootBytes.Data(), 0, rootBytes.Length());
    auto rootVectors = std::make_shared<BasicVectorSet>(
        rootBytes, m_options.m_valueType, m_options.m_dim, physicalCount);
    if (metadataRoot->BuildIndex(rootVectors, nullptr, false, true, true) != ErrorCode::Success)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "Failed to build physical KDT metadata root.\n");
        return ErrorCode::Fail;
    }

    // Validate every existing bundle before overwriting the old global root on disk.
    m_index = std::move(metadataRoot);
    m_metadataOnlyHeadStore = false;
    if (InitializeHeadBundleRuntime(m_options.m_indexDirectory) != ErrorCode::Success)
    {
        return ErrorCode::Fail;
    }
    for (const auto& node : m_headBundleNodes)
    {
        if (node.headCount == 0) {
            continue;
        }
        if (node.nodeId < 0 ||
            node.nodeId >= static_cast<int>(m_loadedHeadBundleIndexes.size()) ||
            EnsureHeadBundleNodeLoaded(node.nodeId) != ErrorCode::Success)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Failed to validate head bundle node %d before metadata-root activation.\n",
                         node.nodeId);
            return ErrorCode::Fail;
        }

        const size_t slot = static_cast<size_t>(node.nodeId);
        const auto& bundle = m_loadedHeadBundleIndexes[slot];
        const auto& localToGlobal = m_headBundleLocalToGlobalHIDs[slot];
        if (bundle == nullptr || bundle->GetVectorValueType() != m_options.m_valueType ||
            bundle->GetFeatureDim() != m_options.m_dim ||
            bundle->GetNumSamples() != node.headCount ||
            static_cast<SizeType>(localToGlobal.size()) != node.headCount)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Head bundle node %d does not match the resumed checkpoint.\n",
                         node.nodeId);
            return ErrorCode::Fail;
        }
    }
    {
        std::lock_guard<std::mutex> lock(m_globalHeadVIDToLocalHIDMutex);
        if (static_cast<SizeType>(m_globalHeadVIDToLocalHID.size()) != totalHeads)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Top-level head IDs are not a unique complete bundle-head set.\n");
            return ErrorCode::Fail;
        }
    }

    const std::string headDir =
        m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder;
    if (!EnsureDirectory(headDir) || m_index->SaveIndex(headDir) != ErrorCode::Success)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "Failed to save KDT metadata root to %s.\n", headDir.c_str());
        return ErrorCode::Fail;
    }
    if (!WriteMetadataOnlyHeadStore(
            headDir + FolderSep + "head_metaonly.bin", totalHeads, m_options.m_dim))
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "Failed to write metadata-root sidecar.\n");
        return ErrorCode::Fail;
    }
    if (SetupMetadataOnlyHeadStore(m_options.m_indexDirectory) != ErrorCode::Success)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "Failed to bind bundle samples to KDT metadata root.\n");
        return ErrorCode::Fail;
    }
    // The atomic manifest is the last BuildHead-complete commit marker.
    if (SaveHeadBundleManifest(m_options.m_indexDirectory) != ErrorCode::Success)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "Failed to commit bundle-head manifest.\n");
        return ErrorCode::Fail;
    }

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                 "Bundle STATIC build: activated KDT metadata root for %d logical heads.\n",
                 static_cast<int>(totalHeads));
    return ErrorCode::Success;
}

template <typename T> ErrorCode Index<T>::TryResumeCompletedBundleHeads(bool& p_resumed)
{
    p_resumed = false;
    if (InitializeHeadBundleNodesFromSelections() != ErrorCode::Success) {
        return ErrorCode::Fail;
    }

    const SizeType totalHeads = TotalHeadSampleCount();
    if (totalHeads <= 0 ||
        m_pendingHeadVectorOwners.size() != static_cast<size_t>(totalHeads))
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "Resumed bundle heads have an incomplete owner map.\n");
        return ErrorCode::Fail;
    }
    if (!m_pendingHeadRoles.empty() &&
        (m_pendingHeadRoles.size() != static_cast<size_t>(totalHeads) ||
         std::any_of(m_pendingHeadRoles.begin(), m_pendingHeadRoles.end(),
                     [](uint8_t role) { return role != 0; })))
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "Resuming completed bundle heads does not support U_extra roles.\n");
        return ErrorCode::Fail;
    }

    const std::string headDir =
        m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder;
    const std::string topLevelIDs =
        m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile;
    const std::string manifestPath =
        HeadBundleManifestPath(m_options, m_options.m_indexDirectory);
    const std::string metadataSidecar = headDir + FolderSep + "head_metaonly.bin";

    size_t expectedNonemptyNodes = 0;
    size_t completeNodes = 0;
    bool anyNodeArtifact = false;
    bool partialNodeArtifact = false;
    for (const auto& node : m_headBundleNodes)
    {
        if (node.headCount == 0) continue;
        ++expectedNonemptyNodes;
        const std::string nodeDir =
            JoinPath(m_options.m_indexDirectory, node.headIndexRelativePath);
        const bool hasDirectory = direxists(nodeDir.c_str());
        const bool hasConfig = fileexists((nodeDir + FolderSep + "indexloader.ini").c_str());
        const bool hasIDs = fileexists((nodeDir + FolderSep + m_options.m_headIDFile).c_str());
        const bool hasAny = hasDirectory || hasConfig || hasIDs;
        anyNodeArtifact = anyNodeArtifact || hasAny;
        if (!hasAny) continue;
        if (!hasConfig || !hasIDs) {
            partialNodeArtifact = true;
            continue;
        }
        ++completeNodes;
    }

    if (!anyNodeArtifact)
    {
        if (fileexists(manifestPath.c_str()) || fileexists(metadataSidecar.c_str())) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Bundle-head completion marker exists without bundle graph artifacts.\n");
            return ErrorCode::Fail;
        }
        m_headBundleNodes.clear();
        return ErrorCode::Success; // No completed BuildHead: use the normal rebuild path.
    }
    if (partialNodeArtifact || completeNodes != expectedNonemptyNodes ||
        !fileexists(topLevelIDs.c_str()))
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "Found partial completed bundle-head artifacts; refusing to overwrite them.\n");
        return ErrorCode::Fail;
    }

    const std::vector<HeadBundleNodeInfo> expectedNodes = m_headBundleNodes;
    if (fileexists(manifestPath.c_str()))
    {
        if (LoadHeadBundleManifest(m_options.m_indexDirectory) != ErrorCode::Success ||
            m_headBundleNodes.size() != expectedNodes.size())
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Existing bundle-head manifest is invalid.\n");
            return ErrorCode::Fail;
        }
        for (size_t i = 0; i < expectedNodes.size(); ++i)
        {
            const auto& actual = m_headBundleNodes[i];
            const auto& expected = expectedNodes[i];
            if (actual.nodeId != expected.nodeId ||
                actual.headIndexRelativePath != expected.headIndexRelativePath ||
                actual.headOffset != expected.headOffset ||
                actual.headCount != expected.headCount ||
                actual.postingOffset != expected.postingOffset ||
                actual.postingCount != expected.postingCount ||
                actual.assignmentCount != expected.assignmentCount)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                             "Existing bundle-head manifest does not match the SelectHead checkpoint.\n");
                return ErrorCode::Fail;
            }
        }
        m_headBundleNodes = expectedNodes;
    }

    if (LoadTopLevelHeadIDMap(totalHeads) != ErrorCode::Success) {
        return ErrorCode::Fail;
    }
    for (SizeType hid = 0; hid < totalHeads; ++hid)
    {
        const SizeType globalVID = static_cast<SizeType>(*(m_vectorTranslateMap[hid]));
        const auto owner = m_pendingHeadVectorOwners.find(globalVID);
        if (owner == m_pendingHeadVectorOwners.end() || owner->second < 0 ||
            owner->second >= static_cast<int>(m_headBundleNodes.size()))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Top-level head ID %d is absent from the SelectHead owner map.\n",
                         globalVID);
            return ErrorCode::Fail;
        }
    }

    for (const auto& node : m_headBundleNodes)
    {
        if (node.headCount == 0) continue;
        const auto& expectedIDs =
            m_pendingNodeHeadSelections[static_cast<size_t>(node.nodeId)];
        if (!std::is_sorted(expectedIDs.begin(), expectedIDs.end()) ||
            std::adjacent_find(expectedIDs.begin(), expectedIDs.end()) != expectedIDs.end())
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Checkpoint head IDs for bundle node %d are not sorted and unique.\n",
                         node.nodeId);
            return ErrorCode::Fail;
        }

        COMMON::Dataset<std::uint64_t> nodeHeadIDs;
        nodeHeadIDs.SetName("ResumeBundleNodeIDs");
        const std::string nodeDir =
            JoinPath(m_options.m_indexDirectory, node.headIndexRelativePath);
        if (nodeHeadIDs.Load(nodeDir + FolderSep + m_options.m_headIDFile,
                             m_options.m_datasetRowsInBlock,
                             m_options.m_datasetCapacity) != ErrorCode::Success ||
            nodeHeadIDs.R() != node.headCount)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Bundle node %d head-ID file count does not match its checkpoint.\n",
                         node.nodeId);
            return ErrorCode::Fail;
        }
        for (SizeType localHid = 0; localHid < node.headCount; ++localHid)
        {
            const SizeType globalVID =
                static_cast<SizeType>(*(nodeHeadIDs[localHid]));
            if (globalVID != expectedIDs[static_cast<size_t>(localHid)])
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                             "Bundle node %d head-ID order differs from its checkpoint.\n",
                             node.nodeId);
                return ErrorCode::Fail;
            }
            const auto owner = m_pendingHeadVectorOwners.find(globalVID);
            if (owner == m_pendingHeadVectorOwners.end() || owner->second != node.nodeId)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                             "Bundle node %d contains head %d owned by another node.\n",
                             node.nodeId, globalVID);
                return ErrorCode::Fail;
            }
        }
    }

    if (ActivateMetadataOnlyBundleRoot() != ErrorCode::Success) {
        return ErrorCode::Fail;
    }
    p_resumed = true;
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                 "Resume: reusing %zu completed bundle head graphs; skipping BuildHead.\n",
                 m_headBundleNodes.size());
    return ErrorCode::Success;
}

template <typename T> ErrorCode Index<T>::InitializeHeadBundleRuntime(const std::string& p_baseDir)
{
    std::lock_guard<std::mutex> lock(m_headBundleLoadLock);

    m_headBundleBaseDir = p_baseDir;
    m_loadedHeadBundleIndexes.clear();
    m_headBundleLocalToGlobalHIDs.clear();
    {
        std::lock_guard<std::mutex> mapLock(m_globalHeadVIDToLocalHIDMutex);
        m_globalHeadVIDToLocalHID.clear();
    }
    m_headCrossEdgeOffsetByB.clear();
    m_headCrossEdgeCountByB.clear();
    m_headCrossEdgeTargetsB.clear();
    m_headBundleNodeByB.clear();
    m_headBundleLocalByB.clear();
    m_headCrossEdgesLoaded.store(false, std::memory_order_release);
    m_headCrossEdgesDirty.store(false, std::memory_order_release);
    {
        std::lock_guard<std::mutex> mapLock(m_globalVIDToBundleLocMutex);
        m_globalVIDToBundleLoc.clear();
    }

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
    const bool isMonolithicDefaultBundle =
        !m_metadataOnlyHeadStore && p_nodeId == 0 && m_headBundleNodes.size() == 1 &&
        nodeInfo.headIndexRelativePath == m_options.m_headIndexFolder &&
        nodeInfo.headCount == m_index->GetNumSamples();
    if (isMonolithicDefaultBundle) {
        // The root index is already this bundle. Let the caller use the direct
        // root search path instead of trying to load HeadIndex/HeadIndex.
        return ErrorCode::Fail;
    }

    std::lock_guard<std::mutex> lock(m_headBundleLoadLock);
    std::lock_guard<std::mutex> headMapLock(m_globalHeadVIDToLocalHIDMutex);
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

    const std::string bundleBaseDir =
        m_headBundleBaseDir.empty() ? m_options.m_indexDirectory : m_headBundleBaseDir;
    const std::string nodeDir = JoinPath(bundleBaseDir, nodeInfo.headIndexRelativePath);
    if (localToGlobalHIDs.empty())
    {
        COMMON::Dataset<std::uint64_t> nodeHeadIDs;
        nodeHeadIDs.SetName("HeadBundleNodeIDs");
        if (nodeHeadIDs.Load(nodeDir + FolderSep + m_options.m_headIDFile,
                             m_options.m_datasetRowsInBlock,
                             m_options.m_datasetCapacity) != ErrorCode::Success)
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
        nodeIndex->SetParameter("HashTableExponent", std::to_string(m_options.m_hashExp));
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

template <typename T> ErrorCode Index<T>::SetupMetadataOnlyHeadStore(const std::string& p_baseDir)
{
    // Detect the dual-pool slim head store sidecar. Absent => normal full head index
    // (zero behavior change for every existing/vanilla index).
    std::string headDir = p_baseDir.empty() ? m_options.m_indexDirectory : p_baseDir;
    if (!headDir.empty() && headDir.back() != FolderSep) headDir += FolderSep;
    headDir += m_options.m_headIndexFolder;
    const std::string sidecar = headDir + FolderSep + "head_metaonly.bin";

    FILE* fp = std::fopen(sidecar.c_str(), "rb");
    if (fp == nullptr) return ErrorCode::Success; // not a slim index

    // Sidecar layout is packed (no struct padding): u32 magic, i32 version,
    // i64 totalHeads, i64 h1Split, i32 dim  => 28 bytes. Read field-by-field so
    // host struct alignment can never desync from the writer.
    std::uint32_t magic = 0;
    std::int32_t version = 0;
    std::int64_t totalHeads = 0, h1Split = 0;
    std::int32_t dim = 0;
    bool ok = std::fread(&magic, sizeof(magic), 1, fp) == 1
           && std::fread(&version, sizeof(version), 1, fp) == 1
           && std::fread(&totalHeads, sizeof(totalHeads), 1, fp) == 1
           && std::fread(&h1Split, sizeof(h1Split), 1, fp) == 1
           && std::fread(&dim, sizeof(dim), 1, fp) == 1;
    std::fclose(fp);
    if (!ok || magic != 0x484D4F31u /* 'HMO1' */ || version != 1) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Invalid head_metaonly.bin sidecar at %s\n", sidecar.c_str());
        return ErrorCode::Fail;
    }

    auto* kdt = dynamic_cast<KDT::Index<T>*>(m_index.get());
    if (kdt == nullptr) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Slim head store requires a KDT head index.\n");
        return ErrorCode::Fail;
    }

    const SizeType total = static_cast<SizeType>(totalHeads);
    const SizeType h1Split_ = static_cast<SizeType>(h1Split);
    (void)dim;

    // Eager-load every bundle node so the globalVID -> (node, local) reverse map is
    // fully populated before any search-time H1 GetSample resolution.
    for (size_t nodeId = 0; nodeId < m_headBundleNodes.size(); ++nodeId) {
        if (EnsureHeadBundleNodeLoaded(static_cast<int>(nodeId)) != ErrorCode::Success) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Metadata-only head root failed to load bundle node %zu.\n",
                         nodeId);
            return ErrorCode::Fail;
        }
    }

    kdt->SetMetadataOnly(total, h1Split_);

    // Precompute every H1 head's resolved bundle vector pointer ONCE. All bundle nodes
    // are eager-loaded above and never evicted, so these pointers are stable for the
    // lifetime of the index and the search-time resolver becomes a lock-free O(1) lookup.
    m_metaOnlyHeadVectorPtrs.assign(static_cast<size_t>(h1Split_), nullptr);
    for (SizeType hid = 0; hid < h1Split_ && hid < m_vectorTranslateMap.R(); ++hid) {
        SizeType globalVID = static_cast<SizeType>(*(m_vectorTranslateMap[hid]));
        if (globalVID == MaxSize && m_index->HasHeadNodeMeta()) {
            globalVID = m_index->GetHeadNodeGlobalVID(hid);
        }
        if (globalVID == MaxSize) continue;
        auto it = m_globalVIDToBundleLoc.find(globalVID);
        if (it == m_globalVIDToBundleLoc.end()) continue;
        const std::pair<int, SizeType>& loc = it->second;
        if (EnsureHeadBundleNodeLoaded(loc.first) != ErrorCode::Success) continue;
        auto& bidx = m_loadedHeadBundleIndexes[static_cast<size_t>(loc.first)];
        if (bidx) m_metaOnlyHeadVectorPtrs[static_cast<size_t>(hid)] = bidx->GetSample(loc.second);
    }
    kdt->SetExternalSampleResolver([this](SizeType hid) -> const void* {
        // hid in [0, h1Split): resolve the H1 head vector from the precomputed table.
        if (hid < 0 || hid >= static_cast<SizeType>(m_metaOnlyHeadVectorPtrs.size())) return nullptr;
        return m_metaOnlyHeadVectorPtrs[static_cast<size_t>(hid)];
    });

    m_metadataOnlyHeadStore = true;

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
        "Dual-pool slim head store enabled: total=%d H1Split=%d (U_extra from root, H1 from bundles).\n",
        static_cast<int>(total), static_cast<int>(h1Split_));
    return ErrorCode::Success;
}

template <typename T> ErrorCode Index<T>::LoadHeadCrossEdges() const
{
    if (m_headCrossEdgesDirty.load(std::memory_order_acquire)) {
        return ErrorCode::Success;
    }
    if (m_headCrossEdgesLoaded.load(std::memory_order_acquire)) {
        return ErrorCode::Success;
    }
    std::lock_guard<std::mutex> lock(m_headCrossEdgesMutex);
    if (m_headCrossEdgesDirty.load(std::memory_order_relaxed)) {
        m_headCrossEdgesLoaded.store(true, std::memory_order_release);
        return ErrorCode::Success;
    }
    if (m_headCrossEdgesLoaded.load(std::memory_order_relaxed)) {
        return ErrorCode::Success;
    }

    // The resolved adjacency is keyed by dense head-local-id B and target edges
    // are also stored as B ids. Ensure every bundle is loaded first so both
    // globalVID->B and B->(bundle node, local id) maps are complete.
    for (const auto& bundleNodeInfo : m_headBundleNodes) {
        if (EnsureHeadBundleNodeLoaded(bundleNodeInfo.nodeId) != ErrorCode::Success) {
            return ErrorCode::Fail;
        }
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
    const std::string dirtyPath = path + Helper::kHeadCrossEdgesDirtyFileName;
    FILE* dirtyFile = std::fopen(dirtyPath.c_str(), "rb");
    if (dirtyFile != nullptr) {
        std::fclose(dirtyFile);
        m_headCrossEdgesDirty.store(true, std::memory_order_release);
        m_headCrossEdgesLoaded.store(true, std::memory_order_release);
        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                     "[TaggedUpdate] ignoring stale cross-edge snapshot; rebuild it with augmentheadgraph --overwrite "
                     "after topology maintenance (%s).\n",
                     dirtyPath.c_str());
        return ErrorCode::Success;
    }
    path += Helper::kHeadCrossEdgesFileName;

    FILE* fp = std::fopen(path.c_str(), "rb");
    if (fp == nullptr) {
        m_headCrossEdgesLoaded.store(true, std::memory_order_release);
        return ErrorCode::Success;
    }
    const auto invalidate = [this]() {
        m_headCrossEdgeOffsetByB.clear();
        m_headCrossEdgeCountByB.clear();
        m_headCrossEdgeTargetsB.clear();
        m_headBundleNodeByB.clear();
        m_headBundleLocalByB.clear();
        m_headCrossEdgesLoaded.store(true, std::memory_order_release);
    };

    Helper::HeadCrossEdgesHeader header{};
    if (std::fread(&header, sizeof(header), 1, fp) != 1 ||
        header.magic != Helper::kHeadCrossEdgesMagic ||
        header.version != Helper::kHeadCrossEdgesVersion) {
        std::fclose(fp);
        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                     "head_cross_edges.bin format mismatch at %s — ignoring.\n", path.c_str());
        invalidate();
        return ErrorCode::Fail;
    }

    const SizeType headCount = (m_vectorTranslateMap.R() > 0)
        ? static_cast<SizeType>(m_vectorTranslateMap.R())
        : (m_index ? m_index->GetNumSamples() : 0);
    if (headCount <= 0) {
        std::fclose(fp);
        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                     "head_cross_edges.bin loaded before head id map is ready — ignoring.\n");
        invalidate();
        return ErrorCode::Fail;
    }
    if (header.totalHeads != headCount || header.maxEdgesPerHead < 0 ||
        header.searchTopK <= 0) {
        std::fclose(fp);
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Warning,
            "head_cross_edges.bin topology mismatch at %s: records=%d expected=%d M=%d K=%d.\n",
            path.c_str(), header.totalHeads, static_cast<int>(headCount),
            header.maxEdgesPerHead, header.searchTopK);
        invalidate();
        return ErrorCode::Fail;
    }

    m_headCrossEdgeOffsetByB.assign(static_cast<size_t>(headCount), UINT32_MAX);
    m_headCrossEdgeCountByB.assign(static_cast<size_t>(headCount), 0);
    m_headBundleNodeByB.assign(static_cast<size_t>(headCount), static_cast<std::int16_t>(-1));
    m_headBundleLocalByB.assign(static_cast<size_t>(headCount), static_cast<SizeType>(-1));

    for (size_t nodeId = 0; nodeId < m_headBundleLocalToGlobalHIDs.size(); ++nodeId) {
        const auto& localToB = m_headBundleLocalToGlobalHIDs[nodeId];
        for (SizeType local = 0; local < static_cast<SizeType>(localToB.size()); ++local) {
            SizeType b = localToB[static_cast<size_t>(local)];
            if (b >= 0 && b < headCount) {
                m_headBundleNodeByB[static_cast<size_t>(b)] = static_cast<std::int16_t>(nodeId);
                m_headBundleLocalByB[static_cast<size_t>(b)] = local;
            }
        }
    }

    std::lock_guard<std::mutex> headMapLock(m_globalHeadVIDToLocalHIDMutex);
    if (static_cast<SizeType>(m_globalHeadVIDToLocalHID.size()) != headCount) {
        m_globalHeadVIDToLocalHID.clear();
        m_globalHeadVIDToLocalHID.reserve(static_cast<size_t>(headCount) * 2 + 1);
        for (SizeType b = 0; b < headCount; ++b) {
            const SizeType globalVID = static_cast<SizeType>(*(m_vectorTranslateMap[b]));
            if (globalVID != MaxSize && globalVID >= 0) {
                m_globalHeadVIDToLocalHID[globalVID] = b;
            }
        }
    }

    size_t edgeReserve = 0;
    {
        struct stat st {};
        if (stat(path.c_str(), &st) == 0) {
            const size_t fixedBytes = sizeof(Helper::HeadCrossEdgesHeader)
                + static_cast<size_t>(header.totalHeads) * 2 * sizeof(std::int32_t);
            if (static_cast<size_t>(st.st_size) > fixedBytes) {
                edgeReserve = (static_cast<size_t>(st.st_size) - fixedBytes)
                    / sizeof(Helper::HeadCrossEdgeEntry);
            }
        }
    }
    m_headCrossEdgeTargetsB.clear();
    if (edgeReserve > 0) m_headCrossEdgeTargetsB.reserve(edgeReserve);

    bool ok = true;
    size_t nonEmpty = 0;
    size_t rawEdges = 0;
    size_t keptEdges = 0;
    std::vector<std::uint8_t> sourceSeen(static_cast<size_t>(headCount), 0);
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
        rawEdges += static_cast<size_t>(edgeCount);
        SizeType srcB = MaxSize;
        if (globalVID >= 0) {
            const auto source = m_globalHeadVIDToLocalHID.find(static_cast<SizeType>(globalVID));
            if (source != m_globalHeadVIDToLocalHID.end()) {
                srcB = source->second;
            }
        }
        if (srcB == MaxSize || srcB < 0 || srcB >= headCount ||
            sourceSeen[static_cast<size_t>(srcB)] != 0 ||
            m_headBundleNodeByB[static_cast<size_t>(srcB)] < 0) {
            ok = false;
            break;
        }
        sourceSeen[static_cast<size_t>(srcB)] = 1;
        const size_t begin = m_headCrossEdgeTargetsB.size();
        for (const auto& e : entries) {
            SizeType nbrGlobal = static_cast<SizeType>(e.neighborGlobalVID);
            if (nbrGlobal < 0) {
                ok = false;
                break;
            }
            const auto neighbor = m_globalHeadVIDToLocalHID.find(nbrGlobal);
            const SizeType nbrB =
                neighbor == m_globalHeadVIDToLocalHID.end() ? MaxSize : neighbor->second;
            if (nbrB == MaxSize || nbrB < 0 || nbrB >= headCount ||
                m_headBundleNodeByB[static_cast<size_t>(nbrB)] < 0 ||
                m_headBundleNodeByB[static_cast<size_t>(nbrB)] ==
                    m_headBundleNodeByB[static_cast<size_t>(srcB)]) {
                ok = false;
                break;
            }
            m_headCrossEdgeTargetsB.push_back(nbrB);
        }
        if (!ok) break;
        const size_t count = m_headCrossEdgeTargetsB.size() - begin;
        if (count > 0) {
            if (begin > UINT32_MAX || count > UINT16_MAX) {
                ok = false;
                break;
            }
            m_headCrossEdgeOffsetByB[static_cast<size_t>(srcB)] = static_cast<std::uint32_t>(begin);
            m_headCrossEdgeCountByB[static_cast<size_t>(srcB)] = static_cast<std::uint16_t>(count);
            ++nonEmpty;
            keptEdges += count;
        }
    }
    std::fclose(fp);
    if (ok && std::find(sourceSeen.begin(), sourceSeen.end(), 0) != sourceSeen.end()) {
        ok = false;
    }

    if (!ok) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                     "head_cross_edges.bin is incomplete or stale at %s — rejecting it.\n",
                     path.c_str());
        invalidate();
        return ErrorCode::Fail;
    }
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                 "Loaded head_cross_edges.bin resolved: records=%d nonEmpty=%zu rawEdges=%zu keptEdges=%zu "
                 "M=%d K=%d (dense B adjacency).\n",
                 header.totalHeads, nonEmpty, rawEdges, keptEdges,
                 header.maxEdgesPerHead, header.searchTopK);
    m_headCrossEdgesLoaded.store(true, std::memory_order_release);
    return ErrorCode::Success;
}

template <typename T> ErrorCode Index<T>::EnsureStaticTailCrossEdges()
{
    if (!m_options.m_buildCrossEdges || m_headBundleNodes.size() <= 1) {
        return ErrorCode::Success;
    }
    if (m_index == nullptr || m_vectorTranslateMap.R() == 0 ||
        m_loadedHeadBundleIndexes.size() != m_headBundleNodes.size() ||
        m_headBundleLocalToGlobalHIDs.size() != m_headBundleNodes.size()) {
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Error,
            "Cannot build STATIC cross edges before the bundle runtime and head-ID map are ready.\n");
        return ErrorCode::Fail;
    }

    if (!m_headCrossEdgesDirty.load(std::memory_order_acquire) &&
        LoadHeadCrossEdges() == ErrorCode::Success &&
        !m_headCrossEdgeTargetsB.empty()) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                     "Reusing validated pre-BuildSSD cross-edge sidecar.\n");
        return ErrorCode::Success;
    }

    std::vector<HeadCrossEdgeBuildNode> nodes;
    nodes.reserve(m_headBundleNodes.size());
    for (const auto& bundleNode : m_headBundleNodes) {
        if (bundleNode.headCount == 0) {
            continue;
        }
        const int nodeId = bundleNode.nodeId;
        if (nodeId < 0 || nodeId >= static_cast<int>(m_loadedHeadBundleIndexes.size()) ||
            EnsureHeadBundleNodeLoaded(nodeId) != ErrorCode::Success) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Cannot load bundle node %d for STATIC cross-edge construction.\n",
                         nodeId);
            return ErrorCode::Fail;
        }
        const size_t slot = static_cast<size_t>(nodeId);
        const auto& index = m_loadedHeadBundleIndexes[slot];
        const auto& localToGlobal = m_headBundleLocalToGlobalHIDs[slot];
        if (index == nullptr ||
            index->GetNumSamples() != static_cast<SizeType>(localToGlobal.size()) ||
            bundleNode.headCount < 0 || bundleNode.headCount > index->GetNumSamples()) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Invalid bundle node %d for STATIC cross-edge construction.\n",
                         nodeId);
            return ErrorCode::Fail;
        }
        nodes.push_back(
            {nodeId, bundleNode.headCount, index, &localToGlobal, &m_vectorTranslateMap});
    }
    if (nodes.empty()) {
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Error,
            "Cannot build STATIC cross edges without non-empty bundle nodes.\n");
        return ErrorCode::Fail;
    }

    std::string headDirectory = m_options.m_indexDirectory;
    if (!headDirectory.empty() && headDirectory.back() != FolderSep) headDirectory += FolderSep;
    headDirectory += m_options.m_headIndexFolder;
    const std::string outputPath =
        headDirectory + FolderSep + Helper::kHeadCrossEdgesFileName;
    const std::string dirtyPath =
        headDirectory + FolderSep + Helper::kHeadCrossEdgesDirtyFileName;
    const HeadCrossEdgeBuildOptions options{
        (std::max)(15, m_options.m_crossExtraEdges),
        (std::max)(1, m_options.m_crossExtraEdges),
        (std::max)(1, m_options.m_iSSDNumberOfThreads),
        true};

    {
        std::lock_guard<std::mutex> lock(m_headCrossEdgesMutex);
        m_headCrossEdgeOffsetByB.clear();
        m_headCrossEdgeCountByB.clear();
        m_headCrossEdgeTargetsB.clear();
        m_headBundleNodeByB.clear();
        m_headBundleLocalByB.clear();
        m_headCrossEdgesLoaded.store(false, std::memory_order_release);
        m_headCrossEdgesDirty.store(false, std::memory_order_release);
    }

    SPTAGLIB_LOG(
        Helper::LogLevel::LL_Info,
        "Building cross-edge sidecar before STATIC tail: nodes=%zu K=%d M=%d threads=%d.\n",
        nodes.size(), options.searchTopK, options.extraEdges, options.threads);
    if (!BuildHeadCrossEdges(nodes, outputPath, dirtyPath, options)) {
        return ErrorCode::Fail;
    }

    if (LoadHeadCrossEdges() != ErrorCode::Success || m_headCrossEdgeTargetsB.empty()) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "Pre-BuildSSD cross-edge sidecar did not resolve to usable adjacency.\n");
        return ErrorCode::Fail;
    }
    return ErrorCode::Success;
}

template <typename T>
bool Index<T>::SearchStaticTailCrossGraph(
    const T* p_target,
    int p_ownerNode,
    int p_candidateCount,
    std::vector<std::pair<SizeType, float>>& p_candidates) const
{
    p_candidates.clear();
    if (p_target == nullptr || p_ownerNode < 0 || p_candidateCount <= 0 ||
        m_headCrossEdgesDirty.load(std::memory_order_acquire) ||
        LoadHeadCrossEdges() != ErrorCode::Success || m_headCrossEdgeTargetsB.empty()) {
        return false;
    }

    QueryResult query(p_target, p_candidateCount, false);
    COMMON::QueryResultSet<T> results(p_target, p_candidateCount);
    int scanned = 0;
    const std::vector<int> ownerNode{p_ownerNode};
    if (CrossSubgraphGraphSearch(
            query, &results, ownerNode, nullptr, 0, p_candidateCount, scanned) !=
        ErrorCode::Success) {
        return false;
    }

    p_candidates.reserve(static_cast<size_t>(p_candidateCount));
    for (int rank = 0; rank < results.GetResultNum(); ++rank) {
        BasicResult* result = results.GetResult(rank);
        if (result == nullptr || result->VID < 0) break;
        p_candidates.emplace_back(result->VID, result->Dist);
    }
    return !p_candidates.empty();
}

// Helper: access RNG graph + samples uniformly for either KDT or BKT head-bundle index.
template <typename T>
struct HeadBundleAccess {
    const COMMON::RelativeNeighborhoodGraph* graph = nullptr;
    DimensionType nbrSize = 0;
    const void* sampleSource = nullptr;
    const T* (*getSample)(const void*, SizeType) = nullptr;
    SizeType numSamples = 0;
    bool valid = false;

    const T* GetSample(SizeType p_local) const
    {
        return getSample == nullptr ? nullptr : getSample(sampleSource, p_local);
    }
};

template <typename T>
static const T* GetBKTBundleSample(const void* p_source, SizeType p_local)
{
    const auto* index = static_cast<const BKT::Index<T>*>(p_source);
    return static_cast<const T*>(index->GetSample(p_local));
}

template <typename T>
static const T* GetKDTBundleSample(const void* p_source, SizeType p_local)
{
    const auto* index = static_cast<const KDT::Index<T>*>(p_source);
    return static_cast<const T*>(index->GetSample(p_local));
}

template <typename T>
static HeadBundleAccess<T> AccessHeadBundleIndex(VectorIndex* p_idx) {
    HeadBundleAccess<T> a;
    if (p_idx == nullptr) return a;
    if (auto* b = dynamic_cast<BKT::Index<T>*>(p_idx)) {
        a.graph = &b->GetGraph();
        a.nbrSize = b->GetNeighborhoodSize();
        a.sampleSource = b;
        a.getSample = &GetBKTBundleSample<T>;
        a.numSamples = b->GetNumSamples();
        a.valid = true;
    } else if (auto* k = dynamic_cast<KDT::Index<T>*>(p_idx)) {
        a.graph = &k->GetGraph();
        a.nbrSize = k->GetNeighborhoodSize();
        a.sampleSource = k;
        a.getSample = &GetKDTBundleSample<T>;
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
    if (m_headCrossEdgesDirty.load(std::memory_order_acquire) ||
        LoadHeadCrossEdges() != ErrorCode::Success || m_headCrossEdgeTargetsB.empty()) {
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
    const auto& entryL2G = m_headBundleLocalToGlobalHIDs[static_cast<size_t>(entryNode)];
    if (entryL2G.empty()) {
        return ErrorCode::Fail;
    }

    std::vector<HeadBundleAccess<T>> bundleAccesses(m_loadedHeadBundleIndexes.size());
    for (size_t nodeId = 0; nodeId < m_loadedHeadBundleIndexes.size(); ++nodeId) {
        bundleAccesses[nodeId] = AccessHeadBundleIndex<T>(m_loadedHeadBundleIndexes[nodeId].get());
    }
    if (!bundleAccesses[static_cast<size_t>(entryNode)].valid) {
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

    // Build routed node mask from candidateNodeSet.
    // Per-node byte allow-list (empty = no routing constraint). The previous
    // uint32 bitmask silently dropped any node id >= 32, which broke ablations
    // at nodeCount=64/256 (most queries ended up with no representable mask
    // → search either rejected every head or skipped routing entirely).
    std::vector<uint8_t> routedNodeMask;
    if (!p_candidateNodes.empty()) {
        int maxNid = -1;
        for (int nid : p_candidateNodes) if (nid > maxNid) maxNid = nid;
        if (maxNid >= 0) {
            routedNodeMask.assign(static_cast<size_t>(maxNid) + 1, 0);
            for (int nid : p_candidateNodes) {
                if (nid >= 0) routedNodeMask[static_cast<size_t>(nid)] = 1;
            }
        }
    }

    // Reuse BKT's per-thread compact deduper while preserving the prior
    // std::priority_queue tie behavior for this finite-budget traversal. Unlike
    // BKT's result-gated traversal, nonmatching heads remain valid bridges.
    // Keep the allocation large enough to retain the requested result set, but
    // never turn that allocation requirement into a larger traversal budget.
    const int maxChecks = std::max(1, m_options.m_maxCheck);
    const int workspaceCapacity = std::max(maxChecks, p_graphResultNum * 4);
    auto crossGraphWorkspace = m_crossGraphWorkSpaceFactory->GetWorkSpace();
    if (!crossGraphWorkspace) {
        crossGraphWorkspace.reset(new CrossGraphWorkSpace());
    }
    crossGraphWorkspace->Prepare(workspaceCapacity, p_graphResultNum, m_options.m_hashExp);

    size_t visitedCount = 0;
    auto pushKnownCandidate = [&](SizeType headB, float dist) -> bool {
        if (crossGraphWorkspace->CheckAndSet(headB)) return false;
        crossGraphWorkspace->PushFrontier(NodeDistPair(headB, dist));
        ++visitedCount;
        return true;
    };

    auto pushSampleCandidate = [&](SizeType headB, const T* sample) -> bool {
        if (sample == nullptr || crossGraphWorkspace->CheckAndSet(headB)) return false;
        const float dist = m_fComputeDistance(qTarget, sample, dim);
        crossGraphWorkspace->PushFrontier(NodeDistPair(headB, dist));
        ++visitedCount;
        return true;
    };

    // Phase 1: seed from EVERY routed node. Cross-edges alone are too sparse
    // a bridge when candidateNodes is small (e.g. 2-node dept queries) — the
    // entry-only seed leaves the other half of the routed scope unreachable.
    // We split the seed budget across all routed nodes and let each node's
    // BKT contribute its local nearest heads to the priority queue.
    //
    // The native MaxCheck budget also governs the seed search unless an
    // explicit SeedMaxCheck override is supplied.
    // Unfilter behaves like one logical graph search, so seed from the entry
    // bundle only and let dense cross edges carry the traversal. Filtered queries
    // seed every routed bundle to preserve per-filter coverage. CrossSingleSeed
    // keeps an explicit diagnostic override: -1 auto, 0 off, 1 on.
    const int crossSingleSeedOverride =
        (m_options.m_crossSingleSeed == 0 || m_options.m_crossSingleSeed == 1)
            ? m_options.m_crossSingleSeed
            : -1;
    const bool defaultSingleSeed = (p_queryTags == nullptr || p_numQueryTags == 0);
    const bool useSingleSeed = (crossSingleSeedOverride >= 0)
        ? (crossSingleSeedOverride == 1)
        : defaultSingleSeed;
    std::vector<int> seedNodes;
    if (useSingleSeed) {
        seedNodes.push_back(entryNode);
    } else {
        seedNodes = p_candidateNodes;
    }
    int totalSeedK = std::max(16, std::min(p_graphResultNum, 64));
    int perNodeSeed = std::max(4, totalSeedK / static_cast<int>(seedNodes.size()));
    const int seedMaxCheck = (m_options.m_seedMaxCheck > 0)
        ? m_options.m_seedMaxCheck
        : std::max(1, m_options.m_maxCheck);
    int scanned = 0;
    int totalSeeded = 0;
    int seedDroppedByTag = 0;
    int seedIndexMaxCheck = 0;

    auto _bktT0 = std::chrono::high_resolution_clock::now();

    for (int nodeId : seedNodes) {
        if (nodeId < 0 || nodeId >= static_cast<int>(m_loadedHeadBundleIndexes.size())) continue;
        const auto& nodeIdx = m_loadedHeadBundleIndexes[static_cast<size_t>(nodeId)];
        const auto& nodeL2G = m_headBundleLocalToGlobalHIDs[static_cast<size_t>(nodeId)];
        if (nodeIdx == nullptr || nodeL2G.empty()) continue;
        const auto& nodeAcc = bundleAccesses[static_cast<size_t>(nodeId)];
        if (!nodeAcc.valid) continue;

        int seedK = std::min(perNodeSeed, static_cast<int>(nodeL2G.size()));
        if (seedK <= 0) continue;
        COMMON::QueryResultSet<T> seedResults(qTarget, seedK);

        ErrorCode seedStatus = ErrorCode::Fail;
        if (auto* bkt = dynamic_cast<BKT::Index<T>*>(nodeIdx.get())) {
            seedIndexMaxCheck = std::max(seedIndexMaxCheck, bkt->GetCurrMaxCheck());
            seedStatus = bkt->SearchIndexWithFilter(seedResults, nullptr, seedMaxCheck);
        } else if (auto* kdt = dynamic_cast<KDT::Index<T>*>(nodeIdx.get())) {
            seedIndexMaxCheck = std::max(seedIndexMaxCheck, kdt->GetCurrMaxCheck());
            seedStatus = kdt->SearchIndexWithMaxCheck(seedResults, seedMaxCheck);
        }

        if (seedStatus != ErrorCode::Success) continue;
        scanned += seedResults.GetScanned();

        for (int i = 0; i < seedResults.GetResultNum(); ++i) {
            auto* r = seedResults.GetResult(i);
            if (r == nullptr || r->VID < 0) continue;
            if (static_cast<size_t>(r->VID) >= nodeL2G.size()) continue;
            SizeType bundleLocal = r->VID;
            SizeType m_idx_B = nodeL2G[static_cast<size_t>(bundleLocal)];
            pushKnownCandidate(m_idx_B, r->Dist);
            ++totalSeeded;
        }
    }

    if (totalSeeded == 0) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "CrossSubgraph DBG: totalSeeded=0 nodes=%zu seedMaxCheck=%d perNodeSeed=%d\n",
                     p_candidateNodes.size(), seedMaxCheck, perNodeSeed);
        m_crossGraphWorkSpaceFactory->ReturnWorkSpace(std::move(crossGraphWorkspace));
        return ErrorCode::Fail;
    }

    auto _bktT1 = std::chrono::high_resolution_clock::now();
    g_bktSeedMs = std::chrono::duration<double, std::milli>(_bktT1 - _bktT0).count();

    int checks = 0;
    int crossHopCount = 0;
    int crossEdgesSeen = 0;
    int crossDroppedByTag = 0;
    // Dual-pool U_extra diagnostics: count how often the unfilter traversal
    // reaches role==1 (U_extra) heads.
    const bool logUExtra = m_options.m_logUExtra;
    const bool s_hasRoles = HasHeadRoles();
    int uextraCrossTargets = 0;  // cross-edges pointing at a U_extra head
    int uextraChecked = 0;       // U_extra heads popped/visited in the PQ
    int uextraKept = 0;          // U_extra heads committed to the result heap

    // Match the original BKT termination rule: once the requested result heap
    // can be full, let worstDist stop a settled frontier. The old 256-pop floor
    // was custom to the cross path and forced extra graph work at TopK=10.
    const int minChecks = p_graphResultNum;

    while (!crossGraphWorkspace->FrontierEmpty() && checks < maxChecks) {
        const NodeDistPair cur = crossGraphWorkspace->PopFrontier();
        const SizeType curHeadB = cur.node;

        // Nodes are deduped at push time, so every popped candidate is unique and
        // expanded exactly once (no stale-duplicate skip needed here).
        ++checks;
        const bool curIsUExtra = logUExtra && s_hasRoles && m_extraSearcher &&
            m_extraSearcher->IsUnfilterOnlyHead(static_cast<int>(curHeadB));
        if (curIsUExtra) ++uextraChecked;

        // Early termination: once we've done minChecks and the closest remaining
        // PQ candidate is already worse than the worst kept top-graphResultNum
        // distance, no future candidate can improve the result heap. This is the
        // same idea KDT uses (gnode.distance > p_query.worstDist()) but applied
        // to the cross-subgraph unified PQ.
        if (checks >= minChecks && cur.distance > p_queryResults->worstDist()) {
            break;
        }

        // Push to result heap. AddPoint uses headB (m_index local hid space) to
        // match the existing post-graph code path that translates B -> tenant
        // data globalVID downstream via translateHeadVID.
        // Tag-aware in-filter: only commit head as a result (and thus its posting
        // for I/O) if its hier_mask intersects the query mask. Continue walking
        // (RNG / cross-edge expansion) regardless so navigation isn't cut.
        bool keepHead = true;
        if (p_queryTags != nullptr && p_numQueryTags > 0 && m_index->HasHeadNodeMeta()) {
            keepHead = m_index->HeadPostingHierMaskMayIntersect(curHeadB, queryHierMask);
        }
        if (keepHead) {
            p_queryResults->AddPoint(curHeadB, cur.distance);
            if (curIsUExtra) ++uextraKept;
        }

        // Expand RNG neighbors within the home node's head index (BKT or KDT).
        if (curHeadB >= 0 && static_cast<size_t>(curHeadB) < m_headBundleNodeByB.size()) {
            const int homeNodeId =
                static_cast<int>(m_headBundleNodeByB[static_cast<size_t>(curHeadB)]);
            const SizeType homeLocal = m_headBundleLocalByB[static_cast<size_t>(curHeadB)];
            if (homeNodeId >= 0 &&
                homeNodeId < static_cast<int>(bundleAccesses.size())) {
                const auto& homeL2G =
                    m_headBundleLocalToGlobalHIDs[static_cast<size_t>(homeNodeId)];
                const auto& homeAcc = bundleAccesses[static_cast<size_t>(homeNodeId)];
                if (homeAcc.valid && homeLocal >= 0 && homeLocal < homeAcc.numSamples) {
                    const SizeType* nbrs = (*homeAcc.graph)[homeLocal];
                    COMMON::PrefetchGraphNeighbors(
                        nbrs, homeAcc.nbrSize,
                        [&homeAcc](SizeType neighbor) -> const void* {
                            return neighbor >= 0 && neighbor < homeAcc.numSamples
                                ? homeAcc.GetSample(neighbor)
                                : nullptr;
                        });
                    for (DimensionType i = 0; i < homeAcc.nbrSize; ++i) {
                        SizeType nbrLocal = nbrs[i];
                        if (nbrLocal < 0) break;
                        if (nbrLocal >= homeAcc.numSamples) continue;
                        if (static_cast<size_t>(nbrLocal) >= homeL2G.size()) continue;
                        SizeType nbr_B = homeL2G[static_cast<size_t>(nbrLocal)];
                        if (nbr_B < 0 || crossGraphWorkspace->Contains(nbr_B)) continue;
                        pushSampleCandidate(nbr_B, homeAcc.GetSample(nbrLocal));
                    }
                }
            }
        }

        // Expand cross-subgraph edges. Cross-edges store space A (tenant data
        // global VID). Translate the source key from B to A via m_index meta.
        //
        // Policy (2026-05): in filter mode (p_queryTags != nullptr && p_numQueryTags > 0)
        // we skip cross-edge expansion by default and rely solely on intra-bundle
        // RNG navigation seeded from every routed bundle (Phase 1). Rationale:
        //   - routedNodeMask already restricts cross neighbors to routed bundles,
        //     so cross edges only act as in-scope navigational shortcuts.
        //   - filter recall is governed by (a) routing completeness and (b) per-
        //     bundle RNG connectivity from its own seed, neither of which depends
        //     on cross edges.
        //   - Removing the cross-edge branch cuts a hot lookup + hier-mask check
        //     per candidate, and reduces visited-set / PQ churn under heavy
        //     fan-out subgraphs.
        // FilterKeepCross restores the previous behavior (filter walks cross
        // edges, gated by routedNodeMask + signature) for A/B comparison.
        // DisableCrossEdges still disables cross edges globally.
        const bool disableCross = m_options.m_disableCrossEdges;
        const bool filterKeepCross = m_options.m_filterKeepCross;
        // Cross-edge expansion is memory-scattered and dominates the walk cost.
        // Cross-edges emanating from the far (late-popped) candidates rarely
        // introduce new top-graphResultNum heads. Gate expansion to the nearest
        // N popped candidates (candidates pop nearest-first from the min-heap).
        // CrossExpandLimit=0 retains unlimited legacy behavior.
        const int crossExpandLimit = std::max(0, m_options.m_crossExpandLimit);
        const bool filterMode = (p_queryTags != nullptr && p_numQueryTags > 0);
        const bool withinCrossBudget = (crossExpandLimit <= 0) || (checks <= crossExpandLimit);
        const bool useCrossHere = !disableCross && (!filterMode || filterKeepCross) && withinCrossBudget;
        if (useCrossHere && curHeadB >= 0 &&
            static_cast<size_t>(curHeadB) < m_headCrossEdgeOffsetByB.size()) {
            std::uint32_t off = m_headCrossEdgeOffsetByB[static_cast<size_t>(curHeadB)];
            std::uint16_t cnt = m_headCrossEdgeCountByB[static_cast<size_t>(curHeadB)];
            if (off != UINT32_MAX && cnt > 0) {
                const SizeType* crossNeighbors =
                    m_headCrossEdgeTargetsB.data() + static_cast<size_t>(off);
                _mm_prefetch((const char*)crossNeighbors, _MM_HINT_T0);
                for (std::uint16_t ei = 0; ei < cnt; ++ei) {
                    const SizeType nbrB = crossNeighbors[ei];
                    if (nbrB < 0 || static_cast<size_t>(nbrB) >= m_headBundleNodeByB.size()) continue;
                    const int nodeId =
                        static_cast<int>(m_headBundleNodeByB[static_cast<size_t>(nbrB)]);
                    const SizeType local = m_headBundleLocalByB[static_cast<size_t>(nbrB)];
                    if (nodeId < 0 || nodeId >= static_cast<int>(bundleAccesses.size())) continue;
                    const auto& access = bundleAccesses[static_cast<size_t>(nodeId)];
                    if (!access.valid || local < 0 || local >= access.numSamples) continue;
                    const T* sample = access.GetSample(local);
                    if (sample != nullptr) {
                        _mm_prefetch((const char*)sample, _MM_HINT_T0);
                    }
                }

                for (std::uint16_t ei = 0; ei < cnt; ++ei) {
                    const SizeType nbrB = crossNeighbors[ei];
                    if (nbrB < 0) continue;
                    ++crossEdgesSeen;
                    if (logUExtra && s_hasRoles && m_extraSearcher &&
                        m_extraSearcher->IsUnfilterOnlyHead(static_cast<int>(nbrB))) {
                        ++uextraCrossTargets;
                    }
                    if (crossGraphWorkspace->Contains(nbrB)) continue;

                    // Hierarchical mask filter: skip neighbors that don't match query.
                    if (p_queryTags != nullptr && p_numQueryTags > 0 && m_index->HasHeadNodeMeta()) {
                        if (!routedNodeMask.empty()) {
                            const int16_t bundleNodeId = m_index->GetHeadNodeBundleNodeId(nbrB);
                            if (bundleNodeId >= 0 &&
                                (static_cast<size_t>(bundleNodeId) >= routedNodeMask.size() ||
                                 routedNodeMask[static_cast<size_t>(bundleNodeId)] == 0)) {
                                ++crossDroppedByTag;
                                continue;
                            }
                        }
                        if (!m_index->HeadPostingHierMaskMayIntersect(nbrB, queryHierMask)) {
                            ++crossDroppedByTag;
                            continue;
                        }
                    }

                    if (static_cast<size_t>(nbrB) >= m_headBundleNodeByB.size()) continue;
                    const int nodeId =
                        static_cast<int>(m_headBundleNodeByB[static_cast<size_t>(nbrB)]);
                    const SizeType local = m_headBundleLocalByB[static_cast<size_t>(nbrB)];
                    if (nodeId < 0 || nodeId >= static_cast<int>(bundleAccesses.size())) continue;
                    const auto& access = bundleAccesses[static_cast<size_t>(nodeId)];
                    if (!access.valid || local < 0 || local >= access.numSamples) continue;
                    if (pushSampleCandidate(nbrB, access.GetSample(local))) {
                        ++crossHopCount;
                    }
                }
            }
        }
    }

    p_queryResults->SortResult();
    p_scannedOut = scanned + checks;

    if (logUExtra) {
        int uextraInFinal = 0;
        int finalCount = p_queryResults->GetResultNum();
        for (int i = 0; i < finalCount; ++i) {
            auto* r = p_queryResults->GetResult(i);
            if (r == nullptr || r->VID < 0) continue;
            if (s_hasRoles && m_extraSearcher &&
                m_extraSearcher->IsUnfilterOnlyHead(static_cast<int>(r->VID))) ++uextraInFinal;
        }
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
            "UExtraDBG: hasRoles=%d filterMode=%d crossSeen=%d uextraCrossTargets=%d "
            "checks=%d uextraChecked=%d uextraKept=%d uextraInFinalHeads=%d/%d\n",
            (int)s_hasRoles, (p_numQueryTags > 0) ? 1 : 0,
            crossEdgesSeen, uextraCrossTargets,
            checks, uextraChecked, uextraKept, uextraInFinal, finalCount);
    }

    if (m_options.m_logAdaptiveNprobe) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
            "CrossSubgraph: nodes=%zu totalSeeded=%d checks=%d crossHops=%d nodesVisited=%zu "
            "crossEdgesSeen=%d crossDroppedByTag=%d (tagPass=%.3f)\n",
            p_candidateNodes.size(), totalSeeded, checks, crossHopCount, visitedCount,
            crossEdgesSeen, crossDroppedByTag,
            crossEdgesSeen > 0 ? 1.0 - (double)crossDroppedByTag / crossEdgesSeen : 0.0);
    }
    {
        auto _pqT1 = std::chrono::high_resolution_clock::now();
        g_pqGraphMs = std::chrono::duration<double, std::milli>(_pqT1 - _bktT1).count();
    }
    if (m_options.m_logPhaseTime || m_options.m_logCrossStats) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
            "CSStats: nodes=%zu seedNodes=%zu seeded=%d seedScanned=%d seedBudget=%d "
            "seedConfiguredMaxC=%d seedIndexMaxC=%d seedDropTag=%d checks=%d cross=%d "
            "crossDropTag=%d visited=%zu maxC=%d\n",
            p_candidateNodes.size(), seedNodes.size(), totalSeeded, scanned, seedMaxCheck,
            m_options.m_seedMaxCheck, seedIndexMaxCheck, seedDroppedByTag, checks,
            crossEdgesSeen, crossDroppedByTag, visitedCount, maxChecks);
    }
    m_crossGraphWorkSpaceFactory->ReturnWorkSpace(std::move(crossGraphWorkspace));
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

    // Preserve and apply the documented runtime section after construction
    // settings. SetParameter maps its aliases to mutable SSD options.
    for (const auto& entry : p_reader.GetParameters("SearchSSDIndex"))
    {
        SetParameter(entry.first.c_str(), entry.second.c_str(), "SearchSSDIndex");
    }

    const std::string metadataRootSidecar = JoinPath(
        JoinPath(m_options.m_indexDirectory, m_options.m_headIndexFolder),
        "head_metaonly.bin");
    if (fileexists(metadataRootSidecar.c_str()))
    {
        // The parent SPANN configuration retains the original BKT algorithm so bundle
        // graphs reload as BKT. The persisted root itself is intentionally a tiny KDT.
        auto metadataRoot = CreateInstance(IndexAlgoType::KDT, valueType);
        if (metadataRoot == nullptr)
            return ErrorCode::FailedParseValue;
        metadataRoot->SetParameter(
            "DistCalcMethod",
            SPTAG::Helper::Convert::ConvertToString(m_options.m_distCalcMethod));
        m_index = std::move(metadataRoot);
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
    if (m_options.m_recovery) {
        const std::string headIDPath =
            m_options.m_persistentBufferPath + FolderSep + m_options.m_headIDFile;
        std::shared_ptr<Helper::DiskIO> headIDStream = SPTAG::f_createIO();
        if (headIDStream == nullptr ||
            !headIDStream->Initialize(headIDPath.c_str(), std::ios::binary | std::ios::in) ||
            m_vectorTranslateMap.Load(headIDStream,
                                      m_options.m_datasetRowsInBlock,
                                      m_options.m_datasetCapacity) != ErrorCode::Success) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Cannot load checkpoint head-ID map %s.\n", headIDPath.c_str());
            return ErrorCode::Fail;
        }
    } else if (m_vectorTranslateMap.Load(
                   p_indexStreams[m_index->GetIndexFiles()->size()],
                   m_options.m_datasetRowsInBlock,
                   m_options.m_datasetCapacity) != ErrorCode::Success) {
        return ErrorCode::Fail;
    }

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

    if (SetupMetadataOnlyHeadStore(bundleBaseDir) != ErrorCode::Success)
        return ErrorCode::Fail;

    return ErrorCode::Success;
}

template <typename T>
bool Index<T>::BuildPrimaryHeadCSRBackfill(const void* vectors, SizeType vectorCount,
                                           const uint32_t* tags, int numTagsPerVec)
{
    if (vectors == nullptr || tags == nullptr || vectorCount <= 0 || numTagsPerVec < 5 ||
        m_headBundleNodes.empty() || m_loadedHeadBundleIndexes.empty()) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "[PrimaryHeadCSR] backfill requires vectors, five tags, and loaded head bundles.\n");
        return false;
    }

    const SizeType headCount = TotalHeadSampleCount();
    if (headCount <= 0 || headCount > std::numeric_limits<std::uint32_t>::max()) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[PrimaryHeadCSR] invalid head count for backfill.\n");
        return false;
    }

    std::uint32_t tagBases[4] = {
        std::numeric_limits<std::uint32_t>::max(),
        std::numeric_limits<std::uint32_t>::max(),
        std::numeric_limits<std::uint32_t>::max(),
        std::numeric_limits<std::uint32_t>::max()
    };
    for (SizeType vid = 0; vid < vectorCount; ++vid) {
        const uint32_t* row = tags + static_cast<size_t>(vid) * numTagsPerVec;
        for (int level = 0; level < 4; ++level) tagBases[level] = std::min(tagBases[level], row[level]);
    }
    for (SizeType vid = 0; vid < vectorCount; ++vid) {
        const uint32_t* row = tags + static_cast<size_t>(vid) * numTagsPerVec;
        for (int level = 0; level < 4; ++level) {
            if (row[level] - tagBases[level] > 0xffU) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                             "[PrimaryHeadCSR] categorical level %d exceeds uint8 packing range.\n", level);
                return false;
            }
        }
    }

    // The persisted SIFT bundle layout routes level-0 (org) tag 0..3 to
    // bundle node 0..3. Refuse a schema that cannot satisfy that invariant.
    std::vector<int> bundleForOrg(256, -1);
    for (size_t bundle = 0; bundle < m_headBundleNodes.size(); ++bundle) {
        const int nodeId = m_headBundleNodes[bundle].nodeId;
        if (nodeId >= 0 && nodeId < static_cast<int>(bundleForOrg.size())) {
            bundleForOrg[static_cast<size_t>(nodeId)] = nodeId;
        }
    }

    std::vector<std::uint32_t> primaryHeads(static_cast<size_t>(vectorCount),
                                            std::numeric_limits<std::uint32_t>::max());
    std::atomic<SizeType> nextVID(0);
    std::atomic<bool> failed(false);
    const T* typedVectors = reinterpret_cast<const T*>(vectors);
    const int workers = std::max(1, m_options.m_iSSDNumberOfThreads);
    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (int worker = 0; worker < workers; ++worker) {
        threads.emplace_back([&, worker]() {
            (void)worker;
            while (!failed.load(std::memory_order_relaxed)) {
                const SizeType vid = nextVID.fetch_add(1, std::memory_order_relaxed);
                if (vid >= vectorCount) return;
                const uint32_t* row = tags + static_cast<size_t>(vid) * numTagsPerVec;
                const std::uint32_t org = row[0] - tagBases[0];
                if (org >= bundleForOrg.size()) {
                    failed.store(true, std::memory_order_relaxed);
                    return;
                }
                const int nodeId = bundleForOrg[org];
                if (nodeId < 0 || nodeId >= static_cast<int>(m_loadedHeadBundleIndexes.size()) ||
                    m_loadedHeadBundleIndexes[static_cast<size_t>(nodeId)] == nullptr ||
                    m_headBundleLocalToGlobalHIDs[static_cast<size_t>(nodeId)].empty()) {
                    failed.store(true, std::memory_order_relaxed);
                    return;
                }

                COMMON::QueryResultSet<T> result(
                    typedVectors + static_cast<size_t>(vid) * m_options.m_dim, 1);
                if (m_loadedHeadBundleIndexes[static_cast<size_t>(nodeId)]->SearchIndex(result) != ErrorCode::Success) {
                    failed.store(true, std::memory_order_relaxed);
                    return;
                }
                BasicResult* nearest = result.GetResult(0);
                if (nearest == nullptr || nearest->VID < 0 ||
                    nearest->VID >= static_cast<SizeType>(
                        m_headBundleLocalToGlobalHIDs[static_cast<size_t>(nodeId)].size())) {
                    failed.store(true, std::memory_order_relaxed);
                    return;
                }
                primaryHeads[static_cast<size_t>(vid)] = static_cast<std::uint32_t>(
                    m_headBundleLocalToGlobalHIDs[static_cast<size_t>(nodeId)]
                                                   [static_cast<size_t>(nearest->VID)]);

                if ((vid & ((1 << 20) - 1)) == 0) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                 "[PrimaryHeadCSR] assignment %d/%d\n",
                                 static_cast<int>(vid), static_cast<int>(vectorCount));
                }
            }
        });
    }
    for (auto& thread : threads) thread.join();
    if (failed.load(std::memory_order_relaxed)) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[PrimaryHeadCSR] nearest-head assignment failed.\n");
        return false;
    }

    std::vector<std::uint32_t> offsets(static_cast<size_t>(headCount) + 1, 0);
    for (std::uint32_t head : primaryHeads) {
        if (head >= headCount || offsets[static_cast<size_t>(head) + 1] == std::numeric_limits<std::uint32_t>::max()) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[PrimaryHeadCSR] invalid primary head assignment.\n");
            return false;
        }
        ++offsets[static_cast<size_t>(head) + 1];
    }
    for (SizeType head = 0; head < headCount; ++head) {
        const std::uint64_t next = static_cast<std::uint64_t>(offsets[static_cast<size_t>(head)]) +
                                   offsets[static_cast<size_t>(head) + 1];
        if (next > std::numeric_limits<std::uint32_t>::max()) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[PrimaryHeadCSR] offset overflow.\n");
            return false;
        }
        offsets[static_cast<size_t>(head) + 1] = static_cast<std::uint32_t>(next);
    }

    std::unique_ptr<std::atomic_uint32_t[]> cursors(
        new std::atomic_uint32_t[static_cast<size_t>(headCount)]);
    for (SizeType head = 0; head < headCount; ++head) {
        cursors[static_cast<size_t>(head)].store(offsets[static_cast<size_t>(head)], std::memory_order_relaxed);
    }
    std::vector<PrimaryHeadCSREntry> entries(static_cast<size_t>(vectorCount));
    nextVID.store(0, std::memory_order_relaxed);
    threads.clear();
    for (int worker = 0; worker < workers; ++worker) {
        threads.emplace_back([&, worker]() {
            (void)worker;
            while (true) {
                const SizeType vid = nextVID.fetch_add(1, std::memory_order_relaxed);
                if (vid >= vectorCount) return;
                const std::uint32_t head = primaryHeads[static_cast<size_t>(vid)];
                const std::uint32_t pos =
                    cursors[static_cast<size_t>(head)].fetch_add(1, std::memory_order_relaxed);
                const uint32_t* row = tags + static_cast<size_t>(vid) * numTagsPerVec;
                std::uint32_t packedTags = 0;
                for (int level = 0; level < 4; ++level) {
                    packedTags |= ((row[level] - tagBases[level]) & 0xffU) << (level * 8);
                }
                entries[pos].vid = static_cast<std::uint32_t>(vid);
                entries[pos].attributes = static_cast<std::uint64_t>(packedTags) |
                                          (static_cast<std::uint64_t>(row[4]) << 32);
            }
        });
    }
    for (auto& thread : threads) thread.join();

    PrimaryHeadCSRHeader header;
    header.headCount = static_cast<std::uint32_t>(headCount);
    header.entryCount = static_cast<std::uint64_t>(vectorCount);
    for (int level = 0; level < 4; ++level) header.tagBases[level] = tagBases[level];
    const std::string path = m_options.m_indexDirectory + FolderSep + m_options.m_primaryHeadCSRFile;
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[PrimaryHeadCSR] cannot create %s.\n", path.c_str());
        return false;
    }
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    output.write(reinterpret_cast<const char*>(offsets.data()),
                 static_cast<std::streamsize>(offsets.size() * sizeof(std::uint32_t)));
    output.write(reinterpret_cast<const char*>(entries.data()),
                 static_cast<std::streamsize>(entries.size() * sizeof(PrimaryHeadCSREntry)));
    output.close();
    if (!output) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[PrimaryHeadCSR] write failed for %s.\n", path.c_str());
        return false;
    }
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                 "[PrimaryHeadCSR] backfilled %d vectors across %d heads to %s.\n",
                 static_cast<int>(vectorCount), static_cast<int>(headCount), path.c_str());
    return true;
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
    auto buildSSDParameterValue = [this](const char* p_name, const std::string& p_fallback) {
        for (const auto& parameter : m_buildSSDParameters)
        {
            if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(parameter.first.c_str(), p_name))
            {
                return parameter.second;
            }
        }
        return p_fallback;
    };
    IOSTRING(p_configOut, WriteString, "[BuildSSDIndex]\n");
#define DefineSSDParameter(VarName, VarType, DefaultValue, RepresentStr)                                               \
    IOSTRING(p_configOut, WriteString,                                                                                 \
             (std::string(RepresentStr) + std::string("=") +                                                          \
              buildSSDParameterValue(RepresentStr, SPTAG::Helper::Convert::ConvertToString(m_options.VarName)) +       \
              std::string("\n"))                                                                                       \
                 .c_str());

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineSSDParameter

    if (!m_searchSSDParameters.empty())
    {
        IOSTRING(p_configOut, WriteString, "\n[SearchSSDIndex]\n");
        for (const auto& parameter : m_searchSSDParameters)
        {
            const std::string line = parameter.first + "=" + parameter.second + "\n";
            IOSTRING(p_configOut, WriteString, line.c_str());
        }
    }

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

    std::unique_lock<std::mutex> dataLock(m_dataAddLock);
    ErrorCode ret;
    if ((ret = DrainTaggedMergeMaintenance()) != ErrorCode::Success)
        return ret;

    if ((ret = m_index->SaveIndexData(p_indexStreams)) != ErrorCode::Success)
        return ret;

    if ((ret = m_vectorTranslateMap.Save(p_indexStreams[m_index->GetIndexFiles()->size()])) != ErrorCode::Success)
        return ret;

    if ((ret = m_extraSearcher->Checkpoint(m_options.m_indexDirectory)) != ErrorCode::Success)
        return ret;

    if ((ret = SaveLoadedHeadBundles(m_options.m_indexDirectory)) != ErrorCode::Success)
        return ret;
    return ErrorCode::Success;
}

#pragma region K - NN search

template <typename T> ErrorCode Index<T>::SearchIndex(QueryResult &p_query, bool p_searchDeleted) const
{
    if (!m_bReady)
        return ErrorCode::EmptyIndex;

    std::shared_lock<std::shared_timed_mutex> topologyLock(m_headTopologyLock);
    SPTAG::VectorIndex::ResetThreadLocalPostingScanStats();

    const auto* threadLocalSearchContext = SPTAG::VectorIndex::GetThreadLocalSearchContext();
    static const std::vector<SizeType> kEmptyDirectPostingIDs;
    static const std::function<bool(int)> kEmptyPostingFilter;
    const std::vector<SizeType>& directPostingIDs = threadLocalSearchContext != nullptr
        ? threadLocalSearchContext->m_directPostingIDs
        : kEmptyDirectPostingIDs;
    const std::vector<SizeType>& directHeadLocalIDs = threadLocalSearchContext != nullptr
        ? threadLocalSearchContext->m_directHeadLocalIDs
        : kEmptyDirectPostingIDs;
    const uint32_t* queryTags = threadLocalSearchContext != nullptr
        ? threadLocalSearchContext->QueryTags()
        : nullptr;
    const int numQueryTags = threadLocalSearchContext != nullptr
        ? threadLocalSearchContext->NumQueryTags()
        : 0;
    const SPTAG::Cache::DNFPredicate* queryDNF = threadLocalSearchContext != nullptr
        ? threadLocalSearchContext->DNF()
        : nullptr;
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
        if (directPostingIDs.size() > static_cast<size_t>((std::numeric_limits<int>::max)()) ||
            directHeadLocalIDs.size() > static_cast<size_t>((std::numeric_limits<int>::max)())) {
            return ErrorCode::Fail;
        }
        auto workSpace = m_workSpaceFactory->GetWorkSpace();
        if (!workSpace) {
            workSpace.reset(new ExtraWorkSpace());
            m_extraSearcher->InitWorkSpace(workSpace.get(), false);
        } else {
            m_extraSearcher->InitWorkSpace(workSpace.get(), true);
        }
        workSpace->m_queryTags = queryTags;
        workSpace->m_numQueryTags = numQueryTags;
        workSpace->m_dnf = queryDNF;
        workSpace->m_deduper.clear();
        workSpace->m_postingIDs.clear();
        workSpace->m_postingFilter = nullptr;  // no PS needed, we know exact postings
        workSpace->m_postingProbeStats.Reset();

        const int directPostingCount = static_cast<int>(directPostingIDs.size());
        if (directPostingCount > m_options.m_searchInternalResultNum) {
            const bool isStaticStorage = m_options.m_storage == Storage::STATIC;
            const int maxPages = isStaticStorage
                ? (std::max)(m_options.m_postingPageLimit, m_options.m_searchPostingPageLimit + 1) << PageSizeEx
                : ((std::max)(m_options.m_postingPageLimit, m_options.m_searchPostingPageLimit)
                    + m_options.m_bufferLength + m_options.m_unfilterTailBufferLength) << PageSizeEx;
            // ExtraStaticSearcher indexes m_diskRequests by posting ordinal, so
            // its workspace must retain one request per posting rather than the
            // dynamic block-I/O request layout.
            workSpace->Clear(directPostingCount, maxPages, !isStaticStorage,
                             m_options.m_enableDataCompression);
            // Clear() may have allocated new requests; initialize their static
            // I/O context IDs before the scan.
            m_extraSearcher->InitWorkSpace(workSpace.get(), true);
        }

        const int directResultNum = (std::max)(
            m_options.m_searchInternalResultNum,
            static_cast<int>(directHeadLocalIDs.size()));
        COMMON::QueryResultSet<T> *p_queryResults;
        if (p_query.GetResultNum() >= directResultNum)
            p_queryResults = (COMMON::QueryResultSet<T> *)&p_query;
        else
            p_queryResults = new COMMON::QueryResultSet<T>((const T *)p_query.GetTarget(), directResultNum);

        if (!directHeadLocalIDs.empty() && m_index != nullptr)
        {
            const bool hasTagFilter = queryTags != nullptr && numQueryTags > 0;
            SPTAG::Cache::HierarchicalPostingMask queryHierMask;
            if (hasTagFilter) {
                queryHierMask.Clear();
                for (int i = 0; i < numQueryTags; ++i) {
                    queryHierMask.Insert(TagLevelFromId(queryTags[i]), queryTags[i]);
                }
            }

            auto translateHeadVID = [&](SizeType localHid) -> SizeType {
                if (localHid < 0 || localHid >= m_index->GetNumSamples()) return MaxSize;
                if (m_index->HasHeadNodeMeta()) {
                    SizeType metaVID = m_index->GetHeadNodeGlobalVID(localHid);
                    if (metaVID != MaxSize) return metaVID;
                }
                if (localHid < static_cast<SizeType>(m_vectorTranslateMap.R()))
                    return static_cast<SizeType>(*(m_vectorTranslateMap[localHid]));
                return MaxSize;
            };
            auto shouldKeepHeadResult = [&](SizeType localHid) -> bool {
                if (!hasTagFilter) return true;
                static const std::vector<uint8_t> kNoRouteMask;
                return m_index->HasHeadNodeMeta() &&
                    m_index->HeadNodeMatchesQuery(localHid, queryHierMask, kNoRouteMask);
            };

            for (SizeType localHid : directHeadLocalIDs) {
                if (localHid < 0 || localHid >= m_index->GetNumSamples() ||
                    !shouldKeepHeadResult(localHid)) {
                    continue;
                }
                const void* headSample = m_index->GetSample(localHid);
                const SizeType globalVID = translateHeadVID(localHid);
                if (headSample == nullptr || globalVID == MaxSize || m_versionMap.Deleted(globalVID) ||
                    workSpace->m_deduper.CheckAndSet(globalVID)) {
                    continue;
                }
                const float distance = m_index->ComputeDistance(
                    p_queryResults->GetQuantizedTarget(), headSample);
                p_queryResults->AddPoint(globalVID, distance);
            }
        }

        // Directly inject all target posting IDs for sparse brute-force.
        int maxPostings = directPostingCount;
        for (SizeType pid : directPostingIDs) {
            if ((int)workSpace->m_postingIDs.size() >= maxPostings) break;
            if (m_extraSearcher->CheckValidPosting(pid)) {
                workSpace->m_postingIDs.emplace_back(pid);
            }
        }

        // Read postings and scan with inline tag filter
        ErrorCode ret = m_extraSearcher->SearchIndex(workSpace.get(), *p_queryResults,
                                                     m_index, nullptr, nullptr, nullptr);
        SPTAG::VectorIndex::SetThreadLocalPostingScanStats(
            workSpace->m_postingProbeStats.m_readPostings,
            workSpace->m_postingProbeStats.m_matchedPostings,
            workSpace->m_postingProbeStats.m_prePSPostings,
            workSpace->m_postingProbeStats.m_scannedVectors,
            workSpace->m_postingProbeStats.m_matchedVectors,
            workSpace->m_postingProbeStats.m_primaryHeadCandidates,
            workSpace->m_postingProbeStats.m_postingPageReads,
            workSpace->m_postingProbeStats.m_postingLogicalBytes,
            workSpace->m_postingProbeStats.m_postingPhysicalBytes,
            workSpace->m_postingProbeStats.m_adcScannedVectors,
            workSpace->m_postingProbeStats.m_adcSurvivors,
            workSpace->m_postingProbeStats.m_rerankCandidates,
            workSpace->m_postingProbeStats.m_rerankReadRequests,
            workSpace->m_postingProbeStats.m_rerankPhysicalBytes,
            workSpace->m_postingProbeStats.m_uniqueMatchedPostings,
            workSpace->m_postingProbeStats.m_uniqueMatchedVectors);

        if (ret == ErrorCode::Success &&
            queryTags != nullptr &&
            numQueryTags > 0 &&
            directHeadLocalIDs.empty() &&
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
                // Pass empty routedNodeMask to skip node check (we're scanning all heads)
                static const std::vector<uint8_t> kNoRouteMask;
                if (!m_index->HeadNodeMatchesQuery(sampleId, queryHierMask, kNoRouteMask)) {
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

                // Dedup against posting-scan results: a head VID that also lives
                // in a scanned posting (via its replicas) would otherwise be
                // AddPoint'd twice, pushing valid GT off top-K.
                if (workSpace->m_deduper.CheckAndSet(globalVID)) {
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

    const int dumpHeadsLimit = std::max(0, m_options.m_dumpHeads);
    static std::atomic<int> s_dumpHeadsQueryCount{0};
    const int dumpHeadsQueryId =
        dumpHeadsLimit > 0 ? s_dumpHeadsQueryCount.fetch_add(1) : -1;
    const bool dumpHeads =
        dumpHeadsQueryId >= 0 && dumpHeadsQueryId < dumpHeadsLimit;

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

    const bool useHeadBundleRuntime = !m_headBundleNodes.empty() &&
        (m_metadataOnlyHeadStore || m_headBundleNodes.size() > 1);
    std::vector<int> candidateNodes;
    if (useHeadBundleRuntime && !searchHeadBundleNodes.empty())
    {
        // Routed (filtered) queries restrict navigation to the bundle nodes
        // their tags map to. Single-bundle indexes (size==1) route to node 0
        // and must be handled here too — the previous `size > 1` guard left
        // them with an empty candidate set (no head search ran). When the
        // routed set covers all nodes the result is identical to the
        // unfilter all-nodes branch below.
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
    else if (useHeadBundleRuntime && searchHeadBundleNodes.empty())
    {
        // v5: unfilter (no tag scope) always routes through cross-edge unified
        // traversal across all per-bundle subgraphs. The global m_index is no
        // longer used for navigation in any code path. A single-bundle index
        // (size==1) is also handled here so its sole node is searched instead
        // of falling through to the (metadata-only-disabled) global fallback.
        candidateNodes.reserve(m_headBundleNodes.size());
        for (size_t i = 0; i < m_headBundleNodes.size(); ++i) {
            const auto& bn = m_headBundleNodes[i];
            if (bn.headCount == 0 || bn.postingCount == 0) continue;
            candidateNodes.push_back(static_cast<int>(bn.nodeId));
        }
    }
    if (adaptiveFilteredNprobeEnabled && filterSelectivity < 1.0f) {
        const SizeType totalHeads = TotalHeadSampleCount();
        const double globalTenantSize = static_cast<double>(m_options.m_vectorSize > 0 ? m_options.m_vectorSize : totalHeads);
        const SizeType globalPostingCount = std::max<SizeType>(1, totalHeads);
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
        // UnifiedNprobeBudget makes all routed
        // queries (including multi-subindex) trust the single aggregate budget,
        // which matches the unified cross-subgraph PQ search and avoids
        // amplifying nprobe by the number of routed subindexes.
        if (candidateNodes.size() > 1 && !m_options.m_unifiedNprobeBudget) {
            const double configuredKeepRatio = m_options.m_multiNodeBudgetKeepRatio;
            const double multiNodeBudgetKeepRatio =
                (std::isfinite(configuredKeepRatio) &&
                 configuredKeepRatio > 0.0 &&
                 configuredKeepRatio <= 1.0)
                    ? configuredKeepRatio
                    : 0.60;
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
            static_cast<int>(TotalHeadSampleCount()),
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
    // Primary-head CSR owns an independent one-vector-per-head assignment.
    // Its exact attributes are evaluated after graph navigation, so the old
    // physical-posting tag mask must not remove a primary owner before CSR
    // expansion.
    const bool primaryHeadBypassRequested =
        m_options.m_enablePrimaryHeadBypass && m_extraSearcher != nullptr &&
        m_extraSearcher->HasPrimaryHeadCSR();
    const bool s_phaseTime = m_options.m_logPhaseTime;
    g_bktSeedMs = 0.0;
    g_pqGraphMs = 0.0;
    auto _phT0 = s_phaseTime ? std::chrono::high_resolution_clock::now()
                             : std::chrono::high_resolution_clock::time_point{};

    if (!usedHeadBundleGraphSearch && !candidateNodes.empty())
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
            // available, (b) the query spans more than one routing node,
            // run a single best-first search across all bundle nodes' BKTs
            // joined by cross-edges instead of the serial per-node fanout.
            // The tag-aware in-filter inside CrossSubgraphGraphSearch is
            // self-guarded (no-op when numQueryTags==0), so unfiltered queries
            // can flow through the same path.
            bool useCrossSubgraph = !m_headCrossEdgesDirty.load(std::memory_order_acquire)
                && (candidateNodes.size() > 1)
                && (LoadHeadCrossEdges() == ErrorCode::Success)
                && (!m_headCrossEdgeTargetsB.empty())
                && !m_options.m_disableCrossSubgraph;

            if (m_options.m_logPathStats) {
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
                // Keep graph expansion and physical posting I/O as separate
                // budgets. BKT/KDT retain their native MaxCheck-adaptive
                // frontier; postingTarget remains the number of matching
                // postings ultimately admitted for I/O.
                const bool hasTagFilterLocal = (queryTags != nullptr && numQueryTags > 0);
                SPTAG::Cache::HierarchicalPostingMask tagAwareQueryMask;
                const bool usePostingMaskResultQueue = !primaryHeadBypassRequested &&
                    hasTagFilterLocal && m_options.m_enableHierPostingFilter
                    && m_index != nullptr && m_index->HasHeadNodeMeta();
                if (usePostingMaskResultQueue) {
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

                COMMON::QueryResultSet<T> nodeResults((const T*)p_query.GetTarget(), nodeGraphResultNum);
                ErrorCode nodeSearchStatus = ErrorCode::Fail;
                if (usePostingMaskResultQueue)
                {
                    const auto resultFilter = [this, &localToGlobalHIDs, &tagAwareQueryMask, queryDNF](SizeType localHid) {
                        if (localHid < 0 || static_cast<size_t>(localHid) >= localToGlobalHIDs.size()) {
                            return false;
                        }
                        const SizeType globalHid = localToGlobalHIDs[static_cast<size_t>(localHid)];
                        if (queryDNF != nullptr && !queryDNF->Empty()) {
                            const auto* postingMask = m_index->GetHeadNodePostingHierMask(globalHid);
                            return postingMask == nullptr || queryDNF->MayMatchHier(*postingMask);
                        }
                        return m_index->HeadPostingHierMaskMayIntersect(globalHid, tagAwareQueryMask);
                    };
                    if (auto* bkt = dynamic_cast<BKT::Index<T>*>(nodeIndex.get()))
                    {
                        nodeSearchStatus = bkt->SearchIndexWithResultFilter(nodeResults, resultFilter);
                    }
                    else if (auto* kdt = dynamic_cast<KDT::Index<T>*>(nodeIndex.get()))
                    {
                        nodeSearchStatus = kdt->SearchIndexWithResultFilter(nodeResults, resultFilter);
                    }
                }
                else
                {
                    nodeSearchStatus = nodeIndex->SearchIndex(nodeResults);
                }
                if (nodeSearchStatus != ErrorCode::Success)
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

                    if (usePostingMaskResultQueue
                        && !m_index->HeadPostingHierMaskMayIntersect(globalHID, tagAwareQueryMask)) {
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
        // In the dual-pool slim head store the root index physically holds only the
        // U_extra heads, so its tree cannot navigate H1; skip this dead fallback.
        if (!m_metadataOnlyHeadStore &&
            (ret = m_index->SearchIndex(*p_queryResults)) != ErrorCode::Success)
            return ret;
    }

    // Diagnostic: dump the selected head set (m_index-local hid + dist) for the
    // first DumpHeads queries so cross-subgraph vs fallback head selection can
    // be diffed in the same id space.
    if (dumpHeads) {
        int n = p_queryResults->GetResultNum();
        std::string line = "DUMPHEADS q=" + std::to_string(dumpHeadsQueryId) +
            " cross=" + std::to_string(usedHeadBundleGraphSearch ? 1 : 0) +
            " n=" + std::to_string(n) + " :";
        for (int di = 0; di < n; ++di) {
            auto* r = p_queryResults->GetResult(di);
            if (r == nullptr || r->VID < 0) continue;
            line += " " + std::to_string(r->VID) + ":" +
                    std::to_string(r->Dist);
        }
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "%s\n", line.c_str());
    }

    auto _phT1 = s_phaseTime ? std::chrono::high_resolution_clock::now()
                             : std::chrono::high_resolution_clock::time_point{};

    int phaseHeadCandidates = 0;
    int phaseHeadsAt110 = 0;
    int phaseHeadsAt125 = 0;
    int phaseHeadsAt150 = 0;
    int phaseHeadsAt200 = 0;
    int phaseHeadsAt400 = 0;
    int phaseHeadsAt800 = 0;
    float phaseHeadRatioP50 = 0.0f;
    float phaseHeadRatioP90 = 0.0f;
    float phaseHeadRatioMax = 0.0f;
    if (s_phaseTime && graphResultNum > 0) {
        const auto* nearestHead = p_queryResults->GetResult(0);
        const float nearestHeadDist = nearestHead != nullptr ? nearestHead->Dist : 0.0f;
        if (nearestHead != nullptr && nearestHead->VID != -1 && nearestHeadDist > 0.0f) {
            // Distances are squared L2 values. Record their ratios before pruning
            // so a native-INI cutoff can be calibrated from one loaded index.
            for (; phaseHeadCandidates < graphResultNum; ++phaseHeadCandidates) {
                const auto* head = p_queryResults->GetResult(phaseHeadCandidates);
                if (head == nullptr || head->VID == -1) break;

                const float ratio = head->Dist / nearestHeadDist;
                if (ratio <= 1.10f) ++phaseHeadsAt110;
                if (ratio <= 1.25f) ++phaseHeadsAt125;
                if (ratio <= 1.50f) ++phaseHeadsAt150;
                if (ratio <= 2.00f) ++phaseHeadsAt200;
                if (ratio <= 4.00f) ++phaseHeadsAt400;
                if (ratio <= 8.00f) ++phaseHeadsAt800;
            }

            if (phaseHeadCandidates > 0) {
                phaseHeadRatioP50 = p_queryResults->GetResult(
                    (phaseHeadCandidates - 1) / 2)->Dist / nearestHeadDist;
                phaseHeadRatioP90 = p_queryResults->GetResult(
                    (phaseHeadCandidates - 1) * 9 / 10)->Dist / nearestHeadDist;
                phaseHeadRatioMax = p_queryResults->GetResult(
                    phaseHeadCandidates - 1)->Dist / nearestHeadDist;
            }
        }
    }

    SearchStats _phExtraStats;
    SearchStats* _phExtraStatsPtr = s_phaseTime ? &_phExtraStats : nullptr;
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

        // Grow the pooled workspace's page-buffer / disk-request arrays to hold
        // postingTarget postings. InitWorkSpace(.,true) above reset them to only
        // m_searchInternalResultNum entries, so any postingTarget beyond that
        // must expand here — otherwise the posting scan below indexes past the
        // end of m_pageBuffers/m_diskRequests, truncating reads and collapsing
        // recall. The prior guard compared against nprobeBase
        // (=max(searchInternalResultNum, topk)), which left postingTarget in
        // (searchInternalResultNum, topk] under-sized — the k=100 dead-band.
        // Clear() only grows (never shrinks), so this is a no-op when already
        // large enough.
        if (postingTarget > m_options.m_searchInternalResultNum) {
            int maxPages = (std::max(m_options.m_postingPageLimit, m_options.m_searchPostingPageLimit)
                           + m_options.m_bufferLength + m_options.m_unfilterTailBufferLength) << PageSizeEx;
            workSpace->Clear(postingTarget, maxPages, true, m_options.m_enableDataCompression);
        }

        // Propagate posting-level PS pre-filter (applied in ExtraDynamicSearcher before MultiGet)
        workSpace->m_postingFilter = postingFilter;
        // Propagate inline tag filter (for per-vector exact tag check in posting scan)
        workSpace->m_queryTags = queryTags;
        workSpace->m_numQueryTags = numQueryTags;
        workSpace->m_dnf = queryDNF;
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
            // Intentionally uses HeadNodeMatchesQuery (with IsHeadNodeHeadOnly
            // gate) so that only ghost head-only vectors can be returned as
            // top-K results. For real heads (centroids of postings) the head
            // VID's own tag is NOT guaranteed to match the query even when its
            // posting members do; rejecting them here ensures top-K is sourced
            // only from posting scans (and the rare head-only ghost vectors).
            static const std::vector<uint8_t> kNoRouteMask;
            if (queryDNF != nullptr && !queryDNF->Empty()) {
                // Head-only metadata stores categorical own-tags only. A numeric
                // predicate has no exact head-only representation, so it must not
                // admit a coarse graph candidate into top-K.
                if (queryDNF->HasNumericLiteral()) return false;
                if (m_index == nullptr || !m_index->HasHeadNodeMeta() ||
                    !m_index->IsHeadNodeHeadOnly(localHid)) {
                    return false;
                }
                const auto* ownTags = m_index->GetHeadNodeHierMask(localHid);
                return ownTags != nullptr &&
                    queryDNF->Matches(ownTags->tag, SPTAG::Cache::HIER_LEVELS);
            }
            if (!hasTagFilter) return true;
            return m_index != nullptr &&
                   m_index->HasHeadNodeMeta() &&
                   m_index->HeadNodeMatchesQuery(localHid, queryHierMask, kNoRouteMask);
        };

        float limitDist = p_queryResults->GetResult(0)->Dist * m_options.m_maxDistRatio;
        const bool usePrimaryHeadBypass = primaryHeadBypassRequested;
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
                // The primary-owner sidecar is independent of the retained physical
                // posting membership. It must receive every graph head, including
                // heads whose current posting mask rejects this query.
                if (usePrimaryHeadBypass ||
                    !workSpace->m_postingFilter || workSpace->m_postingFilter(localHid)) {
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

        if (usePrimaryHeadBypass) {
        ret = m_extraSearcher->SearchPrimaryHeadCandidates(workSpace.get(), *p_queryResults, m_index);
        if (ret == ErrorCode::Success) {
            SPTAG::VectorIndex::SetThreadLocalPostingScanStats(
                0, 0, 0, 0, 0, workSpace->m_postingProbeStats.m_primaryHeadCandidates);
            m_workSpaceFactory->ReturnWorkSpace(std::move(workSpace));
            p_queryResults->SortResult();
            if (p_queryResults != (COMMON::QueryResultSet<T> *)&p_query) {
                std::copy(p_queryResults->GetResults(),
                          p_queryResults->GetResults() + p_query.GetResultNum(),
                          p_query.GetResults());
                p_query.SetScanned(p_queryResults->GetScanned());
                delete p_queryResults;
            }
            return ErrorCode::Success;
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
        if (dumpHeadsLimit > 0) {
            std::string s = "HEADDUMP:";
            s.reserve(workSpace->m_postingIDs.size() * 8 + 16);
            for (auto h : workSpace->m_postingIDs) {
                s.push_back(' ');
                s += std::to_string(static_cast<long long>(h));
            }
            fprintf(stderr, "%s\n", s.c_str());
            fflush(stderr);
        }
        ret = m_extraSearcher->SearchIndex(workSpace.get(), *p_queryResults, m_index, _phExtraStatsPtr);
        SPTAG::VectorIndex::SetThreadLocalPostingScanStats(
            workSpace->m_postingProbeStats.m_readPostings,
            workSpace->m_postingProbeStats.m_matchedPostings,
            workSpace->m_postingProbeStats.m_prePSPostings,
            workSpace->m_postingProbeStats.m_scannedVectors,
            workSpace->m_postingProbeStats.m_matchedVectors,
            workSpace->m_postingProbeStats.m_primaryHeadCandidates,
            workSpace->m_postingProbeStats.m_postingPageReads,
            workSpace->m_postingProbeStats.m_postingLogicalBytes,
            workSpace->m_postingProbeStats.m_postingPhysicalBytes,
            workSpace->m_postingProbeStats.m_adcScannedVectors,
            workSpace->m_postingProbeStats.m_adcSurvivors,
            workSpace->m_postingProbeStats.m_rerankCandidates,
            workSpace->m_postingProbeStats.m_rerankReadRequests,
            workSpace->m_postingProbeStats.m_rerankPhysicalBytes,
            workSpace->m_postingProbeStats.m_uniqueMatchedPostings,
            workSpace->m_postingProbeStats.m_uniqueMatchedVectors);
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
            "PhaseTime: tag=%u nprobe=%d heads=%d headR50=%.3f headR90=%.3f headRMax=%.3f "
            "headAt110=%d headAt125=%d headAt150=%d headAt200=%d headAt400=%d headAt800=%d "
            "bkt=%.3f pq=%.3f graphOther=%.3f post=%.3f io=%.3f scan=%.3f postOther=%.3f total=%.3f\n",
            firstTag, postingTarget,
            phaseHeadCandidates, phaseHeadRatioP50, phaseHeadRatioP90, phaseHeadRatioMax,
            phaseHeadsAt110, phaseHeadsAt125, phaseHeadsAt150, phaseHeadsAt200,
            phaseHeadsAt400, phaseHeadsAt800,
            g_bktSeedMs, g_pqGraphMs,
            graphTotalMs - g_bktSeedMs - g_pqGraphMs, postMs,
            _phExtraStats.m_diskReadLatency, _phExtraStats.m_compLatency,
            postMs - _phExtraStats.m_diskReadLatency - _phExtraStats.m_compLatency,
            graphTotalMs + postMs);
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
    if (m_metadataOnlyHeadStore) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "ITERATIVE NOT SUPPORT FOR METADATA-ONLY SPANN HEAD ROOT\n");
        return ErrorCode::Undefined;
    }

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
    ErrorCode headSearchStatus = ErrorCode::Fail;
    if (m_metadataOnlyHeadStore) {
        std::vector<int> candidateNodes;
        candidateNodes.reserve(m_headBundleNodes.size());
        for (const auto& node : m_headBundleNodes) {
            candidateNodes.push_back(node.nodeId);
        }
        if (candidateNodes.empty()) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Metadata-only head root has no bundle nodes for generic filtering.\n");
            return ErrorCode::Fail;
        }

        headResults.Reset();
        int scanned = 0;
        const bool canUseCrossSubgraph =
            !m_headCrossEdgesDirty.load(std::memory_order_acquire) &&
            candidateNodes.size() > 1 &&
            LoadHeadCrossEdges() == ErrorCode::Success &&
            !m_headCrossEdgeTargetsB.empty() &&
            !m_options.m_disableCrossSubgraph;
        if (canUseCrossSubgraph) {
            headSearchStatus = CrossSubgraphGraphSearch(
                p_query, &headResults, candidateNodes, nullptr, 0, headSearchNum, scanned);
        } else {
            headSearchStatus = ErrorCode::Success;
            for (int nodeId : candidateNodes) {
                if (EnsureHeadBundleNodeLoaded(nodeId) != ErrorCode::Success) {
                    headSearchStatus = ErrorCode::Fail;
                    break;
                }
                const auto& nodeIndex =
                    m_loadedHeadBundleIndexes[static_cast<size_t>(nodeId)];
                const auto& localToGlobal =
                    m_headBundleLocalToGlobalHIDs[static_cast<size_t>(nodeId)];
                if (nodeIndex == nullptr || localToGlobal.empty()) {
                    headSearchStatus = ErrorCode::Fail;
                    break;
                }

                COMMON::QueryResultSet<T> nodeResults(
                    static_cast<const T*>(p_query.GetTarget()),
                    (std::min)(headSearchNum, static_cast<int>(localToGlobal.size())));
                if (nodeIndex->SearchIndex(nodeResults) != ErrorCode::Success) {
                    headSearchStatus = ErrorCode::Fail;
                    break;
                }
                scanned += nodeResults.GetScanned();
                for (int i = 0; i < nodeResults.GetResultNum(); ++i) {
                    BasicResult* result = nodeResults.GetResult(i);
                    if (result == nullptr || result->VID == -1) break;
                    if (result->VID < 0 ||
                        static_cast<size_t>(result->VID) >= localToGlobal.size()) {
                        continue;
                    }
                    headResults.AddPoint(
                        localToGlobal[static_cast<size_t>(result->VID)], result->Dist);
                }
            }
            if (headSearchStatus == ErrorCode::Success) {
                headResults.SortResult();
            }
        }
        if (headSearchStatus == ErrorCode::Success) {
            headResults.SetScanned(scanned);
        }
    } else {
        headSearchStatus = m_index->SearchIndex(headResults);
    }
    if (headSearchStatus != ErrorCode::Success) {
        return headSearchStatus;
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
        // Native clustering on raw vectors: do NOT route head selection through the
        // quantizer reconstruct path (that's for posting encoding at BuildSSDIndex).
        bkt->m_pQuantizer = nullptr;
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start invoking BuildTrees.\n");
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Info,
            "BKTKmeansK: %d, BKTLeafSize: %d, Samples: %d, BKTLambdaFactor:%f TreeNumber: %d, ThreadNum: %d.\n",
            bkt->m_iBKTKmeansK, bkt->m_iBKTLeafSize, bkt->m_iSamples, bkt->m_fBalanceFactor, bkt->m_iTreeNumber,
            m_options.m_iSelectHeadNumberOfThreads);

        if (m_options.m_parallelBKTBuild)
            bkt->BuildTreesParallel<InternalDataType>(data, m_options.m_distCalcMethod, m_options.m_iSelectHeadNumberOfThreads,
                                              nullptr, nullptr, true);
        else
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
    else if (Helper::StrUtils::StrEqualIgnoreCase(m_options.m_selectType.c_str(), "PerTagBKT"))
    {
        // Per-tag head selection (the deprecated cross-tag "merge" variants were
        // removed 2026-06-11):
        //   Partition vectors by a per-vector grouping tag (the routing key,
        //   e.g. SIFT project / YFCC country). For each group build a BKTree on
        //   that subset and run SelectHeadDynamically at the target head ratio.
        //   The union of all per-tag heads is the final head set. No merge.
        // PerVectorTagsFile contains one grouping tag per vector and is supplied
        // through native [SelectHead] configuration.
        if (m_options.m_perVectorTagsFile.empty())
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "PerTagBKT requires [SelectHead] PerVectorTagsFile\n");
            return false;
        }
        const std::string& tagsFile = m_options.m_perVectorTagsFile;
        // PerTagBKT historically used a 1.6% fallback, unlike vanilla BKT's
        // 20% default. Preserve that behavior unless native Ratio was supplied.
        const double configuredRatio =
            m_options.m_ratioExplicitlySet ? m_options.m_ratio : 0.016;
        const double finalRatio = std::clamp(configuredRatio, 1e-5, 0.9);

        // ---- Read per-vector tag column (one int per line) ----
        std::vector<int> perVecTag(data.R(), -1);
        {
            std::ifstream fin(tagsFile);
            if (!fin.good())
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                             "PerTagBKT failed to open %s\n", tagsFile.c_str());
                return false;
            }
            int v; int idx = 0;
            while (idx < data.R() && (fin >> v))
                perVecTag[idx++] = v;
            if (idx != data.R())
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                             "PerTagBKT tag file has %d entries, expected %d\n",
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
                     "PerTagBKT: %zu distinct tag values\n", tagGroups.size());

        // ---- Per-tag BKT + SelectHeadDynamically ----
        // Target ratio for SelectHeadDynamically per tag = finalRatio directly.
        const double perTagTarget = std::min(0.9, finalRatio);
        const auto savedSelTh    = m_options.m_selectThreshold;
        const auto savedSplTh    = m_options.m_splitThreshold;
        const auto savedSplFa    = m_options.m_splitFactor;
        const auto savedKmeansK  = m_options.m_iBKTKmeansK;
        const auto savedSamples  = m_options.m_iSamples;

        std::vector<int> initialHeads;     // global VIDs

        for (auto& kv : tagGroups)
        {
            std::vector<SizeType>& subIdx = kv.second;
            int subSize = static_cast<int>(subIdx.size());
            if (subSize <= 1)
            {
                if (subSize == 1)
                    initialHeads.push_back(static_cast<int>(subIdx[0]));
                continue;
            }

            // Per-tag SelectHead reuses the user-configured ratio AND thresholds
            // exactly as vanilla path. Do NOT reset selectThreshold/splitThreshold
            // — that was an oversample-era artifact that forced threshold
            // re-derivation from ratio (=1/ratio), producing 10x fewer heads
            // than vanilla which uses user-set defaults (e.g. 6/25).
            m_options.m_ratio = perTagTarget;
            m_options.m_iBKTKmeansK = savedKmeansK;
            m_options.m_iSamples = std::min(savedSamples, subSize);
            SelectHeadAdjustOptions(subSize);

            std::shared_ptr<COMMON::BKTree> bkt = std::make_shared<COMMON::BKTree>();
            bkt->m_iBKTKmeansK   = m_options.m_iBKTKmeansK;
            bkt->m_iBKTLeafSize  = m_options.m_iBKTLeafSize;
            bkt->m_iSamples      = std::min(m_options.m_iSamples, subSize);
            bkt->m_iTreeNumber   = m_options.m_iTreeNumber;
            bkt->m_fBalanceFactor = m_options.m_fBalanceFactor;
            // Native clustering on raw vectors (OPQ is posting-only, see BKT branch).
            bkt->m_pQuantizer    = nullptr;
            if (m_options.m_parallelBKTBuild)
                bkt->BuildTreesParallel<InternalDataType>(data, m_options.m_distCalcMethod,
                                                  m_options.m_iSelectHeadNumberOfThreads,
                                                  &subIdx, nullptr, true);
            else
                bkt->BuildTrees<InternalDataType>(data, m_options.m_distCalcMethod,
                                              m_options.m_iSelectHeadNumberOfThreads,
                                              &subIdx, nullptr, true);

            std::vector<int> subSelected;
            SelectHeadDynamically(bkt, subSize, subSelected);
            if (subSelected.empty())
                subSelected.push_back(static_cast<int>(subIdx[0]));

            for (int h : subSelected)
                initialHeads.push_back(h);
        }

        // Restore temporary selection knobs but retain the effective target for
        // the persisted indexloader.ini.
        m_options.m_ratio = perTagTarget;
        m_options.m_selectThreshold = savedSelTh;
        m_options.m_splitThreshold = savedSplTh;
        m_options.m_splitFactor = savedSplFa;
        m_options.m_iBKTKmeansK = savedKmeansK;
        m_options.m_iSamples = savedSamples;

        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                     "PerTagBKT: %zu per-tag heads "
                     "(finalRatio=%.3f%%, perTagTarget=%.3f%%, achieved=%.3f%%)\n",
                     initialHeads.size(),
                     100.0 * finalRatio,
                     100.0 * perTagTarget,
                     100.0 * initialHeads.size() / data.R());

        // [merge path removed 2026-06-11] No cross-tag merge: the per-tag
        // heads ARE the final head set. Dedup + sort for downstream phases.
        std::sort(initialHeads.begin(), initialHeads.end());
        initialHeads.erase(std::unique(initialHeads.begin(), initialHeads.end()),
                           initialHeads.end());
        selected.swap(initialHeads);

        if (selected.empty()) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "PerTagBKT produced no heads\n");
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

    // --- Dual-pool v3 augmentation ---------------------------------------------
    // [SelectHead] DualPoolAugment appends DualPoolExtraRatio * |H1| non-head
    // VIDs as unfilter-only heads (role==1). Filter queries gate them out.
    if (m_options.m_dualPoolAugment && !selected.empty()) {
        const double extraRatio = m_options.m_dualPoolExtraRatio;
        if (extraRatio > 0.0) {
                size_t n_h1 = selected.size();
                std::unordered_set<int> h1set(selected.begin(), selected.end());
                std::vector<int> nonHeads;
                nonHeads.reserve(static_cast<size_t>(data.R()) - h1set.size());
                for (int i = 0; i < data.R(); ++i)
                    if (!h1set.count(i)) nonHeads.push_back(i);

                size_t n_extras = static_cast<size_t>(
                    std::round(extraRatio * static_cast<double>(n_h1)));
                if (n_extras > nonHeads.size()) n_extras = nonHeads.size();

                // If UExtraIDFile points to a binary file (int32 count, then
                // int32[count] tenant-local VIDs), use that explicit U_extra set.
                // Otherwise retain deterministic random selection.
                bool uxLoaded = false;
                if (!m_options.m_uExtraIDFile.empty()) {
                    std::ifstream uf(m_options.m_uExtraIDFile, std::ios::binary);
                    if (uf) {
                        std::int32_t cnt = 0;
                        uf.read(reinterpret_cast<char*>(&cnt), sizeof(cnt));
                        std::vector<std::int32_t> tmp;
                        if (cnt > 0) {
                            tmp.resize(static_cast<size_t>(cnt));
                            uf.read(reinterpret_cast<char*>(tmp.data()),
                                    static_cast<std::streamsize>(cnt) * sizeof(std::int32_t));
                        }
                        std::vector<int> picked;
                        picked.reserve(tmp.size());
                        for (std::int32_t v : tmp)
                            if (v >= 0 && v < data.R() && !h1set.count(v))
                                picked.push_back(static_cast<int>(v));
                        std::sort(picked.begin(), picked.end());
                        picked.erase(std::unique(picked.begin(), picked.end()), picked.end());
                        nonHeads.swap(picked);
                        n_extras = nonHeads.size();
                        uxLoaded = true;
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                            "DualPoolAugment: loaded %zu U_extra VIDs from %s (RNG-selected)\n",
                            n_extras, m_options.m_uExtraIDFile.c_str());
                    } else {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                            "DualPoolAugment: cannot open UExtraIDFile=%s; "
                            "falling back to random selection\n",
                            m_options.m_uExtraIDFile.c_str());
                    }
                }

                if (!uxLoaded) {
                    std::mt19937 rng(12345);
                    std::shuffle(nonHeads.begin(), nonHeads.end(), rng);
                    nonHeads.resize(n_extras);
                    std::sort(nonHeads.begin(), nonHeads.end());
                }

                selected.insert(selected.end(), nonHeads.begin(), nonHeads.end());

                m_pendingHeadRoles.clear();
                m_pendingHeadRoles.resize(selected.size(), 0);
                for (size_t ri = n_h1; ri < selected.size(); ++ri)
                    m_pendingHeadRoles[ri] = 1;

                m_pendingNodeUExtraSelections.clear();
                if (!m_pendingPrimaryNodeVectorAssignments.empty()) {
                    std::vector<int> primaryOwner(static_cast<size_t>(data.R()), -1);
                    for (size_t ni = 0; ni < m_pendingPrimaryNodeVectorAssignments.size(); ++ni)
                        for (SizeType vid : m_pendingPrimaryNodeVectorAssignments[ni])
                            if (vid >= 0 && vid < data.R())
                                primaryOwner[static_cast<size_t>(vid)] = static_cast<int>(ni);
                    m_pendingNodeUExtraSelections.assign(
                        m_pendingPrimaryNodeVectorAssignments.size(), std::vector<SizeType>());
                    for (int vid : nonHeads)
                        if (vid >= 0 && vid < data.R()) {
                            int node = primaryOwner[static_cast<size_t>(vid)];
                            if (node >= 0 && node < static_cast<int>(m_pendingNodeUExtraSelections.size()))
                                m_pendingNodeUExtraSelections[static_cast<size_t>(node)].push_back(
                                    static_cast<SizeType>(vid));
                        }
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                    "DualPoolAugment v3: H1=%zu, U_extra=%zu, final=%zu\n",
                    n_h1, n_extras, selected.size());
        }
    }
    // --------------------------------------------------------------------------

    if (!m_options.m_noOutput)
    {
        // Dual-pool: H1 already sorted; U_extra appended sorted. Sorting across
        // the boundary would corrupt the role->ordinal mapping in head_role.bin.
        if (m_pendingHeadRoles.empty()) {
            std::sort(selected.begin(), selected.end());
        }
        const bool hasBundleUExtra = std::any_of(
            m_pendingNodeUExtraSelections.begin(),
            m_pendingNodeUExtraSelections.end(),
            [](const std::vector<SizeType>& p_selection) { return !p_selection.empty(); });
        const bool buildMetadataOnlyBundleRoot =
            m_options.m_storage == Storage::STATIC &&
            !m_pendingNodeHeadSelections.empty() &&
            m_pQuantizer == nullptr &&
            !m_options.m_enableDeltaEncoding &&
            !hasBundleUExtra;
        if (buildMetadataOnlyBundleRoot) {
            for (auto& nodeHeads : m_pendingNodeHeadSelections) {
                std::sort(nodeHeads.begin(), nodeHeads.end());
            }
        }
        if (buildMetadataOnlyBundleRoot) {
            if (!WriteSelectedHeadIDs(
                    selected,
                    m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile)) {
                return false;
            }
            SPTAGLIB_LOG(
                Helper::LogLevel::LL_Info,
                "Bundle STATIC build: wrote top-level head IDs only; bundle vectors are the sole head-vector source.\n");
        } else if (!WriteSelectedHeadFiles(
                       data,
                       std::vector<SizeType>(selected.begin(), selected.end()),
                       m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile,
                       m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile)) {
            return false;
        }

        if (!m_pendingHeadRoles.empty()) {
            std::string rolePath = m_options.m_indexDirectory + FolderSep + m_options.m_headRoleFile;
            FILE* fp = fopen(rolePath.c_str(), "wb");
            if (fp) {
                fwrite(m_pendingHeadRoles.data(), 1, m_pendingHeadRoles.size(), fp);
                fclose(fp);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                    "DualPool v3: wrote head_role.bin (%zu bytes) to %s\n",
                    m_pendingHeadRoles.size(), rolePath.c_str());
            } else {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "DualPool v3: failed to write head_role.bin to %s\n", rolePath.c_str());
            }
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

                // Dual-pool approach-1: append this bundle's U_extra heads after
                // its H1 heads in both the head vector file and the head ID file,
                // so they become routable graph nodes inside the bundle subgraph.
                std::vector<SizeType> nodeSel = m_pendingNodeHeadSelections[nodeId];
                if (nodeId < m_pendingNodeUExtraSelections.size() &&
                    !m_pendingNodeUExtraSelections[nodeId].empty()) {
                    const auto& ux = m_pendingNodeUExtraSelections[nodeId];
                    nodeSel.insert(nodeSel.end(), ux.begin(), ux.end());
                }
                if (!WriteSelectedHeadFiles(data,
                                            nodeSel,
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

    // SelectHead resume checkpoint: once head selection + per-node head files are on
    // disk, persist the derived in-memory state so a later BuildHead/BuildSSDIndex
    // failure can be retried without re-running the expensive BKT head selection.
    if (!m_options.m_noOutput && SpannEnvFlagOn("SPTAG_PERSIST_SELECTHEAD")) {
        const std::string statePath = HeadSelectStatePath(m_options, m_options.m_indexDirectory);
        if (SaveHeadSelectState(statePath) == ErrorCode::Success) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                         "SelectHead checkpoint written: %s (resume with SPTAG_RESUME_BUILD=1 to skip BKT)\n",
                         statePath.c_str());
        } else {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                         "SelectHead checkpoint FAILED to write: %s\n", statePath.c_str());
        }
    }
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

    // Resume path: if a SelectHead checkpoint exists and resume is requested, reload
    // the derived build state and skip the expensive BKT head selection. The per-node
    // head vector files (preserved during the prior BuildHead run because persistence
    // was enabled) and the global head files remain on disk for BuildHead/BuildSSDIndex.
    bool resumedSelectHead = false;
    if (SpannEnvFlagOn("SPTAG_PERSIST_SELECTHEAD") && SpannEnvFlagOn("SPTAG_RESUME_BUILD")) {
        const std::string statePath = HeadSelectStatePath(m_options, m_options.m_indexDirectory);
        if (fileexists(statePath.c_str())) {
            if (LoadHeadSelectState(statePath) == ErrorCode::Success) {
                resumedSelectHead = true;
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                             "Resume: loaded SelectHead checkpoint %s (nodes=%zu, headRoles=%zu); "
                             "skipping BKT head selection\n",
                             statePath.c_str(), m_pendingNodeHeadSelections.size(),
                             m_pendingHeadRoles.size());
            } else {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                             "Resume: failed to load checkpoint %s; re-running SelectHead\n",
                             statePath.c_str());
            }
        } else {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                         "Resume requested but no checkpoint at %s; running SelectHead normally\n",
                         statePath.c_str());
        }
    }

    if (m_options.m_selectHead && !resumedSelectHead)
    {
        // Head selection clusters the RAW vectors using their true ValueType T
        // (native uint8/int8/float clustering). The previous `if (m_pQuantizer)`
        // branch forced InternalDataType=std::uint8_t whenever a quantizer was set,
        // which (a) silently reinterpreted signed int8 vectors as uint8 — flipping
        // every negative coordinate (e.g. -10 -> 246) and producing a degenerate
        // one-mega-cluster BKT — and (b) built a uint8 head index for an int8 main
        // index (type-inconsistent). OPQ is applied to the SSD postings at
        // BuildSSDIndex, NOT to head selection, so we always cluster on T here.
        bool success = SelectHeadInternal<T>(p_reader);
        if (!success)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SelectHead Failed!\n");
            return ErrorCode::Fail;
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double selectHeadTime = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "select head time: %.2lfs\n", selectHeadTime);

    const bool hasBundleUExtra = std::any_of(
        m_pendingNodeUExtraSelections.begin(),
        m_pendingNodeUExtraSelections.end(),
        [](const std::vector<SizeType>& p_selection) { return !p_selection.empty(); });
    const bool buildMetadataOnlyBundleRoot =
        m_options.m_storage == Storage::STATIC &&
        !m_pendingNodeHeadSelections.empty() &&
        m_pQuantizer == nullptr &&
        !m_options.m_enableDeltaEncoding &&
        !hasBundleUExtra;
    bool resumedCompletedBundleHeads = false;
    if (resumedSelectHead && m_options.m_buildHead && buildMetadataOnlyBundleRoot)
    {
        if (TryResumeCompletedBundleHeads(resumedCompletedBundleHeads) != ErrorCode::Success) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Failed to validate completed bundle heads for resume.\n");
            return ErrorCode::Fail;
        }
    }

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Begin Build Head...\n");
    if (m_options.m_buildHead && !resumedCompletedBundleHeads)
    {
        // QuantizeHead config gate: only build the head index on quantized vectors when
        // a global quantizer is present AND m_quantizeHead is set. With in-posting
        // quantization (the default for posting-OPQ/RaBitQ) m_pQuantizer is null and the
        // head always stays full-precision; QuantizeHead only controls the native
        // global-quantizer head path.
        bool headQuant = (m_pQuantizer != nullptr) && m_options.m_quantizeHead;
        auto valueType = headQuant ? SPTAG::VectorValueType::UInt8 : m_options.m_valueType;
        auto dims = headQuant ? m_pQuantizer->GetNumSubvectors() : m_options.m_dim;
        auto buildHeadIndexFromFile = [&](const std::string& vectorFilePath, const std::string& saveDir,
                                          SizeType n_h1_split = -1) -> bool {
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
            localHeadIndex->SetQuantizer(headQuant ? m_pQuantizer : nullptr);
            for (const auto& iter : m_headParameters)
            {
                localHeadIndex->SetParameter(iter.first.c_str(), iter.second.c_str());
            }

            auto localHeadVectorSet = localVectorReader->GetVectorSet();
            // Dual-pool approach-1: the bundle head file is laid out as
            // [H1 heads][bundle U_extra heads]. Build the BKT tree + RNG graph
            // over H1 only, then append U_extra as graph nodes WITHOUT inserting
            // back-edges into H1 (AddIndexIdxNoBackEdge). This gives each U_extra
            // out-edges toward its host H1 while keeping H1's RNG neighbor lists
            // pristine — H1 can only reach U_extra via cross-edges (built later
            // by AugmentHeadGraph). U_extra also stay out of the BKT tree, so
            // query seeding never lands on them directly.
            // SPTAG_UEXTRA_FULL_GRAPH (default OFF): when set =1, build the BKT tree + RNG
            // graph over the full [H1 + U_extra] head set so U_extra become first-class graph
            // nodes (seedable via BKT + normal RNG back-edges). DEFAULT (OFF) keeps the
            // intended dual-pool design: BKT over H1 only, U_extra appended with NoBackEdge
            // so their only in-edges are the cross-edges added by AugmentHeadGraph (second-
            // class graph citizens). Posting assignment is governed separately by
            // SPTAG_UEXTRA_FULL_POSTING. head_role.bin is always written for filter gating.
            static const bool s_uextraFullGraph = []() {
                const char* v = std::getenv("SPTAG_UEXTRA_FULL_GRAPH");
                return (v && (v[0] == '1' || v[0] == 't' || v[0] == 'T'));
            }();
            SizeType totalCount = localHeadVectorSet->Count();
            SizeType n_h1 = (n_h1_split >= 0 && n_h1_split <= totalCount) ? n_h1_split : totalCount;
            SizeType n_extra = totalCount - n_h1;
            std::shared_ptr<VectorSet> buildSet = localHeadVectorSet;
            if (!s_uextraFullGraph && n_extra > 0) {
                ByteArray h1Bytes = ByteArray::Alloc(
                    static_cast<size_t>(n_h1) * static_cast<size_t>(dims) * sizeof(T));
                std::memcpy(h1Bytes.Data(), localHeadVectorSet->GetData(),
                            static_cast<size_t>(n_h1) * static_cast<size_t>(dims) * sizeof(T));
                buildSet = std::make_shared<BasicVectorSet>(
                    h1Bytes, valueType, static_cast<DimensionType>(dims), static_cast<SizeType>(n_h1));
            }
            if (localHeadIndex->BuildIndex(buildSet, nullptr, false, true, true) != ErrorCode::Success)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to build node head index for %s.\n", saveDir.c_str());
                return false;
            }
            if (!s_uextraFullGraph && n_extra > 0) {
                const T* uBase = static_cast<const T*>(localHeadVectorSet->GetData())
                                 + static_cast<size_t>(n_h1) * static_cast<size_t>(dims);
                localHeadIndex->SetAddCountForRebuild(std::numeric_limits<int>::max());
                int beginU = 0, endU = 0;
                if (localHeadIndex->AddIndexId(uBase, n_extra,
                                               static_cast<DimensionType>(dims), beginU, endU) == ErrorCode::Success)
                    localHeadIndex->AddIndexIdxNoBackEdge(static_cast<SizeType>(beginU), static_cast<SizeType>(endU));
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                    "DualPool approach-1: bundle %s built %d H1 + %d U_extra (no-back-edge)\n",
                    saveDir.c_str(), (int)n_h1, (int)n_extra);
            }
            if (headQuant && !m_options.m_quantizerFilePath.empty()) {
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

        if (!buildMetadataOnlyBundleRoot) {
        m_index = SPTAG::VectorIndex::CreateInstance(m_options.m_indexAlgoType, valueType);
        m_index->SetParameter("DistCalcMethod", SPTAG::Helper::Convert::ConvertToString(m_options.m_distCalcMethod));
        m_index->SetQuantizer(headQuant ? m_pQuantizer : nullptr);
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
            // Dual-pool v3: build global m_index BKT from H1-only heads (first n_h1 in file),
            // then extend the RNG graph with U_extra via AddIndexId+AddIndexIdx so unfilter
            // can reach U_extra while filter never sees them in the BKT tree.
            static const bool s_uextraFullGraph = []() {
                const char* v = std::getenv("SPTAG_UEXTRA_FULL_GRAPH");
                return (v && (v[0] == '1' || v[0] == 't' || v[0] == 'T'));
            }();
            std::shared_ptr<VectorSet> h1VecSet = headvectorset;
            SizeType n_extra_global = 0;
            if (!s_uextraFullGraph && !m_pendingHeadRoles.empty()) {
                SizeType n_h1 = 0;
                for (auto r : m_pendingHeadRoles) if (r == 0) ++n_h1;
                if (n_h1 < headvectorset->Count()) {
                    auto valueType = headQuant ? SPTAG::VectorValueType::UInt8 : m_options.m_valueType;
                    auto dims = headQuant ? m_pQuantizer->GetNumSubvectors() : m_options.m_dim;
                    ByteArray h1Bytes = ByteArray::Alloc(
                        static_cast<size_t>(n_h1) * static_cast<size_t>(dims) * sizeof(T));
                    std::memcpy(h1Bytes.Data(), headvectorset->GetData(),
                                static_cast<size_t>(n_h1) * static_cast<size_t>(dims) * sizeof(T));
                    h1VecSet = std::make_shared<BasicVectorSet>(
                        h1Bytes, valueType,
                        static_cast<DimensionType>(dims), static_cast<SizeType>(n_h1));
                    n_extra_global = headvectorset->Count() - n_h1;
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "DualPool v3: global m_index H1-only build: %d H1 + %d U_extra\n",
                        (int)n_h1, (int)n_extra_global);
                }
            }
            if (m_index->BuildIndex(h1VecSet, nullptr, false, true, true) != ErrorCode::Success)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to build head index.\n");
                return ErrorCode::Fail;
            }
            if (n_extra_global > 0) {
                auto dims = headQuant ? m_pQuantizer->GetNumSubvectors() : m_options.m_dim;
                SizeType n_h1 = headvectorset->Count() - n_extra_global;
                const T* uBase = static_cast<const T*>(headvectorset->GetData())
                                 + static_cast<size_t>(n_h1) * static_cast<size_t>(dims);
                m_index->SetAddCountForRebuild(std::numeric_limits<int>::max());
                int beginU = 0, endU = 0;
                if (m_index->AddIndexId(uBase, n_extra_global,
                                        static_cast<DimensionType>(dims), beginU, endU) == ErrorCode::Success)
                    m_index->AddIndexIdx(static_cast<SizeType>(beginU), static_cast<SizeType>(endU));
            }
            if (headQuant && !m_options.m_quantizerFilePath.empty())
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
                    const std::string nodeCanonicalVectorFile = nodeDir + FolderSep + "vectors.bin";
                    const std::string buildInputVectorFile =
                        fileexists(nodeVectorFile.c_str()) ? nodeVectorFile : nodeCanonicalVectorFile;
                    // Pass the H1 split so the bundle BKT tree is built over H1
                    // only and U_extra are appended graph-only (no back-edges).
                    SizeType n_h1_node = static_cast<SizeType>(m_pendingNodeHeadSelections[nodeId].size());
                    if (!buildHeadIndexFromFile(buildInputVectorFile, nodeDir, n_h1_node)) {
                        return ErrorCode::Fail;
                    }
                    // The bundle subgraph's SaveIndex has written node vectors.bin,
                    // which is byte-identical to this SPTAGHeadVectors.bin build input.
                    // Nothing at load/search/augment time reads the per-node
                    // SPTAGHeadVectors.bin (EnsureHeadBundleNodeLoaded uses vectors.bin
                    // via LoadIndex + SPTAGHeadVectorIDs.bin; AugmentHeadGraph likewise).
                    // Remove the redundant copy to halve per-node head-vector storage.
                    // BuildHead can resume from vectors.bin if the transient input is
                    // already gone, so never keep this duplicate in the final index.
                    if (fileexists(nodeVectorFile.c_str()) && remove(nodeVectorFile.c_str()) != 0) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                                     "Failed to remove redundant node head vector file %s\n",
                                     nodeVectorFile.c_str());
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

        if (buildMetadataOnlyBundleRoot)
        {
            if (ActivateMetadataOnlyBundleRoot() != ErrorCode::Success) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                             "Failed to activate the bundle metadata root.\n");
                return ErrorCode::Fail;
            }
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

        // Tags are optional, but bundle ownership is required whenever static
        // placement/tails use multiple preselected head bundles.
        if (!m_pendingVectorTags.empty() && m_pendingNumTagsPerVec > 0) {
            int numVecs = (int)(m_pendingVectorTags.size() / m_pendingNumTagsPerVec);
            m_extraSearcher->SetVectorTags(
                m_pendingVectorTags.data(), numVecs, m_pendingNumTagsPerVec);
        }
        if (!m_pendingNodeVectorAssignments.empty()) {
            m_extraSearcher->SetNodeVectorAssignments(m_pendingNodeVectorAssignments);
        }
        if (!m_pendingPrimaryNodeVectorAssignments.empty()) {
            m_extraSearcher->SetPrimaryNodeVectorAssignments(
                m_pendingPrimaryNodeVectorAssignments);
        }
        if (auto* staticSearcher =
                dynamic_cast<ExtraStaticSearcher<T>*>(m_extraSearcher.get())) {
            staticSearcher->SetHeadVectorOwnersView(&m_pendingHeadVectorOwners);
            if (m_metadataOnlyHeadStore) {
                staticSearcher->SetHeadBundleBuildView(
                    m_loadedHeadBundleIndexes,
                    &m_headBundleLocalToGlobalHIDs,
                    &m_pendingNodeHeadSelections);
            }
        } else if (!m_pendingHeadVectorOwners.empty()) {
            m_extraSearcher->SetHeadVectorOwners(m_pendingHeadVectorOwners);
        }
        auto* eds = dynamic_cast<ExtraDynamicSearcher<T>*>(m_extraSearcher.get());
        if (eds && !m_pendingHeadRoles.empty()) {
            // Dual-pool v3: pass head roles so SSD build can route U_extra
            // through the k-NN posting path (ExtraDynamicSearcher.h:2493+,2630+).
            eds->SetHeadRoles(m_pendingHeadRoles);
        }

       if (m_vectorTranslateMap.R() == 0) {
            std::shared_ptr<Helper::DiskIO> ptr = SPTAG::f_createIO();
            if (ptr == nullptr ||
                !ptr->Initialize((m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str(),
                                 std::ios::binary | std::ios::in))
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to open headIDFile file:%s\n",
                             (m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str());
                return ErrorCode::Fail;
            }
            m_vectorTranslateMap.Load(
                ptr,
                m_options.m_datasetRowsInBlock,
                m_options.m_datasetCapacity);
        }

        const bool usePrebuiltCrossTail =
            m_options.m_storage == Storage::STATIC &&
            m_options.m_buildSsdIndex &&
            m_options.m_buildCrossEdges &&
            m_options.m_tailReplicaCount > 0 &&
            m_headBundleNodes.size() > 1;
        if (usePrebuiltCrossTail) {
            if (EnsureStaticTailCrossEdges() != ErrorCode::Success) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                             "Failed to prepare cross edges before STATIC tail construction.\n");
                return ErrorCode::Fail;
            }
            auto* staticSearcher =
                dynamic_cast<ExtraStaticSearcher<T>*>(m_extraSearcher.get());
            if (staticSearcher == nullptr) {
                SPTAGLIB_LOG(
                    Helper::LogLevel::LL_Error,
                    "Single-seed STATIC cross-tail requires an unquantized static searcher.\n");
                return ErrorCode::Fail;
            }
            staticSearcher->SetStaticCrossGraphSearch(
                [this](
                    const T* p_target,
                    int p_ownerNode,
                    int p_candidateCount,
                    std::vector<std::pair<SizeType, float>>& p_candidates) {
                    return SearchStaticTailCrossGraph(
                        p_target, p_ownerNode, p_candidateCount, p_candidates);
                });
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
            if (m_metadataOnlyHeadStore &&
                SetupMetadataOnlyHeadStore(m_options.m_indexDirectory) != ErrorCode::Success) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                             "Failed to bind bundle samples to metadata-only head root.\n");
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

    // Remove the top-level head vector file (SPTAGHeadVectors.bin) when explicitly
    // requested, OR for bundle builds where it is pure redundancy: it is byte-identical
    // to HeadIndex/vectors.bin and at search-load is only a never-triggered fallback for
    // head count (SPTAGHeadVectorIDs.bin is always present). Dropping it removes one full
    // copy of the head vectors from disk.
    if (m_options.m_deleteHeadVectors || !m_pendingNodeHeadSelections.empty())
    {
        if (fileexists((m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile).c_str()) &&
            remove((m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile).c_str()) != 0)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "Head vector file can't be removed.\n");
        }
    }

    // ---- Dual-pool slim (metadata-only) head root ----
    // For bundle builds, rewrite HeadIndex/{vectors,tree,graph,deletes}.bin to a tiny
    // compatibility KDT and resolve ALL real head vectors from the per-bundle subgraph
    // stores (see SetupMetadataOnlyHeadStore). A head_metaonly.bin sidecar records
    // totalHeads/h1Split/dim. h1Split is set to totalHeads so no real head vector is
    // physically stored in the root. This removes the obsolete full global head graph
    // from the persisted index.
    if (!m_pendingNodeHeadSelections.empty() && m_index != nullptr &&
        !m_metadataOnlyHeadStore && m_pQuantizer == nullptr)
    {
        SizeType total = m_vectorTranslateMap.R();
        SizeType n_h1 = total;
        SizeType n_extra = 0;
        bool rolesOk = true;
        if (!m_pendingHeadRoles.empty())
        {
            if (static_cast<SizeType>(m_pendingHeadRoles.size()) != total)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                    "Slim head root: head_role size %zu != total heads %d; skipping slim root.\n",
                    m_pendingHeadRoles.size(), (int)total);
                rolesOk = false;
            }
            else
            {
                n_h1 = total;   // all heads resolve from bundle stores
                n_extra = 0;
            }
        }
        if (rolesOk)
        {
            auto slimValueType = m_options.m_valueType;
            auto slimDims = m_options.m_dim;
            SizeType physCount = 1; // compatibility dummy only; real heads live in bundles
            ByteArray slimBytes = ByteArray::Alloc(
                static_cast<size_t>(physCount) * static_cast<size_t>(slimDims) * sizeof(T));
            std::memset(slimBytes.Data(), 0, slimBytes.Length());
            auto slimVecSet = std::make_shared<BasicVectorSet>(
                slimBytes, slimValueType, static_cast<DimensionType>(slimDims), physCount);
            auto slim = SPTAG::VectorIndex::CreateInstance(
                SPTAG::IndexAlgoType::KDT, slimValueType);
            slim->SetParameter("DistCalcMethod",
                SPTAG::Helper::Convert::ConvertToString(m_options.m_distCalcMethod));
            slim->SetQuantizer(nullptr);
            for (const auto& iter : m_headParameters)
                slim->SetParameter(iter.first.c_str(), iter.second.c_str());
            if (slim->BuildIndex(slimVecSet, nullptr, false, true, true) != ErrorCode::Success)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Slim head root: failed to build slim KDT.\n");
                return ErrorCode::Fail;
            }
            // Swap the in-memory root for the slim KDT so any later serialization
            // (e.g. SaveAll -> SaveIndexData) writes ONLY the slim physical samples.
            // KDT::SaveIndexData serializes m_pSamples (physical), so this persists the
            // metadata-only root. Full H1 resolution is restored on load via the sidecar.
            m_index = slim;
            const std::string headDir =
                m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder;
            if (!WriteMetadataOnlyHeadStore(
                    headDir + FolderSep + "head_metaonly.bin",
                    total,
                    static_cast<DimensionType>(slimDims)))
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "Slim head root: failed writing metadata sidecar.\n");
                return ErrorCode::Fail;
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "Slim head root: wrote metadata-only root total=%d h1Split=%d U_extra=%d phys=%d dim=%d\n",
                (int)total, (int)n_h1, (int)n_extra, (int)physCount, (int)slimDims);
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
    if (!m_metadataOnlyHeadStore) {
        if (InitializeHeadBundleRuntime(m_options.m_indexDirectory) != ErrorCode::Success)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to initialize head bundle runtime.\n");
            return ErrorCode::Fail;
        }
        const std::string metadataRootSidecar =
            m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder +
            FolderSep + "head_metaonly.bin";
        if (fileexists(metadataRootSidecar.c_str()) &&
            SetupMetadataOnlyHeadStore(m_options.m_indexDirectory) != ErrorCode::Success) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Failed to initialize metadata-only head root.\n");
            return ErrorCode::Fail;
        }
    } else if (m_metaOnlyHeadVectorPtrs.empty() &&
               SetupMetadataOnlyHeadStore(m_options.m_indexDirectory) != ErrorCode::Success) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "Failed to initialize metadata-only head root.\n");
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
    auto storeParameter = [](std::vector<std::pair<std::string, std::string>>& p_parameters,
                             const char* p_name,
                             const char* p_newValue) {
        for (auto& parameter : p_parameters)
        {
            if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(parameter.first.c_str(), p_name))
            {
                parameter.second = p_newValue;
                return;
            }
        }
        p_parameters.emplace_back(p_name, p_newValue);
    };

    if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_section, "SearchSSDIndex"))
    {
        storeParameter(m_searchSSDParameters, p_param, p_value);

        // These are SSDServing control flags, not mutable runtime options.
        if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "isExecute") ||
            SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "BuildSsdIndex"))
        {
            return ErrorCode::Success;
        }

        const char* runtimeParam = p_param;
        if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "PostingPageLimit"))
        {
            runtimeParam = "SearchPostingPageLimit";
        }
        else if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "InternalResultNum"))
        {
            runtimeParam = "SearchInternalResultNum";
        }
        else if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "NumberOfThreads"))
        {
            runtimeParam = "SearchThreadNum";
        }
        // m_options is shared by construction and runtime. Capture the effective
        // construction value before the runtime overlay changes it so SaveConfig
        // can write the two sections independently.
        bool hasBuildValue = false;
        for (const auto& parameter : m_buildSSDParameters)
        {
            if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(parameter.first.c_str(), runtimeParam))
            {
                hasBuildValue = true;
                break;
            }
        }
        if (!hasBuildValue)
        {
            const std::string buildValue =
                m_options.GetParameter("BuildSSDIndex", runtimeParam);
            storeParameter(m_buildSSDParameters, runtimeParam, buildValue.c_str());
        }
        m_options.SetParameter("BuildSSDIndex", runtimeParam, p_value);
        return ErrorCode::Success;
    }

    if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_section, "BuildSSDIndex"))
    {
        storeParameter(m_buildSSDParameters, p_param, p_value);
    }

    if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_section, "BuildHead") &&
        !SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "isExecute"))
    {
        if (m_index != nullptr)
            return m_index->SetParameter(p_param, p_value);
        for (auto& parameter : m_headParameters) {
            if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(parameter.first.c_str(), p_param)) {
                parameter.second = p_value;
                return ErrorCode::Success;
            }
        }
        m_headParameters[p_param] = p_value;
    }
    else
    {
        m_options.SetParameter(p_section, p_param, p_value);
        double configuredRatio = 0.0;
        if (p_section != nullptr && p_param != nullptr && p_value != nullptr
            && SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_section, "SelectHead")
            && SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "Ratio")
            && SPTAG::Helper::Convert::ConvertStringTo<double>(p_value, configuredRatio)) {
            m_options.m_ratioExplicitlySet = true;
        }
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
    if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_section, "SearchSSDIndex"))
    {
        const char* canonicalParam = p_param;
        for (const auto& parameter : m_searchSSDParameters)
        {
            if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(parameter.first.c_str(), canonicalParam))
            {
                return parameter.second;
            }
        }

        const char* runtimeParam = canonicalParam;
        if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(canonicalParam, "PostingPageLimit"))
        {
            runtimeParam = "SearchPostingPageLimit";
        }
        else if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(canonicalParam, "InternalResultNum"))
        {
            runtimeParam = "SearchInternalResultNum";
        }
        return m_options.GetParameter("BuildSSDIndex", runtimeParam);
    }
    else if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_section, "BuildHead") &&
        !SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "isExecute"))
    {
        if (m_index != nullptr)
            return m_index->GetParameter(p_param);
        for (const auto& parameter : m_headParameters) {
            if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(parameter.first.c_str(), p_param)) {
                return parameter.second;
            }
        }
        return "Undefined!";
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
    if (m_options.m_numTagsPerVec > 0 || !m_headBundleNodes.empty()) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "[TaggedUpdate] this index requires AddIndexWithTags so posting tags and "
                     "pure/tail membership stay consistent.\n");
        return ErrorCode::Undefined;
    }

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
ErrorCode Index<T>::GetTaggedHeadLocation(SizeType p_headID, TaggedHeadLocation& p_location) const
{
    p_location = {};
    if (p_headID < 0) return ErrorCode::Key_OverFlow;

    // A non-slim single-head index is the original SPANN topology: the root
    // graph owns the posting IDs directly.
    if (!m_metadataOnlyHeadStore &&
        (m_headBundleNodes.empty() || m_headBundleNodes.size() == 1)) {
        if (m_index == nullptr || p_headID >= m_index->GetNumSamples()) {
            return ErrorCode::Key_OverFlow;
        }
        p_location.m_localHeadID = p_headID;
        return ErrorCode::Success;
    }

    for (size_t slot = 0; slot < m_headBundleNodes.size(); ++slot) {
        if (EnsureHeadBundleNodeLoaded(static_cast<int>(slot)) != ErrorCode::Success) {
            return ErrorCode::Fail;
        }
        const auto& localToGlobal = m_headBundleLocalToGlobalHIDs[slot];
        const auto it = std::find(localToGlobal.begin(), localToGlobal.end(), p_headID);
        if (it != localToGlobal.end()) {
            p_location.m_bundleSlot = static_cast<int>(slot);
            p_location.m_localHeadID = static_cast<SizeType>(it - localToGlobal.begin());
            return ErrorCode::Success;
        }
    }
    return ErrorCode::Key_OverFlow;
}

template <typename T>
ErrorCode Index<T>::MarkCrossEdgesDirty(const std::string& p_baseDir)
{
    if (m_headBundleNodes.size() <= 1) return ErrorCode::Success;
    m_headCrossEdgesDirty.store(true, std::memory_order_release);

    const std::string baseDir = p_baseDir.empty()
        ? (m_headBundleBaseDir.empty() ? m_options.m_indexDirectory : m_headBundleBaseDir)
        : p_baseDir;
    if (baseDir.empty()) return ErrorCode::FailedCreateFile;
    std::string headDir = baseDir;
    if (!headDir.empty() && headDir.back() != FolderSep) headDir += FolderSep;
    headDir += m_options.m_headIndexFolder;
    if (!EnsureDirectory(headDir)) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                     "[TaggedUpdate] cannot create cross-edge marker directory %s.\n",
                     headDir.c_str());
        return ErrorCode::FailedCreateFile;
    }
    const std::string marker = headDir + FolderSep + Helper::kHeadCrossEdgesDirtyFileName;
    std::ofstream out(marker + ".tmp", std::ios::trunc);
    if (!out) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                     "[TaggedUpdate] cannot mark cross-edge snapshot dirty at %s.\n",
                     marker.c_str());
        return ErrorCode::FailedCreateFile;
    }
    out << "subset-local head topology changed; rebuild head_cross_edges.bin before the next deployment\n";
    out.close();
    if (std::rename((marker + ".tmp").c_str(), marker.c_str()) != 0) {
        std::remove((marker + ".tmp").c_str());
        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                     "[TaggedUpdate] cannot publish cross-edge dirty marker %s.\n",
                     marker.c_str());
        return ErrorCode::FailedCreateFile;
    }
    return ErrorCode::Success;
}

template <typename T>
ErrorCode Index<T>::AddTaggedHeadToBundle(int p_bundleSlot, const T* p_center,
                                          SizeType p_anchorVID, SizeType p_templateHeadID,
                                          SizeType& p_headID)
{
    p_headID = -1;
    if (p_center == nullptr || p_anchorVID < 0 || m_extraSearcher == nullptr ||
        m_index == nullptr) {
        return ErrorCode::Fail;
    }
    if (m_metadataOnlyHeadStore && dynamic_cast<KDT::Index<T>*>(m_index.get()) == nullptr) {
        return ErrorCode::Fail;
    }

    std::shared_ptr<VectorIndex> graph;
    if (p_bundleSlot < 0) {
        graph = m_index;
    } else {
        if (p_bundleSlot >= static_cast<int>(m_headBundleNodes.size()) ||
            EnsureHeadBundleNodeLoaded(p_bundleSlot) != ErrorCode::Success) {
            return ErrorCode::Fail;
        }
        graph = m_loadedHeadBundleIndexes[static_cast<size_t>(p_bundleSlot)];
    }
    if (graph == nullptr) return ErrorCode::Fail;

    // Cross-edge snapshots cannot represent a newly appended local graph head.
    // Publish the dirty marker before any topology mutation becomes visible.
    const ErrorCode dirtyRet = MarkCrossEdgesDirty();
    if (dirtyRet != ErrorCode::Success) return dirtyRet;
    const SizeType newHeadID = m_vectorTranslateMap.R();
    int begin = -1;
    int end = -1;
    ErrorCode ret = graph->AddIndexId(p_center, 1, m_options.m_dim, begin, end);
    if (ret != ErrorCode::Success || begin < 0 || end != begin + 1) {
        return ret == ErrorCode::Success ? ErrorCode::Fail : ret;
    }
    ret = graph->AddIndexIdx(static_cast<SizeType>(begin), static_cast<SizeType>(end));
    if (ret != ErrorCode::Success) {
        graph->DeleteIndex(static_cast<SizeType>(begin));
        return ret;
    }

    ret = m_extraSearcher->ReserveTaggedPosting(newHeadID);
    if (ret != ErrorCode::Success) {
        graph->DeleteIndex(static_cast<SizeType>(begin));
        return ret;
    }
    if (m_vectorTranslateMap.AddBatch(1) != ErrorCode::Success) {
        graph->DeleteIndex(static_cast<SizeType>(begin));
        return ErrorCode::MemoryOverFlow;
    }
    *(m_vectorTranslateMap.At(newHeadID)) = static_cast<std::uint64_t>(p_anchorVID);

    if (m_index->HasHeadNodeMeta()) {
        const SizeType oldMetaCount = m_index->GetHeadNodeMetaSampleCount();
        const size_t stride = m_index->GetHeadNodeMetaStride();
        if (oldMetaCount != newHeadID || stride == 0) {
            graph->DeleteIndex(static_cast<SizeType>(begin));
            return ErrorCode::Fail;
        }
        auto& blob = m_index->GetHeadNodeMetaBlob();
        const size_t oldBytes = blob.size();
        blob.resize(oldBytes + stride, 0);
        if (p_templateHeadID >= 0 && p_templateHeadID < oldMetaCount) {
            std::memcpy(blob.data() + oldBytes,
                        blob.data() + static_cast<size_t>(p_templateHeadID) * stride,
                        stride);
        }
        m_index->SetHeadNodeGlobalVID(newHeadID, p_anchorVID);
        m_index->SetHeadNodeBundleNodeId(
            newHeadID, static_cast<std::int16_t>(p_bundleSlot < 0 ? 0 : p_bundleSlot));
    }

    if (p_bundleSlot >= 0) {
        auto& localToGlobal = m_headBundleLocalToGlobalHIDs[static_cast<size_t>(p_bundleSlot)];
        localToGlobal.push_back(newHeadID);
        auto& node = m_headBundleNodes[static_cast<size_t>(p_bundleSlot)];
        ++node.headCount;
        ++node.postingCount;
        {
            std::lock_guard<std::mutex> lock(m_globalHeadVIDToLocalHIDMutex);
            m_globalHeadVIDToLocalHID[p_anchorVID] = newHeadID;
        }
        {
            std::lock_guard<std::mutex> lock(m_globalVIDToBundleLocMutex);
            m_globalVIDToBundleLoc[p_anchorVID] =
                std::make_pair(p_bundleSlot, static_cast<SizeType>(begin));
        }
    } else if (!m_headBundleNodes.empty()) {
        ++m_headBundleNodes.front().headCount;
        ++m_headBundleNodes.front().postingCount;
    }

    {
        std::lock_guard<std::mutex> lock(m_headCrossEdgesMutex);
        if (!m_headBundleNodeByB.empty()) {
            m_headBundleNodeByB.resize(static_cast<size_t>(newHeadID) + 1, -1);
            m_headBundleLocalByB.resize(static_cast<size_t>(newHeadID) + 1, -1);
            m_headCrossEdgeOffsetByB.resize(static_cast<size_t>(newHeadID) + 1, UINT32_MAX);
            m_headCrossEdgeCountByB.resize(static_cast<size_t>(newHeadID) + 1, 0);
            m_headBundleNodeByB[static_cast<size_t>(newHeadID)] =
                static_cast<std::int16_t>(p_bundleSlot < 0 ? 0 : p_bundleSlot);
            m_headBundleLocalByB[static_cast<size_t>(newHeadID)] = static_cast<SizeType>(begin);
        }
    }
    if (m_metadataOnlyHeadStore) {
        auto* metadataOnlyIndex = dynamic_cast<KDT::Index<T>*>(m_index.get());
        metadataOnlyIndex->SetMetadataOnly(newHeadID + 1, newHeadID + 1);
        m_metaOnlyHeadVectorPtrs.resize(static_cast<size_t>(newHeadID) + 1, nullptr);
        m_metaOnlyHeadVectorPtrs[static_cast<size_t>(newHeadID)] =
            graph->GetSample(static_cast<SizeType>(begin));
    }

    p_headID = newHeadID;
    return ErrorCode::Success;
}

template <typename T>
void Index<T>::TombstoneTaggedHeads(ExtraWorkSpace* p_workspace,
                                    const std::vector<SizeType>& p_headIDs,
                                    const TaggedPostingSnapshot* p_restorePosting)
{
    if (p_workspace == nullptr || p_headIDs.empty() || m_extraSearcher == nullptr) return;

    if (MarkCrossEdgesDirty() != ErrorCode::Success) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "[TaggedUpdate] could not persist the cross-edge dirty marker during split rollback.\n");
    }
    std::vector<TaggedPostingSnapshot> rewrites;
    rewrites.reserve(p_headIDs.size() + (p_restorePosting == nullptr ? 0 : 1));
    if (p_restorePosting != nullptr) {
        rewrites.push_back(*p_restorePosting);
    }

    for (SizeType headID : p_headIDs) {
        TaggedHeadLocation location;
        if (GetTaggedHeadLocation(headID, location) == ErrorCode::Success) {
            std::shared_ptr<VectorIndex> graph = location.m_bundleSlot < 0
                ? m_index
                : m_loadedHeadBundleIndexes[static_cast<size_t>(location.m_bundleSlot)];
            if (graph != nullptr && graph->ContainSample(location.m_localHeadID)) {
                const ErrorCode ret = graph->DeleteIndex(location.m_localHeadID);
                if (ret != ErrorCode::Success && ret != ErrorCode::VectorNotFound) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                                 "[TaggedUpdate] failed to tombstone rollback head %d (error %d).\n",
                                 headID, static_cast<int>(ret));
                }
            }
        }
        TaggedPostingSnapshot empty;
        empty.m_headID = headID;
        rewrites.emplace_back(std::move(empty));
    }

    if (m_extraSearcher->RewriteTaggedPostings(p_workspace, rewrites) != ErrorCode::Success) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "[TaggedUpdate] failed to restore posting state while rolling back split heads.\n");
    }
}

template <typename T>
ErrorCode Index<T>::SplitTaggedPosting(ExtraWorkSpace* p_workspace, SizeType p_headID,
                                       const T* p_preferredCenter, SizeType p_preferredVID)
{
    if (p_workspace == nullptr || p_preferredCenter == nullptr || m_extraSearcher == nullptr) {
        return ErrorCode::Fail;
    }

    TaggedHeadLocation location;
    ErrorCode ret = GetTaggedHeadLocation(p_headID, location);
    if (ret != ErrorCode::Success) return ret;
    std::shared_ptr<VectorIndex> graph = location.m_bundleSlot < 0
        ? m_index
        : m_loadedHeadBundleIndexes[static_cast<size_t>(location.m_bundleSlot)];
    if (graph == nullptr || !graph->ContainSample(location.m_localHeadID)) {
        return ErrorCode::VectorNotFound;
    }

    TaggedPostingSnapshot source;
    ret = m_extraSearcher->GetTaggedPostingSnapshot(p_workspace, p_headID, source);
    if (ret != ErrorCode::Success) return ret;
    const int stride = m_extraSearcher->GetTaggedRecordSize();
    const int pureCapacity = m_extraSearcher->GetTaggedPureCapacity();
    if (stride <= 0 || pureCapacity <= 0 || source.m_records.empty()) {
        return ErrorCode::Posting_OverFlow;
    }
    const int total = static_cast<int>(source.m_records.size() / static_cast<size_t>(stride));

    struct LiveRecord
    {
        int m_record = -1;
        SizeType m_vid = -1;
    };
    std::vector<LiveRecord> pureRecords;
    std::vector<LiveRecord> tailRecords;
    pureRecords.reserve(static_cast<size_t>(source.m_pureCount));
    tailRecords.reserve(static_cast<size_t>(std::max(0, total - source.m_pureCount)));
    for (int record = 0; record < total; ++record) {
        const char* bytes = source.m_records.data() + static_cast<size_t>(record) * stride;
        SizeType vid = -1;
        std::uint8_t version = 0;
        std::memcpy(&vid, bytes, sizeof(vid));
        std::memcpy(&version, bytes + sizeof(vid), sizeof(version));
        if (vid < 0 || vid >= m_versionMap.Count() || m_versionMap.Deleted(vid) ||
            m_versionMap.GetVersion(vid) != version) {
            continue;
        }
        if (record < source.m_pureCount) pureRecords.push_back({record, vid});
        else tailRecords.push_back({record, vid});
    }

    if (static_cast<int>(pureRecords.size()) <= pureCapacity) {
        // Stale records consumed the buffer. Compact without changing topology.
        TaggedPostingSnapshot compacted;
        compacted.m_headID = p_headID;
        compacted.m_pureCount = static_cast<int>(pureRecords.size());
        for (const auto& record : pureRecords) {
            compacted.m_records.append(
                source.m_records.data() + static_cast<size_t>(record.m_record) * stride,
                static_cast<size_t>(stride));
        }
        for (const auto& record : tailRecords) {
            compacted.m_records.append(
                source.m_records.data() + static_cast<size_t>(record.m_record) * stride,
                static_cast<size_t>(stride));
        }
        return m_extraSearcher->RewriteTaggedPostings(p_workspace, {compacted});
    }
    if (pureRecords.size() < 2) return ErrorCode::Posting_OverFlow;

    // Centroids are learned only from the filtered-visible pure prefix. Tail
    // records are subsequently classified against those centroids.
    std::vector<SizeType> pureVIDs;
    pureVIDs.reserve(pureRecords.size());
    for (const auto& record : pureRecords) pureVIDs.push_back(record.m_vid);
    ByteArray pureVectors;
    ret = m_extraSearcher->ReadTaggedFullVectors(pureVIDs, pureVectors);
    if (ret != ErrorCode::Success) return ret;

    const SizeType pureCount = static_cast<SizeType>(pureRecords.size());
    COMMON::Dataset<T> samples(pureCount, m_options.m_dim,
                               std::max<SizeType>(1, m_index->m_iDataBlockSize),
                               std::max<SizeType>(pureCount + 1, m_index->m_iDataCapacity),
                               pureVectors.Data(), true);
    std::vector<int> localIndices(static_cast<size_t>(pureCount));
    std::iota(localIndices.begin(), localIndices.end(), 0);
    SPTAG::COMMON::KmeansArgs<T> args(2, samples.C(), pureCount, 1,
                                      graph->GetDistCalcMethod(), graph->m_pQuantizer);
    std::shuffle(localIndices.begin(), localIndices.end(), std::mt19937(std::random_device()()));
    const int clusters = SPTAG::COMMON::KmeansClustering(
        samples, localIndices, 0, pureCount, args, 1000, 100.0F, false, nullptr);
    if (clusters != 2 || args.counts[0] == 0 || args.counts[1] == 0) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "[TaggedUpdate] split of posting %d produced %d clusters (%d, %d).\n",
                     p_headID, clusters, args.counts[0], args.counts[1]);
        return ErrorCode::Posting_OverFlow;
    }

    std::vector<int> pureCluster(static_cast<size_t>(pureCount), -1);
    if (std::max(args.counts[0], args.counts[1]) > pureCapacity) {
        if (pureCount > static_cast<SizeType>(pureCapacity) * 2) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "[TaggedUpdate] split posting %d has %d live pure records, exceeding "
                         "the two-posting capacity %d.\n",
                         p_headID, pureCount, pureCapacity * 2);
            return ErrorCode::Posting_OverFlow;
        }

        // The original two-means implementation can isolate a small group of
        // old records and leave the new cluster above the hard posting limit.
        // Preserve every update by making the bisection capacity-safe, then
        // recompute both pure-derived centroids before graph insertion.
        std::vector<std::pair<float, SizeType>> ordered;
        ordered.reserve(static_cast<size_t>(pureCount));
        const T* vectors = reinterpret_cast<const T*>(pureVectors.Data());
        for (SizeType i = 0; i < pureCount; ++i) {
            const T* vector = vectors + static_cast<size_t>(i) * m_options.m_dim;
            ordered.emplace_back(graph->ComputeDistance(vector, args.centers) -
                                     graph->ComputeDistance(
                                         vector, args.centers + m_options.m_dim),
                                 i);
        }
        std::sort(ordered.begin(), ordered.end());
        std::array<std::vector<double>, 2> sums = {
            std::vector<double>(static_cast<size_t>(m_options.m_dim), 0.0),
            std::vector<double>(static_cast<size_t>(m_options.m_dim), 0.0)};
        args.counts[0] = 0;
        args.counts[1] = 0;
        for (SizeType rank = 0; rank < pureCount; ++rank) {
            const int cluster = rank < pureCount / 2 ? 0 : 1;
            const SizeType pureIndex = ordered[static_cast<size_t>(rank)].second;
            pureCluster[static_cast<size_t>(pureIndex)] = cluster;
            ++args.counts[cluster];
            const T* vector = vectors + static_cast<size_t>(pureIndex) * m_options.m_dim;
            for (DimensionType dim = 0; dim < m_options.m_dim; ++dim) {
                sums[static_cast<size_t>(cluster)][static_cast<size_t>(dim)] += vector[dim];
            }
        }
        for (int cluster = 0; cluster < 2; ++cluster) {
            T* center = args.centers + static_cast<size_t>(cluster) * m_options.m_dim;
            for (DimensionType dim = 0; dim < m_options.m_dim; ++dim) {
                center[dim] = static_cast<T>(
                    sums[static_cast<size_t>(cluster)][static_cast<size_t>(dim)] /
                    args.counts[cluster]);
            }
            if (graph->GetDistCalcMethod() == DistCalcMethod::Cosine) {
                COMMON::Utils::Normalize(center, m_options.m_dim, COMMON::Utils::GetBase<T>());
            }
        }
    } else {
        int offset = 0;
        for (int cluster = 0; cluster < 2; ++cluster) {
            for (int i = 0; i < args.counts[cluster]; ++i) {
                pureCluster[static_cast<size_t>(localIndices[static_cast<size_t>(offset + i)])] = cluster;
            }
            offset += args.counts[cluster];
        }
    }

    const T* oldCenter = reinterpret_cast<const T*>(graph->GetSample(location.m_localHeadID));
    const float center0Distance = graph->ComputeDistance(args.centers, oldCenter);
    const float center1Distance =
        graph->ComputeDistance(args.centers + m_options.m_dim, oldCenter);
    const int retainedCluster = center0Distance <= center1Distance ? 0 : 1;
    const bool retainOldHead =
        std::min(center0Distance, center1Distance) < Epsilon;

    std::array<std::vector<int>, 2> pureByCluster;
    std::array<std::vector<std::pair<float, int>>, 2> tailByCluster;
    for (SizeType i = 0; i < pureCount; ++i) {
        const int cluster = pureCluster[static_cast<size_t>(i)];
        if (cluster < 0) return ErrorCode::Fail;
        pureByCluster[static_cast<size_t>(cluster)].push_back(static_cast<int>(i));
    }

    if (!tailRecords.empty()) {
        std::vector<SizeType> tailVIDs;
        tailVIDs.reserve(tailRecords.size());
        for (const auto& record : tailRecords) tailVIDs.push_back(record.m_vid);
        ByteArray tailVectors;
        ret = m_extraSearcher->ReadTaggedFullVectors(tailVIDs, tailVectors);
        if (ret != ErrorCode::Success) return ret;
        const T* tailData = reinterpret_cast<const T*>(tailVectors.Data());
        for (size_t i = 0; i < tailRecords.size(); ++i) {
            const T* vector = tailData + i * static_cast<size_t>(m_options.m_dim);
            const float distance0 = graph->ComputeDistance(vector, args.centers);
            const float distance1 = graph->ComputeDistance(
                vector, args.centers + m_options.m_dim);
            const int cluster = distance0 <= distance1 ? 0 : 1;
            tailByCluster[static_cast<size_t>(cluster)].emplace_back(
                std::min(distance0, distance1), static_cast<int>(i));
        }
    }

    auto isKnownHeadVID = [&](SizeType vid) {
        if (vid == MaxSize || vid < 0) return true;
        {
            std::lock_guard<std::mutex> lock(m_globalHeadVIDToLocalHIDMutex);
            if (m_globalHeadVIDToLocalHID.find(vid) != m_globalHeadVIDToLocalHID.end()) return true;
        }
        for (SizeType head = 0; head < m_vectorTranslateMap.R(); ++head) {
            if (static_cast<SizeType>(*(m_vectorTranslateMap[head])) == vid) return true;
        }
        return false;
    };
    std::unordered_set<SizeType> reservedAnchors;
    auto selectAnchor = [&](int cluster, bool preferIncoming) -> SizeType {
        if (preferIncoming && !isKnownHeadVID(p_preferredVID) &&
            reservedAnchors.insert(p_preferredVID).second) {
            return p_preferredVID;
        }
        for (int pureIndex : pureByCluster[static_cast<size_t>(cluster)]) {
            const SizeType candidate = pureRecords[static_cast<size_t>(pureIndex)].m_vid;
            if (!isKnownHeadVID(candidate) && reservedAnchors.insert(candidate).second) {
                return candidate;
            }
        }
        return -1;
    };

    const SizeType firstNewHeadID = m_vectorTranslateMap.R();
    auto rollbackNewHeads = [&](const TaggedPostingSnapshot* restorePosting) {
        std::vector<SizeType> newHeads;
        for (SizeType headID = firstNewHeadID; headID < m_vectorTranslateMap.R(); ++headID) {
            newHeads.push_back(headID);
        }
        TombstoneTaggedHeads(p_workspace, newHeads, restorePosting);
    };

    std::array<SizeType, 2> heads = {-1, -1};
    if (retainOldHead) {
        heads[static_cast<size_t>(retainedCluster)] = p_headID;
        const int newCluster = 1 - retainedCluster;
        const SizeType anchor = selectAnchor(newCluster, true);
        if (anchor < 0) return ErrorCode::Posting_OverFlow;
        ret = AddTaggedHeadToBundle(location.m_bundleSlot,
                                    args.centers + static_cast<size_t>(newCluster) * m_options.m_dim,
                                    anchor, p_headID, heads[static_cast<size_t>(newCluster)]);
        if (ret != ErrorCode::Success) {
            rollbackNewHeads(nullptr);
            return ret;
        }
    } else {
        for (int cluster = 0; cluster < 2; ++cluster) {
            const SizeType anchor = selectAnchor(cluster, cluster == 0);
            if (anchor < 0) {
                rollbackNewHeads(nullptr);
                return ErrorCode::Posting_OverFlow;
            }
            ret = AddTaggedHeadToBundle(location.m_bundleSlot,
                                        args.centers + static_cast<size_t>(cluster) * m_options.m_dim,
                                        anchor, p_headID, heads[static_cast<size_t>(cluster)]);
            if (ret != ErrorCode::Success) {
                rollbackNewHeads(nullptr);
                return ret;
            }
        }
    }

    auto makeRewrite = [&](int cluster, SizeType targetHead) {
        TaggedPostingSnapshot rewrite;
        rewrite.m_headID = targetHead;
        std::unordered_set<SizeType> seen;
        for (int pureIndex : pureByCluster[static_cast<size_t>(cluster)]) {
            const auto& live = pureRecords[static_cast<size_t>(pureIndex)];
            if (seen.insert(live.m_vid).second) {
                rewrite.m_records.append(
                    source.m_records.data() + static_cast<size_t>(live.m_record) * stride,
                    static_cast<size_t>(stride));
                ++rewrite.m_pureCount;
            }
        }

        auto& tails = tailByCluster[static_cast<size_t>(cluster)];
        std::sort(tails.begin(), tails.end(),
                  [](const std::pair<float, int>& left, const std::pair<float, int>& right) {
                      return left.first < right.first;
                  });
        const size_t purePages =
            (static_cast<size_t>(rewrite.m_pureCount) * stride + PageSize - 1) / PageSize;
        const size_t maxRecords =
            ((purePages + static_cast<size_t>(m_options.m_unfilterTailBufferLength)) * PageSize) /
            static_cast<size_t>(stride);
        for (const auto& tail : tails) {
            if (rewrite.m_records.size() / static_cast<size_t>(stride) >= maxRecords) break;
            const auto& live = tailRecords[static_cast<size_t>(tail.second)];
            if (seen.insert(live.m_vid).second) {
                rewrite.m_records.append(
                    source.m_records.data() + static_cast<size_t>(live.m_record) * stride,
                    static_cast<size_t>(stride));
            }
        }
        return rewrite;
    };

    std::vector<TaggedPostingSnapshot> rewrites;
    rewrites.reserve(retainOldHead ? 2 : 3);
    for (int cluster = 0; cluster < 2; ++cluster) {
        rewrites.emplace_back(makeRewrite(cluster, heads[static_cast<size_t>(cluster)]));
    }
    if (!retainOldHead) {
        TaggedPostingSnapshot deleted;
        deleted.m_headID = p_headID;
        rewrites.emplace_back(std::move(deleted));
    }

    ret = m_extraSearcher->RewriteTaggedPostings(p_workspace, rewrites);
    if (ret != ErrorCode::Success) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "[TaggedUpdate] failed to rewrite split posting %d (error %d).\n",
                     p_headID, static_cast<int>(ret));
        for (const auto& rewrite : rewrites) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "[TaggedUpdate] rejected split target %d: pure=%d total=%zu capacity=%d.\n",
                         rewrite.m_headID, rewrite.m_pureCount,
                         rewrite.m_records.size() / static_cast<size_t>(stride), pureCapacity);
        }
        rollbackNewHeads(&source);
        return ret;
    }
    if (!retainOldHead) {
        graph->DeleteIndex(location.m_localHeadID);
    }
    return ErrorCode::Success;
}

template <typename T>
ErrorCode Index<T>::EnsureTaggedPureCapacity(ExtraWorkSpace* p_workspace,
                                             std::shared_ptr<VectorSet>& p_vectors,
                                             SizeType p_begin,
                                             PostingUpdateTargets& p_targets,
                                             bool& p_topologyChanged)
{
    p_topologyChanged = false;
    if (p_workspace == nullptr || p_vectors == nullptr || m_extraSearcher == nullptr) {
        return ErrorCode::Fail;
    }
    const int pureCapacity = m_extraSearcher->GetTaggedPureCapacity();
    if (pureCapacity <= 0) return ErrorCode::Posting_OverFlow;

    struct PendingPure
    {
        int m_count = 0;
        SizeType m_firstVectorOffset = -1;
    };
    std::unordered_map<SizeType, PendingPure> pending;
    for (SizeType vectorOffset = 0; vectorOffset < p_vectors->Count(); ++vectorOffset) {
        for (const PostingUpdateTarget& target : p_targets[static_cast<size_t>(vectorOffset)]) {
            if (target.m_kind != PostingUpdateKind::Pure) continue;
            PendingPure& targetPending = pending[target.m_headID];
            ++targetPending.m_count;
            if (targetPending.m_firstVectorOffset < 0) {
                targetPending.m_firstVectorOffset = vectorOffset;
            }
        }
    }

    for (const auto& pendingEntry : pending) {
        TaggedPostingSnapshot snapshot;
        ErrorCode ret = m_extraSearcher->GetTaggedPostingSnapshot(
            p_workspace, pendingEntry.first, snapshot);
        if (ret != ErrorCode::Success) return ret;
        if (snapshot.m_pureCount <= pureCapacity) continue;

        const SizeType vectorOffset = pendingEntry.second.m_firstVectorOffset;
        ret = SplitTaggedPosting(
            p_workspace, pendingEntry.first,
            reinterpret_cast<const T*>(p_vectors->GetVector(vectorOffset)),
            p_begin + vectorOffset);
        if (ret != ErrorCode::Success) return ret;
        p_topologyChanged = true;
        return ErrorCode::Success;
    }
    return ErrorCode::Success;
}

template <typename T>
ErrorCode Index<T>::MergeTaggedPosting(ExtraWorkSpace* p_workspace, SizeType p_headID)
{
    if (p_workspace == nullptr || m_extraSearcher == nullptr) return ErrorCode::Fail;

    TaggedHeadLocation currentLocation;
    ErrorCode ret = GetTaggedHeadLocation(p_headID, currentLocation);
    if (ret != ErrorCode::Success) return ret;
    std::shared_ptr<VectorIndex> graph = currentLocation.m_bundleSlot < 0
        ? m_index
        : m_loadedHeadBundleIndexes[static_cast<size_t>(currentLocation.m_bundleSlot)];
    if (graph == nullptr || !graph->ContainSample(currentLocation.m_localHeadID)) {
        return ErrorCode::VectorNotFound;
    }

    TaggedPostingSnapshot current;
    ret = m_extraSearcher->GetTaggedPostingSnapshot(p_workspace, p_headID, current);
    if (ret != ErrorCode::Success) return ret;
    const int stride = m_extraSearcher->GetTaggedRecordSize();
    const int pureCapacity = m_extraSearcher->GetTaggedPureCapacity();
    if (stride <= 0 || pureCapacity <= 0) return ErrorCode::Fail;

    struct RecordRef
    {
        SizeType m_vid = -1;
        const char* m_record = nullptr;
    };
    auto extractLive = [&](const TaggedPostingSnapshot& snapshot,
                           std::vector<RecordRef>& pure,
                           std::vector<RecordRef>& tail) -> ErrorCode {
        if (snapshot.m_records.size() % static_cast<size_t>(stride) != 0) {
            return ErrorCode::Posting_SizeError;
        }
        const int count = static_cast<int>(snapshot.m_records.size() / static_cast<size_t>(stride));
        if (snapshot.m_pureCount < 0 || snapshot.m_pureCount > count) {
            return ErrorCode::Posting_SizeError;
        }
        for (int i = 0; i < count; ++i) {
            const char* record = snapshot.m_records.data() + static_cast<size_t>(i) * stride;
            SizeType vid = -1;
            std::uint8_t version = 0;
            std::memcpy(&vid, record, sizeof(vid));
            std::memcpy(&version, record + sizeof(vid), sizeof(version));
            if (vid < 0 || vid >= m_versionMap.Count() || m_versionMap.Deleted(vid) ||
                m_versionMap.GetVersion(vid) != version) {
                continue;
            }
            (i < snapshot.m_pureCount ? pure : tail).push_back({vid, record});
        }
        return ErrorCode::Success;
    };

    std::vector<RecordRef> currentPure;
    std::vector<RecordRef> currentTail;
    ret = extractLive(current, currentPure, currentTail);
    if (ret != ErrorCode::Success) return ret;

    auto compactCurrent = [&]() -> ErrorCode {
        TaggedPostingSnapshot compacted;
        compacted.m_headID = p_headID;
        std::unordered_set<SizeType> seen;
        for (const RecordRef& record : currentPure) {
            if (seen.insert(record.m_vid).second) {
                compacted.m_records.append(record.m_record, static_cast<size_t>(stride));
                ++compacted.m_pureCount;
            }
        }
        for (const RecordRef& record : currentTail) {
            if (seen.insert(record.m_vid).second) {
                compacted.m_records.append(record.m_record, static_cast<size_t>(stride));
            }
        }
        return m_extraSearcher->RewriteTaggedPostings(p_workspace, {compacted});
    };

    if (static_cast<int>(currentPure.size()) > m_extraSearcher->GetTaggedMergeThreshold()) {
        return compactCurrent();
    }

    COMMON::QueryResultSet<T> nearby(
        const_cast<T*>(reinterpret_cast<const T*>(graph->GetSample(currentLocation.m_localHeadID))),
        std::max(2, m_options.m_internalResultNum));
    std::shared_ptr<std::uint8_t> reconstructed;
    if (graph->m_pQuantizer) {
        reconstructed.reset(
            static_cast<std::uint8_t*>(ALIGN_ALLOC(graph->m_pQuantizer->ReconstructSize())),
            [](std::uint8_t* ptr) { ALIGN_FREE(ptr); });
        if (!reconstructed) return ErrorCode::MemoryOverFlow;
        graph->m_pQuantizer->ReconstructVector(
            reinterpret_cast<const std::uint8_t*>(nearby.GetTarget()), reconstructed.get());
        nearby.SetTarget(reinterpret_cast<T*>(reconstructed.get()), graph->m_pQuantizer);
    }
    ret = graph->SearchIndex(nearby);
    if (ret != ErrorCode::Success) return ret;

    SizeType neighborHeadID = -1;
    SizeType neighborLocalID = -1;
    TaggedPostingSnapshot neighbor;
    std::vector<RecordRef> neighborPure;
    std::vector<RecordRef> neighborTail;
    for (int result = 0; result < nearby.GetResultNum(); ++result) {
        const BasicResult* candidate = nearby.GetResult(result);
        if (candidate == nullptr || candidate->VID < 0 ||
            candidate->VID == currentLocation.m_localHeadID ||
            !graph->ContainSample(candidate->VID)) {
            continue;
        }
        SizeType candidateGlobal = candidate->VID;
        if (currentLocation.m_bundleSlot >= 0) {
            const auto& localToGlobal =
                m_headBundleLocalToGlobalHIDs[static_cast<size_t>(currentLocation.m_bundleSlot)];
            if (static_cast<size_t>(candidate->VID) >= localToGlobal.size()) continue;
            candidateGlobal = localToGlobal[static_cast<size_t>(candidate->VID)];
        }
        if (candidateGlobal == p_headID) continue;

        TaggedPostingSnapshot candidateSnapshot;
        ret = m_extraSearcher->GetTaggedPostingSnapshot(
            p_workspace, candidateGlobal, candidateSnapshot);
        if (ret != ErrorCode::Success) return ret;
        std::vector<RecordRef> candidatePure;
        std::vector<RecordRef> candidateTail;
        ret = extractLive(candidateSnapshot, candidatePure, candidateTail);
        if (ret != ErrorCode::Success) return ret;

        std::unordered_set<SizeType> distinctPure;
        for (const RecordRef& record : currentPure) distinctPure.insert(record.m_vid);
        for (const RecordRef& record : candidatePure) distinctPure.insert(record.m_vid);
        if (static_cast<int>(distinctPure.size()) > pureCapacity) continue;

        neighborHeadID = candidateGlobal;
        neighborLocalID = candidate->VID;
        neighbor = std::move(candidateSnapshot);
        neighborPure = std::move(candidatePure);
        neighborTail = std::move(candidateTail);
        break;
    }
    if (neighborHeadID < 0) return compactCurrent();

    const bool currentSurvives = currentPure.size() >= neighborPure.size();
    const SizeType survivorHeadID = currentSurvives ? p_headID : neighborHeadID;
    const SizeType loserHeadID = currentSurvives ? neighborHeadID : p_headID;
    const SizeType survivorLocalID =
        currentSurvives ? currentLocation.m_localHeadID : neighborLocalID;
    if (survivorLocalID < 0) return ErrorCode::Fail;

    const std::vector<RecordRef>& firstPure = currentSurvives ? currentPure : neighborPure;
    const std::vector<RecordRef>& secondPure = currentSurvives ? neighborPure : currentPure;
    const std::vector<RecordRef>& firstTail = currentSurvives ? currentTail : neighborTail;
    const std::vector<RecordRef>& secondTail = currentSurvives ? neighborTail : currentTail;
    TaggedPostingSnapshot merged;
    merged.m_headID = survivorHeadID;
    std::unordered_set<SizeType> seen;
    for (const RecordRef& record : firstPure) {
        if (seen.insert(record.m_vid).second) {
            merged.m_records.append(record.m_record, static_cast<size_t>(stride));
            ++merged.m_pureCount;
        }
    }
    for (const RecordRef& record : secondPure) {
        if (seen.insert(record.m_vid).second) {
            merged.m_records.append(record.m_record, static_cast<size_t>(stride));
            ++merged.m_pureCount;
        }
    }
    if (merged.m_pureCount > pureCapacity) return ErrorCode::Posting_OverFlow;

    std::vector<RecordRef> tailRecords;
    tailRecords.reserve(firstTail.size() + secondTail.size());
    for (const RecordRef& record : firstTail) {
        if (seen.insert(record.m_vid).second) tailRecords.push_back(record);
    }
    for (const RecordRef& record : secondTail) {
        if (seen.insert(record.m_vid).second) tailRecords.push_back(record);
    }
    if (!tailRecords.empty()) {
        std::vector<SizeType> tailVIDs;
        tailVIDs.reserve(tailRecords.size());
        for (const RecordRef& record : tailRecords) tailVIDs.push_back(record.m_vid);
        ByteArray tailVectors;
        ret = m_extraSearcher->ReadTaggedFullVectors(tailVIDs, tailVectors);
        if (ret != ErrorCode::Success) return ret;
        const T* vectors = reinterpret_cast<const T*>(tailVectors.Data());
        const T* survivorCenter =
            reinterpret_cast<const T*>(graph->GetSample(survivorLocalID));
        std::vector<std::pair<float, size_t>> sortedTails;
        sortedTails.reserve(tailRecords.size());
        for (size_t i = 0; i < tailRecords.size(); ++i) {
            sortedTails.emplace_back(
                graph->ComputeDistance(vectors + i * static_cast<size_t>(m_options.m_dim),
                                       survivorCenter),
                i);
        }
        std::sort(sortedTails.begin(), sortedTails.end());
        const size_t purePages =
            (static_cast<size_t>(merged.m_pureCount) * stride + PageSize - 1) / PageSize;
        const size_t maxRecords =
            ((purePages + static_cast<size_t>(m_options.m_unfilterTailBufferLength)) * PageSize) /
            static_cast<size_t>(stride);
        for (const auto& tail : sortedTails) {
            if (merged.m_records.size() / static_cast<size_t>(stride) >= maxRecords) break;
            merged.m_records.append(tailRecords[tail.second].m_record,
                                    static_cast<size_t>(stride));
        }
    }

    TaggedPostingSnapshot deleted;
    deleted.m_headID = loserHeadID;
    ret = MarkCrossEdgesDirty();
    if (ret != ErrorCode::Success) return ret;
    ret = m_extraSearcher->RewriteTaggedPostings(p_workspace, {merged, deleted});
    if (ret != ErrorCode::Success) return ret;
    const SizeType loserLocalID =
        currentSurvives ? neighborLocalID : currentLocation.m_localHeadID;
    ret = graph->DeleteIndex(loserLocalID);
    if (ret != ErrorCode::Success) return ret;

    // Preserve the local-ID/anchor mapping for the tombstoned graph sample.
    // As in original SPANN, existing graph/tree/cross-edge references resolve
    // normally and ContainSample suppresses the deleted head at query time.
    return ErrorCode::Success;
}

template <typename T>
ErrorCode Index<T>::DrainTaggedMergeMaintenance()
{
    if (m_extraSearcher == nullptr) return ErrorCode::Success;
    std::vector<SizeType> candidates;
    m_extraSearcher->DrainTaggedMergeCandidates(candidates);
    if (candidates.empty()) return ErrorCode::Success;
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                 "[TaggedUpdate] processing %zu subset-local merge candidates.\n",
                 candidates.size());

    auto workspace = m_workSpaceFactory->GetWorkSpace();
    if (!workspace) {
        workspace.reset(new ExtraWorkSpace());
        m_extraSearcher->InitWorkSpace(workspace.get(), false);
    } else {
        m_extraSearcher->InitWorkSpace(workspace.get(), true);
    }
    workspace->m_deduper.clear();
    workspace->m_postingIDs.clear();
    std::unique_lock<std::shared_timed_mutex> topologyLock(m_headTopologyLock);
    for (SizeType headID : candidates) {
        const ErrorCode ret = MergeTaggedPosting(workspace.get(), headID);
        if (ret != ErrorCode::Success && ret != ErrorCode::VectorNotFound &&
            ret != ErrorCode::Key_OverFlow) {
            return ret;
        }
    }
    return ErrorCode::Success;
}

template <typename T>
ErrorCode Index<T>::AddIndexWithTags(const void* p_data, SizeType p_vectorNum,
                                     DimensionType p_dimension, const uint32_t* p_tags,
                                     int p_numTagsPerVec, bool p_normalized)
{
    if (m_options.m_storage == Storage::STATIC || m_extraSearcher == nullptr) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[TaggedUpdate] only FileIO/KV postings support updates.\n");
        return ErrorCode::Fail;
    }
    if (p_data == nullptr || p_tags == nullptr || p_vectorNum <= 0 ||
        p_dimension != GetFeatureDim() || p_numTagsPerVec != m_options.m_numTagsPerVec) {
        return p_dimension != GetFeatureDim() ? ErrorCode::DimensionSizeMismatch : ErrorCode::Fail;
    }
    std::shared_lock<std::shared_timed_mutex> checkpointLock(m_checkPointLock);
    std::unique_lock<std::mutex> addLock(m_dataAddLock);
    std::unique_lock<std::shared_timed_mutex> topologyLock(m_headTopologyLock);

    std::shared_ptr<VectorSet> vectorSet;
    if (m_options.m_distCalcMethod == DistCalcMethod::Cosine && !p_normalized) {
        ByteArray arr = ByteArray::Alloc(sizeof(T) * p_vectorNum * p_dimension);
        memcpy(arr.Data(), p_data, sizeof(T) * p_vectorNum * p_dimension);
        vectorSet.reset(new BasicVectorSet(arr, GetEnumValueType<T>(), p_dimension, p_vectorNum));
        const int base = COMMON::Utils::GetBase<T>();
        for (SizeType i = 0; i < p_vectorNum; ++i) {
            COMMON::Utils::Normalize(reinterpret_cast<T*>(vectorSet->GetVector(i)), p_dimension, base);
        }
    } else {
        vectorSet.reset(new BasicVectorSet(
            ByteArray(reinterpret_cast<std::uint8_t*>(const_cast<void*>(p_data)),
                      sizeof(T) * p_vectorNum * p_dimension, false),
            GetEnumValueType<T>(), p_dimension, p_vectorNum));
    }

    struct HeadCandidate {
        SizeType global = -1;
        SizeType local = -1;
        int bundleSlot = -1;
        float distance = std::numeric_limits<float>::max();
    };

    auto findBundleSlot = [&](int nodeId) -> int {
        for (size_t slot = 0; slot < m_headBundleNodes.size(); ++slot) {
            if (m_headBundleNodes[slot].nodeId == nodeId) return static_cast<int>(slot);
        }
        return -1;
    };
    const bool useBundleRuntime = !m_headBundleNodes.empty() &&
                                  (m_headBundleNodes.size() > 1 || m_metadataOnlyHeadStore);

    int routingPivotLevel = -1;
    std::unordered_map<std::uint32_t, std::vector<int>> tagToNodes;
    const std::string routingBaseDir = m_options.m_recovery
        ? m_options.m_persistentBufferPath
        : m_options.m_indexDirectory;
    if (useBundleRuntime && m_headBundleNodes.size() > 1 &&
        !LoadUpdateRoutingNodes(routingBaseDir, m_options.m_headIndexFolder,
                                routingPivotLevel, tagToNodes)) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "[TaggedUpdate] multi-bundle index is missing a valid tag_node_index.bin routing map.\n");
        return ErrorCode::Fail;
    }

    auto collectCandidates = [&](const T* vector, int bundleSlot, int requestCount,
                                 std::vector<HeadCandidate>& output) -> ErrorCode {
        std::shared_ptr<VectorIndex> headIndex;
        const std::vector<SizeType>* localToGlobal = nullptr;
        if (bundleSlot >= 0) {
            if (EnsureHeadBundleNodeLoaded(bundleSlot) != ErrorCode::Success) return ErrorCode::Fail;
            headIndex = m_loadedHeadBundleIndexes[static_cast<size_t>(bundleSlot)];
            localToGlobal = &m_headBundleLocalToGlobalHIDs[static_cast<size_t>(bundleSlot)];
        } else {
            headIndex = m_index;
        }
        if (headIndex == nullptr || headIndex->GetNumSamples() == 0) return ErrorCode::Fail;

        COMMON::QueryResultSet<T> results(const_cast<T*>(vector),
                                          std::min(requestCount, static_cast<int>(headIndex->GetNumSamples())));
        std::shared_ptr<std::uint8_t> reconstructed;
        if (headIndex->m_pQuantizer) {
            reconstructed.reset(static_cast<std::uint8_t*>(ALIGN_ALLOC(headIndex->m_pQuantizer->ReconstructSize())),
                                [](std::uint8_t* ptr) { ALIGN_FREE(ptr); });
            headIndex->m_pQuantizer->ReconstructVector(
                reinterpret_cast<const std::uint8_t*>(results.GetTarget()), reconstructed.get());
            results.SetTarget(reinterpret_cast<T*>(reconstructed.get()), headIndex->m_pQuantizer);
        }
        if (headIndex->SearchIndex(results) != ErrorCode::Success) return ErrorCode::Fail;
        for (int i = 0; i < results.GetResultNum(); ++i) {
            const BasicResult* result = results.GetResult(i);
            if (result == nullptr || result->VID < 0) continue;
            if (localToGlobal != nullptr &&
                static_cast<size_t>(result->VID) >= localToGlobal->size()) {
                return ErrorCode::Fail;
            }
            output.push_back({localToGlobal == nullptr ? result->VID
                                                        : (*localToGlobal)[static_cast<size_t>(result->VID)],
                              result->VID, bundleSlot, result->Dist});
        }
        return ErrorCode::Success;
    };

    const int pureRequestCount = std::max(m_options.m_internalResultNum,
                                          std::max(8, m_options.m_replicaCount * 4));
    const int tailRequestCount = std::max(m_options.m_internalResultNum,
                                          std::max(8, m_options.m_tailReplicaCount * 4));
    auto buildPlans = [&](PostingUpdateTargets& p_plans) -> ErrorCode {
        p_plans.assign(static_cast<size_t>(p_vectorNum), {});
        for (SizeType vectorOffset = 0; vectorOffset < p_vectorNum; ++vectorOffset) {
            const std::uint32_t* tags =
                p_tags + static_cast<size_t>(vectorOffset) * static_cast<size_t>(p_numTagsPerVec);
            int pureBundleSlot = -1;
            if (useBundleRuntime) {
                if (m_headBundleNodes.size() == 1) {
                    pureBundleSlot = 0;
                } else {
                    if (routingPivotLevel < 0 || routingPivotLevel >= p_numTagsPerVec) {
                        return ErrorCode::Fail;
                    }
                    const auto routeIt = tagToNodes.find(tags[routingPivotLevel]);
                    if (routeIt == tagToNodes.end() || routeIt->second.size() != 1) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                     "[TaggedUpdate] tag %u does not resolve to one bundle at pivot level %d.\n",
                                     tags[routingPivotLevel], routingPivotLevel);
                        return ErrorCode::Fail;
                    }
                    pureBundleSlot = findBundleSlot(routeIt->second.front());
                    if (pureBundleSlot < 0) return ErrorCode::Fail;
                }
            }

            std::vector<HeadCandidate> pureCandidates;
            ErrorCode ret = collectCandidates(
                reinterpret_cast<const T*>(vectorSet->GetVector(vectorOffset)), pureBundleSlot,
                pureRequestCount, pureCandidates);
            if (ret != ErrorCode::Success) return ret;
            std::sort(pureCandidates.begin(), pureCandidates.end(),
                      [](const HeadCandidate& left, const HeadCandidate& right) {
                          return left.distance < right.distance;
                      });

            std::vector<HeadCandidate> pureHeads;
            for (const HeadCandidate& candidate : pureCandidates) {
                if (static_cast<int>(pureHeads.size()) >= m_options.m_replicaCount) break;
                bool duplicate = false;
                for (const HeadCandidate& existing : pureHeads) {
                    if (existing.global == candidate.global) {
                        duplicate = true;
                        break;
                    }
                    if (candidate.bundleSlot == existing.bundleSlot) {
                        const std::shared_ptr<VectorIndex>& bundleIndex =
                            candidate.bundleSlot < 0 ? m_index
                                                    : m_loadedHeadBundleIndexes[static_cast<size_t>(candidate.bundleSlot)];
                        const float headDistance = bundleIndex->ComputeDistance(
                            bundleIndex->GetSample(candidate.local), bundleIndex->GetSample(existing.local));
                        if (m_options.m_rngFactor * headDistance <= candidate.distance) {
                            duplicate = true;
                            break;
                        }
                    }
                }
                if (!duplicate) pureHeads.push_back(candidate);
            }
            if (pureHeads.empty()) return ErrorCode::Fail;
            for (const HeadCandidate& head : pureHeads) {
                p_plans[static_cast<size_t>(vectorOffset)].push_back(
                    {head.global, PostingUpdateKind::Pure});
            }

            if (m_options.m_tailReplicaCount <= 0) continue;
            std::vector<HeadCandidate> tailCandidates;
            if (!useBundleRuntime) {
                ret = collectCandidates(reinterpret_cast<const T*>(vectorSet->GetVector(vectorOffset)), -1,
                                        tailRequestCount, tailCandidates);
                if (ret != ErrorCode::Success) return ret;
            } else {
                for (size_t slot = 0; slot < m_headBundleNodes.size(); ++slot) {
                    ret = collectCandidates(reinterpret_cast<const T*>(vectorSet->GetVector(vectorOffset)),
                                           static_cast<int>(slot), tailRequestCount, tailCandidates);
                    if (ret != ErrorCode::Success) return ret;
                }
            }
            std::sort(tailCandidates.begin(), tailCandidates.end(),
                      [](const HeadCandidate& left, const HeadCandidate& right) {
                          return left.distance < right.distance;
                      });
            int tailCount = 0;
            for (const HeadCandidate& candidate : tailCandidates) {
                if (tailCount >= m_options.m_tailReplicaCount) break;
                bool alreadyPure = false;
                for (const HeadCandidate& pure : pureHeads) {
                    if (pure.global == candidate.global) {
                        alreadyPure = true;
                        break;
                    }
                }
                if (alreadyPure) continue;
                bool duplicateTail = false;
                for (const PostingUpdateTarget& existing : p_plans[static_cast<size_t>(vectorOffset)]) {
                    if (existing.m_headID == candidate.global) {
                        duplicateTail = true;
                        break;
                    }
                }
                if (duplicateTail) continue;
                p_plans[static_cast<size_t>(vectorOffset)].push_back(
                    {candidate.global, PostingUpdateKind::Tail});
                ++tailCount;
            }
        }
        return ErrorCode::Success;
    };

    PostingUpdateTargets plans;
    ErrorCode planningRet = buildPlans(plans);
    if (planningRet != ErrorCode::Success) return planningRet;

    const SizeType begin = m_versionMap.GetVectorNum();
    if (begin == 0 || m_versionMap.AddBatch(p_vectorNum) != ErrorCode::Success) {
        return begin == 0 ? ErrorCode::EmptyIndex : ErrorCode::MemoryOverFlow;
    }
    if (m_pMetadata != nullptr) {
        for (SizeType i = 0; i < p_vectorNum; ++i) m_pMetadata->Add(ByteArray::c_empty);
    }

    auto workSpace = m_workSpaceFactory->GetWorkSpace();
    if (!workSpace) {
        workSpace.reset(new ExtraWorkSpace());
        m_extraSearcher->InitWorkSpace(workSpace.get(), false);
    } else {
        m_extraSearcher->InitWorkSpace(workSpace.get(), true);
    }
    workSpace->m_deduper.clear();
    workSpace->m_postingIDs.clear();

    // Preserve original SPANN ordering: append the selected records first,
    // then cluster the genuinely overfull posting, including this update.
    ErrorCode ret = m_extraSearcher->AddIndexWithTargets(
        workSpace.get(), vectorSet, plans, p_tags, p_numTagsPerVec, begin);
    if (ret == ErrorCode::Success) {
        for (int splitRound = 0; splitRound < 64; ++splitRound) {
            bool topologyChanged = false;
            ret = EnsureTaggedPureCapacity(
                workSpace.get(), vectorSet, begin, plans, topologyChanged);
            if (ret != ErrorCode::Success || !topologyChanged) break;
            if (splitRound == 63) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                             "[TaggedUpdate] pure posting split did not converge after 64 rounds.\n");
                ret = ErrorCode::Posting_OverFlow;
                break;
            }
        }
        if (ret == ErrorCode::Success) {
            m_options.m_vectorSize = m_versionMap.GetVectorNum();
        }
    }
    if (ret != ErrorCode::Success) {
        // A multi-posting write can fail after earlier targets have been rewritten.
        // Version invalidation makes any partial record unreachable without
        // leaving a partially searchable update behind.
        for (SizeType i = 0; i < p_vectorNum; ++i) {
            m_versionMap.Delete(begin + i);
        }
    }
    return ret;
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
