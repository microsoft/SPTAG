// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/CoreInterface.h"
#include "inc/Helper/StringConvert.h"
#include "inc/Helper/TenantPrefixedKeyValueIO.h"
#include "inc/Core/SPANN/Index.h"
#ifdef ROCKSDB
#include "inc/Core/SPANN/ExtraRocksDBController.h"
#endif

#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <thread>
#include "inc/Core/SPANN/Options.h"
#include "inc/Core/SPANN/ExtraFileController.h"
#include "inc/Core/SPANN/Index.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <vector>
#include <sstream>
#include <cstring>
#include <unordered_set>
#include <limits>
#include <sys/stat.h>
#include <sys/types.h>
#include <cstdlib>
#ifdef __linux__
#include <unistd.h>
#endif
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

namespace {

struct TagRoutingStatRecord {
    uint32_t tag;
    int32_t vectorCount;
    int32_t postingCount;
};

static_assert(sizeof(TagRoutingStatRecord) == sizeof(uint32_t) + 2 * sizeof(int32_t),
              "Unexpected TagRoutingStatRecord layout");

bool EnsureDir(const std::string& path)
{
    if (path.empty()) return false;

    std::string cmd = "mkdir -p \"" + path + "\"";
    return std::system(cmd.c_str()) == 0;
}

bool RemovePathRecursive(const std::string& path)
{
    if (path.empty()) return false;

    std::string cmd = "rm -rf \"" + path + "\"";
    return std::system(cmd.c_str()) == 0;
}

bool CopyDirRecursive(const std::string& src, const std::string& dst)
{
    if (src.empty() || dst.empty()) return false;

    if (!RemovePathRecursive(dst)) return false;

    std::string cmd = "cp -a \"" + src + "\" \"" + dst + "\"";
    return std::system(cmd.c_str()) == 0;
}

uint64_t GetPathSizeBytes(const std::string& path)
{
    struct stat st;
    if (lstat(path.c_str(), &st) != 0) {
        return 0;
    }

    if (S_ISREG(st.st_mode)) {
        return static_cast<uint64_t>(st.st_size);
    }

    if (!S_ISDIR(st.st_mode)) {
        return 0;
    }

    DIR* dir = opendir(path.c_str());
    if (dir == nullptr) {
        return 0;
    }

    uint64_t totalBytes = 0;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        totalBytes += GetPathSizeBytes(path + "/" + entry->d_name);
    }

    closedir(dir);
    return totalBytes;
}

uint64_t GetCurrentProcessRSSBytes()
{
#ifdef __linux__
    long rssPages = 0;
    FILE* statm = fopen("/proc/self/statm", "r");
    if (statm == nullptr) {
        return 0;
    }

    int scanned = fscanf(statm, "%*s %ld", &rssPages);
    fclose(statm);
    if (scanned != 1 || rssPages <= 0) {
        return 0;
    }

    long pageSize = sysconf(_SC_PAGESIZE);
    if (pageSize <= 0) {
        return 0;
    }

    return static_cast<uint64_t>(rssPages) * static_cast<uint64_t>(pageSize);
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc))) {
        return static_cast<uint64_t>(pmc.WorkingSetSize);
    }
    return 0;
#else
    return 0;
#endif
}

} // namespace

namespace {

int TagLevel(uint32_t tag)
{
    return static_cast<int>(tag / 1000U);
}

float EstimateQueryVectorSelectivity(
    int tenantSize,
    const std::unordered_map<uint32_t, TenantIndexManager::TagRoutingStats>* tagStats,
    const uint32_t* queryTags,
    int numQueryTags)
{
    if (tenantSize <= 0 || tagStats == nullptr || queryTags == nullptr || numQueryTags <= 0) {
        return 1.0f;
    }

    std::unordered_set<uint32_t> seenTags;
    std::unordered_map<int, double> levelSelectivities;
    for (int index = 0; index < numQueryTags; ++index) {
        uint32_t tag = queryTags[index];
        if (!seenTags.insert(tag).second) {
            continue;
        }

        auto statIt = tagStats->find(tag);
        if (statIt == tagStats->end()) {
            continue;
        }

        double tagSelectivity = static_cast<double>(statIt->second.vectorCount) / static_cast<double>(tenantSize);
        int level = TagLevel(tag);
        double& levelSel = levelSelectivities[level];
        levelSel = std::min(1.0, levelSel + tagSelectivity);
    }

    if (levelSelectivities.empty()) {
        return 1.0f;
    }

    double productNotSelected = 1.0;
    for (const auto& [level, selectivity] : levelSelectivities) {
        (void)level;
        productNotSelected *= std::max(0.0, 1.0 - selectivity);
    }

    double unionSelectivity = 1.0 - productNotSelected;
    unionSelectivity = std::clamp(unionSelectivity, 1e-6, 1.0);
    return static_cast<float>(unionSelectivity);
}

struct PivotEstimatorLevelData {
    std::vector<uint32_t> uniqueTags;
    std::vector<int> counts;
    std::unordered_map<uint32_t, uint32_t> parentByTag;
};

struct PivotEstimatorCandidate {
    int pivotLevel = -1;
    int nodeCount = 0;
    double latencyCost = 0.0;
    double recallPenalty = 0.0;
    double totalCost = std::numeric_limits<double>::infinity();
    std::vector<std::vector<uint32_t>> nodePivotTags;
    std::vector<int> nodeSizes;
};

std::string JsonEscape(const std::string& input)
{
    std::string out;
    out.reserve(input.size() + 8);
    for (char ch : input)
    {
        switch (ch)
        {
        case '\\': out += "\\\\"; break;
        case '"': out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default: out.push_back(ch); break;
        }
    }
    return out;
}

bool ParseLevelWeights(const std::string& csv, int levelCount, std::vector<double>& outWeights)
{
    outWeights.clear();
    if (levelCount <= 0) return false;

    if (csv.empty()) {
        outWeights.assign(levelCount, 1.0 / static_cast<double>(levelCount));
        return true;
    }

    std::stringstream ss(csv);
    std::string token;
    while (std::getline(ss, token, ','))
    {
        if (token.empty()) continue;
        try {
            outWeights.push_back(std::stod(token));
        }
        catch (...) {
            return false;
        }
    }

    if (static_cast<int>(outWeights.size()) != levelCount) return false;

    double sum = 0.0;
    for (double value : outWeights)
    {
        if (value < 0.0) return false;
        sum += value;
    }
    if (sum <= 0.0) return false;

    for (double& value : outWeights)
    {
        value /= sum;
    }
    return true;
}

bool TryGetAncestorTag(uint32_t tag,
                       int fromLevel,
                       int targetLevel,
                       const std::vector<PivotEstimatorLevelData>& levelData,
                       uint32_t& ancestorTag)
{
    ancestorTag = tag;
    if (fromLevel == targetLevel) return true;
    if (fromLevel < targetLevel || fromLevel < 0 || targetLevel < 0 ||
        fromLevel >= static_cast<int>(levelData.size()) ||
        targetLevel >= static_cast<int>(levelData.size())) {
        return false;
    }

    uint32_t currentTag = tag;
    for (int level = fromLevel; level > targetLevel; --level)
    {
        const auto parentIt = levelData[level].parentByTag.find(currentTag);
        if (parentIt == levelData[level].parentByTag.end()) {
            return false;
        }
        currentTag = parentIt->second;
    }

    ancestorTag = currentTag;
    return true;
}

} // namespace

constexpr int32_t kHeadNodeMetaVersion = 2;

struct HeadNodeMetaFileHeader {
    int32_t version;
    int32_t numSamples;
    int32_t numTagsPerSample;
    int32_t stride;
};

std::string HeadNodeMetaPath(const std::string& workDir)
{
    return workDir + "/HeadIndex/head_node_meta.bin";
}

std::shared_ptr<SPTAG::VectorIndex> GetMemoryIndexForInternal(const std::shared_ptr<SPTAG::VectorIndex>& internalIndex)
{
    auto* spannInternalIdx = dynamic_cast<SPTAG::SPANN::Index<float>*>(internalIndex.get());
    if (spannInternalIdx == nullptr) return nullptr;
    return spannInternalIdx->GetMemoryIndex();
}

bool SaveHeadNodeMetaFile(const std::string& workDir, const std::shared_ptr<SPTAG::VectorIndex>& headIndex)
{
    if (headIndex == nullptr || !headIndex->HasHeadNodeMeta()) return false;

    std::string metaPath = HeadNodeMetaPath(workDir);
    FILE* f = fopen(metaPath.c_str(), "wb");
    if (!f) return false;

    HeadNodeMetaFileHeader header{};
    header.version = kHeadNodeMetaVersion;
    header.numSamples = headIndex->GetHeadNodeMetaSampleCount();
    header.numTagsPerSample = 0;  // No longer used in V2, repurposed/reserved
    header.stride = static_cast<int32_t>(headIndex->GetHeadNodeMetaStride());

    const auto& blob = headIndex->GetHeadNodeMetaBlob();
    bool ok =
        fwrite(&header, sizeof(header), 1, f) == 1 &&
        fwrite(blob.data(), 1, blob.size(), f) == blob.size();
    fclose(f);
    return ok;
}

bool LoadHeadNodeMetaFile(const std::string& workDir, const std::shared_ptr<SPTAG::VectorIndex>& headIndex)
{
    if (headIndex == nullptr) return false;

    std::string metaPath = HeadNodeMetaPath(workDir);
    FILE* f = fopen(metaPath.c_str(), "rb");
    if (!f) return false;

    HeadNodeMetaFileHeader header{};
    bool ok = fread(&header, sizeof(header), 1, f) == 1;
    if (!ok || header.numSamples < 0 || header.stride <= 0 ||
        header.numSamples != headIndex->GetNumSamples()) {
        fclose(f);
        return false;
    }

    // Version check with detailed error message
    if (header.version != kHeadNodeMetaVersion) {
        fprintf(stderr, "[ERROR] head_node_meta.bin version mismatch: file has version %d, expected version %d. "
                        "Please rebuild the index to use the new hierarchical mask format.\n",
                header.version, kHeadNodeMetaVersion);
        fclose(f);
        return false;
    }

    headIndex->InitializeHeadNodeMeta(header.numSamples);
    if (static_cast<int32_t>(headIndex->GetHeadNodeMetaStride()) != header.stride) {
        fprintf(stderr, "[ERROR] head_node_meta.bin stride mismatch: file has stride %d, expected %zu. "
                        "Binary layout has changed.\n",
                header.stride, headIndex->GetHeadNodeMetaStride());
        headIndex->ClearHeadNodeMeta();
        fclose(f);
        return false;
    }

    auto& blob = headIndex->GetHeadNodeMetaBlob();
    ok = fread(blob.data(), 1, blob.size(), f) == blob.size();
    fclose(f);
    if (!ok) {
        headIndex->ClearHeadNodeMeta();
        return false;
    }
    return true;
}

bool LoadPostingSignaturesIntoHeadIndex(const std::string& workDir,
                                        const std::shared_ptr<SPTAG::VectorIndex>& internalIndex)
{
    if (internalIndex == nullptr) return false;

    auto headIndex = GetMemoryIndexForInternal(internalIndex);
    auto* spannInternalIdx = dynamic_cast<SPTAG::SPANN::Index<float>*>(internalIndex.get());
    if (headIndex == nullptr || spannInternalIdx == nullptr) return false;

    SPTAG::Cache::TenantBitmaskPS sigs;
    std::string sigPath = workDir + "/signatures_bitmask.bin";
    if (!sigs.Load(sigPath)) return false;

    const SizeType numHeadSamples = headIndex->GetNumSamples();
    if (!headIndex->HasHeadNodeMeta()) {
        headIndex->InitializeHeadNodeMeta(numHeadSamples);
    }

    for (SizeType hid = 0; hid < numHeadSamples; ++hid) {
        SizeType globalVID = spannInternalIdx->GetGlobalVID(hid);
        headIndex->SetHeadNodeGlobalVID(hid, globalVID);
        if (hid < sigs.num_postings) {
            headIndex->SetHeadNodePS(hid, sigs.ps[hid]);
        }
    }
    return true;
}

bool EnsureHeadNodeMetaLoaded(const std::string& workDir, const std::shared_ptr<SPTAG::VectorIndex>& internalIndex)
{
    auto headIndex = GetMemoryIndexForInternal(internalIndex);
    if (headIndex == nullptr) return false;
    if (headIndex->HasHeadNodeMeta()) return true;
    if (LoadHeadNodeMetaFile(workDir, headIndex)) return true;
    return LoadPostingSignaturesIntoHeadIndex(workDir, internalIndex);
}

constexpr int32_t kHeadNodeRoutingIndexVersion = 1;

struct HeadNodeRoutingIndexFileHeader {
    int32_t version;
    int32_t pivotLevel;
    int32_t nodeCount;
    int32_t numHeadSamples;
    int32_t numTagMappings;
};

struct PivotEstimatorComputation {
    std::vector<PivotEstimatorLevelData> levelData;
    std::vector<double> levelWeights;
    std::vector<PivotEstimatorCandidate> candidates;
};

constexpr double kGreedyLeafMinLocalSelectivity = 0.05;

struct LeafGreedyPlanEntry {
    uint32_t leafTag = 0;
    int leafCount = 0;
    std::vector<uint32_t> ancestorPath;
};

std::string HeadNodeRoutingIndexPath(const std::string& workDir)
{
    return workDir + "/HeadIndex/tag_node_index.bin";
}

void BuildNodeByPivotTag(const PivotEstimatorCandidate& candidate,
                         std::unordered_map<uint32_t, int>& nodeByPivotTag)
{
    nodeByPivotTag.clear();
    for (int nodeId = 0; nodeId < candidate.nodeCount; ++nodeId)
    {
        if (nodeId < 0 || nodeId >= static_cast<int>(candidate.nodePivotTags.size())) continue;
        for (uint32_t pivotTag : candidate.nodePivotTags[nodeId])
        {
            nodeByPivotTag[pivotTag] = nodeId;
        }
    }
}

void CollectNodesForTag(int tagLevel,
                        uint32_t tag,
                        const PivotEstimatorCandidate& candidate,
                        const std::vector<PivotEstimatorLevelData>& levelData,
                        const std::unordered_map<uint32_t, int>& nodeByPivotTag,
                        std::vector<int>& outNodes)
{
    std::unordered_set<int> touchedNodes;

    if (tagLevel < candidate.pivotLevel)
    {
        for (const auto& nodePivotTags : candidate.nodePivotTags)
        {
            for (uint32_t pivotTag : nodePivotTags)
            {
                uint32_t ancestor = 0;
                if (TryGetAncestorTag(pivotTag, candidate.pivotLevel, tagLevel, levelData, ancestor) &&
                    ancestor == tag)
                {
                    auto nodeIt = nodeByPivotTag.find(pivotTag);
                    if (nodeIt != nodeByPivotTag.end()) {
                        touchedNodes.insert(nodeIt->second);
                    }
                }
            }
        }
    }
    else if (tagLevel == candidate.pivotLevel)
    {
        auto nodeIt = nodeByPivotTag.find(tag);
        if (nodeIt != nodeByPivotTag.end()) {
            touchedNodes.insert(nodeIt->second);
        }
    }
    else
    {
        uint32_t ancestorPivot = 0;
        if (TryGetAncestorTag(tag, tagLevel, candidate.pivotLevel, levelData, ancestorPivot)) {
            auto nodeIt = nodeByPivotTag.find(ancestorPivot);
            if (nodeIt != nodeByPivotTag.end()) {
                touchedNodes.insert(nodeIt->second);
            }
        }
    }

    outNodes.assign(touchedNodes.begin(), touchedNodes.end());
    std::sort(outNodes.begin(), outNodes.end());
}

bool BuildPivotEstimatorComputation(const uint32_t* tags,
                                    int numVectors,
                                    int numTagsPerVec,
                                    int maxNodes,
                                    double recallTarget,
                                    double lambdaRecall,
                                    double estimatedRecall,
                                    const std::string& weightsCsv,
                                    PivotEstimatorComputation& out)
{
    out = PivotEstimatorComputation();
    if (tags == nullptr || numVectors <= 0 || numTagsPerVec <= 0) {
        return false;
    }

    (void)maxNodes;

    out.levelData.resize(numTagsPerVec);
    for (int level = 0; level < numTagsPerVec; ++level)
    {
        std::unordered_map<uint32_t, int> levelCounts;
        levelCounts.reserve(static_cast<size_t>(numVectors) / 4 + 8);
        std::unordered_map<uint32_t, uint32_t> parentByTag;
        if (level > 0) {
            parentByTag.reserve(static_cast<size_t>(numVectors) / 4 + 8);
        }

        for (int vectorId = 0; vectorId < numVectors; ++vectorId)
        {
            const size_t vectorOffset = static_cast<size_t>(vectorId) * static_cast<size_t>(numTagsPerVec);
            uint32_t tag = tags[vectorOffset + static_cast<size_t>(level)];
            levelCounts[tag] += 1;

            if (level > 0)
            {
                uint32_t parentTag = tags[vectorOffset + static_cast<size_t>(level - 1)];
                auto parentIt = parentByTag.find(tag);
                if (parentIt == parentByTag.end()) {
                    parentByTag.emplace(tag, parentTag);
                } else if (parentIt->second != parentTag) {
                    return false;
                }
            }
        }

        std::vector<std::pair<uint32_t, int>> pairs(levelCounts.begin(), levelCounts.end());
        std::sort(pairs.begin(), pairs.end(), [](const auto& left, const auto& right) { return left.first < right.first; });

        out.levelData[level].uniqueTags.reserve(pairs.size());
        out.levelData[level].counts.reserve(pairs.size());
        for (size_t i = 0; i < pairs.size(); ++i)
        {
            out.levelData[level].uniqueTags.push_back(pairs[i].first);
            out.levelData[level].counts.push_back(pairs[i].second);
        }
        if (level > 0) {
            out.levelData[level].parentByTag = std::move(parentByTag);
        }
    }

    if (!ParseLevelWeights(weightsCsv, numTagsPerVec, out.levelWeights)) {
        out.levelWeights.assign(numTagsPerVec, 1.0 / static_cast<double>(numTagsPerVec));
    }

    const double totalVectors = static_cast<double>(numVectors);

    const int leafLevel = numTagsPerVec - 1;
    const auto& leafTags = out.levelData[leafLevel].uniqueTags;
    const auto& leafCounts = out.levelData[leafLevel].counts;
    if (leafTags.empty() || leafTags.size() != leafCounts.size()) {
        return false;
    }

    std::vector<LeafGreedyPlanEntry> leafEntries;
    leafEntries.reserve(leafTags.size());
    for (size_t idx = 0; idx < leafTags.size(); ++idx)
    {
        if (leafCounts[idx] <= 0) {
            continue;
        }

        LeafGreedyPlanEntry entry;
        entry.leafTag = leafTags[idx];
        entry.leafCount = leafCounts[idx];
        entry.ancestorPath.resize(static_cast<size_t>(numTagsPerVec));

        bool validPath = true;
        for (int level = 0; level <= leafLevel; ++level)
        {
            uint32_t ancestorTag = 0;
            if (!TryGetAncestorTag(entry.leafTag, leafLevel, level, out.levelData, ancestorTag)) {
                validPath = false;
                break;
            }
            entry.ancestorPath[static_cast<size_t>(level)] = ancestorTag;
        }

        if (!validPath) {
            return false;
        }

        leafEntries.emplace_back(std::move(entry));
    }

    if (leafEntries.empty()) {
        return false;
    }

    std::sort(leafEntries.begin(), leafEntries.end(), [](const LeafGreedyPlanEntry& left, const LeafGreedyPlanEntry& right) {
        if (std::lexicographical_compare(left.ancestorPath.begin(), left.ancestorPath.end(),
                                         right.ancestorPath.begin(), right.ancestorPath.end())) {
            return true;
        }
        if (std::lexicographical_compare(right.ancestorPath.begin(), right.ancestorPath.end(),
                                         left.ancestorPath.begin(), left.ancestorPath.end())) {
            return false;
        }
        return left.leafTag < right.leafTag;
    });

    PivotEstimatorCandidate candidate;
    candidate.pivotLevel = leafLevel;

    std::vector<uint32_t> currentNodeLeafTags;
    int currentNodeSize = 0;
    int currentMinLeafCount = std::numeric_limits<int>::max();
    auto flushCurrentNode = [&]() {
        if (currentNodeLeafTags.empty()) {
            return;
        }
        std::sort(currentNodeLeafTags.begin(), currentNodeLeafTags.end());
        candidate.nodePivotTags.push_back(currentNodeLeafTags);
        currentNodeLeafTags.clear();
        currentNodeSize = 0;
        currentMinLeafCount = std::numeric_limits<int>::max();
    };

    for (const auto& leafEntry : leafEntries)
    {
        int nextNodeSize = currentNodeSize + leafEntry.leafCount;
        int nextMinLeafCount = std::min(currentMinLeafCount, leafEntry.leafCount);
        bool canMerge = currentNodeLeafTags.empty() ||
            (static_cast<double>(nextMinLeafCount) / static_cast<double>(nextNodeSize) >= kGreedyLeafMinLocalSelectivity);

        if (!canMerge) {
            flushCurrentNode();
            nextNodeSize = leafEntry.leafCount;
            nextMinLeafCount = leafEntry.leafCount;
        }

        currentNodeLeafTags.push_back(leafEntry.leafTag);
        currentNodeSize = nextNodeSize;
        currentMinLeafCount = nextMinLeafCount;
    }
    flushCurrentNode();

    candidate.nodeCount = static_cast<int>(candidate.nodePivotTags.size());
    if (candidate.nodeCount <= 0) {
        return false;
    }

    std::unordered_map<uint32_t, int> nodeByPivotTag;
    BuildNodeByPivotTag(candidate, nodeByPivotTag);
    candidate.nodeSizes.assign(candidate.nodeCount, 0);
    for (int vectorId = 0; vectorId < numVectors; ++vectorId)
    {
        uint32_t pivotTag = tags[static_cast<size_t>(vectorId) * static_cast<size_t>(numTagsPerVec) + static_cast<size_t>(candidate.pivotLevel)];
        auto nodeIt = nodeByPivotTag.find(pivotTag);
        if (nodeIt != nodeByPivotTag.end()) {
            candidate.nodeSizes[nodeIt->second] += 1;
        }
    }

    double latencyCost = 0.0;
    for (int queryLevel = 0; queryLevel < numTagsPerVec; ++queryLevel)
    {
        double levelCost = 0.0;
        const auto& qTags = out.levelData[queryLevel].uniqueTags;
        const auto& qCounts = out.levelData[queryLevel].counts;

        for (size_t qIdx = 0; qIdx < qTags.size(); ++qIdx)
        {
            std::vector<int> touchedNodes;
            CollectNodesForTag(queryLevel, qTags[qIdx], candidate, out.levelData, nodeByPivotTag, touchedNodes);
            if (touchedNodes.empty()) continue;

            double touchedSize = 0.0;
            for (int nodeId : touchedNodes)
            {
                if (nodeId >= 0 && nodeId < static_cast<int>(candidate.nodeSizes.size())) {
                    touchedSize += static_cast<double>(candidate.nodeSizes[nodeId]);
                }
            }
            if (touchedSize <= 0.0) continue;

            double selectivity = std::max(1e-9, static_cast<double>(qCounts[qIdx]) / totalVectors);
            double baseLatency = std::log2(touchedSize + 1.0);
            double tagLatency = baseLatency * (1.0 / selectivity);
            double probability = static_cast<double>(qCounts[qIdx]) / totalVectors;
            levelCost += probability * tagLatency;
        }

        latencyCost += out.levelWeights[queryLevel] * levelCost;
    }

    candidate.latencyCost = latencyCost;
    candidate.recallPenalty = lambdaRecall * std::max(0.0, recallTarget - estimatedRecall);
    candidate.totalCost = candidate.latencyCost + candidate.recallPenalty;
    out.candidates.push_back(std::move(candidate));

    return !out.candidates.empty();
}

const PivotEstimatorCandidate* FindBestPivotEstimatorCandidate(const std::vector<PivotEstimatorCandidate>& candidates)
{
    if (candidates.empty()) return nullptr;

    const PivotEstimatorCandidate* best = &candidates.front();
    for (const auto& candidate : candidates)
    {
        if (candidate.totalCost < best->totalCost) {
            best = &candidate;
        }
    }
    return best;
}

void BuildTagToNodeIndexForCandidate(const PivotEstimatorCandidate& candidate,
                                     const std::vector<PivotEstimatorLevelData>& levelData,
                                     std::unordered_map<uint32_t, std::vector<int>>& tagToNodes)
{
    tagToNodes.clear();

    std::unordered_map<uint32_t, int> nodeByPivotTag;
    BuildNodeByPivotTag(candidate, nodeByPivotTag);

    for (int level = 0; level < static_cast<int>(levelData.size()); ++level)
    {
        for (uint32_t tag : levelData[level].uniqueTags)
        {
            std::vector<int> nodes;
            CollectNodesForTag(level, tag, candidate, levelData, nodeByPivotTag, nodes);
            if (!nodes.empty()) {
                tagToNodes[tag] = std::move(nodes);
            }
        }
    }
}

void BuildNodeVectorAssignmentsForTagToNodes(const uint32_t* tags,
                                             int numVectors,
                                             int numTagsPerVec,
                                             const std::unordered_map<uint32_t, std::vector<int>>& tagToNodes,
                                             int nodeCount,
                                             std::vector<std::vector<int>>& nodeVectors)
{
    nodeVectors.clear();
    if (tags == nullptr || numVectors <= 0 || numTagsPerVec <= 0 || nodeCount <= 0) {
        return;
    }

    nodeVectors.assign(nodeCount, std::vector<int>());
    for (int vectorId = 0; vectorId < numVectors; ++vectorId)
    {
        std::unordered_set<int> touchedNodes;
        const size_t vectorOffset = static_cast<size_t>(vectorId) * static_cast<size_t>(numTagsPerVec);
        for (int tagIdx = 0; tagIdx < numTagsPerVec; ++tagIdx)
        {
            auto tagIt = tagToNodes.find(tags[vectorOffset + static_cast<size_t>(tagIdx)]);
            if (tagIt == tagToNodes.end()) {
                continue;
            }

            touchedNodes.insert(tagIt->second.begin(), tagIt->second.end());
        }

        if (touchedNodes.empty()) {
            continue;
        }

        for (int nodeId : touchedNodes)
        {
            if (nodeId >= 0 && nodeId < nodeCount) {
                nodeVectors[static_cast<size_t>(nodeId)].push_back(vectorId);
            }
        }
    }
}

void BuildPrimaryNodeVectorAssignmentsForCandidate(const PivotEstimatorCandidate& candidate,
                                                   const uint32_t* tags,
                                                   int numVectors,
                                                   int numTagsPerVec,
                                                   std::vector<std::vector<int>>& nodeVectors)
{
    nodeVectors.clear();
    if (tags == nullptr || numVectors <= 0 || numTagsPerVec <= 0 || candidate.nodeCount <= 0) {
        return;
    }

    std::unordered_map<uint32_t, int> nodeByPivotTag;
    BuildNodeByPivotTag(candidate, nodeByPivotTag);

    nodeVectors.assign(candidate.nodeCount, std::vector<int>());
    for (int vectorId = 0; vectorId < numVectors; ++vectorId)
    {
        uint32_t pivotTag = tags[static_cast<size_t>(vectorId) * static_cast<size_t>(numTagsPerVec) + static_cast<size_t>(candidate.pivotLevel)];
        auto nodeIt = nodeByPivotTag.find(pivotTag);
        if (nodeIt == nodeByPivotTag.end()) {
            continue;
        }

        int nodeId = nodeIt->second;
        if (nodeId >= 0 && nodeId < candidate.nodeCount) {
            nodeVectors[static_cast<size_t>(nodeId)].push_back(vectorId);
        }
    }
}

void BuildHeadNodeToNodeIndexForCandidate(const PivotEstimatorCandidate& candidate,
                                          const uint32_t* tags,
                                          int numVectors,
                                          int numTagsPerVec,
                                          const std::shared_ptr<SPTAG::VectorIndex>& memoryIndex,
                                          SPTAG::SPANN::Index<float>* spannInternalIdx,
                                          std::vector<int>& headNodeToNode)
{
    headNodeToNode.clear();
    if (tags == nullptr || numVectors <= 0 || numTagsPerVec <= 0 || memoryIndex == nullptr || spannInternalIdx == nullptr) {
        return;
    }

    std::unordered_map<uint32_t, int> nodeByPivotTag;
    BuildNodeByPivotTag(candidate, nodeByPivotTag);

    const SizeType numHeadSamples = memoryIndex->GetNumSamples();
    headNodeToNode.assign(numHeadSamples, -1);
    for (SizeType hid = 0; hid < numHeadSamples; ++hid)
    {
        SizeType globalVID = spannInternalIdx->GetGlobalVID(hid);
        if (globalVID == SPTAG::MaxSize || globalVID >= static_cast<SizeType>(numVectors)) {
            continue;
        }

        uint32_t pivotTag = tags[static_cast<size_t>(globalVID) * static_cast<size_t>(numTagsPerVec) + static_cast<size_t>(candidate.pivotLevel)];
        auto nodeIt = nodeByPivotTag.find(pivotTag);
        if (nodeIt != nodeByPivotTag.end()) {
            headNodeToNode[hid] = nodeIt->second;
        }
    }
}

bool TryCollectRoutingNodesForQuery(const std::unordered_map<uint32_t, std::vector<int>>& tagToNodes,
                                    const uint32_t* queryTags,
                                    int numQueryTags,
                                    std::vector<int>& outNodes)
{
    outNodes.clear();
    if (queryTags == nullptr || numQueryTags <= 0) {
        return false;
    }

    std::unordered_set<int> unionNodes;
    for (int idx = 0; idx < numQueryTags; ++idx)
    {
        auto tagIt = tagToNodes.find(queryTags[idx]);
        if (tagIt == tagToNodes.end() || tagIt->second.empty()) {
            continue;
        }

        unionNodes.insert(tagIt->second.begin(), tagIt->second.end());
    }

    outNodes.assign(unionNodes.begin(), unionNodes.end());
    std::sort(outNodes.begin(), outNodes.end());
    return !outNodes.empty();
}

bool SaveHeadNodeRoutingIndexFile(const std::string& workDir,
                                  int pivotLevel,
                                  const std::vector<std::vector<uint32_t>>& nodePivotTags,
                                  const std::unordered_map<uint32_t, std::vector<int>>& tagToNodes,
                                  const std::vector<int>& headNodeToNode)
{
    std::string path = HeadNodeRoutingIndexPath(workDir);
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;

    HeadNodeRoutingIndexFileHeader header{};
    header.version = kHeadNodeRoutingIndexVersion;
    header.pivotLevel = pivotLevel;
    header.nodeCount = static_cast<int32_t>(nodePivotTags.size());
    header.numHeadSamples = static_cast<int32_t>(headNodeToNode.size());
    header.numTagMappings = static_cast<int32_t>(tagToNodes.size());

    bool ok = fwrite(&header, sizeof(header), 1, f) == 1;
    for (const auto& tagsForNode : nodePivotTags)
    {
        int32_t tagCount = static_cast<int32_t>(tagsForNode.size());
        ok = ok && fwrite(&tagCount, sizeof(tagCount), 1, f) == 1;
        if (tagCount > 0) {
            ok = ok && fwrite(tagsForNode.data(), sizeof(uint32_t), tagCount, f) == static_cast<size_t>(tagCount);
        }
    }

    std::vector<std::pair<uint32_t, std::vector<int>>> mappings(tagToNodes.begin(), tagToNodes.end());
    std::sort(mappings.begin(), mappings.end(), [](const auto& left, const auto& right) { return left.first < right.first; });
    for (auto& [tag, nodes] : mappings)
    {
        std::sort(nodes.begin(), nodes.end());
        nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());

        int32_t nodeCount = static_cast<int32_t>(nodes.size());
        ok = ok && fwrite(&tag, sizeof(tag), 1, f) == 1;
        ok = ok && fwrite(&nodeCount, sizeof(nodeCount), 1, f) == 1;
        if (nodeCount > 0) {
            ok = ok && fwrite(nodes.data(), sizeof(int32_t), nodeCount, f) == static_cast<size_t>(nodeCount);
        }
    }

    if (!headNodeToNode.empty()) {
        ok = ok && fwrite(headNodeToNode.data(), sizeof(int32_t), headNodeToNode.size(), f) == headNodeToNode.size();
    }
    fclose(f);
    return ok;
}

bool LoadHeadNodeRoutingIndexFile(const std::string& workDir,
                                  int& pivotLevel,
                                  int& nodeCount,
                                  std::vector<std::vector<uint32_t>>& nodePivotTags,
                                  std::unordered_map<uint32_t, std::vector<int>>& tagToNodes,
                                  std::vector<int>& headNodeToNode)
{
    pivotLevel = -1;
    nodeCount = 0;
    nodePivotTags.clear();
    tagToNodes.clear();
    headNodeToNode.clear();

    std::string path = HeadNodeRoutingIndexPath(workDir);
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;

    HeadNodeRoutingIndexFileHeader header{};
    bool ok = fread(&header, sizeof(header), 1, f) == 1;
    if (!ok || header.version != kHeadNodeRoutingIndexVersion || header.nodeCount < 0 ||
        header.numHeadSamples < 0 || header.numTagMappings < 0) {
        fclose(f);
        return false;
    }

    pivotLevel = header.pivotLevel;
    nodeCount = header.nodeCount;
    nodePivotTags.assign(header.nodeCount, std::vector<uint32_t>());
    for (int nodeId = 0; nodeId < header.nodeCount; ++nodeId)
    {
        int32_t tagCount = 0;
        ok = fread(&tagCount, sizeof(tagCount), 1, f) == 1;
        if (!ok || tagCount < 0) {
            fclose(f);
            return false;
        }
        nodePivotTags[nodeId].resize(tagCount);
        if (tagCount > 0) {
            ok = fread(nodePivotTags[nodeId].data(), sizeof(uint32_t), tagCount, f) == static_cast<size_t>(tagCount);
            if (!ok) {
                fclose(f);
                return false;
            }
        }
    }

    for (int mappingId = 0; mappingId < header.numTagMappings; ++mappingId)
    {
        uint32_t tag = 0;
        int32_t mappingNodeCount = 0;
        ok = fread(&tag, sizeof(tag), 1, f) == 1;
        ok = ok && fread(&mappingNodeCount, sizeof(mappingNodeCount), 1, f) == 1;
        if (!ok || mappingNodeCount < 0) {
            fclose(f);
            return false;
        }

        std::vector<int> nodes(mappingNodeCount);
        if (mappingNodeCount > 0) {
            ok = fread(nodes.data(), sizeof(int32_t), mappingNodeCount, f) == static_cast<size_t>(mappingNodeCount);
            if (!ok) {
                fclose(f);
                return false;
            }
        }
        tagToNodes.emplace(tag, std::move(nodes));
    }

    headNodeToNode.resize(header.numHeadSamples);
    if (header.numHeadSamples > 0) {
        ok = fread(headNodeToNode.data(), sizeof(int32_t), header.numHeadSamples, f) == static_cast<size_t>(header.numHeadSamples);
    }
    fclose(f);
    return ok;
}

AnnIndex::AnnIndex(DimensionType p_dimension)
    : m_algoType(SPTAG::IndexAlgoType::BKT), m_inputValueType(SPTAG::VectorValueType::Float), m_dimension(p_dimension)
{
    m_inputVectorSize = SPTAG::GetValueTypeSize(m_inputValueType) * m_dimension;
}

AnnIndex::AnnIndex(const char *p_algoType, const char *p_valueType, DimensionType p_dimension)
    : m_algoType(SPTAG::IndexAlgoType::Undefined), m_inputValueType(SPTAG::VectorValueType::Undefined),
      m_dimension(p_dimension)
{
    SPTAG::Helper::Convert::ConvertStringTo<SPTAG::IndexAlgoType>(p_algoType, m_algoType);
    SPTAG::Helper::Convert::ConvertStringTo<SPTAG::VectorValueType>(p_valueType, m_inputValueType);
    m_inputVectorSize = SPTAG::GetValueTypeSize(m_inputValueType) * m_dimension;
}

AnnIndex::AnnIndex(const std::shared_ptr<SPTAG::VectorIndex> &p_index)
    : m_algoType(p_index->GetIndexAlgoType()), m_inputValueType(p_index->GetVectorValueType()),
      m_dimension(p_index->GetFeatureDim()), m_index(p_index)
{
    m_inputVectorSize = p_index->m_pQuantizer ? p_index->m_pQuantizer->GetNumSubvectors()
                                              : SPTAG::GetValueTypeSize(m_inputValueType) * m_dimension;
}

AnnIndex::~AnnIndex()
{
}

bool AnnIndex::BuildSPANN(bool p_normalized)
{
    if (nullptr == m_index)
    {
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
    }
    if (nullptr == m_index)
        return false;

    return (SPTAG::ErrorCode::Success == m_index->BuildIndex(p_normalized));
}

bool AnnIndex::BuildSPANNWithMetaData(ByteArray p_meta, SizeType p_num, bool p_withMetaIndex, bool p_normalized)
{
    if (nullptr == m_index)
    {
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
    }
    if (nullptr == m_index)
        return false;

    std::uint64_t *offsets = new std::uint64_t[p_num + 1]{0};
    if (!SPTAG::MetadataSet::GetMetadataOffsets(p_meta.Data(), p_meta.Length(), offsets, p_num + 1, '\n'))
        return false;

    m_index->SetMetadata((new SPTAG::MemMetadataSet(
        p_meta, ByteArray((std::uint8_t *)offsets, (p_num + 1) * sizeof(std::uint64_t), true), (SPTAG::SizeType)p_num,
        m_index->m_iDataBlockSize, m_index->m_iDataCapacity, m_index->m_iMetaRecordSize)));
    if (p_withMetaIndex)
        m_index->BuildMetaMapping(false);

    return (SPTAG::ErrorCode::Success == m_index->BuildIndex(p_normalized));
}

// Build SPANN index with both vector data and metadata (for attribute filtering support)
bool AnnIndex::BuildSPANNWithDataAndMeta(ByteArray p_data, ByteArray p_meta, SizeType p_num,
                                          bool p_withMetaIndex, bool p_normalized)
{
    if (nullptr == m_index)
    {
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
    }
    if (nullptr == m_index || p_num == 0 || m_dimension == 0 || p_data.Length() != p_num * m_inputVectorSize)
        return false;

    // Set metadata first (before build, so it's available during search)
    std::uint64_t *offsets = new std::uint64_t[p_num + 1]{0};
    if (!SPTAG::MetadataSet::GetMetadataOffsets(p_meta.Data(), p_meta.Length(), offsets, p_num + 1, '\n'))
        return false;

    m_index->SetMetadata((new SPTAG::MemMetadataSet(
        p_meta, ByteArray((std::uint8_t *)offsets, (p_num + 1) * sizeof(std::uint64_t), true), (SPTAG::SizeType)p_num,
        m_index->m_iDataBlockSize, m_index->m_iDataCapacity, m_index->m_iMetaRecordSize)));
    if (p_withMetaIndex)
        m_index->BuildMetaMapping(false);

    // Build with in-memory vector data
    return (SPTAG::ErrorCode::Success == m_index->BuildIndex(p_data.Data(), (SPTAG::SizeType)p_num,
                                                              (SPTAG::DimensionType)m_dimension, p_normalized));
}

bool AnnIndex::Build(ByteArray p_data, SizeType p_num, bool p_normalized)
{
    if (nullptr == m_index)
    {
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
    }
    if (nullptr == m_index || p_num == 0 || m_dimension == 0 || p_data.Length() != p_num * m_inputVectorSize)
    {
        return false;
    }
    return (SPTAG::ErrorCode::Success == m_index->BuildIndex(p_data.Data(), (SPTAG::SizeType)p_num,
                                                             (SPTAG::DimensionType)m_dimension, p_normalized));
}

bool AnnIndex::BuildWithMetaData(ByteArray p_data, ByteArray p_meta, SizeType p_num, bool p_withMetaIndex,
                                 bool p_normalized)
{
    if (nullptr == m_index)
    {
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
    }
    if (nullptr == m_index || p_num == 0 || m_dimension == 0 || p_data.Length() != p_num * m_inputVectorSize)
    {
        return false;
    }

    auto vectorType = m_index->m_pQuantizer ? SPTAG::VectorValueType::UInt8 : m_inputValueType;
    auto vectorSize = m_index->m_pQuantizer ? m_index->m_pQuantizer->GetNumSubvectors() : m_dimension;
    std::shared_ptr<SPTAG::VectorSet> vectors(new SPTAG::BasicVectorSet(
        p_data, vectorType, static_cast<SPTAG::DimensionType>(vectorSize), static_cast<SPTAG::SizeType>(p_num)));

    std::uint64_t *offsets = new std::uint64_t[p_num + 1]{0};
    if (!SPTAG::MetadataSet::GetMetadataOffsets(p_meta.Data(), p_meta.Length(), offsets, p_num + 1, '\n'))
        return false;
    std::shared_ptr<SPTAG::MetadataSet> meta(new SPTAG::MemMetadataSet(
        p_meta, ByteArray((std::uint8_t *)offsets, (p_num + 1) * sizeof(std::uint64_t), true), (SPTAG::SizeType)p_num,
        m_index->m_iDataBlockSize, m_index->m_iDataCapacity, m_index->m_iMetaRecordSize));
    return (SPTAG::ErrorCode::Success == m_index->BuildIndex(vectors, meta, p_withMetaIndex, p_normalized));
}

void AnnIndex::SetBuildParam(const char *p_name, const char *p_value, const char *p_section)
{
    if (nullptr == m_index)
    {
        if (SPTAG::IndexAlgoType::Undefined == m_algoType || SPTAG::VectorValueType::Undefined == m_inputValueType)
        {
            return;
        }
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
    }
    m_index->SetParameter(p_name, p_value, p_section);
}

void AnnIndex::SetSearchParam(const char *p_name, const char *p_value, const char *p_section)
{
    if (nullptr != m_index)
        m_index->SetParameter(p_name, p_value, p_section);
}

std::shared_ptr<ResultIterator> AnnIndex::GetIterator(ByteArray p_target)
{
    if (nullptr != m_index)
        return m_index->GetIterator(p_target.Data());
    return nullptr;
}

bool AnnIndex::LoadQuantizer(const char *p_quantizerFile)
{
    if (nullptr == m_index)
    {
        if (SPTAG::IndexAlgoType::Undefined == m_algoType || SPTAG::VectorValueType::Undefined == m_inputValueType)
        {
            return false;
        }
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
    }

    auto ret = (m_index->LoadQuantizer(p_quantizerFile) == SPTAG::ErrorCode::Success);
    if (ret)
    {
        m_inputVectorSize = m_index->m_pQuantizer->QuantizeSize();
    }
    return ret;
}

void AnnIndex::SetQuantizerADC(bool p_adc)
{
    if (nullptr != m_index)
        return m_index->SetQuantizerADC(p_adc);
}

ByteArray AnnIndex::QuantizeVector(ByteArray p_data, int p_num)
{
    if (nullptr != m_index && m_index->GetQuantizer() != nullptr)
    {
        size_t outsize = m_index->GetQuantizer()->GetNumSubvectors() * (size_t)p_num;
        std::uint8_t *outdata = new std::uint8_t[outsize];
        if (SPTAG::ErrorCode::Success !=
            m_index->QuantizeVector(p_data.Data(), p_num, ByteArray(outdata, outsize, false)))
            return ByteArray::c_empty;
        return ByteArray(outdata, outsize, false);
    }
    return ByteArray::c_empty;
}

ByteArray AnnIndex::ReconstructVector(ByteArray p_data, int p_num)
{
    if (nullptr != m_index && m_index->GetQuantizer() != nullptr)
    {
        size_t outsize = m_index->GetQuantizer()->ReconstructSize() * (size_t)p_num;
        std::uint8_t *outdata = new std::uint8_t[outsize];
        if (SPTAG::ErrorCode::Success !=
            m_index->ReconstructVector(p_data.Data(), p_num, ByteArray(outdata, outsize, false)))
            return ByteArray::c_empty;
        return ByteArray(outdata, outsize, false);
    }
    return ByteArray::c_empty;
}

std::shared_ptr<QueryResult> AnnIndex::Search(ByteArray p_data, int p_resultNum)
{
    std::shared_ptr<QueryResult> results = std::make_shared<QueryResult>(p_data.Data(), p_resultNum, false);

    if (nullptr != m_index)
    {
        m_index->SearchIndex(*results);
    }
    return std::move(results);
}

std::shared_ptr<QueryResult> AnnIndex::SearchWithMetaData(ByteArray p_data, int p_resultNum)
{
    std::shared_ptr<QueryResult> results = std::make_shared<QueryResult>(p_data.Data(), p_resultNum, true);

    if (nullptr != m_index)
    {
        m_index->SearchIndex(*results);
    }
    return std::move(results);
}

std::shared_ptr<QueryResult> AnnIndex::BatchSearch(ByteArray p_data, int p_vectorNum, int p_resultNum,
                                                   bool p_withMetaData)
{
    std::shared_ptr<QueryResult> results =
        std::make_shared<QueryResult>(p_data.Data(), p_vectorNum * p_resultNum, p_withMetaData);
    if (nullptr != m_index)
    {
        m_index->SearchIndex(p_data.Data(), p_vectorNum, p_resultNum, p_withMetaData, results->GetResults());
    }
    return std::move(results);
}

std::shared_ptr<QueryResult> AnnIndex::SearchWithTenantFilter(ByteArray p_data, int p_resultNum, const char* p_tenantId)
{
    std::shared_ptr<QueryResult> results = std::make_shared<QueryResult>(p_data.Data(), p_resultNum, true);
    
    if (nullptr != m_index && nullptr != p_tenantId)
    {
        // Create filter function that checks if metadata exactly matches tenantId
        std::string tenantId(p_tenantId);
        auto filterFunc = [tenantId](const SPTAG::ByteArray& metadata) -> bool {
            if (metadata.Length() == 0) return false;
            std::string meta(reinterpret_cast<const char*>(metadata.Data()), metadata.Length());
            // Trim trailing whitespace/newline
            while (!meta.empty() && (meta.back() == '\n' || meta.back() == '\r' || meta.back() == ' '))
                meta.pop_back();
            return meta == tenantId;
        };
        
        m_index->SearchIndexWithFilter(*results, filterFunc, 0, false);
    }
    return std::move(results);
}

std::shared_ptr<QueryResult> AnnIndex::BatchSearchWithTenantFilter(ByteArray p_data, int p_vectorNum, 
                                                                    int p_resultNum, const char* p_tenantId)
{
    std::shared_ptr<QueryResult> results = 
        std::make_shared<QueryResult>(p_data.Data(), p_vectorNum * p_resultNum, true);
    
    if (nullptr != m_index && nullptr != p_tenantId && nullptr != p_data.Data())
    {
        // For batch search with filter, we need to process each vector separately
        // since the batch SearchIndex doesn't support filtering
        std::string tenantId(p_tenantId);
        auto filterFunc = [tenantId](const SPTAG::ByteArray& metadata) -> bool {
            if (metadata.Length() == 0) return false;
            std::string meta(reinterpret_cast<const char*>(metadata.Data()), metadata.Length());
            return meta.find(tenantId) != std::string::npos;
        };
        
        SPTAG::BasicResult* results_array = results->GetResults();
        const char* data = reinterpret_cast<const char*>(p_data.Data());
        size_t vectorSize = p_data.Length() / p_vectorNum;
        
        for (int i = 0; i < p_vectorNum; i++)
        {
            SPTAG::QueryResult singleQuery(data + i * vectorSize, p_resultNum, true);
            m_index->SearchIndexWithFilter(singleQuery, filterFunc, 0, false);
            
            // Copy results
            for (int j = 0; j < p_resultNum && j < singleQuery.GetResultNum(); j++)
            {
                auto* one = singleQuery.GetResult(j);
                if (one != nullptr)
                {
                    results_array[i * p_resultNum + j] = *one;
                }
            }
        }
    }
    return std::move(results);
}

bool AnnIndex::ReadyToServe() const
{
    return m_index != nullptr;
}

void AnnIndex::SetVectorTags(const uint32_t* tags, int numVecs, int numTagsPerVec)
{
    if (!m_index) return;
    // Cast to SPANN Index<float> to access SetVectorTags
    auto* spannIdx = dynamic_cast<SPTAG::SPANN::Index<float>*>(m_index.get());
    if (spannIdx) {
        spannIdx->SetVectorTags(tags, numVecs, numTagsPerVec);
    }
}

void AnnIndex::SetNodeVectorAssignments(const std::vector<std::vector<int>>& nodeVectorAssignments)
{
    if (!m_index) return;
    auto* spannIdx = dynamic_cast<SPTAG::SPANN::Index<float>*>(m_index.get());
    if (!spannIdx) return;

    std::vector<std::vector<SPTAG::SizeType>> convertedAssignments;
    convertedAssignments.reserve(nodeVectorAssignments.size());
    for (const auto& nodeVectors : nodeVectorAssignments)
    {
        std::vector<SPTAG::SizeType> convertedNode;
        convertedNode.reserve(nodeVectors.size());
        for (int vectorId : nodeVectors)
        {
            if (vectorId >= 0) {
                convertedNode.push_back(static_cast<SPTAG::SizeType>(vectorId));
            }
        }
        convertedAssignments.emplace_back(std::move(convertedNode));
    }

    spannIdx->SetNodeVectorAssignments(convertedAssignments);
}

void AnnIndex::SetPrimaryNodeVectorAssignments(const std::vector<std::vector<int>>& primaryNodeVectorAssignments)
{
    if (!m_index) return;
    auto* spannIdx = dynamic_cast<SPTAG::SPANN::Index<float>*>(m_index.get());
    if (!spannIdx) return;

    std::vector<std::vector<SPTAG::SizeType>> convertedAssignments;
    convertedAssignments.reserve(primaryNodeVectorAssignments.size());
    for (const auto& nodeVectors : primaryNodeVectorAssignments)
    {
        std::vector<SPTAG::SizeType> convertedNode;
        convertedNode.reserve(nodeVectors.size());
        for (int vectorId : nodeVectors)
        {
            if (vectorId >= 0) {
                convertedNode.push_back(static_cast<SPTAG::SizeType>(vectorId));
            }
        }
        convertedAssignments.emplace_back(std::move(convertedNode));
    }

    spannIdx->SetPrimaryNodeVectorAssignments(convertedAssignments);
}

bool AnnIndex::SetSharedDB(std::shared_ptr<SPTAG::Helper::KeyValueIO> p_db)
{
    if (m_index == nullptr)
    {
        if (m_algoType == SPTAG::IndexAlgoType::Undefined ||
            m_inputValueType == SPTAG::VectorValueType::Undefined)
            return false;
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
        if (m_index == nullptr) return false;
    }
    if (m_algoType != SPTAG::IndexAlgoType::SPANN) return false;
    if (m_inputValueType == SPTAG::VectorValueType::Float)
    {
        auto spann = std::dynamic_pointer_cast<SPTAG::SPANN::Index<float>>(m_index);
        if (spann == nullptr) return false;
        spann->SetSharedDB(std::move(p_db));
        return true;
    }
    return false;
}

void AnnIndex::UpdateIndex()
{
    m_index->UpdateIndex();
}

bool AnnIndex::Save(const char *p_savefile) const
{
    return SPTAG::ErrorCode::Success == m_index->SaveIndex(p_savefile);
}

bool AnnIndex::Add(ByteArray p_data, SizeType p_num, bool p_normalized)
{
    if (nullptr == m_index)
    {
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
    }
    if (nullptr == m_index || p_num == 0 || m_dimension == 0 || p_data.Length() != p_num * m_inputVectorSize)
    {
        return false;
    }

    return (SPTAG::ErrorCode::Success == m_index->AddIndex(p_data.Data(), (SPTAG::SizeType)p_num,
                                                           (SPTAG::DimensionType)m_dimension, nullptr, false,
                                                           p_normalized));
}

bool AnnIndex::AddWithMetaData(ByteArray p_data, ByteArray p_meta, SizeType p_num, bool p_withMetaIndex,
                               bool p_normalized)
{
    if (nullptr == m_index)
    {
        m_index = SPTAG::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
    }
    if (nullptr == m_index || p_num == 0 || m_dimension == 0 || p_data.Length() != p_num * m_inputVectorSize)
    {
        return false;
    }

    std::shared_ptr<SPTAG::VectorSet> vectors(new SPTAG::BasicVectorSet(
        p_data, m_inputValueType, static_cast<SPTAG::DimensionType>(m_dimension), static_cast<SPTAG::SizeType>(p_num)));

    std::uint64_t *offsets = new std::uint64_t[p_num + 1]{0};
    if (!SPTAG::MetadataSet::GetMetadataOffsets(p_meta.Data(), p_meta.Length(), offsets, p_num + 1, '\n'))
        return false;
    std::shared_ptr<SPTAG::MetadataSet> meta(new SPTAG::MemMetadataSet(
        p_meta, ByteArray((std::uint8_t *)offsets, (p_num + 1) * sizeof(std::uint64_t), true), (SPTAG::SizeType)p_num));
    return (SPTAG::ErrorCode::Success == m_index->AddIndex(vectors, meta, p_withMetaIndex, p_normalized));
}

bool AnnIndex::Delete(ByteArray p_data, SizeType p_num)
{
    if (nullptr == m_index || p_num == 0 || m_dimension == 0 || p_data.Length() != p_num * m_inputVectorSize)
    {
        return false;
    }

    return (SPTAG::ErrorCode::Success == m_index->DeleteIndex(p_data.Data(), (SPTAG::SizeType)p_num));
}

bool AnnIndex::DeleteByMetaData(ByteArray p_meta)
{
    if (nullptr == m_index)
        return false;

    return (SPTAG::ErrorCode::Success == m_index->DeleteIndex(p_meta));
}

uint64_t AnnIndex::CalculateBufferSize()
{
    if (nullptr == m_index)
        return 0;

    std::shared_ptr<std::vector<uint64_t>> buffersize = m_index->CalculateBufferSize();
    uint64_t total = sizeof(int) + sizeof(uint64_t) * buffersize->size();
    for (uint64_t bs : *buffersize)
    {
        total += bs;
    }
    return total;
}

ByteArray AnnIndex::Dump(ByteArray p_blobs)
{
    if (nullptr == m_index)
        return ByteArray::c_empty;

    std::shared_ptr<std::vector<uint64_t>> buffersize = m_index->CalculateBufferSize();
    std::uint8_t *ptr = p_blobs.Data(), *pdata = ptr + sizeof(int) + sizeof(uint64_t) * buffersize->size();
    *((int *)ptr) = (int)(buffersize->size());
    ptr += sizeof(int);

    std::vector<SPTAG::ByteArray> indexBlobs;
    for (size_t i = 0; i < buffersize->size(); i++)
    {
        *((uint64_t *)ptr) = buffersize->at(i);
        ptr += sizeof(uint64_t);
        indexBlobs.push_back(SPTAG::ByteArray(pdata, buffersize->at(i), false));
        pdata += buffersize->at(i);
    }

    std::string config;
    if (SPTAG::ErrorCode::Success != m_index->SaveIndex(config, indexBlobs))
    {
        return ByteArray::c_empty;
    }
    std::uint8_t *newdata = new std::uint8_t[config.size()];
    memcpy(newdata, config.c_str(), config.size());
    return ByteArray(newdata, config.size(), false);
}

AnnIndex AnnIndex::LoadFromDump(ByteArray p_config, ByteArray p_blobs)
{
    if (p_config.Length() == 0)
        return AnnIndex(0);

    std::uint8_t *ptr = p_blobs.Data();
    int streamNum = *((int *)ptr);
    ptr += sizeof(int);
    std::uint8_t *pdata = ptr + sizeof(uint64_t) * streamNum;

    std::vector<SPTAG::ByteArray> p_indexBlobs;
    for (int i = 0; i < streamNum; i++)
    {
        std::uint64_t streamSize = *((uint64_t *)ptr);
        ptr += sizeof(uint64_t);
        p_indexBlobs.push_back(SPTAG::ByteArray((std::uint8_t *)pdata, streamSize, false));
        pdata += streamSize;
    }

    std::shared_ptr<SPTAG::VectorIndex> vecIndex;
    std::string config((char *)p_config.Data(), p_config.Length());
    if (SPTAG::ErrorCode::Success != SPTAG::VectorIndex::LoadIndex(config, p_indexBlobs, vecIndex) ||
        nullptr == vecIndex)
    {
        return AnnIndex(0);
    }
    return AnnIndex(vecIndex);
}

AnnIndex AnnIndex::Load(const char *p_loaderFile)
{
    std::shared_ptr<SPTAG::VectorIndex> vecIndex;
    auto ret = SPTAG::VectorIndex::LoadIndex(p_loaderFile, vecIndex);
    if (SPTAG::ErrorCode::Success != ret || nullptr == vecIndex)
    {
        return AnnIndex(0);
    }

    return AnnIndex(vecIndex);
}

AnnIndex AnnIndex::Merge(const char *p_indexFilePath1, const char *p_indexFilePath2)
{
    std::shared_ptr<SPTAG::VectorIndex> vecIndex, addIndex;
    if (SPTAG::ErrorCode::Success != SPTAG::VectorIndex::LoadIndex(p_indexFilePath1, vecIndex) ||
        SPTAG::ErrorCode::Success != SPTAG::VectorIndex::LoadIndex(p_indexFilePath2, addIndex) ||
        SPTAG::ErrorCode::Success !=
            vecIndex->MergeIndex(addIndex.get(), std::atoi(vecIndex->GetParameter("NumberOfThreads").c_str()), nullptr))
        return AnnIndex(0);

    return AnnIndex(vecIndex);
}

// ============================================================================
// TenantIndexManager Implementation
// ============================================================================

TenantIndexManager::TenantIndexManager(DimensionType p_dimension, const char* p_algoType, const char* p_valueType)
    : m_dimension(p_dimension), m_algoType(SPTAG::IndexAlgoType::Undefined), 
    m_valueType(SPTAG::VectorValueType::Undefined),
    m_headIndexCacheLimitBytes(1024*1024*1024),  // Default 1GB cache limit
    m_headIndexCacheSafetyFactor(1.3),
    m_headCache(nullptr)
{
    SPTAG::Helper::Convert::ConvertStringTo<SPTAG::IndexAlgoType>(p_algoType, m_algoType);
    SPTAG::Helper::Convert::ConvertStringTo<SPTAG::VectorValueType>(p_valueType, m_valueType);
    m_inputVectorSize = SPTAG::GetValueTypeSize(m_valueType) * m_dimension;

    // Initialize shared AIO pool: 4 contexts, 1024 events each
    // Must be large enough for concurrent MultiBatchSearch across multiple tenants
    // Each tenant's BatchSearch submits nprobe(64) × batch_threads IO requests
    SPTAG::Helper::SharedAIOPool::Instance().Initialize(4, 1024);
}

TenantIndexManager::~TenantIndexManager()
{
    if (m_headCache) m_headCache->Clear();
    m_tenantIndices.clear();
    m_lruList.clear();
    m_lruMap.clear();
    m_tenantHeadIndexAccountedBytes.clear();
    m_loadedHeadIndexBytes = 0;
    m_tenantVectorCounts.clear();
    m_tenantSpannWorkDirs.clear();
    m_tenantPostingOffsets.clear();
    m_tenantHeadCounts.clear();
    m_tenantTagRoutingStats.clear();
    m_tenantPivotLevels.clear();
    m_tenantPivotNodeCounts.clear();
    m_tenantNodePivotTags.clear();
    m_tenantPlannedNodeVectors.clear();
    m_tenantPlannedPrimaryNodeVectors.clear();
    m_tenantTagToNodes.clear();
    m_tenantHeadNodeToNode.clear();

    // Shut the shared RocksDB down only after every tenant index (which holds
    // a TenantPrefixedKeyValueIO referencing the shared DB through a
    // shared_ptr) has been released.
    if (m_sharedDB)
    {
        m_sharedDB->ShutDown();
        m_sharedDB.reset();
    }
}

bool TenantIndexManager::EnsureSharedDB()
{
#ifndef ROCKSDB
    fprintf(stderr, "[ERROR] TenantIndexManager: shared RocksDB requested but binary built without ROCKSDB.\n");
    return false;
#else
    std::lock_guard<std::mutex> lk(m_sharedDBMutex);
    if (m_sharedDB) return true;
    std::string base = m_baseStoragePath.empty() ? std::string("./tenant_index") : m_baseStoragePath;
    EnsureDir(base);
    if (m_baseStoragePath.empty()) m_baseStoragePath = base;

    // Single shared DB at <baseDir>/rocksdb_shared_0/. The trailing _0 mirrors
    // PrepareDB()'s per-layer suffix; only layer 0 is exercised today.
    std::string dbPath = base + "/rocksdb_shared_0";
    std::shared_ptr<SPTAG::SPANN::RocksDBIO> db;
    try
    {
        db = std::make_shared<SPTAG::SPANN::RocksDBIO>(
            dbPath.c_str(), m_useDirectIO, m_enableWAL, /*recovery=*/false);
    }
    catch (...)
    {
        fprintf(stderr, "[ERROR] TenantIndexManager: failed to open shared RocksDB at %s\n", dbPath.c_str());
        return false;
    }
    if (!db || !db->Available())
    {
        fprintf(stderr, "[ERROR] TenantIndexManager: shared RocksDB unavailable at %s\n", dbPath.c_str());
        return false;
    }
    m_sharedDB = db;
    fprintf(stderr, "[INFO] TenantIndexManager: opened shared RocksDB at %s\n", dbPath.c_str());
    return true;
#endif
}

bool TenantIndexManager::InjectSharedDB(const std::shared_ptr<AnnIndex>& p_idx, int p_internalId)
{
    if (!p_idx) return false;
    if (!m_sharedDB)
    {
        fprintf(stderr, "[ERROR] TenantIndexManager: shared DB not initialised before InjectSharedDB.\n");
        return false;
    }
    auto wrapper = std::make_shared<SPTAG::Helper::TenantPrefixedKeyValueIO>(m_sharedDB, p_internalId);
    if (!p_idx->SetSharedDB(wrapper))
    {
        fprintf(stderr, "[ERROR] TenantIndexManager: tenant %d index is not SPANN<Float>; cannot share DB.\n", p_internalId);
        return false;
    }
    return true;
}

std::shared_ptr<AnnIndex> TenantIndexManager::LoadSpannWithSharedDB(const std::string& p_folder, int p_internalId)
{
    using namespace SPTAG;

    std::string folderPath = p_folder;
    if (!folderPath.empty() && folderPath.back() != FolderSep) folderPath += FolderSep;

    Helper::IniReader iniReader;
    {
        auto fp = SPTAG::f_createIO();
        if (fp == nullptr || !fp->Initialize((folderPath + "indexloader.ini").c_str(), std::ios::in))
            return nullptr;
        if (ErrorCode::Success != iniReader.LoadIni(fp)) return nullptr;
    }

    IndexAlgoType algoType = iniReader.GetParameter("Index", "IndexAlgoType", IndexAlgoType::Undefined);
    VectorValueType valueType = iniReader.GetParameter("Index", "ValueType", VectorValueType::Undefined);
    std::shared_ptr<VectorIndex> vecIndex = VectorIndex::CreateInstance(algoType, valueType);
    if (vecIndex == nullptr) return nullptr;

    if (vecIndex->LoadIndexConfig(iniReader) != ErrorCode::Success) return nullptr;

    if (algoType == IndexAlgoType::SPANN)
    {
        vecIndex->SetParameter("IndexDirectory", p_folder.c_str(), "Base");
        // Disable lazy per-tenant DB creation; the searcher must use m_externalDB.
        vecIndex->SetParameter("ShareDB", "true", "BuildSSDIndex");
        if (!EnsureSharedDB()) return nullptr;
        if (valueType == VectorValueType::Float)
        {
            auto spann = std::dynamic_pointer_cast<SPTAG::SPANN::Index<float>>(vecIndex);
            if (!spann) return nullptr;
            auto wrapper = std::make_shared<SPTAG::Helper::TenantPrefixedKeyValueIO>(m_sharedDB, p_internalId);
            spann->SetSharedDB(wrapper);
        }
        else
        {
            return nullptr;
        }
    }

    auto indexfiles = vecIndex->GetIndexFiles();
    if (iniReader.DoesSectionExist("MetaData"))
    {
        indexfiles->push_back("metadata.bin");
        indexfiles->push_back("metadataIndex.bin");
    }
    if (iniReader.DoesSectionExist("Quantizer"))
    {
        indexfiles->push_back("quantizer.bin");
    }

    std::vector<std::shared_ptr<Helper::DiskIO>> handles;
    for (std::string& f : *indexfiles)
    {
        auto ptr = SPTAG::f_createIO();
        if (ptr == nullptr || !ptr->Initialize((folderPath + f).c_str(),
                                                std::ios::binary | std::ios::in))
        {
            ptr = nullptr;
        }
        handles.push_back(std::move(ptr));
    }

    if (vecIndex->LoadIndexData(handles) != ErrorCode::Success) return nullptr;

    size_t metaStart = vecIndex->GetIndexFiles()->size();
    if (iniReader.DoesSectionExist("MetaData"))
    {
        vecIndex->SetMetadata(new SPTAG::MemMetadataSet(handles[metaStart], handles[metaStart + 1],
                                                        vecIndex->m_iDataBlockSize, vecIndex->m_iDataCapacity,
                                                        vecIndex->m_iMetaRecordSize));
        if (!(vecIndex->GetMetadata()->Available())) return nullptr;
        if (iniReader.GetParameter("MetaData", "MetaDataToVectorIndex", std::string()) == "true")
            vecIndex->BuildMetaMapping();
        metaStart += 2;
    }
    if (iniReader.DoesSectionExist("Quantizer"))
    {
        vecIndex->SetQuantizer(SPTAG::COMMON::IQuantizer::LoadIQuantizer(handles[metaStart]));
        if (!vecIndex->m_pQuantizer) return nullptr;
    }
    vecIndex->SetReady(true);
    return std::make_shared<AnnIndex>(AnnIndex(vecIndex));
}

bool TenantIndexManager::BuildFromData(ByteArray p_vectors, ByteArray p_metadata, SizeType p_vectorNum,
                                       bool p_withMetaIndex, bool p_normalized)
{
    if (p_vectorNum == 0 || m_dimension == 0 || p_vectors.Length() != p_vectorNum * m_inputVectorSize)
    {
        return false;
    }

    m_tenantIndices.clear();
    m_lruList.clear();
    m_lruMap.clear();
    m_tenantHeadIndexAccountedBytes.clear();
    m_loadedHeadIndexBytes = 0;
    m_tenantVectorCounts.clear();
    m_tenantSpannWorkDirs.clear();
    m_tenantTagRoutingStats.clear();
    m_tenantPivotLevels.clear();
    m_tenantPivotNodeCounts.clear();
    m_tenantNodePivotTags.clear();
    m_tenantTagToNodes.clear();
    m_tenantHeadNodeToNode.clear();

    std::map<int, std::vector<std::pair<const uint8_t*, size_t>>> tenantVectorRanges;
    std::map<int, std::vector<std::string>> tenantMetadataLines;

    const char* metaPtr = reinterpret_cast<const char*>(p_metadata.Data());
    const char* metaEnd = metaPtr + p_metadata.Length();
    const uint8_t* vectorPtr = p_vectors.Data();

    SizeType globalIdx = 0;
    while (metaPtr < metaEnd && globalIdx < p_vectorNum)
    {
        const char* lineEnd = metaPtr;
        while (lineEnd < metaEnd && *lineEnd != '\n')
        {
            lineEnd++;
        }

        if (lineEnd == metaPtr)
        {
            return false;
        }

        std::string metaLine(metaPtr, lineEnd - metaPtr);
        int tenantId = RegisterTenantId(metaLine.c_str());

        tenantVectorRanges[tenantId].push_back({vectorPtr, m_inputVectorSize});
        tenantMetadataLines[tenantId].push_back(metaLine);
        m_tenantGlobalIndices[tenantId].push_back(globalIdx);
        vectorPtr += m_inputVectorSize;
        metaPtr = (lineEnd < metaEnd) ? (lineEnd + 1) : lineEnd;
        globalIdx++;
    }

    std::string algoTypeStr = SPTAG::Helper::Convert::ConvertToString(m_algoType);
    std::string valueTypeStr = SPTAG::Helper::Convert::ConvertToString(m_valueType);

    for (auto& tenantEntry : tenantVectorRanges)
    {
        int tenantId = tenantEntry.first;
        std::vector<std::pair<const uint8_t*, size_t>>& vectorRanges = tenantEntry.second;
        if (vectorRanges.empty())
        {
            continue;
        }

        size_t totalVectorSize = vectorRanges.size() * m_inputVectorSize;
        uint8_t* tenantVectorBuffer = new uint8_t[totalVectorSize];
        uint8_t* out = tenantVectorBuffer;
        for (const auto& vec : vectorRanges)
        {
            memcpy(out, vec.first, vec.second);
            out += vec.second;
        }
        ByteArray tenantVectors(tenantVectorBuffer, totalVectorSize, true);

        std::string metaStr;
        for (size_t i = 0; i < tenantMetadataLines[tenantId].size(); ++i)
        {
            if (i > 0) metaStr.push_back('\n');
            metaStr += tenantMetadataLines[tenantId][i];
        }
        metaStr.push_back('\n');

        uint8_t* metaBuffer = new uint8_t[metaStr.size()];
        memcpy(metaBuffer, metaStr.data(), metaStr.size());
        ByteArray tenantMetadata(metaBuffer, metaStr.size(), true);

        auto tenantIndex = std::make_shared<AnnIndex>(algoTypeStr.c_str(), valueTypeStr.c_str(), m_dimension);
        bool buildOk = false;
        SizeType tenantVecCount = static_cast<SizeType>(vectorRanges.size());
        std::vector<uint32_t> tenantLocalTags;

        if (m_buildNumTagsPerVec > 0 && m_buildTags.Data() != nullptr) {
            const uint32_t* globalTags = reinterpret_cast<const uint32_t*>(m_buildTags.Data());
            auto gidIt = m_tenantGlobalIndices.find(tenantId);
            if (gidIt != m_tenantGlobalIndices.end()) {
                const auto& gids = gidIt->second;
                tenantLocalTags.resize(static_cast<size_t>(tenantVecCount) * static_cast<size_t>(m_buildNumTagsPerVec));
                for (int i = 0; i < tenantVecCount && i < static_cast<int>(gids.size()); ++i) {
                    int gid = gids[i];
                    for (int t = 0; t < m_buildNumTagsPerVec; ++t) {
                        tenantLocalTags[static_cast<size_t>(i) * static_cast<size_t>(m_buildNumTagsPerVec) + static_cast<size_t>(t)] =
                            globalTags[static_cast<size_t>(gid) * static_cast<size_t>(m_buildNumTagsPerVec) + static_cast<size_t>(t)];
                    }
                }
            }
        }

        // Choose index type based on tenant size (hybrid strategy)
        TenantIndexType indexType = ChooseIndexType(tenantVecCount);
        m_tenantIndexTypes[tenantId] = indexType;

        if (indexType == TenantIndexType::SPANN)
        {
            if (!tenantLocalTags.empty()) {
                PivotEstimatorComputation pivotComputation;
                const PivotEstimatorCandidate* pivotCandidate = nullptr;
                if (BuildPivotEstimatorComputation(tenantLocalTags.data(),
                                                   static_cast<int>(tenantVecCount),
                                                   m_buildNumTagsPerVec,
                                                   0,
                                                   0.99,
                                                   10.0,
                                                   1.0,
                                                   std::string(),
                                                   pivotComputation)) {
                    pivotCandidate = FindBestPivotEstimatorCandidate(pivotComputation.candidates);
                }

                if (pivotCandidate != nullptr) {
                    m_tenantPivotLevels[tenantId] = pivotCandidate->pivotLevel;
                    m_tenantPivotNodeCounts[tenantId] = pivotCandidate->nodeCount;
                    m_tenantNodePivotTags[tenantId] = pivotCandidate->nodePivotTags;
                    BuildTagToNodeIndexForCandidate(*pivotCandidate,
                                                    pivotComputation.levelData,
                                                    m_tenantTagToNodes[tenantId]);
                    BuildPrimaryNodeVectorAssignmentsForCandidate(*pivotCandidate,
                                                                  tenantLocalTags.data(),
                                                                  static_cast<int>(tenantVecCount),
                                                                  m_buildNumTagsPerVec,
                                                                  m_tenantPlannedPrimaryNodeVectors[tenantId]);

                    // Keep tree-structured tag-to-node merges for query routing, but
                    // partition postings by the pivot-layer owner only so vectors are
                    // evenly distributed across nodes instead of being replicated by
                    // higher-level ancestor tags.
                    m_tenantPlannedNodeVectors[tenantId] = m_tenantPlannedPrimaryNodeVectors[tenantId];

                    fprintf(stderr,
                            "[INFO] Tenant %d: planned %d routing nodes before SPANN build\n",
                            tenantId,
                            pivotCandidate->nodeCount);
                }
            }

            auto planIt = m_tenantPlannedNodeVectors.find(tenantId);
            auto primaryPlanIt = m_tenantPlannedPrimaryNodeVectors.find(tenantId);
            bool hasNodeAwarePlan = (planIt != m_tenantPlannedNodeVectors.end() && !planIt->second.empty());

            int64_t postingAssignmentCount = static_cast<int64_t>(tenantVecCount);
            if (hasNodeAwarePlan) {
                int64_t plannedAssignmentTotal = 0;
                for (const auto& nodeAssignments : planIt->second) {
                    plannedAssignmentTotal += static_cast<int64_t>(nodeAssignments.size());
                }

                if (plannedAssignmentTotal > 0) {
                    postingAssignmentCount = plannedAssignmentTotal;
                }
            }

            std::string spannWorkDir = "/tmp/sptag_spann_tenant_" + std::to_string(tenantId);
            RemovePathRecursive(spannWorkDir);
            EnsureDir(spannWorkDir);
            tenantIndex->SetBuildParam("IndexDirectory", spannWorkDir.c_str(), "Base");
            tenantIndex->SetBuildParam("DistCalcMethod", "Cosine", "Base");
            tenantIndex->SetBuildParam("isExecute", "true", "SelectHead");
            tenantIndex->SetBuildParam("isExecute", "true", "BuildHead");
            tenantIndex->SetBuildParam("isExecute", "true", "BuildSSDIndex");
            tenantIndex->SetBuildParam("BuildSsdIndex", "true", "BuildSSDIndex");
            tenantIndex->SetBuildParam("Storage", m_storageBackend.c_str(), "BuildSSDIndex");

            // Scale DataCapacity and SSD file size to tenant size
            // Block pool uses 4KB pages; each vector with replication needs multiple blocks
            // Use the routed posting assignment count instead of raw tenant size,
            // because node-aware builds can replicate vectors across routing nodes.
            int64_t dataCapacity64 = std::max<int64_t>(postingAssignmentCount * 8LL, 4096LL);
            int dataCapacity = static_cast<int>(std::min<int64_t>(dataCapacity64, std::numeric_limits<int>::max()));
            tenantIndex->SetBuildParam("DataCapacity", std::to_string(dataCapacity).c_str(), "Base");
            tenantIndex->SetBuildParam("DataBlockSize", std::to_string(std::min(dataCapacity, 1024 * 1024)).c_str(), "Base");

            // Scale SSD file size: each posting can hold ~PostingVectorLimit(118) vectors
            // Each vector in posting: dim*sizeof(float) + metadata overhead ~= dim*4+64 bytes
            // With ReplicaCount=8, total data ~ assignments * replica * vec_bytes / page_size blocks.
            int64_t estimatedBytes = postingAssignmentCount * static_cast<int64_t>(m_dimension * 4 + 64) * 10LL;
                int startFileSizeGB = std::max(1, (int)(estimatedBytes / (1024LL * 1024LL * 1024LL)) + 1);
                if (hasNodeAwarePlan) {
                startFileSizeGB = std::max(startFileSizeGB, 4);
                }
                int maxFileSizeGB = std::max(startFileSizeGB * 3, hasNodeAwarePlan ? 32 : 10);
            tenantIndex->SetBuildParam("StartFileSizeGB", std::to_string(startFileSizeGB).c_str(), "BuildSSDIndex");
                tenantIndex->SetBuildParam("MaxFileSizeGB", std::to_string(maxFileSizeGB).c_str(), "BuildSSDIndex");

            fprintf(stderr,
                    "[INFO] Tenant %d: posting assignments=%lld raw_vectors=%d StartFileSizeGB=%d MaxFileSizeGB=%d DataCapacity=%d\n",
                    tenantId,
                    static_cast<long long>(postingAssignmentCount),
                    tenantVecCount,
                    startFileSizeGB,
                    maxFileSizeGB,
                    dataCapacity);

            // Scale graph build parameters by tenant size to avoid fixed overhead on small tenants
            // TPTNumber: controls how many random partition trees for initial KNN graph
            // RefineIterations: controls graph refinement passes
            int tptNumber = 32;
            int refineIter = 2;
            if (tenantVecCount < 10000) {
                tptNumber = 8;
                refineIter = 2;
            } else if (tenantVecCount < 50000) {
                tptNumber = 8;
                refineIter = 2;
            } else if (tenantVecCount < 200000) {
                tptNumber = 16;
                refineIter = 2;
            } else if (tenantVecCount < 500000) {
                tptNumber = 16;
                refineIter = 2;
            }
            tenantIndex->SetBuildParam("TPTNumber", std::to_string(tptNumber).c_str(), "BuildHead");
            tenantIndex->SetBuildParam("RefineIterations", std::to_string(refineIter).c_str(), "BuildHead");

            // Set per-vector tags to embed in posting metadata (if available from BuildFromDataWithTags)
            if (!tenantLocalTags.empty()) {
                tenantIndex->SetBuildParam("NumTagsPerVec", std::to_string(m_buildNumTagsPerVec).c_str(), "BuildSSDIndex");
                tenantIndex->SetVectorTags(tenantLocalTags.data(), tenantVecCount, m_buildNumTagsPerVec);
            }
            if (hasNodeAwarePlan) {
                tenantIndex->SetNodeVectorAssignments(planIt->second);
            }
            if (primaryPlanIt != m_tenantPlannedPrimaryNodeVectors.end() && !primaryPlanIt->second.empty()) {
                tenantIndex->SetPrimaryNodeVectorAssignments(primaryPlanIt->second);
            }

            // Shared RocksDB: when enabled, inject a tenant-prefixed wrapper
            // BEFORE Build so SPANN's ExtraDynamicSearcher reuses the shared
            // store instead of opening a per-tenant RocksDB.
            if (m_storageBackend == "ROCKSDBIO" && m_useSharedDB)
            {
                tenantIndex->SetBuildParam("ShareDB", "true", "BuildSSDIndex");
                if (!EnsureSharedDB()) return false;
                if (!InjectSharedDB(tenantIndex, tenantId)) return false;
            }

            buildOk = tenantIndex->Build(tenantVectors, tenantVecCount, p_normalized);
            m_tenantSpannWorkDirs[tenantId] = spannWorkDir;
            fprintf(stderr, "[INFO] Tenant %d: SPANN build (%d vectors)\n", tenantId, tenantVecCount);
        }
        else if (indexType == TenantIndexType::BKT)
        {
            // Medium tenant: build in-memory BKT index
            tenantIndex = std::make_shared<AnnIndex>("BKT", valueTypeStr.c_str(), m_dimension);
            tenantIndex->SetBuildParam("DistCalcMethod", "Cosine", "Index");
            buildOk = tenantIndex->Build(tenantVectors, tenantVecCount, p_normalized);
            fprintf(stderr, "[INFO] Tenant %d: BKT build (%d vectors)\n", tenantId, tenantVecCount);
        }
        else // BRUTEFORCE
        {
            // Small tenant: build trivial BKT index (effectively brute force at this scale)
            tenantIndex = std::make_shared<AnnIndex>("BKT", valueTypeStr.c_str(), m_dimension);
            tenantIndex->SetBuildParam("DistCalcMethod", "Cosine", "Index");
            buildOk = tenantIndex->Build(tenantVectors, tenantVecCount, p_normalized);
            fprintf(stderr, "[INFO] Tenant %d: BruteForce build (%d vectors)\n", tenantId, tenantVecCount);
        }

        if (!buildOk)
        {
            return false;
        }

        m_tenantVectorCounts[tenantId] = static_cast<int>(vectorRanges.size());

        // For SPANN: save the index to its work dir right away, then release the
        // AnnIndex object.  This closes the SSD file descriptor and frees the
        // HeadIndex memory, preventing fd exhaustion when building many tenants.
        if (indexType == TenantIndexType::SPANN)
        {
            std::string workDir = m_tenantSpannWorkDirs[tenantId];
            tenantIndex->Save(workDir.c_str());
            fprintf(stderr, "[INFO] Tenant %d: built & released (%d vectors, dir=%s)\n",
                tenantId, (int)vectorRanges.size(), workDir.c_str());
            tenantIndex.reset();
            continue;
        }

        m_tenantIndices[tenantId] = tenantIndex;
    }

    // For SPANN tenants: compute posting offsets and record head counts
    if (m_algoType == SPTAG::IndexAlgoType::SPANN)
    {
        m_tenantPostingOffsets.clear();
        m_tenantHeadCounts.clear();
        m_totalPostingCount = 0;

        // Iterate all tenants (not just loaded ones — released tenants have work dirs)
        for (const auto& kv : m_tenantVectorCounts)
        {
            int tenantId = kv.first;
            auto typeIt = m_tenantIndexTypes.find(tenantId);
            if (typeIt == m_tenantIndexTypes.end() || typeIt->second != TenantIndexType::SPANN)
            {
                // Non-SPANN tenants don't have postings in the shared SSD
                m_tenantPostingOffsets[tenantId] = -1;
                m_tenantHeadCounts[tenantId] = 0;
                continue;
            }
            // Get head count from SPTAGHeadVectorIDs.bin file size
            // Each entry is sizeof(uint64_t) = 8 bytes
            std::string headIDFile = m_tenantSpannWorkDirs[tenantId] + "/SPTAGHeadVectorIDs.bin";
            int headCount = 0;

            if (fileexists(headIDFile.c_str()))
            {
                int64_t fsize = filesize(headIDFile.c_str());
                headCount = static_cast<int>(fsize / sizeof(uint64_t));
            }

            if (headCount <= 0)
            {
                // Fallback: read vector file
                std::string headVecFile = m_tenantSpannWorkDirs[tenantId] + "/SPTAGHeadVectors.bin";
                if (fileexists(headVecFile.c_str()))
                {
                    int64_t fsize = filesize(headVecFile.c_str());
                    headCount = static_cast<int>(fsize / (m_dimension * SPTAG::GetValueTypeSize(m_valueType)));
                }
            }

            if (headCount <= 0)
            {
                fprintf(stderr, "[ERROR] Cannot determine head count for tenant %d\n", tenantId);
                return false;
            }

            m_tenantPostingOffsets[tenantId] = m_totalPostingCount;
            m_tenantHeadCounts[tenantId] = headCount;
            m_totalPostingCount += headCount;
            fprintf(stderr, "[INFO] Tenant %d: headCount=%d, postingOffset=%d\n",
                tenantId, headCount, m_tenantPostingOffsets[tenantId]);
        }

        fprintf(stderr, "[INFO] Total posting count across %d tenants: %d\n",
            (int)m_tenantVectorCounts.size(), m_totalPostingCount);
    }

    return !m_tenantVectorCounts.empty();
}

bool TenantIndexManager::BuildFromDataWithTags(ByteArray p_vectors, ByteArray p_metadata, SizeType p_vectorNum,
                                                ByteArray p_tags, int p_numTagsPerVec,
                                                bool p_withMetaIndex, bool p_normalized)
{
    // Store tags and numTagsPerVec for the build process.
    // BuildFromData will be modified to pass tags to each SPANN index.
    m_buildTags = p_tags;
    m_buildNumTagsPerVec = p_numTagsPerVec;

    // Build SPANN indexes — tags will be embedded in postings via SetVectorTags
    if (!BuildFromData(p_vectors, p_metadata, p_vectorNum, p_withMetaIndex, p_normalized))
        return false;

    // Also build PS (Posting Signature) Bloom filters for posting-level pre-filter
    const char* metaPtr = reinterpret_cast<const char*>(p_metadata.Data());
    const char* metaEnd = metaPtr + p_metadata.Length();
    const uint32_t* tagsPtr = reinterpret_cast<const uint32_t*>(p_tags.Data());

    std::map<int, std::vector<int>> tenantGlobalIndices;
    SizeType globalIdx = 0;
    const char* mp = metaPtr;
    while (mp < metaEnd && globalIdx < p_vectorNum)
    {
        const char* lineEnd = mp;
        while (lineEnd < metaEnd && *lineEnd != '\n') lineEnd++;
        std::string metaLine(mp, lineEnd - mp);

        int tenantId = -1;
        {
            std::lock_guard<std::mutex> lock(m_tenantIdMutex);
            auto it = m_tenantStrToInt.find(metaLine);
            if (it != m_tenantStrToInt.end()) tenantId = it->second;
        }
        if (tenantId >= 0)
            tenantGlobalIndices[tenantId].push_back(globalIdx);

        mp = (lineEnd < metaEnd) ? (lineEnd + 1) : lineEnd;
        globalIdx++;
    }

    for (auto& [tenantId, globalIds] : tenantGlobalIndices)
    {
        int n = (int)globalIds.size();
        std::vector<uint32_t> tenantTags(n * p_numTagsPerVec);
        for (int i = 0; i < n; i++)
            for (int t = 0; t < p_numTagsPerVec; t++)
                tenantTags[i * p_numTagsPerVec + t] = tagsPtr[globalIds[i] * p_numTagsPerVec + t];

        uint8_t* tagBuf = new uint8_t[tenantTags.size() * sizeof(uint32_t)];
        memcpy(tagBuf, tenantTags.data(), tenantTags.size() * sizeof(uint32_t));
        ByteArray tagBytes(tagBuf, tenantTags.size() * sizeof(uint32_t), true);
        BuildSignatures(tenantId, tagBytes, n, p_numTagsPerVec);
    }

    m_buildTags = ByteArray();  // release reference
    m_buildNumTagsPerVec = 0;

    fprintf(stderr, "[INFO] BuildFromDataWithTags: tags embedded in postings + PS signatures for %d tenants\n",
            (int)tenantGlobalIndices.size());
    return true;
}

std::shared_ptr<QueryResult> TenantIndexManager::Search(ByteArray p_queryVector, int p_tenantId, int p_resultNum)
{
    if (!EnsureTenantLoaded(p_tenantId))
    {
        return nullptr;
    }

    // Get index under shared lock (concurrent reads safe)
    std::shared_ptr<AnnIndex> indexPtr;
    {
        std::shared_lock<std::shared_mutex> rlock(m_tenantIndicesMutex);
        auto it = m_tenantIndices.find(p_tenantId);
        if (it == m_tenantIndices.end()) return nullptr;
        indexPtr = it->second;  // shared_ptr copy under lock, search outside lock
    }

    return indexPtr->Search(p_queryVector, p_resultNum);
}

std::shared_ptr<QueryResult> TenantIndexManager::BatchSearch(ByteArray p_queryVectors, int p_vectorNum,
                                                              int p_tenantId, int p_resultNum)
{
    if (!EnsureTenantLoaded(p_tenantId))
    {
        return nullptr;
    }

    std::shared_ptr<AnnIndex> indexPtr;
    {
        std::shared_lock<std::shared_mutex> rlock(m_tenantIndicesMutex);
        auto it = m_tenantIndices.find(p_tenantId);
        if (it == m_tenantIndices.end()) return nullptr;
        indexPtr = it->second;
    }
    return indexPtr->BatchSearch(p_queryVectors, p_vectorNum, p_resultNum, false);
}

std::shared_ptr<QueryResult> TenantIndexManager::MultiBatchSearch(
    ByteArray p_queryVectors, int p_vectorNum, ByteArray p_tenantIds, int p_resultNum)
{
    const int32_t* tenantIds = reinterpret_cast<const int32_t*>(p_tenantIds.Data());
    const uint8_t* vectors = p_queryVectors.Data();
    size_t vecSize = m_inputVectorSize;

    // Group queries by tenant: tenant_id → [(original_index, vector_ptr)]
    // Using ordered map ensures deterministic tenant processing order
    std::map<int, std::vector<std::pair<int, const uint8_t*>>> groups;
    for (int i = 0; i < p_vectorNum; i++)
    {
        groups[tenantIds[i]].emplace_back(i, vectors + i * vecSize);
    }

    // OPTIMIZATION 1: Sort tenants by batch count (most queries first)
    // and group same-tenant queries together for better cache locality.
    std::vector<std::pair<int, int>> tenantOrder;  // (tenant_id, query_count)
    for (auto& [tid, qs] : groups)
        tenantOrder.emplace_back(tid, (int)qs.size());
    std::sort(tenantOrder.begin(), tenantOrder.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Allocate output: p_vectorNum × p_resultNum results
    auto output = std::make_shared<QueryResult>(nullptr, p_vectorNum * p_resultNum, false);
    BasicResult* outResults = output->GetResults();

    for (int i = 0; i < p_vectorNum * p_resultNum; i++)
    {
        outResults[i].VID = -1;
        outResults[i].Dist = SPTAG::MaxDist;
    }

    // Pre-load each tenant and IMMEDIATELY grab shared_ptr to prevent
    // subsequent loads from evicting it.  This lets the cache temporarily
    // exceed the limit for one batch; excess is reclaimed on next batch.
    std::map<int, std::shared_ptr<AnnIndex>> heldIndices;
    for (auto& [tid, _] : tenantOrder)
    {
        EnsureTenantLoaded(tid);
        // Immediately pin so the next EnsureTenantLoaded won't evict this one
        std::shared_lock<std::shared_mutex> rlock(m_tenantIndicesMutex);
        auto it = m_tenantIndices.find(tid);
        if (it != m_tenantIndices.end())
            heldIndices[tid] = it->second;  // use_count > 1 → eviction-proof
    }

    // Dispatch BatchSearch per tenant in parallel
    std::vector<std::thread> threads;

    for (auto& [tid, queryList] : groups)
    {
        auto idxIt = heldIndices.find(tid);
        if (idxIt == heldIndices.end()) continue;
        auto indexPtr = idxIt->second;  // shared_ptr copy → ref count held during search

        threads.emplace_back([&, tid, indexPtr]() {
            int n = (int)queryList.size();
            if (n == 0) return;

            std::vector<uint8_t> buf(n * vecSize);
            for (int i = 0; i < n; i++)
            {
                memcpy(buf.data() + i * vecSize, queryList[i].second, vecSize);
            }

            ByteArray batchData(buf.data(), buf.size(), false);
            auto batchResult = indexPtr->BatchSearch(batchData, n, p_resultNum, false);
            if (!batchResult) return;

            BasicResult* batchRes = batchResult->GetResults();
            for (int i = 0; i < n; i++)
            {
                int origIdx = queryList[i].first;
                memcpy(outResults + origIdx * p_resultNum,
                       batchRes + i * p_resultNum,
                       p_resultNum * sizeof(BasicResult));
            }
        });
    }

    for (auto& t : threads) t.join();

    // Release held indices after all searches complete
    heldIndices.clear();

    return output;
}

void TenantIndexManager::GetTenantIds(int* p_tenants, int* p_count) const
{
    int idx = 0;
    for (const auto& [tenantId, _] : m_tenantVectorCounts)
    {
        p_tenants[idx++] = tenantId;
    }
    *p_count = (int)m_tenantVectorCounts.size();
}

int TenantIndexManager::GetTenantCount() const
{
    return (int)m_tenantVectorCounts.size();
}

int TenantIndexManager::GetTenantVectorCount(int p_tenantId) const
{
    auto it = m_tenantVectorCounts.find(p_tenantId);
    if (it != m_tenantVectorCounts.end())
    {
        return it->second;
    }
    return 0;  // Tenant not found
}

uint64_t TenantIndexManager::GetTenantHeadIndexSize(int p_tenantId) const
{
    auto workIt = m_tenantSpannWorkDirs.find(p_tenantId);
    if (workIt == m_tenantSpannWorkDirs.end()) {
        return 0;
    }

    return GetPathSizeBytes(workIt->second + "/HeadIndex");
}

uint64_t TenantIndexManager::EstimateTenantHeadIndexBytes(int p_tenantId) const
{
    uint64_t onDiskBytes = GetTenantHeadIndexSize(p_tenantId);
    if (onDiskBytes > 0) {
        return static_cast<uint64_t>(std::ceil(static_cast<double>(onDiskBytes) * m_headIndexCacheSafetyFactor));
    }

    auto vcIt = m_tenantVectorCounts.find(p_tenantId);
    if (vcIt != m_tenantVectorCounts.end()) {
        return static_cast<uint64_t>(vcIt->second) * 128ULL;
    }

    return 1024ULL * 1024ULL;
}

ByteArray TenantIndexManager::GetTagRoutingStatsBlob(int p_tenantId) const
{
    auto routeIt = m_tenantTagRoutingStats.find(p_tenantId);
    if (routeIt == m_tenantTagRoutingStats.end() || routeIt->second.empty()) {
        return ByteArray();
    }

    std::vector<TagRoutingStatRecord> entries;
    entries.reserve(routeIt->second.size());
    for (const auto& [tag, stats] : routeIt->second) {
        entries.push_back(TagRoutingStatRecord{
            tag,
            static_cast<int32_t>(stats.vectorCount),
            static_cast<int32_t>(stats.postingCount),
        });
    }

    std::sort(entries.begin(), entries.end(), [](const TagRoutingStatRecord& left, const TagRoutingStatRecord& right) {
        return left.tag < right.tag;
    });

    ByteArray payload = ByteArray::Alloc(entries.size() * sizeof(TagRoutingStatRecord));
    std::memcpy(payload.Data(), entries.data(), payload.Length());
    return payload;
}

ByteArray TenantIndexManager::EstimatePivotBuildPlan(ByteArray p_tags,
                                                     int p_numVectors,
                                                     int p_numTagsPerVec,
                                                     int p_maxNodes,
                                                     float p_recallTarget,
                                                     float p_lambdaRecall,
                                                     float p_estimatedRecall,
                                                     ByteArray p_levelWeightsCsv) const
{
    if (p_numVectors <= 0 || p_numTagsPerVec <= 0 || p_tags.Data() == nullptr) {
        return ByteArray();
    }

    size_t expectedBytes = static_cast<size_t>(p_numVectors) * static_cast<size_t>(p_numTagsPerVec) * sizeof(uint32_t);
    if (p_tags.Length() < expectedBytes) {
        return ByteArray();
    }

    std::string weightsCsv;
    if (p_levelWeightsCsv.Data() != nullptr && p_levelWeightsCsv.Length() > 0) {
        weightsCsv.assign(reinterpret_cast<const char*>(p_levelWeightsCsv.Data()), p_levelWeightsCsv.Length());
    }

    PivotEstimatorComputation computation;
    if (!BuildPivotEstimatorComputation(reinterpret_cast<const uint32_t*>(p_tags.Data()),
                                        p_numVectors,
                                        p_numTagsPerVec,
                                        p_maxNodes,
                                        std::clamp(static_cast<double>(p_recallTarget), 0.01, 1.0),
                                        std::max(0.0, static_cast<double>(p_lambdaRecall)),
                                        std::clamp(static_cast<double>(p_estimatedRecall), 0.0, 1.0),
                                        weightsCsv,
                                        computation)) {
        return ByteArray();
    }

    const PivotEstimatorCandidate* best = FindBestPivotEstimatorCandidate(computation.candidates);
    if (best == nullptr) return ByteArray();

    std::ostringstream json;
    json << "{";
    json << "\"planner_strategy\":\"greedy_leaf_packing\",";
    json << "\"min_local_selectivity\":" << kGreedyLeafMinLocalSelectivity << ",";
    json << "\"num_vectors\":" << p_numVectors << ",";
    json << "\"num_levels\":" << p_numTagsPerVec << ",";
    json << "\"requested_max_nodes\":" << p_maxNodes << ",";
    json << "\"recall_target\":" << std::clamp(static_cast<double>(p_recallTarget), 0.01, 1.0) << ",";
    json << "\"lambda_recall\":" << std::max(0.0, static_cast<double>(p_lambdaRecall)) << ",";
    json << "\"estimated_recall\":" << std::clamp(static_cast<double>(p_estimatedRecall), 0.0, 1.0) << ",";
    json << "\"best_plan\":{";
    json << "\"pivot_level\":" << best->pivotLevel << ",";
    json << "\"node_count\":" << best->nodeCount << ",";
    json << "\"latency_cost\":" << best->latencyCost << ",";
    json << "\"recall_penalty\":" << best->recallPenalty << ",";
    json << "\"total_cost\":" << best->totalCost << "},";
    json << "\"candidates\":[";
    for (size_t idx = 0; idx < computation.candidates.size(); ++idx)
    {
        const auto& candidate = computation.candidates[idx];
        if (idx > 0) json << ",";
        json << "{";
        json << "\"pivot_level\":" << candidate.pivotLevel << ",";
        json << "\"node_count\":" << candidate.nodeCount << ",";
        json << "\"latency_cost\":" << candidate.latencyCost << ",";
        json << "\"recall_penalty\":" << candidate.recallPenalty << ",";
        json << "\"total_cost\":" << candidate.totalCost << ",";
        json << "\"node_sizes\":[";
        for (size_t i = 0; i < candidate.nodeSizes.size(); ++i) {
            if (i > 0) json << ",";
            json << candidate.nodeSizes[i];
        }
        json << "],";
        json << "\"node_pivot_tags\":[";
        for (size_t node = 0; node < candidate.nodePivotTags.size(); ++node)
        {
            if (node > 0) json << ",";
            json << "[";
            for (size_t tagIdx = 0; tagIdx < candidate.nodePivotTags[node].size(); ++tagIdx)
            {
                if (tagIdx > 0) json << ",";
                json << candidate.nodePivotTags[node][tagIdx];
            }
            json << "]";
        }
        json << "]";
        json << "}";
    }
    json << "]";
    json << "}";

    const std::string payload = json.str();
    ByteArray output = ByteArray::Alloc(payload.size());
    std::memcpy(output.Data(), payload.data(), payload.size());
    return output;
}

bool TenantIndexManager::SaveAll(const char* p_baseDir)
{
    m_baseStoragePath = std::string(p_baseDir);
    if (!EnsureDir(m_baseStoragePath))
    {
        return false;
    }

    // For SPANN: copy shared SSD infrastructure once
    // For other algos: save each tenant's data sequentially without directory overhead
    
    if (!SaveUnifiedStorage(p_baseDir))
    {
        return false;
    }

    // Checkpoint the shared RocksDB into <baseDir>/rocksdb_shared_0/ when
    // saving to a directory other than where the live DB lives. RocksDB
    // persists in place, so same-dir saves are a no-op.
    if (m_useSharedDB && m_sharedDB)
    {
        if (m_sharedDB->Checkpoint(std::string(p_baseDir)) != SPTAG::ErrorCode::Success)
        {
            fprintf(stderr, "[ERROR] TenantIndexManager::SaveAll: failed to checkpoint shared RocksDB to %s\n", p_baseDir);
            return false;
        }
    }

    // Write manifest for all tenants
    std::string manifestPath = m_baseStoragePath + "/manifest.txt";
    FILE* manifestFile = fopen(manifestPath.c_str(), "w");
    if (!manifestFile)
    {
        return false;
    }

    fprintf(manifestFile, "dimension %d\n", static_cast<int>(m_dimension));
    fprintf(manifestFile, "algorithm %s\n", m_algoType == SPTAG::IndexAlgoType::SPANN ? "SPANN" : 
            (m_algoType == SPTAG::IndexAlgoType::BKT ? "BKT" : "KDT"));
    fprintf(manifestFile, "unified_storage 1\n");
    fprintf(manifestFile, "total_postings %d\n", m_totalPostingCount);
    
    for (const auto& kv : m_tenantVectorCounts)
    {
        int tenantId = kv.first;
        int count = kv.second;
        int postingOffset = 0;
        int headCount = 0;
        auto offIt = m_tenantPostingOffsets.find(tenantId);
        if (offIt != m_tenantPostingOffsets.end()) postingOffset = offIt->second;
        auto hcIt = m_tenantHeadCounts.find(tenantId);
        if (hcIt != m_tenantHeadCounts.end()) headCount = hcIt->second;
        int typeInt = 0;
        auto typeIt = m_tenantIndexTypes.find(tenantId);
        if (typeIt != m_tenantIndexTypes.end()) typeInt = static_cast<int>(typeIt->second);
        // Format: tenant <id> <vecCount> <postingOffset> <headCount> <indexType>
        fprintf(manifestFile, "tenant %d %d %d %d %d\n", tenantId, count, postingOffset, headCount, typeInt);
    }

    // Save string tenant ID ↔ internal ID mapping
    {
        std::lock_guard<std::mutex> lock(m_tenantIdMutex);
        for (const auto& kv : m_tenantStrToInt)
        {
            // Format: tenant_mapping <internalId> <stringId>
            fprintf(manifestFile, "tenant_mapping %d %s\n", kv.second, kv.first.c_str());
        }
    }

    fclose(manifestFile);

    return true;
}

bool TenantIndexManager::LoadAll(const char* p_baseDir)
{
    m_tenantIndices.clear();
    m_lruList.clear();
    m_lruMap.clear();
    m_tenantHeadIndexAccountedBytes.clear();
    m_loadedHeadIndexBytes = 0;
    m_tenantVectorCounts.clear();
    m_tenantIndexPaths.clear();
    m_tenantSpannWorkDirs.clear();
    m_tenantTagRoutingStats.clear();
    m_tenantPivotLevels.clear();
    m_tenantPivotNodeCounts.clear();
    m_tenantNodePivotTags.clear();
    m_tenantTagToNodes.clear();
    m_tenantHeadNodeToNode.clear();

    // Clear tenant ID mapping
    {
        std::lock_guard<std::mutex> lock(m_tenantIdMutex);
        m_tenantStrToInt.clear();
        m_tenantIntToStr.clear();
        m_nextInternalId = 0;
    }

    std::string baseDir(p_baseDir);
    m_baseStoragePath = baseDir;
    std::string manifestPath = baseDir + "/manifest.txt";
    std::ifstream in(manifestPath.c_str());
    if (!in)
    {
        return false;
    }

    // Read manifest
    std::string line;
    bool unifiedStorage = false;
    while (std::getline(in, line))
    {
        std::istringstream iss(line);
        std::string key;
        iss >> key;
        if (key == "dimension")
        {
            int dim = 0;
            if (!(iss >> dim) || dim != m_dimension)
            {
                return false;
            }
        }
        else if (key == "unified_storage")
        {
            int val = 0;
            if (iss >> val)
            {
                unifiedStorage = (val != 0);
            }
        }
        else if (key == "total_postings")
        {
            int val = 0;
            if (iss >> val) m_totalPostingCount = val;
        }
        else if (key == "tenant")
        {
            int tenantId = 0;
            int count = 0;
            int postingOffset = 0;
            int headCount = 0;
            int typeInt = 0;
            if (!(iss >> tenantId >> count))
            {
                return false;
            }
            // Optional fields: postingOffset, headCount, indexType
            iss >> postingOffset >> headCount >> typeInt;
            m_tenantVectorCounts[tenantId] = count;
            m_tenantPostingOffsets[tenantId] = postingOffset;
            m_tenantHeadCounts[tenantId] = headCount;
            m_tenantIndexTypes[tenantId] = static_cast<TenantIndexType>(typeInt);
        }
        else if (key == "tenant_mapping")
        {
            int internalId = 0;
            std::string strId;
            if (iss >> internalId >> strId)
            {
                std::lock_guard<std::mutex> lock(m_tenantIdMutex);
                m_tenantStrToInt[strId] = internalId;
                m_tenantIntToStr[internalId] = strId;
                if (internalId >= m_nextInternalId)
                    m_nextInternalId = internalId + 1;
            }
        }
    }
    in.close();

    // Load tenant indices based on storage type
    if (unifiedStorage)
    {
        return LoadUnifiedStorage(p_baseDir);
    }
    else
    {
        // Legacy: load from tenant_XX directories for backward compatibility
        for (const auto& kv : m_tenantVectorCounts)
        {
            int tenantId = kv.first;
            m_tenantSpannWorkDirs[tenantId] = baseDir + "/tenant_" + std::to_string(tenantId) + "/index";
        }
        LoadTenantSparseIndices();
        return true;
    }
}

void TenantIndexManager::LoadTenantSparseIndices()
{
    // Sparse-tag fast-path index is small (<<1MB/tenant) but the saved
    // m_tenantSparseIdx map is only populated at build time. Without this
    // load step, query-side sparse routing in SearchWithTags is a no-op on
    // any process that only Load()s the index.
    int loadedCount = 0;
    for (const auto& kv : m_tenantSpannWorkDirs)
    {
        int tenantId = kv.first;
        if (m_tenantSparseIdx.count(tenantId)) continue;
        const std::string sparsePath = kv.second + "/sparse_tags.bin";
        struct stat st{};
        if (stat(sparsePath.c_str(), &st) != 0) continue;
        auto sparseIdx = std::make_shared<SPTAG::Cache::SparseTagIndex>();
        if (!sparseIdx->Load(sparsePath))
        {
            fprintf(stderr, "[WARN] Tenant %d: failed to load sparse_tags.bin (%s)\n",
                    tenantId, sparsePath.c_str());
            continue;
        }
        m_tenantSparseIdx[tenantId] = std::move(sparseIdx);
        ++loadedCount;
    }
    if (loadedCount > 0)
    {
        fprintf(stderr, "[INFO] Loaded sparse tag indices for %d tenants\n", loadedCount);
    }
}

bool TenantIndexManager::SaveUnifiedStorage(const char* p_baseDir)
{
    std::string baseDir(p_baseDir);

    // Save tenants that are still in memory
    for (const auto& kv : m_tenantIndices)
    {
        int tenantId = kv.first;
        std::string dstTenantDir = baseDir + "/tenant_" + std::to_string(tenantId);
        if (!EnsureDir(dstTenantDir))
            return false;

        auto typeIt = m_tenantIndexTypes.find(tenantId);
        TenantIndexType indexType = (typeIt != m_tenantIndexTypes.end()) ? typeIt->second : TenantIndexType::SPANN;

        if (indexType == TenantIndexType::SPANN)
        {
            if (!kv.second->Save(dstTenantDir.c_str()))
            {
                fprintf(stderr, "[ERROR] Failed to save SPANN index for tenant %d\n", tenantId);
                return false;
            }
            fprintf(stderr, "[INFO] Tenant %d: saved full SPANN index\n", tenantId);
        }
        else
        {
            std::string indexPath = dstTenantDir + "/index";
            if (!kv.second->Save(indexPath.c_str()))
            {
                fprintf(stderr, "[ERROR] Failed to save BKT/BF index for tenant %d\n", tenantId);
                return false;
            }
            fprintf(stderr, "[INFO] Tenant %d: saved BKT/BF index\n", tenantId);
        }
    }

    // Copy tenants that were already saved-and-released during build
    // (they exist in m_tenantSpannWorkDirs but not in m_tenantIndices)
    for (const auto& kv : m_tenantSpannWorkDirs)
    {
        int tenantId = kv.first;
        if (m_tenantIndices.count(tenantId)) continue;  // Already saved above

        std::string srcDir = kv.second;
        std::string dstDir = baseDir + "/tenant_" + std::to_string(tenantId);

        if (srcDir == dstDir) {
            // Already in the right place (saved directly to output dir)
            fprintf(stderr, "[INFO] Tenant %d: already saved in place\n", tenantId);
            continue;
        }

        if (!EnsureDir(dstDir)) return false;
        if (!CopyDirRecursive(srcDir, dstDir))
        {
            fprintf(stderr, "[ERROR] Failed to copy tenant %d from %s to %s\n", tenantId, srcDir.c_str(), dstDir.c_str());
            return false;
        }
        // Update work dir to point to final location
        m_tenantSpannWorkDirs[tenantId] = dstDir;
        fprintf(stderr, "[INFO] Tenant %d: copied from build dir\n", tenantId);
    }

    int totalSaved = (int)m_tenantIndices.size();
    for (const auto& kv : m_tenantSpannWorkDirs)
        if (!m_tenantIndices.count(kv.first)) totalSaved++;
    fprintf(stderr, "[INFO] Unified storage saved: %d tenants (%d SPANN)\n",
        totalSaved,
        (int)std::count_if(m_tenantIndexTypes.begin(), m_tenantIndexTypes.end(),
            [](const auto& kv) { return kv.second == TenantIndexType::SPANN; }));

    return true;
}

bool TenantIndexManager::LoadUnifiedStorage(const char* p_baseDir)
{
    std::string baseDir(p_baseDir);
    m_sharedSpannWorkDir = baseDir + "/shared_ssd";

    for (const auto& kv : m_tenantVectorCounts)
    {
        int tenantId = kv.first;
        auto typeIt = m_tenantIndexTypes.find(tenantId);
        TenantIndexType indexType = (typeIt != m_tenantIndexTypes.end()) ? typeIt->second : TenantIndexType::SPANN;

        std::string tenantDir = baseDir + "/tenant_" + std::to_string(tenantId);

        if (indexType == TenantIndexType::SPANN)
        {
            m_tenantSpannWorkDirs[tenantId] = tenantDir;
        }
        else
        {
            // BKT / BruteForce: store index path for lazy loading
            m_tenantIndexPaths[tenantId] = tenantDir + "/index";
        }
    }

    LoadTenantSparseIndices();

    return true;
}

void TenantIndexManager::SetBuildParam(const char* p_name, const char* p_value, const char* p_section)
{
    for (auto& tenantEntry : m_tenantIndices)
    {
        tenantEntry.second->SetBuildParam(p_name, p_value, p_section);
    }
}

void TenantIndexManager::SetSearchParam(const char* p_name, const char* p_value, const char* p_section)
{
    if (p_name == nullptr || p_value == nullptr || p_section == nullptr) {
        return;
    }

    std::unique_lock<std::shared_mutex> wlock(m_tenantIndicesMutex);
    bool updated = false;
    for (auto& pendingParam : m_pendingSearchParams)
    {
        if (std::get<0>(pendingParam) == p_name && std::get<2>(pendingParam) == p_section)
        {
            std::get<1>(pendingParam) = p_value;
            updated = true;
            break;
        }
    }
    if (!updated)
    {
        m_pendingSearchParams.emplace_back(p_name, p_value, p_section);
    }

    for (auto& tenantEntry : m_tenantIndices)
    {
        tenantEntry.second->SetSearchParam(p_name, p_value, p_section);
    }
}

bool TenantIndexManager::EnsureTenantLoaded(int p_tenantId)
{
    // Fast path: shared lock check (hot cache)
    {
        std::shared_lock<std::shared_mutex> rlock(m_tenantIndicesMutex);
        if (m_tenantIndices.count(p_tenantId))
        {
            // Skip LRU update on fast path — splice is not thread-safe under shared_lock.
            // LRU order is approximate; only updated on slow path (exclusive lock).
            return true;
        }
    }

    // Slow path: exclusive lock
    std::unique_lock<std::shared_mutex> wlock(m_tenantIndicesMutex);
    // Double-check hot cache
    if (m_tenantIndices.count(p_tenantId))
    {
        auto it = m_lruMap.find(p_tenantId);
        if (it != m_lruMap.end())
            m_lruList.splice(m_lruList.end(), m_lruList, it->second);
        return true;
    }

    // Estimate loaded HeadIndex bytes from on-disk bytes and a safety factor.
    uint64_t estimatedBytes = EstimateTenantHeadIndexBytes(p_tenantId);

    // Soft-evict LRU hot tenants until we have room
    if (m_headIndexCacheLimitBytes > 0)
    {
        int retries = 0;
        while (m_loadedHeadIndexBytes + estimatedBytes > m_headIndexCacheLimitBytes
               && !m_lruList.empty())
        {
            int evictId = m_lruList.front();
            if (evictId == p_tenantId) break;
            bool evicted = UnloadTenantLocked(evictId);
            if (!evicted)
            {
                // Tenant is in use (use_count > 1). Try next LRU candidate.
                // Move to back of LRU to avoid spinning on same tenant.
                m_lruList.pop_front();
                m_lruList.push_back(evictId);
                m_lruMap[evictId] = std::prev(m_lruList.end());
                retries++;
                if (retries > (int)m_lruList.size())
                {
                    // All LRU candidates are pinned (held by caller or another thread).
                    // Break and allow the cache to temporarily exceed the limit.
                    break;
                }
            }
        }
    }

    // Best-effort RSS guard using current process RSS instead of only estimated cache bytes.
    if (m_rssHighWaterMarkBytes > 0)
    {
        uint64_t currentRSSBytes = GetCurrentProcessRSSBytes();
        int retries = 0;
        while (currentRSSBytes > 0
               && currentRSSBytes + estimatedBytes > m_rssHighWaterMarkBytes
               && !m_lruList.empty())
        {
            int evictId = m_lruList.front();
            if (evictId == p_tenantId) break;
            bool evicted = UnloadTenantLocked(evictId);
            if (!evicted)
            {
                m_lruList.pop_front();
                m_lruList.push_back(evictId);
                m_lruMap[evictId] = std::prev(m_lruList.end());
                retries++;
                if (retries > (int)m_lruList.size())
                {
                    break;
                }
            }
            else
            {
                retries = 0;
                currentRSSBytes = GetCurrentProcessRSSBytes();
            }
        }

        if (currentRSSBytes > 0 && currentRSSBytes + estimatedBytes > m_rssHighWaterMarkBytes)
        {
            fprintf(stderr,
                    "[WARN] Rejecting tenant %d load: current RSS %.2f MB + estimated HeadIndex %.2f MB exceeds RSS high-water %.2f MB\n",
                    p_tenantId,
                    currentRSSBytes / (1024.0 * 1024.0),
                    estimatedBytes / (1024.0 * 1024.0),
                    m_rssHighWaterMarkBytes / (1024.0 * 1024.0));
            return false;
        }
    }

    // Full load from disk
    auto typeIt = m_tenantIndexTypes.find(p_tenantId);
    TenantIndexType indexType = (typeIt != m_tenantIndexTypes.end()) ? typeIt->second : TenantIndexType::SPANN;
    std::string loadPath;
    if (indexType == TenantIndexType::SPANN)
    {
        auto workIt = m_tenantSpannWorkDirs.find(p_tenantId);
        if (workIt == m_tenantSpannWorkDirs.end()) return false;
        loadPath = workIt->second;
    }
    else
    {
        auto pathIt = m_tenantIndexPaths.find(p_tenantId);
        if (pathIt == m_tenantIndexPaths.end()) return false;
        loadPath = pathIt->second;
    }

    AnnIndex loadedIndex = AnnIndex::Load(loadPath.c_str());
    (void)loadedIndex;
    std::shared_ptr<AnnIndex> indexPtr;
    if (indexType == TenantIndexType::SPANN && m_storageBackend == "ROCKSDBIO" && m_useSharedDB)
    {
        indexPtr = LoadSpannWithSharedDB(loadPath, p_tenantId);
        if (indexPtr == nullptr || !indexPtr->ReadyToServe())
        {
            fprintf(stderr, "[ERROR] Failed to load tenant %d (shared-DB) from %s\n", p_tenantId, loadPath.c_str());
            return false;
        }
    }
    else
    {
        AnnIndex tmp = AnnIndex::Load(loadPath.c_str());
        if (!tmp.ReadyToServe())
        {
            fprintf(stderr, "[ERROR] Failed to load tenant %d from %s\n", p_tenantId, loadPath.c_str());
            return false;
        }
        indexPtr = std::make_shared<AnnIndex>(tmp);
    }
    if (m_rssHighWaterMarkBytes > 0)
    {
        uint64_t currentRSSBytes = GetCurrentProcessRSSBytes();
        if (currentRSSBytes > 0 && currentRSSBytes > m_rssHighWaterMarkBytes)
        {
            fprintf(stderr,
                    "[WARN] Rejecting tenant %d after load: current RSS %.2f MB exceeds RSS high-water %.2f MB\n",
                    p_tenantId,
                    currentRSSBytes / (1024.0 * 1024.0),
                    m_rssHighWaterMarkBytes / (1024.0 * 1024.0));
            return false;
        }
    }

    for (const auto& pendingParam : m_pendingSearchParams)
    {
        indexPtr->SetSearchParam(std::get<0>(pendingParam).c_str(),
                                 std::get<1>(pendingParam).c_str(),
                                 std::get<2>(pendingParam).c_str());
    }
    m_tenantIndices[p_tenantId] = indexPtr;
    m_loadedHeadIndexBytes += estimatedBytes;
    m_tenantHeadIndexAccountedBytes[p_tenantId] = estimatedBytes;
    m_lruList.push_back(p_tenantId);
    m_lruMap[p_tenantId] = std::prev(m_lruList.end());

    EnsureHeadNodeMetaLoaded(loadPath, indexPtr->GetInternalIndex());
    if (indexType == TenantIndexType::SPANN) {
        EnsureTenantPivotIndexLoaded(p_tenantId);
    }

    return true;
}

bool TenantIndexManager::EnsureTenantPivotIndexLoaded(int p_tenantId)
{
    if (m_tenantPivotLevels.count(p_tenantId) &&
        m_tenantPivotNodeCounts.count(p_tenantId) &&
        m_tenantNodePivotTags.count(p_tenantId) &&
        m_tenantTagToNodes.count(p_tenantId) &&
        m_tenantHeadNodeToNode.count(p_tenantId)) {
        return true;
    }

    auto wdIt = m_tenantSpannWorkDirs.find(p_tenantId);
    if (wdIt == m_tenantSpannWorkDirs.end()) return false;

    int pivotLevel = -1;
    int nodeCount = 0;
    std::vector<std::vector<uint32_t>> nodePivotTags;
    std::unordered_map<uint32_t, std::vector<int>> tagToNodes;
    std::vector<int> headNodeToNode;
    if (!LoadHeadNodeRoutingIndexFile(wdIt->second,
                                      pivotLevel,
                                      nodeCount,
                                      nodePivotTags,
                                      tagToNodes,
                                      headNodeToNode)) {
        return false;
    }

    m_tenantPivotLevels[p_tenantId] = pivotLevel;
    m_tenantPivotNodeCounts[p_tenantId] = nodeCount;
    m_tenantNodePivotTags[p_tenantId] = std::move(nodePivotTags);
    m_tenantTagToNodes[p_tenantId] = std::move(tagToNodes);
    m_tenantHeadNodeToNode[p_tenantId] = std::move(headNodeToNode);
    return true;
}

void TenantIndexManager::InitCache()
{
    SPTAG::Cache::HeadIndexCache::Config cfg;
    cfg.capacity_bytes = m_headIndexCacheLimitBytes;
    cfg.ttl = std::chrono::seconds(600);
    cfg.load_timeout = std::chrono::milliseconds(30000);
    m_headCache = std::make_unique<SPTAG::Cache::HeadIndexCache>(cfg);
}

void TenantIndexManager::SetHeadIndexCacheLimit(uint64_t p_bytesLimit)
{
    m_headIndexCacheLimitBytes = p_bytesLimit;
    if (m_headCache) {
        m_headCache->SetCapacity(p_bytesLimit);
    }
    fprintf(stderr, "[INFO] HeadIndex cache limit set to %lu bytes (%.1f MB)\n",
            (unsigned long)p_bytesLimit, p_bytesLimit / (1024.0 * 1024.0));
}

void TenantIndexManager::SetHeadIndexCacheSafetyFactor(double p_factor)
{
    if (p_factor < 1.0) p_factor = 1.0;
    if (p_factor > 8.0) p_factor = 8.0;
    m_headIndexCacheSafetyFactor = p_factor;
    fprintf(stderr, "[INFO] HeadIndex cache safety factor set to %.3f\n", m_headIndexCacheSafetyFactor);
}

double TenantIndexManager::GetHeadIndexCacheSafetyFactor() const
{
    return m_headIndexCacheSafetyFactor;
}

uint64_t TenantIndexManager::GetHeadIndexCacheUsage() const
{
    std::shared_lock<std::shared_mutex> rlock(m_tenantIndicesMutex);
    return m_loadedHeadIndexBytes;
}

uint64_t TenantIndexManager::GetCurrentRSSBytes() const
{
    return GetCurrentProcessRSSBytes();
}

void TenantIndexManager::SetRSSHighWaterMark(uint64_t p_bytesLimit)
{
    m_rssHighWaterMarkBytes = p_bytesLimit;
    fprintf(stderr, "[INFO] RSS high-water mark set to %lu bytes (%.1f MB)\n",
            (unsigned long)p_bytesLimit, p_bytesLimit / (1024.0 * 1024.0));
}

uint64_t TenantIndexManager::GetRSSHighWaterMark() const
{
    return m_rssHighWaterMarkBytes;
}

uint64_t TenantIndexManager::GetLastPostingReadCount() const
{
    return SPTAG::VectorIndex::GetThreadLocalPostingScanStats().m_readPostings;
}

uint64_t TenantIndexManager::GetLastPostingMatchCount() const
{
    return SPTAG::VectorIndex::GetThreadLocalPostingScanStats().m_matchedPostings;
}

uint64_t TenantIndexManager::GetLastPostingFP() const
{
    return SPTAG::VectorIndex::GetThreadLocalPostingScanStats().FalsePositivePostings();
}

bool TenantIndexManager::UnloadTenant(int p_tenantId)
{
    std::unique_lock<std::shared_mutex> wlock(m_tenantIndicesMutex);
    return UnloadTenantLocked(p_tenantId);
}

bool TenantIndexManager::UnloadTenantLocked(int p_tenantId)
{
    // Must be called under exclusive lock (m_tenantIndicesMutex)
    auto it = m_tenantIndices.find(p_tenantId);
    if (it == m_tenantIndices.end()) return false;

    // SAFETY: If another thread holds a shared_ptr to this index (e.g. BatchSearch
    // in progress), skip eviction. The search thread's shared_ptr keeps the object alive.
    // use_count > 1 means: 1 (in map) + N (held by search threads).
    if (it->second.use_count() > 1)
    {
        return false;  // Skip: tenant in use
    }

    uint64_t freedBytes = 0;
    auto accountedIt = m_tenantHeadIndexAccountedBytes.find(p_tenantId);
    if (accountedIt != m_tenantHeadIndexAccountedBytes.end()) {
        freedBytes = accountedIt->second;
        m_tenantHeadIndexAccountedBytes.erase(accountedIt);
    } else {
        freedBytes = EstimateTenantHeadIndexBytes(p_tenantId);
    }

    // With SharedAIOPool: destruction only does close(fd) + free memory (~1ms).
    // AIO contexts are shared and never destroyed.

    // DisableCheckpoint=true (from ini/default): ShutDown never writes back.
    it->second.reset();
    m_tenantIndices.erase(it);

    // Update cache accounting
    if (m_loadedHeadIndexBytes >= freedBytes)
        m_loadedHeadIndexBytes -= freedBytes;
    else
        m_loadedHeadIndexBytes = 0;

    // Remove from LRU
    auto lruIt = m_lruMap.find(p_tenantId);
    if (lruIt != m_lruMap.end())
    {
        m_lruList.erase(lruIt->second);
        m_lruMap.erase(lruIt);
    }

    // Drop OS page cache for this tenant's HeadIndex files.
    // This ensures next load hits real disk IO, not page cache.
    if (m_dropPageCacheOnEvict)
    {
        std::string hiDir;
        auto wdIt = m_tenantSpannWorkDirs.find(p_tenantId);
        if (wdIt != m_tenantSpannWorkDirs.end())
            hiDir = wdIt->second + "/HeadIndex";
        if (!hiDir.empty())
        {
            DIR* dir = opendir(hiDir.c_str());
            if (dir) {
                struct dirent* ent;
                while ((ent = readdir(dir)) != nullptr) {
                    if (ent->d_name[0] == '.') continue;
                    std::string fp = hiDir + "/" + ent->d_name;
                    int fd = open(fp.c_str(), O_RDONLY);
                    if (fd >= 0) {
                        struct stat st;
                        fstat(fd, &st);
                        posix_fadvise(fd, 0, st.st_size, POSIX_FADV_DONTNEED);
                        close(fd);
                    }
                }
                closedir(dir);
            }
        }
    }

    return true;
}

void TenantIndexManager::TouchLRU(int p_tenantId)
{
    // No-op: S3-FIFO handles promotion internally via freq counter
}

void TenantIndexManager::EvictIfNeeded()
{
    // No-op: HeadIndexCache handles eviction internally
}

// ============================================================================
// ACL / Tag Filtered Search — Two-Level Signature Implementation
// ============================================================================

bool TenantIndexManager::BuildSignatures(int p_tenantId, ByteArray p_tags, int p_numVectors, int p_numTagsPerVec)
{
    const uint32_t* p_tagsPtr = reinterpret_cast<const uint32_t*>(p_tags.Data());

    auto wdIt = m_tenantSpannWorkDirs.find(p_tenantId);
    if (wdIt == m_tenantSpannWorkDirs.end()) return false;
    std::string workDir = wdIt->second;

    // Read head count from HeadIndex vectors.bin
    auto hcIt = m_tenantHeadCounts.find(p_tenantId);
    int numHeads = (hcIt != m_tenantHeadCounts.end()) ? hcIt->second : 0;
    if (numHeads <= 0) {
        std::string vecPath = workDir + "/HeadIndex/vectors.bin";
        FILE* vf = fopen(vecPath.c_str(), "rb");
        if (vf) {
            int32_t rows = 0;
            if (fread(&rows, sizeof(int32_t), 1, vf) == 1) numHeads = rows;
            fclose(vf);
        }
    }
    if (numHeads <= 0) return false;

    // ── Read real posting→vector assignment from SPANN on-disk data ──
    // Files: ssdinfo (posting sizes), ssdmapping (block addresses), ssdmapping_postings (block data)
    // Posting format per vector: [VID(4B) | Version(1B) | Tags(N*4B) | VectorData(dim*sizeof(T))]
    const int PAGE_SIZE = 4096;
    const int TAG_BYTES = p_numTagsPerVec * (int)sizeof(uint32_t);
    const int META_SIZE = sizeof(int32_t) + sizeof(uint8_t) + TAG_BYTES;
    const int VEC_INFO_SIZE = m_inputVectorSize + META_SIZE;

    // 1. Read ssdinfo: header (rows, cols=1) then rows × int32 posting sizes
    std::string ssdinfoPath = workDir + "/ssdinfo";
    FILE* infoF = fopen(ssdinfoPath.c_str(), "rb");
    if (!infoF) {
        fprintf(stderr, "[ERROR] Cannot open %s\n", ssdinfoPath.c_str());
        return false;
    }
    int32_t infoHeader[2];
    if (fread(infoHeader, sizeof(int32_t), 2, infoF) != 2) { fclose(infoF); return false; }
    int numPostings = infoHeader[0];
    std::vector<int32_t> postingSizes(numPostings);
    if ((int)fread(postingSizes.data(), sizeof(int32_t), numPostings, infoF) != numPostings) {
        fclose(infoF); return false;
    }
    fclose(infoF);

    // 2. Read ssdmapping: header (rows, cols) then rows × cols × int64 block addresses
    //    addrs[pid][0] = data size in bytes, addrs[pid][1..] = block addresses
    std::string mappingPath = workDir + "/ssdmapping";
    FILE* mapF = fopen(mappingPath.c_str(), "rb");
    if (!mapF) {
        fprintf(stderr, "[ERROR] Cannot open %s\n", mappingPath.c_str());
        return false;
    }
    int32_t mapHeader[2];
    if (fread(mapHeader, sizeof(int32_t), 2, mapF) != 2) { fclose(mapF); return false; }
    int mapRows = mapHeader[0], mapCols = mapHeader[1];
    std::vector<int64_t> addrFlat((size_t)mapRows * mapCols);
    if ((int)fread(addrFlat.data(), sizeof(int64_t), (size_t)mapRows * mapCols, mapF)
        != mapRows * mapCols) {
        fclose(mapF); return false;
    }
    fclose(mapF);

    // 3. Open posting data file
    std::string postingPath = workDir + "/ssdmapping_postings";
    FILE* postF = fopen(postingPath.c_str(), "rb");
    if (!postF) {
        fprintf(stderr, "[ERROR] Cannot open %s\n", postingPath.c_str());
        return false;
    }

    // 4. For each posting, read its blocks and extract vector IDs
    std::vector<std::vector<uint32_t>> posting_tags(numHeads);
    std::vector<SPTAG::Cache::HierarchicalPostingMask> posting_hier_masks(numHeads);
    int totalAssignments = 0;
    std::vector<uint8_t> blockBuf(PAGE_SIZE);

    for (int pid = 0; pid < std::min(numPostings, numHeads); pid++) {
        int nVecs = postingSizes[pid];
        if (nVecs <= 0) continue;

        // Gather block addresses (skip index 0 which is data size)
        int64_t* rowAddrs = addrFlat.data() + (int64_t)pid * mapCols;
        // rowAddrs[0] = data size, rowAddrs[1..] = block addresses

        // Read blocks into contiguous buffer
        int dataSize = nVecs * VEC_INFO_SIZE;
        std::vector<uint8_t> raw;
        raw.reserve(dataSize + PAGE_SIZE);
        for (int b = 1; b < mapCols; b++) {
            int64_t blkAddr = rowAddrs[b];
            if (blkAddr < 0) break;  // -1 marks end of block list; 0 is valid
            fseek(postF, blkAddr * PAGE_SIZE, SEEK_SET);
            raw.resize(raw.size() + PAGE_SIZE);
            size_t readBytes = fread(raw.data() + raw.size() - PAGE_SIZE, 1, PAGE_SIZE, postF);
            (void)readBytes;
        }

        if ((int)raw.size() < dataSize) continue;

        // Extract VIDs and map to tags
        for (int j = 0; j < nVecs; j++) {
            int offset = j * VEC_INFO_SIZE;
            int32_t vid;
            memcpy(&vid, raw.data() + offset, sizeof(int32_t));
            if (vid < 0 || vid >= p_numVectors) continue;
            // Insert ALL tags for this vector into this posting's Bloom AND hierarchical mask
            for (int t = 0; t < p_numTagsPerVec; t++) {
                uint32_t tag = p_tagsPtr[vid * p_numTagsPerVec + t];
                posting_tags[pid].push_back(tag);
                // Also insert into hierarchical mask at level t
                posting_hier_masks[pid].Insert(t, tag);
            }
            totalAssignments++;
        }
    }
    fclose(postF);

    auto sigs = std::make_shared<SPTAG::Cache::TenantBitmaskPS>();
    sigs->Build(numHeads, posting_tags);

    std::string sigPath = workDir + "/signatures_bitmask.bin";
    sigs->Save(sigPath);

    std::unordered_map<uint32_t, int> tagVectorCounts;
    tagVectorCounts.reserve(static_cast<size_t>(p_numVectors) * p_numTagsPerVec);
    for (int vid = 0; vid < p_numVectors; ++vid) {
        std::unordered_set<uint32_t> seenTags;
        for (int t = 0; t < p_numTagsPerVec; ++t) {
            uint32_t tag = p_tagsPtr[vid * p_numTagsPerVec + t];
            if (seenTags.insert(tag).second) {
                ++tagVectorCounts[tag];
            }
        }
    }

    std::unordered_map<uint32_t, int> tagPostingCounts;
    tagPostingCounts.reserve(tagVectorCounts.size());
    for (int pid = 0; pid < numHeads; ++pid) {
        std::unordered_set<uint32_t> seenTags;
        for (uint32_t tag : posting_tags[pid]) {
            if (seenTags.insert(tag).second) {
                ++tagPostingCounts[tag];
            }
        }
    }

    auto& routeStats = m_tenantTagRoutingStats[p_tenantId];
    routeStats.clear();
    routeStats.reserve(tagVectorCounts.size());
    for (const auto& [tag, vectorCount] : tagVectorCounts) {
        routeStats[tag] = TagRoutingStats{vectorCount, tagPostingCounts[tag]};
    }

    // Sparse-path single knob:
    //   SPTAG_SPARSE_MAX_POSTINGS = N (default 1024)
    // A tag is materialized into sparse_tags.bin iff it appears in ≤ N postings.
    // At query time, materialized tags ALWAYS route through the sparse path
    // (no second-stage union-size gate) - this is the single fixed threshold.
    int kSparseIndexBuildMaxPostings = 1024;
    if (const char* env = std::getenv("SPTAG_SPARSE_MAX_POSTINGS")) {
        int parsed = 0;
        if (SPTAG::Helper::Convert::ConvertStringTo<int>(env, parsed) && parsed > 0) {
            kSparseIndexBuildMaxPostings = parsed;
        }
    }
    auto sparseIdx = std::make_shared<SPTAG::Cache::SparseTagIndex>();
    sparseIdx->Build(numHeads, posting_tags, tagPostingCounts, kSparseIndexBuildMaxPostings);

    std::string sparsePath = workDir + "/sparse_tags.bin";
    sparseIdx->Save(sparsePath);
    m_tenantSparseIdx[p_tenantId] = sparseIdx;

    constexpr int kPivotEstimatorDefaultMaxNodes = 0;
    constexpr double kPivotEstimatorDefaultRecallTarget = 0.99;
    constexpr double kPivotEstimatorDefaultLambdaRecall = 10.0;
    constexpr double kPivotEstimatorDefaultEstimatedRecall = 1.0;

    PivotEstimatorComputation pivotComputation;
    const PivotEstimatorCandidate* pivotCandidate = nullptr;
    if (BuildPivotEstimatorComputation(p_tagsPtr,
                                       p_numVectors,
                                       p_numTagsPerVec,
                                       kPivotEstimatorDefaultMaxNodes,
                                       kPivotEstimatorDefaultRecallTarget,
                                       kPivotEstimatorDefaultLambdaRecall,
                                       kPivotEstimatorDefaultEstimatedRecall,
                                       std::string(),
                                       pivotComputation)) {
        pivotCandidate = FindBestPivotEstimatorCandidate(pivotComputation.candidates);
    }

    if (pivotCandidate != nullptr) {
        m_tenantPivotLevels[p_tenantId] = pivotCandidate->pivotLevel;
        m_tenantPivotNodeCounts[p_tenantId] = pivotCandidate->nodeCount;
        m_tenantNodePivotTags[p_tenantId] = pivotCandidate->nodePivotTags;
        BuildTagToNodeIndexForCandidate(*pivotCandidate,
                                        pivotComputation.levelData,
                                        m_tenantTagToNodes[p_tenantId]);
    } else {
        m_tenantPivotLevels.erase(p_tenantId);
        m_tenantPivotNodeCounts.erase(p_tenantId);
        m_tenantNodePivotTags.erase(p_tenantId);
        m_tenantTagToNodes.erase(p_tenantId);
        m_tenantHeadNodeToNode.erase(p_tenantId);
    }

    // Build head tag table: VIDs NOT found in any posting are head vectors.
    // They need tag metadata for filtered search since inline tag filter
    // can't check them (they're not in posting data).
    std::unordered_set<int> postingVIDs;
    // Re-derive from posting_tags: impossible to get VIDs from tags alone.
    // Instead, use the totalAssignments count: if a VID was found in postings,
    // it contributed to posting_tags. Just re-scan the posting file.
    {
        FILE* pf2 = fopen(postingPath.c_str(), "rb");
        if (pf2) {
            for (int pid = 0; pid < std::min(numPostings, numHeads); pid++) {
                int nVecs = postingSizes[pid];
                if (nVecs <= 0) continue;
                int64_t* rowAddrs2 = addrFlat.data() + (int64_t)pid * mapCols;
                std::vector<uint8_t> raw2;
                raw2.reserve(nVecs * VEC_INFO_SIZE + PAGE_SIZE);
                for (int b = 1; b < mapCols; b++) {
                    int64_t blkAddr = rowAddrs2[b];
                    if (blkAddr < 0) break;
                    fseek(pf2, blkAddr * PAGE_SIZE, SEEK_SET);
                    raw2.resize(raw2.size() + PAGE_SIZE);
                    size_t r = fread(raw2.data() + raw2.size() - PAGE_SIZE, 1, PAGE_SIZE, pf2);
                    (void)r;
                }
                for (int j = 0; j < nVecs && j * VEC_INFO_SIZE + 4 <= (int)raw2.size(); j++) {
                    int32_t vid;
                    memcpy(&vid, raw2.data() + j * VEC_INFO_SIZE, sizeof(int32_t));
                    if (vid >= 0 && vid < p_numVectors) postingVIDs.insert(vid);
                }
            }
            fclose(pf2);
        }
    }

    // Store per-head-node metadata on the inner head index (if loaded).
    // First ensure the tenant is loaded.
    EnsureTenantLoaded(p_tenantId);
    std::shared_ptr<AnnIndex> idxPtr;
    {
        std::shared_lock<std::shared_mutex> rlock(m_tenantIndicesMutex);
        auto it = m_tenantIndices.find(p_tenantId);
        if (it != m_tenantIndices.end()) idxPtr = it->second;
    }
    int headTagCount = 0;
    if (idxPtr) {
        auto internalIdx = idxPtr->GetInternalIndex();
        auto memoryIndex = GetMemoryIndexForInternal(internalIdx);
        auto* spannInternalIdx = dynamic_cast<SPTAG::SPANN::Index<float>*>(internalIdx.get());
        if (memoryIndex != nullptr && spannInternalIdx != nullptr) {
            const SizeType numHeadSamples = memoryIndex->GetNumSamples();
            memoryIndex->InitializeHeadNodeMeta(numHeadSamples);
            for (SizeType hid = 0; hid < numHeadSamples; ++hid) {
                SizeType globalVID = spannInternalIdx->GetGlobalVID(hid);
                memoryIndex->SetHeadNodeGlobalVID(hid, globalVID);
                if (hid < sigs->num_postings) {
                    memoryIndex->SetHeadNodePS(hid, sigs->ps[hid]);
                }
                // Set hierarchical mask for this head
                if (hid < (SizeType)posting_hier_masks.size()) {
                    memoryIndex->SetHeadNodeHierMask(hid, posting_hier_masks[hid]);
                }

                if (globalVID == SPTAG::MaxSize || globalVID >= static_cast<SizeType>(p_numVectors)) {
                    continue;
                }
                if (postingVIDs.count(static_cast<int>(globalVID)) != 0) {
                    continue;
                }

                // This is a head-only vector (not in any posting)
                memoryIndex->SetHeadNodeHeadOnly(hid, true);
                // Build its hierarchical mask from its tags
                SPTAG::Cache::HierarchicalPostingMask headMask;
                headMask.Clear();
                for (int t = 0; t < p_numTagsPerVec; ++t) {
                    uint32_t tag = p_tagsPtr[static_cast<size_t>(globalVID) * static_cast<size_t>(p_numTagsPerVec) + static_cast<size_t>(t)];
                    headMask.Insert(t, tag);
                }
                memoryIndex->SetHeadNodeHierMask(hid, headMask);
                headTagCount++;
            }

            if (pivotCandidate != nullptr) {
                std::vector<int> headNodeToNode;
                BuildHeadNodeToNodeIndexForCandidate(*pivotCandidate,
                                                     p_tagsPtr,
                                                     p_numVectors,
                                                     p_numTagsPerVec,
                                                     memoryIndex,
                                                     spannInternalIdx,
                                                     headNodeToNode);
                m_tenantHeadNodeToNode[p_tenantId] = headNodeToNode;

                // Populate bundleNodeId for each head
                for (SizeType hid = 0; hid < numHeadSamples; ++hid) {
                    int16_t nid = (hid < (SizeType)headNodeToNode.size() && headNodeToNode[hid] >= 0)
                                  ? (int16_t)headNodeToNode[hid] : (int16_t)-1;
                    memoryIndex->SetHeadNodeBundleNodeId(hid, nid);
                }

                SaveHeadNodeRoutingIndexFile(workDir,
                                             pivotCandidate->pivotLevel,
                                             pivotCandidate->nodePivotTags,
                                             m_tenantTagToNodes[p_tenantId],
                                             headNodeToNode);
            }
            // Save meta file AFTER bundleNodeId is populated
            SaveHeadNodeMetaFile(workDir, memoryIndex);
        }
    }

    if (pivotCandidate != nullptr) {
        fprintf(stderr,
                "[INFO] Tenant %d: pivot estimator selected level=%d nodes=%d\n",
                p_tenantId,
                pivotCandidate->pivotLevel,
                pivotCandidate->nodeCount);
    }

    fprintf(stderr, "[INFO] Tenant %d: built PS + sparse index + %d head tags (%d postings, %d assignments, sparse_max_postings=%d)\n",
            p_tenantId, headTagCount, numHeads, totalAssignments, kSparseIndexBuildMaxPostings);
    return true;
}

std::shared_ptr<QueryResult> TenantIndexManager::SearchWithACL(
    ByteArray p_queryVector, int p_tenantId, int p_resultNum,
    ByteArray p_queryTags, int p_numTags)
{
    static const bool s_wrapperTime = (std::getenv("SPTAG_LOG_WRAPPER_TIME") != nullptr);
    auto _wrTotal0 = s_wrapperTime ? std::chrono::high_resolution_clock::now()
                                   : std::chrono::high_resolution_clock::time_point{};
    const uint32_t* queryTagsPtr = reinterpret_cast<const uint32_t*>(p_queryTags.Data());
    auto _ck_a = s_wrapperTime ? std::chrono::high_resolution_clock::now()
                               : std::chrono::high_resolution_clock::time_point{};
    if (!EnsureTenantLoaded(p_tenantId)) return nullptr;
    auto _ck_b = s_wrapperTime ? std::chrono::high_resolution_clock::now()
                               : std::chrono::high_resolution_clock::time_point{};
    auto wdIt = m_tenantSpannWorkDirs.find(p_tenantId);
    if (wdIt == m_tenantSpannWorkDirs.end()) return nullptr;
    const std::string& workDir = wdIt->second;

    std::shared_ptr<AnnIndex> indexPtr;
    {
        std::shared_lock<std::shared_mutex> rlock(m_tenantIndicesMutex);
        auto it = m_tenantIndices.find(p_tenantId);
        if (it == m_tenantIndices.end()) return nullptr;
        indexPtr = it->second;
    }
    auto _ck_c = s_wrapperTime ? std::chrono::high_resolution_clock::now()
                               : std::chrono::high_resolution_clock::time_point{};

    auto internalIdx = indexPtr->GetInternalIndex();
    auto _ck_d = s_wrapperTime ? std::chrono::high_resolution_clock::now()
                               : std::chrono::high_resolution_clock::time_point{};
    bool forceDenseTagSearch = false;
    float filteredSearchNprobeSafety = 1.0f;
    if (internalIdx != nullptr) {
        const std::string forceDenseParam = internalIdx->GetParameter("ForceDenseTagSearch", "BuildSSDIndex");
        if (!forceDenseParam.empty()) {
            SPTAG::Helper::Convert::ConvertStringTo<bool>(forceDenseParam.c_str(), forceDenseTagSearch);
        }

        const std::string filteredSearchNprobeSafetyParam = internalIdx->GetParameter("FilteredSearchNprobeSafety", "BuildSSDIndex");
        if (!filteredSearchNprobeSafetyParam.empty()) {
            float parsedFilteredSearchNprobeSafety = 1.0f;
            if (SPTAG::Helper::Convert::ConvertStringTo<float>(filteredSearchNprobeSafetyParam.c_str(), parsedFilteredSearchNprobeSafety)
                && parsedFilteredSearchNprobeSafety > 0.0f) {
                filteredSearchNprobeSafety = parsedFilteredSearchNprobeSafety;
            }
        }
    }

    auto _ck_afterIdx = s_wrapperTime ? std::chrono::high_resolution_clock::now()
                                      : std::chrono::high_resolution_clock::time_point{};
    if (s_wrapperTime) {
        double t_a = std::chrono::duration<double, std::milli>(_ck_a - _wrTotal0).count();
        double t_b = std::chrono::duration<double, std::milli>(_ck_b - _ck_a).count();
        double t_c = std::chrono::duration<double, std::milli>(_ck_c - _ck_b).count();
        double t_d = std::chrono::duration<double, std::milli>(_ck_d - _ck_c).count();
        double t_e = std::chrono::duration<double, std::milli>(_ck_afterIdx - _ck_d).count();
        fprintf(stdout,
            "[1] WrapperEntry: tag0=%u  preEnsure=%.4f ensureTenant=%.4f tenantLookups=%.4f getInternal=%.4f getParams=%.4f\n",
            (p_numTags > 0 && queryTagsPtr) ? queryTagsPtr[0] : 0u,
            t_a, t_b, t_c, t_d, t_e);
        fflush(stdout);
    }
    const auto tagStatsIt = m_tenantTagRoutingStats.find(p_tenantId);
    const auto* tagStats = (tagStatsIt != m_tenantTagRoutingStats.end()) ? &tagStatsIt->second : nullptr;

    auto _ck_routingStart = s_wrapperTime ? std::chrono::high_resolution_clock::now()
                                          : std::chrono::high_resolution_clock::time_point{};

    const auto tagToNodesIt = m_tenantTagToNodes.find(p_tenantId);
    const auto headNodeToNodeIt = m_tenantHeadNodeToNode.find(p_tenantId);
    const std::vector<int>* headNodeToNode = (headNodeToNodeIt != m_tenantHeadNodeToNode.end())
        ? &headNodeToNodeIt->second
        : nullptr;
    std::vector<int> routedNodes;
    std::vector<uint8_t> allowedNodeMask;
    bool hasRoutingNodeFilter = false;
    if (p_numTags > 0 &&
        tagToNodesIt != m_tenantTagToNodes.end() &&
        headNodeToNode != nullptr &&
        !headNodeToNode->empty() &&
        TryCollectRoutingNodesForQuery(tagToNodesIt->second, queryTagsPtr, p_numTags, routedNodes)) {
        int nodeCount = 0;
        auto nodeCountIt = m_tenantPivotNodeCounts.find(p_tenantId);
        if (nodeCountIt != m_tenantPivotNodeCounts.end()) {
            nodeCount = nodeCountIt->second;
        }
        if (nodeCount <= 0) {
            for (int nodeId : routedNodes) {
                nodeCount = std::max(nodeCount, nodeId + 1);
            }
        }

        if (nodeCount > 0) {
            allowedNodeMask.assign(static_cast<size_t>(nodeCount), 0);
            for (int nodeId : routedNodes) {
                if (nodeId >= 0 && nodeId < nodeCount) {
                    allowedNodeMask[static_cast<size_t>(nodeId)] = 1;
                    hasRoutingNodeFilter = true;
                }
            }
        }
    }

    // Check if ALL query tags are sparse → use brute-force path
    auto _ck_routing = s_wrapperTime ? std::chrono::high_resolution_clock::now()
                                     : std::chrono::high_resolution_clock::time_point{};
    if (s_wrapperTime) {
        double t_idxParam = std::chrono::duration<double, std::milli>(_ck_afterIdx - _wrTotal0).count();
        double t_tagStats = std::chrono::duration<double, std::milli>(_ck_routingStart - _ck_afterIdx).count();
        double t_routeNodes = std::chrono::duration<double, std::milli>(_ck_routing - _ck_routingStart).count();
        fprintf(stdout,
            "[1] WrapperRouting: tag0=%u rn=%zu  idxParam=%.3f tagStats=%.3f routeNodes=%.3f\n",
            (p_numTags > 0 && queryTagsPtr) ? queryTagsPtr[0] : 0u,
            (size_t)routedNodes.size(), t_idxParam, t_tagStats, t_routeNodes);
        fflush(stdout);
    }
    static const bool s_disableSparsePath = []() {
        const char* v = std::getenv("SPTAG_DISABLE_SPARSE_PATH");
        return v && (v[0] == '1' || v[0] == 't' || v[0] == 'T');
    }();
    auto sparseIt = m_tenantSparseIdx.find(p_tenantId);
    if (!s_disableSparsePath && !forceDenseTagSearch && sparseIt != m_tenantSparseIdx.end() && p_numTags > 0) {
        auto& sparseIdx = sparseIt->second;
        bool hasDirectPostingListsForAllTags = true;
        // Collect posting IDs for all query tags
        std::unordered_set<int> bfPostings;
        for (int i = 0; i < p_numTags; i++) {
            auto* pids = sparseIdx->GetPostings(queryTagsPtr[i]);
            if (!pids) {
                hasDirectPostingListsForAllTags = false;
                break;
            }
            bfPostings.insert(pids->begin(), pids->end());
        }

        // Sparse fast-path policy: build-time `kSparseIndexBuildMaxPostings` is
        // the single source of truth - if a tag's posting list was materialized
        // at build, query-time always routes through the sparse path. No
        // second-stage union-size cap: that would create a window where the
        // sidecar was paid for but the path silently fell back to ANN.
        if (hasDirectPostingListsForAllTags && !bfPostings.empty()) {
            SPTAG::VectorIndex::ThreadLocalSearchContext searchContext;
            searchContext.m_queryTags.assign(queryTagsPtr, queryTagsPtr + p_numTags);
            searchContext.m_directPostingIDs.assign(bfPostings.begin(), bfPostings.end());
            SPTAG::VectorIndex::ThreadLocalSearchContextGuard searchContextGuard(std::move(searchContext));

            auto result = indexPtr->Search(p_queryVector, p_resultNum);
            return result;
        }
    }

    // Dense tag path: SPANN graph + bitmask PS + inline filter
    // Build query bitmask from requested tags
    auto _ck_sparse = s_wrapperTime ? std::chrono::high_resolution_clock::now()
                                    : std::chrono::high_resolution_clock::time_point{};
    SPTAG::Cache::PostingBitmask queryMask;
    queryMask.Clear();
    for (int i = 0; i < p_numTags; i++) {
        queryMask.Insert(queryTagsPtr[i]);
    }
    auto memoryIndex = GetMemoryIndexForInternal(internalIdx);
    EnsureHeadNodeMetaLoaded(workDir, internalIdx);
    auto _ck_qmask = s_wrapperTime ? std::chrono::high_resolution_clock::now()
                                   : std::chrono::high_resolution_clock::time_point{};
    SPTAG::VectorIndex::ThreadLocalSearchContext searchContext;
    if (p_numTags > 0 && queryTagsPtr != nullptr) {
        searchContext.m_queryTags.assign(queryTagsPtr, queryTagsPtr + p_numTags);
    }
    if (internalIdx) {
            // Dense path uses the pivot routing index as a coarse posting filter.
            // If routing metadata is missing or the node intersection collapses to
            // empty, fall back to the original exact-filter-only behavior.
            if (hasRoutingNodeFilter && headNodeToNode != nullptr) {
                searchContext.m_postingFilter =
                    [allowedNodeMask = std::move(allowedNodeMask), headNodeToNode](int localHid) {
                        if (localHid < 0 || localHid >= static_cast<int>(headNodeToNode->size())) {
                            return true;
                        }

                        int nodeId = (*headNodeToNode)[static_cast<size_t>(localHid)];
                        if (nodeId < 0 || nodeId >= static_cast<int>(allowedNodeMask.size())) {
                            return true;
                        }

                        return allowedNodeMask[static_cast<size_t>(nodeId)] != 0;
                    };
                searchContext.m_searchHeadBundleNodes = routedNodes;
            }

        auto vcIt2 = m_tenantVectorCounts.find(p_tenantId);
        int tenantSize2 = (vcIt2 != m_tenantVectorCounts.end()) ? vcIt2->second : 1;
        float vectorSel = EstimateQueryVectorSelectivity(tenantSize2, tagStats, queryTagsPtr, p_numTags);
        vectorSel = std::clamp(vectorSel / std::max(1.0f, filteredSearchNprobeSafety), 1e-6f, 1.0f);
        searchContext.m_filterSelectivity = vectorSel;

        if (memoryIndex != nullptr && memoryIndex->HasHeadNodeMeta() && tenantSize2 > 0 && searchContext.m_filterSelectivity >= 1.0f) {
            int passCount = 0;
            SizeType totalHeads = memoryIndex->GetHeadNodeMetaSampleCount();
            for (SizeType pid = 0; pid < totalHeads; ++pid) {
                if (memoryIndex->HeadNodePSMayIntersect(pid, queryMask)) passCount++;
            }
            // FIX: divide by total head count (same units), not tenantSize.
            // passCount/tenantSize mixed head-count and vector-count units, so the
            // fallback always reported pathologically small selectivity (~1/avgPosting).
            // passCount/totalHeads is the fraction of heads that *may* match — a
            // sane proxy for vector selectivity when tag distribution across
            // postings is roughly uniform.
            float fallbackVectorSel = (totalHeads > 0)
                ? static_cast<float>(passCount) / static_cast<float>(totalHeads)
                : 1.0f;
            searchContext.m_filterSelectivity = std::clamp(fallbackVectorSel, 1e-6f, 1.0f);
        }
    }
    SPTAG::VectorIndex::ThreadLocalSearchContextGuard searchContextGuard(std::move(searchContext));
    auto _ck_denseEnd = s_wrapperTime ? std::chrono::high_resolution_clock::now()
                                      : std::chrono::high_resolution_clock::time_point{};

    auto _wrSearch0 = s_wrapperTime ? std::chrono::high_resolution_clock::now()
                                    : std::chrono::high_resolution_clock::time_point{};
    if (s_wrapperTime) {
        double t_routing = std::chrono::duration<double, std::milli>(_ck_routing - _wrTotal0).count();
        double t_sparse  = std::chrono::duration<double, std::milli>(_ck_sparse  - _ck_routing).count();
        double t_qmask   = std::chrono::duration<double, std::milli>(_ck_qmask   - _ck_sparse ).count();
        double t_dense   = std::chrono::duration<double, std::milli>(_ck_denseEnd- _ck_qmask  ).count();
        fprintf(stdout,
            "[1] WrapperPre: tag0=%u rn=%zu  routing=%.3f sparse=%.3f qmask=%.3f denseSetup=%.3f\n",
            (p_numTags > 0 && queryTagsPtr) ? queryTagsPtr[0] : 0u,
            (size_t)routedNodes.size(), t_routing, t_sparse, t_qmask, t_dense);
        fflush(stdout);
    }
    auto _wrT0 = s_wrapperTime ? std::chrono::high_resolution_clock::now()
                               : std::chrono::high_resolution_clock::time_point{};
    auto result = indexPtr->Search(p_queryVector, p_resultNum);
    if (s_wrapperTime) {
        auto _wrT1 = std::chrono::high_resolution_clock::now();
        double searchMs = std::chrono::duration<double, std::milli>(_wrT1 - _wrSearch0).count();
        double totalMs  = std::chrono::duration<double, std::milli>(_wrT1 - _wrTotal0).count();
        double preMs    = std::chrono::duration<double, std::milli>(_wrSearch0 - _wrTotal0).count();
        fprintf(stdout,
            "[1] WrapperCall: tag0=%u nTags=%d routedNodes=%zu preMs=%.3f searchMs=%.3f totalMs=%.3f\n",
            (p_numTags > 0 && queryTagsPtr) ? queryTagsPtr[0] : 0u,
            p_numTags, (size_t)routedNodes.size(), preMs, searchMs, totalMs);
        fflush(stdout);
    }

    return result;
}

bool TenantIndexManager::EnsureTenantCached(int p_tenantId)
{
    return EnsureTenantLoaded(p_tenantId);
}

TenantIndexType TenantIndexManager::ChooseIndexType(int vectorCount) const
{
    // All tenants use SPANN: HeadIndex in memory, postings on SSD
    (void)vectorCount;
    return TenantIndexType::SPANN;
}

// --- String tenant ID mapping ---

int TenantIndexManager::RegisterTenantId(const char* p_tenantStr)
{
    if (p_tenantStr == nullptr) return -1;
    std::string key(p_tenantStr);
    std::lock_guard<std::mutex> lock(m_tenantIdMutex);
    auto it = m_tenantStrToInt.find(key);
    if (it != m_tenantStrToInt.end())
    {
        return it->second;
    }
    int id = m_nextInternalId++;
    m_tenantStrToInt[key] = id;
    m_tenantIntToStr[id] = key;
    return id;
}

int TenantIndexManager::GetInternalTenantId(const char* p_tenantStr) const
{
    if (p_tenantStr == nullptr) return -1;
    std::lock_guard<std::mutex> lock(m_tenantIdMutex);
    auto it = m_tenantStrToInt.find(std::string(p_tenantStr));
    return (it != m_tenantStrToInt.end()) ? it->second : -1;
}

const char* TenantIndexManager::GetTenantIdStr(int p_internalId) const
{
    std::lock_guard<std::mutex> lock(m_tenantIdMutex);
    auto it = m_tenantIntToStr.find(p_internalId);
    return (it != m_tenantIntToStr.end()) ? it->second.c_str() : nullptr;
}

std::shared_ptr<QueryResult> TenantIndexManager::SearchByTenant(
    ByteArray p_queryVector, const char* p_tenantStr, int p_resultNum)
{
    int internalId = GetInternalTenantId(p_tenantStr);
    if (internalId < 0) return nullptr;
    return Search(p_queryVector, internalId, p_resultNum);
}

bool TenantIndexManager::InitSharedFileIO()
{
    if (m_sharedFileIO) return true;
    if (m_sharedSpannWorkDir.empty()) return false;

    SPTAG::SPANN::Options sharedOpts;
    sharedOpts.m_indexDirectory = m_sharedSpannWorkDir;
    sharedOpts.m_ssdMappingFile = "ssdmapping";
    sharedOpts.m_storage = SPTAG::Storage::FILEIO;
    sharedOpts.m_datasetCapacity = std::max(m_totalPostingCount * 8, 4096);
    sharedOpts.m_datasetRowsInBlock = std::min(sharedOpts.m_datasetCapacity, 1024 * 1024);

    // Estimate file size from total posting count
    int64_t totalEstBytes = (int64_t)m_totalPostingCount * (int64_t)(m_dimension * 4 + 64) * 10;
    int startGB = std::max(1, (int)(totalEstBytes / (1024LL * 1024LL * 1024LL)) + 1);
    sharedOpts.m_startFileSize = startGB;
    sharedOpts.m_maxFileSize = std::max(startGB * 3, 10);
    sharedOpts.m_spdkBatchSize = 64;

    m_sharedFileIO = std::make_shared<SPTAG::SPANN::FileIO>(sharedOpts);
    if (!m_sharedFileIO->Available())
    {
        fprintf(stderr, "[ERROR] Failed to initialize shared FileIO at %s\n", m_sharedSpannWorkDir.c_str());
        m_sharedFileIO.reset();
        return false;
    }
    fprintf(stderr, "[INFO] Shared FileIO initialized: %s (%d total postings)\n",
        m_sharedSpannWorkDir.c_str(), m_totalPostingCount);
    return true;
}

std::shared_ptr<QueryResult> TenantIndexManager::SearchSharedSPANN(
    ByteArray p_queryVector, int p_tenantId, int p_resultNum)
{
    // Ensure shared FileIO is initialized
    if (!InitSharedFileIO()) return nullptr;

    // Ensure tenant's HeadIndex is loaded
    auto headIt = m_tenantHeadIndices.find(p_tenantId);
    if (headIt == m_tenantHeadIndices.end())
    {
        // Load HeadIndex from saved directory
        auto workIt = m_tenantSpannWorkDirs.find(p_tenantId);
        if (workIt == m_tenantSpannWorkDirs.end()) return nullptr;

        std::string headIdxDir = workIt->second + "/HeadIndex";
        std::shared_ptr<SPTAG::VectorIndex> headIdx;
        if (SPTAG::VectorIndex::LoadIndex(headIdxDir, headIdx) != SPTAG::ErrorCode::Success || !headIdx)
        {
            fprintf(stderr, "[ERROR] Failed to load HeadIndex for tenant %d from %s\n",
                p_tenantId, headIdxDir.c_str());
            return nullptr;
        }
        headIdx->SetReady(true);
        m_tenantHeadIndices[p_tenantId] = headIdx;
        headIt = m_tenantHeadIndices.find(p_tenantId);
    }

    auto& headIdx = headIt->second;
    int postingOffset = 0;
    auto offIt = m_tenantPostingOffsets.find(p_tenantId);
    if (offIt != m_tenantPostingOffsets.end()) postingOffset = offIt->second;

    // Step 1: Search HeadIndex for top candidate postings
    int internalResultNum = std::min(64, headIdx->GetNumSamples());
    SPTAG::QueryResult headQuery(p_queryVector.Data(), internalResultNum, false);
    headIdx->SearchIndex(headQuery);

    // Collect valid posting IDs (apply offset)
    std::vector<int> postingIDs;
    for (int i = 0; i < internalResultNum; i++)
    {
        auto res = headQuery.GetResult(i);
        if (res->VID == -1) break;
        postingIDs.push_back(res->VID + postingOffset);
    }

    if (postingIDs.empty())
    {
        fprintf(stderr, "[DEBUG] SearchSharedSPANN tenant %d: HeadIndex returned 0 results\n", p_tenantId);
        return nullptr;
    }
    fprintf(stderr, "[DEBUG] SearchSharedSPANN tenant %d: %d posting IDs (first=%d, offset=%d)\n",
        p_tenantId, (int)postingIDs.size(), postingIDs[0], postingOffset);

    // Step 2: Read postings from shared FileIO and compute distances
    const float* queryVec = reinterpret_cast<const float*>(p_queryVector.Data());
    size_t vectorSize = m_dimension * sizeof(float);
    int metaDataSize = sizeof(SPTAG::SizeType) + sizeof(uint8_t);  // 4 + 1 = 5
    int vectorInfoSize = (int)vectorSize + metaDataSize;

    // Priority queue for top-K results
    struct Result { int vid; float dist; };
    auto cmp = [](const Result& a, const Result& b) { return a.dist < b.dist; };
    std::priority_queue<Result, std::vector<Result>, decltype(cmp)> topK(cmp);

    int totalVecsSeen = 0;
    int getFailures = 0;
    for (int globalPostingId : postingIDs)
    {
        std::string postingData;
        auto ret = m_sharedFileIO->Get(globalPostingId, &postingData,
            SPTAG::MaxTimeout, nullptr);
        if (ret != SPTAG::ErrorCode::Success || postingData.empty()) { getFailures++; continue; }

        int numVectors = (int)postingData.size() / vectorInfoSize;
        totalVecsSeen += numVectors;
        const char* ptr = postingData.data();

        for (int j = 0; j < numVectors; j++)
        {
            SPTAG::SizeType vid;
            memcpy(&vid, ptr, sizeof(vid));
            const float* vec = reinterpret_cast<const float*>(ptr + metaDataSize);

            // Cosine distance (SPTAG uses negative inner product for cosine on normalized vectors)
            float dist = 0;
            for (int d = 0; d < m_dimension; d++)
            {
                dist -= queryVec[d] * vec[d];
            }

            if ((int)topK.size() < p_resultNum)
            {
                topK.push({vid, dist});
            }
            else if (dist < topK.top().dist)
            {
                topK.pop();
                topK.push({vid, dist});
            }

            ptr += vectorInfoSize;
        }
    }

    if (totalVecsSeen == 0)
    {
        fprintf(stderr, "[DEBUG] SearchSharedSPANN tenant %d: 0 vectors read from %d postings (%d Get failures)\n",
            p_tenantId, (int)postingIDs.size(), getFailures);
    }

    // Step 3: Build QueryResult
    auto result = std::make_shared<SPTAG::QueryResult>(p_queryVector.Data(), p_resultNum, false);
    int count = (int)topK.size();
    std::vector<Result> sorted;
    while (!topK.empty()) { sorted.push_back(topK.top()); topK.pop(); }
    std::reverse(sorted.begin(), sorted.end());

    for (int i = 0; i < p_resultNum; i++)
    {
        if (i < count)
        {
            result->SetResult(i, sorted[i].vid, sorted[i].dist);
        }
        else
        {
            result->SetResult(i, -1, SPTAG::MaxDist);
        }
    }

    return result;
}
