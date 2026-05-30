// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// augmentheadgraph
//
// Stand-alone post-processing tool that, after a multi-tenant SPANN index
// has been built, walks every per-subgraph head BKT and computes for each
// head h a small set of "cross-subgraph" shortcut edges: the M nearest
// heads that live in OTHER subgraphs.
//
// The result is written to <head_index_dir>/head_cross_edges.bin (next to
// head_bundle_manifest.bin) using the format declared in
// inc/Helper/HeadCrossEdges.h. Indexes that do not have this file load
// unchanged; the file is therefore fully backward-compatible.
//
// Per design (see plan.md, "Active Phase: Cross-Subgraph Edge
// Augmentation"), this is build-only MVP. The search path is unchanged.

#include "inc/Core/Common.h"
#include "inc/Core/Common/Dataset.h"
#include "inc/Core/SearchQuery.h"
#include "inc/Core/SearchResult.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Helper/ArgumentsParser.h"
#include "inc/Helper/HeadCrossEdges.h"
#include "inc/Helper/Logging.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
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

std::string JoinPath(const std::string& base, const std::string& rel)
{
    if (rel.empty()) return base;
    std::string out = base;
    if (!out.empty() && out.back() != FolderSep) out += FolderSep;
    out += rel;
    return out;
}

bool LoadManifest(const std::string& headIndexDir, std::vector<BundleNode>& outNodes)
{
    const std::string manifestPath = JoinPath(headIndexDir, "head_bundle_manifest.bin");
    FILE* fp = fopen(manifestPath.c_str(), "rb");
    if (fp == nullptr) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot open manifest: %s\n", manifestPath.c_str());
        return false;
    }

    HeadBundleManifestHeader header{};
    if (fread(&header, sizeof(header), 1, fp) != 1) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read manifest header.\n");
        fclose(fp);
        return false;
    }
    if (header.magic != kHeadBundleManifestMagic || header.version != kHeadBundleManifestVersion) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "Manifest magic/version mismatch (magic=0x%x version=%d).\n",
                     header.magic, header.version);
        fclose(fp);
        return false;
    }

    outNodes.clear();
    outNodes.reserve(static_cast<size_t>(header.nodeCount));
    // Bundle manifest paths are relative to the index *base* dir, not to
    // the head index dir. Walk one level up from headIndexDir.
    std::string baseDir = headIndexDir;
    if (!baseDir.empty() && baseDir.back() == FolderSep) baseDir.pop_back();
    auto sep = baseDir.find_last_of(FolderSep);
    if (sep != std::string::npos) {
        baseDir = baseDir.substr(0, sep);
    } else {
        baseDir = ".";
    }

    for (std::int32_t i = 0; i < header.nodeCount; ++i) {
        HeadBundleManifestNodeRecordV2 rec{};
        if (fread(&rec, sizeof(rec), 1, fp) != 1) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read manifest record %d.\n", i);
            fclose(fp);
            return false;
        }
        std::string rel(static_cast<size_t>(rec.pathLength), '\0');
        if (rec.pathLength > 0 &&
            fread(&rel[0], 1, static_cast<size_t>(rec.pathLength), fp)
                != static_cast<size_t>(rec.pathLength)) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read manifest path %d.\n", i);
            fclose(fp);
            return false;
        }
        BundleNode node;
        node.nodeId = rec.nodeId;
        node.relativePath = rel;
        node.absolutePath = JoinPath(baseDir, rel);
        node.headCount = rec.headCount;
        outNodes.emplace_back(std::move(node));
    }

    fclose(fp);
    return true;
}

bool LoadAllSubgraphs(std::vector<BundleNode>& nodes,
                      const std::string& headIDFileName,
                      int blockSize,
                      int capacity,
                      int searchTopK)
{
    for (auto& n : nodes) {
        if (n.headCount == 0) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                         "Subgraph node %d has 0 heads — skipping load.\n", n.nodeId);
            continue;
        }
        std::shared_ptr<VectorIndex> idx;
        if (VectorIndex::LoadIndex(n.absolutePath, idx) != ErrorCode::Success || idx == nullptr) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Failed to load subgraph node %d from %s\n",
                         n.nodeId, n.absolutePath.c_str());
            return false;
        }
        // Generous MaxCheck so we approach the BKT recall ceiling.
        idx->SetParameter("MaxCheck", std::to_string(std::max(8192, searchTopK * 64)));
        idx->UpdateIndex();
        idx->SetReady(true);
        n.index = std::move(idx);

        COMMON::Dataset<std::uint64_t> headIDs;
        headIDs.SetName("HeadBundleNodeIDs");
        const std::string idsPath = JoinPath(n.absolutePath, headIDFileName);
        if (headIDs.Load(idsPath, blockSize, capacity) != ErrorCode::Success) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Failed to load head IDs file %s\n", idsPath.c_str());
            return false;
        }
        n.localHidToGlobalVID.resize(headIDs.R());
        for (SizeType lh = 0; lh < headIDs.R(); ++lh) {
            n.localHidToGlobalVID[lh] = static_cast<SizeType>(*(headIDs[lh]));
        }
        if (static_cast<SizeType>(n.localHidToGlobalVID.size()) != n.index->GetNumSamples()) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Subgraph %d: head id count (%zu) != BKT samples (%d)\n",
                         n.nodeId, n.localHidToGlobalVID.size(),
                         (int)n.index->GetNumSamples());
            return false;
        }
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                     "Loaded subgraph %d: %d heads, dim=%d, type=%d\n",
                     n.nodeId, (int)n.index->GetNumSamples(),
                     (int)n.index->GetFeatureDim(),
                     (int)n.index->GetVectorValueType());
    }
    return true;
}

class AugmentOptions : public Helper::ArgumentsParser
{
public:
    AugmentOptions()
    {
        AddRequiredOption(m_headIndexDir, "-d", "--head_index_dir",
                          "Directory containing head_bundle_manifest.bin (e.g. <index>/HeadVectors).");
        AddOptionalOption(m_searchTopK, "-k", "--search_topk",
                          "Per-subgraph BKT search top (default 15).");
        AddOptionalOption(m_extraEdges, "-m", "--extra_edges",
                          "Cross-subgraph edges to keep per head (default 10).");
        AddOptionalOption(m_threads, "-t", "--threads",
                          "Worker threads (default hardware_concurrency).");
        AddOptionalOption(m_overwrite, "-w", "--overwrite",
                          "Overwrite existing head_cross_edges.bin (default false).");
        AddOptionalOption(m_headIDFile, "-i", "--head_id_file",
                          "HeadVectorIDs file name inside each subgraph (default SPTAGHeadVectorIDs.bin).");
    }

    std::string m_headIndexDir;
    int m_searchTopK = 15;
    int m_extraEdges = 10;
    int m_threads = static_cast<int>(std::thread::hardware_concurrency());
    bool m_overwrite = false;
    std::string m_headIDFile = "SPTAGHeadVectorIDs.bin";
};

} // namespace

int main(int argc, char** argv)
{
    auto opts = std::make_shared<AugmentOptions>();
    if (!opts->Parse(argc - 1, argv + 1)) {
        return 1;
    }
    if (opts->m_threads <= 0) opts->m_threads = 1;
    if (opts->m_searchTopK <= 0) opts->m_searchTopK = 15;
    if (opts->m_extraEdges <= 0) opts->m_extraEdges = 10;

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                 "augmentheadgraph: dir=%s topk=%d M=%d threads=%d overwrite=%d\n",
                 opts->m_headIndexDir.c_str(), opts->m_searchTopK, opts->m_extraEdges,
                 opts->m_threads, (int)opts->m_overwrite);

    const std::string outPath = JoinPath(opts->m_headIndexDir, Helper::kHeadCrossEdgesFileName);
    if (!opts->m_overwrite) {
        FILE* probe = fopen(outPath.c_str(), "rb");
        if (probe != nullptr) {
            fclose(probe);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "%s already exists. Pass --overwrite to regenerate.\n",
                         outPath.c_str());
            return 2;
        }
    }

    std::vector<BundleNode> nodes;
    if (!LoadManifest(opts->m_headIndexDir, nodes)) return 3;
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Manifest: %zu subgraph nodes.\n", nodes.size());

    if (nodes.size() <= 1) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                     "Only %zu subgraph(s) — writing empty cross-edge file.\n", nodes.size());
        FILE* fp = fopen(outPath.c_str(), "wb");
        if (fp == nullptr) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot open %s for write.\n", outPath.c_str());
            return 4;
        }
        Helper::HeadCrossEdgesHeader header{};
        header.magic = Helper::kHeadCrossEdgesMagic;
        header.version = Helper::kHeadCrossEdgesVersion;
        header.totalHeads = 0;
        header.maxEdgesPerHead = opts->m_extraEdges;
        header.searchTopK = opts->m_searchTopK;
        fwrite(&header, sizeof(header), 1, fp);
        fclose(fp);
        return 0;
    }

    constexpr int kDefaultBlockSize = 1024 * 1024;
    constexpr int kDefaultCapacity = 1024 * 1024;
    if (!LoadAllSubgraphs(nodes, opts->m_headIDFile,
                          kDefaultBlockSize, kDefaultCapacity,
                          opts->m_searchTopK)) {
        return 5;
    }

    // Sanity: all subgraphs share value type and dim.
    VectorValueType vt = VectorValueType::Undefined;
    DimensionType dim = -1;
    size_t totalHeads = 0;
    for (const auto& n : nodes) {
        if (n.index == nullptr) continue;
        if (vt == VectorValueType::Undefined) {
            vt = n.index->GetVectorValueType();
            dim = n.index->GetFeatureDim();
        } else if (vt != n.index->GetVectorValueType() || dim != n.index->GetFeatureDim()) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Subgraph %d type/dim mismatch.\n", n.nodeId);
            return 6;
        }
        totalHeads += n.index->GetNumSamples();
    }
    if (totalHeads == 0) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "No heads across subgraphs; nothing to do.\n");
        return 0;
    }

    // Flatten work list: (subgraphIdx, localHid).
    struct WorkItem { int subIdx; SizeType localHid; };
    std::vector<WorkItem> work;
    work.reserve(totalHeads);
    for (size_t s = 0; s < nodes.size(); ++s) {
        if (nodes[s].index == nullptr) continue;
        const SizeType n = nodes[s].index->GetNumSamples();
        for (SizeType h = 0; h < n; ++h) {
            work.push_back({static_cast<int>(s), h});
        }
    }

    struct Record {
        SizeType globalVID = -1;
        std::vector<Helper::HeadCrossEdgeEntry> edges;
    };
    std::vector<Record> records(work.size());

    std::atomic<size_t> nextIdx{0};
    std::atomic<size_t> doneCount{0};
    std::atomic<size_t> nonEmpty{0};
    std::atomic<size_t> fullCount{0};

    auto worker = [&]() {
        std::vector<BasicResult> buf(static_cast<size_t>(opts->m_searchTopK));
        std::vector<Helper::HeadCrossEdgeEntry> merged;
        merged.reserve(static_cast<size_t>(opts->m_searchTopK) * (nodes.size() - 1));
        while (true) {
            size_t i = nextIdx.fetch_add(1);
            if (i >= work.size()) return;
            const WorkItem w = work[i];
            const auto& srcNode = nodes[w.subIdx];
            if (srcNode.index == nullptr) continue;
            const void* qVec = srcNode.index->GetSample(w.localHid);
            if (qVec == nullptr) continue;

            merged.clear();
            for (size_t j = 0; j < nodes.size(); ++j) {
                if (static_cast<int>(j) == w.subIdx) continue;
                const auto& tgt = nodes[j];
                if (tgt.index == nullptr || tgt.index->GetNumSamples() == 0) continue;
                std::fill(buf.begin(), buf.end(), BasicResult());
                QueryResult qr(qVec, opts->m_searchTopK, false, buf.data());
                if (tgt.index->SearchIndex(qr) != ErrorCode::Success) continue;
                for (int r = 0; r < opts->m_searchTopK; ++r) {
                    const BasicResult& br = buf[r];
                    if (br.VID < 0 || br.VID >= static_cast<SizeType>(tgt.localHidToGlobalVID.size())) continue;
                    if (br.Dist >= MaxDist) continue;
                    Helper::HeadCrossEdgeEntry e;
                    e.neighborGlobalVID = static_cast<std::int32_t>(tgt.localHidToGlobalVID[br.VID]);
                    e.dist = br.Dist;
                    merged.push_back(e);
                }
            }
            std::sort(merged.begin(), merged.end(),
                      [](const Helper::HeadCrossEdgeEntry& a, const Helper::HeadCrossEdgeEntry& b) {
                          return a.dist < b.dist;
                      });
            const int keep = std::min<int>(opts->m_extraEdges, static_cast<int>(merged.size()));
            Record& rec = records[i];
            rec.globalVID = srcNode.localHidToGlobalVID[w.localHid];
            rec.edges.assign(merged.begin(), merged.begin() + keep);
            if (keep > 0) nonEmpty.fetch_add(1);
            if (keep >= opts->m_extraEdges) fullCount.fetch_add(1);
            const size_t d = doneCount.fetch_add(1) + 1;
            if (d % 5000 == 0) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                             "Progress: %zu/%zu heads processed.\n", d, work.size());
            }
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(opts->m_threads);
    for (int t = 0; t < opts->m_threads; ++t) pool.emplace_back(worker);
    for (auto& t : pool) t.join();

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                 "Augmentation complete: %zu heads, %zu with edges, %zu fully filled.\n",
                 work.size(), nonEmpty.load(), fullCount.load());

    // Dual-pool approach-1: add reverse H1->U_extra cross-edges. U_extra heads
    // are appended graph-only in each bundle (localHid >= manifest headCount)
    // with out-edges toward H1 but NO back-edges into H1's RNG lists, so during
    // unfilter traversal H1 nodes cannot reach U_extra unless we explicitly wire
    // cross-edges H1->U_extra. For every U_extra source, take its forward edges
    // to H1 heads in other bundles and register the reverse edge on those H1
    // records. This is the only path by which an H1 (in another bundle) reaches
    // a U_extra head; filter-mode never follows cross-edges, keeping U_extra
    // unfilter-only.
    {
        auto isUExtra = [&](int subIdx, SizeType localHid) -> bool {
            return localHid >= static_cast<SizeType>(nodes[subIdx].headCount);
        };
        std::unordered_map<SizeType, size_t> vidToRec;
        vidToRec.reserve(records.size() * 2);
        for (size_t i = 0; i < records.size(); ++i) {
            if (records[i].globalVID >= 0) vidToRec[records[i].globalVID] = i;
        }
        std::unordered_set<SizeType> uExtraVIDs;
        for (size_t i = 0; i < work.size(); ++i) {
            if (isUExtra(work[i].subIdx, work[i].localHid) && records[i].globalVID >= 0)
                uExtraVIDs.insert(records[i].globalVID);
        }
        size_t reverseAdded = 0;
        for (size_t i = 0; i < work.size(); ++i) {
            if (!isUExtra(work[i].subIdx, work[i].localHid)) continue;
            const SizeType uVID = records[i].globalVID;
            if (uVID < 0) continue;
            for (const auto& e : records[i].edges) {
                const SizeType nbrVID = static_cast<SizeType>(e.neighborGlobalVID);
                if (uExtraVIDs.count(nbrVID)) continue; // only H1 targets get reverse edge
                auto it = vidToRec.find(nbrVID);
                if (it == vidToRec.end()) continue;
                Record& hrec = records[it->second];
                bool exists = false;
                for (const auto& he : hrec.edges) {
                    if (he.neighborGlobalVID == static_cast<std::int32_t>(uVID)) { exists = true; break; }
                }
                if (exists) continue;
                Helper::HeadCrossEdgeEntry re;
                re.neighborGlobalVID = static_cast<std::int32_t>(uVID);
                re.dist = e.dist;
                hrec.edges.push_back(re);
                ++reverseAdded;
            }
        }
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                     "DualPool approach-1: added %zu reverse H1->U_extra cross-edges.\n", reverseAdded);
    }

    // Cross-edge loader rejects records whose edge count exceeds the header's
    // maxEdgesPerHead, so record the true maximum after reverse-edge merging.
    int actualMaxEdges = opts->m_extraEdges;
    for (const auto& r : records) {
        actualMaxEdges = std::max<int>(actualMaxEdges, static_cast<int>(r.edges.size()));
    }

    FILE* fp = fopen(outPath.c_str(), "wb");
    if (fp == nullptr) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot open %s for write.\n", outPath.c_str());
        return 7;
    }
    Helper::HeadCrossEdgesHeader header{};
    header.magic = Helper::kHeadCrossEdgesMagic;
    header.version = Helper::kHeadCrossEdgesVersion;
    header.totalHeads = static_cast<std::int32_t>(records.size());
    header.maxEdgesPerHead = actualMaxEdges;
    header.searchTopK = opts->m_searchTopK;
    if (fwrite(&header, sizeof(header), 1, fp) != 1) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Write header failed.\n");
        fclose(fp);
        return 8;
    }
    for (const auto& r : records) {
        std::int32_t vid = static_cast<std::int32_t>(r.globalVID);
        std::int32_t cnt = static_cast<std::int32_t>(r.edges.size());
        bool ok = fwrite(&vid, sizeof(vid), 1, fp) == 1 &&
                  fwrite(&cnt, sizeof(cnt), 1, fp) == 1;
        if (ok && cnt > 0) {
            ok = fwrite(r.edges.data(), sizeof(Helper::HeadCrossEdgeEntry),
                        static_cast<size_t>(cnt), fp) == static_cast<size_t>(cnt);
        }
        if (!ok) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Write record failed.\n");
            fclose(fp);
            return 9;
        }
    }
    fclose(fp);
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                 "Wrote %s (%zu records).\n", outPath.c_str(), records.size());
    return 0;
}
