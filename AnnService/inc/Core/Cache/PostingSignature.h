// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// PostingSignature: Two-level Bloom filter signatures for tenant-internal
// ACL/tag filtering in SPANN.
//
// PS (Posting Signature): per-posting Bloom filter, hard reject before SSD read.
// NS (Navigation Signature): per-head-node Bloom filter (1-hop OR of PS),
//   soft filtering to guide graph traversal priority.
//
// Stored alongside HeadIndex, loaded into memory for zero-IO filtering.
//
#ifndef _SPTAG_POSTING_SIGNATURE_H_
#define _SPTAG_POSTING_SIGNATURE_H_

#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <memory>

namespace SPTAG {
namespace Cache {

// 128-bit Bloom filter with k=3 hash functions.
// Designed for ACL/tag sets with |S| ≤ 64 tags per posting.
// False positive rate ≈ 3% at |S|=50.
struct Bloom128 {
    uint64_t bits[2] = {0, 0};

    void Clear() { bits[0] = bits[1] = 0; }

    // Insert a tag ID into the Bloom filter (k=3 hashes)
    void Insert(uint32_t tag) {
        uint64_t h = Hash64(tag);
        uint32_t h0 = (h >>  0) & 127;  // bit position 0-127
        uint32_t h1 = (h >> 16) & 127;
        uint32_t h2 = (h >> 32) & 127;
        bits[h0 >> 6] |= (1ULL << (h0 & 63));
        bits[h1 >> 6] |= (1ULL << (h1 & 63));
        bits[h2 >> 6] |= (1ULL << (h2 & 63));
    }

    // Check if a tag ID might be in the set (may have false positive, no false negative)
    bool MayContain(uint32_t tag) const {
        uint64_t h = Hash64(tag);
        uint32_t h0 = (h >>  0) & 127;
        uint32_t h1 = (h >> 16) & 127;
        uint32_t h2 = (h >> 32) & 127;
        return (bits[h0 >> 6] & (1ULL << (h0 & 63))) &&
               (bits[h1 >> 6] & (1ULL << (h1 & 63))) &&
               (bits[h2 >> 6] & (1ULL << (h2 & 63)));
    }

    // Check if ANY tag in the query mask might be present.
    // query_bloom is the Bloom of all requested tags.
    bool MayIntersect(const Bloom128& query_bloom) const {
        // If any bit set in query_bloom is also set in this → possible intersection
        return (bits[0] & query_bloom.bits[0]) || (bits[1] & query_bloom.bits[1]);
    }

    // OR-merge another Bloom into this one (for NS aggregation)
    void MergeOR(const Bloom128& other) {
        bits[0] |= other.bits[0];
        bits[1] |= other.bits[1];
    }

    // Popcount (for saturation monitoring)
    int Popcount() const {
        return __builtin_popcountll(bits[0]) + __builtin_popcountll(bits[1]);
    }

    bool IsSaturated() const { return Popcount() > 96; } // >75% bits set

private:
    static uint64_t Hash64(uint32_t x) {
        // MurmurHash3 finalizer
        uint64_t h = x;
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= h >> 33;
        return h;
    }
};

// Posting Signatures for one tenant's SPANN index.
// PS[posting_id] = Bloom of all tags in that posting.
// NS[head_id] = PS[head_id] | OR(PS[neighbor]) for 1-hop.
struct TenantSignatures {
    int num_postings = 0;
    std::vector<Bloom128> ps;   // Posting Signatures, indexed by posting_id (= head VID)

    // Build PS from per-vector tags.
    // posting_tags[posting_id] = list of tag IDs for vectors in that posting.
    void BuildPS(int num_posts, const std::vector<std::vector<uint32_t>>& posting_tags) {
        num_postings = num_posts;
        ps.resize(num_posts);
        for (int i = 0; i < num_posts; i++) {
            ps[i].Clear();
            for (uint32_t tag : posting_tags[i]) {
                ps[i].Insert(tag);
            }
        }
    }

    // Save to file
    bool Save(const std::string& path) const {
        FILE* f = fopen(path.c_str(), "wb");
        if (!f) return false;
        int32_t n = num_postings;
        fwrite(&n, sizeof(int32_t), 1, f);
        fwrite(ps.data(), sizeof(Bloom128), n, f);
        fclose(f);
        return true;
    }

    // Load from file
    bool Load(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) return false;
        int32_t n = 0;
        fread(&n, sizeof(int32_t), 1, f);
        num_postings = n;
        ps.resize(n);
        fread(ps.data(), sizeof(Bloom128), n, f);
        fclose(f);
        return true;
    }

    // Memory usage in bytes
    size_t MemoryBytes() const {
        return sizeof(*this) + ps.capacity() * sizeof(Bloom128);
    }

    // Check if posting should be read from SSD (PS hard reject)
    bool ShouldReadPosting(int posting_id, const Bloom128& query_bloom) const {
        if (posting_id < 0 || posting_id >= num_postings) return true;
        return ps[posting_id].MayIntersect(query_bloom);
    }
};

// ═══════════════════════════════════════════════════════════════════
// Deterministic Bitmask PS: replaces Bloom128 for ACL tag filtering.
//
// Each tag maps to exactly 1 bit via: bit_pos = tag_id % N_BITS.
// Each posting's bitmask is the OR of all its vectors' tag bits.
//
// Query: check if bitmask[tag % N_BITS] is set.
// For hierarchical query (team → its projects), check OR of all leaf bits.
//
// FP rate ≈ (n_tags_in_posting) / N_BITS  (much lower than Bloom128).
// N_BITS = 256 (32 bytes per posting) → 40K postings = 1.25 MB.
// ═══════════════════════════════════════════════════════════════════

static constexpr int PS_BITMASK_BITS = 256;
static constexpr int PS_BITMASK_WORDS = PS_BITMASK_BITS / 64;  // 4

struct PostingBitmask {
    uint64_t bits[PS_BITMASK_WORDS] = {};

    void Clear() { for (int i = 0; i < PS_BITMASK_WORDS; i++) bits[i] = 0; }

    // Deterministic single-bit insert: tag → bit position = tag % N_BITS
    void Insert(uint32_t tag) {
        uint32_t pos = tag % PS_BITMASK_BITS;
        bits[pos >> 6] |= (1ULL << (pos & 63));
    }

    // Check if a specific tag's bit is set
    bool MayContain(uint32_t tag) const {
        uint32_t pos = tag % PS_BITMASK_BITS;
        return (bits[pos >> 6] & (1ULL << (pos & 63))) != 0;
    }

    // Check if ANY of the query tags might be present
    bool MayIntersect(const PostingBitmask& query) const {
        for (int i = 0; i < PS_BITMASK_WORDS; i++)
            if (bits[i] & query.bits[i]) return true;
        return false;
    }

    void MergeOR(const PostingBitmask& other) {
        for (int i = 0; i < PS_BITMASK_WORDS; i++) bits[i] |= other.bits[i];
    }

    int Popcount() const {
        int c = 0;
        for (int i = 0; i < PS_BITMASK_WORDS; i++) c += __builtin_popcountll(bits[i]);
        return c;
    }
};

// ═══════════════════════════════════════════════════════════════════
// Hierarchical Posting Mask: 4-level tag hierarchy filter
//
// Each level has its own bit array sized to cover the typical tag range:
//   Level 0 (org):     1000-1099 → 8 bits   (supports up to 8 orgs)
//   Level 1 (dept):    2000-2999 → 32 bits  (supports up to 32 depts)
//   Level 2 (team):    3000-3999 → 128 bits (supports up to 128 teams)
//   Level 3 (project): 4000-4999 → 128 bits (supports up to 128 projects)
//
// Insert(level, tag) sets bit at position: tag % LEVEL_BITS
// MayIntersect checks OR-across-levels (matches existing HeadNodeMatchesAnyQueryTag
// semantic: pass if ANY of head's tags equals ANY of query's tags).
// ═══════════════════════════════════════════════════════════════════

static constexpr int HIER_ORG_BITS  = 8;
static constexpr int HIER_DEPT_BITS = 32;
static constexpr int HIER_TEAM_BITS = 128;
static constexpr int HIER_PROJ_BITS = 128;

struct HierarchicalPostingMask {
    uint8_t  orgMask = 0;        // 8 bits for org tags
    uint32_t deptMask = 0;       // 32 bits for dept tags
    uint64_t teamMask[2] = {};   // 128 bits for team tags
    uint64_t projectMask[2] = {}; // 128 bits for project tags

    void Clear() {
        orgMask = 0;
        deptMask = 0;
        teamMask[0] = teamMask[1] = 0;
        projectMask[0] = projectMask[1] = 0;
    }

    // level: 0=org, 1=dept, 2=team, 3=project. tag is the raw tag id.
    void Insert(int level, uint32_t tag) {
        switch (level) {
            case 0: {  // org
                uint32_t pos = tag % HIER_ORG_BITS;
                orgMask |= (1u << pos);
                break;
            }
            case 1: {  // dept
                uint32_t pos = tag % HIER_DEPT_BITS;
                deptMask |= (1u << pos);
                break;
            }
            case 2: {  // team
                uint32_t pos = tag % HIER_TEAM_BITS;
                teamMask[pos >> 6] |= (1ULL << (pos & 63));
                break;
            }
            case 3: {  // project
                uint32_t pos = tag % HIER_PROJ_BITS;
                projectMask[pos >> 6] |= (1ULL << (pos & 63));
                break;
            }
        }
    }

    // OR-across-levels semantic: returns true if ANY level has a non-zero AND.
    // This matches the existing HeadNodeMatchesAnyQueryTag behavior.
    bool MayIntersect(const HierarchicalPostingMask& q) const {
        if ((orgMask & q.orgMask) != 0) return true;
        if ((deptMask & q.deptMask) != 0) return true;
        if ((teamMask[0] & q.teamMask[0]) != 0) return true;
        if ((teamMask[1] & q.teamMask[1]) != 0) return true;
        if ((projectMask[0] & q.projectMask[0]) != 0) return true;
        if ((projectMask[1] & q.projectMask[1]) != 0) return true;
        return false;
    }
};

// Bitmask-based Posting Signatures for one tenant.
struct TenantBitmaskPS {
    int num_postings = 0;
    std::vector<PostingBitmask> ps;

    void Build(int num_posts, const std::vector<std::vector<uint32_t>>& posting_tags) {
        num_postings = num_posts;
        ps.resize(num_posts);
        for (int i = 0; i < num_posts; i++) {
            ps[i].Clear();
            for (uint32_t tag : posting_tags[i]) {
                ps[i].Insert(tag);
            }
        }
    }

    bool Save(const std::string& path) const {
        FILE* f = fopen(path.c_str(), "wb");
        if (!f) return false;
        int32_t n = num_postings;
        fwrite(&n, sizeof(int32_t), 1, f);
        fwrite(ps.data(), sizeof(PostingBitmask), n, f);
        fclose(f);
        return true;
    }

    bool Load(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) return false;
        int32_t n = 0;
        if (fread(&n, sizeof(int32_t), 1, f) != 1) { fclose(f); return false; }
        num_postings = n;
        ps.resize(n);
        if ((int)fread(ps.data(), sizeof(PostingBitmask), n, f) != n) { fclose(f); return false; }
        fclose(f);
        return true;
    }

    size_t MemoryBytes() const {
        return sizeof(*this) + ps.capacity() * sizeof(PostingBitmask);
    }

    bool ShouldReadPosting(int posting_id, const PostingBitmask& query_mask) const {
        if (posting_id < 0 || posting_id >= num_postings) return true;
        return ps[posting_id].MayIntersect(query_mask);
    }
};

// Compute the sparse-tag threshold: if a tag has fewer matching vectors
// than this value, brute-force search over those vectors gives better
// recall than SPANN graph routing.
//
// Approximate dense-path filtered recall under a uniform-coverage model:
//   expected_matches_scanned ≈ nprobe * avg_posting * n_match / tenant_size
//   filtered_recall@topk ≈ min(1, expected_matches_scanned / topk)
//
// Solving filtered_recall@topk >= target_recall for n_match gives:
//   threshold = target_recall * topk * tenant_size / (nprobe * avg_posting)
//
// Tags with n_match <= threshold are routed to the sparse direct path.
// Lower target_recall keeps more medium-selectivity tags on the dense path;
// higher target_recall makes sparse routing more aggressive.
//
// The threshold is also capped so that brute-force latency does not
// exceed estimated SPANN latency.
//
// Parameters:
//   tenant_size    – total vectors in the tenant
//   nprobe         – number of postings read (default 64)
//   avg_posting    – average vectors per posting (0 = fallback auto-estimate)
//   target_recall  – approximate filtered Recall@topk target for dense path
//   topk           – filtered top-k used by the recall target
//   dim            – vector dimensionality (for BF latency cap)
//
// Returns: threshold count.  Tags with ≤ threshold matches → BF.
inline int SparseTagThreshold(int tenant_size,
                              int nprobe = 64,
                              double avg_posting = 0.0,
                              float target_recall = 0.95f,
                              int topk = 10,
                              int dim = 128)
{
    // Auto-estimate avg posting size from empirical data
    if (avg_posting <= 0.0) {
        if (tenant_size < 1000)       avg_posting = 3;
        else if (tenant_size < 10000) avg_posting = 5;
        else if (tenant_size < 100000) avg_posting = 20;
        else                          avg_posting = 28;
    }

    double vecs_scanned = static_cast<double>(nprobe) * avg_posting;
    if (vecs_scanned <= 0.0) vecs_scanned = 1.0;

    if (target_recall < 0.0f) target_recall = 0.0f;
    if (target_recall > 1.0f) target_recall = 1.0f;
    if (topk <= 0) topk = 10;

    double required_matches = static_cast<double>(target_recall) * static_cast<double>(topk);
    if (required_matches < 1.0) required_matches = 1.0;

    // Core formula: threshold = target_recall * topk * tenant_size / vecs_scanned
    int threshold = static_cast<int>(required_matches * static_cast<double>(tenant_size) /
                                     vecs_scanned);

    // Cap: brute-force must not exceed SPANN latency
    // SPANN latency ≈ 50us (graph) + nprobe * pages * 100us (SSD)
    int pages_per_posting = static_cast<int>((avg_posting * (dim * 4 + 21) + 4095.0) / 4096.0);
    int spann_latency_us = 50 + nprobe * pages_per_posting * 100;
    // BF: ~0.5us per vector (128-dim float, AVX2)
    int max_bf_vecs = spann_latency_us * 2;  // 0.5us per vec → divide by 0.5
    if (threshold > max_bf_vecs) threshold = max_bf_vecs;

    return threshold;
}

// Sparse tag index: for tags below the selectivity threshold,
// store a direct mapping tag → [posting_ids] so that SearchWithACL
// can read exactly those postings and brute-force scan them.
//
// No extra vector storage — reuses the existing SPANN postings.
// At query time the inline tag filter in ExtraDynamicSearcher
// still runs to pick out matching vectors from the postings.
struct SparseTagIndex {
    // tag_id → list of posting IDs that contain vectors with this tag
    std::unordered_map<uint32_t, std::vector<int>> tag_to_postings;
    // Set of tag IDs whose posting fanout is small enough for direct sparse routing.
    std::unordered_set<uint32_t> sparse_tags;

    // Build from per-posting tag lists and exact per-tag posting counts.
    // posting_tags[pid] = list of tag IDs in that posting.
    // tag_posting_counts[tag] = number of postings that contain this tag.
    // max_postings = build-time fanout cap for materializing direct posting lists.
    void Build(int num_postings,
               const std::vector<std::vector<uint32_t>>& posting_tags,
               const std::unordered_map<uint32_t, int>& tag_posting_counts,
               int max_postings)
    {
        // Materialize direct posting lists only for tags with bounded posting fanout.
        tag_to_postings.clear();
        sparse_tags.clear();
        for (const auto& [tag, posting_count] : tag_posting_counts) {
            if (posting_count > 0 && posting_count <= max_postings) {
                sparse_tags.insert(tag);
            }
        }

        // For each sparse tag, collect all posting IDs.
        for (int pid = 0; pid < num_postings; pid++) {
            std::unordered_set<uint32_t> seen;
            for (uint32_t t : posting_tags[pid]) {
                if (seen.insert(t).second && sparse_tags.count(t)) {
                    tag_to_postings[t].push_back(pid);
                }
            }
        }
    }

    bool IsSparse(uint32_t tag) const {
        return sparse_tags.count(tag) > 0;
    }

    // Get posting IDs for a sparse tag. Returns nullptr if not sparse.
    const std::vector<int>* GetPostings(uint32_t tag) const {
        auto it = tag_to_postings.find(tag);
        if (it != tag_to_postings.end()) return &it->second;
        return nullptr;
    }

    size_t MemoryBytes() const {
        size_t bytes = sparse_tags.size() * (sizeof(uint32_t) + 32);  // hash overhead
        for (auto& [t, pids] : tag_to_postings) {
            bytes += sizeof(uint32_t) + sizeof(std::vector<int>) + pids.size() * sizeof(int);
        }
        return bytes;
    }

    bool Save(const std::string& path) const {
        FILE* f = fopen(path.c_str(), "wb");
        if (!f) return false;
        int32_t n = (int32_t)tag_to_postings.size();
        fwrite(&n, sizeof(int32_t), 1, f);
        for (auto& [tag, pids] : tag_to_postings) {
            uint32_t t = tag;
            int32_t cnt = (int32_t)pids.size();
            fwrite(&t, sizeof(uint32_t), 1, f);
            fwrite(&cnt, sizeof(int32_t), 1, f);
            fwrite(pids.data(), sizeof(int32_t), cnt, f);
        }
        fclose(f);
        return true;
    }

    bool Load(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) return false;
        int32_t n = 0;
        if (fread(&n, sizeof(int32_t), 1, f) != 1) { fclose(f); return false; }
        tag_to_postings.clear();
        sparse_tags.clear();
        for (int32_t i = 0; i < n; i++) {
            uint32_t tag; int32_t cnt;
            if (fread(&tag, sizeof(uint32_t), 1, f) != 1) break;
            if (fread(&cnt, sizeof(int32_t), 1, f) != 1) break;
            std::vector<int> pids(cnt);
            if ((int)fread(pids.data(), sizeof(int32_t), cnt, f) != cnt) break;
            tag_to_postings[tag] = std::move(pids);
            sparse_tags.insert(tag);
        }
        fclose(f);
        return true;
    }
};

// Tag-pure posting (chunked, KV-backed): for very sparse tags
// (selectivity < threshold), materialize the full list of
// (VID, normalized-vector) tuples for vectors that carry the tag and
// store them inside the same KeyValueIO that holds regular postings
// (FileIO ShardedLRUCache or RocksDB block cache provides caching).
//
// Layout per chunk value:
//   repeated { int32_t vid; float normVec[dim]; }
//
// In-memory metadata kept per tag:
//   dim          : vector dimensionality
//   count        : total #entries across all chunks
//   chunkKeys    : KV keys for this tag's chunks (allocated by builder)
//   chunkCounts  : number of entries packed inside each chunk
//
// At query time, MultiGet all chunks, decode and flat-scan; R=1.0 by
// construction since each chunk holds exact members of the tag.
struct TagPurePosting {
    int dim = 0;
    int count = 0;
    std::vector<int>     chunkKeys;
    std::vector<int>     chunkCounts;

    // Pack vids+normVecs into one or more byte buffers, each holding at
    // most chunkCap entries. chunkCap must be > 0; recommended:
    //   chunkCap = floor(postingPageLimit * 4096 / (4 + dim*4))
    // Returns the packed chunks in order. Also fills chunkCounts.
    void Pack(const std::vector<int>& vids,
              const std::vector<float>& normVecs,
              int chunkCap,
              std::vector<std::string>& outChunks) {
        outChunks.clear();
        chunkCounts.clear();
        count = (int)vids.size();
        if (count == 0 || dim <= 0 || chunkCap <= 0) return;
        const size_t recSize = sizeof(int32_t) + (size_t)dim * sizeof(float);
        int idx = 0;
        while (idx < count) {
            int n = std::min(chunkCap, count - idx);
            std::string blob;
            blob.resize((size_t)n * recSize);
            char* p = blob.data();
            for (int j = 0; j < n; ++j) {
                int32_t vid = vids[idx + j];
                std::memcpy(p, &vid, sizeof(int32_t));
                std::memcpy(p + sizeof(int32_t),
                            normVecs.data() + (size_t)(idx + j) * (size_t)dim,
                            (size_t)dim * sizeof(float));
                p += recSize;
            }
            outChunks.emplace_back(std::move(blob));
            chunkCounts.push_back(n);
            idx += n;
        }
    }

    // Decode all chunks (already fetched from KV) and flat-scan against q.
    // chunkValues[i] must be the raw value blob for chunkKeys[i], holding
    // chunkCounts[i] records of (int32 vid + float[dim] normVec).
    // Distance: 1.0 - cos(q_normalized, v) → matches SPTAG cosine.
    void SearchTopK(const float* q,
                    const std::vector<std::string>& chunkValues,
                    int topK,
                    std::vector<std::pair<float, int>>& out) const {
        float qn2 = 0.0f;
        for (int i = 0; i < dim; ++i) qn2 += q[i] * q[i];
        float qInv = (qn2 > 1e-30f) ? 1.0f / std::sqrt(qn2) : 0.0f;
        std::vector<float> qn(dim);
        for (int i = 0; i < dim; ++i) qn[i] = q[i] * qInv;

        const size_t recSize = sizeof(int32_t) + (size_t)dim * sizeof(float);
        out.clear();
        out.reserve(count);
        for (size_t c = 0; c < chunkValues.size() && c < chunkCounts.size(); ++c) {
            int n = chunkCounts[c];
            const char* base = chunkValues[c].data();
            if (chunkValues[c].size() < (size_t)n * recSize) continue;
            for (int j = 0; j < n; ++j) {
                const char* rec = base + (size_t)j * recSize;
                int32_t vid;
                std::memcpy(&vid, rec, sizeof(int32_t));
                const float* v = reinterpret_cast<const float*>(rec + sizeof(int32_t));
                float ip = 0.0f;
                for (int i = 0; i < dim; ++i) ip += qn[i] * v[i];
                out.emplace_back(1.0f - ip, (int)vid);
            }
        }
        int k = std::min(topK, (int)out.size());
        if (k < (int)out.size()) {
            std::partial_sort(out.begin(), out.begin() + k, out.end(),
                              [](const std::pair<float,int>& a,
                                 const std::pair<float,int>& b) { return a.first < b.first; });
            out.resize(k);
        } else {
            std::sort(out.begin(), out.end(),
                      [](const std::pair<float,int>& a,
                         const std::pair<float,int>& b) { return a.first < b.first; });
        }
    }
};

// Persistence bundle for an entire tenant's tag-pure postings.
// File layout (little-endian):
//   [magic   uint32 = 'TPUR' = 0x52555054]
//   [version uint32 = 1]
//   [dim     int32 ]
//   [numTags uint32]
//   repeated numTags times:
//     [tag_id      uint32]
//     [count       int32 ]
//     [numChunks   uint32]
//     [chunkKeys   int32 × numChunks]
//     [chunkCounts int32 × numChunks]
struct TagPureBundle {
    static constexpr uint32_t kMagic   = 0x52555054u;  // 'TPUR' little-endian
    static constexpr uint32_t kVersion = 1u;

    static bool Save(const std::string& path,
                     int dim,
                     const std::unordered_map<uint32_t,
                         std::shared_ptr<TagPurePosting>>& tags)
    {
        FILE* f = std::fopen(path.c_str(), "wb");
        if (f == nullptr) return false;
        uint32_t magic = kMagic, version = kVersion;
        int32_t dim32 = dim;
        uint32_t numTags = 0;
        for (const auto& kv : tags) if (kv.second && kv.second->count > 0) ++numTags;
        if (std::fwrite(&magic,   sizeof(magic),   1, f) != 1) { std::fclose(f); return false; }
        if (std::fwrite(&version, sizeof(version), 1, f) != 1) { std::fclose(f); return false; }
        if (std::fwrite(&dim32,   sizeof(dim32),   1, f) != 1) { std::fclose(f); return false; }
        if (std::fwrite(&numTags, sizeof(numTags), 1, f) != 1) { std::fclose(f); return false; }
        for (const auto& kv : tags) {
            if (!kv.second || kv.second->count <= 0) continue;
            uint32_t tagId = kv.first;
            const auto& p = *kv.second;
            int32_t cnt = p.count;
            uint32_t nChunks = (uint32_t)p.chunkKeys.size();
            if (std::fwrite(&tagId,   sizeof(tagId),   1, f) != 1) { std::fclose(f); return false; }
            if (std::fwrite(&cnt,     sizeof(cnt),     1, f) != 1) { std::fclose(f); return false; }
            if (std::fwrite(&nChunks, sizeof(nChunks), 1, f) != 1) { std::fclose(f); return false; }
            if (nChunks > 0) {
                if (std::fwrite(p.chunkKeys.data(),   sizeof(int32_t), nChunks, f) != nChunks)
                { std::fclose(f); return false; }
                if (std::fwrite(p.chunkCounts.data(), sizeof(int32_t), nChunks, f) != nChunks)
                { std::fclose(f); return false; }
            }
        }
        std::fclose(f);
        return true;
    }

    static bool Load(const std::string& path,
                     int& outDim,
                     std::unordered_map<uint32_t,
                         std::shared_ptr<TagPurePosting>>& outTags)
    {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (f == nullptr) return false;
        uint32_t magic = 0, version = 0, numTags = 0;
        int32_t dim32 = 0;
        if (std::fread(&magic,   sizeof(magic),   1, f) != 1) { std::fclose(f); return false; }
        if (std::fread(&version, sizeof(version), 1, f) != 1) { std::fclose(f); return false; }
        if (std::fread(&dim32,   sizeof(dim32),   1, f) != 1) { std::fclose(f); return false; }
        if (std::fread(&numTags, sizeof(numTags), 1, f) != 1) { std::fclose(f); return false; }
        if (magic != kMagic || version != kVersion || dim32 <= 0) { std::fclose(f); return false; }
        outDim = dim32;
        outTags.clear();
        outTags.reserve(numTags);
        for (uint32_t i = 0; i < numTags; ++i) {
            uint32_t tagId = 0, nChunks = 0;
            int32_t cnt = 0;
            if (std::fread(&tagId,   sizeof(tagId),   1, f) != 1) { std::fclose(f); return false; }
            if (std::fread(&cnt,     sizeof(cnt),     1, f) != 1) { std::fclose(f); return false; }
            if (std::fread(&nChunks, sizeof(nChunks), 1, f) != 1) { std::fclose(f); return false; }
            auto pp = std::make_shared<TagPurePosting>();
            pp->dim = dim32;
            pp->count = cnt;
            pp->chunkKeys.resize(nChunks);
            pp->chunkCounts.resize(nChunks);
            if (nChunks > 0) {
                if (std::fread(pp->chunkKeys.data(),   sizeof(int32_t), nChunks, f) != nChunks)
                { std::fclose(f); return false; }
                if (std::fread(pp->chunkCounts.data(), sizeof(int32_t), nChunks, f) != nChunks)
                { std::fclose(f); return false; }
            }
            outTags[tagId] = std::move(pp);
        }
        std::fclose(f);
        return true;
    }
};

}  // namespace Cache
}  // namespace SPTAG

#endif // _SPTAG_POSTING_SIGNATURE_H_
