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
#include <unordered_map>
#include <unordered_set>

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
    // Set of tag IDs that are sparse (below threshold)
    std::unordered_set<uint32_t> sparse_tags;

    // Build from per-posting tag lists and exact per-tag vector counts.
    // posting_tags[pid] = list of tag IDs in that posting.
    // tag_vector_counts[tag] = number of tenant vectors that carry this tag.
    // threshold = SparseTagThreshold(tenant_size, ...).
    void Build(int num_postings,
               const std::vector<std::vector<uint32_t>>& posting_tags,
               const std::unordered_map<uint32_t, int>& tag_vector_counts,
               int threshold)
    {
        // Identify sparse tags by their exact vector counts.
        tag_to_postings.clear();
        sparse_tags.clear();
        for (const auto& [tag, vector_count] : tag_vector_counts) {
            if (vector_count <= threshold) {
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

}  // namespace Cache
}  // namespace SPTAG

#endif // _SPTAG_POSTING_SIGNATURE_H_
