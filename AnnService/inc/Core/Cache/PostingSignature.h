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

}  // namespace Cache
}  // namespace SPTAG

#endif // _SPTAG_POSTING_SIGNATURE_H_
