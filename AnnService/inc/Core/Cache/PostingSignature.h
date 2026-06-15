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
// Wider page-level signature (collision-free for >256 distinct tags).
// Used by the page-selective IO directory: the org+dept+team+project tag
// space spans up to ~340 ids, so a 256-bit mask aliases project ids
// 256.. onto lower-level bits (false positives -> over-read). 512 bits
// keeps all of them collision-free while costing only 64B/page. Measured:
// at 46 vec/posting the over-read is dominated by genuine tag dilution, not
// bit collisions, so 256 bits (32B/page) is the lean operating point.
// ═══════════════════════════════════════════════════════════════════
static constexpr int PS_PAGE_BITS  = 256;
static constexpr int PS_PAGE_WORDS = PS_PAGE_BITS / 64;  // 4

struct PageBitmask {
    uint64_t bits[PS_PAGE_WORDS] = {};

    void Clear() { for (int i = 0; i < PS_PAGE_WORDS; i++) bits[i] = 0; }

    void Insert(uint32_t tag) {
        uint32_t pos = tag % PS_PAGE_BITS;
        bits[pos >> 6] |= (1ULL << (pos & 63));
    }

    bool MayContain(uint32_t tag) const {
        uint32_t pos = tag % PS_PAGE_BITS;
        return (bits[pos >> 6] & (1ULL << (pos & 63))) != 0;
    }

    bool MayIntersect(const PageBitmask& query) const {
        for (int i = 0; i < PS_PAGE_WORDS; i++)
            if (bits[i] & query.bits[i]) return true;
        return false;
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

// Generalised hierarchical posting mask: HIER_LEVELS independent levels, each a
// HIER_LEVEL_BITS-wide bit array. A "level" is a categorical tag column
// (0=org/country, 1=dept/year, ...). Bit position within a level is tag %
// HIER_LEVEL_BITS. With 256 bits/level any column whose distinct values span a
// contiguous range <= 256 maps collision-free (e.g. YFCC country 1000..1245).
//
// Backwards compatible with the legacy 4-level org/dept/team/project layout:
// those columns (0..3) still map to levels 0..3, and 256 bits is >= every legacy
// width (8/32/128/128) so pruning is equal-or-better. The wider/extra levels
// change the persisted head_node_meta byte size (offsets derive from sizeof), so
// indexes must be rebuilt -- there is no on-disk back-compat with old indexes.
//
// HIER_LEVELS is sized to the widest schema in use: YFCC scope-A = 5 facets
// (year/month/camera/country/us_state), SIFT ACL = 4 (org/dept/team/project).
// 5 covers both with no spare; raise it if a schema needs more categorical cols.
static constexpr int HIER_LEVELS     = 5;
static constexpr int HIER_LEVEL_BITS = 256;                   // max width / level
static constexpr int HIER_LEVEL_WORDS = HIER_LEVEL_BITS / 64;  // 4
// Maximum total 64-bit words across all levels (used as the fixed storage
// capacity of HierarchicalPostingMask so its compile-time sizeof stays stable).
static constexpr int HIER_MAX_WORDS  = HIER_LEVELS * HIER_LEVEL_WORDS;  // 20

// Legacy names kept for any external reference / documentation.
static constexpr int HIER_ORG_BITS  = HIER_LEVEL_BITS;
static constexpr int HIER_DEPT_BITS = HIER_LEVEL_BITS;
static constexpr int HIER_TEAM_BITS = HIER_LEVEL_BITS;
static constexpr int HIER_PROJ_BITS = HIER_LEVEL_BITS;

// ═══════════════════════════════════════════════════════════════════
// Runtime per-column mask width table.
//
// Each categorical column (level) gets its own bit-width, chosen at build time
// from the column's value cardinality (collision-free iff width >= value span).
// Low-cardinality columns (year=12, month=13) need only 64 bits, not the full
// 256, so the per-head posting-content mask packs tighter: e.g. YFCC widths
// [256,64,64,128,64] -> 576 bits = 9 words = 72 B instead of 5*256 = 160 B.
//
// The table is a process-global set once per index:
//   * at build  -> SetHierWidths() from the SPTAG_HIER_LEVEL_WIDTHS env list.
//   * at load   -> SetHierWidths() from the widths persisted in the
//                  head_node_meta.bin V5 header (so the byte layout matches).
// Default (never set, or legacy V3/V4 indexes) is uniform HIER_LEVEL_BITS,
// which reproduces the previous fixed 256-bit-per-level layout bit-for-bit.
// Widths are clamped to [64, HIER_LEVEL_BITS] and rounded up to a multiple of
// 64, guaranteeing totalWords <= HIER_MAX_WORDS (no storage overflow).
// ═══════════════════════════════════════════════════════════════════
struct HierWidthTable {
    int bits[HIER_LEVELS];
    int wordOff[HIER_LEVELS];
    int totalWords;

    void Recompute() {
        int off = 0;
        for (int l = 0; l < HIER_LEVELS; ++l) { wordOff[l] = off; off += bits[l] / 64; }
        totalWords = off;
    }
    void SetUniform() { for (int l = 0; l < HIER_LEVELS; ++l) bits[l] = HIER_LEVEL_BITS; Recompute(); }
};

inline HierWidthTable& MutableHierWidths() {
    static HierWidthTable t = [] { HierWidthTable x; x.SetUniform(); return x; }();
    return t;
}
inline const HierWidthTable& HierWidths() { return MutableHierWidths(); }

// Set per-level widths (bits). n entries are consumed (n <= HIER_LEVELS);
// remaining levels default to HIER_LEVEL_BITS. Each width is clamped to
// [64, HIER_LEVEL_BITS] and rounded up to a multiple of 64.
inline void SetHierWidths(const int* bitsPerLevel, int n) {
    auto& t = MutableHierWidths();
    for (int l = 0; l < HIER_LEVELS; ++l) {
        int b = (l < n) ? bitsPerLevel[l] : HIER_LEVEL_BITS;
        if (b < 64) b = 64;
        if (b > HIER_LEVEL_BITS) b = HIER_LEVEL_BITS;
        b = ((b + 63) / 64) * 64;
        t.bits[l] = b;
    }
    t.Recompute();
}
inline void SetHierWidthsUniform() { MutableHierWidths().SetUniform(); }

// Number of bytes actually occupied by one HierarchicalPostingMask given the
// current width table. The per-head meta record uses this (NOT sizeof) so the
// stored mask shrinks for narrow schemas.
inline size_t HierPostingMaskBytes() { return static_cast<size_t>(HierWidths().totalWords) * sizeof(uint64_t); }

struct HierarchicalPostingMask {
    // Flat storage of up to HIER_MAX_WORDS words. Only the first
    // HierWidths().totalWords words are ever used; level L occupies words
    // [wordOff[L], wordOff[L] + bits[L]/64). The compile-time sizeof is fixed
    // (160 B) so the type is a stable POD, but per-head persistence copies only
    // HierPostingMaskBytes() bytes.
    uint64_t mask[HIER_MAX_WORDS] = {};

    void Clear() {
        for (int w = 0; w < HIER_MAX_WORDS; ++w) mask[w] = 0;
    }

    // level: categorical tag column index (0=org/country, 1=dept/year, ...).
    // tag is the raw tag id. Columns >= HIER_LEVELS are ignored (fail-open at
    // query time via MayContain returning true), preserving correctness.
    void Insert(int level, uint32_t tag) {
        if (level < 0 || level >= HIER_LEVELS) return;
        const auto& t = HierWidths();
        uint32_t pos = tag % static_cast<uint32_t>(t.bits[level]);
        mask[t.wordOff[level] + (pos >> 6)] |= (1ULL << (pos & 63));
    }

    // OR-across-levels semantic: returns true if ANY level has a non-zero AND.
    // Because levels partition the used words (wordOff), a flat AND over the
    // used words [0, totalWords) is identical to the per-level OR-of-ANDs.
    bool MayIntersect(const HierarchicalPostingMask& q) const {
        const int tw = HierWidths().totalWords;
        for (int w = 0; w < tw; ++w)
            if ((mask[w] & q.mask[w]) != 0) return true;
        return false;
    }

    // Single (level, tag) membership test. No false negatives. A column beyond
    // the supported level count fails open (returns true) so such predicates are
    // never wrongly pruned here -- they are still enforced by the exact post-filter.
    bool MayContain(int level, uint32_t tag) const {
        if (level < 0 || level >= HIER_LEVELS) return true;
        const auto& t = HierWidths();
        uint32_t pos = tag % static_cast<uint32_t>(t.bits[level]);
        return (mask[t.wordOff[level] + (pos >> 6)] & (1ULL << (pos & 63))) != 0;
    }
};

// ═══════════════════════════════════════════════════════════════════
// Compact own-tags signature for a HEAD centroid.
//
// A head carries exactly ONE categorical value per column (single-valued
// facets), so its own-tag mask has at most HIER_LEVELS bits set. Storing it as
// a full HierarchicalPostingMask bitmap (HIER_LEVELS*HIER_LEVEL_BITS bits)
// wastes ~97% of the space. Keep the raw value per level instead: 4 B/level vs
// 32 B/level. Empty levels (column index >= numTagsPerVec) use OWN_TAG_EMPTY and
// are skipped at match time. Categorical tag ids are small (<< OWN_TAG_EMPTY),
// so the sentinel never collides with a real value.
// ═══════════════════════════════════════════════════════════════════
static constexpr uint32_t OWN_TAG_EMPTY = 0xFFFFFFFFu;

struct HierarchicalOwnTags {
    uint32_t tag[HIER_LEVELS];

    HierarchicalOwnTags() { Clear(); }

    void Clear() { for (int l = 0; l < HIER_LEVELS; ++l) tag[l] = OWN_TAG_EMPTY; }

    // level: categorical tag column index. Columns >= HIER_LEVELS are ignored
    // (matching HierarchicalPostingMask::Insert; enforced by the exact
    // post-filter, never wrongly pruned here).
    void Insert(int level, uint32_t t) {
        if (level < 0 || level >= HIER_LEVELS) return;
        tag[level] = t;
    }

    // Equivalent to the former (single-bit-per-level own bitmap) MayIntersect:
    // true iff ANY level's own value is admitted by the query mask at that
    // level. q.MayContain(l, v) reproduces the exact bit-collision behaviour of
    // the old bitmap AND (pos = v % HIER_LEVEL_BITS), so pruning is unchanged.
    bool MayIntersect(const HierarchicalPostingMask& q) const {
        for (int l = 0; l < HIER_LEVELS; ++l) {
            if (tag[l] == OWN_TAG_EMPTY) continue;
            if (q.MayContain(l, tag[l])) return true;
        }
        return false;
    }
};

// ═══════════════════════════════════════════════════════════════════
// Literal comparison operator. EQ is the categorical (tag) semantic and the
// historical default. LT/LE/GT/GE are numerical range comparisons evaluated on
// the per-vector value stored in column `col`. Numerical values are stored as
// uint32 with an order-preserving encoding applied at the wrapper boundary
// (signed/float -> monotone uint32), so the engine only ever does unsigned
// uint32 comparisons here.
enum DNFOp : uint8_t { DNF_EQ = 0, DNF_LT = 1, DNF_LE = 2, DNF_GT = 3, DNF_GE = 4 };

static inline bool DNFEvalOp(uint8_t op, uint32_t a, uint32_t b) {
    switch (op) {
        case DNF_EQ: return a == b;
        case DNF_LT: return a <  b;
        case DNF_LE: return a <= b;
        case DNF_GT: return a >  b;
        case DNF_GE: return a >= b;
        default:     return false;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Quantized Numeric Signature (range pruning via the signature layer).
//
// A numeric attribute is range-queryable, which a set-membership bitmap cannot
// express directly. We bridge that gap by QUANTIZING each numeric column into
// NUM_QUANT_BITS uniform buckets over its [lo,hi] domain and treating the bucket
// id as a discrete signature symbol. A posting's NumericQuantMask is the OR of
// its member vectors' buckets (per column); a range predicate sets every bucket
// overlapping the query interval, then the usual bitmap intersection prunes
// postings whose members all fall outside the range. The boundary bucket may
// admit false positives, which the exact per-vector post-filter removes — so the
// pre-filter is conservative (no false negatives), exactly like the categorical
// signatures. Multiple numeric columns each get their own 256-bit lane, so the
// mask is M*NUM_QUANT_WORDS uint64 (variable M, stored in head_node_meta).
// ═══════════════════════════════════════════════════════════════════
static constexpr int NUM_QUANT_BITS  = 256;
static constexpr int NUM_QUANT_WORDS = NUM_QUANT_BITS / 64;  // 4

// Per-column quantization domain [lo, hi] (order-preserving uint32 values).
struct NumQuantParam { uint32_t lo = 0; uint32_t hi = 0; };

// Map a value to its bucket id in [0, NUM_QUANT_BITS). Clamped at the ends so an
// out-of-domain value lands in the first/last bucket (still no false negatives).
static inline int NumQuantBucket(const NumQuantParam& p, uint32_t v) {
    if (p.hi <= p.lo) return 0;
    if (v <= p.lo) return 0;
    if (v >= p.hi) return NUM_QUANT_BITS - 1;
    uint64_t b = (uint64_t)(v - p.lo) * (uint64_t)NUM_QUANT_BITS / (uint64_t)(p.hi - p.lo);
    return (b >= NUM_QUANT_BITS) ? (NUM_QUANT_BITS - 1) : (int)b;
}

// Set bit `bucket` in column `col` of a flat M*NUM_QUANT_WORDS uint64 mask.
static inline void NumQuantInsert(uint64_t* mask, int col, int bucket) {
    uint64_t* lane = mask + (size_t)col * NUM_QUANT_WORDS;
    lane[bucket >> 6] |= (1ULL << (bucket & 63));
}

// Bucket range [blo, bhi] that a range predicate (op, val) may touch. Returns
// false if the predicate selects nothing representable.
static inline bool NumQuantPredBuckets(const NumQuantParam& p, uint8_t op, uint32_t val,
                                       int& blo, int& bhi) {
    int bv = NumQuantBucket(p, val);
    switch (op) {
        case DNF_EQ: blo = bv; bhi = bv; return true;
        case DNF_LT: case DNF_LE: blo = 0;  bhi = bv; return true;
        case DNF_GT: case DNF_GE: blo = bv; bhi = NUM_QUANT_BITS - 1; return true;
        default: blo = 0; bhi = NUM_QUANT_BITS - 1; return true;
    }
}

// True iff column `col` of `mask` has ANY set bit in bucket range [blo, bhi].
static inline bool NumQuantAnyInRange(const uint64_t* mask, int col, int blo, int bhi) {
    if (blo < 0) blo = 0;
    if (bhi > NUM_QUANT_BITS - 1) bhi = NUM_QUANT_BITS - 1;
    if (blo > bhi) return false;
    const uint64_t* lane = mask + (size_t)col * NUM_QUANT_WORDS;
    for (int w = blo >> 6; w <= (bhi >> 6); ++w) {
        int lo = (w == (blo >> 6)) ? (blo & 63) : 0;
        int hi = (w == (bhi >> 6)) ? (bhi & 63) : 63;
        uint64_t m = (hi - lo == 63) ? ~0ULL : (((1ULL << (hi - lo + 1)) - 1) << lo);
        if (lane[w] & m) return true;
    }
    return false;
}

// ═══════════════════════════════════════════════════════════════════
// DNF (disjunctive normal form) predicate over per-vector tags.
//   predicate = OR of clauses;  clause = AND of literals;
//   literal   = (col, val)  meaning  vector.tag[col] == val.
// `col` is the hierarchy level / tag column (0=org,1=dept,2=team,3=project).
// A vector with m_numTagsPerVec tag columns matches the predicate iff it
// satisfies ANY clause (all that clause's literals).
//
// Pre-filter helpers (MayMatch*) are conservative (no false negatives): a
// posting/page is kept iff SOME clause has all its literal values present in
// the corresponding signature. If a vector truly satisfies a clause, all of
// that clause's literal values are tags on the vector, so its signature bits
// are set — hence the test never drops a matching posting/page.
// ═══════════════════════════════════════════════════════════════════
// A literal is either CATEGORICAL (kind=0: a tag-column equality, the historical
// default) or NUMERIC (kind=1: a range comparison on a numeric attribute).
//   - kind==0: `col` indexes the categorical tag columns (0=org,1=dept,...); the
//     tag is one logical ACL attribute physically split across these columns.
//     These drive the categorical (hier/flat) signatures.
//   - kind==1: `col` ALSO indexes the per-vector tag columns -- numeric attributes
//     are stored inline as additional tag columns (col >= numBaseCols) holding the
//     RAW order-preserving value. The exact post-filter reads vecTags[col] for both
//     kinds; the only difference is `op` (range vs equality). For signature pruning
//     a numeric literal is matched against the QUANTIZED numeric signature, not the
//     categorical bitmaps.
// op defaults to DNF_EQ so existing aggregate initialisers `{col, val}` keep
// their categorical meaning unchanged.
struct DNFLiteral { uint32_t col; uint32_t val; uint8_t op = DNF_EQ; uint8_t kind = 0; };

struct DNFClause { std::vector<DNFLiteral> lits; };

struct DNFPredicate {
    std::vector<DNFClause> clauses;

    bool Empty() const { return clauses.empty(); }

    void Clear() { clauses.clear(); }

    // True iff some clause is a real conjunction (>1 literal). For pure-OR
    // predicates (every clause a single literal) the flat union of literal
    // values is logically EQUIVALENT to the DNF, so the coarse union mask used
    // to admit head-graph candidates can never admit a non-matching vector --
    // i.e. there is no "head leak" and the exact-DNF result drop pass is a
    // pure no-op that would only risk discarding legitimately-matching heads.
    bool HasAndClause() const {
        for (const auto& c : clauses)
            if (c.lits.size() > 1) return true;
        return false;
    }

    // True iff any literal references a numeric attribute (kind==1). Such a
    // predicate cannot be lowered to the legacy flat-OR (categorical equality)
    // path because its values are ranges, not tag equalities.
    bool HasNumericLiteral() const {
        for (const auto& c : clauses)
            for (const auto& l : c.lits)
                if (l.kind != 0) return true;
        return false;
    }

    // All distinct CATEGORICAL (tag, kind==0) literal values across every clause
    // (union) -- used to build the coarse union signatures, routing and the
    // selectivity estimate. Numeric literals are excluded: their values are raw
    // numeric values, not tag ids, and must never enter a categorical bitmask.
    std::vector<uint32_t> AllValues() const {
        std::vector<uint32_t> v;
        for (const auto& c : clauses)
            for (const auto& l : c.lits) if (l.kind == 0) v.push_back(l.val);
        return v;
    }

    // Exact per-vector evaluation (post-filter). Both categorical and numeric
    // literals read the per-vector tag columns (numeric attributes are inlined as
    // extra tag columns holding the raw order-preserving value); the literal's
    // `op` selects equality vs range. Uniform, sidecar-free.
    bool Matches(const uint32_t* vecTags, int numTags) const {
        for (const auto& c : clauses) {
            if (c.lits.empty()) continue;
            bool all = true;
            for (const auto& l : c.lits) {
                if ((int)l.col >= numTags || !DNFEvalOp(l.op, vecTags[l.col], l.val)) { all = false; break; }
            }
            if (all) return true;
        }
        return false;
    }

    // ── Signature pre-filters (categorical only) ──────────────────────────
    // Conservative (no false negatives): a posting/page/head is kept iff SOME
    // clause has all its CATEGORICAL (kind==0) literal values present in the
    // corresponding signature. Numeric (kind==1) literals are never tested here
    // -- they are unconstrained at the categorical signature level and enforced
    // by the quantized numeric pre-filter (MayMatchHierQuant) and/or the exact
    // post-filter, so no posting is ever wrongly dropped.
    bool MayMatchPage(const PageBitmask& page) const {
        for (const auto& c : clauses) {
            if (c.lits.empty()) continue;
            bool all = true;
            for (const auto& l : c.lits)
                if (l.kind == 0 && !page.MayContain(l.val)) { all = false; break; }
            if (all) return true;
        }
        return false;
    }

    bool MayMatchPostingFlat(const PostingBitmask& ps) const {
        for (const auto& c : clauses) {
            if (c.lits.empty()) continue;
            bool all = true;
            for (const auto& l : c.lits)
                if (l.kind == 0 && !ps.MayContain(l.val)) { all = false; break; }
            if (all) return true;
        }
        return false;
    }

    bool MayMatchHier(const HierarchicalPostingMask& h) const {
        for (const auto& c : clauses) {
            if (c.lits.empty()) continue;
            bool all = true;
            for (const auto& l : c.lits)
                if (l.kind == 0 && !h.MayContain((int)l.col, l.val)) { all = false; break; }
            if (all) return true;
        }
        return false;
    }

    // Combined categorical + quantized-numeric posting pre-filter. A clause
    // passes iff EVERY one of its literals may match: categorical literals against
    // the hierarchical mask `h`, numeric literals against the posting's quantized
    // numeric mask `quant` (M*NUM_QUANT_WORDS uint64). `qp` holds the per-column
    // [lo,hi] domains; `numBaseCols` is the first numeric tag-column index (so a
    // numeric literal at absolute column `col` maps to quant lane col-numBaseCols).
    // Conservative: a numeric literal "may match" iff some bucket overlapping its
    // range is set. When `quant` is null (no numeric signature present) numeric
    // literals are treated as always-may-match (fail open).
    bool MayMatchHierQuant(const HierarchicalPostingMask& h,
                           const uint64_t* quant, int numQuantCols,
                           const NumQuantParam* qp, int numBaseCols) const {
        for (const auto& c : clauses) {
            if (c.lits.empty()) continue;
            bool all = true;
            for (const auto& l : c.lits) {
                if (l.kind == 0) {
                    if (!h.MayContain((int)l.col, l.val)) { all = false; break; }
                } else if (quant != nullptr && qp != nullptr) {
                    int lane = (int)l.col - numBaseCols;
                    if (lane < 0 || lane >= numQuantCols) continue;  // unknown col: fail open
                    int blo, bhi;
                    NumQuantPredBuckets(qp[lane], l.op, l.val, blo, bhi);
                    if (!NumQuantAnyInRange(quant, lane, blo, bhi)) { all = false; break; }
                }
            }
            if (all) return true;
        }
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
