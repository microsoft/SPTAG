// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_EXTRADYNAMICSEARCHER_H_
#define _SPTAG_SPANN_EXTRADYNAMICSEARCHER_H_

#include "inc/Helper/VectorSetReader.h"
#include "inc/Helper/AsyncFileReader.h"
#include "IExtraSearcher.h"
#include "ExtraStaticSearcher.h"
#include "inc/Core/Common/TruthSet.h"
#include "inc/Helper/KeyValueIO.h"
#include "inc/Helper/ConcurrentSet.h"
#include "inc/Core/Common/FineGrainedLock.h"
#include "inc/Core/Common/Checksum.h"
#include "PersistentBuffer.h"
#include "inc/Core/Common/PostingSizeRecord.h"
#include "inc/Core/Cache/PostingSignature.h"
#include "ExtraFileController.h"
#include "SlimVectorKV.h"
#include "RaBitQ2.h"
#include "PipePQ.h"
#include "PrimaryHeadCSR.h"
#include <chrono>
#include <cstdint>
#include <map>
#include <cmath>
#include <cstring>
#include <cstdio>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#include <climits>
#include <future>
#include <numeric>
#include <utility>
#include <random>
#include <fstream>
#include <sstream>
#include <queue>
#include <unordered_set>
#include <shared_mutex>
#include <atomic>
#include <limits>
#include "inc/Core/Common/IQuantizer.h"
#include "inc/Core/Common/DistanceUtils.h"
#ifndef _MSC_VER
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#ifdef SPDK
#include "ExtraSPDKController.h"
#endif

#ifdef ROCKSDB
#include "ExtraRocksDBController.h"
// enable rocksdb io_uring: parallel MultiRead so the L survivor-vector blob
// fetches in a single rerank MultiGet are submitted concurrently (the analog of
// a graph beam-width). On by default; set SPTAG_ROCKSDB_NO_IOURING=1 to force the
// serial FSRandomAccessFile::MultiRead fallback (A/B to isolate IO parallelism).
extern "C" bool RocksDbIOUringEnable() {
    static const bool off = []() {
        const char* e = std::getenv("SPTAG_ROCKSDB_NO_IOURING");
        return e && e[0] == '1';
    }();
    return !off;
}
#endif

namespace SPTAG::SPANN {
    template <typename ValueType>
    class ExtraDynamicSearcher : public IExtraSearcher
    {
        struct AppendPair
        {
            std::string BKTID;
            int headID;
            std::string posting;

            AppendPair(std::string p_BKTID = "", int p_headID = -1, std::string p_posting = "") : BKTID(p_BKTID), headID(p_headID), posting(p_posting) {}
            inline bool operator < (const AppendPair& rhs) const
            {
                return std::strcmp(BKTID.c_str(), rhs.BKTID.c_str()) < 0;
            }

            inline bool operator > (const AppendPair& rhs) const
            {
                return std::strcmp(BKTID.c_str(), rhs.BKTID.c_str()) > 0;
            }
        };

        class MergeAsyncJob : public Helper::ThreadPool::Job
        {
        private:
            VectorIndex* m_index;
            ExtraDynamicSearcher<ValueType>* m_extraIndex;
            SizeType headID;
            bool disableReassign;
            std::function<void()> m_callback;
        public:
            MergeAsyncJob(VectorIndex* headIndex, ExtraDynamicSearcher<ValueType>* extraIndex, SizeType headID, bool disableReassign, std::function<void()> p_callback)
                : m_index(headIndex), m_extraIndex(extraIndex), headID(headID), disableReassign(disableReassign), m_callback(std::move(p_callback)) {}

            ~MergeAsyncJob() {}
            inline void exec(IAbortOperation* p_abort) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot support job.exec(abort)!\n");
            }
            inline void exec(void* p_workSpace, IAbortOperation* p_abort) override {
                ErrorCode ret = m_extraIndex->MergePostings((ExtraWorkSpace*)p_workSpace, m_index, headID, !disableReassign);
                if (ret != ErrorCode::Success)
                    m_extraIndex->m_asyncStatus = ret;
                if (m_callback != nullptr) {
                    m_callback();
                }
            }
        };

        class SplitAsyncJob : public Helper::ThreadPool::Job
        {
        private:
            VectorIndex* m_index;
            ExtraDynamicSearcher<ValueType>* m_extraIndex;
            SizeType headID;
            bool disableReassign;
            std::function<void()> m_callback;
        public:
            SplitAsyncJob(VectorIndex* headIndex, ExtraDynamicSearcher<ValueType>* extraIndex, SizeType headID, bool disableReassign, std::function<void()> p_callback)
                : m_index(headIndex), m_extraIndex(extraIndex), headID(headID), disableReassign(disableReassign), m_callback(std::move(p_callback)) {}

            ~SplitAsyncJob() {}
            inline void exec(IAbortOperation* p_abort) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot support job.exec(abort)!\n");
            }
            inline void exec(void* p_workSpace, IAbortOperation* p_abort) override {
                ErrorCode ret = m_extraIndex->Split((ExtraWorkSpace*)p_workSpace, m_index, headID, !disableReassign);
                if (ret != ErrorCode::Success)
                    m_extraIndex->m_asyncStatus = ret;
                if (m_callback != nullptr) {
                    m_callback();
                }
            }
        };

        class ReassignAsyncJob : public Helper::ThreadPool::Job
        {
        private:
            VectorIndex* m_index;
            ExtraDynamicSearcher<ValueType>* m_extraIndex;
            std::shared_ptr<std::string> vectorInfo;
            SizeType HeadPrev;
            std::function<void()> m_callback;
        public:
            ReassignAsyncJob(VectorIndex* headIndex, ExtraDynamicSearcher<ValueType>* extraIndex,
                std::shared_ptr<std::string> vectorInfo, SizeType HeadPrev, std::function<void()> p_callback)
                : m_index(headIndex), m_extraIndex(extraIndex), vectorInfo(std::move(vectorInfo)), HeadPrev(HeadPrev), m_callback(std::move(p_callback)) {}

            ~ReassignAsyncJob() {}
            
            inline void exec(IAbortOperation* p_abort) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot support job.exec(abort)!\n");
            }

            void exec(void* p_workSpace, IAbortOperation* p_abort) override {
                ErrorCode ret = m_extraIndex->Reassign((ExtraWorkSpace*)p_workSpace, m_index, vectorInfo, HeadPrev);
                if (ret != ErrorCode::Success)
                    m_extraIndex->m_asyncStatus = ret;
                if (m_callback != nullptr) {
                    m_callback();
                }
            }
        };

        class SPDKThreadPool : public Helper::ThreadPool
        {
        public:
            void initSPDK(int numberOfThreads, ExtraDynamicSearcher<ValueType>* extraIndex) 
            {
                m_abort.SetAbort(false);
                for (int i = 0; i < numberOfThreads; i++)
                {
                    m_threads.emplace_back([this, extraIndex] {
                        Job *j;
                        ExtraWorkSpace workSpace;
                        extraIndex->InitWorkSpace(&workSpace);
                        while (get(j))
                        {
                            try 
                            {
                                j->exec(&workSpace, &m_abort);
                            }
                            catch (std::exception& e) {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "ThreadPool: exception in %s %s\n", typeid(*j).name(), e.what());
                            }
                            delete j;
                            currentJobs--;
                        }
                    });
                }
            }
        };

    private:
        std::shared_ptr<Helper::Concurrent::ConcurrentQueue<int>> m_freeWorkSpaceIds;
        std::atomic<int> m_workspaceCount = 0;

        Helper::Concurrent::ConcurrentPriorityQueue<AppendPair> m_asyncAppendQueue;
        std::mutex m_asyncAppendLock;

        std::shared_ptr<Helper::KeyValueIO> db;

        COMMON::VersionLabel* m_versionMap;
        Options* m_opt;

        std::mutex m_dataAddLock;

        std::mutex m_mergeLock;

        COMMON::FineGrainedRWLock m_rwLocks;

        COMMON::PostingSizeRecord m_postingSizes;

        // Unfilter-tail sidecar: per-head count of "filtered-visible" prefix.
        // Vectors at indices [0, m_postingPureCounts[h]) are tag-pure and scanned
        // by filtered queries. Vectors at [m_postingPureCounts[h], m_postingSizes[h])
        // are unfilter-only replicas, scanned ONLY by unfiltered queries.
        // Legacy / missing sidecar => pure_count = total_size (no tail, original behaviour).
        COMMON::PostingSizeRecord m_postingPureCounts;
        bool m_hasPostingPureCounts = false;

        COMMON::Checksum m_checkSum;
        COMMON::Dataset<ChecksumType> m_checkSums;

        IndexStats m_stat;

        std::shared_ptr<PersistentBuffer> m_wal;

        std::mutex m_runningLock;
        std::unordered_set<SizeType>m_splitList;

        Helper::Concurrent::ConcurrentMap<SizeType, SizeType> m_mergeList;
        std::shared_timed_mutex m_mergeListLock;

        // Explicit pure/tail updates require SPANN::Index to merge in the
        // owning bundle. Search only records candidates; checkpoint maintenance
        // performs the topology mutation under the index-level lock.
        std::atomic<bool> m_taggedMaintenance{false};
        std::mutex m_taggedMergeCandidatesLock;
        std::unordered_set<SizeType> m_taggedMergeCandidates;

        // Per-vector tags stored alongside posting data
        std::vector<uint32_t> m_vectorTags;  // [vid * m_numTagsPerVec + t]
        int m_numTagsPerVec = 0;
        int m_tagBytesPerVec = 0;  // m_numTagsPerVec * sizeof(uint32_t)

        // In-posting quantization: when SPTAG_INPOST_QUANT_BITS is set the posting
        // record stores a quantized code (m_inpostPackedBytes) in place of the full
        // ValueType vector, shrinking m_vectorInfoSize and per-query posting IO with
        // ZERO extra resident memory. Distance is computed by ADC over the code
        // (InpostL2). The on-disk postings are rewritten once by QuantizeInPostings()
        // (env SPTAG_INPOST_QUANT_BUILD=1), guarded by the inpost_quant.bin marker.
        int m_inpostQuantBits = 0;     // 0 = off; else bits/dim (currently 4)
        int m_inpostPackedBytes = 0;   // (dim*bits+7)/8

        // In-posting RaBitQ (b1): the posting record stores a 1-bit RaBitQ code
        // (m_inpostRbqBinBytes) in place of the full ValueType vector. The screen
        // estimate is computed from the in-posting code (zero resident codes); the
        // top-L survivors are exact-reranked by cold O_DIRECT reads from the
        // full-precision base file (vid-indexed, NEVER page-cache resident).
        // Postings are rewritten once by
        // TransformInPostingsRbq() (env SPTAG_INPOST_RBQ_BUILD=1), guarded by the
        // inpost_rbq.bin marker. Enabled by env SPTAG_INPOST_RBQ=1.
        bool m_inpostRbq = false;
        int m_inpostRbqBinBytes = 0;
        int m_inpostRbqExBytes = 0;          // ex code bytes per vec (b>=2); 0 for b1
        std::string m_inpostRbqFile = "rabitq2_b1.bin";  // code sidecar (env SPTAG_INPOST_RBQ_FILE)
        int m_inpostRerankL = 30;
        std::shared_ptr<RaBitQ2> m_inpostRbq2;   // meta-only: rotator+centroid, NO resident codes
        // Full-precision rerank base is NEVER page-cache resident: opened O_DIRECT and
        // read per-survivor (cold device read every time) via ReadBaseVecDirect().
        int m_inpostBaseFd = -1;                 // O_DIRECT fd (vid -> dim uint8), for cold rerank
        size_t m_inpostBaseN = 0;
        int m_inpostBaseDim = 0;

        // New vectors cannot be reranked from the immutable FullVectorFile. Keep their
        // normalized full-precision payloads in a small, append-only sidecar keyed by VID.
        struct DynamicVectorStoreHeader
        {
            std::uint32_t magic = 0x53505556u; // SPUV
            std::uint32_t version = 1;
            std::uint32_t valueSize = 0;
            std::uint32_t dimension = 0;
            std::int64_t baseVID = -1;
            std::uint64_t slotCount = 0;
        };
        int m_dynamicVectorFd = -1;
        SizeType m_dynamicVectorBaseVID = -1;
        size_t m_dynamicVectorSlotCount = 0;
        std::string m_dynamicVectorPath;
        bool m_dynamicVectorWritable = false;
        mutable std::shared_mutex m_dynamicVectorLock;

        // Build-time slim (native): write [meta | RaBitQ-code] postings DIRECTLY during
        // the fresh build, never materializing the full-vector posting store. This is the
        // billion-scale path: the full ~1TB intermediate (replicas x full-vector records)
        // is never written; only the slim end-state (replicas x [meta|code]) hits disk.
        // Posting MEMBERSHIP is still full-stride-based (m_postingSizeLimit computed from
        // the full record), so the result is byte-compatible with the post-build transform
        // TransformInPostingsRbqContig (same members, slim records). The per-vector codes
        // come from a pre-encoded sidecar (rabitq2_encode_stream) mmap'd, indexed by VID.
        // Env: SPTAG_INPOST_RBQ=1 + SPTAG_INPOST_RBQ_BUILD_NATIVE=1 + SPTAG_INPOST_RBQ_FILE.
        bool   m_buildSlimRbq = false;
        void*  m_buildRbqMap = nullptr;
        size_t m_buildRbqMapSize = 0;
        const uint8_t* m_buildRbqCodes = nullptr;
        int    m_buildRbqCodeBytes = 0;
        int    m_buildRbqN = 0;
        int    m_buildSlimStride = 0;            // m_metaDataSize + codeBytes (slim record)
        int    m_quantFullVectorInfoSize = 0;    // full [meta|vector] stride, for build-time membership sizing
        std::string m_inpostRbqPathResolved;     // resolved code sidecar path, for BuildIndex slim writer setup

        // Build-time slim (native) for in-posting OPQ: same single-pass mechanism as the
        // RaBitQ path above, but the per-VID codes come from the raw opq_codes_m<M>.bin
        // sidecar (N*M uint8, vid-indexed, no header) instead of the rabitq2 stream. The
        // build writes [meta | M-byte OPQ code] postings directly, so the full-vector
        // posting store is never materialized (billion-scale path). Byte-compatible with
        // the post-build transform TransformInPostingsOpq (same members, slim records).
        bool   m_buildSlimOpq = false;
        void*  m_buildOpqMap = nullptr;
        size_t m_buildOpqMapSize = 0;
        const uint8_t* m_buildOpqCodes = nullptr;
        int    m_buildOpqM = 0;
        SizeType m_buildOpqN = 0;
        std::string m_opqCodesPathResolved;      // resolved opq_codes_m<M>.bin path, for BuildIndex slim writer setup

        // Build-time slim for PipeANN-style PQ: same [meta | M-byte code] posting
        // layout as OPQ, but the ADC LUT/codebook are PipeANN fixed-chunk PQ pivots.
        bool   m_buildSlimPipePQ = false;
        void*  m_buildPipePQMap = nullptr;
        size_t m_buildPipePQMapSize = 0;
        const uint8_t* m_buildPipePQCodes = nullptr;
        int    m_buildPipePQM = 0;
        SizeType m_buildPipePQN = 0;
        size_t m_buildPipePQCodeOffset = 0;      // 8 for PipeANN compressed.bin, 0 for raw N*M
        std::string m_pipePQCodesPathResolved;
        std::string m_pipePQPivotsPathResolved;

        // Page-selective directory: per posting, a 256-bit signature per 4KB page
        // (OR of the tags of the records whose bytes fall in that page). Used by
        // filtered queries (env SPTAG_PAGE_SELECT=1) to read only the pages that
        // may contain a queried tag instead of the whole posting. Built once from
        // the authoritative posting bytes and cached to page_signatures.bin.
        std::vector<std::vector<SPTAG::Cache::PageBitmask>> m_pagePS;
        std::atomic<int> m_pagePSState{0};  // 0=unbuilt, 1=building, 2=ready, -1=failed
        std::mutex m_pagePSMutex;

        // Phase 2: one-time within-posting reorder (env SPTAG_REORDER_POSTINGS=1).
        std::atomic<int> m_reorderState{0};  // 0=not done, 2=done, -1=failed
        std::mutex m_reorderMutex;


        // Dual-pool v3: head role sidecar (role==0: H1 filter+unfilter, role==1: U_extra unfilter-only)
        std::vector<uint8_t> m_headRole;
        bool m_hasHeadRole = false;

        // One nearest-head owner per vector. Optional sparse-filter sidecar.
        PrimaryHeadCSR m_primaryHeadCSR;

    public:
        void SetVectorTags(const uint32_t* tags, int numVecs, int numTagsPerVec) {
            m_numTagsPerVec = numTagsPerVec;
            m_tagBytesPerVec = numTagsPerVec * sizeof(uint32_t);
            m_vectorTags.assign(tags, tags + (size_t)numVecs * numTagsPerVec);
        }

        void SetNodeVectorAssignments(const std::vector<std::vector<SizeType>>& nodeVectorAssignments) {
            m_plannedNodeVectorAssignments = nodeVectorAssignments;
        }

        void SetPrimaryNodeVectorAssignments(const std::vector<std::vector<SizeType>>& primaryNodeVectorAssignments) {
            m_primaryNodeVectorAssignments = primaryNodeVectorAssignments;
        }

        void SetHeadVectorOwners(const std::unordered_map<SizeType, int>& headVectorOwners) {
            m_headVectorOwners = headVectorOwners;
        }

        bool HasPrimaryHeadCSR() const override { return m_primaryHeadCSR.Loaded(); }

        // Dual-pool v3: head role sidecar management
        void SetHeadRoles(const std::vector<uint8_t>& roles) {
            m_headRole = roles;
            m_hasHeadRole = !roles.empty();
        }
        void LoadHeadRole() {
            std::string path = m_opt->m_indexDirectory + FolderSep + m_opt->m_headRoleFile;
            if (!fileexists(path.c_str())) {
                m_hasHeadRole = false;
                return;
            }
            FILE* fp = fopen(path.c_str(), "rb");
            if (!fp) { m_hasHeadRole = false; return; }
            fseek(fp, 0, SEEK_END);
            long sz = ftell(fp);
            fseek(fp, 0, SEEK_SET);
            if (sz <= 0) { fclose(fp); m_hasHeadRole = false; return; }
            m_headRole.resize(static_cast<size_t>(sz));
            size_t nread = fread(m_headRole.data(), 1, static_cast<size_t>(sz), fp);
            fclose(fp);
            m_hasHeadRole = (nread == static_cast<size_t>(sz));
        }

        std::uint64_t PackPrimaryHeadAttributes(SizeType vid, const std::uint32_t tagBases[4]) const
        {
            const size_t tagOffset = static_cast<size_t>(vid) * static_cast<size_t>(m_numTagsPerVec);
            std::uint32_t packedTags = 0;
            for (int level = 0; level < 4; ++level) {
                const std::uint32_t rawTag = m_vectorTags[tagOffset + static_cast<size_t>(level)];
                const std::uint32_t localTag = rawTag - tagBases[level];
                packedTags |= (localTag & 0xffU) << (level * 8);
            }
            const std::uint32_t numeric = m_vectorTags[tagOffset + 4];
            return static_cast<std::uint64_t>(packedTags) |
                   (static_cast<std::uint64_t>(numeric) << 32);
        }

        bool WritePrimaryHeadCSR(Selection& selections,
                                 const std::unordered_map<SizeType, SizeType>& headVectorIDs,
                                 SizeType fullCount,
                                 SizeType headCount)
        {
            if (m_numTagsPerVec < 5 ||
                m_vectorTags.size() < static_cast<size_t>(fullCount) * static_cast<size_t>(m_numTagsPerVec)) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                             "[PrimaryHeadCSR] requires four categorical tags plus one numeric attribute.\n");
                return false;
            }

            if (selections.m_start != 0 || selections.m_end != selections.m_selections.size()) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                             "[PrimaryHeadCSR] batched selections are unsupported; use Batches=1.\n");
                return false;
            }

            std::uint32_t tagBases[4] = {
                std::numeric_limits<std::uint32_t>::max(),
                std::numeric_limits<std::uint32_t>::max(),
                std::numeric_limits<std::uint32_t>::max(),
                std::numeric_limits<std::uint32_t>::max()
            };
            for (SizeType vid = 0; vid < fullCount; ++vid) {
                const size_t tagOffset = static_cast<size_t>(vid) * static_cast<size_t>(m_numTagsPerVec);
                for (int level = 0; level < 4; ++level) {
                    tagBases[level] = std::min(tagBases[level], m_vectorTags[tagOffset + static_cast<size_t>(level)]);
                }
            }
            for (SizeType vid = 0; vid < fullCount; ++vid) {
                const size_t tagOffset = static_cast<size_t>(vid) * static_cast<size_t>(m_numTagsPerVec);
                for (int level = 0; level < 4; ++level) {
                    if (m_vectorTags[tagOffset + static_cast<size_t>(level)] - tagBases[level] > 0xffU) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                     "[PrimaryHeadCSR] categorical level %d exceeds uint8 range.\n", level);
                        return false;
                    }
                }
            }

            std::vector<SizeType> selfVIDs(static_cast<size_t>(headCount), MaxSize);
            if (m_opt->m_excludehead) {
                for (const auto& pair : headVectorIDs) {
                    if (pair.first >= 0 && pair.first < fullCount &&
                        pair.second >= 0 && pair.second < headCount) {
                        selfVIDs[static_cast<size_t>(pair.second)] = pair.first;
                    }
                }
            }

            std::vector<std::uint32_t> counts(static_cast<size_t>(headCount), 0);
            for (SizeType h = 0; h < headCount; ++h) {
                if (selfVIDs[static_cast<size_t>(h)] != MaxSize) {
                    ++counts[static_cast<size_t>(h)];
                }
            }
            for (const Edge& edge : selections.m_selections) {
                if (!std::signbit(edge.distance) || edge.node < 0 || edge.node >= headCount ||
                    edge.tonode < 0 || edge.tonode >= fullCount) {
                    continue;
                }
                if (m_opt->m_excludehead && headVectorIDs.find(edge.tonode) != headVectorIDs.end()) {
                    continue;
                }
                ++counts[static_cast<size_t>(edge.node)];
            }

            std::vector<std::uint32_t> offsets(static_cast<size_t>(headCount) + 1, 0);
            std::uint64_t entryCount = 0;
            for (SizeType h = 0; h < headCount; ++h) {
                entryCount += counts[static_cast<size_t>(h)];
                if (entryCount > std::numeric_limits<std::uint32_t>::max()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "[PrimaryHeadCSR] entry count exceeds uint32 offset capacity.\n");
                    return false;
                }
                offsets[static_cast<size_t>(h) + 1] = static_cast<std::uint32_t>(entryCount);
            }

            PrimaryHeadCSRHeader header;
            header.headCount = static_cast<std::uint32_t>(headCount);
            header.entryCount = entryCount;
            for (int level = 0; level < 4; ++level) header.tagBases[level] = tagBases[level];

            const std::string path = m_opt->m_indexDirectory + FolderSep + m_opt->m_primaryHeadCSRFile;
            std::ofstream output(path, std::ios::binary | std::ios::trunc);
            if (!output) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                             "[PrimaryHeadCSR] cannot create %s.\n", path.c_str());
                return false;
            }
            output.write(reinterpret_cast<const char*>(&header), sizeof(header));
            output.write(reinterpret_cast<const char*>(offsets.data()),
                         static_cast<std::streamsize>(offsets.size() * sizeof(std::uint32_t)));

            std::vector<PrimaryHeadCSREntry> writeBuffer;
            writeBuffer.reserve(1 << 20);
            auto appendEntry = [&](SizeType vid) {
                PrimaryHeadCSREntry entry;
                entry.vid = static_cast<std::uint32_t>(vid);
                entry.attributes = PackPrimaryHeadAttributes(vid, tagBases);
                writeBuffer.push_back(entry);
                if (writeBuffer.size() == writeBuffer.capacity()) {
                    output.write(reinterpret_cast<const char*>(writeBuffer.data()),
                                 static_cast<std::streamsize>(writeBuffer.size() * sizeof(PrimaryHeadCSREntry)));
                    writeBuffer.clear();
                }
            };

            size_t selectionPos = 0;
            std::uint64_t written = 0;
            for (SizeType h = 0; h < headCount; ++h) {
                while (selectionPos < selections.m_selections.size() &&
                       selections.m_selections[selectionPos].node < h) {
                    selections.m_selections[selectionPos].distance =
                        std::fabs(selections.m_selections[selectionPos].distance);
                    ++selectionPos;
                }
                if (selfVIDs[static_cast<size_t>(h)] != MaxSize) {
                    appendEntry(selfVIDs[static_cast<size_t>(h)]);
                    ++written;
                }
                while (selectionPos < selections.m_selections.size() &&
                       selections.m_selections[selectionPos].node == h) {
                    Edge& edge = selections.m_selections[selectionPos++];
                    if (std::signbit(edge.distance) && edge.tonode >= 0 && edge.tonode < fullCount &&
                        (!m_opt->m_excludehead || headVectorIDs.find(edge.tonode) == headVectorIDs.end())) {
                        appendEntry(edge.tonode);
                        ++written;
                    }
                    edge.distance = std::fabs(edge.distance);
                }
                if (written != offsets[static_cast<size_t>(h) + 1]) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "[PrimaryHeadCSR] head %d count mismatch: %llu vs %u.\n",
                                 static_cast<int>(h),
                                 static_cast<unsigned long long>(written),
                                 offsets[static_cast<size_t>(h) + 1]);
                    return false;
                }
            }
            while (selectionPos < selections.m_selections.size()) {
                selections.m_selections[selectionPos].distance =
                    std::fabs(selections.m_selections[selectionPos].distance);
                ++selectionPos;
            }
            if (!writeBuffer.empty()) {
                output.write(reinterpret_cast<const char*>(writeBuffer.data()),
                             static_cast<std::streamsize>(writeBuffer.size() * sizeof(PrimaryHeadCSREntry)));
            }
            output.close();
            if (!output || written != entryCount) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                             "[PrimaryHeadCSR] write failed for %s.\n", path.c_str());
                return false;
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                         "[PrimaryHeadCSR] wrote %llu entries across %d heads to %s.\n",
                         static_cast<unsigned long long>(entryCount),
                         static_cast<int>(headCount), path.c_str());
            return true;
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "DualPool: loaded head_role.bin (%zu heads, hasRole=%d)\n",
                m_headRole.size(), (int)m_hasHeadRole);
        }
        bool IsUnfilterOnlyHead(int headOrd) const override {
            if (!m_hasHeadRole || headOrd < 0 || headOrd >= (int)m_headRole.size()) return false;
            return m_headRole[headOrd] == 1;
        }
        bool HasHeadRoles() const override { return m_hasHeadRole; }

        // --- Unfilter-tail sidecar accessors -----------------------------------
        // Returns the number of records to scan for a query with the given
        // filter mode. For filtered queries we return pure_count (skip tail);
        // for unfiltered we return total_size (scan everything).
        // Defensive: clamp to total to handle stale / corrupt sidecars.
        inline int GetScanLimit(const SizeType& headID, bool unfiltered) {
            int total = m_postingSizes.GetSize(headID);
            if (unfiltered || !m_hasPostingPureCounts) return total;
            int pure = m_postingPureCounts.GetSize(headID);
            return (pure <= 0 || pure > total) ? total : pure;
        }
        inline int GetPureCount(const SizeType& headID) {
            int total = m_postingSizes.GetSize(headID);
            if (!m_hasPostingPureCounts) return total;
            int pure = m_postingPureCounts.GetSize(headID);
            return (pure <= 0 || pure > total) ? total : pure;
        }
        inline bool HasPostingPureCounts() const { return m_hasPostingPureCounts; }
        // Used by the build path (PerTagBKT head selection) to write the sidecar.
        inline void SetPureCount(const SizeType& headID, int pure_count) {
            m_postingPureCounts.UpdateSize(headID, pure_count);
        }
        // Initialize pure_count = total_size for every head (no-tail fallback).
        // Safe to call repeatedly.
        void InitializePureCountsFromTotals(SizeType numHeads) {
            m_postingPureCounts.Initialize(numHeads, m_opt->m_datasetRowsInBlock, m_opt->m_datasetCapacity);
            for (SizeType h = 0; h < numHeads; ++h) {
                m_postingPureCounts.UpdateSize(h, m_postingSizes.GetSize(h));
            }
            m_hasPostingPureCounts = true;
        }
        // Try to load the sidecar; if absent or unreadable, fall back to totals.
        void LoadOrInitPostingPureCounts() {
            const std::string baseDir = m_opt->m_recovery
                ? m_opt->m_persistentBufferPath
                : m_opt->m_indexDirectory;
            std::string path = baseDir + FolderSep + m_opt->m_postingPureCountsFile;
            if (fileexists(path.c_str())) {
                if (m_postingPureCounts.Load(path, m_opt->m_datasetRowsInBlock, m_opt->m_datasetCapacity) == ErrorCode::Success) {
                    m_hasPostingPureCounts = true;
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loaded posting_pure_counts sidecar from %s (numHeads=%d).\n",
                                 path.c_str(), m_postingPureCounts.GetPostingNum());
                    return;
                }
                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "Failed to load %s; falling back to pure=total.\n", path.c_str());
            }
            InitializePureCountsFromTotals(m_postingSizes.GetPostingNum());
            m_hasPostingPureCounts = false;  // not loaded => legacy mode; tail effectively absent
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "posting_pure_counts sidecar not present; using pure=total fallback.\n");
        }
        void SavePostingPureCounts() {
            if (!m_hasPostingPureCounts) return;
            std::string path = m_opt->m_indexDirectory + FolderSep + m_opt->m_postingPureCountsFile;
            m_postingPureCounts.Save(path);
        }
        // -----------------------------------------------------------------------


        ExtraDynamicSearcher(SPANN::Options& p_opt) {
            m_opt = &p_opt;
            m_numTagsPerVec = p_opt.m_numTagsPerVec;
            m_tagBytesPerVec = m_numTagsPerVec * sizeof(uint32_t);
            m_metaDataSize = sizeof(int) + sizeof(uint8_t) + m_tagBytesPerVec;
            m_vectorInfoSize = p_opt.m_dim * sizeof(ValueType) + m_metaDataSize;
            // In-posting quantization: shrink the posting record to a quantized code.
            // Both build (transform) and search modes size the posting store to the
            // quantized stride; the full ValueType stride is only used transiently
            // inside QuantizeInPostings() as a local.
            {
                const char* qb = std::getenv("SPTAG_INPOST_QUANT_BITS");
                if (qb && *qb) {
                    int bits = std::atoi(qb);
                    if (bits > 0 && bits < 8 && sizeof(ValueType) == 1) {
                        m_inpostQuantBits = bits;
                        m_inpostPackedBytes = (p_opt.m_dim * bits + 7) / 8;
                        m_vectorInfoSize = m_metaDataSize + m_inpostPackedBytes;
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                            "[InpostQuant] bits=%d packedBytes=%d vectorInfoSize=%d (full=%d)\n",
                            bits, m_inpostPackedBytes, m_vectorInfoSize,
                            (int)(p_opt.m_dim * sizeof(ValueType) + m_metaDataSize));
                    } else {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                            "[InpostQuant] ignored: bits=%d valueTypeSize=%d (need 0<bits<8, uint8 data)\n",
                            bits, (int)sizeof(ValueType));
                    }
                }
            }
            // In-posting RaBitQ b1: size the record to [meta | b1-code]. Loads the
            // RaBitQ2 rotator/centroid meta-only (zero resident codes) to learn the
            // code byte width, and mmaps the full-precision base file for rerank.
            {
                if (Helper::StrUtils::StrEqualIgnoreCase(p_opt.m_postingQuantizer.c_str(), "RaBitQ") && sizeof(ValueType) == 1) {
                    std::string dir = p_opt.m_indexDirectory + FolderSep;
                    if (!p_opt.m_postingQuantFile.empty()) m_inpostRbqFile = p_opt.m_postingQuantFile;
                    // The sidecar may be given as an absolute path (build-time slim, where
                    // the codes are pre-encoded into a known location independent of the
                    // index work dir) or as a name relative to the index directory.
                    std::string rbqPath = (!m_inpostRbqFile.empty() && m_inpostRbqFile[0] == '/')
                        ? m_inpostRbqFile : (dir + m_inpostRbqFile);
                    auto store = std::make_shared<RaBitQ2>();
                    if (store->LoadMeta(rbqPath)) {
                        m_inpostRbq = true;
                        m_inpostRbq2 = store;
                        m_inpostRbqBinBytes = store->GetBinBytes();
                        m_inpostRbqExBytes = store->GetExBytes();
                        // The on-disk postings are always slim [meta|code]; size the search
                        // stride accordingly. During a fresh build, BuildIndex restores the
                        // full stride for membership and installs the slim writer
                        // (SetupBuildSlimRbq) using m_inpostRbqPathResolved.
                        m_quantFullVectorInfoSize = p_opt.m_dim * sizeof(ValueType) + m_metaDataSize;
                        m_vectorInfoSize = m_metaDataSize + m_inpostRbqBinBytes + m_inpostRbqExBytes;
                        m_inpostRbqPathResolved = rbqPath;
                        if (p_opt.m_rerankL > 0) m_inpostRerankL = p_opt.m_rerankL;
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                            "[InpostRBQ] %s binBytes=%d exBytes=%d vectorInfoSize=%d (full=%d) rerankL=%d\n",
                            rbqPath.c_str(), m_inpostRbqBinBytes, m_inpostRbqExBytes, m_vectorInfoSize,
                            (int)m_quantFullVectorInfoSize, m_inpostRerankL);
                        // Open the full-precision base O_DIRECT for cold per-survivor
                        // rerank reads. It is NEVER mmap'd / page-cache resident, so every
                        // rerank vector fetch is a device read (apples-to-apples with
                        // PipeANN's on-disk rerank).
                        const char* bp = p_opt.m_fullVectorFile.empty() ? nullptr : p_opt.m_fullVectorFile.c_str();
                        if (bp && *bp) {
                            int hfd = open(bp, O_RDONLY);
                            if (hfd >= 0) {
                                int32_t h[2] = { 0, 0 };
                                if (pread(hfd, h, 8, 0) == 8) {
                                    m_inpostBaseN = (size_t)h[0];
                                    m_inpostBaseDim = (int)h[1];
                                }
                                close(hfd);
#ifdef O_DIRECT
                                m_inpostBaseFd = open(bp, O_RDONLY | O_DIRECT);
#else
                                m_inpostBaseFd = open(bp, O_RDONLY);
#endif
                                if (m_inpostBaseFd >= 0) {
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                        "[InpostRBQ] O_DIRECT base %s N=%zu dim=%d (cold rerank, no residency)\n",
                                        bp, m_inpostBaseN, m_inpostBaseDim);
                                } else {
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[InpostRBQ] O_DIRECT base open failed %s\n", bp);
                                }
                            } else {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[InpostRBQ] open base failed %s\n", bp);
                            }
                        }
                        // Prefer the canonical RocksDB vid->vector store for rerank: a single
                        // batched async MultiGet (libaio) over the L survivors instead of L
                        // serial O_DIRECT preads. With SPTAG_ROCKSDB_DIRECT_IO=1 (block cache 0)
                        // it is still a cold device read (no residency) but PARALLEL. The
                        // O_DIRECT flat base above remains the fallback when no vecstore exists.
#ifdef ROCKSDB
                        {
                            std::string vdir = dir + "opq_vecstore";
                            struct stat vst;
                            if (stat(vdir.c_str(), &vst) == 0) {
                                m_opqVecDB.reset(new RocksDBIO(vdir.c_str(), false, false, false, true));
                                if (m_opqVecDB->Available()) {
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                        "[InpostRBQ] rerank via RocksDB vecstore MultiGet (batched async cold)\n");
                                } else {
                                    m_opqVecDB.reset();
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                        "[InpostRBQ] vecstore open failed; rerank falls back to O_DIRECT flat base\n");
                                }
                            }
                        }
#endif
                    } else {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                            "[InpostRBQ] SPTAG_INPOST_RBQ=1 but %s meta load failed\n", m_inpostRbqFile.c_str());
                    }
                }
            }
            // In-posting OPQ (DB-resident): the slim [meta | M-byte OPQ code] records
            // live IN the posting store db and are read via the SAME async/batched
            // MultiGet the baseline uses (FileIO libaio) -- NOT the serial opq_slim.bin
            // mmap. Sizing the posting stride to the slim record here; the full
            // ValueType stride is only a transient local in TransformInPostingsOpq().
            // Enabled by env SPTAG_OPQ_INPOST_DB=<M> (M = OPQ subvector/code-byte count).
            {
                if (Helper::StrUtils::StrEqualIgnoreCase(p_opt.m_postingQuantizer.c_str(), "OPQ") && sizeof(ValueType) == 1) {
                    int M = p_opt.m_postingQuantM;
                    if (M > 0) {
                        m_opqInpostDb = true;
                        m_opqInpostDbM = M;
                        m_vectorInfoSize = m_metaDataSize + M;
                        // Build-time slim (native): BuildIndex restores the full stride for
                        // membership and installs the slim writer (SetupBuildSlimOpq), which
                        // reads the per-VID codes from opq_codes_m<M>.bin. Resolve that path
                        // now: an absolute PostingQuantizerFile overrides; otherwise the code
                        // sidecar lives in the index directory under its canonical name.
                        m_quantFullVectorInfoSize = p_opt.m_dim * sizeof(ValueType) + m_metaDataSize;
                        if (!p_opt.m_postingQuantFile.empty() && p_opt.m_postingQuantFile[0] == '/') {
                            m_opqCodesPathResolved = p_opt.m_postingQuantFile;
                        } else {
                            char codeName[64];
                            snprintf(codeName, sizeof(codeName), "opq_codes_m%d.bin", M);
                            m_opqCodesPathResolved = p_opt.m_indexDirectory + FolderSep + codeName;
                        }
                        if (p_opt.m_rerankL > 0) m_inpostRerankL = p_opt.m_rerankL;
                        // Open the flat O_DIRECT base so OPQ rerank uses the SAME deep-queue
                        // libaio path as RaBitQ (fair estimator comparison at equal IO).
                        EnsureInpostBaseFd();
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                            "[InpostOPQ-DB] M=%d vectorInfoSize=%d (full=%d): slim codes in posting store, async MultiGet scan\n",
                            M, m_vectorInfoSize, (int)(p_opt.m_dim * sizeof(ValueType) + m_metaDataSize));
                    }
                }
            }
            // In-posting PipeANN-style PQ: same DB-resident slim layout as OPQ, with
            // PipeANN fixed-chunk PQ pivots driving the per-query ADC LUT. The code sidecar
            // may be raw N*M bytes or PipeANN's compressed.bin format [uint32 N][uint32 M].
            {
                const bool wantPipePQ =
                    (Helper::StrUtils::StrEqualIgnoreCase(p_opt.m_postingQuantizer.c_str(), "PipePQ") ||
                     Helper::StrUtils::StrEqualIgnoreCase(p_opt.m_postingQuantizer.c_str(), "PQ"));
                if (wantPipePQ) {
                    int M = p_opt.m_postingQuantM;
                    if (M > 0) {
                        auto resolveInIndex = [&](const std::string& path) -> std::string {
                            if (path.empty()) return path;
                            if (path[0] == '/') return path;
                            return p_opt.m_indexDirectory + FolderSep + path;
                        };
                        auto looksLikePivots = [](const std::string& path) -> bool {
                            return path.find("pivot") != std::string::npos ||
                                   path.find("PIVOT") != std::string::npos;
                        };
                        m_pipePQ = true;
                        m_opqInpostDb = true;   // reuse the async [meta|code] scan/rerank path
                        m_opqInpostDbM = M;
                        m_opqM = M;
                        m_vectorInfoSize = m_metaDataSize + M;
                        m_quantFullVectorInfoSize = p_opt.m_dim * sizeof(ValueType) + m_metaDataSize;

                        if (!p_opt.m_pipePQPivotsFile.empty()) {
                            m_pipePQPivotsPathResolved = resolveInIndex(p_opt.m_pipePQPivotsFile);
                            if (!p_opt.m_postingQuantFile.empty()) {
                                m_pipePQCodesPathResolved = resolveInIndex(p_opt.m_postingQuantFile);
                            } else {
                                char codeName[64];
                                snprintf(codeName, sizeof(codeName), "pipepq_codes_m%d.bin", M);
                                m_pipePQCodesPathResolved = p_opt.m_indexDirectory + FolderSep + codeName;
                            }
                        } else if (!p_opt.m_postingQuantFile.empty() && looksLikePivots(p_opt.m_postingQuantFile)) {
                            m_pipePQPivotsPathResolved = resolveInIndex(p_opt.m_postingQuantFile);
                            char codeName[64];
                            snprintf(codeName, sizeof(codeName), "pipepq_codes_m%d.bin", M);
                            m_pipePQCodesPathResolved = p_opt.m_indexDirectory + FolderSep + codeName;
                        } else {
                            if (!p_opt.m_postingQuantFile.empty()) {
                                m_pipePQCodesPathResolved = resolveInIndex(p_opt.m_postingQuantFile);
                            } else {
                                char codeName[64];
                                snprintf(codeName, sizeof(codeName), "pipepq_codes_m%d.bin", M);
                                m_pipePQCodesPathResolved = p_opt.m_indexDirectory + FolderSep + codeName;
                            }
                            if (const char* piv = std::getenv("SPTAG_PIPEPQ_PIVOTS")) {
                                m_pipePQPivotsPathResolved = piv;
                            } else {
                                m_pipePQPivotsPathResolved = p_opt.m_indexDirectory + FolderSep + "pipepq_pivots.bin";
                            }
                        }

                        if (p_opt.m_rerankL > 0) m_inpostRerankL = p_opt.m_rerankL;
                        EnsureInpostBaseFd();
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                            "[InpostPipePQ-DB] M=%d vectorInfoSize=%d (full=%d): pivots=%s codes=%s\n",
                            M, m_vectorInfoSize, (int)(p_opt.m_dim * sizeof(ValueType) + m_metaDataSize),
                            m_pipePQPivotsPathResolved.c_str(), m_pipePQCodesPathResolved.c_str());
                    }
                }
            }
            p_opt.m_searchPostingPageLimit = p_opt.m_postingPageLimit;
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Setting index with posting page limit:%d\n", p_opt.m_postingPageLimit);
            m_postingSizeLimit = p_opt.m_postingPageLimit * PageSize / m_vectorInfoSize;
            m_bufferSizeLimit = p_opt.m_bufferLength * PageSize / m_vectorInfoSize;
            // ini is the single source of truth: unfilter-tail extra buffer pages come
            // only from the native SSD param UnfilterTailBufferLength (no env override).
            m_tailBufferSizeLimit = p_opt.m_unfilterTailBufferLength * PageSize / m_vectorInfoSize;

            if(p_opt.m_storage == Storage::FILEIO) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "ExtraDynamicSearcher:UseFileIO\n");
                db.reset(new FileIO(p_opt));
            }
            else if (p_opt.m_storage == Storage::SPDKIO) {
#ifdef SPDK
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "ExtraDynamicSearcher:UseSPDK\n");
                db.reset(new SPDKIO(p_opt));
#else
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "ExtraDynamicSearcher:SPDK unsupport! Use -DSPDK to enable SPDK when doing cmake.\n");
                return;
#endif
            } 
            else if (p_opt.m_storage == Storage::ROCKSDBIO) {
#ifdef ROCKSDB
                if (p_opt.m_shareDB && p_opt.m_externalDB) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "ExtraDynamicSearcher:UseSharedRocksDB\n");
                    db = p_opt.m_externalDB;
                } else {
                    std::string indexDir = (p_opt.m_recovery)? p_opt.m_persistentBufferPath + FolderSep: p_opt.m_indexDirectory + FolderSep;
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "ExtraDynamicSearcher:UseKV\n");
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "ExtraDynamicSearcher:dbPath:%s\n", (indexDir + p_opt.m_KVFile).c_str());
                    db.reset(new RocksDBIO((indexDir + p_opt.m_KVFile).c_str(), p_opt.m_useDirectIO, p_opt.m_enableWAL, p_opt.m_recovery));
                }
#else
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "ExtraDynamicSearcher:RocksDB unsupport! Use -DROCKSDB to enable RocksDB when doing cmake.\n");
                return;
#endif
            }

            m_mergeThreshold = p_opt.m_mergeThreshold;
            m_checkSum.Initialize(!p_opt.m_checksumCheck, 0, 0);

            int maxIOThreads =  max(p_opt.m_ioThreads, (2 * max(p_opt.m_searchThreadNum, p_opt.m_iSSDNumberOfThreads) +
                                    p_opt.m_insertThreadNum + p_opt.m_reassignThreadNum + p_opt.m_appendThreadNum));
            m_freeWorkSpaceIds.reset(new Helper::Concurrent::ConcurrentQueue<int>());
            for (int i = 0; i < maxIOThreads; i++) {
                m_freeWorkSpaceIds->push(i);
            }
            m_workspaceCount = maxIOThreads;
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Posting size limit: %d, search limit: %f, merge threshold: %d\n", m_postingSizeLimit, p_opt.m_latencyLimit, m_mergeThreshold);
        }

        ~ExtraDynamicSearcher() {
#ifndef _MSC_VER
            if (m_inpostBaseFd >= 0) close(m_inpostBaseFd);
            std::unique_lock<std::shared_mutex> lock(m_dynamicVectorLock);
            if (m_dynamicVectorFd >= 0) close(m_dynamicVectorFd);
#endif
        }

        std::shared_ptr<Helper::KeyValueIO> GetKVStore() override { return db; }

        virtual bool Available() override
        {
            return db->Available();
        }

        //headCandidates: search data structrue for "vid" vector
        //headID: the head vector that stands for vid
        bool IsAssumptionBroken(VectorIndex* p_index, SizeType headID, ValueType* vector, SizeType vid)
        {
            COMMON::QueryResultSet<ValueType> headCandidates(vector, m_opt->m_reassignK);
            std::shared_ptr<std::uint8_t> rec_query;
            if (p_index->m_pQuantizer) {
                rec_query.reset((uint8_t*)ALIGN_ALLOC(p_index->m_pQuantizer->ReconstructSize()), [=](std::uint8_t* ptr) { ALIGN_FREE(ptr); });
                p_index->m_pQuantizer->ReconstructVector((const uint8_t*)headCandidates.GetTarget(), rec_query.get());
                headCandidates.SetTarget((ValueType*)(rec_query.get()), p_index->m_pQuantizer);
            }
            p_index->SearchIndex(headCandidates);
            int replicaCount = 0;
            BasicResult* queryResults = headCandidates.GetResults();
            std::vector<Edge> selections(static_cast<size_t>(m_opt->m_replicaCount));
            for (int i = 0; i < headCandidates.GetResultNum() && replicaCount < m_opt->m_replicaCount; ++i) {
                if (queryResults[i].VID == -1) {
                    break;
                }
                // RNG Check.
                bool rngAccpeted = true;
                for (int j = 0; j < replicaCount; ++j) {
                    float nnDist = p_index->ComputeDistance(
                        p_index->GetSample(queryResults[i].VID),
                        p_index->GetSample(selections[j].node));
                    if (nnDist < queryResults[i].Dist) {
                        rngAccpeted = false;
                        break;
                    }
                }
                if (!rngAccpeted)
                    continue;

                selections[replicaCount].node = queryResults[i].VID;
                // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "head:%d\n", queryResults[i].VID);
                if (selections[replicaCount].node == headID) return false;
                ++replicaCount;
            }
            return true;
        }

        //Measure that in "headID" posting list, how many vectors break their assumption
        int QuantifyAssumptionBroken(VectorIndex* p_index, SizeType headID, std::string& postingList, SizeType SplitHead, std::vector<SizeType>& newHeads, std::set<int>& brokenID, int topK = 0, float ratio = 1.0)
        {
            int assumptionBrokenNum = 0;
            int postVectorNum = postingList.size() / m_vectorInfoSize;
            uint8_t* postingP = reinterpret_cast<uint8_t*>(postingList.data());
            float minDist;
            float maxDist;
            float avgDist = 0;
            std::vector<float> distanceSet;

            for (int j = 0; j < postVectorNum; j++) {
                uint8_t* vectorId = postingP + j * m_vectorInfoSize;
                SizeType vid = *(reinterpret_cast<int*>(vectorId));
                uint8_t version = *(reinterpret_cast<uint8_t*>(vectorId + sizeof(int)));
                float_t dist = p_index->ComputeDistance(reinterpret_cast<ValueType*>(vectorId + m_metaDataSize), p_index->GetSample(headID));
                // if (dist < Epsilon) SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "head found: vid: %d, head: %d\n", vid, headID);
                avgDist += dist;
                distanceSet.push_back(dist);
                if (m_versionMap->Deleted(vid) || m_versionMap->GetVersion(vid) != version) continue;
                
                if (brokenID.find(vid) == brokenID.end() && IsAssumptionBroken(p_index, headID, reinterpret_cast<ValueType*>(vectorId + m_metaDataSize), vid)) {
                    /*
                    float_t headDist = p_index->ComputeDistance(headCandidates.GetTarget(), p_index->GetSample(SplitHead));
                    float_t newHeadDist_1 = p_index->ComputeDistance(headCandidates.GetTarget(), p_index->GetSample(newHeads[0]));
                    float_t newHeadDist_2 = p_index->ComputeDistance(headCandidates.GetTarget(), p_index->GetSample(newHeads[1]));

                    float_t splitDist = p_index->ComputeDistance(p_index->GetSample(SplitHead), p_index->GetSample(headID));

                    float_t headToNewHeadDist_1 = p_index->ComputeDistance(p_index->GetSample(headID), p_index->GetSample(newHeads[0]));
                    float_t headToNewHeadDist_2 = p_index->ComputeDistance(p_index->GetSample(headID), p_index->GetSample(newHeads[1]));

                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "broken vid to head distance: %f, to split head distance: %f\n", dist, headDist);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "broken vid to new head 1 distance: %f, to new head 2 distance: %f\n", newHeadDist_1, newHeadDist_2);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "head to spilit head distance: %f\n", splitDist);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "head to new head 1 distance: %f, to new head 2 distance: %f\n", headToNewHeadDist_1, headToNewHeadDist_2);
                    */
                    assumptionBrokenNum++;
                    brokenID.insert(vid);
                }
            }

            if (assumptionBrokenNum != 0) {
                std::sort(distanceSet.begin(), distanceSet.end());
                minDist = distanceSet[1];
                maxDist = distanceSet.back();
                // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "distance: min: %f, max: %f, avg: %f, 50th: %f\n", minDist, maxDist, avgDist/postVectorNum, distanceSet[distanceSet.size() * 0.5]);
                // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "assumption broken num: %d\n", assumptionBrokenNum);
                float_t splitDist = p_index->ComputeDistance(p_index->GetSample(SplitHead), p_index->GetSample(headID));

                float_t headToNewHeadDist_1 = p_index->ComputeDistance(p_index->GetSample(headID), p_index->GetSample(newHeads[0]));
                float_t headToNewHeadDist_2 = p_index->ComputeDistance(p_index->GetSample(headID), p_index->GetSample(newHeads[1]));

                // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "head to spilt head distance: %f/%d/%.2f\n", splitDist, topK, ratio);
                // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "head to new head 1 distance: %f, to new head 2 distance: %f\n", headToNewHeadDist_1, headToNewHeadDist_2);
            }

            return assumptionBrokenNum;
        }

        int QuantifySplitCaseA(VectorIndex* p_index, std::vector<SizeType>& newHeads, std::vector<std::string>& postingLists, SizeType SplitHead, int split_order, std::set<int>& brokenID)
        {
            int assumptionBrokenNum = 0;
            assumptionBrokenNum += QuantifyAssumptionBroken(p_index, newHeads[0], postingLists[0], SplitHead, newHeads, brokenID);
            assumptionBrokenNum += QuantifyAssumptionBroken(p_index, newHeads[1], postingLists[1], SplitHead, newHeads, brokenID);
            int vectorNum = (postingLists[0].size() + postingLists[1].size()) / m_vectorInfoSize;
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "After Split%d, Top0 nearby posting lists, caseA : %d/%d\n", split_order, assumptionBrokenNum, vectorNum);
            return assumptionBrokenNum;
        }

        //Measure that around "headID", how many vectors break their assumption
        //"headID" is the head vector before split
        void QuantifySplitCaseB(ExtraWorkSpace* p_exWorkSpace, VectorIndex* p_index, SizeType headID, std::vector<SizeType>& newHeads, SizeType SplitHead, int split_order, int assumptionBrokenNum_top0, std::set<int>& brokenID)
        {
            COMMON::QueryResultSet<ValueType> nearbyHeads((ValueType*)(p_index->GetSample(headID)), m_opt->m_reassignK);
            std::shared_ptr<std::uint8_t> rec_query;
            if (p_index->m_pQuantizer) {
                rec_query.reset((uint8_t*)ALIGN_ALLOC(p_index->m_pQuantizer->ReconstructSize()), [=](std::uint8_t* ptr) { ALIGN_FREE(ptr); });
                p_index->m_pQuantizer->ReconstructVector((const uint8_t*)nearbyHeads.GetTarget(), rec_query.get());
                nearbyHeads.SetTarget((ValueType*)(rec_query.get()), p_index->m_pQuantizer);
            }
            p_index->SearchIndex(nearbyHeads);
            std::vector<std::string> postingLists;
            std::string postingList;
            BasicResult* queryResults = nearbyHeads.GetResults();
            int topk = 8;
            int assumptionBrokenNum = assumptionBrokenNum_top0;
            int assumptionBrokenNum_topK = assumptionBrokenNum_top0;
            int i;
            int containedHead = 0;
            if (assumptionBrokenNum_top0 != 0) containedHead++;
            int vectorNum = 0;
            float furthestDist = 0;
            for (i = 0; i < nearbyHeads.GetResultNum(); i++) {
                if (queryResults[i].VID == -1) {
                    break;
                }
                furthestDist = queryResults[i].Dist;
                if (i == topk) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "After Split%d, Top%d nearby posting lists, caseB : %d in %d/%d\n", split_order, i, assumptionBrokenNum, containedHead, vectorNum);
                    topk *= 2;
                }
                if (queryResults[i].VID == newHeads[0] || queryResults[i].VID == newHeads[1]) continue;
                db->Get(queryResults[i].VID, &postingList, MaxTimeout, &(p_exWorkSpace->m_diskRequests));
                vectorNum += postingList.size() / m_vectorInfoSize;
                int tempNum = QuantifyAssumptionBroken(p_index, queryResults[i].VID, postingList, SplitHead, newHeads, brokenID, i, queryResults[i].Dist / queryResults[1].Dist);
                assumptionBrokenNum += tempNum;
                if (tempNum != 0) containedHead++;
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "After Split%d, Top%d nearby posting lists, caseB : %d in %d/%d\n", split_order, i, assumptionBrokenNum, containedHead, vectorNum);
        }

        void QuantifySplit(ExtraWorkSpace* p_exWorkSpace, VectorIndex* p_index, SizeType headID, std::vector<std::string>& postingLists, std::vector<SizeType>& newHeads, SizeType SplitHead, int split_order)
        {
            std::set<int> brokenID;
            brokenID.clear();
            // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Split Quantify: %d, head1:%d, head2:%d\n", split_order, newHeads[0], newHeads[1]);
            int assumptionBrokenNum = QuantifySplitCaseA(p_index, newHeads, postingLists, SplitHead, split_order, brokenID);
            QuantifySplitCaseB(p_exWorkSpace, p_index, headID, newHeads, SplitHead, split_order, assumptionBrokenNum, brokenID);
        }

        bool CheckIsNeedReassign(VectorIndex* p_index, std::vector<SizeType>& newHeads, ValueType* data, SizeType splitHead, float_t headToSplitHeadDist, float_t currentHeadDist, bool isInSplitHead, SizeType currentHead)
        {

            float_t splitHeadDist = p_index->ComputeDistance(data, p_index->GetSample(splitHead));

            if (isInSplitHead) {
                if (splitHeadDist >= currentHeadDist) return false;
            }
            else {
                float_t newHeadDist_1 = p_index->ComputeDistance(data, p_index->GetSample(newHeads[0]));
                float_t newHeadDist_2 = p_index->ComputeDistance(data, p_index->GetSample(newHeads[1]));
                if (splitHeadDist <= newHeadDist_1 && splitHeadDist <= newHeadDist_2) return false;
                if (currentHeadDist <= newHeadDist_1 && currentHeadDist <= newHeadDist_2) return false;
            }
            return true;
        }

        inline void Serialize(char* ptr, SizeType VID, std::uint8_t version, const void* vector) {
            memcpy(ptr, &VID, sizeof(VID));
            memcpy(ptr + sizeof(VID), &version, sizeof(version));
            // Write per-vector tags if available
            if (m_tagBytesPerVec > 0 && VID >= 0 && (size_t)VID * m_numTagsPerVec < m_vectorTags.size()) {
                memcpy(ptr + sizeof(VID) + sizeof(version),
                       &m_vectorTags[(size_t)VID * m_numTagsPerVec],
                       m_tagBytesPerVec);
            } else if (m_tagBytesPerVec > 0) {
                memset(ptr + sizeof(VID) + sizeof(version), 0, m_tagBytesPerVec);
            }
            memcpy(ptr + m_metaDataSize, vector, m_vectorInfoSize - m_metaDataSize);
        }

        bool SerializeDynamicPosting(char* ptr, SizeType VID, std::uint8_t version,
                                     const ValueType* vector, const std::uint32_t* tags,
                                     int numTagsPerVec)
        {
            if (tags == nullptr || numTagsPerVec != m_numTagsPerVec) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                             "[TaggedUpdate] expected %d tags per vector, got %d.\n",
                             m_numTagsPerVec, numTagsPerVec);
                return false;
            }

            memcpy(ptr, &VID, sizeof(VID));
            memcpy(ptr + sizeof(VID), &version, sizeof(version));
            if (m_tagBytesPerVec > 0) {
                memcpy(ptr + sizeof(VID) + sizeof(version), tags, m_tagBytesPerVec);
            }

            char* payload = ptr + m_metaDataSize;
            if (m_pipePQ) {
                if (!m_pipePQTable || m_pipePQTable->Dim() != m_opt->m_dim ||
                    m_opqInpostDbM != m_pipePQTable->Chunks()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "[TaggedUpdate] PipePQ table is not initialized for online encoding.\n");
                    return false;
                }
                std::vector<float> values(static_cast<size_t>(m_opt->m_dim));
                for (int d = 0; d < m_opt->m_dim; ++d) {
                    values[static_cast<size_t>(d)] = static_cast<float>(vector[d]);
                }
                m_pipePQTable->Encode(values.data(), reinterpret_cast<std::uint8_t*>(payload));
                return true;
            }

            if (m_opqInpostDb || m_inpostRbq || m_inpostQuantBits > 0) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                             "[TaggedUpdate] online encoding is implemented for PipePQ only.\n");
                return false;
            }

            memcpy(payload, vector, m_vectorInfoSize - m_metaDataSize);
            return true;
        }

        // Build-time slim setup: mmap the pre-encoded RaBitQ2 sidecar (rabitq2_encode_stream
        // output) and expose its per-VID code region. Same on-disk layout as the post-build
        // transform reads (TransformInPostingsRbqContig): header 7xint32 [magic,N,dim,pdim,
        // ex_bits,rotator_type,rotator_bytes], then rotator dump, then float32 centroid_rot
        // [pdim], then per-vec [bin|ex]. The slim record written into postings is
        // [meta | bin|ex] of stride m_buildSlimStride.
        bool SetupBuildSlimRbq(const std::string& rbqPath) {
            if (m_inpostRbqBinBytes <= 0) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[InpostRBQ-native] meta not loaded, cannot slim-build\n");
                return false;
            }
            int cfd = open(rbqPath.c_str(), O_RDONLY);
            if (cfd < 0) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[InpostRBQ-native] open sidecar %s fail\n", rbqPath.c_str());
                return false;
            }
            off_t csz = lseek(cfd, 0, SEEK_END);
            void* cmap = mmap(nullptr, (size_t)csz, PROT_READ, MAP_SHARED, cfd, 0);
            close(cfd);
            if (cmap == MAP_FAILED) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[InpostRBQ-native] mmap sidecar fail\n");
                return false;
            }
            const int32_t* h = reinterpret_cast<const int32_t*>(cmap);
            int32_t N = h[1], pdim = h[3], rbytes = h[6];
            int codeBytes = m_inpostRbqBinBytes + m_inpostRbqExBytes;
            size_t codeBase = (size_t)7 * 4 + (size_t)rbytes + (size_t)pdim * sizeof(float);
            m_buildRbqMap = cmap;
            m_buildRbqMapSize = (size_t)csz;
            m_buildRbqCodes = reinterpret_cast<const uint8_t*>(cmap) + codeBase;
            m_buildRbqCodeBytes = codeBytes;
            m_buildRbqN = N;
            m_buildSlimStride = m_metaDataSize + codeBytes;
            m_buildSlimRbq = true;
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "[InpostRBQ-native] build-time slim ENABLED: N=%d codeBytes=%d slimStride=%d (full record kept for membership)\n",
                N, codeBytes, m_buildSlimStride);
            return true;
        }

        // Slim posting record writer used by the build-time-slim path: writes
        // [id | version | tags | RaBitQ-code(VID)] of stride m_buildSlimStride. The code is
        // fetched from the mmapped sidecar by global VID. Mirrors Serialize's meta layout.
        inline void SerializeSlim(char* ptr, SizeType VID, std::uint8_t version) {
            memcpy(ptr, &VID, sizeof(VID));
            memcpy(ptr + sizeof(VID), &version, sizeof(version));
            if (m_tagBytesPerVec > 0 && VID >= 0 && (size_t)VID * m_numTagsPerVec < m_vectorTags.size()) {
                memcpy(ptr + sizeof(VID) + sizeof(version),
                       &m_vectorTags[(size_t)VID * m_numTagsPerVec],
                       m_tagBytesPerVec);
            } else if (m_tagBytesPerVec > 0) {
                memset(ptr + sizeof(VID) + sizeof(version), 0, m_tagBytesPerVec);
            }
            if (VID >= 0 && VID < m_buildRbqN) {
                memcpy(ptr + m_metaDataSize,
                       m_buildRbqCodes + (size_t)VID * m_buildRbqCodeBytes,
                       m_buildRbqCodeBytes);
            } else {
                memset(ptr + m_metaDataSize, 0, m_buildRbqCodeBytes);
            }
        }

        // Build-time slim setup for in-posting OPQ: mmap the raw opq_codes_m<M>.bin sidecar
        // (N*M uint8, vid-indexed, no header) and expose its per-VID code region. The slim
        // record written into postings is [meta | M-byte OPQ code] of stride
        // m_buildSlimStride. mmap (not resident) so the billion-scale codes (e.g. 1B*25 =
        // 25GB) page in on demand.
        bool SetupBuildSlimOpq(int M) {
            if (M <= 0) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[InpostOPQ-native] invalid M=%d, cannot slim-build\n", M);
                return false;
            }
            int cfd = open(m_opqCodesPathResolved.c_str(), O_RDONLY);
            if (cfd < 0) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[InpostOPQ-native] open codes %s fail\n", m_opqCodesPathResolved.c_str());
                return false;
            }
            off_t csz = lseek(cfd, 0, SEEK_END);
            void* cmap = mmap(nullptr, (size_t)csz, PROT_READ, MAP_SHARED, cfd, 0);
            close(cfd);
            if (cmap == MAP_FAILED) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[InpostOPQ-native] mmap codes fail\n");
                return false;
            }
            m_buildOpqMap = cmap;
            m_buildOpqMapSize = (size_t)csz;
            m_buildOpqCodes = reinterpret_cast<const uint8_t*>(cmap);
            m_buildOpqM = M;
            m_buildOpqN = (SizeType)((size_t)csz / (size_t)M);
            m_buildSlimStride = m_metaDataSize + M;
            m_buildSlimOpq = true;
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "[InpostOPQ-native] build-time slim ENABLED: codes=%s N=%d M=%d slimStride=%d (full record kept for membership)\n",
                m_opqCodesPathResolved.c_str(), (int)m_buildOpqN, M, m_buildSlimStride);
            return true;
        }

        // Slim posting record writer for the OPQ build-time-slim path: writes
        // [id | version | tags | OPQ-code(VID)] of stride m_buildSlimStride. The code is
        // fetched from the mmapped sidecar by global VID. Mirrors SerializeSlim's meta layout.
        inline void SerializeSlimOpq(char* ptr, SizeType VID, std::uint8_t version) {
            memcpy(ptr, &VID, sizeof(VID));
            memcpy(ptr + sizeof(VID), &version, sizeof(version));
            if (m_tagBytesPerVec > 0 && VID >= 0 && (size_t)VID * m_numTagsPerVec < m_vectorTags.size()) {
                memcpy(ptr + sizeof(VID) + sizeof(version),
                       &m_vectorTags[(size_t)VID * m_numTagsPerVec],
                       m_tagBytesPerVec);
            } else if (m_tagBytesPerVec > 0) {
                memset(ptr + sizeof(VID) + sizeof(version), 0, m_tagBytesPerVec);
            }
            if (VID >= 0 && VID < m_buildOpqN) {
                memcpy(ptr + m_metaDataSize,
                       m_buildOpqCodes + (size_t)VID * m_buildOpqM,
                       m_buildOpqM);
            } else {
                memset(ptr + m_metaDataSize, 0, m_buildOpqM);
            }
        }

        bool MmapPipePQCodes(const std::string& path, int M, void*& map, size_t& mapSize,
                             const std::uint8_t*& codes, SizeType& n, size_t& codeOffset,
                             const char* label)
        {
            if (M <= 0) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[%s] invalid M=%d\n", label, M);
                return false;
            }
            int cfd = open(path.c_str(), O_RDONLY);
            if (cfd < 0) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[%s] open codes %s fail\n", label, path.c_str());
                return false;
            }
            off_t csz = lseek(cfd, 0, SEEK_END);
            if (csz <= 0) {
                close(cfd);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[%s] empty codes %s\n", label, path.c_str());
                return false;
            }
            void* cmap = mmap(nullptr, (size_t)csz, PROT_READ, MAP_SHARED, cfd, 0);
            close(cfd);
            if (cmap == MAP_FAILED) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[%s] mmap codes fail\n", label);
                return false;
            }
            const auto* base = reinterpret_cast<const std::uint8_t*>(cmap);
            size_t offset = 0;
            SizeType rows = (SizeType)((size_t)csz / (size_t)M);
            if (csz >= 8) {
                std::uint32_t hdrN = 0, hdrM = 0;
                std::memcpy(&hdrN, base, sizeof(hdrN));
                std::memcpy(&hdrM, base + sizeof(hdrN), sizeof(hdrM));
                if (hdrM == (std::uint32_t)M &&
                    (size_t)csz == 8 + (size_t)hdrN * (size_t)M) {
                    offset = 8;
                    rows = (SizeType)hdrN;
                }
            }
            if (offset == 0 && ((size_t)csz % (size_t)M) != 0) {
                munmap(cmap, (size_t)csz);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "[%s] codes %s size %zu is neither raw N*M nor PipeANN [N,M]+codes for M=%d\n",
                    label, path.c_str(), (size_t)csz, M);
                return false;
            }
            map = cmap;
            mapSize = (size_t)csz;
            codes = base + offset;
            n = rows;
            codeOffset = offset;
            return true;
        }

        bool SetupBuildSlimPipePQ(int M) {
            if (!MmapPipePQCodes(m_pipePQCodesPathResolved, M, m_buildPipePQMap,
                                 m_buildPipePQMapSize, m_buildPipePQCodes,
                                 m_buildPipePQN, m_buildPipePQCodeOffset,
                                 "InpostPipePQ-native")) {
                return false;
            }
            m_buildPipePQM = M;
            m_buildSlimStride = m_metaDataSize + M;
            m_buildSlimPipePQ = true;
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "[InpostPipePQ-native] build-time slim ENABLED: codes=%s N=%d M=%d offset=%zu slimStride=%d\n",
                m_pipePQCodesPathResolved.c_str(), (int)m_buildPipePQN, M,
                m_buildPipePQCodeOffset, m_buildSlimStride);
            return true;
        }

        inline void SerializeSlimPipePQ(char* ptr, SizeType VID, std::uint8_t version) {
            memcpy(ptr, &VID, sizeof(VID));
            memcpy(ptr + sizeof(VID), &version, sizeof(version));
            if (m_tagBytesPerVec > 0 && VID >= 0 && (size_t)VID * m_numTagsPerVec < m_vectorTags.size()) {
                memcpy(ptr + sizeof(VID) + sizeof(version),
                       &m_vectorTags[(size_t)VID * m_numTagsPerVec],
                       m_tagBytesPerVec);
            } else if (m_tagBytesPerVec > 0) {
                memset(ptr + sizeof(VID) + sizeof(version), 0, m_tagBytesPerVec);
            }
            if (VID >= 0 && VID < m_buildPipePQN) {
                memcpy(ptr + m_metaDataSize,
                       m_buildPipePQCodes + (size_t)VID * m_buildPipePQM,
                       m_buildPipePQM);
            } else {
                memset(ptr + m_metaDataSize, 0, m_buildPipePQM);
            }
        }

        void CalculatePostingDistribution(VectorIndex* p_index)
        {
            int top = m_postingSizeLimit / 10 + 1;
            int page = m_opt->m_postingPageLimit + 1;
            std::vector<int> lengthDistribution(top, 0);
            std::vector<int> sizeDistribution(page + 2, 0);
            int deletedHead = 0;
            for (int i = 0; i < p_index->GetNumSamples(); i++) {
                if (!p_index->ContainSample(i)) deletedHead++;
                lengthDistribution[m_postingSizes.GetSize(i) / 10]++;
                int size = m_postingSizes.GetSize(i) * m_vectorInfoSize;
                if (size < PageSize) {
                    if (size < 512) sizeDistribution[0]++;
                    else if (size < 1024) sizeDistribution[1]++;
                    else sizeDistribution[2]++;
                }
                else {
                    sizeDistribution[size / PageSize + 2]++;
                }
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Posting Length (Vector Num):\n");
            for (int i = 0; i < top; ++i)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "%d ~ %d: %d, \n", i * 10, (i + 1) * 10 - 1, lengthDistribution[i]);
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Posting Length (Data Size):\n");
            for (int i = 0; i < page + 2; ++i)
            {
                if (i <= 2) {
                    if (i == 0) SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "0 ~ 512 B: %d, \n", sizeDistribution[0] - deletedHead);
                    else if (i == 1) SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "512 B ~ 1 KB: %d, \n", sizeDistribution[1]);
                    else SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "1 KB ~ 4 KB: %d, \n", sizeDistribution[2]);
                }
                else
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "%d ~ %d KB: %d, \n", (i - 2) * 4, (i - 1) * 4, sizeDistribution[i]);
            }
        }

        void PrintErrorInPosting(std::string &posting, SizeType headID)
        {
            SizeType postVectorNum = posting.size() / m_vectorInfoSize;
            uint8_t *vectorId = reinterpret_cast<uint8_t *>(posting.data());
            for (int j = 0; j < postVectorNum; j++, vectorId += m_vectorInfoSize)
            {
                SizeType VID = *((SizeType *)(vectorId));
                if (VID < 0 || VID >= m_versionMap->Count())
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "PrintErrorInPosting found wrong VID:%d in headID:%d (should be less than %d)\n", VID,
                                 headID, m_versionMap->Count());
                }
            }
        }

        // TODO
        ErrorCode RefineIndex(std::shared_ptr<VectorIndex>& p_index,
                              bool p_prereassign, std::vector<SizeType> *p_headmapping, std::vector<SizeType> *p_mapping) override
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Begin RefineIndex\n");

            COMMON::PostingSizeRecord new_postingSizes;
            COMMON::Dataset<ChecksumType> new_checkSums;
            if (!p_prereassign)
            {
                new_postingSizes.Initialize(p_index->GetNumSamples() - p_index->GetNumDeleted(),
                            p_index->m_iDataBlockSize, p_index->m_iDataCapacity);
                new_checkSums.Initialize(p_index->GetNumSamples() - p_index->GetNumDeleted(), 1,
                                            p_index->m_iDataBlockSize, p_index->m_iDataCapacity);
            }
            std::atomic_bool doneReassign = false;
            Helper::Concurrent::ConcurrentSet<SizeType> mergelist;
            while (!doneReassign) {
                auto preReassignTimeBegin = std::chrono::high_resolution_clock::now();
                std::atomic<ErrorCode> finalcode = ErrorCode::Success;
                doneReassign = true;
                std::vector<std::thread> threads;
                std::atomic_int nextPostingID(0);
                int currentPostingNum = p_index->GetNumSamples();
                int limit = m_postingSizeLimit * m_opt->m_preReassignRatio;
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Batch PreReassign, Current PostingNum: %d, Current Limit: %d\n", currentPostingNum, limit);
                auto func = [&]()
                {
                    ErrorCode ret;
                    int index = 0;
                    ExtraWorkSpace workSpace;
                    InitWorkSpace(&workSpace);
                    while (true)
                    {
                        index = nextPostingID.fetch_add(1);
                        if (index < currentPostingNum)
                        {
                            if ((index & ((1 << 14) - 1)) == 0)
                            {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Sent %.2lf%%...\n", index * 100.0 / currentPostingNum);
                            }
                            if (p_prereassign)
                            {
                                if (m_postingSizes.GetSize(index) >= limit)
                                {
                                    doneReassign = false;
                                    Split(&workSpace, p_index.get(), index, false, p_prereassign);
                                }
                            }
                            else
                            {
                                if (!p_index->ContainSample(index))
                                    continue;

                                // ForceCompaction
                                std::string postingList;
                                if ((ret = db->Get(index, &postingList, MaxTimeout, &(workSpace.m_diskRequests))) !=
                                        ErrorCode::Success ||
                                    !m_checkSum.ValidateChecksum(postingList.c_str(), (int)(postingList.size()),
                                                                *m_checkSums[index]))
                                {
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                                 "RefineIndex failed to get posting %d, required size:%d, read size:%d "
                                                 "checksum issue:%d\n",
                                                 index, (int)(m_postingSizes.GetSize(index) * m_vectorInfoSize),
                                                 (int)(postingList.size()), (int)(ret == ErrorCode::Success));
                                    PrintErrorInPosting(postingList, index);
                                    finalcode = ErrorCode::Fail;
                                    //return;
                                }
                                SizeType postVectorNum = (SizeType)(postingList.size() / m_vectorInfoSize);
                                auto *postingP = reinterpret_cast<uint8_t *>(postingList.data());
                                uint8_t *vectorId = postingP;
                                int vectorCount = 0;
                                for (int j = 0; j < postVectorNum;
                                     j++, vectorId += m_vectorInfoSize)
                                {
                                    uint8_t version = *(vectorId + sizeof(SizeType));
                                    SizeType VID = *((SizeType *)(vectorId));

                                    if (m_versionMap->Deleted(VID) || m_versionMap->GetVersion(VID) != version)
                                        continue;

                                    *((SizeType *)(vectorId)) = (*p_mapping)[VID];
                                    if (j != vectorCount)
                                    {
                                        memcpy(postingP + vectorCount * m_vectorInfoSize, vectorId, m_vectorInfoSize);
                                    }
                                    vectorCount++;
                                }

                                if (vectorCount <= m_mergeThreshold) mergelist.insert(p_headmapping->at(index));

                                postingList.resize(vectorCount * m_vectorInfoSize);
                                if ((ret = db->Put(p_headmapping->at(index), postingList, MaxTimeout,
                                                       &(workSpace.m_diskRequests))) !=
                                    ErrorCode::Success)
                                {
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                                    "RefineIndex Failed to write back compacted posting\n");
                                    finalcode = ret;
                                    return;
                                }
                                new_postingSizes.UpdateSize(p_headmapping->at(index), vectorCount);
                                *new_checkSums[p_headmapping->at(index)] =
                                    m_checkSum.CalcChecksum(postingList.c_str(), (int)(postingList.size()));
                                if (m_opt->m_consistencyCheck && (ret = db->Check(p_headmapping->at(index), new_postingSizes.GetSize(p_headmapping->at(index)) * m_vectorInfoSize, nullptr)) != ErrorCode::Success)
                                {
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                                    "RefineIndex: Check failed after Put %d\n",
                                                    p_headmapping->at(index));
                                    finalcode = ret;
                                    return;
                                }
                            }
                        }
                        else
                        {
                            return;
                        }
                    }
                };
                for (int j = 0; j < m_opt->m_iSSDNumberOfThreads; j++) { threads.emplace_back(func); }
                for (auto& thread : threads) { thread.join(); }
                auto preReassignTimeEnd = std::chrono::high_resolution_clock::now();
                double elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(preReassignTimeEnd - preReassignTimeBegin).count();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "rebuild cost: %.2lf s\n", elapsedSeconds);

                if (finalcode != ErrorCode::Success)
                    return finalcode;

                if (p_prereassign)
                {
                    Checkpoint(m_opt->m_indexDirectory);
                    p_index->SaveIndex(m_opt->m_indexDirectory + FolderSep + m_opt->m_headIndexFolder);
                    CalculatePostingDistribution(p_index.get());
                }
                else
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Saving posting size\n");
                    std::string p_persistenRecord = m_opt->m_indexDirectory + FolderSep + m_opt->m_ssdInfoFile;
                    new_postingSizes.Save(p_persistenRecord);
                    std::string p_checksumPath = m_opt->m_indexDirectory + FolderSep + m_opt->m_checksumFile;
                    new_checkSums.Save(p_checksumPath);
                    db->Checkpoint(m_opt->m_indexDirectory);

                    if ((finalcode = m_postingSizes.Load(p_persistenRecord, p_index->m_iDataBlockSize,
                                                         p_index->m_iDataCapacity)) != ErrorCode::Success)
                        return finalcode;
                    if ((finalcode = m_checkSums.Load(p_checksumPath, p_index->m_iDataBlockSize,
                                                      p_index->m_iDataCapacity)) != ErrorCode::Success)
                        return finalcode;
                    
                    if ((finalcode = m_versionMap->Load(m_opt->m_indexDirectory + FolderSep + m_opt->m_deleteIDFile,
                                                        p_index->m_iDataBlockSize, p_index->m_iDataCapacity)) !=
                        ErrorCode::Success)
                        return finalcode;
                    if ((finalcode = m_vectorTranslateMap->Load(
                             m_opt->m_indexDirectory + FolderSep + m_opt->m_headIDFile, p_index->m_iDataBlockSize,
                             p_index->m_iDataCapacity)) != ErrorCode::Success)
                        return finalcode;
                    if ((finalcode =
                             VectorIndex::LoadIndex(m_opt->m_indexDirectory + FolderSep + m_opt->m_headIndexFolder,
                                                    p_index)) != ErrorCode::Success)
                        return finalcode;

                    if (mergelist.size() > 0)
                    {
                        for (SizeType pid : mergelist)
                        {
                            MergeAsync(p_index.get(), pid);
                        }
                        Checkpoint(m_opt->m_indexDirectory);
                        p_index->SaveIndex(m_opt->m_indexDirectory + FolderSep + m_opt->m_headIndexFolder);
                        m_vectorTranslateMap->Save(m_opt->m_indexDirectory + FolderSep + m_opt->m_headIDFile);
                    }
                }
                
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPFresh: ReWriting SSD Info\n");
            }
            return ErrorCode::Success;
        }
        
        ErrorCode Split(ExtraWorkSpace* p_exWorkSpace, VectorIndex* p_index, const SizeType headID, bool reassign = false, bool preReassign = false, bool requirelock = true)
        {
            auto splitBegin = std::chrono::high_resolution_clock::now();
            std::vector<SizeType> newHeadsID;
            std::vector<std::string> newPostingLists;
            ErrorCode ret;
            bool theSameHead = false;
            double elapsedMSeconds;
            {
                std::unique_lock<std::shared_timed_mutex> lock(m_rwLocks[headID], std::defer_lock);
                if (requirelock) lock.lock();

                int retry = 0;
             Retry:
                if (!p_index->ContainSample(headID)) return ErrorCode::Success;

                std::string postingList;
                auto splitGetBegin = std::chrono::high_resolution_clock::now();
                if ((ret=db->Get(headID, &postingList, MaxTimeout, &(p_exWorkSpace->m_diskRequests))) !=
                    ErrorCode::Success || !m_checkSum.ValidateChecksum(postingList.c_str(), (int)(postingList.size()), *m_checkSums[headID]))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "Split fail to get oversized postings: key=%d required size=%d read size=%d checksum "
                                 "issue=%d\n",
                                 headID, (int)(m_postingSizes.GetSize(headID) * m_vectorInfoSize),
                                 (int)(postingList.size()), (int)(ret == ErrorCode::Success));
                    return ret;
                }
                auto splitGetEnd = std::chrono::high_resolution_clock::now();
                elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(splitGetEnd - splitGetBegin).count();
                m_stat.m_getCost += elapsedMSeconds;
                // reinterpret postingList to vectors and IDs
                auto* postingP = reinterpret_cast<uint8_t*>(postingList.data());
                SizeType postVectorNum = (SizeType)(postingList.size() / m_vectorInfoSize);
               
                //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "DEBUG: db get Posting %d successfully with length %d real length:%d vectorNum:%d\n", headID, (int)(postingList.size()), m_postingSizes.GetSize(headID), postVectorNum);
                COMMON::Dataset<ValueType> smallSample(postVectorNum, m_opt->m_dim, p_index->m_iDataBlockSize, p_index->m_iDataCapacity, (ValueType*)postingP, true, nullptr, m_metaDataSize, m_vectorInfoSize);
                //COMMON::Dataset<ValueType> smallSample(0, m_opt->m_dim, p_index->m_iDataBlockSize, p_index->m_iDataCapacity);  // smallSample[i] -> VID
                //std::vector<int> localIndicesInsert(postVectorNum);  // smallSample[i] = j <-> localindices[j] = i
                //std::vector<uint8_t> localIndicesInsertVersion(postVectorNum);
                std::vector<int> localIndices;
                localIndices.reserve(postVectorNum);
                uint8_t* vectorId = postingP;
                for (int j = 0; j < postVectorNum; j++, vectorId += m_vectorInfoSize)
                {
                    //LOG(Helper::LogLevel::LL_Info, "vector index/total:id: %d/%d:%d\n", j, m_postingSizes[headID].load(), *(reinterpret_cast<int*>(vectorId)));
                    uint8_t version = *(vectorId + sizeof(int));
                    int VID = *((int*)(vectorId));
                    if (VID < 0 || VID >= m_versionMap->Count())
                    {
                        if (retry < 3)
                        {
                            retry++;
                            goto Retry;
                        }
                        else
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                         "Split fail: Get posting %d fail after 3 times retries.\n", headID);
                            return ErrorCode::DiskIOFail;
                        }
                    }
                        
		    //if (VID >= m_versionMap->Count()) SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "DEBUG: vector ID:%d total size:%d\n", VID, m_versionMap->Count());
                    if (m_versionMap->Deleted(VID) || m_versionMap->GetVersion(VID) != version) continue;

                    //localIndicesInsert[index] = VID;
                    //localIndicesInsertVersion[index] = version;
                    //smallSample.AddBatch(1, (ValueType*)(vectorId + m_metaDataSize));
                    localIndices.push_back(j);
                }
                // double gcEndTime = sw.getElapsedMs();
                // m_splitGcCost += gcEndTime;
		
                if (!preReassign && localIndices.size() < m_postingSizeLimit)
                {

                    //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "DEBUG: in place or not prereassign & index < m_postingSizeLimit. GC begin...\n");
                    char* ptr = (char*)(postingList.c_str());
                    for (int j = 0; j < localIndices.size(); j++, ptr += m_vectorInfoSize)
                    {
                        if (j == localIndices[j]) continue;
                        memcpy(ptr, postingList.c_str() + localIndices[j] * m_vectorInfoSize, m_vectorInfoSize);
                        //Serialize(ptr, localIndicesInsert[j], localIndicesInsertVersion[j], smallSample[j]);
                    }
                    postingList.resize(localIndices.size() * m_vectorInfoSize);
                    if ((ret=db->Put(headID, postingList, MaxTimeout, &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Split Fail to write back postings\n");
                        return ret;
                    }
                    m_postingSizes.UpdateSize(headID, localIndices.size());
                    *m_checkSums[headID] = m_checkSum.CalcChecksum(postingList.c_str(), (int)(postingList.size()));
                    if (m_opt->m_consistencyCheck && (ret = db->Check(headID, m_postingSizes.GetSize(headID) * m_vectorInfoSize, nullptr)) != ErrorCode::Success)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Split: Check failed after Put %d\n", headID);
                        return ret;
                    }
                    m_stat.m_garbageNum++;
                    auto GCEnd = std::chrono::high_resolution_clock::now();
                    elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(GCEnd - splitBegin).count();
                    m_stat.m_garbageCost += elapsedMSeconds;
                    {
                        std::lock_guard<std::mutex> tmplock(m_runningLock);
                        // SPTAGLIB_LOG(Helper::LogLevel::LL_Info,"erase: %d\n", headID);
                        m_splitList.erase(headID);
                    }
                    //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "GC triggered: %d, new length: %d\n", headID, index);
                    return ErrorCode::Success;
                }

                auto clusterBegin = std::chrono::high_resolution_clock::now();
                // k = 2, maybe we can change the split number, now it is fixed
                SPTAG::COMMON::KmeansArgs<ValueType> args(2, smallSample.C(), (SizeType)localIndices.size(), 1, p_index->GetDistCalcMethod(), p_index->m_pQuantizer);
                std::shuffle(localIndices.begin(), localIndices.end(), std::mt19937(std::random_device()()));

                int numClusters = SPTAG::COMMON::KmeansClustering(smallSample, localIndices, 0, (SizeType)localIndices.size(), args, 1000, 100.0F, false, nullptr);

                auto clusterEnd = std::chrono::high_resolution_clock::now();
                elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(clusterEnd - clusterBegin).count();
                m_stat.m_clusteringCost += elapsedMSeconds;
                // int numClusters = ClusteringSPFresh(smallSample, localIndices, 0, localIndices.size(), args, 10, false, m_opt->m_virtualHead);
                if (numClusters <= 1)
                {
                    int cut = 1;
                    if (m_opt->m_oneClusterCutMax) cut = m_postingSizeLimit;
                    std::string newpostingList(cut * m_vectorInfoSize, '\0');
                    char* ptr = (char*)(newpostingList.c_str());
                    float totaldist = 0.0f;
                    for (int j = 0; j < cut; j++, ptr += m_vectorInfoSize)
                    {
                        totaldist += p_index->ComputeDistance(ptr + sizeof(int) + 1, args.centers);
                        memcpy(ptr, postingList.c_str() + localIndices[j] * m_vectorInfoSize, m_vectorInfoSize);
                        //Serialize(ptr, localIndicesInsert[j], localIndicesInsertVersion[j], smallSample[j]);
                    }
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Cluserting Failed (The same vector), Cluster total dist:%f Only Keep %d vectors.\n", totaldist, cut);
                   
                    if ((ret=db->Put(headID, newpostingList, MaxTimeout, &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Split fail to override postings cut to limit\n");
                        return ret;
                    }
                    m_postingSizes.UpdateSize(headID, cut);
                    *m_checkSums[headID] =
                        m_checkSum.CalcChecksum(newpostingList.c_str(), (int)(newpostingList.size()));
                    if (m_opt->m_consistencyCheck && (ret = db->Check(headID, m_postingSizes.GetSize(headID) * m_vectorInfoSize, nullptr)) != ErrorCode::Success)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Split: Consolidate Check failed after Put %d\n", headID);
                        return ret;
                    }
                    {
                        std::lock_guard<std::mutex> tmplock(m_runningLock);
                        m_splitList.erase(headID);
                    }
                    return ErrorCode::Success;
                }

                long long newHeadVID = -1;
                int first = 0;                
                newPostingLists.resize(2);
                for (int k = 0; k < 2; k++) {
                    if (args.counts[k] == 0)	continue;
                    
                    newPostingLists[k].resize(args.counts[k] * m_vectorInfoSize);
                    char* ptr = (char*)(newPostingLists[k].c_str());
                    for (int j = 0; j < args.counts[k]; j++, ptr += m_vectorInfoSize)
                    {
                        memcpy(ptr, postingList.c_str() + localIndices[first + j] * m_vectorInfoSize, m_vectorInfoSize);
                        //Serialize(ptr, localIndicesInsert[localIndices[first + j]], localIndicesInsertVersion[localIndices[first + j]], smallSample[localIndices[first + j]]);
                    }
                    if (!theSameHead && p_index->ComputeDistance(args.centers + k * args._D, p_index->GetSample(headID)) < Epsilon) {
                        newHeadsID.push_back(headID);
                        newHeadVID = headID;
                        theSameHead = true;
                        auto splitPutBegin = std::chrono::high_resolution_clock::now();
                        if (!preReassign && (ret=db->Put(newHeadVID, newPostingLists[k], MaxTimeout, &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to override postings\n");
                            return ret;
                        }
                        m_postingSizes.UpdateSize(newHeadVID, args.counts[k]);
                        *m_checkSums[newHeadVID] =
                            m_checkSum.CalcChecksum(newPostingLists[k].c_str(), (int)(newPostingLists[k].size()));
                        if (m_opt->m_consistencyCheck && (ret = db->Check(newHeadVID, m_postingSizes.GetSize(newHeadVID) * m_vectorInfoSize, nullptr)) != ErrorCode::Success)
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Split: Cluster Write Check failed after Put %d\n", newHeadVID);
                            return ret;
                        }
                        auto splitPutEnd = std::chrono::high_resolution_clock::now();
                        elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(splitPutEnd - splitPutBegin).count();
                        m_stat.m_putCost += elapsedMSeconds;
                        m_stat.m_theSameHeadNum++;
                    }
                    else {
                        int begin, end = 0;
                        p_index->AddIndexId(args.centers + k * args._D, 1, m_opt->m_dim, begin, end);
                        {
                            std::lock_guard<std::mutex> tmplock(m_runningLock);
                            m_vectorTranslateMap->AddBatch(1);
                        }
                        if (m_opt->m_excludehead)
                        {
                            SizeType VID = *((SizeType*)(postingP + args.clusterIdx[k] * m_vectorInfoSize));
                            uint8_t version = *((uint8_t*)(postingP + args.clusterIdx[k] * m_vectorInfoSize + sizeof(SizeType)));
                            *(m_vectorTranslateMap->At(begin)) = VID;
                            m_versionMap->IncVersion(VID, &version);
                        }
                        else
                        {
                            *(m_vectorTranslateMap->At(begin)) = MaxSize;
                        }
                        newHeadVID = begin;
                        newHeadsID.push_back(begin);
                        auto splitPutBegin = std::chrono::high_resolution_clock::now();
                        if (!preReassign && (ret=db->Put(newHeadVID, newPostingLists[k], MaxTimeout, &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to add new postings\n");
                            return ret;
                        }                        
                        auto splitPutEnd = std::chrono::high_resolution_clock::now();
                        elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(splitPutEnd - splitPutBegin).count();
                        m_stat.m_putCost += elapsedMSeconds;

                        std::lock_guard<std::mutex> tmplock(m_dataAddLock);
                        if (m_postingSizes.AddBatch(1) == ErrorCode::MemoryOverFlow || m_checkSums.AddBatch(1) == ErrorCode::MemoryOverFlow)
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "MemoryOverFlow: NnewHeadVID: %d, Map Size:%d\n",
                                         newHeadVID, m_postingSizes.BufferSize());
                            return ErrorCode::MemoryOverFlow;
                        }
                        m_postingSizes.UpdateSize(newHeadVID, args.counts[k]);
                        *m_checkSums[newHeadVID] =
                            m_checkSum.CalcChecksum(newPostingLists[k].c_str(), (int)(newPostingLists[k].size()));
                        if (m_opt->m_consistencyCheck && (ret = db->Check(newHeadVID, m_postingSizes.GetSize(newHeadVID) * m_vectorInfoSize, nullptr)) != ErrorCode::Success)
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Split: Cluster Write Check failed after Put %d\n", newHeadVID);
                            return ret;
                        }
                         
                        auto updateHeadBegin = std::chrono::high_resolution_clock::now();
                        p_index->AddIndexIdx(begin, end);
                        auto updateHeadEnd = std::chrono::high_resolution_clock::now();
                        elapsedMSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(updateHeadEnd - updateHeadBegin).count();
                        m_stat.m_updateHeadCost += elapsedMSeconds;
                    }
                    //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Head id: %d split into : %d, length: %d\n", headID, newHeadVID, args.counts[k]);
                    first += args.counts[k];
                }
                if (!theSameHead) {
                    p_index->DeleteIndex(headID);
                    m_postingSizes.UpdateSize(headID, 0);
                    *m_checkSums[headID] = 0;
                    if ((ret=db->Delete(headID)) != ErrorCode::Success)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to delete old posting in Split\n");
                        return ret;
                    }
                }
            }
            {
                std::lock_guard<std::mutex> tmplock(m_runningLock);
                //SPTAGLIB_LOG(Helper::LogLevel::LL_Info,"erase: %d\n", headID);
                m_splitList.erase(headID);
            }
            m_stat.m_splitNum++;
            if (reassign) {
                auto reassignScanBegin = std::chrono::high_resolution_clock::now();

                CollectReAssign(p_exWorkSpace, p_index, headID, newPostingLists, newHeadsID, theSameHead);

                auto reassignScanEnd = std::chrono::high_resolution_clock::now();
                elapsedMSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(reassignScanEnd - reassignScanBegin).count();

                m_stat.m_reassignScanCost += elapsedMSeconds;
            }
            auto splitEnd = std::chrono::high_resolution_clock::now();
            elapsedMSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(splitEnd - splitBegin).count();
            m_stat.m_splitCost += elapsedMSeconds;
            return ErrorCode::Success;
        }

        ErrorCode MergePostings(ExtraWorkSpace *p_exWorkSpace, VectorIndex* p_index, SizeType headID, bool reassign = false)
        {
            {
                if (!m_mergeLock.try_lock()) {
                    auto* curJob = new MergeAsyncJob(p_index, this, headID, reassign, nullptr);
                    m_splitThreadPool->add(curJob);
                    return ErrorCode::Success;
                }
                std::unique_lock<std::shared_timed_mutex> lock(m_rwLocks[headID]);

                if (!p_index->ContainSample(headID)) {
                    m_mergeLock.unlock();
                    return ErrorCode::Success;
                }

                std::string mergedPostingList;
                std::set<SizeType> vectorIdSet;

                std::string currentPostingList;
                ErrorCode ret;
                if ((ret = db->Get(headID, &currentPostingList, MaxTimeout, &(p_exWorkSpace->m_diskRequests))) !=
                        ErrorCode::Success ||
                    !m_checkSum.ValidateChecksum(currentPostingList.c_str(), (int)(currentPostingList.size()), *m_checkSums[headID]))
                {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "Fail to get original merge postings: %d, required size:%d, get size:%d, checksum issue:%d\n",
                        headID, (int)(m_postingSizes.GetSize(headID) * m_vectorInfoSize),
                        (int)(currentPostingList.size()), (int)(ret == ErrorCode::Success));
                    PrintErrorInPosting(currentPostingList, headID);
                    m_mergeLock.unlock();
                    return ret;
                }

                auto* postingP = reinterpret_cast<uint8_t*>(currentPostingList.data());
                size_t postVectorNum = currentPostingList.size() / m_vectorInfoSize;
                int currentLength = 0;
                uint8_t* vectorId = postingP;
                for (int j = 0; j < postVectorNum; j++, vectorId += m_vectorInfoSize)
                {
                    int VID = *((int*)(vectorId));
                    uint8_t version = *(vectorId + sizeof(int));
                    if (m_versionMap->Deleted(VID) || m_versionMap->GetVersion(VID) != version) continue;
                    vectorIdSet.insert(VID);
                    mergedPostingList += currentPostingList.substr(j * m_vectorInfoSize, m_vectorInfoSize);
                    currentLength++;
                }
                int totalLength = currentLength;

                if (currentLength > m_mergeThreshold)
                {
                    if ((ret=db->Put(headID, mergedPostingList, MaxTimeout, &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Merge Fail to write back postings\n");
                        m_mergeLock.unlock();
                        return ret;
                    }

                    m_postingSizes.UpdateSize(headID, currentLength);
                    *m_checkSums[headID] =
                        m_checkSum.CalcChecksum(mergedPostingList.c_str(), (int)(mergedPostingList.size()));

                    if (m_opt->m_consistencyCheck && (ret = db->Check(headID, m_postingSizes.GetSize(headID) * m_vectorInfoSize, nullptr)) != ErrorCode::Success)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Merge: Check failed after Put %d\n", headID);
                        m_mergeLock.unlock();
                        return ret;
                    }
                    {
                        std::unique_lock<std::shared_timed_mutex> lock(m_mergeListLock);
                        m_mergeList.unsafe_erase(headID);
                    }
                    m_mergeLock.unlock();
                    return ErrorCode::Success;
                }

                COMMON::QueryResultSet<ValueType> queryResults((ValueType*)(p_index->GetSample(headID)), m_opt->m_internalResultNum);
                std::shared_ptr<std::uint8_t> rec_query;
                if (p_index->m_pQuantizer) {
                    rec_query.reset((uint8_t*)ALIGN_ALLOC(p_index->m_pQuantizer->ReconstructSize()), [=](std::uint8_t* ptr) { ALIGN_FREE(ptr); });
                    p_index->m_pQuantizer->ReconstructVector((const uint8_t*)queryResults.GetTarget(), rec_query.get());
                    queryResults.SetTarget((ValueType*)(rec_query.get()), p_index->m_pQuantizer);
                }
                p_index->SearchIndex(queryResults);

                std::string nextPostingList;
                for (int i = 1; i < queryResults.GetResultNum(); ++i)
                {
                    BasicResult* queryResult = queryResults.GetResult(i);
                    int nextLength = m_postingSizes.GetSize(queryResult->VID);
                    bool listContains = false;
                    {
                        std::shared_lock<std::shared_timed_mutex> anotherLock(m_mergeListLock);
                        listContains = (m_mergeList.find(queryResult->VID) != m_mergeList.end());
                    }
                    if (currentLength + nextLength < m_postingSizeLimit && !listContains)
                    {
                        {
                            std::unique_lock<std::shared_timed_mutex> anotherLock(m_rwLocks[queryResult->VID], std::defer_lock);
                            // SPTAGLIB_LOG(Helper::LogLevel::LL_Info,"Locked: %d, to be lock: %d\n", headID, queryResult->VID);
                            if (m_rwLocks.hash_func(queryResult->VID) != m_rwLocks.hash_func(headID)) anotherLock.lock();
                            if (!p_index->ContainSample(queryResult->VID)) continue;
                            if ((ret=db->Get(queryResult->VID, &nextPostingList, MaxTimeout, &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success || 
                                !m_checkSum.ValidateChecksum(nextPostingList.c_str(), (int)(nextPostingList.size()), *m_checkSums[queryResult->VID])) {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                             "Fail to get to be merged postings: %d, required size:%d get size:%d, "
                                             "checksum issue:%d\n",
                                             queryResult->VID,
                                             (int)(m_postingSizes.GetSize(queryResult->VID) * m_vectorInfoSize),
                                             (int)(nextPostingList.size()), (int)(ret == ErrorCode::Success));
                                PrintErrorInPosting(nextPostingList, queryResult->VID);
                                m_mergeLock.unlock();
                                return ret;
                            }
                            postingP = reinterpret_cast<uint8_t*>(nextPostingList.data());
                            postVectorNum = nextPostingList.size() / m_vectorInfoSize;
                            nextLength = 0;
                            vectorId = postingP;
                            for (int j = 0; j < postVectorNum; j++, vectorId += m_vectorInfoSize)
                            {
                                int VID = *((int*)(vectorId));
                                uint8_t version = *(vectorId + sizeof(int));
                                if (m_versionMap->Deleted(VID) || m_versionMap->GetVersion(VID) != version) continue;
                                if (vectorIdSet.find(VID) == vectorIdSet.end()) {
                                    mergedPostingList += nextPostingList.substr(j * m_vectorInfoSize, m_vectorInfoSize);
                                    totalLength++;
                                }
                                nextLength++;
                            }
                            if (currentLength > nextLength) 
                            {
                                p_index->DeleteIndex(queryResult->VID);
                                if ((ret=db->Put(headID, mergedPostingList, MaxTimeout, &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success) {
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "MergePostings fail to override postings after merge\n");
                                    m_mergeLock.unlock();
                                    return ret;
                                }
                                m_postingSizes.UpdateSize(queryResult->VID, 0);
                                *m_checkSums[queryResult->VID] = 0;
                                m_postingSizes.UpdateSize(headID, totalLength);
                                *m_checkSums[headID] =
                                    m_checkSum.CalcChecksum(mergedPostingList.c_str(), (int)(mergedPostingList.size()));
                                if (m_opt->m_consistencyCheck && (ret = db->Check(headID, m_postingSizes.GetSize(headID) * m_vectorInfoSize, nullptr)) != ErrorCode::Success)
                                {
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "MergePostings fail to check old posting %d in Merge\n", headID);
                                    m_mergeLock.unlock();
                                    return ret;
                                }
                                if ((ret=db->Delete(queryResult->VID)) != ErrorCode::Success)
                                {
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to delete old posting in Merge\n");
                                    m_mergeLock.unlock();
                                    return ret;
                                }
                            } else
                            {
                                p_index->DeleteIndex(headID);
                                if ((ret=db->Put(queryResult->VID, mergedPostingList, MaxTimeout, &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success) {
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "MergePostings fail to override postings after merge\n");
                                    m_mergeLock.unlock();
                                    return ret;
                                }
                                m_postingSizes.UpdateSize(queryResult->VID, totalLength);
                                *m_checkSums[queryResult->VID] =
                                    m_checkSum.CalcChecksum(mergedPostingList.c_str(), (int)(mergedPostingList.size()));
                                m_postingSizes.UpdateSize(headID, 0);
                                *m_checkSums[headID] = 0;
                                if (m_opt->m_consistencyCheck && (ret = db->Check(queryResult->VID, m_postingSizes.GetSize(queryResult->VID) * m_vectorInfoSize, nullptr)) != ErrorCode::Success)
                                {
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "MergePostings fail to check nearby posting %d in Merge\n", queryResult->VID);
                                    m_mergeLock.unlock();
                                    return ret;
                                }
                                if ((ret = db->Delete(headID)) != ErrorCode::Success)
                                {
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to delete old posting in Merge\n");
                                    m_mergeLock.unlock();
                                    return ret;
                                }
                            }
                            if (m_rwLocks.hash_func(queryResult->VID) != m_rwLocks.hash_func(headID)) anotherLock.unlock();
                        }

                        // SPTAGLIB_LOG(Helper::LogLevel::LL_Info,"Release: %d, Release: %d\n", headID, queryResult->VID);
                        lock.unlock();
                        m_mergeLock.unlock();

                        if (reassign) 
                        {
                            SizeType deletedHead = -1;
                            /* ReAssign */
                            if (currentLength > nextLength) 
                            {
                                /* ReAssign queryResult->VID*/
                                postingP = reinterpret_cast<uint8_t*>(nextPostingList.data());
                                for (int j = 0; j < nextLength; j++) {
                                    uint8_t* vectorId = postingP + j * m_vectorInfoSize;
                                    // SizeType vid = *(reinterpret_cast<SizeType*>(vectorId));
                                    ValueType* vector = reinterpret_cast<ValueType*>(vectorId + m_metaDataSize);
                                    float origin_dist = p_index->ComputeDistance(p_index->GetSample(queryResult->VID), vector);
                                    float current_dist = p_index->ComputeDistance(p_index->GetSample(headID), vector);
                                    if (current_dist > origin_dist)
                                        ReassignAsync(p_index, std::make_shared<std::string>((char*)vectorId, m_vectorInfoSize), headID);
                                }
                                deletedHead = queryResult->VID;
                            } else
                            {
                                /* ReAssign headID*/
                                postingP = reinterpret_cast<uint8_t*>(currentPostingList.data());
                                for (int j = 0; j < currentLength; j++) {
                                    uint8_t* vectorId = postingP + j * m_vectorInfoSize;
                                    // SizeType vid = *(reinterpret_cast<SizeType*>(vectorId));
                                    ValueType* vector = reinterpret_cast<ValueType*>(vectorId + m_metaDataSize);
                                    float origin_dist = p_index->ComputeDistance(p_index->GetSample(headID), vector);
                                    float current_dist = p_index->ComputeDistance(p_index->GetSample(queryResult->VID), vector);
                                    if (current_dist > origin_dist)
                                        ReassignAsync(p_index, std::make_shared<std::string>((char*)vectorId, m_vectorInfoSize), queryResult->VID);
                                }
                                deletedHead = headID;
                            }

                            if (m_opt->m_excludehead)
                            {
                                SizeType vid = (SizeType)(*(m_vectorTranslateMap->At(deletedHead)));
                                if (vid != MaxSize && !m_versionMap->Deleted(vid))
                                {
                                    std::shared_ptr<std::string> vectorinfo =
                                        std::make_shared<std::string>(m_vectorInfoSize, ' ');
                                    Serialize(vectorinfo->data(), vid, m_versionMap->GetVersion(vid),
                                              p_index->GetSample(deletedHead));
                                    ReassignAsync(p_index, vectorinfo, -1);
                                }
                            }
                        }

                        {
                            std::unique_lock<std::shared_timed_mutex> lock(m_mergeListLock);
                            m_mergeList.unsafe_erase(headID);
                        }
                        m_stat.m_mergeNum++;

                        return ErrorCode::Success;
                    }
                }
                mergedPostingList.resize(currentLength * m_vectorInfoSize);
                if ((ret=db->Put(headID, mergedPostingList, MaxTimeout, &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Merge Fail to write back postings\n");
                    return ret;
                }

                m_postingSizes.UpdateSize(headID, currentLength);
                *m_checkSums[headID] =
                    m_checkSum.CalcChecksum(mergedPostingList.c_str(), (int)(mergedPostingList.size()));

                if (m_opt->m_consistencyCheck && (ret = db->Check(headID, m_postingSizes.GetSize(headID) * m_vectorInfoSize, nullptr)) != ErrorCode::Success)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Merge: Check failed after put original posting %d\n", headID);
                    return ret;
                }
                {
                    std::unique_lock<std::shared_timed_mutex> lock(m_mergeListLock);
                    m_mergeList.unsafe_erase(headID);
                }
                m_mergeLock.unlock();
            }
            return ErrorCode::Success;
        }

        inline void SplitAsync(VectorIndex* p_index, SizeType headID, std::function<void()> p_callback = nullptr)
        {
            // SPTAGLIB_LOG(Helper::LogLevel::LL_Info,"Into SplitAsync, current headID: %d, size: %d\n", headID, m_postingSizes.GetSize(headID));
            // tbb::concurrent_hash_map<SizeType, SizeType>::const_accessor headIDAccessor;
            // if (m_splitList.find(headIDAccessor, headID)) {
            //     return;
            // }
            // tbb::concurrent_hash_map<SizeType, SizeType>::value_type workPair(headID, headID);
            // m_splitList.insert(workPair);
            {
                std::lock_guard<std::mutex> tmplock(m_runningLock);

                if (m_splitList.find(headID) != m_splitList.end()) {
                    // SPTAGLIB_LOG(Helper::LogLevel::LL_Info,"Already in queue\n");
                    return;
                }
                m_splitList.insert(headID);
            }

            auto* curJob = new SplitAsyncJob(p_index, this, headID, m_opt->m_disableReassign, p_callback);
            m_splitThreadPool->add(curJob);
            // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Add to thread pool\n");
        }

        inline void MergeAsync(VectorIndex* p_index, SizeType headID, std::function<void()> p_callback = nullptr)
        {
            Helper::Concurrent::ConcurrentMap<SizeType, SizeType>::value_type workPair(headID, headID);
            {
                std::shared_lock<std::shared_timed_mutex> lock(m_mergeListLock);
                auto res = m_mergeList.insert(workPair);
                if (!res.second)
                {
                    // Already in queue
                    return;
                }
            }

            auto* curJob = new MergeAsyncJob(p_index, this, headID, m_opt->m_disableReassign, p_callback);
            m_splitThreadPool->add(curJob);
        }

        inline void ReassignAsync(VectorIndex* p_index, std::shared_ptr<std::string> vectorInfo, SizeType HeadPrev, std::function<void()> p_callback = nullptr)
        {
            auto* curJob = new ReassignAsyncJob(p_index, this, std::move(vectorInfo), HeadPrev, p_callback);
            m_splitThreadPool->add(curJob);
        }

        ErrorCode CollectReAssign(ExtraWorkSpace *p_exWorkSpace, VectorIndex *p_index, SizeType headID,
                                  std::vector<std::string> &postingLists, std::vector<SizeType> &newHeadsID,
                                  bool theSameHead)
        {
            auto headVector = reinterpret_cast<const ValueType*>(p_index->GetSample(headID));
            if (m_opt->m_excludehead && !theSameHead)
            {
                SizeType vid = (SizeType)(*(m_vectorTranslateMap->At(headID)));
                if (vid != MaxSize && !m_versionMap->Deleted(vid))
                {
                    std::shared_ptr<std::string> vectorinfo = std::make_shared<std::string>(m_vectorInfoSize, ' ');
                    Serialize(vectorinfo->data(), vid, m_versionMap->GetVersion(vid), headVector);
                    ReassignAsync(p_index, vectorinfo, -1);
                }
            }
            std::vector<float> newHeadsDist;
            std::set<SizeType> reAssignVectorsTopK;
            newHeadsDist.push_back(p_index->ComputeDistance(p_index->GetSample(headID), p_index->GetSample(newHeadsID[0])));
            newHeadsDist.push_back(p_index->ComputeDistance(p_index->GetSample(headID), p_index->GetSample(newHeadsID[1])));
            for (int i = 0; i < postingLists.size(); i++) {
                auto& postingList = postingLists[i];
                size_t postVectorNum = postingList.size() / m_vectorInfoSize;
                auto* postingP = reinterpret_cast<uint8_t*>(postingList.data());
                for (int j = 0; j < postVectorNum; j++) {
                    uint8_t* vectorId = postingP + j * m_vectorInfoSize;
                    SizeType vid = *(reinterpret_cast<SizeType*>(vectorId));
                    // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "VID: %d, Head: %d\n", vid, newHeadsID[i]);
                    uint8_t version = *(reinterpret_cast<uint8_t*>(vectorId + sizeof(int)));
                    ValueType* vector = reinterpret_cast<ValueType*>(vectorId + m_metaDataSize);
                    if (reAssignVectorsTopK.find(vid) == reAssignVectorsTopK.end() && !m_versionMap->Deleted(vid) && m_versionMap->GetVersion(vid) == version) {
                        m_stat.m_reAssignScanNum++;
                        float dist = p_index->ComputeDistance(p_index->GetSample(newHeadsID[i]), vector);
                        if (CheckIsNeedReassign(p_index, newHeadsID, vector, headID, newHeadsDist[i], dist, true, newHeadsID[i])) {
                            ReassignAsync(p_index, std::make_shared<std::string>((char*)vectorId, m_vectorInfoSize), newHeadsID[i]);
                            reAssignVectorsTopK.insert(vid);
                        }
                    }
                }
            }
            if (m_opt->m_reassignK > 0) {
                std::vector<SizeType> HeadPrevTopK;
                newHeadsDist.clear();
                newHeadsDist.resize(0);
                COMMON::QueryResultSet<ValueType> nearbyHeads((ValueType*)headVector, m_opt->m_reassignK);
                std::shared_ptr<std::uint8_t> rec_query;
                if (p_index->m_pQuantizer) {
                    rec_query.reset((uint8_t*)ALIGN_ALLOC(p_index->m_pQuantizer->ReconstructSize()), [=](std::uint8_t* ptr) { ALIGN_FREE(ptr); });
                    p_index->m_pQuantizer->ReconstructVector((const uint8_t*)nearbyHeads.GetTarget(), rec_query.get());
                    nearbyHeads.SetTarget((ValueType*)(rec_query.get()), p_index->m_pQuantizer);
                }
                p_index->SearchIndex(nearbyHeads);
                BasicResult* queryResults = nearbyHeads.GetResults();
                for (int i = 0; i < nearbyHeads.GetResultNum(); i++) {
                    auto vid = queryResults[i].VID;
                    if (vid == -1) break;

                    if (find(newHeadsID.begin(), newHeadsID.end(), vid) == newHeadsID.end()) {
                        HeadPrevTopK.push_back(vid);
                        newHeadsID.push_back(vid);
                        newHeadsDist.push_back(queryResults[i].Dist);
                    }
                }
                auto reassignScanIOBegin = std::chrono::high_resolution_clock::now();
                ErrorCode ret;
                if ((ret = db->MultiGet(HeadPrevTopK, p_exWorkSpace->m_pageBuffers, HardLatencyLimit(),
                                        &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success ||
                    !ValidatePostings(HeadPrevTopK, p_exWorkSpace->m_pageBuffers))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "ReAssign can't get all the near postings\n");
                    return ret;
                }

                auto reassignScanIOEnd = std::chrono::high_resolution_clock::now();
                auto elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(reassignScanIOEnd - reassignScanIOBegin).count();
                m_stat.m_reassignScanIOCost += elapsedMSeconds;

                for (int i = 0; i < HeadPrevTopK.size(); i++)
                {
                    auto &buffer = (p_exWorkSpace->m_pageBuffers[i]);
                    size_t postVectorNum = (int)(buffer.GetAvailableSize() / m_vectorInfoSize);
                    auto *postingP = buffer.GetBuffer();
                    for (int j = 0; j < postVectorNum; j++) {
                        uint8_t* vectorId = postingP + j * m_vectorInfoSize;
                        SizeType vid = *(reinterpret_cast<SizeType*>(vectorId));
                        // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "%d: VID: %d, Head: %d, size:%d/%d\n", i, vid, HeadPrevTopK[i], postingLists.size(), HeadPrevTopK.size());
                        uint8_t version = *(reinterpret_cast<uint8_t*>(vectorId + sizeof(SizeType)));
                        ValueType* vector = reinterpret_cast<ValueType*>(vectorId + m_metaDataSize);
                        if (reAssignVectorsTopK.find(vid) == reAssignVectorsTopK.end() && !m_versionMap->Deleted(vid) && m_versionMap->GetVersion(vid) == version) {
                            m_stat.m_reAssignScanNum++;
                            float dist = p_index->ComputeDistance(p_index->GetSample(HeadPrevTopK[i]), vector);
                            if (CheckIsNeedReassign(p_index, newHeadsID, vector, headID, newHeadsDist[i], dist, false, HeadPrevTopK[i])) {
                                ReassignAsync(p_index, std::make_shared<std::string>((char*)vectorId, m_vectorInfoSize), HeadPrevTopK[i]);
                                reAssignVectorsTopK.insert(vid);
                            }
                        }
                    }
                }
            }
            return ErrorCode::Success;
        }

        bool RNGSelection(std::vector<Edge>& selections, ValueType* queryVector, VectorIndex* p_index, SizeType p_fullID, int& replicaCount, int checkHeadID = -1, const std::vector<uint8_t>* p_allowedHeads = nullptr, SizeType p_allowedHeadCount = -1)
        {
            COMMON::QueryResultSet<ValueType> queryResults(queryVector, m_opt->m_internalResultNum);
            std::shared_ptr<std::uint8_t> rec_query;
            if (p_index->m_pQuantizer) {
                rec_query.reset((uint8_t*)ALIGN_ALLOC(p_index->m_pQuantizer->ReconstructSize()), [=](std::uint8_t* ptr) { ALIGN_FREE(ptr); });
                p_index->m_pQuantizer->ReconstructVector((const uint8_t*)queryResults.GetTarget(), rec_query.get());
                queryResults.SetTarget((ValueType*)(rec_query.get()), p_index->m_pQuantizer);
            }

            replicaCount = 0;

            // When an allowed-head mask is given (subset / node-pure build), do a
            // direct linear scan over only the allowed heads instead of "search
            // global BKT → filter by mask". The global-search approach asks for
            // m_internalResultNum (≈32) closest heads anywhere; for small subsets
            // (e.g. N=256 leaves ~80 heads per subset out of ~27k total) ~88% of
            // those candidates are masked out, leaving most vectors with 0
            // replicas and effectively dropped from the index. Per-subset linear
            // scan guarantees every vector finds its true top-K allowed heads.
            if (p_allowedHeads != nullptr)
            {
                const SizeType numHeads = static_cast<SizeType>(p_allowedHeads->size());
                const SizeType allowedCount = (p_allowedHeadCount >= 0)
                    ? static_cast<SizeType>(p_allowedHeadCount) : numHeads;

                // Large node-pure subset fast path. When this node owns a large
                // fraction of all heads (e.g. a "missing-value" sentinel bucket
                // holding ~half the dataset), the per-vector linear scan below is
                // O(V*H) and dominates the whole node-aware build. Instead use the
                // head BKT graph search (O(log H)) for an expanded candidate set,
                // then keep only in-node heads. The expansion factor guarantees that
                // >= replicaCount in-node heads survive the mask filter. Threshold is
                // an absolute head count (env SPTAG_GRAPH_SCAN_MIN_HEADS, default
                // 100000); below it the linear scan is cheap and exact.
                static const SizeType kGraphScanMinHeads = []() {
                    const char* e = std::getenv("SPTAG_GRAPH_SCAN_MIN_HEADS");
                    return e ? static_cast<SizeType>(std::max(1, atoi(e))) : static_cast<SizeType>(100000);
                }();

                if (allowedCount >= kGraphScanMinHeads)
                {
                    int targetK = m_opt->m_internalResultNum;
                    if (allowedCount > 0) {
                        long long expand = static_cast<long long>(m_opt->m_replicaCount) * 4LL
                                           * static_cast<long long>(numHeads)
                                           / static_cast<long long>(allowedCount);
                        if (expand > targetK) targetK = static_cast<int>(std::min<long long>(expand, static_cast<long long>(numHeads)));
                    }
                    COMMON::QueryResultSet<ValueType> bigResults(queryVector, targetK);
                    if (p_index->m_pQuantizer) {
                        bigResults.SetTarget((ValueType*)(rec_query.get()), p_index->m_pQuantizer);
                    }
                    p_index->SearchIndex(bigResults);

                    replicaCount = 0;
                    for (int i = 0; i < bigResults.GetResultNum() && replicaCount < m_opt->m_replicaCount; ++i)
                    {
                        BasicResult* r = bigResults.GetResult(i);
                        if (r->VID == -1) break;
                        if (r->VID < 0 || r->VID >= static_cast<SizeType>(p_allowedHeads->size()) ||
                            (*p_allowedHeads)[static_cast<size_t>(r->VID)] == 0) {
                            continue;
                        }
                        bool rngAccepted = true;
                        for (int j = 0; j < replicaCount; ++j) {
                            float nnDist = p_index->ComputeDistance(p_index->GetSample(r->VID),
                                                                    p_index->GetSample(selections[j].node));
                            if (m_opt->m_rngFactor * nnDist <= r->Dist) {
                                rngAccepted = false;
                                break;
                            }
                        }
                        if (!rngAccepted) continue;
                        selections[replicaCount].node = r->VID;
                        selections[replicaCount].tonode = p_fullID;
                        selections[replicaCount].distance = r->Dist;
                        if (selections[replicaCount].node == checkHeadID) {
                            return false;
                        }
                        ++replicaCount;
                    }
                    return true;
                }

                std::vector<std::pair<float, SizeType>> candidates;
                candidates.reserve(64);
                const void* queryTarget = queryResults.GetTarget();
                for (SizeType h = 0; h < numHeads; ++h) {
                    if ((*p_allowedHeads)[static_cast<size_t>(h)] == 0) continue;
                    const void* headSample = p_index->GetSample(h);
                    if (headSample == nullptr) continue;
                    float dist = p_index->ComputeDistance(queryTarget, headSample);
                    candidates.emplace_back(dist, h);
                }
                // Only the nearest m_internalResultNum allowed heads can plausibly
                // win an RNG replica slot, exactly mirroring the global SearchIndex
                // path below (which returns m_internalResultNum candidates). For
                // large node-pure subsets (e.g. a big country with ~10^5 heads) a
                // full std::sort of every allowed head per vector is O(H log H) and
                // dominates the whole node-aware build. Bound the candidate set with
                // nth_element first, then sort only the retained head.
                auto candCmp = [](const std::pair<float, SizeType>& a, const std::pair<float, SizeType>& b) {
                    return a.first < b.first;
                };
                const size_t keepK = static_cast<size_t>(std::max(1, m_opt->m_internalResultNum));
                if (candidates.size() > keepK) {
                    std::nth_element(candidates.begin(), candidates.begin() + keepK, candidates.end(), candCmp);
                    candidates.resize(keepK);
                }
                std::sort(candidates.begin(), candidates.end(), candCmp);

                for (const auto& cand : candidates) {
                    if (replicaCount >= m_opt->m_replicaCount) break;
                    SizeType candVID = cand.second;
                    float candDist = cand.first;
                    // RNG dedup check (same as the global path below).
                    bool rngAccepted = true;
                    for (int j = 0; j < replicaCount; ++j) {
                        float nnDist = p_index->ComputeDistance(p_index->GetSample(candVID),
                                                                p_index->GetSample(selections[j].node));
                        if (m_opt->m_rngFactor * nnDist <= candDist) {
                            rngAccepted = false;
                            break;
                        }
                    }
                    if (!rngAccepted) continue;
                    selections[replicaCount].node = candVID;
                    selections[replicaCount].tonode = p_fullID;
                    selections[replicaCount].distance = candDist;
                    if (selections[replicaCount].node == checkHeadID) {
                        return false;
                    }
                    ++replicaCount;
                }
                return true;
            }

            p_index->SearchIndex(queryResults);

            replicaCount = 0;
            for (int i = 0; i < queryResults.GetResultNum() && replicaCount < m_opt->m_replicaCount; ++i)
            {
                BasicResult* queryResult = queryResults.GetResult(i);
                if (queryResult->VID == -1) {
                    break;
                }
                if (p_allowedHeads != nullptr)
                {
                    if (queryResult->VID < 0 || queryResult->VID >= static_cast<SizeType>(p_allowedHeads->size()) ||
                        (*p_allowedHeads)[static_cast<size_t>(queryResult->VID)] == 0)
                    {
                        continue;
                    }
                }
                // RNG Check.
                bool rngAccpeted = true;
                for (int j = 0; j < replicaCount; ++j)
                {
                    float nnDist = p_index->ComputeDistance(p_index->GetSample(queryResult->VID),
                        p_index->GetSample(selections[j].node));
                    if (m_opt->m_rngFactor * nnDist <= queryResult->Dist)
                    {
                        rngAccpeted = false;
                        break;
                    }
                }
                if (!rngAccpeted) continue;
                selections[replicaCount].node = queryResult->VID;
                selections[replicaCount].tonode = p_fullID;
                selections[replicaCount].distance = queryResult->Dist;
                if (selections[replicaCount].node == checkHeadID) {
                    return false;
                }
                ++replicaCount;
            }
            return true;
        }

        void InitWorkSpace(ExtraWorkSpace* p_exWorkSpace, bool clear = false) override
        {
            if (clear) {
                p_exWorkSpace->Clear(m_opt->m_searchInternalResultNum, (max(m_opt->m_postingPageLimit, m_opt->m_searchPostingPageLimit) + m_opt->m_bufferLength + m_opt->m_unfilterTailBufferLength) << PageSizeEx, true, m_opt->m_enableDataCompression);
            }
            else {
                p_exWorkSpace->Initialize(m_opt->m_maxCheck, m_opt->m_hashExp, max(m_opt->m_searchInternalResultNum, m_opt->m_reassignK), (max(m_opt->m_postingPageLimit, m_opt->m_searchPostingPageLimit) + m_opt->m_bufferLength + m_opt->m_unfilterTailBufferLength) << PageSizeEx, true, m_opt->m_enableDataCompression);
                int wid = 0;
                if (m_freeWorkSpaceIds == nullptr)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "FreeWorkSpaceIds is not initialized; allocating a new workspace channel.\n");
                    wid = m_workspaceCount.fetch_add(1);
                }
                else if (!m_freeWorkSpaceIds->try_pop(wid))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                                 "Workspace channel pool exhausted; allocating an additional channel.\n");
                    wid = m_workspaceCount.fetch_add(1);
                }
                p_exWorkSpace->m_diskRequests[0].m_status = wid;
                p_exWorkSpace->m_callback = [m_freeWorkSpaceIds = m_freeWorkSpaceIds, wid] () {
                    if (m_freeWorkSpaceIds) m_freeWorkSpaceIds->push(wid);
                };
            }
        }

        ErrorCode AsyncAppend(ExtraWorkSpace* p_exWorkSpace, VectorIndex* p_index, SizeType headID, int appendNum, std::string& appendPosting, int reassignThreshold = 0)
        {
            if (m_asyncAppendQueue.size() >= m_opt->m_asyncAppendQueueSize) {
                std::lock_guard<std::mutex> lock(m_asyncAppendLock);
                if (m_asyncAppendQueue.size() < m_opt->m_asyncAppendQueueSize) {
                    m_asyncAppendQueue.push(AppendPair(p_index->GetPriorityID(headID), headID, appendPosting));
                    return ErrorCode::Success;
                }

                AppendPair workPair;
                ErrorCode ret;
                while (m_asyncAppendQueue.try_pop(workPair)) {
                    if ((ret = Append(p_exWorkSpace, p_index, workPair.headID, 1, workPair.posting, reassignThreshold)) != ErrorCode::Success) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "AsyncAppend: Append failed in async queue processing, headID: %d\n", workPair.headID);
                        return ret;
                    }
                }
            } else {
                m_asyncAppendQueue.push(AppendPair(p_index->GetPriorityID(headID), headID, appendPosting));
            }
            return ErrorCode::Success;
        }

        ErrorCode Append(ExtraWorkSpace* p_exWorkSpace, VectorIndex* p_index, SizeType headID, int appendNum, std::string& appendPosting, int reassignThreshold = 0)
        {
            auto appendBegin = std::chrono::high_resolution_clock::now();
            if (appendPosting.empty()) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Error! empty append posting!\n");
            }

            if (appendNum == 0) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Error!, headID :%d, appendNum:%d\n", headID, appendNum);
            }

        checkDeleted:
            if (!p_index->ContainSample(headID)) {
                for (int i = 0; i < appendNum; i++)
                {
                    uint32_t idx = i * m_vectorInfoSize;
                    SizeType VID = *(int*)(&appendPosting[idx]);
                    uint8_t version = *(uint8_t*)(&appendPosting[idx + sizeof(int)]);
                    auto vectorInfo = std::make_shared<std::string>(appendPosting.c_str() + idx, m_vectorInfoSize);
                    if (m_versionMap->GetVersion(VID) == version) {
                        // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Head Miss To ReAssign: VID: %d, current version: %d\n", *(int*)(&appendPosting[idx]), version);
                        m_stat.m_headMiss++;
                        ReassignAsync(p_index, vectorInfo, headID);
                    }
                    // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Head Miss Do Not To ReAssign: VID: %d, version: %d, current version: %d\n", *(int*)(&appendPosting[idx]), m_versionMap->GetVersion(*(int*)(&appendPosting[idx])), version);
                }
                return ErrorCode::Success;
            }
            double appendIOSeconds = 0;
            {
                //std::shared_lock<std::shared_timed_mutex> lock(m_rwLocks[headID]); //ROCKSDB
                std::unique_lock<std::shared_timed_mutex> lock(m_rwLocks[headID]); //SPDK
                ErrorCode ret;
                if (!p_index->ContainSample(headID)) {
                    lock.unlock();
                    goto checkDeleted;
                }
                if (m_postingSizes.GetSize(headID) + appendNum > (m_postingSizeLimit + m_bufferSizeLimit)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "After appending, the number of vectors in %d exceeds the postingsize + buffersize (%d + %d)! Do split now...\n", headID, m_postingSizeLimit, m_bufferSizeLimit);
                    ret = Split(p_exWorkSpace, p_index, headID, !m_opt->m_disableReassign, false, false);
                    if (ret != ErrorCode::Success)
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Split %d failed!\n", headID);
                    lock.unlock();
                    goto checkDeleted;
                }

                auto appendIOBegin = std::chrono::high_resolution_clock::now();
                if ((ret = db->Merge(
                         headID, appendPosting, MaxTimeout, &(p_exWorkSpace->m_diskRequests),
                         [this, prefixChecksum = *m_checkSums[headID]](const void *val, const int size) -> bool {
                    return this->m_checkSum.ValidateChecksum((const char*)val, size, prefixChecksum);
                })) != ErrorCode::Success)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Merge failed for %d! Posting Size:%d, limit: %d\n", headID, m_postingSizes.GetSize(headID), m_postingSizeLimit);
                    GetDBStats();
                    return ret;
                }
                auto appendIOEnd = std::chrono::high_resolution_clock::now();
                appendIOSeconds = std::chrono::duration_cast<std::chrono::microseconds>(appendIOEnd - appendIOBegin).count();
                *m_checkSums[headID] =
                    m_checkSum.AppendChecksum(*m_checkSums[headID], appendPosting.c_str(), (int)(appendPosting.size()));
                m_postingSizes.IncSize(headID, appendNum);
                if (m_opt->m_consistencyCheck && (ret = db->Check(headID, m_postingSizes.GetSize(headID) * m_vectorInfoSize, nullptr)) != ErrorCode::Success)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Append: Check failed after Merge %d, append %d vectors with size %d\n", headID, appendNum, (int)(appendPosting.size()));
                    return ret;
                }
            }
            if (m_postingSizes.GetSize(headID) > (m_postingSizeLimit + reassignThreshold)) {
                // SizeType VID = *(int*)(&appendPosting[0]);
                // SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Split Triggered by inserting VID: %d, reAssign: %d\n", VID, reassignThreshold);
                // GetDBStats();
                // if (m_postingSizes.GetSize(headID) > 120) {
                //     GetDBStats();
                // }
                if (!reassignThreshold) SplitAsync(p_index, headID);
                else Split(p_exWorkSpace, p_index, headID, !m_opt->m_disableReassign);
                // SplitAsync(p_index, headID);
            }
            auto appendEnd = std::chrono::high_resolution_clock::now();
            double elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(appendEnd - appendBegin).count();
            if (!reassignThreshold) {
                m_stat.m_appendTaskNum++;
                m_stat.m_appendIOCost += appendIOSeconds;
                m_stat.m_appendCost += elapsedMSeconds;
            }
            // } else {
            //     SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "ReAssign Append To: %d\n", headID);
            // }
            return ErrorCode::Success;
        }
        
        ErrorCode Reassign(ExtraWorkSpace* p_exWorkSpace, VectorIndex* p_index, std::shared_ptr<std::string> vectorInfo, SizeType HeadPrev)
        {
            SizeType VID = *((SizeType*)vectorInfo->c_str());
            uint8_t version = *((uint8_t*)(vectorInfo->c_str() + sizeof(VID)));
            // return;
            // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "ReassignID: %d, version: %d, current version: %d, HeadPrev: %d\n", VID, version, m_versionMap->GetVersion(VID), HeadPrev);
            if (m_versionMap->Deleted(VID) || m_versionMap->GetVersion(VID) != version) {
                return ErrorCode::Success;
            }
            auto reassignBegin = std::chrono::high_resolution_clock::now();

            m_stat.m_reAssignNum++;

            auto selectBegin = std::chrono::high_resolution_clock::now();
            std::vector<Edge> selections(static_cast<size_t>(m_opt->m_replicaCount));
            int replicaCount;
            bool isNeedReassign = RNGSelection(selections, (ValueType*)(vectorInfo->c_str() + m_metaDataSize), p_index, VID, replicaCount, HeadPrev);
            auto selectEnd = std::chrono::high_resolution_clock::now();
            auto elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(selectEnd - selectBegin).count();
            m_stat.m_selectCost += elapsedMSeconds;

            auto reassignAppendBegin = std::chrono::high_resolution_clock::now();
            // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Need ReAssign\n");
            if (isNeedReassign && m_versionMap->GetVersion(VID) == version) {
                // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Update Version: VID: %d, version: %d, current version: %d\n", VID, version, m_versionMap.GetVersion(VID));
                m_versionMap->IncVersion(VID, &version);
                (*vectorInfo)[sizeof(VID)] = version;

                //LOG(Helper::LogLevel::LL_Info, "Reassign: oldVID:%d, replicaCount:%d, candidateNum:%d, dist0:%f\n", oldVID, replicaCount, i, selections[0].distance);
                for (int i = 0; i < replicaCount && m_versionMap->GetVersion(VID) == version; i++) {
                    //LOG(Helper::LogLevel::LL_Info, "Reassign: headID :%d, oldVID:%d, newVID:%d, posting length: %d, dist: %f, string size: %d\n", headID, oldVID, VID, m_postingSizes[headID].load(), selections[i].distance, newPart.size());
                    ErrorCode tmp = Append(p_exWorkSpace, p_index, selections[i].node, 1, *vectorInfo, 3);
                    if (ErrorCode::Success != tmp) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Head Miss: VID: %d, current version: %d, another re-assign\n", VID, version);
                        return tmp;
                    }
                }
            }
            auto reassignAppendEnd = std::chrono::high_resolution_clock::now();
            elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(reassignAppendEnd - reassignAppendBegin).count();
            m_stat.m_reAssignAppendCost += elapsedMSeconds;

            auto reassignEnd = std::chrono::high_resolution_clock::now();
            elapsedMSeconds = std::chrono::duration_cast<std::chrono::microseconds>(reassignEnd - reassignBegin).count();
            m_stat.m_reAssignCost += elapsedMSeconds;
            return ErrorCode::Success;
        }

        bool LoadIndex(Options& p_opt, COMMON::VersionLabel& p_versionMap, COMMON::Dataset<std::uint64_t>& p_vectorTranslateMap,  std::shared_ptr<VectorIndex> m_index) override {
            m_versionMap = &p_versionMap;
            m_opt = &p_opt;
	        m_vectorTranslateMap = &p_vectorTranslateMap;
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "DataBlockSize: %d, Capacity: %d\n", m_opt->m_datasetRowsInBlock, m_opt->m_datasetCapacity);
            std::string versionmapPath = m_opt->m_indexDirectory + FolderSep + m_opt->m_deleteIDFile;
            std::string postingSizePath = m_opt->m_indexDirectory + FolderSep + m_opt->m_ssdInfoFile;
            std::string checksumPath = m_opt->m_indexDirectory + FolderSep + m_opt->m_checksumFile;
            if (m_opt->m_recovery) {
                versionmapPath = m_opt->m_persistentBufferPath + FolderSep + m_opt->m_deleteIDFile;
                postingSizePath = m_opt->m_persistentBufferPath + FolderSep + m_opt->m_ssdInfoFile;
                checksumPath = m_opt->m_persistentBufferPath + FolderSep + m_opt->m_checksumFile;
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Recovery: Loading version map\n");
                m_versionMap->Load(versionmapPath, m_opt->m_datasetRowsInBlock, m_opt->m_datasetCapacity);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Recovery: Loading posting size\n");
                m_postingSizes.Load(postingSizePath, m_opt->m_datasetRowsInBlock, m_opt->m_datasetCapacity);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Recovery: Loading posting checksum\n");
                m_checkSums.Load(checksumPath, m_opt->m_datasetRowsInBlock, m_opt->m_datasetCapacity);

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Recovery: Current vector num: %d.\n", m_versionMap->Count());
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Recovery:Current posting num: %d.\n", m_postingSizes.GetPostingNum());
            }
            else if (m_opt->m_storage == Storage::ROCKSDBIO) {
                m_versionMap->Load(versionmapPath, m_opt->m_datasetRowsInBlock, m_opt->m_datasetCapacity);
                m_postingSizes.Load(postingSizePath, m_opt->m_datasetRowsInBlock, m_opt->m_datasetCapacity);
                m_checkSums.Load(checksumPath, m_opt->m_datasetRowsInBlock, m_opt->m_datasetCapacity);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Current vector num: %d.\n", m_versionMap->Count());
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Current posting num: %d.\n", m_postingSizes.GetPostingNum());
            } else if (m_opt->m_storage == Storage::SPDKIO || m_opt->m_storage == Storage::FILEIO) {
		        if (fileexists((m_opt->m_indexDirectory + FolderSep + m_opt->m_ssdIndex).c_str())) {
                	m_versionMap->Initialize(m_opt->m_vectorSize, m_opt->m_datasetRowsInBlock, m_opt->m_datasetCapacity);
			        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Copying data from static to SPDK\n");
			        std::shared_ptr<IExtraSearcher> storeExtraSearcher;
			        storeExtraSearcher.reset(new ExtraStaticSearcher<ValueType>());
			        if (!storeExtraSearcher->LoadIndex(*m_opt, *m_versionMap, p_vectorTranslateMap, m_index)) {
			            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Load Static Index Initialize Error\n");
			            return false;
			        }
			        int totalPostingNum = m_index->GetNumSamples();

			        m_postingSizes.Initialize((SizeType)(totalPostingNum), m_opt->m_datasetRowsInBlock, m_opt->m_datasetCapacity);
                    m_checkSums.Initialize((SizeType)(totalPostingNum), 1, m_opt->m_datasetRowsInBlock,
                                           m_opt->m_datasetCapacity);

			        std::vector<std::thread> threads;
			        std::atomic_size_t vectorsSent(0);
                    ErrorCode ret = ErrorCode::Success;
			        auto func = [&]() {
                        ExtraWorkSpace workSpace;
                        InitWorkSpace(&workSpace);
                        size_t index = 0;
                        while (true)
                        {
                            index = vectorsSent.fetch_add(1);
                            if (index < totalPostingNum)
                            {

                                if ((index & ((1 << 14) - 1)) == 0)
                                {
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Copy to SPDK: Sent %.2lf%%...\n",
                                                 index * 100.0 / totalPostingNum);
                                }
                                std::string tempPosting;
                                if (storeExtraSearcher->GetWritePosting(&workSpace, index, tempPosting) !=
                                    ErrorCode::Success)
                                {
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Static Index Read Posting fail\n");
                                    ret = ErrorCode::Fail;
                                    return;
                                }
                                int vectorNum = (int)(tempPosting.size() / (m_vectorInfoSize - sizeof(uint8_t)));

                                if (vectorNum > m_postingSizeLimit)
                                    vectorNum = m_postingSizeLimit;
                                auto *postingP = reinterpret_cast<char *>(tempPosting.data());
                                std::string newPosting(m_vectorInfoSize * vectorNum, '\0');
                                char *ptr = (char *)(newPosting.c_str());
                                for (int j = 0; j < vectorNum; ++j, ptr += m_vectorInfoSize)
                                {
                                    char *vectorInfo = postingP + j * (m_vectorInfoSize - sizeof(uint8_t));
                                    int VID = *(reinterpret_cast<int *>(vectorInfo));
                                    uint8_t version = m_versionMap->GetVersion(VID);
                                    memcpy(ptr, &VID, sizeof(int));
                                    memcpy(ptr + sizeof(int), &version, sizeof(uint8_t));
                                    memcpy(ptr + sizeof(int) + sizeof(uint8_t), vectorInfo + sizeof(int),
                                           m_vectorInfoSize - sizeof(uint8_t) - sizeof(int));
                                }
                                if (GetWritePosting(&workSpace, index, newPosting, true) != ErrorCode::Success)
                                {
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Index Write Posting fail\n");                                  
                                    ret = ErrorCode::Fail;
                                    return;
                                }
                            }
                            else
                            {
                                return;
                            }
                        }
                    };
			    for (int j = 0; j < m_opt->m_iSSDNumberOfThreads; j++) { threads.emplace_back(func); }
			    for (auto& thread : threads) { thread.join(); }
                if (ret != ErrorCode::Success)
                    return false;
		    } else {
                        m_versionMap->Load(versionmapPath, m_opt->m_datasetRowsInBlock, m_opt->m_datasetCapacity);
                        m_postingSizes.Load(postingSizePath, m_opt->m_datasetRowsInBlock, m_opt->m_datasetCapacity);
                        m_checkSums.Load(checksumPath, m_opt->m_datasetRowsInBlock, m_opt->m_datasetCapacity);
		    } 
	    }
            // Unfilter-tail sidecar: load if present, fall back to pure=total.
            LoadOrInitPostingPureCounts();
            // Dual-pool v3: head role sidecar (optional; absent = all heads are H1).
            LoadHeadRole();
            if (m_opt->m_enablePrimaryHeadBypass) {
                const std::string primaryPath =
                    m_opt->m_indexDirectory + FolderSep + m_opt->m_primaryHeadCSRFile;
                if (m_primaryHeadCSR.Load(primaryPath, static_cast<std::uint32_t>(m_postingSizes.GetPostingNum()))) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                 "[PrimaryHeadCSR] loaded %s (%llu entries).\n",
                                 primaryPath.c_str(),
                                 static_cast<unsigned long long>(m_primaryHeadCSR.Header().entryCount));
                } else {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                                 "[PrimaryHeadCSR] bypass enabled but sidecar unavailable or invalid: %s.\n",
                                 primaryPath.c_str());
                }
            }
            // OPQ prefilter: optional offline sidecar export, or load-for-search. In-posting
            // OPQ-DB mode (config PostingQuantizer=OPQ) auto-loads the codebook for the ADC
            // screen + rerank, so the index is searchable config-only (no env needed).
            {
                const char* ex = std::getenv("SPTAG_OPQ_EXPORT");
                if (ex && ex[0] == '1') ExportOPQSidecars();
                const char* pf = std::getenv("SPTAG_OPQ_PREFILTER");
                if ((pf && pf[0] == '1') || m_opqInpostDb) LoadOPQPrefilter();
            }
            if (!OpenDynamicVectorStore(false)) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                             "[TaggedUpdate] failed to load dynamic vector sidecar.\n");
                return false;
            }
            // In-posting quantization: one-time offline transform (rewrites postings
            // to slim quantized records, writes the inpost_quant.bin marker).
            if (m_inpostQuantBits > 0) {
                const char* qbuild = std::getenv("SPTAG_INPOST_QUANT_BUILD");
                if (qbuild && qbuild[0] == '1') QuantizeInPostings();
            }
            // In-posting RaBitQ b1: one-time offline transform (rewrites postings to
            // [meta | b1-code], writes the inpost_rbq.bin marker).
            if (m_inpostRbq) {
                const char* rbuild = std::getenv("SPTAG_INPOST_RBQ_BUILD");
                if (rbuild && rbuild[0] == '1') {
                    const char* contig = std::getenv("SPTAG_INPOST_RBQ_CONTIG");
                    if (contig && contig[0] == '1') TransformInPostingsRbqContig();
                    else TransformInPostingsRbq();
                }
            }
            // In-posting OPQ/PipePQ (DB-resident): one-time offline transform that
            // rewrites the posting store records to [meta | code] (codes vid-indexed).
            if (m_opqInpostDb) {
                if (m_pipePQ) {
                    const char* pbuild = std::getenv("SPTAG_PIPEPQ_INPOST_DB_BUILD");
                    if (pbuild && pbuild[0] == '1') TransformInPostingsPipePQ();
                } else {
                    const char* obuild = std::getenv("SPTAG_OPQ_INPOST_DB_BUILD");
                    if (obuild && obuild[0] == '1') TransformInPostingsOpq();
                }
            }
            {
                const char* tailRewrite = std::getenv("SPTAG_TAIL_REWRITE_ONLY");
                if (tailRewrite && tailRewrite[0] == '1') RewriteTailOnly(m_index);
            }
            if (m_opt->m_update) {
                if (m_splitThreadPool == nullptr) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPFresh: initialize thread pools, append: %d, reassign %d\n", m_opt->m_appendThreadNum, m_opt->m_reassignThreadNum);

                    m_splitThreadPool = std::make_shared<SPDKThreadPool>();
                    m_splitThreadPool->initSPDK(m_opt->m_appendThreadNum, this);
                    //m_reassignThreadPool = std::make_shared<SPDKThreadPool>();
                    //m_reassignThreadPool->initSPDK(m_opt->m_reassignThreadNum, this);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPFresh: finish initialization\n");
                }
                
                if (m_opt->m_enableWAL && !m_opt->m_persistentBufferPath.empty()) {
                    std::string p_persistenWAL = m_opt->m_persistentBufferPath + FolderSep + "WAL";
                    std::shared_ptr<Helper::KeyValueIO> pdb;
#ifdef ROCKSDB
                    pdb.reset(new RocksDBIO(p_persistenWAL.c_str(), false, false));
                    m_wal.reset(new PersistentBuffer(pdb));
#else
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SPFresh: Wal only support RocksDB! Please use -DROCKSDB when doing cmake.\n");
                    return false;
#endif
                } 
            }

            /** recover the previous WAL **/
            if (m_opt->m_recovery && m_opt->m_enableWAL && m_wal) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Recovery: WAL\n");
                std::string assignment;
                int countAssignment = 0;
                if (!m_wal->StartToScan(assignment)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Recovery: No log\n");
                    return true;
                }
                ExtraWorkSpace workSpace;
                InitWorkSpace(&workSpace);
                do {
                    countAssignment++;
                    if (countAssignment % 10000 == 0) SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Process %d logs\n", countAssignment);
                    char* ptr = (char*)(assignment.c_str());
                    SizeType VID = *(reinterpret_cast<SizeType*>(ptr));
                    if (assignment.size() == m_vectorInfoSize) {
                        if (VID >= m_versionMap->GetVectorNum()) {
                            if (m_versionMap->AddBatch(VID - m_versionMap->GetVectorNum() + 1) != ErrorCode::Success) {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "MemoryOverFlow: VID: %d, Map Size:%d\n", VID, m_versionMap->BufferSize());
                                return false;
                            }
                        }
                        std::shared_ptr<VectorSet> vectorSet;
                        vectorSet.reset(new BasicVectorSet(ByteArray((std::uint8_t*)ptr + sizeof(SizeType) + sizeof(uint8_t), sizeof(ValueType) * 1 * m_opt->m_dim, false),
                            GetEnumValueType<ValueType>(), m_opt->m_dim, 1));
                        AddIndex(&workSpace, vectorSet, m_index, VID);
                    } else {
                        m_versionMap->Delete(VID);
                    }
                } while (m_wal->NextToScan(assignment));
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Recovery: No more to repeat, wait for rebalance\n");
                while(!AllFinished())
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                }
            }
            return true;
        }
        bool ValidatePostings(
            std::vector<SizeType> &pids, std::vector<Helper::PageBuffer<std::uint8_t>> &postings,
            bool allowPurePrefix = false)
        {
            if (!m_opt->m_checksumInRead) return true;

            for (int i = 0; i < pids.size(); i++)
            {
                const size_t fullBytes = static_cast<size_t>(m_postingSizes.GetSize(pids[i])) *
                                         static_cast<size_t>(m_vectorInfoSize);
                const size_t readBytes = postings[i].GetAvailableSize();
                if (allowPurePrefix && readBytes < fullBytes && m_hasPostingPureCounts) {
                    const int pureCount = m_postingPureCounts.GetSize(pids[i]);
                    const size_t expectedPrefix = IsUnfilterOnlyHead(pids[i])
                        ? (std::min)(fullBytes, static_cast<size_t>(m_vectorInfoSize))
                        : (pureCount > 0
                            ? (std::min)(fullBytes, static_cast<size_t>(pureCount) *
                                                         static_cast<size_t>(m_vectorInfoSize))
                            : fullBytes);
                    // A full-posting checksum cannot validate an intentionally truncated
                    // pure-prefix read. Only accept the exact prefix requested by the
                    // filtered IO path; other short reads remain errors.
                    if (readBytes == expectedPrefix) {
                        continue;
                    }
                }
                if (!m_checkSum.ValidateChecksum((const char *)(postings[i].GetBuffer()),
                                                 readBytes, *m_checkSums[pids[i]]))
                {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "ValidatePostings fail: posting id:%d, required size:%d, buffer size:%d, checksum:%d\n",
                        pids[i], (int)fullBytes, (int)readBytes, (int)(*m_checkSums[pids[i]]));
                    return false;
                }
            }
            return true;
        }

        bool ValidatePostings(std::vector<SizeType> &pids, std::vector<std::string> &postings)
        {
            if (!m_opt->m_checksumInRead) return true;

            ErrorCode ret;
            for (int i = 0; i < pids.size(); i++)
            {
                if (!m_checkSum.ValidateChecksum(postings[i].c_str(),
                                                 postings[i].size(), *m_checkSums[pids[i]]))
                {
                    SPTAGLIB_LOG(
                        Helper::LogLevel::LL_Error,
                        "ValidatePostings fail: posting id:%d, required size:%d, buffer size:%d, checksum:%d\n",
                        pids[i], (int)(m_postingSizes.GetSize(pids[i]) * m_vectorInfoSize),
                        (int)(postings[i].size()), (int)(*m_checkSums[pids[i]]));
                    PrintErrorInPosting(postings[i], pids[i]);
                    return false;
                }
            }
            return true;
        }

        // Phase 2: reorder each posting's PURE prefix [0,pure) so vectors are
        // sorted by the hierarchy tuple (org,dept,team,project). Because the tags
        // are a strict nested hierarchy, sorting by the tuple makes EVERY level's
        // vectors contiguous simultaneously, so a tag's vectors collapse onto the
        // minimum number of pages (read amplification floor). The unfilter-only
        // tail [pure,total) is left untouched. One-time, persisted via Checkpoint.
        bool ReorderPostingsByTag(ExtraWorkSpace* ws) {
            int st = m_reorderState.load();
            if (st == 2) return true;
            if (st == -1) return false;
            std::lock_guard<std::mutex> g(m_reorderMutex);
            st = m_reorderState.load();
            if (st == 2) return true;
            if (st == -1) return false;

            // SPTAG_REORDER_ATTR=k pins the PRIMARY sort key to tag column k
            // (e.g. year, month) instead of the full lexicographic tuple. For
            // NON-hierarchical facets the tuple-sort only makes column 0
            // contiguous; a single chosen column makes THAT column's values
            // collapse onto the minimum pages so page-select can prune reads for
            // queries on that column. Remaining columns follow as stable tiebreak.
            static const int s_reorderAttr = []() {
                const char* e = std::getenv("SPTAG_REORDER_ATTR");
                return e ? std::atoi(e) : -1;
            }();
            const int reorderAttr =
                (s_reorderAttr >= 0 && s_reorderAttr < m_numTagsPerVec) ? s_reorderAttr : -1;

            SizeType numHeads = m_postingSizes.GetPostingNum();
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "[Reorder] sorting pure region of %d postings by %s (one-time)...\n",
                (int)numHeads,
                reorderAttr >= 0 ? ("attr-col " + std::to_string(reorderAttr)).c_str() : "tag tuple");
            // Simple one-time offline pass: for each head, read its posting,
            // stable-sort the pure region by the chosen attribute column (full
            // tuple as tiebreak), write it back, refresh the checksum. This is a
            // one-time operation triggered on first query and checkpointed, so a
            // straightforward sequential loop is sufficient.
            size_t reordered = 0, vecsSorted = 0;
            std::string buf;
            for (SizeType hid = 0; hid < numHeads; hid++) {
                buf.clear();
                if (db->Get(hid, &buf, MaxTimeout, &(ws->m_diskRequests)) != ErrorCode::Success) continue;
                int total = (int)(buf.size() / m_vectorInfoSize);
                if (total <= 1) continue;
                int pure = m_hasPostingPureCounts ? m_postingPureCounts.GetSize(hid) : total;
                if (pure < 0) pure = 0;
                if (pure > total) pure = total;
                if (pure <= 1) continue;
                const char* base = buf.data();
                auto tagsOf = [&](int i) -> const uint32_t* {
                    return reinterpret_cast<const uint32_t*>(
                        base + (size_t)i * m_vectorInfoSize + sizeof(int) + sizeof(uint8_t));
                };
                std::vector<int> order(pure);
                for (int i = 0; i < pure; i++) order[i] = i;
                std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
                    const uint32_t* ta = tagsOf(a);
                    const uint32_t* tb = tagsOf(b);
                    if (reorderAttr >= 0 && ta[reorderAttr] != tb[reorderAttr])
                        return ta[reorderAttr] < tb[reorderAttr];
                    for (int t = 0; t < m_numTagsPerVec; t++)
                        if (ta[t] != tb[t]) return ta[t] < tb[t];
                    return a < b;
                });
                bool changed = false;
                for (int i = 0; i < pure; i++) if (order[i] != i) { changed = true; break; }
                if (!changed) continue;
                std::string nb(buf.size(), '\0');
                char* dst = &nb[0];
                for (int i = 0; i < pure; i++)
                    memcpy(dst + (size_t)i * m_vectorInfoSize,
                           base + (size_t)order[i] * m_vectorInfoSize, m_vectorInfoSize);
                if (total > pure)
                    memcpy(dst + (size_t)pure * m_vectorInfoSize,
                           base + (size_t)pure * m_vectorInfoSize,
                           (size_t)(total - pure) * m_vectorInfoSize);
                if (db->Put(hid, nb, MaxTimeout, &(ws->m_diskRequests)) != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[Reorder] Put failed at head %d\n", (int)hid);
                    m_reorderState.store(-1);
                    return false;
                }
                *m_checkSums[hid] = m_checkSum.CalcChecksum(nb.c_str(), (int)nb.size());
                reordered++;
                vecsSorted += (size_t)pure;
                if ((hid % 100000) == 0)
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "[Reorder] progress %d/%d heads, %zu reordered\n",
                        (int)hid, (int)numHeads, reordered);
            }
            ErrorCode cp = Checkpoint(m_opt->m_indexDirectory);
            if (cp != ErrorCode::Success) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[Reorder] Checkpoint failed\n");
                m_reorderState.store(-1);
                return false;
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "[Reorder] done: %zu/%d postings reordered, %zu vectors sorted, checkpointed\n",
                reordered, (int)numHeads, vecsSorted);
            m_reorderState.store(2);
            return true;
        }


        // ── Page-selective directory (env SPTAG_PAGE_SELECT=1) ──────────────
        // Build one 256-bit signature per 4KB page of a posting from the
        // authoritative on-disk bytes. A page's signature is the OR of the tag
        // bits of every record whose bytes fall (wholly or partly) in that page.
        void BuildPagePSFromBuffer(SizeType hid, const std::uint8_t* data, size_t postingSize) {
            auto& pages = m_pagePS[hid];
            if (postingSize == 0 || data == nullptr) { pages.clear(); return; }
            int numPages = (int)((postingSize + PageSize - 1) >> PageSizeEx);
            int n = (int)(postingSize / m_vectorInfoSize);
            pages.assign(numPages, SPTAG::Cache::PageBitmask{});
            for (int j = 0; j < n; j++) {
                size_t sb = (size_t)j * m_vectorInfoSize;
                size_t eb = sb + m_vectorInfoSize - 1;
                int p0 = (int)(sb >> PageSizeEx);
                int p1 = (int)(eb >> PageSizeEx);
                const uint32_t* vt = reinterpret_cast<const uint32_t*>(
                    data + sb + sizeof(int) + sizeof(uint8_t));
                for (int p = p0; p <= p1 && p < numPages; p++)
                    for (int t = 0; t < m_numTagsPerVec; t++) pages[p].Insert(vt[t]);
            }
        }

        bool LoadPagePS(const std::string& path, SizeType numHeads) {
            std::ifstream in(path, std::ios::binary);
            if (!in.is_open()) return false;
            std::uint32_t magic = 0; std::int32_t nh = 0, bits = 0;
            in.read(reinterpret_cast<char*>(&magic), 4);
            in.read(reinterpret_cast<char*>(&nh), 4);
            in.read(reinterpret_cast<char*>(&bits), 4);
            if (!in || magic != 0x32534750u || nh != (std::int32_t)numHeads ||
                bits != SPTAG::Cache::PS_PAGE_BITS) return false;
            for (SizeType h = 0; h < numHeads; h++) {
                std::int32_t np = 0;
                in.read(reinterpret_cast<char*>(&np), 4);
                if (!in || np < 0) return false;
                m_pagePS[h].assign(np, SPTAG::Cache::PageBitmask{});
                if (np > 0) in.read(reinterpret_cast<char*>(m_pagePS[h].data()),
                                    (std::streamsize)np * sizeof(SPTAG::Cache::PageBitmask));
                if (!in) return false;
            }
            return true;
        }

        void SavePagePS(const std::string& path, SizeType numHeads) {
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            if (!out.is_open()) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                    "[PageSelect] cannot write %s (directory cached only)\n", path.c_str());
                return;
            }
            std::uint32_t magic = 0x32534750u;
            std::int32_t nh = (std::int32_t)numHeads, bits = SPTAG::Cache::PS_PAGE_BITS;
            out.write(reinterpret_cast<const char*>(&magic), 4);
            out.write(reinterpret_cast<const char*>(&nh), 4);
            out.write(reinterpret_cast<const char*>(&bits), 4);
            for (SizeType h = 0; h < numHeads; h++) {
                std::int32_t np = (std::int32_t)m_pagePS[h].size();
                out.write(reinterpret_cast<const char*>(&np), 4);
                if (np > 0) out.write(reinterpret_cast<const char*>(m_pagePS[h].data()),
                                      (std::streamsize)np * sizeof(SPTAG::Cache::PageBitmask));
            }
        }

        // Build (or load) the per-page signature directory once, reading every
        // posting through the provided workspace. Returns true when ready.
        bool EnsurePagePS(ExtraWorkSpace* ws) {
            int st = m_pagePSState.load();
            if (st == 2) return true;
            if (st == -1) return false;
            std::lock_guard<std::mutex> g(m_pagePSMutex);
            st = m_pagePSState.load();
            if (st == 2) return true;
            if (st == -1) return false;

            SizeType numHeads = m_postingSizes.GetPostingNum();
            m_pagePS.assign(numHeads, {});
            std::string path = m_opt->m_indexDirectory + FolderSep + "page_signatures.bin";

            if (LoadPagePS(path, numHeads)) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                    "[PageSelect] loaded page-signature directory (%d heads) from %s\n",
                    (int)numHeads, path.c_str());
                m_pagePSState.store(2);
                return true;
            }

            int batch = (int)ws->m_pageBuffers.size();
            if (batch <= 0) { m_pagePSState.store(-1); return false; }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "[PageSelect] building page-signature directory for %d heads (one-time full posting scan)...\n",
                (int)numHeads);
            std::chrono::microseconds buildTimeout(3600000000LL);
            std::vector<SizeType> keys; keys.reserve(batch);
            for (SizeType start = 0; start < numHeads; start += batch) {
                keys.clear();
                SizeType end = (std::min)(start + (SizeType)batch, numHeads);
                for (SizeType h = start; h < end; h++) keys.push_back(h);
                ErrorCode err = db->MultiGet(keys, ws->m_pageBuffers, buildTimeout,
                                             &(ws->m_diskRequests));
                if (err != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                        "[PageSelect] build read fail at head %d\n", (int)start);
                    m_pagePSState.store(-1);
                    return false;
                }
                for (int li = 0; li < (int)keys.size(); li++) {
                    auto& buf = ws->m_pageBuffers[li];
                    BuildPagePSFromBuffer(keys[li],
                        reinterpret_cast<const std::uint8_t*>(buf.GetBuffer()),
                        buf.GetAvailableSize());
                }
            }
            SavePagePS(path, numHeads);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "[PageSelect] page-signature directory ready (%d heads)\n", (int)numHeads);
            m_pagePSState.store(2);
            return true;
        }

        ErrorCode SearchPrimaryHeadCandidates(ExtraWorkSpace* p_exWorkSpace,
                                               QueryResult& p_queryResults,
                                               std::shared_ptr<VectorIndex> /*p_index*/) override
        {
            if (!m_primaryHeadCSR.Loaded() || p_exWorkSpace == nullptr) {
                return ErrorCode::Fail;
            }

            const SPTAG::Cache::DNFPredicate* dnf = p_exWorkSpace->m_dnf;
            std::uint32_t projectTag = 0;
            if (dnf != nullptr && !dnf->Empty()) {
                // A project equality literal is a safe sparse candidate generator
                // only when it constrains the sole DNF clause. Remaining
                // categorical/numeric literals are evaluated exactly below.
                if (dnf->clauses.size() != 1) return ErrorCode::Fail;
                bool foundProjectAnchor = false;
                for (const auto& literal : dnf->clauses.front().lits) {
                    if (literal.kind == 0 && literal.col == 3 &&
                        literal.op == SPTAG::Cache::DNF_EQ &&
                        m_primaryHeadCSR.IsProjectTag(literal.val)) {
                        projectTag = literal.val;
                        foundProjectAnchor = true;
                        break;
                    }
                }
                if (!foundProjectAnchor) return ErrorCode::Fail;
            } else {
                if (p_exWorkSpace->m_numQueryTags != 1 || p_exWorkSpace->m_queryTags == nullptr) {
                    return ErrorCode::Fail;
                }
                projectTag = p_exWorkSpace->m_queryTags[0];
            }

            if (!m_primaryHeadCSR.IsProjectTag(projectTag)) {
                return ErrorCode::Fail;
            }

            COMMON::QueryResultSet<ValueType>& queryResults =
                *((COMMON::QueryResultSet<ValueType>*) & p_queryResults);
            std::vector<int> candidates;
            candidates.reserve(p_exWorkSpace->m_postingIDs.size() * 10);
            p_exWorkSpace->m_deduper.clear();

            for (SizeType headId : p_exWorkSpace->m_postingIDs) {
                if (headId < 0 || headId >= m_primaryHeadCSR.HeadCount()) continue;
                const PrimaryHeadCSREntry* begin =
                    m_primaryHeadCSR.Begin(static_cast<std::uint32_t>(headId));
                const PrimaryHeadCSREntry* end =
                    m_primaryHeadCSR.End(static_cast<std::uint32_t>(headId));
                for (const PrimaryHeadCSREntry* entry = begin; entry != end; ++entry) {
                    const int vid = static_cast<int>(entry->vid);
                    std::uint32_t vecTags[5];
                    m_primaryHeadCSR.UnpackAttributes(*entry, vecTags);
                    if (vid < 0 || vid >= m_versionMap->Count() ||
                        m_versionMap->Deleted(vid) ||
                        !m_primaryHeadCSR.MatchesProject(*entry, projectTag) ||
                        (dnf != nullptr && !dnf->Matches(vecTags, 5)) ||
                        p_exWorkSpace->m_deduper.CheckAndSet(vid)) {
                        continue;
                    }
                    candidates.push_back(vid);
                }
            }

            const int rerankLimit = m_opt->m_primaryHeadBypassRerankL;
            if (rerankLimit > 0 && static_cast<int>(candidates.size()) > rerankLimit) {
                candidates.resize(static_cast<size_t>(rerankLimit));
            }

            p_exWorkSpace->m_postingProbeStats.m_primaryHeadCandidates = candidates.size();
            queryResults.Reset();
            RerankFromVecDB(candidates, queryResults.GetTarget(), m_opt->m_dim, queryResults);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Debug,
                         "[PrimaryHeadCSR] heads=%zu project=%u candidates=%zu\n",
                         p_exWorkSpace->m_postingIDs.size(), projectTag, candidates.size());
            return ErrorCode::Success;
        }

        virtual ErrorCode SearchIndex(ExtraWorkSpace* p_exWorkSpace,
            QueryResult& p_queryResults,
            std::shared_ptr<VectorIndex> p_index,
            SearchStats* p_stats, std::set<int>* truth, std::map<int, std::set<int>>* found) override
        {
            if (m_opqPF) return SearchIndexOPQ(p_exWorkSpace, p_queryResults, p_index, p_stats, truth, found);
            if (p_stats) p_stats->m_exSetUpLatency = 0;

            // Phase 2 one-time within-posting reorder (env SPTAG_REORDER_POSTINGS=1).
            // Runs before the page directory is built so signatures reflect sorted bytes.
            static const bool s_reorderPostings = []() {
                const char* env = std::getenv("SPTAG_REORDER_POSTINGS");
                return env && env[0] == '1';
            }();
            if (s_reorderPostings && m_reorderState.load() != 2) ReorderPostingsByTag(p_exWorkSpace);


            COMMON::QueryResultSet<ValueType>& queryResults = *((COMMON::QueryResultSet<ValueType>*) & p_queryResults);
            int diskRead = 0;
            int diskIO = 0;
            int listElements = 0;
            // In-posting RaBitQ b1: estimate from the in-posting code during the scan,
            // collect survivors, then exact-rerank the top-L from the mmap'd base.
            void* rbqCtx = nullptr;
            std::vector<std::pair<float, int>> rbqSurv;
            if (m_inpostRbq && m_inpostRbq2 && m_inpostRbq2->Loaded()) {
                rbqCtx = m_inpostRbq2->AllocQuery();
                int dim = m_opt->m_dim;
                std::vector<float> qf = WidenQuery(
                    reinterpret_cast<const ValueType*>(queryResults.GetQuantizedTarget()), dim);
                m_inpostRbq2->PrepareQuery(rbqCtx, qf.data());
                rbqSurv.reserve(4096);
            }
            // VIDs that passed the exact inline DNF filter during the posting scan.
            // Used by the final pass to drop head-graph candidates that were added
            // under the coarse union mask but do not satisfy the DNF predicate.
            std::unordered_set<SizeType> dnfMatched;

            double compLatency = 0;
            double readLatency = 0;
            const std::chrono::microseconds hardLatencyLimit = HardLatencyLimit();
            std::chrono::microseconds remainLimit;
            if (p_stats) remainLimit = hardLatencyLimit - std::chrono::microseconds((int)p_stats->m_totalLatency);
            else remainLimit = hardLatencyLimit;

            auto readStart = std::chrono::high_resolution_clock::now();

            // PS posting-level pre-filter: remove posting IDs that cannot contain
            // matching ACL/tags BEFORE reading from SSD.
            if (p_exWorkSpace->m_postingFilter) {
                auto& ids = p_exWorkSpace->m_postingIDs;
                p_exWorkSpace->m_postingProbeStats.m_prePSPostings += ids.size();
                ids.erase(
                    std::remove_if(ids.begin(), ids.end(),
                        [&](int pid) { return !p_exWorkSpace->m_postingFilter(pid); }),
                    ids.end());
            } else {
                p_exWorkSpace->m_postingProbeStats.m_prePSPostings += p_exWorkSpace->m_postingIDs.size();
            }

            // Decide unfilter-tail BEFORE issuing IO so filtered queries can
            // request a shorter read (skip tail blocks) in a single MultiGet.
            const bool hasInlineTagFilter =
                m_tagBytesPerVec > 0 &&
                p_exWorkSpace->m_queryTags != nullptr &&
                p_exWorkSpace->m_numQueryTags > 0;
            const bool hasDNF =
                m_tagBytesPerVec > 0 &&
                p_exWorkSpace->m_dnf != nullptr &&
                !p_exWorkSpace->m_dnf->Empty();
            {
                static const bool s_dnfDbg = (std::getenv("SPTAG_DNF_DEBUG") != nullptr);
                if (s_dnfDbg) {
                    static std::atomic<int> g_d{0};
                    if (g_d++ < 4)
                        fprintf(stderr, "[DNFdbg] hasDNF=%d dnfPtr=%p numQ=%d tagBytes=%d\n",
                                (int)hasDNF, (void*)p_exWorkSpace->m_dnf,
                                p_exWorkSpace->m_numQueryTags, m_tagBytesPerVec);
                }
            }
            static const bool s_trackAllStats = (std::getenv("SPTAG_TRACK_ALL_STATS") != nullptr);
            const bool trackPostingStats = hasInlineTagFilter || hasDNF || s_trackAllStats;

            // U_extra (unfilter-only) heads are infrastructure for unfiltered
            // recall only. Filtered queries must behave identically to a build
            // without U_extra, so drop role==1 heads from the candidate posting
            // list entirely. Disable for A/B via SPTAG_FILTER_KEEP_UEXTRA=1.
            static const bool s_filterKeepUextra = []() {
                const char* env = std::getenv("SPTAG_FILTER_KEEP_UEXTRA");
                return env && env[0] == '1';
            }();
            if (hasInlineTagFilter && HasHeadRoles() && !s_filterKeepUextra) {
                auto& ids = p_exWorkSpace->m_postingIDs;
                ids.erase(std::remove_if(ids.begin(), ids.end(),
                    [&](int pid) { return IsUnfilterOnlyHead(pid); }), ids.end());
            }
            // Pure/tail split for filtered queries is ON by default whenever the
            // pure-count sidecar exists: filtered queries must never scan the
            // unfilter-only tail. Set SPTAG_UNFILTER_TAIL=0 to force-disable.
            static const bool s_unfilterTailEnabled = []() {
                const char* env = std::getenv("SPTAG_UNFILTER_TAIL");
                return !(env && env[0] == '0');
            }();
            const bool useUnfilterTail =
                s_unfilterTailEnabled && m_hasPostingPureCounts && hasInlineTagFilter;

            // Page-selective IO (env SPTAG_PAGE_SELECT=1): for filtered queries,
            // read only the posting pages whose per-page signature may contain a
            // queried tag, then skip records on unread pages during the scan. The
            // exact tag filter in the scan loop guards correctness (false positives
            // only cost extra reads; the directory has no false negatives).
            static const bool s_pageSelect = []() {
                const char* env = std::getenv("SPTAG_PAGE_SELECT");
                return env && env[0] == '1';
            }();
            bool usePageSelect = s_pageSelect && hasInlineTagFilter && m_numTagsPerVec > 0;
            if (usePageSelect && !EnsurePagePS(p_exWorkSpace)) usePageSelect = false;

            // Diagnostic: full read, but measure the page-floor (pages that truly
            // contain a matching vector) vs the signature-selected pages, to
            // attribute over-read to false positives vs genuine tag dilution.
            static const bool s_pageDiag = []() {
                const char* env = std::getenv("SPTAG_PAGE_DIAG");
                return env && env[0] == '1';
            }();
            const bool runDiag = s_pageDiag && hasInlineTagFilter && m_numTagsPerVec > 0
                                 && !usePageSelect && EnsurePagePS(p_exWorkSpace);

            // Per-posting page selectors (only populated when usePageSelect).
            std::vector<std::vector<std::uint8_t>> pageSel;
            SPTAG::Cache::PageBitmask qmask;
            if (usePageSelect || runDiag) {
                for (int qi = 0; qi < p_exWorkSpace->m_numQueryTags; qi++)
                    qmask.Insert(static_cast<uint32_t>(p_exWorkSpace->m_queryTags[qi]));
            }
            if (usePageSelect) {
                const auto& ids = p_exWorkSpace->m_postingIDs;
                pageSel.resize(ids.size());
                for (size_t i = 0; i < ids.size(); ++i) {
                    SizeType hid = ids[i];
                    auto& sel = pageSel[i];
                    if (hid < 0 || hid >= (SizeType)m_pagePS.size()) { sel.clear(); continue; }
                    const auto& pages = m_pagePS[hid];
                    int numPages = (int)pages.size();
                    int pStart = 0, pEnd = numPages;
                    if (m_hasPostingPureCounts) {
                        int pure = m_postingPureCounts.GetSize(hid);
                        if (pure > 0)
                            pEnd = (int)((std::min)((size_t)numPages,
                                (size_t)(((size_t)pure * m_vectorInfoSize + PageSize - 1) >> PageSizeEx)));
                    }
                    sel.assign(numPages, 0);
                    for (int p = pStart; p < pEnd; ++p) {
                        bool keep = hasDNF ? p_exWorkSpace->m_dnf->MayMatchPage(pages[p])
                                           : pages[p].MayIntersect(qmask);
                        if (keep) sel[p] = 1;
                    }
                }
            }

            ErrorCode mgErr;
            if (usePageSelect) {
                mgErr = db->MultiGet(p_exWorkSpace->m_postingIDs,
                                     p_exWorkSpace->m_pageBuffers,
                                     pageSel,
                                     remainLimit,
                                     &(p_exWorkSpace->m_diskRequests));
            } else if (useUnfilterTail) {
                // Build per-posting byte cap = pure_count * vectorInfoSize.
                // Block layer rounds up to ceil(cap / PageSize) blocks.
                std::vector<std::uint32_t> maxBytes(p_exWorkSpace->m_postingIDs.size(), 0);
                for (size_t i = 0; i < maxBytes.size(); ++i) {
                    SizeType hid = p_exWorkSpace->m_postingIDs[i];
                    if (IsUnfilterOnlyHead((int)hid)) {
                        // Tail-only (U_extra) head: filtered queries scan nothing
                        // from it, so cap the read to a single vector (minimal IO).
                        maxBytes[i] = static_cast<std::uint32_t>(m_vectorInfoSize);
                        continue;
                    }
                    int pure = m_postingPureCounts.GetSize(hid);
                    if (pure > 0) {
                        maxBytes[i] = static_cast<std::uint32_t>(pure) *
                                      static_cast<std::uint32_t>(m_vectorInfoSize);
                    }
                }
                mgErr = db->MultiGet(p_exWorkSpace->m_postingIDs,
                                     p_exWorkSpace->m_pageBuffers,
                                     maxBytes,
                                     remainLimit,
                                     &(p_exWorkSpace->m_diskRequests));
            } else {
                mgErr = db->MultiGet(p_exWorkSpace->m_postingIDs,
                                     p_exWorkSpace->m_pageBuffers,
                                     remainLimit,
                                     &(p_exWorkSpace->m_diskRequests));
            }
            if (mgErr != ErrorCode::Success ||
                (!usePageSelect &&
                 !ValidatePostings(p_exWorkSpace->m_postingIDs, p_exWorkSpace->m_pageBuffers,
                                   useUnfilterTail)))
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[SearchIndex] read postings fail!\n");
                return ErrorCode::DiskIOFail;
            }
            auto readEnd = std::chrono::high_resolution_clock::now();
            readLatency += ((double)std::chrono::duration_cast<std::chrono::microseconds>(readEnd - readStart).count());

            const auto postingListCount = static_cast<uint32_t>(p_exWorkSpace->m_postingIDs.size());
            if (trackPostingStats) {
                p_exWorkSpace->m_postingProbeStats.m_readPostings += postingListCount;
            }
            for (uint32_t pi = 0; pi < postingListCount; ++pi) {
                auto curPostingID = p_exWorkSpace->m_postingIDs[pi];
                auto& buffer = (p_exWorkSpace->m_pageBuffers[pi]);
                char* p_postingListFullData = (char*)(buffer.GetBuffer());
                int vectorNum = (int)(buffer.GetAvailableSize() / m_vectorInfoSize);
                int scanStart = 0;
                int scanLimit = vectorNum;
                if (useUnfilterTail) {
                    if (IsUnfilterOnlyHead((int)curPostingID)) {
                        // Tail-only (U_extra) head: never scanned by filtered queries.
                        scanLimit = 0;
                    } else {
                        int pure = m_postingPureCounts.GetSize(curPostingID);
                        if (pure > 0 && pure < scanLimit) scanLimit = pure;
                    }
                }
                bool postingHasExactMatch = false;

                if (runDiag) {
                    // Full buffer is present: compute (a) total pages, (b) pages the
                    // signature selects, (c) pages that truly hold a matching vector.
                    int numPages = (int)((buffer.GetAvailableSize() + PageSize - 1) >> PageSizeEx);
                    SizeType hid = curPostingID;
                    const std::vector<SPTAG::Cache::PageBitmask>* pgs =
                        (hid >= 0 && hid < (SizeType)m_pagePS.size()) ? &m_pagePS[hid] : nullptr;
                    std::vector<uint8_t> sigSel(numPages, 0), trueNeed(numPages, 0);
                    int purePages = numPages;
                    if (m_hasPostingPureCounts) {
                        int pure = m_postingPureCounts.GetSize(hid);
                        purePages = (pure <= 0) ? 0 : (std::min)(numPages,
                            (int)(((size_t)pure * m_vectorInfoSize + PageSize - 1) >> PageSizeEx));
                    }
                    if (pgs) for (int p = 0; p < purePages && p < (int)pgs->size(); ++p)
                        if ((*pgs)[p].MayIntersect(qmask)) sigSel[p] = 1;
                    for (int i = 0; i < scanLimit; i++) {
                        const char* vi = p_postingListFullData + (size_t)i * m_vectorInfoSize;
                        const uint32_t* vt = reinterpret_cast<const uint32_t*>(vi + sizeof(int) + sizeof(uint8_t));
                        bool m = false;
                        if (hasDNF) {
                            m = p_exWorkSpace->m_dnf->Matches(vt, m_numTagsPerVec);
                        } else {
                            for (int t = 0; t < m_numTagsPerVec && !m; t++)
                                for (int qi = 0; qi < p_exWorkSpace->m_numQueryTags && !m; qi++)
                                    if (vt[t] == p_exWorkSpace->m_queryTags[qi]) m = true;
                        }
                        if (m) { int p = (int)(((size_t)i * m_vectorInfoSize) >> PageSizeEx); if (p < numPages) trueNeed[p] = 1; }
                    }
                    int tot = numPages, sig = 0, need = 0, sigAndNeed = 0;
                    for (int p = 0; p < numPages; ++p) {
                        sig += sigSel[p]; need += trueNeed[p];
                        if (sigSel[p] && trueNeed[p]) ++sigAndNeed;
                    }
                    // Posting-level accounting:
                    //   need==0 -> posting holds NO matching vector (a "false
                    //             positive" posting that the centroid search picked).
                    //   sig>0   -> signature would KEEP this posting (read >=1 page).
                    //   sig>0 && need==0 -> signature FAILED to prune a non-matching
                    //             posting = posting-level signature false positive.
                    static std::atomic<size_t> g_tot{0}, g_sig{0}, g_need{0}, g_post{0}, g_q{0};
                    static std::atomic<size_t> g_postNeed0{0}, g_postSigKeep{0}, g_postSigFP{0};
                    g_tot += tot; g_sig += sig; g_need += need; g_post += 1;
                    if (need == 0) g_postNeed0 += 1;
                    if (sig > 0) g_postSigKeep += 1;
                    if (sig > 0 && need == 0) g_postSigFP += 1;
                    (void)sigAndNeed;
                    if (pi + 1 == postingListCount) {
                        size_t q = ++g_q;
                        if (q % 500 == 0) {
                            size_t T=g_tot,S=g_sig,N=g_need,P=g_post;
                            size_t PN0=g_postNeed0, PSK=g_postSigKeep, PSF=g_postSigFP;
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                "[PageDiag] q=%zu postings=%zu pages/posting total=%.2f sigSelected=%.2f trueNeeded=%.2f  FP-overread=%.2fx  floor-prune=%.2fx\n",
                                q, P/q?P/q:0, (double)T/P, (double)S/P, (double)N/P,
                                N>0?(double)S/N:0.0, N>0?(double)T/N:0.0);
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                "[PageDiag] posting-level: read=%zu  noMatch(need0)=%zu (%.1f%%)  sigKeeps=%zu  sigFP(keep&noMatch)=%zu  posting-FP-rate=%.1f%%\n",
                                P, PN0, 100.0*PN0/P, PSK, PSF,
                                PSK>0?100.0*PSF/PSK:0.0);
                        }
                    }
                }

                if (usePageSelect) {
                    // Count only the pages actually read (selected) for honest IO stats.
                    int selPages = 0;
                    const std::vector<std::uint8_t>& sel = pageSel[pi];
                    for (size_t p = 0; p < sel.size(); ++p) if (sel[p]) ++selPages;
                    diskIO += selPages;
                    diskRead += selPages * PageSize;
                } else {
                    diskIO += ((buffer.GetAvailableSize() + PageSize - 1) >> PageSizeEx);
                    diskRead += (int)(buffer.GetAvailableSize());
                }
                
                //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "DEBUG: postingList %d size:%d m_vectorInfoSize:%d vectorNum:%d\n", pi, (int)(postingList.size()), m_vectorInfoSize, vectorNum);
                int realNum = vectorNum;
                listElements += (scanLimit - scanStart);
                auto compStart = std::chrono::high_resolution_clock::now();
                for (int i = scanStart; i < scanLimit; i++) {
                    char* vectorInfo = p_postingListFullData + i * m_vectorInfoSize;
                    if (usePageSelect) {
                        // Skip records whose pages were not read (page-selective IO):
                        // their bytes in the pooled buffer are stale/garbage.
                        const std::vector<std::uint8_t>& sel = pageSel[pi];
                        size_t sb = (size_t)i * m_vectorInfoSize;
                        int p0 = (int)(sb >> PageSizeEx);
                        int p1 = (int)((sb + m_vectorInfoSize - 1) >> PageSizeEx);
                        bool pageOk = true;
                        for (int p = p0; p <= p1; ++p) {
                            if (p >= (int)sel.size() || sel[p] == 0) { pageOk = false; break; }
                        }
                        if (!pageOk) { listElements--; continue; }
                    }
                    int vectorID = *(reinterpret_cast<int*>(vectorInfo));

		            //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "DEBUG: vectorID:%d\n", vectorID);
                    if (m_versionMap->Deleted(vectorID)) {
                        realNum--;
                        listElements--;
                        continue;
                    }

                    bool tagMatch = true;
                    if (hasDNF) {
                        const uint32_t* vecTags = reinterpret_cast<const uint32_t*>(vectorInfo + sizeof(int) + sizeof(uint8_t));
                        tagMatch = p_exWorkSpace->m_dnf->Matches(vecTags, m_numTagsPerVec);
                    } else if (hasInlineTagFilter) {
                        tagMatch = false;
                        const uint32_t* vecTags = reinterpret_cast<const uint32_t*>(vectorInfo + sizeof(int) + sizeof(uint8_t));
                        for (int ti = 0; ti < m_numTagsPerVec && !tagMatch; ti++) {
                            for (int qi = 0; qi < p_exWorkSpace->m_numQueryTags && !tagMatch; qi++) {
                                if (vecTags[ti] == p_exWorkSpace->m_queryTags[qi]) tagMatch = true;
                            }
                        }
                    }

                    if (trackPostingStats) {
                        ++p_exWorkSpace->m_postingProbeStats.m_scannedVectors;
                        if (tagMatch) ++p_exWorkSpace->m_postingProbeStats.m_matchedVectors;
                    }

                    if (tagMatch) {
                        postingHasExactMatch = true;
                        if (hasDNF) dnfMatched.insert((SizeType)vectorID);
                    }

                    if(p_exWorkSpace->m_deduper.CheckAndSet(vectorID)) {
                        listElements--;
                        continue;
                    }
                    // Inline tag filter: check tags embedded in posting metadata
                    if (!tagMatch) {
                        listElements--;
                        continue;
                    }
                    if (rbqCtx) {
                        // In-posting RaBitQ: screen by estimate, defer exact dist to rerank.
                        const void* binPtr = vectorInfo + m_metaDataSize;
                        const void* exPtr = (m_inpostRbqExBytes > 0)
                            ? (const void*)(vectorInfo + m_metaDataSize + m_inpostRbqBinBytes) : nullptr;
                        float est = m_inpostRbq2->EstimateCode(rbqCtx, binPtr, exPtr);
                        rbqSurv.emplace_back(est, vectorID);
                        continue;
                    }
                    auto distance2leaf = (m_inpostQuantBits > 0)
                        ? InpostL2(queryResults.GetQuantizedTarget(), vectorInfo + m_metaDataSize, m_opt->m_dim)
                        : p_index->ComputeDistance(queryResults.GetQuantizedTarget(), vectorInfo + m_metaDataSize);
                    queryResults.AddPoint(vectorID, distance2leaf);
                }
                if (trackPostingStats && postingHasExactMatch) {
                    ++p_exWorkSpace->m_postingProbeStats.m_matchedPostings;
                }
                auto compEnd = std::chrono::high_resolution_clock::now();
                if (realNum <= m_mergeThreshold && m_taggedMaintenance.load(std::memory_order_acquire)) {
                    std::lock_guard<std::mutex> lock(m_taggedMergeCandidatesLock);
                    m_taggedMergeCandidates.insert(curPostingID);
                }
                // Async merge requires update-mode thread pools; in read-only serving they are not initialized.
                else if (m_opt->m_update && m_opt->m_asyncMergeInSearch &&
                         m_splitThreadPool != nullptr && realNum <= m_mergeThreshold) {
                    MergeAsync(p_index.get(), curPostingID);
                }

                compLatency += ((double)std::chrono::duration_cast<std::chrono::microseconds>(compEnd - compStart).count());

                if (truth) {
                    for (int i = 0; i < vectorNum; ++i) {
                        if (usePageSelect) {
                            const std::vector<std::uint8_t>& sel = pageSel[pi];
                            size_t sb = (size_t)i * m_vectorInfoSize;
                            int p0 = (int)(sb >> PageSizeEx);
                            int p1 = (int)((sb + m_vectorInfoSize - 1) >> PageSizeEx);
                            bool pageOk = true;
                            for (int p = p0; p <= p1; ++p) {
                                if (p >= (int)sel.size() || sel[p] == 0) { pageOk = false; break; }
                            }
                            if (!pageOk) continue;
                        }
                        char* vectorInfo = p_postingListFullData + i * m_vectorInfoSize;
                        int vectorID = *(reinterpret_cast<int*>(vectorInfo));
                        if (truth->count(vectorID) != 0)
                            (*found)[curPostingID].insert(vectorID);
                    }
                }
            }

            // In-posting RaBitQ: exact-rerank the top-L survivors (by estimate) from
            // the mmap'd full-precision base file. Head-graph seeds already in
            // queryResults carry exact distances and are kept (not reset).
            if (rbqCtx) {
                int L = m_inpostRerankL;
                int total = (int)rbqSurv.size();
                int lim = (total < L) ? total : L;
                if (total > lim) {
                    std::nth_element(rbqSurv.begin(), rbqSurv.begin() + lim, rbqSurv.end(),
                        [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first < b.first; });
                }
                const int dim = m_opt->m_dim;
                std::vector<int> rvids(lim);
                for (int i = 0; i < lim; i++) rvids[i] = rbqSurv[i].second;
                // Deep-queue libaio batch over the flat O_DIRECT base (PipeANN-style
                // full queue depth) is the default. SPTAG_INPOST_LIBAIO_RERANK=0
                // reverts to RocksDB MultiGet / serial O_DIRECT (which falls back
                // automatically here too if the flat base / AIO pool is unavailable).
                static const bool s_libaioRerank = []() {
                    const char* e = std::getenv("SPTAG_INPOST_LIBAIO_RERANK");
                    return (e == nullptr) || (std::atoi(e) != 0);
                }();
                bool reranked = false;
                if (s_libaioRerank) {
                    reranked = RerankBaseDirectBatch(rvids, queryResults.GetQuantizedTarget(), dim, queryResults);
                }
                if (!reranked) {
                    if (m_opqVecDB) {
                        // Batched async cold rerank: ONE MultiGet over all L survivors
                        // (libaio parallel, DIRECT_IO => no residency). No L-serial preads.
                        RerankFromVecDB(rvids, queryResults.GetQuantizedTarget(), dim, queryResults);
                    } else {
                        for (int i = 0; i < lim; i++) {
                            int vid = rbqSurv[i].second;
                            const ValueType* fv = ReadBaseVecDirect(vid, dim);
                            if (fv) {
                                float d = p_index->ComputeDistance(queryResults.GetQuantizedTarget(), fv);
                                queryResults.AddPoint(vid, d);
                            }
                        }
                    }
                }
                m_inpostRbq2->FreeQuery(rbqCtx);
                rbqCtx = nullptr;
            }

            if (p_stats)
            {
                p_stats->m_compLatency = compLatency / 1000;
                p_stats->m_diskReadLatency = readLatency / 1000;
                p_stats->m_totalListElementsCount = listElements;
                p_stats->m_diskIOCount = diskIO;
                p_stats->m_diskAccessCount = diskRead / 1024;
            }
            {
                static const bool s_vanStats = []() { const char* e = std::getenv("SPTAG_OPQ_STATS"); return e && e[0] == '1'; }();
                if (s_vanStats) {
                    static std::atomic<size_t> g_bytes{ 0 }, g_scan{ 0 }, g_q{ 0 };
                    size_t q = ++g_q;
                    g_bytes += (size_t)diskRead;
                    g_scan += (size_t)listElements;
                    if (q % 1000 == 0) {
                        size_t by = g_bytes, sc = g_scan;
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                            "[VAN stats] q=%zu diskBytes/q=%.0f scanned/q=%.1f\n",
                            q, (double)by / q, (double)sc / q);
                    }
                }
            }
            // Final exact DNF pass: head-graph candidates are added to the result
            // set using only the coarse union hier-mask (they must be added to drive
            // which postings get scanned). Under a DNF predicate a head's OWN vector
            // may satisfy the union but not the DNF, so re-check every surviving
            // result against the exact predicate and drop the ones that fail. The
            // inline posting-scan filter already guarantees posting members are
            // DNF-correct; this only removes the head leak. No-op without DNF.
            static const bool s_dnfNoDrop = []() { const char* e = std::getenv("SPTAG_DNF_NODROP"); return e && e[0] == '1'; }();
            // The drop pass fixes a "head leak": head-graph candidates are added
            // to the result set using only the coarse categorical union mask (a
            // posting-level OR of member tags), which never guarantees the head's
            // OWN vector satisfies the predicate -- so a head whose own vector
            // fails the DNF can leak into the results. This affects categorical
            // AND-clauses, numeric ranges (numeric values are excluded from the
            // categorical union mask entirely), AND pure categorical OR (the union
            // is a posting-membership test, not a per-vector test).
            //
            // We re-evaluate the exact DNF directly against each surviving
            // result's inline per-vector tags (m_vectorTags) when available. This
            // is exact and independent of whether the result's posting happened to
            // be scanned, so it removes leaks with NO recall loss (legitimately
            // matching results are kept). Falls back to the dnfMatched membership
            // test only when inline tags are unavailable.
            if (hasDNF && !s_dnfNoDrop) {
                int rn = queryResults.GetResultNum();
                bool anyDropped = false;
                for (int i = 0; i < rn; ++i) {
                    BasicResult* r = queryResults.GetResult(i);
                    if (r == nullptr || r->VID < 0) continue;
                    bool matches;
                    if (m_tagBytesPerVec > 0 && r->VID >= 0 &&
                        (size_t)r->VID * m_numTagsPerVec < m_vectorTags.size()) {
                        matches = p_exWorkSpace->m_dnf->Matches(
                            &m_vectorTags[(size_t)r->VID * m_numTagsPerVec], m_numTagsPerVec);
                    } else {
                        matches = (dnfMatched.find((SizeType)r->VID) != dnfMatched.end());
                    }
                    if (!matches) {
                        r->VID = -1;
                        r->Dist = MaxDist;
                        anyDropped = true;
                    }
                }
                if (anyDropped) queryResults.SortResult();
            }
            queryResults.SetScanned(listElements);
            return ErrorCode::Success;
        }

        virtual ErrorCode SearchIndexWithoutParsing(ExtraWorkSpace* p_exWorkSpace)
        {
            int retry = 0;
            ErrorCode ret = ErrorCode::Undefined;
            while (retry < 2 && ret != ErrorCode::Success)
            {
                ret = db->MultiGet(p_exWorkSpace->m_postingIDs, p_exWorkSpace->m_pageBuffers, HardLatencyLimit(),
                                   &(p_exWorkSpace->m_diskRequests));
                retry++;
            }
            if (ret == ErrorCode::Success &&
                !ValidatePostings(p_exWorkSpace->m_postingIDs, p_exWorkSpace->m_pageBuffers))
            {
                return ErrorCode::DiskIOFail;
            }
            return ret;
        }

        virtual ErrorCode SearchNextInPosting(ExtraWorkSpace* p_exWorkSpace, QueryResult& p_headResults,
            QueryResult& p_queryResults,
            std::shared_ptr<VectorIndex>& p_index, const VectorIndex* p_spann)
        {
            COMMON::QueryResultSet<ValueType>& headResults = *((COMMON::QueryResultSet<ValueType>*) & p_headResults);
            COMMON::QueryResultSet<ValueType>& queryResults = *((COMMON::QueryResultSet<ValueType>*) & p_queryResults);
            bool foundResult = false;
            BasicResult* head = headResults.GetResult(p_exWorkSpace->m_ri);
            while (!foundResult && p_exWorkSpace->m_pi < p_exWorkSpace->m_postingIDs.size()) {
                if (head && head->VID != -1 && p_exWorkSpace->m_ri <= p_exWorkSpace->m_pi) {
                    if (!m_versionMap->Deleted(head->VID) && !p_exWorkSpace->m_deduper.CheckAndSet(head->VID)) {
                        queryResults.AddPoint(head->VID, head->Dist);
                        foundResult = true;
                    }
                    head = headResults.GetResult(++p_exWorkSpace->m_ri);
                    continue;
                }
                auto& buffer = (p_exWorkSpace->m_pageBuffers[p_exWorkSpace->m_pi]);
                char* p_postingListFullData = (char*)(buffer.GetBuffer());
                int vectorNum = (int)(buffer.GetAvailableSize() / m_vectorInfoSize);
                while (p_exWorkSpace->m_offset < vectorNum) {
                    char* vectorInfo = p_postingListFullData + p_exWorkSpace->m_offset * m_vectorInfoSize;
                    p_exWorkSpace->m_offset++;

                    int vectorID = *(reinterpret_cast<int*>(vectorInfo));
                    if (vectorID >= m_versionMap->Count()) return ErrorCode::Key_OverFlow;
                    if (m_versionMap->Deleted(vectorID)) continue;
                    if (p_exWorkSpace->m_deduper.CheckAndSet(vectorID)) continue;

                    float distance2leaf;
                    if (m_inpostRbq && m_inpostBaseFd >= 0 && (size_t)vectorID < m_inpostBaseN) {
                        // In-posting RaBitQ: this streaming path has no rerank buffer,
                        // so cold-read the exact vector from the O_DIRECT base (no residency).
                        const ValueType* fv = ReadBaseVecDirect((int)vectorID, m_opt->m_dim);
                        if (!fv) continue;
                        distance2leaf = p_index->ComputeDistance(queryResults.GetQuantizedTarget(), fv);
                    } else {
                        distance2leaf = (m_inpostQuantBits > 0)
                            ? InpostL2(queryResults.GetQuantizedTarget(), vectorInfo + m_metaDataSize, m_opt->m_dim)
                            : p_index->ComputeDistance(queryResults.GetQuantizedTarget(), vectorInfo + m_metaDataSize);
                    }
                    queryResults.AddPoint(vectorID, distance2leaf);
                    foundResult = true;
                    break;
                }
                if (p_exWorkSpace->m_offset == vectorNum) {
                    p_exWorkSpace->m_pi++;
                    p_exWorkSpace->m_offset = 0;
                }
            }
            while (!foundResult && head && head->VID != -1) {
                if (!m_versionMap->Deleted(head->VID) && !p_exWorkSpace->m_deduper.CheckAndSet(head->VID)) {
                    queryResults.AddPoint(head->VID, head->Dist);
                    foundResult = true;
                }
                head = headResults.GetResult(++p_exWorkSpace->m_ri);
            }
            if (foundResult) p_queryResults.SetScanned(p_queryResults.GetScanned() + 1);
            return (foundResult) ? ErrorCode::Success : ErrorCode::VectorNotFound;
        }

        virtual ErrorCode SearchIterativeNext(ExtraWorkSpace* p_exWorkSpace, QueryResult& p_headResults,
            QueryResult& p_query,
            std::shared_ptr<VectorIndex> p_index, const VectorIndex* p_spann)
        {
            if (p_exWorkSpace->m_loadPosting) {
                ErrorCode ret = SearchIndexWithoutParsing(p_exWorkSpace);
                if (ret != ErrorCode::Success) return ret;
                p_exWorkSpace->m_ri = 0;
                p_exWorkSpace->m_pi = 0;
                p_exWorkSpace->m_offset = 0;
                p_exWorkSpace->m_loadPosting = false;
            }

            return SearchNextInPosting(p_exWorkSpace, p_headResults, p_query, p_index, p_spann);
        }

        std::string GetPostingListFullData(
            int postingListId,
            size_t p_postingListSize,
            Selection& p_selections,
            std::shared_ptr<VectorSet> p_fullVectors,
            bool p_enableDeltaEncoding = false,
            bool p_enablePostingListRearrange = false,
            const ValueType* headVector = nullptr)
        {
            std::string postingListFullData("");
            std::string vectors("");
            std::string vectorIDs("");
            size_t selectIdx = p_selections.lower_bound(postingListId);
            // iterate over all the vectors in the posting list
            for (int i = 0; i < p_postingListSize; ++i)
            {
                if (p_selections[selectIdx].node != postingListId)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Selection ID NOT MATCH! node:%d offset:%zu\n", postingListId, selectIdx);
                    throw std::runtime_error("Selection ID mismatch");
                }
                std::string vectorID("");
                std::string vector("");

                int vid = p_selections[selectIdx++].tonode;
                vectorID.append(reinterpret_cast<char*>(&vid), sizeof(int));

                ValueType* p_vector = reinterpret_cast<ValueType*>(p_fullVectors->GetVector(vid));
                if (p_enableDeltaEncoding)
                {
                    DimensionType n = p_fullVectors->Dimension();
                    std::vector<ValueType> p_vector_delta(n);
                    for (auto j = 0; j < n; j++)
                    {
                        p_vector_delta[j] = p_vector[j] - headVector[j];
                    }
                    vector.append(reinterpret_cast<char*>(&p_vector_delta[0]), p_fullVectors->PerVectorDataSize());
                }
                else
                {
                    vector.append(reinterpret_cast<char*>(p_vector), p_fullVectors->PerVectorDataSize());
                }

                if (p_enablePostingListRearrange)
                {
                    vectorIDs += vectorID;
                    vectors += vector;
                }
                else
                {
                    postingListFullData += (vectorID + vector);
                }
            }
            if (p_enablePostingListRearrange)
            {
                return vectors + vectorIDs;
            }
            return postingListFullData;
        }

        bool BuildIndex(std::shared_ptr<Helper::VectorSetReader>& p_reader, std::shared_ptr<VectorIndex> p_headIndex, Options& p_opt, COMMON::VersionLabel& p_versionMap, COMMON::Dataset<std::uint64_t>& p_vectorTranslateMap, SizeType upperBound = -1) override {
            m_versionMap = &p_versionMap;
            m_vectorTranslateMap = &p_vectorTranslateMap;
            m_opt = &p_opt;

            int numThreads = m_opt->m_iSSDNumberOfThreads;
            int candidateNum = m_opt->m_internalResultNum;
            std::unordered_map<SizeType, SizeType> headVectorIDS;
            if (m_opt->m_headIDFile.empty()) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Not found VectorIDTranslate!\n");
                return false;
            }

            for (int i = 0; i < p_vectorTranslateMap.R(); i++)
            {
                headVectorIDS[static_cast<SizeType>(*(p_vectorTranslateMap[i]))] = i;
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loaded %u Vector IDs\n", static_cast<uint32_t>(headVectorIDS.size()));

            SizeType fullCount = 0;
            {
                auto fullVectors = p_reader->GetVectorSet();
                fullCount = fullVectors->Count();
            }
            if (upperBound > 0) fullCount = upperBound;

            // m_metaDataSize and m_vectorInfoSize already set in constructor
            // (includes tag bytes if m_numTagsPerVec > 0)
            m_vectorInfoSize = m_opt->m_dim * sizeof(ValueType) + m_metaDataSize;

            // In-posting quantization: the constructor sized the search stride to the
            // SLIM record. A fresh build needs FULL-stride membership (so each posting
            // holds the same vectors as a full build), then writes SLIM codes. Restore
            // the full-stride size limits here and install the build-time slim writer.
            if (m_inpostRbq && m_quantFullVectorInfoSize > 0) {
                m_postingSizeLimit = m_opt->m_postingPageLimit * PageSize / m_vectorInfoSize;
                m_bufferSizeLimit = m_opt->m_bufferLength * PageSize / m_vectorInfoSize;
                m_tailBufferSizeLimit = m_opt->m_unfilterTailBufferLength * PageSize / m_vectorInfoSize;
                if (!m_buildSlimRbq && !m_inpostRbqPathResolved.empty()) {
                    SetupBuildSlimRbq(m_inpostRbqPathResolved);
                }
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                    "[InpostRBQ] build-time slim: full stride=%d postingSizeLimit=%d, writing slim stride=%d\n",
                    m_vectorInfoSize, m_postingSizeLimit, m_buildSlimStride);
            }
            // In-posting OPQ: same single-pass slim build as RaBitQ above (full-stride
            // membership, slim [meta|OPQ-code] records written via SerializeSlimOpq).
            else if (m_opqInpostDb && m_quantFullVectorInfoSize > 0) {
                m_postingSizeLimit = m_opt->m_postingPageLimit * PageSize / m_vectorInfoSize;
                m_bufferSizeLimit = m_opt->m_bufferLength * PageSize / m_vectorInfoSize;
                m_tailBufferSizeLimit = m_opt->m_unfilterTailBufferLength * PageSize / m_vectorInfoSize;
                if (m_pipePQ) {
                    if (!m_buildSlimPipePQ) {
                        SetupBuildSlimPipePQ(m_opqInpostDbM);
                    }
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "[InpostPipePQ] build-time slim: full stride=%d postingSizeLimit=%d, writing slim stride=%d\n",
                        m_vectorInfoSize, m_postingSizeLimit, m_buildSlimStride);
                } else if (!m_buildSlimOpq) {
                    SetupBuildSlimOpq(m_opqInpostDbM);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "[InpostOPQ] build-time slim: full stride=%d postingSizeLimit=%d, writing slim stride=%d\n",
                        m_vectorInfoSize, m_postingSizeLimit, m_buildSlimStride);
                }
            }

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Build SSD Index.\n");

            std::vector<std::vector<SizeType>> plannedNodeVectors;
            std::vector<std::vector<int>> vectorMemberships;
            bool useNodeAwareBuild = !m_plannedNodeVectorAssignments.empty();
            size_t plannedAssignmentCount = static_cast<size_t>(fullCount);
            // Prefer primary assignments (each vector owned by exactly one node) for
            // posting placement so each vector contributes a unique posting footprint.
            // Multi-membership planned assignments are kept only for head-routing/ACL.
            const std::vector<std::vector<SizeType>>& postingPlacementSource =
                !m_primaryNodeVectorAssignments.empty()
                    ? m_primaryNodeVectorAssignments
                    : m_plannedNodeVectorAssignments;
            if (useNodeAwareBuild)
            {
                plannedNodeVectors.resize(postingPlacementSource.size());
                vectorMemberships.assign(fullCount, std::vector<int>());
                plannedAssignmentCount = 0;

                std::vector<uint8_t> claimedVector(fullCount, 0);
                for (size_t nodeId = 0; nodeId < postingPlacementSource.size(); ++nodeId)
                {
                    for (SizeType vectorId : postingPlacementSource[nodeId])
                    {
                        if (vectorId < 0 || vectorId >= fullCount) {
                            continue;
                        }
                        if (claimedVector[static_cast<size_t>(vectorId)]) {
                            continue;
                        }
                        claimedVector[static_cast<size_t>(vectorId)] = 1;

                        plannedNodeVectors[nodeId].push_back(vectorId);
                        vectorMemberships[vectorId].push_back(static_cast<int>(nodeId));
                        ++plannedAssignmentCount;
                    }
                }

                useNodeAwareBuild = plannedAssignmentCount > 0;
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                             "Node-aware build: posting placement source=%s, unique assignments=%zu across %zu nodes\n",
                             (!m_primaryNodeVectorAssignments.empty() ? "primary" : "planned"),
                             plannedAssignmentCount,
                             postingPlacementSource.size());
            }

            Selection selections((useNodeAwareBuild ? plannedAssignmentCount : static_cast<size_t>(fullCount)) * m_opt->m_replicaCount, m_opt->m_tmpdir);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Full vector count:%d Edge bytes:%llu selection size:%zu, capacity size:%zu\n", fullCount, sizeof(Edge), selections.m_selections.size(), selections.m_selections.capacity());
            std::vector<std::atomic_int> replicaCount(fullCount);
            std::vector<std::atomic_int> postingListSize(p_headIndex->GetNumSamples());
            for (auto& rc : replicaCount) rc = 0;
            for (auto& pls : postingListSize) pls = 0;
            std::unordered_set<SizeType> emptySet;
            SizeType batchSize = (fullCount + m_opt->m_batches - 1) / m_opt->m_batches;

            auto t1 = std::chrono::high_resolution_clock::now();
            if (!useNodeAwareBuild && p_opt.m_batches > 1)
            {
                if (selections.SaveBatch() != ErrorCode::Success)
                {
                    return false;
                }
            }
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Preparation done, start candidate searching.\n");
                if (useNodeAwareBuild)
                {
                    auto fullVectors = p_reader->GetVectorSet();
                    if (m_opt->m_distCalcMethod == DistCalcMethod::Cosine && !p_reader->IsNormalized()) {
                        fullVectors->Normalize(m_opt->m_iSSDNumberOfThreads);
                    }

                    std::vector<int> primaryOwner(fullCount, -1);
                    if (!m_primaryNodeVectorAssignments.empty())
                    {
                        for (size_t nodeId = 0; nodeId < m_primaryNodeVectorAssignments.size(); ++nodeId)
                        {
                            for (SizeType vectorId : m_primaryNodeVectorAssignments[nodeId])
                            {
                                if (vectorId >= 0 && vectorId < fullCount) {
                                    primaryOwner[vectorId] = static_cast<int>(nodeId);
                                }
                            }
                        }
                    }

                    std::vector<int> headToNode(p_headIndex->GetNumSamples(), -1);
                    for (const auto& pair : headVectorIDS)
                    {
                        if (pair.first < 0 || pair.first >= fullCount) {
                            continue;
                        }

                        int assignedNode = -1;
                        auto ownerIt = m_headVectorOwners.find(pair.first);
                        if (ownerIt != m_headVectorOwners.end()) {
                            assignedNode = ownerIt->second;
                        } else if (pair.first >= 0 && pair.first < static_cast<SizeType>(primaryOwner.size()) && primaryOwner[pair.first] >= 0) {
                            assignedNode = primaryOwner[pair.first];
                        } else if (!vectorMemberships[pair.first].empty()) {
                            assignedNode = vectorMemberships[pair.first].front();
                        }
                        if (assignedNode < 0) {
                            assignedNode = 0;
                        }
                        headToNode[pair.second] = assignedNode;
                    }

                    std::vector<std::vector<uint8_t>> allowedHeadMasks(plannedNodeVectors.size(), std::vector<uint8_t>(p_headIndex->GetNumSamples(), 0));
                    std::vector<size_t> nodeHeadCounts(plannedNodeVectors.size(), 0);
                    // Dual-pool v3: collect U_extra head ordinals per node (legacy: skipped from normal RNGSelection)
                    std::vector<std::vector<SizeType>> uExtraOrdPerNode(plannedNodeVectors.size());
                    // By default U_extra heads now participate in the normal inverse RNG
                    // assignment, so they receive full-sized postings (~replicaCount * N / heads)
                    // that compete with H1 for replicas -- same as ordinary heads. Set
                    // SPTAG_UEXTRA_FULL_POSTING=0 to restore the legacy behavior (exclude
                    // U_extra from the mask, then give each only a k=replicaCount kNN posting).
                    static const bool s_uextraFullPosting = []() {
                        const char* v = std::getenv("SPTAG_UEXTRA_FULL_POSTING");
                        return !(v && (v[0] == '0' || v[0] == 'f' || v[0] == 'F'));
                    }();
                    // Tail-only U_extra (corrected design, default ON): U_extra heads are
                    // NOT pinned into any subset's pure clustering. They are excluded from
                    // the pure mask (so their pure_count stays 0) and receive members ONLY
                    // through the global tag-agnostic unfilter-tail (K_replica / Phase 4).
                    // Each subset thus clusters purely over its own H1 heads, while U_extra
                    // act as global unfilter-only heads filled by the tail mechanism.
                    // Set SPTAG_UEXTRA_TAIL_ONLY=0 to restore legacy pure-posting behavior.
                    static const bool s_uextraTailOnly = []() {
                        const char* v = std::getenv("SPTAG_UEXTRA_TAIL_ONLY");
                        return !(v && (v[0] == '0' || v[0] == 'f' || v[0] == 'F'));
                    }();
                    for (SizeType headId = 0; headId < static_cast<SizeType>(headToNode.size()); ++headId)
                    {
                        int nodeId = headToNode[headId];
                        if (nodeId >= 0 && nodeId < static_cast<int>(allowedHeadMasks.size())) {
                            if (m_hasHeadRole && (s_uextraTailOnly || !s_uextraFullPosting) && IsUnfilterOnlyHead(static_cast<int>(headId))) {
                                // U_extra excluded from this subset's pure clustering.
                                uExtraOrdPerNode[static_cast<size_t>(nodeId)].push_back(headId);
                            } else {
                                allowedHeadMasks[static_cast<size_t>(nodeId)][static_cast<size_t>(headId)] = 1;
                                ++nodeHeadCounts[static_cast<size_t>(nodeId)];
                            }
                        }
                    }
                    for (size_t nodeId = 0; nodeId < allowedHeadMasks.size(); ++nodeId)
                    {
                        if (!plannedNodeVectors[nodeId].empty() && nodeHeadCounts[nodeId] == 0)
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                         "Node-aware build requires dedicated heads, but node %d has none.\n",
                                         static_cast<int>(nodeId));
                            return false;
                        }
                    }

                    std::vector<std::pair<int, SizeType>> assignmentEntries;
                    assignmentEntries.reserve(plannedAssignmentCount);
                    for (size_t nodeId = 0; nodeId < plannedNodeVectors.size(); ++nodeId)
                    {
                        for (SizeType vectorId : plannedNodeVectors[nodeId])
                        {
                            assignmentEntries.emplace_back(static_cast<int>(nodeId), vectorId);
                        }
                    }

                    std::atomic_size_t sent(0);
                    std::vector<std::thread> mythreads;
                    mythreads.reserve(numThreads);
                    for (int tid = 0; tid < numThreads; ++tid)
                    {
                        mythreads.emplace_back([&, tid]() {
                            std::vector<Edge> localSelections(static_cast<size_t>(m_opt->m_replicaCount));
                            while (true)
                            {
                                size_t assignmentIdx = sent.fetch_add(1);
                                if (assignmentIdx >= assignmentEntries.size()) {
                                    return;
                                }

                                const auto& assignment = assignmentEntries[assignmentIdx];
                                int nodeId = assignment.first;
                                SizeType vectorId = assignment.second;
                                size_t selectionOffset = assignmentIdx * static_cast<size_t>(m_opt->m_replicaCount);
                                int assignedReplicaCount = 0;

                                auto headIt = headVectorIDS.find(vectorId);
                                int headNodeId = -1;
                                if (headIt != headVectorIDS.end()) {
                                    headNodeId = headToNode[headIt->second];
                                }

                                if (!p_opt.m_excludehead && headIt != headVectorIDS.end() && headNodeId == nodeId)
                                {
                                    Edge& selfSelection = selections.m_selections[selectionOffset];
                                    selfSelection.node = headIt->second;
                                    selfSelection.tonode = vectorId;
                                    selfSelection.distance = 0.0f;
                                    ++postingListSize[selfSelection.node];
                                    ++replicaCount[vectorId];
                                    assignedReplicaCount = 1;
                                }

                                if (assignedReplicaCount >= m_opt->m_replicaCount) {
                                    if (m_opt->m_buildPrimaryHeadCSR) {
                                        Edge& primary = selections.m_selections[selectionOffset];
                                        primary.distance = std::copysign(std::fabs(primary.distance), -1.0f);
                                    }
                                    continue;
                                }

                                std::fill(localSelections.begin(), localSelections.end(), Edge());
                                int localReplicaCount = 0;
                                // Bundle structure constrains RNG replica placement by default:
                                // when a per-node assignment is set, every vector in node N is
                                // only allowed to land on heads that also belong to node N, so
                                // postings are strictly node-pure and PostingSignature can prune
                                // effectively. Set SPTAG_NODE_REPLICA_MASK=0 to fall back to
                                // legacy globally-optimal RNG layout that ignores node identity
                                // (relies on signature filtering at query time instead).
                                static const bool s_disableNodeReplicaMask = []() {
                                    const char* v = std::getenv("SPTAG_NODE_REPLICA_MASK");
                                    return (v && (v[0] == '0' || v[0] == 'f' || v[0] == 'F'));
                                }();
                                const std::vector<uint8_t>* replicaMask = s_disableNodeReplicaMask
                                    ? nullptr
                                    : &allowedHeadMasks[static_cast<size_t>(nodeId)];
                                SizeType replicaMaskHeadCount = (replicaMask != nullptr)
                                    ? static_cast<SizeType>(nodeHeadCounts[static_cast<size_t>(nodeId)])
                                    : static_cast<SizeType>(-1);
                                RNGSelection(localSelections,
                                             (ValueType*)(fullVectors->GetVector(vectorId)),
                                             p_headIndex.get(),
                                             vectorId,
                                             localReplicaCount,
                                             -1,
                                             replicaMask,
                                             replicaMaskHeadCount);

                                for (int selIdx = 0; selIdx < localReplicaCount && assignedReplicaCount < m_opt->m_replicaCount; ++selIdx)
                                {
                                    const Edge& candidate = localSelections[static_cast<size_t>(selIdx)];
                                    bool duplicateHead = false;
                                    for (int prevIdx = 0; prevIdx < assignedReplicaCount; ++prevIdx)
                                    {
                                        if (selections.m_selections[selectionOffset + static_cast<size_t>(prevIdx)].node == candidate.node)
                                        {
                                            duplicateHead = true;
                                            break;
                                        }
                                    }
                                    if (duplicateHead) {
                                        continue;
                                    }

                                    Edge& target = selections.m_selections[selectionOffset + static_cast<size_t>(assignedReplicaCount)];
                                    target = candidate;
                                    ++postingListSize[target.node];
                                    ++replicaCount[vectorId];
                                    ++assignedReplicaCount;
                                }

                                if (m_opt->m_buildPrimaryHeadCSR && assignedReplicaCount > 0) {
                                    Edge& primary = selections.m_selections[selectionOffset];
                                    primary.distance = std::copysign(std::fabs(primary.distance), -1.0f);
                                }
                            }
                        });
                    }
                    for (auto& thread : mythreads) { thread.join(); }
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                 "Node-aware candidate search finished with %zu node/vector assignments across %zu nodes.\n",
                                 assignmentEntries.size(),
                                 plannedNodeVectors.size());

                    // Dual-pool v3: build U_extra postings via k-NN within each bundle.
                    // For each U_extra head u in bundle N, find m_opt->m_replicaCount nearest
                    // base vectors from plannedNodeVectors[N] and add them as postings.
                    // This gives U_extra their own coverage without stealing from H1 postings.
                    // NOTE: skipped in tail-only mode (default) -- U_extra must have
                    // pure_count=0; the unfilter-tail (Phase 4 / K_replica) fills them instead.
                    if (m_hasHeadRole && !s_uextraFullPosting && !s_uextraTailOnly) {
                        size_t totalUExtra = 0;
                        for (auto& v : uExtraOrdPerNode) totalUExtra += v.size();
                        if (totalUExtra > 0) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                "DualPool v3: building U_extra postings via k-NN (%zu total U_extra heads)\n",
                                totalUExtra);
                            int K = m_opt->m_replicaCount;
                            // Grow selections to accommodate U_extra postings
                            size_t origSize = assignmentEntries.size();
                            size_t newSize = origSize + totalUExtra;
                            selections.m_selections.resize(newSize * static_cast<size_t>(K));
                            // Update m_end so operator[] bounds check covers U_extra entries.
                            selections.m_totalsize = newSize * static_cast<size_t>(K);
                            selections.m_end = newSize * static_cast<size_t>(K);

                            size_t uExtraOffset = origSize;
                            for (size_t nodeId = 0; nodeId < uExtraOrdPerNode.size(); ++nodeId) {
                                const auto& uOrdList = uExtraOrdPerNode[nodeId];
                                if (uOrdList.empty()) continue;
                                const auto& nodeVecs = plannedNodeVectors[nodeId];
                                for (SizeType uOrd : uOrdList) {
                                    const ValueType* uVec = (const ValueType*)p_headIndex->GetSample(uOrd);
                                    if (!uVec) { ++uExtraOffset; continue; }
                                    // Find K nearest base vectors in this bundle via brute force
                                    using PFV = std::pair<float, SizeType>;
                                    std::priority_queue<PFV> topK;
                                    for (SizeType vid : nodeVecs) {
                                        const ValueType* bVec = (const ValueType*)fullVectors->GetVector(vid);
                                        if (!bVec) continue;
                                        float d = (float)p_headIndex->ComputeDistance(
                                            static_cast<const void*>(uVec),
                                            static_cast<const void*>(bVec));
                                        if ((int)topK.size() < K)
                                            topK.push({d, vid});
                                        else if (d < topK.top().first) {
                                            topK.pop(); topK.push({d, vid});
                                        }
                                    }
                                    size_t selOff = uExtraOffset * static_cast<size_t>(K);
                                    int kk = 0;
                                    while (!topK.empty() && kk < K) {
                                        auto [d, vid] = topK.top(); topK.pop();
                                        Edge& e = selections.m_selections[selOff + kk];
                                        e.node = uOrd;
                                        e.tonode = vid;
                                        e.distance = d;
                                        ++postingListSize[uOrd];
                                        ++kk;
                                    }
                                    ++uExtraOffset;
                                }
                            }
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                "DualPool v3: U_extra posting build done.\n");
                        }
                    }
                }
                else
                {
                    SizeType sampleSize = m_opt->m_samples;
                    std::vector<SizeType> samples(sampleSize, 0);
                    for (int i = 0; i < m_opt->m_batches; i++) {
                        SizeType start = i * batchSize;
                        SizeType end = min(start + batchSize, fullCount);
                        auto fullVectors = p_reader->GetVectorSet(start, end);
                        if (m_opt->m_distCalcMethod == DistCalcMethod::Cosine && !p_reader->IsNormalized()) fullVectors->Normalize(m_opt->m_iSSDNumberOfThreads);

                        if (p_opt.m_batches > 1) {
                            if (selections.LoadBatch(static_cast<size_t>(start) * p_opt.m_replicaCount, static_cast<size_t>(end) * p_opt.m_replicaCount) != ErrorCode::Success)
                            {
                                return false;
                            }
                            emptySet.clear();
                            for (auto& pair : headVectorIDS) {
                                if (pair.first >= start && pair.first < end) emptySet.insert(pair.first - start);
                            }
                        }
                        else {
                            for (auto& pair : headVectorIDS) {
                                emptySet.insert(pair.first);
                            }
                        }

                        int sampleNum = 0;
                        for (int j = start; j < end && sampleNum < sampleSize; j++)
                        {
                            if (headVectorIDS.count(j) == 0) samples[sampleNum++] = j - start;
                        }

                        float acc = 0;
                        for (int j = 0; j < sampleNum; j++)
                        {
                            COMMON::Utils::atomic_float_add(&acc, COMMON::TruthSet::CalculateRecall(p_headIndex.get(), fullVectors->GetVector(samples[j]), candidateNum));
                        }
                        acc = acc / sampleNum;
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Batch %d vector(%d,%d) loaded with %d vectors (%zu) HeadIndex acc @%d:%f.\n", i, start, end, fullVectors->Count(), selections.m_selections.size(), candidateNum, acc);

                        p_headIndex->ApproximateRNG(fullVectors, emptySet, candidateNum, selections.m_selections.data(), m_opt->m_replicaCount, numThreads, m_opt->m_gpuSSDNumTrees, m_opt->m_gpuSSDLeafSize, m_opt->m_rngFactor, m_opt->m_numGPUs);
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Batch %d finished!\n", i);

                        for (SizeType j = start; j < end; j++) {
                            replicaCount[j] = 0;
                            size_t vecOffset = j * (size_t)m_opt->m_replicaCount;
                            if (headVectorIDS.count(j) == 0) {
                                for (int resNum = 0; resNum < m_opt->m_replicaCount && selections[vecOffset + resNum].node != INT_MAX; resNum++) {
                                    ++postingListSize[selections[vecOffset + resNum].node];
                                    selections[vecOffset + resNum].tonode = j;
                                    ++replicaCount[j];
                                }
                            } else if (!p_opt.m_excludehead) {
                                    selections[vecOffset].node = headVectorIDS[j];
                                    selections[vecOffset].tonode = j;
                                    ++postingListSize[selections[vecOffset].node];
                                    ++replicaCount[j];
                            }
                            if (m_opt->m_buildPrimaryHeadCSR && replicaCount[j] > 0) {
                                Edge& primary = selections[vecOffset];
                                primary.distance = std::copysign(std::fabs(primary.distance), -1.0f);
                            }
                        }

                        if (p_opt.m_batches > 1)
                        {
                            if (selections.SaveBatch() != ErrorCode::Success)
                            {
                                return false;
                            }
                        }
                    }
                }
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Searching replicas ended. Search Time: %.2lf mins\n", ((double)std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()) / 60.0);

            if (!useNodeAwareBuild && p_opt.m_batches > 1)
            {
                if (selections.LoadBatch(0, static_cast<size_t>(fullCount) * p_opt.m_replicaCount) != ErrorCode::Success)
                {
                    return false;
                }
            }

            // Sort results either in CPU or GPU
            VectorIndex::SortSelections(&selections.m_selections);

            if (m_opt->m_buildPrimaryHeadCSR &&
                !WritePrimaryHeadCSR(selections, headVectorIDS, fullCount, p_headIndex->GetNumSamples())) {
                return false;
            }

            auto t3 = std::chrono::high_resolution_clock::now();
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Time to sort selections:%.2lf sec.\n", ((double)std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()) / 1000);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Posting size limit: %d\n", m_postingSizeLimit);
            {
                int maxReplicaSlots = m_opt->m_replicaCount;
                if (useNodeAwareBuild)
                {
                    for (const auto& memberships : vectorMemberships)
                    {
                        maxReplicaSlots = max(maxReplicaSlots, static_cast<int>(memberships.size()) * m_opt->m_replicaCount);
                    }
                }
                std::vector<int> replicaCountDist(maxReplicaSlots + 1, 0);
                for (int i = 0; i < replicaCount.size(); ++i)
                {
                    if (headVectorIDS.count(i) > 0) continue;
                    int rc = replicaCount[i].load();
                    rc = min<int>(rc, static_cast<int>(replicaCountDist.size()) - 1);
                    ++replicaCountDist[rc];
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Before Posting Cut:\n");
                for (int i = 0; i < replicaCountDist.size(); ++i)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Replica Count Dist: %d, %d\n", i, replicaCountDist[i]);
                }
            }

            Helper::Concurrent::ConcurrentSet<SizeType> zeroReplicaSet;
            std::atomic_int64_t originalSize(0), relaxSize(0);
            {
                std::vector<std::thread> mythreads;
                mythreads.reserve(m_opt->m_iSSDNumberOfThreads);
                std::atomic_size_t sent(0);
                int relaxLimit = m_postingSizeLimit + m_bufferSizeLimit;
                for (int tid = 0; tid < m_opt->m_iSSDNumberOfThreads; tid++)
                {
                    mythreads.emplace_back([&, tid]() {
                        size_t i = 0;
                        while (true)
                        {
                            i = sent.fetch_add(1);
                            if (i < postingListSize.size())
                            {
                                if (postingListSize[i] <= m_postingSizeLimit)
                                    originalSize += postingListSize[i];
                                else
                                    originalSize += m_postingSizeLimit;

                                if (postingListSize[i] <= relaxLimit)
                                {
                                    relaxSize += postingListSize[i];
                                    continue;
                                }
                                relaxSize += relaxLimit;

                                std::size_t selectIdx =
                                    std::lower_bound(selections.m_selections.begin(), selections.m_selections.end(), i,
                                                     Selection::g_edgeComparer) -
                                    selections.m_selections.begin();

                                for (size_t dropID = relaxLimit;
                                     dropID < postingListSize[i]; ++dropID)
                                {
                                    int tonode = selections.m_selections[selectIdx + dropID].tonode;
                                    --replicaCount[tonode];
                                    if (replicaCount[tonode] == 0)
                                    {
                                        zeroReplicaSet.insert(tonode);
                                    }
                                }
                                postingListSize[i] = relaxLimit;
                            }
                            else
                            {
                                return;
                            }
                        }
                    });
                }
                for (auto &t : mythreads)
                {
                    t.join();
                }
                mythreads.clear();
            }
            {
                int maxReplicaSlots = m_opt->m_replicaCount;
                if (useNodeAwareBuild)
                {
                    for (const auto& memberships : vectorMemberships)
                    {
                        maxReplicaSlots = max(maxReplicaSlots, static_cast<int>(memberships.size()) * m_opt->m_replicaCount);
                    }
                }
                std::vector<int> replicaCountDist(maxReplicaSlots + 1, 0);
                for (int i = 0; i < replicaCount.size(); ++i)
                {
                    int rc = replicaCount[i].load();
                    rc = min<int>(rc, static_cast<int>(replicaCountDist.size()) - 1);
                    ++replicaCountDist[rc];
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "After Posting Cut:\n");
                for (int i = 0; i < replicaCountDist.size(); ++i)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Replica Count Dist: %d, %d\n", i, replicaCountDist[i]);
                }
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Posting cut original:%lld relax:%lld\n", originalSize.load(),
                         relaxSize.load());

    //         if (m_opt->m_outputEmptyReplicaID)
    //         {
    //             std::vector<int> replicaCountDist(m_opt->m_replicaCount + 1, 0);
    //             auto ptr = SPTAG::f_createIO();
    //             if (ptr == nullptr || !ptr->Initialize("EmptyReplicaID.bin", std::ios::binary | std::ios::out)) {
    //                 SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to create EmptyReplicaID.bin!\n");
    //                 return false;
    //             }
    //             for (int i = 0; i < replicaCount.size(); ++i)
    //             {
    //                 if (headVectorIDS.count(i) > 0) continue;

    //                 ++replicaCountDist[replicaCount[i]];

    //                 if (replicaCount[i] < 2)
    //                 {
    //                     long long vid = i;
    //                     if (ptr->WriteBinary(sizeof(vid), reinterpret_cast<char*>(&vid)) != sizeof(vid)) {
    //                         SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failt to write EmptyReplicaID.bin!");
    //                         return false;
    //                     }
    //                 }
    //             }

    //             SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "After Posting Cut:\n");
    //             for (int i = 0; i < replicaCountDist.size(); ++i)
    //             {
    //                 SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Replica Count Dist: %d, %d\n", i, replicaCountDist[i]);
    //             }
    //         }


            auto t4 = std::chrono::high_resolution_clock::now();
            SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Info, "Time to perform posting cut:%.2lf sec.\n", ((double)std::chrono::duration_cast<std::chrono::seconds>(t4 - t3).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count()) / 1000);

            auto fullVectors = p_reader->GetVectorSet();
            if (m_opt->m_distCalcMethod == DistCalcMethod::Cosine && !p_reader->IsNormalized() && !p_headIndex->m_pQuantizer) fullVectors->Normalize(m_opt->m_iSSDNumberOfThreads);

            // ====================================================================
            // Phase 4 (Unfilter-tail): for each base vector, append K_replica
            // tag-agnostic nearest-head replicas. These are scanned ONLY by
            // unfilter queries; filtered queries stop at pure_count.
            //
            // Layout trick: tail edges are given a distance of FLT_MAX so they
            // sort AFTER all pure edges within the same head posting.
            //   pure region:  edges sorted by distance ascending  (idx < pure_count)
            //   tail region:  edges with dist=FLT_MAX             (idx >= pure_count)
            // ====================================================================
            std::vector<int> pure_count_per_head;
            {
                // ini is the single source of truth: K_replica comes only from the
                // native SSD param TailReplicaCount (no env override).
                int k_replica = m_opt->m_tailReplicaCount;
                // ε-closure dynamic replica: when SPTAG_TAIL_CLOSURE_FACTOR>0, k_replica is
                // re-interpreted as Kmax (an upper cap), and each base vector v gets a
                // VARIABLE number of tail replicas: the nearest head (always), plus every
                // further head h within an additive margin tau of the nearest head:
                //   replicate h  iff  dist(v,h) - dist(v,nearest) < tau .
                // tau is NOT hand-picked; it is data-driven: sample a subset of base
                // vectors, compute the mean of their top-k nearest-head distances (Dbar,
                // the typical local head spacing), and set  tau = factor * Dbar .
                // The cutoff is anchored to the NEAREST head's distance (not the previous
                // replica) so dense interiors stop at 1 replica while dense boundaries /
                // sparse points near several equidistant heads get more. Min 1, max Kmax.
                const char* env_factor = std::getenv("SPTAG_TAIL_CLOSURE_FACTOR");
                double closure_factor = env_factor ? std::atof(env_factor) : 0.0;
                const bool closureMode = (closure_factor > 0.0);
                double closure_tau = 0.0;   // = closure_factor * Dbar, computed by sampling below
                double closure_Dbar = 0.0;
                if (k_replica <= 0 && m_hasHeadRole) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                                 "Phase 4 (unfilter-tail) DISABLED (K_replica=0) but U_extra heads exist: "
                                 "in tail-only mode their postings will be EMPTY. Set TailReplicaCount"
                                 " > 0 to populate U_extra tails.\n");
                }
                if (k_replica > 0) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                 "Phase 4 (unfilter-tail): %s=%d (source=%s)%s, scanning %d base vectors against %d heads\n",
                                 (closureMode ? "Kmax" : "K_replica"), k_replica,
                                 "TailReplicaCount param",
                                 (closureMode ? (" closure factor=" + std::to_string(closure_factor)).c_str() : ""),
                                 fullCount, p_headIndex->GetNumSamples());

                    // Snapshot pure counts (post-cut, pre-tail)
                    pure_count_per_head.resize(postingListSize.size());
                    for (size_t h = 0; h < postingListSize.size(); ++h) {
                        pure_count_per_head[h] = postingListSize[h].load();
                    }

                    // For each vector v, find K_replica nearest heads (tag-agnostic),
                    // then append only tail candidates that fit the page budget.
                    std::atomic_size_t vec_cursor(0);
                    std::atomic_size_t tail_added(0), tail_skipped_dup(0), tail_skipped_cap(0);
                    int n_threads = m_opt->m_iSSDNumberOfThreads;
                    auto recordsForPages = [&](int pages) -> int {
                        return std::max(0, (pages * PageSize) / std::max(1, m_vectorInfoSize));
                    };
                    auto pagesForRecords = [&](int records) -> int {
                        if (records <= 0) return 0;
                        return (records * m_vectorInfoSize + PageSize - 1) / PageSize;
                    };
                    auto sparseTailLastPageKeep = [&](int pure, int keep) -> int {
                        if (keep <= pure) return pure;
                        const int totalBytes = keep * m_vectorInfoSize;
                        const int totalPages = (totalBytes + PageSize - 1) / PageSize;
                        if (totalPages <= 1) return keep;
                        const int lastPageStart = (totalPages - 1) * PageSize;
                        const int pureBytes = pure * m_vectorInfoSize;
                        // Only drop the last page when it contains tail exclusively.
                        if (pureBytes > lastPageStart) return keep;
                        const int lastPageBytes = totalBytes - lastPageStart;
                        if (lastPageBytes >= (PageSize + 9) / 10) return keep;
                        return std::max(pure, lastPageStart / m_vectorInfoSize);
                    };
                    const int recordsPerPage = recordsForPages(1);
                    // Tail capacity is expressed relative to the already-built pure
                    // prefix. The buffer setting means extra physical tail pages, not
                    // an absolute posting-size target derived from the old pure
                    // PostingPageLimit/BufferLength budget. Tail may still fill slack
                    // in the pure prefix's final page at no extra page cost.
                    const int extraTailPages = std::max(0, m_opt->m_unfilterTailBufferLength);
                    auto tailHardCapForHead = [&](SizeType h) -> int {
                        if (h < 0 || static_cast<size_t>(h) >= pure_count_per_head.size()) return 0;
                        const int pure = pure_count_per_head[static_cast<size_t>(h)];
                        const int purePages = pagesForRecords(pure);
                        const int capPages = purePages + extraTailPages;
                        return std::max(pure, recordsForPages(capPages));
                    };
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                 "Phase 4 page-budget tail: recordBytes=%d recordsPerPage=%d "
                                 "extraTailPages=%d cap=purePages+extra sparseLastPageThreshold=10%%\n",
                                 m_vectorInfoSize, recordsPerPage, extraTailPages);
                    SizeType numHeadsLocal = p_headIndex->GetNumSamples();

                    // The pre-tail selection array still contains RNG candidates that
                    // were excluded by the posting cut. Compact it to the persisted pure
                    // prefixes before adding tails so discarded pure candidates never
                    // consume tail-page capacity.
                    const size_t preTailSelectionCount = selections.m_selections.size();
                    size_t pureWrite = 0;
                    size_t pureRead = 0;
                    while (pureRead < selections.m_selections.size()) {
                        const int h = selections.m_selections[pureRead].node;
                        size_t pureEnd = pureRead + 1;
                        while (pureEnd < selections.m_selections.size() &&
                               selections.m_selections[pureEnd].node == h) {
                            ++pureEnd;
                        }
                        if (h >= 0 && static_cast<size_t>(h) < pure_count_per_head.size()) {
                            const int pure = std::max(0, std::min(
                                pure_count_per_head[static_cast<size_t>(h)],
                                static_cast<int>(pureEnd - pureRead)));
                            for (int i = 0; i < pure; ++i) {
                                if (pureWrite != pureRead + static_cast<size_t>(i)) {
                                    selections.m_selections[pureWrite] =
                                        selections.m_selections[pureRead + static_cast<size_t>(i)];
                                }
                                ++pureWrite;
                            }
                            postingListSize[static_cast<size_t>(h)] = pure;
                        }
                        pureRead = pureEnd;
                    }
                    selections.m_selections.resize(pureWrite);
                    const size_t pureSelectionCount = pureWrite;
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                 "Phase 4 compacted pure prefixes: kept=%zu dropped=%zu\n",
                                 pureSelectionCount, preTailSelectionCount - pureSelectionCount);

                    std::vector<std::atomic_int> tailCountPerHead(static_cast<size_t>(numHeadsLocal));
                    for (auto& count : tailCountPerHead) count.store(0, std::memory_order_relaxed);
                    std::vector<Edge> tailCandidates;
                    std::mutex tailMutex;

                    // Data-driven tau: sample base vectors, average their top-k nearest-head
                    // distances -> Dbar (typical local head spacing), then tau = factor * Dbar.
                    if (closureMode) {
                        const size_t sampleTarget = std::min<size_t>(8192, (size_t)fullCount);
                        const size_t stride = std::max<size_t>(1, (size_t)fullCount / std::max<size_t>(1, sampleTarget));
                        COMMON::QueryResultSet<ValueType> sres(nullptr, k_replica);
                        double sum = 0.0; size_t cnt = 0;
                        for (size_t v = 0; v < (size_t)fullCount; v += stride) {
                            if (headVectorIDS.count((SizeType)v)) continue;
                            const ValueType* vd = (const ValueType*)fullVectors->GetVector((SizeType)v);
                            if (vd == nullptr) continue;
                            sres.SetTarget(vd, p_headIndex->m_pQuantizer);
                            sres.Reset();
                            p_headIndex->SearchIndex(sres);
                            BasicResult* sr = sres.GetResults();
                            double vsum = 0.0; int got = 0;
                            for (int r = 0; r < k_replica; ++r) {
                                if (sr[r].VID < 0 || sr[r].VID >= numHeadsLocal) continue;
                                vsum += sr[r].Dist; ++got;
                            }
                            if (got > 0) { sum += vsum / got; ++cnt; }
                        }
                        closure_Dbar = cnt ? sum / (double)cnt : 0.0;
                        closure_tau = closure_factor * closure_Dbar;
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                     "Phase 4 closure: sampled %zu vectors, Dbar(top-%d mean head dist)=%.6f -> tau=factor(%.3f)*Dbar=%.6f\n",
                                     cnt, k_replica, closure_Dbar, closure_factor, closure_tau);
                        if (closure_tau <= 0.0) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                                         "Phase 4 closure: tau<=0 after sampling; closure will reduce to K=1 (nearest only).\n");
                        }
                    }

                    // Closure diagnostics (merged from workers under tailMutex):
                    //  - margin_hist: histogram of (d2 - d1) over non-head base vectors,
                    //    used to pick tau (look for the knee).
                    //  - replica_dist: distribution of per-vector replica counts after closure.
                    const int kHistBins = 101;        // bins [0,0.5) width 0.005 + overflow bin
                    const double kHistW = 0.005;
                    std::vector<uint64_t> margin_hist(kHistBins, 0);
                    std::vector<uint64_t> replica_dist((size_t)k_replica + 1, 0);

                    // Build a per-head set of vector-IDs already present in pure region,
                    // for fast dup check. Memory: ~ sum(pure_count) * 8 bytes ≈ 400 MB worst case.
                    // To keep it cheap, we instead do binary-search lookup on the sorted
                    // `selections.m_selections` (sorted by node first).
                    auto worker = [&]() {
                        COMMON::QueryResultSet<ValueType> nearbyHeads(nullptr, k_replica);
                        std::vector<Edge> local_appends;
                        local_appends.reserve(4096);
                        std::vector<uint64_t> local_margin_hist(kHistBins, 0);
                        std::vector<uint64_t> local_replica_dist((size_t)k_replica + 1, 0);
                        const std::vector<Edge>& pureSelections = selections.m_selections;
                        size_t v;
                        while (true) {
                            v = vec_cursor.fetch_add(1);
                            if (v >= (size_t)fullCount) break;
                            // Skip if v is itself a head (it's already its own posting member)
                            // — those entries have no "base posting" and don't need a tail.
                            if (headVectorIDS.count((SizeType)v)) continue;

                            const ValueType* vec_data = (const ValueType*)fullVectors->GetVector((SizeType)v);
                            if (vec_data == nullptr) continue;
                            nearbyHeads.SetTarget(vec_data, p_headIndex->m_pQuantizer);
                            nearbyHeads.Reset();
                            p_headIndex->SearchIndex(nearbyHeads);
                            BasicResult* res = nearbyHeads.GetResults();
                            if (res[0].VID < 0 || res[0].VID >= numHeadsLocal) continue;
                            float d1c = res[0].Dist;
                            // margin histogram: (d2 - d1)
                            if (k_replica >= 2 && res[1].VID >= 0 && res[1].VID < numHeadsLocal) {
                                float mg = res[1].Dist - d1c;
                                int mb = (mg <= 0.f) ? 0 : (int)(mg / (float)kHistW);
                                if (mb >= kHistBins) mb = kHistBins - 1;
                                ++local_margin_hist[mb];
                            }
                            int v_replicas = 0;
                            for (int r = 0; r < k_replica; ++r) {
                                SizeType h = res[r].VID;
                                if (h < 0 || h >= numHeadsLocal) continue;
                                // ε-closure cutoff: always keep nearest (r==0); for further
                                // heads, replicate only while within tau of the NEAREST head's
                                // distance (anchored to d1, not the previous replica).
                                if (closureMode && r > 0 && (res[r].Dist - d1c) >= (float)closure_tau) break;
                                // Dup check: scan pure region of head h's posting
                                size_t lo = std::lower_bound(pureSelections.begin(), pureSelections.end(), (int)h,
                                                              Selection::g_edgeComparer) - pureSelections.begin();
                                size_t pure_end = std::min(
                                    pureSelectionCount,
                                    lo + static_cast<size_t>(pure_count_per_head[h]));
                                bool dup = false;
                                for (size_t k = lo; k < pure_end; ++k) {
                                    if (pureSelections[k].node != h) break;
                                    if (pureSelections[k].tonode == (SizeType)v) { dup = true; break; }
                                }
                                if (dup) { ++tail_skipped_dup; continue; }

                                const int tailCapacity = std::max(
                                    0,
                                    tailHardCapForHead(h) - pure_count_per_head[static_cast<size_t>(h)]);
                                if (tailCapacity == 0) {
                                    ++tail_skipped_cap;
                                    continue;
                                }
                                const int prior = tailCountPerHead[static_cast<size_t>(h)].fetch_add(
                                    1, std::memory_order_relaxed);
                                if (prior >= tailCapacity) {
                                    ++tail_skipped_cap;
                                    continue;
                                }

                                // Keep true distance until the candidates have been
                                // sorted per head. The final merge below restores the
                                // FLT_MAX tail marker so the pure/tail boundary stays
                                // physically contiguous in the posting.
                                Edge e; e.node = h; e.tonode = (SizeType)v; e.distance = res[r].Dist;
                                local_appends.push_back(e);
                                ++v_replicas;
                                ++tail_added;
                            }
                            if (v_replicas >= (int)local_replica_dist.size()) v_replicas = (int)local_replica_dist.size() - 1;
                            ++local_replica_dist[v_replicas];
                            if (local_appends.size() >= 4096) {
                                std::lock_guard<std::mutex> lk(tailMutex);
                                tailCandidates.insert(tailCandidates.end(), local_appends.begin(), local_appends.end());
                                local_appends.clear();
                            }
                        }
                        if (!local_appends.empty()) {
                            std::lock_guard<std::mutex> lk(tailMutex);
                            tailCandidates.insert(tailCandidates.end(), local_appends.begin(), local_appends.end());
                        }
                        {
                            std::lock_guard<std::mutex> lk(tailMutex);
                            for (int b = 0; b < kHistBins; ++b) margin_hist[b] += local_margin_hist[b];
                            for (size_t b = 0; b < local_replica_dist.size(); ++b) replica_dist[b] += local_replica_dist[b];
                        }
                    };

                    std::vector<std::thread> phase4_threads;
                    for (int t = 0; t < n_threads; ++t) phase4_threads.emplace_back(worker);
                    for (auto& th : phase4_threads) th.join();

                    // RewriteTailOnly orders each head's retained tail by true
                    // head distance before writing it after the pure prefix. Do the
                    // same in the direct build so a newly built index needs no
                    // post-build tail rewrite for ordering.
                    std::sort(tailCandidates.begin(), tailCandidates.end(), EdgeCompare());
                    const size_t tailCandidateCount = tailCandidates.size();
                    selections.m_selections.resize(pureSelectionCount + tailCandidateCount);
                    size_t purePos = pureSelectionCount;
                    size_t tailPos = tailCandidateCount;
                    size_t outputPos = pureSelectionCount + tailCandidateCount;
                    while (purePos > 0 || tailPos > 0) {
                        const int pureHead = purePos > 0 ? selections.m_selections[purePos - 1].node : -1;
                        const int tailHead = tailPos > 0 ? tailCandidates[tailPos - 1].node : -1;
                        const int head = std::max(pureHead, tailHead);

                        size_t pureBegin = purePos;
                        while (pureBegin > 0 && selections.m_selections[pureBegin - 1].node == head) {
                            --pureBegin;
                        }
                        size_t tailBegin = tailPos;
                        while (tailBegin > 0 && tailCandidates[tailBegin - 1].node == head) {
                            --tailBegin;
                        }
                        const size_t pureCount = purePos - pureBegin;
                        const size_t tailCount = tailPos - tailBegin;

                        while (tailPos > tailBegin) {
                            Edge tail = tailCandidates[--tailPos];
                            tail.distance = std::numeric_limits<float>::max();
                            selections.m_selections[--outputPos] = tail;
                        }
                        while (purePos > pureBegin) {
                            selections.m_selections[--outputPos] = selections.m_selections[--purePos];
                        }
                        if (head >= 0 && static_cast<size_t>(head) < postingListSize.size()) {
                            postingListSize[static_cast<size_t>(head)] =
                                static_cast<int>(pureCount + tailCount);
                        }
                    }
                    std::vector<Edge>().swap(tailCandidates);

                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                 "Phase 4 done: tail_added=%zu tail_skipped_dup=%zu tail_skipped_cap=%zu "
                                 "pure=%zu final=%zu\n",
                                 tail_added.load(), tail_skipped_dup.load(), tail_skipped_cap.load(),
                                 pureSelectionCount, selections.m_selections.size());

                    // Replica-count distribution after closure (Σ over base vectors).
                    {
                        uint64_t tot = 0, weighted = 0;
                        for (size_t r = 0; r < replica_dist.size(); ++r) {
                            if (replica_dist[r]) {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                             "Phase 4 replica-count: %zu -> %llu vectors\n",
                                             r, (unsigned long long)replica_dist[r]);
                            }
                            tot += replica_dist[r];
                            weighted += (uint64_t)r * replica_dist[r];
                        }
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                     "Phase 4 replica-count: total=%llu avg=%.3f (mode=%s factor=%.3f Dbar=%.6f tau=%.6f Kmax=%d)\n",
                                     (unsigned long long)tot,
                                     tot ? (double)weighted / (double)tot : 0.0,
                                     (closureMode ? "closure" : "fixed-K"),
                                     closure_factor, closure_Dbar, closure_tau, k_replica);
                    }
                    // Margin (d2-d1) histogram — pick tau near the knee of this curve.
                    {
                        uint64_t cum = 0, tot = 0;
                        for (int b = 0; b < kHistBins; ++b) tot += margin_hist[b];
                        for (int b = 0; b < kHistBins; ++b) {
                            if (!margin_hist[b]) continue;
                            cum += margin_hist[b];
                            double lo = b * kHistW;
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                         "Phase 4 margin-hist: [%.3f,%.3f)%s -> %llu (cum %.1f%%)\n",
                                         lo, lo + kHistW, (b == kHistBins - 1 ? "+" : " "),
                                         (unsigned long long)margin_hist[b],
                                         tot ? 100.0 * (double)cum / (double)tot : 0.0);
                        }
                    }

                    // Page-budget final trim: keep tail within the storage page budget.
                    // Tail may add pages. If the final page contains only tail and is
                    // <10% occupied, drop tail records from that final page. Pure records
                    // are never dropped; if pure already uses many pages, tail can still
                    // fill slack in those already-paid pages.
                    {
                        size_t write = 0;
                        size_t read = 0;
                        size_t tailTrimmedByTwoPage = 0;
                        size_t tailTrimmedBySparseSecondPage = 0;
                        size_t headsTrimmedByTwoPage = 0;
                        size_t headsTrimmedBySparseSecondPage = 0;
                        size_t headsWithTail = 0;
                        while (read < selections.m_selections.size()) {
                            const int h = selections.m_selections[read].node;
                            size_t end = read + 1;
                            while (end < selections.m_selections.size() &&
                                   selections.m_selections[end].node == h) {
                                ++end;
                            }
                            if (h < 0 || static_cast<size_t>(h) >= postingListSize.size()) {
                                read = end;
                                continue;
                            }
                            const int total = static_cast<int>(end - read);
                            const int pure = (h >= 0 && static_cast<size_t>(h) < pure_count_per_head.size())
                                ? pure_count_per_head[static_cast<size_t>(h)]
                                : total;
                            int keep = total;
                            bool trimmedTwoPage = false;
                            bool trimmedSparseSecond = false;
                            if (total > pure) ++headsWithTail;
                            const int hardCap = tailHardCapForHead(static_cast<SizeType>(h));
                            if (keep > hardCap) {
                                keep = hardCap;
                                trimmedTwoPage = true;
                            }
                            int sparseKeep = sparseTailLastPageKeep(pure, keep);
                            if (sparseKeep < keep) {
                                keep = sparseKeep;
                                trimmedSparseSecond = true;
                            }
                            keep = std::max(0, std::min(keep, total));
                            const size_t drop = static_cast<size_t>(total - keep);
                            if (drop > 0) {
                                if (trimmedTwoPage) {
                                    ++headsTrimmedByTwoPage;
                                    tailTrimmedByTwoPage += drop;
                                } else if (trimmedSparseSecond) {
                                    ++headsTrimmedBySparseSecondPage;
                                    tailTrimmedBySparseSecondPage += drop;
                                }
                            }
                            for (int i = 0; i < keep; ++i) {
                                if (write != read + static_cast<size_t>(i)) {
                                    selections.m_selections[write] =
                                        selections.m_selections[read + static_cast<size_t>(i)];
                                }
                                ++write;
                            }
                            postingListSize[static_cast<size_t>(h)] = keep;
                            read = end;
                        }
                        selections.m_selections.resize(write);
                        // Phase 4 changes the backing vector cardinality. Keep the
                        // Selection bounds in sync so WriteDownAllPostingToDB can
                        // address the newly appended tail records without emitting
                        // false out-of-range diagnostics.
                        selections.m_end = selections.m_start + selections.m_selections.size();
                        selections.m_totalsize = selections.m_end;
                        size_t pageBudgetViolations = 0;
                        for (size_t h = 0; h < postingListSize.size(); ++h) {
                            const int pure = pure_count_per_head[h];
                            const int total = postingListSize[h].load();
                            const int maxPages = pagesForRecords(pure) + extraTailPages;
                            if (total < pure || pagesForRecords(total) > maxPages) {
                                if (pageBudgetViolations < 5) {
                                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                                 "Phase 4 page-budget violation: head=%d pure=%d total=%d "
                                                 "pages=%d maxPages=%d\n",
                                                 static_cast<int>(h), pure, total,
                                                 pagesForRecords(total), maxPages);
                                }
                                ++pageBudgetViolations;
                            }
                        }
                        if (pageBudgetViolations != 0) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                         "Phase 4 page-budget violation count=%zu\n",
                                         pageBudgetViolations);
                            return false;
                        }
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                     "Phase 4 page-budget verified: heads=%zu cap=purePages+%d\n",
                                     postingListSize.size(), extraTailPages);
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                     "Phase 4 page-budget trim: headsWithTail=%zu hardCapTrimHeads=%zu hardCapTrimRecords=%zu "
                                     "sparseSecondTrimHeads=%zu sparseSecondTrimRecords=%zu finalSelections=%zu\n",
                                     headsWithTail,
                                     headsTrimmedByTwoPage, tailTrimmedByTwoPage,
                                     headsTrimmedBySparseSecondPage, tailTrimmedBySparseSecondPage,
                                     selections.m_selections.size());
                    }

                    // Initialize pure_counts sidecar (will be persisted at save).
                    m_postingPureCounts.Initialize((SizeType)postingListSize.size(),
                                                   p_headIndex->m_iDataBlockSize, p_headIndex->m_iDataCapacity);
                    for (size_t h = 0; h < postingListSize.size(); ++h) {
                        m_postingPureCounts.UpdateSize((SizeType)h, pure_count_per_head[h]);
                    }
                    m_hasPostingPureCounts = true;
                }
            }
            // ====================================================================

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPFresh: initialize versionMap\n");
            m_versionMap->Initialize(fullCount, p_headIndex->m_iDataBlockSize, p_headIndex->m_iDataCapacity);

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPFresh: Writing values to DB\n");

            m_postingSizes.Initialize((SizeType)(postingListSize.size()), p_headIndex->m_iDataBlockSize,
                                      p_headIndex->m_iDataCapacity);
            for (int i = 0; i < postingListSize.size(); i++)
            {
                m_postingSizes.UpdateSize(i, postingListSize[i].load());
            }

            m_checkSums.Initialize((SizeType)(postingListSize.size()), 1, p_headIndex->m_iDataBlockSize,
                                   p_headIndex->m_iDataCapacity);

            if (ErrorCode::Success != WriteDownAllPostingToDB(selections, fullVectors)) return false;

            if (m_opt->m_update && !m_opt->m_allowZeroReplica && zeroReplicaSet.size() > 0)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPFresh: initialize thread pools, append: %d, reassign %d\n", m_opt->m_appendThreadNum, m_opt->m_reassignThreadNum);
                m_splitThreadPool = std::make_shared<SPDKThreadPool>();
                m_splitThreadPool->initSPDK(m_opt->m_appendThreadNum, this);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPFresh: finish initialization, zeroReplicaCount:%d\n", (int)(zeroReplicaSet.size()));

                ExtraWorkSpace workSpace;
                InitWorkSpace(&workSpace);
                for (SizeType it : zeroReplicaSet)
                {
                    std::shared_ptr<VectorSet> vectorSet(new BasicVectorSet(ByteArray((std::uint8_t*)fullVectors->GetVector(it), sizeof(ValueType) * m_opt->m_dim, false),
                        GetEnumValueType<ValueType>(), m_opt->m_dim, 1));
                    if (AddIndex(&workSpace, vectorSet, p_headIndex, it) != ErrorCode::Success) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to add index for zero replica ID: %d\n", it);
                        return false;
                    }
                }
                while (!AllFinished())
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                }

                if (p_headIndex->SaveIndex(m_opt->m_indexDirectory + FolderSep + m_opt->m_headIndexFolder) != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to save head index!\n");
                    return false;
                }

                if (m_vectorTranslateMap->Save(m_opt->m_indexDirectory + FolderSep + m_opt->m_headIDFile) != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to save vector ID translate map!\n");
                    return false;
                }
            }

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPFresh: Writing SSD Info and checkSum\n");
            // Build-time slim: postings were written as [meta|RaBitQ-code]. Reflect the slim
            // stride in m_vectorInfoSize (so ssdinfo/metadata and any reload math are slim)
            // and drop the inpost_rbq.bin marker so the post-build transform is a no-op and
            // the index loads in in-posting-RBQ mode.
            if (m_buildSlimRbq) {
                m_vectorInfoSize = m_buildSlimStride;
                std::string marker = m_opt->m_indexDirectory + FolderSep + "inpost_rbq.bin";
                std::ofstream mf(marker, std::ios::binary);
                int hdr[2] = { 1, m_inpostRbqBinBytes };
                mf.write((const char*)hdr, sizeof(hdr));
                mf.close();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                    "[InpostRBQ-native] build-time slim DONE: slimStride=%d, marker written\n",
                    m_buildSlimStride);
            }
            // In-posting OPQ build-time slim: drop the inpost_opq.bin marker (same format
            // TransformInPostingsOpq writes) so the post-build transform is a no-op and the
            // index loads in in-posting-OPQ-DB mode.
            if (m_buildSlimOpq) {
                m_vectorInfoSize = m_buildSlimStride;
                std::string marker = m_opt->m_indexDirectory + FolderSep + "inpost_opq.bin";
                std::ofstream mf(marker, std::ios::binary);
                int hdr[2] = { m_buildOpqM, m_buildSlimStride };
                mf.write((const char*)hdr, sizeof(hdr));
                mf.close();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                    "[InpostOPQ-native] build-time slim DONE: M=%d slimStride=%d, marker written\n",
                    m_buildOpqM, m_buildSlimStride);
            }
            if (m_buildSlimPipePQ) {
                m_vectorInfoSize = m_buildSlimStride;
                std::string marker = m_opt->m_indexDirectory + FolderSep + "inpost_pipepq.bin";
                std::ofstream mf(marker, std::ios::binary);
                int hdr[2] = { m_buildPipePQM, m_buildSlimStride };
                mf.write((const char*)hdr, sizeof(hdr));
                mf.close();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                    "[InpostPipePQ-native] build-time slim DONE: M=%d slimStride=%d, marker written\n",
                    m_buildPipePQM, m_buildSlimStride);
            }
            m_postingSizes.Save(m_opt->m_indexDirectory + FolderSep + m_opt->m_ssdInfoFile);
            m_checkSums.Save(m_opt->m_indexDirectory + FolderSep + m_opt->m_checksumFile);
            SavePostingPureCounts();

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPFresh: save versionMap\n");
            m_versionMap->Save(m_opt->m_indexDirectory + FolderSep + m_opt->m_deleteIDFile);

            auto t5 = std::chrono::high_resolution_clock::now();
            double elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(t5 - t1).count();
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Total used time: %.2lf minutes (about %.2lf hours).\n", elapsedSeconds / 60.0, elapsedSeconds / 3600.0);
            return true;
        }

        ErrorCode WriteDownAllPostingToDB(Selection& p_postingSelections, std::shared_ptr<VectorSet> p_fullVectors) {

            std::vector<std::thread> threads;
            std::atomic_size_t vectorsSent(0);
            ErrorCode ret = ErrorCode::Success;
            auto func = [&]()
            {
                ExtraWorkSpace workSpace;
                InitWorkSpace(&workSpace);
                size_t index = 0;
                while (true)
                {
                    index = vectorsSent.fetch_add(1);
                    if (index < m_postingSizes.GetPostingNum()) {
                        // Build-time slim: write [meta|RaBitQ-code] records (stride
                        // m_buildSlimStride) instead of [meta|full-vector] (m_vectorInfoSize),
                        // so the full-vector posting store is never materialized. Membership
                        // count (m_postingSizes.GetSize) is unchanged (full-stride-based),
                        // so the on-disk result matches the post-build slim transform.
                        const int stride = (m_buildSlimRbq || m_buildSlimOpq || m_buildSlimPipePQ) ? m_buildSlimStride : m_vectorInfoSize;
                        std::string postinglist((size_t)stride * m_postingSizes.GetSize(index), '\0');
                        char* ptr = (char*)postinglist.c_str();
			            std::size_t selectIdx = p_postingSelections.lower_bound((int)index);
                        for (int j = 0; j < m_postingSizes.GetSize(index); ++j)
                        {
                            if (p_postingSelections[selectIdx].node != index) {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Selection ID NOT MATCH\n");
                                ret = ErrorCode::Fail;
                                return;
                            }
                            SizeType fullID = p_postingSelections[selectIdx++].tonode;
                            // if (id == 0) SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "ID: %d\n", fullID);
                            uint8_t version = m_versionMap->GetVersion(fullID);
                            // First Vector ID, then version, then Vector (or slim code)
                            if (m_buildSlimRbq) {
                                SerializeSlim(ptr, fullID, version);
                            } else if (m_buildSlimOpq) {
                                SerializeSlimOpq(ptr, fullID, version);
                            } else if (m_buildSlimPipePQ) {
                                SerializeSlimPipePQ(ptr, fullID, version);
                            } else {
                                Serialize(ptr, fullID, version, p_fullVectors->GetVector(fullID));
                            }
                            ptr += stride;
                        }
                        ErrorCode tmp;
                        if ((tmp = db->Put(index, postinglist, MaxTimeout, &(workSpace.m_diskRequests))) !=
                            ErrorCode::Success)
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[WriteDB] Put fail!\n");
                            ret = tmp;
                            return;
                        }
                        *m_checkSums[index] = m_checkSum.CalcChecksum(postinglist.c_str(), (int)(postinglist.size()));
                        if (m_opt->m_consistencyCheck && (tmp = db->Check(index, m_postingSizes.GetSize(index) * stride, nullptr)) !=
                            ErrorCode::Success)
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "WriteDB: Check failed after Put %d\n", index);
                            ret = tmp;
                            return;
                        }
                    }
                    else
                    {
                        return;
                    }
                }
            };

            for (int j = 0; j < m_opt->m_iSSDNumberOfThreads; j++) { threads.emplace_back(func); }
            for (auto& thread : threads) { thread.join(); }
	        return ret;
        }

        ErrorCode AddIndex(ExtraWorkSpace* p_exWorkSpace, std::shared_ptr<VectorSet>& p_vectorSet,
            std::shared_ptr<VectorIndex> p_index, SizeType begin) override {

            for (int v = 0; v < p_vectorSet->Count(); v++) {
                SizeType VID = begin + v;
                std::vector<Edge> selections(static_cast<size_t>(m_opt->m_replicaCount));
                int replicaCount;
                RNGSelection(selections, (ValueType*)(p_vectorSet->GetVector(v)), p_index.get(), VID, replicaCount);

                uint8_t version = m_versionMap->GetVersion(VID);
                std::string appendPosting(m_vectorInfoSize, '\0');
                Serialize((char*)(appendPosting.c_str()), VID, version, p_vectorSet->GetVector(v));
                if (m_opt->m_enableWAL && m_wal) {
                    m_wal->PutAssignment(appendPosting);
                }
                for (int i = 0; i < replicaCount; i++)
                {
                    // AppendAsync(selections[i].node, 1, appendPosting_ptr);
                    ErrorCode ret;
                    if (m_opt->m_asyncAppendQueueSize > 0) {
                        if ((ret = AsyncAppend(p_exWorkSpace, p_index.get(), selections[i].node, 1, appendPosting)) != ErrorCode::Success)
                            return ret;
                    } else {
                        if ((ret = Append(p_exWorkSpace, p_index.get(), selections[i].node, 1, appendPosting)) !=
                            ErrorCode::Success)
                            return ret;
                    }
                }

                // Make the new vector visible to the OPQ tag-pure search path. The vector
                // itself already lands in the canonical vector store via the SlimVectorKV
                // deflate-on-write inside Append above; here we maintain the resident OPQ
                // codes + tag->vids map. Tags come from m_vectorTags when present for this
                // VID (build-time-populated); runtime public inserts that do not supply
                // tags leave the vid unregistered under any tag (still vecstore-resident).
                if (m_opqDynamic) {
                    const uint32_t* tags = nullptr;
                    int numTags = 0;
                    if (m_tagBytesPerVec > 0 && (size_t)VID * m_numTagsPerVec < m_vectorTags.size()) {
                        tags = &m_vectorTags[(size_t)VID * m_numTagsPerVec];
                        numTags = m_numTagsPerVec;
                    }
                    OPQInsertMaintain(VID, (const ValueType*)p_vectorSet->GetVector(v), tags, numTags);
                }
            }
            return ErrorCode::Success;
        }

        ErrorCode AddIndexWithTargets(ExtraWorkSpace* p_exWorkSpace,
                                      std::shared_ptr<VectorSet>& p_vectorSet,
                                      const PostingUpdateTargets& p_targets,
                                      const std::uint32_t* p_tags,
                                      int p_numTagsPerVec,
                                      SizeType p_begin) override
        {
            if (p_vectorSet == nullptr || p_tags == nullptr ||
                p_targets.size() != static_cast<size_t>(p_vectorSet->Count()) ||
                p_numTagsPerVec != m_numTagsPerVec) {
                return ErrorCode::Fail;
            }
            m_taggedMaintenance.store(true, std::memory_order_release);
            if (m_opt->m_enableWAL) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                             "[TaggedUpdate] WAL cannot replay explicit pure/tail targets; disable EnableWAL "
                             "or use a target-aware WAL before inserting.\n");
                return ErrorCode::Undefined;
            }
            if (!m_hasPostingPureCounts) {
                InitializePureCountsFromTotals(m_postingSizes.GetPostingNum());
            }
            if (!AppendDynamicVectors(p_vectorSet, p_begin)) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                             "[TaggedUpdate] failed to persist full-precision update vectors.\n");
                return ErrorCode::Fail;
            }

            auto writePosting = [&](SizeType headID, PostingUpdateKind kind,
                                    const std::string& record) -> ErrorCode {
                if (headID < 0 || headID >= m_postingSizes.GetPostingNum()) {
                    return ErrorCode::Key_OverFlow;
                }

                std::unique_lock<std::shared_timed_mutex> lock(m_rwLocks[headID]);
                std::string posting;
                ErrorCode ret = db->Get(headID, &posting, MaxTimeout, &(p_exWorkSpace->m_diskRequests));
                if (ret != ErrorCode::Success ||
                    !m_checkSum.ValidateChecksum(posting.c_str(), static_cast<int>(posting.size()),
                                                 *m_checkSums[headID])) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "[TaggedUpdate] cannot read a valid posting %d.\n", headID);
                    return ret == ErrorCode::Success ? ErrorCode::Fail : ret;
                }
                if (posting.size() % static_cast<size_t>(m_vectorInfoSize) != 0) {
                    return ErrorCode::Posting_SizeError;
                }

                const int total = static_cast<int>(posting.size() / static_cast<size_t>(m_vectorInfoSize));
                int pure = GetPureCount(headID);
                if (pure < 0 || pure > total) return ErrorCode::Posting_SizeError;

                SizeType newVID = -1;
                memcpy(&newVID, record.data(), sizeof(newVID));
                SizeType existingVID = -1;
                for (int i = 0; i < total; ++i) {
                    memcpy(&existingVID, posting.data() + static_cast<size_t>(i) * m_vectorInfoSize,
                           sizeof(existingVID));
                    if (existingVID == newVID) {
                        return ErrorCode::Success; // idempotent retry
                    }
                }

                if (kind == PostingUpdateKind::Pure) {
                    const long long pureLimit = static_cast<long long>(m_postingSizeLimit) +
                                                static_cast<long long>(m_bufferSizeLimit);
                    if (pureLimit >= 0 && pure >= pureLimit &&
                        !m_taggedMaintenance.load(std::memory_order_acquire)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                     "[TaggedUpdate] pure posting %d reached its static-head capacity (%lld).\n",
                                     headID, pureLimit);
                        return ErrorCode::Posting_OverFlow;
                    }
                    posting.insert(static_cast<size_t>(pure) * m_vectorInfoSize, record);
                    ++pure;

                    const size_t purePages =
                        (static_cast<size_t>(pure) * m_vectorInfoSize + PageSize - 1) / PageSize;
                    const size_t maxRecords =
                        ((purePages + static_cast<size_t>(m_opt->m_unfilterTailBufferLength)) * PageSize) /
                        static_cast<size_t>(m_vectorInfoSize);
                    if (posting.size() / static_cast<size_t>(m_vectorInfoSize) > maxRecords) {
                        posting.resize(maxRecords * static_cast<size_t>(m_vectorInfoSize));
                    }
                } else {
                    const size_t purePages =
                        (static_cast<size_t>(pure) * m_vectorInfoSize + PageSize - 1) / PageSize;
                    const size_t maxRecords =
                        ((purePages + static_cast<size_t>(m_opt->m_unfilterTailBufferLength)) * PageSize) /
                        static_cast<size_t>(m_vectorInfoSize);
                    if (static_cast<size_t>(total) >= maxRecords) {
                        return ErrorCode::Success; // no tail capacity; preserve the pure prefix
                    }
                    posting.append(record);
                }

                ret = db->Put(headID, posting, MaxTimeout, &(p_exWorkSpace->m_diskRequests));
                if (ret != ErrorCode::Success) return ret;
                m_postingSizes.UpdateSize(headID,
                                          static_cast<int>(posting.size() / static_cast<size_t>(m_vectorInfoSize)));
                m_postingPureCounts.UpdateSize(headID, pure);
                *m_checkSums[headID] =
                    m_checkSum.CalcChecksum(posting.c_str(), static_cast<int>(posting.size()));
                if (m_opt->m_consistencyCheck && total > 0) {
                    ret = db->Check(headID, m_postingSizes.GetSize(headID) * m_vectorInfoSize, nullptr);
                }
                return ret;
            };

            for (int v = 0; v < p_vectorSet->Count(); ++v) {
                const SizeType vid = p_begin + v;
                std::string record(static_cast<size_t>(m_vectorInfoSize), '\0');
                if (!SerializeDynamicPosting(record.data(), vid, m_versionMap->GetVersion(vid),
                                             reinterpret_cast<const ValueType*>(p_vectorSet->GetVector(v)),
                                             p_tags + static_cast<size_t>(v) * p_numTagsPerVec,
                                             p_numTagsPerVec)) {
                    return ErrorCode::Undefined;
                }

                for (const PostingUpdateTarget& target : p_targets[static_cast<size_t>(v)]) {
                    ErrorCode ret = writePosting(target.m_headID, target.m_kind, record);
                    if (ret != ErrorCode::Success) return ret;
                }
            }
            return ErrorCode::Success;
        }

        ErrorCode GetTaggedPostingSnapshot(ExtraWorkSpace* p_exWorkSpace,
                                           SizeType p_headID,
                                           TaggedPostingSnapshot& p_snapshot) override
        {
            if (p_exWorkSpace == nullptr || p_headID < 0 ||
                p_headID >= m_postingSizes.GetPostingNum()) {
                return ErrorCode::Key_OverFlow;
            }

            std::shared_lock<std::shared_timed_mutex> lock(m_rwLocks[p_headID]);
            const int total = m_postingSizes.GetSize(p_headID);
            p_snapshot = {};
            p_snapshot.m_headID = p_headID;
            if (total == 0) {
                return ErrorCode::Success;
            }

            ErrorCode ret = db->Get(p_headID, &p_snapshot.m_records, MaxTimeout,
                                    &(p_exWorkSpace->m_diskRequests));
            if (ret != ErrorCode::Success ||
                !m_checkSum.ValidateChecksum(p_snapshot.m_records.c_str(),
                                             static_cast<int>(p_snapshot.m_records.size()),
                                             *m_checkSums[p_headID])) {
                return ret == ErrorCode::Success ? ErrorCode::Fail : ret;
            }
            if (p_snapshot.m_records.size() % static_cast<size_t>(m_vectorInfoSize) != 0) {
                return ErrorCode::Posting_SizeError;
            }
            p_snapshot.m_pureCount = GetPureCount(p_headID);
            const int recordCount = static_cast<int>(
                p_snapshot.m_records.size() / static_cast<size_t>(m_vectorInfoSize));
            if (p_snapshot.m_pureCount < 0 || p_snapshot.m_pureCount > recordCount) {
                return ErrorCode::Posting_SizeError;
            }
            return ErrorCode::Success;
        }

        ErrorCode ReserveTaggedPosting(SizeType p_expectedHeadID) override
        {
            std::lock_guard<std::mutex> lock(m_dataAddLock);
            if (p_expectedHeadID != m_postingSizes.GetPostingNum()) {
                return ErrorCode::Key_OverFlow;
            }
            if (!m_hasPostingPureCounts) {
                InitializePureCountsFromTotals(m_postingSizes.GetPostingNum());
            }
            if (m_postingSizes.AddBatch(1) != ErrorCode::Success ||
                m_checkSums.AddBatch(1) != ErrorCode::Success ||
                m_postingPureCounts.AddBatch(1) != ErrorCode::Success) {
                return ErrorCode::MemoryOverFlow;
            }
            m_postingSizes.UpdateSize(p_expectedHeadID, 0);
            m_postingPureCounts.UpdateSize(p_expectedHeadID, 0);
            *m_checkSums[p_expectedHeadID] = 0;
            return ErrorCode::Success;
        }

        ErrorCode RewriteTaggedPostings(ExtraWorkSpace* p_exWorkSpace,
                                        const std::vector<TaggedPostingSnapshot>& p_rewrites) override
        {
            if (p_exWorkSpace == nullptr || p_rewrites.empty()) {
                return p_rewrites.empty() ? ErrorCode::Success : ErrorCode::Fail;
            }

            std::vector<SizeType> ids;
            ids.reserve(p_rewrites.size());
            for (const auto& rewrite : p_rewrites) {
                if (rewrite.m_headID < 0 || rewrite.m_headID >= m_postingSizes.GetPostingNum() ||
                    rewrite.m_records.size() % static_cast<size_t>(m_vectorInfoSize) != 0) {
                    return ErrorCode::Posting_SizeError;
                }
                const int total = static_cast<int>(
                    rewrite.m_records.size() / static_cast<size_t>(m_vectorInfoSize));
                if (rewrite.m_pureCount < 0 || rewrite.m_pureCount > total) {
                    return ErrorCode::Posting_SizeError;
                }
                const int pureLimit = m_postingSizeLimit + m_bufferSizeLimit;
                if (pureLimit >= 0 && rewrite.m_pureCount > pureLimit) {
                    return ErrorCode::Posting_OverFlow;
                }
                const size_t purePages =
                    (static_cast<size_t>(rewrite.m_pureCount) * m_vectorInfoSize + PageSize - 1) / PageSize;
                const size_t maxRecords =
                    ((purePages + static_cast<size_t>(m_opt->m_unfilterTailBufferLength)) * PageSize) /
                    static_cast<size_t>(m_vectorInfoSize);
                if (static_cast<size_t>(total) > maxRecords) {
                    return ErrorCode::Posting_OverFlow;
                }
                ids.push_back(rewrite.m_headID);
            }
            std::sort(ids.begin(), ids.end());
            if (std::adjacent_find(ids.begin(), ids.end()) != ids.end()) {
                return ErrorCode::Fail;
            }

            std::vector<std::pair<unsigned, SizeType>> lockKeys;
            lockKeys.reserve(ids.size());
            for (SizeType id : ids) {
                lockKeys.emplace_back(
                    COMMON::FineGrainedRWLock::hash_func(static_cast<unsigned>(id)), id);
            }
            std::sort(lockKeys.begin(), lockKeys.end());
            std::vector<std::unique_lock<std::shared_timed_mutex>> locks;
            locks.reserve(lockKeys.size());
            for (size_t i = 0; i < lockKeys.size();) {
                locks.emplace_back(m_rwLocks[lockKeys[i].second]);
                const unsigned bucket = lockKeys[i].first;
                do {
                    ++i;
                } while (i < lockKeys.size() && lockKeys[i].first == bucket);
            }

            struct PreviousPosting
            {
                SizeType m_headID = -1;
                int m_pureCount = 0;
                std::string m_records;
            };
            std::vector<PreviousPosting> previous;
            previous.reserve(p_rewrites.size());
            for (const auto& rewrite : p_rewrites) {
                PreviousPosting old;
                old.m_headID = rewrite.m_headID;
                old.m_pureCount = GetPureCount(rewrite.m_headID);
                const int total = m_postingSizes.GetSize(rewrite.m_headID);
                if (total > 0) {
                    ErrorCode ret = db->Get(rewrite.m_headID, &old.m_records, MaxTimeout,
                                            &(p_exWorkSpace->m_diskRequests));
                    if (ret != ErrorCode::Success ||
                        !m_checkSum.ValidateChecksum(old.m_records.c_str(),
                                                     static_cast<int>(old.m_records.size()),
                                                     *m_checkSums[rewrite.m_headID])) {
                        return ret == ErrorCode::Success ? ErrorCode::Fail : ret;
                    }
                }
                previous.emplace_back(std::move(old));
            }

            auto restorePrevious = [&]() {
                for (const auto& old : previous) {
                    if (old.m_records.empty()) {
                        db->Delete(old.m_headID);
                    } else {
                        db->Put(old.m_headID, old.m_records, MaxTimeout,
                                &(p_exWorkSpace->m_diskRequests));
                    }
                }
            };

            size_t applied = 0;
            ErrorCode ret = ErrorCode::Success;
            for (; applied < p_rewrites.size(); ++applied) {
                const auto& rewrite = p_rewrites[applied];
                if (rewrite.m_records.empty()) {
                    ret = db->Delete(rewrite.m_headID);
                } else {
                    ret = db->Put(rewrite.m_headID, rewrite.m_records, MaxTimeout,
                                  &(p_exWorkSpace->m_diskRequests));
                }
                if (ret != ErrorCode::Success) break;
            }
            if (ret != ErrorCode::Success) {
                restorePrevious();
                return ret;
            }

            for (const auto& rewrite : p_rewrites) {
                const int total = static_cast<int>(
                    rewrite.m_records.size() / static_cast<size_t>(m_vectorInfoSize));
                if (m_opt->m_consistencyCheck && total > 0) {
                    ret = db->Check(rewrite.m_headID,
                                    total * m_vectorInfoSize, nullptr);
                    if (ret != ErrorCode::Success) {
                        restorePrevious();
                        return ret;
                    }
                }
            }
            for (const auto& rewrite : p_rewrites) {
                const int total = static_cast<int>(
                    rewrite.m_records.size() / static_cast<size_t>(m_vectorInfoSize));
                m_postingSizes.UpdateSize(rewrite.m_headID, total);
                m_postingPureCounts.UpdateSize(rewrite.m_headID, rewrite.m_pureCount);
                *m_checkSums[rewrite.m_headID] = rewrite.m_records.empty()
                    ? 0
                    : m_checkSum.CalcChecksum(rewrite.m_records.c_str(),
                                               static_cast<int>(rewrite.m_records.size()));
            }
            return ErrorCode::Success;
        }

        ErrorCode ReadTaggedFullVectors(const std::vector<SizeType>& p_vids,
                                        ByteArray& p_vectors) override
        {
            if (m_opt == nullptr || p_vids.empty()) {
                return p_vids.empty() ? ErrorCode::Success : ErrorCode::Fail;
            }
            const size_t vectorBytes = static_cast<size_t>(m_opt->m_dim) * sizeof(ValueType);
            p_vectors = ByteArray::Alloc(vectorBytes * p_vids.size());
            if (p_vectors.Data() == nullptr) return ErrorCode::MemoryOverFlow;
            auto* output = p_vectors.Data();
            for (size_t i = 0; i < p_vids.size(); ++i) {
                const SizeType vid = p_vids[i];
                if (vid < 0 || vid > std::numeric_limits<int>::max()) {
                    return ErrorCode::Key_OverFlow;
                }
                const ValueType* vector = ReadBaseVecDirect(static_cast<int>(vid), m_opt->m_dim);
                if (vector == nullptr) return ErrorCode::Fail;
                std::memcpy(output + i * vectorBytes, vector, vectorBytes);
            }
            return ErrorCode::Success;
        }

        void DrainTaggedMergeCandidates(std::vector<SizeType>& p_candidates) override
        {
            std::lock_guard<std::mutex> lock(m_taggedMergeCandidatesLock);
            p_candidates.reserve(p_candidates.size() + m_taggedMergeCandidates.size());
            for (SizeType headID : m_taggedMergeCandidates) {
                p_candidates.push_back(headID);
            }
            m_taggedMergeCandidates.clear();
        }

        SizeType GetTaggedPostingCount() override
        {
            return m_postingSizes.GetPostingNum();
        }

        int GetTaggedRecordSize() const override
        {
            return m_vectorInfoSize;
        }

        int GetTaggedPureCapacity() const override
        {
            return m_postingSizeLimit + m_bufferSizeLimit;
        }

        int GetTaggedMergeThreshold() const override
        {
            return m_mergeThreshold;
        }

        ErrorCode DeleteIndex(SizeType p_id) override {
            if (m_opt->m_enableWAL && m_wal) {
                std::string assignment(sizeof(SizeType), '\0');
                memcpy((char*)assignment.c_str(), &p_id, sizeof(SizeType));
                m_wal->PutAssignment(assignment);
            }
            if (m_versionMap->Delete(p_id)) return ErrorCode::Success;
            return ErrorCode::VectorNotFound;
        }

        SizeType SearchVector(ExtraWorkSpace* p_exWorkSpace, std::shared_ptr<VectorSet>& p_vectorSet,
            std::shared_ptr<VectorIndex> p_index, int testNum = 64, SizeType VID = -1) override {
            
            QueryResult queryResults(p_vectorSet->GetVector(0), testNum, false);
            p_index->SearchIndex(queryResults);
            
            std::set<SizeType> checked;
            std::string postingList;
            for (int i = 0; i < queryResults.GetResultNum(); ++i)
            {
                if (db->Get(queryResults.GetResult(i)->VID, &postingList, MaxTimeout,
                            &(p_exWorkSpace->m_diskRequests)) != ErrorCode::Success ||
                    !m_checkSum.ValidateChecksum(postingList.c_str(), (int)(postingList.size()), *m_checkSums[queryResults.GetResult(i)->VID]))
                {
                    continue;
                }
                int vectorNum = (int)(postingList.size() / m_vectorInfoSize);

                for (int j = 0; j < vectorNum; j++) {
                    char* vectorInfo = (char* )postingList.data() + j * m_vectorInfoSize;
                    int vectorID = *(reinterpret_cast<int*>(vectorInfo));
                    if(checked.find(vectorID) != checked.end() || m_versionMap->Deleted(vectorID)) {
                        continue;
                    }
                    checked.insert(vectorID);
                    if (VID != -1 && VID == vectorID) SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Find %d in %dth posting\n", VID, i);
                    auto distance2leaf = p_index->ComputeDistance(queryResults.GetQuantizedTarget(), vectorInfo + m_metaDataSize);
                    if (distance2leaf < 1e-6) return vectorID;
                }
            }
            return -1;
        }

        void ForceGC(ExtraWorkSpace* p_exWorkSpace, VectorIndex* p_index) override {
            for (int i = 0; i < p_index->GetNumSamples(); i++) {
                if (!p_index->ContainSample(i)) continue;
                Split(p_exWorkSpace, p_index, i, false);
            }
        }

        bool AllFinished() { return m_splitThreadPool ? m_splitThreadPool->allClear() : true; } // && m_reassignThreadPool->allClear(); }
        void ForceCompaction() override { db->ForceCompaction(); }
        void GetDBStats() override { 
            db->GetStat();
            if (m_splitThreadPool)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "remain splitJobs: %d, reassignJobs: %d, running split: %d, running reassign: %d\n", m_splitThreadPool->jobsize(), 0, m_splitThreadPool->runningJobs(), 0);
            }
        }

        int64_t GetNumBlocks() override
        {
            return db->GetNumBlocks();   
        }

        void GetIndexStats(int finishedInsert, bool cost, bool reset) override { m_stat.PrintStat(finishedInsert, cost, reset); }

        bool CheckValidPosting(SizeType postingID) override {
            return (postingID < m_postingSizes.GetPostingNum()) && (m_postingSizes.GetSize(postingID) > 0);
        }

        virtual ErrorCode CheckPosting(SizeType postingID, std::vector<std::uint8_t> *visited = nullptr,
                                       ExtraWorkSpace *p_exWorkSpace = nullptr) override
        {
            if (postingID < 0 || postingID >= m_postingSizes.GetPostingNum())
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[CheckPosting]: Error postingID %d (should be 0 ~ %d)\n",
                             postingID, m_postingSizes.GetPostingNum());
                return ErrorCode::Key_OverFlow;
            }
            if (m_postingSizes.GetSize(postingID) < 0)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[CheckPosting]: postingID %d has wrong size:%d\n", postingID,
                             m_postingSizes.GetSize(postingID));
                return ErrorCode::Posting_SizeError;
            }
            ErrorCode ret = db->Check(postingID, m_postingSizes.GetSize(postingID) * m_vectorInfoSize, visited);
            if (ret != ErrorCode::Success)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[CheckPosting]: postingID %d has wrong meta data\n",
                             postingID);
                return ret;
            }

                        
            if (m_opt->m_checksumInRead && p_exWorkSpace != nullptr)
            {
                std::string posting;
                if ((ret = db->Get(postingID, &posting, MaxTimeout, &(p_exWorkSpace->m_diskRequests))) !=
                        ErrorCode::Success ||
                    !m_checkSum.ValidateChecksum(posting.c_str(), (int)(posting.size()), *m_checkSums[postingID]))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[CheckPosting] Get checksum fail %d!\n", postingID);
                    PrintErrorInPosting(posting, postingID);
                    return ret;
                }
            }
            return ErrorCode::Success;
        }

        ErrorCode GetWritePosting(ExtraWorkSpace* p_exWorkSpace, SizeType pid, std::string& posting, bool write = false) override {
            ErrorCode ret;
            if (write) {
                if ((ret = db->Put(pid, posting, MaxTimeout, &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[GetWritePosting] Put fail!\n");
                    return ret;
                }
                    
                m_postingSizes.UpdateSize(pid, posting.size() / m_vectorInfoSize);
                *m_checkSums[pid] = m_checkSum.CalcChecksum(posting.c_str(), (int)(posting.size()));
                if (m_opt->m_consistencyCheck && (ret = db->Check(pid, m_postingSizes.GetSize(pid) * m_vectorInfoSize, nullptr)) != ErrorCode::Success)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[GetWritePosting] Check fail!\n");
                    return ret;
                }
                // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "PostingSize: %d\n", m_postingSizes.GetSize(pid));
            } else {
                if ((ret = db->Get(pid, &posting, MaxTimeout, &(p_exWorkSpace->m_diskRequests))) != ErrorCode::Success ||
                    !m_checkSum.ValidateChecksum(posting.c_str(), (int)(posting.size()), *m_checkSums[pid])) 
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[GetWritePosting] Get fail!\n");
                    return ret;
                }
            }
            return ErrorCode::Success;
        }

        ErrorCode Checkpoint(std::string prefix) override {
            /**flush SPTAG, versionMap, block mapping, block pool**/
            /** Wait **/
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Waiting for index update complete\n");
            while(!AllFinished())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            if (m_asyncStatus != ErrorCode::Success)
                return m_asyncStatus;

            std::string p_persistenMap = prefix + FolderSep + m_opt->m_deleteIDFile;
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Saving version map\n");
            
            ErrorCode ret;
            if ((ret = m_versionMap->Save(p_persistenMap)) != ErrorCode::Success)
                return ret;

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Saving posting size\n");
            std::string p_persistenRecord = prefix + FolderSep + m_opt->m_ssdInfoFile;
            if ((ret = m_postingSizes.Save(p_persistenRecord)) != ErrorCode::Success)
                return ret;

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Saving posting checksum\n");
            std::string p_checksumPath = prefix + FolderSep + m_opt->m_checksumFile;
            if ((ret = m_checkSums.Save(p_checksumPath)) != ErrorCode::Success)
                return ret;

            if (m_hasPostingPureCounts) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Saving posting_pure_counts sidecar\n");
                std::string p_pureCountsPath = prefix + FolderSep + m_opt->m_postingPureCountsFile;
                if ((ret = m_postingPureCounts.Save(p_pureCountsPath)) != ErrorCode::Success)
                    return ret;
            }

#ifndef _MSC_VER
            std::shared_lock<std::shared_mutex> dynamicVectorLock(m_dynamicVectorLock);
            if (m_dynamicVectorFd >= 0 && fsync(m_dynamicVectorFd) != 0) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                             "[TaggedUpdate] failed to flush dynamic vector sidecar.\n");
                return ErrorCode::Fail;
            }
            if (!CopyDynamicVectorStore(prefix)) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                             "[TaggedUpdate] failed to checkpoint dynamic vector sidecar.\n");
                return ErrorCode::Fail;
            }
#endif
            if ((ret = db->Checkpoint(prefix)) != ErrorCode::Success)
                return ret;
            if (m_opt->m_enableWAL && m_wal) {
                /** delete all the previous record **/
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Checkpoint done, delete previous record\n");
                m_wal->ClearPreviousRecord();
            }
            return ErrorCode::Success;
        }

        ErrorCode GetPostingDebug(ExtraWorkSpace* p_exWorkSpace, std::shared_ptr<VectorIndex> p_index, SizeType vid, std::vector<SizeType>& VIDs, std::shared_ptr<VectorSet>& vecs) {
            std::string posting;
            db->Get(vid, &posting, MaxTimeout, &(p_exWorkSpace->m_diskRequests));
            int vectorNum = (int)(posting.size() / m_vectorInfoSize);
            int vectorNum_real = vectorNum;
            for (int j = 0; j < vectorNum; j++) {
                char* vectorInfo = (char*)posting.data() + j * m_vectorInfoSize;
                int vectorID = *(reinterpret_cast<int*>(vectorInfo));
                uint8_t version = *(reinterpret_cast<uint8_t*>(vectorInfo + sizeof(int)));
                if(m_versionMap->GetVersion(vectorID) != version) {
                    vectorNum_real--;
                }
                
            }
            VIDs.resize(vectorNum_real);
            ByteArray vector_array = ByteArray::Alloc(sizeof(ValueType) * vectorNum_real * m_opt->m_dim);
            vecs.reset(new BasicVectorSet(vector_array, GetEnumValueType<ValueType>(), m_opt->m_dim, vectorNum_real));

            for (int j = 0, i = 0; j < vectorNum; j++) {
                char* vectorInfo = (char*)posting.data() + j * m_vectorInfoSize;
                int vectorID = *(reinterpret_cast<int*>(vectorInfo));
                uint8_t version = *(reinterpret_cast<uint8_t*>(vectorInfo + sizeof(int)));
                if(m_versionMap->GetVersion(vectorID) != version) {
                    continue;
                }
                VIDs[i] = vectorID;
                auto outVec = vecs->GetVector(i);
                memcpy(outVec, (void*)(vectorInfo + sizeof(int) + sizeof(uint8_t)), sizeof(ValueType) * m_opt->m_dim);
                i++;
            }
            return ErrorCode::Success;
        }

        // =====================================================================
        // OPQ prefilter: metadata-only postings + resident OPQ codes + point store
        // =====================================================================
        // ===================================================================
        // In-posting quantization (zero-resident-memory posting compression)
        // ===================================================================
        // Quantize a full uint8 vector -> packed 4-bit code (2 dims/byte).
        inline void InpostQuantizeVec(const ValueType* v, char* code, int dim) const {
            const uint8_t* uv = reinterpret_cast<const uint8_t*>(v);
            uint8_t* c = reinterpret_cast<uint8_t*>(code);
            for (int d = 0; d + 1 < dim; d += 2) {
                c[d >> 1] = (uint8_t)(((uv[d] >> 4) << 4) | (uv[d + 1] >> 4));
            }
            if (dim & 1) c[dim >> 1] = (uint8_t)((uv[dim - 1] >> 4) << 4);
        }
        // ADC squared-L2 between a full uint8 query and a packed 4-bit code.
        inline float InpostL2(const ValueType* q, const char* code, int dim) const {
            const uint8_t* uq = reinterpret_cast<const uint8_t*>(q);
            const uint8_t* c = reinterpret_cast<const uint8_t*>(code);
            float s = 0;
            for (int d = 0; d + 1 < dim; d += 2) {
                uint8_t b = c[d >> 1];
                int r0 = (int)((b >> 4) << 4) | 8;     // bucket midpoint
                int r1 = (int)((b & 0xf) << 4) | 8;
                int e0 = (int)uq[d] - r0;
                int e1 = (int)uq[d + 1] - r1;
                s += (float)(e0 * e0 + e1 * e1);
            }
            if (dim & 1) {
                int r = (int)((c[dim >> 1] >> 4) << 4) | 8;
                int e = (int)uq[dim - 1] - r;
                s += (float)(e * e);
            }
            return s;
        }
        // One-time offline rewrite: read each full posting, replace each record's
        // ValueType vector with the packed quantized code, write the slim posting
        // back. Guarded by the inpost_quant.bin marker (skips if already done).
        // Requires the constructor to have sized m_vectorInfoSize to the quantized
        // stride (env SPTAG_INPOST_QUANT_BITS); the full stride is a local here.
        void QuantizeInPostings() {
            if (m_inpostQuantBits <= 0) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[InpostQuant] build requested but bits not set\n");
                return;
            }
            std::string dir = m_opt->m_indexDirectory + FolderSep;
            std::string marker = dir + "inpost_quant.bin";
            {
                std::ifstream mf(marker, std::ios::binary);
                if (mf.good()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[InpostQuant] marker present, skip transform\n");
                    return;
                }
            }
            int dim = m_opt->m_dim;
            int fullStride = m_metaDataSize + dim * (int)sizeof(ValueType);
            int slimStride = m_vectorInfoSize;  // = m_metaDataSize + m_inpostPackedBytes
            SizeType postingNum = m_postingSizes.GetPostingNum();
            ExtraWorkSpace ws; InitWorkSpace(&ws);
            std::string blob, out;
            size_t totalRecs = 0, slimHeads = 0;
            for (SizeType h = 0; h < postingNum; h++) {
                int n = m_postingSizes.GetSize(h);
                if (n <= 0) continue;
                if (db->Get(h, &blob, MaxTimeout, &(ws.m_diskRequests)) != ErrorCode::Success) continue;
                int avail = (int)(blob.size() / fullStride);
                if (avail < n) n = avail;
                out.assign((size_t)slimStride * n, '\0');
                const char* src = blob.data();
                char* dst = (char*)out.data();
                for (int i = 0; i < n; i++) {
                    const char* e = src + (size_t)i * fullStride;
                    char* o = dst + (size_t)i * slimStride;
                    memcpy(o, e, m_metaDataSize);  // [id|ver|tags]
                    InpostQuantizeVec(reinterpret_cast<const ValueType*>(e + m_metaDataSize),
                                      o + m_metaDataSize, dim);
                }
                if (GetWritePosting(&ws, h, out, true) != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[InpostQuant] write posting %d fail\n", (int)h);
                    return;
                }
                totalRecs += n; slimHeads++;
            }
            Checkpoint(m_opt->m_indexDirectory);
            {
                std::ofstream mf(marker, std::ios::binary);
                int hdr[2] = { m_inpostQuantBits, m_inpostPackedBytes };
                mf.write((const char*)hdr, sizeof(hdr));
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "[InpostQuant] DONE heads=%zu recs=%zu fullStride=%d slimStride=%d (%.2fx smaller record)\n",
                slimHeads, totalRecs, fullStride, slimStride, (double)fullStride / slimStride);
        }

        // One-time offline rewrite for in-posting RaBitQ b1: read each full posting,
        // replace each record's ValueType vector with that vid's 1-bit RaBitQ code
        // (read from rabitq2_b1.bin, indexed by vid), write the slim posting back.
        // Guarded by the inpost_rbq.bin marker. Requires the constructor to have
        // sized m_vectorInfoSize to the slim stride (env SPTAG_INPOST_RBQ).
        void TransformInPostingsRbq() {
            if (!m_inpostRbq || m_inpostRbqBinBytes <= 0) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[InpostRBQ] build requested but mode not set\n");
                return;
            }
            std::string dir = m_opt->m_indexDirectory + FolderSep;
            std::string marker = dir + "inpost_rbq.bin";
            {
                std::ifstream mf(marker, std::ios::binary);
                if (mf.good()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[InpostRBQ] marker present, skip transform\n");
                    return;
                }
            }
            // mmap the code sidecar (header + rotator + centroid, then per-vec bin[+ex]).
            std::string codePath = dir + m_inpostRbqFile;
            int cfd = open(codePath.c_str(), O_RDONLY);
            if (cfd < 0) { SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[InpostRBQ] open %s fail\n", codePath.c_str()); return; }
            off_t csz = lseek(cfd, 0, SEEK_END);
            void* cmap = mmap(nullptr, (size_t)csz, PROT_READ, MAP_SHARED, cfd, 0);
            close(cfd);
            if (cmap == MAP_FAILED) { SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[InpostRBQ] mmap codes fail\n"); return; }
            // Parse header to find the per-vector code region offset.
            const int32_t* h = reinterpret_cast<const int32_t*>(cmap);
            int32_t N = h[1], pdim = h[3], rbytes = h[6];
            int binBytes = m_inpostRbqBinBytes;
            int exBytes = m_inpostRbqExBytes;
            int codeBytes = binBytes + exBytes;   // per-vec stride in sidecar
            // layout: 7*int32 header, rbytes rotator, pdim*float centroid, then N*codeBytes
            size_t codeBase = (size_t)7 * 4 + (size_t)rbytes + (size_t)pdim * sizeof(float);
            const uint8_t* codes = reinterpret_cast<const uint8_t*>(cmap) + codeBase;

            int dim = m_opt->m_dim;
            int fullStride = m_metaDataSize + dim * (int)sizeof(ValueType);
            int slimStride = m_vectorInfoSize;  // = m_metaDataSize + binBytes + exBytes
            SizeType postingNum = m_postingSizes.GetPostingNum();
            ExtraWorkSpace ws; InitWorkSpace(&ws);
            std::string blob, out;
            size_t totalRecs = 0, slimHeads = 0;
            for (SizeType hh = 0; hh < postingNum; hh++) {
                int n = m_postingSizes.GetSize(hh);
                if (n <= 0) continue;
                if (db->Get(hh, &blob, MaxTimeout, &(ws.m_diskRequests)) != ErrorCode::Success) continue;
                int avail = (int)(blob.size() / fullStride);
                if (avail < n) n = avail;
                out.assign((size_t)slimStride * n, '\0');
                const char* src = blob.data();
                char* dst = (char*)out.data();
                for (int i = 0; i < n; i++) {
                    const char* e = src + (size_t)i * fullStride;
                    char* o = dst + (size_t)i * slimStride;
                    memcpy(o, e, m_metaDataSize);  // [id|ver|tags]
                    int vid = *reinterpret_cast<const int*>(e);
                    if (vid >= 0 && vid < N)
                        memcpy(o + m_metaDataSize, codes + (size_t)vid * codeBytes, codeBytes);
                }
                if (GetWritePosting(&ws, hh, out, true) != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[InpostRBQ] write posting %d fail\n", (int)hh);
                    munmap(cmap, (size_t)csz);
                    return;
                }
                totalRecs += n; slimHeads++;
                if (slimHeads % 100000 == 0)
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[InpostRBQ] transformed %zu heads\n", slimHeads);
            }
            munmap(cmap, (size_t)csz);
            Checkpoint(m_opt->m_indexDirectory);
            {
                std::ofstream mf(marker, std::ios::binary);
                int hdr[2] = { 1, m_inpostRbqBinBytes };
                mf.write((const char*)hdr, sizeof(hdr));
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "[InpostRBQ] DONE heads=%zu recs=%zu fullStride=%d slimStride=%d (%.2fx smaller record)\n",
                slimHeads, totalRecs, fullStride, slimStride, (double)fullStride / slimStride);
        }

        // Contiguous variant of the in-posting RaBitQ transform. The in-place
        // TransformInPostingsRbq() rewrites postings via db->Put, which (for an already
        // populated store) allocates slim records from the EXISTING fragmented free pool
        // -> slim postings get scattered -> poor cold locality. This variant instead
        // writes every slim posting ONCE, in head order, into a FRESH FileIO store whose
        // blockpool is initialized sequentially (0,1,2,..) -> each posting's blocks are
        // contiguous AND postings are laid out in head order. The fresh store files are
        // then renamed over the originals. Enabled by SPTAG_INPOST_RBQ_CONTIG=1 together
        // with SPTAG_INPOST_RBQ_BUILD=1 (and SPTAG_INPOST_RBQ=1 to size the slim stride).
        void TransformInPostingsRbqContig() {
            if (!m_inpostRbq || m_inpostRbqBinBytes <= 0) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[InpostRBQ-contig] build requested but mode not set\n");
                return;
            }
            std::string dir = m_opt->m_indexDirectory + FolderSep;
            std::string marker = dir + "inpost_rbq.bin";
            {
                std::ifstream mf(marker, std::ios::binary);
                if (mf.good()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[InpostRBQ-contig] marker present, skip transform\n");
                    return;
                }
            }
            // mmap the code sidecar (header + rotator + centroid, then per-vec bin[+ex]).
            std::string codePath = dir + m_inpostRbqFile;
            int cfd = open(codePath.c_str(), O_RDONLY);
            if (cfd < 0) { SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[InpostRBQ-contig] open %s fail\n", codePath.c_str()); return; }
            off_t csz = lseek(cfd, 0, SEEK_END);
            void* cmap = mmap(nullptr, (size_t)csz, PROT_READ, MAP_SHARED, cfd, 0);
            close(cfd);
            if (cmap == MAP_FAILED) { SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[InpostRBQ-contig] mmap codes fail\n"); return; }
            const int32_t* h = reinterpret_cast<const int32_t*>(cmap);
            int32_t N = h[1], pdim = h[3], rbytes = h[6];
            int binBytes = m_inpostRbqBinBytes;
            int exBytes = m_inpostRbqExBytes;
            int codeBytes = binBytes + exBytes;
            size_t codeBase = (size_t)7 * 4 + (size_t)rbytes + (size_t)pdim * sizeof(float);
            const uint8_t* codes = reinterpret_cast<const uint8_t*>(cmap) + codeBase;

            int dim = m_opt->m_dim;
            int fullStride = m_metaDataSize + dim * (int)sizeof(ValueType);
            int slimStride = m_vectorInfoSize;  // = m_metaDataSize + binBytes + exBytes

            // Fresh contiguous store: distinct file prefix -> non-existent files ->
            // sequential blockpool init -> contiguous first-writes.
            SPANN::Options optCopy = *m_opt;
            optCopy.m_ssdMappingFile = m_opt->m_ssdMappingFile + "_contig";
            optCopy.m_recovery = false;
            {
                // Right-size the prealloc to the slim footprint (slim/full of the original
                // StartFileSize), plus headroom; growth covers any underestimate.
                double ratio = (double)slimStride / (double)fullStride;
                int est = (int)std::ceil(m_opt->m_startFileSize * ratio) + 2;
                if (est < 2) est = 2;
                optCopy.m_startFileSize = est;
            }
            for (const char* suf : { "", "_postings", "_postings_blockpool" }) {
                std::string p = dir + optCopy.m_ssdMappingFile + suf;
                std::remove(p.c_str());
            }
            FileIO newdb(optCopy);
            if (!newdb.Available()) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[InpostRBQ-contig] fresh store init failed\n");
                munmap(cmap, (size_t)csz);
                return;
            }

            SizeType postingNum = m_postingSizes.GetPostingNum();
            ExtraWorkSpace ws; InitWorkSpace(&ws);
            std::string blob, out;
            size_t totalRecs = 0, slimHeads = 0;
            for (SizeType hh = 0; hh < postingNum; hh++) {
                int n = m_postingSizes.GetSize(hh);
                if (n <= 0) continue;
                if (db->Get(hh, &blob, MaxTimeout, &(ws.m_diskRequests)) != ErrorCode::Success) continue;
                int avail = (int)(blob.size() / fullStride);
                if (avail < n) n = avail;
                out.assign((size_t)slimStride * n, '\0');
                const char* src = blob.data();
                char* dst = (char*)out.data();
                for (int i = 0; i < n; i++) {
                    const char* e = src + (size_t)i * fullStride;
                    char* o = dst + (size_t)i * slimStride;
                    memcpy(o, e, m_metaDataSize);  // [id|ver|tags]
                    int vid = *reinterpret_cast<const int*>(e);
                    if (vid >= 0 && vid < N)
                        memcpy(o + m_metaDataSize, codes + (size_t)vid * codeBytes, codeBytes);
                }
                if (newdb.Put(hh, out, MaxTimeout, &(ws.m_diskRequests)) != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[InpostRBQ-contig] write posting %d fail\n", (int)hh);
                    munmap(cmap, (size_t)csz);
                    return;
                }
                // ssdinfo record COUNTS are unchanged by slimming; only checksums change.
                if (hh < m_checkSums.R())
                    *m_checkSums[hh] = m_checkSum.CalcChecksum(out.c_str(), (int)out.size());
                totalRecs += n; slimHeads++;
                if (slimHeads % 100000 == 0)
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[InpostRBQ-contig] transformed %zu heads\n", slimHeads);
            }
            munmap(cmap, (size_t)csz);

            // Persist the fresh store (mapping + blockpool under the _contig prefix).
            if (newdb.Checkpoint(m_opt->m_indexDirectory) != ErrorCode::Success) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[InpostRBQ-contig] checkpoint fresh store failed\n");
                return;
            }
            newdb.ShutDown();
            m_checkSums.Save(dir + m_opt->m_checksumFile);

            // Swap the fresh contiguous files over the originals. The live `db` still holds
            // the old (now-unlinked) inode open; it is released at process exit.
            for (const char* suf : { "", "_postings", "_postings_blockpool" }) {
                std::string from = dir + optCopy.m_ssdMappingFile + suf;
                std::string to   = dir + m_opt->m_ssdMappingFile + suf;
                if (std::rename(from.c_str(), to.c_str()) != 0)
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                        "[InpostRBQ-contig] rename %s -> %s failed\n", from.c_str(), to.c_str());
            }
            {
                std::ofstream mf(marker, std::ios::binary);
                int hdr[2] = { 1, m_inpostRbqBinBytes };
                mf.write((const char*)hdr, sizeof(hdr));
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "[InpostRBQ-contig] DONE heads=%zu recs=%zu fullStride=%d slimStride=%d "
                "(%.2fx smaller record, fresh contiguous store)\n",
                slimHeads, totalRecs, fullStride, slimStride, (double)fullStride / slimStride);
        }

        // One-time offline rewrite for in-posting OPQ (DB-resident): read each full
        // posting (vector inline), replace each record's ValueType vector with that
        // vid's M-byte OPQ code (read from opq_codes_m<M>.bin, raw N*M uint8, vid-indexed),
        // and write the slim [meta | code] posting back into the SAME posting store db.
        // Search then reads these slim records via the baseline async MultiGet path.
        // Guarded by the inpost_opq.bin marker. Requires the constructor to have sized
        // m_vectorInfoSize to the slim stride (env SPTAG_OPQ_INPOST_DB=<M>).
        void TransformInPostingsOpq() {
            if (!m_opqInpostDb || m_opqInpostDbM <= 0) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[InpostOPQ-DB] build requested but mode not set\n");
                return;
            }
            int M = m_opqInpostDbM;
            std::string dir = m_opt->m_indexDirectory + FolderSep;
            std::string marker = dir + "inpost_opq.bin";
            {
                std::ifstream mf(marker, std::ios::binary);
                if (mf.good()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[InpostOPQ-DB] marker present, skip transform\n");
                    return;
                }
            }
            // Load the raw OPQ codes sidecar (N*M uint8, vid-indexed, no header).
            char codeName[64];
            snprintf(codeName, sizeof(codeName), "opq_codes_m%d.bin", M);
            std::string codePath = dir + codeName;
            std::ifstream cin(codePath, std::ios::binary);
            if (!cin) { SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[InpostOPQ-DB] open %s fail\n", codePath.c_str()); return; }
            SizeType N = m_opt->m_vectorSize;
            std::vector<std::uint8_t> codes((size_t)N * M);
            cin.read((char*)codes.data(), (std::streamsize)codes.size());
            if ((size_t)cin.gcount() != codes.size()) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[InpostOPQ-DB] %s short read (%zd / %zu)\n",
                    codePath.c_str(), (ssize_t)cin.gcount(), codes.size());
                return;
            }
            int dim = m_opt->m_dim;
            int fullStride = m_metaDataSize + dim * (int)sizeof(ValueType);
            int slimStride = m_vectorInfoSize;  // = m_metaDataSize + M
            SizeType postingNum = m_postingSizes.GetPostingNum();
            ExtraWorkSpace ws; InitWorkSpace(&ws);
            std::string blob, out;
            size_t totalRecs = 0, slimHeads = 0;
            for (SizeType hh = 0; hh < postingNum; hh++) {
                int n = m_postingSizes.GetSize(hh);
                if (n <= 0) continue;
                if (db->Get(hh, &blob, MaxTimeout, &(ws.m_diskRequests)) != ErrorCode::Success) continue;
                int avail = (int)(blob.size() / fullStride);
                if (avail < n) n = avail;
                out.assign((size_t)slimStride * n, '\0');
                const char* src = blob.data();
                char* dst = (char*)out.data();
                for (int i = 0; i < n; i++) {
                    const char* e = src + (size_t)i * fullStride;
                    char* o = dst + (size_t)i * slimStride;
                    memcpy(o, e, m_metaDataSize);  // [id|ver|tags]
                    int vid = *reinterpret_cast<const int*>(e);
                    if (vid >= 0 && vid < N)
                        memcpy(o + m_metaDataSize, &codes[(size_t)vid * M], M);
                }
                if (GetWritePosting(&ws, hh, out, true) != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[InpostOPQ-DB] write posting %d fail\n", (int)hh);
                    return;
                }
                totalRecs += n; slimHeads++;
                if (slimHeads % 100000 == 0)
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[InpostOPQ-DB] transformed %zu heads\n", slimHeads);
            }
            Checkpoint(m_opt->m_indexDirectory);
            {
                std::ofstream mf(marker, std::ios::binary);
                int hdr[2] = { M, slimStride };
                mf.write((const char*)hdr, sizeof(hdr));
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "[InpostOPQ-DB] DONE heads=%zu recs=%zu fullStride=%d slimStride=%d (%.2fx smaller record)\n",
                slimHeads, totalRecs, fullStride, slimStride, (double)fullStride / slimStride);
        }

        // One-time offline rewrite for PipeANN-style PQ over an existing posting layout.
        // It preserves posting membership, ordering, and pure/tail split, but can change
        // record stride (e.g. OPQ25 [meta25|code25] -> PipePQ32 [meta25|code32]).
        void TransformInPostingsPipePQ() {
            if (!m_pipePQ || !m_opqInpostDb || m_opqInpostDbM <= 0) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[InpostPipePQ-DB] transform requested but mode not set\n");
                return;
            }
            int M = m_opqInpostDbM;
            std::string dir = m_opt->m_indexDirectory + FolderSep;
            std::string marker = dir + "inpost_pipepq.bin";
            {
                std::ifstream mf(marker, std::ios::binary);
                if (mf.good()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[InpostPipePQ-DB] marker present, skip transform\n");
                    return;
                }
            }

            void* codeMap = nullptr;
            size_t codeMapSize = 0, codeOffset = 0;
            const std::uint8_t* codes = nullptr;
            SizeType codeN = 0;
            if (!MmapPipePQCodes(m_pipePQCodesPathResolved, M, codeMap, codeMapSize, codes, codeN,
                                 codeOffset, "InpostPipePQ-DB")) {
                return;
            }

            const int dstStride = m_vectorInfoSize;  // = m_metaDataSize + M
            if (dstStride != m_metaDataSize + M) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "[InpostPipePQ-DB] unsupported dst stride=%d meta=%d M=%d\n",
                    dstStride, m_metaDataSize, M);
                munmap(codeMap, codeMapSize);
                return;
            }
            int srcStride = dstStride;
            {
                std::ifstream om(dir + "inpost_opq.bin", std::ios::binary);
                int hdr[2] = { 0, 0 };
                if (om.good()) {
                    om.read((char*)hdr, sizeof(hdr));
                    if (om && hdr[1] >= m_metaDataSize) srcStride = hdr[1];
                }
            }
            if (srcStride < m_metaDataSize) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "[InpostPipePQ-DB] bad source stride=%d meta=%d\n", srcStride, m_metaDataSize);
                munmap(codeMap, codeMapSize);
                return;
            }

            SizeType postingNum = m_postingSizes.GetPostingNum();
            ExtraWorkSpace ws; InitWorkSpace(&ws);
            std::string blob, out;
            size_t totalRecs = 0, slimHeads = 0;
            auto fileDb = std::dynamic_pointer_cast<FileIO>(db);
            if (fileDb) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                    "[InpostPipePQ-DB] using FileIO::RewriteInPlace (delta-page rewrite, no full copy-on-write)\n");
            }
            for (SizeType hh = 0; hh < postingNum; hh++) {
                int n = m_postingSizes.GetSize(hh);
                if (n <= 0) continue;
                if (db->Get(hh, &blob, MaxTimeout, &(ws.m_diskRequests)) != ErrorCode::Success) continue;
                int avail = (int)(blob.size() / srcStride);
                if (avail < n) n = avail;
                out.assign((size_t)dstStride * n, '\0');
                const char* src = blob.data();
                char* dst = (char*)out.data();
                for (int i = 0; i < n; i++) {
                    const char* e = src + (size_t)i * srcStride;
                    char* o = dst + (size_t)i * dstStride;
                    memcpy(o, e, m_metaDataSize);
                    int vid = *reinterpret_cast<const int*>(e);
                    if (vid >= 0 && vid < codeN)
                        memcpy(o + m_metaDataSize, codes + (size_t)vid * M, M);
                }
                ErrorCode writeCode = ErrorCode::Success;
                if (fileDb) {
                    writeCode = fileDb->RewriteInPlace(hh, out, MaxTimeout, &(ws.m_diskRequests));
                    if (writeCode == ErrorCode::Success) {
                        m_postingSizes.UpdateSize(hh, out.size() / m_vectorInfoSize);
                        *m_checkSums[hh] = m_checkSum.CalcChecksum(out.c_str(), (int)out.size());
                    }
                } else {
                    writeCode = GetWritePosting(&ws, hh, out, true);
                }
                if (writeCode != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                        "[InpostPipePQ-DB] write posting %d fail code=%d\n", (int)hh, (int)writeCode);
                    munmap(codeMap, codeMapSize);
                    return;
                }
                totalRecs += n; slimHeads++;
                if (slimHeads % 100000 == 0)
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[InpostPipePQ-DB] transformed %zu heads\n", slimHeads);
            }
            Checkpoint(m_opt->m_indexDirectory);
            {
                std::ofstream mf(marker, std::ios::binary);
                int hdr[2] = { M, dstStride };
                mf.write((const char*)hdr, sizeof(hdr));
            }
            munmap(codeMap, codeMapSize);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "[InpostPipePQ-DB] DONE heads=%zu recs=%zu srcStride=%d dstStride=%d codeOffset=%zu "
                "(membership/order/pure-tail unchanged)\n",
                slimHeads, totalRecs, srcStride, dstStride, codeOffset);
        }

        // Tail-only rewrite for an already-built in-posting-code index. Keeps every
        // head's pure prefix unchanged, discards the old tail, generates new tail
        // candidates from the head index (Kmax from TailReplicaCount), and rewrites
        // postings in place. This is an experimental billion-scale path; use
        // SPTAG_TAIL_REWRITE_MAX_VECTORS for a small smoke run.
        void RewriteTailOnly(std::shared_ptr<VectorIndex> p_headIndex) {
            if (p_headIndex == nullptr || !m_hasPostingPureCounts) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "[TailRewrite] head index or posting_pure_counts missing\n");
                return;
            }
            const int kmax = m_opt->m_tailReplicaCount;
            if (kmax <= 0) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "[TailRewrite] TailReplicaCount=%d, nothing to rewrite\n", kmax);
                return;
            }
            const SizeType numHeads = m_postingSizes.GetPostingNum();
            const SizeType N = m_opt->m_vectorSize;
            const int recBytes = m_vectorInfoSize;
            auto recordsForPages = [&](int pages) -> int {
                return std::max(0, (pages * PageSize) / std::max(1, recBytes));
            };
            auto pagesForRecords = [&](int records) -> int {
                if (records <= 0) return 0;
                return (records * recBytes + PageSize - 1) / PageSize;
            };
            auto sparseTailLastPageKeep = [&](int pure, int keep) -> int {
                if (keep <= pure) return pure;
                const int totalBytes = keep * recBytes;
                const int totalPages = (totalBytes + PageSize - 1) / PageSize;
                if (totalPages <= 1) return keep;
                const int lastPageStart = (totalPages - 1) * PageSize;
                const int pureBytes = pure * recBytes;
                // Drop the final page only when it contains tail exclusively.
                if (pureBytes > lastPageStart) return keep;
                const int lastPageBytes = totalBytes - lastPageStart;
                if (lastPageBytes >= (PageSize + 9) / 10) return keep;
                return std::max(pure, lastPageStart / recBytes);
            };
            const int recordsPerPage = recordsForPages(1);
            const int extraTailPages = std::max(0, m_opt->m_unfilterTailBufferLength);
            SizeType maxVec = N;
            if (const char* e = std::getenv("SPTAG_TAIL_REWRITE_MAX_VECTORS")) {
                long long v = std::atoll(e);
                if (v > 0 && v < maxVec) maxVec = static_cast<SizeType>(v);
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "[TailRewrite] START N=%d maxVec=%d heads=%d recBytes=%d recordsPerPage=%d "
                "extraTailPages=%d cap=purePages+extra Kmax=%d\n",
                (int)N, (int)maxVec, (int)numHeads, recBytes, recordsPerPage,
                extraTailPages, kmax);

            auto fileDb = std::dynamic_pointer_cast<FileIO>(db);
            if (!fileDb) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                    "[TailRewrite] db is not FileIO; falling back to Put copy-on-write\n");
            }

            std::vector<std::uint8_t> recSeen(static_cast<size_t>(N), 0);
            std::vector<std::uint8_t> recByVid;
            try {
                recByVid.assign(static_cast<size_t>(N) * static_cast<size_t>(recBytes), 0);
            } catch (const std::bad_alloc&) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "[TailRewrite] failed to allocate recByVid (%zu bytes)\n",
                    static_cast<size_t>(N) * static_cast<size_t>(recBytes));
                return;
            }

            std::vector<std::uint8_t> pureDeg(static_cast<size_t>(N), 0);
            ExtraWorkSpace ws; InitWorkSpace(&ws);
            std::string blob;
            size_t pureRecords = 0;
            for (SizeType h = 0; h < numHeads; ++h) {
                int pure = m_postingPureCounts.GetSize(h);
                if (pure <= 0) continue;
                if (db->Get(h, &blob, MaxTimeout, &(ws.m_diskRequests)) != ErrorCode::Success) continue;
                int avail = static_cast<int>(blob.size() / static_cast<size_t>(recBytes));
                if (pure > avail) pure = avail;
                const char* p = blob.data();
                for (int i = 0; i < pure; ++i) {
                    const char* e = p + static_cast<size_t>(i) * recBytes;
                    SizeType vid = *reinterpret_cast<const SizeType*>(e);
                    if (vid < 0 || vid >= N) continue;
                    if (pureDeg[static_cast<size_t>(vid)] < 255) ++pureDeg[static_cast<size_t>(vid)];
                    if (!recSeen[static_cast<size_t>(vid)]) {
                        std::memcpy(recByVid.data() + static_cast<size_t>(vid) * recBytes, e, recBytes);
                        recSeen[static_cast<size_t>(vid)] = 1;
                    }
                    ++pureRecords;
                }
                if (h > 0 && (h % 1000000) == 0) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "[TailRewrite] pass1 counted pure heads=%d pureRecords=%zu\n", (int)h, pureRecords);
                }
            }

            std::vector<std::uint64_t> pureOff(static_cast<size_t>(N) + 1, 0);
            for (SizeType v = 0; v < N; ++v) {
                pureOff[static_cast<size_t>(v) + 1] = pureOff[static_cast<size_t>(v)] + pureDeg[static_cast<size_t>(v)];
            }
            std::vector<SizeType> pureHeads;
            try {
                pureHeads.assign(static_cast<size_t>(pureOff.back()), 0);
            } catch (const std::bad_alloc&) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "[TailRewrite] failed to allocate pureHeads (%zu entries)\n",
                    static_cast<size_t>(pureOff.back()));
                return;
            }
            std::vector<std::uint8_t> fillDeg(static_cast<size_t>(N), 0);
            for (SizeType h = 0; h < numHeads; ++h) {
                int pure = m_postingPureCounts.GetSize(h);
                if (pure <= 0) continue;
                if (db->Get(h, &blob, MaxTimeout, &(ws.m_diskRequests)) != ErrorCode::Success) continue;
                int avail = static_cast<int>(blob.size() / static_cast<size_t>(recBytes));
                if (pure > avail) pure = avail;
                const char* p = blob.data();
                for (int i = 0; i < pure; ++i) {
                    const char* e = p + static_cast<size_t>(i) * recBytes;
                    SizeType vid = *reinterpret_cast<const SizeType*>(e);
                    if (vid < 0 || vid >= N) continue;
                    std::uint8_t pos = fillDeg[static_cast<size_t>(vid)]++;
                    pureHeads[static_cast<size_t>(pureOff[static_cast<size_t>(vid)] + pos)] = h;
                }
                if (h > 0 && (h % 1000000) == 0) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "[TailRewrite] pass2 filled pure heads=%d\n", (int)h);
                }
            }
            std::vector<std::uint8_t>().swap(fillDeg);
            std::vector<std::uint8_t>().swap(pureDeg);

            auto isPureDup = [&](SizeType vid, SizeType h) -> bool {
                if (vid < 0 || vid >= N) return true;
                std::uint64_t b = pureOff[static_cast<size_t>(vid)];
                std::uint64_t e = pureOff[static_cast<size_t>(vid) + 1];
                for (std::uint64_t i = b; i < e; ++i) if (pureHeads[static_cast<size_t>(i)] == h) return true;
                return false;
            };
            auto genCapForHead = [&](SizeType h) -> int {
                int pure = m_postingPureCounts.GetSize(h);
                int purePages = pagesForRecords(pure);
                int capPages = purePages + extraTailPages;
                return std::max(0, recordsForPages(capPages) - pure);
            };

            int fd = open(m_opt->m_fullVectorFile.c_str(), O_RDONLY);
            if (fd < 0) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "[TailRewrite] open FullVectorFile %s failed\n", m_opt->m_fullVectorFile.c_str());
                return;
            }
            struct stat st {};
            if (fstat(fd, &st) != 0) { close(fd); return; }
            void* map = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_SHARED, fd, 0);
            close(fd);
            if (map == MAP_FAILED) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[TailRewrite] mmap full vectors failed\n");
                return;
            }
            const ValueType* base = reinterpret_cast<const ValueType*>(
                reinterpret_cast<const std::uint8_t*>(map) + 8);

            std::vector<std::atomic<int>> tailCounts(static_cast<size_t>(numHeads));
            for (auto& c : tailCounts) c.store(0, std::memory_order_relaxed);
            std::vector<Edge> tailPairs;
            std::mutex tailMutex;
            std::atomic<SizeType> cursor(0);
            std::atomic<size_t> generated(0), skippedDup(0), skippedCap(0), skippedNoRecord(0);
            int nThreads = std::max(1, m_opt->m_iSSDNumberOfThreads);
            auto worker = [&]() {
                COMMON::QueryResultSet<ValueType> heads(nullptr, kmax);
                std::vector<Edge> local;
                local.reserve(1 << 20);
                while (true) {
                    SizeType v = cursor.fetch_add(1);
                    if (v >= maxVec) break;
                    if (!recSeen[static_cast<size_t>(v)]) { ++skippedNoRecord; continue; }
                    const ValueType* vec = base + static_cast<size_t>(v) * m_opt->m_dim;
                    heads.SetTarget(vec, p_headIndex->m_pQuantizer);
                    heads.Reset();
                    if (p_headIndex->SearchIndex(heads) != ErrorCode::Success) continue;
                    BasicResult* res = heads.GetResults();
                    for (int r = 0; r < kmax; ++r) {
                        SizeType h = res[r].VID;
                        if (h < 0 || h >= numHeads) continue;
                        if (isPureDup(v, h)) { ++skippedDup; continue; }
                        int cap = genCapForHead(h);
                        if (cap <= 0) { ++skippedCap; continue; }
                        int old = tailCounts[static_cast<size_t>(h)].fetch_add(1, std::memory_order_relaxed);
                        if (old >= cap) { ++skippedCap; continue; }
                        Edge e;
                        e.node = h;
                        e.tonode = v;
                        e.distance = res[r].Dist;
                        local.push_back(e);
                        ++generated;
                        if (local.size() >= (1 << 20)) {
                            std::lock_guard<std::mutex> lk(tailMutex);
                            tailPairs.insert(tailPairs.end(), local.begin(), local.end());
                            local.clear();
                        }
                    }
                    if (v > 0 && (v % 10000000) == 0) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                            "[TailRewrite] generated up to vid=%d pairs=%zu dup=%zu cap=%zu noRecord=%zu\n",
                            (int)v, generated.load(), skippedDup.load(), skippedCap.load(), skippedNoRecord.load());
                    }
                }
                if (!local.empty()) {
                    std::lock_guard<std::mutex> lk(tailMutex);
                    tailPairs.insert(tailPairs.end(), local.begin(), local.end());
                }
            };
            std::vector<std::thread> threads;
            for (int t = 0; t < nThreads; ++t) threads.emplace_back(worker);
            for (auto& t : threads) t.join();
            munmap(map, static_cast<size_t>(st.st_size));

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "[TailRewrite] generation done pairs=%zu generated=%zu dup=%zu cap=%zu noRecord=%zu; sorting\n",
                tailPairs.size(), generated.load(), skippedDup.load(), skippedCap.load(), skippedNoRecord.load());
            std::sort(tailPairs.begin(), tailPairs.end(), EdgeCompare());

            auto finalKeepTail = [&](SizeType h, size_t tailAvail) -> size_t {
                int pure = m_postingPureCounts.GetSize(h);
                int purePages = pagesForRecords(pure);
                int capPages = purePages + extraTailPages;
                int hardCap = recordsForPages(capPages);
                int keep = pure + static_cast<int>(std::min<size_t>(tailAvail, static_cast<size_t>(std::max(0, hardCap - pure))));
                keep = sparseTailLastPageKeep(pure, keep);
                return static_cast<size_t>(std::max(0, keep - pure));
            };

            size_t pairPos = 0;
            size_t rewrittenHeads = 0, finalTail = 0, sparseTrim = 0;
            for (SizeType h = 0; h < numHeads; ++h) {
                int pure = m_postingPureCounts.GetSize(h);
                if (pure < 0) pure = 0;
                if (db->Get(h, &blob, MaxTimeout, &(ws.m_diskRequests)) != ErrorCode::Success) continue;
                int avail = static_cast<int>(blob.size() / static_cast<size_t>(recBytes));
                if (pure > avail) pure = avail;
                while (pairPos < tailPairs.size() && tailPairs[pairPos].node < h) ++pairPos;
                size_t begin = pairPos;
                while (pairPos < tailPairs.size() && tailPairs[pairPos].node == h) ++pairPos;
                size_t keepTail = finalKeepTail(h, pairPos - begin);
                if (keepTail < pairPos - begin) sparseTrim += (pairPos - begin - keepTail);
                std::string out;
                out.reserve(static_cast<size_t>(pure + keepTail) * recBytes);
                out.append(blob.data(), static_cast<size_t>(pure) * recBytes);
                for (size_t i = 0; i < keepTail; ++i) {
                    SizeType vid = tailPairs[begin + i].tonode;
                    out.append(reinterpret_cast<const char*>(
                        recByVid.data() + static_cast<size_t>(vid) * recBytes), recBytes);
                }
                ErrorCode code = ErrorCode::Success;
                if (fileDb) code = fileDb->RewriteInPlace(h, out, MaxTimeout, &(ws.m_diskRequests));
                else code = db->Put(h, out, MaxTimeout, &(ws.m_diskRequests));
                if (code != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                        "[TailRewrite] write head=%d failed code=%d\n", (int)h, (int)code);
                    return;
                }
                m_postingSizes.UpdateSize(h, static_cast<int>(out.size() / recBytes));
                *m_checkSums[h] = m_checkSum.CalcChecksum(out.c_str(), static_cast<int>(out.size()));
                finalTail += keepTail;
                ++rewrittenHeads;
                if (h > 0 && (h % 1000000) == 0) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "[TailRewrite] rewritten heads=%d finalTail=%zu sparseTrim=%zu\n",
                        (int)h, finalTail, sparseTrim);
                }
            }
            Checkpoint(m_opt->m_indexDirectory);
            std::ofstream marker(m_opt->m_indexDirectory + FolderSep + "tail_rewrite_pagebudget.done", std::ios::binary);
            int hdr[4] = { kmax, recBytes, recordsPerPage, extraTailPages };
            marker.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
            marker.close();
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "[TailRewrite] DONE rewrittenHeads=%zu finalTail=%zu sparseTrim=%zu\n",
                rewrittenHeads, finalTail, sparseTrim);
        }

        void ExportOPQSidecars() {
            std::string dir = m_opt->m_indexDirectory + FolderSep;
            auto qio = SPTAG::f_createIO();
            std::string qpath = dir + "opq_quantizer.bin";
            if (!qio || !qio->Initialize(qpath.c_str(), std::ios::binary | std::ios::in)) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[OPQ export] cannot open %s\n", qpath.c_str());
                return;
            }
            auto q = COMMON::IQuantizer::LoadIQuantizer(qio);
            if (!q) { SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[OPQ export] load quantizer failed\n"); return; }
            q->SetEnableADC(false);
            int M = q->GetNumSubvectors();
            int recSize = m_metaDataSize;
            // In-posting-code mode: emit slim records as [meta | M-byte OPQ code] so the
            // search-time ADC screen reads the code inline (no resident codes array).
            const char* icx = std::getenv("SPTAG_OPQ_INPOST_CODE");
            bool inpostCode = (icx && icx[0] == '1');
            int vecBytes = m_opt->m_dim * (int)sizeof(ValueType);
            SizeType N = m_opt->m_vectorSize;
            SizeType postingNum = m_postingSizes.GetPostingNum();

            std::vector<std::uint8_t> codes((size_t)N * M, 0);
            std::vector<char> seen(N, 0);
            std::vector<std::uint64_t> idx((size_t)postingNum + 1, 0);
            std::string slim; slim.reserve((size_t)1 << 26);
            std::unordered_map<uint32_t, std::vector<int>> tagVids;   // tag value -> vids (exhaustive)

            // Canonical vid -> vector store. Prefer a KV store (RocksDB) when compiled
            // in; otherwise fall back to a single mmap'd point store (.bin). Either way
            // there is exactly ONE vector copy, fetched by vid (vector-level IO).
            std::shared_ptr<Helper::KeyValueIO> vecDB;
#ifdef ROCKSDB
            vecDB.reset(new RocksDBIO((dir + "opq_vecstore").c_str(), false, false, false));
#endif
            const bool useKV = (vecDB != nullptr);
            std::vector<ValueType> pointstore;
            if (!useKV) pointstore.assign((size_t)N * m_opt->m_dim, (ValueType)0);
            size_t vecPuts = 0;

            ExtraWorkSpace ws; InitWorkSpace(&ws);
            std::string blob;
            std::uint64_t off = 0;
            for (SizeType h = 0; h < postingNum; h++) {
                idx[h] = off;
                int sz = m_postingSizes.GetSize(h);
                if (sz <= 0) continue;
                if (db->Get(h, &blob, MaxTimeout, &(ws.m_diskRequests)) != ErrorCode::Success) continue;
                int n = (int)(blob.size() / m_vectorInfoSize);
                const char* p = blob.data();
                for (int i = 0; i < n; i++) {
                    const char* e = p + (size_t)i * m_vectorInfoSize;
                    int vid = *(reinterpret_cast<const int*>(e));
                    slim.append(e, recSize);
                    off += recSize;
                    if (vid >= 0 && vid < N && !seen[vid]) {
                        seen[vid] = 1;
                        const ValueType* v = reinterpret_cast<const ValueType*>(e + m_metaDataSize);
                        if (useKV) {
                            vecDB->Put((SizeType)vid, std::string((const char*)v, vecBytes), MaxTimeout, nullptr);
                            vecPuts++;
                        } else {
                            memcpy(&pointstore[(size_t)vid * m_opt->m_dim], v, vecBytes);
                        }
                        // OPQ quantizer is float-typed; widen uint8 vectors to float so
                        // QuantizeVector (which reads its input as float*) encodes correctly.
                        std::vector<float> vf(m_opt->m_dim);
                        for (int d = 0; d < m_opt->m_dim; d++) vf[d] = (float)v[d];
                        q->QuantizeVector(vf.data(), &codes[(size_t)vid * M], false);
                        const uint32_t* vt = reinterpret_cast<const uint32_t*>(e + sizeof(int) + sizeof(uint8_t));
                        for (int t = 0; t < m_numTagsPerVec; t++) tagVids[vt[t]].push_back(vid);
                    }
                    // Append the inline OPQ code after the meta prefix. The code is
                    // computed on a vid's first sight (above); repeats reference the
                    // already-filled codes[] (first sight always precedes repeats).
                    if (inpostCode && vid >= 0 && vid < N) {
                        slim.append((const char*)&codes[(size_t)vid * M], M);
                        off += M;
                    }
                }
            }
            idx[postingNum] = off;

            auto wbin = [&](const std::string& name, const void* data, size_t bytes) {
                std::ofstream o(dir + name, std::ios::binary);
                o.write((const char*)data, bytes);
                o.close();
            };
            wbin("opq_codes.bin", codes.data(), (size_t)N * M);
            wbin("opq_slim.bin", slim.data(), slim.size());
            wbin("opq_slim.idx", idx.data(), idx.size() * sizeof(std::uint64_t));
            if (!useKV) wbin("opq_pointstore.bin", pointstore.data(), (size_t)N * m_opt->m_dim * sizeof(ValueType));
            {
                // opq_tagpure.bin: exhaustive tag -> vids map (for narrow tag-pure path)
                std::ofstream o(dir + "opq_tagpure.bin", std::ios::binary);
                int numTags = (int)tagVids.size();
                o.write((const char*)&numTags, sizeof(int));
                size_t totalVids = 0;
                for (auto& kv : tagVids) {
                    uint32_t tagVal = kv.first;
                    int cnt = (int)kv.second.size();
                    o.write((const char*)&tagVal, sizeof(uint32_t));
                    o.write((const char*)&cnt, sizeof(int));
                    o.write((const char*)kv.second.data(), (size_t)cnt * sizeof(int));
                    totalVids += cnt;
                }
                o.close();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                    "[OPQ export] tagpure tags=%d totalVids=%zu\n", numTags, totalVids);
            }
            size_t seenCount = 0; for (SizeType i = 0; i < N; i++) seenCount += seen[i];
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "[OPQ export] N=%d seen=%zu M=%d heads=%d slimBytes=%zu vecStore=%s(puts=%zu) codes=%.1fMB\n",
                (int)N, seenCount, M, (int)postingNum, slim.size(),
                useKV ? "RocksDB" : "mmap", vecPuts, (double)N * M / 1e6);
        }

        // ===================================================================
        // RaBitQ prefilter helpers (offline dump + sidecar load + query estimate)
        // ===================================================================
        void DumpVectorsForRaBitQ(const std::string& dir) {
            if (!m_opqVecDB) { SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[RaBitQ dump] no vecstore\n"); return; }
            int dim = m_opt->m_dim;
            size_t vecBytes = (size_t)dim * sizeof(ValueType);
            std::ofstream o(dir + "opq_vectors.bin", std::ios::binary);
            if (!o) { SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[RaBitQ dump] open out failed\n"); return; }
            int hdr[2] = { (int)m_opqN, dim };
            o.write((const char*)hdr, sizeof(hdr));
            std::string val;
            std::vector<Helper::AsyncReadRequest> reqs;
            std::vector<ValueType> zero(dim, (ValueType)0);
            size_t miss = 0;
            for (SizeType vid = 0; vid < m_opqN; vid++) {
                val.clear();
                if (m_opqVecDB->Get(vid, &val, MaxTimeout, &reqs) == ErrorCode::Success && val.size() >= vecBytes)
                    o.write(val.data(), vecBytes);
                else { o.write((const char*)zero.data(), vecBytes); miss++; }
            }
            o.close();
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "[RaBitQ dump] wrote opq_vectors.bin N=%d dim=%d miss=%zu\n", (int)m_opqN, dim, miss);
        }

        void LoadRaBitQ(const std::string& dir) {
            const char* e = std::getenv("SPTAG_RABITQ");
            if (!(e && e[0] == '1')) return;
            std::ifstream in(dir + "rabitq.bin", std::ios::binary);
            if (!in) { SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[RaBitQ] rabitq.bin not found; staying on PQ\n"); return; }
            int hdr[3] = { 0, 0, 0 };
            in.read((char*)hdr, sizeof(hdr));
            int N = hdr[0], pdim = hdr[1], bits = hdr[2];
            if (N != (int)m_opqN) { SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[RaBitQ] N mismatch %d vs %d\n", N, (int)m_opqN); return; }
            m_rbqDim = pdim; m_rbqBits = bits;
            m_rbqRot.resize((size_t)m_opt->m_dim * pdim);
            in.read((char*)m_rbqRot.data(), m_rbqRot.size() * sizeof(float));
            m_rbqCodes.resize((size_t)N * pdim);
            m_rbqDelta.resize(N); m_rbqVl.resize(N);
            for (int i = 0; i < N; i++) {
                in.read((char*)&m_rbqCodes[(size_t)i * pdim], pdim);
                in.read((char*)&m_rbqDelta[i], sizeof(float));
                in.read((char*)&m_rbqVl[i], sizeof(float));
            }
            if (!in) { SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[RaBitQ] read truncated\n"); m_rbqRot.clear(); m_rbqCodes.clear(); return; }
            m_rbq = true;
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "[RaBitQ] ENABLED N=%d dim=%d bits=%d codeMB=%.1f\n", N, pdim, bits, (double)N * pdim / 1e6);
        }

        // Widen the index ValueType query (e.g. uint8) to float[dim]. The OPQ/RaBitQ
        // quantizers are float-typed; passing the raw ValueType buffer reinterpreted as
        // float* yields a garbage estimator (near-random screen). Single source of truth
        // for every search fork's query preparation -- do NOT re-inline this loop.
        std::vector<float> WidenQuery(const ValueType* rawQuery, int dim) const {
            std::vector<float> out(dim);
            for (int d = 0; d < dim; d++) out[d] = (float)rawQuery[d];
            return out;
        }

        // Unit-normalize (cosine) the query, then rotate into RaBitQ space: rq = qn * R.
        void RaBitQRotateQuery(const ValueType* rawQuery, std::vector<float>& rq) const {
            int dim = m_opt->m_dim, pdim = m_rbqDim;
            std::vector<float> qn = WidenQuery(rawQuery, dim);
            if (m_opt->m_distCalcMethod == DistCalcMethod::Cosine) {
                double nrm = 0; for (int i = 0; i < dim; i++) nrm += (double)qn[i] * qn[i];
                nrm = std::sqrt(nrm); if (nrm > 0) { float inv = (float)(1.0 / nrm); for (int i = 0; i < dim; i++) qn[i] *= inv; }
            }
            rq.assign(pdim, 0.f);
            for (int i = 0; i < dim; i++) {
                float qi = qn[i];
                const float* Ri = &m_rbqRot[(size_t)i * pdim];
                for (int j = 0; j < pdim; j++) rq[j] += qi * Ri[j];
            }
        }

        // L2 distance in rotated space between query and the reconstructed code of vid.
        // recon[j] = code[j]*delta + vl ; dist = sum_j (rq[j] - recon[j])^2.
#if defined(__x86_64__) || defined(_M_X64)
        __attribute__((target("avx2,fma")))
        static float RaBitQL2AVX2(const float* rq, const std::uint8_t* code, int pdim, float delta, float vl) {
            __m256 vdelta = _mm256_set1_ps(delta);
            __m256 vvl = _mm256_set1_ps(vl);
            __m256 acc = _mm256_setzero_ps();
            int j = 0;
            for (; j + 8 <= pdim; j += 8) {
                __m128i c8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(code + j));
                __m256 cf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(c8));
                __m256 recon = _mm256_fmadd_ps(cf, vdelta, vvl);
                __m256 d = _mm256_sub_ps(_mm256_loadu_ps(rq + j), recon);
                acc = _mm256_fmadd_ps(d, d, acc);
            }
            __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
            s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
            float total = _mm_cvtss_f32(s);
            for (; j < pdim; j++) { float r = (float)code[j] * delta + vl; float d = rq[j] - r; total += d * d; }
            return total;
        }
#endif

        inline float RaBitQDist(const std::vector<float>& rq, int vid) const {
            int pdim = m_rbqDim;
            const std::uint8_t* c = &m_rbqCodes[(size_t)vid * pdim];
            float delta = m_rbqDelta[vid], vl = m_rbqVl[vid];
#if defined(__x86_64__) || defined(_M_X64)
            static const bool s_avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
            if (s_avx2) return RaBitQL2AVX2(rq.data(), c, pdim, delta, vl);
#endif
            float s = 0;
            for (int j = 0; j < pdim; j++) {
                float recon = (float)c[j] * delta + vl;
                float d = rq[j] - recon;
                s += d * d;
            }
            return s;
        }

        bool MmapRO(const std::string& path, const std::uint8_t*& ptr, size_t& bytes) {
#ifndef _MSC_VER
            int fd = open(path.c_str(), O_RDONLY);
            if (fd < 0) return false;
            struct stat st; if (fstat(fd, &st) != 0) { close(fd); return false; }
            bytes = (size_t)st.st_size;
            void* m = mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
            close(fd);
            if (m == MAP_FAILED) return false;
            ptr = (const std::uint8_t*)m;
            return true;
#else
            return false;
#endif
        }

        void LoadOPQPrefilter() {
            std::string dir = m_opt->m_indexDirectory + FolderSep;
            if (m_pipePQ) {
                m_pipePQTable.reset(new PipePQTable());
                if (!m_pipePQTable->Load(m_pipePQPivotsPathResolved, m_opqInpostDbM)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                        "[PipePQ prefilter] cannot load pivots %s\n", m_pipePQPivotsPathResolved.c_str());
                    return;
                }
                m_opqM = m_pipePQTable->Chunks();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                    "[PipePQ prefilter] loaded pivots=%s dim=%d chunks=%d\n",
                    m_pipePQPivotsPathResolved.c_str(), m_pipePQTable->Dim(), m_opqM);
            } else {
                auto qio = SPTAG::f_createIO();
                if (!qio || !qio->Initialize((dir + "opq_quantizer.bin").c_str(), std::ios::binary | std::ios::in)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[OPQ prefilter] cannot open quantizer\n"); return;
                }
                m_opqQ = COMMON::IQuantizer::LoadIQuantizer(qio);
                if (!m_opqQ) { SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[OPQ prefilter] load quantizer failed\n"); return; }
                m_opqQ->SetEnableADC(true);
                m_opqM = m_opqQ->GetNumSubvectors();
            }
            m_opqKs = 256;
            m_opqN = m_opt->m_vectorSize;
            {
                const char* ic = std::getenv("SPTAG_OPQ_INPOST_CODE");
                m_opqInpostCode = (ic && ic[0] == '1') || m_opqInpostDb;
            }
            // In-posting-code mode: the slim record is [meta | M-byte OPQ code]; the
            // ADC screen reads the code inline (no resident m_opqCodes array). Default
            // (resident) keeps the slim record meta-only with codes in opq_codes.bin.
            m_slimRec = m_metaDataSize + (m_opqInpostCode ? m_opqM : 0);
            // Slim-postings update mode: the posting store on disk holds only the
            // metadata prefix and the single vector copy lives in the vector store.
            // This requires the vector store to be writable so inserts/splits can
            // push new/relocated vectors into it.
            {
                const char* sp = std::getenv("SPTAG_SLIM_POSTINGS");
                m_slimPostings = (sp && sp[0] == '1');
                const char* st = std::getenv("SPTAG_SLIM_SELFTEST");
                m_slimSelfTest = (st && st[0] == '1');
            }

            const std::uint8_t* p; size_t b;
#ifdef ROCKSDB
            {
                struct stat st;
                if (stat((dir + "opq_vecstore").c_str(), &st) == 0) {
                    // Slim-update mode wants a writable vector store so the decorator can
                    // push new/relocated vectors on insert/split. But there can be several
                    // ExtraDynamicSearcher instances per tenant (e.g. main + pivot), and
                    // RocksDB allows only ONE read-write (exclusive-lock) open per process.
                    // So: try RW first; if the lock is already held, fall back to a
                    // read-only open (no lock) so this instance can still serve OPQ search.
                    // The single instance that wins RW owns the update path.
                    if (m_slimPostings) {
                        m_opqVecDB.reset(new RocksDBIO((dir + "opq_vecstore").c_str(), false, false, false, false));
                        if (m_opqVecDB->Available()) {
                            m_slimWritable = true;
                        } else {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                "[slim postings] vecstore RW open unavailable (lock held by another instance); opening read-only for search.\n");
                            m_opqVecDB.reset(new RocksDBIO((dir + "opq_vecstore").c_str(), false, false, false, true));
                            if (!m_opqVecDB->Available()) m_opqVecDB.reset();
                        }
                    } else {
                        // Query path is read-only: OpenForReadOnly takes no exclusive LOCK,
                        // allows concurrent opens within the process.
                        m_opqVecDB.reset(new RocksDBIO((dir + "opq_vecstore").c_str(), false, false, false, true));
                        if (!m_opqVecDB->Available()) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                "[OPQ prefilter] vecstore open failed; falling back to pointstore\n");
                            m_opqVecDB.reset();
                        }
                    }
                }
            }
#endif
            if (!m_opqVecDB) {
                // Fallback: single mmap'd point store (one vector copy, vid-indexed).
                // In-posting-DB build-native mode reranks survivors from the FullVectorFile
                // O_DIRECT base (m_inpostBaseFd) instead, so neither a vecstore nor a
                // pointstore is required when that base is open.
                if (!MmapRO(dir + "opq_pointstore.bin", p, b)) {
                    if (m_opqInpostDb && m_inpostBaseFd >= 0) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                            "[OPQ prefilter] no vecstore/pointstore; rerank via FullVectorFile O_DIRECT base\n");
                    } else {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[OPQ prefilter] no vector store (vecstore/pointstore) found\n"); return;
                    }
                } else {
                    m_psVec = (const ValueType*)p;
                }
            }
            if (!m_opqInpostDb) {
                // Resident / mmap-slim modes read the slim postings from the opq_slim.bin
                // file. In-posting-DB mode keeps the slim [meta|code] records IN the
                // posting store db (read via async MultiGet at search time), so there is
                // no opq_slim.bin to map.
                if (!MmapRO(dir + "opq_slim.bin", m_slim, b)) { SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[OPQ prefilter] mmap slim failed\n"); return; }
                if (!MmapRO(dir + "opq_slim.idx", p, b)) { SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[OPQ prefilter] mmap slim.idx failed\n"); return; }
                m_slimOff = (const std::uint64_t*)p;
            }

            // Fair device-bound mode: read slim postings via O_DIRECT (bypass OS page
            // cache) so each posting scan pays real device IO, matching the cache-disabled
            // comparison model (mirrors SPTAG_ROCKSDB_DIRECT_IO for the vector store).
            // Default off => the mmap fast path (page-cache resident) is used.
            {
                const char* sd = std::getenv("SPTAG_SLIM_DIRECT_IO");
                if (sd && sd[0] == '1') {
#ifdef O_DIRECT
                    int fd = open((dir + "opq_slim.bin").c_str(), O_RDONLY | O_DIRECT);
#else
                    int fd = open((dir + "opq_slim.bin").c_str(), O_RDONLY);
#endif
                    if (fd < 0) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[OPQ prefilter] O_DIRECT slim open failed; falling back to mmap\n");
                    } else {
                        m_slimDirectFd = fd;
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[OPQ prefilter] slim postings: O_DIRECT device-bound reads ENABLED\n");
                    }
                }
            }

            if (!m_opqInpostCode) {
                m_opqCodes.resize((size_t)m_opqN * m_opqM);
                std::ifstream in(dir + "opq_codes.bin", std::ios::binary);
                if (!in) { SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[OPQ prefilter] open codes failed\n"); return; }
                in.read((char*)m_opqCodes.data(), m_opqCodes.size());
            } else {
                m_opqCodes.clear(); m_opqCodes.shrink_to_fit();   // codes live inline in slim records
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                    "[OPQ prefilter] in-posting code mode: slimRec=%d (meta=%d + code=%d), zero resident codes\n",
                    m_slimRec, m_metaDataSize, m_opqM);
            }
            // RerankL is a native index parameter and the sole source of truth for
            // the in-posting survivor count.
            if (m_opqInpostDb && m_opt->m_rerankL > 0) m_opqL = m_opt->m_rerankL;
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "[OPQ prefilter] rerank survivors L=%d (RerankL from index config)\n",
                m_opqL);
            m_opqPF = true;
            // Optional: dump vid-ordered raw vectors (for offline RaBitQ encoding), then
            // optionally load a RaBitQ sidecar to replace the PQ ADC scan at search time.
            if (const char* d = std::getenv("SPTAG_RABITQ_DUMP")) { if (d[0] == '1') DumpVectorsForRaBitQ(dir); }
            LoadRaBitQ(dir);
            // Real extended RaBitQ (rabitq2.bin): packed 1-bit + ex-bits estimator with
            // exact rerank. Takes precedence over the SQ-style m_rbq path when present
            // (and SPTAG_RABITQ2 != 0).
            {
                const char* r2 = std::getenv("SPTAG_RABITQ2");
                bool want2 = !(r2 && r2[0] == '0');
                if (want2) {
                    auto store = std::make_unique<RaBitQ2>();
                    if (store->Load(dir + "rabitq2.bin")) {
                        if (store->GetN() != (int)m_opqN) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                "[RaBitQ2] N mismatch %d vs %d; ignoring rabitq2.bin\n",
                                store->GetN(), (int)m_opqN);
                        } else {
                            m_rbq2 = std::move(store);
                            m_rbq2on = true;
                            m_rbq = true;  // reuse the RaBitQ screen+rerank plumbing
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                "[RaBitQ2] ENABLED N=%d dim=%d ex_bits=%d (total_bits=%d)\n",
                                m_rbq2->GetN(), m_rbq2->GetDim(), m_rbq2->GetExBits(),
                                m_rbq2->GetExBits() + 1);
                        }
                    }
                }
            }
            // optional exhaustive tag->vids map for the narrow tag-pure path
            {
                std::ifstream tin(dir + "opq_tagpure.bin", std::ios::binary);
                if (tin) {
                    int numTags = 0;
                    tin.read((char*)&numTags, sizeof(int));
                    size_t totalVids = 0;
                    for (int t = 0; t < numTags && tin; t++) {
                        uint32_t tagVal = 0; int cnt = 0;
                        tin.read((char*)&tagVal, sizeof(uint32_t));
                        tin.read((char*)&cnt, sizeof(int));
                        if (cnt < 0) break;
                        std::vector<int>& v = m_opqTagVids[tagVal];
                        v.resize(cnt);
                        if (cnt) tin.read((char*)v.data(), (size_t)cnt * sizeof(int));
                        totalVids += cnt;
                    }
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "[OPQ prefilter] tagpure map tags=%zu totalVids=%zu\n",
                        m_opqTagVids.size(), totalVids);
                }
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[OPQ prefilter] ENABLED N=%d M=%d Ks=%d L=%d\n",
                (int)m_opqN, m_opqM, m_opqKs, m_opqL);

            // Install the slim-posting decorator: from here on every db->Get/Put/Merge/
            // MultiGet on the posting store transparently splits the vector out to (or
            // splices it back from) the single canonical vector store. Existing
            // maintenance code (Split/Append/Reassign/AddIndex) keeps operating on full
            // in-memory records and needs no change.
            if (m_slimPostings) {
                if (!m_opqVecDB || !m_opqVecDB->Available()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                        "[slim postings] no vector store available; slim mode disabled\n");
                    m_slimPostings = false;
                } else {
                    // One-time migration: the existing posting store holds full records
                    // (vector inline). Rewrite each posting to its slim metadata prefix;
                    // the canonical vectors already live in the vector store (exported by
                    // ExportOPQSidecars). A marker file makes this idempotent across loads
                    // and instances.
                    MigratePostingsToSlim();
                    // Wrap on every instance so posting reads inflate correctly. Only the
                    // instance holding the RW vector store (m_slimWritable) can serve the
                    // update path; read-only instances still inflate on read for search.
                    db = std::make_shared<SlimVectorKV>(db, m_opqVecDB, m_metaDataSize, m_vectorInfoSize);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "[slim postings] ENABLED (%s): posting store wrapped (metaSize=%d recSize=%d vecBytes=%d)\n",
                        m_slimWritable ? "writable" : "read-only",
                        m_metaDataSize, m_vectorInfoSize, m_vectorInfoSize - m_metaDataSize);
                    // On the writable instance, load a second quantizer in encode mode
                    // (ADC disabled) so inserts can compute PQ codes for new vids without
                    // disturbing the query-LUT quantizer (m_opqQ, ADC enabled). This makes
                    // brand-new inserts visible to the OPQ tag-pure search path.
                    if (m_slimWritable) {
                        auto eio = SPTAG::f_createIO();
                        if (eio && eio->Initialize((dir + "opq_quantizer.bin").c_str(), std::ios::binary | std::ios::in)) {
                            m_opqQEnc = COMMON::IQuantizer::LoadIQuantizer(eio);
                            if (m_opqQEnc) {
                                m_opqQEnc->SetEnableADC(false);
                                m_opqDynamic = true;
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                                    "[slim postings] incremental OPQ maintenance ENABLED (encoder loaded)\n");
                            }
                        }
                        if (!m_opqQEnc)
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                                "[slim postings] encoder load failed; inserts will not be OPQ-visible\n");
                    }
                    if (m_slimSelfTest && m_slimWritable) SlimSelfTest();
                    {
                        const char* it = std::getenv("SPTAG_SLIM_INSERT_SELFTEST");
                        if (it && it[0] == '1' && m_slimWritable) OPQInsertSelfTest();
                    }
                }
            }
        }

        // Rewrite every full posting (vector inline) into its slim metadata prefix
        // ([vid|version|tag] per vector). Idempotent: guarded by a marker file and a
        // per-head size check, so re-running on an already-slim store is a no-op.
        void MigratePostingsToSlim() {
            std::string marker = m_opt->m_indexDirectory + FolderSep + "opq_slim_postings.done";
            std::string compactMarker = m_opt->m_indexDirectory + FolderSep + "opq_slim_compacted.done";
            if (fileexists(marker.c_str())) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                    "[slim postings] already migrated (marker present); skipping.\n");
                // Already-migrated stores may still hold obsolete full records as RocksDB
                // garbage if a prior run predated the compaction step. Compact once.
                if (m_slimWritable && !fileexists(compactMarker.c_str())) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[slim postings] forcing one-time posting-store compaction\n");
                    db->ForceCompaction();
                    FILE* cf = fopen(compactMarker.c_str(), "wb");
                    if (cf) { fputc('1', cf); fclose(cf); }
                }
                return;
            }
            ExtraWorkSpace ws; InitWorkSpace(&ws);
            SizeType postingNum = m_postingSizes.GetPostingNum();
            std::string full, slim;
            size_t migrated = 0, alreadySlim = 0;
            size_t fullBytes = 0, slimBytes = 0;
            for (SizeType h = 0; h < postingNum; h++) {
                int cnt = m_postingSizes.GetSize(h);
                if (cnt <= 0) continue;
                if (db->Get(h, &full, MaxTimeout, &(ws.m_diskRequests)) != ErrorCode::Success) continue;
                size_t expectFull = (size_t)cnt * m_vectorInfoSize;
                size_t expectSlim = (size_t)cnt * m_metaDataSize;
                if (full.size() == expectSlim) { alreadySlim++; continue; }
                if (full.size() != expectFull) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                        "[slim postings] head %d unexpected size %zu (full=%zu slim=%zu); skipping.\n",
                        h, full.size(), expectFull, expectSlim);
                    continue;
                }
                // Snapshot a few original full postings for the decorator self-test.
                if (m_slimSelfTest && m_slimSelfTestSnapshot.size() < 8) {
                    m_slimSelfTestSnapshot[h] = full;
                }
                slim.clear();
                slim.reserve(expectSlim);
                const char* p = full.data();
                for (int i = 0; i < cnt; i++) {
                    slim.append(p + (size_t)i * m_vectorInfoSize, (size_t)m_metaDataSize);
                }
                if (db->Put(h, slim, MaxTimeout, &(ws.m_diskRequests)) != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                        "[slim postings] failed to write slim posting for head %d\n", h);
                    continue;
                }
                fullBytes += full.size();
                slimBytes += slim.size();
                migrated++;
            }
            FILE* fp = fopen(marker.c_str(), "wb");
            if (fp) { fputc('1', fp); fclose(fp); }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "[slim postings] migration done: heads migrated=%zu alreadySlim=%zu fullBytes=%zu slimBytes=%zu (%.1fx)\n",
                migrated, alreadySlim, fullBytes, slimBytes,
                slimBytes ? (double)fullBytes / slimBytes : 0.0);
            // The migration overwrote each posting key in place; on a RocksDB posting
            // store (Storage=ROCKSDBIO) the obsolete full records linger as garbage until
            // compaction. Force a compaction now so the slim shrink is physically realized
            // (FileIO block stores ignore this — ForceCompaction is a no-op there).
            if (migrated > 0) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[slim postings] forcing posting-store compaction to reclaim space\n");
                db->ForceCompaction();
                FILE* cf = fopen(compactMarker.c_str(), "wb");
                if (cf) { fputc('1', cf); fclose(cf); }
            }
        }

        // In-process verification of the slim decorator. Exercises both the read
        // round-trip (inflate equals the original full posting) and the write
        // round-trip (a synthetic appended record whose vector exists only because
        // the decorator's deflate pushed it to the vector store, then is read back
        // byte-identically). Gated by SPTAG_SLIM_SELFTEST=1.
        void SlimSelfTest() {
            ExtraWorkSpace ws; InitWorkSpace(&ws);
            int readPass = 0, readFail = 0;
            for (auto& kv : m_slimSelfTestSnapshot) {
                std::string got;
                if (db->Get(kv.first, &got, MaxTimeout, &(ws.m_diskRequests)) != ErrorCode::Success) {
                    readFail++; continue;
                }
                if (got == kv.second) readPass++;
                else {
                    readFail++;
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                        "[slim selftest] READ mismatch head %d: got=%zu orig=%zu\n",
                        kv.first, got.size(), kv.second.size());
                }
            }

            bool writeOk = false, restoreOk = false;
            SizeType h = -1; std::string orig;
            if (!m_slimSelfTestSnapshot.empty()) {
                h = m_slimSelfTestSnapshot.begin()->first;
                orig = m_slimSelfTestSnapshot.begin()->second;
            } else {
                for (SizeType x = 0; x < m_postingSizes.GetPostingNum(); x++) {
                    if (m_postingSizes.GetSize(x) > 0) {
                        h = x; db->Get(x, &orig, MaxTimeout, &(ws.m_diskRequests)); break;
                    }
                }
            }
            SizeType synVid = m_opqN + 12345;  // synthetic vid, unused by the index
            if (h >= 0 && orig.size() >= (size_t)m_vectorInfoSize) {
                std::string rec(m_vectorInfoSize, '\0');
                memcpy(&rec[0], orig.data(), (size_t)m_metaDataSize);   // reuse meta (ver|tag) from rec0
                memcpy(&rec[0], &synVid, sizeof(SizeType));             // override vid
                int dim = m_opt->m_dim;
                std::vector<ValueType> pat(dim);
                for (int d = 0; d < dim; d++) pat[d] = (ValueType)((d % 7) + 1);
                memcpy(&rec[m_metaDataSize], pat.data(), (size_t)dim * sizeof(ValueType));
                std::string mod = orig + rec;  // append one synthetic record
                if (db->Put(h, mod, MaxTimeout, &(ws.m_diskRequests)) == ErrorCode::Success) {
                    std::string back;
                    if (db->Get(h, &back, MaxTimeout, &(ws.m_diskRequests)) == ErrorCode::Success && back == mod)
                        writeOk = true;
                }
                // restore the original posting and remove the synthetic vector
                if (db->Put(h, orig, MaxTimeout, &(ws.m_diskRequests)) == ErrorCode::Success) {
                    std::string back2;
                    if (db->Get(h, &back2, MaxTimeout, &(ws.m_diskRequests)) == ErrorCode::Success && back2 == orig)
                        restoreOk = true;
                }
                if (m_opqVecDB) m_opqVecDB->Delete(synVid);
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "[slim selftest] READ pass=%d fail=%d | WRITE append-roundtrip=%s restore=%s\n",
                readPass, readFail, writeOk ? "PASS" : "FAIL", restoreOk ? "PASS" : "FAIL");
        }

        // Make a brand-new inserted vector visible to the OPQ tag-pure search path.
        // The canonical vector is already written to the vector store by the SlimVectorKV
        // decorator on the posting Append (deflate-on-write). Here we maintain the resident
        // OPQ state: compute the new vid's PQ code, grow the code-coverage bound (m_opqN),
        // and register the vid under each of its tags. Gated by m_opqDynamic (writable
        // instance with an encoder). Called from AddIndex per inserted vid.
        void OPQInsertMaintain(SizeType vid, const ValueType* vec, const uint32_t* tags, int numTags) {
            if (!m_opqDynamic || !m_opqQEnc || vid < 0 || m_opqM <= 0) return;

            // Encode outside the lock (codebook is fixed/immutable).
            std::vector<ValueType> qn(vec, vec + m_opt->m_dim);
            if (m_opt->m_distCalcMethod == DistCalcMethod::Cosine)
                COMMON::Utils::Normalize<ValueType>(qn.data(), m_opt->m_dim, COMMON::Utils::GetBase<ValueType>());
            std::vector<std::uint8_t> code(m_opqM);
            m_opqQEnc->QuantizeVector(qn.data(), code.data(), false);

            std::unique_lock<std::shared_timed_mutex> wlock(m_opqMaintLock);
            size_t need = (size_t)(vid + 1) * m_opqM;
            if (m_opqCodes.size() < need) m_opqCodes.resize(need, 0);
            memcpy(&m_opqCodes[(size_t)vid * m_opqM], code.data(), m_opqM);
            if (vid + 1 > m_opqN) m_opqN = vid + 1;
            for (int t = 0; t < numTags; t++)
                m_opqTagVids[tags[t]].push_back((int)vid);
            m_opqMutated.store(true, std::memory_order_release);
        }

        // Verify brand-new-insert visibility on the OPQ tag-pure path WITHOUT mutating the
        // index: register a synthetic vid (= next slot) under a fresh unused tag, place its
        // vector in the vector store, then run OPQTagPureSearch for that tag and confirm the
        // synthetic vid is returned. Rolls back all state. Gated SPTAG_SLIM_INSERT_SELFTEST=1.
        void OPQInsertSelfTest() {
            if (!m_opqDynamic) { SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "[insert selftest] not dynamic; skipped\n"); return; }
            // borrow an existing vector from the vector store
            int srcVid = -1;
            for (auto& kv : m_opqTagVids) { if (!kv.second.empty()) { srcVid = kv.second[0]; break; } }
            if (srcVid < 0) { SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "[insert selftest] no source vid; skipped\n"); return; }
            std::vector<int> one{ srcVid };
            std::vector<std::string> vals; std::vector<Helper::AsyncReadRequest> reqs;
            if (m_opqVecDB->MultiGet(one, &vals, MaxTimeout, &reqs) != ErrorCode::Success || vals.empty()
                || vals[0].size() < (size_t)m_opt->m_dim * sizeof(ValueType)) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "[insert selftest] source fetch failed; skipped\n"); return;
            }
            int dim = m_opt->m_dim;
            std::vector<ValueType> vec(dim);
            memcpy(vec.data(), vals[0].data(), (size_t)dim * sizeof(ValueType));

            // pick a fresh tag not currently present
            uint32_t newTag = 0xDEAD0000u;
            while (m_opqTagVids.find(newTag) != m_opqTagVids.end()) newTag++;
            // Reserve a real version-map slot for the new vid (this is what AddIndex does via
            // m_versionMap.AddBatch before calling the extra searcher). Without it the search
            // path's Deleted(vid) bounds-check would treat the new vid as deleted.
            SizeType synVid = m_versionMap->GetVectorNum();
            m_versionMap->AddBatch(1);
            SizeType savedN = m_opqN;

            // place the vector in the canonical store (what the decorator deflate would do)
            std::string vbytes((const char*)vec.data(), (size_t)dim * sizeof(ValueType));
            std::vector<Helper::AsyncReadRequest> putReqs;
            m_opqVecDB->Put(synVid, vbytes, MaxTimeout, &putReqs);

            OPQInsertMaintain(synVid, vec.data(), &newTag, 1);

            // query for the fresh tag with the exact vector; expect synVid back
            QueryResult qr(vec.data(), 10, false);
            bool ok = OPQTagPureSearch(qr, newTag);
            bool found = false;
            for (int i = 0; i < qr.GetResultNum(); i++) {
                if (qr.GetResult(i)->VID == synVid) { found = true; break; }
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "[insert selftest] new vid=%d tag=0x%X search-returned=%s (results=%d, m_opqN %d->%d)\n",
                (int)synVid, newTag, (ok && found) ? "PASS" : "FAIL", qr.GetResultNum(), (int)savedN, (int)m_opqN);

            // rollback: drop the synthetic tag list, code coverage, and stored vector
            {
                std::unique_lock<std::shared_timed_mutex> wlock(m_opqMaintLock);
                m_opqTagVids.erase(newTag);
                m_opqN = savedN;
            }
            m_opqMutated.store(false, std::memory_order_release);
            m_opqVecDB->Delete(synVid);
        }

        ErrorCode SearchIndexOPQ(ExtraWorkSpace* p_exWorkSpace, QueryResult& p_queryResults,
            std::shared_ptr<VectorIndex> p_index, SearchStats* p_stats,
            std::set<int>* truth, std::map<int, std::set<int>>* found)
        {
            COMMON::QueryResultSet<ValueType>& queryResults = *((COMMON::QueryResultSet<ValueType>*) & p_queryResults);
            const bool logPhaseTime = p_stats != nullptr && m_opt != nullptr && m_opt->m_logPhaseTime;
            const auto opqStart = logPhaseTime ? std::chrono::high_resolution_clock::now()
                                               : std::chrono::high_resolution_clock::time_point{};
            double lutMs = 0.0;
            double postingIoMs = 0.0;
            double adcMs = 0.0;
            double rerankMs = 0.0;

            // ---- candidate posting prep (mirrors SearchIndex) ----
            if (p_exWorkSpace->m_postingFilter) {
                auto& ids = p_exWorkSpace->m_postingIDs;
                ids.erase(std::remove_if(ids.begin(), ids.end(),
                    [&](int pid) { return !p_exWorkSpace->m_postingFilter(pid); }), ids.end());
            }
            const bool hasInlineTagFilter =
                m_tagBytesPerVec > 0 && p_exWorkSpace->m_queryTags != nullptr && p_exWorkSpace->m_numQueryTags > 0;
            static const bool s_filterKeepUextra = []() {
                const char* env = std::getenv("SPTAG_FILTER_KEEP_UEXTRA");
                return env && env[0] == '1';
            }();
            if (hasInlineTagFilter && HasHeadRoles() && !s_filterKeepUextra) {
                auto& ids = p_exWorkSpace->m_postingIDs;
                ids.erase(std::remove_if(ids.begin(), ids.end(),
                    [&](int pid) { return IsUnfilterOnlyHead(pid); }), ids.end());
            }
            // Per-query-type posting READ scale (same on-disk layout, different read
            // length): filter reads only the pages covering the PURE prefix (tail
            // pages skipped -> less IO); unfilter reads the FULL posting (pure+tail)
            // for the extra boundary coverage. Because ~79% of heads' tail fits in
            // the slack of the pure prefix's last page, unfilter's full read adds a
            // page for only ~21% of heads (+~16% IO total) -- cheap for the recall it
            // buys, and mostly unavoidable. The read-length cap is applied per key in
            // a single MultiGet (maxBytesPerKey), so each posting reads its own scale.
            //
            // Diagnostic ablation toggles (DEFAULT OFF -> unfilter keeps tail+U_extra):
            //   SPTAG_ABLATE_UEXTRA=1 -> drop role==1 (U_extra, tail-only) heads from
            //     the unfilter candidate list (no IO, no compute for them).
            //   SPTAG_ABLATE_TAIL=1   -> cap unfilter reads+scan to the pure prefix
            //     (skip tail pages), matching the filter read scale.
            static const bool s_ablateUextra = []() {
                const char* env = std::getenv("SPTAG_ABLATE_UEXTRA");
                return env && env[0] == '1';   // default OFF: keep U_extra for unfilter
            }();
            static const bool s_ablateTail = []() {
                const char* env = std::getenv("SPTAG_ABLATE_TAIL");
                return env && env[0] == '1';   // default OFF: unfilter reads full posting
            }();
            static const bool s_unfilterPurePages = []() {
                const char* env = std::getenv("SPTAG_UNFILTER_PURE_PAGES");
                return env && env[0] == '1';
            }();
            static const int s_unfilterExtraTailPages = []() {
                const char* env = std::getenv("SPTAG_UNFILTER_EXTRA_TAIL_PAGES");
                return env ? std::max(0, std::atoi(env)) : 0;
            }();
            if (s_ablateUextra && HasHeadRoles() && !hasInlineTagFilter) {
                auto& ids = p_exWorkSpace->m_postingIDs;
                ids.erase(std::remove_if(ids.begin(), ids.end(),
                    [&](int pid) { return IsUnfilterOnlyHead(pid); }), ids.end());
            }
            static const bool s_unfilterTailEnabled = []() {
                const char* env = std::getenv("SPTAG_UNFILTER_TAIL");
                return !(env && env[0] == '0');
            }();
            // Filter: read+scan only the pure prefix (skip tail pages). Unfilter
            // reads the full posting unless the tail-ablation diagnostic is on.
            const bool useUnfilterTail = s_unfilterTailEnabled && m_hasPostingPureCounts && hasInlineTagFilter;
            const bool capScanToPure = useUnfilterTail || (s_ablateTail && m_hasPostingPureCounts && !hasInlineTagFilter);
            // Experimental unfilter mode: read only the pages that cover the pure
            // prefix, but scan all records present in those pages. This keeps tail
            // records that fit in already-paid pure pages and avoids extra tail-page IO.
            const bool capUnfilterToPurePages =
                (s_unfilterPurePages || s_unfilterExtraTailPages > 0) &&
                m_hasPostingPureCounts && !hasInlineTagFilter;
            static const bool s_trackAllStatsOPQ = (std::getenv("SPTAG_TRACK_ALL_STATS") != nullptr);
            const bool trackStatsOPQ = hasInlineTagFilter || s_trackAllStatsOPQ;

            // ---- query ADC LUT (normalized query copy to match training space) ----
            const ValueType* rawQuery = queryResults.GetTarget();
            int dim = m_opt->m_dim;
            std::vector<float> lut;
            std::vector<float> rq;
            // Real extended RaBitQ: prepare a per-query estimator context (auto-freed).
            void* rbq2ctx = nullptr;
            std::shared_ptr<void> rbq2guard;
            std::vector<float> rbq2qf;
            if (m_rbq2on) {
                // PrepareQuery wants m_dim un-normalized floats; widen via WidenQuery.
                rbq2qf = WidenQuery(rawQuery, dim);
                rbq2ctx = m_rbq2->AllocQuery();
                m_rbq2->PrepareQuery(rbq2ctx, rbq2qf.data());
                rbq2guard = std::shared_ptr<void>(rbq2ctx, [this](void* p) { m_rbq2->FreeQuery(p); });
            }
            const auto lutStart = logPhaseTime ? std::chrono::high_resolution_clock::now()
                                               : std::chrono::high_resolution_clock::time_point{};
            if (m_rbq) {
                RaBitQRotateQuery(rawQuery, rq);
                // Exhaustive RaBitQ over ALL resident vids (the no-tag/unfilter analogue of
                // tag-pure). The nprobe-limited posting scan plateaus well below 0.95 for
                // broad/unfilter, so SPTAG_RBQ_EXHAUSTIVE=1 scans every vid for full recall
                // with zero survivor IO. Only meaningful without an inline tag filter.
                static const bool s_rbqExhaustive = []() { const char* e = std::getenv("SPTAG_RBQ_EXHAUSTIVE"); return e && e[0] == '1'; }();
                if (s_rbqExhaustive && !hasInlineTagFilter && !m_rbq2on) {
                    queryResults.Reset();
                    for (SizeType vid = 0; vid < m_opqN; ++vid) {
                        if (m_versionMap->Deleted(vid)) continue;
                        queryResults.AddPoint(vid, RaBitQDist(rq, vid));
                    }
                    // NOTE: do NOT SortResult() here. The caller (SPANNIndex::SearchIndex)
                    // calls SortResult() exactly once after the extra search. SortResult
                    // assumes a max-heap; sorting here would leave an ascending array that
                    // the caller's second SortResult corrupts (ejecting the best result).
                    // AddPoint maintains the max-heap, so leave it for the caller.
                    queryResults.SetScanned((int)m_opqN);
                    return ErrorCode::Success;
                }
            } else {
                lut.resize((size_t)m_opqM * m_opqKs);
                // The posting quantizers are float-typed. When the index ValueType is
                // byte-sized, widen raw query bytes before building the ADC LUT.
                std::vector<float> qf = WidenQuery(rawQuery, dim);
                if (m_opt->m_distCalcMethod == DistCalcMethod::Cosine)
                    COMMON::Utils::Normalize<float>(qf.data(), dim, COMMON::Utils::GetBase<float>());
                if (m_pipePQ) {
                    if (!m_pipePQTable || m_pipePQTable->Dim() != dim) return ErrorCode::Fail;
                    m_pipePQTable->PopulateDistances(qf.data(), lut.data(), m_opt->m_distCalcMethod);
                } else {
                    m_opqQ->QuantizeVector(qf.data(), (std::uint8_t*)lut.data(), true);
                }
            }
            if (logPhaseTime) {
                lutMs = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - lutStart).count();
            }
            // ---- ADC screen survivors, keep best-L by ADC (max-heap) ----
            std::priority_queue<std::pair<float, int>> heap;
            const SizeType postingNum = m_postingSizes.GetPostingNum();
            auto queueTaggedMerge = [&](SizeType headID, int liveCount, bool scannedFullPosting) {
                if (!scannedFullPosting || liveCount > m_mergeThreshold ||
                    !m_taggedMaintenance.load(std::memory_order_acquire)) {
                    return;
                }
                std::lock_guard<std::mutex> lock(m_taggedMergeCandidatesLock);
                m_taggedMergeCandidates.insert(headID);
            };
            size_t slimBytesRead = 0;
            const auto postingListCount = (uint32_t)p_exWorkSpace->m_postingIDs.size();
            if (trackStatsOPQ) p_exWorkSpace->m_postingProbeStats.m_readPostings += postingListCount;
            // SPTAG_OPQ_ASYNC_FULL=1: read the FULL postings (vector inline) from the
            // existing posting store via the SAME async/batched MultiGet the baseline
            // uses (FileIO libaio), then run the ADC screen over the resident codes.
            // Reads the same bytes as baseline but proves whether the cold loss was the
            // serial mmap slim scan (async should recover baseline-class cold QPS).
            // In-posting-DB mode (m_opqInpostDb) ALWAYS takes this async path: the slim
            // [meta|code] records live in db (stride m_vectorInfoSize), read via the same
            // MultiGet, with the OPQ code taken inline from each record.
            static const bool s_asyncFull = []() { const char* e = std::getenv("SPTAG_OPQ_ASYNC_FULL"); return e && e[0] == '1'; }();
            const bool asyncScan = !m_rbq && !m_rbq2on &&
                ((s_asyncFull && !m_opqInpostCode && !m_opqCodes.empty()) || m_opqInpostDb);
            const auto adcStart = logPhaseTime ? std::chrono::high_resolution_clock::now()
                                               : std::chrono::high_resolution_clock::time_point{};
            if (asyncScan) {
                const auto postingIoStart = logPhaseTime ? std::chrono::high_resolution_clock::now()
                                                         : std::chrono::high_resolution_clock::time_point{};
                if ((capScanToPure || capUnfilterToPurePages) && m_hasPostingPureCounts) {
                    // Cap each posting's READ to its pure prefix so tail-replica
                    // records are never fetched from SSD (saves IO, not just the
                    // scan compute). The block layer rounds each cap up to whole
                    // pages. U_extra (tail-only) heads read a single record's worth
                    // (minimal IO); when ablating U_extra they are already dropped
                    // from m_postingIDs above, so this is a harmless floor for them.
                    std::vector<std::uint32_t> maxBytes(p_exWorkSpace->m_postingIDs.size(), 0);
                    for (size_t i = 0; i < maxBytes.size(); ++i) {
                        SizeType hid = p_exWorkSpace->m_postingIDs[i];
                        if (IsUnfilterOnlyHead((int)hid) && !capUnfilterToPurePages) {
                            maxBytes[i] = (std::uint32_t)m_vectorInfoSize;
                            continue;
                        }
                        int pure = m_postingPureCounts.GetSize(hid);
                        if (pure > 0 || capUnfilterToPurePages) {
                            std::uint32_t bytes = (pure > 0)
                                ? (std::uint32_t)pure * (std::uint32_t)m_vectorInfoSize
                                : 0;
                            if (capUnfilterToPurePages) {
                                bytes = ((bytes + (std::uint32_t)PageSize - 1) / (std::uint32_t)PageSize
                                    + (std::uint32_t)s_unfilterExtraTailPages) * (std::uint32_t)PageSize;
                                if (bytes == 0) bytes = (std::uint32_t)m_vectorInfoSize;
                            } else if (IsUnfilterOnlyHead((int)hid)) {
                                bytes = (std::uint32_t)m_vectorInfoSize;
                            }
                            maxBytes[i] = bytes;
                        }
                    }
                    db->MultiGet(p_exWorkSpace->m_postingIDs, p_exWorkSpace->m_pageBuffers,
                                 maxBytes, HardLatencyLimit(), &(p_exWorkSpace->m_diskRequests));
                } else {
                    db->MultiGet(p_exWorkSpace->m_postingIDs, p_exWorkSpace->m_pageBuffers,
                                 HardLatencyLimit(), &(p_exWorkSpace->m_diskRequests));
                }
                if (logPhaseTime) {
                    postingIoMs = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - postingIoStart).count();
                }
                for (uint32_t pi = 0; pi < postingListCount; ++pi) {
                    const std::uint64_t logicalBytes =
                        p_exWorkSpace->m_pageBuffers[pi].GetAvailableSize();
                    if (logicalBytes == 0) continue;
                    const std::uint64_t pageReads =
                        (logicalBytes + PageSize - 1) / PageSize;
                    p_exWorkSpace->m_postingProbeStats.m_postingPageReads += pageReads;
                    p_exWorkSpace->m_postingProbeStats.m_postingLogicalBytes += logicalBytes;
                    // FileIO issues PageSize reads even for a posting's partial final page.
                    p_exWorkSpace->m_postingProbeStats.m_postingPhysicalBytes +=
                        pageReads * PageSize;
                }
                for (uint32_t pi = 0; pi < postingListCount; ++pi) {
                    SizeType h = p_exWorkSpace->m_postingIDs[pi];
                    if (h < 0 || h >= postingNum) continue;
                    auto& buffer = p_exWorkSpace->m_pageBuffers[pi];
                    const std::uint8_t* data = (const std::uint8_t*)buffer.GetBuffer();
                    int n = (int)(buffer.GetAvailableSize() / m_vectorInfoSize);
                    int scanLimit = n;
                    if (capScanToPure) {
                        if (IsUnfilterOnlyHead((int)h)) scanLimit = 0;
                        else { int pure = m_postingPureCounts.GetSize(h); if (pure > 0 && pure < scanLimit) scanLimit = pure; }
                    }
                    p_exWorkSpace->m_postingProbeStats.m_adcScannedVectors +=
                        static_cast<std::uint64_t>(scanLimit);
                    int liveCount = n;
                    for (int i = 0; i < scanLimit; i++) {
                        const std::uint8_t* e = data + (size_t)i * m_vectorInfoSize;
                        int vid = *(reinterpret_cast<const int*>(e));
                        // PipePQ records carry their ADC code inline. New tagged-update
                        // VIDs therefore need only be present in the version map; they do
                        // not exist in the immutable build-time code sidecar (m_opqN).
                        if (vid < 0 || vid >= m_versionMap->Count()) {
                            --liveCount;
                            continue;
                        }
                        if (m_versionMap->Deleted(vid)) {
                            --liveCount;
                            continue;
                        }
                        if (trackStatsOPQ) ++p_exWorkSpace->m_postingProbeStats.m_scannedVectors;
                        if (hasInlineTagFilter) {
                            bool tagMatch = false;
                            const uint32_t* vt = reinterpret_cast<const uint32_t*>(e + sizeof(int) + sizeof(uint8_t));
                            for (int ti = 0; ti < m_numTagsPerVec && !tagMatch; ti++)
                                for (int qi = 0; qi < p_exWorkSpace->m_numQueryTags && !tagMatch; qi++)
                                    if (vt[ti] == p_exWorkSpace->m_queryTags[qi]) tagMatch = true;
                            if (!tagMatch) continue;
                        }
                        if (trackStatsOPQ) ++p_exWorkSpace->m_postingProbeStats.m_matchedVectors;
                        if (p_exWorkSpace->m_deduper.CheckAndSet(vid)) continue;
                        const std::uint8_t* c = m_opqInpostCode
                            ? (e + m_metaDataSize)
                            : &m_opqCodes[(size_t)vid * m_opqM];
                        float adc = 0;
                        for (int m = 0; m < m_opqM; m++) adc += lut[(size_t)m * m_opqKs + c[m]];
                        if ((int)heap.size() < m_opqL) heap.push({ adc, vid });
                        else if (adc < heap.top().first) { heap.pop(); heap.push({ adc, vid }); }
                    }
                    queueTaggedMerge(h, liveCount,
                                     scanLimit == n && n == m_postingSizes.GetSize(h));
                }
            } else
            for (uint32_t pi = 0; pi < postingListCount; ++pi) {
                SizeType h = p_exWorkSpace->m_postingIDs[pi];
                if (h < 0 || h >= postingNum) continue;
                std::uint64_t o0 = m_slimOff[h], o1 = m_slimOff[h + 1];
                int n = (int)((o1 - o0) / m_slimRec);
                int scanLimit = n;
                if (capScanToPure) {
                    if (IsUnfilterOnlyHead((int)h)) scanLimit = 0;
                    else { int pure = m_postingPureCounts.GetSize(h); if (pure > 0 && pure < scanLimit) scanLimit = pure; }
                }
                slimBytesRead += (size_t)scanLimit * m_slimRec;
                int liveCount = n;
                const std::uint8_t* base;
                if (m_slimDirectFd >= 0 && scanLimit > 0) {
                    // O_DIRECT device-bound read of the scanned posting range [o0, o0+scanLimit*rec).
                    // Align the offset/length to the device block size into a per-thread bounce buffer.
                    const size_t ALIGN = 4096;
                    std::uint64_t readEnd = o0 + (std::uint64_t)scanLimit * m_slimRec;
                    std::uint64_t aStart = o0 & ~(std::uint64_t)(ALIGN - 1);
                    std::uint64_t aEnd = (readEnd + ALIGN - 1) & ~(std::uint64_t)(ALIGN - 1);
                    size_t aLen = (size_t)(aEnd - aStart);
                    static thread_local std::uint8_t* t_buf = nullptr;
                    static thread_local size_t t_cap = 0;
                    if (aLen > t_cap) {
                        if (t_buf) free(t_buf);
                        if (posix_memalign((void**)&t_buf, ALIGN, aLen) != 0) { t_buf = nullptr; t_cap = 0; }
                        else t_cap = aLen;
                    }
                    if (t_buf && pread(m_slimDirectFd, t_buf, aLen, (off_t)aStart) > 0) {
                        base = t_buf + (o0 - aStart);
                    } else {
                        base = m_slim + o0;
                    }
                } else {
                    base = m_slim + o0;
                }
                for (int i = 0; i < scanLimit; i++) {
                    const std::uint8_t* e = base + (size_t)i * m_slimRec;
                    int vid = *(reinterpret_cast<const int*>(e));
                    if (vid < 0 || vid >= m_opqN) {
                        --liveCount;
                        continue;
                    }
                    if (m_versionMap->Deleted(vid)) {
                        --liveCount;
                        continue;
                    }
                    if (trackStatsOPQ) ++p_exWorkSpace->m_postingProbeStats.m_scannedVectors;
                    if (hasInlineTagFilter) {
                        bool tagMatch = false;
                        const uint32_t* vt = reinterpret_cast<const uint32_t*>(e + sizeof(int) + sizeof(uint8_t));
                        for (int ti = 0; ti < m_numTagsPerVec && !tagMatch; ti++)
                            for (int qi = 0; qi < p_exWorkSpace->m_numQueryTags && !tagMatch; qi++)
                                if (vt[ti] == p_exWorkSpace->m_queryTags[qi]) tagMatch = true;
                        if (!tagMatch) continue;
                    }
                    if (trackStatsOPQ) ++p_exWorkSpace->m_postingProbeStats.m_matchedVectors;
                    if (p_exWorkSpace->m_deduper.CheckAndSet(vid)) continue;
                    float adc;
                    if (m_rbq2on) {
                        adc = m_rbq2->Estimate(rbq2ctx, vid);
                    } else if (m_rbq) {
                        adc = RaBitQDist(rq, vid);
                    } else {
                        const std::uint8_t* c = m_opqInpostCode
                            ? (e + m_metaDataSize)
                            : &m_opqCodes[(size_t)vid * m_opqM];
                        adc = 0;
                        for (int m = 0; m < m_opqM; m++) adc += lut[(size_t)m * m_opqKs + c[m]];
                    }
                    if ((int)heap.size() < m_opqL) heap.push({ adc, vid });
                    else if (adc < heap.top().first) { heap.pop(); heap.push({ adc, vid }); }
                }
                queueTaggedMerge(h, liveCount,
                                 scanLimit == n && n == m_postingSizes.GetSize(h));
            }

            if (logPhaseTime) {
                adcMs = std::max(0.0, std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - adcStart).count() - postingIoMs);
            }

            // ---- exact rerank of the L survivors: fetch vectors from the RocksDB
            //      vid->vector store (single shared copy, vector-level fine-grained IO) ----
            // SPTAG_OPQ_ADC_ONLY=1: skip the full-vector rerank entirely and return the
            // top-k by ADC (quantized) distance. Zero survivor device IO; recall drops to
            // the OPQ quantization ceiling but QPS becomes purely in-RAM.
            static const bool s_adcOnly = []() { const char* e = std::getenv("SPTAG_OPQ_ADC_ONLY"); return e && e[0] == '1'; }();
            // Diagnostic: RaBitQ does the ADC screen (identical candidate set to no-rerank),
            // but the L survivors get an EXACT rerank from the vector store. This isolates
            // whether the no-rerank plateau is a COVERAGE problem (true NN absent from the
            // screened candidates -> this also stays low) or a RANKING problem (true NN
            // present but RaBitQ mis-scores it -> this recovers to the candidate ceiling).
            static const bool s_rbqScreenRerank = []() { const char* e = std::getenv("SPTAG_RBQ_SCREEN_RERANK"); return e && e[0] == '1'; }();
            int fetched = (int)heap.size();
            int listElements = fetched;
            p_exWorkSpace->m_postingProbeStats.m_adcSurvivors +=
                static_cast<std::uint64_t>(fetched);
            // Harvest the head-search seed vids BEFORE any Reset(). On the dense graph path
            // these exact-distance seeds carry most of the recall (PQ keeps them implicitly
            // because RerankFromVecDB does not Reset); the RaBitQ no-rerank branch must Reset
            // to avoid mixing RaBitQ-L2 with cosine-scale seeds, so we instead re-score the
            // harvested seed vids in RaBitQ space and merge them back as equal candidates.
            std::vector<int> headVids;
            if (m_rbq || m_rbq2on) {
                int rn = queryResults.GetResultNum();
                headVids.reserve(rn);
                for (int r = 0; r < rn; r++) {
                    BasicResult* hr = queryResults.GetResult(r);
                    if (hr && hr->VID >= 0) headVids.push_back(hr->VID);
                }
            }
            const auto rerankStart = logPhaseTime ? std::chrono::high_resolution_clock::now()
                                                  : std::chrono::high_resolution_clock::time_point{};
            if ((m_rbq && s_rbqScreenRerank) || (m_rbq2on && !s_adcOnly)) {
                // Mirror the PQ path: the head-search seeds already carry EXACT cosine
                // distances (head nav scores them with full-float head vectors), and
                // RerankFromVecDB produces exact cosine for the screen survivors -- the
                // SAME scale. So do NOT Reset and do NOT re-fetch the head seeds. The old
                // code Reset() then re-ranked survivors+headVids, which re-fetched every
                // head seed (~head-count redundant vector reads/query) -- the dominant
                // cost under direct IO (unfilter 164 vs PQ 393). Screen survivors already
                // exclude head VIDs (marked in m_deduper during head processing), so
                // AddPoint cleanly merges the reranked survivors with the head seeds.
                std::vector<int> survivors;
                survivors.reserve(fetched);
                while (!heap.empty()) { survivors.push_back(heap.top().second); heap.pop(); }
                RerankFromVecDB(survivors, rawQuery, dim, queryResults);
                // No SortResult here: the caller sorts once (see note in the exhaustive
                // branch). AddPoint inside RerankFromVecDB leaves a valid max-heap.
            } else if (s_adcOnly || m_rbq) {
                // No-rerank output: results come solely from the approximate (PQ-ADC or
                // RaBitQ) screen, on a different distance scale than the head-seeded
                // candidates, so start from a clean result set.
                queryResults.Reset();
                std::unordered_set<int> seen;
                while (!heap.empty()) {
                    int hv2 = heap.top().second; float hd2 = heap.top().first; heap.pop();
                    if (seen.insert(hv2).second) queryResults.AddPoint(hv2, hd2);
                }
                // Re-include the head-search seeds (dropped by Reset) scored in RaBitQ space.
                // NOTE: head VIDs were already marked in m_deduper during head processing
                // (SPANNIndex), so we must NOT gate on the deduper here (that silently
                // discarded every head candidate). Dedup against the local 'seen' set only.
                if (m_rbq2on) {
                    for (int hv : headVids) {
                        if (hv < 0 || hv >= m_opqN) continue;
                        if (m_versionMap->Deleted(hv)) continue;
                        if (!seen.insert(hv).second) continue;
                        queryResults.AddPoint(hv, m_rbq2->Estimate(rbq2ctx, hv));
                    }
                } else if (m_rbq) {
                    for (int hv : headVids) {
                        if (hv < 0 || hv >= m_opqN) continue;
                        if (m_versionMap->Deleted(hv)) continue;
                        if (!seen.insert(hv).second) continue;
                        queryResults.AddPoint(hv, RaBitQDist(rq, hv));
                    }
                }
                // No SortResult here: AddPoint above keeps a valid max-heap and the caller
                // (SPANNIndex::SearchIndex) calls SortResult exactly once. Sorting twice
                // corrupts the heap and ejects the best (rank-0) result.
            } else {
                std::vector<int> survivors;
                survivors.reserve(fetched);
                while (!heap.empty()) { survivors.push_back(heap.top().second); heap.pop(); }
                static const bool s_dbgPqReset = []() { const char* e = std::getenv("SPTAG_DBG_PQ_RESET"); return e && e[0] == '1'; }();
                if (s_dbgPqReset) queryResults.Reset();
                RerankFromVecDB(survivors, rawQuery, dim, queryResults,
                                &p_exWorkSpace->m_postingProbeStats);
            }
            if (logPhaseTime) {
                rerankMs = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - rerankStart).count();
            }

            static const bool s_rbqDbg = []() { const char* e = std::getenv("SPTAG_RBQ_DBG"); return e && e[0] == '1'; }();
            if (s_rbqDbg) {
                static std::atomic<int> g_dq{0};
                int qn2 = g_dq++;
                if (qn2 < 6) {
                    fprintf(stderr, "[RBQ-DBG dense] q=%d top1=(id=%d,dist=%.5f) rbq=%d screenRerank=%d\n",
                        qn2, queryResults.GetResult(0)->VID, queryResults.GetResult(0)->Dist,
                        (int)m_rbq, (int)s_rbqScreenRerank);
                    fflush(stderr);
                }
            }
            if (p_stats) {
                p_stats->m_totalListElementsCount = listElements;
                p_stats->m_diskAccessCount =
                    (int)((slimBytesRead + (size_t)fetched * dim * sizeof(ValueType)) / 1024);
                p_stats->m_diskIOCount = fetched;
            }
            static const bool s_opqStats = []() { const char* e = std::getenv("SPTAG_OPQ_STATS"); return e && e[0] == '1'; }();
            if (s_opqStats) {
                static std::atomic<size_t> g_slim{ 0 }, g_fetch{ 0 }, g_q{ 0 };
                size_t q = ++g_q;
                g_slim += slimBytesRead;
                g_fetch += (size_t)fetched;
                if (q % 1000 == 0) {
                    size_t sb = g_slim, ft = g_fetch;
                    size_t bytes = sb + ft * dim * sizeof(ValueType);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "[OPQ stats] q=%zu slimBytes/q=%.0f fetched/q=%.1f totalBytes/q=%.0f\n",
                        q, (double)sb / q, (double)ft / q, (double)bytes / q);
                }
            }
            queryResults.SetScanned(listElements);
            if (logPhaseTime) {
                const double totalMs = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - opqStart).count();
                const double otherMs = std::max(0.0, totalMs - lutMs - postingIoMs - adcMs - rerankMs);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                    "OPQPhaseTime: async=%d postings=%u survivors=%d lut=%.3f postingIO=%.3f adc=%.3f rerank=%.3f other=%.3f total=%.3f\n",
                    asyncScan ? 1 : 0, postingListCount, fetched, lutMs, postingIoMs, adcMs,
                    rerankMs, otherMs, totalMs);
            }
            return ErrorCode::Success;
        }

        // Cold O_DIRECT read of one full-precision vector (vid -> dim ValueType) into a
        // thread-local aligned bounce buffer. Every call hits the device (the rerank base
        // is NEVER page-cache resident). Returns nullptr on OOB / read failure.
        // Open the flat full-precision base file (vid -> dim uint8, 8B header) O_DIRECT
        // for cold rerank reads (never page-cache resident). Idempotent. Used by BOTH
        // the in-posting RaBitQ path and the in-posting OPQ-DB path so their rerank IO
        // is identical (deep-queue libaio over this fd) and only the estimator differs.
        void EnsureInpostBaseFd()
        {
            if (m_inpostBaseFd >= 0) return;
            const char* bp = (m_opt && !m_opt->m_fullVectorFile.empty()) ? m_opt->m_fullVectorFile.c_str() : nullptr;
            if (!bp || !*bp) return;
            int hfd = open(bp, O_RDONLY);
            if (hfd >= 0) {
                int32_t h[2] = { 0, 0 };
                if (pread(hfd, h, 8, 0) == 8) { m_inpostBaseN = (size_t)h[0]; m_inpostBaseDim = (int)h[1]; }
                close(hfd);
#ifdef O_DIRECT
                m_inpostBaseFd = open(bp, O_RDONLY | O_DIRECT);
#else
                m_inpostBaseFd = open(bp, O_RDONLY);
#endif
                if (m_inpostBaseFd >= 0)
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "[InpostBase] O_DIRECT base %s N=%zu dim=%d (cold rerank, no residency)\n",
                        bp, m_inpostBaseN, m_inpostBaseDim);
                else
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[InpostBase] O_DIRECT base open failed %s\n", bp);
            } else {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[InpostBase] open base failed %s\n", bp);
            }
        }

        std::string GetDynamicVectorPath() const
        {
            if (!m_dynamicVectorPath.empty()) return m_dynamicVectorPath;
            if (m_opt->m_updateVectorFile.empty()) return std::string();
            if (m_opt->m_recovery) {
                return GetDynamicVectorCheckpointPath(m_opt->m_persistentBufferPath);
            }
            if (m_opt->m_updateVectorFile[0] == '/') return m_opt->m_updateVectorFile;
            return m_opt->m_indexDirectory + FolderSep + m_opt->m_updateVectorFile;
        }

        std::string GetDynamicVectorCheckpointPath(const std::string& p_baseDir) const
        {
            if (p_baseDir.empty() || m_opt->m_updateVectorFile.empty()) return std::string();
            std::string fileName = m_opt->m_updateVectorFile;
            const size_t separator = fileName.find_last_of("/\\");
            if (separator != std::string::npos) fileName = fileName.substr(separator + 1);
            return p_baseDir + FolderSep + fileName;
        }

        bool CopyDynamicVectorStore(const std::string& p_checkpointDir)
        {
            if (m_dynamicVectorFd < 0) return true;
            const std::string sourcePath = GetDynamicVectorPath();
            const std::string targetPath = GetDynamicVectorCheckpointPath(p_checkpointDir);
            if (sourcePath.empty() || targetPath.empty() || sourcePath == targetPath) return true;

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

        bool WriteDynamicVectorHeader()
        {
#ifdef _MSC_VER
            return false;
#else
            if (m_dynamicVectorFd < 0) return false;
            DynamicVectorStoreHeader header;
            header.valueSize = sizeof(ValueType);
            header.dimension = static_cast<std::uint32_t>(m_opt->m_dim);
            header.baseVID = m_dynamicVectorBaseVID;
            header.slotCount = m_dynamicVectorSlotCount;
            return pwrite(m_dynamicVectorFd, &header, sizeof(header), 0) ==
                   static_cast<ssize_t>(sizeof(header));
#endif
        }

        bool OpenDynamicVectorStore(bool create)
        {
            std::unique_lock<std::shared_mutex> lock(m_dynamicVectorLock);
            return OpenDynamicVectorStoreLocked(create);
        }

        bool OpenDynamicVectorStoreLocked(bool create)
        {
#ifdef _MSC_VER
            (void)create;
            return false;
#else
            if (m_dynamicVectorFd >= 0 && (!create || m_dynamicVectorWritable)) return true;
            if (m_dynamicVectorFd >= 0) {
                close(m_dynamicVectorFd);
                m_dynamicVectorFd = -1;
                m_dynamicVectorWritable = false;
            }
            m_dynamicVectorPath = GetDynamicVectorPath();
            if (m_dynamicVectorPath.empty()) return !create;

            int flags = create ? (O_RDWR | O_CREAT) : O_RDONLY;
            m_dynamicVectorFd = open(m_dynamicVectorPath.c_str(), flags, 0644);
            if (m_dynamicVectorFd < 0) {
                return !create && errno == ENOENT;
            }
            m_dynamicVectorWritable = create;

            struct stat st {};
            if (fstat(m_dynamicVectorFd, &st) != 0) {
                close(m_dynamicVectorFd);
                m_dynamicVectorFd = -1;
                m_dynamicVectorWritable = false;
                return false;
            }
            if (st.st_size == 0) {
                if (!create) {
                    close(m_dynamicVectorFd);
                    m_dynamicVectorFd = -1;
                    m_dynamicVectorWritable = false;
                    return false;
                }
                return true;
            }

            DynamicVectorStoreHeader header;
            if (pread(m_dynamicVectorFd, &header, sizeof(header), 0) !=
                    static_cast<ssize_t>(sizeof(header)) ||
                header.magic != DynamicVectorStoreHeader{}.magic ||
                header.version != 1 || header.valueSize != sizeof(ValueType) ||
                header.dimension != static_cast<std::uint32_t>(m_opt->m_dim) ||
                header.baseVID < 0) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                             "[TaggedUpdate] invalid dynamic vector sidecar %s.\n",
                             m_dynamicVectorPath.c_str());
                close(m_dynamicVectorFd);
                m_dynamicVectorFd = -1;
                m_dynamicVectorWritable = false;
                return false;
            }
            m_dynamicVectorBaseVID = static_cast<SizeType>(header.baseVID);
            m_dynamicVectorSlotCount = static_cast<size_t>(header.slotCount);
            return true;
#endif
        }

        bool AppendDynamicVectors(std::shared_ptr<VectorSet>& vectors, SizeType begin)
        {
#ifdef _MSC_VER
            (void)vectors;
            (void)begin;
            return false;
#else
            if (vectors == nullptr || vectors->Count() <= 0) {
                return false;
            }
            std::unique_lock<std::shared_mutex> lock(m_dynamicVectorLock);
            if (!OpenDynamicVectorStoreLocked(true)) return false;
            if (m_dynamicVectorFd < 0) return false;

            if (m_dynamicVectorBaseVID < 0) {
                m_dynamicVectorBaseVID = begin;
                m_dynamicVectorSlotCount = 0;
                if (!WriteDynamicVectorHeader()) return false;
            }
            if (begin < m_dynamicVectorBaseVID) return false;

            const size_t recordBytes = static_cast<size_t>(m_opt->m_dim) * sizeof(ValueType);
            const size_t beginSlot = static_cast<size_t>(begin - m_dynamicVectorBaseVID);
            std::vector<std::uint8_t> zero(recordBytes, 0);
            for (size_t slot = m_dynamicVectorSlotCount; slot < beginSlot; ++slot) {
                const off_t offset = static_cast<off_t>(sizeof(DynamicVectorStoreHeader) + slot * recordBytes);
                if (pwrite(m_dynamicVectorFd, zero.data(), recordBytes, offset) !=
                    static_cast<ssize_t>(recordBytes)) {
                    return false;
                }
            }

            for (int i = 0; i < vectors->Count(); ++i) {
                const size_t slot = beginSlot + static_cast<size_t>(i);
                const off_t offset = static_cast<off_t>(sizeof(DynamicVectorStoreHeader) + slot * recordBytes);
                if (pwrite(m_dynamicVectorFd, vectors->GetVector(i), recordBytes, offset) !=
                    static_cast<ssize_t>(recordBytes)) {
                    return false;
                }
            }
            m_dynamicVectorSlotCount =
                std::max(m_dynamicVectorSlotCount, beginSlot + static_cast<size_t>(vectors->Count()));
            return WriteDynamicVectorHeader();
#endif
        }

        const ValueType* ReadDynamicVector(SizeType vid, int dim)
        {
#ifdef _MSC_VER
            (void)vid;
            (void)dim;
            return nullptr;
#else
            std::shared_lock<std::shared_mutex> lock(m_dynamicVectorLock);
            if (m_dynamicVectorFd < 0 || m_dynamicVectorBaseVID < 0 || dim != m_opt->m_dim ||
                vid < m_dynamicVectorBaseVID) {
                return nullptr;
            }
            const size_t slot = static_cast<size_t>(vid - m_dynamicVectorBaseVID);
            if (slot >= m_dynamicVectorSlotCount) return nullptr;
            static thread_local std::vector<ValueType> buffer;
            buffer.resize(static_cast<size_t>(dim));
            const size_t recordBytes = static_cast<size_t>(dim) * sizeof(ValueType);
            const off_t offset = static_cast<off_t>(sizeof(DynamicVectorStoreHeader) + slot * recordBytes);
            if (pread(m_dynamicVectorFd, buffer.data(), recordBytes, offset) !=
                static_cast<ssize_t>(recordBytes)) {
                return nullptr;
            }
            return buffer.data();
#endif
        }

        bool HasDynamicVector(SizeType vid) const
        {
#ifdef _MSC_VER
            (void)vid;
            return false;
#else
            std::shared_lock<std::shared_mutex> lock(m_dynamicVectorLock);
            return m_dynamicVectorFd >= 0 && m_dynamicVectorBaseVID >= 0 &&
                   vid >= m_dynamicVectorBaseVID;
#endif
        }

        bool HasDynamicVectorStore() const
        {
#ifdef _MSC_VER
            return false;
#else
            std::shared_lock<std::shared_mutex> lock(m_dynamicVectorLock);
            return m_dynamicVectorFd >= 0 && m_dynamicVectorBaseVID >= 0;
#endif
        }

        const ValueType* ReadBaseVecDirect(int vid, int dim)
        {
            if (vid < 0) return nullptr;
            if (const ValueType* dynamicVector = ReadDynamicVector(vid, dim)) return dynamicVector;
            if (m_inpostBaseFd < 0 || (size_t)vid >= m_inpostBaseN) return nullptr;
            const size_t ALIGN = 4096;
            std::uint64_t recOff = 8 + (std::uint64_t)vid * (std::uint64_t)dim * sizeof(ValueType);
            std::uint64_t recEnd = recOff + (std::uint64_t)dim * sizeof(ValueType);
            std::uint64_t aStart = recOff & ~(std::uint64_t)(ALIGN - 1);
            std::uint64_t aEnd = (recEnd + ALIGN - 1) & ~(std::uint64_t)(ALIGN - 1);
            size_t aLen = (size_t)(aEnd - aStart);
            static thread_local std::uint8_t* t_buf = nullptr;
            static thread_local size_t t_cap = 0;
            if (aLen > t_cap) {
                if (t_buf) free(t_buf);
                if (posix_memalign((void**)&t_buf, ALIGN, aLen) != 0) { t_buf = nullptr; t_cap = 0; return nullptr; }
                t_cap = aLen;
            }
            if (!t_buf) return nullptr;
            if (pread(m_inpostBaseFd, t_buf, aLen, (off_t)aStart) <= 0) return nullptr;
            return reinterpret_cast<const ValueType*>(t_buf + (recOff - aStart));
        }

        // Deep-queue libaio rerank: issue ALL L survivor reads against the flat
        // O_DIRECT base in ONE io_submit, then io_getevents-wait them together.
        // Unlike RocksDB BlobDB MultiGet (which in 7.6 reads blob payloads serially
        // -> ~4x effective queue depth) this gives the device its full queue depth
        // (PipeANN-style) so the L random rerank reads overlap instead of trickling.
        // Returns true on success; falls back to the caller's path on any error.
        bool RerankBaseDirectBatch(const std::vector<int>& vids, const ValueType* rawQuery,
            int dim, COMMON::QueryResultSet<ValueType>& queryResults,
            ExtraWorkSpace::PostingProbeStats* p_workStats = nullptr)
        {
            int L = (int)vids.size();
            if (L <= 0 || m_inpostBaseFd < 0) return false;
            for (int vid : vids) {
                if (HasDynamicVector(vid)) {
                    return false;
                }
            }
            auto& aioPool = Helper::SharedAIOPool::Instance();
            if (!aioPool.IsUsable()) {
                static std::atomic<bool> s_reportedUnavailablePool{false};
                bool expected = false;
                if (s_reportedUnavailablePool.compare_exchange_strong(expected, true)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "RerankBaseDirectBatch: shared AIO pool is not usable; "
                                 "falling back to the configured rerank path\n");
                }
                return false;
            }
            const size_t ALIGN = 4096;
            const size_t recBytes = (size_t)dim * sizeof(ValueType);
            // Per-thread reusable aligned buffer (max 2 pages per survivor) + iocb arrays.
            static thread_local std::uint8_t* t_buf = nullptr;
            static thread_local size_t t_cap = 0;
            static thread_local int t_ctxId = -1;
            if (t_ctxId < 0) {
                static std::atomic<int> s_next{0};
                t_ctxId = s_next.fetch_add(1);
            }
            size_t needCap = (size_t)L * (2 * ALIGN);
            if (needCap > t_cap) {
                if (t_buf) free(t_buf);
                if (posix_memalign((void**)&t_buf, ALIGN, needCap) != 0) { t_buf = nullptr; t_cap = 0; return false; }
                t_cap = needCap;
            }
            if (!t_buf) return false;

            std::vector<struct iocb> cbs(L);
            std::vector<struct iocb*> cbps(L);
            std::vector<struct io_event> evs(L);
            std::vector<std::uint64_t> aStartOf(L);
            std::vector<size_t> aLenOf(L);
            int n = 0;
            for (int i = 0; i < L; i++) {
                int vid = vids[i];
                if (vid < 0 || (size_t)vid >= m_inpostBaseN) continue;
                std::uint64_t recOff = 8 + (std::uint64_t)vid * recBytes;
                std::uint64_t recEnd = recOff + recBytes;
                std::uint64_t aStart = recOff & ~(std::uint64_t)(ALIGN - 1);
                std::uint64_t aEnd = (recEnd + ALIGN - 1) & ~(std::uint64_t)(ALIGN - 1);
                size_t aLen = (size_t)(aEnd - aStart);
                std::uint8_t* buf = t_buf + (size_t)n * (2 * ALIGN);
                struct iocb* cb = &cbs[n];
                memset(cb, 0, sizeof(struct iocb));
                cb->aio_lio_opcode = IOCB_CMD_PREAD;
                cb->aio_fildes = m_inpostBaseFd;
                cb->aio_buf = reinterpret_cast<std::uint64_t>(buf);
                cb->aio_nbytes = aLen;
                cb->aio_offset = (std::int64_t)aStart;
                cb->aio_data = (std::uint64_t)i;   // index back into vids
                cbps[n] = cb;
                aStartOf[n] = aStart;
                aLenOf[n] = aLen;
                n++;
            }
            if (n == 0) return true;

            aio_context_t ctx = aioPool.GetContext(t_ctxId);
            std::unique_lock<std::mutex> contextLock(
                aioPool.GetContextMutex(t_ctxId));
            int submitted = 0;
            while (submitted < n) {
                long s = syscall(__NR_io_submit, ctx, (long)(n - submitted), cbps.data() + submitted);
                if (s <= 0) {
                    if (submitted == 0) return false;   // total failure -> caller fallback
                    break;
                }
                submitted += (int)s;
            }
            int done = 0;
            while (done < submitted) {
                long d = syscall(__NR_io_getevents, ctx, (long)(submitted - done), (long)(submitted - done),
                    evs.data() + done, nullptr);
                if (d <= 0) break;
                done += (int)d;
            }
            if (done != n) return false;
            if (p_workStats != nullptr) {
                std::uint64_t physicalBytes = 0;
                for (int i = 0; i < n; ++i) {
                    physicalBytes += aLenOf[i];
                }
                p_workStats->m_rerankReadRequests += static_cast<std::uint64_t>(n);
                p_workStats->m_rerankPhysicalBytes += physicalBytes;
            }

            auto rawDist = COMMON::DistanceCalcSelector<ValueType>(m_opt->m_distCalcMethod);
            for (int e = 0; e < done; e++) {
                int i = (int)evs[e].data;             // original vids index
                if (i < 0 || i >= L) continue;
                if ((long)evs[e].res <= 0) continue;
                // Recover this request's slot to locate its buffer + record offset.
                // cb index == submission order; find it via the matching iocb pointer.
                struct iocb* cb = reinterpret_cast<struct iocb*>(evs[e].obj);
                int slot = (int)(cb - cbs.data());
                if (slot < 0 || slot >= n) continue;
                std::uint8_t* buf = t_buf + (size_t)slot * (2 * ALIGN);
                std::uint64_t recOff = 8 + (std::uint64_t)vids[i] * recBytes;
                const ValueType* v = reinterpret_cast<const ValueType*>(buf + (recOff - aStartOf[slot]));
                queryResults.AddPoint(vids[i], rawDist(rawQuery, v, dim));
            }
            return true;
        }

        // Fetch the survivor vectors from the single canonical vector store (RocksDB
        // KV when compiled, else mmap point store) and exact-rerank them by vid.
        void RerankFromVecDB(std::vector<int>& vids, const ValueType* rawQuery, int dim,
            COMMON::QueryResultSet<ValueType>& queryResults,
            ExtraWorkSpace::PostingProbeStats* p_workStats = nullptr)
        {
            if (vids.empty()) return;
            if (p_workStats != nullptr) {
                p_workStats->m_rerankCandidates += static_cast<std::uint64_t>(vids.size());
            }
            const bool hasDynamic = std::any_of(vids.begin(), vids.end(), [&](int vid) {
                return HasDynamicVector(vid);
            });
            // Prefer the deep-queue libaio batch over the flat O_DIRECT base (full
            // device queue depth, PipeANN-style). Default on; SPTAG_INPOST_LIBAIO_RERANK=0
            // or an unavailable base/AIO pool falls through to RocksDB / mmap. This makes
            // every rerank site (RaBitQ + OPQ in-posting) use the same fast IO path so
            // estimator comparisons are not confounded by rerank-IO parallelism.
            static const bool s_libaioRerank = []() {
                const char* e = std::getenv("SPTAG_INPOST_LIBAIO_RERANK");
                return (e == nullptr) || (std::atoi(e) != 0);
            }();
            if (!hasDynamic && s_libaioRerank && m_inpostBaseFd >= 0 &&
                RerankBaseDirectBatch(vids, rawQuery, dim, queryResults, p_workStats)) {
                return;
            }
            auto rawDist = COMMON::DistanceCalcSelector<ValueType>(m_opt->m_distCalcMethod);
            if (!hasDynamic && m_opqVecDB) {
                std::vector<std::string> vals;
                std::vector<Helper::AsyncReadRequest> reqs;
                if (m_opqVecDB->MultiGet(vids, &vals, MaxTimeout, &reqs) == ErrorCode::Success
                    && vals.size() == vids.size()) {
                    for (size_t i = 0; i < vids.size(); i++) {
                        const ValueType* v = reinterpret_cast<const ValueType*>(vals[i].data());
                        queryResults.AddPoint(vids[i], rawDist(rawQuery, v, dim));
                    }
                }
            } else if (!hasDynamic && m_psVec) {
                for (int vid : vids) {
                    const ValueType* v = &m_psVec[(size_t)vid * dim];
                    queryResults.AddPoint(vid, rawDist(rawQuery, v, dim));
                }
            } else if (m_inpostBaseFd >= 0 || HasDynamicVectorStore()) {
                for (int vid : vids) {
                    const ValueType* v = ReadBaseVecDirect(vid, dim);
                    if (v != nullptr) queryResults.AddPoint(vid, rawDist(rawQuery, v, dim));
                }
            }
        }

        // Exhaustive OPQ search over a single narrow tag's vids: load ids -> ADC screen
        // all of them -> fetch best-L survivors from the point store -> exact rerank.
        // Returns false (caller falls back) when OPQ off or the tag has no resident vid list.
        bool OPQTagPureSearch(QueryResult& p_queryResults, uint32_t tag)
        {
            static const bool s_dbg = []() { const char* e = std::getenv("SPTAG_SLIM_DEBUG"); return e && e[0] == '1'; }();
            static std::atomic<int> s_dbgCount{0};
            if (s_dbg && s_dbgCount++ < 3)
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                    "[slim dbg] OPQTagPureSearch called tag=%u m_opqPF=%d mapSize=%zu found=%d\n",
                    tag, (int)m_opqPF, m_opqTagVids.size(), (int)(m_opqTagVids.find(tag) != m_opqTagVids.end()));
            if (!m_opqPF) return false;

            // Incremental inserts can push_back into m_opqTagVids[tag] and resize
            // m_opqCodes concurrently. Take a shared lock spanning the map lookup and the
            // candidate scan when the index has been mutated (zero overhead otherwise:
            // before any insert m_opqMutated is false and verified QPS is preserved).
            const bool needLock = m_opqMutated.load(std::memory_order_acquire);
            std::shared_lock<std::shared_timed_mutex> rlock(m_opqMaintLock, std::defer_lock);
            if (needLock) rlock.lock();

            auto it = m_opqTagVids.find(tag);
            if (it == m_opqTagVids.end()) return false;
            if (m_pipePQ && m_opqCodes.empty()) return false;
            const std::vector<int>& vids = it->second;

            COMMON::QueryResultSet<ValueType>& queryResults = *((COMMON::QueryResultSet<ValueType>*) & p_queryResults);
            queryResults.Reset();
            const ValueType* rawQuery = queryResults.GetTarget();
            int dim = m_opt->m_dim;

            // Real extended RaBitQ (rabitq2.bin): exhaustive estimator screen over the
            // tag's vids -> best-L survivors -> exact rerank from the canonical vector
            // store. Half the SQ memory, recall restored to ~1.0 via the rerank.
            if (m_rbq2on) {
                static const bool s_rbq2AdcOnly = []() { const char* e = std::getenv("SPTAG_OPQ_ADC_ONLY"); return e && e[0] == '1'; }();
                void* qctx = m_rbq2->AllocQuery();
                // Widen uint8 query to float before PrepareQuery (see WidenQuery).
                std::vector<float> rbq2qf = WidenQuery(rawQuery, dim);
                m_rbq2->PrepareQuery(qctx, rbq2qf.data());
                std::priority_queue<std::pair<float, int>> heap;
                for (int vid : vids) {
                    if (vid < 0 || vid >= m_opqN) continue;
                    if (m_versionMap->Deleted(vid)) continue;
                    float adc = m_rbq2->Estimate(qctx, vid);
                    if ((int)heap.size() < m_opqL) heap.push({ adc, vid });
                    else if (adc < heap.top().first) { heap.pop(); heap.push({ adc, vid }); }
                }
                m_rbq2->FreeQuery(qctx);
                if (needLock) rlock.unlock();
                if (s_rbq2AdcOnly) {
                    // True no-rerank: emit the top-L by the RaBitQ estimator directly,
                    // zero survivor IO. Recall is the estimator's coverage*ranking ceiling.
                    while (!heap.empty()) { queryResults.AddPoint(heap.top().second, heap.top().first); heap.pop(); }
                } else {
                    std::vector<int> survivors;
                    survivors.reserve(heap.size());
                    while (!heap.empty()) { survivors.push_back(heap.top().second); heap.pop(); }
                    RerankFromVecDB(survivors, rawQuery, dim, queryResults);
                }
                queryResults.SortResult();
                return true;
            }

            // RaBitQ path: high-recall, no rerank, no survivor IO. Scan the tenant's vids,
            // reconstruct each code in rotated space and take top-k by L2.
            if (m_rbq) {
                std::vector<float> rq;
                RaBitQRotateQuery(rawQuery, rq);
                int scanned = 0;
                for (int vid : vids) {
                    if (vid < 0 || vid >= m_opqN) continue;
                    if (m_versionMap->Deleted(vid)) continue;
                    queryResults.AddPoint(vid, RaBitQDist(rq, vid));
                    ++scanned;
                }
                if (needLock) rlock.unlock();
                queryResults.SortResult();
                return true;
            }

            std::vector<float> lut((size_t)m_opqM * m_opqKs);
            {
                std::vector<float> qf = WidenQuery(rawQuery, dim);
                if (m_opt->m_distCalcMethod == DistCalcMethod::Cosine)
                    COMMON::Utils::Normalize<float>(qf.data(), dim, COMMON::Utils::GetBase<float>());
                if (m_pipePQ) {
                    if (!m_pipePQTable || m_pipePQTable->Dim() != dim) return false;
                    m_pipePQTable->PopulateDistances(qf.data(), lut.data(), m_opt->m_distCalcMethod);
                } else {
                    m_opqQ->QuantizeVector(qf.data(), (std::uint8_t*)lut.data(), true);
                }
            }
            auto rawDist = COMMON::DistanceCalcSelector<ValueType>(m_opt->m_distCalcMethod);
            (void)rawDist;

            std::priority_queue<std::pair<float, int>> heap;
            for (int vid : vids) {
                if (vid < 0 || vid >= m_opqN) continue;
                if (m_versionMap->Deleted(vid)) continue;
                const std::uint8_t* c = &m_opqCodes[(size_t)vid * m_opqM];
                float adc = 0;
                for (int m = 0; m < m_opqM; m++) adc += lut[(size_t)m * m_opqKs + c[m]];
                if ((int)heap.size() < m_opqL) heap.push({ adc, vid });
                else if (adc < heap.top().first) { heap.pop(); heap.push({ adc, vid }); }
            }
            const size_t vidsCount = vids.size();
            if (needLock) rlock.unlock();

            int fetched = (int)heap.size();
            int listElements = fetched;
            static const bool s_adcOnly = []() { const char* e = std::getenv("SPTAG_OPQ_ADC_ONLY"); return e && e[0] == '1'; }();
            if (s_adcOnly) {
                while (!heap.empty()) { queryResults.AddPoint(heap.top().second, heap.top().first); heap.pop(); }
            } else {
                std::vector<int> survivors;
                survivors.reserve(fetched);
                while (!heap.empty()) { survivors.push_back(heap.top().second); heap.pop(); }
                RerankFromVecDB(survivors, rawQuery, dim, queryResults);
            }
            queryResults.SetScanned(listElements);
            queryResults.SortResult();

            static const bool s_opqStats = []() { const char* e = std::getenv("SPTAG_OPQ_STATS"); return e && e[0] == '1'; }();
            if (s_opqStats) {
                static std::atomic<size_t> g_idbytes{ 0 }, g_fetch{ 0 }, g_q{ 0 };
                size_t q = ++g_q;
                g_idbytes += vidsCount * sizeof(int);
                g_fetch += (size_t)fetched;
                if (q % 1000 == 0) {
                    size_t ib = g_idbytes, ft = g_fetch;
                    size_t bytes = ib + ft * dim * sizeof(ValueType);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "[OPQ tagpure stats] q=%zu idBytes/q=%.0f fetched/q=%.1f totalBytes/q=%.0f\n",
                        q, (double)ib / q, (double)ft / q, (double)bytes / q);
                }
            }
            return true;
        }

        std::int64_t GetOPQTagVidCount(std::uint32_t tag) override
        {
            if (!m_opqPF) return -1;
            const bool needLock = m_opqMutated.load(std::memory_order_acquire);
            std::shared_lock<std::shared_timed_mutex> rlock(m_opqMaintLock, std::defer_lock);
            if (needLock) rlock.lock();
            auto it = m_opqTagVids.find(tag);
            if (it == m_opqTagVids.end()) return -1;
            return (std::int64_t)it->second.size();
        }

        std::int64_t GetOPQTotalVectors() override
        {
            return m_opqPF ? (std::int64_t)m_opqN : -1;
        }

        bool GetRaBitQEnabled() override { return m_rbq; }

    private:

        int m_metaDataSize = 0;

        
        int m_vectorInfoSize = 0;

        int m_postingSizeLimit = INT_MAX;

        int m_bufferSizeLimit = INT_MAX;
        int m_tailBufferSizeLimit = 0;

        // ---- OPQ prefilter state (metadata-only posting + resident codes + point store) ----
        bool m_opqPF = false;
        bool m_slimPostings = false;
        bool m_slimWritable = false;
        bool m_slimSelfTest = false;
        std::unordered_map<SizeType, std::string> m_slimSelfTestSnapshot;
        std::shared_ptr<COMMON::IQuantizer> m_opqQ;
        std::shared_ptr<COMMON::IQuantizer> m_opqQEnc;  // ADC-disabled encoder for incremental inserts
        bool m_pipePQ = false;                          // PipeANN fixed-chunk PQ screen in the OPQ-compatible path
        std::unique_ptr<PipePQTable> m_pipePQTable;
        bool m_opqDynamic = false;                      // incremental OPQ maintenance enabled (writable + encoder)
        std::atomic<bool> m_opqMutated{false};          // set true after first insert; gates search-side lock
        std::shared_timed_mutex m_opqMaintLock;         // search: shared; insert maintenance: unique
        int m_opqM = 0;
        int m_opqKs = 256;
        int m_opqL = 64;
        SizeType m_opqN = 0;
        int m_slimRec = 0;
        bool m_opqInpostCode = false;                  // SPTAG_OPQ_INPOST_CODE=1: code lives in the slim record [meta|code], no resident m_opqCodes
        bool m_opqInpostDb = false;                    // SPTAG_OPQ_INPOST_DB=<M>: slim [meta|code] records live IN the posting store db, read via async MultiGet (NOT mmap opq_slim.bin)
        int m_opqInpostDbM = 0;                        // OPQ subvector count M for the in-db slim code
        std::vector<std::uint8_t> m_opqCodes;          // [vid*M], resident (unused when m_opqInpostCode)
        const ValueType* m_psVec = nullptr;            // point store mmap [vid*dim]
        const std::uint8_t* m_slim = nullptr;          // slim postings mmap
        const std::uint64_t* m_slimOff = nullptr;      // per-head byte offsets, len postingNum+1
        int m_slimDirectFd = -1;                       // O_DIRECT fd for fair device-bound posting IO (SPTAG_SLIM_DIRECT_IO=1)
        std::shared_ptr<Helper::KeyValueIO> m_opqVecDB;  // canonical vid -> vector store (RocksDB)
        std::unordered_map<uint32_t, std::vector<int>> m_opqTagVids;  // exhaustive tag -> vids (narrow path)

        // ---- RaBitQ prefilter (drop-in replacement for PQ ADC; high-recall, no rerank) ----
        bool m_rbq = false;                    // RaBitQ search enabled (SPTAG_RABITQ=1)
        int m_rbqBits = 0;                     // bits per dim
        int m_rbqDim = 0;                      // rotated/padded dim (== dim for MatrixRotator)
        std::vector<float> m_rbqRot;           // rotation matrix [dim*pdim], row-major: rq = v * R
        std::vector<std::uint8_t> m_rbqCodes;  // [vid*pdim] per-dim scalar codes
        std::vector<float> m_rbqDelta;         // [vid] reconstruction step
        std::vector<float> m_rbqVl;            // [vid] reconstruction low bound

        // ---- Real extended RaBitQ (split: 1-bit packed + ex-bits) + exact rerank ----
        // When rabitq2.bin is present, this replaces the SQ-style m_rbq screen: the
        // estimator screens candidates (best-L max-heap) and the L survivors are exact-
        // reranked from m_opqVecDB. Smaller memory & near-exact recall vs SQ7.
        std::unique_ptr<RaBitQ2> m_rbq2;       // null unless rabitq2.bin loaded
        bool m_rbq2on = false;

        // LatencyLimit is a mutable search parameter. Derive the deadline at
        // use time so TenantIndexManager::SetSearchParam takes effect after
        // the index and its extra searcher are already loaded.
        std::chrono::microseconds HardLatencyLimit() const
        {
            return std::chrono::microseconds(
                static_cast<int>(std::max(0.0f, m_opt->m_latencyLimit) * 1000.0f));
        }

        int m_mergeThreshold = 10;
        ErrorCode m_asyncStatus = ErrorCode::Success;

	    COMMON::Dataset<std::uint64_t>* m_vectorTranslateMap;

        std::vector<std::vector<SizeType>> m_plannedNodeVectorAssignments;
        std::vector<std::vector<SizeType>> m_primaryNodeVectorAssignments;
        std::unordered_map<SizeType, int> m_headVectorOwners;

        std::shared_ptr<SPDKThreadPool> m_splitThreadPool;
        std::shared_ptr<SPDKThreadPool> m_reassignThreadPool;
    };
} // namespace SPTAG
#endif // _SPTAG_SPANN_EXTRADYNAMICSEARCHER_H_
