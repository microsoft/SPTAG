// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_IEXTRASEARCHER_H_
#define _SPTAG_SPANN_IEXTRASEARCHER_H_

#include "Options.h"

#include "inc/Core/VectorIndex.h"
#include "inc/Core/Common/VersionLabel.h"
#include "inc/Helper/AsyncFileReader.h"

#include <memory>
#include <vector>
#include <chrono>
#include <atomic>
#include <set>

namespace SPTAG {
    namespace SPANN {

        struct SearchStats
        {
            SearchStats()
                : m_check(0),
                m_exCheck(0),
                m_totalListElementsCount(0),
                m_diskIOCount(0),
                m_diskAccessCount(0),
                m_totalSearchLatency(0),
                m_totalLatency(0),
                m_exLatency(0),
                m_asyncLatency0(0),
                m_asyncLatency1(0),
                m_asyncLatency2(0),
                m_queueLatency(0),
                m_sleepLatency(0)
            {
            }

            int m_check;

            int m_exCheck;

            int m_totalListElementsCount;

            int m_diskIOCount;

            int m_diskAccessCount;

            double m_totalSearchLatency;

            double m_totalLatency;

            double m_exLatency;

            double m_asyncLatency0;

            double m_asyncLatency1;

            double m_asyncLatency2;

            double m_queueLatency;

            double m_sleepLatency;

            double m_compLatency;

            double m_diskReadLatency;

            double m_exSetUpLatency;

            std::chrono::steady_clock::time_point m_searchRequestTime;

            int m_threadID;
        };

        struct IndexStats {
            std::atomic_uint32_t m_headMiss{ 0 };
            uint32_t m_appendTaskNum{ 0 };
            uint32_t m_splitNum{ 0 };
            uint32_t m_theSameHeadNum{ 0 };
            uint32_t m_reAssignNum{ 0 };
            uint32_t m_garbageNum{ 0 };
            uint64_t m_reAssignScanNum{ 0 };
            uint32_t m_mergeNum{ 0 };

            //Split
            double m_splitCost{ 0 };
            double m_getCost{ 0 };
            double m_putCost{ 0 };
            double m_clusteringCost{ 0 };
            double m_updateHeadCost{ 0 };
            double m_reassignScanCost{ 0 };
            double m_reassignScanIOCost{ 0 };

            // Append
            double m_appendCost{ 0 };
            double m_appendIOCost{ 0 };

            // reAssign
            double m_reAssignCost{ 0 };
            double m_selectCost{ 0 };
            double m_reAssignAppendCost{ 0 };

            // GC
            double m_garbageCost{ 0 };

            void PrintStat(int finishedInsert, bool cost = false, bool reset = false) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "After %d insertion, head vectors split %d times, head missing %d times, same head %d times, reassign %d times, reassign scan %ld times, garbage collection %d times, merge %d times\n",
                    finishedInsert, m_splitNum, m_headMiss.load(), m_theSameHeadNum, m_reAssignNum, m_reAssignScanNum, m_garbageNum, m_mergeNum);

                if (cost) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "AppendTaskNum: %d, TotalCost: %.3lf us, PerCost: %.3lf us\n", m_appendTaskNum, m_appendCost, m_appendCost / m_appendTaskNum);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "AppendTaskNum: %d, AppendIO TotalCost: %.3lf us, PerCost: %.3lf us\n", m_appendTaskNum, m_appendIOCost, m_appendIOCost / m_appendTaskNum);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SplitNum: %d, TotalCost: %.3lf ms, PerCost: %.3lf ms\n", m_splitNum, m_splitCost, m_splitCost / m_splitNum);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SplitNum: %d, Read TotalCost: %.3lf us, PerCost: %.3lf us\n", m_splitNum, m_getCost, m_getCost / m_splitNum);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SplitNum: %d, Clustering TotalCost: %.3lf us, PerCost: %.3lf us\n", m_splitNum, m_clusteringCost, m_clusteringCost / m_splitNum);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SplitNum: %d, UpdateHead TotalCost: %.3lf ms, PerCost: %.3lf ms\n", m_splitNum, m_updateHeadCost, m_updateHeadCost / m_splitNum);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SplitNum: %d, Write TotalCost: %.3lf us, PerCost: %.3lf us\n", m_splitNum, m_putCost, m_putCost / m_splitNum);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SplitNum: %d, ReassignScan TotalCost: %.3lf ms, PerCost: %.3lf ms\n", m_splitNum, m_reassignScanCost, m_reassignScanCost / m_splitNum);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SplitNum: %d, ReassignScanIO TotalCost: %.3lf us, PerCost: %.3lf us\n", m_splitNum, m_reassignScanIOCost, m_reassignScanIOCost / m_splitNum);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "GCNum: %d, TotalCost: %.3lf us, PerCost: %.3lf us\n", m_garbageNum, m_garbageCost, m_garbageCost / m_garbageNum);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "ReassignNum: %d, TotalCost: %.3lf us, PerCost: %.3lf us\n", m_reAssignNum, m_reAssignCost, m_reAssignCost / m_reAssignNum);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "ReassignNum: %d, Select TotalCost: %.3lf us, PerCost: %.3lf us\n", m_reAssignNum, m_selectCost, m_selectCost / m_reAssignNum);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "ReassignNum: %d, ReassignAppend TotalCost: %.3lf us, PerCost: %.3lf us\n", m_reAssignNum, m_reAssignAppendCost, m_reAssignAppendCost / m_reAssignNum);
                }

                if (reset) {
                    m_splitNum = 0;
                    m_headMiss = 0;
                    m_theSameHeadNum = 0;
                    m_reAssignNum = 0;
                    m_reAssignScanNum = 0;
                    m_mergeNum = 0;
                    m_garbageNum = 0;
                    m_appendTaskNum = 0;
                    m_splitCost = 0;
                    m_clusteringCost = 0;
                    m_garbageCost = 0;
                    m_updateHeadCost = 0;
                    m_getCost = 0;
                    m_putCost = 0;
                    m_reassignScanCost = 0;
                    m_reassignScanIOCost = 0;
                    m_appendCost = 0;
                    m_appendIOCost = 0;
                    m_reAssignCost = 0;
                    m_selectCost = 0;
                    m_reAssignAppendCost = 0;
                }
            }
        };

        template<typename T>
        class PageBuffer
        {
        public:
            PageBuffer()
                : m_pageBufferSize(0)
            {
            }

            void ReservePageBuffer(std::size_t p_size)
            {
                if (m_pageBufferSize < p_size)
                {
                    m_pageBufferSize = p_size;
                    m_pageBuffer.reset(static_cast<T*>(PAGE_ALLOC(sizeof(T) * m_pageBufferSize)), [=](T* ptr) { PAGE_FREE(ptr); });
                }
            }

            T* GetBuffer()
            {
                return m_pageBuffer.get();
            }

            std::size_t GetPageSize()
            {
                return m_pageBufferSize;
            }

        private:
            std::shared_ptr<T> m_pageBuffer;

            std::size_t m_pageBufferSize;
        };

        struct ExtraWorkSpace : public SPTAG::COMMON::IWorkSpace
        {
            ExtraWorkSpace() {}

            ~ExtraWorkSpace() { g_spaceCount--; }

            ExtraWorkSpace(ExtraWorkSpace& other) {
                Initialize(other.m_deduper.MaxCheck(), other.m_deduper.HashTableExponent(), (int)other.m_pageBuffers.size(), (int)(other.m_pageBuffers[0].GetPageSize()), other.m_enableDataCompression);
            }

            void Initialize(int p_maxCheck, int p_hashExp, int p_internalResultNum, int p_maxPages, bool enableDataCompression) {
                m_postingIDs.reserve(p_internalResultNum);
                m_deduper.Init(p_maxCheck, p_hashExp);
                m_processIocp.reset(p_internalResultNum);
                m_pageBuffers.resize(p_internalResultNum);
                for (int pi = 0; pi < p_internalResultNum; pi++) {
                    m_pageBuffers[pi].ReservePageBuffer(p_maxPages);
                }
                m_diskRequests.resize(p_internalResultNum);
                for (int pi = 0; pi < p_internalResultNum; pi++) {
                    m_diskRequests[pi].m_extension = m_processIocp.handle();
                }
                m_enableDataCompression = enableDataCompression;
                if (enableDataCompression) {
                    m_decompressBuffer.ReservePageBuffer(p_maxPages);
                }
                m_spaceID = g_spaceCount++;
                m_relaxedMono = false;
            }

            void Initialize(va_list& arg) {
                int maxCheck = va_arg(arg, int);
                int hashExp = va_arg(arg, int);
                int internalResultNum = va_arg(arg, int);
                int maxPages = va_arg(arg, int);
                bool enableDataCompression = bool(va_arg(arg, int));
                Initialize(maxCheck, hashExp, internalResultNum, maxPages, enableDataCompression);
            }

            void Clear(int p_internalResultNum, int p_maxPages, bool enableDataCompression) {
                if (p_internalResultNum > m_pageBuffers.size()) {
                    m_postingIDs.reserve(p_internalResultNum);
                    m_processIocp.reset(p_internalResultNum);
                    m_pageBuffers.resize(p_internalResultNum);
                    for (int pi = 0; pi < p_internalResultNum; pi++) {
                        m_pageBuffers[pi].ReservePageBuffer(p_maxPages);
                    }
                    m_diskRequests.resize(p_internalResultNum);
                    for (int pi = 0; pi < p_internalResultNum; pi++) {
                        m_diskRequests[pi].m_extension = m_processIocp.handle();
                    }
                } else if (p_maxPages > m_pageBuffers[0].GetPageSize()) {
                    for (int pi = 0; pi < m_pageBuffers.size(); pi++) m_pageBuffers[pi].ReservePageBuffer(p_maxPages);
                }

                m_enableDataCompression = enableDataCompression;
                if (enableDataCompression) {
                    m_decompressBuffer.ReservePageBuffer(p_maxPages);
                }
            }

            static void Reset() { g_spaceCount = 0; }

            std::vector<int> m_postingIDs;

            COMMON::OptHashPosVector m_deduper;

            Helper::RequestQueue m_processIocp;

            std::vector<PageBuffer<std::uint8_t>> m_pageBuffers;

            bool m_enableDataCompression;
            PageBuffer<std::uint8_t> m_decompressBuffer;

            std::vector<Helper::AsyncReadRequest> m_diskRequests;

            int m_spaceID;

            uint32_t m_pi;

            int m_offset;

            bool m_loadPosting;

            bool m_relaxedMono;

            int m_loadedPostingNum;

            static std::atomic_int g_spaceCount;
        };

        class IExtraSearcher
        {
        public:
            IExtraSearcher()
            {
            }

            ~IExtraSearcher()
            {
            }

            virtual bool LoadIndex(Options& p_options, COMMON::VersionLabel& p_versionMap, std::shared_ptr<std::uint64_t> m_vectorTranslateMap,  std::shared_ptr<VectorIndex> m_index) = 0;

            virtual void SearchIndex(ExtraWorkSpace* p_exWorkSpace,
                QueryResult& p_queryResults,
                std::shared_ptr<VectorIndex> p_index,
                SearchStats* p_stats, std::set<int>* truth = nullptr, std::map<int, std::set<int>>* found = nullptr) = 0;

            virtual bool SearchIterativeNext(ExtraWorkSpace* p_exWorkSpace,
                QueryResult& p_queryResults,
                std::shared_ptr<VectorIndex> p_index) = 0;

            virtual void SearchIndexWithoutParsing(ExtraWorkSpace* p_exWorkSpace) = 0;

            virtual bool SearchNextInPosting(ExtraWorkSpace* p_exWorkSpace,
                QueryResult& p_queryResults,
		std::shared_ptr<VectorIndex>& p_index) = 0;

            virtual bool BuildIndex(std::shared_ptr<Helper::VectorSetReader>& p_reader, 
                std::shared_ptr<VectorIndex> p_index, 
                Options& p_opt, COMMON::VersionLabel& p_versionMap, SizeType upperBound = -1) = 0;

            virtual ErrorCode GetPostingDebug(ExtraWorkSpace* p_exWorkSpace, std::shared_ptr<VectorIndex> p_index, SizeType vid, std::vector<SizeType>& VIDs, std::shared_ptr<VectorSet>& vecs) = 0;
            
            virtual void RefineIndex(std::shared_ptr<Helper::VectorSetReader>& p_reader,
                std::shared_ptr<VectorIndex> p_index) { return; }
            virtual ErrorCode AddIndex(std::shared_ptr<VectorSet>& p_vectorSet,
                std::shared_ptr<VectorIndex> p_index, SizeType p_begin) { return ErrorCode::Undefined; }
            virtual ErrorCode DeleteIndex(SizeType p_id) { return ErrorCode::Undefined; }

            virtual bool AllFinished() { return false; }
            virtual void GetDBStats() { return; }
            virtual void GetIndexStats(int finishedInsert, bool cost, bool reset) { return; }
            virtual void ForceCompaction() { return; }

            virtual bool CheckValidPosting(SizeType postingID) = 0;
            virtual SizeType SearchVector(std::shared_ptr<VectorSet>& p_vectorSet,
                std::shared_ptr<VectorIndex> p_index, int testNum = 64, SizeType VID = -1) { return -1; }
            virtual void ForceGC(VectorIndex* p_index) { return; }

            virtual void GetWritePosting(SizeType pid, std::string& posting, bool write = false) { return; }

            virtual bool Initialize() { return false; }

            virtual bool ExitBlockController() { return false; }

            virtual void Checkpoint(std::string prefix) { return; }
        };
    } // SPANN
} // SPTAG

#endif // _SPTAG_SPANN_IEXTRASEARCHER_H_
