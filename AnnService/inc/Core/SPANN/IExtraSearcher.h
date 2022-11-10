// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_IEXTRASEARCHER_H_
#define _SPTAG_SPANN_IEXTRASEARCHER_H_

#include "Options.h"

#include "inc/Core/VectorIndex.h"
#include "inc/Helper/AsyncFileReader.h"

#include <memory>
#include <vector>
#include <chrono>
#include <atomic>

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

            std::chrono::steady_clock::time_point m_searchRequestTime;

            int m_threadID;
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

        struct ExtraWorkSpace
        {
            ExtraWorkSpace() {}

            ~ExtraWorkSpace() {}

            ExtraWorkSpace(ExtraWorkSpace& other) {
                Initialize(other.m_deduper.MaxCheck(), other.m_deduper.HashTableExponent(), (int)other.m_pageBuffers.size(), (int)(other.m_pageBuffers[0].GetPageSize()), other.m_enableDataCompression);
                m_spaceID = g_spaceCount++;
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
                m_enableDataCompression = enableDataCompression;
                if (enableDataCompression) {
                    m_decompressBuffer.ReservePageBuffer(p_maxPages);
                }
            }

            void Initialize(va_list& arg) {
                int maxCheck = va_arg(arg, int);
                int hashExp = va_arg(arg, int);
                int internalResultNum = va_arg(arg, int);
                int maxPages = va_arg(arg, int);
                bool enableDataCompression = bool(va_arg(arg, int));
                Initialize(maxCheck, hashExp, internalResultNum, maxPages, enableDataCompression);
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

            static std::atomic_int g_spaceCount;
        };

        class IExtraSearcher
        {
        public:
            IExtraSearcher()
            {
            }

            virtual ~IExtraSearcher()
            {
            }

            virtual bool LoadIndex(Options& p_options) = 0;

            virtual void SearchIndex(ExtraWorkSpace* p_exWorkSpace,
                QueryResult& p_queryResults,
                std::shared_ptr<VectorIndex> p_index,
                SearchStats* p_stats,
                std::set<int>* truth = nullptr,
                std::map<int, std::set<int>>* found = nullptr) = 0;

            virtual bool BuildIndex(std::shared_ptr<Helper::VectorSetReader>& p_reader, 
                std::shared_ptr<VectorIndex> p_index, 
                Options& p_opt) = 0;

            virtual bool CheckValidPosting(SizeType postingID) = 0;
        };
    } // SPANN
} // SPTAG

#endif // _SPTAG_SPANN_IEXTRASEARCHER_H_