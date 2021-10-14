#pragma once
#include "inc/Core/VectorIndex.h"
#include "inc/Core/Common/WorkSpace.h"
#include "inc/Core/Common/QueryResultSet.h"
#include "inc/SSDServing/VectorSearch/SearchStats.h"

#ifndef _MSC_VER
#include "inc/SSDServing/VectorSearch/AsyncFileReaderLinux.h"
#else
#include "inc/SSDServing/VectorSearch/AsyncFileReader.h"
#endif

#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif // defined(__GNUC__)

#include <memory>
#include <vector>

namespace SPTAG {
    namespace SSDServing {
        namespace VectorSearch {
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
                        m_pageBuffer.reset(static_cast<T*>(_mm_malloc(sizeof(T) * m_pageBufferSize, 512)), [=](T* ptr) { _mm_free(ptr); });
                    }
                }

                T* GetBuffer()
                {
                    return m_pageBuffer.get();
                }


            private:
                std::shared_ptr<T> m_pageBuffer;

                std::size_t m_pageBufferSize;
            };

            struct DiskListRequest : public SPTAG::Helper::AsyncReadRequest
            {
                void* m_pListInfo;
            };
            
            struct ExtraWorkSpace
            {
                ExtraWorkSpace() {}

                ~ExtraWorkSpace() {}

                std::vector<int> m_postingIDs;

                COMMON::OptHashPosVector m_deduper;

                RequestQueue m_processIocp;

                std::vector<PageBuffer<std::uint8_t>> m_pageBuffers;

                std::vector<DiskListRequest> m_diskRequests;
            };


            template<typename ValueType>
            class IExtraSearcher
            {
            public:
                IExtraSearcher()
                {
                }


                virtual ~IExtraSearcher()
                {
                }

                virtual size_t GetMaxListSize() const = 0;

                virtual void Search(ExtraWorkSpace* p_exWorkSpace,
                    COMMON::QueryResultSet<ValueType>& p_queryResults,
                    std::shared_ptr<VectorIndex> p_index,
                    SearchStats& p_stats) = 0;
            };
        }
    }
}