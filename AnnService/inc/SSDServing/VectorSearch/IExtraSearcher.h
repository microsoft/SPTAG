#pragma once
#include "inc/Core/VectorIndex.h"

#include "inc/SSDServing/VectorSearch/SearchStats.h"
#include "inc/SSDServing/VectorSearch/VectorSearchUtils.h"
#include "inc/SSDServing/VectorSearch/Options.h"
#include "inc/Core/Common/QueryResultSet.h"
#include "inc/SSDServing/VectorSearch/DiskListCommonUtils.h"
#include "inc/SSDServing/VectorSearch/DiskFileReader.h"

#include <memory>
#include <vector>
#include <functional>
#include <windows.h>

namespace SPTAG {
    namespace SSDServing {
        namespace VectorSearch{
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
                        m_pageBuffer.reset(new T[m_pageBufferSize], std::default_delete<T[]>());
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

            struct DiskListRequest : public DiskFileReadRequest
            {
                bool m_success;

                uint32_t m_requestID;
            };
            
            struct ExtraWorkSpace
            {
                ExtraWorkSpace() {
                    m_processIocp.Reset(::CreateIoCompletionPort(INVALID_HANDLE_VALUE, NULL, NULL, 0));
                }

                ~ExtraWorkSpace() {

                }

                std::vector<int> m_postingIDs;

                HashBasedDeduper m_deduper;

                HandleWrapper m_processIocp;

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

                virtual void InitWorkSpace(ExtraWorkSpace* p_space, int p_resNumHint) = 0;

                virtual void FinishPrepare()
                {
                }

                virtual void Search(ExtraWorkSpace* p_exWorkSpace,
                    COMMON::QueryResultSet<ValueType>& p_queryResults,
                    std::shared_ptr<VectorIndex> p_index,
                    SearchStats& p_stats) = 0;

                virtual void Search(ExtraWorkSpace* p_exWorkSpace,
                    COMMON::QueryResultSet<ValueType>& p_queryResults,
                    std::shared_ptr<VectorIndex> p_index) = 0;
            };
        }
    }
}