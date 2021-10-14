#pragma once
#include <memory>
#include <vector>

#include "inc/SSDServing/VectorSearch/SearchStats.h"
#include "inc/SSDServing/VectorSearch/VectorSearchUtils.h"
#include "inc/Core/Common/QueryResultSet.h"
#include "inc/Core/VectorIndex.h"

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


            struct ExtraWorkSpace
            {
                ExtraWorkSpace() {
                }

                ~ExtraWorkSpace() {

                }

                std::vector<int> m_postingIDs;

                HashBasedDeduper m_deduper;

                std::vector<PageBuffer<std::uint8_t>> m_pageBuffers;
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

                virtual void Setup(Options& p_config) = 0;

                virtual void FinishPrepare()
                {
                }

                virtual void Search(ExtraWorkSpace* p_exWorkSpace,
                    SPTAG::COMMON::QueryResultSet<ValueType>& p_queryResults,
                    std::shared_ptr<VectorIndex> p_index,
                    SearchStats& p_stats) = 0;
            };
        }
    }
}