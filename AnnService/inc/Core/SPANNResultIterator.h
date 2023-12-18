// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_RESULT_ITERATOR_H
#define _SPTAG_SPANN_RESULT_ITERATOR_H

#include<memory>

#include "SPANN/Index.h"
#include "SearchQuery.h"
#include "Common/WorkSpace.h"
#include "SPANN/IExtraSearcher.h"

namespace SPTAG
{
	namespace SPANN
	{
		template<typename T>
		class SPANNResultIterator
		{
		public:
			SPANNResultIterator(const Index<T>* p_index, const void* p_target,
				std::shared_ptr<COMMON::WorkSpace> headWorkspace,
				std::shared_ptr<SPANN::ExtraWorkSpace> extraWorkspace,
				int batch)
				:m_index(p_index),
			        m_target(p_target),
			        m_headWorkspace(headWorkspace),
			        m_extraWorkspace(extraWorkspace),
			        m_batch(batch)
		        {
			        m_headQueryResult = std::make_unique<QueryResult>(p_target, batch, false);
			        m_queryResult = std::make_unique<QueryResult>(p_target, 1, true);
			        m_isFirstResult = true;
		        }
			~SPANNResultIterator()
			{
                                if (m_index != nullptr && m_headWorkspace != nullptr && m_extraWorkspace != nullptr) {
				    m_index->SearchIndexIterativeEnd(m_headWorkspace, m_extraWorkspace);
			        }
			        m_headQueryResult = nullptr;
			        m_queryResult = nullptr;
			}
			bool Next(BasicResult& result)
			{
			     m_queryResult->Reset();
			     m_index->SearchIndexIterative(*m_headQueryResult, *m_queryResult, m_headWorkspace, m_extraWorkspace, m_isFirstResult);
			     m_isFirstResult = false;
			     if (m_queryResult->GetResult(0) == nullptr || m_queryResult->GetResult(0)->VID < 0)
			     {
				return false;
			     }
			     result.VID = m_queryResult->GetResult(0)->VID;
			     result.Dist = m_queryResult->GetResult(0)->Dist;
			     result.Meta = m_queryResult->GetResult(0)->Meta;
			     return true;	
			}
			void Close()
			{
			   if (m_headWorkspace != nullptr && m_extraWorkspace != nullptr) {
				m_index->SearchIndexIterativeEnd(m_headWorkspace, m_extraWorkspace);
				m_headWorkspace = nullptr;
				m_extraWorkspace = nullptr;
			   }
			}
			QueryResult* GetQuery() const
			{
			   return m_queryResult.get();
			}
		private:
			const Index<T>* m_index;
			const void* m_target;
			ByteArray m_byteTarget;
			std::unique_ptr<QueryResult> m_headQueryResult;
			std::unique_ptr<QueryResult> m_queryResult;
			std::shared_ptr<COMMON::WorkSpace> m_headWorkspace;
			std::shared_ptr<SPANN::ExtraWorkSpace> m_extraWorkspace;
			bool m_isFirstResult;
			int m_batch;
		};
	}// namespace SPTAG
} // namespace SPTAG
#endif
