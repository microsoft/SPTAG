// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_RESULT_ITERATOR_H
#define _SPTAG_SPANN_RESULT_ITERATOR_H

#include<memory>

#include "inc/Core/SPANN/Index.h"
#include "inc/Core/SearchQuery.h"
#include "inc/Core/ResultIterator.h"
#include "inc/Core/Common/WorkSpace.h"
#include "inc/Core/SPANN/IExtraSearcher.h"

namespace SPTAG
{
	namespace SPANN
	{
		template<typename T>
		class SPANNResultIterator : public ResultIterator
		{
		public:
			SPANNResultIterator(const Index<T>* p_spannIndex, const VectorIndex* p_index, const void* p_target,
				std::unique_ptr<SPANN::ExtraWorkSpace> p_extraWorkspace,
				int p_batch): ResultIterator(p_index, p_target, false, p_batch),
				m_spannIndex(p_spannIndex),
				m_extraWorkspace(std::move(p_extraWorkspace))
		    {
			    m_headQueryResult = std::make_unique<QueryResult>(p_target, p_batch, false);
		    }

			~SPANNResultIterator()
			{
				Close();
			}

			virtual std::shared_ptr<QueryResult> Next(int batch)
			{
				if (m_queryResult == nullptr) {
					m_queryResult = std::make_unique<QueryResult>(m_target, batch, true);
				}
				else if (batch <= m_queryResult->GetResultNum()) {
					m_queryResult->SetResultNum(batch);
				}
				else {
					batch = m_queryResult->GetResultNum();
				}

			     m_queryResult->Reset();
				 if (m_workspace == nullptr) return m_queryResult;

				 int resultCount = 0;
			     m_spannIndex->SearchIndexIterative(*m_headQueryResult, *m_queryResult,
					 (COMMON::WorkSpace*)GetWorkSpace(), m_extraWorkspace.get(), batch, resultCount, m_isFirstResult);
			     m_isFirstResult = false;

				 for (int i = 0; i < resultCount; i++)
				 {
					 m_queryResult->GetResult(i)->RelaxedMono = m_extraWorkspace->m_relaxedMono;
				 }
				 m_queryResult->SetResultNum(resultCount);
			     return m_queryResult;	
			}

			virtual bool GetRelaxedMono() 
			{
				if (m_extraWorkspace == nullptr) return false;

				return m_extraWorkspace->m_relaxedMono;
			}

			virtual void Close()
			{
				ResultIterator::Close();
			   if (m_extraWorkspace != nullptr) {
				   m_spannIndex->SearchIndexIterativeEnd(std::move(m_extraWorkspace));
				   m_extraWorkspace = nullptr;
			   }
			}

		private:
			const Index<T>* m_spannIndex;
			std::unique_ptr<QueryResult> m_headQueryResult;
			std::unique_ptr<SPANN::ExtraWorkSpace> m_extraWorkspace;
		};
	}// namespace SPANN
} // namespace SPTAG
#endif
