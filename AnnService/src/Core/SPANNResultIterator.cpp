/*
#include "inc/Core/SPANNResultIterator.h"

namespace SPTAG
{
	namespace SPANN
	{
		template<typename T>
		SPANNResultIterator<T>::SPANNResultIterator(const Index<T>* p_index, const void* p_target,
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
                
		template<typename T>
		SPANNResultIterator<T>::~SPANNResultIterator()
		{
			if (m_index != nullptr && m_headWorkspace != nullptr && m_extraWorkspace != nullptr) {
				m_index->SearchIndexIterativeEnd(m_headWorkspace, m_extraWorkspace);
			}
			m_headQueryResult = nullptr;
			m_queryResult = nullptr;
		}

		template<typename T>
		bool SPANNResultIterator<T>::Next(BasicResult& result)
		{
			m_queryResult->Reset();
			m_index->SearchIndexIterativeNext(*m_headQueryResult, *m_queryResult, m_headWorkspace, m_extraWorkspace, m_isFirstResult);
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

		// Add end into destructor.
		template<typename T>
		void SPANNResultIterator<T>::Close()
		{
			if (m_headWorkspace != nullptr && m_extraWorkspace != nullptr) {
				m_index->SearchIndexIterativeEnd(m_headWorkspace, m_extraWorkspace);
				m_headWorkspace = nullptr;
				m_extraWorkspace = nullptr;
			}
		}

                template<typename T>
		QueryResult* SPANNResultIterator<T>::GetQuery() const
		{
			return m_queryResult.get();
		}
	} // namespace SPANN
} // namespace SPTAG
*/
