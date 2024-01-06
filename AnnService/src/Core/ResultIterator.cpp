#include "inc/Core/ResultIterator.h"



	struct UniqueHandler {
		std::unique_ptr<SPTAG::COMMON::WorkSpace> m_handler;
	};

	ResultIterator::ResultIterator(const void* index, const void* p_target, bool searchDeleted)
		:m_index((const VectorIndex*)index),
		m_target(p_target),
		m_searchDeleted(searchDeleted)
	{
		m_workspace = new UniqueHandler;
		((UniqueHandler*)m_workspace)->m_handler = std::move(m_index->RentWorkSpace(1));
		m_isFirstResult = true;
	}

	ResultIterator::~ResultIterator()
	{
		if (m_index != nullptr && m_workspace != nullptr) {
			m_index->SearchIndexIterativeEnd(std::move(((UniqueHandler*)m_workspace)->m_handler));
			delete m_workspace;
			m_workspace = nullptr;
		}
		m_queryResult = nullptr;
	}

	std::shared_ptr<QueryResult> ResultIterator::Next(int batch)
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
        
		if (m_workspace == nullptr) {
			m_queryResult->Reset();
			return m_queryResult;
		}
		
		int resultCount = 0;
		m_index->SearchIndexIterativeNextBatch(*m_queryResult, (((UniqueHandler*)m_workspace)->m_handler).get(), batch, resultCount, m_isFirstResult, m_searchDeleted);
		m_isFirstResult = false;
		for (int i = 0; i < resultCount; i++)
		{
			m_queryResult->GetResult(i)->RelaxedMono = (((UniqueHandler*)m_workspace)->m_handler)->m_relaxedMono;
		}
		m_queryResult->SetResultNum(resultCount);
		return m_queryResult;
	}

	bool ResultIterator::GetRelaxedMono()
	{
		if (m_workspace == nullptr) return false;

		return (((UniqueHandler*)m_workspace)->m_handler)->m_relaxedMono;
	}

	// Add end into destructor.
	void ResultIterator::Close()
	{
		if (m_workspace != nullptr) {
			m_index->SearchIndexIterativeEnd(std::move(((UniqueHandler*)m_workspace)->m_handler));
			delete m_workspace;
			m_workspace = nullptr;
		}
	}

	const void* ResultIterator::GetTarget()
	{
		return m_target;
	}
 // namespace SPTAG
