#include "inc/Core/ResultIterator.h"

namespace SPTAG
{
	ResultIterator::ResultIterator(const VectorIndex* index, const void* p_target,
		std::shared_ptr<COMMON::WorkSpace> workspace, bool searchDeleted)
		:m_index(index),
		m_target(p_target),
		m_workspace(workspace),
		m_searchDeleted(searchDeleted)
	{
		// TODO(qiazh): optimize batch instead of 1
		m_queryResult = std::make_unique<QueryResult>(p_target, 1, true);
		m_isFirstResult = true;
	}

	ResultIterator::~ResultIterator()
	{
		if (m_index != nullptr && m_workspace != nullptr) {
			m_index->SearchIndexIterativeEnd(m_workspace);
		}
		m_queryResult = nullptr;
	}

	bool ResultIterator::Next(BasicResult& result)
	{
		m_queryResult->Reset();
		m_index->SearchIndexIterativeNext(*m_queryResult, m_workspace, m_isFirstResult, m_searchDeleted);
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
	void ResultIterator::Close()
	{
		if (m_workspace != nullptr) {
			m_index->SearchIndexIterativeEnd(m_workspace);
			m_workspace = nullptr;
		}
	}

	QueryResult* ResultIterator::GetQuery() const
	{
		return m_queryResult.get();
	}


} // namespace SPTAG
