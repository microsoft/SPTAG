// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_RESULT_ITERATOR_H
#define _SPTAG_RESULT_ITERATOR_H

#include<memory>

#include "VectorIndex.h"
#include "SearchQuery.h"
#include "Common/WorkSpace.h"

namespace SPTAG
{
class ResultIterator
{
public:
	ResultIterator(const VectorIndex* index, const void* p_target,
		std::shared_ptr<COMMON::WorkSpace> workspace, bool searchDeleted);
	~ResultIterator();
	bool Next(BasicResult& result);
	void Close();
	QueryResult* GetQuery() const;
private:
	const VectorIndex* m_index;
	const void* m_target;
	ByteArray m_byte_target;
	std::unique_ptr<QueryResult> m_queryResult;
	std::shared_ptr<COMMON::WorkSpace> m_workspace;
	bool m_searchDeleted;
	bool m_isFirstResult;
	int m_batch = 1;
};
} // namespace SPTAG
#endif