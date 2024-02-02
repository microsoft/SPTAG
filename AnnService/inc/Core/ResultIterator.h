// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_RESULT_ITERATOR_H
#define _SPTAG_RESULT_ITERATOR_H

#include<memory>

#include "VectorIndex.h"
#include "SearchQuery.h"

typedef SPTAG::VectorIndex VectorIndex;
typedef SPTAG::ByteArray ByteArray;
typedef SPTAG::QueryResult QueryResult;

class ResultIterator
{
public:
	ResultIterator(const void* p_index, const void* p_target, bool p_searchDeleted, int p_workspaceBatch);

	~ResultIterator();
	
	void* GetWorkSpace();

	virtual std::shared_ptr<QueryResult> Next(int batch);
	
	virtual bool GetRelaxedMono();
	
	virtual void Close();

	const void* GetTarget();

protected:
	const VectorIndex* m_index;
	const void* m_target;
	ByteArray m_byte_target;
	std::shared_ptr<QueryResult> m_queryResult;
	void* m_workspace;
	bool m_searchDeleted;
	bool m_isFirstResult;
	int m_batch = 1;
};

#endif