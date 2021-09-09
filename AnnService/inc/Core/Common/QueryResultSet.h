// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_QUERYRESULTSET_H_
#define _SPTAG_COMMON_QUERYRESULTSET_H_

#include "../SearchQuery.h"
#include "DistanceUtils.h"
#include <algorithm>

namespace SPTAG
{
namespace COMMON
{

inline bool operator < (const BasicResult& lhs, const BasicResult& rhs)
{
    return ((lhs.Dist < rhs.Dist) || ((lhs.Dist == rhs.Dist) && (lhs.VID < rhs.VID)));
}


inline bool Compare(const BasicResult& lhs, const BasicResult& rhs)
{
    return ((lhs.Dist < rhs.Dist) || ((lhs.Dist == rhs.Dist) && (lhs.VID < rhs.VID)));
}


// Space to save temporary answer, similar with TopKCache
template<typename T>
class QueryResultSet : public QueryResult
{
public:
    QueryResultSet(const T* _target, int _K) : QueryResult(_target, _K, false)
    {
    }

    QueryResultSet(const QueryResultSet& other) : QueryResult(other)
    {
    }

    ~QueryResultSet()
    {
        if (m_quantizedTarget)
        {
            _mm_free(m_quantizedTarget);
        }
    }

    inline void SetTarget(const T *p_target)
    {
        m_target = p_target;
        if (m_quantizedTarget)
        {
            COMMON::DistanceUtils::Quantizer->QuantizeVector((void*)m_target, (uint8_t*)m_quantizedTarget);
        }
    }

    inline const T* GetTarget() const
    {
        return reinterpret_cast<const T*>(m_target);
    }

    inline const T* GetQuantizedTarget()
    {
        if (nullptr != COMMON::DistanceUtils::Quantizer)
        {
            if (!m_quantizedTarget)
            {
                m_quantizedTarget = (T*) _mm_malloc(COMMON::DistanceUtils::Quantizer->QuantizeSize(), ALIGN);
                COMMON::DistanceUtils::Quantizer->QuantizeVector((void*)m_target, (uint8_t*)m_quantizedTarget);
            }
            return reinterpret_cast<const T*>(m_quantizedTarget);
        }
        else
        {
            return reinterpret_cast<const T*>(m_target);
        }
    }

    inline float worstDist() const
    {
        return m_results[0].Dist;
    }

    bool AddPoint(const SizeType index, float dist)
    {
        if (dist < m_results[0].Dist || (dist == m_results[0].Dist && index < m_results[0].VID))
        {
            m_results[0].VID = index;
            m_results[0].Dist = dist;
            Heapify(m_resultNum);
            return true;
        }
        return false;
    }

    inline void SortResult()
    {
        for (int i = m_resultNum - 1; i >= 0; i--)
        {
            std::swap(m_results[0], m_results[i]);
            Heapify(i);
        }
    }

    void Reverse()
    {
        std::reverse(m_results.Data(), m_results.Data() + m_resultNum);
    }

private:
    void Heapify(int count)
    {
        int parent = 0, next = 1, maxidx = count - 1;
        while (next < maxidx)
        {
            if (m_results[next] < m_results[next + 1]) next++;
            if (m_results[parent] < m_results[next])
            {
                std::swap(m_results[next], m_results[parent]);
                parent = next;
                next = (parent << 1) + 1;
            }
            else break;
        }
        if (next == maxidx && m_results[parent] < m_results[next]) std::swap(m_results[parent], m_results[next]);
    }

    T* m_quantizedTarget = nullptr;
};
}
}

#endif // _SPTAG_COMMON_QUERYRESULTSET_H_
