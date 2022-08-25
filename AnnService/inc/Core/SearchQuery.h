// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SEARCHQUERY_H_
#define _SPTAG_SEARCHQUERY_H_

#include "SearchResult.h"

#include <cstring>

namespace SPTAG
{

// Space to save temporary answer, similar with TopKCache
class QueryResult
{
public:
    typedef BasicResult* iterator;
    typedef const BasicResult* const_iterator;

    QueryResult()
        : m_target(nullptr),
          m_resultNum(0),
          m_withMeta(false),
          m_quantizedTarget(nullptr),
          m_quantizedSize(0)
    {
    }


    QueryResult(const void* p_target, int p_resultNum, bool p_withMeta)
    {
        Init(p_target, p_resultNum, p_withMeta);
    }

    
    QueryResult(const void* p_target, int p_resultNum, bool p_withMeta, BasicResult* p_results)
        : m_target(p_target),
          m_resultNum(p_resultNum),
          m_withMeta(p_withMeta),
          m_quantizedTarget((void*)p_target),
          m_quantizedSize(0)
    {
        m_results.Set(p_results, p_resultNum, false);
    }


    QueryResult(const QueryResult& p_other)
    {
        Init(p_other.m_target, p_other.m_resultNum, p_other.m_withMeta);
        if (m_resultNum > 0)
        {
            std::copy(p_other.m_results.Data(), p_other.m_results.Data() + m_resultNum, m_results.Data());
        }
        if (p_other.m_target != p_other.m_quantizedTarget)
        {
            m_quantizedSize = p_other.m_quantizedSize;
            m_quantizedTarget = ALIGN_ALLOC(m_quantizedSize);
            std::copy(reinterpret_cast<std::uint8_t*>(p_other.m_quantizedTarget), reinterpret_cast<std::uint8_t*>(p_other.m_quantizedTarget) + m_quantizedSize, reinterpret_cast<std::uint8_t*>(m_quantizedTarget));
        }
    }


    QueryResult& operator=(const QueryResult& p_other)
    {
        if (m_target != m_quantizedTarget) ALIGN_FREE(m_quantizedTarget);

        Init(p_other.m_target, p_other.m_resultNum, p_other.m_withMeta);
        if (m_resultNum > 0)
        {
            std::copy(p_other.m_results.Data(), p_other.m_results.Data() + m_resultNum, m_results.Data());
        }
        if (p_other.m_target != p_other.m_quantizedTarget)
        {
            m_quantizedSize = p_other.m_quantizedSize;
            m_quantizedTarget = ALIGN_ALLOC(m_quantizedSize);
            std::copy(reinterpret_cast<std::uint8_t*>(p_other.m_quantizedTarget), reinterpret_cast<std::uint8_t*>(p_other.m_quantizedTarget) + m_quantizedSize, reinterpret_cast<std::uint8_t*>(m_quantizedTarget));
        }
        return *this;
    }


    ~QueryResult()
    {
        if (m_target != m_quantizedTarget)
        {
            ALIGN_FREE(m_quantizedTarget);
        }
    }


    inline void Init(const void* p_target, int p_resultNum, bool p_withMeta)
    {
        m_target = p_target;
        m_resultNum = p_resultNum;
        m_withMeta = p_withMeta;
        m_quantizedTarget = (void*)p_target;
        m_quantizedSize = 0;

        m_results = Array<BasicResult>::Alloc(p_resultNum);
    }


    inline int GetResultNum() const
    {
        return m_resultNum;
    }


    inline const void* GetTarget()
    {
        return m_target;
    }


    inline void* GetQuantizedTarget()
    {
        return m_quantizedTarget;
    }


    inline void SetTarget(const void* p_target)
    {
        if (m_target != m_quantizedTarget)
        {
            ALIGN_FREE(m_quantizedTarget);
        }
        m_target = p_target;
        m_quantizedTarget = (void*)p_target;
        m_quantizedSize = 0;
    }
    

    inline bool HasQuantizedTarget()
    {
        return m_target != m_quantizedTarget;
    }


    inline void CleanQuantizedTarget()
    {
        if (m_target != m_quantizedTarget) {
            ALIGN_FREE(m_quantizedTarget);
            m_quantizedTarget = (void*)m_target;
        }
    }


    inline BasicResult* GetResult(int i) const
    {
        return i < m_resultNum ? m_results.Data() + i : nullptr;
    }


    inline void SetResult(int p_index, SizeType p_VID, float p_dist)
    {
        if (p_index < m_resultNum)
        {
            m_results[p_index].VID = p_VID;
            m_results[p_index].Dist = p_dist;
        }
    }


    inline BasicResult* GetResults() const
    {
        return m_results.Data();
    }


    inline bool WithMeta() const
    {
        return m_withMeta;
    }


    inline const ByteArray& GetMetadata(int p_index) const
    {
        if (p_index < m_resultNum && m_withMeta)
        {
            return m_results[p_index].Meta;
        }

        return ByteArray::c_empty;
    }


    inline void SetMetadata(int p_index, ByteArray p_metadata)
    {
        if (p_index < m_resultNum && m_withMeta)
        {
            m_results[p_index].Meta = std::move(p_metadata);
        }
    }


    inline void Reset()
    {
        for (int i = 0; i < m_resultNum; i++)
        {
            m_results[i].VID = -1;
            m_results[i].Dist = MaxDist;
            m_results[i].Meta.Clear();
        }
    }


    iterator begin()
    {
        return m_results.Data();
    }


    iterator end()
    {
        return m_results.Data() + m_resultNum;
    }


    const_iterator begin() const
    {
        return m_results.Data();
    }


    const_iterator end() const
    {
        return m_results.Data() + m_resultNum;
    }


protected:
    const void* m_target;

    void* m_quantizedTarget;

    SizeType m_quantizedSize;

    int m_resultNum;

    bool m_withMeta;

    Array<BasicResult> m_results;
};
} // namespace SPTAG

#endif // _SPTAG_SEARCHQUERY_H_
