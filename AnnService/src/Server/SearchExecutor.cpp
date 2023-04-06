// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Server/SearchExecutor.h"

using namespace SPTAG;
using namespace SPTAG::Service;


SearchExecutor::SearchExecutor(std::string p_queryString,
                               std::shared_ptr<ServiceContext> p_serviceContext,
                               const CallBack& p_callback)
    : m_callback(p_callback),
      c_serviceContext(std::move(p_serviceContext)),
      m_queryString(std::move(p_queryString))
{
}


SearchExecutor::~SearchExecutor()
{
}


void
SearchExecutor::Execute()
{
    ExecuteInternal();
    if (bool(m_callback))
    {
        m_callback(std::move(m_executionContext));
    }
}


void
SearchExecutor::ExecuteInternal()
{
    m_executionContext.reset(new SearchExecutionContext(c_serviceContext->GetServiceSettings()));

    if (m_executionContext->ParseQuery(m_queryString) != ErrorCode::Success) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to parse query:%s!\n", m_queryString.c_str());
        return;
    }

    m_executionContext->ExtractOption();

    SelectIndex();

    if (m_selectedIndex.empty())
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Empty selected index!\n");
        return;
    }

    const auto& firstIndex = m_selectedIndex.front();

    if (ErrorCode::Success != m_executionContext->ExtractVector(firstIndex->GetVectorValueType()))
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to extract vector!\n");
        return;
    }

    if (m_executionContext->GetVectorDimension() != firstIndex->GetFeatureDim())
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to match vector dimension!\n");
        return;
    } 

    QueryResult query(m_executionContext->GetVector().Data(),
                      m_executionContext->GetResultNum(),
                      m_executionContext->GetExtractMetadata());

    for (const auto& vectorIndex : m_selectedIndex)
    {
        if (vectorIndex->GetVectorValueType() != firstIndex->GetVectorValueType()
            || vectorIndex->GetFeatureDim() != firstIndex->GetFeatureDim())
        {
            continue;
        }

        query.Reset();
        if (ErrorCode::Success == vectorIndex->SearchIndex(query))
        {
            m_executionContext->AddResults(vectorIndex->GetIndexName(), query);
        }
        else {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to execute SearchIndex!\n");
        }
    }
}


void
SearchExecutor::SelectIndex()
{
    const auto& indexNames = m_executionContext->GetSelectedIndexNames();
    const auto& indexMap = c_serviceContext->GetIndexMap();
    if (indexMap.empty())
    {
        return;
    }

    if (indexNames.empty())
    {
        if (indexMap.size() == 1)
        {
            m_selectedIndex.push_back(indexMap.begin()->second);
        }
    }
    else
    {
        for (const auto& indexName : indexNames)
        {
            auto iter = indexMap.find(indexName);
            if (iter != indexMap.cend())
            {
                m_selectedIndex.push_back(iter->second);
            }
        }
    }
}

