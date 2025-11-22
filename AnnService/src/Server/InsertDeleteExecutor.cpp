#include "inc/Server/InsertDeleteExecutor.h"
#include "inc/Core/MetadataSet.h"
#include "inc/Core/VectorSet.h"
#include "inc/Helper/CommonHelper.h"

using namespace SPTAG;
using namespace SPTAG::Service;

InsertExecutionContext::InsertExecutionContext(std::shared_ptr<ServiceSettings> p_settings)
    : m_settings(std::move(p_settings))
{
    m_result.m_status = Socket::RemoteInsertDeleteResult::ResultStatus::Success;
    m_result.m_processedCount = 0;
}

InsertExecutionContext::~InsertExecutionContext()
{
}

ErrorCode InsertExecutionContext::ParseQuery(const Socket::RemoteInsertQuery& p_query)
{
    m_query = p_query;
    
    if (m_query.m_indexName.empty())
    {
        m_result.m_status = Socket::RemoteInsertDeleteResult::ResultStatus::InvalidIndex;
        m_result.m_message = "Index name is empty";
        return ErrorCode::Fail;
    }
    
    if (m_query.m_vectorData.empty())
    {
        m_result.m_status = Socket::RemoteInsertDeleteResult::ResultStatus::InvalidData;
        m_result.m_message = "Vector data is empty";
        return ErrorCode::Fail;
    }
    
    return ErrorCode::Success;
}

DeleteExecutionContext::DeleteExecutionContext(std::shared_ptr<ServiceSettings> p_settings)
    : m_settings(std::move(p_settings))
{
    m_result.m_status = Socket::RemoteInsertDeleteResult::ResultStatus::Success;
    m_result.m_processedCount = 0;
}

DeleteExecutionContext::~DeleteExecutionContext()
{
}

ErrorCode DeleteExecutionContext::ParseQuery(const Socket::RemoteDeleteQuery& p_query)
{
    m_query = p_query;
    
    if (m_query.m_indexName.empty())
    {
        m_result.m_status = Socket::RemoteInsertDeleteResult::ResultStatus::InvalidIndex;
        m_result.m_message = "Index name is empty";
        return ErrorCode::Fail;
    }
    
    if (m_query.m_type == Socket::RemoteDeleteQuery::DeleteType::ByVector && m_query.m_vectorData.empty())
    {
        m_result.m_status = Socket::RemoteInsertDeleteResult::ResultStatus::InvalidData;
        m_result.m_message = "Vector data is empty for ByVector delete";
        return ErrorCode::Fail;
    }
    
    if (m_query.m_type == Socket::RemoteDeleteQuery::DeleteType::ByVectorId && m_query.m_vectorIds.empty())
    {
        m_result.m_status = Socket::RemoteInsertDeleteResult::ResultStatus::InvalidData;
        m_result.m_message = "Vector IDs are empty for ByVectorId delete";
        return ErrorCode::Fail;
    }
    
    if (m_query.m_type == Socket::RemoteDeleteQuery::DeleteType::ByMetadata && m_query.m_metadataData.empty())
    {
        m_result.m_status = Socket::RemoteInsertDeleteResult::ResultStatus::InvalidData;
        m_result.m_message = "Metadata is empty for ByMetadata delete";
        return ErrorCode::Fail;
    }
    
    return ErrorCode::Success;
}

InsertExecutor::InsertExecutor(Socket::RemoteInsertQuery p_query, std::shared_ptr<ServiceContext> p_serviceContext,
                               const CallBack& p_callback)
    : m_callback(p_callback), c_serviceContext(std::move(p_serviceContext)), m_query(std::move(p_query))
{
}

InsertExecutor::~InsertExecutor()
{
}

void InsertExecutor::Execute()
{
    ExecuteInternal();
    if (bool(m_callback))
    {
        m_callback(std::move(m_executionContext));
    }
}

void InsertExecutor::ExecuteInternal()
{
    m_executionContext.reset(new InsertExecutionContext(c_serviceContext->GetServiceSettings()));

    if (m_executionContext->ParseQuery(m_query) != ErrorCode::Success)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to parse insert query!\n");
        return;
    }

    SelectIndex();

    if (m_selectedIndex.empty())
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Empty selected index for insert!\n");
        m_executionContext->GetResult().m_status = Socket::RemoteInsertDeleteResult::ResultStatus::InvalidIndex;
        m_executionContext->GetResult().m_message = "Index not found: " + m_query.m_indexName;
        return;
    }

    const auto& index = m_selectedIndex.front();

    // Validate dimension compatibility
    if (m_query.m_dimension != index->GetFeatureDim())
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Insert: dimension mismatch - expected %d, got %d\n", 
                     index->GetFeatureDim(), m_query.m_dimension);
        m_executionContext->GetResult().m_status = Socket::RemoteInsertDeleteResult::ResultStatus::DimensionMismatch;
        m_executionContext->GetResult().m_message = "Dimension mismatch";
        return;
    }

    // Validate value type compatibility
    if (m_query.m_valueType != index->GetVectorValueType())
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Insert: value type mismatch\n");
        m_executionContext->GetResult().m_status = Socket::RemoteInsertDeleteResult::ResultStatus::InvalidData;
        m_executionContext->GetResult().m_message = "Value type mismatch";
        return;
    }

    // Validate vector count
    if (m_query.m_vectorCount <= 0)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Insert: invalid vector count %d\n", m_query.m_vectorCount);
        m_executionContext->GetResult().m_status = Socket::RemoteInsertDeleteResult::ResultStatus::InvalidData;
        m_executionContext->GetResult().m_message = "Invalid vector count";
        return;
    }

    std::shared_ptr<MetadataSet> metadataSet = nullptr;
    if (m_query.m_type == Socket::RemoteInsertQuery::InsertType::VectorWithMetadata && !m_query.m_metadataData.empty())
    {
        std::vector<std::uint64_t> offsets(m_query.m_vectorCount + 1);
        if (!MetadataSet::GetMetadataOffsets(m_query.m_metadataData.data(), m_query.m_metadataData.size(), 
                                           offsets.data(), offsets.size(), '\n'))
        {
            m_executionContext->GetResult().m_status = Socket::RemoteInsertDeleteResult::ResultStatus::InvalidData;
            m_executionContext->GetResult().m_message = "Invalid metadata format";
            return;
        }
        
        metadataSet.reset(new MemMetadataSet(
            ByteArray(m_query.m_metadataData.data(), m_query.m_metadataData.size(), false),
            ByteArray((std::uint8_t*)offsets.data(), offsets.size() * sizeof(std::uint64_t), false),
            m_query.m_vectorCount));
    }

    ErrorCode result = ErrorCode::Undefined;
    int beginHead = -1, endHead = -1;
    
    // Try to use AddIndexId first to get vector ID range if available
    // Note: AddIndexId doesn't handle metadata, so only use it for basic vector insertion
    if (metadataSet == nullptr && index->AddIndexId(m_query.m_vectorData.data(), m_query.m_vectorCount, m_query.m_dimension, beginHead, endHead) == ErrorCode::Success)
    {
        result = ErrorCode::Success;
        // Populate the new vector IDs if we got a valid range
        if (beginHead >= 0 && endHead > beginHead)
        {
            m_executionContext->GetResult().m_newVectorIds.reserve(endHead - beginHead);
            for (int i = beginHead; i < endHead; ++i)
            {
                m_executionContext->GetResult().m_newVectorIds.push_back(static_cast<SizeType>(i));
            }
        }
    }
    else
    {
        // Use regular AddIndex for metadata insertion or when AddIndexId is not supported
        result = index->AddIndex(m_query.m_vectorData.data(), m_query.m_vectorCount, m_query.m_dimension,
                                metadataSet, m_query.m_withMetaIndex, m_query.m_normalized);
    }

    switch (result)
    {
    case ErrorCode::Success:
        m_executionContext->GetResult().m_status = Socket::RemoteInsertDeleteResult::ResultStatus::Success;
        m_executionContext->GetResult().m_processedCount = m_query.m_vectorCount;
        m_executionContext->GetResult().m_message = "Insert completed successfully";
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Inserted %d vectors, assigned IDs: %d to %d\n",
                     m_query.m_vectorCount, beginHead, endHead - 1);
        break;
    case ErrorCode::MemoryOverFlow:
        m_executionContext->GetResult().m_status = Socket::RemoteInsertDeleteResult::ResultStatus::MemoryOverflow;
        m_executionContext->GetResult().m_message = "Memory overflow during insert";
        break;
    case ErrorCode::DimensionSizeMismatch:
        m_executionContext->GetResult().m_status = Socket::RemoteInsertDeleteResult::ResultStatus::DimensionMismatch;
        m_executionContext->GetResult().m_message = "Dimension size mismatch";
        break;
    default:
        m_executionContext->GetResult().m_status = Socket::RemoteInsertDeleteResult::ResultStatus::Failed;
        m_executionContext->GetResult().m_message = "Insert operation failed";
        break;
    }

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Insert operation completed with status: %d, processed %d vectors, assigned %zu new IDs\n",
                 static_cast<int>(m_executionContext->GetResult().m_status), 
                 m_executionContext->GetResult().m_processedCount,
                 m_executionContext->GetResult().m_newVectorIds.size());
}

void InsertExecutor::SelectIndex()
{
    const auto &indexMap = c_serviceContext->GetIndexMap();
    if (indexMap.empty())
    {
        return;
    }

    auto iter = indexMap.find(m_query.m_indexName);
    if (iter != indexMap.cend())
    {
        m_selectedIndex.push_back(iter->second);
    }
}

DeleteExecutor::DeleteExecutor(Socket::RemoteDeleteQuery p_query, std::shared_ptr<ServiceContext> p_serviceContext,
                               const CallBack& p_callback)
    : m_callback(p_callback), c_serviceContext(std::move(p_serviceContext)), m_query(std::move(p_query))
{
}

DeleteExecutor::~DeleteExecutor()
{
}

void DeleteExecutor::Execute()
{
    ExecuteInternal();
    if (bool(m_callback))
    {
        m_callback(std::move(m_executionContext));
    }
}

void DeleteExecutor::ExecuteInternal()
{
    m_executionContext.reset(new DeleteExecutionContext(c_serviceContext->GetServiceSettings()));

    if (m_executionContext->ParseQuery(m_query) != ErrorCode::Success)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to parse delete query!\n");
        return;
    }

    SelectIndex();

    if (m_selectedIndex.empty())
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Empty selected index for delete!\n");
        m_executionContext->GetResult().m_status = Socket::RemoteInsertDeleteResult::ResultStatus::InvalidIndex;
        m_executionContext->GetResult().m_message = "Index not found: " + m_query.m_indexName;
        return;
    }

    const auto& index = m_selectedIndex.front();
    ErrorCode result = ErrorCode::Success;
    SizeType processedCount = 0;

    switch (m_query.m_type)
    {
    case Socket::RemoteDeleteQuery::DeleteType::ByVector:
        {
            if (m_query.m_dimension != index->GetFeatureDim())
            {
                m_executionContext->GetResult().m_status = Socket::RemoteInsertDeleteResult::ResultStatus::DimensionMismatch;
                m_executionContext->GetResult().m_message = "Dimension mismatch";
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Delete by vector: dimension mismatch - expected %d, got %d\n",
                             index->GetFeatureDim(), m_query.m_dimension);
                return;
            }

            if (m_query.m_valueType != index->GetVectorValueType())
            {
                m_executionContext->GetResult().m_status = Socket::RemoteInsertDeleteResult::ResultStatus::InvalidData;
                m_executionContext->GetResult().m_message = "Value type mismatch";
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Delete by vector: value type mismatch\n");
                return;
            }

            result = index->DeleteIndex(m_query.m_vectorData.data(), m_query.m_vectorCount);
            processedCount = m_query.m_vectorCount;
            
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Delete by vector: attempted to delete %d vectors\n", m_query.m_vectorCount);
        }
        break;

    case Socket::RemoteDeleteQuery::DeleteType::ByVectorId:
        {
            SizeType successCount = 0;
            std::vector<SizeType> successfullyDeleted;
            successfullyDeleted.reserve(m_query.m_vectorIds.size());
            
            for (SizeType id : m_query.m_vectorIds)
            {
                ErrorCode deleteResult = index->DeleteIndex(id);
                if (deleteResult == ErrorCode::Success)
                {
                    successCount++;
                    successfullyDeleted.push_back(id);
                }
            }
            
            m_executionContext->GetResult().m_newVectorIds = std::move(successfullyDeleted);
            processedCount = successCount;
            result = (successCount > 0) ? ErrorCode::Success : ErrorCode::VectorNotFound;
            
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Delete by ID: requested %zu, successfully deleted %d vectors\n",
                         m_query.m_vectorIds.size(), successCount);
        }
        break;

    case Socket::RemoteDeleteQuery::DeleteType::ByMetadata:
        {
            // TODO: For metadata-based deletion, need to implement this functionality
            // This would require extending the VectorIndex interface to support metadata queries
            // and finding vectors that match the specified metadata criteria
            m_executionContext->GetResult().m_status = Socket::RemoteInsertDeleteResult::ResultStatus::Failed;
            m_executionContext->GetResult().m_message = "Delete by metadata not yet implemented - requires VectorIndex interface extension";
            SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "Delete by metadata requested but not yet implemented\n");
            return;
        }
        break;

    default:
        m_executionContext->GetResult().m_status = Socket::RemoteInsertDeleteResult::ResultStatus::InvalidData;
        m_executionContext->GetResult().m_message = "Invalid delete type";
        return;
    }

    switch (result)
    {
    case ErrorCode::Success:
        m_executionContext->GetResult().m_status = Socket::RemoteInsertDeleteResult::ResultStatus::Success;
        m_executionContext->GetResult().m_processedCount = processedCount;
        m_executionContext->GetResult().m_message = "Delete completed successfully";
        break;
    case ErrorCode::VectorNotFound:
        m_executionContext->GetResult().m_status = Socket::RemoteInsertDeleteResult::ResultStatus::Failed;
        m_executionContext->GetResult().m_message = "Vector(s) not found";
        break;
    default:
        m_executionContext->GetResult().m_status = Socket::RemoteInsertDeleteResult::ResultStatus::Failed;
        m_executionContext->GetResult().m_message = "Delete operation failed";
        break;
    }

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Delete operation completed with status: %d, processed %d items, affected %zu vector IDs\n",
                 static_cast<int>(m_executionContext->GetResult().m_status), 
                 m_executionContext->GetResult().m_processedCount,
                 m_executionContext->GetResult().m_newVectorIds.size());
}

void DeleteExecutor::SelectIndex()
{
    const auto &indexMap = c_serviceContext->GetIndexMap();
    if (indexMap.empty())
    {
        return;
    }

    auto iter = indexMap.find(m_query.m_indexName);
    if (iter != indexMap.cend())
    {
        m_selectedIndex.push_back(iter->second);
    }
}