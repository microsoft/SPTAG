#include "inc/Socket/RemoteInsertDeleteQuery.h"
#include "inc/Socket/SimpleSerialization.h"

using namespace SPTAG::Socket;

RemoteInsertQuery::RemoteInsertQuery()
    : m_type(InsertType::Vector), m_dimension(0), m_valueType(VectorValueType::Undefined), 
      m_vectorCount(0), m_normalized(false), m_withMetaIndex(false)
{
}

std::size_t RemoteInsertQuery::EstimateBufferSize() const
{
    return sizeof(std::uint16_t) * 2 // version
         + sizeof(InsertType)
         + sizeof(std::uint32_t) + m_indexName.size() // index name
         + sizeof(DimensionType)
         + sizeof(VectorValueType)
         + sizeof(SizeType)
         + sizeof(std::uint32_t) + m_vectorData.size() // vector data size + data
         + sizeof(std::uint32_t) + m_metadataData.size() // metadata size + data
         + sizeof(bool) * 2; // flags
}

std::uint8_t* RemoteInsertQuery::Write(std::uint8_t* p_buffer) const
{
    std::uint8_t* buff = p_buffer;
    
    // Write version
    buff = SimpleSerialization::SimpleWriteBuffer(MajorVersion(), buff);
    buff = SimpleSerialization::SimpleWriteBuffer(MirrorVersion(), buff);
    
    // Write data
    buff = SimpleSerialization::SimpleWriteBuffer(m_type, buff);
    buff = SimpleSerialization::SimpleWriteBuffer(m_indexName, buff);
    buff = SimpleSerialization::SimpleWriteBuffer(m_dimension, buff);
    buff = SimpleSerialization::SimpleWriteBuffer(m_valueType, buff);
    buff = SimpleSerialization::SimpleWriteBuffer(m_vectorCount, buff);
    
    // Write vector data (size + raw bytes)
    buff = SimpleSerialization::SimpleWriteBuffer(static_cast<std::uint32_t>(m_vectorData.size()), buff);
    if (!m_vectorData.empty())
    {
        std::memcpy(buff, m_vectorData.data(), m_vectorData.size());
        buff += m_vectorData.size();
    }
    
    // Write metadata data (size + raw bytes)
    buff = SimpleSerialization::SimpleWriteBuffer(static_cast<std::uint32_t>(m_metadataData.size()), buff);
    if (!m_metadataData.empty())
    {
        std::memcpy(buff, m_metadataData.data(), m_metadataData.size());
        buff += m_metadataData.size();
    }
    
    buff = SimpleSerialization::SimpleWriteBuffer(m_normalized, buff);
    buff = SimpleSerialization::SimpleWriteBuffer(m_withMetaIndex, buff);
    
    return buff;
}

const std::uint8_t* RemoteInsertQuery::Read(const std::uint8_t* p_buffer)
{
    const std::uint8_t* buff = p_buffer;
    
    std::uint16_t majorVersion, mirrorVersion;
    buff = SimpleSerialization::SimpleReadBuffer(buff, majorVersion);
    buff = SimpleSerialization::SimpleReadBuffer(buff, mirrorVersion);
    
    if (majorVersion != MajorVersion())
    {
        return nullptr;
    }
    
    buff = SimpleSerialization::SimpleReadBuffer(buff, m_type);
    buff = SimpleSerialization::SimpleReadBuffer(buff, m_indexName);
    buff = SimpleSerialization::SimpleReadBuffer(buff, m_dimension);
    buff = SimpleSerialization::SimpleReadBuffer(buff, m_valueType);
    buff = SimpleSerialization::SimpleReadBuffer(buff, m_vectorCount);
    
    std::uint32_t vectorDataSize = 0;
    buff = SimpleSerialization::SimpleReadBuffer(buff, vectorDataSize);
    m_vectorData.resize(vectorDataSize);
    if (vectorDataSize > 0)
    {
        std::memcpy(m_vectorData.data(), buff, vectorDataSize);
        buff += vectorDataSize;
    }
    
    std::uint32_t metadataSize = 0;
    buff = SimpleSerialization::SimpleReadBuffer(buff, metadataSize);
    m_metadataData.resize(metadataSize);
    if (metadataSize > 0)
    {
        std::memcpy(m_metadataData.data(), buff, metadataSize);
        buff += metadataSize;
    }
    
    buff = SimpleSerialization::SimpleReadBuffer(buff, m_normalized);
    buff = SimpleSerialization::SimpleReadBuffer(buff, m_withMetaIndex);
    
    return buff;
}

RemoteDeleteQuery::RemoteDeleteQuery()
    : m_type(DeleteType::ByVector), m_dimension(0), m_valueType(VectorValueType::Undefined), 
      m_vectorCount(0), m_normalized(false)
{
}

std::size_t RemoteDeleteQuery::EstimateBufferSize() const
{
    return sizeof(std::uint16_t) * 2 // version
         + sizeof(DeleteType)
         + sizeof(std::uint32_t) + m_indexName.size() // index name
         + sizeof(DimensionType)
         + sizeof(VectorValueType)
         + sizeof(SizeType)
         + sizeof(std::uint32_t) + m_vectorData.size() // vector data size + data
         + sizeof(std::uint32_t) + m_vectorIds.size() * sizeof(SizeType) // vector IDs size + data
         + sizeof(std::uint32_t) + m_metadataData.size() // metadata size + data
         + sizeof(bool); // normalized flag
}

std::uint8_t* RemoteDeleteQuery::Write(std::uint8_t* p_buffer) const
{
    std::uint8_t* buff = p_buffer;

    buff = SimpleSerialization::SimpleWriteBuffer(MajorVersion(), buff);
    buff = SimpleSerialization::SimpleWriteBuffer(MirrorVersion(), buff);

    buff = SimpleSerialization::SimpleWriteBuffer(m_type, buff);
    buff = SimpleSerialization::SimpleWriteBuffer(m_indexName, buff);
    buff = SimpleSerialization::SimpleWriteBuffer(m_dimension, buff);
    buff = SimpleSerialization::SimpleWriteBuffer(m_valueType, buff);
    buff = SimpleSerialization::SimpleWriteBuffer(m_vectorCount, buff);
    
    buff = SimpleSerialization::SimpleWriteBuffer(static_cast<std::uint32_t>(m_vectorData.size()), buff);
    if (!m_vectorData.empty())
    {
        std::memcpy(buff, m_vectorData.data(), m_vectorData.size());
        buff += m_vectorData.size();
    }
    
    buff = SimpleSerialization::SimpleWriteBuffer(static_cast<std::uint32_t>(m_vectorIds.size()), buff);
    for (const auto& id : m_vectorIds)
    {
        buff = SimpleSerialization::SimpleWriteBuffer(id, buff);
    }
    
    buff = SimpleSerialization::SimpleWriteBuffer(static_cast<std::uint32_t>(m_metadataData.size()), buff);
    if (!m_metadataData.empty())
    {
        std::memcpy(buff, m_metadataData.data(), m_metadataData.size());
        buff += m_metadataData.size();
    }
    
    buff = SimpleSerialization::SimpleWriteBuffer(m_normalized, buff);
    
    return buff;
}

const std::uint8_t* RemoteDeleteQuery::Read(const std::uint8_t* p_buffer)
{
    const std::uint8_t* buff = p_buffer;
    
    std::uint16_t majorVersion, mirrorVersion;
    buff = SimpleSerialization::SimpleReadBuffer(buff, majorVersion);
    buff = SimpleSerialization::SimpleReadBuffer(buff, mirrorVersion);
    
    if (majorVersion != MajorVersion())
    {
        return nullptr;
    }
    
    buff = SimpleSerialization::SimpleReadBuffer(buff, m_type);
    buff = SimpleSerialization::SimpleReadBuffer(buff, m_indexName);
    buff = SimpleSerialization::SimpleReadBuffer(buff, m_dimension);
    buff = SimpleSerialization::SimpleReadBuffer(buff, m_valueType);
    buff = SimpleSerialization::SimpleReadBuffer(buff, m_vectorCount);
    
    std::uint32_t vectorDataSize = 0;
    buff = SimpleSerialization::SimpleReadBuffer(buff, vectorDataSize);
    m_vectorData.resize(vectorDataSize);
    if (vectorDataSize > 0)
    {
        std::memcpy(m_vectorData.data(), buff, vectorDataSize);
        buff += vectorDataSize;
    }

    std::uint32_t vectorIdCount = 0;
    buff = SimpleSerialization::SimpleReadBuffer(buff, vectorIdCount);
    m_vectorIds.resize(vectorIdCount);
    for (std::uint32_t i = 0; i < vectorIdCount; ++i)
    {
        buff = SimpleSerialization::SimpleReadBuffer(buff, m_vectorIds[i]);
    }

    std::uint32_t metadataSize = 0;
    buff = SimpleSerialization::SimpleReadBuffer(buff, metadataSize);
    m_metadataData.resize(metadataSize);
    if (metadataSize > 0)
    {
        std::memcpy(m_metadataData.data(), buff, metadataSize);
        buff += metadataSize;
    }
    
    buff = SimpleSerialization::SimpleReadBuffer(buff, m_normalized);
    
    return buff;
}

RemoteInsertDeleteResult::RemoteInsertDeleteResult()
    : m_status(ResultStatus::Success), m_processedCount(0)
{
}

RemoteInsertDeleteResult::RemoteInsertDeleteResult(const RemoteInsertDeleteResult& p_right)
    : m_status(p_right.m_status), m_message(p_right.m_message), m_processedCount(p_right.m_processedCount),
      m_newVectorIds(p_right.m_newVectorIds)
{
}

RemoteInsertDeleteResult::RemoteInsertDeleteResult(RemoteInsertDeleteResult&& p_right)
    : m_status(std::move(p_right.m_status)), m_message(std::move(p_right.m_message)), 
      m_processedCount(std::move(p_right.m_processedCount)), m_newVectorIds(std::move(p_right.m_newVectorIds))
{
}

RemoteInsertDeleteResult& RemoteInsertDeleteResult::operator=(RemoteInsertDeleteResult&& p_right)
{
    m_status = std::move(p_right.m_status);
    m_message = std::move(p_right.m_message);
    m_processedCount = std::move(p_right.m_processedCount);
    m_newVectorIds = std::move(p_right.m_newVectorIds);
    return *this;
}

std::size_t RemoteInsertDeleteResult::EstimateBufferSize() const
{
    return sizeof(std::uint16_t) * 2 // version
         + sizeof(ResultStatus)
         + sizeof(std::uint32_t) + m_message.size() // message
         + sizeof(SizeType)
         + sizeof(std::uint32_t) + m_newVectorIds.size() * sizeof(SizeType); // new vector IDs
}

std::uint8_t* RemoteInsertDeleteResult::Write(std::uint8_t* p_buffer) const
{
    std::uint8_t* buff = p_buffer;

    buff = SimpleSerialization::SimpleWriteBuffer(MajorVersion(), buff);
    buff = SimpleSerialization::SimpleWriteBuffer(MirrorVersion(), buff);
    
    buff = SimpleSerialization::SimpleWriteBuffer(m_status, buff);
    buff = SimpleSerialization::SimpleWriteBuffer(m_message, buff);
    buff = SimpleSerialization::SimpleWriteBuffer(m_processedCount, buff);

    buff = SimpleSerialization::SimpleWriteBuffer(static_cast<std::uint32_t>(m_newVectorIds.size()), buff);
    for (const auto& id : m_newVectorIds)
    {
        buff = SimpleSerialization::SimpleWriteBuffer(id, buff);
    }
    
    return buff;
}

const std::uint8_t* RemoteInsertDeleteResult::Read(const std::uint8_t* p_buffer)
{
    const std::uint8_t* buff = p_buffer;
    
    std::uint16_t majorVersion, mirrorVersion;
    buff = SimpleSerialization::SimpleReadBuffer(buff, majorVersion);
    buff = SimpleSerialization::SimpleReadBuffer(buff, mirrorVersion);
    
    if (majorVersion != MajorVersion())
    {
        return nullptr;
    }
    
    buff = SimpleSerialization::SimpleReadBuffer(buff, m_status);
    buff = SimpleSerialization::SimpleReadBuffer(buff, m_message);
    buff = SimpleSerialization::SimpleReadBuffer(buff, m_processedCount);
    
    std::uint32_t vectorIdCount = 0;
    buff = SimpleSerialization::SimpleReadBuffer(buff, vectorIdCount);
    m_newVectorIds.resize(vectorIdCount);
    for (std::uint32_t i = 0; i < vectorIdCount; ++i)
    {
        buff = SimpleSerialization::SimpleReadBuffer(buff, m_newVectorIds[i]);
    }
    
    return buff;
}