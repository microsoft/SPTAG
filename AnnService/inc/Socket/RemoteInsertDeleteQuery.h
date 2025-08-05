#ifndef _SPTAG_SOCKET_REMOTEINSERTDELETEQUERY_H_
#define _SPTAG_SOCKET_REMOTEINSERTDELETEQUERY_H_

#include "inc/Core/CommonDataStructure.h"
#include "inc/Core/VectorIndex.h"

#include <cstdint>
#include <memory>
#include <functional>
#include <vector>
#include <string>

namespace SPTAG
{
namespace Socket
{

struct RemoteInsertQuery
{
    static constexpr std::uint16_t MajorVersion() { return 1; }
    static constexpr std::uint16_t MirrorVersion() { return 0; }

    enum class InsertType : std::uint8_t
    {
        Vector = 0,
        VectorWithMetadata = 1
    };

    RemoteInsertQuery();

    std::size_t EstimateBufferSize() const;

    std::uint8_t* Write(std::uint8_t* p_buffer) const;

    const std::uint8_t* Read(const std::uint8_t* p_buffer);

    InsertType m_type;
    std::string m_indexName;
    DimensionType m_dimension;
    VectorValueType m_valueType;
    SizeType m_vectorCount;
    std::vector<std::uint8_t> m_vectorData;
    std::vector<std::uint8_t> m_metadataData;
    bool m_normalized;
    bool m_withMetaIndex;
};

struct RemoteDeleteQuery
{
    static constexpr std::uint16_t MajorVersion() { return 1; }
    static constexpr std::uint16_t MirrorVersion() { return 0; }

    enum class DeleteType : std::uint8_t
    {
        ByVector = 0,
        ByVectorId = 1,
        ByMetadata = 2
    };

    RemoteDeleteQuery();

    std::size_t EstimateBufferSize() const;

    std::uint8_t* Write(std::uint8_t* p_buffer) const;

    const std::uint8_t* Read(const std::uint8_t* p_buffer);

    DeleteType m_type;
    std::string m_indexName;
    DimensionType m_dimension;
    VectorValueType m_valueType;
    SizeType m_vectorCount;
    std::vector<std::uint8_t> m_vectorData;
    std::vector<SizeType> m_vectorIds;
    std::vector<std::uint8_t> m_metadataData;
    bool m_normalized;
};

struct RemoteInsertDeleteResult
{
    static constexpr std::uint16_t MajorVersion() { return 1; }
    static constexpr std::uint16_t MirrorVersion() { return 0; }

    enum class ResultStatus : std::uint8_t
    {
        Success = 0,
        Failed = 1,
        InvalidIndex = 2,
        InvalidData = 3,
        MemoryOverflow = 4,
        DimensionMismatch = 5
    };

    RemoteInsertDeleteResult();

    RemoteInsertDeleteResult(const RemoteInsertDeleteResult& p_right);

    RemoteInsertDeleteResult(RemoteInsertDeleteResult&& p_right);

    RemoteInsertDeleteResult& operator=(RemoteInsertDeleteResult&& p_right);

    std::size_t EstimateBufferSize() const;

    std::uint8_t* Write(std::uint8_t* p_buffer) const;

    const std::uint8_t* Read(const std::uint8_t* p_buffer);

    ResultStatus m_status;
    std::string m_message;
    SizeType m_processedCount;
    std::vector<SizeType> m_newVectorIds; // For insert operations
};

} // namespace Socket
} // namespace SPTAG

#endif // _SPTAG_SOCKET_REMOTEINSERTDELETEQUERY_H_ 