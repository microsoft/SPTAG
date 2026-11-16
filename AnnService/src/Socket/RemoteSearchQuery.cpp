// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Socket/RemoteSearchQuery.h"
#include "inc/Socket/Packet.h"
#include "inc/Socket/SimpleSerialization.h"

using namespace SPTAG;
using namespace SPTAG::Socket;

RemoteQuery::RemoteQuery() : m_type(QueryType::String)
{
}

std::size_t RemoteQuery::EstimateBufferSize() const
{
    std::size_t sum = 0;
    sum += SimpleSerialization::EstimateBufferSize(MajorVersion());
    sum += SimpleSerialization::EstimateBufferSize(MirrorVersion());
    sum += SimpleSerialization::EstimateBufferSize(m_type);
    sum += SimpleSerialization::EstimateBufferSize(m_queryString);

    return sum;
}

std::uint8_t *RemoteQuery::Write(std::uint8_t *p_buffer) const
{
    p_buffer = SimpleSerialization::SimpleWriteBuffer(MajorVersion(), p_buffer);
    p_buffer = SimpleSerialization::SimpleWriteBuffer(MirrorVersion(), p_buffer);

    p_buffer = SimpleSerialization::SimpleWriteBuffer(m_type, p_buffer);
    p_buffer = SimpleSerialization::SimpleWriteBuffer(m_queryString, p_buffer);

    return p_buffer;
}

const std::uint8_t *RemoteQuery::Read(const std::uint8_t *p_buffer)
{
    decltype(MajorVersion()) majorVer = 0;
    decltype(MirrorVersion()) mirrorVer = 0;

    p_buffer = SimpleSerialization::SimpleReadBuffer(p_buffer, majorVer);
    p_buffer = SimpleSerialization::SimpleReadBuffer(p_buffer, mirrorVer);
    if (majorVer != MajorVersion())
    {
        return nullptr;
    }

    p_buffer = SimpleSerialization::SimpleReadBuffer(p_buffer, m_type);
    p_buffer = SimpleSerialization::SimpleReadBuffer(p_buffer, m_queryString);

    return p_buffer;
}

RemoteSearchResult::RemoteSearchResult() : m_status(ResultStatus::Timeout)
{
}

RemoteSearchResult::RemoteSearchResult(const RemoteSearchResult &p_right)
    : m_status(p_right.m_status), m_allIndexResults(p_right.m_allIndexResults)
{
}

RemoteSearchResult::RemoteSearchResult(RemoteSearchResult &&p_right)
    : m_status(std::move(p_right.m_status)), m_allIndexResults(std::move(p_right.m_allIndexResults))
{
}

RemoteSearchResult &RemoteSearchResult::operator=(RemoteSearchResult &&p_right)
{
    m_status = p_right.m_status;
    m_allIndexResults = std::move(p_right.m_allIndexResults);

    return *this;
}

std::size_t RemoteSearchResult::EstimateBufferSize() const
{
    std::size_t sum = 0;
    sum += SimpleSerialization::EstimateBufferSize(MajorVersion());
    sum += SimpleSerialization::EstimateBufferSize(MirrorVersion());

    sum += SimpleSerialization::EstimateBufferSize(m_status);

    sum += sizeof(std::uint32_t);
    for (const auto &indexRes : m_allIndexResults)
    {
        sum += SimpleSerialization::EstimateBufferSize(indexRes.m_indexName);
        sum += sizeof(std::uint32_t);
        sum += sizeof(bool);

        for (const auto &res : indexRes.m_results)
        {
            sum += SimpleSerialization::EstimateBufferSize(res.VID);
            sum += SimpleSerialization::EstimateBufferSize(res.Dist);
        }

        if (indexRes.m_results.WithMeta())
        {
            for (int i = 0; i < indexRes.m_results.GetResultNum(); ++i)
            {
                sum += SimpleSerialization::EstimateBufferSize(indexRes.m_results.GetMetadata(i));
            }
        }
    }

    return sum;
}

std::uint8_t *RemoteSearchResult::Write(std::uint8_t *p_buffer) const
{
    p_buffer = SimpleSerialization::SimpleWriteBuffer(MajorVersion(), p_buffer);
    p_buffer = SimpleSerialization::SimpleWriteBuffer(MirrorVersion(), p_buffer);

    p_buffer = SimpleSerialization::SimpleWriteBuffer(m_status, p_buffer);
    p_buffer = SimpleSerialization::SimpleWriteBuffer(static_cast<std::uint32_t>(m_allIndexResults.size()), p_buffer);
    for (const auto &indexRes : m_allIndexResults)
    {
        p_buffer = SimpleSerialization::SimpleWriteBuffer(indexRes.m_indexName, p_buffer);

        p_buffer = SimpleSerialization::SimpleWriteBuffer(static_cast<std::uint32_t>(indexRes.m_results.GetResultNum()),
                                                          p_buffer);
        p_buffer = SimpleSerialization::SimpleWriteBuffer(indexRes.m_results.WithMeta(), p_buffer);

        for (const auto &res : indexRes.m_results)
        {
            p_buffer = SimpleSerialization::SimpleWriteBuffer(res.VID, p_buffer);
            p_buffer = SimpleSerialization::SimpleWriteBuffer(res.Dist, p_buffer);
        }

        if (indexRes.m_results.WithMeta())
        {
            for (int i = 0; i < indexRes.m_results.GetResultNum(); ++i)
            {
                p_buffer = SimpleSerialization::SimpleWriteBuffer(indexRes.m_results.GetMetadata(i), p_buffer);
            }
        }
    }

    return p_buffer;
}

const std::uint8_t *RemoteSearchResult::Read(const std::uint8_t *p_buffer)
{
    decltype(MajorVersion()) majorVer = 0;
    decltype(MirrorVersion()) mirrorVer = 0;

    p_buffer = SimpleSerialization::SimpleReadBuffer(p_buffer, majorVer);
    p_buffer = SimpleSerialization::SimpleReadBuffer(p_buffer, mirrorVer);
    if (majorVer != MajorVersion())
    {
        return nullptr;
    }

    p_buffer = SimpleSerialization::SimpleReadBuffer(p_buffer, m_status);

    std::uint32_t len = 0;
    p_buffer = SimpleSerialization::SimpleReadBuffer(p_buffer, len);
    m_allIndexResults.resize(len);

    for (auto &indexRes : m_allIndexResults)
    {
        p_buffer = SimpleSerialization::SimpleReadBuffer(p_buffer, indexRes.m_indexName);

        std::uint32_t resNum = 0;
        p_buffer = SimpleSerialization::SimpleReadBuffer(p_buffer, resNum);

        bool withMeta = false;
        p_buffer = SimpleSerialization::SimpleReadBuffer(p_buffer, withMeta);

        indexRes.m_results.Init(nullptr, resNum, withMeta, false);
        for (auto &res : indexRes.m_results)
        {
            p_buffer = SimpleSerialization::SimpleReadBuffer(p_buffer, res.VID);
            p_buffer = SimpleSerialization::SimpleReadBuffer(p_buffer, res.Dist);
        }

        if (withMeta)
        {
            for (int i = 0; i < indexRes.m_results.GetResultNum(); ++i)
            {
                ByteArray meta;
                p_buffer = SimpleSerialization::SimpleReadBuffer(p_buffer, meta);
                indexRes.m_results.SetMetadata(i, std::move(meta));
            }
        }
    }

    return p_buffer;
}

namespace
{
// Match the existing socket protocol's native-endian scalar encoding, without unaligned loads.
class TenantReader
{
  public:
    TenantReader(const std::uint8_t *data, size_t size) : m_data(data), m_remaining(data ? size : 0)
    {
    }

    template <typename T> bool Read(T &value)
    {
        if (m_remaining < sizeof(T))
            return false;
        memcpy(&value, m_data, sizeof(T));
        m_data += sizeof(T);
        m_remaining -= sizeof(T);
        return true;
    }

    bool Bytes(void *target, size_t size)
    {
        if (size > m_remaining)
            return false;
        if (size)
            memcpy(target, m_data, size);
        m_data += size;
        m_remaining -= size;
        return true;
    }

    size_t Remaining() const
    {
        return m_remaining;
    }

  private:
    const std::uint8_t *m_data;
    size_t m_remaining;
};

template <typename T> void TenantWrite(std::uint8_t *&data, T value)
{
    memcpy(data, &value, sizeof(T));
    data += sizeof(T);
}

constexpr std::uint16_t TenantWireVersion = 1;
constexpr std::uint16_t TenantHeadWireVersion = 2;
constexpr size_t TenantRequestHeaderSize = 29;
constexpr size_t TenantResultHeaderSize = 16;
} // namespace

bool RemoteTenantRequest::Valid() const
{
    const size_t headBytes = headIndexAlgo == IndexAlgoType::Undefined ? 0 : 1;
    if (tenant.empty() || tenant.size() > 1024 ||
        vectors.Length() > c_maxTenantPacketBytes - TenantRequestHeaderSize - headBytes - tenant.size() ||
        (headBytes && (operation != TenantOperation::Create ||
                       (headIndexAlgo != IndexAlgoType::BKT && headIndexAlgo != IndexAlgoType::KDT))))
        return false;
    if (operation == TenantOperation::Delete || operation == TenantOperation::DeleteVector)
        return vectors.Length() == 0 && dimension == 0 && count == 0 && k == 0 &&
               valueType == VectorValueType::Undefined && metric == DistCalcMethod::Undefined &&
               (operation == TenantOperation::Delete ? vectorID == -1 : vectorID >= 0);
    if (dimension <= 0 || count <= 0 || GetValueTypeSize(valueType) == 0 || vectorID != -1 || !vectors.Data() ||
        static_cast<std::uint64_t>(dimension) * count * GetValueTypeSize(valueType) != vectors.Length())
        return false;
    switch (operation)
    {
    case TenantOperation::Create:
        return count >= 64 && k == 0 && (metric == DistCalcMethod::L2 || metric == DistCalcMethod::Cosine);
    case TenantOperation::Insert:
        return k == 0 && metric == DistCalcMethod::Undefined;
    case TenantOperation::Search:
        return count == 1 && k > 0 && static_cast<size_t>(k) <= (c_maxTenantPacketBytes - TenantResultHeaderSize) / 8 &&
               metric == DistCalcMethod::Undefined;
    default:
        return false;
    }
}

bool RemoteTenantRequest::Write(ByteArray &body) const
{
    if (!Valid())
        return false;
    const bool explicitHead = headIndexAlgo != IndexAlgoType::Undefined;
    body = ByteArray::Alloc(TenantRequestHeaderSize + (explicitHead ? 1 : 0) + tenant.size() + vectors.Length());
    auto *data = body.Data();
    TenantWrite(data, explicitHead ? TenantHeadWireVersion : TenantWireVersion);
    TenantWrite(data, static_cast<std::uint8_t>(operation));
    TenantWrite(data, static_cast<std::uint8_t>(valueType));
    TenantWrite(data, static_cast<std::uint8_t>(metric));
    TenantWrite(data, dimension);
    TenantWrite(data, count);
    TenantWrite(data, vectorID);
    TenantWrite(data, k);
    TenantWrite(data, static_cast<std::uint32_t>(tenant.size()));
    TenantWrite(data, static_cast<std::uint32_t>(vectors.Length()));
    if (explicitHead)
        TenantWrite(data, static_cast<std::uint8_t>(headIndexAlgo));
    memcpy(data, tenant.data(), tenant.size());
    if (vectors.Length())
        memcpy(data + tenant.size(), vectors.Data(), vectors.Length());
    return true;
}

bool RemoteTenantRequest::Read(const std::uint8_t *body, size_t size)
{
    if (size > c_maxTenantPacketBytes)
        return false;
    TenantReader reader(body, size);
    RemoteTenantRequest request;
    std::uint16_t version;
    std::uint8_t op, type, distance;
    std::uint32_t tenantLength, vectorLength;
    if (!reader.Read(version) || (version != TenantWireVersion && version != TenantHeadWireVersion) ||
        !reader.Read(op) || !reader.Read(type) || !reader.Read(distance) || !reader.Read(request.dimension) ||
        !reader.Read(request.count) || !reader.Read(request.vectorID) || !reader.Read(request.k) ||
        !reader.Read(tenantLength) || !reader.Read(vectorLength) || tenantLength == 0 || tenantLength > 1024)
        return false;
    request.operation = static_cast<TenantOperation>(op);
    request.valueType = static_cast<VectorValueType>(type);
    request.metric = static_cast<DistCalcMethod>(distance);
    if (version == TenantHeadWireVersion)
    {
        std::uint8_t head;
        if (!reader.Read(head) || request.operation != TenantOperation::Create)
            return false;
        request.headIndexAlgo = static_cast<IndexAlgoType>(head);
        if (request.headIndexAlgo != IndexAlgoType::BKT && request.headIndexAlgo != IndexAlgoType::KDT)
            return false;
    }
    if (static_cast<std::uint64_t>(tenantLength) + vectorLength != reader.Remaining())
        return false;
    request.tenant.resize(tenantLength);
    reader.Bytes(&request.tenant[0], tenantLength);
    // Validate the shape before allocating/copying the payload.
    request.vectors = ByteArray(const_cast<std::uint8_t *>(body + size - vectorLength), vectorLength, false);
    if (!request.Valid())
        return false;
    request.vectors = ByteArray::Alloc(vectorLength);
    reader.Bytes(request.vectors.Data(), vectorLength);
    *this = std::move(request);
    return true;
}

const char *RemoteTenantResult::StatusName() const
{
    switch (status)
    {
#define DefineErrorCode(Name, Value)                                                                                   \
    case ErrorCode::Name:                                                                                              \
        return #Name;
#include "inc/Core/DefinitionList.h"
#undef DefineErrorCode
    default:
        return "Unknown";
    }
}

bool RemoteTenantResult::Write(ByteArray &body) const
{
    if (results.size() > (c_maxTenantPacketBytes - TenantResultHeaderSize) / 8)
        return false;
    body = ByteArray::Alloc(TenantResultHeaderSize + results.size() * 8);
    auto *data = body.Data();
    TenantWrite(data, TenantWireVersion);
    TenantWrite(data, static_cast<std::uint16_t>(status));
    TenantWrite(data, firstID);
    TenantWrite(data, count);
    TenantWrite(data, static_cast<std::uint32_t>(results.size()));
    for (const auto &result : results)
    {
        TenantWrite(data, result.VID);
        TenantWrite(data, result.Dist);
    }
    return true;
}

bool RemoteTenantResult::Read(const std::uint8_t *body, size_t size)
{
    if (size > c_maxTenantPacketBytes)
        return false;
    TenantReader reader(body, size);
    RemoteTenantResult result;
    std::uint16_t version, code;
    std::uint32_t length;
    if (!reader.Read(version) || version != TenantWireVersion || !reader.Read(code) || !reader.Read(result.firstID) ||
        !reader.Read(result.count) || !reader.Read(length) ||
        static_cast<std::uint64_t>(length) * 8 != reader.Remaining() || result.firstID < -1 || result.count < 0)
        return false;
    result.status = static_cast<ErrorCode>(code);
    result.results.resize(length);
    for (auto &entry : result.results)
    {
        reader.Read(entry.VID);
        reader.Read(entry.Dist);
    }
    *this = std::move(result);
    return true;
}
