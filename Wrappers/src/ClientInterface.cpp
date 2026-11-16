// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/ClientInterface.h"
#include "inc/Helper/Base64Encode.h"
#include "inc/Helper/CommonHelper.h"
#include "inc/Helper/Concurrent.h"
#include "inc/Helper/StringConvert.h"

#include <boost/asio.hpp>
#include <future>

AnnClient::AnnClient(const char *p_serverAddr, const char *p_serverPort)
    : m_connectionID(SPTAG::Socket::c_invalidConnectionID), m_timeoutInMilliseconds(9000)
{
    using namespace SPTAG;

    m_socketClient.reset(new Socket::Client(GetHandlerMap(), 2, 30));

    if (nullptr == p_serverAddr || nullptr == p_serverPort)
    {
        return;
    }

    m_server = p_serverAddr;
    m_port = p_serverPort;

    auto connectCallback = [this](Socket::ConnectionID p_cid, ErrorCode p_ec) {
        m_connectionID = p_cid;

        if (ErrorCode::Socket_FailedResolveEndPoint == p_ec)
        {
            return;
        }

        RetryConnect();
    };

    m_socketClient->AsyncConnectToServer(m_server, m_port, connectCallback);

    m_socketClient->SetEventOnConnectionClose([this](Socket::ConnectionID p_cid) {
        m_connectionID = Socket::c_invalidConnectionID;
        RetryConnect();
    });
}

AnnClient::~AnnClient()
{
    m_stopping = true;
    m_reconnectWake.notify_all();
    {
        std::lock_guard<std::mutex> lock(m_reconnectMutex);
    }
    // Stop socket callbacks before destroying either callback registry.
    m_socketClient.reset();
}

void AnnClient::RetryConnect()
{
    std::unique_lock<std::mutex> lock(m_reconnectMutex);
    while (!m_stopping && m_connectionID == SPTAG::Socket::c_invalidConnectionID)
    {
        if (m_reconnectWake.wait_for(lock, std::chrono::seconds(10), [this] { return m_stopping.load(); }))
            return;
        SPTAG::ErrorCode status;
        m_connectionID = m_socketClient->ConnectToServer(m_server, m_port, status);
    }
}

void AnnClient::SetTimeoutMilliseconds(int p_timeout)
{
    m_timeoutInMilliseconds = p_timeout;
}

void AnnClient::SetSearchParam(const char *p_name, const char *p_value)
{
    std::lock_guard<std::mutex> guard(m_paramMutex);

    if (nullptr == p_name || '\0' == *p_name)
    {
        return;
    }

    std::string name(p_name);
    SPTAG::Helper::StrUtils::ToLowerInPlace(name);

    if (nullptr == p_value || '\0' == *p_value)
    {
        m_params.erase(name);
        return;
    }

    m_params[name] = p_value;
}

void AnnClient::ClearSearchParam()
{
    std::lock_guard<std::mutex> guard(m_paramMutex);
    m_params.clear();
}

std::shared_ptr<RemoteSearchResult> AnnClient::Search(ByteArray p_data, int p_resultNum, const char *p_valueType,
                                                      bool p_withMetaData)
{
    using namespace SPTAG;

    SPTAG::Socket::RemoteSearchResult ret;

    SPTAG::VectorValueType valueType = SPTAG::VectorValueType::Undefined;
    SPTAG::Helper::Convert::ConvertStringTo<SPTAG::VectorValueType>(p_valueType, valueType);

    if (Socket::c_invalidConnectionID != m_connectionID && SPTAG::VectorValueType::Undefined != valueType)
    {

        auto signal = std::make_shared<Helper::Concurrent::WaitSignal>(1);

        auto callback = [&ret, signal](RemoteSearchResult p_result) {
            if (RemoteSearchResult::ResultStatus::Success == p_result.m_status)
            {
                ret = std::move(p_result);
            }

            signal->FinishOne();
        };

        auto timeoutCallback = [this](std::shared_ptr<Callback> p_callback) {
            if (nullptr != p_callback)
            {
                RemoteSearchResult result;
                result.m_status = RemoteSearchResult::ResultStatus::Timeout;

                (*p_callback)(std::move(result));
            }
        };

        auto connectCallback = [callback, this](bool p_connectSucc) {
            if (!p_connectSucc)
            {
                RemoteSearchResult result;
                result.m_status = RemoteSearchResult::ResultStatus::FailedNetwork;

                callback(std::move(result));
            }
        };

        Socket::Packet packet;
        packet.Header().m_connectionID = Socket::c_invalidConnectionID;
        packet.Header().m_packetType = Socket::PacketType::SearchRequest;
        packet.Header().m_processStatus = Socket::PacketProcessStatus::Ok;
        packet.Header().m_resourceID = m_callbackManager.Add(std::make_shared<Callback>(std::move(callback)),
                                                             m_timeoutInMilliseconds, std::move(timeoutCallback));

        Socket::RemoteQuery query;
        query.m_queryString = CreateSearchQuery(p_data, p_resultNum, p_withMetaData, valueType);

        packet.Header().m_bodyLength = static_cast<std::uint32_t>(query.EstimateBufferSize());
        packet.AllocateBuffer(packet.Header().m_bodyLength);
        query.Write(packet.Body());
        packet.Header().WriteBuffer(packet.HeaderBuffer());

        m_socketClient->SendPacket(m_connectionID, std::move(packet), connectCallback);

        signal->Wait();
    }
    else
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Error connection or data type!");
    }
    return std::make_shared<RemoteSearchResult>(ret);
}

bool AnnClient::IsConnected() const
{
    return m_connectionID != SPTAG::Socket::c_invalidConnectionID;
}

SPTAG::Socket::PacketHandlerMapPtr AnnClient::GetHandlerMap()
{
    using namespace SPTAG;

    Socket::PacketHandlerMapPtr handlerMap(new Socket::PacketHandlerMap);
    handlerMap->emplace(Socket::PacketType::SearchResponse, std::bind(&AnnClient::SearchResponseHanlder, this,
                                                                      std::placeholders::_1, std::placeholders::_2));
    handlerMap->emplace(Socket::PacketType::TenantResponse, std::bind(&AnnClient::TenantResponseHandler, this,
                                                                      std::placeholders::_1, std::placeholders::_2));

    return handlerMap;
}

void AnnClient::SearchResponseHanlder(SPTAG::Socket::ConnectionID p_localConnectionID, SPTAG::Socket::Packet p_packet)
{
    using namespace SPTAG;

    std::shared_ptr<Callback> callback = m_callbackManager.GetAndRemove(p_packet.Header().m_resourceID);
    if (nullptr == callback)
    {
        return;
    }

    if (p_packet.Header().m_processStatus != Socket::PacketProcessStatus::Ok || 0 == p_packet.Header().m_bodyLength)
    {
        Socket::RemoteSearchResult result;
        result.m_status = Socket::RemoteSearchResult::ResultStatus::FailedExecute;

        (*callback)(std::move(result));
    }
    else
    {
        Socket::RemoteSearchResult result;
        result.Read(p_packet.Body());
        (*callback)(std::move(result));
    }
}

std::string AnnClient::CreateSearchQuery(const ByteArray &p_data, int p_resultNum, bool p_extractMetadata,
                                         SPTAG::VectorValueType p_valueType)
{
    std::stringstream out;

    out << "#";
    std::size_t encLen;
    SPTAG::Helper::Base64::Encode(p_data.Data(), p_data.Length(), out, encLen);

    out << " $datatype:" << SPTAG::Helper::Convert::ConvertToString(p_valueType);
    out << " $resultnum:" << std::to_string(p_resultNum);
    out << " $extractmetadata:" << (p_extractMetadata ? "true" : "false");

    {
        std::lock_guard<std::mutex> guard(m_paramMutex);
        for (const auto &param : m_params)
        {
            out << " $" << param.first << ":" << param.second;
        }
    }

    return out.str();
}

namespace
{
SPTAG::Socket::RemoteTenantRequest TenantRequest(SPTAG::Socket::TenantOperation operation, const char *tenant,
                                                 ByteArray vectors = ByteArray(), int count = 0,
                                                 const char *valueType = nullptr)
{
    SPTAG::Socket::RemoteTenantRequest request;
    request.operation = operation;
    request.tenant = tenant ? tenant : "";
    request.vectors = vectors;
    request.count = count;
    if (valueType)
        SPTAG::Helper::Convert::ConvertStringTo<SPTAG::VectorValueType>(valueType, request.valueType);
    const auto elementSize = SPTAG::GetValueTypeSize(request.valueType);
    if (count > 0 && elementSize > 0)
    {
        const auto stride = static_cast<std::uint64_t>(count) * elementSize;
        if (vectors.Length() % stride == 0 && vectors.Length() / stride <= static_cast<size_t>(SPTAG::MaxSize))
            request.dimension = static_cast<SPTAG::DimensionType>(vectors.Length() / stride);
    }
    return request;
}
} // namespace

std::shared_ptr<RemoteTenantResult> AnnClient::CreateTenant(const char *tenant, ByteArray vectors, int count,
                                                            int dimension, const char *valueType, const char *metric,
                                                            const char *headIndexAlgo)
{
    auto request = TenantRequest(SPTAG::Socket::TenantOperation::Create, tenant, vectors, count, valueType);
    request.dimension = dimension;
    if (metric)
        SPTAG::Helper::Convert::ConvertStringTo<SPTAG::DistCalcMethod>(metric, request.metric);
    if (headIndexAlgo && *headIndexAlgo &&
        (!SPTAG::Helper::Convert::ConvertStringTo<SPTAG::IndexAlgoType>(headIndexAlgo, request.headIndexAlgo) ||
         (request.headIndexAlgo != SPTAG::IndexAlgoType::BKT && request.headIndexAlgo != SPTAG::IndexAlgoType::KDT)))
    {
        auto result = std::make_shared<RemoteTenantResult>();
        result->status = SPTAG::ErrorCode::FailedParseValue;
        return result;
    }
    return SendTenantRequest(request);
}

std::shared_ptr<RemoteTenantResult> AnnClient::DeleteTenant(const char *tenant)
{
    return SendTenantRequest(TenantRequest(SPTAG::Socket::TenantOperation::Delete, tenant));
}

std::shared_ptr<RemoteTenantResult> AnnClient::InsertTenant(const char *tenant, ByteArray vectors, int count,
                                                            const char *valueType)
{
    return SendTenantRequest(TenantRequest(SPTAG::Socket::TenantOperation::Insert, tenant, vectors, count, valueType));
}

std::shared_ptr<RemoteTenantResult> AnnClient::DeleteTenantVector(const char *tenant, int vectorID)
{
    auto request = TenantRequest(SPTAG::Socket::TenantOperation::DeleteVector, tenant);
    request.vectorID = vectorID;
    return SendTenantRequest(request);
}

std::shared_ptr<RemoteTenantResult> AnnClient::SearchTenant(const char *tenant, ByteArray query, int k,
                                                            const char *valueType)
{
    auto request = TenantRequest(SPTAG::Socket::TenantOperation::Search, tenant, query, 1, valueType);
    request.k = k;
    return SendTenantRequest(request);
}

std::shared_ptr<RemoteTenantResult> AnnClient::SendTenantRequest(const SPTAG::Socket::RemoteTenantRequest &request)
{
    using namespace SPTAG;
    auto failure = std::make_shared<RemoteTenantResult>();
    ByteArray body;
    if (!request.Write(body))
    {
        failure->status = ErrorCode::FailedParseValue;
        return failure;
    }
    if (!IsConnected())
    {
        failure->status = ErrorCode::Socket_FailedNetwork;
        return failure;
    }

    auto promise = std::make_shared<std::promise<RemoteTenantResult>>();
    auto future = promise->get_future();
    auto callback = std::make_shared<TenantCallback>(
        [promise](RemoteTenantResult result) { promise->set_value(std::move(result)); });
    Socket::Packet packet;
    packet.Header().m_packetType = Socket::PacketType::TenantRequest;
    // A nonpositive legacy timeout must not leave new synchronous writes waiting forever.
    auto timeout = m_timeoutInMilliseconds.load();
    timeout = timeout && timeout <= MaxSize ? timeout : 9000;
    packet.Header().m_resourceID = m_tenantCallbacks.Add(callback, 0, nullptr);
    const auto resource = packet.Header().m_resourceID;
    packet.Header().m_bodyLength = static_cast<std::uint32_t>(body.Length());
    packet.AllocateBuffer(packet.Header().m_bodyLength);
    memcpy(packet.Body(), body.Data(), body.Length());
    packet.Header().WriteBuffer(packet.HeaderBuffer());
    m_socketClient->SendPacket(m_connectionID, std::move(packet), [this, resource](bool sent) {
        if (!sent)
        {
            auto callback = m_tenantCallbacks.GetAndRemove(resource);
            if (callback)
            {
                RemoteTenantResult result;
                result.status = ErrorCode::Socket_FailedNetwork;
                (*callback)(std::move(result));
            }
        }
    });
    if (future.wait_for(std::chrono::milliseconds(timeout)) != std::future_status::ready)
    {
        m_tenantCallbacks.Remove(resource);
        failure->status = ErrorCode::Socket_Timeout;
        return failure;
    }
    return std::make_shared<RemoteTenantResult>(future.get());
}

void AnnClient::TenantResponseHandler(SPTAG::Socket::ConnectionID, SPTAG::Socket::Packet packet)
{
    auto callback = m_tenantCallbacks.GetAndRemove(packet.Header().m_resourceID);
    if (!callback)
        return;
    RemoteTenantResult result;
    if (packet.Header().m_processStatus != SPTAG::Socket::PacketProcessStatus::Ok)
        result.status = SPTAG::ErrorCode::Socket_UnsupportedRequest;
    else if (!result.Read(packet.Body(), packet.Header().m_bodyLength))
        result.status = SPTAG::ErrorCode::Socket_InvalidMessage;
    (*callback)(std::move(result));
}
