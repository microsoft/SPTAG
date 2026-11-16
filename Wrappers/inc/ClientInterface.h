// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_PW_CLIENTINTERFACE_H_
#define _SPTAG_PW_CLIENTINTERFACE_H_

#include "TransferDataType.h"
#include "inc/Socket/Client.h"
#include "inc/Socket/ResourceManager.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <unordered_map>

class AnnClient
{
public:
    AnnClient(const char* p_serverAddr, const char* p_serverPort);

    ~AnnClient();

    void SetTimeoutMilliseconds(int p_timeout);

    void SetSearchParam(const char* p_name, const char* p_value);

    void ClearSearchParam();

    std::shared_ptr<RemoteSearchResult> Search(ByteArray p_data, int p_resultNum, const char* p_valueType, bool p_withMetaData);

    std::shared_ptr<RemoteTenantResult> CreateTenant(const char *tenant, ByteArray vectors, int count, int dimension,
                                                     const char *valueType, const char *metric,
                                                     const char *headIndexAlgo = nullptr);
    std::shared_ptr<RemoteTenantResult> DeleteTenant(const char *tenant);
    std::shared_ptr<RemoteTenantResult> InsertTenant(const char *tenant, ByteArray vectors, int count,
                                                     const char *valueType);
    std::shared_ptr<RemoteTenantResult> DeleteTenantVector(const char *tenant, int vectorID);
    std::shared_ptr<RemoteTenantResult> SearchTenant(const char *tenant, ByteArray query, int k, const char *valueType);

    bool IsConnected() const;

private:
    std::string CreateSearchQuery(const ByteArray& p_data,
                                  int p_resultNum,
                                  bool p_extractMetadata,
                                  SPTAG::VectorValueType p_valueType);

    SPTAG::Socket::PacketHandlerMapPtr GetHandlerMap();

    void SearchResponseHanlder(SPTAG::Socket::ConnectionID p_localConnectionID,
                               SPTAG::Socket::Packet p_packet);

    std::shared_ptr<RemoteTenantResult> SendTenantRequest(const SPTAG::Socket::RemoteTenantRequest &request);
    void TenantResponseHandler(SPTAG::Socket::ConnectionID connectionID, SPTAG::Socket::Packet packet);
    void RetryConnect();

  private:
    typedef std::function<void(SPTAG::Socket::RemoteSearchResult)> Callback;

    std::atomic<std::uint32_t> m_timeoutInMilliseconds;

    std::string m_server;

    std::string m_port;

    std::unique_ptr<SPTAG::Socket::Client> m_socketClient;

    std::atomic<SPTAG::Socket::ConnectionID> m_connectionID;

    SPTAG::Socket::ResourceManager<Callback> m_callbackManager;

    using TenantCallback = std::function<void(RemoteTenantResult)>;
    SPTAG::Socket::ResourceManager<TenantCallback> m_tenantCallbacks;
    std::atomic<bool> m_stopping{false};
    std::mutex m_reconnectMutex;
    std::condition_variable m_reconnectWake;

    std::unordered_map<std::string, std::string> m_params;

    std::mutex m_paramMutex;
};

#endif // _SPTAG_PW_CLIENTINTERFACE_H_
