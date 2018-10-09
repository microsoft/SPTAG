#ifndef _SPTAG_PW_CLIENTINTERFACE_H_
#define _SPTAG_PW_CLIENTINTERFACE_H_

#ifndef SWIG

#include "TransferDataType.h"
#include "inc/Core/CommonDataStructure.h"
#include "inc/Socket/Client.h"
#include "inc/Socket/RemoteSearchQuery.h"
#include "inc/Socket/ResourceManager.h"

#include <unordered_map>
#include <atomic>
#include <mutex>

#else
%module SPTAGClient

%{
#include "inc/ClientInterface.h"
%}

%include <std_shared_ptr.i>
%shared_ptr(AnnClient)

%include "PyByteArray.i"

%{
#define SWIG_FILE_WITH_INIT
%}

#endif // SWIG

typedef unsigned int SizeType;

class AnnClient
{
public:
    AnnClient(const char* p_serverAddr, const char* p_serverPort);

    ~AnnClient();

    void SetTimeoutMilliseconds(SizeType p_timeout);

    void SetSearchParam(const char* p_name, const char* p_value);

    void ClearSearchParam();

    RemoteSearchResult Search(ByteArray p_data, SizeType p_resultNum, const char* p_valueType, bool p_withMetaData);

    bool IsConnected() const;

private:
    std::string CreateSearchQuery(const ByteArray& p_data,
                                  SizeType p_resultNum,
                                  bool p_extractMetadata,
                                  SPTAG::VectorValueType p_valueType);

    SPTAG::Socket::PacketHandlerMapPtr GetHandlerMap();

    void SearchResponseHanlder(SPTAG::Socket::ConnectionID p_localConnectionID,
                               SPTAG::Socket::Packet p_packet);

private:
    typedef std::function<void(SPTAG::Socket::RemoteSearchResult)> Callback;

    std::uint32_t m_timeoutInMilliseconds;

    std::string m_server;

    std::string m_port;

    std::unique_ptr<SPTAG::Socket::Client> m_socketClient;

    std::atomic<SPTAG::Socket::ConnectionID> m_connectionID;

    SPTAG::Socket::ResourceManager<Callback> m_callbackManager;

    std::unordered_map<std::string, std::string> m_params;

    std::mutex m_paramMutex;
};

#endif // _SPTAG_PW_CLIENTINTERFACE_H_
