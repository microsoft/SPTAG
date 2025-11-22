// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SERVER_SERVICESTTINGS_H_
#define _SPTAG_SERVER_SERVICESTTINGS_H_

#include "../Core/Common.h"

#include <string>

namespace SPTAG
{
namespace Service
{

struct ServiceSettings
{
    ServiceSettings();

    std::string m_vectorSeparator;

    std::string m_listenAddr;

    std::string m_listenPort;

    SizeType m_defaultMaxResultNumber;

    SizeType m_threadNum;

    SizeType m_socketThreadNum;
    
    // HTTP Configuration
    std::string m_httpListenAddr;

    std::string m_httpListenPort;

    SizeType m_httpThreadNum;

    SizeType m_maxHttpConnections;

    bool m_enableHTTP;

    bool m_enableWebSocket;

    bool m_enableSocket;
    
    // HTTP Performance Tuning
    SizeType m_httpBufferSize;

    SizeType m_httpTimeout;

    SizeType m_httpKeepAlive;

    bool m_httpPipelining;
    
    bool m_httpCompression;
};




} // namespace Server
} // namespace AnnService


#endif // _SPTAG_SERVER_SERVICESTTINGS_H_

