// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Server/ServiceSettings.h"

using namespace SPTAG;
using namespace SPTAG::Service;

ServiceSettings::ServiceSettings() 
    : m_defaultMaxResultNumber(10)
    , m_threadNum(12)
    , m_socketThreadNum(8)
    , m_httpListenAddr("0.0.0.0")
    , m_httpListenPort("8080")
    , m_httpThreadNum(8)
    , m_maxHttpConnections(10000)
    , m_enableHTTP(true)
    , m_enableWebSocket(false)
    , m_enableSocket(true)
    , m_httpBufferSize(65536)
    , m_httpTimeout(60)
    , m_httpKeepAlive(300)
    , m_httpPipelining(true)
    , m_httpCompression(false)
{
}
