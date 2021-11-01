// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Aggregator/AggregatorContext.h"
#include "inc/Helper/SimpleIniReader.h"

#include <fstream>

using namespace SPTAG;
using namespace SPTAG::Aggregator;

RemoteMachine::RemoteMachine()
    : m_connectionID(Socket::c_invalidConnectionID),
      m_status(RemoteMachineStatus::Disconnected)
{
}


AggregatorContext::AggregatorContext(const std::string& p_filePath)
    : m_initialized(false)
{
    Helper::IniReader iniReader;
    if (ErrorCode::Success != iniReader.LoadIniFile(p_filePath))
    {
        return;
    }

    m_settings.reset(new AggregatorSettings);

    m_settings->m_listenAddr = iniReader.GetParameter("Service", "ListenAddr", std::string("0.0.0.0"));
    m_settings->m_listenPort = iniReader.GetParameter("Service", "ListenPort", std::string("8100"));
    m_settings->m_threadNum = iniReader.GetParameter("Service", "ThreadNumber", static_cast<std::uint32_t>(8));
    m_settings->m_socketThreadNum = iniReader.GetParameter("Service", "SocketThreadNumber", static_cast<std::uint32_t>(8));
    m_settings->m_centers = iniReader.GetParameter("Service", "Centers", std::string("centers"));
    m_settings->m_valueType = iniReader.GetParameter("Service", "ValueType", VectorValueType::Float);
    m_settings->m_topK = iniReader.GetParameter("Service", "TopK", static_cast<SizeType>(-1));
    m_settings->m_distMethod = iniReader.GetParameter("Service", "DistCalcMethod", DistCalcMethod::L2);
    const std::string emptyStr;

    SizeType serverNum = iniReader.GetParameter("Servers", "Number", static_cast<SizeType>(0));

    for (SizeType i = 0; i < serverNum; ++i)
    {
        std::string sectionName("Server_");
        sectionName += std::to_string(i);
        if (!iniReader.DoesSectionExist(sectionName))
        {
            continue;
        }

        std::shared_ptr<RemoteMachine> remoteMachine(new RemoteMachine);

        remoteMachine->m_address = iniReader.GetParameter(sectionName, "Address", emptyStr);
        remoteMachine->m_port = iniReader.GetParameter(sectionName, "Port", emptyStr);

        if (remoteMachine->m_address.empty() || remoteMachine->m_port.empty())
        {
            continue;
        }

        m_remoteServers.push_back(std::move(remoteMachine));
    }

    if (m_settings->m_topK > 0) {
        std::ifstream inputStream(m_settings->m_centers, std::ifstream::binary);
        if (!inputStream.is_open()) {
            LOG(Helper::LogLevel::LL_Error, "Failed to read file %s.\n", m_settings->m_centers.c_str());
            exit(1);
        }

        SizeType row;
        DimensionType col;
        inputStream.read((char*)&row, sizeof(SizeType));
        inputStream.read((char*)&col, sizeof(DimensionType));
        if (row > serverNum) row = serverNum;
        std::uint64_t totalRecordVectorBytes = ((std::uint64_t)GetValueTypeSize(m_settings->m_valueType)) * row * col;
        ByteArray vectorSet = ByteArray::Alloc(totalRecordVectorBytes);
        char* vecBuf = reinterpret_cast<char*>(vectorSet.Data());
        inputStream.read(vecBuf, totalRecordVectorBytes);
        inputStream.close();

        m_centers.reset(new BasicVectorSet(vectorSet, m_settings->m_valueType, col, row));
    }
    m_initialized = true;
}


AggregatorContext::~AggregatorContext()
{
}


bool
AggregatorContext::IsInitialized() const
{
    return m_initialized;
}


const std::vector<std::shared_ptr<RemoteMachine>>&
AggregatorContext::GetRemoteServers() const
{
    return m_remoteServers;
}


const std::shared_ptr<AggregatorSettings>&
AggregatorContext::GetSettings() const
{
    return m_settings;
}

const std::shared_ptr<VectorSet>&
AggregatorContext::GetCenters() const
{
    return m_centers;
}