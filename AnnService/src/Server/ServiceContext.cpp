// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Server/ServiceContext.h"
#include "inc/Helper/CommonHelper.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Helper/StringConvert.h"

using namespace SPTAG;
using namespace SPTAG::Service;

ServiceContext::ServiceContext(const std::string &p_configFilePath) : m_initialized(false)
{
    Helper::IniReader iniReader;
    if (ErrorCode::Success != iniReader.LoadIniFile(p_configFilePath))
    {
        return;
    }

    m_settings.reset(new ServiceSettings);

    m_settings->m_listenAddr = iniReader.GetParameter("Service", "ListenAddr", std::string("0.0.0.0"));
    m_settings->m_listenPort = iniReader.GetParameter("Service", "ListenPort", std::string("8000"));
    m_settings->m_threadNum = iniReader.GetParameter("Service", "ThreadNumber", static_cast<std::uint32_t>(8));
    m_settings->m_socketThreadNum =
        iniReader.GetParameter("Service", "SocketThreadNumber", static_cast<std::uint32_t>(8));

    m_settings->m_defaultMaxResultNumber =
        iniReader.GetParameter("QueryConfig", "DefaultMaxResultNumber", static_cast<SizeType>(10));
    m_settings->m_vectorSeparator = iniReader.GetParameter("QueryConfig", "DefaultSeparator", std::string("|"));

    const std::string emptyStr;

    auto tenantRoot = iniReader.GetParameter("MultiTenant", "Root", emptyStr);
    if (!tenantRoot.empty())
    {
#ifdef MULTITENANT
        SPANN::TenantIndexManager::Limits limits;
        limits.maxTenants = iniReader.GetParameter("MultiTenant", "MaxTenants", limits.maxTenants);
        limits.maxVectorsPerTenant =
            iniReader.GetParameter("MultiTenant", "MaxVectorsPerTenant", limits.maxVectorsPerTenant);
        limits.maxDimension = iniReader.GetParameter("MultiTenant", "MaxDimension", limits.maxDimension);
        limits.maxResults = iniReader.GetParameter("MultiTenant", "MaxResults", limits.maxResults);
        limits.cacheBytes = iniReader.GetParameter("MultiTenant", "CacheBytes", limits.cacheBytes);
        limits.maxResidentTenants =
            iniReader.GetParameter("MultiTenant", "MaxResidentTenants", limits.maxResidentTenants);
        limits.maxResidentBytes = iniReader.GetParameter("MultiTenant", "MaxResidentBytes", limits.maxResidentBytes);
        limits.residentWaitMilliseconds =
            iniReader.GetParameter("MultiTenant", "ResidentWaitMilliseconds", limits.residentWaitMilliseconds);
        limits.autoCheckpoint = iniReader.GetParameter("MultiTenant", "AutoCheckpoint", limits.autoCheckpoint);
        limits.checkpointLogBytes =
            iniReader.GetParameter("MultiTenant", "CheckpointLogBytes", limits.checkpointLogBytes);
        limits.checkpointPressureBytes =
            iniReader.GetParameter("MultiTenant", "CheckpointPressureBytes", limits.checkpointPressureBytes);
        limits.checkpointMinIntervalSeconds =
            iniReader.GetParameter("MultiTenant", "CheckpointMinIntervalSeconds", limits.checkpointMinIntervalSeconds);
        const auto storage = iniReader.GetParameter("MultiTenant", "Storage", std::string("ROCKSDBIO"));
        if (!Helper::Convert::ConvertStringTo<Storage>(storage.c_str(), limits.posting.storage))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Invalid MultiTenant.Storage\n");
            return;
        }
        limits.posting.tikvPDAddresses =
            iniReader.GetParameter("MultiTenant", "TiKVPDAddresses", limits.posting.tikvPDAddresses);
        limits.posting.tikvKeyPrefix =
            iniReader.GetParameter("MultiTenant", "TiKVKeyPrefix", limits.posting.tikvKeyPrefix);
        m_initialized = SPANN::TenantIndexManager::Open(tenantRoot, limits, m_tenantManager) == ErrorCode::Success;
#else
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "MultiTenant.Root requires a MULTITENANT build\n");
#endif
        return;
    }

    std::string indexListStr = iniReader.GetParameter("Index", "List", emptyStr);
    const auto &indexList = Helper::StrUtils::SplitString(indexListStr, ",");

    for (const auto &indexName : indexList)
    {
        std::string sectionName("Index_");
        sectionName += indexName.c_str();
        if (!iniReader.DoesParameterExist(sectionName, "IndexFolder"))
        {
            continue;
        }

        std::string indexFolder = iniReader.GetParameter(sectionName, "IndexFolder", emptyStr);

        std::shared_ptr<VectorIndex> vectorIndex;
        if (ErrorCode::Success == VectorIndex::LoadIndex(indexFolder, vectorIndex))
        {
            vectorIndex->SetIndexName(indexName);
            m_fullIndexList.emplace(indexName, vectorIndex);
        }
        else
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed loading index: %s\n", indexName.c_str());
        }
    }

    m_initialized = true;
}

ServiceContext::~ServiceContext()
{
}

const std::map<std::string, std::shared_ptr<VectorIndex>> &ServiceContext::GetIndexMap() const
{
    return m_fullIndexList;
}

const std::shared_ptr<ServiceSettings> &ServiceContext::GetServiceSettings() const
{
    return m_settings;
}

bool ServiceContext::IsInitialized() const
{
    return m_initialized;
}
