// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SERVER_SERVICECONTEX_H_
#define _SPTAG_SERVER_SERVICECONTEX_H_

#include "inc/Core/VectorIndex.h"
#include "ServiceSettings.h"

#include <memory>
#include <map>
#ifdef MULTITENANT
#include "inc/Core/SPANN/TenantIndexManager.h"
#endif

namespace SPTAG
{
namespace Service
{

class ServiceContext
{
public:
    ServiceContext(const std::string& p_configFilePath);

    ~ServiceContext();

    const std::map<std::string, std::shared_ptr<VectorIndex>>& GetIndexMap() const;

    const std::shared_ptr<ServiceSettings>& GetServiceSettings() const;

    bool IsInitialized() const;
#ifdef MULTITENANT
    SPANN::TenantIndexManager *GetTenantManager() const
    {
        return m_tenantManager.get();
    }
#endif

  private:
    bool m_initialized;

    std::shared_ptr<ServiceSettings> m_settings;

    std::map<std::string, std::shared_ptr<VectorIndex>> m_fullIndexList;
#ifdef MULTITENANT
    std::unique_ptr<SPANN::TenantIndexManager> m_tenantManager;
#endif
};


} // namespace Server
} // namespace AnnService

#endif // _SPTAG_SERVER_SERVICECONTEX_H_
