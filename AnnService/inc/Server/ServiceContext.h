#ifndef _SPACEV_SERVER_SERVICECONTEX_H_
#define _SPACEV_SERVER_SERVICECONTEX_H_

#include "inc/Core/VectorIndex.h"
#include "ServiceSettings.h"

#include <memory>
#include <map>

namespace SpaceV
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

private:
    bool m_initialized;

    std::shared_ptr<ServiceSettings> m_settings;

    std::map<std::string, std::shared_ptr<VectorIndex>> m_fullIndexList;
};


} // namespace Server
} // namespace AnnService

#endif // _SPACEV_SERVER_SERVICECONTEX_H_

