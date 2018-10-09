#ifndef _SPACEV_SERVER_SERVICESTTINGS_H_
#define _SPACEV_SERVER_SERVICESTTINGS_H_

#include "../Core/Common.h"

#include <string>

namespace SpaceV
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
};




} // namespace Server
} // namespace AnnService


#endif // _SPACEV_SERVER_SERVICESTTINGS_H_

