#ifndef _SPACEV_CLIENT_OPTIONS_H_
#define _SPACEV_CLIENT_OPTIONS_H_

#include "inc/Helper/ArgumentsParser.h"

#include <string>
#include <vector>
#include <memory>

namespace SpaceV
{
namespace Client
{

class ClientOptions : public Helper::ArgumentsParser
{
public:
    ClientOptions();

    virtual ~ClientOptions();

    std::string m_serverAddr;

    std::string m_serverPort;

    // in milliseconds.
    std::uint32_t m_searchTimeout;

    std::uint32_t m_threadNum;

    std::uint32_t m_socketThreadNum;

};


} // namespace Socket
} // namespace SpaceV

#endif // _SPACEV_CLIENT_OPTIONS_H_
