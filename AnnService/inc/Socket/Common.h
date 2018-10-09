#ifndef _SPACEV_SOCKET_COMMON_H_
#define _SPACEV_SOCKET_COMMON_H_

#include <cstdint>

namespace SpaceV
{
namespace Socket
{

typedef std::uint32_t ConnectionID;

typedef std::uint32_t ResourceID;

extern const ConnectionID c_invalidConnectionID;

extern const ResourceID c_invalidResourceID;

} // namespace Socket
} // namespace SpaceV

#endif // _SPACEV_SOCKET_COMMON_H_
