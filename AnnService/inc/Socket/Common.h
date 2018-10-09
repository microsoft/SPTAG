#ifndef _SPTAG_SOCKET_COMMON_H_
#define _SPTAG_SOCKET_COMMON_H_

#include <cstdint>

namespace SPTAG
{
namespace Socket
{

typedef std::uint32_t ConnectionID;

typedef std::uint32_t ResourceID;

extern const ConnectionID c_invalidConnectionID;

extern const ResourceID c_invalidResourceID;

} // namespace Socket
} // namespace SPTAG

#endif // _SPTAG_SOCKET_COMMON_H_
