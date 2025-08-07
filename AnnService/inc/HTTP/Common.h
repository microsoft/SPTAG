// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_HTTP_COMMON_H_
#define _SPTAG_HTTP_COMMON_H_

// Prevent Windows.h from defining macros that conflict with our code
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifdef DELETE
#undef DELETE
#endif
#endif

#include <cstdint>
#include <string>
#include <chrono>

namespace SPTAG {
namespace HTTP {

// HTTP specific connection ID type
using HTTPConnectionID = std::uint64_t;
constexpr HTTPConnectionID c_invalidHTTPConnectionID = 0;

// HTTP status codes we commonly use
enum class StatusCode : std::uint16_t {
    OK = 200,
    Created = 201,
    Accepted = 202,
    NoContent = 204,
    BadRequest = 400,
    Unauthorized = 401,
    Forbidden = 403,
    NotFound = 404,
    MethodNotAllowed = 405,
    RequestTimeout = 408,
    TooManyRequests = 429,
    InternalServerError = 500,
    BadGateway = 502,
    ServiceUnavailable = 503,
    GatewayTimeout = 504
};

// Request method type
enum class Method : std::uint8_t {
    HTTP_GET,
    HTTP_POST,
    HTTP_PUT,
    HTTP_DELETE,
    HTTP_HEAD,
    HTTP_OPTIONS,
    HTTP_PATCH
};

// Connection state
enum class ConnectionState : std::uint8_t {
    Connecting,
    Connected,
    Closing,
    Closed
};

// Performance metrics
struct RequestMetrics {
    std::chrono::steady_clock::time_point startTime;
    std::chrono::steady_clock::time_point endTime;
    std::size_t bytesReceived{0};
    std::size_t bytesSent{0};
    StatusCode statusCode{StatusCode::OK};
    std::string path;
    Method method{Method::HTTP_GET};
    
    std::chrono::milliseconds GetLatency() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    }
};

// Helper functions
inline const char* MethodToString(Method m) {
    switch (m) {
        case Method::HTTP_GET: return "GET";
        case Method::HTTP_POST: return "POST";
        case Method::HTTP_PUT: return "PUT";
        case Method::HTTP_DELETE: return "DELETE";
        case Method::HTTP_HEAD: return "HEAD";
        case Method::HTTP_OPTIONS: return "OPTIONS";
        case Method::HTTP_PATCH: return "PATCH";
        default: return "UNKNOWN";
    }
}

inline Method StringToMethod(const std::string& s) {
    if (s == "GET") return Method::HTTP_GET;
    if (s == "POST") return Method::HTTP_POST;
    if (s == "PUT") return Method::HTTP_PUT;
    if (s == "DELETE") return Method::HTTP_DELETE;
    if (s == "HEAD") return Method::HTTP_HEAD;
    if (s == "OPTIONS") return Method::HTTP_OPTIONS;
    if (s == "PATCH") return Method::HTTP_PATCH;
    return Method::HTTP_GET;
}

} // namespace HTTP
} // namespace SPTAG

#endif // _SPTAG_HTTP_COMMON_H_
