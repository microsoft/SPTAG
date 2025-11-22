// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_HTTP_REQUESTHANDLER_H_
#define _SPTAG_HTTP_REQUESTHANDLER_H_

#include "../Server/ServiceContext.h"
#include "../Server/SearchExecutor.h"
#include "../Server/InsertDeleteExecutor.h"
#include "../Socket/RemoteSearchQuery.h"
#include "../Socket/RemoteInsertDeleteQuery.h"
#include "ResponseBuilder.h"

#include <boost/beast/http.hpp>
#include <memory>
#include <functional>
#include <string>

namespace SPTAG {
namespace HTTP {

namespace http = boost::beast::http;

class RequestHandler
{
public:
    RequestHandler(std::shared_ptr<Service::ServiceContext> p_context);
    ~RequestHandler();
    
    // Main request router
    http::response<http::string_body> HandleRequest(
        const http::request<http::string_body>& p_request);
    
    // Async handlers with callbacks
    void HandleSearchAsync(const http::request<http::string_body>& p_request,
                          std::function<void(http::response<http::string_body>)> p_callback);
    
    void HandleInsertAsync(const http::request<http::string_body>& p_request,
                          std::function<void(http::response<http::string_body>)> p_callback);
    
    void HandleDeleteAsync(const http::request<http::string_body>& p_request,
                          std::function<void(http::response<http::string_body>)> p_callback);
    
    void HandleBatchAsync(const http::request<http::string_body>& p_request,
                         std::function<void(http::response<http::string_body>)> p_callback);
    
    void HandleUpdateAsync(const http::request<http::string_body>& p_request,
                          std::function<void(http::response<http::string_body>)> p_callback);
    
    // Health check and metrics
    http::response<http::string_body> HandleHealthCheck(const http::request<http::string_body>& p_request);
    http::response<http::string_body> HandleMetrics(const http::request<http::string_body>& p_request);

private:
    // Parse JSON request body - returns parsed JSON as string for simplicity
    bool ParseJsonBody(const std::string& p_body, std::string& p_query, 
                      std::string& p_index, int& p_k, std::string& p_error);
    
    // Parse insert request
    bool ParseInsertBody(const std::string& p_body, Socket::RemoteInsertQuery& p_query, 
                        std::string& p_error);
    
    // Parse delete request
    bool ParseDeleteBody(const std::string& p_body, Socket::RemoteDeleteQuery& p_query,
                        std::string& p_error);
    
    // Parse update request
    bool ParseUpdateBody(const std::string& p_body, Socket::RemoteDeleteQuery& p_deleteQuery,
                        Socket::RemoteInsertQuery& p_insertQuery, std::string& p_error);
    
    // Convert HTTP request to internal query format
    Socket::RemoteQuery ParseSearchRequest(const std::string& p_body);
    
    // Error responses
    http::response<http::string_body> MakeBadRequest(const http::request<http::string_body>& p_request,
                                                     const std::string& p_message);
    http::response<http::string_body> MakeNotFound(const http::request<http::string_body>& p_request);
    http::response<http::string_body> MakeServerError(const http::request<http::string_body>& p_request,
                                                      const std::string& p_message);
    
    // Success response
    http::response<http::string_body> MakeSuccessResponse(const http::request<http::string_body>& p_request,
                                                          const std::string& p_body);
    
private:
    std::shared_ptr<Service::ServiceContext> m_context;
    std::unique_ptr<ResponseBuilder> m_responseBuilder;
};

} // namespace HTTP
} // namespace SPTAG

#endif // _SPTAG_HTTP_REQUESTHANDLER_H_

