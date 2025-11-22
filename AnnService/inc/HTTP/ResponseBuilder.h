// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_HTTP_RESPONSEBUILDER_H_
#define _SPTAG_HTTP_RESPONSEBUILDER_H_

#include "../Socket/RemoteSearchQuery.h"
#include "../Socket/RemoteInsertDeleteQuery.h"
#include "../Server/SearchExecutionContext.h"
#include "../Server/InsertDeleteExecutor.h"

#include <string>
#include <sstream>
#include <memory>

namespace SPTAG {
namespace HTTP {

class ResponseBuilder
{
public:
    ResponseBuilder();
    ~ResponseBuilder();
    
    // Build JSON response from search results
    std::string BuildSearchResponse(const std::vector<Socket::IndexSearchResult>& p_results,
                                   bool p_success = true,
                                   const std::string& p_error = "",
                                   int64_t p_timingMs = -1);
    
    // Build JSON response from insert results
    std::string BuildInsertResponse(const Socket::RemoteInsertDeleteResult& p_result,
                                   bool p_success = true,
                                   const std::string& p_error = "",
                                   int64_t p_timingMs = -1);
    
    // Build JSON response from delete results
    std::string BuildDeleteResponse(const Socket::RemoteInsertDeleteResult& p_result,
                                   bool p_success = true,
                                   const std::string& p_error = "",
                                   int64_t p_timingMs = -1);
    
    // Build batch response
    std::string BuildBatchResponse(const std::vector<std::string>& p_results,
                                  bool p_success = true,
                                  const std::string& p_error = "");
    
    // Build error response
    std::string BuildErrorResponse(const std::string& p_error,
                                  int p_code = 500);
    
    // Build metrics response
    std::string BuildMetricsResponse(uint64_t p_totalRequests,
                                    uint64_t p_activeConnections,
                                    uint64_t p_bytesReceived,
                                    uint64_t p_bytesSent,
                                    uint64_t p_errors,
                                    uint64_t p_avgLatency);
    
    // Build health check response
    std::string BuildHealthResponse(bool p_healthy = true,
                                   const std::string& p_status = "healthy");

private:
    // Helper to escape JSON strings
    std::string EscapeJson(const std::string& p_str);
    
    // Helper to format vector results
    std::string FormatVectorResult(const BasicResult& p_result, const QueryResult& p_queryResult, int p_idx);
};

} // namespace HTTP
} // namespace SPTAG

#endif // _SPTAG_HTTP_RESPONSEBUILDER_H_