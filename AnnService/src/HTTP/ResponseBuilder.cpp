// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/HTTP/ResponseBuilder.h"
#include <sstream>
#include <iomanip>

namespace SPTAG {
namespace HTTP {

ResponseBuilder::ResponseBuilder()
{
}

ResponseBuilder::~ResponseBuilder()
{
}

std::string ResponseBuilder::BuildSearchResponse(const std::vector<Socket::IndexSearchResult>& p_results,
                                                bool p_success,
                                                const std::string& p_error,
                                                int64_t p_timingMs)
{
    std::stringstream ss;
    ss << "{";
    
    if (p_success) {
        ss << "\"status\":\"success\",";
        ss << "\"results\":[";
        
        bool firstIndex = true;
        for (const auto& indexResult : p_results) {
            if (!firstIndex) ss << ",";
            firstIndex = false;
            
            ss << "{";
            ss << "\"index\":\"" << EscapeJson(indexResult.m_indexName) << "\",";
            ss << "\"items\":[";
            
            bool firstItem = true;
            int idx = 0;
            for (const auto& result : indexResult.m_results) {
                if (!firstItem) ss << ",";
                firstItem = false;
                
                ss << FormatVectorResult(result, indexResult.m_results, idx);
                idx++;
            }
            
            ss << "]"; // items
            ss << "}"; // index result
        }
        
        ss << "]"; // results
        
        if (p_timingMs >= 0) {
            ss << ",\"timing_ms\":" << p_timingMs;
        }
    } else {
        ss << "\"status\":\"error\",";
        ss << "\"error\":\"" << EscapeJson(p_error) << "\"";
    }
    
    ss << "}";
    return ss.str();
}

std::string ResponseBuilder::BuildInsertResponse(const Socket::RemoteInsertDeleteResult& p_result,
                                                bool p_success,
                                                const std::string& p_error,
                                                int64_t p_timingMs)
{
    std::stringstream ss;
    ss << "{";
    
    if (p_success) {
        ss << "\"status\":\"success\",";
        ss << "\"inserted\":" << p_result.m_processedCount << ",";
        
        // Include inserted vector IDs if available
        if (!p_result.m_newVectorIds.empty()) {
            ss << "\"inserted_ids\":[";
            for (size_t i = 0; i < p_result.m_newVectorIds.size(); ++i) {
                if (i > 0) ss << ",";
                ss << p_result.m_newVectorIds[i];
            }
            ss << "],";
        }
        
        if (p_result.m_status == Socket::RemoteInsertDeleteResult::ResultStatus::Success) {
            ss << "\"result\":\"completed\"";
        } else {
            ss << "\"result\":\"partial\",";
            ss << "\"errors\":[";
            // Add error details if available
            ss << "]";
        }
        
        if (p_timingMs >= 0) {
            ss << ",\"timing_ms\":" << p_timingMs;
        }
    } else {
        ss << "\"status\":\"error\",";
        ss << "\"error\":\"" << EscapeJson(p_error) << "\"";
    }
    
    ss << "}";
    return ss.str();
}

std::string ResponseBuilder::BuildDeleteResponse(const Socket::RemoteInsertDeleteResult& p_result,
                                                bool p_success,
                                                const std::string& p_error,
                                                int64_t p_timingMs)
{
    std::stringstream ss;
    ss << "{";
    
    if (p_success) {
        ss << "\"status\":\"success\",";
        ss << "\"deleted\":" << p_result.m_processedCount << ",";
        
        // Include deleted vector IDs if available
        if (!p_result.m_newVectorIds.empty()) {
            ss << "\"deleted_ids\":[";
            for (size_t i = 0; i < p_result.m_newVectorIds.size(); ++i) {
                if (i > 0) ss << ",";
                ss << p_result.m_newVectorIds[i];
            }
            ss << "],";
        }
        
        if (p_result.m_status == Socket::RemoteInsertDeleteResult::ResultStatus::Success) {
            ss << "\"result\":\"completed\"";
        } else {
            ss << "\"result\":\"partial\",";
            ss << "\"errors\":[";
            // Add error details if available
            ss << "]";
        }
        
        if (p_timingMs >= 0) {
            ss << ",\"timing_ms\":" << p_timingMs;
        }
    } else {
        ss << "\"status\":\"error\",";
        ss << "\"error\":\"" << EscapeJson(p_error) << "\"";
    }
    
    ss << "}";
    return ss.str();
}

std::string ResponseBuilder::BuildBatchResponse(const std::vector<std::string>& p_results,
                                               bool p_success,
                                               const std::string& p_error)
{
    std::stringstream ss;
    ss << "{";
    
    if (p_success) {
        ss << "\"status\":\"success\",";
        ss << "\"results\":[";
        
        bool first = true;
        for (const auto& result : p_results) {
            if (!first) ss << ",";
            first = false;
            ss << result;
        }
        
        ss << "]";
    } else {
        ss << "\"status\":\"error\",";
        ss << "\"error\":\"" << EscapeJson(p_error) << "\"";
    }
    
    ss << "}";
    return ss.str();
}

std::string ResponseBuilder::BuildErrorResponse(const std::string& p_error, int p_code)
{
    std::stringstream ss;
    ss << "{";
    ss << "\"status\":\"error\",";
    ss << "\"code\":" << p_code << ",";
    ss << "\"error\":\"" << EscapeJson(p_error) << "\"";
    ss << "}";
    return ss.str();
}

std::string ResponseBuilder::BuildMetricsResponse(uint64_t p_totalRequests,
                                                 uint64_t p_activeConnections,
                                                 uint64_t p_bytesReceived,
                                                 uint64_t p_bytesSent,
                                                 uint64_t p_errors,
                                                 uint64_t p_avgLatency)
{
    std::stringstream ss;
    ss << "{";
    ss << "\"total_requests\":" << p_totalRequests << ",";
    ss << "\"active_connections\":" << p_activeConnections << ",";
    ss << "\"bytes_received\":" << p_bytesReceived << ",";
    ss << "\"bytes_sent\":" << p_bytesSent << ",";
    ss << "\"errors\":" << p_errors << ",";
    ss << "\"avg_latency_ms\":" << p_avgLatency;
    ss << "}";
    return ss.str();
}

std::string ResponseBuilder::BuildHealthResponse(bool p_healthy, const std::string& p_status)
{
    std::stringstream ss;
    ss << "{";
    ss << "\"status\":\"" << (p_healthy ? "healthy" : "unhealthy") << "\",";
    ss << "\"service\":\"AnnService\",";
    ss << "\"details\":\"" << EscapeJson(p_status) << "\"";
    ss << "}";
    return ss.str();
}

std::string ResponseBuilder::EscapeJson(const std::string& p_str)
{
    std::stringstream ss;
    for (char c : p_str) {
        switch (c) {
            case '"': ss << "\\\""; break;
            case '\\': ss << "\\\\"; break;
            case '\b': ss << "\\b"; break;
            case '\f': ss << "\\f"; break;
            case '\n': ss << "\\n"; break;
            case '\r': ss << "\\r"; break;
            case '\t': ss << "\\t"; break;
            default:
                if (c >= 0x20 && c <= 0x7E) {
                    ss << c;
                } else {
                    ss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c;
                }
                break;
        }
    }
    return ss.str();
}

std::string ResponseBuilder::FormatVectorResult(const BasicResult& p_result, const QueryResult& p_queryResult, int p_idx)
{
    std::stringstream ss;
    ss << "{";
    ss << "\"id\":" << p_result.VID << ",";
    ss << "\"distance\":" << p_result.Dist;
    
    // Add metadata if available
    if (p_queryResult.WithMeta() && p_result.Meta.Length() > 0) {
        ss << ",\"metadata\":\"" << EscapeJson(std::string((char*)p_result.Meta.Data(), p_result.Meta.Length())) << "\"";
    }
    
    ss << "}";
    return ss.str();
}

} // namespace HTTP
} // namespace SPTAG