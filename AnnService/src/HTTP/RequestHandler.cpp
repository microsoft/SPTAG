// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/HTTP/RequestHandler.h"
#include "inc/Helper/Logging.h"
#include <sstream>
#include <algorithm>

namespace SPTAG {
namespace HTTP {

RequestHandler::RequestHandler(std::shared_ptr<Service::ServiceContext> p_context)
    : m_context(p_context)
    , m_responseBuilder(std::make_unique<ResponseBuilder>())
{
}

RequestHandler::~RequestHandler()
{
}

http::response<http::string_body> RequestHandler::HandleRequest(
    const http::request<http::string_body>& p_request)
{
    // This is a simple router for synchronous requests
    // Most requests should use the async handlers
    
    std::string target = std::string(p_request.target());
    
    if (target == "/health") {
        return HandleHealthCheck(p_request);
    } else if (target == "/metrics") {
        return HandleMetrics(p_request);
    } else {
        return MakeNotFound(p_request);
    }
}

void RequestHandler::HandleSearchAsync(const http::request<http::string_body>& p_request,
                                      std::function<void(http::response<http::string_body>)> p_callback)
{
    try {
        std::string query, index, error;
        int k = 10;
        
        if (!ParseJsonBody(p_request.body(), query, index, k, error)) {
            p_callback(MakeBadRequest(p_request, error));
            return;
        }
        
        auto callback = [this, p_callback, &p_request](std::shared_ptr<Service::SearchExecutionContext> p_exeContext) {
            if (!p_exeContext) {
                p_callback(MakeServerError(p_request, "Search execution failed"));
                return;
            }
            
            std::string responseBody = m_responseBuilder->BuildSearchResponse(
                p_exeContext->GetResults(), true);
            
            p_callback(MakeSuccessResponse(p_request, responseBody));
        };
        
        // Execute search
        Service::SearchExecutor executor(query.c_str(), m_context, callback);
        executor.Execute();
        
    } catch (const std::exception& e) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "Search handler exception: %s", e.what());
        p_callback(MakeServerError(p_request, e.what()));
    }
}

void RequestHandler::HandleInsertAsync(const http::request<http::string_body>& p_request,
                                      std::function<void(http::response<http::string_body>)> p_callback)
{
    try {
        Socket::RemoteInsertQuery query;
        std::string error;
        
        if (!ParseInsertBody(p_request.body(), query, error)) {
            p_callback(MakeBadRequest(p_request, error));
            return;
        }
        
        auto callback = [this, p_callback, &p_request](std::shared_ptr<Service::InsertExecutionContext> p_exeContext) {
            if (!p_exeContext) {
                p_callback(MakeServerError(p_request, "Insert execution failed"));
                return;
            }
            
            std::string responseBody = m_responseBuilder->BuildInsertResponse(
                p_exeContext->GetResult(), true);
            
            p_callback(MakeSuccessResponse(p_request, responseBody));
        };

        Service::InsertExecutor executor(std::move(query), m_context, callback);
        executor.Execute();
        
    } catch (const std::exception& e) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "Insert handler exception: %s", e.what());
        p_callback(MakeServerError(p_request, e.what()));
    }
}

void RequestHandler::HandleDeleteAsync(const http::request<http::string_body>& p_request,
                                      std::function<void(http::response<http::string_body>)> p_callback)
{
    try {
        Socket::RemoteDeleteQuery query;
        std::string error;
        
        if (!ParseDeleteBody(p_request.body(), query, error)) {
            p_callback(MakeBadRequest(p_request, error));
            return;
        }
        
        auto callback = [this, p_callback, &p_request](std::shared_ptr<Service::DeleteExecutionContext> p_exeContext) {
            if (!p_exeContext) {
                p_callback(MakeServerError(p_request, "Delete execution failed"));
                return;
            }

            std::string responseBody = m_responseBuilder->BuildDeleteResponse(
                p_exeContext->GetResult(), true);
            
            p_callback(MakeSuccessResponse(p_request, responseBody));
        };
        
        Service::DeleteExecutor executor(std::move(query), m_context, callback);
        executor.Execute();
        
    } catch (const std::exception& e) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "Delete handler exception: %s", e.what());
        p_callback(MakeServerError(p_request, e.what()));
    }
}

void RequestHandler::HandleBatchAsync(const http::request<http::string_body>& p_request,
                                     std::function<void(http::response<http::string_body>)> p_callback)
{
    // TODO: Implement batch operations
    p_callback(MakeServerError(p_request, "Batch operations not yet implemented"));
}

http::response<http::string_body> RequestHandler::HandleHealthCheck(
    const http::request<http::string_body>& p_request)
{
    std::string body = m_responseBuilder->BuildHealthResponse(true, "healthy");
    return MakeSuccessResponse(p_request, body);
}

http::response<http::string_body> RequestHandler::HandleMetrics(
    const http::request<http::string_body>& p_request)
{
    // TODO: Get actual metrics from server
    std::string body = m_responseBuilder->BuildMetricsResponse(
        0, 0, 0, 0, 0, 0);
    return MakeSuccessResponse(p_request, body);
}

bool RequestHandler::ParseJsonBody(const std::string& p_body, 
                                  std::string& p_query,
                                  std::string& p_index, 
                                  int& p_k, 
                                  std::string& p_error)
{
    // TODO : Will use a proper JSON library
    
    if (p_body.empty()) {
        p_error = "Empty request body";
        return false;
    }
    
    // Extract query field
    size_t queryPos = p_body.find("\"query\"");
    if (queryPos == std::string::npos) {
        p_error = "Missing 'query' field";
        return false;
    }
    
    size_t queryStart = p_body.find("\"", queryPos + 7) + 1;
    size_t queryEnd = p_body.find("\"", queryStart);
    if (queryStart == std::string::npos || queryEnd == std::string::npos) {
        p_error = "Invalid 'query' field";
        return false;
    }
    
    p_query = p_body.substr(queryStart, queryEnd - queryStart);
    
    // Extract k field (optional)
    size_t kPos = p_body.find("\"k\"");
    if (kPos != std::string::npos) {
        size_t kStart = p_body.find(":", kPos) + 1;
        size_t kEnd = p_body.find_first_of(",}", kStart);
        if (kStart != std::string::npos && kEnd != std::string::npos) {
            std::string kStr = p_body.substr(kStart, kEnd - kStart);
            // Remove whitespace
            kStr.erase(std::remove_if(kStr.begin(), kStr.end(), ::isspace), kStr.end());
            try {
                p_k = std::stoi(kStr);
            } catch (...) {
                p_k = 10; // Default
            }
        }
    }
    
    // Extract index field (optional)
    size_t indexPos = p_body.find("\"index\"");
    if (indexPos != std::string::npos) {
        size_t indexStart = p_body.find("\"", indexPos + 7) + 1;
        size_t indexEnd = p_body.find("\"", indexStart);
        if (indexStart != std::string::npos && indexEnd != std::string::npos) {
            p_index = p_body.substr(indexStart, indexEnd - indexStart);
        }
    }
    
    return true;
}

bool RequestHandler::ParseInsertBody(const std::string& p_body, 
                                    Socket::RemoteInsertQuery& p_query,
                                    std::string& p_error)
{
    if (p_body.empty()) {
        p_error = "Empty request body";
        return false;
    }
    
    // Extract index field
    size_t indexPos = p_body.find("\"index\"");
    if (indexPos != std::string::npos) {
        size_t indexStart = p_body.find("\"", indexPos + 7) + 1;
        size_t indexEnd = p_body.find("\"", indexStart);
        if (indexStart != std::string::npos && indexEnd != std::string::npos) {
            p_query.m_indexName = p_body.substr(indexStart, indexEnd - indexStart);
        }
    }
    
    if (p_query.m_indexName.empty()) {
        p_error = "Missing or empty 'index' field";
        return false;
    }
    
    // Extract vectors array
    size_t vectorsPos = p_body.find("\"vectors\"");
    if (vectorsPos == std::string::npos) {
        p_error = "Missing 'vectors' field";
        return false;
    }
    
    // Find first vector's data field
    size_t dataPos = p_body.find("\"data\"", vectorsPos);
    if (dataPos == std::string::npos) {
        p_error = "Missing 'data' field in vector";
        return false;
    }
    
    size_t dataStart = p_body.find("\"", dataPos + 6) + 1;
    size_t dataEnd = p_body.find("\"", dataStart);
    if (dataStart == std::string::npos || dataEnd == std::string::npos) {
        p_error = "Invalid 'data' field in vector";
        return false;
    }
    
    std::string vectorDataStr = p_body.substr(dataStart, dataEnd - dataStart);
    
    // Parse pipe-separated vector data
    std::vector<int8_t> vectorData;
    std::stringstream ss(vectorDataStr);
    std::string token;
    
    while (std::getline(ss, token, '|')) {
        try {
            vectorData.push_back(static_cast<int8_t>(std::stoi(token)));
        } catch (...) {
            p_error = "Invalid vector data format";
            return false;
        }
    }
    
    if (vectorData.empty()) {
        p_error = "Empty vector data";
        return false;
    }
    
    // Set up the query
    p_query.m_type = Socket::RemoteInsertQuery::InsertType::Vector;
    p_query.m_valueType = SPTAG::VectorValueType::Int8;
    p_query.m_dimension = static_cast<DimensionType>(vectorData.size());
    p_query.m_vectorCount = 1; // For now, support single vector
    p_query.m_vectorData.resize(vectorData.size());
    std::memcpy(p_query.m_vectorData.data(), vectorData.data(), vectorData.size());
    p_query.m_normalized = false;
    p_query.m_withMetaIndex = false;
    
    return true;
}

bool RequestHandler::ParseDeleteBody(const std::string& p_body,
                                    Socket::RemoteDeleteQuery& p_query,
                                    std::string& p_error)
{
    if (p_body.empty()) {
        p_error = "Empty request body";
        return false;
    }
    
    // Extract index field
    size_t indexPos = p_body.find("\"index\"");
    if (indexPos != std::string::npos) {
        size_t indexStart = p_body.find("\"", indexPos + 7) + 1;
        size_t indexEnd = p_body.find("\"", indexStart);
        if (indexStart != std::string::npos && indexEnd != std::string::npos) {
            p_query.m_indexName = p_body.substr(indexStart, indexEnd - indexStart);
        }
    }
    
    if (p_query.m_indexName.empty()) {
        p_error = "Missing or empty 'index' field";
        return false;
    }
    
    // Check for vector_id field
    size_t vectorIdPos = p_body.find("\"vector_id\"");
    if (vectorIdPos != std::string::npos) {
        size_t idStart = p_body.find(":", vectorIdPos);
        if (idStart != std::string::npos) {
            idStart++;
            // Skip whitespace
            while (idStart < p_body.length() && (p_body[idStart] == ' ' || p_body[idStart] == '\t')) {
                idStart++;
            }
            
            size_t idEnd = idStart;
            while (idEnd < p_body.length() && std::isdigit(p_body[idEnd])) {
                idEnd++;
            }
            
            if (idEnd > idStart) {
                try {
                    SizeType vectorId = static_cast<SizeType>(std::stoul(p_body.substr(idStart, idEnd - idStart)));
                    p_query.m_type = Socket::RemoteDeleteQuery::DeleteType::ByVectorId;
                    p_query.m_vectorIds.push_back(vectorId);
                    return true;
                } catch (...) {
                    p_error = "Invalid vector_id format";
                    return false;
                }
            }
        }
    }
    
    // If no vector_id found, check for vectors array with data
    size_t vectorsPos = p_body.find("\"vectors\"");
    if (vectorsPos != std::string::npos) {
        size_t dataPos = p_body.find("\"data\"", vectorsPos);
        if (dataPos != std::string::npos) {
            size_t dataStart = p_body.find("\"", dataPos + 6) + 1;
            size_t dataEnd = p_body.find("\"", dataStart);
            if (dataStart != std::string::npos && dataEnd != std::string::npos) {
                std::string vectorDataStr = p_body.substr(dataStart, dataEnd - dataStart);
                
                // Parse pipe-separated vector data
                std::vector<int8_t> vectorData;
                std::stringstream ss(vectorDataStr);
                std::string token;
                
                while (std::getline(ss, token, '|')) {
                    try {
                        vectorData.push_back(static_cast<int8_t>(std::stoi(token)));
                    } catch (...) {
                        p_error = "Invalid vector data format";
                        return false;
                    }
                }
                
                if (!vectorData.empty()) {
                    p_query.m_type = Socket::RemoteDeleteQuery::DeleteType::ByVector;
                    p_query.m_valueType = SPTAG::VectorValueType::Int8;
                    p_query.m_dimension = static_cast<DimensionType>(vectorData.size());
                    p_query.m_vectorCount = 1;
                    p_query.m_vectorData.resize(vectorData.size());
                    std::memcpy(p_query.m_vectorData.data(), vectorData.data(), vectorData.size());
                    return true;
                }
            }
        }
    }
    
    p_error = "Missing or invalid vector_id or vectors field";
    return false;
}

http::response<http::string_body> RequestHandler::MakeBadRequest(
    const http::request<http::string_body>& p_request,
    const std::string& p_message)
{
    http::response<http::string_body> resp{http::status::bad_request, p_request.version()};
    resp.set(http::field::server, "SPTAG-HTTP/1.0");
    resp.set(http::field::content_type, "application/json");
    resp.body() = m_responseBuilder->BuildErrorResponse(p_message, 400);
    resp.keep_alive(p_request.keep_alive());
    resp.prepare_payload();
    return resp;
}

http::response<http::string_body> RequestHandler::MakeNotFound(
    const http::request<http::string_body>& p_request)
{
    http::response<http::string_body> resp{http::status::not_found, p_request.version()};
    resp.set(http::field::server, "SPTAG-HTTP/1.0");
    resp.set(http::field::content_type, "application/json");
    resp.body() = m_responseBuilder->BuildErrorResponse("Not found", 404);
    resp.keep_alive(p_request.keep_alive());
    resp.prepare_payload();
    return resp;
}

http::response<http::string_body> RequestHandler::MakeServerError(
    const http::request<http::string_body>& p_request,
    const std::string& p_message)
{
    http::response<http::string_body> resp{http::status::internal_server_error, p_request.version()};
    resp.set(http::field::server, "SPTAG-HTTP/1.0");
    resp.set(http::field::content_type, "application/json");
    resp.body() = m_responseBuilder->BuildErrorResponse(p_message, 500);
    resp.keep_alive(p_request.keep_alive());
    resp.prepare_payload();
    return resp;
}

http::response<http::string_body> RequestHandler::MakeSuccessResponse(
    const http::request<http::string_body>& p_request,
    const std::string& p_body)
{
    http::response<http::string_body> resp{http::status::ok, p_request.version()};
    resp.set(http::field::server, "SPTAG-HTTP/1.0");
    resp.set(http::field::content_type, "application/json");
    resp.body() = p_body;
    resp.keep_alive(p_request.keep_alive());
    resp.prepare_payload();
    return resp;
}

} // namespace HTTP
} // namespace SPTAG
