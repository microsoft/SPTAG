// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/HTTP/Connection.h"
#include "inc/HTTP/ConnectionManager.h"
#include "inc/HTTP/Server.h"
#include "inc/Helper/Logging.h"
#include <boost/asio/bind_executor.hpp>

namespace SPTAG {
namespace HTTP {

Connection::Connection(HTTPConnectionID p_id,
                      tcp::socket&& p_socket,
                      std::weak_ptr<ConnectionManager> p_manager,
                      std::weak_ptr<Server> p_server)
    : m_id(p_id)
    , m_socket(std::move(p_socket))
    , m_strand(static_cast<net::io_context&>(m_socket.get_executor().context()))
    , m_manager(p_manager)
    , m_server(p_server)
    , m_timer(static_cast<net::io_context&>(m_socket.get_executor().context()))
    , m_stopped(false)
    , m_isWebSocket(false)
    , m_writing(false)
{
    m_stats.connectedTime = std::chrono::steady_clock::now();
    m_stats.lastActivityTime = m_stats.connectedTime;
    
    // Set TCP no delay for lower latency
    beast::error_code ec;
    m_socket.set_option(tcp::no_delay(true), ec);
    if (ec) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                    "Failed to set TCP_NODELAY: %s", ec.message().c_str());
    }
}

Connection::~Connection()
{
    Stop();
}

void Connection::Start()
{
    SPTAGLIB_LOG(Helper::LogLevel::LL_Debug,
                "Connection %llu started from %s:%u",
                m_id, GetRemoteAddress().c_str(), GetRemotePort());
    
    ReadRequest();
    
    SetupTimeout();
}

void Connection::Stop()
{
    if (m_stopped.exchange(true)) {
        return; // Already stopped
    }
    
    SPTAGLIB_LOG(Helper::LogLevel::LL_Debug,
                "Connection %llu stopping", m_id);
    
    // Cancel timer
    beast::error_code ec;
    m_timer.cancel(ec);
    
    // Close socket
    m_socket.shutdown(tcp::socket::shutdown_both, ec);
    m_socket.close(ec);
    
    // Notify manager
    if (auto manager = m_manager.lock()) {
        manager->RemoveConnection(m_id);
    }
    
    // Update server metrics
    if (auto server = m_server.lock()) {
        server->GetMetrics().activeConnections--;
    }
}

void Connection::ReadRequest()
{
    if (m_stopped.load()) return;
    
    // Clear previous request
    m_request = {};
    
    // Read HTTP request
    http::async_read(m_socket, m_buffer, m_request,
        boost::asio::bind_executor(
            m_strand,
            [self = shared_from_this()](beast::error_code ec, std::size_t bytes) {
                self->HandleRequest(ec, bytes);
            }));
}

void Connection::HandleRequest(beast::error_code ec, std::size_t bytes_transferred)
{
    if (ec == http::error::end_of_stream) {
        // Connection closed gracefully
        Stop();
        return;
    }
    
    if (ec) {
        OnError(ec, "read");
        return;
    }
    
    // Update stats
    {
        std::lock_guard<std::mutex> lock(m_statsMutex);
        m_stats.bytesReceived += bytes_transferred;
        m_stats.requestsHandled++;
        m_stats.lastActivityTime = std::chrono::steady_clock::now();
    }
    
    // Update server metrics
    if (auto server = m_server.lock()) {
        server->GetMetrics().totalBytesReceived += bytes_transferred;
    }
    
    // Reset timeout
    CancelTimeout();
    
    // Process the request
    ProcessRequest();
    
    // Continue reading if keep-alive
    if (!m_stopped.load() && m_request.keep_alive()) {
        ReadRequest();
        SetupTimeout();
    }
}

void Connection::ProcessRequest()
{
    auto server = m_server.lock();
    if (!server) {
        Stop();
        return;
    }
    
    // Get method and target
    std::string method = std::string(m_request.method_string());
    std::string target = std::string(m_request.target());
    
    // Remove query parameters from target
    auto query_pos = target.find('?');
    if (query_pos != std::string::npos) {
        target = target.substr(0, query_pos);
    }
    
    SPTAGLIB_LOG(Helper::LogLevel::LL_Debug,
                "Connection %llu: %s %s",
                m_id, method.c_str(), target.c_str());
    
    // Check for WebSocket upgrade
    if (server->IsWebSocketPath(target) && 
        m_request[http::field::upgrade] == "websocket") {
        UpgradeToWebSocket(std::move(m_request));
        return;
    }
    
    // Get route handler
    auto handler = server->GetRouteHandler(method, target);
    
    if (handler) {
        // Execute handler
        try {
            handler(m_id, std::move(m_request), shared_from_this());
        } catch (const std::exception& e) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                        "Handler exception: %s", e.what());
            
            // Send error response
            http::response<http::string_body> resp{http::status::internal_server_error, m_request.version()};
            resp.set(http::field::server, "SPTAG-HTTP/1.0");
            resp.set(http::field::content_type, "application/json");
            resp.body() = R"({"error":"Internal server error"})";
            resp.prepare_payload();
            SendResponse(std::move(resp));
        }
    } else {
        // Send 404 response
        http::response<http::string_body> resp{http::status::not_found, m_request.version()};
        resp.set(http::field::server, "SPTAG-HTTP/1.0");
        resp.set(http::field::content_type, "application/json");
        resp.body() = R"({"error":"Not found"})";
        resp.keep_alive(m_request.keep_alive());
        resp.prepare_payload();
        SendResponse(std::move(resp));
    }
}

void Connection::SendResponse(http::response<http::string_body>&& p_response,
                             std::function<void(bool)> p_callback)
{
    if (m_stopped.load()) {
        if (p_callback) p_callback(false);
        return;
    }
    
    // Queue the response
    m_strand.post(
        [self = shared_from_this(), resp = std::move(p_response), callback = std::move(p_callback)]() mutable {
            self->m_responseQueue.push({std::move(resp), std::move(callback)});
            
            // Start writing if not already writing
            if (!self->m_writing) {
                self->m_writing = true;
                self->WriteResponse();
            }
        });
}

void Connection::WriteResponse()
{
    if (m_responseQueue.empty()) {
        m_writing = false;
        return;
    }
    
    // Get next response
    auto item = std::move(m_responseQueue.front());
    m_responseQueue.pop();
    
    // Store response for async write
    m_response = std::make_shared<http::response<http::string_body>>(std::move(item.response));
    
    // Update stats
    {
        std::lock_guard<std::mutex> lock(m_statsMutex);
        m_stats.bytesSent += m_response->body().size();
    }
    
    // Update server metrics
    if (auto server = m_server.lock()) {
        server->GetMetrics().totalBytesSent += m_response->body().size();
    }
    
    // Async write
    http::async_write(m_socket, *m_response,
        boost::asio::bind_executor(
            m_strand,
            [self = shared_from_this(), callback = std::move(item.callback)]
            (beast::error_code ec, std::size_t bytes) {
                self->HandleWrite(ec, bytes, callback);
            }));
}

void Connection::HandleWrite(beast::error_code ec, std::size_t bytes_transferred,
                            std::function<void(bool)> callback)
{
    if (ec) {
        OnError(ec, "write");
        if (callback) callback(false);
        return;
    }
    
    // Success callback
    if (callback) callback(true);
    
    // Check if we should close after sending
    if (m_response && !m_response->keep_alive()) {
        Stop();
        return;
    }
    
    // Process next response in queue
    WriteResponse();
}

void Connection::UpgradeToWebSocket(http::request<http::string_body>&& p_request)
{
    // TODO: implement WebSocket upgrade
    // Send error response
    http::response<http::string_body> resp{http::status::not_implemented, p_request.version()};
    resp.set(http::field::server, "SPTAG-HTTP/1.0");
    resp.set(http::field::content_type, "application/json");
    resp.body() = R"({"error":"WebSocket not implemented"})";
    resp.prepare_payload();
    SendResponse(std::move(resp));
}

void Connection::SetupTimeout()
{
    if (m_stopped.load()) return;
    
    m_timer.expires_after(TIMEOUT_DURATION);
    m_timer.async_wait(
        boost::asio::bind_executor(
            m_strand,
            [self = shared_from_this()](beast::error_code ec) {
                self->HandleTimeout(ec);
            }));
}

void Connection::CancelTimeout()
{
    beast::error_code ec;
    m_timer.cancel(ec);
}

void Connection::HandleTimeout(beast::error_code ec)
{
    if (ec && ec != boost::asio::error::operation_aborted) {
        return;
    }
    
    if (!ec) {
        // Timeout occurred
        SPTAGLIB_LOG(Helper::LogLevel::LL_Debug,
                    "Connection %llu timed out", m_id);
        Stop();
    }
}

void Connection::OnError(beast::error_code ec, const char* what)
{
    if (ec == net::error::operation_aborted) {
        return; // Normal during shutdown
    }
    
    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                "Connection %llu error during %s: %s",
                m_id, what, ec.message().c_str());
    
    // Update server metrics
    if (auto server = m_server.lock()) {
        server->GetMetrics().requestErrors++;
    }
    
    Stop();
}

std::string Connection::GetRemoteAddress() const
{
    try {
        return m_socket.remote_endpoint().address().to_string();
    } catch (...) {
        return "unknown";
    }
}

uint16_t Connection::GetRemotePort() const
{
    try {
        return m_socket.remote_endpoint().port();
    } catch (...) {
        return 0;
    }
}

} // namespace HTTP
} // namespace SPTAG
