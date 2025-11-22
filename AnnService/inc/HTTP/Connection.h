// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_HTTP_CONNECTION_H_
#define _SPTAG_HTTP_CONNECTION_H_

#include "Common.h"
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/io_context_strand.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/asio/bind_executor.hpp>
#include <memory>
#include <queue>
#include <atomic>
#include <mutex>
#include <functional>

namespace SPTAG {
namespace HTTP {

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
using tcp = net::ip::tcp;

class ConnectionManager;
class Server;

class Connection : public std::enable_shared_from_this<Connection>
{
public:
    Connection(HTTPConnectionID p_id,
              tcp::socket&& p_socket,
              std::weak_ptr<ConnectionManager> p_manager,
              std::weak_ptr<Server> p_server);
    
    ~Connection();
    
    void Start();
    void Stop();
    
    HTTPConnectionID GetID() const { return m_id; }
    
    // Async send response
    void SendResponse(http::response<http::string_body>&& p_response,
                     std::function<void(bool)> p_callback = nullptr);
    
    // Upgrade to WebSocket
    void UpgradeToWebSocket(http::request<http::string_body>&& p_request);
    
    // Get remote endpoint info
    std::string GetRemoteAddress() const;
    uint16_t GetRemotePort() const;
    
    // Connection state
    bool IsAlive() const { return !m_stopped.load(); }
    bool IsWebSocket() const { return m_isWebSocket; }
    
    // Performance tracking
    struct Stats {
        uint64_t bytesReceived{0};
        uint64_t bytesSent{0};
        uint64_t requestsHandled{0};
        std::chrono::steady_clock::time_point connectedTime;
        std::chrono::steady_clock::time_point lastActivityTime;
    };
    
    const Stats& GetStats() const { return m_stats; }
    Stats& GetStats() { return m_stats; }

private:
    void ReadRequest();
    void HandleRequest(beast::error_code ec, std::size_t bytes_transferred);
    void ProcessRequest();
    void WriteResponse();
    void HandleWrite(beast::error_code ec, std::size_t bytes_transferred, 
                    std::function<void(bool)> callback);
    
    void SetupTimeout();
    void CancelTimeout();
    void HandleTimeout(beast::error_code ec);
    
    void OnError(beast::error_code ec, const char* what);
    
private:
    HTTPConnectionID m_id;
    tcp::socket m_socket;
    net::io_context::strand m_strand;
    beast::flat_buffer m_buffer;
    
    std::weak_ptr<ConnectionManager> m_manager;
    std::weak_ptr<Server> m_server;
    
    // HTTP parser and serializer
    http::request<http::string_body> m_request;
    std::shared_ptr<http::response<http::string_body>> m_response;
    
    // Response queue for pipelining
    struct ResponseItem {
        http::response<http::string_body> response;
        std::function<void(bool)> callback;
    };
    std::queue<ResponseItem> m_responseQueue;
    bool m_writing{false};
    
    // Timeout handling
    net::steady_timer m_timer;
    static constexpr auto TIMEOUT_DURATION = std::chrono::seconds(60);
    
    // State
    std::atomic<bool> m_stopped{false};
    bool m_isWebSocket{false};
    
    // Stats
    mutable std::mutex m_statsMutex;
    Stats m_stats;
    
    // WebSocket upgrade (if needed)
    std::shared_ptr<void> m_wsConnection; // WebSocketConnection
};

} // namespace HTTP
} // namespace SPTAG

#endif // _SPTAG_HTTP_CONNECTION_H_
