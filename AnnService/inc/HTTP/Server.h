// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_HTTP_SERVER_H_
#define _SPTAG_HTTP_SERVER_H_

#include "Common.h"
#include "Connection.h"
#include "ConnectionManager.h"
#include "../Socket/Packet.h"
#include "../Server/ServiceContext.h"

#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/strand.hpp>

#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>
#include <set>
#include <mutex>
#include <atomic>
#include <functional>

namespace SPTAG {
namespace HTTP {

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
using tcp = net::ip::tcp;

// Forward declarations
class RequestHandler;
class Connection;

class Server : public std::enable_shared_from_this<Server>
{
public:
    using RouteHandler = std::function<void(HTTPConnectionID, 
                                              http::request<http::string_body>&&,
                                              std::shared_ptr<Connection>)>;
    
    Server(const std::string& p_address,
           const std::string& p_port,
           std::shared_ptr<Service::ServiceContext> p_context,
           std::size_t p_threadNum,
           std::size_t p_maxConnections = 10000);
    
    ~Server();
    
    void Start();
    void Stop();
    
    // Register HTTP route handlers
    void RegisterRoute(const std::string& p_method,
                      const std::string& p_path,
                      RouteHandler p_handler);
    
    // Get a route handler
    RouteHandler GetRouteHandler(const std::string& p_method,
                                  const std::string& p_path) const;
    
    // Send async response
    void SendResponse(HTTPConnectionID p_connection,
                     http::response<http::string_body>&& p_response,
                     std::function<void(bool)> p_callback = nullptr);
    
    // Enable WebSocket upgrade on specific path
    void EnableWebSocket(const std::string& p_path);
    
    // Check if path is WebSocket enabled
    bool IsWebSocketPath(const std::string& p_path) const;
    
    // Check if server is running
    bool IsRunning() const { return m_running.load(); }
    
    // Get service context
    std::shared_ptr<Service::ServiceContext> GetServiceContext() const { return m_serviceContext; }
    
    // Performance metrics
    struct Metrics {
        std::atomic<uint64_t> totalRequests{0};
        std::atomic<uint64_t> activeConnections{0};
        std::atomic<uint64_t> totalBytesReceived{0};
        std::atomic<uint64_t> totalBytesSent{0};
        std::atomic<uint64_t> requestErrors{0};
        std::atomic<uint64_t> avgLatencyMs{0};
    };
    
    const Metrics& GetMetrics() const { return m_metrics; }
    Metrics& GetMetrics() { return m_metrics; }

private:
    void AcceptLoop();
    void HandleAccept(tcp::socket socket, beast::error_code ec);
    void RunIOContext();
    void RegisterDefaultRoutes();
    
private:
    net::io_context m_ioContext;
    tcp::acceptor m_acceptor;
    tcp::endpoint m_endpoint;
    
    std::shared_ptr<ConnectionManager> m_connectionManager;
    std::shared_ptr<Service::ServiceContext> m_serviceContext;
    
    std::vector<std::thread> m_threadPool;
    std::size_t m_threadNum;
    std::size_t m_maxConnections;
    
    // Route table: METHOD -> PATH -> Handler
    mutable std::mutex m_routeMutex;
    std::unordered_map<std::string, 
        std::unordered_map<std::string, RouteHandler>> m_routes;
    
    std::set<std::string> m_websocketPaths;
    
    std::shared_ptr<RequestHandler> m_requestHandler;
    Metrics m_metrics;
    std::atomic<bool> m_running{false};
};

} // namespace HTTP
} // namespace SPTAG

#endif // _SPTAG_HTTP_SERVER_H_
