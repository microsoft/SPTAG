// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/HTTP/Server.h"
#include "inc/HTTP/RequestHandler.h"
#include "inc/Helper/Logging.h"
#include <boost/asio/signal_set.hpp>

namespace SPTAG {
namespace HTTP {

Server::Server(const std::string& p_address,
               const std::string& p_port,
               std::shared_ptr<Service::ServiceContext> p_context,
               std::size_t p_threadNum,
               std::size_t p_maxConnections)
    : m_serviceContext(p_context)
    , m_threadNum(p_threadNum)
    , m_maxConnections(p_maxConnections)
    , m_acceptor(m_ioContext)
    , m_connectionManager(std::make_shared<ConnectionManager>(p_maxConnections))
    , m_requestHandler(std::make_shared<RequestHandler>(p_context))
{
    // Resolve address and port
    tcp::resolver resolver(m_ioContext);
    beast::error_code ec;
    auto const results = resolver.resolve(p_address, p_port, ec);
    
    if (ec) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, 
                    "Failed to resolve address %s:%s - %s",
                    p_address.c_str(), p_port.c_str(), ec.message().c_str());
        throw std::runtime_error("Failed to resolve address");
    }
    
    m_endpoint = results.begin()->endpoint();
    
    // Setup acceptor
    m_acceptor.open(m_endpoint.protocol(), ec);
    if (ec) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, 
                    "Failed to open acceptor - %s", ec.message().c_str());
        throw std::runtime_error("Failed to open acceptor");
    }
    
    m_acceptor.set_option(net::socket_base::reuse_address(true), ec);
    if (ec) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, 
                    "Failed to set reuse_address - %s", ec.message().c_str());
    }
    
    // Enable SO_REUSEPORT on Linux for better load distribution
    #ifdef __linux__
    typedef boost::asio::detail::socket_option::boolean<SOL_SOCKET, SO_REUSEPORT> reuse_port;
    m_acceptor.set_option(reuse_port(true), ec);
    if (ec) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, 
                    "Failed to set SO_REUSEPORT - %s", ec.message().c_str());
    }
    #endif
    
    m_acceptor.bind(m_endpoint, ec);
    if (ec) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, 
                    "Failed to bind to %s:%s - %s",
                    p_address.c_str(), p_port.c_str(), ec.message().c_str());
        throw std::runtime_error("Failed to bind address");
    }
    
    m_acceptor.listen(net::socket_base::max_listen_connections, ec);
    if (ec) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, 
                    "Failed to listen - %s", ec.message().c_str());
        throw std::runtime_error("Failed to listen");
    }
    
    // Register default routes
    RegisterDefaultRoutes();
    
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, 
                "HTTP Server initialized on %s:%s with %zu threads",
                p_address.c_str(), p_port.c_str(), m_threadNum);
}

Server::~Server()
{
    Stop();
}

void Server::Start()
{
    if (m_running.exchange(true)) {
        return; // Already running
    }
    
    // Start accept loop
    AcceptLoop();
    
    // Start worker threads
    m_threadPool.reserve(m_threadNum);
    for (std::size_t i = 0; i < m_threadNum; ++i) {
        m_threadPool.emplace_back([this]() { RunIOContext(); });
    }
    
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "HTTP Server started");
}

void Server::Stop()
{
    if (!m_running.exchange(false)) {
        return; // Already stopped
    }
    
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Stopping HTTP Server...");
    
    // Stop accepting new connections
    beast::error_code ec;
    m_acceptor.close(ec);
    if (ec) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, 
                    "Error closing acceptor: %s", ec.message().c_str());
    }
    
    // Stop all existing connections
    m_connectionManager->StopAll();
    
    // Stop IO context
    m_ioContext.stop();
    
    // Wait for threads
    for (auto& thread : m_threadPool) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "HTTP Server stopped");
}

void Server::AcceptLoop()
{
    if (!m_running.load()) return;
    
    m_acceptor.async_accept(
        net::make_strand(m_ioContext),
        [self = shared_from_this()](beast::error_code ec, tcp::socket socket) {
            self->HandleAccept(std::move(socket), ec);
        });
}

void Server::HandleAccept(tcp::socket socket, beast::error_code ec)
{
    if (ec) {
        if (ec != net::error::operation_aborted) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, 
                        "Accept failed: %s", ec.message().c_str());
        }
        return;
    }
    
    // Check connection limit
    if (m_connectionManager->GetConnectionCount() >= m_maxConnections) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, 
                    "Connection limit reached (%zu/%zu), rejecting new connection",
                    m_connectionManager->GetConnectionCount(), m_maxConnections);
        beast::error_code close_ec;
        socket.close(close_ec);
    } else {
        // Create and start new connection
        auto conn = m_connectionManager->AddConnection(
            std::move(socket), 
            weak_from_this());
        
        if (conn) {
            conn->Start();
            m_metrics.activeConnections++;
            m_metrics.totalRequests++;
        }
    }
    
    // Continue accepting
    AcceptLoop();
}

void Server::RegisterRoute(const std::string& p_method,
                          const std::string& p_path,
                          RouteHandler p_handler)
{
    std::lock_guard<std::mutex> lock(m_routeMutex);
    m_routes[p_method][p_path] = std::move(p_handler);
    
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, 
                "Registered route: %s %s", p_method.c_str(), p_path.c_str());
}

Server::RouteHandler Server::GetRouteHandler(const std::string& p_method,
                                              const std::string& p_path) const
{
    std::lock_guard<std::mutex> lock(m_routeMutex);
    
    auto methodIt = m_routes.find(p_method);
    if (methodIt != m_routes.end()) {
        auto pathIt = methodIt->second.find(p_path);
        if (pathIt != methodIt->second.end()) {
            return pathIt->second;
        }
    }
    
    return nullptr;
}

void Server::RegisterDefaultRoutes()
{
    // Search endpoint
    RegisterRoute("POST", "/v1/search", 
        [this](HTTPConnectionID id, 
               http::request<http::string_body>&& req,
               std::shared_ptr<Connection> conn) {
            m_requestHandler->HandleSearchAsync(std::move(req), 
                [conn](http::response<http::string_body> resp) {
                    conn->SendResponse(std::move(resp));
                });
        });
    
    // Insert endpoint
    RegisterRoute("POST", "/v1/insert",
        [this](HTTPConnectionID id, 
               http::request<http::string_body>&& req,
               std::shared_ptr<Connection> conn) {
            m_requestHandler->HandleInsertAsync(std::move(req),
                [conn](http::response<http::string_body> resp) {
                    conn->SendResponse(std::move(resp));
                });
        });
    
    // Delete endpoint
    RegisterRoute("POST", "/v1/delete",
        [this](HTTPConnectionID id, 
               http::request<http::string_body>&& req,
               std::shared_ptr<Connection> conn) {
            m_requestHandler->HandleDeleteAsync(std::move(req),
                [conn](http::response<http::string_body> resp) {
                    conn->SendResponse(std::move(resp));
                });
        });
    
    // Batch operations
    RegisterRoute("POST", "/v1/batch",
        [this](HTTPConnectionID id, 
               http::request<http::string_body>&& req,
               std::shared_ptr<Connection> conn) {
            m_requestHandler->HandleBatchAsync(std::move(req),
                [conn](http::response<http::string_body> resp) {
                    conn->SendResponse(std::move(resp));
                });
        });
    
    // Health check
    RegisterRoute("GET", "/health",
        [this](HTTPConnectionID id, 
               http::request<http::string_body>&& req,
               std::shared_ptr<Connection> conn) {
            auto resp = m_requestHandler->HandleHealthCheck(std::move(req));
            conn->SendResponse(std::move(resp));
        });
    
    // Metrics
    RegisterRoute("GET", "/metrics",
        [this](HTTPConnectionID id, 
               http::request<http::string_body>&& req,
               std::shared_ptr<Connection> conn) {
            auto resp = m_requestHandler->HandleMetrics(std::move(req));
            conn->SendResponse(std::move(resp));
        });
}

void Server::EnableWebSocket(const std::string& p_path)
{
    std::lock_guard<std::mutex> lock(m_routeMutex);
    m_websocketPaths.insert(p_path);
    
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, 
                "WebSocket enabled on path: %s", p_path.c_str());
}

bool Server::IsWebSocketPath(const std::string& p_path) const
{
    std::lock_guard<std::mutex> lock(m_routeMutex);
    return m_websocketPaths.find(p_path) != m_websocketPaths.end();
}

void Server::RunIOContext()
{
    // Set thread name for debugging
    #ifdef __linux__
    pthread_setname_np(pthread_self(), "http-worker");
    #endif
    
    try {
        m_ioContext.run();
    } catch (const std::exception& e) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, 
                    "IO context error: %s", e.what());
    }
}

} // namespace HTTP
} // namespace SPTAG
