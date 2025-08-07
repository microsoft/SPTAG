// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_HTTP_CONNECTIONMANAGER_H_
#define _SPTAG_HTTP_CONNECTIONMANAGER_H_

#include "Common.h"
#include "Connection.h"
#include <boost/asio/ip/tcp.hpp>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <atomic>

namespace SPTAG {
namespace HTTP {

namespace net = boost::asio;
using tcp = net::ip::tcp;

class Server;

class ConnectionManager : public std::enable_shared_from_this<ConnectionManager>
{
public:
    ConnectionManager(std::size_t p_maxConnections);
    ~ConnectionManager();
    
    // Add a new connection
    std::shared_ptr<Connection> AddConnection(tcp::socket&& p_socket,
                                             std::weak_ptr<Server> p_server);
    
    // Remove a connection
    void RemoveConnection(HTTPConnectionID p_id);
    
    // Get a connection by ID
    std::shared_ptr<Connection> GetConnection(HTTPConnectionID p_id) const;
    
    // Stop all connections
    void StopAll();
    
    // Get current connection count
    std::size_t GetConnectionCount() const { return m_connectionCount.load(); }
    
    // Get max connections
    std::size_t GetMaxConnections() const { return m_maxConnections; }
    
    // Set connection close callback
    void SetOnConnectionClose(std::function<void(HTTPConnectionID)> p_callback);

private:
    HTTPConnectionID GenerateConnectionID();
    
private:
    mutable std::mutex m_mutex;
    std::unordered_map<HTTPConnectionID, std::shared_ptr<Connection>> m_connections;
    
    std::atomic<HTTPConnectionID> m_nextConnectionID{1};
    std::atomic<std::size_t> m_connectionCount{0};
    std::size_t m_maxConnections;
    
    std::function<void(HTTPConnectionID)> m_onConnectionClose;
};

} // namespace HTTP
} // namespace SPTAG

#endif // _SPTAG_HTTP_CONNECTIONMANAGER_H_
