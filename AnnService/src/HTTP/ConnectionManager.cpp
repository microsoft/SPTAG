// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/HTTP/ConnectionManager.h"
#include "inc/HTTP/Server.h"
#include "inc/Helper/Logging.h"

namespace SPTAG {
namespace HTTP {

ConnectionManager::ConnectionManager(std::size_t p_maxConnections)
    : m_maxConnections(p_maxConnections)
    , m_nextConnectionID(1)
    , m_connectionCount(0)
{
}

ConnectionManager::~ConnectionManager()
{
    StopAll();
}

std::shared_ptr<Connection> ConnectionManager::AddConnection(tcp::socket&& p_socket,
                                                            std::weak_ptr<Server> p_server)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (m_connectionCount >= m_maxConnections) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                    "Connection limit reached: %zu/%zu",
                    m_connectionCount.load(), m_maxConnections);
        return nullptr;
    }
    
    HTTPConnectionID id = GenerateConnectionID();
    
    auto connection = std::make_shared<Connection>(
        id, 
        std::move(p_socket),
        weak_from_this(),
        p_server);
    
    m_connections[id] = connection;
    m_connectionCount++;
    
    SPTAGLIB_LOG(Helper::LogLevel::LL_Debug,
                "Added connection %llu (total: %zu)",
                id, m_connectionCount.load());
    
    return connection;
}

void ConnectionManager::RemoveConnection(HTTPConnectionID p_id)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    
    auto it = m_connections.find(p_id);
    if (it != m_connections.end()) {
        m_connections.erase(it);
        m_connectionCount--;
        
        SPTAGLIB_LOG(Helper::LogLevel::LL_Debug,
                    "Removed connection %llu (total: %zu)",
                    p_id, m_connectionCount.load());
        
        // Notify callback if set
        if (m_onConnectionClose) {
            m_onConnectionClose(p_id);
        }
    }
}

std::shared_ptr<Connection> ConnectionManager::GetConnection(HTTPConnectionID p_id) const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    
    auto it = m_connections.find(p_id);
    if (it != m_connections.end()) {
        return it->second;
    }
    
    return nullptr;
}

void ConnectionManager::StopAll()
{
    std::vector<std::shared_ptr<Connection>> connections;
    
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        // Copy all connections
        for (const auto& pair : m_connections) {
            connections.push_back(pair.second);
        }
        
        m_connections.clear();
        m_connectionCount = 0;
    }
    
    // Stop all connections outside the lock
    for (auto& conn : connections) {
        if (conn) {
            conn->Stop();
        }
    }
    
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "Stopped all %zu connections", connections.size());
}

void ConnectionManager::SetOnConnectionClose(std::function<void(HTTPConnectionID)> p_callback)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_onConnectionClose = std::move(p_callback);
}

HTTPConnectionID ConnectionManager::GenerateConnectionID()
{
    return m_nextConnectionID.fetch_add(1);
}

} // namespace HTTP
} // namespace SPTAG
