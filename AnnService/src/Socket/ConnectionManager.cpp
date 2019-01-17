// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Socket/ConnectionManager.h"

using namespace SPTAG::Socket;


ConnectionManager::ConnectionItem::ConnectionItem()
    : m_isEmpty(true)
{
}


ConnectionManager::ConnectionManager()
    : m_nextConnectionID(1),
      m_connectionCount(0)
{
}


ConnectionID
ConnectionManager::AddConnection(boost::asio::ip::tcp::socket&& p_socket,
                                 const PacketHandlerMapPtr& p_handler,
                                 std::uint32_t p_heartbeatIntervalSeconds)
{
    ConnectionID currID = m_nextConnectionID.fetch_add(1);
    while (c_invalidConnectionID == currID || !m_connections[GetPosition(currID)].m_isEmpty.exchange(false))
    {
        if (m_connectionCount >= c_connectionPoolSize)
        {
            return c_invalidConnectionID;
        }

        currID = m_nextConnectionID.fetch_add(1);
    }

    ++m_connectionCount;

    auto connection = std::make_shared<Connection>(currID,
                                                   std::move(p_socket),
                                                   p_handler,
                                                   std::weak_ptr<ConnectionManager>(shared_from_this()));

    m_spinLock.Lock();
    m_connections[GetPosition(currID)].m_connection = connection;
    m_spinLock.Unlock();

    connection->Start();
    if (p_heartbeatIntervalSeconds > 0)
    {
        connection->StartHeartbeat(p_heartbeatIntervalSeconds);
    }

    return currID;
}


void
ConnectionManager::RemoveConnection(ConnectionID p_connectionID)
{
    auto position = GetPosition(p_connectionID);
    if (m_connections[position].m_isEmpty.exchange(true))
    {
        return;
    }

    Connection::Ptr conn;

    m_spinLock.Lock();
    conn = std::move(m_connections[position].m_connection);
    m_spinLock.Unlock();

    --m_connectionCount;

    conn->Stop();
    conn.reset();

    if (bool(m_eventOnRemoving))
    {
        m_eventOnRemoving(p_connectionID);
    }
}


Connection::Ptr
ConnectionManager::GetConnection(ConnectionID p_connectionID)
{
    auto position = GetPosition(p_connectionID);
    Connection::Ptr ret;

    m_spinLock.Lock();
    ret = m_connections[position].m_connection;
    m_spinLock.Unlock();

    if (nullptr == ret || ret->GetConnectionID() != p_connectionID)
    {
        return nullptr;
    }

    return ret;
}


void
ConnectionManager::SetEventOnRemoving(std::function<void(ConnectionID)> p_event)
{
    m_eventOnRemoving = std::move(p_event);
}


void
ConnectionManager::StopAll()
{
    m_spinLock.Lock();
    for (auto& connection : m_connections)
    {
        if (nullptr != connection.m_connection)
        {
            connection.m_connection->Stop();
        }
    }

    m_spinLock.Unlock();
}


std::uint32_t
ConnectionManager::GetPosition(ConnectionID p_connectionID)
{
    return static_cast<std::uint32_t>(p_connectionID) & c_connectionPoolMask;
}
