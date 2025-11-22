#ifndef _SPTAG_SERVER_INSERTDELETEEXECUTORH
#define _SPTAG_SERVER_INSERTDELETEEXECUTORH

#include "ServiceContext.h"
#include "inc/Socket/RemoteInsertDeleteQuery.h"
#include "inc/Core/VectorIndex.h"

#include <memory>
#include <string>
#include <functional>

namespace SPTAG
{
namespace Service
{

class InsertExecutionContext
{
public:
    InsertExecutionContext(std::shared_ptr<ServiceSettings> p_settings);
    ~InsertExecutionContext();

    ErrorCode ParseQuery(const Socket::RemoteInsertQuery& p_query);
    const Socket::RemoteInsertDeleteResult& GetResult() const { return m_result; }
    Socket::RemoteInsertDeleteResult& GetResult() { return m_result; }

private:
    std::shared_ptr<ServiceSettings> m_settings;
    Socket::RemoteInsertDeleteResult m_result;
    Socket::RemoteInsertQuery m_query;
};

class DeleteExecutionContext
{
public:
    DeleteExecutionContext(std::shared_ptr<ServiceSettings> p_settings);
    ~DeleteExecutionContext();

    ErrorCode ParseQuery(const Socket::RemoteDeleteQuery& p_query);
    const Socket::RemoteInsertDeleteResult& GetResult() const { return m_result; }
    Socket::RemoteInsertDeleteResult& GetResult() { return m_result; }

private:
    std::shared_ptr<ServiceSettings> m_settings;
    Socket::RemoteInsertDeleteResult m_result;
    Socket::RemoteDeleteQuery m_query;
};

class InsertExecutor
{
public:
    typedef std::function<void(std::shared_ptr<InsertExecutionContext>)> CallBack;

    InsertExecutor(Socket::RemoteInsertQuery p_query, std::shared_ptr<ServiceContext> p_serviceContext,
                   const CallBack& p_callback);

    ~InsertExecutor();

    void Execute();

private:
    void ExecuteInternal();
    void SelectIndex();

private:
    CallBack m_callback;
    std::shared_ptr<ServiceContext> c_serviceContext;
    Socket::RemoteInsertQuery m_query;
    std::shared_ptr<InsertExecutionContext> m_executionContext;
    std::vector<std::shared_ptr<VectorIndex>> m_selectedIndex;
};

class DeleteExecutor
{
public:
    typedef std::function<void(std::shared_ptr<DeleteExecutionContext>)> CallBack;

    DeleteExecutor(Socket::RemoteDeleteQuery p_query, std::shared_ptr<ServiceContext> p_serviceContext,
                   const CallBack& p_callback);

    ~DeleteExecutor();

    void Execute();

private:
    void ExecuteInternal();
    void SelectIndex();

private:
    CallBack m_callback;
    std::shared_ptr<ServiceContext> c_serviceContext;
    Socket::RemoteDeleteQuery m_query;
    std::shared_ptr<DeleteExecutionContext> m_executionContext;
    std::vector<std::shared_ptr<VectorIndex>> m_selectedIndex;
};

} // namespace Service
} // namespace SPTAG

#endif // _SPTAG_SERVER_INSERTDELETEEXECUTORH