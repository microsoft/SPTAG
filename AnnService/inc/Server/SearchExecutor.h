#ifndef _SPACEV_SERVER_SEARCHEXECUTOR_H_
#define _SPACEV_SERVER_SEARCHEXECUTOR_H_

#include "ServiceContext.h"
#include "ServiceSettings.h"
#include "SearchExecutionContext.h"
#include "QueryParser.h"

#include <functional>
#include <memory>
#include <vector>

namespace SpaceV
{
namespace Service
{

class SearchExecutor
{
public:
    typedef std::function<void(std::shared_ptr<SearchExecutionContext>)> CallBack;

    SearchExecutor(std::string p_queryString,
                   std::shared_ptr<ServiceContext> p_serviceContext,
                   const CallBack& p_callback);

    ~SearchExecutor();

    void Execute();

private:
    void ExecuteInternal();

    void SelectIndex();

private:
    CallBack m_callback;

    const std::shared_ptr<ServiceContext> c_serviceContext;

    std::shared_ptr<SearchExecutionContext> m_executionContext;

    std::string m_queryString;

    std::vector<std::shared_ptr<VectorIndex>> m_selectedIndex;
};


} // namespace Server
} // namespace AnnService


#endif // _SPACEV_SERVER_SEARCHEXECUTOR_H_
