#ifndef _SPACEV_BKT_WORKSPACEPOOL_H_
#define _SPACEV_BKT_WORKSPACEPOOL_H_

#include "WorkSpace.h"

#include <list>
#include <mutex>
#include <memory>

namespace SpaceV
{
namespace BKT
{

class WorkSpacePool
{
public:
    WorkSpacePool(int p_maxCheck, int p_vectorCount);

    virtual ~WorkSpacePool();

    std::shared_ptr<WorkSpace> Rent();

    void Return(const std::shared_ptr<WorkSpace>& p_workSpace);

    void Init(int size);

private:
    std::list<std::shared_ptr<WorkSpace>> m_workSpacePool;

    std::mutex m_workSpacePoolMutex;

    int m_maxCheck;

    int m_vectorCount;
};

}
}

#endif // _SPACEV_BKT_WORKSPACEPOOL_H_
