#ifndef _SPACEV_INDEXBUILDER_THREADPOOL_H_
#define _SPACEV_INDEXBUILDER_THREADPOOL_H_

#include <functional>
#include <cstdint>

namespace SpaceV
{
namespace IndexBuilder
{
namespace ThreadPool
{

void Init(std::uint32_t p_threadNum);

bool Queue(std::function<void()> p_workItem);

std::uint32_t CurrentThreadNum();

}
} // namespace IndexBuilder
} // namespace SpaceV

#endif // _SPACEV_INDEXBUILDER_THREADPOOL_H_
