#ifndef _SPACEV_HELPER_CONCURRENT_H_
#define _SPACEV_HELPER_CONCURRENT_H_

#include <cstdint>
#include <cstddef> 
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <memory>

namespace SpaceV
{
namespace Helper
{
namespace Concurrent
{

class SpinLock
{
public:
    SpinLock();
    ~SpinLock();

    void Lock();
    void Unlock();

    SpinLock(const SpinLock&) = delete;
    SpinLock& operator = (const SpinLock&) = delete;

private:
    std::atomic_flag m_lock;
};


class WaitSignal
{
public:
    WaitSignal();

    WaitSignal(std::uint32_t p_unfinished);

    ~WaitSignal();

    void Reset(std::uint32_t p_unfinished);

    void Wait();

    void FinishOne();

private:
    std::atomic<std::uint32_t> m_unfinished;

    std::atomic_bool m_isWaiting;

    std::mutex m_mutex;

    std::condition_variable m_cv;
};


} // namespace Base64
} // namespace Helper
} // namespace SpaceV

#endif // _SPACEV_HELPER_CONCURRENT_H_
