// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>

namespace SPTAG::SPANN
{
// Only managed tenant writes install a context. Descendant maintenance jobs inherit it.
class UpdateContext
{
  public:
    class Scope
    {
      public:
        explicit Scope(std::shared_ptr<UpdateContext> context) : m_previous(std::move(Current()))
        {
            Current() = std::move(context);
        }
        ~Scope()
        {
            Current() = std::move(m_previous);
        }

      private:
        std::shared_ptr<UpdateContext> m_previous;
    };

    static std::shared_ptr<UpdateContext> &Current()
    {
        static thread_local std::shared_ptr<UpdateContext> context;
        return context;
    }

    void Add()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        ++m_pending;
    }
    void Complete()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (--m_pending == 0)
            m_done.notify_all();
    }
    void Fail()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_failed = true;
    }
    bool Wait()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_done.wait(lock, [&] { return m_pending == 0; });
        return !m_failed;
    }

  private:
    std::mutex m_mutex;
    std::condition_variable m_done;
    size_t m_pending = 0;
    bool m_failed = false;
};
} // namespace SPTAG::SPANN
