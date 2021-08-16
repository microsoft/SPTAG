// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_HELPER_THREADPOOL_H_
#define _SPTAG_HELPER_THREADPOOL_H_

#include <queue>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace SPTAG
{
    namespace Helper
    {
        class ThreadPool
        {
        public:
            class Abort : public IAbortOperation
            {
            private:
                bool m_stopped;

            public:
                Abort(bool p_status = true) { m_stopped = p_status; }
                ~Abort() {}
                virtual bool ShouldAbort() { return m_stopped; }
                void SetAbort(bool p_status) { m_stopped = p_status; }
            };

            class Job
            {
            public:
                virtual ~Job() {}
                virtual void exec(IAbortOperation* p_abort) = 0;
            };

            ThreadPool() {}

            ~ThreadPool() 
            {
                m_abort.SetAbort(true);
                m_cond.notify_all();
                for (auto&& t : m_threads) t.join();
                m_threads.clear();
            }

            void init(int numberOfThreads = 1)
            {
                m_abort.SetAbort(false);
                for (int i = 0; i < numberOfThreads; i++)
                {
                    m_threads.emplace_back([this] {
                        Job *j;
                        while (get(j))
                        {
                            try 
                            {
                                j->exec(&m_abort);
                            }
                            catch (std::exception& e) {
                                LOG(Helper::LogLevel::LL_Error, "ThreadPool: exception in %s %s\n", typeid(*j).name(), e.what());
                            }
                            
                            delete j;
                        }
                    });
                }
            }

            void add(Job* j)
            {
                {
                    std::lock_guard<std::mutex> lock(m_lock);
                    m_jobs.push(j);
                }
                m_cond.notify_one();
            }

            bool get(Job*& j)
            {
                std::unique_lock<std::mutex> lock(m_lock);
                while (m_jobs.empty() && !m_abort.ShouldAbort()) m_cond.wait(lock);
                if (!m_abort.ShouldAbort()) {
                    j = m_jobs.front();
                    m_jobs.pop();
                }
                return !m_abort.ShouldAbort();
            }

            size_t jobsize()
            {
                std::lock_guard<std::mutex> lock(m_lock);
                return m_jobs.size();
            }

        protected:
            std::queue<Job*> m_jobs;
            Abort m_abort;
            std::mutex m_lock;
            std::condition_variable m_cond;
            std::vector<std::thread> m_threads;
        };
    }
}

#endif // _SPTAG_HELPER_THREADPOOL_H_