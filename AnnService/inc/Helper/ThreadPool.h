// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_HELPER_THREADPOOL_H_
#define _SPTAG_HELPER_THREADPOOL_H_

#include <iostream>
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
            class Job
            {
            public:
                virtual ~Job() {}
                virtual void exec() = 0;
            };

            ThreadPool(): m_stopped(true) {}

            ~ThreadPool() 
            {
                m_stopped = true;
                m_cond.notify_all();
                for (auto && t : m_threads) t.join();
                m_threads.clear();

                /*
                while (!m_jobs.empty())
                {
                    Job* j = m_jobs.front();
                    m_jobs.pop();
                    
                    try
                    {
                        j->exec(); 
                    }
                    catch (std::exception& e) {
                        std::cout << "ThreadPool: exception in " << typeid(*j).name() << " " << e.what() << std::endl;
                    }
                    
                    delete j;
                }
                */
            }

            void init(int numberOfThreads = 1)
            {
                m_stopped = false;
                for (int i = 0; i < numberOfThreads; i++)
                {
                    m_threads.push_back(std::thread([this] {
                        Job *j;
                        while (get(j))
                        {
                            try 
                            {
                                j->exec();
                            }
                            catch (std::exception& e) {
                                std::cout << "ThreadPool: exception in " << typeid(*j).name() << " " << e.what() << std::endl;
                            }
                            
                            delete j;
                        }
                    }));
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
                while (m_jobs.empty() && !m_stopped) m_cond.wait(lock);
                if (!m_stopped) {
                    j = m_jobs.front();
                    m_jobs.pop();
                }
                return !m_stopped;
            }

            size_t jobsize()
            {
                std::lock_guard<std::mutex> lock(m_lock);
                return m_jobs.size();
            }

        protected:
            std::queue<Job*> m_jobs;
            bool m_stopped;
            std::mutex m_lock;
            std::condition_variable m_cond;
            std::vector<std::thread> m_threads;
        };
    }
}

#endif // _SPTAG_HELPER_THREADPOOL_H_