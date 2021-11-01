// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "inc/Helper/DiskIO.h"
#include "inc/Core/Common.h"

#include <fcntl.h>
#include <memory>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include <thread>
#include <stdint.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <sys/syscall.h>
#include <linux/aio_abi.h>

namespace SPTAG {
    namespace SSDServing {
        namespace VectorSearch {

            template <typename T>
            class ConcurrentQueue
            {
            public:

                ConcurrentQueue() {}

                ~ConcurrentQueue() {}

                void Push(T* j)
                {
                    {
                        std::lock_guard<std::mutex> lock(m_lock);
                        m_queue.push(j);
                    }
                    m_cond.notify_one();
                }

                bool Pop(T*& j)
                {
                    std::unique_lock<std::mutex> lock(m_lock);
                    while (m_queue.empty()) m_cond.wait(lock);
                    j = m_queue.front();
                    m_queue.pop();
                    return true;
                }

            protected:
                std::queue<T*> m_queue;
                std::mutex m_lock;
                std::condition_variable m_cond;
            };

            struct DiskListRequest;
            typedef ConcurrentQueue<DiskListRequest> RequestQueue;

            class AsyncFileIO : public SPTAG::Helper::DiskPriorityIO
            {
            public:
                AsyncFileIO(SPTAG::Helper::DiskIOScenario scenario = SPTAG::Helper::DiskIOScenario::DIS_UserRead) {}

                virtual ~AsyncFileIO() { ShutDown(); }

                virtual bool Initialize(const char* filePath, int openMode,
                    std::uint64_t maxIOSize = (1 << 20),
                    std::uint32_t maxReadRetries = 2,
                    std::uint32_t maxWriteRetries = 2,
                    std::uint16_t threadPoolSize = 4)
                {
                    m_fileHandle = open(filePath, O_RDONLY | O_NOATIME);
                    if (m_fileHandle == -1) {
                        LOG(SPTAG::Helper::LogLevel::LL_Error, "Failed to create file handle: %s\n", filePath);
                        return false;
                    }
                    memset(&m_iocp, 0, sizeof(m_iocp));
                    auto ret = syscall(__NR_io_setup, 64, &m_iocp);
                    if (ret < 0) {
                        LOG(SPTAG::Helper::LogLevel::LL_Error, "Cannot setup aio: %s\n", strerror(errno));
                        return false;
                    }

                    int iocpThreads = threadPoolSize;
                    for (int i = 0; i < iocpThreads; ++i)
                    {
                        m_fileIocpThreads.emplace_back(std::thread(std::bind(&AsyncFileIO::ListionIOCP, this)));
                    }
                    return true;
                }

                virtual std::uint64_t ReadBinary(std::uint64_t readSize, char* buffer, std::uint64_t offset = UINT64_MAX)
                {
                    return pread(m_fileHandle, (void*)buffer, readSize, offset);
                }

                virtual std::uint64_t WriteBinary(std::uint64_t writeSize, const char* buffer, std::uint64_t offset = UINT64_MAX)
                {
                    return 0;
                }

                virtual std::uint64_t ReadString(std::uint64_t& readSize, std::unique_ptr<char[]>& buffer, char delim = '\n', std::uint64_t offset = UINT64_MAX)
                {
                    return 0;
                }

                virtual std::uint64_t WriteString(const char* buffer, std::uint64_t offset = UINT64_MAX)
                {
                    return 0;
                }

                virtual bool ReadFileAsync(SPTAG::Helper::AsyncReadRequest& readRequest)
                {
                    struct iocb myiocb = { 0 };
                    myiocb.aio_data = reinterpret_cast<uintptr_t>(&readRequest);
                    myiocb.aio_lio_opcode = IOCB_CMD_PREAD;
                    myiocb.aio_fildes = m_fileHandle;
                    myiocb.aio_buf = (std::uint64_t)(readRequest.m_buffer);
                    myiocb.aio_nbytes = readRequest.m_readSize;
                    myiocb.aio_offset = static_cast<std::int64_t>(readRequest.m_offset);

                    struct iocb* iocbs[1] = { &myiocb };
                    int res = syscall(__NR_io_submit, m_iocp, 1, iocbs);
                    if (res != 1)
                    {
                        //LOG(Helper::LogLevel::LL_Error, "ReadFileAsync Failed! res = %d\n", (int)res);
                        return false;
                    }
                    return true;
                }

                virtual std::uint64_t TellP() { return 0; }

                virtual void ShutDown()
                {
                    syscall(__NR_io_destroy, m_iocp);
                    close(m_fileHandle);
                    for (auto& th : m_fileIocpThreads)
                    {
                        if (th.joinable())
                        {
                            th.join();
                        }
                    }
                }

            private:
                void ListionIOCP() {
                    struct timespec timeout;
                    timeout.tv_sec = 1;
                    timeout.tv_nsec = 500000000;
                    struct io_event events[1];
                    while (true)
                    {
                        int numEvents = syscall(__NR_io_getevents, m_iocp, 1, 1, events, &timeout);
                        if (numEvents != 1)
                        {
                            break;
                        }
                        SPTAG::Helper::AsyncReadRequest* req = reinterpret_cast<SPTAG::Helper::AsyncReadRequest*>((events[0].data));
                        auto callback = &(req->m_callback);
                        if (nullptr != callback && (*callback))
                        {
                            (*callback)(true);
                        }
                    }
                }
            private:
                int m_fileHandle;

                aio_context_t m_iocp;

                std::vector<std::thread> m_fileIocpThreads;
            };
        }
    }
}