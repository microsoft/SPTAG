// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_HELPER_ASYNCFILEREADER_H_
#define _SPTAG_HELPER_ASYNCFILEREADER_H_

#include "inc/Helper/DiskIO.h"
#include "inc/Helper/ConcurrentSet.h"
#include "inc/Core/Common.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include <thread>
#include <stdint.h>

#define ASYNC_READ 1

#ifdef _MSC_VER
#include <tchar.h>
#include <Windows.h>
#define BATCH_READ 1
#else
#define BATCH_READ 1
#include <fcntl.h>
#include <sys/syscall.h>
#include <linux/aio_abi.h>
#endif

namespace SPTAG
{
    namespace Helper
    {

#ifdef _MSC_VER
        namespace DiskUtils
        {

            struct PrioritizedDiskFileReaderResource;

            struct CallbackOverLapped : public OVERLAPPED
            {
                PrioritizedDiskFileReaderResource* const c_registeredResource;

                void* m_data;

                CallbackOverLapped(PrioritizedDiskFileReaderResource* p_registeredResource)
                    : c_registeredResource(p_registeredResource),
                    m_data(nullptr)
                {
                }
            };


            struct PrioritizedDiskFileReaderResource
            {
                CallbackOverLapped m_col;

                PrioritizedDiskFileReaderResource()
                    : m_col(this)
                {
                }
            };
        }

        class HandleWrapper
        {
        public:
            HandleWrapper(HANDLE p_handle) : m_handle(p_handle) {}
            HandleWrapper(HandleWrapper&& p_right) : m_handle(std::move(p_right.m_handle)) {}
            HandleWrapper() : m_handle(INVALID_HANDLE_VALUE) {}
            ~HandleWrapper() {}

            void Reset(HANDLE p_handle) { m_handle.reset(p_handle); }
            HANDLE GetHandle() { return m_handle.get(); }
            bool IsValid() const { return m_handle.get() != INVALID_HANDLE_VALUE; };
            void Close() { m_handle.reset(INVALID_HANDLE_VALUE); }

            HandleWrapper(const HandleWrapper&) = delete;
            HandleWrapper& operator=(const HandleWrapper&) = delete;

            struct HandleDeleter
            {
                void operator()(HANDLE p_handle) const
                {
                    if (p_handle != INVALID_HANDLE_VALUE)
                    {
                        ::CloseHandle(p_handle);
                    }

                    p_handle = INVALID_HANDLE_VALUE;
                }
            };

        private:
            typedef std::unique_ptr<typename std::remove_pointer<HANDLE>::type, HandleDeleter> UniqueHandle;
            UniqueHandle m_handle;
        };

        class RequestQueue
        {
        public:
            RequestQueue() {
                m_handle.Reset(::CreateIoCompletionPort(INVALID_HANDLE_VALUE, NULL, NULL, 0));
            }

            void reset(int capacity) {}

            void push(AsyncReadRequest* j) {
                ::PostQueuedCompletionStatus(m_handle.GetHandle(),
                    0,
                    NULL,
                    reinterpret_cast<LPOVERLAPPED>(j));
            }

            bool pop(AsyncReadRequest*& j) {
                DWORD cBytes;
                ULONG_PTR key;
                OVERLAPPED* ol;
                BOOL ret = ::GetQueuedCompletionStatus(m_handle.GetHandle(),
                    &cBytes,
                    &key,
                    &ol,
                    INFINITE);
                if (FALSE == ret || nullptr == ol) return false;
                j = reinterpret_cast<AsyncReadRequest*>(ol);
                return true;
            }

        private:
            HandleWrapper m_handle;
        };

        class AsyncFileIO : public DiskIO
        {
        public:
            AsyncFileIO(DiskIOScenario scenario = DiskIOScenario::DIS_UserRead) {}

            virtual ~AsyncFileIO() { ShutDown(); }

            virtual bool Initialize(const char* filePath, int openMode,
                std::uint64_t maxIOSize = (1 << 20),
                std::uint32_t maxReadRetries = 2,
                std::uint32_t maxWriteRetries = 2,
                std::uint16_t threadPoolSize = 4)
            {
                m_fileHandle.Reset(::CreateFile(filePath,
                    GENERIC_READ,
                    FILE_SHARE_READ,
                    NULL,
                    OPEN_EXISTING,
                    FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,
                    NULL));

                if (!m_fileHandle.IsValid()) return false;

                m_diskSectorSize = static_cast<uint32_t>(GetSectorSize(filePath));
                LOG(LogLevel::LL_Info, "Success open file handle: %s DiskSectorSize: %u\n", filePath, m_diskSectorSize);

                PreAllocQueryContext();

#ifndef BATCH_READ
                int iocpThreads = threadPoolSize;
                m_fileIocp.Reset(::CreateIoCompletionPort(m_fileHandle.GetHandle(), NULL, NULL, iocpThreads));
                for (int i = 0; i < iocpThreads; ++i)
                {
                    m_fileIocpThreads.emplace_back(std::thread(std::bind(&AsyncFileIO::ListionIOCP, this)));
                }
                return m_fileIocp.IsValid();
#else
                return true;
#endif
            }

            virtual std::uint64_t ReadBinary(std::uint64_t readSize, char* buffer, std::uint64_t offset = UINT64_MAX)
            {
                ResourceType* resource = GetResource();

                DiskUtils::CallbackOverLapped& col = resource->m_col;
                memset(&col, 0, sizeof(col));
                col.Offset = (offset & 0xffffffff);
                col.OffsetHigh = (offset >> 32);
                col.m_data = nullptr;
                col.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
                if (!::ReadFile(m_fileHandle.GetHandle(),
                    buffer,
                    static_cast<DWORD>(readSize),
                    nullptr,
                    &col) && GetLastError() != ERROR_IO_PENDING) {
                    if (col.hEvent) CloseHandle(col.hEvent);
                    ReturnResource(resource);
                    return 0;
                }

                DWORD cBytes = 0;
                ::GetOverlappedResult(m_fileHandle.GetHandle(), &col, &cBytes, TRUE);
                if (col.hEvent) CloseHandle(col.hEvent);
                ReturnResource(resource);
                return cBytes;
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

            virtual bool ReadFileAsync(AsyncReadRequest& readRequest)
            {
                ResourceType* resource = GetResource();

                DiskUtils::CallbackOverLapped& col = resource->m_col;
                memset(&col, 0, sizeof(col));
                col.Offset = (readRequest.m_offset & 0xffffffff);
                col.OffsetHigh = (readRequest.m_offset >> 32);
                col.m_data = (void*)&readRequest;

                if (!::ReadFile(m_fileHandle.GetHandle(),
                    readRequest.m_buffer,
                    static_cast<DWORD>(readRequest.m_readSize),
                    nullptr,
                    &col) && GetLastError() != ERROR_IO_PENDING)
                {
                    ReturnResource(resource);
                    return false;
                }
                return true;
            }

            virtual bool BatchReadFile(AsyncReadRequest* readRequests, std::uint32_t requestCount)
            {
                std::vector<ResourceType*> waitResources;
                for (std::uint32_t i = 0; i < requestCount; i++) {
                    AsyncReadRequest* readRequest = &(readRequests[i]);

                    ResourceType* resource = GetResource();
                    DiskUtils::CallbackOverLapped& col = resource->m_col;
                    memset(&col, 0, sizeof(col));
                    col.Offset = (readRequest->m_offset & 0xffffffff);
                    col.OffsetHigh = (readRequest->m_offset >> 32);
                    col.m_data = (void*)readRequest;

                    col.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
                    if (!::ReadFile(m_fileHandle.GetHandle(),
                        readRequest->m_buffer,
                        static_cast<DWORD>(readRequest->m_readSize),
                        nullptr,
                        &col) && GetLastError() != ERROR_IO_PENDING)
                    {
                        if (col.hEvent) CloseHandle(col.hEvent);
                        ReturnResource(resource);
                        LOG(Helper::LogLevel::LL_Error, "Cannot read request %d error:%u\n", i, GetLastError());
                    }
                    else {
                        waitResources.push_back(resource);
                    }
                }

                for (int i = 0; i < waitResources.size(); i++) {
                    DWORD readbytes = 0;
                    DiskUtils::CallbackOverLapped* col = &(waitResources[i]->m_col);
                    AsyncReadRequest* readRequest = (AsyncReadRequest*)(col->m_data);

                    if (!::GetOverlappedResult(m_fileHandle.GetHandle(), col, &readbytes, TRUE) || readbytes != readRequest->m_readSize) {
                        LOG(Helper::LogLevel::LL_Error, "Cannot wait request %d readbytes:%llu expect:%llu error:%u\n", i, readbytes, readRequest->m_readSize, GetLastError());
                    }
                    else {
                        readRequest->m_success = true;
                    }
                    if (col->hEvent) CloseHandle(col->hEvent);
                    ReturnResource(waitResources[i]);
                }
                return true;
            }

            virtual std::uint64_t TellP() { return 0; }

            virtual void ShutDown()
            {
                m_fileHandle.Close();
                m_fileIocp.Close();

                for (auto& th : m_fileIocpThreads)
                {
                    if (th.joinable())
                    {
                        th.join();
                    }
                }

                ResourceType* res = nullptr;
                while (m_resources.try_pop(res))
                {
                    if (res != nullptr)
                    {
                        delete res;
                    }
                }
            }

        private:
            typedef DiskUtils::PrioritizedDiskFileReaderResource ResourceType;

            static uint64_t GetSectorSize(const char* p_filePath)
            {
                DWORD dwSectorSize = 0;
                LPTSTR lpFilePart;
                // Get file sector size
                DWORD dwSize = GetFullPathName(p_filePath, 0, NULL, &lpFilePart);
                if (dwSize > 0)
                {
                    LPTSTR lpBuffer = new TCHAR[dwSize];
                    if (lpBuffer)
                    {
                        DWORD dwResult = GetFullPathName(p_filePath, dwSize, lpBuffer, &lpFilePart);
                        if (dwResult > 0 && dwResult <= dwSize)
                        {
                            bool nameValid = false;
                            if (lpBuffer[0] == _T('\\'))
                            {
                                if (lpBuffer[1] == _T('\\'))
                                {
                                    DWORD i;
                                    if (dwSize > 2)
                                    {
                                        for (i = 2; lpBuffer[i] != 0 && lpBuffer[i] != _T('\\'); i++);
                                        if (lpBuffer[i] == _T('\\'))
                                        {
                                            for (i++; lpBuffer[i] != 0 && lpBuffer[i] != _T('\\'); i++);
                                            if (lpBuffer[i] == _T('\\'))
                                            {
                                                lpBuffer[i + 1] = 0;
                                                nameValid = true;
                                            }
                                        }
                                    }
                                }
                            }
                            else
                            {
                                if (((lpBuffer[0] >= _T('a') && lpBuffer[0] <= _T('z')) || (lpBuffer[0] >= _T('A') && lpBuffer[0] <= _T('Z'))) &&
                                    lpBuffer[1] == _T(':'))
                                {
                                    if (lpBuffer[2] != 0)
                                    {
                                        lpBuffer[2] = _T('\\'); lpBuffer[3] = 0;
                                        nameValid = true;
                                    }
                                }
                            }
                            if (nameValid)
                            {
                                DWORD dwSPC, dwNOFC, dwTNOC;
                                GetDiskFreeSpace(lpBuffer, &dwSPC, &dwSectorSize, &dwNOFC, &dwTNOC);
                            }
                        }
                        delete[] lpBuffer;
                    }
                }

                return dwSectorSize;
            }

            void ErrorExit()
            {
                LPVOID lpMsgBuf;
                DWORD dw = GetLastError();

                FormatMessage(
                    FORMAT_MESSAGE_ALLOCATE_BUFFER |
                    FORMAT_MESSAGE_FROM_SYSTEM |
                    FORMAT_MESSAGE_IGNORE_INSERTS,
                    NULL,
                    dw,
                    0,
                    (LPTSTR)&lpMsgBuf,
                    0, NULL);

                // Display the error message and exit the process

                LOG(Helper::LogLevel::LL_Error, "Failed with: %s\n", (char*)lpMsgBuf);

                LocalFree(lpMsgBuf);
                ExitProcess(dw);
            }

            void ListionIOCP()
            {
                DWORD cBytes;
                ULONG_PTR key;
                OVERLAPPED* ol;
                DiskUtils::CallbackOverLapped* col;

                while (true)
                {
                    BOOL ret = ::GetQueuedCompletionStatus(this->m_fileIocp.GetHandle(),
                        &cBytes,
                        &key,
                        &ol,
                        INFINITE);
                    if (FALSE == ret || nullptr == ol)
                    {
                        //We exit the thread
                        return;
                    }

                    col = (DiskUtils::CallbackOverLapped*)ol;
                    auto req = static_cast<Helper::AsyncReadRequest*>(col->m_data);
                    ReturnResource(col->c_registeredResource);

                    if (nullptr != req)
                    {
                        req->m_callback(true);
                    }
                }
            }

            ResourceType* GetResource()
            {
                ResourceType* ret = nullptr;
                if (m_resources.try_pop(ret))
                {
                }
                else
                {
                    ret = new ResourceType();
                }

                return ret;
            }

            void ReturnResource(ResourceType* p_res)
            {
                if (p_res != nullptr)
                {
                    m_resources.push(p_res);
                }
            }

            void PreAllocQueryContext()
            {
                const size_t num = 64 * 64;
                typedef ResourceType* ResourcePtr;

                ResourcePtr* contextArray = new ResourcePtr[num];

                for (int i = 0; i < num; ++i)
                {
                    contextArray[i] = GetResource();
                }

                for (int i = 0; i < num; ++i)
                {
                    ReturnResource(contextArray[i]);
                    contextArray[i] = nullptr;
                }

                delete[] contextArray;

            }

        private:
            HandleWrapper m_fileHandle;

            HandleWrapper m_fileIocp;

            std::vector<std::thread> m_fileIocpThreads;

            uint32_t m_diskSectorSize;

            Helper::Concurrent::ConcurrentQueue<ResourceType*> m_resources;
        };
#else
        extern struct timespec AIOTimeout;

        class RequestQueue
        {
        public:

            RequestQueue() :m_front(0), m_end(0), m_capacity(0) {}

            ~RequestQueue() {}

            void reset(int capacity) {
                if (capacity > m_capacity) {
                    m_capacity = capacity + 1;
                    m_queue.reset(new AsyncReadRequest * [m_capacity]);
                }
            }

            void push(AsyncReadRequest* j)
            {
                m_queue[m_end++] = j;
                if (m_end == m_capacity) m_end = 0;
            }

            bool pop(AsyncReadRequest*& j)
            {
                while (m_front == m_end) usleep(AIOTimeout.tv_nsec / 1000);
                j = m_queue[m_front++];
                if (m_front == m_capacity) m_front = 0;
                return true;
            }

        protected:
            int m_front, m_end, m_capacity;
            std::unique_ptr<AsyncReadRequest* []> m_queue;
        };

        class AsyncFileIO : public DiskIO
        {
        public:
            AsyncFileIO(DiskIOScenario scenario = DiskIOScenario::DIS_UserRead) {}

            virtual ~AsyncFileIO() { ShutDown(); }

            virtual bool Initialize(const char* filePath, int openMode,
                std::uint64_t maxIOSize = (1 << 20),
                std::uint32_t maxReadRetries = 2,
                std::uint32_t maxWriteRetries = 2,
                std::uint16_t threadPoolSize = 4)
            {
                m_fileHandle = open(filePath, O_RDONLY | O_DIRECT);
                if (m_fileHandle <= 0) {
                    LOG(LogLevel::LL_Error, "Failed to create file handle: %s\n", filePath);
                    return false;
                }

                m_iocps.resize(threadPoolSize);
                memset(m_iocps.data(), 0, sizeof(aio_context_t) * threadPoolSize);
                for (int i = 0; i < threadPoolSize; i++) {
                    auto ret = syscall(__NR_io_setup, (int)maxIOSize, &(m_iocps[i]));
                    if (ret < 0) {
                        LOG(LogLevel::LL_Error, "Cannot setup aio: %s\n", strerror(errno));
                        return false;
                    }
                }

#ifndef BATCH_READ
                m_shutdown = false;
                for (int i = 0; i < threadPoolSize; ++i)
                {
                    m_fileIocpThreads.emplace_back(std::thread(std::bind(&AsyncFileIO::ListionIOCP, this, i)));
                }
#endif
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

            virtual bool ReadFileAsync(AsyncReadRequest& readRequest)
            {
                struct iocb myiocb = { 0 };
                myiocb.aio_data = reinterpret_cast<uintptr_t>(&readRequest);
                myiocb.aio_lio_opcode = IOCB_CMD_PREAD;
                myiocb.aio_fildes = m_fileHandle;
                myiocb.aio_buf = (std::uint64_t)(readRequest.m_buffer);
                myiocb.aio_nbytes = readRequest.m_readSize;
                myiocb.aio_offset = static_cast<std::int64_t>(readRequest.m_offset);

                struct iocb* iocbs[1] = { &myiocb };
                int curTry = 0, maxTry = 10;
                while (curTry < maxTry && syscall(__NR_io_submit, m_iocps[(readRequest.m_status & 0xffff) % m_iocps.size()], 1, iocbs) < 1) {
                    usleep(AIOTimeout.tv_nsec / 1000);
                    curTry++;
                }
                if (curTry == maxTry) return false;
                return true;
            }

            virtual std::uint64_t TellP() { return 0; }

            virtual void ShutDown()
            {
                for (int i = 0; i < m_iocps.size(); i++) syscall(__NR_io_destroy, m_iocps[i]);
                close(m_fileHandle);
#ifndef BATCH_READ
                m_shutdown = true;
                for (auto& th : m_fileIocpThreads)
                {
                    if (th.joinable())
                    {
                        th.join();
                    }
                }
#endif
            }

            aio_context_t& GetIOCP(int i) { return m_iocps[i]; }

            int GetFileHandler() { return m_fileHandle; }

        private:
#ifndef BATCH_READ
            void ListionIOCP(int i) {
                int b = 10;
                std::vector<struct io_event> events(b);
                while (!m_shutdown)
                {
                    int numEvents = syscall(__NR_io_getevents, m_iocps[i], b, b, events.data(), &AIOTimeout);

                    for (int r = 0; r < numEvents; r++) {
                        AsyncReadRequest* req = reinterpret_cast<AsyncReadRequest*>((events[r].data));
                        if (nullptr != req)
                        {
                            req->m_callback(true);
                        }
                    }
                }
            }

            bool m_shutdown;

            std::vector<std::thread> m_fileIocpThreads;
#endif
            int m_fileHandle;

            std::vector<aio_context_t> m_iocps;
        };
#endif
        void BatchReadFileAsync(std::vector<std::shared_ptr<Helper::DiskIO>>& handlers, AsyncReadRequest* readRequests, int num);
    }
}

#endif // _SPTAG_HELPER_ASYNCFILEREADER_H_
