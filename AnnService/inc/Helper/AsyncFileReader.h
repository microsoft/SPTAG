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

#ifdef _MSC_VER
#include <tchar.h>
#include <Windows.h>
#else
#include <fcntl.h>
#include <sys/syscall.h>
#include <linux/aio_abi.h>
#ifdef URING
#include <liburing.h>
#endif
#ifdef NUMA
#include <numa.h>
#endif
#endif

#define ASYNC_READ 1
#define BATCH_READ 1

namespace SPTAG
{
    namespace Helper
    {
        struct AsyncReadRequest
        {
            std::uint64_t m_offset;
            std::uint64_t m_readSize;
            char* m_buffer;
            std::function<void(bool)> m_callback;
            int m_status;

            // Carry items like counter for callback to process.
            void* m_payload;
            bool m_success;

            // Carry exension metadata needed by some DiskIO implementations
            void* m_extension;
            void* m_ctrl;

#ifdef _MSC_VER
            DiskUtils::PrioritizedDiskFileReaderResource myres;
#else
            struct iocb myiocb;
#endif

            AsyncReadRequest() : m_offset(0), m_readSize(0), m_buffer(nullptr), m_status(0), m_payload(nullptr), m_success(false), m_extension(nullptr) {}
        };

        void SetThreadAffinity(int threadID, std::thread& thread, NumaStrategy socketStrategy = NumaStrategy::LOCAL, OrderStrategy idStrategy = OrderStrategy::ASC);
#ifdef _MSC_VER
        class HandleWrapper
        {
        public:
            HandleWrapper(HANDLE p_handle) : m_handle(p_handle) {}
            HandleWrapper(HandleWrapper&& p_right) noexcept: m_handle(std::move(p_right.m_handle)) {}
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

            bool try_pop(AsyncReadRequest*& j) {
                DWORD cBytes;
                ULONG_PTR key;
                OVERLAPPED* ol;
                BOOL ret = ::GetQueuedCompletionStatus(m_handle.GetHandle(),
                    &cBytes,
                    &key,
                    &ol,
                    0);
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

            virtual bool Available()
            {
                return !m_shutdown;
            }

            virtual bool NewFile(const char* filePath, uint64_t maxFileSize) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "AsyncFileIO: Create a file\n");
                m_fileHandle.Reset(::CreateFile(filePath,
                    GENERIC_WRITE,
                    0,
                    NULL,
                    CREATE_ALWAYS,
                    FILE_ATTRIBUTE_NORMAL,
                    NULL));
                if (!m_fileHandle.IsValid()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "AsyncFileIO::InitializeFileIo failed\n");
                    return false;
                }

                LARGE_INTEGER pos = { 0 };
                pos.QuadPart = maxFileSize;
                if (!SetFilePointerEx(m_fileHandle.GetHandle(), pos, NULL, FILE_BEGIN) || 
                    !SetEndOfFile(m_fileHandle.GetHandle())) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "AsyncFileIO::InitializeFileIo failed: fallocate failed\n");
                    m_fileHandle.Close();
                    return false;
                }

                m_fileHandle.Close();
                m_currSize = (uint64_t)filesize(filePath);
                if (m_currSize != maxFileSize) SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot fallocate enough space actural size (%llu) < max size (%llu)\n", m_currSize, maxFileSize);
                return true;
            }

            virtual bool ExpandFile(uint64_t expandSize) {
                LARGE_INTEGER new_pos = { 0 };
                new_pos.QuadPart = m_currSize + expandSize;
                if (!SetFilePointerEx(m_fileHandle.GetHandle(), new_pos, NULL, FILE_BEGIN) ||
                    !SetEndOfFile(m_fileHandle.GetHandle())) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                        "AsyncFileReader:[ExpandFile] fallocate failed at offset %lld for size %lld bytes\n",
                        static_cast<long long>(m_currSize),
                        static_cast<long long>(expandSize));
                    m_fileHandle.Close();
                    return false;
                }
                m_currSize += expandSize;
                return true;
            }

            virtual bool Initialize(const char* filePath, int openMode,
                std::uint64_t maxNumBlocks = (1 << 20),
                std::uint32_t maxReadRetries = 2,
                std::uint32_t maxWriteRetries = 2,
                std::uint16_t threadPoolSize = 4,
                std::uint64_t maxFileSize = (300ULL << 30))
            {
                if (!fileexists(filePath)) {
                    if (openMode == GENERIC_READ) {
                        SPTAGLIB_LOG(LogLevel::LL_Error, "Failed to open file handle: %s\n", filePath);
                        return false;
                    }
                    if (!NewFile(filePath, maxFileSize)) return false;
                }
                else {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "AsyncFileIO::Open a file\n");
                    m_currSize = filesize(filePath);
                    if (openMode == GENERIC_READ || m_currSize >= maxFileSize) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "AsyncFileIO::InitializeFileIo: file has been created with enough space.\n");
                    }
                    else {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "AsyncFileIO::Open failed! currSize(%llu) < maxFileSize(%llu)\n", m_currSize, maxFileSize);
                        if (!NewFile(filePath, maxFileSize)) return false;
                    }
                }

                m_fileHandle.Reset(::CreateFile(filePath,
                    GENERIC_READ | GENERIC_WRITE,
                    FILE_SHARE_READ,
                    NULL,
                    OPEN_EXISTING,
                    FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,
                    NULL));

                if (!m_fileHandle.IsValid()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "AsyncFileIO::Open failed for %s!\n", filePath);
                    return false;
                }

                m_diskSectorSize = static_cast<uint32_t>(GetSectorSize(filePath));
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "AsyncFileIO::InitializeFileIo: file %s opened, DiskSectorSize=%u threads=%d maxNumBlocks=%d\n", filePath, m_diskSectorSize, threadPoolSize, maxNumBlocks);

                m_shutdown = false;
                int iocpThreads = threadPoolSize;
                m_fileIocp.Reset(::CreateIoCompletionPort(m_fileHandle.GetHandle(), NULL, NULL, iocpThreads));
                for (int i = 0; i < iocpThreads; ++i)
                {
                    m_fileIocpThreads.emplace_back(std::thread(std::bind(&AsyncFileIO::ListionIOCP, this, i)));
                }
                return m_fileIocp.IsValid();
            }

            virtual std::uint64_t ReadBinary(std::uint64_t readSize, char* buffer, std::uint64_t offset = UINT64_MAX)
            {
                ResourceType resource;

                DiskUtils::CallbackOverLapped& col = resource.m_col;
                memset(&col, 0, sizeof(OVERLAPPED));
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
                    return 0;
                }

                DWORD cBytes = 0;
                ::GetOverlappedResult(m_fileHandle.GetHandle(), &col, &cBytes, TRUE);
                if (col.hEvent) CloseHandle(col.hEvent);
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
                DiskUtils::CallbackOverLapped& col = readRequest.myres.m_col;
                col.Offset = (readRequest.m_offset & 0xffffffff);
                col.OffsetHigh = (readRequest.m_offset >> 32);

                if (!::ReadFile(m_fileHandle.GetHandle(),
                    readRequest.m_buffer,
                    static_cast<DWORD>(ROUND_UP(readRequest.m_readSize, PageSize)),
                    nullptr,
                    &col) && GetLastError() != ERROR_IO_PENDING)
                {
                    return false;
                }
                return true;
            }

            virtual void Wait(AsyncReadRequest& readRequest)
            {
                // currently not used anywhere, effective only when ASYNC_READ is defined and BATCH_READ is not defined
                throw std::runtime_error("Not implemented");
            }

            virtual std::uint32_t BatchReadFile(AsyncReadRequest* readRequests, std::uint32_t requestCount, const std::chrono::microseconds& timeout, int batchSize = -1)
            {
                if (requestCount <= 0) return 0;

                auto t1 = std::chrono::high_resolution_clock::now();
                if (batchSize < 0 || batchSize > requestCount) batchSize = requestCount;

                RequestQueue* completeQueue = reinterpret_cast<RequestQueue*>(readRequests[0].m_extension);
                uint32_t batchTotalDone = 0, skip = 0;
                uint32_t currId = 0;
                AsyncReadRequest* req;
                while (currId < requestCount) {
                    int totalSubmitted = 0, totalDone = 0;
                    uint32_t batchStart = currId;
                    while (currId < requestCount && totalSubmitted < batchSize) {
                        AsyncReadRequest* readRequest = &(readRequests[currId++]);
                        if (readRequest->m_readSize == 0) {
                            skip++;
                            continue;
                        }
                        if (ReadFileAsync(*readRequest)) totalSubmitted++;

                    }
                    while (totalDone < totalSubmitted) {
                        if (!completeQueue->pop(req)) break;
                        totalDone++;
                        if (req->m_callback) req->m_callback(true);
                    }
                 
                    batchTotalDone += totalDone;
                    auto t2 = std::chrono::high_resolution_clock::now();
                    if (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1) > timeout) {
                        if (batchTotalDone < requestCount - skip) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "AsyncFileReader::ReadBlocks (batch[%u:%u]) : timeout, continue for next batch...\n", batchStart, currId);
                        }
                        //break;
                    }
                }
                //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "AsyncFileReader::ReadBlocks: finish\n");
                return batchTotalDone;
            }

            virtual std::uint32_t BatchWriteFile(AsyncReadRequest* readRequests, std::uint32_t requestCount, const std::chrono::microseconds& timeout, int batchSize = -1)
            {
                if (requestCount <= 0) return 0;

                auto t1 = std::chrono::high_resolution_clock::now();
                if (batchSize < 0 || batchSize > requestCount) batchSize = requestCount;

                RequestQueue* completeQueue = reinterpret_cast<RequestQueue*>(readRequests[0].m_extension);
                uint32_t batchTotalDone = 0;
                uint32_t currId = 0;
                AsyncReadRequest* req;
                while (currId < requestCount) {
                    int totalSubmitted = 0, totalDone = 0;
                    uint32_t batchStart = currId;
                    while (currId < requestCount && totalSubmitted < batchSize) {
                        AsyncReadRequest* readRequest = &(readRequests[currId++]);

                        DiskUtils::CallbackOverLapped& col = readRequest->myres.m_col;
                        col.Offset = (readRequest->m_offset & 0xffffffff);
                        col.OffsetHigh = (readRequest->m_offset >> 32);

                        if (!::WriteFile(m_fileHandle.GetHandle(),
                            readRequest->m_buffer,
                            static_cast<DWORD>(PageSize),
                            nullptr,
                            &col) && GetLastError() != ERROR_IO_PENDING)
                        {
                            continue;
                        }
                        totalSubmitted++;
                    }
                    while (totalDone < totalSubmitted) {
                        if (!completeQueue->pop(req)) break;
                        totalDone++;
                        if (req->m_callback) req->m_callback(true);
                    }

                    batchTotalDone += totalDone;
                    auto t2 = std::chrono::high_resolution_clock::now();
                    if (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1) > timeout) {
                        if (batchTotalDone < requestCount) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "AsyncFileReader::WriteBlocks (batch[%u:%u]) : timeout, continue for next batch...\n", batchStart, currId);
                        }
                        //break;
                    }
                }
                //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "AsyncFileReader::WriteBlocks: %d reqs finish\n", requestCount);
                return batchTotalDone;
            }

            virtual std::uint64_t TellP() { return 0; }

            virtual void ShutDown()
            {
                if (m_shutdown) return;

                m_shutdown = true;
                m_fileHandle.Close();
                m_fileIocp.Close();

                for (auto& th : m_fileIocpThreads)
                {
                    if (th.joinable())
                    {
                        th.join();
                    }
                }
                m_fileIocpThreads.clear();
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

                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed with: %s\n", (char*)lpMsgBuf);

                LocalFree(lpMsgBuf);
                ExitProcess(dw); 
                ShutDown();
            }

            void ListionIOCP(int i)
            {
                SetThreadAffinity(i, m_fileIocpThreads[i], NumaStrategy::SCATTER, OrderStrategy::DESC); // avoid IO threads overlap with search threads

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
                        //SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "AsyncFileReader::ListionIOCP thread failed!\n");
                        return;
                    }

                    col = (DiskUtils::CallbackOverLapped*)ol;
                    auto req = static_cast<Helper::AsyncReadRequest*>(col->m_data);

                    if (nullptr != req)
                    {
                        reinterpret_cast<RequestQueue*>(req->m_extension)->push(req);
                    }
                }
            }

        private:
            bool m_shutdown = true;

            HandleWrapper m_fileHandle;

            uint64_t m_currSize;
            
            HandleWrapper m_fileIocp;

            std::vector<std::thread> m_fileIocpThreads;

            uint32_t m_diskSectorSize = 512;
        };
#else
        extern struct timespec AIOTimeout;

        using RequestQueue = Helper::Concurrent::ConcurrentQueue<Helper::AsyncReadRequest*>;

        class AsyncFileIO : public DiskIO
        {
        public:
            AsyncFileIO(DiskIOScenario scenario = DiskIOScenario::DIS_UserRead) {}

            virtual ~AsyncFileIO() {
		        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "AsyncFileReader: Destroying fd=%d!\n", m_fileHandle);
                ShutDown(); 
	        }

            virtual bool Available()
            {
                return !m_shutdown;
            }

            virtual bool NewFile(const char* filePath, uint64_t maxFileSize) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "AsyncFileIO: Create a file\n");
                m_fileHandle = open(filePath, O_CREAT | O_WRONLY, 0666);
                if (m_fileHandle == -1) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "AsyncFileIO::InitializeFileIo failed\n");
                    return false;
                }

                if (fallocate(m_fileHandle, 0, 0, maxFileSize) == -1) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "AsyncFileIO::InitializeFileIo failed: fallocate failed\n");
                    m_fileHandle = -1;
                    return false;
                }
                close(m_fileHandle);
		        m_currSize = filesize(filePath);
                if (m_currSize != maxFileSize) SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot fallocate enough space actural size (%llu) < max size (%llu)\n", m_currSize, maxFileSize);
                return true;
            }

            virtual bool ExpandFile(uint64_t expandSize) {
                if (fallocate(m_fileHandle, 0, m_currSize, expandSize) != 0) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                        "AsyncFileReader:[ExpandFile] fallocate failed at offset %lld for size %lld bytes: %s\n",
                        static_cast<long long>(m_currSize),
                        static_cast<long long>(expandSize),
                        strerror(errno));
                    return false;
                }
                m_currSize += expandSize;
                return true;
            }

            virtual bool Initialize(const char* filePath, int openMode,
                std::uint64_t maxNumBlocks = (1 << 20),
                std::uint32_t maxReadRetries = 2,
                std::uint32_t maxWriteRetries = 2,
                std::uint16_t threadPoolSize = 4,
                std::uint64_t maxFileSize = (300ULL << 30))
            {
                if (!fileexists(filePath)) {
                    if (openMode == (O_RDONLY | O_DIRECT)) {
                        SPTAGLIB_LOG(LogLevel::LL_Error, "Failed to open file handle: %s\n", filePath);
                        return false;
                    }
                    if (!NewFile(filePath, maxFileSize)) return false;
                }
                else {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "AsyncFileIO::Open a file\n");
                    m_currSize = filesize(filePath);
                    if (openMode == (O_RDONLY | O_DIRECT) || m_currSize >= maxFileSize) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "AsyncFileIO::InitializeFileIo: file has been created with enough space.\n");
                    } else {
			            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "AsyncFileIO::Open failed! currSize(%llu) < maxFileSize(%llu)\n", m_currSize, maxFileSize);
                        if (!NewFile(filePath, maxFileSize)) return false;
                   }
                }
                m_fileHandle = open(filePath, O_RDWR | O_DIRECT);
                if (m_fileHandle == -1) {
                    auto err_str = strerror(errno);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "AsyncFileIO::Open failed: %s\n", err_str);
                    return false;
                }
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "AsyncFileIO::InitializeFileIo: file %s opened, fd=%d threads=%d maxNumBlocks=%d\n", filePath, m_fileHandle, threadPoolSize, maxNumBlocks);
                m_iocps.resize(threadPoolSize);
#ifdef URING
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "AsyncFileIO::InitializeFileIo: using io uring for read!\n");
                m_uring.resize(threadPoolSize);
#endif
                memset(m_iocps.data(), 0, sizeof(aio_context_t) * threadPoolSize);
                for (int i = 0; i < threadPoolSize; i++) {
                    auto ret = syscall(__NR_io_setup, (int)maxNumBlocks, &(m_iocps[i]));
                    if (ret < 0) {
                        SPTAGLIB_LOG(LogLevel::LL_Error, "Cannot setup aio: %s\n", strerror(errno));
                        return false;
                    }
#ifdef URING
                    ret = io_uring_queue_init((int)maxNumBlocks, &m_uring[i], 0);
                    if (ret < 0)
                    {
                        SPTAGLIB_LOG(LogLevel::LL_Error, "Cannot setup io_uring: %s\n", strerror(-ret));
                        return false;
                    }
#endif
                }
                m_shutdown = false;

#ifndef BATCH_READ
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

            virtual std::uint32_t BatchReadFile(AsyncReadRequest* readRequests, std::uint32_t requestCount, const std::chrono::microseconds& timeout, int batchSize = -1)
            {
                if (requestCount <= 0) return 0;

                auto t1 = std::chrono::high_resolution_clock::now();
                if (batchSize < 0 || batchSize > requestCount) batchSize = requestCount;
                std::vector<struct iocb*> iocbs(batchSize);
                std::vector<struct io_event> events(batchSize);
                if (readRequests[0].m_status < 0)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "The iocp number is not enough! Please increase iothread number.\n");
                    return 0;
                }
                int iocp = readRequests[0].m_status % m_iocps.size();
		        //struct timespec timeout_ts {0, 0};
		        //while (syscall(__NR_io_getevents, m_iocps[iocp], batchSize, batchSize, events.data(), &timeout_ts) > 0);
 
                uint32_t realCount = 0;
                for (int i = 0; i < requestCount; i++) {
                    AsyncReadRequest* readRequest = readRequests + i;
#ifndef URING
                    struct iocb* myiocb = &(readRequest->myiocb);
 		            memset(myiocb, 0, sizeof(struct iocb));
                    myiocb->aio_data = reinterpret_cast<uintptr_t>(&readRequest);
                    myiocb->aio_lio_opcode = IOCB_CMD_PREAD;
                    myiocb->aio_fildes = m_fileHandle;
                    myiocb->aio_buf = (std::uint64_t)(readRequest->m_buffer);
                    myiocb->aio_nbytes = PageSize;
                    myiocb->aio_offset = static_cast<std::int64_t>(readRequest->m_offset);
#endif
                    if (readRequest->m_readSize > 0) realCount++;
                }
                uint32_t batchTotalDone = 0;
                uint32_t reqidx = 0;
                for (int currSubIoStartId = 0; currSubIoStartId < realCount; currSubIoStartId += batchSize) {
                    int currSubIoEndId = (currSubIoStartId + batchSize) > realCount ? realCount : currSubIoStartId + batchSize;
                    int totalToSubmit = currSubIoEndId - currSubIoStartId;
                    int totalSubmitted = 0, totalDone = 0;
                    for (int i = 0; i < totalToSubmit; i++) {
                        while (reqidx < requestCount && readRequests[reqidx].m_readSize == 0) reqidx++;
			            if (reqidx >= requestCount) SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "AsyncFileReader::ReadBlocks: error reqidx(%d) >= requestCount(%d)\n", reqidx, requestCount);
			            else {                                                     
#ifdef URING
                            auto sqe = io_uring_get_sqe(&m_uring[iocp]);
                            if (!sqe)
                            {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "AsyncFileReader::ReadBlocks: io_uring_get_sqe failed\n");
                                return 0;
                            }
                            io_uring_prep_read(sqe, m_fileHandle, readRequests[reqidx].m_buffer, PageSize,
                                               readRequests[reqidx].m_offset);
                            io_uring_sqe_set_data(sqe, &(readRequests[reqidx]));
#else
                            iocbs[i] = &(readRequests[reqidx].myiocb);   
#endif
                            reqidx++;
			            }
                    }

                    //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "AsyncFileReader::ReadBlocks: iocp:%d totalToSubmit:%d\n", iocp, totalToSubmit);
#ifdef URING
                    if (io_uring_submit(&m_uring[iocp]) < 0)
                    {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "AsyncFileReader::ReadBlocks: io_uring_submit failed!\n");
                        return 0;
                    }
                    struct io_uring_cqe *cqe;
                    for (auto i = 0; i < totalToSubmit; i++)
                    {
                        int ret = io_uring_wait_cqe(&m_uring[iocp], &cqe);
                        if (ret < 0)
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "AsyncFileReader::ReadBlocks: io_uring_wait_cqe failed: %s\n", strerror(-ret));
                            break;
                        }
                        ret = cqe->res;
                        io_uring_cqe_seen(&m_uring[iocp], cqe);
                        if (ret < 0)
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "AsyncFileReader::ReadBlocks: io_uring_wait_cqe failed with res=%d: %s\n", ret, strerror(-ret));
                            continue;
                        }
                        // AsyncReadRequest* readRequest =
                        // reinterpret_cast<AsyncReadRequest*>(io_uring_cqe_get_data(cqe));
                        totalDone++;
                    }
#else
                    while (totalDone < totalToSubmit) {
                        // Submit all I/Os
                        if (totalSubmitted < totalToSubmit) {
                            int s = syscall(__NR_io_submit, m_iocps[iocp], totalToSubmit - totalSubmitted, iocbs.data() + totalSubmitted);
                            if (s > 0) {
                                totalSubmitted += s;
                            } else {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "AsyncFileReader::ReadBlocks: io_submit failed\n");
                                return batchTotalDone;
			                } 
                        }
                        int wait = totalSubmitted - totalDone;
                        auto d = syscall(__NR_io_getevents, m_iocps[iocp], wait, wait, events.data() + totalDone, &AIOTimeout);
			            if (d >= 0) {
                            totalDone += d;
                        }
                        else
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "AsyncFileReader::ReadBlocks: io_getevents failed\n");
                            return batchTotalDone;
                        }
                    }
#endif
                    batchTotalDone += totalDone;
                    auto t2 = std::chrono::high_resolution_clock::now();
                    if (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1) > timeout) {
                        if (batchTotalDone < requestCount) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "AsyncFileReader::ReadBlocks (batch[%d:%d]) : timeout, continue for next batch...\n", currSubIoStartId, currSubIoEndId);
                        }
                        //break;
                    }
                }
                //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "AsyncFileReader::ReadBlocks: finish\n");
                return batchTotalDone;
            }

            virtual std::uint32_t BatchWriteFile(AsyncReadRequest* readRequests, std::uint32_t requestCount, const std::chrono::microseconds& timeout, int batchSize = -1)
            {
                if (requestCount <= 0) return 0;

                auto t1 = std::chrono::high_resolution_clock::now();
                if (batchSize < 0 || batchSize > requestCount) batchSize = requestCount;
		
                std::vector<struct iocb*> iocbs(batchSize);
                std::vector<struct io_event> events(batchSize);
                if (readRequests[0].m_status < 0)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "The iocp number is not enough! Please increase iothread number.\n");
                    return 0;
                }
                int iocp = readRequests[0].m_status % m_iocps.size();
	         	//struct timespec timeout_ts {0, 0};
		        //while (syscall(__NR_io_getevents, m_iocps[iocp], batchSize, batchSize, events.data(), &timeout_ts) > 0);
		
 
                for (int i = 0; i < requestCount; i++) {
                    AsyncReadRequest* readRequest = readRequests + i;
                    struct iocb* myiocb = &(readRequest->myiocb);
		            memset(myiocb, 0, sizeof(struct iocb));
                    myiocb->aio_data = reinterpret_cast<uintptr_t>(&readRequest);
                    myiocb->aio_lio_opcode = IOCB_CMD_PWRITE;
                    myiocb->aio_fildes = m_fileHandle;
                    myiocb->aio_buf = (std::uint64_t)(readRequest->m_buffer);
                    //myiocb->aio_nbytes = readRequest->m_readSize;
                    myiocb->aio_nbytes = PageSize;
                    myiocb->aio_offset = static_cast<std::int64_t>(readRequest->m_offset);
                }

                uint32_t batchTotalDone = 0;
                for (int currSubIoStartId = 0; currSubIoStartId < requestCount; currSubIoStartId += batchSize) {
                    int currSubIoEndId = (currSubIoStartId + batchSize) > requestCount ? requestCount : currSubIoStartId + batchSize;
                    int totalToSubmit = currSubIoEndId - currSubIoStartId;
                    int totalSubmitted = 0, totalDone = 0;
                    for (int i = 0; i < totalToSubmit; i++) {
                        iocbs[i] = &(readRequests[currSubIoStartId + i].myiocb);
                    }

                    //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "AsyncFileReader::WriteBlocks: iocp:%d totalToSubmit:%d\n", iocp, totalToSubmit);
                    while (totalDone < totalToSubmit) {
                        // Submit all I/Os
                        if (totalSubmitted < totalToSubmit) {
                            int s = syscall(__NR_io_submit, m_iocps[iocp], totalToSubmit - totalSubmitted, iocbs.data() + totalSubmitted);
                            if (s > 0) {
                                totalSubmitted += s;
                            } else {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "AsyncFileReader::WriteBlocks: io_submit failed\n");
                                return batchTotalDone;
			                }
                        }
                        int wait = totalSubmitted - totalDone;
                        auto d = syscall(__NR_io_getevents, m_iocps[iocp], wait, wait, events.data() + totalDone, &AIOTimeout);
			            if (d >= 0) {
                            totalDone += d;
                        }
                        else
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "AsyncFileReader::WriteBlocks: io_getevents failed\n");
                            return batchTotalDone;
                        }
                    }
                    batchTotalDone += totalDone;
                    auto t2 = std::chrono::high_resolution_clock::now();
                    if (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1) > timeout) {
                        if (batchTotalDone < requestCount) {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "AsyncFileReader::WriteBlocks (batch[%d:%d]) : timeout, continue for next batch...\n", currSubIoStartId, currSubIoEndId);
                        }
                        //break;
                    }
                }
                //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "AsyncFileReader::WriteBlocks: %d reqs finish\n", requestCount);
                return batchTotalDone;
            }

            virtual bool ReadFileAsync(AsyncReadRequest& readRequest)
            {
                struct iocb *myiocb = &(readRequest.myiocb);
                myiocb->aio_data = reinterpret_cast<uintptr_t>(&readRequest);
                myiocb->aio_lio_opcode = IOCB_CMD_PREAD;
                myiocb->aio_fildes = m_fileHandle;
                myiocb->aio_buf = (std::uint64_t)(readRequest.m_buffer);
                myiocb->aio_nbytes = readRequest.m_readSize;
                myiocb->aio_offset = static_cast<std::int64_t>(readRequest.m_offset);

                int iocp = (readRequest.m_status & 0xffff) % m_iocps.size();
                struct iocb* iocbs[1] = { myiocb };
                int curTry = 0, maxTry = 10;
                while (curTry < maxTry && syscall(__NR_io_submit, m_iocps[iocp], 1, iocbs) < 1) {
                    usleep(AIOTimeout.tv_nsec / 1000);
                    curTry++;
                }
                if (curTry == maxTry) return false;
                return true;
            }

            virtual std::uint64_t TellP() { return 0; }

            virtual void ShutDown()
            {
                if (m_shutdown) return;

                m_shutdown = true;
                for (int i = 0; i < m_iocps.size(); i++) syscall(__NR_io_destroy, m_iocps[i]);
#ifdef URING
                for (int i = 0; i < m_uring.size(); i++) io_uring_queue_exit(&m_uring[i]);
#endif
                close(m_fileHandle);
		SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "AsyncFileReader: ShutDown!\n");
#ifndef BATCH_READ
                for (auto& th : m_fileIocpThreads)
                {
                    if (th.joinable())
                    {
                        th.join();
                    }
                }
#endif
            }

            aio_context_t& GetIOCP(int i) { return m_iocps[i % m_iocps.size()]; }

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

            std::vector<std::thread> m_fileIocpThreads;
#endif
            bool m_shutdown = true;

            int m_fileHandle;

            uint64_t m_currSize;

	        std::vector<aio_context_t> m_iocps;

#ifdef URING
            std::vector<struct io_uring> m_uring;
#endif
        };
#endif
        bool BatchReadFileAsync(std::vector<std::shared_ptr<Helper::DiskIO>>& handlers, AsyncReadRequest* readRequests, int num);
    }
}

#endif // _SPTAG_HELPER_ASYNCFILEREADER_H_
