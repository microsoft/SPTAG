// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_HELPER_DISKIO_H_
#define _SPTAG_HELPER_DISKIO_H_

#include <functional>
#include <fstream>

namespace SPTAG
{
    namespace Helper
    {
        enum class DiskIOScenario
        {
            DIS_BulkRead = 0,
            DIS_UserRead,
            DIS_HighPriorityUserRead,
            DIS_Count
        };

        struct AsyncReadRequest
        {
            unsigned __int64 m_offset;
            unsigned __int32 m_readSize;
            __int8* m_buffer;
            std::function<void(bool)> m_callback;
            
            // Carry items like counter for callback to process.
            void* m_payload;
            bool m_success;

            AsyncReadRequest() : m_offset(0), m_readSize(0), m_buffer(nullptr), m_payload(nullptr), m_success(false) {}
        };

        class DiskPriorityIO
        {
        public:
            DiskPriorityIO(DiskIOScenario scenario = DiskIOScenario::DIS_UserRead) {}

            virtual ~DiskPriorityIO() {}

            virtual bool Initialize(const char* filePath, bool isBinary,
                // Max read/write buffer size.
                unsigned __int32 maxIOSize = (1 << 20),
                unsigned __int32 maxReadRetries = 2,
                unsigned __int32 maxWriteRetries = 2,
                unsigned __int16 threadPoolSize = 4) = 0;

            virtual unsigned __int32 ReadFile(unsigned __int64 offset, unsigned __int32 readSize, __int8* buffer) = 0;

            virtual unsigned __int32 WriteFile(unsigned __int64 offset, unsigned __int32 writeSize, __int8* buffer) = 0;

            virtual unsigned __int32 Read(unsigned __int32 readSize, __int8* buffer) = 0;

            virtual unsigned __int32 ReadLine(unsigned __int32 readSize, __int8* buffer, char delim = '\n') = 0;
            
            virtual unsigned __int32 Write(unsigned __int32 writeSize, __int8* buffer) = 0;

            virtual bool ReadFileAsync(AsyncReadRequest& readRequest) = 0;

            virtual void ShutDown() = 0;

        };

        class SimpleFileIO : public DiskPriorityIO
        {
        public:
            SimpleFileIO(DiskIOScenario scenario) {}

            virtual ~SimpleFileIO()
            {
                ShutDown();
            }

            virtual bool Initialize(const char* filePath, bool isBinary,
                // Max read/write buffer size.
                unsigned __int32 maxIOSize = (1 << 20),
                unsigned __int32 maxReadRetries = 2,
                unsigned __int32 maxWriteRetries = 2,
                unsigned __int16 threadPoolSize = 4)
            {
                m_handle.reset(new std::fstream(filePath, (isBinary ? std::ios::binary : 0) | std::ios::out | std::ios::in));
                return m_handle->is_open();
            }

            virtual unsigned __int32 ReadFile(unsigned __int64 offset, unsigned __int32 readSize, __int8* buffer)
            {
                m_handle->seekg(offset, std::ios::beg);
                m_handle->read((char*)buffer, readSize);
                if (m_handle->fail() || m_handle->bad() || m_handle->eof()) return 0;
                return m_handle->gcount();
            }

            virtual unsigned __int32 WriteFile(unsigned __int64 offset, unsigned __int32 writeSize, __int8* buffer)
            {
                m_handle->seekp(offset, std::ios::beg);
                m_handle->write((char*)buffer, writeSize);
                if (m_handle->fail() || m_handle->bad()) return 0;
                return writeSize;
            }

            virtual unsigned __int32 Read(unsigned __int32 readSize, __int8* buffer)
            {
                m_handle->read((char*)buffer, readSize);
                if (m_handle->fail() || m_handle->bad() || m_handle->eof()) return 0;
                return m_handle->gcount();
            }

            virtual unsigned __int32 ReadLine(unsigned __int32 readSize, __int8* buffer, char delim = '\n')
            {
                m_handle->getline((char*)buffer, readSize, delim);
                if (m_handle->fail() || m_handle->bad() || m_handle->eof()) return 0;
                return m_handle->gcount();
            }

            virtual unsigned __int32 Write(unsigned __int32 writeSize, __int8* buffer)
            {
                m_handle->write((char*)buffer, writeSize);
                if (m_handle->fail() || m_handle->bad()) return 0;
                return writeSize;
            }

            virtual bool ReadFileAsync(AsyncReadRequest& readRequest)
            {
                return false;
            }

            virtual void ShutDown()
            {
                m_handle->close();
            }

        private:
            std::unique_ptr<std::fstream> m_handle;
        };

    } // namespace Helper
} // namespace SPTAG

#endif // _SPTAG_HELPER_DISKIO_H_
