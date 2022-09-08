// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_HELPER_DISKIO_H_
#define _SPTAG_HELPER_DISKIO_H_

#include <functional>
#include <fstream>
#include <string.h>
#include <memory>

namespace SPTAG
{
    namespace Helper
    {
        enum class DiskIOScenario
        {
            DIS_BulkRead = 0,
            DIS_UserRead,
            DIS_HighPriorityUserRead,
            DIS_BulkWrite,
            DIS_UserWrite,
            DIS_HighPriorityUserWrite,
            DIS_Count
        };

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

            AsyncReadRequest() : m_offset(0), m_readSize(0), m_buffer(nullptr), m_status(0), m_payload(nullptr), m_success(false), m_extension(nullptr) {}
        };

        class DiskIO
        {
        public:
            DiskIO(DiskIOScenario scenario = DiskIOScenario::DIS_UserRead) {}

            virtual ~DiskIO() {}

            virtual bool Initialize(const char* filePath, int openMode,
                // Max read/write buffer size.
                std::uint64_t maxIOSize = (1 << 20),
                std::uint32_t maxReadRetries = 2,
                std::uint32_t maxWriteRetries = 2,
                std::uint16_t threadPoolSize = 4) = 0;

            virtual std::uint64_t ReadBinary(std::uint64_t readSize, char* buffer, std::uint64_t offset = UINT64_MAX) = 0;

            virtual std::uint64_t WriteBinary(std::uint64_t writeSize, const char* buffer, std::uint64_t offset = UINT64_MAX) = 0;

            virtual std::uint64_t ReadString(std::uint64_t& readSize, std::unique_ptr<char[]>& buffer, char delim = '\n', std::uint64_t offset = UINT64_MAX) = 0;

            virtual std::uint64_t WriteString(const char* buffer, std::uint64_t offset = UINT64_MAX) = 0;

            virtual bool ReadFileAsync(AsyncReadRequest& readRequest) { return false; }
            
            virtual bool BatchReadFile(AsyncReadRequest* readRequests, std::uint32_t requestCount) { return false; }

            virtual bool BatchCleanRequests(SPTAG::Helper::AsyncReadRequest* readRequests, std::uint32_t requestCount) { return false; }

            virtual std::uint64_t TellP() = 0;

            virtual void ShutDown() = 0; 
        };

        class SimpleFileIO : public DiskIO
        {
        public:
            SimpleFileIO(DiskIOScenario scenario = DiskIOScenario::DIS_UserRead) {}

            virtual ~SimpleFileIO() { ShutDown(); }

            virtual bool Initialize(const char* filePath, int openMode,
                // Max read/write buffer size.
                std::uint64_t maxIOSize = (1 << 20),
                std::uint32_t maxReadRetries = 2,
                std::uint32_t maxWriteRetries = 2,
                std::uint16_t threadPoolSize = 4)
            {
                m_handle.reset(new std::fstream(filePath, (std::ios::openmode)openMode));
                return m_handle->is_open();
            }

            virtual std::uint64_t ReadBinary(std::uint64_t readSize, char* buffer, std::uint64_t offset = UINT64_MAX)
            {
                if (offset != UINT64_MAX) m_handle->seekg(offset, std::ios::beg);
                m_handle->read((char*)buffer, readSize);
                return m_handle->gcount();
            }

            virtual std::uint64_t WriteBinary(std::uint64_t writeSize, const char* buffer, std::uint64_t offset = UINT64_MAX)
            {
                if (offset != UINT64_MAX) m_handle->seekp(offset, std::ios::beg);
                m_handle->write((const char*)buffer, writeSize);
                if (m_handle->fail() || m_handle->bad()) return 0;
                return writeSize;
            }

            virtual std::uint64_t ReadString(std::uint64_t& readSize, std::unique_ptr<char[]>& buffer, char delim = '\n', std::uint64_t offset = UINT64_MAX)
            {
                if (offset != UINT64_MAX) m_handle->seekg(offset, std::ios::beg);
                std::uint64_t readCount = 0;
                for (int _Meta = m_handle->get();; _Meta = m_handle->get()) {
                    if (_Meta == '\r') _Meta = '\n';

                    if (readCount >= readSize) { // buffer full
                        readSize *= 2;
                        std::unique_ptr<char[]> newBuffer(new char[readSize]);
                        memcpy(newBuffer.get(), buffer.get(), readCount);
                        buffer.swap(newBuffer);
                    }

                    if (_Meta == EOF) { // eof
                        buffer[readCount] = '\0';
                        break;
                    }
                    else if (_Meta == delim) { // got a delimiter, discard it and quit
                        buffer[readCount++] = '\0';
                        if (delim == '\n' && m_handle->peek() == '\n') {
                            readCount++;
                            m_handle->ignore();
                        }
                        break;
                    }
                    else { // got a character, add it to string
                        buffer[readCount++] = std::char_traits<char>::to_char_type(_Meta);
                    }
                }
                return readCount;
            }

            virtual std::uint64_t WriteString(const char* buffer, std::uint64_t offset = UINT64_MAX)
            {
                return WriteBinary(strlen(buffer), (const char*)buffer, offset);
            }

            virtual std::uint64_t TellP()
            {
                return m_handle->tellp();
            }

            virtual void ShutDown()
            {
                if (m_handle != nullptr) m_handle->close();
            }

        private:
            std::unique_ptr<std::fstream> m_handle;
        };

        class SimpleBufferIO : public DiskIO
        {
        public:
            struct streambuf : public std::basic_streambuf<char>
            {
                streambuf() {}

                streambuf(char* buffer, size_t size)
                {
                    setg(buffer, buffer, buffer + size);
                }

                std::uint64_t tellp()
                {
                    if (pptr()) return pptr() - pbase();
                    return 0;
                }
            };

            SimpleBufferIO(DiskIOScenario scenario = DiskIOScenario::DIS_UserRead) {}

            virtual ~SimpleBufferIO()
            {
                ShutDown();
            }

            virtual bool Initialize(const char* filePath, int openMode,
                // Max read/write buffer size.
                std::uint64_t maxIOSize = (1 << 20),
                std::uint32_t maxReadRetries = 2,
                std::uint32_t maxWriteRetries = 2,
                std::uint16_t threadPoolSize = 4)
            {
                if (filePath != nullptr)
                    m_handle.reset(new streambuf((char*)filePath, maxIOSize));
                else
                    m_handle.reset(new streambuf());
                return true;
            }

            virtual std::uint64_t ReadBinary(std::uint64_t readSize, char* buffer, std::uint64_t offset = UINT64_MAX)
            {
                if (offset != UINT64_MAX) m_handle->pubseekpos(offset);
                return m_handle->sgetn((char*)buffer, readSize);
            }

            virtual std::uint64_t WriteBinary(std::uint64_t writeSize, const char* buffer, std::uint64_t offset = UINT64_MAX)
            {
                if (offset != UINT64_MAX) m_handle->pubseekpos(offset);
                if ((std::uint64_t)m_handle->sputn((const char*)buffer, writeSize) < writeSize) return 0;
                return writeSize;
            }

            virtual std::uint64_t ReadString(std::uint64_t& readSize, std::unique_ptr<char[]>& buffer, char delim = '\n', std::uint64_t offset = UINT64_MAX)
            {
                if (offset != UINT64_MAX) m_handle->pubseekpos(offset);
                std::uint64_t readCount = 0;
                for (int _Meta = m_handle->sgetc();; _Meta = m_handle->snextc()) {
                    if (_Meta == '\r') _Meta = '\n';

                    if (readCount >= readSize) { // buffer full
                        readSize *= 2;
                        std::unique_ptr<char[]> newBuffer(new char[readSize]);
                        memcpy(newBuffer.get(), buffer.get(), readCount);
                        buffer.swap(newBuffer);
                    }

                    if (_Meta == EOF) { // eof
                        buffer[readCount] = '\0';
                        break;
                    }
                    else if (_Meta == delim) { // got a delimiter, discard it and quit
                        buffer[readCount++] = '\0';
                        m_handle->sbumpc();
                        if (delim == '\n' && m_handle->sgetc() == '\n') {
                            readCount++;
                            m_handle->sbumpc();
                        }
                        break;
                    }
                    else { // got a character, add it to string
                        buffer[readCount++] = std::char_traits<char>::to_char_type(_Meta);
                    }
                }
                return readCount;
            }

            virtual std::uint64_t WriteString(const char* buffer, std::uint64_t offset = UINT64_MAX)
            {
                return WriteBinary(strlen(buffer), (const char*)buffer, offset);
            }

            virtual std::uint64_t TellP()
            { 
                return m_handle->tellp(); 
            }

            virtual void ShutDown() {}

        private:
            std::unique_ptr<streambuf> m_handle;
        };
    } // namespace Helper
} // namespace SPTAG

#endif // _SPTAG_HELPER_DISKIO_H_
