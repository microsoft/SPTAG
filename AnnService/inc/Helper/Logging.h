// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_HELPER_LOGGING_H_
#define _SPTAG_HELPER_LOGGING_H_

#include <stdarg.h>
#include <string.h>
#include <stdio.h>
#include <fstream>
#include <atomic>
#include <mutex>
#include <ctime>

#pragma warning(disable : 4996) // 'function': was declared deprecated
#pragma warning(disable : 4018) // '<' : signed/unsigned mismatch
#pragma warning(disable : 4242) // '=' : conversion from 'int' to 'short', possible loss of data
#pragma warning(disable : 4244) // '=' : conversion from 'int' to 'short', possible loss of data
#pragma warning(disable : 4267) // 'var' : conversion from 'size_t' to 'DWORD', possible loss of data
#pragma warning(disable : 4127) // conditional expression is constant

namespace SPTAG
{
    namespace Helper
    {
        enum class LogLevel
        {
            LL_Debug = 0,
            LL_Info,
            LL_Status,
            LL_Warning,
            LL_Error,
            LL_Assert,
            LL_Count,
            LL_Empty
        };

        class Logger 
        {
        public:
            virtual void Logging(const char* title, LogLevel level, const char* file, int line, const char* func, const char* format, ...) = 0;
        };

        class LoggerHolder
        {
#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 	202002L) || __cplusplus >= 	202002L)
        private:
            std::atomic<std::shared_ptr<Logger>> m_logger;
        public:
            LoggerHolder(std::shared_ptr<Logger> logger) : m_logger(logger) {}

            void SetLogger(std::shared_ptr<Logger> p_logger)
            {
                m_logger = p_logger;
            }

            std::shared_ptr<Logger> GetLogger()
            {
                return m_logger;
            }
#else
        private:
            std::shared_ptr<Logger> m_logger;
        public:
            LoggerHolder(std::shared_ptr<Logger> logger) : m_logger(logger) {}

            void SetLogger(std::shared_ptr<Logger> p_logger)
            {
                std::atomic_store(&m_logger, p_logger);
            }

            std::shared_ptr<Logger> GetLogger()
            {
                return std::atomic_load(&m_logger);
            }
#endif
        };


        class SimpleLogger : public Logger {
        public:
            SimpleLogger(LogLevel level) : m_level(level) {}

            virtual void Logging(const char* title, LogLevel level, const char* file, int line, const char* func, const char* format, ...)
            {
                if (level < m_level) return;

                std::time_t now = std::time(nullptr);
                std::tm* local = std::localtime(&now);
                if (level != LogLevel::LL_Empty) printf("[%d] %04d-%02d-%02d %02d:%02d:%02d ", (int)level, local->tm_year + 1900, local->tm_mon + 1, local->tm_mday, local->tm_hour, local->tm_min, local->tm_sec);

                va_list args;
                va_start(args, format);
                
                vprintf(format, args);
                fflush(stdout);

                va_end(args);
            }
        private:
            LogLevel m_level;
        };

        class FileLogger : public Logger {
        public:
            FileLogger(LogLevel level, const char* file) : m_level(level)
            {
                m_handle.reset(new std::fstream(file, std::ios::out));
            }

            ~FileLogger()
            {
                if (m_handle != nullptr) m_handle->close();
            }

            virtual void Logging(const char* title, LogLevel level, const char* file, int line, const char* func, const char* format, ...)
            {
                if (level < m_level || m_handle == nullptr || !m_handle->is_open()) return;

                va_list args;
                va_start(args, format);

                char buffer[1024];
                int ret = vsprintf(buffer, format, args);
                if (ret > 0)
                {
                    m_handle->write(buffer, strlen(buffer));
                }
                else
                {
                    std::string msg("Buffer size is not enough!\n");
                    m_handle->write(msg.c_str(), msg.size());
                }

                m_handle->flush();
                va_end(args);
            }
        private:
            LogLevel m_level;
            std::unique_ptr<std::fstream> m_handle;
        };
    } // namespace Helper
} // namespace SPTAG

#endif // _SPTAG_HELPER_LOGGING_H_
