// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_HELPER_LOGGING_H_
#define _SPTAG_HELPER_LOGGING_H_

#include <stdarg.h>
#include <string.h>
#include <stdio.h>
#include <fstream>

#pragma warning(disable:4996)

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


        class SimpleLogger : public Logger {
        public:
            SimpleLogger(LogLevel level) : m_level(level) {}

            virtual void Logging(const char* title, LogLevel level, const char* file, int line, const char* func, const char* format, ...)
            {
                if (level < m_level) return;

                if (level != LogLevel::LL_Empty) printf("[%d] ", (int)level);

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
