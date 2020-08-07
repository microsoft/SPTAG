// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_HELPER_LOGGING_H_
#define _SPTAG_HELPER_LOGGING_H_

#include <stdarg.h>
#include <stdio.h>

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
    } // namespace Helper
} // namespace SPTAG

#endif // _SPTAG_HELPER_LOGGING_H_
