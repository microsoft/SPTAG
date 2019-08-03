// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_HELPER_BUFFERSTREAM_H_
#define _SPTAG_HELPER_BUFFERSTREAM_H_

#include <streambuf>
#include <ostream>
#include <memory>

namespace SPTAG
{
    namespace Helper
    {
        struct streambuf : public std::basic_streambuf<char> 
        {
            streambuf(char* buffer, size_t size)
            {
                setp(buffer, buffer + size);
            }
        };

        class obufferstream : public std::ostream
        {
        public:
            obufferstream(char* buffer, size_t size) : m_buf(new streambuf(buffer, size)), std::ostream(m_buf.get()) {}
        private:
            std::unique_ptr<streambuf> m_buf;
        };
    } // namespace Helper
} // namespace SPTAG

#endif // _SPTAG_HELPER_BUFFERSTREAM_H_

