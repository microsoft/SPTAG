#pragma once

#include <cstdint>
#include <functional>

namespace SPTAG {
    namespace SSDServing {
        namespace VectorSearch {
            struct DiskFileReadRequest
            {
                DiskFileReadRequest();

                uint64_t m_offset;

                uint64_t m_readSize;

                uint8_t* m_buffer;

                std::function<void()> m_callback;
            };


            class IDiskFileReader
            {
            public:
                IDiskFileReader();
                virtual ~IDiskFileReader();

                IDiskFileReader(const IDiskFileReader&) = delete;
                IDiskFileReader& operator=(const IDiskFileReader&) = delete;

                virtual bool IsOpened() const = 0;

                virtual bool ReadFileAsync(const DiskFileReadRequest& p_request) = 0;
            };

        }
    }
}