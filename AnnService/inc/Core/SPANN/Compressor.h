// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_COMPRESSOR_H_
#define _SPTAG_SPANN_COMPRESSOR_H_

#include <string>
#include "zstd.h"
#include "../Common.h"

namespace SPTAG {
    namespace SPANN {
        class Compressor
        {
        public:
            Compressor(int level=0)
            {
                compress_level = level;
            }

            virtual ~Compressor(){}
            
            std::string Compress(const std::string &src)
            {
                size_t est_comp_size = ZSTD_compressBound(src.size());
                std::string buffer{};
                buffer.resize(est_comp_size);
                size_t compressed_size = ZSTD_compress((void*)buffer.data(), est_comp_size,
                    src.data(), src.size(), compress_level);
                if (ZSTD_isError(compressed_size))
                {
                    LOG(Helper::LogLevel::LL_Error, "ZSTD compress error %s, \n", ZSTD_getErrorName(compressed_size));
                    exit(1);
                }
                buffer.resize(compressed_size);
                buffer.shrink_to_fit();

                return buffer;
            }

            std::string Decompress(const char * src, size_t srcSize)
            {
                size_t est_decomp_size = ZSTD_getFrameContentSize(src, srcSize);
                if (est_decomp_size == ZSTD_CONTENTSIZE_ERROR) {
                    LOG(Helper::LogLevel::LL_Error, "not compressed by zstd!\n");
                    exit(1);
                }
                else if (est_decomp_size == ZSTD_CONTENTSIZE_UNKNOWN) {
                    LOG(Helper::LogLevel::LL_Error, "original size unknown!\n");
                    exit(1);
                }
                std::string dst{};
                est_decomp_size *= 10;
                dst.resize(est_decomp_size);
                size_t const decomp_size = ZSTD_decompress(
                    (void*)dst.data(), est_decomp_size, src, srcSize);
                if (ZSTD_isError(decomp_size))
                {
                    LOG(Helper::LogLevel::LL_Error, "ZSTD decompress error %s, \n", ZSTD_getErrorName(decomp_size));
                    exit(1);
                }
                dst.resize(decomp_size);
                dst.shrink_to_fit();

                return dst;
            }

            // return the compressed sie
            size_t GetCompressedSize(const std::string &src)
            {
                std::string dst = Compress(src);
                return dst.size();
            }

        private:
            int compress_level;
            std::string dictionary{};
        };
    } // SPANN
} // SPTAG

#endif // _SPTAG_SPANN_COMPRESSOR_H_
