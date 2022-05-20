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
            Compressor()
            {
                compress_level = 1;
            }

            virtual ~Compressor(){}
            
            std::string Compress(const std::string &src)
            {
                size_t est_comp_size = ZSTD_compressBound(src.size());
                std::string buffer{};
                buffer.resize(est_comp_size); // TODO: reuse buffer
                size_t compressed_size = ZSTD_compress((void*)buffer.data(), est_comp_size,
                    src.data(), src.size(), 1); // TODO: change compress level
                buffer.resize(compressed_size);
                buffer.shrink_to_fit();

                return buffer;
            }

            std::string Decompress(const char * src, size_t srcSize)
            {
                size_t est_decomp_size = ZSTD_getDecompressedSize(src, srcSize);
                std::string dst{};
                dst.resize(est_decomp_size);
                size_t const decomp_size = ZSTD_decompress(
                    (void*)dst.data(), est_decomp_size, src, srcSize);
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
            //std::string buffer{};
            std::string dictionary{};
        };
    } // SPANN
} // SPTAG

#endif // _SPTAG_SPANN_COMPRESSOR_H_
