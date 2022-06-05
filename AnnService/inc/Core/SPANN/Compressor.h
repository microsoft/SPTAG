// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_COMPRESSOR_H_
#define _SPTAG_SPANN_COMPRESSOR_H_

#include <string>
#include "zstd.h"
#include "zdict.h"
#include "../Common.h"

namespace SPTAG
{
    namespace SPANN
    {
        class Compressor
        {
        private:
            void CreateCDict()
            {
                cdict = ZSTD_createCDict((void *)dictBuffer.data(), dictBuffer.size(), compress_level);
                if (cdict == NULL)
                {
                    LOG(Helper::LogLevel::LL_Error, "ZSTD_createCDict() failed! \n");
                    exit(1);
                }
            }

            void CreateDDict()
            {
                ddict = ZSTD_createDDict((void *)dictBuffer.data(), dictBuffer.size());
                if (ddict == NULL)
                {
                    LOG(Helper::LogLevel::LL_Error, "ZSTD_createDDict() failed! \n");
                    exit(1);
                }
            }

            std::string CompressWithDict(const std::string &src)
            {
                size_t est_compress_size = ZSTD_compressBound(src.size());
                std::string comp_buffer{};
                comp_buffer.resize(est_compress_size);

                ZSTD_CCtx *const cctx = ZSTD_createCCtx();
                if (cctx == NULL)
                {
                    LOG(Helper::LogLevel::LL_Error, "ZSTD_createCCtx() failed! \n");
                    exit(1);
                }
                size_t compressed_size = ZSTD_compress_usingCDict(cctx, (void *)comp_buffer.data(), est_compress_size, src.data(), src.size(), cdict);
                if (ZSTD_isError(compressed_size))
                {
                    LOG(Helper::LogLevel::LL_Error, "ZSTD compress error %s, \n", ZSTD_getErrorName(compressed_size));
                    exit(1);
                }
                ZSTD_freeCCtx(cctx);
                comp_buffer.resize(compressed_size);
                comp_buffer.shrink_to_fit();

                return comp_buffer;
            }

            std::string DecompressWithDict(const char *src, size_t srcSize)
            {
                auto const est_decomp_size =
                    ZSTD_getFrameContentSize(src, srcSize);

                std::string decomp_buffer{};
                decomp_buffer.resize(est_decomp_size);

                ZSTD_DCtx *const dctx = ZSTD_createDCtx();
                if (dctx == NULL)
                {
                    LOG(Helper::LogLevel::LL_Error, "ZSTD_createDCtx() failed! \n");
                    exit(1);
                }
                size_t const decomp_size = ZSTD_decompress_usingDDict(dctx,
                                                                      (void *)decomp_buffer.data(), est_decomp_size, src, srcSize, ddict);
                if (ZSTD_isError(decomp_size))
                {
                    LOG(Helper::LogLevel::LL_Error, "ZSTD decompress error %s, \n", ZSTD_getErrorName(decomp_size));
                    exit(1);
                }

                ZSTD_freeDCtx(dctx);
                decomp_buffer.resize(decomp_size);
                decomp_buffer.shrink_to_fit();
                return decomp_buffer;
            }

            std::string CompressWithoutDict(const std::string &src)
            {
                size_t est_comp_size = ZSTD_compressBound(src.size());
                std::string buffer{};
                buffer.resize(est_comp_size);
                size_t compressed_size = ZSTD_compress((void *)buffer.data(), est_comp_size,
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

            std::string DecompressWithoutDict(const char *src, size_t srcSize)
            {
                size_t est_decomp_size = ZSTD_getFrameContentSize(src, srcSize);
                if (est_decomp_size == ZSTD_CONTENTSIZE_ERROR)
                {
                    LOG(Helper::LogLevel::LL_Error, "not compressed by zstd!\n");
                    exit(1);
                }
                else if (est_decomp_size == ZSTD_CONTENTSIZE_UNKNOWN)
                {
                    LOG(Helper::LogLevel::LL_Error, "original size unknown!\n");
                    exit(1);
                }
                std::string dst{};
                dst.resize(est_decomp_size);
                size_t const decomp_size = ZSTD_decompress(
                    (void *)dst.data(), est_decomp_size, src, srcSize);
                if (ZSTD_isError(decomp_size))
                {
                    LOG(Helper::LogLevel::LL_Error, "ZSTD decompress error %s, \n", ZSTD_getErrorName(decomp_size));
                    exit(1);
                }
                dst.resize(decomp_size);
                dst.shrink_to_fit();

                return dst;
            }

        public:
            Compressor(int level = 0, int bufferCapacity = 102400)
            {
                compress_level = level;
                dictBufferCapacity = bufferCapacity;
                cdict = nullptr;
                ddict = nullptr;
            }

            virtual ~Compressor() {}

            std::size_t TrainDict(const std::string &samplesBuffer, const size_t *samplesSizes, unsigned nbSamples)
            {
                dictBuffer.resize(dictBufferCapacity);
                size_t dictSize = ZDICT_trainFromBuffer((void *)dictBuffer.data(), dictBufferCapacity, (void *)samplesBuffer.data(), &samplesSizes[0], nbSamples);
                if (ZDICT_isError(dictSize))
                {
                    LOG(Helper::LogLevel::LL_Error, "ZDICT_trainFromBuffer() failed: %s \n", ZDICT_getErrorName(dictSize));
                    exit(1);
                }
                dictBuffer.resize(dictSize);
                dictBuffer.shrink_to_fit();

                CreateCDict();

                return dictSize;
            }

            std::string GetDictBuffer()
            {
                return dictBuffer;
            }

            void SetDictBuffer(const std::string &buffer)
            {
                dictBuffer = buffer;
                CreateDDict();
            }

            std::string Compress(const std::string &src, const bool useDict)
            {
                return useDict ? CompressWithDict(src) : CompressWithoutDict(src);
            }

            std::string Decompress(const char *src, size_t srcSize, const bool useDict)
            {
                return useDict ? DecompressWithDict(src, srcSize) : DecompressWithoutDict(src, srcSize);
            }

            // return the compressed sie
            size_t GetCompressedSize(const std::string &src, bool useDict)
            {
                std::string dst = Compress(src, useDict);
                return dst.size();
            }

        private:
            int compress_level;

            std::string dictBuffer;
            size_t dictBufferCapacity;
            ZSTD_CDict *cdict;
            ZSTD_DDict *ddict;
        };
    } // SPANN
} // SPTAG

#endif // _SPTAG_SPANN_COMPRESSOR_H_
