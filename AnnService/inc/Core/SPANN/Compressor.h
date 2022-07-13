// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_COMPRESSOR_H_
#define _SPTAG_SPANN_COMPRESSOR_H_

#include <string>
#include "zstd.h"
#include "zdict.h"
#include "inc/Core/Common.h"

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
                    throw std::runtime_error("ZSTD_createCDict() failed!");
                }
            }

            void CreateDDict()
            {
                ddict = ZSTD_createDDict((void *)dictBuffer.data(), dictBuffer.size());
                if (ddict == NULL)
                {
                    LOG(Helper::LogLevel::LL_Error, "ZSTD_createDDict() failed! \n");
                    throw std::runtime_error("ZSTD_createDDict() failed!");
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
                    throw std::runtime_error("ZSTD_createCCtx() failed!");
                }
                size_t compressed_size = ZSTD_compress_usingCDict(cctx, (void *)comp_buffer.data(), est_compress_size, src.data(), src.size(), cdict);
                if (ZSTD_isError(compressed_size))
                {
                    LOG(Helper::LogLevel::LL_Error, "ZSTD compress error %s, \n", ZSTD_getErrorName(compressed_size));
                    throw std::runtime_error("ZSTD compress error");
                }
                ZSTD_freeCCtx(cctx);
                comp_buffer.resize(compressed_size);
                comp_buffer.shrink_to_fit();

                return comp_buffer;
            }

            std::size_t DecompressWithDict(const char* src, size_t srcSize, char* dst, size_t dstCapacity)
            {
                ZSTD_DCtx* const dctx = ZSTD_createDCtx();
                if (dctx == NULL)
                {
                    LOG(Helper::LogLevel::LL_Error, "ZSTD_createDCtx() failed! \n");
                    throw std::runtime_error("ZSTD_createDCtx() failed!");
                }
                std::size_t const decomp_size = ZSTD_decompress_usingDDict(dctx,
                    (void*)dst, dstCapacity, src, srcSize, ddict);
                if (ZSTD_isError(decomp_size))
                {
                    LOG(Helper::LogLevel::LL_Error, "ZSTD decompress error %s, \n", ZSTD_getErrorName(decomp_size));
                    throw std::runtime_error("ZSTD decompress failed.");
                }
                ZSTD_freeDCtx(dctx);
                return decomp_size;
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
                    throw std::runtime_error("ZSTD compress error");
                }
                buffer.resize(compressed_size);
                buffer.shrink_to_fit();

                return buffer;
            }

            std::size_t DecompressWithoutDict(const char *src, size_t srcSize, char* dst, size_t dstCapacity)
            {
                std::size_t const decomp_size = ZSTD_decompress(
                    (void *)dst, dstCapacity, src, srcSize);
                if (ZSTD_isError(decomp_size))
                {
                    LOG(Helper::LogLevel::LL_Error, "ZSTD decompress error %s, \n", ZSTD_getErrorName(decomp_size));
                    throw std::runtime_error("ZSTD decompress failed.");
                }

                return decomp_size;
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
                    throw std::runtime_error("ZDICT_trainFromBuffer() failed");
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

            std::size_t Decompress(const char *src, size_t srcSize, char* dst, size_t dstCapacity, const bool useDict)
            {
                return useDict ? DecompressWithDict(src, srcSize, dst, dstCapacity) : DecompressWithoutDict(src, srcSize, dst, dstCapacity);
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
