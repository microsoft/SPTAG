// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_DATASET_H_
#define _SPTAG_COMMON_DATASET_H_

#include <fstream>

#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif // defined(__GNUC__)

#define ALIGN 32

#define aligned_malloc(a, b) _mm_malloc(a, b)
#define aligned_free(a) _mm_free(a)

#pragma warning(disable:4996)  // 'fopen': This function or variable may be unsafe. Consider using fopen_s instead. To disable deprecation, use _CRT_SECURE_NO_WARNINGS. See online help for details.

namespace SPTAG
{
    namespace COMMON
    {
        // structure to save Data and Graph
        template <typename T>
        class Dataset
        {
        private:
            SizeType rows = 0;
            SizeType cols = 1;
            bool ownData = false;
            T* data = nullptr;
            SizeType incRows = 0;
            std::vector<T*> incBlocks;
            static const SizeType rowsInBlock = 1024 * 1024;
        public:
            Dataset() {}
            Dataset(SizeType rows_, SizeType cols_, T* data_ = nullptr, bool transferOnwership_ = true)
            {
                Initialize(rows_, cols_, data_, transferOnwership_);
            }
            ~Dataset()
            {
                if (ownData) aligned_free(data);
                for (T* ptr : incBlocks) aligned_free(ptr);
                incBlocks.clear();
            }
            void Initialize(SizeType rows_, SizeType cols_, T* data_ = nullptr, bool transferOnwership_ = true)
            {
                rows = rows_;
                cols = cols_;
                data = data_;
                if (data_ == nullptr || !transferOnwership_)
                {
                    ownData = true;
                    data = (T*)aligned_malloc(((size_t)rows) * cols * sizeof(T), ALIGN);
                    if (data_ != nullptr) memcpy(data, data_, ((size_t)rows) * cols * sizeof(T));
                    else std::memset(data, -1, ((size_t)rows) * cols * sizeof(T));
                }
                incBlocks.reserve((std::numeric_limits<SizeType>::max)() / rowsInBlock + 1);
            }
            void SetR(SizeType R_) 
            {
                if (R_ >= rows)
                    incRows = R_ - rows;
                else 
                {
                    rows = R_;
                    incRows = 0;
                }
            }
            inline SizeType R() const { return rows + incRows; }
            inline SizeType C() const { return cols; }
            
            inline const T* At(SizeType index) const
            {
                if (index >= rows) {
                    SizeType incIndex = index - rows;
                    return incBlocks[incIndex / rowsInBlock] + ((size_t)(incIndex % rowsInBlock)) * cols;
                }
                return data + ((size_t)index) * cols;
            }

            T* operator[](SizeType index)
            {
                return (T*)At(index);
            }
            
            const T* operator[](SizeType index) const
            {
                return At(index);
            }

            void AddBatch(const T* pData, SizeType num)
            {
                if (R() > (std::numeric_limits<SizeType>::max)() - num) return;

                SizeType written = 0;
                while (written < num) {
                    SizeType curBlockIdx = (incRows + written) / rowsInBlock;
                    if (curBlockIdx >= incBlocks.size()) {
                        incBlocks.push_back((T*)aligned_malloc(((size_t)rowsInBlock) * cols * sizeof(T), ALIGN));
                    }
                    SizeType curBlockPos = (incRows + written) % rowsInBlock;
                    SizeType toWrite = min(rowsInBlock - curBlockPos, num - written);
                    std::memcpy(incBlocks[curBlockIdx] + ((size_t)curBlockPos) * cols, pData + ((size_t)written) * cols, ((size_t)toWrite) * cols * sizeof(T));
                    written += toWrite;
                }
                incRows += written;
            }

            void AddBatch(SizeType num)
            {
                if (R() > (std::numeric_limits<SizeType>::max)() - num) return;

                SizeType written = 0;
                while (written < num) {
                    SizeType curBlockIdx = (incRows + written) / rowsInBlock;
                    if (curBlockIdx >= incBlocks.size()) {
                        incBlocks.push_back((T*)aligned_malloc(((size_t)rowsInBlock) * cols * sizeof(T), ALIGN));
                    }
                    SizeType curBlockPos = (incRows + written) % rowsInBlock;
                    SizeType toWrite = min(rowsInBlock - curBlockPos, num - written);
                    std::memset(incBlocks[curBlockIdx] + ((size_t)curBlockPos) * cols, -1, ((size_t)toWrite) * cols * sizeof(T));
                    written += toWrite;
                }
                incRows += written;
            }

            bool Save(std::string sDataPointsFileName)
            {
                std::cout << "Save Data To " << sDataPointsFileName << std::endl;
                FILE * fp = fopen(sDataPointsFileName.c_str(), "wb");
                if (fp == NULL) return false;

                SizeType CR = R();
                fwrite(&CR, sizeof(SizeType), 1, fp);
                fwrite(&cols, sizeof(SizeType), 1, fp);

                SizeType written = 0;
                while (written < rows) 
                {
                    written += (SizeType)fwrite(data + ((size_t)written) * cols, sizeof(T) * cols, rows - written, fp);
                }

                written = 0;
                while (written < incRows)
                {
                    SizeType pos = written % rowsInBlock;
                    written += (SizeType)fwrite(incBlocks[written / rowsInBlock] + ((size_t)pos) * cols, sizeof(T) * cols, min(rowsInBlock - pos, incRows - written), fp);
                }
                fclose(fp);

                std::cout << "Save Data (" << CR << ", " << cols << ") Finish!" << std::endl;
                return true;
            }

            bool Load(std::string sDataPointsFileName)
            {
                std::cout << "Load Data From " << sDataPointsFileName << std::endl;
                FILE * fp = fopen(sDataPointsFileName.c_str(), "rb");
                if (fp == NULL) return false;

                SizeType R, C;
                fread(&R, sizeof(SizeType), 1, fp);
                fread(&C, sizeof(SizeType), 1, fp);

                Initialize(R, C);
                R = 0;
                while (R < rows) {
                    R += (SizeType)fread(data + ((size_t)R) * C, sizeof(T) * C, rows - R, fp);
                }
                fclose(fp);
                std::cout << "Load Data (" << rows << ", " << cols << ") Finish!" << std::endl;
                return true;
            }

            // Functions for loading models from memory mapped files
            bool Load(char* pDataPointsMemFile)
            {
                SizeType R, C;
                R = *((SizeType*)pDataPointsMemFile);
                pDataPointsMemFile += sizeof(SizeType);

                C = *((SizeType*)pDataPointsMemFile);
                pDataPointsMemFile += sizeof(SizeType);

                Initialize(R, C, (T*)pDataPointsMemFile);
                return true;
            }

            bool Refine(const std::vector<SizeType>& indices, std::string sDataPointsFileName)
            {
                std::cout << "Save Refine Data To " << sDataPointsFileName << std::endl;
                FILE * fp = fopen(sDataPointsFileName.c_str(), "wb");
                if (fp == NULL) return false;

                SizeType R = (SizeType)(indices.size());
                fwrite(&R, sizeof(SizeType), 1, fp);
                fwrite(&cols, sizeof(SizeType), 1, fp);

                // write point one by one in case for cache miss
                for (SizeType i = 0; i < R; i++) {
                    fwrite(At(indices[i]), sizeof(T) * cols, 1, fp);
                }
                fclose(fp);

                std::cout << "Save Refine Data (" << R << ", " << cols << ") Finish!" << std::endl;
                return true;
            }
        };
    }
}

#endif // _SPTAG_COMMON_DATASET_H_
