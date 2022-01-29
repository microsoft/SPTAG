// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_OPTIMIZEDDATASET_H_
#define _SPTAG_COMMON_OPTIMIZEDDATASET_H_

#include "Dataset.h"
#include <cstdint>

namespace SPTAG
{
    namespace COMMON
    {
        // structure to save Data and Graph
        template <typename T>
        class OptimizedDataset : public Dataset<T>
        {
        private:
            DimensionType colStart;
            DimensionType mycols;

        public:
            OptimizedDataset() {}

            OptimizedDataset(SizeType rows_, DimensionType cols_, SizeType rowsInBlock_, SizeType capacity_, char* data_ = nullptr, bool transferOnwership_ = true)
            {
                Initialize(rows_, cols_, rowsInBlock_, capacity_, data_, transferOnwership_);
            }
            ~OptimizedDataset()
            {
            }

            void Initialize(SizeType rows_, DimensionType cols_, SizeType rowsInBlock_, SizeType capacity_, char* data_ = nullptr, bool transferOnwership_ = true)
            {
                rows = rows_;
                cols = cols_;
                data = data_;
                if (data_ == nullptr || !transferOnwership_)
                {
                    ownData = true;
                    data = _mm_malloc(((size_t)rows) * cols, ALIGN_SPTAG);
                    if (data_ != nullptr) memcpy(data, data_, ((size_t)rows) * cols);
                    else std::memset(data, -1, ((size_t)rows) * cols);
                }
                maxRows = capacity_;
                rowsInBlockEx = static_cast<SizeType>(ceil(log2(rowsInBlock_)));
                rowsInBlock = (1 << rowsInBlockEx) - 1;
                incBlocks.reserve((static_cast<std::int64_t>(capacity_) + rowsInBlock) >> rowsInBlockEx);
            }


            void SetTypeRange(int colStart_, int colEnd_) {
                colStart = colStart_;
                mycols = (colEnd_ - colStart_) / sizeof(T);
            }

            virtual inline DimensionType C() const { return mycols; }

            virtual inline const T* At(SizeType index) const
            {
                if (index >= rows) {
                    SizeType incIndex = index - rows;
                    return (const T*)(incBlocks[incIndex >> rowsInBlockEx] + ((size_t)(incIndex & rowsInBlock)) * cols + colStart);
                }
                return (const T*)(data + ((size_t)index) * cols + colStart);
            }

            virtual ErrorCode AddBatch(const T* pData, SizeType num)
            {
                if (R() > maxRows - num) return ErrorCode::MemoryOverFlow;

                SizeType written = 0;
                while (written < num) {
                    SizeType curBlockIdx = ((incRows + written) >> rowsInBlockEx);
                    if (curBlockIdx >= (SizeType)incBlocks.size()) {
                        T* newBlock = (T*)_mm_malloc(((size_t)rowsInBlock + 1) * cols, ALIGN_SPTAG);
                        if (newBlock == nullptr) return ErrorCode::MemoryOverFlow;
                        std::memset(newBlock, -1, ((size_t)rowsInBlock + 1) * cols);
                        incBlocks.push_back(newBlock);
                    }
                    SizeType curBlockPos = ((incRows + written) & rowsInBlock);
                    SizeType toWrite = min(rowsInBlock + 1 - curBlockPos, num - written);
                    for (int i = 0; i < toWrite; i++) {
                        std::memcpy(incBlocks[curBlockIdx] + ((size_t)curBlockPos + i) * cols + colStart, pData + ((size_t)written + i) * mycols, mycols * sizeof(T));
                    }
                    written += toWrite;
                }
                incRows += written;
                return ErrorCode::Success;
            }

            virtual ErrorCode Save(std::shared_ptr<Helper::DiskPriorityIO> p_out) const
            {
                SizeType CR = R();
                IOBINARY(p_out, WriteBinary, sizeof(SizeType), (char*)&CR);
                IOBINARY(p_out, WriteBinary, sizeof(DimensionType), (char*)&mycols);
                for (SizeType i = 0; i < CR; i++) {
                    IOBINARY(output, WriteBinary, sizeof(T) * mycols, (char*)At(i));
                }

                LOG(Helper::LogLevel::LL_Info, "Save %s (%d,%d) Finish!\n", name.c_str(), CR, cols);
                return ErrorCode::Success;
            }

            virtual ErrorCode Load(std::shared_ptr<Helper::DiskPriorityIO> pInput, SizeType blockSize, SizeType capacity)
            {
                IOBINARY(pInput, ReadBinary, sizeof(SizeType), (char*)&rows);
                IOBINARY(pInput, ReadBinary, sizeof(DimensionType), (char*)&mycols);

                for (SizeType i = 0; i < rows; i++) {
                    IOBINARY(pInput, ReadBinary, sizeof(T) * mycols, (char*)At(i));
                }
                LOG(Helper::LogLevel::LL_Info, "Load %s (%d,%d) Finish!\n", name.c_str(), rows, cols);
                return ErrorCode::Success;
            }

            // Functions for loading models from memory mapped files
            virtual ErrorCode Load(char* pDataPointsMemFile, SizeType blockSize, SizeType capacity)
            {
                rows = *((SizeType*)pDataPointsMemFile);
                pDataPointsMemFile += sizeof(SizeType);

                mycols = *((DimensionType*)pDataPointsMemFile);
                pDataPointsMemFile += sizeof(DimensionType);

                for (SizeType i = 0; i < rows; i++) {
                    std::memcpy((void*)At(i), pDataPointsMemFile + sizeof(T) * mycols * i, sizeof(T) * mycols);
                }
                LOG(Helper::LogLevel::LL_Info, "Load %s (%d,%d) Finish!\n", name.c_str(), rows, mycols);
                return ErrorCode::Success;
            }

            virtual ErrorCode Refine(const std::vector<SizeType>& indices, OptimizedDataset<T>& data) const
            {
                SizeType R = (SizeType)(indices.size());
                for (SizeType i = 0; i < R; i++) {
                    std::memcpy((void*)data.At(i), (void*)this->At(indices[i]), sizeof(T) * mycols);
                }
                return ErrorCode::Success;
            }

            virtual ErrorCode Refine(const std::vector<SizeType>& indices, std::shared_ptr<Helper::DiskPriorityIO> output) const
            {
                SizeType R = (SizeType)(indices.size());
                IOBINARY(output, WriteBinary, sizeof(SizeType), (char*)&R);
                IOBINARY(output, WriteBinary, sizeof(DimensionType), (char*)&mycols);

                for (SizeType i = 0; i < R; i++) {
                    IOBINARY(output, WriteBinary, sizeof(T) * mycols, (char*)At(indices[i]));
                }
                LOG(Helper::LogLevel::LL_Info, "Save Refine %s (%d,%d) Finish!\n", name.c_str(), R, mycols);
                return ErrorCode::Success;
            }
        };
    }
}
#endif //_SPTAG_COMMON_OPTIMIZEDDATASET_H_