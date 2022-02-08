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
            DimensionType colStart = 0;
            DimensionType mycols = 0;

        public:
            OptimizedDataset() {}

            OptimizedDataset(SizeType rows_, DimensionType cols_, SizeType rowsInBlock_, SizeType capacity_, char* data_ = nullptr, bool transferOnwership_ = true, int colStart_ = 0, int colEnd_ = -1, std::string name_ = "Data")
            {
                Initialize(rows_, cols_, rowsInBlock_, capacity_, data_, transferOnwership_, colStart_, colEnd_, name_);
            }
            ~OptimizedDataset()
            {
            }

            void Initialize(SizeType rows_, DimensionType cols_, SizeType rowsInBlock_, SizeType capacity_, char* data_ = nullptr, bool transferOnwership_ = true, int colStart_ = 0, int colEnd_ = -1, std::string name_ = "Data")
            {
                this->name = name_;
                this->rows = rows_;
                this->cols = cols_;
                this->data = data_;
                this->ownData = transferOnwership_;
                if (data_ == nullptr)
                {
                    this->ownData = true;
                    this->data = (char*)_mm_malloc(((size_t)this->rows) * this->cols, ALIGN_SPTAG);
                    std::memset(this->data, -1, ((size_t)this->rows) * this->cols);
                }
                this->maxRows = capacity_;
                this->rowsInBlockEx = static_cast<SizeType>(ceil(log2(rowsInBlock_)));
                this->rowsInBlock = (1 << this->rowsInBlockEx) - 1;
                this->incBlocks.reserve((static_cast<std::int64_t>(capacity_) + this->rowsInBlock) >> this->rowsInBlockEx);

                colStart = colStart_;
                mycols = (colEnd_ - colStart_) / sizeof(T);
            }

            virtual inline const DimensionType& C() const { return mycols; }

#define GETITEM(index) \
            if (index >= this->rows) { \
            SizeType incIndex = index - this->rows; \
            return (T*)(this->incBlocks[incIndex >> this->rowsInBlockEx] + ((size_t)(incIndex & this->rowsInBlock)) * this->cols + colStart); \
            } \
            return (T*)(this->data + ((size_t)index) * this->cols + colStart); \

            virtual inline const T* At(SizeType index) const
            {
                GETITEM(index)
            }

            virtual inline T* At(SizeType index)
            {
                GETITEM(index)
            }
#undef GETITEM

            virtual ErrorCode AddBatch(SizeType num, const T* pData = nullptr)
            {
                if (colStart != 0) return ErrorCode::Success;
                if (this->R() > this->maxRows - num) return ErrorCode::MemoryOverFlow;

                SizeType written = 0;
                while (written < num) {
                    SizeType curBlockIdx = ((this->incRows + written) >> this->rowsInBlockEx);
                    if (curBlockIdx >= (SizeType)(this->incBlocks).size()) {
                        char* newBlock = (char*)_mm_malloc(((size_t)this->rowsInBlock + 1) * this->cols, ALIGN_SPTAG);
                        if (newBlock == nullptr) return ErrorCode::MemoryOverFlow;
                        std::memset(newBlock, -1, ((size_t)this->rowsInBlock + 1) * this->cols);
                        (this->incBlocks).push_back(newBlock);
                    }
                    SizeType curBlockPos = ((this->incRows + written) & this->rowsInBlock);
                    SizeType toWrite = min(this->rowsInBlock + 1 - curBlockPos, num - written);
                    if (pData) {
                        for (int i = 0; i < toWrite; i++) {
                            std::memcpy(this->incBlocks[curBlockIdx] + ((size_t)curBlockPos + i) * this->cols + colStart, pData + ((size_t)written + i) * mycols, mycols * sizeof(T));
                        }
                    }
                    written += toWrite;
                }
                this->incRows += written;
                return ErrorCode::Success;
            }

            virtual ErrorCode Save(std::shared_ptr<Helper::DiskPriorityIO> p_out) const
            {
                SizeType CR = this->R();
                IOBINARY(p_out, WriteBinary, sizeof(SizeType), (char*)&CR);
                IOBINARY(p_out, WriteBinary, sizeof(DimensionType), (char*)&mycols);
                for (SizeType i = 0; i < CR; i++) {
                    IOBINARY(p_out, WriteBinary, sizeof(T) * mycols, (char*)At(i));
                }

                LOG(Helper::LogLevel::LL_Info, "Save %s (%d,%d) Finish!\n", this->name.c_str(), CR, mycols);
                return ErrorCode::Success;
            }

            virtual ErrorCode Load(std::shared_ptr<Helper::DiskPriorityIO> pInput, SizeType blockSize, SizeType capacity)
            {
                IOBINARY(pInput, ReadBinary, sizeof(SizeType), (char*)&(this->rows));
                IOBINARY(pInput, ReadBinary, sizeof(DimensionType), (char*)&mycols);

                for (SizeType i = 0; i < this->rows; i++) {
                    IOBINARY(pInput, ReadBinary, sizeof(T) * mycols, (char*)At(i));
                }
                LOG(Helper::LogLevel::LL_Info, "Load %s (%d,%d) Finish!\n", this->name.c_str(), this->rows, mycols);
                return ErrorCode::Success;
            }
        };

        template <typename T>
        ErrorCode LoadOptimizedDatasets(std::shared_ptr<Helper::DiskPriorityIO> pVectorsInput, std::shared_ptr<Helper::DiskPriorityIO> pGraphInput, 
            std::shared_ptr<COMMON::Dataset<T>>& pVectors, std::shared_ptr<COMMON::Dataset<SizeType>>& pGraph, DimensionType pNeighborhoodSize,
            SizeType blockSize, SizeType capacity) {
            if (pVectorsInput == nullptr || pGraphInput == nullptr) return ErrorCode::LackOfInputs;

            SizeType VR, GR;
            DimensionType VC, GC;
            IOBINARY(pVectorsInput, ReadBinary, sizeof(SizeType), (char*)&VR);
            IOBINARY(pVectorsInput, ReadBinary, sizeof(DimensionType), (char*)&VC);

            char* data = (char*)_mm_malloc((sizeof(T) * VC + sizeof(SizeType) * pNeighborhoodSize) * VR, ALIGN_SPTAG);
            pVectors.reset(new OptimizedDataset<T>(VR, sizeof(T) * VC + sizeof(SizeType) * pNeighborhoodSize, blockSize, capacity, data, true, 0, sizeof(T) * VC, "Opt" + pVectors->Name()));
            for (SizeType i = 0; i < VR; i++) {
                IOBINARY(pVectorsInput, ReadBinary, sizeof(T) * VC, (char*)(pVectors->At(i)));
            }

            IOBINARY(pGraphInput, ReadBinary, sizeof(SizeType), (char*)&GR);
            IOBINARY(pGraphInput, ReadBinary, sizeof(DimensionType), (char*)&GC);
            if (GR != VR || GC != pNeighborhoodSize) return ErrorCode::DiskIOFail;

            pGraph.reset(new OptimizedDataset<SizeType>(GR, sizeof(T) * VC + sizeof(SizeType) * GC, blockSize, capacity, data, false, sizeof(T) * VC, sizeof(T) * VC + sizeof(SizeType) * GC, "Opt" + pGraph->Name()));
            for (SizeType i = 0; i < VR; i++) {
                IOBINARY(pGraphInput, ReadBinary, sizeof(SizeType) * GC, (char*)(pGraph->At(i)));
            }

            LOG(Helper::LogLevel::LL_Info, "Load %s (%d,%d) Finish!\n", pVectors->Name().c_str(), pVectors->R(), pVectors->C());
            LOG(Helper::LogLevel::LL_Info, "Load %s (%d,%d) Finish!\n", pGraph->Name().c_str(), pGraph->R(), pGraph->C());

            return ErrorCode::Success;
        }
    }
}
#endif //_SPTAG_COMMON_OPTIMIZEDDATASET_H_