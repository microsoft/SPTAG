// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_DATASET_H_
#define _SPTAG_COMMON_DATASET_H_

#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif // defined(__GNUC__)

#define ALIGN 32

#define aligned_malloc(a, b) _mm_malloc(a, b)
#define aligned_free(a) _mm_free(a)

namespace SPTAG
{
    namespace COMMON
    {
        // structure to save Data and Graph
        template <typename T>
        class Dataset
        {
        private:
            std::string name = "Data";
            SizeType rows = 0;
            DimensionType cols = 1;
            bool ownData = false;
            T* data = nullptr;
            SizeType incRows = 0;
            std::vector<T*> incBlocks;
            static const SizeType rowsInBlock = 1024 * 1024;
        public:
            Dataset()
            {
                incBlocks.reserve(MaxSize / rowsInBlock + 1); 
            }
            Dataset(SizeType rows_, DimensionType cols_, T* data_ = nullptr, bool transferOnwership_ = true)
            {
                Initialize(rows_, cols_, data_, transferOnwership_);
                incBlocks.reserve(MaxSize / rowsInBlock + 1);
            }
            ~Dataset()
            {
                if (ownData) aligned_free(data);
                for (T* ptr : incBlocks) aligned_free(ptr);
                incBlocks.clear();
            }
            void Initialize(SizeType rows_, DimensionType cols_, T* data_ = nullptr, bool transferOnwership_ = true)
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
            }
            void SetName(const std::string& name_) { name = name_; }
            const std::string& Name() const { return name; }

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
            inline DimensionType C() const { return cols; }
            inline std::uint64_t BufferSize() const { return sizeof(SizeType) + sizeof(DimensionType) + sizeof(T) * R() * C(); }

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

            ErrorCode AddBatch(const T* pData, SizeType num)
            {
                if (R() > MaxSize - num) return ErrorCode::MemoryOverFlow;

                SizeType written = 0;
                while (written < num) {
                    SizeType curBlockIdx = (incRows + written) / rowsInBlock;
                    if (curBlockIdx >= (SizeType)incBlocks.size()) {
                        T* newBlock = (T*)aligned_malloc(((size_t)rowsInBlock) * cols * sizeof(T), ALIGN);
                        if (newBlock == nullptr) return ErrorCode::MemoryOverFlow;
                        incBlocks.push_back(newBlock);
                    }
                    SizeType curBlockPos = (incRows + written) % rowsInBlock;
                    SizeType toWrite = min(rowsInBlock - curBlockPos, num - written);
                    std::memcpy(incBlocks[curBlockIdx] + ((size_t)curBlockPos) * cols, pData + ((size_t)written) * cols, ((size_t)toWrite) * cols * sizeof(T));
                    written += toWrite;
                }
                incRows += written;
                return ErrorCode::Success;
            }

            ErrorCode AddBatch(SizeType num)
            {
                if (R() > MaxSize - num) return ErrorCode::MemoryOverFlow;

                SizeType written = 0;
                while (written < num) {
                    SizeType curBlockIdx = (incRows + written) / rowsInBlock;
                    if (curBlockIdx >= (SizeType)incBlocks.size()) {
                        T* newBlock = (T*)aligned_malloc(sizeof(T) * rowsInBlock * cols, ALIGN);
                        if (newBlock == nullptr) return ErrorCode::MemoryOverFlow;
                        std::memset(newBlock, -1, sizeof(T) * rowsInBlock * cols);
                        incBlocks.push_back(newBlock);
                    }
                    written += min(rowsInBlock - ((incRows + written) % rowsInBlock), num - written);
                }
                incRows += written;
                return ErrorCode::Success;
            }

            ErrorCode Save(std::shared_ptr<Helper::DiskPriorityIO> p_out) const
            {
                SizeType CR = R();
                IOBINARY(p_out, WriteBinary, sizeof(SizeType), (char*)&CR);
                IOBINARY(p_out, WriteBinary, sizeof(DimensionType), (char*)&cols);
                IOBINARY(p_out, WriteBinary, sizeof(T) * cols * rows, (char*)data);
                
                SizeType blocks = incRows / rowsInBlock;
                for (int i = 0; i < blocks; i++)
                    IOBINARY(p_out, WriteBinary, sizeof(T) * cols * rowsInBlock, (char*)incBlocks[i]);

                SizeType remain = incRows % rowsInBlock;
                if (remain > 0) IOBINARY(p_out, WriteBinary, sizeof(T) * cols * remain, (char*)incBlocks[blocks]);
                LOG(Helper::LogLevel::LL_Info, "Save %s (%d,%d) Finish!\n", name.c_str(), CR, cols);
                return ErrorCode::Success;
            }

            ErrorCode Save(std::string sDataPointsFileName) const
            {
                LOG(Helper::LogLevel::LL_Info, "Save %s To %s\n", name.c_str(), sDataPointsFileName.c_str());
                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(sDataPointsFileName.c_str(), std::ios::binary | std::ios::out)) return ErrorCode::FailedCreateFile;
                return Save(ptr);
            }

            ErrorCode Load(std::shared_ptr<Helper::DiskPriorityIO> p_input)
            {
                IOBINARY(p_input, ReadBinary, sizeof(SizeType), (char*)&rows);
                IOBINARY(p_input, ReadBinary, sizeof(DimensionType), (char*)&cols);

                Initialize(rows, cols);
                IOBINARY(p_input, ReadBinary, sizeof(T) * cols * rows, (char*)data);
                LOG(Helper::LogLevel::LL_Info, "Load %s (%d,%d) Finish!\n", name.c_str(), rows, cols);
                return ErrorCode::Success;
            }

            ErrorCode Load(std::string sDataPointsFileName)
            {
                LOG(Helper::LogLevel::LL_Info, "Load %s From %s\n", name.c_str(), sDataPointsFileName.c_str());
                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(sDataPointsFileName.c_str(), std::ios::binary | std::ios::in)) return ErrorCode::FailedOpenFile;
                return Load(ptr);
            }

            // Functions for loading models from memory mapped files
            ErrorCode Load(char* pDataPointsMemFile)
            {
                SizeType R;
                DimensionType C;
                R = *((SizeType*)pDataPointsMemFile);
                pDataPointsMemFile += sizeof(SizeType);

                C = *((DimensionType*)pDataPointsMemFile);
                pDataPointsMemFile += sizeof(DimensionType);

                Initialize(R, C, (T*)pDataPointsMemFile);
                LOG(Helper::LogLevel::LL_Info, "Load %s (%d,%d) Finish!\n", name.c_str(), R, C);
                return ErrorCode::Success;
            }

            ErrorCode Refine(const std::vector<SizeType>& indices, Dataset<T>& data) const
            {
                SizeType R = (SizeType)(indices.size());
                data.Initialize(R, cols);
                for (SizeType i = 0; i < R; i++) {
                    std::memcpy((void*)data.At(i), (void*)this->At(indices[i]), sizeof(T) * cols);
                }
                return ErrorCode::Success;
            }

            ErrorCode Refine(const std::vector<SizeType>& indices, std::shared_ptr<Helper::DiskPriorityIO> output) const
            {
                SizeType R = (SizeType)(indices.size());
                IOBINARY(output, WriteBinary, sizeof(SizeType), (char*)&R);
                IOBINARY(output, WriteBinary, sizeof(DimensionType), (char*)&cols);

                for (SizeType i = 0; i < R; i++) {
                    IOBINARY(output, WriteBinary, sizeof(T) * cols, (char*)At(indices[i]));
                }
                LOG(Helper::LogLevel::LL_Info, "Save Refine %s (%d,%d) Finish!\n", name.c_str(), R, cols);
                return ErrorCode::Success;
            }

            ErrorCode Refine(const std::vector<SizeType>& indices, std::string sDataPointsFileName) const
            {
                LOG(Helper::LogLevel::LL_Info, "Save Refine %s To %s\n", name.c_str(), sDataPointsFileName.c_str());
                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(sDataPointsFileName.c_str(), std::ios::binary | std::ios::out)) return ErrorCode::FailedCreateFile;
                return Refine(indices, ptr);
            }
        };
    }
}

#endif // _SPTAG_COMMON_DATASET_H_
