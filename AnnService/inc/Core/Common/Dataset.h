// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_DATASET_H_
#define _SPTAG_COMMON_DATASET_H_

namespace SPTAG
{
    namespace COMMON
    {
        // structure to save Data and Graph
        template <typename T>
        class Dataset
        {
        protected:
            std::string name = "Data";
            SizeType rows = 0;
            DimensionType cols = 1;
            char* data = nullptr;
            bool ownData = false;
            SizeType incRows = 0;
            SizeType maxRows;
            SizeType rowsInBlock;
            SizeType rowsInBlockEx;
            std::vector<char*> incBlocks;

        public:
            Dataset() {}

            Dataset(SizeType rows_, DimensionType cols_, SizeType rowsInBlock_, SizeType capacity_, T* data_ = nullptr, bool transferOnwership_ = true)
            {
                Initialize(rows_, cols_, rowsInBlock_, capacity_, data_, transferOnwership_);
            }
            ~Dataset()
            {
                if (ownData) _mm_free(data);
                for (char* ptr : incBlocks) _mm_free(ptr);
                incBlocks.clear();
            }
            virtual void Initialize(SizeType rows_, DimensionType cols_, SizeType rowsInBlock_, SizeType capacity_, T* data_ = nullptr, bool transferOnwership_ = true)
            {
                rows = rows_;
                cols = cols_;
                data = (char*)data_;
                if (data_ == nullptr || !transferOnwership_)
                {
                    ownData = true;
                    data = (char*)_mm_malloc(((size_t)rows) * cols * sizeof(T), ALIGN_SPTAG);
                    if (data_ != nullptr) memcpy(data, data_, ((size_t)rows) * cols * sizeof(T));
                    else std::memset(data, -1, ((size_t)rows) * cols * sizeof(T));
                }
                maxRows = capacity_;
                rowsInBlockEx = static_cast<SizeType>(ceil(log2(rowsInBlock_)));
                rowsInBlock = (1 << rowsInBlockEx) - 1;
                incBlocks.reserve((static_cast<std::int64_t>(capacity_) + rowsInBlock) >> rowsInBlockEx);
            }
            virtual void SetName(const std::string& name_) { name = name_; }
            virtual const std::string& Name() const { return name; }

            virtual void SetR(SizeType R_) 
            {
                if (R_ >= rows)
                    incRows = R_ - rows;
                else 
                {
                    rows = R_;
                    incRows = 0;
                }
            }
            virtual inline SizeType R() const { return rows + incRows; }
            virtual inline DimensionType C() const { return cols; }
            virtual inline std::uint64_t BufferSize() const { return sizeof(SizeType) + sizeof(DimensionType) + sizeof(T) * R() * C(); }

            virtual inline const T* At(SizeType index) const
            {
                if (index >= rows) {
                    SizeType incIndex = index - rows;
                    return ((T*)incBlocks[incIndex >> rowsInBlockEx]) + ((size_t)(incIndex & rowsInBlock)) * cols;
                }
                return ((T*)data) + ((size_t)index) * cols;
            }

            virtual T* operator[](SizeType index)
            {
                return (T*)At(index);
            }
            
            virtual const T* operator[](SizeType index) const
            {
                return At(index);
            }

            virtual ErrorCode AddBatch(const T* pData, SizeType num)
            {
                if (R() > maxRows - num) return ErrorCode::MemoryOverFlow;

                SizeType written = 0;
                while (written < num) {
                    SizeType curBlockIdx = ((incRows + written) >> rowsInBlockEx);
                    if (curBlockIdx >= (SizeType)incBlocks.size()) {
                        char* newBlock = (char*)_mm_malloc(sizeof(T) * (rowsInBlock + 1) * cols, ALIGN_SPTAG);
                        if (newBlock == nullptr) return ErrorCode::MemoryOverFlow;
                        incBlocks.push_back(newBlock);
                    }
                    SizeType curBlockPos = ((incRows + written) & rowsInBlock);
                    SizeType toWrite = min(rowsInBlock + 1 - curBlockPos, num - written);
                    std::memcpy(((T*)incBlocks[curBlockIdx]) + ((size_t)curBlockPos) * cols, pData + ((size_t)written) * cols, ((size_t)toWrite) * cols * sizeof(T));
                    written += toWrite;
                }
                incRows += written;
                return ErrorCode::Success;
            }

            virtual ErrorCode AddBatch(SizeType num)
            {
                if (R() > maxRows - num) return ErrorCode::MemoryOverFlow;

                SizeType written = 0;
                while (written < num) {
                    SizeType curBlockIdx = (incRows + written) >> rowsInBlockEx;
                    if (curBlockIdx >= (SizeType)incBlocks.size()) {
                        char* newBlock = (char*)_mm_malloc(sizeof(T) * (rowsInBlock + 1) * cols, ALIGN_SPTAG);
                        if (newBlock == nullptr) return ErrorCode::MemoryOverFlow;
                        std::memset(newBlock, -1, sizeof(T) * (rowsInBlock + 1) * cols);
                        incBlocks.push_back(newBlock);
                    }
                    written += min(rowsInBlock + 1 - ((incRows + written) & rowsInBlock), num - written);
                }
                incRows += written;
                return ErrorCode::Success;
            }

            virtual ErrorCode Save(std::shared_ptr<Helper::DiskPriorityIO> p_out) const
            {
                SizeType CR = R();
                IOBINARY(p_out, WriteBinary, sizeof(SizeType), (char*)&CR);
                IOBINARY(p_out, WriteBinary, sizeof(DimensionType), (char*)&cols);
                IOBINARY(p_out, WriteBinary, sizeof(T) * cols * rows, data);
                
                SizeType blocks = (incRows >> rowsInBlockEx);
                for (int i = 0; i < blocks; i++)
                    IOBINARY(p_out, WriteBinary, sizeof(T) * cols * (rowsInBlock + 1), incBlocks[i]);

                SizeType remain = (incRows & rowsInBlock);
                if (remain > 0) IOBINARY(p_out, WriteBinary, sizeof(T) * cols * remain, incBlocks[blocks]);
                LOG(Helper::LogLevel::LL_Info, "Save %s (%d,%d) Finish!\n", name.c_str(), CR, cols);
                return ErrorCode::Success;
            }

            virtual ErrorCode Save(std::string sDataPointsFileName) const
            {
                LOG(Helper::LogLevel::LL_Info, "Save %s To %s\n", name.c_str(), sDataPointsFileName.c_str());
                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(sDataPointsFileName.c_str(), std::ios::binary | std::ios::out)) return ErrorCode::FailedCreateFile;
                return Save(ptr);
            }

            virtual ErrorCode Load(std::shared_ptr<Helper::DiskPriorityIO> pInput, SizeType blockSize, SizeType capacity)
            {
                IOBINARY(pInput, ReadBinary, sizeof(SizeType), (char*)&rows);
                IOBINARY(pInput, ReadBinary, sizeof(DimensionType), (char*)&cols);

                Initialize(rows, cols, blockSize, capacity);
                IOBINARY(pInput, ReadBinary, sizeof(T) * cols * rows, data);
                LOG(Helper::LogLevel::LL_Info, "Load %s (%d,%d) Finish!\n", name.c_str(), rows, cols);
                return ErrorCode::Success;
            }

            virtual ErrorCode Load(std::string sDataPointsFileName, SizeType blockSize, SizeType capacity)
            {
                LOG(Helper::LogLevel::LL_Info, "Load %s From %s\n", name.c_str(), sDataPointsFileName.c_str());
                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(sDataPointsFileName.c_str(), std::ios::binary | std::ios::in)) return ErrorCode::FailedOpenFile;
                return Load(ptr, blockSize, capacity);
            }

            // Functions for loading models from memory mapped files
            virtual ErrorCode Load(char* pDataPointsMemFile, SizeType blockSize, SizeType capacity)
            {
                SizeType R;
                DimensionType C;
                R = *((SizeType*)pDataPointsMemFile);
                pDataPointsMemFile += sizeof(SizeType);

                C = *((DimensionType*)pDataPointsMemFile);
                pDataPointsMemFile += sizeof(DimensionType);

                Initialize(R, C, blockSize, capacity, (T*)pDataPointsMemFile);
                LOG(Helper::LogLevel::LL_Info, "Load %s (%d,%d) Finish!\n", name.c_str(), R, C);
                return ErrorCode::Success;
            }

            virtual ErrorCode Refine(const std::vector<SizeType>& indices, Dataset<T>& dataset) const
            {
                SizeType R = (SizeType)(indices.size());
                dataset.Initialize(R, cols, rowsInBlock + 1, static_cast<SizeType>(incBlocks.capacity() * (rowsInBlock + 1)));
                for (SizeType i = 0; i < R; i++) {
                    std::memcpy((void*)dataset.At(i), (void*)this->At(indices[i]), sizeof(T) * cols);
                }
                return ErrorCode::Success;
            }

            virtual ErrorCode Refine(const std::vector<SizeType>& indices, std::shared_ptr<Helper::DiskPriorityIO> output) const
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

            virtual ErrorCode Refine(const std::vector<SizeType>& indices, std::string sDataPointsFileName) const
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
