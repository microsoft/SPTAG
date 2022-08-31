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
        private:
            std::string name = "Data";
            SizeType rows = 0;
            DimensionType cols = 1;
            T* data = nullptr;
            bool ownData = false;
            SizeType incRows = 0;
            SizeType maxRows;
            SizeType rowsInBlock;
            SizeType rowsInBlockEx;
            std::vector<T*> incBlocks;

        public:
            Dataset() {}

            Dataset(SizeType rows_, DimensionType cols_, SizeType rowsInBlock_, SizeType capacity_, T* data_ = nullptr, bool shareOwnership_ = true)
            {
                Initialize(rows_, cols_, rowsInBlock_, capacity_, data_, shareOwnership_);
            }
            ~Dataset()
            {
                if (ownData) ALIGN_FREE(data);
                for (T* ptr : incBlocks) ALIGN_FREE(ptr);
                incBlocks.clear();
            }
            void Initialize(SizeType rows_, DimensionType cols_, SizeType rowsInBlock_, SizeType capacity_, T* data_ = nullptr, bool shareOwnership_ = true)
            {
                rows = rows_;
                cols = cols_;
                data = data_;
                if (data_ == nullptr || !shareOwnership_)
                {
                    ownData = true;
                    data = (T*)ALIGN_ALLOC(((size_t)rows) * cols * sizeof(T));
                    if (data_ != nullptr) memcpy(data, data_, ((size_t)rows) * cols * sizeof(T));
                    else std::memset(data, -1, ((size_t)rows) * cols * sizeof(T));
                }
                maxRows = capacity_;
                rowsInBlockEx = static_cast<SizeType>(ceil(log2(rowsInBlock_)));
                rowsInBlock = (1 << rowsInBlockEx) - 1;
                incBlocks.reserve((static_cast<std::int64_t>(capacity_) + rowsInBlock) >> rowsInBlockEx);
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
                    return incBlocks[incIndex >> rowsInBlockEx] + ((size_t)(incIndex & rowsInBlock)) * cols;
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
                if (R() > maxRows - num) return ErrorCode::MemoryOverFlow;

                SizeType written = 0;
                while (written < num) {
                    SizeType curBlockIdx = ((incRows + written) >> rowsInBlockEx);
                    if (curBlockIdx >= (SizeType)incBlocks.size()) {
                        T* newBlock = (T*)ALIGN_ALLOC(((size_t)rowsInBlock + 1) * cols * sizeof(T));
                        if (newBlock == nullptr) return ErrorCode::MemoryOverFlow;
                        incBlocks.push_back(newBlock);
                    }
                    SizeType curBlockPos = ((incRows + written) & rowsInBlock);
                    SizeType toWrite = min(rowsInBlock + 1 - curBlockPos, num - written);
                    std::memcpy(incBlocks[curBlockIdx] + ((size_t)curBlockPos) * cols, pData + ((size_t)written) * cols, ((size_t)toWrite) * cols * sizeof(T));
                    written += toWrite;
                }
                incRows += written;
                return ErrorCode::Success;
            }

            ErrorCode AddBatch(SizeType num)
            {
                if (R() > maxRows - num) return ErrorCode::MemoryOverFlow;

                SizeType written = 0;
                while (written < num) {
                    SizeType curBlockIdx = (incRows + written) >> rowsInBlockEx;
                    if (curBlockIdx >= (SizeType)incBlocks.size()) {
                        T* newBlock = (T*)ALIGN_ALLOC(sizeof(T) * (rowsInBlock + 1) * cols);
                        if (newBlock == nullptr) return ErrorCode::MemoryOverFlow;
                        std::memset(newBlock, -1, sizeof(T) * (rowsInBlock + 1) * cols);
                        incBlocks.push_back(newBlock);
                    }
                    written += min(rowsInBlock + 1 - ((incRows + written) & rowsInBlock), num - written);
                }
                incRows += written;
                return ErrorCode::Success;
            }

            ErrorCode Save(std::shared_ptr<Helper::DiskIO> p_out) const
            {
                SizeType CR = R();
                IOBINARY(p_out, WriteBinary, sizeof(SizeType), (char*)&CR);
                IOBINARY(p_out, WriteBinary, sizeof(DimensionType), (char*)&cols);
                IOBINARY(p_out, WriteBinary, sizeof(T) * cols * rows, (char*)data);

                SizeType blocks = (incRows >> rowsInBlockEx);
                for (int i = 0; i < blocks; i++)
                    IOBINARY(p_out, WriteBinary, sizeof(T) * cols * (rowsInBlock + 1), (char*)incBlocks[i]);

                SizeType remain = (incRows & rowsInBlock);
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

            ErrorCode Load(std::shared_ptr<Helper::DiskIO> pInput, SizeType blockSize, SizeType capacity)
            {
                IOBINARY(pInput, ReadBinary, sizeof(SizeType), (char*)&rows);
                IOBINARY(pInput, ReadBinary, sizeof(DimensionType), (char*)&cols);

                Initialize(rows, cols, blockSize, capacity);
                IOBINARY(pInput, ReadBinary, sizeof(T) * cols * rows, (char*)data);
                LOG(Helper::LogLevel::LL_Info, "Load %s (%d,%d) Finish!\n", name.c_str(), rows, cols);
                return ErrorCode::Success;
            }

            ErrorCode Load(std::string sDataPointsFileName, SizeType blockSize, SizeType capacity)
            {
                LOG(Helper::LogLevel::LL_Info, "Load %s From %s\n", name.c_str(), sDataPointsFileName.c_str());
                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(sDataPointsFileName.c_str(), std::ios::binary | std::ios::in)) return ErrorCode::FailedOpenFile;
                return Load(ptr, blockSize, capacity);
            }

            // Functions for loading models from memory mapped files
            ErrorCode Load(char* pDataPointsMemFile, SizeType blockSize, SizeType capacity)
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

            ErrorCode Refine(const std::vector<SizeType>& indices, Dataset<T>& data) const
            {
                SizeType R = (SizeType)(indices.size());
                data.Initialize(R, cols, rowsInBlock + 1, static_cast<SizeType>(incBlocks.capacity() * (rowsInBlock + 1)));
                for (SizeType i = 0; i < R; i++) {
                    std::memcpy((void*)data.At(i), (void*)this->At(indices[i]), sizeof(T) * cols);
                }
                return ErrorCode::Success;
            }

            ErrorCode Refine(const std::vector<SizeType>& indices, std::shared_ptr<Helper::DiskIO> output) const
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