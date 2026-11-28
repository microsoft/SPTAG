// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_DATASET_H_
#define _SPTAG_COMMON_DATASET_H_

#include "ResidentMemory.h"
#include "ResidentMetadata.h"
#include "inc/Core/VectorIndex.h"

namespace SPTAG
{
    namespace COMMON
    {
        // structure to save Data and Graph
    struct DatasetBuffer
    {
        ResidentMemory::Allocation charge;
        char *data = nullptr;
        size_t bytes = 0;

        ~DatasetBuffer()
        {
            ALIGN_FREE(data);
            charge.Reset();
        }

        static std::shared_ptr<DatasetBuffer> Allocate(const std::shared_ptr<ResidentMemory::Account> &account,
                                                       size_t bytes)
        {
            auto result = std::allocate_shared<DatasetBuffer>(ResidentAllocator<DatasetBuffer>(account));
            result->charge = ResidentMemory::Charge(account, bytes);
            result->bytes = bytes;
            result->data = static_cast<char *>(ALIGN_ALLOC(bytes));
            if (!result->data && bytes)
                throw std::bad_alloc();
            return result;
        }
    };

    using DatasetBlocks =
        std::vector<std::shared_ptr<DatasetBuffer>, ResidentAllocator<std::shared_ptr<DatasetBuffer>>>;

    /*
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

        Dataset(SizeType rows_, DimensionType cols_, SizeType rowsInBlock_, SizeType capacity_, T* data_ = nullptr, bool
    shareOwnership_ = true)
        {
            Initialize(rows_, cols_, rowsInBlock_, capacity_, data_, shareOwnership_);
        }
        ~Dataset()
        {
            if (ownData) ALIGN_FREE(data);
            for (T* ptr : incBlocks) ALIGN_FREE(ptr);
            incBlocks.clear();
        }
        void Initialize(SizeType rows_, DimensionType cols_, SizeType rowsInBlock_, SizeType capacity_, T* data_ =
    nullptr, bool shareOwnership_ = true)
        {
            if (data != nullptr) {
                if (ownData) ALIGN_FREE(data);
                for (T* ptr : incBlocks) ALIGN_FREE(ptr);
                incBlocks.clear();
            }

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
        inline std::uint64_t BufferSize() const { return sizeof(SizeType) + sizeof(DimensionType) + sizeof(T) * R() *
    C(); }

        inline const T* At(SizeType index) const
        {
            if (index < R() && index >= 0)
            {
                if (index >= rows) {
                    SizeType incIndex = index - rows;
                    return incBlocks[incIndex >> rowsInBlockEx] + ((size_t)(incIndex & rowsInBlock)) * cols;
                }
                return data + ((size_t)index) * cols;
            }
            else
            {
                std::ostringstream oss;
                oss << "Index out of range in Dataset. Index: " << index << " Size: " << R();
                throw std::out_of_range(oss.str());
            }
        }

        T* operator[](SizeType index)
        {
            return (T*)At(index);
        }

        const T* operator[](SizeType index) const
        {
            return At(index);
        }

        ErrorCode AddBatch(SizeType num, const T* pData = nullptr)
        {
            if (R() > maxRows - num) return ErrorCode::MemoryOverFlow;

            SizeType written = 0;
            while (written < num) {
                SizeType curBlockIdx = ((incRows + written) >> rowsInBlockEx);
                if (curBlockIdx >= (SizeType)incBlocks.size()) {
                    T* newBlock = (T*)ALIGN_ALLOC(sizeof(T) * (rowsInBlock + 1) * cols);
                    if (newBlock == nullptr) return ErrorCode::MemoryOverFlow;
                    std::memset(newBlock, -1, sizeof(T) * (rowsInBlock + 1) * cols);
                    incBlocks.push_back(newBlock);
                }
                SizeType curBlockPos = ((incRows + written) & rowsInBlock);
                SizeType toWrite = min(rowsInBlock + 1 - curBlockPos, num - written);
                if (pData != nullptr) std::memcpy(incBlocks[curBlockIdx] + ((size_t)curBlockPos) * cols, pData +
    ((size_t)written) * cols, ((size_t)toWrite) * cols * sizeof(T)); written += toWrite;
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
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Save %s (%d,%d) Finish!\n", name.c_str(), CR, cols);
            return ErrorCode::Success;
        }

        ErrorCode Save(std::string sDataPointsFileName) const
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Save %s To %s\n", name.c_str(), sDataPointsFileName.c_str());
            auto ptr = f_createIO();
            if (ptr == nullptr || !ptr->Initialize(sDataPointsFileName.c_str(), std::ios::binary | std::ios::out))
    return ErrorCode::FailedCreateFile; return Save(ptr);
        }

        ErrorCode Load(std::shared_ptr<Helper::DiskIO> pInput, SizeType blockSize, SizeType capacity)
        {
            IOBINARY(pInput, ReadBinary, sizeof(SizeType), (char*)&rows);
            IOBINARY(pInput, ReadBinary, sizeof(DimensionType), (char*)&cols);

            Initialize(rows, cols, blockSize, capacity);
            IOBINARY(pInput, ReadBinary, sizeof(T) * cols * rows, (char*)data);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load %s (%d,%d) Finish!\n", name.c_str(), rows, cols);
            return ErrorCode::Success;
        }

        ErrorCode Load(std::string sDataPointsFileName, SizeType blockSize, SizeType capacity)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load %s From %s\n", name.c_str(), sDataPointsFileName.c_str());
            auto ptr = f_createIO();
            if (ptr == nullptr || !ptr->Initialize(sDataPointsFileName.c_str(), std::ios::binary | std::ios::in)) return
    ErrorCode::FailedOpenFile; return Load(ptr, blockSize, capacity);
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
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load %s (%d,%d) Finish!\n", name.c_str(), R, C);
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
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Save Refine %s (%d,%d) Finish!\n", name.c_str(), R, cols);
            return ErrorCode::Success;
        }

        ErrorCode Refine(const std::vector<SizeType>& indices, std::string sDataPointsFileName) const
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Save Refine %s To %s\n", name.c_str(),
    sDataPointsFileName.c_str()); auto ptr = f_createIO(); if (ptr == nullptr ||
    !ptr->Initialize(sDataPointsFileName.c_str(), std::ios::binary | std::ios::out)) return ErrorCode::FailedCreateFile;
            return Refine(indices, ptr);
        }
    };
    */
    template <typename T> class Dataset
    {
      private:
        ResidentStringOwners<1> m_nameOwner;
        std::string name = "Data";
        SizeType rows = 0;
        DimensionType cols = 1;
        char *data = nullptr;
        std::shared_ptr<ResidentMemory::Account> m_account = ResidentMemory::CurrentAccount();
        std::shared_ptr<DatasetBuffer> m_backing;
        SizeType incRows = 0;
        SizeType maxRows = MaxSize;
        SizeType rowsInBlock = 1024 * 1024;
        SizeType rowsInBlockEx = 20;
        std::shared_ptr<DatasetBlocks> incBlocks;

        DimensionType colStart = 0;
        DimensionType mycols = 0;

        uint8_t defaultValue = -1;

      public:
        Dataset()
        {
        }
        Dataset(const Dataset &other) : m_nameOwner(other.m_account)
        {
            *this = other;
        }
        Dataset &operator=(const Dataset &other)
        {
            if (this == &other)
                return *this;
            m_nameOwner.Bind(other.m_account);
            m_nameOwner.Assign(name, other.name);
            rows = other.rows;
            cols = other.cols;
            data = other.data;
            m_account = other.m_account;
            m_backing = other.m_backing;
            incRows = other.incRows;
            maxRows = other.maxRows;
            rowsInBlock = other.rowsInBlock;
            rowsInBlockEx = other.rowsInBlockEx;
            incBlocks = other.incBlocks;
            colStart = other.colStart;
            mycols = other.mycols;
            defaultValue = other.defaultValue;
            return *this;
        }

        Dataset(SizeType rows_, DimensionType cols_, SizeType rowsInBlock_, SizeType capacity_,
                const void *data_ = nullptr, bool shareOwnership_ = true,
                std::shared_ptr<DatasetBlocks> incBlocks_ = nullptr, int colStart_ = 0, int rowEnd_ = -1)
        {
            Initialize(rows_, cols_, rowsInBlock_, capacity_, data_, shareOwnership_, incBlocks_, colStart_, rowEnd_);
        }
        ~Dataset() = default;

        ErrorCode Initialize(SizeType rows_, DimensionType cols_, SizeType rowsInBlock_, SizeType capacity_,
                             const void *data_ = nullptr, bool shareOwnership_ = true,
                             std::shared_ptr<DatasetBlocks> incBlocks_ = nullptr, int colStart_ = 0, int rowEnd_ = -1)
        {
            if (rows_ < 0 || cols_ <= 0 || rowsInBlock_ <= 0 || capacity_ < rows_)
                return ErrorCode::FailedParseValue;
            const auto rowBytes =
                rowEnd_ >= colStart_ ? static_cast<size_t>(rowEnd_) : static_cast<size_t>(cols_) * sizeof(T);
            if (rowBytes > static_cast<size_t>((std::numeric_limits<DimensionType>::max)()) ||
                (rows_ && rowBytes > (std::numeric_limits<size_t>::max)() / static_cast<size_t>(rows_)))
                return ErrorCode::MemoryOverFlow;
            auto account = m_account ? m_account : ResidentMemory::CurrentAccount();
            m_nameOwner.Bind(account);
            m_nameOwner.Assign(name, name);
            std::shared_ptr<DatasetBuffer> backing;
            const auto blockExponent = static_cast<SizeType>(ceil(log2(rowsInBlock_)));
            if (blockExponent >= 31)
                return ErrorCode::MemoryOverFlow;
            const auto blockMask = (1 << blockExponent) - 1;
            try
            {
                if (data_ == nullptr || !shareOwnership_)
                {
                    backing = DatasetBuffer::Allocate(account, static_cast<size_t>(rows_) * rowBytes);
                    if (data_)
                        std::memcpy(backing->data, data_, static_cast<size_t>(rows_) * rowBytes);
                    else if (rows_)
                        std::memset(backing->data, defaultValue, static_cast<size_t>(rows_) * rowBytes);
                }
                if (!incBlocks_)
                    incBlocks_ =
                        std::allocate_shared<DatasetBlocks>(ResidentAllocator<DatasetBlocks>(account),
                                                            ResidentAllocator<std::shared_ptr<DatasetBuffer>>(account));
                incBlocks_->reserve((static_cast<std::int64_t>(capacity_) + blockMask) >> blockExponent);
            }
            catch (const std::bad_alloc &)
            {
                return ErrorCode::MemoryOverFlow;
            }
            m_account = std::move(account);
            m_backing = std::move(backing);
            data = m_backing ? m_backing->data : static_cast<char *>(const_cast<void *>(data_));
            rows = rows_;
            incRows = 0;
            cols = static_cast<DimensionType>(rowBytes);
            maxRows = capacity_;
            rowsInBlockEx = blockExponent;
            rowsInBlock = blockMask;
            incBlocks = std::move(incBlocks_);
            colStart = colStart_;
            mycols = cols_;
            return ErrorCode::Success;
        }

        template <typename U>
        ErrorCode InitializeSharedView(const Dataset<U> &source, DimensionType columns, DimensionType offset)
        {
            if (offset < 0 || static_cast<size_t>(offset) + static_cast<size_t>(columns) * sizeof(T) >
                                  static_cast<size_t>(source.cols))
                return ErrorCode::FailedParseValue;
            auto status = Initialize(source.rows, columns, source.rowsInBlock + 1, source.maxRows, source.data, true,
                                     source.incBlocks, offset, source.cols);
            if (status == ErrorCode::Success)
            {
                m_account = source.m_account;
                m_backing = source.m_backing;
                incRows = source.incRows;
            }
            return status;
        }

        uint64_t AllocatedBackingBytes() const
        {
            uint64_t result = m_backing ? m_backing->bytes : 0;
            if (incBlocks)
                for (const auto &block : *incBlocks)
                    result += block->bytes;
            return result;
        }

        uint64_t ResidentGrowthBytes(SizeType target) const
        {
            if (target < 0 || target > maxRows || !incBlocks)
                return (std::numeric_limits<uint64_t>::max)();
            if (colStart != 0)
                return 0;
            const auto additional = (std::max<int64_t>)(0, static_cast<int64_t>(target) - rows);
            const auto blocks = static_cast<uint64_t>((additional + rowsInBlock) >> rowsInBlockEx);
            if (blocks <= incBlocks->size())
                return 0;
            const auto blockBytes =
                ResidentAddBytes(ResidentMultiplyBytes(static_cast<uint64_t>(rowsInBlock) + 1, cols),
                                 ResidentSharedBytes<DatasetBuffer>());
            auto result = ResidentMultiplyBytes(blocks - incBlocks->size(), blockBytes);
            if (blocks > incBlocks->capacity())
                result = ResidentAddBytes(result,
                                          ResidentAllocator<std::shared_ptr<DatasetBuffer>>::AllocationBytes(blocks));
            return result;
        }

        template <typename> friend class Dataset;

        bool IsReady() const
        {
            return data != nullptr; }

        void SetName(const std::string &name_)
        {
            m_nameOwner.Assign(name, name_);
        }
            void SetDefaultValue(uint8_t value) { defaultValue = value; }
            
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
            inline const DimensionType& C() const { return mycols; }
            inline std::uint64_t BufferSize() const { return sizeof(SizeType) + sizeof(DimensionType) + sizeof(T) * R() * C(); }

#define GETITEM(index)                                                                                             \
            if (index >= rows)                                                                                         \
            {                                                                                                          \
                SizeType incIndex = index - rows;                                                                      \
                return (T *)((*incBlocks)[incIndex >> rowsInBlockEx]->data +                                           \
                             ((size_t)(incIndex & rowsInBlock)) * cols + colStart);                                    \
            }                                                                                                          \
            return (T *)(data + ((size_t)index) * cols + colStart);

            inline const T* At(SizeType index) const
            {
                GETITEM(index)
            }

            inline T* At(SizeType index)
            {
                GETITEM(index)
            }

            inline T* operator[](SizeType index)
            {
                GETITEM(index)
            }

            inline const T *operator[](SizeType index) const {GETITEM(index)}

#undef GETITEM

            ErrorCode AddBatch(SizeType num, const T *pData = nullptr)
            {
                if (colStart != 0)
                    return ErrorCode::Success;
                if (num < 0 || R() > maxRows - num)
                    return ErrorCode::MemoryOverFlow;
                auto status = ReserveRows(R() + num);
                if (status != ErrorCode::Success)
                    return status;

                SizeType written = 0;
                while (written < num)
                {
                    SizeType curBlockIdx = ((incRows + written) >> rowsInBlockEx);
                    SizeType curBlockPos = ((incRows + written) & rowsInBlock);
                    SizeType toWrite = min(rowsInBlock + 1 - curBlockPos, num - written);
                    if (pData)
                    {
                        for (int i = 0; i < toWrite; i++)
                        {
                            std::memcpy((*incBlocks)[curBlockIdx]->data + ((size_t)curBlockPos + i) * cols + colStart,
                                        pData + ((size_t)written + i) * mycols, mycols * sizeof(T));
                        }
                    }
                    written += toWrite;
                }
                incRows += written;
                return ErrorCode::Success;
            }

            ErrorCode ReserveRows(SizeType target)
            {
                if (target < 0 || target > maxRows || !incBlocks)
                    return ErrorCode::MemoryOverFlow;
                if (colStart != 0)
                    return ErrorCode::Success;
                const auto additional = (std::max<int64_t>)(0, static_cast<int64_t>(target) - rows);
                const auto blocks = (additional + rowsInBlock) >> rowsInBlockEx;
                try
                {
                    if (static_cast<size_t>(blocks) > incBlocks->capacity())
                        incBlocks->reserve(static_cast<size_t>(blocks));
                    while (incBlocks->size() < static_cast<size_t>(blocks))
                    {
                        auto block = DatasetBuffer::Allocate(m_account, (static_cast<size_t>(rowsInBlock) + 1) * cols);
                        std::memset(block->data, defaultValue, (static_cast<size_t>(rowsInBlock) + 1) * cols);
                        incBlocks->push_back(std::move(block));
                    }
                }
                catch (const std::bad_alloc &)
                {
                    return ErrorCode::MemoryOverFlow;
                }
                return ErrorCode::Success;
            }

            ErrorCode Save(std::shared_ptr<Helper::DiskIO> p_out) const
            {
                SizeType CR = R();
                IOBINARY(p_out, WriteBinary, sizeof(SizeType), (char*)&CR);
                IOBINARY(p_out, WriteBinary, sizeof(DimensionType), (char*)&mycols);
                for (SizeType i = 0; i < CR; i++) {
                    IOBINARY(p_out, WriteBinary, sizeof(T) * mycols, (char*)At(i));
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Save %s (%d,%d) Finish!\n", name.c_str(), CR, mycols);
                return ErrorCode::Success;
            }

            ErrorCode Save(std::string sDataPointsFileName) const
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Save %s To %s\n", name.c_str(), sDataPointsFileName.c_str());
                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(sDataPointsFileName.c_str(), std::ios::binary | std::ios::out)) return ErrorCode::FailedCreateFile;
                return Save(ptr);
            }

            ErrorCode Load(std::shared_ptr<Helper::DiskIO> pInput, SizeType blockSize, SizeType capacity)
            {
                SizeType r;
                DimensionType c;
                IOBINARY(pInput, ReadBinary, sizeof(SizeType), (char*)&(r));
                IOBINARY(pInput, ReadBinary, sizeof(DimensionType), (char*)(&c));

                if (data == nullptr || r != rows + incRows) Initialize(r, c, blockSize, capacity);
                
                for (SizeType i = 0; i < r; i++) {
                    IOBINARY(pInput, ReadBinary, sizeof(T) * mycols, (char*)At(i));
                }
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load %s (%d,%d) Finish!\n", name.c_str(), r, c);
                return ErrorCode::Success;
            }

            ErrorCode Load(std::string sDataPointsFileName, SizeType blockSize, SizeType capacity)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load %s From %s\n", name.c_str(), sDataPointsFileName.c_str());
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

                Initialize(R, C, blockSize, capacity, (char*)pDataPointsMemFile);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load %s (%d,%d) Finish!\n", name.c_str(), R, C);
                return ErrorCode::Success;
            }

            ErrorCode Refine(const std::vector<SizeType>& indices, COMMON::Dataset<T>& dataset) const
            {
                SizeType newrows = (SizeType)(indices.size());
                if (dataset.data == nullptr) dataset.Initialize(newrows, mycols, rowsInBlock + 1, static_cast<SizeType>(incBlocks->capacity() * (rowsInBlock + 1)));

                for (SizeType i = 0; i < newrows; i++) {
                    std::memcpy((void*)dataset.At(i), (void*)At(indices[i]), sizeof(T) * mycols);
                }
                return ErrorCode::Success;
            }

            virtual ErrorCode Refine(const std::vector<SizeType>& indices, std::shared_ptr<Helper::DiskIO> output) const
            {
                SizeType newrows = (SizeType)(indices.size());
                IOBINARY(output, WriteBinary, sizeof(SizeType), (char*)&newrows);
                IOBINARY(output, WriteBinary, sizeof(DimensionType), (char*)&mycols);

                for (SizeType i = 0; i < newrows; i++) {
                    IOBINARY(output, WriteBinary, sizeof(T) * mycols, (char*)At(indices[i]));
                }
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Save Refine %s (%d,%d) Finish!\n", name.c_str(), newrows, C());
                return ErrorCode::Success;
            }

            virtual ErrorCode Refine(const std::vector<SizeType>& indices, std::string sDataPointsFileName) const
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Save Refine %s To %s\n", name.c_str(), sDataPointsFileName.c_str());
                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(sDataPointsFileName.c_str(), std::ios::binary | std::ios::out)) return ErrorCode::FailedCreateFile;
                return Refine(indices, ptr);
            }
    };

        template <typename V, typename VV>
        class SizeTypeConcurrentMap {
        private:
            std::mutex m_mutex;
            std::atomic<SizeType> m_deleted;
        protected:
            Dataset<V> m_data;
            V m_default = (V)(0xffffffffffffffff);

        public:
            class Iterator {
            private:
                const SizeTypeConcurrentMap<V, VV>* map_ptr;
                std::pair<SizeType, V> node_ptr;

            public:
                using iterator_category = std::forward_iterator_tag;
                using difference_type = std::ptrdiff_t;
                using value_type = std::pair<SizeType, V>;
                using pointer = value_type*;
                using reference = value_type&;

                Iterator(const SizeTypeConcurrentMap<V, VV>* ptr, SizeType idx) : map_ptr(ptr), node_ptr(std::pair<SizeType, V>{idx, map_ptr->m_default}) 
                {
                    if (idx < map_ptr->m_data.R()) node_ptr.second = *(map_ptr->m_data.At(idx));
                }
                reference operator*() { return node_ptr; }
                pointer operator->() { return &(node_ptr); }

                Iterator& operator++() {
                    if (node_ptr.first >= map_ptr->m_data.R()) return *this;

                    SizeType next = node_ptr.first + 1;
                    while (next < map_ptr->m_data.R() && *(map_ptr->m_data.At(next)) == map_ptr->m_default) next++;

                    node_ptr.first = next;
                    node_ptr.second = next < map_ptr->m_data.R()
                        ? *(map_ptr->m_data.At(next))
                        : map_ptr->m_default;
                    return *this;
                }

                Iterator operator++(int) {
                    Iterator tmp = *this;
                    ++(*this);
                    return tmp;
                }

                bool operator==(const Iterator& other) const {
                    return node_ptr.first == other.node_ptr.first;
                }

                bool operator!=(const Iterator& other) const {
                    return node_ptr.first != other.node_ptr.first;
                }
            };

            SizeTypeConcurrentMap(V _defaultfill = (V)0xffffffffffffffff) {
                m_data.SetName("SizeTypeConcurrentMap");
                m_data.SetDefaultValue((uint8_t)_defaultfill);
                m_data.Initialize(1024 * 1024, 1, 1024 * 1024, MaxSize);
                m_deleted.store(m_data.R());
                m_default = _defaultfill;
            }

            Iterator find(const SizeType& k) const
            {
                if (k >= 0 && k < m_data.R() && *(m_data[k]) != m_default)
                    return Iterator(this, k);
                return Iterator(this, m_data.R());
            }

            Iterator begin() const noexcept
            {
                if (m_deleted.load() < m_data.R()) {
                    for (SizeType i = 0; i < m_data.R(); i++) {
                        if (*(m_data[i]) != m_default) {
                            return Iterator(this, i);
                        }
                    }
                }
                return Iterator(this, m_data.R());
            }

            Iterator end() const noexcept
            {
                return Iterator(this, m_data.R());
            }

            std::pair<Iterator, bool> insert_or_assign(SizeType&& k, V&& v) 
            {
                if (k < 0) return std::make_pair(Iterator(this, m_data.R()), false);
                if (k >= m_data.R()) {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    if (k >= m_data.R()) {
                        SizeType oldR = m_data.R();
                        if (m_data.AddBatch(k - oldR + 10000) != ErrorCode::Success) {
                            return std::make_pair(Iterator(this, m_data.R()), false);
                        }
                        m_deleted.fetch_add(k - oldR + 10000);
                    }
                }
                V oldvalue = (V)InterlockedExchange((VV*)(m_data[k]), (VV)v);
                if (oldvalue == m_default && v != m_default) m_deleted.fetch_sub(1);
                return std::make_pair(Iterator(this, k), true);
            }

            bool assign_if_equal(SizeType&& k, const V& expected, V&& desired)
            {
                if (k < 0 || k >= m_data.R()) return false;
                V oldvalue = (V)InterlockedCompareExchange((VV*)(m_data[k]), (VV)desired, (VV)expected);
                if (oldvalue != expected) return false;
                return true;
            }

            size_t erase(const SizeType& k)
            {
                if (k < 0 || k >= m_data.R()) return 0;
                V oldvalue = (V)InterlockedExchange((VV*)(m_data[k]), (VV)m_default);
                if (oldvalue == m_default) return 0;
                m_deleted.fetch_add(1);
                return 1;
            }

            void clear()
            {
                for (SizeType i = 0; i < m_data.R(); i++) {
                    *m_data[i] = m_default;
                }
                m_deleted.store(m_data.R());
            }
            
            size_t size() const
            {
                return m_data.R() - m_deleted.load();
            }

            friend class Iterator;
        };

        template <typename T>
        ErrorCode LoadOptDatasets(std::shared_ptr<Helper::DiskIO> pVectorsInput, std::shared_ptr<Helper::DiskIO> pGraphInput,
            Dataset<T>& pVectors, Dataset<SizeType>& pGraph, DimensionType pNeighborhoodSize,
            SizeType blockSize, SizeType capacity) {
            SizeType VR, GR;
            DimensionType VC, GC;
            IOBINARY(pVectorsInput, ReadBinary, sizeof(SizeType), (char*)&VR);
            IOBINARY(pVectorsInput, ReadBinary, sizeof(DimensionType), (char*)&VC);
            DimensionType totalC = ALIGN_ROUND(sizeof(T) * VC + sizeof(SizeType) * pNeighborhoodSize);

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "OPT TotalC: %d\n", totalC);
            auto status = pVectors.Initialize(VR, VC, blockSize, capacity, nullptr, true, nullptr, 0, totalC);
            if (status != ErrorCode::Success)
                return status;
            pVectors.SetName("Opt" + pVectors.Name());
            for (SizeType i = 0; i < VR; i++) {
                IOBINARY(pVectorsInput, ReadBinary, sizeof(T) * VC, (char*)(pVectors.At(i)));
            }

            IOBINARY(pGraphInput, ReadBinary, sizeof(SizeType), (char*)&GR);
            IOBINARY(pGraphInput, ReadBinary, sizeof(DimensionType), (char*)&GC);
            if (GR != VR || GC != pNeighborhoodSize) return ErrorCode::DiskIOFail;

            status = pGraph.InitializeSharedView(pVectors, GC, sizeof(T) * VC);
            if (status != ErrorCode::Success)
                return status;
            pGraph.SetName("Opt" + pGraph.Name());
            for (SizeType i = 0; i < VR; i++) {
                IOBINARY(pGraphInput, ReadBinary, sizeof(SizeType) * GC, (char*)(pGraph.At(i)));
            }

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load %s (%d,%d) Finish!\n", pVectors.Name().c_str(), pVectors.R(), pVectors.C());
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load %s (%d,%d) Finish!\n", pGraph.Name().c_str(), pGraph.R(), pGraph.C());

            return ErrorCode::Success;
        }
    }
}

#endif // _SPTAG_COMMON_DATASET_H_
