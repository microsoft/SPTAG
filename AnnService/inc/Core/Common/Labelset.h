// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_LABELSET_H_
#define _SPTAG_COMMON_LABELSET_H_

#include <atomic>
#include "Dataset.h"

namespace SPTAG
{
    namespace COMMON
    {
        class Labelset
        {
        private:
            std::atomic<SizeType> m_inserted;
            Dataset<std::int8_t> m_data;
            
        public:
            Labelset() 
            {
                m_inserted = 0;
                m_data.SetName("DeleteID");
            }

            void Initialize(SizeType size, SizeType blockSize, SizeType capacity)
            {
                m_data.Initialize(size, 1, blockSize, capacity);
            }

            inline size_t Count() const { return m_inserted.load(); }

            inline bool Contains(const SizeType& key) const
            {
                return *m_data[key] == 1;
            }

            inline bool Insert(const SizeType& key)
            {
                char oldvalue = InterlockedExchange8((char*)m_data[key], 1);
                if (oldvalue == 1) return false;
                m_inserted++;
                return true;
            }

            inline ErrorCode Save(std::shared_ptr<Helper::DiskIO> output)
            {
                SizeType deleted = m_inserted.load();
                IOBINARY(output, WriteBinary, sizeof(SizeType), (char*)&deleted);
                return m_data.Save(output);
            }

            inline ErrorCode Save(std::string filename)
            {
                LOG(Helper::LogLevel::LL_Info, "Save %s To %s\n", m_data.Name().c_str(), filename.c_str());
                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(filename.c_str(), std::ios::binary | std::ios::out)) return ErrorCode::FailedCreateFile;
                return Save(ptr);
            }

            inline ErrorCode Load(std::shared_ptr<Helper::DiskIO> input, SizeType blockSize, SizeType capacity)
            {
                SizeType deleted;
                IOBINARY(input, ReadBinary, sizeof(SizeType), (char*)&deleted);
                m_inserted = deleted;
                return m_data.Load(input, blockSize, capacity);
            }

            inline ErrorCode Load(std::string filename, SizeType blockSize, SizeType capacity)
            {
                LOG(Helper::LogLevel::LL_Info, "Load %s From %s\n", m_data.Name().c_str(), filename.c_str());
                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(filename.c_str(), std::ios::binary | std::ios::in)) return ErrorCode::FailedOpenFile;
                return Load(ptr, blockSize, capacity);
            }

            inline ErrorCode Load(char* pmemoryFile, SizeType blockSize, SizeType capacity)
            {
                m_inserted = *((SizeType*)pmemoryFile);
                return m_data.Load(pmemoryFile + sizeof(SizeType), blockSize, capacity);
            }

            inline ErrorCode AddBatch(SizeType num)
            {
                return m_data.AddBatch(num);
            }

            inline std::uint64_t BufferSize() const 
            {
                return m_data.BufferSize() + sizeof(SizeType);
            }

            inline void SetR(SizeType num)
            {
                m_data.SetR(num);
            }

            inline SizeType R() const
            {
                return m_data.R();
            }
        };
    }
}

#endif // _SPTAG_COMMON_LABELSET_H_
