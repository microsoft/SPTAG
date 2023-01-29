// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_POSTINGSIZERECORD_H_
#define _SPTAG_COMMON_POSTINGSIZERECORD_H_

#include <atomic>
#include "Dataset.h"

namespace SPTAG
{
    namespace COMMON
    {
        class PostingSizeRecord
        {
        private:
            Dataset<int> m_data;
            
        public:
            PostingSizeRecord() 
            {
                m_data.SetName("PostingSizeRecord");
            }

            void Initialize(SizeType size, SizeType blockSize, SizeType capacity)
            {
                m_data.Initialize(size, 1, blockSize, capacity);
            }

            inline int GetSize(const SizeType& headID)
            {
                return *m_data[headID];
            }

            inline bool UpdateSize(const SizeType& headID, int newSize)
            {
                while (true) {
                    int oldSize = GetSize(headID);
                    if (InterlockedCompareExchange((unsigned*)m_data[headID], (unsigned)newSize, (unsigned)oldSize) == oldSize) {
                        return true;
                    }
                }
            }

            inline bool IncSize(const SizeType& headID, int appendNum)
            {
                while (true) {
                    int oldSize = GetSize(headID);
                    int newSize = oldSize + appendNum;
                    if (InterlockedCompareExchange((unsigned*)m_data[headID], (unsigned)newSize, (unsigned)oldSize) == oldSize) {
                        return true;
                    }
                }
            }
            
            inline SizeType GetPostingNum()
            {
                return m_data.R();
            }

            inline ErrorCode Save(std::shared_ptr<Helper::DiskIO> output)
            {
                return m_data.Save(output);
            }

            inline ErrorCode Save(const std::string& filename)
            {
                LOG(Helper::LogLevel::LL_Info, "Save %s To %s\n", m_data.Name().c_str(), filename.c_str());
                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(filename.c_str(), std::ios::binary | std::ios::out)) return ErrorCode::FailedCreateFile;
                return Save(ptr);
            }

            inline ErrorCode Load(std::shared_ptr<Helper::DiskIO> input, SizeType blockSize, SizeType capacity)
            {
                return m_data.Load(input, blockSize, capacity);
            }

            inline ErrorCode Load(const std::string& filename, SizeType blockSize, SizeType capacity)
            {
                LOG(Helper::LogLevel::LL_Info, "Load %s From %s\n", m_data.Name().c_str(), filename.c_str());
                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(filename.c_str(), std::ios::binary | std::ios::in)) return ErrorCode::FailedOpenFile;
                return Load(ptr, blockSize, capacity);
            }

            inline ErrorCode Load(char* pmemoryFile, SizeType blockSize, SizeType capacity)
            {
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
        };
    }
}

#endif // _SPTAG_COMMON_LABELSET_H_
