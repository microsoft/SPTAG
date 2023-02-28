// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_VERSIONLABEL_H_
#define _SPTAG_COMMON_VERSIONLABEL_H_

#include <atomic>
#include "Dataset.h"

namespace SPTAG
{
    namespace COMMON
    {
        class VersionLabel
        {
        private:
            std::atomic<SizeType> m_deleted;
            Dataset<std::uint8_t> m_data;
            
        public:
            VersionLabel() 
            {
                m_deleted = 0;
                m_data.SetName("versionLabelID");
            }

            void Initialize(SizeType size, SizeType blockSize, SizeType capacity)
            {
                m_data.Initialize(size, 1, blockSize, capacity);
            }

            inline size_t Count() const { return m_data.R() - m_deleted.load(); }

            inline size_t GetDeleteCount() const { return m_deleted.load();}

            inline bool Deleted(const SizeType& key) const
            {
                return *m_data[key] == 0xfe;
            }

            inline bool Delete(const SizeType& key)
            {
                uint8_t oldvalue = (uint8_t)InterlockedExchange8((char*)(m_data[key]), (char)0xfe);
                if (oldvalue == 0xfe) return false;
                m_deleted++;
                return true;
            }

            inline uint8_t GetVersion(const SizeType& key)
            {
                return *m_data[key];
            }

            inline bool IncVersion(const SizeType& key, uint8_t* newVersion)
            {
                while (true) {
                    if (Deleted(key)) return false;
                    uint8_t oldVersion = GetVersion(key);
                    *newVersion = (oldVersion+1) & 0x7f;
                    if (((uint8_t)InterlockedCompareExchange((char*)m_data[key], (char)*newVersion, (char)oldVersion)) == oldVersion) {
                        return true;
                    }
                }
            }

            inline SizeType GetVectorNum()
            {
                return m_data.R();
            }

            inline ErrorCode Save(std::shared_ptr<Helper::DiskIO> output)
            {
                SizeType deleted = m_deleted.load();
                IOBINARY(output, WriteBinary, sizeof(SizeType), (char*)&deleted);
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
                SizeType deleted;
                IOBINARY(input, ReadBinary, sizeof(SizeType), (char*)&deleted);
                m_deleted = deleted;
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
                m_deleted = *((SizeType*)pmemoryFile);
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
