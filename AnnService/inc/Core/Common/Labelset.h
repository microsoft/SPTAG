// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_LABELSET_H_
#define _SPTAG_COMMON_LABELSET_H_

#include <atomic>
#include <shared_mutex>
#include "Dataset.h"

namespace SPTAG
{
    namespace COMMON
    {
        class Labelset
        {
        private:
            std::atomic<SizeType> m_deleted;
            Dataset<std::int8_t> m_data;
            std::unique_ptr<std::shared_timed_mutex> m_lock;

        public:
            Labelset() 
            {
                m_deleted = 0;
                m_data.SetName("DeleteID");
                m_lock.reset(new std::shared_timed_mutex);
            }

            void Initialize(SizeType capacity)
            {
                m_data.Initialize(capacity, 1);
            }

            inline size_t Size() const { return m_deleted.load(); }

            inline bool Contains(const SizeType& key) const
            {
                return *m_data[key] == 1;
            }

            inline void Insert(const SizeType& key)
            {
                if (*m_data[key] != 1)
                {
                    std::shared_lock<std::shared_timed_mutex> lock(*m_lock);
                    m_deleted++;
                    *m_data[key] = 1;
                }
            }

            inline bool Save(std::ostream& output)
            {
                SizeType deleted = m_deleted.load();
                output.write((char*)&deleted, sizeof(SizeType));
                return m_data.Save(output);
            }

            inline bool Save(std::string filename)
            {
                std::cout << "Save " << m_data.Name() << " To " << filename << std::endl;
                std::ofstream output(filename, std::ios::binary);
                if (!output.is_open()) return false;
                Save(output);
                output.close();
                return true;
            }

            inline bool Load(std::string filename)
            {
                std::cout << "Load " << m_data.Name() << " From " << filename << std::endl;
                std::ifstream input(filename, std::ios::binary);
                if (!input.is_open()) return false;          
                SizeType deleted;
                input.read((char*)&deleted, sizeof(SizeType));
                m_deleted = deleted;
                m_data.Load(input);
                input.close();
                return true;
            }

            inline bool Load(char* pmemoryFile) 
            {
                m_deleted = *((SizeType*)pmemoryFile);
                return m_data.Load(pmemoryFile + sizeof(SizeType));
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
            
            inline std::shared_timed_mutex& Lock()
            {
                return *m_lock;
            }
        };
    }
}

#endif // _SPTAG_COMMON_LABELSET_H_
