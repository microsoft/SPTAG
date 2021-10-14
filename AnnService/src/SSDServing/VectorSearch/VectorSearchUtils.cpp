#include <cstring>
#include <limits>
#include <stdlib.h>
#include <inc/Core/Common.h>
#include <inc/SSDServing/VectorSearch/VectorSearchUtils.h>

namespace SPTAG {
    namespace SSDServing {
        namespace VectorSearch {

            HashBasedDeduper::HashBasedDeduper()
            {
                Init();
            }

            HashBasedDeduper::~HashBasedDeduper() {}

            void HashBasedDeduper::Init(int stub)
            {
                m_secondHash = true;
                Clear();
            }

            void HashBasedDeduper::Clear()
            {
                if (!m_secondHash)
                {
                    // Clear first block.
                    memset(&m_hashTable[0], 0, sizeof(int) * (m_poolSize + 1));
                }
                else
                {
                    // Clear all blocks.
                    memset(&m_hashTable[0], 0, 2 * sizeof(int) * (m_poolSize + 1));
                    m_secondHash = false;
                }
            }

            bool HashBasedDeduper::CheckAndSet(int idx)
            {
                // Inner Index is begin from 1
                return _CheckAndSet(&m_hashTable[0], idx + 1) == 0;
            }

            inline unsigned HashBasedDeduper::hash_func2(int idx, int loop)
            {
                return ((unsigned)idx + loop) & m_poolSize;
            }


            inline unsigned HashBasedDeduper::hash_func(unsigned idx)
            {
                return ((unsigned)(idx * 99991) + _rotl(idx, 2) + 101) & m_poolSize;
            }

            inline int HashBasedDeduper::_CheckAndSet(int* hashTable, int idx)
            {
                unsigned index, loop;

                // Get first hash position.
                index = hash_func(idx);
                for (loop = 0; loop < m_maxLoop; ++loop)
                {
                    if (!hashTable[index])
                    {
                        // index first match and record it.
                        hashTable[index] = idx;
                        return 1;
                    }
                    if (hashTable[index] == idx)
                    {
                        // Hit this item in hash table.
                        return 0;
                    }
                    // Get next hash position.
                    index = hash_func2(index, loop);
                }

                if (hashTable == &m_hashTable[0])
                {
                    // Use second hash block.
                    m_secondHash = true;
                    return _CheckAndSet(&m_hashTable[m_poolSize + 1], idx);
                }

                // Do not include this item.
                return -1;
            }
        }
    }
}