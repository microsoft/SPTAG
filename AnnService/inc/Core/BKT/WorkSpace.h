#ifndef _SPTAG_BKT_WORKSPACE_H_
#define _SPTAG_BKT_WORKSPACE_H_

#include "CommonUtils.h"
#include "Heap.h"

namespace SPTAG
{
    namespace BKT
    {
        // node type in the priority queue
        struct HeapCell
        {
            int node;
            float distance;

            HeapCell(int _node = -1, float _distance = 0) : node(_node), distance(_distance) {}

            inline bool operator < (const HeapCell& rhs)
            {
                return distance < rhs.distance;
            }

            inline bool operator > (const HeapCell& rhs)
            {
                return distance > rhs.distance;
            }
        };

        class OptHashPosVector
        {
        protected:
            // Max loop number in one hash block.
            static const int m_maxLoop = 8;

            // Max pool size.
            static const int m_poolSize = 8191;

            // Could we use the second hash block.
            bool m_secondHash;

            // Record 2 hash tables.
            // [0~m_poolSize + 1) is the first block.
            // [m_poolSize + 1, 2*(m_poolSize + 1)) is the second block;
            int m_hashTable[(m_poolSize + 1) * 2];


            inline unsigned hash_func2(int idx, int loop)
            {
                return ((unsigned)idx + loop) & m_poolSize;
            }


            inline unsigned hash_func(unsigned idx)
            {
                return ((unsigned)(idx * 99991) + _rotl(idx, 2) + 101) & m_poolSize;
            }

        public:
            OptHashPosVector() {}

            ~OptHashPosVector() {}


            void Init(int size)
            {
                m_secondHash = true;
                clear();
            }

            void clear()
            {
                if (!m_secondHash)
                {
                    // Clear first block.
                    memset(&m_hashTable[0], 0, sizeof(int)*(m_poolSize + 1));
                }
                else
                {
                    // Clear all blocks.
                    memset(&m_hashTable[0], 0, 2 * sizeof(int) * (m_poolSize + 1));
                    m_secondHash = false;
                }
            }


            inline bool CheckAndSet(int idx)
            {
                // Inner Index is begin from 1
                return _CheckAndSet(&m_hashTable[0], idx + 1) == 0;
            }


            inline int _CheckAndSet(int* hashTable, int idx)
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
        };

        template <typename T>
        class CountVector
        {
            size_t m_bytes;
            T* m_data;
            T m_count;
            T MAX;

        public:
            void Init(int size)
            {
                m_bytes = sizeof(T) * size;
                m_data = new T[size];
                m_count = 0;
                MAX = ((std::numeric_limits<T>::max)());
                memset(m_data, 0, m_bytes);
            }

            CountVector() :m_data(nullptr) {}
            CountVector(int size) { Init(size); }
            ~CountVector() { if (m_data != nullptr) delete[] m_data; }

            inline void clear()
            {
                if (m_count == MAX)
                {
                    memset(m_data, 0, m_bytes);
                    m_count = 1;
                }
                else
                {
                    m_count++;
                }
            }

            inline bool CheckAndSet(int idx)
            {
                if (m_data[idx] == m_count) return true;
                m_data[idx] = m_count;
                return false;
            }
        };

        // Variables for each single NN search
        struct WorkSpace
        {
            void Initialize(int maxCheck, int dataSize)
            {
                nodeCheckStatus.Init(dataSize);
                m_BKTQueue.Resize(maxCheck * 10);
                m_NGQueue.Resize(maxCheck * 30);

                m_iNumberOfCheckedLeaves = 0;
                m_iContinuousLimit = maxCheck / 64;
                m_iMaxCheck = maxCheck;
                m_iNumOfContinuousNoBetterPropagation = 0;
            }

            void Reset(int maxCheck)
            {
                nodeCheckStatus.clear();
                m_BKTQueue.clear();
                m_NGQueue.clear();

                m_iNumberOfCheckedLeaves = 0;
                m_iContinuousLimit = maxCheck / 64;
                m_iMaxCheck = maxCheck;
                m_iNumOfContinuousNoBetterPropagation = 0;
            }

            inline bool CheckAndSet(int idx)
            {
                return nodeCheckStatus.CheckAndSet(idx);
            }

            CountVector<unsigned short> nodeCheckStatus;
            //OptHashPosVector nodeCheckStatus;

            // counter for dynamic pivoting
            int m_iNumOfContinuousNoBetterPropagation;
            int m_iContinuousLimit;
            int m_iNumberOfCheckedLeaves;
            int m_iMaxCheck;

            // Prioriy queue used for neighborhood graph
            Heap<HeapCell> m_NGQueue;

            // Priority queue Used for BKT-Tree
            Heap<HeapCell> m_BKTQueue;
        };
    }
}

#endif // _SPTAG_BKT_WORKSPACE_H_
