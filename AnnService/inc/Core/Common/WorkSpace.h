// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_WORKSPACE_H_
#define _SPTAG_COMMON_WORKSPACE_H_

#include "CommonUtils.h"
#include "Heap.h"

namespace SPTAG
{
    namespace COMMON
    {
        // node type in the priority queue
        struct HeapCell
        {
            SizeType node;
            float distance;

            HeapCell(SizeType _node = -1, float _distance = MaxDist) : node(_node), distance(_distance) {}

            inline bool operator < (const HeapCell& rhs) const
            {
                return distance < rhs.distance;
            }

            inline bool operator > (const HeapCell& rhs) const
            {
                return distance > rhs.distance;
            }
        };

        class OptHashPosVector
        {
        protected:
            // Max loop number in one hash block.
            static const int m_maxLoop = 8;

            // Could we use the second hash block.
            bool m_secondHash;

            int m_exp;

            // Max pool size.
            int m_poolSize;

            // Record 2 hash tables.
            // [0~m_poolSize + 1) is the first block.
            // [m_poolSize + 1, 2*(m_poolSize + 1)) is the second block;
            std::unique_ptr<SizeType[]> m_hashTable;


            inline unsigned hash_func2(unsigned idx, int poolSize, int loop)
            {
                return (idx + loop) & poolSize;
            }


            inline unsigned hash_func(unsigned idx, int poolSize)
            {
                return ((unsigned)(idx * 99991) + _rotl(idx, 2) + 101) & poolSize;
            }

        public:
            OptHashPosVector() {}

            ~OptHashPosVector() {}


            void Init(SizeType size, int exp)
            {
                int ex = 0;
                while (size != 0) {
                    ex++;
                    size >>= 1;
                }
                m_secondHash = true;
                m_exp = exp;
                m_poolSize = (1 << (ex + exp)) - 1;
                m_hashTable.reset(new SizeType[(m_poolSize + 1) * 2]);
                clear();
            }

            void clear()
            {
                if (!m_secondHash)
                {
                    // Clear first block.
                    memset(m_hashTable.get(), 0, sizeof(SizeType) * (m_poolSize + 1));
                }
                else
                {
                    // Clear all blocks.
                    m_secondHash = false;
                    memset(m_hashTable.get(), 0, 2 * sizeof(SizeType) * (m_poolSize + 1));
                }
            }

            inline int HashTableExponent() const { return m_exp; }

            inline bool CheckAndSet(SizeType idx)
            {
                // Inner Index is begin from 1
                return _CheckAndSet(m_hashTable.get(), m_poolSize, true, idx + 1) == 0;
            }

            inline void DoubleSize()
            {
                int new_poolSize = ((m_poolSize + 1) << 1) - 1; 
                SizeType* new_hashTable = new SizeType[(new_poolSize + 1) * 2];
                memset(new_hashTable, 0, sizeof(SizeType) * (new_poolSize + 1) * 2);

                m_secondHash = false;
                for (int i = 0; i <= new_poolSize; i++)
                    if (m_hashTable[i]) _CheckAndSet(new_hashTable, new_poolSize, true, m_hashTable[i]);

                m_exp++;
                m_poolSize = new_poolSize;
                m_hashTable.reset(new_hashTable);
            }

            inline int _CheckAndSet(SizeType* hashTable, int poolSize, bool isFirstTable, SizeType idx)
            {
                unsigned index = hash_func((unsigned)idx, poolSize);
                for (int loop = 0; loop < m_maxLoop; ++loop)
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
                    index = hash_func2(index, poolSize, loop);
                }

                if (isFirstTable)
                {
                    // Use second hash block.
                    m_secondHash = true;
                    return _CheckAndSet(hashTable + poolSize + 1, poolSize, false, idx);
                }

                DoubleSize();
                LOG(Helper::LogLevel::LL_Error, "Hash table is full! Set HashTableExponent to larger value (default is 2). NewHashTableExponent=%d NewPoolSize=%d\n", m_exp, m_poolSize);
                return _CheckAndSet(m_hashTable.get(), m_poolSize, true, idx);
            }
        };
/*
        class DistPriorityQueue {
            std::unique_ptr<float[]> m_data;
            int m_size;
            int m_count;
        public:
            DistPriorityQueue() {}

            void Resize(int size_) {
                m_data.reset(new float[size_ + 1]);
                for (int i = 0; i <= size_; i++) m_data[i] = MaxDist;
                m_size = size_;
                m_count = size_;
            }
            void clear(int count_) {
                if (count_ > m_size) {
                    m_data.reset(new float[count_ + 1]);
                    m_size = count_;
                }
                for (int i = 0; i <= count_; i++) m_data[i] = MaxDist;
                m_count = count_;
            }
            bool insert(float dist) {
                if (dist > m_data[1]) return false;

                m_data[1] = dist;
                int parent = 1, next = 2;
                while (next < m_count) {
                    if (m_data[next] < m_data[next + 1]) next++;
                    if (m_data[parent] < m_data[next]) {
                        std::swap(m_data[next], m_data[parent]);
                        parent = next;
                        next <<= 1;
                    }
                    else break;
                }
                if (next == m_count && m_data[parent] < m_data[next]) std::swap(m_data[parent], m_data[next]);
                return true;
            }
            float worst() {
                return m_data[1];
            }
        };
*/
        // Variables for each single NN search
        struct WorkSpace
        {
            void Initialize(int maxCheck, SizeType dataSize, int hashexp)
            {
                nodeCheckStatus.Init(maxCheck, hashexp);
                m_SPTQueue.Resize(maxCheck * 10);
                m_NGQueue.Resize(maxCheck * 30);
                //m_Results.Resize(maxCheck / 16);

                m_iNumberOfTreeCheckedLeaves = 0;
                m_iNumberOfCheckedLeaves = 0;
                m_iContinuousLimit = maxCheck / 64;
                m_iMaxCheck = maxCheck;
                m_iNumOfContinuousNoBetterPropagation = 0;
            }

            void Reset(int maxCheck)
            {
                nodeCheckStatus.clear();
                m_SPTQueue.clear();
                m_NGQueue.clear();
                //m_Results.clear(maxCheck / 16);

                m_iNumOfContinuousNoBetterPropagation = 0;
                m_iContinuousLimit = maxCheck / 64;
                m_iNumberOfTreeCheckedLeaves = 0;
                m_iNumberOfCheckedLeaves = 0;
                m_iMaxCheck = maxCheck;
            }

            inline bool CheckAndSet(SizeType idx)
            {
                return nodeCheckStatus.CheckAndSet(idx);
            }

            inline int HashTableExponent() const 
            { 
                return nodeCheckStatus.HashTableExponent(); 
            }

            OptHashPosVector nodeCheckStatus;

            // counter for dynamic pivoting
            int m_iNumOfContinuousNoBetterPropagation;
            int m_iContinuousLimit;
            int m_iNumberOfTreeCheckedLeaves;
            int m_iNumberOfCheckedLeaves;
            int m_iMaxCheck;

            // Prioriy queue used for neighborhood graph
            Heap<HeapCell> m_NGQueue;

            // Priority queue Used for Tree
            Heap<HeapCell> m_SPTQueue;

            //DistPriorityQueue m_Results;
        };
    }
}

#endif // _SPTAG_COMMON_WORKSPACE_H_
