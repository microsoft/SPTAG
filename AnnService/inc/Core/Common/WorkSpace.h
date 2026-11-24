// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_WORKSPACE_H_
#define _SPTAG_COMMON_WORKSPACE_H_

#include "CommonUtils.h"
#include "Heap.h"
#include "ResidentMemory.h"
#include "WorkerSizeClassPool.h"
#include "inc/Core/SearchResult.h"

#include <algorithm>
#include <atomic>
#include <functional>
#include <limits>
#include <mutex>
#include <stdarg.h>
#include <unordered_map>

namespace SPTAG
{
    namespace COMMON
    {
        template <typename WorkSpaceType>
        class IWorkSpaceFactory
        {
        public:
            using Ptr = std::unique_ptr<WorkSpaceType>;

            virtual ~IWorkSpaceFactory() = default;
            virtual std::unique_ptr<WorkSpaceType> GetWorkSpace() = 0;
            virtual void ReturnWorkSpace(std::unique_ptr<WorkSpaceType> ws) = 0;
        };

        template <typename WorkSpaceType>
        class ThreadLocalWorkSpaceFactory : public IWorkSpaceFactory<WorkSpaceType>
        {
        public:
            static thread_local std::unique_ptr<WorkSpaceType> m_workspace;

            virtual std::unique_ptr< WorkSpaceType> GetWorkSpace() override
            {
                return std::move(m_workspace);
            }

            virtual void ReturnWorkSpace(std::unique_ptr<WorkSpaceType> ws) override
            {
                m_workspace = std::move(ws);
            }
        };

	template <typename WorkSpaceType>
        class SharedPoolWorkSpaceFactory : public IWorkSpaceFactory<WorkSpaceType> {
        public:
            virtual std::unique_ptr<WorkSpaceType> GetWorkSpace() override
	    {
                std::unique_ptr<WorkSpaceType>  ws;
                std::lock_guard<std::mutex> lock(m_mutex);
                if (!m_pool.empty()) {
                    ws = std::move(m_pool.back());
                    m_pool.pop_back();
                }
		return ws;
	    }

            void ReturnWorkSpace(std::unique_ptr<WorkSpaceType> ws) override {
                if (ws) {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    m_pool.emplace_back(std::move(ws));
                }
            }

        private:
            std::mutex m_mutex;
            std::vector<std::unique_ptr<WorkSpaceType>> m_pool;
        };

        template <typename WorkSpaceType>
        class InstanceThreadLocalWorkSpaceFactory : public IWorkSpaceFactory<WorkSpaceType>
        {
            struct WorkerState
            {
                explicit WorkerState(uint64_t value) : id(value) {}
                const uint64_t id;
                std::atomic<bool> exited{false};
            };
            struct WorkerRegistration
            {
                explicit WorkerRegistration(uint64_t id) : state(std::make_shared<WorkerState>(id))
                {
                }
                WorkerRegistration(const WorkerRegistration &) = delete;
                WorkerRegistration &operator=(const WorkerRegistration &) = delete;
                ~WorkerRegistration()
                {
                    state->exited.store(true, std::memory_order_release);
                }
                std::shared_ptr<WorkerState> state;
            };
            struct Slot
            {
                std::unique_ptr<WorkSpaceType> workspace;
                std::atomic<bool> retired{false};
                uint64_t worker = 0;
                std::shared_ptr<WorkerState> workerLife;
            };
            struct CachedSlot
            {
                std::shared_ptr<Slot> owned;
                Slot *prepared = nullptr;
                Slot *Get() const
                {
                    return prepared ? prepared : owned.get();
                }
            };

            static uint64_t NextID(std::atomic<uint64_t> &source)
            {
                auto id = source.load(std::memory_order_relaxed);
                do
                {
                    if (id == (std::numeric_limits<uint64_t>::max)())
                        throw std::overflow_error("Workspace identity exhausted");
                } while (!source.compare_exchange_weak(id, id + 1, std::memory_order_relaxed));
                return id;
            }
            inline static std::atomic<uint64_t> s_nextID{1};
            inline static std::atomic<uint64_t> s_nextWorkerID{1};
            inline static thread_local const WorkerRegistration s_worker{NextID(s_nextWorkerID)};
            inline static thread_local std::unordered_map<uint64_t, CachedSlot> s_slots;
            const uint64_t m_id = NextID(s_nextID);
            std::mutex m_registration;
            std::vector<std::weak_ptr<Slot>> m_slots;
            const bool m_bounded = bool(ResidentMemory::CurrentAccount());
            ResidentVector<std::shared_ptr<Slot>> m_prepared;

            Slot *LocalSlot()
            {
                auto found = s_slots.find(m_id);
                if (found != s_slots.end())
                    return found->second.Get();
                for (auto it = s_slots.begin(); it != s_slots.end();)
                {
                    if (!it->second.prepared && it->second.owned->retired.load(std::memory_order_acquire))
                        it = s_slots.erase(it);
                    else
                        ++it;
                }
                // Bound idle buffers even when a worker visits more than the default tenant limit.
                if (s_slots.size() >= 64)
                    s_slots.erase(s_slots.begin());
                CachedSlot cached;
                if (m_bounded)
                {
                    std::lock_guard<std::mutex> lock(m_registration);
                    // The exit marker publishes the last workspace return. TLS owns
                    // no factory pointer, so it remains safe after index retirement.
                    for (auto &slot : m_prepared)
                        if (slot->worker && slot->workerLife->exited.load(std::memory_order_acquire))
                        {
                            slot->workspace.reset();
                            slot->worker = 0;
                            slot->workerLife.reset();
                        }
                    for (auto &slot : m_prepared)
                        if (slot->worker == s_worker.state->id)
                        {
                            cached.prepared = slot.get();
                            break;
                        }
                    if (!cached.prepared)
                        for (auto &slot : m_prepared)
                            if (!slot->worker)
                            {
                                slot->worker = s_worker.state->id;
                                slot->workerLife = s_worker.state;
                                cached.prepared = slot.get();
                                break;
                            }
                    if (!cached.prepared)
                        return nullptr;
                }
                else
                {
                    cached.owned = std::make_shared<Slot>();
                    std::lock_guard<std::mutex> lock(m_registration);
                    m_slots.erase(std::remove_if(m_slots.begin(), m_slots.end(),
                                                 [](const auto &entry) { return entry.expired(); }),
                                  m_slots.end());
                    m_slots.emplace_back(cached.owned);
                }
                // Prepared raw entries are only dereferenced by their live,
                // uniquely identified factory, whose users are epoch-pinned.
                // They never retain retired index metadata on a worker thread.
                return s_slots.emplace(m_id, std::move(cached)).first->second.Get();
            }

          public:
            InstanceThreadLocalWorkSpaceFactory()
            {
                if (m_bounded)
                {
                    m_prepared.reserve(64);
                    for (size_t i = 0; i < 64; ++i)
                        m_prepared.push_back(std::allocate_shared<Slot>(ResidentAllocator<Slot>()));
                }
            }
            ~InstanceThreadLocalWorkSpaceFactory() override
            {
                // The index owner drains users before destruction. TLS owns only the slot;
                // eviction releases its buffer now, without waiting for the worker to exit.
                for (auto &entry : m_slots)
                    if (auto slot = entry.lock())
                    {
                        slot->workspace.reset();
                        slot->retired.store(true, std::memory_order_release);
                    }
                for (auto &slot : m_prepared)
                {
                    slot->workspace.reset();
                    slot->retired.store(true, std::memory_order_release);
                }
            }

            std::unique_ptr<WorkSpaceType> GetWorkSpace() override
            {
                auto *slot = LocalSlot();
                return slot ? std::move(slot->workspace) : nullptr;
            }

            void ReturnWorkSpace(std::unique_ptr<WorkSpaceType> workspace) override
            {
                if (workspace)
                {
                    if constexpr (requires { workspace->ReleaseScratch(); })
                        workspace->ReleaseScratch();
                    if (auto *slot = LocalSlot())
                        slot->workspace = std::move(workspace);
                }
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
            std::uint64_t m_poolSize;

            // Record 2 hash tables.
            // [0~m_poolSize + 1) is the first block.
            // [m_poolSize + 1, 2*(m_poolSize + 1)) is the second block;
            std::unique_ptr<SizeType[]> m_hashTable;
            std::size_t m_capacity = 0;
            SizeType m_initialSize = 0;
            bool m_reuseStorage = false;

            inline std::uint64_t hash_func2(std::uint64_t idx, std::uint64_t poolSize, int loop) const
            {
                return (idx + loop) & poolSize;
            }


            inline std::uint64_t hash_func(std::uint64_t idx, std::uint64_t poolSize) const
            {
                return ((std::uint64_t)(idx * 99991) + _rotl64(idx, 2) + 101) & poolSize;
            }

        public:
            OptHashPosVector(): m_secondHash(false), m_exp(2), m_poolSize(8191) {}

            ~OptHashPosVector() { m_hashTable.reset(); }

            static std::size_t RequiredStorageBytes(SizeType size, int exp)
            {
                int bits = 0;
                while (size > 0)
                {
                    ++bits;
                    size >>= 1;
                }
                const int shift = bits + exp + 1;
                if (shift < 0 || shift >= std::numeric_limits<std::size_t>::digits ||
                    (std::size_t(1) << shift) > std::numeric_limits<std::size_t>::max() / sizeof(SizeType))
                    return std::numeric_limits<std::size_t>::max();
                return (std::size_t(1) << shift) * sizeof(SizeType);
            }

            void EnableStorageReuse()
            {
                m_reuseStorage = true;
            }
            std::size_t StorageBytes() const
            {
                return m_capacity * sizeof(SizeType);
            }

            void Prepare(SizeType size, int exp)
            {
                if (!m_hashTable || size != m_initialSize || exp != m_exp)
                    Init(size, exp);
            }

            void Init(SizeType size, int exp)
            {
                m_initialSize = size;
                int ex = 0;
                while (size != 0) {
                    ex++;
                    size >>= 1;
                }
                m_secondHash = true;
                m_exp = exp;
                m_poolSize = (((std::uint64_t)1) << (ex + exp)) - 1;
                const auto count = (m_poolSize + 1) * 2;
                if (!m_reuseStorage || count > m_capacity)
                {
                    m_hashTable.reset(new SizeType[count]);
                    m_capacity = count;
                }
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

            inline int MaxCheck() const { return (1 << (int)(log2(m_poolSize + 1) - m_exp)); }

            inline bool CheckAndSet(SizeType idx)
            {
                // Inner Index is begin from 1
                return _CheckAndSet(m_hashTable.get(), m_poolSize, true, idx + 1) == 0;
            }

            inline void DoubleSize()
            {
                std::uint64_t new_poolSize = ((m_poolSize + 1) << 1) - 1; 
                SizeType* new_hashTable = new SizeType[(new_poolSize + 1) * 2];
                memset(new_hashTable, 0, sizeof(SizeType) * (new_poolSize + 1) * 2);

                m_secondHash = false;
                for (std::uint64_t i = 0; i <= new_poolSize; i++)
                    if (m_hashTable[i]) _CheckAndSet(new_hashTable, new_poolSize, true, m_hashTable[i]);

                m_exp++;
                m_poolSize = new_poolSize;
                m_hashTable.reset(new_hashTable);
                m_capacity = (m_poolSize + 1) * 2;
            }

            inline int _CheckAndSet(SizeType *hashTable, std::uint64_t poolSize, bool isFirstTable, SizeType idx)
            {
                std::uint64_t index = hash_func((std::uint64_t)idx, poolSize);
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
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Hash table is full! Set HashTableExponent to larger value (default is 2). NewHashTableExponent=%d NewPoolSize=%d\n", m_exp, m_poolSize);
                return _CheckAndSet(m_hashTable.get(), m_poolSize, true, idx);
            }
        };

        class DistPriorityQueue {
            int m_size;
            std::unique_ptr<float[]> m_data;
            int m_length;
            int m_count;
            bool m_reuseStorage = false;

          public:
            DistPriorityQueue(): m_size(0), m_length(0), m_count(0) {}

            ~DistPriorityQueue() { m_data.reset(); }

            void EnableStorageReuse()
            {
                m_reuseStorage = true;
            }
            std::size_t StorageBytes() const
            {
                return m_data ? sizeof(float) * (m_size + 1) : 0;
            }

            void Resize(int size_) {
                if (!m_reuseStorage || size_ > m_size)
                {
                    m_size = size_;
                    m_data.reset(new float[size_ + 1]);
                }

                m_data[1] = MaxDist;
                m_length = 1;
                m_count = size_;
            }
            void clear(int count_) {
                if (count_ > m_size) {
                    m_size = count_;
                    m_data.reset(new float[count_ + 1]);
                }
                m_data[1] = MaxDist;
                m_length = 1;
                m_count = count_;
                
            }
            bool insert(float dist) {
                if (dist > m_data[1]) return false;

                if (m_length == m_count) {
                    m_data[1] = dist;
                    int parent = 1, next = 2;
                    while (next < m_length) {
                        if (m_data[next] < m_data[next + 1]) next++;
                        if (m_data[next] > m_data[parent]) {
                            std::swap(m_data[parent], m_data[next]);
                            parent = next;
                            next <<= 1;
                        }
                        else break;
                    }
                    if (next == m_length && m_data[next] > m_data[parent]) std::swap(m_data[parent], m_data[next]);
                }
                else {
                    int next = ++(m_length), parent = (next >> 1);
                    while (parent > 0 && dist > m_data[parent]) {
                        m_data[next] = m_data[parent];
                        next = parent;
                        parent >>= 1;
                    }
                    m_data[next] = dist;
                }
                return true;
            }
            inline float worst() {
                return m_data[1];
            }
        };

        class IWorkSpace {};

        // Variables for each single NN search
        struct WorkSpace : public IWorkSpace
        {
            WorkSpace() {}

            WorkSpace(WorkSpace& other) 
            {
                Initialize(other.m_iMaxCheck, other.nodeCheckStatus.HashTableExponent());
            }

            ~WorkSpace() {
                //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Delete workspace happens!\n");
            }

            void Initialize(int maxCheck, int hashExp)
            {
                m_initialMaxCheck = maxCheck;
                m_initialHashExp = hashExp;
                nodeCheckStatus.Init(maxCheck, hashExp);
                m_SPTQueue.Resize(maxCheck * 10);
                m_NGQueue.Resize(maxCheck * 30);
                m_Results.Resize(maxCheck / 16);

                m_iNumOfContinuousNoBetterPropagation = 0;
                //m_iContinuousLimit = maxCheck / 64;
                m_iNumberOfTreeCheckedLeaves = 0;
                m_iNumberOfCheckedLeaves = 0;
                m_iMaxCheck = maxCheck;
                m_relaxedMono = false;
            }

            void Initialize(va_list& arg)
            {
                int maxCheck = va_arg(arg, int);
                int hashExp = va_arg(arg, int);
                Initialize(maxCheck, hashExp);
            }

            void Reset(int maxCheck, int resultNum)
            {
                nodeCheckStatus.clear();
                m_SPTQueue.clear(maxCheck * 10);
                m_NGQueue.clear(maxCheck * 30);
                m_Results.clear(max(maxCheck / 16, resultNum));

                m_iNumOfContinuousNoBetterPropagation = 0;
                //m_iContinuousLimit = maxCheck / 64;
                m_iNumberOfTreeCheckedLeaves = 0;
                m_iNumberOfCheckedLeaves = 0;
                m_iMaxCheck = maxCheck;
                m_relaxedMono = false;
            }

            void ResetResult(int maxCheck, int resultNum)
            {
                m_Results.clear(max(maxCheck / 16, resultNum));
                m_iNumOfContinuousNoBetterPropagation = 0;
                m_iNumberOfTreeCheckedLeaves = 0;
                m_iNumberOfCheckedLeaves = 0;
            }

            inline bool CheckAndSet(SizeType idx)
            {
                return nodeCheckStatus.CheckAndSet(idx);
            }

            inline int HashTableExponent() const 
            { 
                return nodeCheckStatus.HashTableExponent(); 
            }

            static void Reset() {}

            OptHashPosVector nodeCheckStatus;

            // counter for dynamic pivoting
            int m_iNumOfContinuousNoBetterPropagation = 0;
            int m_iContinuousLimit = 128;
            int m_iNumberOfTreeCheckedLeaves = 0;
            int m_iNumberOfCheckedLeaves = 0;
            int m_iMaxCheck = 8192;
            bool m_relaxedMono = false;

            // Prioriy queue used for neighborhood graph
            Heap<NodeDistPair> m_NGQueue;

            // Priority queue Used for Tree
            Heap<NodeDistPair> m_SPTQueue;
            // Priority queue Used for Tree BFS
            Heap<NodeDistPair> m_currBSPTQueue;
            Heap<NodeDistPair> m_nextBSPTQueue;

            DistPriorityQueue m_Results;
            std::function<bool(const ByteArray&)> m_filterFunc;

            void PrepareScratch(int maxCheck, int hashExp)
            {
                nodeCheckStatus.EnableStorageReuse();
                m_SPTQueue.EnableStorageReuse();
                m_NGQueue.EnableStorageReuse();
                m_currBSPTQueue.EnableStorageReuse();
                m_nextBSPTQueue.EnableStorageReuse();
                m_Results.EnableStorageReuse();
                if (maxCheck != m_initialMaxCheck || hashExp != m_initialHashExp)
                {
                    Initialize(maxCheck, hashExp);
                }
                else
                {
                    nodeCheckStatus.Prepare(maxCheck, hashExp);
                    // A previous borrower may have requested a larger search budget.
                    m_SPTQueue.Resize(maxCheck * 10);
                    m_NGQueue.Resize(maxCheck * 30);
                    m_Results.Resize(maxCheck / 16);
                }
            }

            std::size_t ScratchBytes() const
            {
                return nodeCheckStatus.StorageBytes() + m_SPTQueue.StorageBytes() + m_NGQueue.StorageBytes() +
                       m_currBSPTQueue.StorageBytes() + m_nextBSPTQueue.StorageBytes() + m_Results.StorageBytes();
            }

          private:
            int m_initialMaxCheck = 0;
            int m_initialHashExp = 0;
        };

        class WorkerWorkSpaceFactory : public IWorkSpaceFactory<WorkSpace>
        {
            const int &m_maxCheck;
            const int &m_refineCheck;
            const int &m_hashExp;

          public:
            WorkerWorkSpaceFactory(const int &maxCheck, const int &refineCheck, const int &hashExp)
                : m_maxCheck(maxCheck), m_refineCheck(refineCheck), m_hashExp(hashExp)
            {
            }

            std::unique_ptr<WorkSpace> GetWorkSpace() override
            {
                const int maxCheck = std::max(m_maxCheck, m_refineCheck);
                const auto hashBytes = OptHashPosVector::RequiredStorageBytes(maxCheck, m_hashExp);
                const auto heapBytes =
                    sizeof(NodeDistPair) * (std::size_t(maxCheck) * 40 + 2) + sizeof(float) * (maxCheck / 16 + 1);
                const auto bytes = hashBytes > std::numeric_limits<std::size_t>::max() - heapBytes
                                       ? std::numeric_limits<std::size_t>::max()
                                       : hashBytes + heapBytes;
                auto workspace = WorkerSizeClassPool<WorkSpace>::Rent(bytes);
                if (!workspace)
                    workspace = std::make_unique<WorkSpace>();
                workspace->PrepareScratch(maxCheck, m_hashExp);
                return workspace;
            }

            void ReturnWorkSpace(std::unique_ptr<WorkSpace> workspace) override
            {
                if (!workspace)
                    return;
                workspace->m_filterFunc = nullptr;
                const auto bytes = workspace->ScratchBytes();
                WorkerSizeClassPool<WorkSpace>::Return(std::move(workspace), bytes);
            }
        };
    }
}

#endif // _SPTAG_COMMON_WORKSPACE_H_
