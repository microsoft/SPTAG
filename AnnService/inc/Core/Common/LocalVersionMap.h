// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_LOCALVERSIONMAP_H_
#define _SPTAG_COMMON_LOCALVERSIONMAP_H_

#include "IVersionMap.h"
#include "inc/Helper/ConcurrentSet.h"
#include <array>
#include <atomic>
#include <mutex>
#include <shared_mutex>

namespace SPTAG
{
    namespace COMMON
    {
        /// Sparse local versions for non-TiKV storage modes (FileIO, RocksDB, SPDK).
        class LocalVersionMap : public IVersionMap
        {
        private:
#if defined(TBB) && !defined(_MSC_VER)
            using VersionMap = Helper::Concurrent::ConcurrentHashMap<SizeType, uint8_t>;
            VersionMap m_label;

            // Readers use thread-assigned lanes; only maintenance locks every lane.
            static constexpr std::size_t OperationSlotCount = 64;
            struct alignas(64) OperationSlot {
                std::shared_timed_mutex mutex;
            };
            std::array<OperationSlot, OperationSlotCount> m_operationSlots;
            std::mutex m_maintenanceMutex;
            std::atomic<bool> m_maintenancePending{false};

            class MaintenanceGuard {
                struct Pending {
                    std::atomic<bool>& flag;
                    explicit Pending(std::atomic<bool>& value) : flag(value) {
                        flag.store(true, std::memory_order_release);
                    }
                    ~Pending() { flag.store(false, std::memory_order_release); }
                };

                // Reverse destruction releases slots before reopening admission.
                std::unique_lock<std::mutex> m_serial;
                Pending m_pending;
                std::array<std::unique_lock<std::shared_timed_mutex>, OperationSlotCount> m_slots;

            public:
                explicit MaintenanceGuard(LocalVersionMap& map)
                    : m_serial(map.m_maintenanceMutex), m_pending(map.m_maintenancePending) {
                    for (std::size_t i = 0; i < OperationSlotCount; ++i)
                        m_slots[i] = std::unique_lock<std::shared_timed_mutex>(map.m_operationSlots[i].mutex);
                }
                MaintenanceGuard(const MaintenanceGuard&) = delete;
                MaintenanceGuard& operator=(const MaintenanceGuard&) = delete;
            };

            std::shared_lock<std::shared_timed_mutex> LockOperation() {
                static std::atomic<std::size_t> nextSlot{0};
                static thread_local const std::size_t slot =
                    nextSlot.fetch_add(1, std::memory_order_relaxed) % OperationSlotCount;
                for (;;) {
                    if (m_maintenancePending.load(std::memory_order_acquire)) {
                        std::unique_lock<std::mutex> wait(m_maintenanceMutex);
                        continue;
                    }
                    std::shared_lock<std::shared_timed_mutex> lock(m_operationSlots[slot].mutex);
                    if (!m_maintenancePending.load(std::memory_order_acquire)) return lock;
                }
            }

            MaintenanceGuard LockTable() { return MaintenanceGuard(*this); }
#else
            Helper::Concurrent::ConcurrentMap<SizeType, uint8_t> m_label;
            std::shared_timed_mutex m_updateMutex;

            std::shared_lock<std::shared_timed_mutex> LockOperation() {
                return std::shared_lock<std::shared_timed_mutex>(m_updateMutex);
            }
            std::unique_lock<std::shared_timed_mutex> LockTable() {
                return std::unique_lock<std::shared_timed_mutex>(m_updateMutex);
            }
#endif
        public:
            LocalVersionMap() = default;

            void DeleteAll() override { 
                auto lock = LockTable();
                m_label.clear(); 
            }

            SizeType Count() override { 
                auto lock = LockOperation();
                return (SizeType)(m_label.size()); 
            }
            SizeType GetDeleteCount() override { return 0; }
            std::uint64_t BufferSize() override { 
                auto lock = LockOperation();
                return m_label.size() * (sizeof(uint8_t) + sizeof(SizeType)); 
            }

            bool Deleted(const SizeType& key) override {
                auto lock = LockOperation();
#if defined(TBB) && !defined(_MSC_VER)
                VersionMap::const_accessor entry;
                return !m_label.find(entry, key);
#else
                if (m_label.find(key) != m_label.end()) return false;
                return true;
#endif
            }
            bool Delete(const SizeType& key) override { 
#if defined(TBB) && !defined(_MSC_VER)
                auto lock = LockOperation();
                return m_label.erase(key);
#else
                auto lock = LockTable();
                return m_label.unsafe_erase(key); 
#endif
            }

            ErrorCode GetContainedIDs(std::vector<SizeType>& globalIDs) override {
                auto lock = LockTable();
                globalIDs.clear();
                for (const auto& it : m_label) {
                    globalIDs.push_back(it.first);
                }
                return ErrorCode::Success;
            }

            uint8_t GetVersion(const SizeType& key) override {
                auto lock = LockOperation();
#if defined(TBB) && !defined(_MSC_VER)
                VersionMap::const_accessor entry;
                if (!m_label.find(entry, key)) return 0xfe;
                return entry->second;
#else
                auto iter = m_label.find(key);
                if (iter == m_label.end()) return 0xfe;
                return iter->second; 
#endif
            }
            void SetVersion(const SizeType& key, const uint8_t& version) override { 
#if defined(TBB) && !defined(_MSC_VER)
                auto lock = LockOperation();
                VersionMap::accessor entry;
                m_label.insert(entry, key);
                entry->second = version;
#else
                auto lock = LockTable();
                m_label[key] = version;
#endif
            }
            bool IncVersion(const SizeType& key, uint8_t* newVersion, uint8_t expectedOld = 0xff) override {
#if defined(TBB) && !defined(_MSC_VER)
                auto lock = LockOperation();
                VersionMap::accessor entry;
                if (!m_label.find(entry, key)) return false;
                *newVersion = (entry->second + 1) & 0x7f;
                entry->second = *newVersion;
#else
                auto lock = LockTable();
                auto iter = m_label.find(key);
                if (iter == m_label.end()) return false;
                uint8_t oldVersion = iter->second;
                *newVersion = (oldVersion+1) & 0x7f;
                m_label[key] = *newVersion;
#endif
                return true; 
            }

            ErrorCode Save(std::shared_ptr<Helper::DiskIO> ptr) override { 
                // TBB traversal/clear require quiescence, including concurrent lookups.
                auto lock = LockTable();
                SizeType CR = m_label.size();
                IOBINARY(ptr, WriteBinary, sizeof(SizeType), (char*)&CR);
                for (auto& it : m_label) {
                    IOBINARY(ptr, WriteBinary, sizeof(SizeType), (char*)&(it.first));
                    IOBINARY(ptr, WriteBinary, sizeof(uint8_t), (char*)&(it.second));
                }
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Save mapping (%lld, 1) Finish!\n", (std::int64_t)CR);
                return ErrorCode::Success;
            }
            ErrorCode Save(const std::string& filename) override { 
                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(filename.c_str(), std::ios::binary | std::ios::out)) return ErrorCode::FailedCreateFile;
                return Save(ptr);
            }
            ErrorCode Load(std::shared_ptr<Helper::DiskIO> ptr, SizeType blockSize, SizeType capacity) override { 
                auto lock = LockTable();
                SizeType CR;
                IOBINARY(ptr, ReadBinary, sizeof(SizeType), (char*)&CR);
                for (int i = 0; i < CR; i++) {
                    SizeType key;
                    uint8_t value;
                    IOBINARY(ptr, ReadBinary, sizeof(SizeType), (char*)&key);
                    IOBINARY(ptr, ReadBinary, sizeof(uint8_t), (char*)&value);
#if defined(TBB) && !defined(_MSC_VER)
                    VersionMap::accessor entry;
                    m_label.insert(entry, key);
                    entry->second = value;
#else
                    m_label[key] = value;
#endif
                }
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load mapping (%lld, 1) Finish!\n", (std::int64_t)CR);
                return ErrorCode::Success;
            }
            ErrorCode Load(const std::string& filename, SizeType blockSize, SizeType capacity) override { 
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load mapping From %s\n", filename.c_str());
                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(filename.c_str(), std::ios::binary | std::ios::in)) return ErrorCode::FailedOpenFile;
                return Load(ptr, blockSize, capacity);
            }
        };
    }
}

#endif // _SPTAG_COMMON_LOCALVERSIONMAP_H_
