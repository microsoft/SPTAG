// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_LOCALVERSIONMAP_H_
#define _SPTAG_COMMON_LOCALVERSIONMAP_H_

#include "IVersionMap.h"
#include "inc/Helper/ConcurrentSet.h"
#include <array>
#include <atomic>
#include <cassert>
#include <limits>
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

            // Low 32-bit IDs use lazy 8 KiB pages; wider/negative sparse IDs use the map.
            struct PresencePage {
                static constexpr std::size_t Bits = 65536;
                std::array<std::atomic<std::uint64_t>, Bits / 64> words;

                PresencePage() {
                    for (auto& word : words) word.store(0, std::memory_order_relaxed);
                }
                std::atomic<std::uint64_t>& Word(SizeType key) {
                    return words[(static_cast<std::uint32_t>(key) % Bits) / 64];
                }
                static std::uint64_t Mask(SizeType key) {
                    return std::uint64_t{1} << (static_cast<std::uint32_t>(key) % 64);
                }
            };

            class PresenceDirectory {
                static constexpr std::size_t PageCount = 65536;
                std::unique_ptr<std::atomic<PresencePage*>[]> m_pages;

            public:
                PresenceDirectory()
                    : m_pages(new std::atomic<PresencePage*>[PageCount]) {
                    for (std::size_t i = 0; i < PageCount; ++i)
                        m_pages[i].store(nullptr, std::memory_order_relaxed);
                }
                ~PresenceDirectory() { Clear(); }

                static bool Supports(SizeType key) {
                    return key >= 0 && static_cast<std::uint64_t>(key) <=
                        (std::numeric_limits<std::uint32_t>::max)();
                }
                PresencePage* Get(SizeType key) const {
                    return m_pages[static_cast<std::uint32_t>(key) / PresencePage::Bits]
                        .load(std::memory_order_acquire);
                }
                PresencePage* Ensure(SizeType key) {
                    auto& slot = m_pages[static_cast<std::uint32_t>(key) / PresencePage::Bits];
                    auto* page = slot.load(std::memory_order_acquire);
                    if (page != nullptr) return page;
                    auto created = std::make_unique<PresencePage>();
                    if (slot.compare_exchange_strong(page, created.get(),
                            std::memory_order_acq_rel, std::memory_order_acquire))
                        return created.release();
                    return page;
                }
                // Only destruction or an all-slot maintenance guard may reclaim pages.
                void Clear() {
                    for (std::size_t i = 0; i < PageCount; ++i) {
                        delete m_pages[i].load(std::memory_order_relaxed);
                        m_pages[i].store(nullptr, std::memory_order_relaxed);
                    }
                }
            };
            PresenceDirectory m_presence;

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
#if defined(TBB) && !defined(_MSC_VER)
                m_presence.Clear();
#endif
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
                if (PresenceDirectory::Supports(key)) {
                    auto* page = m_presence.Get(key);
                    return page == nullptr ||
                        (page->Word(key).load(std::memory_order_acquire) & PresencePage::Mask(key)) == 0;
                }
                return m_label.count(key) == 0;
#else
                if (m_label.find(key) != m_label.end()) return false;
                return true;
#endif
            }
            bool Delete(const SizeType& key) override { 
#if defined(TBB) && !defined(_MSC_VER)
                auto lock = LockOperation();
                VersionMap::accessor entry;
                if (!m_label.find(entry, key)) return false;
                if (PresenceDirectory::Supports(key)) {
                    auto* page = m_presence.Get(key);
                    assert(page != nullptr);
                    page->Word(key).fetch_and(~PresencePage::Mask(key), std::memory_order_release);
                }
                // Keep this key exclusively held until both representations are updated.
                return m_label.erase(entry);
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
                auto* page = PresenceDirectory::Supports(key) ? m_presence.Ensure(key) : nullptr;
                VersionMap::accessor entry;
                m_label.insert(entry, key);
                entry->second = version;
                if (page != nullptr)
                    page->Word(key).fetch_or(PresencePage::Mask(key), std::memory_order_release);
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
                    auto* page = PresenceDirectory::Supports(key) ? m_presence.Ensure(key) : nullptr;
                    VersionMap::accessor entry;
                    m_label.insert(entry, key);
                    entry->second = value;
                    if (page != nullptr)
                        page->Word(key).fetch_or(PresencePage::Mask(key), std::memory_order_release);
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
