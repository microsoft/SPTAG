// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_TIKV_VERSIONMAP_H_
#define _SPTAG_COMMON_TIKV_VERSIONMAP_H_

#include "IVersionMap.h"
#include "inc/Helper/KeyValueIO.h"
#include <atomic>
#include <string>
#include <vector>
#include <unordered_set>
#include <mutex>
#include <cstring>
#include <chrono>
#include <sstream>
#include <iomanip>

namespace SPTAG
{
    namespace COMMON
    {
        /// TiKVVersionMap stores each VID version as an individual TiKV key.
        ///
        /// Key schema before the TiKVIO namespace prefix is applied:
        ///   vc:{layer}                -> SizeType vector count
        ///   v:{layer}:{vid}           -> uint8_t version for one VID
        ///
        /// Different benchmark runs should use distinct TiKVKeyPrefix values,
        /// or wipe TiKV data before reuse, so old per-VID keys cannot leak into
        /// a new run with the same layer/VID keyspace.
        class TiKVVersionMap : public IVersionMap
        {
        private:
            std::shared_ptr<Helper::KeyValueIO> m_db;
            int m_layer{0};
            std::atomic<SizeType> m_count{0};
            std::atomic<SizeType> m_deleted{0};
            uint8_t m_defaultVersion{0x00};
            std::atomic<bool> m_metadataDirty{false};
            SizeType m_lastPersistedCount{0};

            static constexpr SizeType kMetadataFlushInterval = 65536;

            static constexpr int kWriteStripes = 1024;
            mutable std::mutex m_writeMutex[kWriteStripes];
            std::mutex& VersionMutex(SizeType vid) const { return m_writeMutex[static_cast<size_t>(vid) % kWriteStripes]; }

            static constexpr auto MaxTimeout = std::chrono::microseconds(60000000); // 60s

            std::string CountKey() const
            {
                return "vc:" + std::to_string(m_layer);
            }

            std::string VersionKey(SizeType vid) const
            {
                std::ostringstream key;
                key << "v:" << m_layer << ':' << std::setw(10) << std::setfill('0') << vid;
                return key.str();
            }

            uint8_t DefaultVersionForLayer() const
            {
                return m_layer == 0 ? 0x00 : 0xfe;
            }

            ErrorCode PutByte(const std::string& key, uint8_t value)
            {
                std::string data(1, static_cast<char>(value));
                return m_db->Put(key, data, MaxTimeout, nullptr);
            }

            ErrorCode PutSizeType(const std::string& key, SizeType value)
            {
                std::string data(reinterpret_cast<const char*>(&value), sizeof(SizeType));
                return m_db->Put(key, data, MaxTimeout, nullptr);
            }

            bool ReadSizeType(const std::string& key, SizeType& value) const
            {
                std::string data;
                auto ret = m_db->Get(key, &data, MaxTimeout, nullptr);
                if (ret == ErrorCode::Success && data.size() >= sizeof(SizeType)) {
                    std::memcpy(&value, data.data(), sizeof(SizeType));
                    return true;
                }
                return false;
            }

            bool ReadByte(const std::string& key, uint8_t& value) const
            {
                std::string data;
                auto ret = m_db->Get(key, &data, MaxTimeout, nullptr);
                if (ret == ErrorCode::Success) {
                    value = data.empty() ? m_defaultVersion : static_cast<uint8_t>(data[0]);
                    return true;
                }
                if (ret == ErrorCode::Key_NotFound) {
                    value = m_defaultVersion;
                    return true;
                }
                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                    "TiKVVersionMap::ReadByte failed key=%s layer=%d ret=%d; treating as deleted.\n",
                    key.c_str(), m_layer, static_cast<int>(ret));
                value = 0xfe;
                return false;
            }

            void SaveCount()
            {
                PutSizeType(CountKey(), m_count.load());
            }

            void SaveMetadata()
            {
                SaveCount();
                m_lastPersistedCount = m_count.load();
                m_metadataDirty.store(false, std::memory_order_release);
            }

            void MarkMetadataDirty()
            {
                m_metadataDirty.store(true, std::memory_order_release);
            }

            void MaybeFlushMetadata()
            {
                SizeType count = m_count.load();
                if (count - m_lastPersistedCount >= kMetadataFlushInterval) {
                    SaveMetadata();
                }
            }

            void EnsureCountAtLeast(SizeType count)
            {
                SizeType current = m_count.load();
                bool updated = false;
                while (current < count) {
                    if (m_count.compare_exchange_weak(current, count)) {
                        updated = true;
                        break;
                    }
                }
                if (updated) {
                    MarkMetadataDirty();
                    MaybeFlushMetadata();
                }
            }

            uint8_t ReadVersionByte(SizeType vid) const
            {
                uint8_t value = 0xfe;
                ReadByte(VersionKey(vid), value);
                return value;
            }

            bool ReadVersionEntry(SizeType vid, uint8_t& value, bool& notExist) const
            {
                std::string data;
                auto ret = m_db->Get(VersionKey(vid), &data, MaxTimeout, nullptr);
                if (ret == ErrorCode::Success) {
                    notExist = false;
                    value = data.empty() ? m_defaultVersion : static_cast<uint8_t>(data[0]);
                    return true;
                }
                if (ret == ErrorCode::Key_NotFound) {
                    notExist = true;
                    value = m_defaultVersion;
                    return true;
                }
                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                    "TiKVVersionMap::ReadVersionEntry failed vid=%d layer=%d ret=%d; treating as deleted.\n",
                    vid, m_layer, static_cast<int>(ret));
                notExist = true;
                value = 0xfe;
                return false;
            }

            bool WriteVersionByte(SizeType vid, uint8_t newVal, uint8_t& oldVal)
            {
                std::lock_guard<std::mutex> lock(VersionMutex(vid));
                if (!ReadByte(VersionKey(vid), oldVal)) {
                    return false;
                }

                auto ret = PutByte(VersionKey(vid), newVal);
                if (ret != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                        "TiKVVersionMap::WriteVersionByte failed vid=%d layer=%d ret=%d\n",
                        vid, m_layer, static_cast<int>(ret));
                    return false;
                }
                return true;
            }

            void UpdateDeleteCount(uint8_t oldVal, uint8_t newVal)
            {
                if (oldVal == 0xfe && newVal != 0xfe) {
                    m_deleted.fetch_sub(1, std::memory_order_relaxed);
                } else if (oldVal != 0xfe && newVal == 0xfe) {
                    m_deleted.fetch_add(1, std::memory_order_relaxed);
                }
            }

        public:
            TiKVVersionMap() = default;

            void SetDB(std::shared_ptr<Helper::KeyValueIO> db) { m_db = db; }
            void SetLayer(int layer) { m_layer = layer; m_defaultVersion = DefaultVersionForLayer(); }
            void SetChunkSize(int) {}

            std::shared_ptr<Helper::KeyValueIO> GetDB() const { return m_db; }

            void Initialize(SizeType size, SizeType blockSize, SizeType capacity, COMMON::Dataset<SizeType>* globalIDs = nullptr)
            {
                (void)blockSize;
                (void)capacity;

                m_count = size;

                if (m_layer > 0 && globalIDs != nullptr && globalIDs->R() > 0) {
                    m_defaultVersion = DefaultVersionForLayer();
                    std::unordered_set<SizeType> aliveIDs;
                    aliveIDs.reserve(static_cast<size_t>(globalIDs->R()));
                    for (SizeType i = 0; i < globalIDs->R(); i++) {
                        SizeType globalID = *(globalIDs->At(i));
                        if (globalID >= 0 && globalID < size) {
                            aliveIDs.insert(globalID);
                        }
                    }

                    m_deleted = size;
                    SaveMetadata();

                    SizeType written = 0;
                    for (SizeType globalID : aliveIDs) {
                        if (PutByte(VersionKey(globalID), 0x00) == ErrorCode::Success) {
                            written++;
                        }
                    }
                    m_deleted = size - written;
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "TiKVVersionMap::Initialize layer=%d: per-VID mode, size=%d, default=deleted, alive=%d, written=%d, deleted=%d\n",
                        m_layer, size, static_cast<int>(aliveIDs.size()), written, m_deleted.load());
                } else {
                    m_defaultVersion = DefaultVersionForLayer();
                    m_deleted = (m_defaultVersion == 0xfe) ? size : 0;
                    SaveMetadata();
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "TiKVVersionMap::Initialize layer=%d: per-VID mode, size=%d, default=%s, deleted=%d\n",
                        m_layer, size, (m_defaultVersion == 0xfe) ? "deleted" : "alive", m_deleted.load());
                }
            }

            ErrorCode GetContainedIDs(std::vector<SizeType>& globalIDs) override {
                return ErrorCode::Success;
            }
            
            void DeleteAll() override
            {
                m_defaultVersion = 0xfe;
                m_deleted = m_count.load();
                SaveMetadata();
                for (SizeType vid = 0; vid < m_count.load(); vid++) {
                    PutByte(VersionKey(vid), 0xfe);
                }
            }

            SizeType Count() override { return m_count.load(); }
            SizeType GetDeleteCount() override { return 0; }
            SizeType GetVectorNum() { return m_count.load(); }
            std::uint64_t BufferSize() override { return static_cast<std::uint64_t>(m_count.load()) + sizeof(SizeType) * 2 + sizeof(uint8_t); }

            bool Deleted(const SizeType& key) override
            {
                return Deleted(key, VersionReadPolicy::UseCache);
            }

            bool Deleted(const SizeType& key, VersionReadPolicy policy) override
            {
                (void)policy;
                if (key < 0 || key >= m_count.load()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVVersionMap::Deleted: invalid key %d (max %d)\n", key, m_count.load());
                    return true;
                }
                return ReadVersionByte(key) == 0xfe;
            }

            bool Delete(const SizeType& key) override
            {
                if (key < 0 || key >= m_count.load()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVVersionMap::Delete: invalid key %d (max %d)\n", key, m_count.load());
                    return false;
                }
                uint8_t oldVal;
                if (!WriteVersionByte(key, 0xfe, oldVal)) return false;
                if (oldVal == 0xfe) return false;
                UpdateDeleteCount(oldVal, 0xfe);
                return true;
            }

            uint8_t GetVersion(const SizeType& key) override
            {
                return GetVersion(key, VersionReadPolicy::UseCache);
            }

            uint8_t GetVersion(const SizeType& key, VersionReadPolicy policy) override
            {
                (void)policy;
                if (key < 0 || key >= m_count.load()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVVersionMap::GetVersion: invalid key %d (max %d)\n", key, m_count.load());
                    return 0xfe;
                }
                return ReadVersionByte(key);
            }

            bool TryGetDefaultVersionForNewVector(uint8_t& version) const override
            {
                if (m_defaultVersion == 0xfe) return false;
                version = m_defaultVersion;
                return true;
            }

            void SetVersion(const SizeType& key, const uint8_t& version) override
            {
                if (key < 0) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVVersionMap::SetVersion: invalid key %d (max %d)\n", key, m_count.load());
                    return;
                }
                uint8_t oldVal;
                uint8_t storedVersion = version;
                if (!WriteVersionByte(key, storedVersion, oldVal)) return;
                EnsureCountAtLeast(key + 1);
                UpdateDeleteCount(oldVal, storedVersion);
            }

            // Per-VID batch write: mirrors SetVersion() for each (vid, ver) pair.
            // The new per-VID-key TiKVVersionMap has no chunked batching path, so
            // this is a thin convenience loop.  Performance-sensitive callers
            // can switch to m_db->MultiPut() directly if profiling requires it.
            void SetVersionBatch(const std::vector<SizeType>& vids, const std::vector<uint8_t>& versions) override
            {
                size_t n = std::min(vids.size(), versions.size());
                if (n == 0) return;
                for (size_t i = 0; i < n; ++i) {
                    SetVersion(vids[i], versions[i]);
                }
            }

            bool IncVersion(const SizeType& key, uint8_t* newVersion, uint8_t expectedOld = 0xff) override
            {
                if (key < 0 || key >= m_count.load()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVVersionMap::IncVersion: invalid key %d (max %d)\n", key, m_count.load());
                    return false;
                }

                uint8_t current;
                bool currentNotExist = false;
                if (!ReadVersionEntry(key, current, currentNotExist)) return false;

                constexpr int kMaxCasConflicts = 16;
                for (int attempt = 0; attempt < kMaxCasConflicts; attempt++) {
                    if (current == 0xfe) return false;

                    uint8_t target;
                    if (expectedOld != 0xff) {
                        target = (expectedOld + 1) & 0x7f;
                        if (current == target) {
                            *newVersion = target;
                            return true;
                        }
                        if (current != expectedOld) return false;
                    } else {
                        target = (current + 1) & 0x7f;
                    }

                    std::string newValue(1, static_cast<char>(target));
                    std::string expectedValue = currentNotExist ? std::string() : std::string(1, static_cast<char>(current));
                    bool swapped = false;
                    bool actualNotExist = false;
                    std::string actualValue;
                    auto ret = m_db->CompareAndSwap(VersionKey(key), newValue,
                        currentNotExist, expectedValue, MaxTimeout, nullptr,
                        &swapped, &actualNotExist, &actualValue);
                    if (ret != ErrorCode::Success) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                            "TiKVVersionMap::IncVersion: CAS failed for key %d, ret=%d\n",
                            key, static_cast<int>(ret));
                        return false;
                    }
                    if (swapped) {
                        *newVersion = target;
                        return true;
                    }

                    currentNotExist = actualNotExist;
                    current = actualNotExist || actualValue.empty() ? m_defaultVersion : static_cast<uint8_t>(actualValue[0]);
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                    "TiKVVersionMap::IncVersion: CAS conflict after %d attempts for key %d\n",
                    kMaxCasConflicts, key);
                return false;
            }

            ErrorCode AddBatch(SizeType num)
            {
                return AddBatch(num, false);
            }

            ErrorCode AddBatch(SizeType num, bool deleted)
            {
                if (num <= 0) return ErrorCode::Success;

                SizeType oldCount = m_count.load();
                SizeType newCount = oldCount + num;

                if (deleted) {
                    if (m_defaultVersion != 0xfe) {
                        for (SizeType vid = oldCount; vid < newCount; vid++) {
                            auto ret = PutByte(VersionKey(vid), 0xfe);
                            if (ret != ErrorCode::Success) return ret;
                        }
                    }
                    m_deleted.fetch_add(num, std::memory_order_relaxed);
                } else if (m_defaultVersion == 0xfe) {
                    for (SizeType vid = oldCount; vid < newCount; vid++) {
                        auto ret = PutByte(VersionKey(vid), 0x00);
                        if (ret != ErrorCode::Success) return ret;
                    }
                }

                m_count = newCount;
                MarkMetadataDirty();
                MaybeFlushMetadata();
                return ErrorCode::Success;
            }

            void SetR(SizeType num) override
            {
                m_count = num;
                MarkMetadataDirty();
                MaybeFlushMetadata();
            }

            ErrorCode Save(std::shared_ptr<Helper::DiskIO> output) override
            {
                (void)output;
                SaveMetadata();
                return ErrorCode::Success;
            }

            ErrorCode Save(const std::string& filename) override
            {
                (void)filename;
                SaveMetadata();
                return ErrorCode::Success;
            }

            ErrorCode Load(std::shared_ptr<Helper::DiskIO> input, SizeType blockSize, SizeType capacity) override
            {
                (void)input;
                (void)blockSize;
                (void)capacity;
                return LoadMetadataFromTiKV();
            }

            ErrorCode Load(const std::string& filename, SizeType blockSize, SizeType capacity) override
            {
                (void)filename;
                (void)blockSize;
                (void)capacity;
                return LoadMetadataFromTiKV();
            }

            ErrorCode Load(char* pmemoryFile, SizeType blockSize, SizeType capacity)
            {
                (void)pmemoryFile;
                (void)blockSize;
                (void)capacity;
                return LoadMetadataFromTiKV();
            }

            void BatchGetVersions(const std::vector<SizeType>& vids, std::vector<uint8_t>& versions) override
            {
                BatchGetVersions(vids, versions, VersionReadPolicy::UseCache);
            }

            void BatchGetVersions(const std::vector<SizeType>& vids, std::vector<uint8_t>& versions, VersionReadPolicy policy) override
            {
                (void)policy;
                versions.assign(vids.size(), 0xfe);
                if (vids.empty()) return;

                SizeType count = m_count.load();
                std::vector<size_t> validIndices;
                std::vector<std::string> keys;
                validIndices.reserve(vids.size());
                keys.reserve(vids.size());

                for (size_t i = 0; i < vids.size(); i++) {
                    if (vids[i] >= 0 && vids[i] < count) {
                        validIndices.push_back(i);
                        keys.push_back(VersionKey(vids[i]));
                    }
                }
                if (keys.empty()) return;

                std::vector<std::string> values;
                auto ret = m_db->MultiGet(keys, &values, MaxTimeout, nullptr);
                if (ret != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                        "TiKVVersionMap::BatchGetVersions: MultiGet failed (layer=%d, ret=%d, keys=%d). Unresolved versions are treated as deleted.\n",
                        m_layer, static_cast<int>(ret), static_cast<int>(keys.size()));
                    return;
                }

                for (size_t i = 0; i < validIndices.size(); i++) {
                    if (i < values.size() && !values[i].empty()) {
                        versions[validIndices[i]] = static_cast<uint8_t>(values[i][0]);
                    } else {
                        versions[validIndices[i]] = m_defaultVersion;
                    }
                }
            }

        private:
            ErrorCode LoadMetadataFromTiKV()
            {
                SizeType count = 0;
                if (!ReadSizeType(CountKey(), count)) {
                    m_count = 0;
                    m_deleted = 0;
                    m_defaultVersion = DefaultVersionForLayer();
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                        "TiKVVersionMap: failed to read count key '%s' (layer=%d); set count=0.\n",
                        CountKey().c_str(), m_layer);
                    return ErrorCode::Success;
                }
                m_count = count;

                m_defaultVersion = DefaultVersionForLayer();

                m_deleted = (m_defaultVersion == 0xfe) ? m_count.load() : 0;
                m_lastPersistedCount = m_count.load();
                m_metadataDirty.store(false, std::memory_order_release);

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                    "TiKVVersionMap: loaded per-VID metadata layer=%d count=%d deleted=%d default=%u\n",
                    m_layer, m_count.load(), m_deleted.load(), static_cast<unsigned>(m_defaultVersion));
                return ErrorCode::Success;
            }
        };
    }
}

#endif // _SPTAG_COMMON_TIKV_VERSIONMAP_H_
