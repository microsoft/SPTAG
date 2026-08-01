// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_TIKV_VERSIONMAP_H_
#define _SPTAG_COMMON_TIKV_VERSIONMAP_H_

#include "IVersionMap.h"
#include "inc/Helper/KeyValueIO.h"
#include <algorithm>
#include <atomic>
#include <string>
#include <vector>
#include <unordered_set>
#include <mutex>
#include <cstring>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace SPTAG
{
    namespace COMMON
    {
        /// TiKVVersionMap stores each VID version as an individual TiKV key.
        ///
        /// Key schema before the TiKVIO namespace prefix is applied:
        ///   vc:{layer}                -> SizeType vector count
        ///   vm:{layer}                -> SizeType maximum allocated VID
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
            std::atomic<SizeType> m_maxVID{-1};
            std::atomic<SizeType> m_deleted{0};
            uint8_t m_defaultVersion{0xff};
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

            std::string MaxVIDKey() const
            {
                return "vm:" + std::to_string(m_layer);
            }

            std::string VersionKey(SizeType vid) const
            {
                std::ostringstream key;
                key << "v:" << m_layer << ':' << std::setw(10) << std::setfill('0') << vid;
                return key.str();
            }

            uint8_t DefaultVersionForLayer() const
            {
                return m_layer == 0 ? 0xff : 0xfe;
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

            void EnsureMaxVIDAtLeast(SizeType maxVID)
            {
                if (maxVID < 0) return;

                SizeType cached = m_maxVID.load(std::memory_order_relaxed);
                while (cached < maxVID &&
                       !m_maxVID.compare_exchange_weak(cached, maxVID,
                                                       std::memory_order_relaxed)) {
                }
                if (cached >= maxVID) return;

                std::string currentValue;
                bool currentNotExist = false;
                auto getRet = m_db->Get(MaxVIDKey(), &currentValue, MaxTimeout, nullptr);
                if (getRet == ErrorCode::Key_NotFound) {
                    currentNotExist = true;
                    currentValue.clear();
                } else if (getRet != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                        "TiKVVersionMap::EnsureMaxVIDAtLeast failed to read max VID layer=%d ret=%d\n",
                        m_layer, static_cast<int>(getRet));
                    return;
                }

                for (int attempt = 0; attempt < 64; ++attempt) {
                    SizeType current = -1;
                    if (!currentNotExist && currentValue.size() >= sizeof(SizeType)) {
                        std::memcpy(&current, currentValue.data(), sizeof(SizeType));
                    }
                    if (current >= maxVID) {
                        m_maxVID.store(current, std::memory_order_relaxed);
                        return;
                    }

                    std::string desired(reinterpret_cast<const char*>(&maxVID), sizeof(SizeType));
                    bool swapped = false;
                    bool actualNotExist = false;
                    std::string actualValue;
                    auto ret = m_db->CompareAndSwap(MaxVIDKey(), desired,
                                                    currentNotExist, currentValue,
                                                    MaxTimeout, nullptr,
                                                    &swapped, &actualNotExist, &actualValue);
                    if (ret != ErrorCode::Success) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                            "TiKVVersionMap::EnsureMaxVIDAtLeast CAS failed layer=%d ret=%d\n",
                            m_layer, static_cast<int>(ret));
                        return;
                    }
                    if (swapped) {
                        m_maxVID.store(maxVID, std::memory_order_relaxed);
                        return;
                    }
                    currentNotExist = actualNotExist;
                    currentValue = std::move(actualValue);
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "TiKVVersionMap::EnsureMaxVIDAtLeast CAS conflict layer=%d maxVID=%d\n",
                    m_layer, maxVID);
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

            void Initialize(SizeType size, SizeType blockSize, SizeType capacity, COMMON::Dataset<SizeType>* globalIDs = nullptr) override
            {
                (void)blockSize;
                (void)capacity;

                m_count = size;
                EnsureMaxVIDAtLeast(size - 1);

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

                    // Batch the alive-marker writes via MultiPut so they
                    // can be grouped per TiKV region and issued in parallel.
                    // Serial PutByte was the build-time hotspot (~1-2ms
                    // per write × ~200K alive heads at 1M-vector scale).
                    std::vector<SizeType> aliveSorted;
                    aliveSorted.reserve(aliveIDs.size());
                    for (SizeType id : aliveIDs) aliveSorted.push_back(id);
                    std::sort(aliveSorted.begin(), aliveSorted.end());

                    SizeType written = 0;
                    constexpr size_t kBatchSize = 4096;
                    std::vector<std::string> keys;
                    std::vector<std::string> values;
                    keys.reserve(kBatchSize);
                    values.reserve(kBatchSize);
                    const std::string aliveByte(1, static_cast<char>(0xff));
                    for (size_t i = 0; i < aliveSorted.size(); i++) {
                        keys.push_back(VersionKey(aliveSorted[i]));
                        values.push_back(aliveByte);
                        if (keys.size() >= kBatchSize || i + 1 == aliveSorted.size()) {
                            auto ret = m_db->MultiPut(keys, values, MaxTimeout, nullptr);
                            if (ret == ErrorCode::Success) {
                                written += static_cast<SizeType>(keys.size());
                            } else if (ret == ErrorCode::Undefined) {
                                // Backend lacks MultiPut: fall back to serial PutByte.
                                for (const auto& k : keys) {
                                    if (PutByte(k, 0xff) == ErrorCode::Success) written++;
                                }
                            } else {
                                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                                    "TiKVVersionMap::Initialize: MultiPut batch failed layer=%d ret=%d size=%zu; falling back to serial PutByte for this batch.\n",
                                    m_layer, static_cast<int>(ret), keys.size());
                                for (const auto& k : keys) {
                                    if (PutByte(k, 0xff) == ErrorCode::Success) written++;
                                }
                            }
                            keys.clear();
                            values.clear();
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

            ErrorCode GetContainedIDs(std::vector<SizeType>& globalIDs) override
            {
                globalIDs.clear();
                const SizeType count = m_count.load();
                if (count <= 0) return ErrorCode::Success;

                constexpr SizeType kBatchSize = 4096;
                ErrorCode result = ErrorCode::Success;
                std::vector<std::string> keys;
                std::vector<std::string> values;
                keys.reserve(kBatchSize);

                for (SizeType batchStart = 0; batchStart < count; batchStart += kBatchSize) {
                    const SizeType batchEnd = (std::min)(batchStart + kBatchSize, count);
                    keys.clear();
                    for (SizeType vid = batchStart; vid < batchEnd; vid++) {
                        keys.emplace_back(VersionKey(vid));
                    }

                    values.clear();
                    auto ret = m_db->MultiGet(keys, &values, MaxTimeout, nullptr);
                    if (ret != ErrorCode::Success) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                            "TiKVVersionMap::GetContainedIDs: MultiGet failed (layer=%d, ret=%d, range=[%d,%d)); treating this range as deleted.\n",
                            m_layer, static_cast<int>(ret), batchStart, batchEnd);
                        result = ErrorCode::Fail;
                        continue;
                    }

                    for (SizeType vid = batchStart; vid < batchEnd; vid++) {
                        size_t index = static_cast<size_t>(vid - batchStart);
                        uint8_t version = (index < values.size() && !values[index].empty())
                            ? static_cast<uint8_t>(values[index][0])
                            : m_defaultVersion;
                        if (version != 0xfe) {
                            globalIDs.push_back(vid);
                        }
                    }
                }

                return result;
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
            SizeType MaxVID() override
            {
                SizeType maxVID = -1;
                if (ReadSizeType(MaxVIDKey(), maxVID)) {
                    m_maxVID.store(maxVID, std::memory_order_relaxed);
                    return maxVID;
                }
                return m_maxVID.load(std::memory_order_relaxed);
            }
            SizeType GetDeleteCount() override { return 0; }
            std::uint64_t BufferSize() override { return static_cast<std::uint64_t>(m_count.load()) + sizeof(SizeType) * 2 + sizeof(uint8_t); }

            bool Deleted(const SizeType& key) override
            {
                return Deleted(key, VersionReadPolicy::UseCache);
            }

            bool Deleted(const SizeType& key, VersionReadPolicy policy) override
            {
                (void)policy;
                if (key < 0) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKVVersionMap::Deleted: invalid key %d (max %d)\n", key, m_count.load());
                    return true;
                }
                return ReadVersionByte(key) == 0xfe;
            }

            bool Delete(const SizeType& key) override
            {
                if (key < 0) {
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
                if (key < 0) {
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
                EnsureMaxVIDAtLeast(key);
                UpdateDeleteCount(oldVal, storedVersion);
            }

            void SetR(SizeType num) override
            {
                EnsureCountAtLeast(num);
                EnsureMaxVIDAtLeast(num - 1);
            }

            // Per-VID batch write: mirrors SetVersion() for each (vid, ver) pair.
            // Uses TiKVIO MultiPut so the writes are grouped per TiKV region
            // and issued in parallel. m_deleted accounting is approximate
            // here (we do not read the old byte to compute the exact delta);
            // GetDeleteCount() returns 0 for the TiKV-backed version map so
            // this approximation is acceptable. Callers that need precise
            // accounting can call SetVersion() per-VID instead.
            void SetVersionBatch(const std::vector<SizeType>& vids, const std::vector<uint8_t>& versions) override
            {
                size_t n = std::min(vids.size(), versions.size());
                if (n == 0) return;

                SizeType count = m_count.load();
                SizeType maxKey = -1;
                std::vector<std::string> keys;
                std::vector<std::string> values;
                keys.reserve(n);
                values.reserve(n);
                for (size_t i = 0; i < n; ++i) {
                    // Only a negative VID is a genuine torn/garbage read. In
                    // distributed mode global VIDs are striped across nodes and
                    // the version map is a shared global keyspace in TiKV, so a
                    // remote-owned VID >= the local Count() is legitimate (these
                    // calls exist precisely to mirror remote-appended records).
                    // Mirror SetVersion(): accept any vid >= 0 and grow the
                    // local count hint to cover it.
                    if (vids[i] < 0) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                            "TiKVVersionMap::SetVersionBatch: invalid key %d (max %d)\n",
                            vids[i], count);
                        continue;
                    }
                    if (vids[i] > maxKey) maxKey = vids[i];
                    keys.push_back(VersionKey(vids[i]));
                    values.push_back(std::string(1, static_cast<char>(versions[i])));
                }
                if (keys.empty()) return;
                if (maxKey >= 0) EnsureCountAtLeast(maxKey + 1);
                if (maxKey >= 0) EnsureMaxVIDAtLeast(maxKey);

                auto ret = m_db->MultiPut(keys, values, MaxTimeout, nullptr);
                if (ret == ErrorCode::Undefined) {
                    // Backend lacks MultiPut: fall back to serial SetVersion
                    // which preserves m_deleted accounting.
                    for (size_t i = 0; i < n; ++i) {
                        if (vids[i] >= 0) {
                            SetVersion(vids[i], versions[i]);
                        }
                    }
                } else if (ret != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                        "TiKVVersionMap::SetVersionBatch: MultiPut failed layer=%d ret=%d keys=%zu; falling back to per-VID SetVersion.\n",
                        m_layer, static_cast<int>(ret), keys.size());
                    for (size_t i = 0; i < n; ++i) {
                        if (vids[i] >= 0) {
                            SetVersion(vids[i], versions[i]);
                        }
                    }
                }
            }

            bool IncVersion(const SizeType& key, uint8_t* newVersion, uint8_t expectedOld = 0xff) override
            {
                if (key < 0) {
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

            void BatchGetVersions(const std::vector<SizeType>& vids, std::vector<uint8_t>& versions) override
            {
                BatchGetVersions(vids, versions, VersionReadPolicy::UseCache);
            }

            void BatchGetVersions(const std::vector<SizeType>& vids, std::vector<uint8_t>& versions, VersionReadPolicy policy) override
            {
                (void)policy;
                versions.assign(vids.size(), 0xfe);
                if (vids.empty()) return;

                std::vector<size_t> validIndices;
                std::vector<std::string> keys;
                validIndices.reserve(vids.size());
                keys.reserve(vids.size());

                for (size_t i = 0; i < vids.size(); i++) {
                    if (vids[i] >= 0) {
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
                SizeType maxVID = -1;
                if (!ReadSizeType(MaxVIDKey(), maxVID)) {
                    maxVID = count > 0 ? count - 1 : -1;
                    if (maxVID >= 0) PutSizeType(MaxVIDKey(), maxVID);
                }
                m_maxVID = maxVID;

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
