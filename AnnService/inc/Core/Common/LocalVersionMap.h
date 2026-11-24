// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_LOCALVERSIONMAP_H_
#define _SPTAG_COMMON_LOCALVERSIONMAP_H_

#include "IVersionMap.h"
#include "ResidentHashMap.h"
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
        
        class LocalVersionMap : public IVersionMap
        {
        private:
            Helper::Concurrent::ConcurrentHashMap<SizeType, uint8_t> m_label;
            //COMMON::SizeTypeConcurrentMap<uint8_t, char> m_label{0xfe};
            uint8_t m_default = 0xff;
        public:
            LocalVersionMap() = default;

            void DeleteAll() override { 
                m_label.clear(); 
            }

            uint8_t& Default() override { return m_default; }

            SizeType Count() override { 
                return (SizeType)(m_label.size()); 
            }
            SizeType GetDeleteCount() override { return 0; }
            std::uint64_t BufferSize() override { 
                return m_label.size() * (sizeof(uint8_t) + sizeof(SizeType)); 
            }

            bool Deleted(const SizeType& key) override {
                if (m_label.find(key) != m_label.end()) return false;
                return true;
            }
            bool Delete(const SizeType& key) override { 
                return m_label.erase(key); 
            }

            ErrorCode GetContainedIDs(std::vector<SizeType>& globalIDs) override {
                globalIDs.clear();
                for (const auto& it : m_label) {
                    globalIDs.push_back(it.first);
                }
                return ErrorCode::Success;
            }

            uint8_t GetVersion(const SizeType& key) override {
                auto iter = m_label.find(key);
                if (iter == m_label.end()) return 0xfe;
                return iter->second; 
            }
            using IVersionMap::BatchGetVersions;
            void BatchGetVersions(const std::vector<SizeType> &ids, std::vector<uint8_t> &versions,
                                  VersionReadPolicy) override
            {
                versions.resize(ids.size());
                for (size_t i = 0; i < ids.size(); ++i)
                    versions[i] = ids[i] < 0 ? 0xfe : GetVersion(ids[i]);
            }
            void SetVersion(SizeType& key, uint8_t& version) override { 
                m_label.insert_or_assign(std::forward<SizeType>(key), std::forward<uint8_t>(version));
            }

            bool IncVersion(SizeType& key, uint8_t* newVersion, uint8_t expectedOld = 0xff) override {
                auto iter = m_label.find(key);
                if (iter == m_label.end()) return false;
                uint8_t oldVersion = iter->second;
                if (expectedOld != 0xff && oldVersion != expectedOld) return false;
                
                *newVersion = (oldVersion+1) & 0x7f;
                auto res = m_label.assign_if_equal(std::forward<SizeType>(key), oldVersion, std::forward<uint8_t>(*newVersion));
                if (res) return true; 
                return false;
            }

            ErrorCode Save(std::shared_ptr<Helper::DiskIO> ptr) override { 
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
                SizeType CR;
                IOBINARY(ptr, ReadBinary, sizeof(SizeType), (char*)&CR);
                for (int i = 0; i < CR; i++) {
                    SizeType key;
                    uint8_t value;
                    IOBINARY(ptr, ReadBinary, sizeof(SizeType), (char*)&key);
                    IOBINARY(ptr, ReadBinary, sizeof(uint8_t), (char*)&value);
                    m_label.insert_or_assign(std::forward<SizeType>(key), std::forward<uint8_t>(value));
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

        // The managed cache uses explicitly sized nodes. Keep the legacy
        // ConcurrentHashMap implementation and its unmanaged behavior unchanged.
        class ManagedLocalVersionMap : public IVersionMap
        {
            ResidentHashMap<SizeType, uint8_t> m_label;
            uint8_t m_default = 0xff;
            // Only exclusive snapshot load widens this proven live range.
            std::atomic<SizeType> m_livePrefix{0};

          public:
            uint64_t ResidentGrowthBytes(SizeType target) const
            {
                // A cascade can erase an old-credit node and re-promote its VID.
                // Reserve its replacement even when the live count does not grow.
                return ResidentAddBytes(m_label.GrowthBytes(target),
                                        ResidentMultiplyBytes(m_label.size(), decltype(m_label)::NodeBytes()));
            }
            void DeleteAll() override
            {
                m_livePrefix.store(0);
                m_label.clear();
            }
            uint8_t &Default() override
            {
                return m_default;
            }
            SizeType Count() override
            {
                return static_cast<SizeType>(m_label.size());
            }
            SizeType GetDeleteCount() override
            {
                return 0;
            }
            std::uint64_t BufferSize() override
            {
                return m_label.size() * (sizeof(SizeType) + sizeof(uint8_t));
            }
            bool Deleted(const SizeType &key) override
            {
                if (key >= 0 && key < m_livePrefix.load())
                    return false;
                uint8_t value;
                return !m_label.Get(key, value);
            }
            bool Delete(const SizeType &key) override
            {
                if (key >= 0)
                {
                    auto prefix = m_livePrefix.load();
                    while (key < prefix && !m_livePrefix.compare_exchange_weak(prefix, key)) {}
                }
                return m_label.erase(key);
            }
            ErrorCode GetContainedIDs(std::vector<SizeType> &ids) override
            {
                ids.clear();
                m_label.ForEach([&](SizeType key, uint8_t) { ids.push_back(key); });
                return ErrorCode::Success;
            }
            uint8_t GetVersion(const SizeType &key) override
            {
                uint8_t value = 0xfe;
                m_label.Get(key, value);
                return value;
            }
            using IVersionMap::BatchGetVersions;
            void BatchGetVersions(const std::vector<SizeType> &ids, std::vector<uint8_t> &versions,
                                  VersionReadPolicy) override
            {
                versions.resize(ids.size());
                for (size_t i = 0; i < ids.size(); ++i)
                    versions[i] = ids[i] < 0 ? 0xfe : GetVersion(ids[i]);
            }
            void SetVersion(SizeType &key, uint8_t &version) override
            {
                m_label.Set(key, version);
            }
            bool IncVersion(SizeType &key, uint8_t *result, uint8_t expected = 0xff) override
            {
                return m_label.Update(key, [&](uint8_t &version) {
                    if (expected != 0xff && expected != version)
                        return false;
                    version = (version + 1) & 0x7f;
                    *result = version;
                    return true;
                });
            }
            ErrorCode Save(std::shared_ptr<Helper::DiskIO> ptr) override
            {
                SizeType count = Count();
                IOBINARY(ptr, WriteBinary, sizeof(count), (char *)&count);
                for (const auto &entry : m_label)
                {
                    IOBINARY(ptr, WriteBinary, sizeof(SizeType), (char *)&entry.first);
                    IOBINARY(ptr, WriteBinary, sizeof(uint8_t), (char *)&entry.second);
                }
                return ErrorCode::Success;
            }
            ErrorCode Save(const std::string &filename) override
            {
                auto ptr = f_createIO();
                if (!ptr || !ptr->Initialize(filename.c_str(), std::ios::binary | std::ios::out))
                    return ErrorCode::FailedCreateFile;
                return Save(ptr);
            }
            ErrorCode Load(std::shared_ptr<Helper::DiskIO> ptr, SizeType, SizeType capacity) override
            {
                SizeType count;
                IOBINARY(ptr, ReadBinary, sizeof(count), (char *)&count);
                if (count < 0 || count > capacity)
                    return ErrorCode::MemoryOverFlow;
                m_livePrefix.store(0);
                m_label.clear();
                bool dense = true;
                for (SizeType i = 0; i < count; ++i)
                {
                    SizeType key;
                    uint8_t value;
                    IOBINARY(ptr, ReadBinary, sizeof(key), (char *)&key);
                    IOBINARY(ptr, ReadBinary, sizeof(value), (char *)&value);
                    if (key < 0 || key >= capacity || (value >= 0x80 && value != m_default) || !Deleted(key))
                        return ErrorCode::FailedParseValue;
                    m_label.Set(key, value);
                    dense = dense && key >= 0 && key < count;
                }
                if (dense && m_label.size() == static_cast<size_t>(count))
                    m_livePrefix.store(count);
                return ErrorCode::Success;
            }
            ErrorCode Load(const std::string &filename, SizeType blockSize, SizeType capacity) override
            {
                auto ptr = f_createIO();
                if (!ptr || !ptr->Initialize(filename.c_str(), std::ios::binary | std::ios::in))
                    return ErrorCode::FailedOpenFile;
                auto status = Load(ptr, blockSize, capacity);
                char trailing;
                if (status == ErrorCode::Success && ptr->ReadBinary(1, &trailing) != 0)
                    return ErrorCode::FailedParseValue;
                return status;
            }
        };

        class VersionLabel : public IVersionMap
        {
        private:
            std::atomic<SizeType> m_deleted;
            Dataset<std::uint8_t> m_data;
            std::mutex m_mutex;
            uint8_t m_default = 0xff;

        public:
            VersionLabel() : m_deleted(0) { 
                m_data.SetName("versionLabelID"); 
                m_data.SetDefaultValue(0xfe);
                m_data.Initialize(1024 * 1024, 1, 1024 * 1024, MaxSize);
                m_deleted = m_data.R();
            }

            void DeleteAll() override
            {
                m_deleted = m_data.R();
                for (SizeType i = 0; i < m_data.R(); i++) {
                    *m_data[i] = 0xfe;
                }
            }

            uint8_t& Default() override { return m_default; }
            
            SizeType Count() override { return m_data.R(); }

            SizeType GetDeleteCount() override { return m_deleted.load();}

            std::uint64_t BufferSize() override { return m_data.BufferSize() + sizeof(SizeType); }

            bool Deleted(const SizeType& key) override
            {
                if (key < 0)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Error vid in VersionLabel Delete check: %d. Max Allowed: %d\n", key, m_data.R());
                    return true;
                }
                if (key >= m_data.R()) return true;
                return *m_data[key] == 0xfe;
            }

            bool Delete(const SizeType& key) override
            {
                if (key < 0 || key >= m_data.R()) 
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Error vid in VersionLabel Delete operation: %d. Max Allowed: %d\n", key, m_data.R());
                    return true;
                }
                uint8_t oldvalue = (uint8_t)InterlockedExchange((char*)(m_data[key]), (char)0xfe);
                if (oldvalue == 0xfe) return false;
                m_deleted++;
                return true;
            }

            ErrorCode GetContainedIDs(std::vector<SizeType>& globalIDs) override {
                globalIDs.clear();
                for (SizeType i = 0; i < m_data.R(); i++)
                {
                    if (!Deleted(i)) {
                        globalIDs.push_back(i);
                    }
                }
                return ErrorCode::Success;
            }

            uint8_t GetVersion(const SizeType& key) override
            {
                if (key < 0 || key >= m_data.R()) 
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Error vid in VersionLabel GetVersion operation: %d. Max Allowed: %d\n", key, m_data.R());
                    return 0xfe;
                }
                return *m_data[key];
            }

            void SetVersion(SizeType& key, uint8_t& version) override
            {
                if (key >= m_data.R()) {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    SizeType oldR = m_data.R();
                    m_data.AddBatch(key -  oldR + 10000);
                    m_deleted += key - oldR + 10000;
                }
                if (key < 0 || key >= m_data.R()) 
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Error vid in VersionLabel SetVersion operation: %d. Max Allowed: %d\n", key, m_data.R());
                    return;
                }
                uint8_t oldvalue = (uint8_t)InterlockedExchange((char*)(m_data[key]), (char)version);
                if (oldvalue == 0xfe && version != 0xfe) m_deleted--;
            }

            bool IncVersion(SizeType& key, uint8_t* newVersion, uint8_t expectedOld = 0xff) override
            {
                if (key < 0 || key >= m_data.R()) 
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Error vid in VersionLabel IncVersion operation: %d. Max Allowed: %d\n", key, m_data.R());
                    return false;
                }

                auto* value = reinterpret_cast<char*>(m_data[key]);
                uint8_t oldVersion = expectedOld != 0xff ? expectedOld :
                    static_cast<uint8_t>(InterlockedCompareExchange(value, char(0), char(0)));
                while (true) {
                    if (oldVersion == 0xfe || (expectedOld != 0xff && oldVersion != expectedOld)) return false;
                    const uint8_t next = (oldVersion + 1) & 0x7f;
                    const uint8_t observed = static_cast<uint8_t>(
                        InterlockedCompareExchange(value, static_cast<char>(next), static_cast<char>(oldVersion)));
                    if (observed == oldVersion) {
                        *newVersion = next;
                        return true;
                    }
                    oldVersion = observed;
                }
            }

            ErrorCode Save(std::shared_ptr<Helper::DiskIO> output) override
            {
                SizeType deleted = m_deleted.load();
                IOBINARY(output, WriteBinary, sizeof(SizeType), (char*)&deleted);
                return m_data.Save(output);
            }

            ErrorCode Save(const std::string& filename) override
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Save %s To %s\n", m_data.Name().c_str(), filename.c_str());
                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(filename.c_str(), std::ios::binary | std::ios::out)) return ErrorCode::FailedCreateFile;
                return Save(ptr);
            }

            ErrorCode Load(std::shared_ptr<Helper::DiskIO> input, SizeType blockSize, SizeType capacity) override
            {
                SizeType deleted;
                IOBINARY(input, ReadBinary, sizeof(SizeType), (char*)&deleted);
                m_deleted = deleted;
                return m_data.Load(input, blockSize, capacity);
            }

            ErrorCode Load(const std::string& filename, SizeType blockSize, SizeType capacity) override
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load %s From %s\n", m_data.Name().c_str(), filename.c_str());
                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(filename.c_str(), std::ios::binary | std::ios::in)) return ErrorCode::FailedOpenFile;
                return Load(ptr, blockSize, capacity);
            }
        };
    }
}

#endif // _SPTAG_COMMON_LOCALVERSIONMAP_H_
