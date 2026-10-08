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
        
        class LocalVersionMap : public IVersionMap
        {
        private:
            Helper::Concurrent::ConcurrentHashMap<SizeType, uint8_t> m_label;
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
            void SetVersion(SizeType& key, uint8_t& version) override { 
                m_label.insert_or_assign(std::forward<SizeType>(key), std::forward<uint8_t>(version));
            }

            bool IncVersion(SizeType& key, uint8_t* newVersion, uint8_t expectedOld = 0xff) override {
                auto iter = m_label.find(key);
                if (iter == m_label.end()) return false;
                uint8_t oldVersion = iter->second;
                *newVersion = (oldVersion+1) & 0x7f;
                m_label.assign_if_equal(std::forward<SizeType>(key), oldVersion, std::forward<uint8_t>(*newVersion));
                return true; 
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
                uint8_t oldvalue = (uint8_t)InterlockedExchange8((char*)(m_data[key]), (char)0xfe);
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
                uint8_t oldvalue = (uint8_t)InterlockedExchange8((char*)(m_data[key]), (char)version);
                if (oldvalue == 0xfe && version != 0xfe) m_deleted--;
            }

            bool IncVersion(SizeType& key, uint8_t* newVersion, uint8_t expectedOld = 0xff) override
            {
                if (key < 0 || key >= m_data.R()) 
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Error vid in VersionLabel IncVersion operation: %d. Max Allowed: %d\n", key, m_data.R());
                    return false;
                }

                while (true) {
                    if (Deleted(key)) return false;
                    uint8_t oldVersion = GetVersion(key);
                    *newVersion = (oldVersion+1) & 0x7f;
                    if (((uint8_t)InterlockedCompareExchange((char*)m_data[key], (char)*newVersion, (char)oldVersion)) == oldVersion) {
                        return true;
                    }
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
