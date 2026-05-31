// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_LOCALVERSIONMAP_H_
#define _SPTAG_COMMON_LOCALVERSIONMAP_H_

#include "IVersionMap.h"
#include "inc/Helper/ConcurrentSet.h"
#include <shared_mutex>

namespace SPTAG
{
    namespace COMMON
    {
        /// LocalVersionMap wraps the existing VersionLabel as an IVersionMap.
        /// Used for non-TiKV storage modes (FileIO, RocksDB, SPDK).
        class LocalVersionMap : public IVersionMap
        {
        private:
            Helper::Concurrent::ConcurrentMap<SizeType, uint8_t> m_label;
            std::shared_timed_mutex m_updateMutex;
        public:
            LocalVersionMap() = default;

            void DeleteAll() override { 
                std::unique_lock<std::shared_timed_mutex> lock(m_updateMutex);
                m_label.clear(); 
            }

            void Initialize(SizeType size, SizeType blockSize, SizeType capacity,
                            COMMON::Dataset<SizeType>* globalIDs = nullptr) override
            {
                (void)size;
                (void)blockSize;
                (void)capacity;
                if (globalIDs == nullptr || globalIDs->R() <= 0) return;

                // Hashmap LocalVersionMap treats missing keys as deleted
                // (Deleted() returns true, GetVersion() returns 0xfe).
                // Layer-1 build calls Initialize with the alive-head global
                // IDs; we must explicitly mark them alive (0x00) so that
                // MergePostings' Deleted()/version-mismatch filter does not
                // strip every base head entry on the first async merge.
                std::unique_lock<std::shared_timed_mutex> lock(m_updateMutex);
                for (SizeType i = 0; i < globalIDs->R(); i++) {
                    SizeType globalID = *(globalIDs->At(i));
                    if (globalID >= 0) {
                        m_label[globalID] = 0x00;
                    }
                }
            }

            SizeType Count() override { 
                std::shared_lock<std::shared_timed_mutex> lock(m_updateMutex);
                return (SizeType)(m_label.size()); 
            }
            SizeType GetDeleteCount() override { return 0; }
            std::uint64_t BufferSize() override { 
                std::shared_lock<std::shared_timed_mutex> lock(m_updateMutex);
                return m_label.size() * (sizeof(uint8_t) + sizeof(SizeType)); 
            }

            bool Deleted(const SizeType& key) override {
                std::shared_lock<std::shared_timed_mutex> lock(m_updateMutex);
                if (m_label.find(key) != m_label.end()) return false;
                return true;
            }
            bool Delete(const SizeType& key) override { 
                std::unique_lock<std::shared_timed_mutex> lock(m_updateMutex);
                return m_label.unsafe_erase(key); 
            }

            ErrorCode GetContainedIDs(std::vector<SizeType>& globalIDs) override {
                std::shared_lock<std::shared_timed_mutex> lock(m_updateMutex);
                globalIDs.clear();
                for (const auto& it : m_label) {
                    globalIDs.push_back(it.first);
                }
                return ErrorCode::Success;
            }

            uint8_t GetVersion(const SizeType& key) override {
                std::shared_lock<std::shared_timed_mutex> lock(m_updateMutex);
                auto iter = m_label.find(key);
                if (iter == m_label.end()) return 0xfe;
                return iter->second; 
            }
            void SetVersion(const SizeType& key, const uint8_t& version) override { 
                std::unique_lock<std::shared_timed_mutex> lock(m_updateMutex);
                m_label[key] = version;
            }
            bool IncVersion(const SizeType& key, uint8_t* newVersion, uint8_t expectedOld = 0xff) override {
                std::shared_lock<std::shared_timed_mutex> lock(m_updateMutex);
                auto iter = m_label.find(key);
                if (iter == m_label.end()) return false;
                uint8_t oldVersion = iter->second;
                *newVersion = (oldVersion+1) & 0x7f;
                m_label[key] = *newVersion;
                return true; 
            }

            ErrorCode Save(std::shared_ptr<Helper::DiskIO> ptr) override { 
                std::shared_lock<std::shared_timed_mutex> lock(m_updateMutex);
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
                    m_label[key] = value;
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
