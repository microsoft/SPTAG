// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_IVERSIONMAP_H_
#define _SPTAG_COMMON_IVERSIONMAP_H_

#include "inc/Core/CommonDataStructure.h"
#include "inc/Core/Common/Dataset.h"
#include "inc/Helper/DiskIO.h"
#include <memory>
#include <vector>
#include <string>

namespace SPTAG
{
    namespace COMMON
    {
        enum class VersionReadPolicy
        {
            UseCache,
            BypassCacheNoFill
        };

        /// Abstract interface for version map, allowing both local (in-memory)
        /// and distributed (TiKV-backed) implementations.
        class IVersionMap
        {
        public:
            virtual ~IVersionMap() = default;

            virtual void DeleteAll() = 0;

            /// One-time per-layer setup performed at the end of BuildIndex.
            ///   size      total VID count for this layer (== m_opt->m_vectorSize)
            ///   blockSize/capacity  hints for array-backed legacy maps; ignored
            ///                       by hashmap / TiKV implementations
            ///   globalIDs (optional) set of GLOBAL VIDs that are alive on
            ///                       this layer. Layers whose "default
            ///                       version" semantics treat unknown VIDs as
            ///                       DELETED (e.g. TiKV layer >0, hashmap
            ///                       LocalVersionMap) MUST persist an
            ///                       explicit alive byte for each globalID;
            ///                       otherwise MergePostings'
            ///                       Deleted()/version-mismatch filter
            ///                       eats every base entry on the first
            ///                       async merge and corrupts the head index.
            /// Default impl: just bump the internal count via SetR.
            virtual void Initialize(SizeType size, SizeType blockSize, SizeType capacity,
                                    COMMON::Dataset<SizeType>* globalIDs = nullptr)
            {
                (void)blockSize;
                (void)capacity;
                (void)globalIDs;
                SetR(size);
            }

            virtual SizeType Count() = 0;
            virtual SizeType GetDeleteCount() = 0;
            virtual std::uint64_t BufferSize() = 0;

            virtual bool Deleted(const SizeType& key) = 0;
            virtual bool Deleted(const SizeType& key, VersionReadPolicy policy) { return Deleted(key); }
            virtual bool Delete(const SizeType& key) = 0;

            virtual ErrorCode GetContainedIDs(std::vector<SizeType>& globalIDs) = 0;

            virtual uint8_t GetVersion(const SizeType& key) = 0;
            virtual uint8_t GetVersion(const SizeType& key, VersionReadPolicy policy) { return GetVersion(key); }
            virtual bool TryGetDefaultVersionForNewVector(uint8_t& version) const { return false; }
            virtual void SetR(SizeType num) {}
            virtual void SetVersion(const SizeType& key, const uint8_t& version) = 0;

            /// Batch SetVersion: apply (vids[i] -> versions[i]) for all i.
            /// Default impl is a per-VID loop. TiKV-backed maps override this
            /// to group writes by chunk so N records in the same chunk only
            /// trigger 1 ReadChunk + 1 WriteChunk RPC pair
            virtual void SetVersionBatch(const std::vector<SizeType>& vids, const std::vector<uint8_t>& versions)
            {
                size_t n = std::min(vids.size(), versions.size());
                for (size_t i = 0; i < n; i++) {
                    SetVersion(vids[i], versions[i]);
                }
            }
            /// Increment the version of a VID.
            /// @param expectedOld If not 0xff, the caller asserts the current version should be this value.
            ///   If TiKV already holds (expectedOld+1)&0x7f, treat as success (another node did the same increment).
            ///   If TiKV holds a different value, return false (conflict).
            ///   If 0xff, just increment whatever the current value is (no check).
            virtual bool IncVersion(const SizeType& key, uint8_t* newVersion, uint8_t expectedOld = 0xff) = 0;

            virtual ErrorCode Save(std::shared_ptr<Helper::DiskIO> output) = 0;
            virtual ErrorCode Save(const std::string& filename) = 0;
            virtual ErrorCode Load(std::shared_ptr<Helper::DiskIO> input, SizeType blockSize, SizeType capacity) = 0;
            virtual ErrorCode Load(const std::string& filename, SizeType blockSize, SizeType capacity) = 0;

            /// Batch version check for a set of VIDs.
            /// Returns a vector of versions (0xfe = deleted) in the same order as vids.
            /// Default implementation does per-VID lookup.
            virtual void BatchGetVersions(const std::vector<SizeType>& vids, std::vector<uint8_t>& versions)
            {
                BatchGetVersions(vids, versions, VersionReadPolicy::UseCache);
            }

            virtual void BatchGetVersions(const std::vector<SizeType>& vids, std::vector<uint8_t>& versions, VersionReadPolicy policy)
            {
                versions.resize(vids.size());
                for (size_t i = 0; i < vids.size(); i++) {
                    if (vids[i] < 0 || vids[i] >= Count()) {
                        versions[i] = 0xfe;
                    } else {
                        versions[i] = GetVersion(vids[i], policy);
                    }
                }
            }
        };
    }
}

#endif // _SPTAG_COMMON_IVERSIONMAP_H_
