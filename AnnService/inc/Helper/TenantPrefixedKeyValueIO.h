// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_HELPER_TENANT_PREFIXED_KEYVALUEIO_H_
#define _SPTAG_HELPER_TENANT_PREFIXED_KEYVALUEIO_H_

#include "inc/Helper/KeyValueIO.h"
#include <memory>
#include <vector>
#include <string>
#include <cstdint>

namespace SPTAG
{
    namespace Helper
    {
        // Wraps an underlying KeyValueIO and transparently prefixes every SizeType key
        // with a per-tenant identifier in the high bits, so multiple tenants can share
        // a single physical KV store without colliding.
        //
        // Key layout:
        //   bits  31..16  (or top 16 of SizeType when LARGEVID): tenant id (up to 65536 tenants)
        //   bits  15..0   (or remaining bits): per-tenant head VID
        //
        // The wrapper never destroys the underlying store on ShutDown(); the owning
        // TenantIndexManager controls the lifetime of the shared KeyValueIO instance.
        class TenantPrefixedKeyValueIO : public KeyValueIO
        {
        public:
            static constexpr int kTenantIdBits = 16;
            static constexpr int kHeadVidBits = static_cast<int>(sizeof(SizeType) * 8) - kTenantIdBits;
            static constexpr SizeType kHeadVidMask = (static_cast<SizeType>(1) << kHeadVidBits) - 1;
            static constexpr int kMaxTenants = 1 << kTenantIdBits;

            TenantPrefixedKeyValueIO(std::shared_ptr<KeyValueIO> underlying, int tenantId)
                : m_underlying(std::move(underlying)),
                  m_tenantId(tenantId),
                  m_prefix(static_cast<SizeType>(tenantId) << kHeadVidBits)
            {
            }

            ~TenantPrefixedKeyValueIO() override = default;

            // Lifetime: do NOT shut down the shared underlying store; manager owns it.
            void ShutDown() override {}

            ErrorCode Get(const SizeType key, std::string* value,
                          const std::chrono::microseconds& timeout,
                          std::vector<Helper::AsyncReadRequest>* reqs) override
            {
                return m_underlying->Get(wrap(key), value, timeout, reqs);
            }

            ErrorCode Get(const SizeType key, Helper::PageBuffer<std::uint8_t>& value,
                          const std::chrono::microseconds& timeout,
                          std::vector<Helper::AsyncReadRequest>* reqs,
                          bool useCache = true) override
            {
                return m_underlying->Get(wrap(key), value, timeout, reqs, useCache);
            }

            ErrorCode MultiGet(const std::vector<SizeType>& keys,
                               std::vector<SPTAG::Helper::PageBuffer<std::uint8_t>>& values,
                               const std::chrono::microseconds& timeout,
                               std::vector<Helper::AsyncReadRequest>* reqs) override
            {
                std::vector<SizeType> wrapped;
                wrapped.reserve(keys.size());
                for (auto k : keys) wrapped.push_back(wrap(k));
                return m_underlying->MultiGet(wrapped, values, timeout, reqs);
            }

            ErrorCode MultiGet(const std::vector<SizeType>& keys,
                               std::vector<std::string>* values,
                               const std::chrono::microseconds& timeout,
                               std::vector<Helper::AsyncReadRequest>* reqs) override
            {
                std::vector<SizeType> wrapped;
                wrapped.reserve(keys.size());
                for (auto k : keys) wrapped.push_back(wrap(k));
                return m_underlying->MultiGet(wrapped, values, timeout, reqs);
            }

            ErrorCode Put(const SizeType key, const std::string& value,
                          const std::chrono::microseconds& timeout,
                          std::vector<Helper::AsyncReadRequest>* reqs) override
            {
                return m_underlying->Put(wrap(key), value, timeout, reqs);
            }

            // Upstream signature: checksum is a callback used by the implementation to
            // decide whether to merge based on existing value.
            ErrorCode Merge(const SizeType key, const std::string& value,
                            const std::chrono::microseconds& timeout,
                            std::vector<Helper::AsyncReadRequest>* reqs,
                            std::function<bool(const void* val, const int size)> checksum) override
            {
                return m_underlying->Merge(wrap(key), value, timeout, reqs, checksum);
            }

            ErrorCode Delete(SizeType key) override
            {
                return m_underlying->Delete(wrap(key));
            }

            ErrorCode DeleteRange(SizeType start, SizeType end) override
            {
                return m_underlying->DeleteRange(wrap(start), wrap(end));
            }

            void ForceCompaction() override { m_underlying->ForceCompaction(); }
            void GetStat() override { m_underlying->GetStat(); }
            int64_t GetNumBlocks() override { return m_underlying->GetNumBlocks(); }
            bool Available() override { return m_underlying && m_underlying->Available(); }

            // Upstream signature includes 'size' parameter.
            ErrorCode Check(const SizeType key, int size, std::vector<std::uint8_t>* visited) override
            {
                return m_underlying->Check(wrap(key), size, visited);
            }

            // Upstream signature is const.
            int64_t GetApproximateMemoryUsage() const override
            {
                return m_underlying->GetApproximateMemoryUsage();
            }

            // No-op: the shared underlying KV store is checkpointed by the
            // owning TenantIndexManager (see Save()), not per-tenant. Each
            // SPANN::Index::SaveIndex() invokes Checkpoint() to snapshot
            // its own DB, but for a tenant-prefixed shared store that
            // would create one duplicate directory per tenant.
            ErrorCode Checkpoint(std::string /*prefix*/) override
            {
                return ErrorCode::Success;
            }

            // Scan helpers are NOT prefix-filtered.
            ErrorCode StartToScan(SizeType& key, std::string* value) override
            {
                return m_underlying->StartToScan(key, value);
            }

            ErrorCode NextToScan(SizeType& key, std::string* value) override
            {
                return m_underlying->NextToScan(key, value);
            }

            int TenantId() const { return m_tenantId; }
            SizeType Prefix() const { return m_prefix; }
            std::shared_ptr<KeyValueIO> Underlying() const { return m_underlying; }

            static int ExtractTenantId(SizeType compositeKey)
            {
                return static_cast<int>((static_cast<uint64_t>(compositeKey) >> kHeadVidBits)
                                        & ((static_cast<uint64_t>(1) << kTenantIdBits) - 1));
            }
            static SizeType ExtractHeadVid(SizeType compositeKey)
            {
                return compositeKey & kHeadVidMask;
            }

        private:
            inline SizeType wrap(SizeType key) const { return (key & kHeadVidMask) | m_prefix; }

            std::shared_ptr<KeyValueIO> m_underlying;
            int m_tenantId;
            SizeType m_prefix;
        };
    }
}

#endif // _SPTAG_HELPER_TENANT_PREFIXED_KEYVALUEIO_H_
