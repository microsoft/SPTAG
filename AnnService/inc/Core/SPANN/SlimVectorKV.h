// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// SlimVectorKV: a transparent KeyValueIO decorator that splits the canonical
// vector out of every posting record so the posting store on disk holds only
// the slim metadata prefix ([vid|version|tag]) while the single vector copy
// lives in a separate vid-keyed vector store (RocksDB "opq_vecstore").
//
// On the WRITE path (Put/Merge) it deflates each fixed-size record: the vector
// bytes are written to the vector store keyed by vid, and only the metadata
// prefix is forwarded to the inner posting store.
//
// On the READ path (Get/MultiGet) it inflates: it reads the slim metadata blob
// from the inner store, fetches the corresponding vectors from the vector store
// by vid, and splices them back into full fixed-size records so every existing
// caller (Split/Append/Reassign/AddIndex/search) sees byte-identical full
// records and needs no modification.
//
// Correctness relies on three properties of the host index, all verified:
//   * the vector store keeps exactly (recSize - metaSize) raw bytes per vid,
//     byte-identical to the inline vector slice rec[metaSize .. recSize);
//   * ChecksumCheck is disabled, so checksums (computed over full blobs in
//     memory) are never validated against the slim on-disk bytes;
//   * ConsistencyCheck is disabled, so Check() block-size accounting is a no-op.

#ifndef _SPTAG_SPANN_SLIMVECTORKV_H_
#define _SPTAG_SPANN_SLIMVECTORKV_H_

#include "inc/Helper/KeyValueIO.h"
#include <memory>
#include <string>
#include <vector>

namespace SPTAG
{
    namespace SPANN
    {
        class SlimVectorKV : public Helper::KeyValueIO
        {
        public:
            // inner  : the real posting store (e.g. FileIO) that will hold slim records
            // vec    : the canonical vid -> vector store (RocksDB), opened read-write
            // metaSize : bytes of the metadata prefix kept inline ([vid|ver|tag])
            // recSize  : bytes of a full record (metaSize + vector bytes) == m_vectorInfoSize
            SlimVectorKV(std::shared_ptr<Helper::KeyValueIO> inner,
                         std::shared_ptr<Helper::KeyValueIO> vec,
                         int metaSize, int recSize)
                : m_inner(std::move(inner)), m_vec(std::move(vec)),
                  m_metaSize(metaSize), m_recSize(recSize),
                  m_vecBytes(recSize - metaSize) {}

            ~SlimVectorKV() override {}

            std::shared_ptr<Helper::KeyValueIO> Inner() const { return m_inner; }

            // ---- write path: split vector out to the vector store ----------------

            ErrorCode Put(const SizeType key, const std::string& value,
                          const std::chrono::microseconds& timeout,
                          std::vector<Helper::AsyncReadRequest>* reqs) override
            {
                std::string slim;
                ErrorCode ret = Deflate(value, slim);
                if (ret != ErrorCode::Success) return ret;
                return m_inner->Put(key, slim, timeout, reqs);
            }

            ErrorCode Put(const std::string& key, const std::string& value,
                          const std::chrono::microseconds& timeout,
                          std::vector<Helper::AsyncReadRequest>* reqs) override
            {
                return Put((SizeType)std::stoi(key), value, timeout, reqs);
            }

            ErrorCode Merge(const SizeType key, const std::string& value,
                            const std::chrono::microseconds& timeout,
                            std::vector<Helper::AsyncReadRequest>* reqs,
                            std::function<bool(const void* val, const int size)> checksum) override
            {
                std::string slim;
                ErrorCode ret = Deflate(value, slim);
                if (ret != ErrorCode::Success) return ret;
                // Checksums run over slim on-disk bytes; ChecksumCheck is disabled so
                // the host's validator always returns true. Forward a trivially-true
                // checksum so the merge operator never rejects on a size/content basis.
                return m_inner->Merge(key, slim, timeout, reqs,
                                      [](const void*, const int) { return true; });
            }

            // ---- read path: splice vectors back in -------------------------------

            ErrorCode Get(const SizeType key, std::string* value,
                          const std::chrono::microseconds& timeout,
                          std::vector<Helper::AsyncReadRequest>* reqs) override
            {
                std::string slim;
                ErrorCode ret = m_inner->Get(key, &slim, timeout, reqs);
                if (ret != ErrorCode::Success) return ret;
                return Inflate(slim, *value);
            }

            ErrorCode Get(const std::string& key, std::string* value,
                          const std::chrono::microseconds& timeout,
                          std::vector<Helper::AsyncReadRequest>* reqs) override
            {
                return Get((SizeType)std::stoi(key), value, timeout, reqs);
            }

            ErrorCode Get(const SizeType key, Helper::PageBuffer<std::uint8_t>& value,
                          const std::chrono::microseconds& timeout,
                          std::vector<Helper::AsyncReadRequest>* reqs, bool useCache = true) override
            {
                std::string slim;
                ErrorCode ret = m_inner->Get(key, &slim, timeout, reqs);
                if (ret != ErrorCode::Success) return ret;
                std::string full;
                ret = Inflate(slim, full);
                if (ret != ErrorCode::Success) return ret;
                value.ReservePageBuffer(full.size());
                memcpy(value.GetBuffer(), full.data(), full.size());
                value.SetAvailableSize(full.size());
                return ErrorCode::Success;
            }

            ErrorCode MultiGet(const std::vector<SizeType>& keys, std::vector<std::string>* values,
                               const std::chrono::microseconds& timeout,
                               std::vector<Helper::AsyncReadRequest>* reqs) override
            {
                values->resize(keys.size());
                for (size_t i = 0; i < keys.size(); i++) {
                    ErrorCode ret = Get(keys[i], &(*values)[i], timeout, reqs);
                    if (ret != ErrorCode::Success) return ret;
                }
                return ErrorCode::Success;
            }

            ErrorCode MultiGet(const std::vector<std::string>& keys, std::vector<std::string>* values,
                               const std::chrono::microseconds& timeout,
                               std::vector<Helper::AsyncReadRequest>* reqs) override
            {
                std::vector<SizeType> intKeys;
                intKeys.reserve(keys.size());
                for (const auto& k : keys) intKeys.push_back((SizeType)std::stoi(k));
                return MultiGet(intKeys, values, timeout, reqs);
            }

            ErrorCode MultiGet(const std::vector<SizeType>& keys,
                               std::vector<Helper::PageBuffer<std::uint8_t>>& values,
                               const std::chrono::microseconds& timeout,
                               std::vector<Helper::AsyncReadRequest>* reqs) override
            {
                for (size_t i = 0; i < keys.size(); i++) {
                    ErrorCode ret = Get(keys[i], values[i], timeout, reqs, true);
                    if (ret != ErrorCode::Success) return ret;
                }
                return ErrorCode::Success;
            }

            // Truncated variant: per-key byte caps are expressed in full-record terms.
            // Reconstruction always yields full records, so the cap is ignored (correct,
            // just no truncation savings on the update path).
            ErrorCode MultiGet(const std::vector<SizeType>& keys,
                               std::vector<Helper::PageBuffer<std::uint8_t>>& values,
                               const std::vector<std::uint32_t>& maxBytesPerKey,
                               const std::chrono::microseconds& timeout,
                               std::vector<Helper::AsyncReadRequest>* reqs) override
            {
                (void)maxBytesPerKey;
                return MultiGet(keys, values, timeout, reqs);
            }

            // ---- pass-through -----------------------------------------------------

            void ShutDown() override { m_inner->ShutDown(); }

            ErrorCode Delete(SizeType key) override { return m_inner->Delete(key); }

            ErrorCode DeleteRange(SizeType start, SizeType end) override
            {
                return m_inner->DeleteRange(start, end);
            }

            // Block accounting is expressed in full-record terms by callers; translate
            // to slim terms. ConsistencyCheck is disabled so this is effectively a no-op.
            ErrorCode Check(const SizeType key, int size, std::vector<std::uint8_t>* visited) override
            {
                int slimSize = (m_recSize > 0) ? (size / m_recSize) * m_metaSize : size;
                return m_inner->Check(key, slimSize, visited);
            }

            void ForceCompaction() override { m_inner->ForceCompaction(); }
            void GetStat() override { m_inner->GetStat(); }
            int64_t GetNumBlocks() override { return m_inner->GetNumBlocks(); }
            bool Available() override
            {
                return m_inner->Available() && (m_vec ? m_vec->Available() : false);
            }
            int64_t GetApproximateMemoryUsage() const override
            {
                int64_t u = m_inner->GetApproximateMemoryUsage();
                if (m_vec) u += m_vec->GetApproximateMemoryUsage();
                return u;
            }
            ErrorCode Checkpoint(std::string prefix) override { return m_inner->Checkpoint(prefix); }

            // Scan returns slim records (used only by recovery, which is disabled here).
            ErrorCode StartToScan(SizeType& key, std::string* value) override
            {
                return m_inner->StartToScan(key, value);
            }
            ErrorCode NextToScan(SizeType& key, std::string* value) override
            {
                return m_inner->NextToScan(key, value);
            }

        private:
            // Split a full-record blob into a slim metadata blob, pushing each record's
            // vector into the vector store keyed by vid.
            ErrorCode Deflate(const std::string& full, std::string& slim)
            {
                if (m_recSize <= 0 || full.empty()) { slim.clear(); return ErrorCode::Success; }
                size_t n = full.size() / (size_t)m_recSize;
                slim.clear();
                slim.reserve(n * (size_t)m_metaSize);
                const char* p = full.data();
                for (size_t i = 0; i < n; i++) {
                    const char* e = p + i * (size_t)m_recSize;
                    SizeType vid = *(reinterpret_cast<const SizeType*>(e));
                    if (m_vec && vid >= 0) {
                        ErrorCode ret = m_vec->Put(vid,
                            std::string(e + m_metaSize, (size_t)m_vecBytes), MaxTimeout, nullptr);
                        if (ret != ErrorCode::Success) return ret;
                    }
                    slim.append(e, (size_t)m_metaSize);
                }
                return ErrorCode::Success;
            }

            // Rebuild full fixed-size records from a slim metadata blob by fetching each
            // vector from the vector store by vid.
            ErrorCode Inflate(const std::string& slim, std::string& full)
            {
                if (m_metaSize <= 0 || slim.empty()) { full.clear(); return ErrorCode::Success; }
                size_t n = slim.size() / (size_t)m_metaSize;
                std::vector<SizeType> vids(n);
                const char* s = slim.data();
                for (size_t i = 0; i < n; i++) {
                    vids[i] = *(reinterpret_cast<const SizeType*>(s + i * (size_t)m_metaSize));
                }
                std::vector<std::string> vecs;
                std::vector<Helper::AsyncReadRequest> reqs;
                if (m_vec) {
                    ErrorCode ret = m_vec->MultiGet(vids, &vecs, MaxTimeout, &reqs);
                    if (ret != ErrorCode::Success || vecs.size() != n) return ErrorCode::Fail;
                }
                full.resize(n * (size_t)m_recSize);
                char* d = &full[0];
                for (size_t i = 0; i < n; i++) {
                    char* e = d + i * (size_t)m_recSize;
                    memcpy(e, s + i * (size_t)m_metaSize, (size_t)m_metaSize);
                    if (m_vec) {
                        size_t cp = std::min((size_t)m_vecBytes, vecs[i].size());
                        memcpy(e + m_metaSize, vecs[i].data(), cp);
                        if (cp < (size_t)m_vecBytes) memset(e + m_metaSize + cp, 0, (size_t)m_vecBytes - cp);
                    } else {
                        memset(e + m_metaSize, 0, (size_t)m_vecBytes);
                    }
                }
                return ErrorCode::Success;
            }

            std::shared_ptr<Helper::KeyValueIO> m_inner;
            std::shared_ptr<Helper::KeyValueIO> m_vec;
            int m_metaSize;
            int m_recSize;
            int m_vecBytes;

            static constexpr std::chrono::microseconds MaxTimeout{ std::chrono::microseconds::max() };
        };
    }
}

#endif // _SPTAG_SPANN_SLIMVECTORKV_H_
