// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef MULTITENANT
#include "inc/Helper/KeyValueIO.h"
#include <mutex>
#include <rocksdb/cache.h>
#include <rocksdb/db.h>
#include <tuple>

namespace SPTAG::SPANN
{
// Engine ownership is separate from index views. Views never close the engine.
class SharedRocksDB : public std::enable_shared_from_this<SharedRocksDB>
{
  public:
    static ErrorCode Open(const std::string &path, const std::vector<std::string> &families,
                          std::shared_ptr<SharedRocksDB> &result, bool create, std::shared_ptr<rocksdb::Cache> cache);
    ~SharedRocksDB();
    SharedRocksDB(const SharedRocksDB &) = delete;
    SharedRocksDB &operator=(const SharedRocksDB &) = delete;

    ErrorCode Get(const std::string &family, const std::string &key, std::string &value) const;
    ErrorCode Get(const std::string &family, const std::string &key, std::string &value, bool fillCache) const;
    ErrorCode MultiGet(const std::string &family, const std::vector<std::string> &keys,
                       std::vector<std::string> &values) const;
    ErrorCode MultiGet(const std::string &family, const std::vector<std::string> &keys,
                       std::vector<Helper::PageBuffer<std::uint8_t>> &values) const;
    ErrorCode Put(const std::string &family, const std::string &key, const std::string &value);
    ErrorCode Merge(const std::string &family, const std::string &key, const std::string &value);
    ErrorCode Delete(const std::string &family, const std::string &key);
    ErrorCode DeleteRange(const std::string &family, const std::string &begin, const std::string &end);
    ErrorCode Write(const std::vector<std::tuple<std::string, std::string, std::string>> &entries);
    ErrorCode Scan(const std::string &family, const std::string &prefix,
                   const std::function<ErrorCode(const std::string &, const std::string &)> &visitor) const;
    ErrorCode Checkpoint(const std::string &directory) const;
    ErrorCode FlushWAL();
    std::shared_ptr<Helper::KeyValueIO> PostingView(const std::string &tenant, std::uint64_t index);

    static std::string EncodeNumber(std::uint64_t value);
    static bool DecodeNumber(const std::string &value, std::uint64_t &number);
    static std::string Namespace(const std::string &tenant, std::uint64_t index);
    static std::string PrefixEnd(std::string prefix);

  private:
    SharedRocksDB() = default;
    rocksdb::ColumnFamilyHandle *Family(const std::string &name) const;
    std::unique_ptr<rocksdb::DB> m_db;
    std::vector<std::string> m_names;
    std::vector<rocksdb::ColumnFamilyHandle *> m_handles;
};
} // namespace SPTAG::SPANN
#endif
