// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_EXTRAROCKSDBCONTROLLER_H_
#define _SPTAG_SPANN_EXTRAROCKSDBCONTROLLER_H_

#include "inc/Helper/KeyValueIO.h"
#include "inc/Helper/StringConvert.h"

#include "rocksdb/db.h"
#include "rocksdb/filter_policy.h"
#include "rocksdb/rate_limiter.h"
#include "rocksdb/slice.h"
#include "rocksdb/options.h"
#include "rocksdb/merge_operator.h"
#include "rocksdb/table.h"
#include "rocksdb/utilities/checkpoint.h"

#include <map>
#include <cmath>
#include <cstdlib>
#include <climits>
#include <future>

namespace SPTAG::SPANN
{
    class RocksDBIO : public Helper::KeyValueIO
    {
        class AnnMergeOperator : public rocksdb::MergeOperator
        {
        public:
            bool FullMergeV2(const rocksdb::MergeOperator::MergeOperationInput& merge_in,
                rocksdb::MergeOperator::MergeOperationOutput* merge_out) const override
            {
                size_t length = (merge_in.existing_value)->size();
                for (const rocksdb::Slice& s : merge_in.operand_list) {
                    length += s.size();
                }
                (merge_out->new_value).resize(length);
                memcpy((char*)((merge_out->new_value).c_str()), 
                    (merge_in.existing_value)->data(), (merge_in.existing_value)->size());
                size_t start = (merge_in.existing_value)->size();
                for (const rocksdb::Slice& s : merge_in.operand_list) {
                    memcpy((char*)((merge_out->new_value).c_str() + start), s.data(), s.size());
                    start += s.size();
                }
                return true;
            }

            bool PartialMergeMulti(const rocksdb::Slice& key,
                const std::deque<rocksdb::Slice>& operand_list,
                std::string* new_value, rocksdb::Logger* logger) const override
            {
                size_t length = 0;
                for (const rocksdb::Slice& s : operand_list) {
                    length += s.size();
                }
                new_value->resize(length);
                size_t start = 0;
                for (const rocksdb::Slice& s : operand_list) {
                    memcpy((char*)(new_value->c_str() + start), s.data(), s.size());
                    start += s.size();
                }
                return true;
            }

            const char* Name() const override {
                return "AnnMergeOperator";
            }
        };

    public:
        RocksDBIO(const char* filePath, bool usdDirectIO, bool wal = false, bool recovery = false, bool readOnly = false) {
            dbPath = std::string(filePath);
            //dbOptions.statistics = rocksdb::CreateDBStatistics();
            dbOptions.create_if_missing = true;
            if (!wal) {
                dbOptions.IncreaseParallelism();
                dbOptions.OptimizeLevelStyleCompaction();
                dbOptions.merge_operator.reset(new AnnMergeOperator);
                // dbOptions.statistics = rocksdb::CreateDBStatistics();

                // SST file size options
                dbOptions.target_file_size_base = 128UL * 1024 * 1024;
                dbOptions.target_file_size_multiplier = 2;
                dbOptions.max_bytes_for_level_base = 16 * 1024UL * 1024 * 1024;
                dbOptions.max_bytes_for_level_multiplier = 4;
                dbOptions.max_subcompactions = 16;
                dbOptions.num_levels = 4;
                dbOptions.level0_file_num_compaction_trigger = 1;
                dbOptions.level_compaction_dynamic_level_bytes = false;
                dbOptions.write_buffer_size = 16UL * 1024 * 1024;

                // rate limiter options
                // dbOptions.rate_limiter.reset(rocksdb::NewGenericRateLimiter(100UL << 20));

                // blob options
                dbOptions.enable_blob_files = true;
                dbOptions.min_blob_size = 64;
                dbOptions.blob_file_size = 8UL << 30;
                dbOptions.blob_compression_type = rocksdb::CompressionType::kNoCompression;
                dbOptions.enable_blob_garbage_collection = true;
                dbOptions.compaction_pri = rocksdb::CompactionPri::kRoundRobin;
                dbOptions.blob_garbage_collection_age_cutoff = 0.4;
                // dbOptions.blob_garbage_collection_force_threshold = 0.5;

                // SPTAG_ROCKSDB_DIRECT_IO=1: bypass the OS page cache for SST/blob
                // reads (O_DIRECT). Forces every posting/vector fetch to hit the
                // device regardless of dataset size, simulating a working-set >> RAM
                // regime for an apples-to-apples comparison vs O_DIRECT baselines.
                if (const char* dio = std::getenv("SPTAG_ROCKSDB_DIRECT_IO")) {
                    if (dio[0] == '1') {
                        dbOptions.use_direct_reads = true;
                        dbOptions.use_direct_io_for_flush_and_compaction = true;
                    }
                }
                // dbOptions.blob_cache = rocksdb::NewLRUCache(5UL << 30);
                // dbOptions.prepopulate_blob_cache = rocksdb::PrepopulateBlobCache::kFlushOnly;

                // dbOptions.env;
                // dbOptions.sst_file_manager = std::shared_ptr<rocksdb::SstFileManager>(rocksdb::NewSstFileManager(dbOptions.env));
                // dbOptions.sst_file_manager->SetStatisticsPtr(dbOptions.statistics);

                // compression options
                // dbOptions.compression = rocksdb::CompressionType::kLZ4Compression;
                // dbOptions.bottommost_compression = rocksdb::CompressionType::kZSTD;

                // block cache options
                rocksdb::BlockBasedTableOptions table_options;
                // table_options.block_cache = rocksdb::NewSimCache(rocksdb::NewLRUCache(1UL << 30), (8UL << 30), -1);
                // Block cache mainly serves cross-tenant block reuse; within-tenant filtering
                // runs off the resident (mmap'd) PQ codes and is unaffected by its size.
                // Default to a modest 64MB to keep the memory floor low; override via env.
                size_t blockCacheMB = 64;
                if (const char* bcEnv = std::getenv("SPTAG_ROCKSDB_BLOCK_CACHE_MB")) {
                    long v = std::atol(bcEnv);
                    if (v >= 0) blockCacheMB = static_cast<size_t>(v);
                }
                if (blockCacheMB == 0) {
                    table_options.no_block_cache = true;
                } else {
                    table_options.block_cache = rocksdb::NewLRUCache(blockCacheMB << 20);
                }

                // filter options
                table_options.filter_policy.reset(rocksdb::NewBloomFilterPolicy(10, true));
                table_options.optimize_filters_for_memory = true;

                dbOptions.table_factory.reset(rocksdb::NewBlockBasedTableFactory(table_options));
            }

            if (usdDirectIO) {
                dbOptions.use_direct_io_for_flush_and_compaction = true;
                dbOptions.use_direct_reads = true;
            }

            auto s = readOnly ? rocksdb::DB::OpenForReadOnly(dbOptions, dbPath, &db)
                              : rocksdb::DB::Open(dbOptions, dbPath, &db);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPFresh: New Rocksdb%s: %s\n", readOnly ? "(RO)" : "", filePath);
            if (s != rocksdb::Status::OK()) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "\e[0;31mRocksdb Open Error\e[0m: %s\n", s.getState());
                ShutDown();
            }
        }

        ~RocksDBIO() override {
            /*
            std::string stats;
            db->GetProperty("rocksdb.stats", &stats);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "RocksDB Status: %s\n%s", dbPath.c_str(),stats.c_str());
            */
            ShutDown();
        }

        void ShutDown() override {
            if (db == nullptr) return;

            db->Close();
            //DestroyDB(dbPath, dbOptions);
            delete db;
	        db = nullptr;
        }

        bool Available() override
        {
            return (db != nullptr);
        }

        // ReadOptions for MultiGet. By default RocksDB's MultiGet (async_io=false)
        // reads the requested keys/blobs ONE AT A TIME -- so a batched rerank
        // MultiGet over L scattered survivors is effectively SERIAL inside RocksDB.
        // async_io=true lets RocksDB issue the underlying file reads in parallel
        // (ReadAsync / io_uring on the posix FileSystem), giving real queue depth
        // -- the parallel-rerank behavior the batched MultiGet was supposed to have.
        // Env SPTAG_ROCKSDB_ASYNC_MULTIGET=0 reverts to the legacy serial path.
        static rocksdb::ReadOptions MultiGetReadOptions() {
            rocksdb::ReadOptions ro;
            static const bool s_async = []() {
                const char* e = std::getenv("SPTAG_ROCKSDB_ASYNC_MULTIGET");
                return (e == nullptr) || (std::atoi(e) != 0);
            }();
            ro.async_io = s_async;
            return ro;
        }

        ErrorCode Get(const std::string& key, std::string* value, const std::chrono::microseconds& timeout, std::vector<Helper::AsyncReadRequest>* reqs) override {
            auto s = db->Get(rocksdb::ReadOptions(), key, value);
            if (s == rocksdb::Status::OK()) {
                return ErrorCode::Success;
            }
            else {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "\e[0;31mError in Get\e[0m: %s, key: %d\n", s.getState(), *((SizeType*)(key.data())));
                return ErrorCode::Fail;
            }
        }

        ErrorCode Get(SizeType key, std::string* value, const std::chrono::microseconds& timeout, std::vector<Helper::AsyncReadRequest>* reqs) override {
            std::string k((char*)&key, sizeof(SizeType));
            return Get(k, value, timeout, reqs);
        }

        ErrorCode MultiGet(const std::vector<SizeType>& keys, std::vector<Helper::PageBuffer<std::uint8_t>>& values,
            const std::chrono::microseconds& timeout, std::vector<Helper::AsyncReadRequest>* reqs) override {
            size_t num_keys = keys.size();

            rocksdb::Slice* slice_keys = new rocksdb::Slice[num_keys];
            rocksdb::PinnableSlice* slice_values = new rocksdb::PinnableSlice[num_keys];
            rocksdb::Status* statuses = new rocksdb::Status[num_keys];

            // RocksDB uses Slice for keys, which doen't own the memory.
            // We need to convert SizeType keys to string slices.

            std::vector<std::string> str_keys(num_keys);

            for (int i = 0; i < num_keys; i++) {
                str_keys[i] = std::string((char*)&keys[i], sizeof(SizeType));
            }

            for (int i = 0; i < num_keys; i++) {
                slice_keys[i] = rocksdb::Slice(str_keys[i]);
            }

            db->MultiGet(MultiGetReadOptions(), db->DefaultColumnFamily(),
                num_keys, slice_keys, slice_values, statuses);

            for (int i = 0; i < num_keys; i++) {
                if (statuses[i] != rocksdb::Status::OK()) {
                    delete[] slice_keys;
                    delete[] slice_values;
                    delete[] statuses;
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "\e[0;31mError in MultiGet\e[0m: %s, key: %d\n", statuses[i].getState(), keys[i]);
                    return ErrorCode::Fail;
                }
                memcpy(values[i].GetBuffer(), slice_values[i].data(), slice_values[i].size());
                values[i].SetAvailableSize(slice_values[i].size());
            }

            delete[] slice_keys;
            delete[] slice_values;
            delete[] statuses;
            return ErrorCode::Success;
        }

        ErrorCode MultiGet(const std::vector<std::string>& keys, std::vector<std::string>* values, 
            const std::chrono::microseconds &timeout, std::vector<Helper::AsyncReadRequest>* reqs) override {
            size_t num_keys = keys.size();

            rocksdb::Slice* slice_keys = new rocksdb::Slice[num_keys];
            rocksdb::PinnableSlice* slice_values = new rocksdb::PinnableSlice[num_keys];
            rocksdb::Status* statuses = new rocksdb::Status[num_keys];

            for (int i = 0; i < num_keys; i++) {
                slice_keys[i] = rocksdb::Slice(keys[i]);
            }

            db->MultiGet(MultiGetReadOptions(), db->DefaultColumnFamily(),
                num_keys, slice_keys, slice_values, statuses);

            for (int i = 0; i < num_keys; i++) {
                if (statuses[i] != rocksdb::Status::OK()) {
                    delete[] slice_keys;
                    delete[] slice_values;
                    delete[] statuses;
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "\e[0;31mError in MultiGet\e[0m: %s, key: %d\n", statuses[i].getState(), *((SizeType*)(keys[i].data())));
                    return ErrorCode::Fail;
                }
                values->push_back(slice_values[i].ToString());
            }

            delete[] slice_keys;
            delete[] slice_values;
            delete[] statuses;
            return ErrorCode::Success;
        }

        ErrorCode MultiGet(const std::vector<SizeType>& keys, std::vector<std::string>* values, 
            const std::chrono::microseconds &timeout, std::vector<Helper::AsyncReadRequest>* reqs) override {
            std::vector<std::string> str_keys;

            for (const auto& key : keys) {
                str_keys.emplace_back((char*)(&key), sizeof(SizeType));
            }

            return MultiGet(str_keys, values, timeout, reqs);
        }

        ErrorCode Put(const std::string& key, const std::string& value, const std::chrono::microseconds& timeout, std::vector<Helper::AsyncReadRequest>* reqs) override {
            auto s = db->Put(rocksdb::WriteOptions(), key, value);
            if (s == rocksdb::Status::OK()) {
                return ErrorCode::Success;
            }
            else {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "\e[0;31mError in Put\e[0m: %s, key: %d\n", s.getState(), *((SizeType*)(key.data())));
                return ErrorCode::Fail;
            }
        }

        ErrorCode Put(const SizeType key, const std::string& value, const std::chrono::microseconds& timeout, std::vector<Helper::AsyncReadRequest>* reqs) override {
            std::string k((char*)&key, sizeof(SizeType));
            return Put(k, value, timeout, reqs);
        }

        ErrorCode Merge(const SizeType key, const std::string &value, const std::chrono::microseconds &timeout,
                        std::vector<Helper::AsyncReadRequest> *reqs,
                        std::function<bool(const void *val, const int size)> checksum) override
        {
            if (value.empty()) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Error! empty append posting!\n");
            }
            std::string k((char*)&key, sizeof(SizeType));
            auto s = db->Merge(rocksdb::WriteOptions(), k, value);
            if (s == rocksdb::Status::OK()) {
                return ErrorCode::Success;
            }
            else {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "\e[0;31mError in Merge\e[0m: %s, key: %d\n", s.getState(), key);
                return ErrorCode::Fail;
            }
        }

        ErrorCode StartToScan(SizeType& key, std::string* value) {
            it = db->NewIterator(rocksdb::ReadOptions());
            it->SeekToFirst();
            if (!it->Valid()) return ErrorCode::Fail;
            key = *((SizeType*)(it->key().ToString().c_str()));
            *value = it->value().ToString();
            return ErrorCode::Success;
        }

        ErrorCode NextToScan(SizeType& key, std::string* value) {
            it->Next();
            if (!it->Valid()) return ErrorCode::Fail;
            key = *((SizeType*)(it->key().ToString().c_str()));
            *value = it->value().ToString();
            return ErrorCode::Success;
        }

        ErrorCode Delete(SizeType key) override {
            std::string k((char*)&key, sizeof(SizeType));
            auto s = db->Delete(rocksdb::WriteOptions(), k);
            if (s == rocksdb::Status::OK()) {
                return ErrorCode::Success;
            }
            else {
                return ErrorCode::Fail;
            }
        }

        ErrorCode DeleteRange(SizeType start, SizeType end) override {
            std::string string_start((char*)&start, sizeof(SizeType));
            rocksdb::Slice slice_start = rocksdb::Slice(string_start);
            std::string string_end((char*)&end, sizeof(SizeType));
            rocksdb::Slice slice_end = rocksdb::Slice(string_end);
            auto s = db->DeleteRange(rocksdb::WriteOptions(), db->DefaultColumnFamily(), slice_start, slice_end);
            if (s == rocksdb::Status::OK()) {
                return ErrorCode::Success;
            }
            else {
                return ErrorCode::Fail;
            }
        }

        void ForceCompaction() {
            /*
            std::string stats;
            db->GetProperty("rocksdb.stats", &stats);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "RocksDB Status:\n%s", stats.c_str());
            */
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start Compaction\n");
            auto s = db->CompactRange(rocksdb::CompactRangeOptions(), nullptr, nullptr);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Finish Compaction\n");

            if (s != rocksdb::Status::OK()) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "\e[0;31mRocksdb Compact Error\e[0m: %s\n", s.getState());
            }
            /*
            db->GetProperty("rocksdb.stats", &stats);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "RocksDB Status:\n%s", stats.c_str());
            */
        }

        void GetStat() {
            if (dbOptions.statistics != nullptr)
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "RocksDB statistics:\n %s\n", dbOptions.statistics->ToString().c_str());
            else
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "DB statistics not set!\n");
        }

        ErrorCode Checkpoint(std::string prefix) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "RocksDB: checkpoint\n");
            rocksdb::Checkpoint* checkpoint_ptr;
            rocksdb::Checkpoint::Create(db, &checkpoint_ptr);
            std::string filename = prefix + FolderSep + dbPath.substr(dbPath.find_last_of(FolderSep) + 1);
            checkpoint_ptr->CreateCheckpoint(filename);
            return ErrorCode::Success;
        }

    private:
        std::string dbPath;
        rocksdb::DB* db{};
        rocksdb::Options dbOptions;
        rocksdb::Iterator* it;
    };
}
#endif // _SPTAG_SPANN_EXTRAROCKSDBCONTROLLER_H_
