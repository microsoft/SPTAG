// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_EXTRADBSEARCHER_H_
#define _SPTAG_SPANN_EXTRADBSEARCHER_H_

#include "inc/Helper/VectorSetReader.h"
#include "inc/Helper/AsyncFileReader.h"
#include "IExtraSearcher.h"
#include "ExtraFullGraphSearcher.h"
#include "../Common/TruthSet.h"
#include "inc/Helper/KeyValueIO.h"

#include "rocksdb/db.h"
#include "rocksdb/filter_policy.h"
#include "rocksdb/rate_limiter.h"
#include "rocksdb/slice.h"
#include "rocksdb/options.h"
#include "rocksdb/merge_operator.h"
#include "rocksdb/table.h"

#include <map>
#include <cmath>
#include <climits>
#include <future>

// enable rocksdb io_uring
extern "C" bool RocksDbIOUringEnable() { return true; }

namespace SPTAG::SPANN
{
    class RocksDBIO : public Helper::KeyValueIO
    {
    public:
        RocksDBIO() = default;

        ~RocksDBIO() override {
            /*
            std::string stats;
            db->GetProperty("rocksdb.stats", &stats);
            LOG(Helper::LogLevel::LL_Info, "RocksDB Status: %s\n%s", dbPath.c_str(),stats.c_str());
            */
            db->Close();
            // DestroyDB(dbPath, dbOptions);
            delete db;
        }

        bool Initialize(const char* filePath, bool usdDirectIO, bool wal = false) override
        {
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
                dbOptions.blob_garbage_collection_age_cutoff = 0.02;
                // dbOptions.blob_garbage_collection_force_threshold = 0.5;
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
                table_options.block_cache = rocksdb::NewLRUCache(3UL << 30);
                // table_options.no_block_cache = true;

                // filter options
                table_options.filter_policy.reset(rocksdb::NewBloomFilterPolicy(10, true));
                table_options.optimize_filters_for_memory = true;

                dbOptions.table_factory.reset(rocksdb::NewBlockBasedTableFactory(table_options));
            }
            
            if (usdDirectIO) {
                dbOptions.use_direct_io_for_flush_and_compaction = true;
                dbOptions.use_direct_reads = true;
            }

            auto s = rocksdb::DB::Open(dbOptions, dbPath, &db);
            LOG(Helper::LogLevel::LL_Info, "SPFresh: New Rocksdb: %s\n", filePath);
            if (s != rocksdb::Status::OK()) {
                LOG(Helper::LogLevel::LL_Error, "\e[0;31mRocksdb Open Error\e[0m: %s\n", s.getState());
            }
            return s == rocksdb::Status::OK();
        }

        void ShutDown() override {
            db->Close();
            DestroyDB(dbPath, dbOptions);
            delete db;
        }

        ErrorCode Get(const std::string& key, std::string* value) override {
            auto s = db->Get(rocksdb::ReadOptions(), key, value);
            if (s == rocksdb::Status::OK()) {
                return ErrorCode::Success;
            } else {
                auto key_int = Helper::Convert::Unserialize<SizeType>(key);
                LOG(Helper::LogLevel::LL_Error, "\e[0;31mError in Get\e[0m: %s, key: %d\n", s.getState(), *(key_int.get()));
                return ErrorCode::Fail;
            }
        }

        ErrorCode Get(SizeType key, std::string* value) override {
            return Get(Helper::Convert::Serialize<SizeType>(&key), value);
        }

        ErrorCode MultiGet(const std::vector<std::string>& keys, std::vector<std::string>* values) {
            size_t num_keys = keys.size();

            rocksdb::Slice* slice_keys = new rocksdb::Slice[num_keys];
            rocksdb::PinnableSlice* slice_values = new rocksdb::PinnableSlice[num_keys];
            rocksdb::Status* statuses = new rocksdb::Status[num_keys];

            for (int i = 0; i < num_keys; i++) {
                slice_keys[i] = rocksdb::Slice(keys[i]);
            }

            db->MultiGet(rocksdb::ReadOptions(), db->DefaultColumnFamily(),
                            num_keys, slice_keys, slice_values, statuses);

            for (int i = 0; i < num_keys; i++) {
                if (statuses[i] != rocksdb::Status::OK()) {
                    delete [] slice_keys;
                    delete [] slice_values;
                    delete [] statuses;
                    auto key = Helper::Convert::Unserialize<SizeType>(keys[i]);
                    LOG(Helper::LogLevel::LL_Error, "\e[0;31mError in MultiGet\e[0m: %s, key: %d\n", statuses[i].getState(), *(key.get()));
                    return ErrorCode::Fail;
                }
                values->push_back(slice_values[i].ToString());
            }

            delete [] slice_keys;
            delete [] slice_values;
            delete [] statuses;
            return ErrorCode::Success;
        }

        ErrorCode MultiGet(const std::vector<SizeType>& keys, std::vector<std::string>* values) {
            std::vector<std::string> str_keys;

            for (const auto &key : keys) {
                str_keys.push_back(Helper::Convert::Serialize<SizeType>(&key));
            }

            return MultiGet(str_keys, values);
        }

        ErrorCode Put(const std::string& key, const std::string& value) override {
            auto s = db->Put(rocksdb::WriteOptions(), key, value);
            if (s == rocksdb::Status::OK()) {
                return ErrorCode::Success;
            } else {
                auto key_int = Helper::Convert::Unserialize<SizeType>(key);
                LOG(Helper::LogLevel::LL_Error, "\e[0;31mError in Put\e[0m: %s, key: %d\n", s.getState(), *(key_int.get()));
                return ErrorCode::Fail;
            }
        }

        ErrorCode Put(SizeType key, const std::string& value) override {
            return Put(Helper::Convert::Serialize<SizeType>(&key), value);
        }

        ErrorCode Put(SizeType key, SizeType id, const void* vector, SizeType dim) override {
            using Helper::Convert::Serialize;
            std::string posting(Serialize<SizeType>(&id) + Serialize<SizeType>(vector, dim));
            return Put(key, posting);
        }

        class AnnMergeOperator : public rocksdb::AssociativeMergeOperator
        {
        public:
            bool Merge(const rocksdb::Slice& key, const rocksdb::Slice* existing_value,
                               const rocksdb::Slice& value, std::string* new_value,
                               rocksdb::Logger* logger) const override {
                std::string newPosting;
                if(existing_value)
                {
                    newPosting += (*existing_value).ToString();
                    newPosting += value.ToString();
                } else
                {
                    newPosting += value.ToString();
                }
                *new_value = newPosting;
                return true;
            }
            const char* Name() const override {
                return "AnnMergeOperator";
            }
        };

        ErrorCode Merge(SizeType key, const std::string& value) {
            if (value.empty()) {
                LOG(Helper::LogLevel::LL_Error, "Error! empty append posting!\n");
            }
            auto s = db->Merge(rocksdb::WriteOptions(), Helper::Convert::Serialize<int>(&key, 1), value);
            if (s == rocksdb::Status::OK()) {
                return ErrorCode::Success;
            } else {
                LOG(Helper::LogLevel::LL_Error, "\e[0;31mError in Merge\e[0m: %s, key: %d\n", s.getState(), key);
                return ErrorCode::Fail;
            }
        }

        ErrorCode Delete(SizeType key) override {
            auto s = db->Delete(rocksdb::WriteOptions(), Helper::Convert::Serialize<int>(&key, 1));
            if (s == rocksdb::Status::OK()) {
                return ErrorCode::Success;
            } else {
                return ErrorCode::Fail;
            }
        }

        void ForceCompaction() {
            /*
            std::string stats;
            db->GetProperty("rocksdb.stats", &stats);
            LOG(Helper::LogLevel::LL_Info, "RocksDB Status:\n%s", stats.c_str());
            */
            LOG(Helper::LogLevel::LL_Info, "Start Compaction\n");
            auto s = db->CompactRange(rocksdb::CompactRangeOptions(), nullptr, nullptr);
            LOG(Helper::LogLevel::LL_Info, "Finish Compaction\n");

            if (s != rocksdb::Status::OK()) {
                LOG(Helper::LogLevel::LL_Error, "\e[0;31mRocksdb Compact Error\e[0m: %s\n", s.getState());
            }
            /*
            db->GetProperty("rocksdb.stats", &stats);
            LOG(Helper::LogLevel::LL_Info, "RocksDB Status:\n%s", stats.c_str());
            */
        }

        void GetDBStat() {
            if (dbOptions.statistics != nullptr)
                LOG(Helper::LogLevel::LL_Info, "RocksDB statistics:\n %s\n", dbOptions.statistics->ToString().c_str());
            else 
                LOG(Helper::LogLevel::LL_Info, "DB statistics not set!\n");
        }

    private:
        std::string dbPath;
        rocksdb::DB* db{};
        rocksdb::Options dbOptions;
    };

    template <typename ValueType>
    class ExtraRocksDBController : public IExtraSearcher
    {
    private:
        RocksDBIO db;
        std::atomic_uint64_t m_postingNum{};
    public:
        ExtraRocksDBController(const char* dbPath, int dim, int vectorlimit, bool useDirectIO, float searchLatencyHardLimit) { 
            db.Initialize(dbPath, useDirectIO); 
            // m_metaDataSize = sizeof(int) + sizeof(uint8_t) + sizeof(float);
            m_metaDataSize = sizeof(int) + sizeof(uint8_t);
            m_vectorInfoSize = dim * sizeof(ValueType) + m_metaDataSize;
            m_postingSizeLimit = vectorlimit;
            LOG(Helper::LogLevel::LL_Info, "Posting size limit: %d\n", m_postingSizeLimit);
            m_hardLatencyLimit = searchLatencyHardLimit;
        }

        ~ExtraRocksDBController() override = default;

        bool LoadIndex(Options& p_opt) override {
            return true;
        }

        virtual void SearchIndex(ExtraWorkSpace* p_exWorkSpace,
                QueryResult& p_queryResults,
                std::shared_ptr<VectorIndex> p_index,
                SearchStats* p_stats, const COMMON::VersionLabel& m_versionMap, std::set<int>* truth, std::map<int, std::set<int>>* found) override
            {
            auto exStart = std::chrono::high_resolution_clock::now();

            const auto postingListCount = static_cast<uint32_t>(p_exWorkSpace->m_postingIDs.size());

            p_exWorkSpace->m_deduper.clear();

            auto exSetUpEnd = std::chrono::high_resolution_clock::now();

            p_stats->m_exSetUpLatency = ((double)std::chrono::duration_cast<std::chrono::microseconds>(exSetUpEnd - exStart).count()) / 1000;

            COMMON::QueryResultSet<ValueType>& queryResults = *((COMMON::QueryResultSet<ValueType>*)&p_queryResults);

            int diskRead = 0;
            int diskIO = 0;
            int listElements = 0;

            double compLatency = 0;
            double readLatency = 0;

            std::vector<std::string> postingLists;

            auto readStart = std::chrono::high_resolution_clock::now();
            db.MultiGet(p_exWorkSpace->m_postingIDs, &postingLists);
            auto readEnd = std::chrono::high_resolution_clock::now();

            diskIO+=postingListCount;

            readLatency += ((double)std::chrono::duration_cast<std::chrono::microseconds>(readEnd - readStart).count());

            for (uint32_t pi = 0; pi < postingListCount; ++pi) {
                auto curPostingID = p_exWorkSpace->m_postingIDs[pi];
                std::string &postingList = postingLists[pi];

                int vectorNum = postingList.size() / m_vectorInfoSize;

                diskRead += postingList.size();
                listElements += vectorNum;

                auto compStart = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < vectorNum; i++) {
                    char* vectorInfo = postingList.data() + i * m_vectorInfoSize;
                    int vectorID = *(reinterpret_cast<int*>(vectorInfo));
                    if (m_versionMap.Contains(vectorID) || p_exWorkSpace->m_deduper.CheckAndSet(vectorID)) {
                        listElements--;
                        continue;
                    }
                    auto distance2leaf = p_index->ComputeDistance(queryResults.GetQuantizedTarget(), vectorInfo + m_metaDataSize);
                    queryResults.AddPoint(vectorID, distance2leaf);
                }
                auto compEnd = std::chrono::high_resolution_clock::now();

                compLatency += ((double)std::chrono::duration_cast<std::chrono::microseconds>(compEnd - compStart).count());

                auto exEnd = std::chrono::high_resolution_clock::now();

                if ((((double)std::chrono::duration_cast<std::chrono::microseconds>(exEnd - exStart).count()) / 1000 + p_stats->m_totalLatency) >= m_hardLatencyLimit) {
                    break;
                }

                if (truth) {
                    for (int i = 0; i < vectorNum; ++i) {
                        char* vectorInfo = postingList.data() + i * m_vectorInfoSize;
                        int vectorID = *(reinterpret_cast<int*>(vectorInfo));
                        if (truth->count(vectorID) != 0)
                            (*found)[curPostingID].insert(vectorID);
                    }
                }
            }

            if (p_stats)
            {
                p_stats->m_compLatency = compLatency / 1000;
                p_stats->m_diskReadLatency = readLatency / 1000;
                p_stats->m_totalListElementsCount = listElements;
                p_stats->m_diskIOCount = diskIO;
                p_stats->m_diskAccessCount = diskRead / 1024;
            }
        }

        bool BuildIndex(std::shared_ptr<Helper::VectorSetReader>& p_reader, std::shared_ptr<VectorIndex> p_headIndex, Options& p_opt) override {
            std::string outputFile = p_opt.m_indexDirectory + FolderSep + p_opt.m_ssdIndex;
            if (outputFile.empty())
            {
                LOG(Helper::LogLevel::LL_Error, "Output file can't be empty!\n");
                return false;
            }

            int numThreads = p_opt.m_iSSDNumberOfThreads;
            int candidateNum = p_opt.m_internalResultNum;

            SizeType fullCount = 0;
            size_t vectorInfoSize = 0;
            {
                auto fullVectors = p_reader->GetVectorSet();
                fullCount = fullVectors->Count();
                // vectorInfoSize = fullVectors->PerVectorDataSize() + sizeof(int);
                vectorInfoSize = fullVectors->PerVectorDataSize() + sizeof(int) + sizeof(uint8_t);
            }

            // m_metaDataSize = sizeof(int) + sizeof(uint8_t) + sizeof(float);
            m_metaDataSize = sizeof(int) + sizeof(uint8_t);

            Selection selections(static_cast<size_t>(fullCount) * p_opt.m_replicaCount, p_opt.m_tmpdir);
            LOG(Helper::LogLevel::LL_Info, "Full vector count:%d Edge bytes:%llu selection size:%zu, capacity size:%zu\n", fullCount, sizeof(Edge), selections.m_selections.size(), selections.m_selections.capacity());
            std::vector<std::atomic_int> replicaCount(fullCount);
            std::vector<std::atomic_int> postingListSize(p_headIndex->GetNumSamples());
            for (auto& pls : postingListSize) pls = 0;
            std::unordered_set<SizeType> emptySet;
            SizeType batchSize = (fullCount + p_opt.m_batches - 1) / p_opt.m_batches;

            auto t1 = std::chrono::high_resolution_clock::now();
            if (p_opt.m_batches > 1) selections.SaveBatch();
            {
                LOG(Helper::LogLevel::LL_Info, "Preparation done, start candidate searching.\n");
                SizeType sampleSize = p_opt.m_samples;
                std::vector<SizeType> samples(sampleSize, 0);
                for (int i = 0; i < p_opt.m_batches; i++) {
                    SizeType start = i * batchSize;
                    SizeType end = min(start + batchSize, fullCount);
                    auto fullVectors = p_reader->GetVectorSet(start, end);
                    if (p_opt.m_distCalcMethod == DistCalcMethod::Cosine && !p_reader->IsNormalized()) fullVectors->Normalize(p_opt.m_iSSDNumberOfThreads);

                    emptySet.clear();

                    p_headIndex->ApproximateRNG(fullVectors, emptySet, candidateNum, selections.m_selections.data(), p_opt.m_replicaCount, numThreads, p_opt.m_gpuSSDNumTrees, p_opt.m_gpuSSDLeafSize, p_opt.m_rngFactor, p_opt.m_numGPUs);

                    for (SizeType j = start; j < end; j++) {
                        replicaCount[j] = 0;
                        size_t vecOffset = j * (size_t)p_opt.m_replicaCount;
                        for (int resNum = 0; resNum < p_opt.m_replicaCount && selections[vecOffset + resNum].node != INT_MAX; resNum++) {
                            ++postingListSize[selections[vecOffset + resNum].node];
                            selections[vecOffset + resNum].tonode = j;
                            ++replicaCount[j];
                        }
                    }

                    if (p_opt.m_batches > 1) selections.SaveBatch();
                }
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            LOG(Helper::LogLevel::LL_Info, "Searching replicas ended. Search Time: %.2lf mins\n", ((double)std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()) / 60.0);

            if (p_opt.m_batches > 1) selections.LoadBatch(0, static_cast<size_t>(fullCount) * p_opt.m_replicaCount);

            // Sort results either in CPU or GPU
            VectorIndex::SortSelections(&selections.m_selections);

            auto t3 = std::chrono::high_resolution_clock::now();
            LOG(Helper::LogLevel::LL_Info, "Time to sort selections:%.2lf sec.\n", ((double)std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()) / 1000);

            auto postingSizeLimit = m_postingSizeLimit;

            if (p_opt.m_postingPageLimit > 0)
            {
                postingSizeLimit = static_cast<int>(p_opt.m_postingPageLimit * PageSize / vectorInfoSize);
            }

            LOG(Helper::LogLevel::LL_Info, "Posting size limit: %d\n", postingSizeLimit);


            {
                std::vector<int> replicaCountDist(p_opt.m_replicaCount + 1, 0);
                for (int i = 0; i < replicaCount.size(); ++i)
                {
                    ++replicaCountDist[replicaCount[i]];
                }

                LOG(Helper::LogLevel::LL_Info, "Before Posting Cut:\n");
                for (int i = 0; i < replicaCountDist.size(); ++i)
                {
                    LOG(Helper::LogLevel::LL_Info, "Replica Count Dist: %d, %d\n", i, replicaCountDist[i]);
                }
            }

#pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < postingListSize.size(); ++i)
            {
                if (postingListSize[i] <= postingSizeLimit) continue;

                std::size_t selectIdx = std::lower_bound(selections.m_selections.begin(), selections.m_selections.end(), i, Selection::g_edgeComparer) - selections.m_selections.begin();

                for (size_t dropID = postingSizeLimit; dropID < postingListSize[i]; ++dropID)
                {
                    int tonode = selections.m_selections[selectIdx + dropID].tonode;
                    --replicaCount[tonode];
                }
                postingListSize[i] = postingSizeLimit;
            }

            {
                std::vector<int> replicaCountDist(p_opt.m_replicaCount + 1, 0);
                for (int i = 0; i < replicaCount.size(); ++i)
                {
                    ++replicaCountDist[replicaCount[i]];
                }

                LOG(Helper::LogLevel::LL_Info, "After Posting Cut:\n");
                for (int i = 0; i < replicaCountDist.size(); ++i)
                {
                    LOG(Helper::LogLevel::LL_Info, "Replica Count Dist: %d, %d\n", i, replicaCountDist[i]);
                }
            }

            if (p_opt.m_outputEmptyReplicaID)
            {
                std::vector<int> replicaCountDist(p_opt.m_replicaCount + 1, 0);
                auto ptr = SPTAG::f_createIO();
                if (ptr == nullptr || !ptr->Initialize("EmptyReplicaID.bin", std::ios::binary | std::ios::out)) {
                    LOG(Helper::LogLevel::LL_Error, "Fail to create EmptyReplicaID.bin!\n");
                    return false;
                }
                for (int i = 0; i < replicaCount.size(); ++i)
                {

                    ++replicaCountDist[replicaCount[i]];

                    if (replicaCount[i] < 2)
                    {
                        long long vid = i;
                        if (ptr->WriteBinary(sizeof(vid), reinterpret_cast<char*>(&vid)) != sizeof(vid)) {
                            LOG(Helper::LogLevel::LL_Error, "Failt to write EmptyReplicaID.bin!");
                            return false;
                        }
                    }
                }
                LOG(Helper::LogLevel::LL_Info, "After Posting Cut:\n");
                for (int i = 0; i < replicaCountDist.size(); ++i)
                {
                    LOG(Helper::LogLevel::LL_Info, "Replica Count Dist: %d, %d\n", i, replicaCountDist[i]);
                }
            }


            auto t4 = std::chrono::high_resolution_clock::now();
            LOG(SPTAG::Helper::LogLevel::LL_Info, "Time to perform posting cut:%.2lf sec.\n", ((double)std::chrono::duration_cast<std::chrono::seconds>(t4 - t3).count()) + ((double)std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count()) / 1000);

            if (p_opt.m_ssdIndexFileNum > 1) selections.SaveBatch();

            auto fullVectors = p_reader->GetVectorSet();
            if (p_opt.m_distCalcMethod == DistCalcMethod::Cosine && !p_reader->IsNormalized()) fullVectors->Normalize(p_opt.m_iSSDNumberOfThreads);

            LOG(Helper::LogLevel::LL_Info, "SPFresh: initialize versionMap\n");
            COMMON::VersionLabel m_versionMap;
            m_versionMap.Initialize(fullCount, p_headIndex->m_iDataBlockSize, p_headIndex->m_iDataCapacity);

            LOG(Helper::LogLevel::LL_Info, "SPFresh: Writing values to DB\n");

            std::vector<int> postingListSize_int(postingListSize.begin(), postingListSize.end());

            WriteDownAllPostingToDB(postingListSize_int, selections, m_versionMap, fullVectors);

            COMMON::PostingSizeRecord m_postingSizes;
            m_postingSizes.Initialize(postingListSize.size(), p_headIndex->m_iDataBlockSize, p_headIndex->m_iDataCapacity);
            for (int i = 0; i < postingListSize.size(); i++) {
                m_postingSizes.UpdateSize(i, postingListSize[i]);
            }
            LOG(Helper::LogLevel::LL_Info, "SPFresh: Writing SSD Info\n");
            m_postingSizes.Save(p_opt.m_ssdInfoFile);
            LOG(Helper::LogLevel::LL_Info, "SPFresh: save versionMap\n");
            m_versionMap.Save(p_opt.m_fullDeletedIDFile);

            auto t5 = std::chrono::high_resolution_clock::now();
            double elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(t5 - t1).count();
            LOG(Helper::LogLevel::LL_Info, "Total used time: %.2lf minutes (about %.2lf hours).\n", elapsedSeconds / 60.0, elapsedSeconds / 3600.0);

            return true;
        }

        void WriteDownAllPostingToDB(const std::vector<int>& p_postingListSizes, Selection& p_postingSelections, COMMON::VersionLabel& m_versionMap, std::shared_ptr<VectorSet> p_fullVectors) {
            size_t dim = p_fullVectors->Dimension();
            #pragma omp parallel for num_threads(10)
            for (int id = 0; id < p_postingListSizes.size(); id++)
            {
                std::string postinglist;
                std::size_t selectIdx = p_postingSelections.lower_bound(id);
                for (int j = 0; j < p_postingListSizes[id]; ++j) {
                    if (p_postingSelections[selectIdx].node != id) {
                        LOG(Helper::LogLevel::LL_Error, "Selection ID NOT MATCH\n");
                        exit(1);
                    }
                    int fullID = p_postingSelections[selectIdx++].tonode;
                    uint8_t version = 0;
                    m_versionMap.UpdateVersion(fullID, 0);
                    // First Vector ID, then version, then Vector
                    postinglist += Helper::Convert::Serialize<int>(&fullID, 1);
                    postinglist += Helper::Convert::Serialize<uint8_t>(&version, 1);
                    postinglist += Helper::Convert::Serialize<ValueType>(p_fullVectors->GetVector(fullID), dim);
                }
                AddIndex(id, postinglist);
            }
        }

        ErrorCode AppendPosting(SizeType headID, const std::string& appendPosting) override {
            if (appendPosting.empty()) {
                LOG(Helper::LogLevel::LL_Error, "Error! empty append posting!\n");
            }
            return db.Merge(headID, appendPosting);
        }

        void ForceCompaction() override { db.ForceCompaction(); }
        void GetDBStats() override { db.GetDBStat(); }

        inline ErrorCode SearchIndex(SizeType headID, std::string& posting) override {  return db.Get(headID, &posting); }
        inline ErrorCode AddIndex(SizeType headID, const std::string& posting) override { m_postingNum++; return db.Put(headID, posting); }
        inline ErrorCode DeleteIndex(SizeType headID) override { m_postingNum--; return db.Delete(headID); }
        inline ErrorCode OverrideIndex(SizeType headID, const std::string& posting) override { return db.Put(headID, posting); }
        inline SizeType  GetIndexSize() override { return m_postingNum; }
        inline SizeType  GetPostingSizeLimit() override { return m_postingSizeLimit;}
        inline SizeType  GetMetaDataSize() override { return m_metaDataSize;}
        inline ErrorCode SearchIndexMulti(const std::vector<SizeType>& keys, std::vector<std::string>* values) override {return db.MultiGet(keys, values);}
    private:
        struct ListInfo
        {
            int listEleCount = 0;

            std::uint16_t listPageCount = 0;

            std::uint64_t listOffset = 0;

            std::uint16_t pageOffset = 0;
        };

        int LoadingHeadInfo(const std::string& p_file, int p_postingPageLimit, std::vector<ListInfo>& m_listInfos)
        {
            auto ptr = SPTAG::f_createIO();
            if (ptr == nullptr || !ptr->Initialize(p_file.c_str(), std::ios::binary | std::ios::in)) {
                LOG(Helper::LogLevel::LL_Error, "Failed to open file: %s\n", p_file.c_str());
                exit(1);
            }

            int m_listCount;
            int m_totalDocumentCount;
            int m_iDataDimension;
            int m_listPageOffset;

            if (ptr->ReadBinary(sizeof(m_listCount), reinterpret_cast<char*>(&m_listCount)) != sizeof(m_listCount)) {
                LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                exit(1);
            }
            if (ptr->ReadBinary(sizeof(m_totalDocumentCount), reinterpret_cast<char*>(&m_totalDocumentCount)) != sizeof(m_totalDocumentCount)) {
                LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                exit(1);
            }
            if (ptr->ReadBinary(sizeof(m_iDataDimension), reinterpret_cast<char*>(&m_iDataDimension)) != sizeof(m_iDataDimension)) {
                LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                exit(1);
            }
            if (ptr->ReadBinary(sizeof(m_listPageOffset), reinterpret_cast<char*>(&m_listPageOffset)) != sizeof(m_listPageOffset)) {
                LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                exit(1);
            }

            if (m_vectorInfoSize == 0) m_vectorInfoSize = m_iDataDimension * sizeof(ValueType) + sizeof(int);
            else if (m_vectorInfoSize != m_iDataDimension * sizeof(ValueType) + sizeof(int)) {
                LOG(Helper::LogLevel::LL_Error, "Failed to read head info file! DataDimension and ValueType are not match!\n");
                exit(1);
            }

            m_listInfos.resize(m_listCount);

            size_t totalListElementCount = 0;

            std::map<int, int> pageCountDist;

            size_t biglistCount = 0;
            size_t biglistElementCount = 0;
            int pageNum;
            for (int i = 0; i < m_listCount; ++i)
            {
                if (ptr->ReadBinary(sizeof(pageNum), reinterpret_cast<char*>(&(pageNum))) != sizeof(pageNum)) {
                    LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                    exit(1);
                }
                if (ptr->ReadBinary(sizeof(m_listInfos[i].pageOffset), reinterpret_cast<char*>(&(m_listInfos[i].pageOffset))) != sizeof(m_listInfos[i].pageOffset)) {
                    LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                    exit(1);
                }
                if (ptr->ReadBinary(sizeof(m_listInfos[i].listEleCount), reinterpret_cast<char*>(&(m_listInfos[i].listEleCount))) != sizeof(m_listInfos[i].listEleCount)) {
                    LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                    exit(1);
                }
                if (ptr->ReadBinary(sizeof(m_listInfos[i].listPageCount), reinterpret_cast<char*>(&(m_listInfos[i].listPageCount))) != sizeof(m_listInfos[i].listPageCount)) {
                    LOG(Helper::LogLevel::LL_Error, "Failed to read head info file!\n");
                    exit(1);
                }

                m_listInfos[i].listOffset = (static_cast<uint64_t>(m_listPageOffset + pageNum) << PageSizeEx);
                m_listInfos[i].listEleCount = min(m_listInfos[i].listEleCount, (min(static_cast<int>(m_listInfos[i].listPageCount), p_postingPageLimit) << PageSizeEx) / m_vectorInfoSize);
                m_listInfos[i].listPageCount = static_cast<std::uint16_t>(ceil((m_vectorInfoSize * m_listInfos[i].listEleCount + m_listInfos[i].pageOffset) * 1.0 / (1 << PageSizeEx)));
                totalListElementCount += m_listInfos[i].listEleCount;
                int pageCount = m_listInfos[i].listPageCount;

                if (pageCount > 1)
                {
                    ++biglistCount;
                    biglistElementCount += m_listInfos[i].listEleCount;
                }

                if (pageCountDist.count(pageCount) == 0)
                {
                    pageCountDist[pageCount] = 1;
                }
                else
                {
                    pageCountDist[pageCount] += 1;
                }
            }

            LOG(Helper::LogLevel::LL_Info,
                "Finish reading header info, list count %d, total doc count %d, dimension %d, list page offset %d.\n",
                m_listCount,
                m_totalDocumentCount,
                m_iDataDimension,
                m_listPageOffset);


            LOG(Helper::LogLevel::LL_Info,
                "Big page (>4K): list count %zu, total element count %zu.\n",
                biglistCount,
                biglistElementCount);

            LOG(Helper::LogLevel::LL_Info, "Total Element Count: %llu\n", totalListElementCount);

            for (auto& ele : pageCountDist)
            {
                LOG(Helper::LogLevel::LL_Info, "Page Count Dist: %d %d\n", ele.first, ele.second);
            }

            return m_listCount;
        }

    private:


        int m_vectorInfoSize = 0;

        int m_postingSizeLimit = INT_MAX;

        int m_metaDataSize = 0;

        float m_hardLatencyLimit = 2;
    };
} // namespace SPTAG

#endif // _SPTAG_SPANN_EXTRADBSEARCHER_H_
