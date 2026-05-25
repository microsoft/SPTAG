#ifndef _SPTAG_SPANN_EXTRAFILECONTROLLER_H_
#define _SPTAG_SPANN_EXTRAFILECONTROLLER_H_
#define USE_ASYNC_IO

#include "inc/Helper/KeyValueIO.h"
#include "inc/Core/Common/Dataset.h"
#include "inc/Core/Common/LabelSet.h"
#include "inc/Core/Common/Checksum.h"
#include "inc/Core/Common/FineGrainedLock.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Helper/ThreadPool.h"
#include "inc/Helper/ConcurrentSet.h"
#include "inc/Helper/AsyncFileReader.h"
#include "inc/Core/SPANN/Options.h"
#include <cstdlib>
#include <memory>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <list>
namespace SPTAG::SPANN {
    typedef std::int64_t AddressType;
    class FileIO : public Helper::KeyValueIO {

        class BlockController {
        private:
            char m_filePath[1024] = "";
            std::shared_ptr <Helper::DiskIO> m_fileHandle = nullptr;

            Helper::Concurrent::ConcurrentQueue<AddressType> m_blockAddresses;
            Helper::Concurrent::ConcurrentQueue<AddressType> m_blockAddresses_reserve;
            COMMON::LabelSet m_available;

	        std::atomic<int64_t> read_complete_vec = 0;
	        std::atomic<int64_t> read_submit_vec = 0;
	        std::atomic<int64_t> write_complete_vec = 0;
	        std::atomic<int64_t> write_submit_vec = 0;
	        std::atomic<int64_t> read_bytes_vec = 0;
	        std::atomic<int64_t> write_bytes_vec = 0;
	        std::atomic<int64_t> read_blocks_time_vec = 0;

            float m_growthThreshold = 0.05F;
            AddressType m_growthBlocks = 0;
            AddressType m_maxBlocks = 0;
            int m_batchSize = 64;
            int64_t m_preIOCompleteCount = 0;
            int64_t m_preIOBytes = 0;
            bool m_disableCheckpoint = false;

            std::chrono::high_resolution_clock::time_point m_startTime;
            std::chrono::time_point<std::chrono::high_resolution_clock> m_preTime = std::chrono::high_resolution_clock::now();

            std::atomic<int64_t> m_batchReadTimes = 0;
            std::atomic<int64_t> m_batchReadTimeouts = 0;

            std::mutex m_expandLock;
            std::atomic<AddressType> m_totalAllocatedBlocks = 0;

        private:
            bool ExpandFile(AddressType blocksToAdd);
            bool NeedsExpansion(int psize);

            // static void Start(void* args);

            // static void FileIoLoop(void *arg);

            // static void FileIoCallback(bool success, void *cb_arg);

            // static void Stop(void* args);

        public:
            bool Initialize(SPANN::Options& p_opt, int p_layer);

            bool GetBlocks(AddressType* p_data, int p_size);

            bool ReleaseBlocks(AddressType* p_data, int p_size);

            bool ReadBlocks(AddressType* p_data, std::string* p_value, const std::chrono::microseconds &timeout, std::vector<Helper::AsyncReadRequest>* reqs);

            bool ReadBlocks(AddressType *p_data, Helper::PageBuffer<std::uint8_t> &p_value, const std::chrono::microseconds &timeout, std::vector<Helper::AsyncReadRequest> *reqs);

            bool ReadBlocks(const std::vector<AddressType*>& p_data, std::vector<std::string>* p_value, const std::chrono::microseconds &timeout, std::vector<Helper::AsyncReadRequest>* reqs);

            bool ReadBlocks(const std::vector<AddressType*>& p_data, std::vector<Helper::PageBuffer<std::uint8_t>>& p_value, const std::chrono::microseconds& timeout, std::vector<Helper::AsyncReadRequest>* reqs);

            bool WriteBlocks(AddressType* p_data, int p_size, const std::string& p_value, const std::chrono::microseconds& timeout, std::vector<Helper::AsyncReadRequest>* reqs);

            bool IOStatistics();

            bool ShutDown();

            int RemainBlocks() {
                return (int)(m_blockAddresses.unsafe_size());
            }

            int ReserveBlocks() {
                return (int)(m_blockAddresses_reserve.unsafe_size());
            }

            int TotalBlocks() {
                return (int)(m_totalAllocatedBlocks.load());
            }

            ErrorCode Checkpoint(std::string prefix) {
                std::string filename = prefix + "_blockpool";
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO::BlockController::Checkpoint - Starting block pool save...\n");

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO::BlockController::Checkpoint - Reload reserved blocks...\n");
                AddressType currBlockAddress = 0;
                int reloadCount = 0;

                while (m_blockAddresses_reserve.try_pop(currBlockAddress)) {
                    m_blockAddresses.push(currBlockAddress);
                    ++reloadCount;
                }
                AddressType blocks = RemainBlocks();
                AddressType totalBlocks = m_totalAllocatedBlocks.load();

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO::BlockController::Checkpoint - Reloaded blocks: %d\n", reloadCount);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO::BlockController::Checkpoint - Total allocated blocks: %llu\n", static_cast<unsigned long long>(totalBlocks));
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO::BlockController::Checkpoint - Remaining free blocks: %llu\n", static_cast<unsigned long long>(blocks));
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO::BlockController::Checkpoint - Saving to file: %s\n", filename.c_str());

                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(filename.c_str(), std::ios::binary | std::ios::out)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::Checkpoint - Failed to open file: %s\n", filename.c_str());
                    return ErrorCode::FailedCreateFile;
                }
                IOBINARY(ptr, WriteBinary, sizeof(AddressType), reinterpret_cast<char*>(&blocks));
                IOBINARY(ptr, WriteBinary, sizeof(AddressType), reinterpret_cast<char*>(&totalBlocks));
                for (auto it = m_blockAddresses.unsafe_begin(); it != m_blockAddresses.unsafe_end(); it++) {
                    IOBINARY(ptr, WriteBinary, sizeof(AddressType), reinterpret_cast<const char*>(&(*it)));
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO::BlockController::Checkpoint - Save Finish!\n");
                return ErrorCode::Success;
            }

	        ErrorCode LoadBlockPool(std::string prefix, AddressType startNumBlocks, bool allowInit, SizeType blockSize, SizeType blockCapacity) {
	            std::string blockfile = prefix + "_blockpool";
                if (allowInit && !fileexists(blockfile.c_str())) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO::BlockController::LoadBlockPool: initializing fresh pool (no existing file found: %s)\n", blockfile.c_str());
                    m_available.Initialize(startNumBlocks, blockSize, blockCapacity);
                    for(AddressType i = 0; i < startNumBlocks; i++) {
                        m_blockAddresses.push(i);
                        m_available.Insert(i);
                    }
                    m_totalAllocatedBlocks.store(startNumBlocks);

                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO::BlockController::LoadBlockPool: initialized with %llu blocks (%.2f GB)\n", 
                        static_cast<unsigned long long>(startNumBlocks), static_cast<float>(startNumBlocks >> (30 - PageSizeEx)));
                } else {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO::BlockController::Load blockpool from %s\n", blockfile.c_str());
                    auto ptr = f_createIO();
                    if (ptr == nullptr || !ptr->Initialize(blockfile.c_str(), std::ios::binary | std::ios::in)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot open the blockpool file: %s\n", blockfile.c_str());
    		            return ErrorCode::Fail;
                    }

                    AddressType blocks = 0;
                    AddressType totalAllocated = 0;

                    // Read block count
                    IOBINARY(ptr, ReadBinary, sizeof(AddressType), reinterpret_cast<char*>(&blocks));
                    IOBINARY(ptr, ReadBinary, sizeof(AddressType), reinterpret_cast<char*>(&totalAllocated));

                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "FileIO::BlockController::LoadBlockPool: reading %llu free blocks into pool (%.2f GB), total allocated: %llu blocks\n",
                        static_cast<unsigned long long>(blocks),
                        static_cast<float>(blocks >> (30 - PageSizeEx)),
                        static_cast<unsigned long long>(totalAllocated));

                    m_available.Initialize(totalAllocated, blockSize, blockCapacity);
                    AddressType currBlockAddress = 0;
                    for (AddressType i = 0; i < blocks; ++i) {
                        IOBINARY(ptr, ReadBinary, sizeof(AddressType), reinterpret_cast<char*>(&currBlockAddress));
                        if (!m_available.Insert(currBlockAddress))
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "LoadBlockPool: Double release blockid:%lld\n",
                                         currBlockAddress);
                            continue;
                        }
                        m_blockAddresses.push(currBlockAddress);
                    }

                    m_totalAllocatedBlocks.store(totalAllocated);

                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "FileIO::BlockController::LoadBlockPool: block pool initialized. Available: %llu, Total allocated: %llu\n",
                        static_cast<unsigned long long>(blocks),
                        static_cast<unsigned long long>(totalAllocated));
    	        }
		        /*
		        int i = 0;
                        for (auto it = m_blockAddresses.unsafe_begin(); it != m_blockAddresses.unsafe_end(); it++) {
		            std::cout << *it << " ";
		            i++;
		            if (i == 10) break;
		        }
		        std::cout << std::endl;
                */
                return ErrorCode::Success;
            } 
        };

        struct CacheCounters
        {
            int64_t query{};
            int64_t query_hits{};
            int64_t put{};
            int64_t evict{};
        };

        struct CacheEntry
        {
            std::string value;
            std::list<SizeType>::iterator iter;
        };

        class LRUCache {
            int64_t capacity;
            int64_t limit;
            int64_t size;
            std::list<SizeType> keys;  // Page Address
            std::unordered_map<SizeType, CacheEntry> cache;    // Page Address -> Page Address in Cache
            FileIO* fileIO;
            Helper::RequestQueue processIocp;
            std::vector<Helper::AsyncReadRequest> reqs;
            Helper::PageBuffer<std::uint8_t> pageBuffer;
            std::atomic<int64_t> query_counter{};
            std::atomic<int64_t> query_hits_counter{};
            std::atomic<int64_t> put_counter{};
            std::atomic<int64_t> evict_counter{};

        public:
            LRUCache(int64_t capacity, int64_t limit, FileIO* fileIO) {
                this->capacity = capacity;
                this->limit = min(capacity, (limit << PageSizeEx));
                this->size = 0;
                this->fileIO = fileIO;
                this->reqs.resize(limit);
                this->pageBuffer.ReservePageBuffer(limit << PageSizeEx);
                for (uint64_t i = 0; i < limit; i++) {
                    auto& req = this->reqs[i];
                    req.m_buffer = (char *)(this->pageBuffer.GetBuffer() + (i << PageSizeEx));
                    req.m_extension = &processIocp;

#ifdef _MSC_VER
                    memset(&(req.myres.m_col), 0, sizeof(OVERLAPPED));
                    req.myres.m_col.m_data = (void*)(&req);
#else
                    memset(&(req.myiocb), 0, sizeof(struct iocb));
                    req.myiocb.aio_buf = reinterpret_cast<uint64_t>(req.m_buffer);
                    req.myiocb.aio_data = reinterpret_cast<uintptr_t>(&req);
#endif
                }
            }

            ~LRUCache() {}

            bool evict(SizeType key, void* value, int vsize, std::unordered_map<SizeType, CacheEntry>::iterator& it) {
                if (value != nullptr) {
                    ++evict_counter;
                    std::string valstr((char*)value, vsize);
                    if (fileIO->Put(key, valstr, MaxTimeout, &reqs, false) != ErrorCode::Success) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "LRUCache: evict key:%d value size:%d to file failed\n", key, vsize);
                        return false;
                    }
                }

                size -= it->second.value.size();
                keys.erase(it->second.iter);
                cache.erase(it);
                return true;
            }

            bool get(SizeType key, Helper::PageBuffer<std::uint8_t>& buffer) {
                ++query_counter;
                auto it = cache.find(key);
                if (it == cache.end()) {
                    return false;
                }

                size_t data_size = it->second.value.size();
                buffer.ReservePageBuffer(data_size);
                memcpy(buffer.GetBuffer(), it->second.value.data(), data_size);
                buffer.SetAvailableSize(data_size);
                ++query_hits_counter;
                return true;
            }

            bool get(SizeType key, std::string& value) {
                ++query_counter;
                auto it = cache.find(key);
                if (it == cache.end()) {
                    return false;
                }

                size_t data_size = it->second.value.size();
                value.resize(data_size);
                memcpy(value.data(), it->second.value.data(), data_size);
                ++query_hits_counter;
                return true;
            }

            bool put(SizeType key, void* value, int put_size) {
                ++put_counter;
                auto it = cache.find(key);
                if (it != cache.end()) {
                    if (put_size > limit) {
                        evict(key, it->second.value.data(), (int)(it->second.value.size()), it);
                        return false;
                    }
                    keys.splice(keys.begin(), keys, it->second.iter);
                    it->second.iter = keys.begin();

                    int delta_size = put_size - (int)(it->second.value.size());
                    while ((int)(capacity - size) < delta_size && (keys.size() > 1)) {
                        auto last = keys.back();
                        auto lastit = cache.find(last);
                        if (!evict(last, lastit->second.value.data(), (int)(lastit->second.value.size()), lastit)) {
                            evict(key, it->second.value.data(), (int)(it->second.value.size()), it);
                            return false;
                        }
                    }
                    it->second.value.resize(put_size);
                    memcpy(it->second.value.data(), (char*)value, put_size);
                    size += delta_size;
                    return true;
                }
                if (put_size > limit) {
                    return false;
                }
                while (put_size > (int)(capacity - size) && (!keys.empty())) {
                    auto last = keys.back();
                    auto lastit = cache.find(last);
                    if (!evict(last, lastit->second.value.data(), (int)(lastit->second.value.size()), lastit)) {
                        return false;
                    }
                }
                auto keys_it = keys.insert(keys.begin(), key);
                cache.insert({key, {std::string((char *)value, put_size), keys_it}});
                size += put_size;
                return true;
            }

            bool del(SizeType key) {
                auto it = cache.find(key);
                if (it == cache.end()) {
                    return false; // If the key does not exist, return false
                }
                evict(key, nullptr, 0, it);
                return true;
            }

            bool merge(SizeType key, void *value, int merge_size)
            {
                ++put_counter;
                // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LRUCache: merge size: %lld\n", merge_size);
                auto it = cache.find(key);
                if (it == cache.end()) {
                    // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LRUCache: merge key not found\n");
                    ErrorCode ret;
                    if ((ret = fileIO->Get(key, pageBuffer, MaxTimeout, &reqs, false)) != ErrorCode::Success) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "LRUCache: merge key %lld not found in file\n", (std::int64_t)key);
                        return false;  // If the key does not exist, return false
                    }
                    std::string valstr(pageBuffer.GetAvailableSize() + merge_size, '\0');
                    memcpy(valstr.data(), pageBuffer.GetBuffer(), pageBuffer.GetAvailableSize());
                    memcpy(valstr.data() + pageBuffer.GetAvailableSize(), value, merge_size);
                    while (valstr.size() > (int)(capacity - size) && (!keys.empty()))
                    {
                        auto last = keys.back();
                        auto lastit = cache.find(last);
                        if (!evict(last, lastit->second.value.data(), (int)(lastit->second.value.size()), lastit))
                        {
                            return false;
                        }
                    }
                    auto keys_it = keys.insert(keys.begin(), key);
                    cache.insert({key, {valstr, keys_it}});
                    size += valstr.size();
                    return true;
                }

                if (merge_size + it->second.value.size() > limit) {
                    evict(key, it->second.value.data(), (int)(it->second.value.size()), it);
                    // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LRUCache: merge size exceeded\n");
                    return false;
                }
                keys.splice(keys.begin(), keys, it->second.iter);
                it->second.iter = keys.begin();
                while((int)(capacity - size) < merge_size && (keys.size() > 1)) {
                    auto last = keys.back();
                    auto lastit = cache.find(last);
                    if (!evict(last, lastit->second.value.data(), (int)(lastit->second.value.size()), lastit)) {
                        evict(key, it->second.value.data(), (int)(it->second.value.size()), it);
                        return false;
                    }
                }
                it->second.value.append((char*)value, merge_size);
                size += merge_size;
                // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LRUCache: merge success\n");
                return true;
            }

            CacheCounters get_stat() {
                return CacheCounters{query_counter.load(), query_hits_counter.load(), put_counter.load(), evict_counter.load()};
            }
            
            bool flush() {
                for (auto it = cache.begin(); it != cache.end(); it++) {
                    if (fileIO->Put(it->first, it->second.value, MaxTimeout, &reqs, false) != ErrorCode::Success) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "LRUCache: evict key:%d value size:%d to file failed\n", it->first, (int)(it->second.value.size()));
                        return false;
                    }
                }
                cache.clear();
                keys.clear();
                size = 0;
                return true;
            }
            
            int64_t GetApproximateMemoryUsage() const
            {
                return static_cast<int64_t>(size);
            }
        }; 

        class ShardedLRUCache {
            int shards;
            std::vector<LRUCache*> caches;
            std::unique_ptr<std::shared_timed_mutex[]> m_rwMutexs;

        public:
            ShardedLRUCache(int shards, int64_t capacity, int64_t limit, FileIO* fileIO) : shards(shards) {
                caches.resize(shards);
                m_rwMutexs.reset(new std::shared_timed_mutex[shards]);
                for (int i = 0; i < shards; i++) {
                    caches[i] = new LRUCache(capacity / shards, limit, fileIO);
                }
                if (capacity % shards != 0) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "LRUCache: capacity is not divisible by shards\n");
                }
            }

            ~ShardedLRUCache() {
                for (int i = 0; i < shards; i++) {
                    delete caches[i];
                }
            }

            bool get(SizeType key, Helper::PageBuffer<std::uint8_t> &buffer)
            {
                SizeType cid = hash(key);
                std::shared_lock<std::shared_timed_mutex> lock(m_rwMutexs[cid]);
                return caches[cid]->get(key, buffer);
            }

            bool get(SizeType key, std::string &value)
            {
                SizeType cid = hash(key);
                std::shared_lock<std::shared_timed_mutex> lock(m_rwMutexs[cid]);
                return caches[cid]->get(key, value);
            }

            bool put(SizeType key, void* value, int put_size) {
                return caches[hash(key)]->put(key, value, put_size);
            }

            bool del(SizeType key) {
                return caches[hash(key)]->del(key);
            }

            bool merge(SizeType key, void *value, int merge_size)
            {
                return caches[hash(key)]->merge(key, value, merge_size);
            }

            bool flush() {
                for (int i = 0; i < shards; i++) {
                	std::unique_lock<std::shared_timed_mutex> lock(m_rwMutexs[i]);
                    if (!caches[i]->flush()) return false;
                }
                return true;
            }

            std::shared_timed_mutex& getlock(SizeType key)
            {
                return m_rwMutexs[hash(key)];
            }
            
            SizeType hash(SizeType key) const {
                return key % shards;
            }
            
            CacheCounters get_stat() {
                CacheCounters result;
                for (int i = 0; i < shards; i++) {
                    auto stat = caches[i]->get_stat();
                    result.query += stat.query;
                    result.query_hits += stat.query_hits;
                    result.put += stat.put;
                    result.evict += stat.evict;
                }

                return result;
            }

            int64_t GetApproximateMemoryUsage() const
            {
                int64_t result = 0;
                for (int i = 0; i < shards; i++)
                {
                    result += caches[i]->GetApproximateMemoryUsage();
                }
                return result;
            }
        };

    public:
        FileIO(SPANN::Options& p_opt, int p_layer) {
            m_mappingPath = p_opt.m_indexDirectory + FolderSep + p_opt.m_ssdMappingFile + "_" + std::to_string(p_layer);
            m_layer = p_layer;
            m_blockLimit = max(p_opt.m_postingPageLimit, p_opt.m_searchPostingPageLimit) + p_opt.m_bufferLength + 1;
            m_bufferLimit = 1024;
            m_shutdownCalled = true;

            m_pShardedLRUCache = nullptr;
            int64_t capacity = static_cast<int64_t>(p_opt.m_cacheSize) << 30;
            if (capacity > 0) {
                m_pShardedLRUCache = new ShardedLRUCache(p_opt.m_cacheShards, capacity, m_blockLimit, this);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO: Using LRU Cache with capacity %d GB, limit %d pages, shards %d\n", p_opt.m_cacheSize, m_blockLimit, p_opt.m_cacheShards);
            }

            if (p_opt.m_recovery) {
                std::string recoverpath = p_opt.m_persistentBufferPath + FolderSep + p_opt.m_ssdMappingFile + "_" + std::to_string(p_layer);
                if (fileexists(recoverpath.c_str())) {
                    Load(recoverpath);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load block mapping successfully from %s!\n", recoverpath.c_str());
                }
                else {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot recover block mapping from %s!\n", recoverpath.c_str());
                    return;
                }
            }
            else {
                if (fileexists(m_mappingPath.c_str())) {
                    Load(m_mappingPath);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load block mapping successfully from %s!\n", m_mappingPath.c_str());
                }
                else {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Initialize block mapping successfully!\n");
                }
            }
            m_checkSumMultiGet = p_opt.m_checksumInRead;
            m_checkSum.Initialize(!p_opt.m_checksumCheck, 0, 0);

            for (int i = 0; i < m_bufferLimit; i++) {
                m_buffer.push((uintptr_t)(new AddressType[m_blockLimit]));
            }
            m_compactionThreadPool = std::make_shared<Helper::ThreadPool>();
            m_compactionThreadPool->init(1);

            if (!m_pBlockController.Initialize(p_opt, p_layer)) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to Initialize FileIO!\n");
                return;
            }

            m_shutdownCalled = false;
        }

        ~FileIO() {
            ShutDown();
        }

        bool Available() override
        {
            return !m_shutdownCalled;
        }

        void ShutDown() override {
            if (m_shutdownCalled) {
                return;
            }
            m_shutdownCalled = true;

            if (!m_mappingPath.empty()) Save(m_mappingPath);
            // TODO: Should we add a lock here?

            for (auto it : m_pBlockMapping) {
                delete[]((AddressType*)(it.second));
            }
            m_pBlockMapping.clear();

            while (!m_buffer.empty()) {
                uintptr_t ptr = InvalidPointer;
                if (m_buffer.try_pop(ptr)) delete[]((AddressType*)ptr);
            }
	        m_pBlockController.ShutDown();
            if (m_pShardedLRUCache) {
                delete m_pShardedLRUCache;
                m_pShardedLRUCache = nullptr;
            }
        }

        inline uintptr_t GetKey(SizeType key) {
            {
                std::shared_lock<std::shared_timed_mutex> lock(m_updateMutex);
                auto it = m_pBlockMapping.find(key);
                if (it != m_pBlockMapping.end()) return it->second;
            }
            return InvalidPointer;
        }
        
        inline uintptr_t& AtKey(SizeType key) {
            uintptr_t* value;
            {
                std::shared_lock<std::shared_timed_mutex> lock(m_updateMutex);
                value = &(m_pBlockMapping[key]);
            }
            return *value;
        }

        ErrorCode Get(const SizeType key, std::string* value, const std::chrono::microseconds &timeout, std::vector<Helper::AsyncReadRequest>* reqs, bool useCache) {
            AddressType* addr = (AddressType*)(GetKey(key));
            if (((uintptr_t)addr) == InvalidPointer) {
                 SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Key NotFound! Key %lld\n", (std::int64_t)key);
                return ErrorCode::Key_NotFound;
            }

            int size = (int)(addr[0]);     
            if (size < 0) return ErrorCode::Posting_SizeError;
            
            ChecksumType checksum = (ChecksumType)(addr[0] >> 32);
            bool result = false;

            if (useCache && m_pShardedLRUCache) {
                if (m_pShardedLRUCache->get(key, *value)) {
                    result = true;
                }
            }
            if (!result) {
                result = m_pBlockController.ReadBlocks((AddressType*)GetKey(key), value, timeout, reqs);
            }

            if (result && m_checkSum.ValidateChecksum(value->c_str(), (int)(value->size()), checksum)) {
                return ErrorCode::Success;
            } else {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Get Key %lld Fail! Checksum error: %d\n", (std::int64_t)key, (int)(result));
                return ErrorCode::Posting_ChecksumError;
            }
        }

        ErrorCode Get(const SizeType key, std::string* value, const std::chrono::microseconds &timeout, std::vector<Helper::AsyncReadRequest>* reqs) override {
            return Get(key, value, timeout, reqs, true);
        }

        ErrorCode Get(const std::string& key, std::string* value, const std::chrono::microseconds& timeout, std::vector<Helper::AsyncReadRequest>* reqs) override {
            return Get(std::stoi(key), value, timeout, reqs, true);
        }

        ErrorCode Get(const SizeType key, Helper::PageBuffer<std::uint8_t> &value,
                      const std::chrono::microseconds &timeout, std::vector<Helper::AsyncReadRequest> *reqs, bool useCache = true) override
        {
            AddressType *addr = (AddressType *)(GetKey(key));
            if (((uintptr_t)addr) == InvalidPointer)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Key NotFound! Key %lld\n", (std::int64_t)key);
                return ErrorCode::Key_NotFound;
            }

            int size = (int)(addr[0]);
            if (size < 0) return ErrorCode::Posting_SizeError;

            ChecksumType checksum = (ChecksumType)(addr[0] >> 32);
            bool result = false;

            if (useCache && m_pShardedLRUCache)
            {
                if (m_pShardedLRUCache->get(key, value))
                {
                    result = true;
                }
            }

            if (!result) {
                result = m_pBlockController.ReadBlocks((AddressType *)GetKey(key), value, timeout, reqs);
            }
            if (result && m_checkSum.ValidateChecksum((const char*)(value.GetBuffer()), (int)(value.GetAvailableSize()), checksum)) {
                return ErrorCode::Success;
            } else {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Get Key %lld Fail! Checksum error: %d\n", (std::int64_t)key, (int)(result));
                return ErrorCode::Posting_ChecksumError;
            }
        }

        ErrorCode MultiGet(const std::vector<SizeType>& keys, std::vector<Helper::PageBuffer<std::uint8_t>>& values,
            const std::chrono::microseconds &timeout, std::vector<Helper::AsyncReadRequest>* reqs) override {
            std::vector<AddressType*> blocks(keys.size(), nullptr);
            std::vector<ChecksumType> checksums(keys.size(), 0);
            for (int i = 0; i < keys.size(); i++) {
                SizeType key = keys[i];
                if (m_pShardedLRUCache && m_pShardedLRUCache->get(key, values[i]))
                {
                    AddressType* addr = (AddressType*)(GetKey(key));
                    checksums[i] = (ChecksumType)(addr[0] >> 32);
                } else {
                    AddressType* addr = (AddressType*)(GetKey(key));
                    blocks[i] = addr;
                    if (((uintptr_t)addr) != InvalidPointer) {
                        checksums[i] = (ChecksumType)(addr[0] >> 32);
                    }
                }
            }
            auto result = m_pBlockController.ReadBlocks(blocks, values, timeout, reqs);
            if (result && m_checkSumMultiGet) {
                for (int i = 0; i < keys.size(); i++)
                {
                    if (!m_checkSum.ValidateChecksum((const char *)(values[i].GetBuffer()),
                                                    values[i].GetAvailableSize(), checksums[i]))
                    {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "MultiGet Checksum fail: Key %lld, buffer size %d\n",
                            (std::int64_t)(keys[i]), (int)(values[i].GetAvailableSize()));
                        return ErrorCode::Posting_ChecksumError;
                    }
                }
            }
            return result ? ErrorCode::Success : ErrorCode::Fail;
        }


        ErrorCode MultiGet(const std::vector<SizeType>& keys, std::vector<std::string>* values,
            const std::chrono::microseconds& timeout, std::vector<Helper::AsyncReadRequest>* reqs) {
            std::vector<AddressType*> blocks(keys.size(), nullptr);
            std::vector<ChecksumType> checksums(keys.size(), 0);
            values->resize(keys.size());
            for (int i = 0; i < keys.size(); i++) {
                SizeType key = keys[i];
                if (m_pShardedLRUCache && m_pShardedLRUCache->get(key, (*values)[i])) {
                    AddressType* addr = (AddressType*)(GetKey(key));
                    checksums[i] = (ChecksumType)(addr[0] >> 32);
                }
                else {
                    AddressType* addr = (AddressType*)(GetKey(key));
                    blocks[i] = addr;
                    if (((uintptr_t)addr) != InvalidPointer) {
                        checksums[i] = (ChecksumType)(addr[0] >> 32);
                    }
                }
            }
            auto result = m_pBlockController.ReadBlocks(blocks, values, timeout, reqs);
            if (result && m_checkSumMultiGet) {
                for (int i = 0; i < keys.size(); i++)
                {
                    if (!m_checkSum.ValidateChecksum((*values)[i].c_str(),
                                                    (*values)[i].size(), checksums[i]))
                    {
                        SPTAGLIB_LOG(
                            Helper::LogLevel::LL_Error,
                            "MultiGet Checksum fail: Key %lld, buffer size %d\n",
                            (std::int64_t)(keys[i]), (int)((*values)[i].size()));
                        return ErrorCode::Posting_ChecksumError;
                    }
                }
            }
            return result ? ErrorCode::Success : ErrorCode::Fail;
        }

        ErrorCode MultiGet(const std::vector<std::string>& keys, std::vector<std::string>* values, 
            const std::chrono::microseconds &timeout, std::vector<Helper::AsyncReadRequest>* reqs) override {
            std::vector<SizeType> int_keys;
            for (const auto& key : keys) {
                int_keys.push_back(std::stoi(key));
            }
            return MultiGet(int_keys, values, timeout, reqs);
        }

        ErrorCode Put(const SizeType key, const std::string& value, const std::chrono::microseconds& timeout, std::vector<Helper::AsyncReadRequest>* reqs, bool useCache) {
            int blocks = (int)(((value.size() + PageSize - 1) >> PageSizeEx));
            if (blocks >= m_blockLimit) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to put Key %lld value %lld since value too long!\n", (std::int64_t)key, (std::int64_t)value.size());
                return ErrorCode::Posting_OverFlow;
            }

            int64_t checksum = ((int64_t)m_checkSum.CalcChecksum(value.c_str(), (int)(value.size()))) << 32;
            int64_t newValueMeta = ((int64_t)(value.size()) | checksum);
            std::shared_timed_mutex *lock = nullptr;
            if (useCache && m_pShardedLRUCache)
            {
                lock = &(m_pShardedLRUCache->getlock(key));
                lock->lock();
            
                if (m_pShardedLRUCache->put(key, (void *)(value.data()), (int)(value.size())))
                {
                    if (GetKey(key) == InvalidPointer)
                    {
                        uintptr_t tmpblocks = InvalidPointer;
                        if (m_buffer.unsafe_size() > m_bufferLimit)
                        {
                            while (!m_buffer.try_pop(tmpblocks));
                        }
                        else
                        {
                            tmpblocks = (uintptr_t)(new AddressType[m_blockLimit]);
                        }
                        // The 0th element of the block address list represents the data size; set it to -1.
                        memset((AddressType *)tmpblocks, -1, sizeof(AddressType) * m_blockLimit);
                        AtKey(key) = tmpblocks;
                    }
                    int64_t *postingSize = (int64_t *)GetKey(key);
                    int oldblocks = (static_cast<int>(*postingSize) < 0) ? 0 : ((static_cast<int>(*postingSize) + PageSize - 1) >> PageSizeEx);
                    if (blocks - oldblocks > 0)
                    {
                        if (!m_pBlockController.GetBlocks(postingSize + oldblocks + 1, blocks - oldblocks))
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[Put] Not enough blocks in the pool can be allocated!\n");
                            if (lock) lock->unlock();
                            return ErrorCode::DiskIOFail;
                        }
                    }
                    else if (blocks - oldblocks < 0)
                    {
                        if (!m_pBlockController.ReleaseBlocks(postingSize + blocks + 1, oldblocks - blocks))
                        {
                            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[Put] Release blocks back to the pool failed!\n");
                            if (lock) lock->unlock();
                            return ErrorCode::DiskIOFail;
                        }
                    }
                    int64_t oldSize = *postingSize;
                    while (InterlockedCompareExchange(postingSize, newValueMeta, oldSize) != oldSize) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[Put] Update posting size for key %lld failed due to concurrent update! Retry...\n", (std::int64_t)key);
                        oldSize = *postingSize;
                    }
                    lock->unlock();
                    return ErrorCode::Success;
                }
            }
            uintptr_t tmpblocks = InvalidPointer;
            // If this key has not been assigned mapping blocks yet, allocate a batch.
            if (GetKey(key) == InvalidPointer) {
                // If there are spare blocks in m_buffer, use them directly; otherwise, allocate a new batch.
                if (m_buffer.unsafe_size() > m_bufferLimit) {
                    while (!m_buffer.try_pop(tmpblocks));
                }
                else {
                    tmpblocks = (uintptr_t)(new AddressType[m_blockLimit]);
                }
                // The 0th element of the block address list represents the data size; set it to -1.
                memset((AddressType*)tmpblocks, -1, sizeof(AddressType) * m_blockLimit);
            } else {
                tmpblocks = GetKey(key);
            }
            int64_t* postingSize = (int64_t*)tmpblocks;
            // If postingSize is less than 0, it means the mapping block is newly allocated—directly
            if (static_cast<int>(*postingSize) < 0) {
                if (!m_pBlockController.GetBlocks(postingSize + 1, blocks))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "[Put] Not enough blocks in the pool can be allocated!\n");
                    if (lock) lock->unlock();
                    return ErrorCode::DiskIOFail;
                }                                
                if (!m_pBlockController.WriteBlocks(postingSize + 1, blocks, value, timeout, reqs))
                {
                    m_pBlockController.ReleaseBlocks(postingSize + 1, blocks);
                    memset(postingSize, -1, sizeof(AddressType) * blocks);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[Put] Write new block failed!\n");
                    if (lock) lock->unlock();
                    return ErrorCode::DiskIOFail;
                }
                int64_t oldSize = *postingSize;
                while (InterlockedCompareExchange(postingSize, newValueMeta, oldSize) != oldSize) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[Put] Update posting size for key %lld failed due to concurrent update! Retry...\n", (std::int64_t)key);
                    oldSize = *postingSize;
                } 
                AtKey(key) = tmpblocks;
            }
            else {
                uintptr_t partialtmpblocks = InvalidPointer;
                // Take a batch of mapping blocks from the buffer, and return a batch back later.
                while (!m_buffer.try_pop(partialtmpblocks));
                // Acquire a new batch of disk blocks and write data directly.
                // To ensure the effectiveness of the checkpoint, new blocks must be allocated for writing here.
                if (!m_pBlockController.GetBlocks((AddressType*)partialtmpblocks + 1, blocks))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[Put] Not enough blocks in the pool can be allocated!\n");
                    m_buffer.push(partialtmpblocks);
                    if (lock) lock->unlock();
                    return ErrorCode::DiskIOFail;
                }
                if (!m_pBlockController.WriteBlocks((AddressType*)partialtmpblocks + 1, blocks, value, timeout, reqs))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[Put] Write new block failed!\n");
                    m_pBlockController.ReleaseBlocks((AddressType*)partialtmpblocks + 1, blocks);
                    m_buffer.push(partialtmpblocks);
                    if (lock) lock->unlock();
                    return ErrorCode::DiskIOFail;
                }
                *((int64_t*)partialtmpblocks) = newValueMeta;
                
                // Release the original blocks
                m_pBlockController.ReleaseBlocks(postingSize + 1, (static_cast<int>(*postingSize) + PageSize - 1) >> PageSizeEx);
                m_buffer.push((uintptr_t)postingSize);
                while (InterlockedCompareExchange(&AtKey(key), partialtmpblocks, (uintptr_t)postingSize) != (uintptr_t)postingSize) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[Put] Update key mapping failed due to concurrent update! Retry...\n");
                    postingSize = (int64_t*)GetKey(key);
                }
            }
            if (lock) lock->unlock();
            return ErrorCode::Success;
        }

        ErrorCode Put(const SizeType key, const std::string& value, const std::chrono::microseconds& timeout, std::vector<Helper::AsyncReadRequest>* reqs) override {
            return Put(key, value, timeout, reqs, true);
        }

        ErrorCode Put(const std::string &key, const std::string& value, const std::chrono::microseconds& timeout, std::vector<Helper::AsyncReadRequest>* reqs) override {
            return Put(std::stoi(key), value, timeout, reqs, true);
        }

        // FileIO is single-process: no cross-process CAS is needed. Internal thread-safety
        // is already handled by Put (per-key lock + InterlockedCompareExchange). Delegate to Put
        // so callers that target the unified KeyValueIO interface (e.g. WriteDownAllPostingToDB)
        // also work for the FILEIO storage backend.
        ErrorCode MergeWithCAS(const SizeType key, const std::string& value,
                               const std::chrono::microseconds& timeout,
                               std::vector<Helper::AsyncReadRequest>* reqs, int& size) override {
            size = static_cast<int>(value.size());
            return Put(key, value, timeout, reqs, true);
        }

        ErrorCode Check(const SizeType key, std::vector<std::uint8_t> *visited) override
        {
            int64_t *postingSize = (int64_t *)GetKey(key);
            if ((uintptr_t)postingSize == InvalidPointer)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[Check] Key error: Key %lld\n", (int64_t)key);
                return ErrorCode::Key_OverFlow;
            }

            int blocks = ((static_cast<int>(*postingSize) + PageSize - 1) >> PageSizeEx);
            for (int i = 1; i <= blocks; i++)
            {
                if (postingSize[i] < 0 || postingSize[i] >= m_pBlockController.TotalBlocks())
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "[Check] Key %lld failed: error block id %d (should be 0 ~ %d)\n", (int64_t)key,
                                 (int)(postingSize[i]), m_pBlockController.TotalBlocks());
                    return ErrorCode::Block_IDError;
                }
                if (visited == nullptr) continue;

                if (postingSize[i] >= visited->size()) 
                {
                	SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[Check] Key %lld failed: BLOCK %lld exceed total block size %zu!\n", (int64_t)key,
                                postingSize[i], visited->size());
                    continue;
                }
                
                if (visited->at(postingSize[i]) > 0)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[Check] Key %lld failed: Block %lld double used!\n", (int64_t)key, postingSize[i]);
                    return ErrorCode::Block_IDError;
                }
                else
                {
                    InterlockedExchange8((char*)(&(visited->at(postingSize[i]))), 1);
                }
            }
            return ErrorCode::Success;
        }

        int64_t GetApproximateMemoryUsage() override
        {
            int64_t result = 0;
            {
                std::shared_lock<std::shared_timed_mutex> lock(m_updateMutex);
                result = m_pBlockMapping.size() * m_blockLimit * sizeof(AddressType);
            }
            if (m_pShardedLRUCache)
            {
                result += m_pShardedLRUCache->GetApproximateMemoryUsage();
            }

            return result;
        }
        
        ErrorCode Merge(const SizeType key, const std::string &value, const std::chrono::microseconds &timeout,
                        std::vector<Helper::AsyncReadRequest> *reqs, int& size) override
        {
            std::shared_timed_mutex *lock = nullptr;
            if (m_pShardedLRUCache)
            {
                lock = &(m_pShardedLRUCache->getlock(key));
                lock->lock();
            }

            int64_t* postingSize = (int64_t*)GetKey(key);
            if (((uintptr_t)postingSize) == InvalidPointer || static_cast<int>(*postingSize) < 0)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[Merge] Key %lld failed: postingSize < 0\n", (int64_t)key);
                if (lock) lock->unlock();
                return ErrorCode::Key_NotFound;
                //return Put(key, value, timeout, reqs);
            }

            auto newSize = static_cast<int>(*postingSize) + value.size();
            int newblocks = (int)(((newSize + PageSize - 1) >> PageSizeEx));
            if (newblocks >= m_blockLimit) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "[Merge] Key %lld failed: new size %d bytes requires %d blocks (limit: %d)\n",
                    (int64_t)key, (int)newSize, newblocks, m_blockLimit);

                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "[Merge] Original size: %d bytes, Merge size: %d bytes\n",
                    static_cast<int>(*postingSize), static_cast<int>(value.size()));
                if (lock) lock->unlock();
                return ErrorCode::Posting_OverFlow;
            }
            
            ChecksumType checksum = (ChecksumType)(*postingSize >> 32);
            checksum = m_checkSum.AppendChecksum(checksum, value.c_str(), (int)(value.size()));
            int64_t newValueMeta = (newSize | (((int64_t)checksum) << 32));
            if (m_pShardedLRUCache && m_pShardedLRUCache->merge(key, (void *)(value.data()), (int)(value.size())))
            {
                int oldblocks = ((static_cast<int>(*postingSize) + PageSize - 1) >> PageSizeEx);
                int allocblocks = newblocks - oldblocks;
                if (allocblocks > 0 && !m_pBlockController.GetBlocks(postingSize + 1 + oldblocks, allocblocks))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[Merge] Not enough blocks in the pool can be allocated!\n");
                    if (lock) lock->unlock();
                    return ErrorCode::DiskIOFail;
                }
                int64_t oldSize = *postingSize;
                while (InterlockedCompareExchange(postingSize, newValueMeta, oldSize) != oldSize) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[Merge] Update posting size for key %lld failed due to concurrent update! Retry...\n", (std::int64_t)key);
                    oldSize = *postingSize;
                }
                if (lock) lock->unlock();
                size = (int)newSize;
                return ErrorCode::Success;
            }

            postingSize = (int64_t *)GetKey(key);
            int sizeInPage = static_cast<int>(*postingSize) % PageSize;    // Actual size of the last block
            int oldblocks = static_cast<int>(*postingSize) >> PageSizeEx;
            int allocblocks = newblocks - oldblocks;
            // If the last block is not full, we need to read it first, then append the new data, and write it back.
            if (sizeInPage != 0) {
                std::string newValue;
                AddressType readreq[] = { sizeInPage, *(postingSize + 1 + oldblocks) };
                if (!m_pBlockController.ReadBlocks(readreq, &newValue, timeout, reqs))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[Merge] Cannot read original posting!\n");
                    if (lock) lock->unlock();
                    return ErrorCode::DiskIOFail;
                }
                //std::string lastblock = before.substr(before.size() - sizeInPage);
                //PrintPostingDiff(lastblock, newValue, "0");
                newValue += value;

                uintptr_t tmpblocks = InvalidPointer;
                while (!m_buffer.try_pop(tmpblocks));
                memcpy((AddressType*)tmpblocks, postingSize, sizeof(AddressType) * (oldblocks + 1));
                if (!m_pBlockController.GetBlocks((AddressType *)tmpblocks + 1 + oldblocks, allocblocks))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[Merge] Not enough blocks in the pool can be allocated!\n");
                    m_buffer.push(tmpblocks);
                    if (lock) lock->unlock();
                    return ErrorCode::DiskIOFail;
                }   
                if (!m_pBlockController.WriteBlocks((AddressType *)tmpblocks + 1 + oldblocks, allocblocks, newValue,
                                                    timeout, reqs))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                "[Merge] Write new block failed!\n");
                    m_pBlockController.ReleaseBlocks((AddressType *)tmpblocks + 1 + oldblocks, allocblocks);
                    m_buffer.push(tmpblocks);
                    if (lock) lock->unlock();
                    return ErrorCode::DiskIOFail;
                }
                *((int64_t *)tmpblocks) = newValueMeta;

                // This is also to ensure checkpoint correctness, so we release the partially used block and allocate a new one.
                m_pBlockController.ReleaseBlocks(postingSize + 1 + oldblocks, 1);
                m_buffer.push((uintptr_t)postingSize);
                while (InterlockedCompareExchange(&AtKey(key), tmpblocks, (uintptr_t)postingSize) != (uintptr_t)postingSize) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[Merge] Posting pointer changed during merge! Key %lld, Retry...\n", (int64_t)key);
                    postingSize = (int64_t*)GetKey(key);
                }
            }
            else {  // Otherwise, directly allocate a new batch of blocks to append after the current ones.
                if (!m_pBlockController.GetBlocks(postingSize + 1 + oldblocks, allocblocks))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "[Merge] Not enough blocks in the pool can be allocated!\n");
                    if (lock) lock->unlock();
                    return ErrorCode::DiskIOFail;
                }
                             
                if (!m_pBlockController.WriteBlocks(postingSize + 1 + oldblocks, allocblocks, value, timeout, reqs))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[Merge] Write new block failed!\n");
                    m_pBlockController.ReleaseBlocks(postingSize + 1 + oldblocks, allocblocks);
                    if (lock) lock->unlock();
                    return ErrorCode::DiskIOFail;
                }
                
                //*postingSize = (newSize | (((int64_t)checksum)<< 32));
                int64_t oldSize = *postingSize;
                while (InterlockedCompareExchange(postingSize, newValueMeta, oldSize) != oldSize) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "[Merge] Posting size changed during merge! Key %lld, old size %d, new size %d, Retry...\n",
                                 (int64_t)key, (int)(oldSize & 0xFFFFFFFF), (int)(newValueMeta & 0xFFFFFFFF));
                    oldSize = *postingSize;
                }
            }
            if (lock) lock->unlock();
            size = (int)newSize;
            return ErrorCode::Success;
        }


        ErrorCode Delete(SizeType key) override {
            std::shared_timed_mutex *lock = nullptr;
            if (m_pShardedLRUCache)
            {
                lock = &(m_pShardedLRUCache->getlock(key));
                lock->lock();
                m_pShardedLRUCache->del(key);
            }

            int64_t* postingSize = (int64_t*)GetKey(key);
            if (((uintptr_t)postingSize) == InvalidPointer || static_cast<int>(*postingSize) < 0)
            {
                if (lock) lock->unlock();
                return ErrorCode::Key_NotFound;
            }

            int blocks = (static_cast<int>(*postingSize)+ PageSize - 1) >> PageSizeEx;
            m_pBlockController.ReleaseBlocks(postingSize + 1, blocks);
            m_buffer.push((uintptr_t)postingSize);

            {
                std::unique_lock<std::shared_timed_mutex> updatelock(m_updateMutex);
                m_pBlockMapping.unsafe_erase(key);
            }
            if (lock) lock->unlock();
            return ErrorCode::Success;
        }

        ErrorCode Delete(const std::string &key) {
            return Delete(std::stoi(key));
        }

        void ForceCompaction() {
            Save(m_mappingPath);
        }

        void GetStat() {
            int remainBlocks = m_pBlockController.RemainBlocks();
            int reserveBlocks = m_pBlockController.ReserveBlocks();
            int totalBlocks = m_pBlockController.TotalBlocks();
            int remainGB = ((long long)(remainBlocks + reserveBlocks) >> (30 - PageSizeEx));
            // int remainGB = remainBlocks >> 20 << 2;
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Total %d blocks, Remain %d blocks, Reserve %d blocks, totally %d GB\n", totalBlocks, remainBlocks, reserveBlocks, remainGB);
            if (m_pShardedLRUCache) {
                auto cache_stat = m_pShardedLRUCache->get_stat();
                if (cache_stat.query + cache_stat.put)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LRUCache query: %lld/%lld (%f), put: %lld/%lld (%f)\n",
                        cache_stat.query_hits,
                        cache_stat.query,
                        cache_stat.query ? cache_stat.query_hits / (float) cache_stat.query : 0.0,
                        cache_stat.evict,
                        cache_stat.put,
                        cache_stat.put ? cache_stat.evict / (float) cache_stat.put : 0.0
                    );
                }
            }
            m_pBlockController.IOStatistics();
        }

        ErrorCode Load(std::string path) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load mapping From %s\n", path.c_str());
            auto ptr = f_createIO();
            if (ptr == nullptr || !ptr->Initialize(path.c_str(), std::ios::binary | std::ios::in)) return ErrorCode::FailedOpenFile;

            SizeType CR, mycols;
            IOBINARY(ptr, ReadBinary, sizeof(SizeType), (char*)&CR);
            IOBINARY(ptr, ReadBinary, sizeof(SizeType), (char*)&mycols);
            if (mycols > m_blockLimit + 1) m_blockLimit = mycols - 1;

            for (int i = 0; i < CR; i++) {
                AddressType key;
                IOBINARY(ptr, ReadBinary, sizeof(AddressType), (char*)&key);
                AtKey((SizeType)key) = (uintptr_t)(new AddressType[m_blockLimit]);
                IOBINARY(ptr, ReadBinary, sizeof(AddressType) * m_blockLimit, (char*)GetKey(key));
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load mapping (%lld,%lld) Finish!\n", (std::int64_t)CR, (std::int64_t)mycols);
            return ErrorCode::Success;
        }
        
        ErrorCode Save(std::string path) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Save mapping To %s\n", path.c_str());
            if (m_pShardedLRUCache) {
                if (!m_pShardedLRUCache->flush()) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to flush cache when saving mapping!\n");
                    return ErrorCode::Fail;
                }
            }

            auto ptr = f_createIO();
            if (ptr == nullptr || !ptr->Initialize(path.c_str(), std::ios::binary | std::ios::out)) return ErrorCode::FailedCreateFile;

            std::unique_lock<std::shared_timed_mutex> updatelock(m_updateMutex);
            SizeType CR = m_pBlockMapping.size();
            SizeType CC = m_blockLimit + 1;
            IOBINARY(ptr, WriteBinary, sizeof(SizeType), (char*)&CR);
            IOBINARY(ptr, WriteBinary, sizeof(SizeType), (char*)&CC);
            for (auto& it : m_pBlockMapping) {
                AddressType key = (AddressType)it.first;
                IOBINARY(ptr, WriteBinary, sizeof(AddressType), (char*)&(key));
                int64_t* postingSize = (int64_t*)(it.second);
                IOBINARY(ptr, WriteBinary, sizeof(AddressType) * m_blockLimit, (char*)postingSize);
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Save mapping (%lld, %lld) Finish!\n", (std::int64_t)CR, (std::int64_t)CC);
            return ErrorCode::Success;
        }

        ErrorCode Checkpoint(std::string prefix) override {
            std::string filename = prefix + FolderSep + m_mappingPath.substr(m_mappingPath.find_last_of(FolderSep) + 1);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO: saving block mapping to %s\n", filename.c_str());
            Save(filename);
            return m_pBlockController.Checkpoint(filename + "_postings");
        }

        int64_t GetNumBlocks() override
        {
            return m_pBlockController.TotalBlocks();
        }

        static const uintptr_t InvalidPointer = 0xffffffffffffffff;

    private:
        std::string m_mappingPath;
        int m_layer;
        SizeType m_blockLimit;
        Helper::Concurrent::ConcurrentMap<SizeType, uintptr_t> m_pBlockMapping;
        bool m_checkSumMultiGet;
        COMMON::Checksum m_checkSum;
        SizeType m_bufferLimit;
        Helper::Concurrent::ConcurrentQueue<uintptr_t> m_buffer;

        std::shared_ptr<Helper::ThreadPool> m_compactionThreadPool;
        BlockController m_pBlockController;
        ShardedLRUCache *m_pShardedLRUCache{nullptr};

        bool m_shutdownCalled;
        std::shared_timed_mutex m_updateMutex;
    };
}
#endif
