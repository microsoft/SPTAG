#ifndef _SPTAG_SPANN_EXTRAFILECONTROLLER_H_
#define _SPTAG_SPANN_EXTRAFILECONTROLLER_H_
#define USE_ASYNC_IO
// #define USE_FILE_DEBUG
#include "inc/Helper/KeyValueIO.h"
#include "inc/Core/Common/Dataset.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Helper/ThreadPool.h"
#include <cstdlib>
#include <memory>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <fcntl.h>
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_hash_map.h>
#include <sys/syscall.h>
#include <linux/aio_abi.h>
#include <list>
namespace SPTAG::SPANN {
    typedef std::int64_t AddressType;
    class FileIO : public Helper::KeyValueIO {
        class BlockController {
        private:
            char m_filePath[1024];
            int fd = -1;

            static constexpr AddressType kFileIoStartBlocks = (3ULL << 30) >> PageSizeEx; // 3GB start pool
            static constexpr AddressType kFileIoGrowthBlocks = (1ULL << 30) >> PageSizeEx; // 1GB growth pool
            static constexpr float fFileIoThreshold = 0.05f;
            static constexpr AddressType kFileIODefaultMaxBlocks = (3000ULL << 30) >> PageSizeEx; // 300 GB            
            
            static constexpr const char* kFileIoDepth = "SPFRESH_FILE_IO_DEPTH";
            static constexpr int kSsdFileIoDefaultIoDepth = 1024;
            static constexpr const char* kFileIoThreadNum = "SPFRESH_FILE_IO_THREAD_NUM";
            static constexpr int kSsdFileIoDefaultIoThreadNum = 64;
            static constexpr const char* kFileIoAlignment = "SPFRESH_FILE_IO_ALIGNMENT";
            static constexpr int kSsdFileIoDefaultAlignment = 4096;

            tbb::concurrent_queue<AddressType> m_blockAddresses;
            tbb::concurrent_queue<AddressType> m_blockAddresses_reserve;

            std::mutex m_expandLock;
            std::atomic<AddressType> m_totalAllocatedBlocks { 0 };
            
            pthread_t m_fileIoTid;
            pthread_t m_ioStatisticsTid;
            volatile bool m_fileIoThreadStartFailed = false;
            volatile bool m_fileIoThreadReady = false;
            volatile bool m_fileIoThreadExiting = false;

            int m_ssdFileIoAlignment = kSsdFileIoDefaultAlignment;
            int m_ssdFileIoDepth = kSsdFileIoDefaultIoDepth;
            int m_ssdFileIoThreadNum = kSsdFileIoDefaultIoThreadNum;
            struct SubIoRequest {
                struct iocb myiocb;
                AddressType real_size;
                AddressType offset;
                void* app_buff;
                BlockController* ctrl;
                int posting_id;
            };
            tbb::concurrent_queue<SubIoRequest *> m_submittedSubIoRequests;
            struct IoContext {
                std::vector<SubIoRequest> sub_io_requests;
                std::queue<SubIoRequest *> free_sub_io_requests;
            };
            thread_local static struct IoContext m_currIoContext;
            thread_local static tbb::concurrent_hash_map<int, std::pair<int, uint64_t>> fd_to_id_iocp;
            thread_local static int debug_fd;
            std::chrono::high_resolution_clock::time_point m_startTime;

            int m_maxId = 0;
            std::queue<int> m_idQueue;

            std::mutex m_uniqueResourceMutex;

            std::unique_ptr<char[]> m_memBuffer;

            std::mutex m_initMutex;
            int m_numInitCalled = 0;

            int m_batchSize;
            static std::atomic<int> m_ioCompleteCount;
            int m_preIOCompleteCount = 0;
            int64_t m_preIOBytes = 0;
            std::chrono::time_point<std::chrono::high_resolution_clock> m_preTime = std::chrono::high_resolution_clock::now();

            std::atomic<int64_t> m_batchReadTimes;
            std::atomic<int64_t> m_batchReadTimeouts;

            void InitializeFileIo();

            static void* IoStatisticsThread(void* args) {
                pthread_exit(NULL);
            };

        private:
            bool ExpandFile(AddressType blocksToAdd);
            bool NeedsExpansion();

        public:
            bool Initialize(std::string filePath, int batchSize);

            bool GetBlocks(AddressType* p_data, int p_size);

            bool ReleaseBlocks(AddressType* p_data, int p_size);

            bool ReadBlocks(AddressType* p_data, std::string* p_value, const std::chrono::microseconds &timeout = std::chrono::microseconds::max());

            bool ReadBlocks(AddressType* p_data, ByteArray& p_value, const std::chrono::microseconds &timeout = std::chrono::microseconds::max());

            bool ReadBlocks(const std::vector<AddressType*>& p_data, std::vector<std::string>* p_value, const std::chrono::microseconds &timeout = std::chrono::microseconds::max());

            bool ReadBlocks(const std::vector<AddressType*>& p_data, std::vector<ByteArray>& p_value, const std::chrono::microseconds &timeout = std::chrono::microseconds::max());

            bool WriteBlocks(AddressType* p_data, int p_size, const std::string& p_value);

            bool WriteBlocks(AddressType* p_data, int p_size, const ByteArray& p_value);

            bool IOStatistics();

            bool ShutDown();

            int RemainBlocks() {
                return m_blockAddresses.unsafe_size();
            };

            ErrorCode Checkpoint(std::string prefix) {
                std::string filename = prefix + "_blockpool";
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO::BlockController::Checkpoint - Starting block pool save...\n");

                // Move reserved blocks back to main queue
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO::BlockController::Checkpoint - Reload reserved blocks...\n");
                AddressType currBlockAddress = 0;
                for (int count = 0; count < m_blockAddresses_reserve.unsafe_size(); count++) {
                    if (m_blockAddresses_reserve.try_pop(currBlockAddress)) {
                        m_blockAddresses.push(currBlockAddress);
                    }
                }

                int blocks = RemainBlocks();
                AddressType totalBlocks = m_totalAllocatedBlocks.load();

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO::BlockController::Checkpoint - Total allocated blocks: %llu\n",
                            static_cast<unsigned long long>(totalBlocks));
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO::BlockController::Checkpoint - Remaining free blocks: %d\n", blocks);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO::BlockController::Checkpoint - Saving to file: %s\n", filename.c_str());

                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(filename.c_str(), std::ios::binary | std::ios::out)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::Checkpoint - Failed to open file: %s\n", filename.c_str());
                    return ErrorCode::FailedCreateFile;
                }

                // Write block count
                IOBINARY(ptr, WriteBinary, sizeof(SizeType), reinterpret_cast<char*>(&blocks));
                // Write total allocated blocks
                IOBINARY(ptr, WriteBinary, sizeof(AddressType), reinterpret_cast<char*>(&totalBlocks));
                // Write individual block addresses
                for (auto it = m_blockAddresses.unsafe_begin(); it != m_blockAddresses.unsafe_end(); ++it) {
                    IOBINARY(ptr, WriteBinary, sizeof(AddressType), reinterpret_cast<const char*>(&(*it)));
                }

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO::BlockController::Checkpoint - Save completed successfully.\n");
                return ErrorCode::Success;
            }

            ErrorCode LoadBlockPool(std::string prefix, bool allowinit) {
                std::string blockfile = prefix + "_blockpool";

                if (allowinit && !fileexists(blockfile.c_str())) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "BlockController::LoadBlockPool: initializing fresh pool (no existing file found: %s)\n",
                        blockfile.c_str());

                    for (AddressType i = 0; i < kFileIoStartBlocks; ++i) {
                        m_blockAddresses.push(i);
                    }

                    m_totalAllocatedBlocks.store(kFileIoStartBlocks);

                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "BlockController::LoadBlockPool: initialized with %llu blocks (%.2f GB)\n",
                        static_cast<unsigned long long>(kFileIoStartBlocks),
                        (kFileIoStartBlocks << PageSizeEx) / double(1 << 30));

                } else {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "BlockController::LoadBlockPool: loading from file %s\n", blockfile.c_str());

                    auto ptr = f_createIO();
                    if (ptr == nullptr || !ptr->Initialize(blockfile.c_str(), std::ios::binary | std::ios::in)) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                            "BlockController::LoadBlockPool: failed to open blockpool file: %s\n", blockfile.c_str());
                        return ErrorCode::Fail;
                    }

                    AddressType blocks = 0;
                    AddressType totalAllocated = 0;

                    // Read block count
                    IOBINARY(ptr, ReadBinary, sizeof(AddressType), reinterpret_cast<char*>(&blocks));
                    // Read total allocated block count
                    IOBINARY(ptr, ReadBinary, sizeof(AddressType), reinterpret_cast<char*>(&totalAllocated));

                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "BlockController::LoadBlockPool: reading %llu blocks into pool (%.2f GB), total allocated: %llu blocks\n",
                        static_cast<unsigned long long>(blocks),
                        (blocks << PageSizeEx) / static_cast<double>(1ULL << 30),
                        static_cast<unsigned long long>(totalAllocated));

                    AddressType currBlockAddress = 0;
                    for (AddressType i = 0; i < blocks; ++i) {
                        IOBINARY(ptr, ReadBinary, sizeof(AddressType), reinterpret_cast<char*>(&currBlockAddress));
                        m_blockAddresses.push(currBlockAddress);
                    }

                    m_totalAllocatedBlocks.store(totalAllocated);

                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "BlockController::LoadBlockPool: block pool initialized. Available: %llu, Total allocated: %llu\n",
                        static_cast<unsigned long long>(blocks),
                        static_cast<unsigned long long>(totalAllocated));
                }
                return ErrorCode::Success;
            }

            ErrorCode Recovery(std::string prefix, int batchSize) {
                std::lock_guard<std::mutex> lock(m_initMutex);
                m_numInitCalled++;
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO Recovery: Loading block pool\n");
                ErrorCode ret = LoadBlockPool(prefix, false);
		        if (ErrorCode::Success != ret) return ret;

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO Recovery: Initializing FileIO\n");
                
                if (m_numInitCalled == 1) {
                    m_batchSize = batchSize;
		            InitializeFileIo();
                    while(!m_fileIoThreadReady && !m_fileIoThreadStartFailed);
                    if (m_fileIoThreadStartFailed) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::Initialize failed\n");
                        return ErrorCode::Fail;
                    }
                }

                // Create sub I/O request pool
                m_currIoContext.sub_io_requests.resize(m_ssdFileIoDepth);
                for (auto &sr : m_currIoContext.sub_io_requests) {
                    sr.app_buff = nullptr;
                    auto buf_ptr = BLOCK_ALLOC(m_ssdFileIoAlignment, PageSize);
                    if (buf_ptr == nullptr) {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::Initialize failed: aligned_alloc failed\n");
                        return ErrorCode::Fail;
                    }
                    sr.myiocb.aio_buf = reinterpret_cast<uint64_t>(buf_ptr);
                    sr.myiocb.aio_fildes = fd;
                    sr.myiocb.aio_data = reinterpret_cast<uintptr_t>(&sr);
                    sr.myiocb.aio_nbytes = PageSize;
                    sr.ctrl = this;
                    m_currIoContext.free_sub_io_requests.push(&sr);
                }
                return ErrorCode::Success;
            }
        };

        class CompactionJob : public Helper::ThreadPool::Job
        {
        private:
            FileIO* m_fileIO;

        public:
            CompactionJob(FileIO* fileIO): m_fileIO(fileIO) {}

            ~CompactionJob() {}

            inline void exec(IAbortOperation* p_abort) override {
                m_fileIO->ForceCompaction();
            }
        };

        class LRUCache {
            int capacity;   // Page Num
            int size;
            std::list<SizeType> keys;  // Page Address
            std::unordered_map<SizeType, std::pair<std::string, std::list<SizeType>::iterator>> cache;    // Page Address -> Page Address in Cache
            std::mutex mu;
            int64_t queries;
            int64_t hits;

        public:
            LRUCache(int capacity) {
                this->capacity = capacity;
                this->size = 0;
                this->queries = 0;
                this->hits = 0;
            }

            bool get(SizeType key, void* value) {
                mu.lock();
                queries++;
                auto it = cache.find(key);
                if (it == cache.end()) {
                    mu.unlock();
                    return false;  // If the key does not exist, return -1
                }
                // Update access order, move the key to the head of the linked list
                memcpy(value, it->second.first.data(), it->second.first.size());
                keys.splice(keys.begin(), keys, it->second.second);
                it->second.second = keys.begin();
                hits++;
                mu.unlock();
                return true;
            }

            bool put(SizeType key, void* value, int put_size) {
                mu.lock();
                auto it = cache.find(key);
                if (it != cache.end()) {
                    if (put_size > capacity) {
                        size -= it->second.first.size();
                        keys.erase(it->second.second);
                        cache.erase(it);
                        mu.unlock();
                        return false;
                    }
                    auto keys_it = it->second.second;
                    keys.splice(keys.begin(), keys, keys_it);
                    it->second.second = keys.begin();
                    keys_it = keys.begin();
                    auto delta_size = put_size - it->second.first.size();
                    while ((capacity - size) < delta_size && (keys.size() > 1)) {
                        auto last = keys.back();
                        auto it = cache.find(last);
                        size -= it->second.first.size();
                        cache.erase(it);
                        keys.pop_back();
                    }
                    it->second.first.resize(put_size);
                    memcpy(it->second.first.data(), value, put_size);
                    size += delta_size;
                    mu.unlock();
                    return true;
                }
                if (put_size > capacity) {
                    mu.unlock();
                    return false;
                }
                while (put_size > (capacity - size) && (!keys.empty())) {
                    auto last = keys.back();
                    auto it = cache.find(last);
                    size -= it->second.first.size();
                    cache.erase(it);
                    keys.pop_back();
                }
                auto keys_it = keys.insert(keys.begin(), key);
                cache.insert({key, {std::string((char*)value, put_size), keys_it}});
                size += put_size;
                mu.unlock();
                return true;
            }

            bool del(SizeType key) {
                mu.lock();
                auto it = cache.find(key);
                if (it == cache.end()) {
                    mu.unlock();
                    return false;  // If the key does not exist, return false
                }
                size -= it->second.first.size();
                keys.erase(it->second.second);
                cache.erase(it);
                mu.unlock();
                return true;
            }

            bool merge(SizeType key, void* value, AddressType merge_size) {
                mu.lock();
                // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LRUCache: merge size: %lld\n", merge_size);
                auto it = cache.find(key);
                if (it == cache.end()) {
                    // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LRUCache: merge key not found\n");
                    mu.unlock();
                    return false;  // If the key does not exist, return false
                }
                if (merge_size + it->second.first.size() > capacity) {
                    size -= it->second.first.size();
                    keys.erase(it->second.second);
                    cache.erase(it);
                    // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LRUCache: merge size exceeded\n");
                    mu.unlock();
                    return false;
                }
                keys.splice(keys.begin(), keys, it->second.second);
                it->second.second = keys.begin();
                while((capacity - size) < merge_size && (keys.size() > 1)) {
                    auto last = keys.back();
                    auto it = cache.find(last);
                    size -= it->second.first.size();
                    cache.erase(it);
                    keys.pop_back();
                }
                it->second.first.append((char*)value, merge_size);
                size += merge_size;
                // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "LRUCache: merge success\n");
                mu.unlock();
                return true;
            }
            
            std::pair<int64_t, int64_t> get_stat() {
                return {queries, hits};
            }
        }; 

        class ShardedLRUCache {
            int shards;
            std::vector<LRUCache*> caches;
            SizeType hash(SizeType key) const {
                return key % shards;
            }
        public:
            ShardedLRUCache(int shards, int capacity) : shards(shards) {
                caches.resize(shards);
                for (int i = 0; i < shards; i++) {
                    caches[i] = new LRUCache(capacity / shards);
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

            bool get(SizeType key, void* value) {
                return caches[hash(key)]->get(key, value);
            }

            bool put(SizeType key, void* value, SizeType put_size) {
                return caches[hash(key)]->put(key, value, put_size);
            }

            bool del(SizeType key) {
                return caches[hash(key)]->del(key);
            }

            bool merge(SizeType key, void* value, AddressType merge_size) {
                return caches[hash(key)]->merge(key, value, merge_size);
            }

            std::pair<int64_t, int64_t> get_stat() {
                int64_t queries = 0, hits = 0;
                for (int i = 0; i < shards; i++) {
                    auto stat = caches[i]->get_stat();
                    queries += stat.first;
                    hits += stat.second;
                }
                return {queries, hits};
            }
        };

    public:
        FileIO(const char* filePath, SizeType blockSize, SizeType capacity, SizeType postingBlocks, SizeType bufferSize = 1024, int batchSize = 64, bool recovery = false, int compactionThreads = 1) {
            m_mappingPath = std::string(filePath);
            m_blockLimit = postingBlocks + 1;
            m_bufferLimit = bufferSize;

            const char* fileIoUseLock = getenv(kFileIoUseLock);
            if(fileIoUseLock) {
                if(strcmp(fileIoUseLock, "True") == 0) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO: Using lock\n");
                    m_fileIoUseLock = true;
                    const char* fileIoLockSize = getenv(kFileIoLockSize);
                    if(fileIoLockSize) {
                        m_fileIoLockSize = atoi(fileIoLockSize);
                    }
                    m_rwMutex = std::vector<std::shared_mutex>(m_fileIoLockSize);
                }
                else {
                    m_fileIoUseLock = false;
                }
            }
            else {
                m_fileIoUseLock = false;
            }
            const char* fileIoUseCache = getenv(kFileIoUseCache);
            if (fileIoUseCache) {
                if (strcmp(fileIoUseCache, "True") == 0) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO: Using cache\n");
                    m_fileIoUseCache = true;
                }
                else {
                    m_fileIoUseCache = false;
                }
            }
            if (m_fileIoUseCache) {
                const char* fileIoCacheSize = getenv(kFileIoCacheSize);
                const char* fileIoCacheShards = getenv(kFileIoCacheShards);
                int capacity = kSsdFileIoDefaultCacheSize;
                int shards = kSsdFileIoDefaultCacheShards;
                if(fileIoCacheSize) {
                    capacity = atoi(fileIoCacheSize);
                } 
                if(fileIoCacheShards) {
                    shards = atoi(fileIoCacheShards);
                }
                m_pShardedLRUCache = new ShardedLRUCache(shards, capacity);
            }

            if (fileexists(m_mappingPath.c_str())) {
                Load(m_mappingPath, blockSize, capacity);
		        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load block mapping successfully from %s!\n", m_mappingPath.c_str());
            } else {
		        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Initialize block mapping successfully!\n");
                m_pBlockMapping.Initialize(0, 1, blockSize, capacity);
            }
            for (int i = 0; i < bufferSize; i++) {
                m_buffer.push((uintptr_t)(new AddressType[m_blockLimit]));
            }
            m_compactionThreadPool = std::make_shared<Helper::ThreadPool>();
            m_compactionThreadPool->init(compactionThreads);
            if (recovery) {
                if (m_pBlockController.Recovery(std::string(filePath), batchSize) != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to Recover FileIO!\n");
                    exit(0);
                }
            } else if (!m_pBlockController.Initialize(std::string(filePath), batchSize)) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to Initialize FileIO!\n");
                exit(0);
            }
            m_shutdownCalled = false;
        }

        ~FileIO() {
            ShutDown();
        }

        void ShutDown() override {
            if (m_shutdownCalled) {
                return;
            }
            if (!m_mappingPath.empty()) Save(m_mappingPath);
            // TODO: Should we add a lock here?
            for (int i = 0; i < m_pBlockMapping.R(); i++) {
                if (At(i) != 0xffffffffffffffff) delete[]((AddressType*)At(i));
            }
            while (!m_buffer.empty()) {
                uintptr_t ptr;
                if (m_buffer.try_pop(ptr)) delete[]((AddressType*)ptr);
            }
            m_pBlockController.ShutDown();
            if (m_fileIoUseCache) {
                delete m_pShardedLRUCache;
            }
            m_shutdownCalled = true;
        }

        inline uintptr_t& At(SizeType key) {
            return *(m_pBlockMapping[key]);
        }

        ErrorCode Get(SizeType key, ByteArray& value) {
            auto get_begin_time = std::chrono::high_resolution_clock::now();
            if (m_fileIoUseLock) {
                m_rwMutex[hash(key)].lock_shared();
            }
            SizeType r;
            if (m_fileIoUseLock) {
                m_updateMutex.lock_shared();
                r = m_pBlockMapping.R();
                m_updateMutex.unlock_shared();
            }
            else {
                r = m_pBlockMapping.R();
            }
            if (key >= r) return ErrorCode::Fail;
            
            if (m_fileIoUseCache) {
                auto size = ((AddressType*)At(key))[0];
                std::uint8_t* outdata = new std::uint8_t[size];
                if (m_pShardedLRUCache->get(key, outdata)) {
                    value.Set(outdata, size, false);
                    if (m_fileIoUseLock) {
                        m_rwMutex[hash(key)].unlock_shared();
                    }
                    auto get_end_time = std::chrono::high_resolution_clock::now();
                    get_time_vec[id] += std::chrono::duration_cast<std::chrono::microseconds>(get_end_time - get_begin_time).count();
                    return ErrorCode::Success;
                }
                delete[] outdata;
            }

            // if (m_pBlockController.ReadBlocks((AddressType*)At(key), value)) {
            //     return ErrorCode::Success;
            // }
            auto begin_time = std::chrono::high_resolution_clock::now();
            auto result = m_pBlockController.ReadBlocks((AddressType*)At(key), value);
            auto end_time = std::chrono::high_resolution_clock::now();
            read_time_vec[id] += std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin_time).count();
            get_times_vec[id]++;
            if (m_fileIoUseCache) {
                m_pShardedLRUCache->put(key, value.Data(), value.Length());
            }
            if (m_fileIoUseLock) {
                m_rwMutex[hash(key)].unlock_shared();
            }
            auto get_end_time = std::chrono::high_resolution_clock::now();
            get_time_vec[id] += std::chrono::duration_cast<std::chrono::microseconds>(get_end_time - get_begin_time).count();
            return result ? ErrorCode::Success : ErrorCode::Fail;
        }

        ErrorCode Get(SizeType key, std::string* value) override {
            auto get_begin_time = std::chrono::high_resolution_clock::now();
            if (m_fileIoUseLock) {
                m_rwMutex[hash(key)].lock_shared();
            }
            SizeType r;
            if (m_fileIoUseLock) {
                m_updateMutex.lock_shared();
                r = m_pBlockMapping.R();
                m_updateMutex.unlock_shared();
            }
            else {
                r = m_pBlockMapping.R();
            }
            if (key >= r) return ErrorCode::Fail;

            if (m_fileIoUseCache) {
                auto size = ((AddressType*)At(key))[0];
                value->resize(size);
                if (m_pShardedLRUCache->get(key, value->data())) {
                    if (m_fileIoUseLock) {
                        m_rwMutex[hash(key)].unlock_shared();
                    }
                    auto get_end_time = std::chrono::high_resolution_clock::now();
                    get_time_vec[id] += std::chrono::duration_cast<std::chrono::microseconds>(get_end_time - get_begin_time).count();
                    return ErrorCode::Success;
                }
            }   
            
            // if (m_pBlockController.ReadBlocks((AddressType*)At(key), value)) {
            //     return ErrorCode::Success;
            // }
            auto begin_time = std::chrono::high_resolution_clock::now();
            auto result = m_pBlockController.ReadBlocks((AddressType*)At(key), value);
            auto end_time = std::chrono::high_resolution_clock::now();
            read_time_vec[id] += std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin_time).count();
            get_times_vec[id]++;
            if (m_fileIoUseLock) {
                m_rwMutex[hash(key)].unlock_shared();
            }
            auto get_end_time = std::chrono::high_resolution_clock::now();
            get_time_vec[id] += std::chrono::duration_cast<std::chrono::microseconds>(get_end_time - get_begin_time).count();
            return result ? ErrorCode::Success : ErrorCode::Fail;
        }

        ErrorCode Get(SizeType key, std::string* value, const std::chrono::microseconds &timeout) {
            auto get_begin_time = std::chrono::high_resolution_clock::now();
            if (m_fileIoUseLock) {
                m_rwMutex[hash(key)].lock_shared();
            }
            SizeType r;
            if (m_fileIoUseLock) {
                m_updateMutex.lock_shared();
                r = m_pBlockMapping.R();
                m_updateMutex.unlock_shared();
            }
            else {
                r = m_pBlockMapping.R();
            }
            if (key >= r) return ErrorCode::Fail;

            if (m_fileIoUseCache) {
                auto size = ((AddressType*)At(key))[0];
                value->resize(size);
                if (m_pShardedLRUCache->get(key, value->data())) {
                    if (m_fileIoUseLock) {
                        m_rwMutex[hash(key)].unlock_shared();
                    }
                    auto get_end_time = std::chrono::high_resolution_clock::now();
                    get_time_vec[id] += std::chrono::duration_cast<std::chrono::microseconds>(get_end_time - get_begin_time).count();
                    return ErrorCode::Success;
                }
            }
            
            
            // if (m_pBlockController.ReadBlocks((AddressType*)At(key), value)) {
            //     return ErrorCode::Success;
            // }
            auto begin_time = std::chrono::high_resolution_clock::now();
            auto result = m_pBlockController.ReadBlocks((AddressType*)At(key), value, timeout);
            auto end_time = std::chrono::high_resolution_clock::now();
            read_time_vec[id] += std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin_time).count();
            get_times_vec[id]++;
            if (m_fileIoUseLock) {
                m_rwMutex[hash(key)].unlock_shared();
            }
            auto get_end_time = std::chrono::high_resolution_clock::now();
            get_time_vec[id] += std::chrono::duration_cast<std::chrono::microseconds>(get_end_time - get_begin_time).count();
            return result ? ErrorCode::Success : ErrorCode::Fail;
        }

        ErrorCode Get(const std::string& key, std::string* value) override {
            return Get(std::stoi(key), value);
        }

        ErrorCode MultiGet(const std::vector<SizeType>& keys, std::vector<std::string>* values, const std::chrono::microseconds &timeout = std::chrono::microseconds::max()) {
            std::vector<AddressType*> blocks;
            std::set<int> lock_keys;
            if (m_fileIoUseLock) {
                for (SizeType key : keys) {
                    m_rwMutex[hash(key)].lock_shared();
                }
            }
            SizeType r;
            values->resize(keys.size());
            int i = 0;
            for (SizeType key : keys) {
                if (m_fileIoUseLock) {
                    m_updateMutex.lock_shared();
                    r = m_pBlockMapping.R();
                    m_updateMutex.unlock_shared();
                }
                else {
                    r = m_pBlockMapping.R();
                }
                if (key < r) {
                    if (m_fileIoUseCache) {
                        auto size = ((AddressType*)At(key))[0];
                        (*values)[i].resize(size);
                        if (m_pShardedLRUCache->get(key, (*values)[i].data())) {
                            blocks.push_back(nullptr);
                        }
                        else {
                            blocks.push_back((AddressType*)At(key));
                        }
                    } else {
                        blocks.push_back((AddressType*)At(key));
                    }
                    
                }
                else {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to read key:%d total key number:%d\n", key, r);
                }
                i++;
            }
            // if (m_pBlockController.ReadBlocks(blocks, values, timeout)) return ErrorCode::Success;
            auto result = m_pBlockController.ReadBlocks(blocks, values, timeout);
            if (m_fileIoUseLock) {
                for (SizeType key : keys) {
                    m_rwMutex[hash(key)].unlock_shared();
                }
            }
            return result ? ErrorCode::Success : ErrorCode::Fail;
        }

        ErrorCode MultiGet(const std::vector<std::string>& keys, std::vector<std::string>* values, const std::chrono::microseconds &timeout = std::chrono::microseconds::max()) override {
            std::vector<SizeType> int_keys;
            for (const auto& key : keys) {
                int_keys.push_back(std::stoi(key));
            }
            return MultiGet(int_keys, values, timeout);
        }

        ErrorCode Scan(const SizeType start_key, const int record_count, std::vector<ByteArray> &values, const std::chrono::microseconds &timeout = std::chrono::microseconds::max()) {
            std::vector<SizeType> keys;
            std::vector<AddressType*> blocks;
            SizeType curr_key = start_key;
            while(keys.size() < record_count && curr_key < m_pBlockMapping.R()) {
                if (m_fileIoUseLock) {
                    m_rwMutex[hash(curr_key)].lock_shared();
                }
                if (At(curr_key) == 0xffffffffffffffff) {
                    if (m_fileIoUseLock) {
                        m_rwMutex[hash(curr_key)].unlock_shared();
                    }
                    curr_key++;
                    continue;
                }
                keys.push_back(curr_key);
                blocks.push_back((AddressType*)At(curr_key));
                curr_key++;
            }
            auto result = m_pBlockController.ReadBlocks(blocks, values, timeout);
            if (m_fileIoUseLock) {
                for (auto key : keys) {
                    m_rwMutex[hash(key)].unlock_shared();
                }
            }
            return result ? ErrorCode::Success : ErrorCode::Fail;
        }

        ErrorCode Put(SizeType key, const std::string& value) override {
            int blocks = ((value.size() + PageSize - 1) >> PageSizeEx);
            if (blocks >= m_blockLimit) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to put key:%d value:%lld since value too long!\n", key, value.size());
                return ErrorCode::Fail;
            }
            // Calculate whether more mapping blocks are needed
            if (m_fileIoUseLock) {
                m_rwMutex[hash(key)].lock();
            }
            int delta;
            if (m_fileIoUseLock) {
                m_updateMutex.lock_shared();
                delta = key + 1 - m_pBlockMapping.R();
                m_updateMutex.unlock_shared();
            }
            else {
                delta = key + 1 - m_pBlockMapping.R();
            }
            if (delta > 0) {
                m_updateMutex.lock();
                delta = key + 1 - m_pBlockMapping.R();
                if (delta > 0) {
                    m_pBlockMapping.AddBatch(delta);
                }
                m_updateMutex.unlock();
            }

            if (m_fileIoUseCache) {
                m_pShardedLRUCache->put(key, (void *)(value.data()), value.size());
            }

            // If this key has not been assigned mapping blocks yet, allocate a batch.
            if (At(key) == 0xffffffffffffffff) {
                // If there are spare blocks in m_buffer, use them directly; otherwise, allocate a new batch.
                if (m_buffer.unsafe_size() > m_bufferLimit) {
                    uintptr_t tmpblocks;
                    while (!m_buffer.try_pop(tmpblocks));
                    At(key) = tmpblocks;
                }
                else {
                    At(key) = (uintptr_t)(new AddressType[m_blockLimit]);
                }
                // The 0th element of the block address list represents the data size; set it to -1.
                memset((AddressType*)At(key), -1, sizeof(AddressType) * m_blockLimit);
            }
            int64_t* postingSize = (int64_t*)At(key);
            // If postingSize is less than 0, it means the mapping block is newly allocated—directly
            if (*postingSize < 0) {
                m_pBlockController.GetBlocks(postingSize + 1, blocks);
                m_pBlockController.WriteBlocks(postingSize + 1, blocks, value);
                *postingSize = value.size();
            }
            else {
                uintptr_t tmpblocks;
                // Take a batch of mapping blocks from the buffer, and return a batch back later.
                while (!m_buffer.try_pop(tmpblocks));
                // Acquire a new batch of disk blocks and write data directly.
                // To ensure the effectiveness of the checkpoint, new blocks must be allocated for writing here.
                m_pBlockController.GetBlocks((AddressType*)tmpblocks + 1, blocks);
                m_pBlockController.WriteBlocks((AddressType*)tmpblocks + 1, blocks, value);
                *((int64_t*)tmpblocks) = value.size();

                // Release the original blocks
                m_pBlockController.ReleaseBlocks(postingSize + 1, (*postingSize + PageSize -1) >> PageSizeEx);
                At(key) = tmpblocks;
                m_buffer.push((uintptr_t)postingSize);
            }
            if (m_fileIoUseLock) {
                m_rwMutex[hash(key)].unlock();
            }
            return ErrorCode::Success;
        }

        ErrorCode Put(SizeType key, const ByteArray& value) {
            int blocks = ((value.Length() + PageSize - 1) >> PageSizeEx);
            if (blocks >= m_blockLimit) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to put key:%d value:%lld since value too long!\n", key, value.Length());
                return ErrorCode::Fail;
            }
            // Calculate whether more mapping blocks are needed
            if (m_fileIoUseLock) {
                m_rwMutex[hash(key)].lock();
            }
            int delta;
            if (m_fileIoUseLock) {
                m_updateMutex.lock_shared();
                delta = key + 1 - m_pBlockMapping.R();
                m_updateMutex.unlock_shared();
            }
            else {
                delta = key + 1 - m_pBlockMapping.R();
            }
            if (delta > 0) {
                m_updateMutex.lock();
                delta = key + 1 - m_pBlockMapping.R();
                if (delta > 0) {
                    m_pBlockMapping.AddBatch(delta);
                }
                m_updateMutex.unlock();
            }

            if (m_fileIoUseCache) {
                m_pShardedLRUCache->put(key, (void *)(value.Data()), value.Length());
            }

            // If this key hasn't been assigned any mapping blocks yet, allocate a batch for it.
            if (At(key) == 0xffffffffffffffff) {
                if (m_buffer.unsafe_size() > m_bufferLimit) {
                    uintptr_t tmpblocks;
                    while (!m_buffer.try_pop(tmpblocks));
                    At(key) = tmpblocks;
                }
                else {
                    At(key) = (uintptr_t)(new AddressType[m_blockLimit]);
                }
                memset((AddressType*)At(key), -1, sizeof(AddressType) * m_blockLimit);
            }
            int64_t* postingSize = (int64_t*)At(key);
            if (*postingSize < 0) {
                m_pBlockController.GetBlocks(postingSize + 1, blocks);
                m_pBlockController.WriteBlocks(postingSize + 1, blocks, value);
                *postingSize = value.Length();
            }
            else {
                uintptr_t tmpblocks;
                while (!m_buffer.try_pop(tmpblocks));
                m_pBlockController.GetBlocks((AddressType*)tmpblocks + 1, blocks);
                m_pBlockController.WriteBlocks((AddressType*)tmpblocks + 1, blocks, value);
                *((int64_t*)tmpblocks) = value.Length();

                // Release the original blocks
                m_pBlockController.ReleaseBlocks(postingSize + 1, (*postingSize + PageSize -1) >> PageSizeEx);
                At(key) = tmpblocks;
                m_buffer.push((uintptr_t)postingSize);
            }
            if (m_fileIoUseLock) {
                m_rwMutex[hash(key)].unlock();
            }
            return ErrorCode::Success;
        }

        ErrorCode Put(const std::string &key, const std::string& value) override {
            return Put(std::stoi(key), value);
        }

        ErrorCode Merge(SizeType key, const std::string& value) {
            if (m_fileIoUseLock) {
                m_rwMutex[hash(key)].lock();
            }
            SizeType r;
            if (m_fileIoUseLock) {
                m_updateMutex.lock_shared();
                r = m_pBlockMapping.R();
                m_updateMutex.unlock_shared();
            }
            else {
                r = m_pBlockMapping.R();
            }
            if (key >= r) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Key range error: key: %d, mapping size: %d\n", key, r);
                if (m_fileIoUseLock) {
                    m_rwMutex[hash(key)].unlock();
                }
                return ErrorCode::Fail;
            }
            
            int64_t* postingSize = (int64_t*)At(key);

            if (m_fileIoUseCache) {
                m_pShardedLRUCache->merge(key, (void *)(value.data()), value.size());
            }

            auto newSize = *postingSize + value.size();
            int newblocks = ((newSize + PageSize - 1) >> PageSizeEx);

            if (newblocks >= m_blockLimit) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "[Merge] Key %d failed: new size %lld bytes requires %d blocks (limit: %d)\n",
                    key, static_cast<long long>(newSize), newblocks, m_blockLimit);

                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "[Merge] Original size: %lld bytes, Merge size: %lld bytes\n",
                    static_cast<long long>(*postingSize), static_cast<long long>(value.size()));

                if (m_fileIoUseLock) {
                    m_rwMutex[hash(key)].unlock();
                }
                return ErrorCode::Fail;
            }

            auto sizeInPage = (*postingSize) % PageSize;    // Actual size of the last block
            int oldblocks = (*postingSize >> PageSizeEx);
            int allocblocks = newblocks - oldblocks;
            // If the last block is not full, we need to read it first, then append the new data, and write it back.
            if (sizeInPage != 0) {
                std::string newValue;
                AddressType readreq[] = { sizeInPage, *(postingSize + 1 + oldblocks) };
                m_pBlockController.ReadBlocks(readreq, &newValue);
                newValue += value;

                uintptr_t tmpblocks;
                while (!m_buffer.try_pop(tmpblocks));
                memcpy((AddressType*)tmpblocks, postingSize, sizeof(AddressType) * (oldblocks + 1));
                m_pBlockController.GetBlocks((AddressType*)tmpblocks + 1 + oldblocks, allocblocks);
                m_pBlockController.WriteBlocks((AddressType*)tmpblocks + 1 + oldblocks, allocblocks, newValue);
                *((int64_t*)tmpblocks) = newSize;

                // This is also to ensure checkpoint correctness, so we release the partially used block and allocate a new one.
                m_pBlockController.ReleaseBlocks(postingSize + 1 + oldblocks, 1);
                At(key) = tmpblocks;
                m_buffer.push((uintptr_t)postingSize);
            }
            else {  // Otherwise, directly allocate a new batch of blocks to append after the current ones.
                m_pBlockController.GetBlocks(postingSize + 1 + oldblocks, allocblocks);
                m_pBlockController.WriteBlocks(postingSize + 1 + oldblocks, allocblocks, value);
                *postingSize = newSize;
            }
            if (m_fileIoUseLock) {
                m_rwMutex[hash(key)].unlock();
            }
            return ErrorCode::Success;
        }

        ErrorCode Merge(const std::string &key, const std::string& value) {
            return Merge(std::stoi(key), value);
        }

        ErrorCode Delete(SizeType key) override {
            if (m_fileIoUseLock) {
                m_rwMutex[hash(key)].lock();
            }
            SizeType r;
            if (m_fileIoUseLock) {
                m_updateMutex.lock_shared();
                r = m_pBlockMapping.R();
                m_updateMutex.unlock_shared();
            }
            else {
                r = m_pBlockMapping.R();
            }
            if (key >= r) return ErrorCode::Fail;

            if (m_fileIoUseCache) {
                m_pShardedLRUCache->del(key);
            }

            int64_t* postingSize = (int64_t*)At(key);
            if (*postingSize < 0) {
                if (m_fileIoUseLock) {
                    m_rwMutex[hash(key)].unlock();
                }
                return ErrorCode::Fail;
            }

            int blocks = ((*postingSize + PageSize - 1) >> PageSizeEx);
            m_pBlockController.ReleaseBlocks(postingSize + 1, blocks);
            m_buffer.push((uintptr_t)postingSize);
            At(key) = 0xffffffffffffffff;
            if (m_fileIoUseLock) {
                m_rwMutex[hash(key)].unlock();
            }
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
            int remainGB = (long long)remainBlocks << PageSizeEx >> 30;

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Remain %d blocks, totally %d GB\n", remainBlocks, remainGB);

            uint64_t get_times = 0;
            uint64_t get_time = 0;
            uint64_t read_time = 0;
            for (int i = 0; i < get_time_vec.size(); i++) {
                get_times += get_times_vec[i];
                get_time += get_time_vec[i];
                read_time += read_time_vec[i];
            }
            double average_read_time = (double)read_time / get_times;
            double average_get_time = (double)get_time / get_times;
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Get times: %llu, get time: %llu us, read time: %llu us\n", get_times, get_time, read_time);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Average read time: %lf us, average get time: %lf us\n", average_read_time, average_get_time);
            if (m_fileIoUseCache) {
                auto cache_stat = m_pShardedLRUCache->get_stat();
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Cache queries: %lld, Cache hits: %lld, Hit rates: %lf\n", cache_stat.first, cache_stat.second, cache_stat.second == 0 ? 0 : (double)cache_stat.second / cache_stat.first);
            }
            m_pBlockController.IOStatistics();
        }

        ErrorCode Load(std::string path, SizeType blockSize, SizeType capacity) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load mapping From %s\n", path.c_str());
            auto ptr = f_createIO();
            if (ptr == nullptr || !ptr->Initialize(path.c_str(), std::ios::binary | std::ios::in)) return ErrorCode::FailedOpenFile;

            SizeType CR, mycols;
            IOBINARY(ptr, ReadBinary, sizeof(SizeType), (char*)&CR);
            IOBINARY(ptr, ReadBinary, sizeof(SizeType), (char*)&mycols);
            if (mycols > m_blockLimit) m_blockLimit = mycols;

            m_pBlockMapping.Initialize(CR, 1, blockSize, capacity);
            for (int i = 0; i < CR; i++) {
                At(i) = (uintptr_t)(new AddressType[m_blockLimit]);
                IOBINARY(ptr, ReadBinary, sizeof(AddressType) * mycols, (char*)At(i));
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load mapping (%d,%d) Finish!\n", CR, mycols);
            return ErrorCode::Success;
        }
        
        ErrorCode Save(std::string path) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Save mapping To %s\n", path.c_str());
            auto ptr = f_createIO();
            if (ptr == nullptr || !ptr->Initialize(path.c_str(), std::ios::binary | std::ios::out)) return ErrorCode::FailedCreateFile;

            SizeType CR = m_pBlockMapping.R();
            IOBINARY(ptr, WriteBinary, sizeof(SizeType), (char*)&CR);
            IOBINARY(ptr, WriteBinary, sizeof(SizeType), (char*)&m_blockLimit);
            std::vector<AddressType> empty(m_blockLimit, 0xffffffffffffffff);
            for (int i = 0; i < CR; i++) {
                if (At(i) == 0xffffffffffffffff) {
                    IOBINARY(ptr, WriteBinary, sizeof(AddressType) * m_blockLimit, (char*)(empty.data()));
                }
                else {
                    int64_t* postingSize = (int64_t*)At(i);
                    IOBINARY(ptr, WriteBinary, sizeof(AddressType) * m_blockLimit, (char*)postingSize);
                }
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Save mapping (%d,%d) Finish!\n", CR, m_blockLimit);
            return ErrorCode::Success;
        }

        bool Initialize(bool debug = false) override {
            if (debug) SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Initialize FileIO for new threads\n");
            m_freeIdMutex.lock();
	        if (id < 0) {
                if (m_freeId.empty()) {
                    id = m_maxId++;
                }
                else {
                    id = m_freeId.front();
                    m_freeId.pop();
                }
	        } else {
                if (id > m_maxId) m_maxId = id + 1;
	        }
            while(read_time_vec.size() < m_maxId) read_time_vec.push_back(0);
            while(get_time_vec.size() < m_maxId) get_time_vec.push_back(0);
            while(get_times_vec.size() < m_maxId) get_times_vec.push_back(0);
            m_freeIdMutex.unlock();
            return m_pBlockController.Initialize(m_mappingPath, 64);
        }

        bool ExitBlockController(bool debug = false) override { 
            if (debug) SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Exit FileIO for thread\n");
            m_freeIdMutex.lock();
            m_freeId.push(id);
            m_freeIdMutex.unlock();
            return m_pBlockController.ShutDown(); 
        }

        ErrorCode Checkpoint(std::string prefix) override {
            std::string filename = prefix + FolderSep + m_mappingPath.substr(m_mappingPath.find_last_of(FolderSep) + 1);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO: saving block mapping\n");
            Save(filename);
            return m_pBlockController.Checkpoint(filename + "_postings");
        }

    private:
        static constexpr const char* kFileIoUseLock = "SPFRESH_FILE_IO_USE_LOCK";
        static constexpr bool kFileIoDefaultUseLock = false;
        static constexpr const char* kFileIoLockSize = "SPFRESH_FILE_IO_LOCK_SIZE";
        static constexpr int kFileIoDefaultLockSize = 1024;
        static constexpr const char* kFileIoUseCache = "SPFRESH_FILE_IO_USE_CACHE";
        static constexpr bool kFileIoDefaultUseCache = false;
        static constexpr const char* kFileIoCacheSize = "SPFRESH_FILE_IO_CACHE_SIZE";
        static constexpr int kSsdFileIoDefaultCacheSize = 8192 << 10;
        static constexpr const char* kFileIoCacheShards = "SPFRESH_FILE_IO_CACHE_SHARDS";
        static constexpr int kSsdFileIoDefaultCacheShards = 4;

        thread_local static int id;
        int m_maxId = 0;
        std::queue<int> m_freeId;
        std::mutex m_freeIdMutex;
        std::vector<uint64_t> read_time_vec;
        std::vector<uint64_t> get_time_vec;
        std::vector<uint64_t> get_times_vec;

        bool m_fileIoUseLock = kFileIoDefaultUseLock;
        int m_fileIoLockSize = kFileIoDefaultLockSize;
        bool m_fileIoUseCache = kFileIoDefaultUseCache;
        std::string m_mappingPath;
        SizeType m_blockLimit;
        COMMON::Dataset<uintptr_t> m_pBlockMapping;
        SizeType m_bufferLimit;
        tbb::concurrent_queue<uintptr_t> m_buffer;

        std::shared_ptr<Helper::ThreadPool> m_compactionThreadPool;
        BlockController m_pBlockController;
        ShardedLRUCache *m_pShardedLRUCache;

        bool m_shutdownCalled;
        std::shared_mutex m_updateMutex;
        std::vector<std::shared_mutex> m_rwMutex;

        inline int hash(int key) {
            return key % m_fileIoLockSize;
        }
    };
}
#endif
