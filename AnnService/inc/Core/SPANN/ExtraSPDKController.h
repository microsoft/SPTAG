// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_EXTRASPDKCONTROLLER_H_
#define _SPTAG_SPANN_EXTRASPDKCONTROLLER_H_

#include "inc/Helper/KeyValueIO.h"
#include "inc/Helper/DiskIO.h"
#include "inc/Helper/ConcurrentSet.h"
#include "inc/Core/Common/Dataset.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Core/SPANN/Options.h"
#include "inc/Helper/ThreadPool.h"
#include <cstdlib>
#include <memory>
#include <atomic>
#include <mutex>
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_hash_map.h>

extern "C" {
#include "spdk/env.h"
#include "spdk/event.h"
#include "spdk/log.h"
#include "spdk/thread.h"
#include "spdk/bdev.h"
}

namespace SPTAG::SPANN
{
    typedef std::int64_t AddressType;
    class SPDKIO : public Helper::KeyValueIO
    {
        class BlockController {
        private:
            static constexpr const char* kUseMemImplEnv = "SPFRESH_SPDK_USE_MEM_IMPL";
            static constexpr const char* kUseSsdImplEnv = "SPFRESH_SPDK_USE_SSD_IMPL";
            static constexpr const char* kSpdkConfEnv = "SPFRESH_SPDK_CONF";
            static constexpr const char* kSpdkBdevNameEnv = "SPFRESH_SPDK_BDEV";

            Helper::Concurrent::ConcurrentQueue<AddressType> m_blockAddresses;
            Helper::Concurrent::ConcurrentQueue<AddressType> m_blockAddresses_reserve;

	        std::string m_filePath;
            bool m_useSsdImpl = false;
            const char* m_ssdSpdkBdevName = nullptr;
            pthread_t m_ssdSpdkTid;
            volatile bool m_ssdSpdkThreadStartFailed = false;
            volatile bool m_ssdSpdkThreadReady = false;
            volatile bool m_ssdSpdkThreadExiting = false;
            struct spdk_bdev *m_ssdSpdkBdev = nullptr;
            struct spdk_bdev_desc *m_ssdSpdkBdevDesc = nullptr;
            struct spdk_io_channel *m_ssdSpdkBdevIoChannel = nullptr;

            Helper::Concurrent::ConcurrentQueue<Helper::AsyncReadRequest*> m_submittedSubIoRequests;

            static int m_ssdInflight;

            bool m_useMemImpl = false;
            static std::unique_ptr<char[]> m_memBuffer;

            float m_growthThreshold = 0.05;
            AddressType m_growthBlocks = 0;
            AddressType m_maxBlocks = 0;
            int m_batchSize = 64;

            static int m_ioCompleteCount;
            int m_preIOCompleteCount = 0;
            std::chrono::time_point<std::chrono::high_resolution_clock> m_preTime = std::chrono::high_resolution_clock::now();

            std::mutex m_expandLock;
            std::atomic<AddressType> m_totalAllocatedBlocks = 0;

            bool ExpandFile(AddressType blocksToAdd);
            
            bool NeedsExpansion();

            static void* InitializeSpdk(void* args);

            static void SpdkStart(void* args);

            static void SpdkIoLoop(void *arg);

            static void SpdkBdevEventCallback(enum spdk_bdev_event_type type, struct spdk_bdev *bdev, void *event_ctx);

            static void SpdkBdevIoCallback(struct spdk_bdev_io *bdev_io, bool success, void *cb_arg);

            static void SpdkStop(void* args);
        public:
            bool Initialize(SPANN::Options& p_opt);

            // get p_size blocks from front, and fill in p_data array
            bool GetBlocks(AddressType* p_data, int p_size);

            // release p_size blocks, put them at the end of the queue
            bool ReleaseBlocks(AddressType* p_data, int p_size);

            // read a posting list. p_data[0] is the total data size, 
            // p_data[1], p_data[2], ..., p_data[((p_data[0] + PageSize - 1) >> PageSizeEx)] are the addresses of the blocks
            // concat all the block contents together into p_value string.
            bool ReadBlocks(AddressType* p_data, std::string* p_value, const std::chrono::microseconds& timeout, std::vector<Helper::AsyncReadRequest>* reqs);

            // parallel read a list of posting lists.
            bool ReadBlocks(std::vector<AddressType*>& p_data, std::vector<std::string>* p_values, const std::chrono::microseconds& timeout, std::vector<Helper::AsyncReadRequest>* reqs);
            bool ReadBlocks(std::vector<AddressType*>& p_data, std::vector<Helper::PageBuffer<std::uint8_t>>& p_values, const std::chrono::microseconds& timeout, std::vector<Helper::AsyncReadRequest>* reqs);

            // write p_value into p_size blocks start from p_data
            bool WriteBlocks(AddressType* p_data, int p_size, const std::string& p_value, const std::chrono::microseconds& timeout, std::vector<Helper::AsyncReadRequest>* reqs);

            bool IOStatistics();

            bool ShutDown();

            int RemainBlocks() {
                return m_blockAddresses.unsafe_size();
            }

            ErrorCode Checkpoint(std::string prefix) {
                std::string filename = prefix + "_blockpool";
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPDK: saving block pool\n");
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Reload reserved blocks!\n");
                AddressType currBlockAddress = 0;
                while (m_blockAddresses_reserve.try_pop(currBlockAddress))
                {
                    m_blockAddresses.push(currBlockAddress);
                }
                AddressType blocks = RemainBlocks();
                AddressType totalBlocks = m_totalAllocatedBlocks.load();

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPDKIO::BlockController::Checkpoint - Total allocated blocks: %llu\n", static_cast<unsigned long long>(totalBlocks));
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPDKIO::BlockController::Checkpoint - Remaining free blocks: %llu\n", blocks);
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPDKIO::BlockController::Checkpoint - Saving to file: %s\n", filename.c_str());

                auto ptr = f_createIO();
                if (ptr == nullptr || !ptr->Initialize(filename.c_str(), std::ios::binary | std::ios::out))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SPDKIO::BlockController::Checkpoint - Failed to open file: %s\n", filename.c_str());
                    return ErrorCode::FailedCreateFile;
                }
                IOBINARY(ptr, WriteBinary, sizeof(AddressType), (char*)&blocks);
                IOBINARY(ptr, WriteBinary, sizeof(AddressType), reinterpret_cast<char*>(&totalBlocks));
                for (auto it = m_blockAddresses.unsafe_begin(); it != m_blockAddresses.unsafe_end(); it++) {
                    IOBINARY(ptr, WriteBinary, sizeof(AddressType), (char*)&(*it));
                }
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Save Finish!\n");
                return ErrorCode::Success;
            }

	    ErrorCode LoadBlockPool(std::string prefix, AddressType startNumBlocks, bool allowinit) {
	        std::string blockfile = prefix + "_blockpool";
            if (allowinit && !fileexists(blockfile.c_str())) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPDKIO::BlockController::LoadBlockPool: initializing fresh pool (no existing file found: %s)\n", blockfile.c_str());
                for(AddressType i = 0; i < startNumBlocks; i++) {
                    m_blockAddresses.push(i);
                }
                m_totalAllocatedBlocks.store(startNumBlocks);

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPDKIO::BlockController::LoadBlockPool: initialized with %llu blocks (%.2f GB)\n",
                    static_cast<unsigned long long>(startNumBlocks), static_cast<float>(startNumBlocks >> (30 - PageSizeEx)));
            } else {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPDKIO::BlockController::Load blockpool from %s\n", blockfile.c_str());
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
                    "SPDKIO::BlockController::LoadBlockPool: reading %llu free blocks into pool (%.2f GB), total allocated: %llu blocks\n",
                    static_cast<unsigned long long>(blocks),
                    static_cast<float>(blocks >> (30 - PageSizeEx)),
                    static_cast<unsigned long long>(totalAllocated));

                AddressType currBlockAddress = 0;
                for (int i = 0; i < blocks; i++) {
                    IOBINARY(ptr, ReadBinary, sizeof(AddressType), (char*)&(currBlockAddress));
                    m_blockAddresses.push(currBlockAddress);
	            }
                m_totalAllocatedBlocks.store(totalAllocated);

                SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                    "SPDKIO::BlockController::LoadBlockPool: block pool initialized. Available: %llu, Total allocated: %llu\n",
                    static_cast<unsigned long long>(blocks),
                    static_cast<unsigned long long>(totalAllocated));
    	    }
		    return ErrorCode::Success;
	    }
    };

    public:
        SPDKIO(SPANN::Options& p_opt)
            //const char* filePath, SizeType blockSize, SizeType capacity, SizeType postingBlocks, SizeType bufferSize = 1024, int batchSize = 64, bool recovery = false, int compactionThreads = 1)
        {
            m_opt = &p_opt;
            m_mappingPath = p_opt.m_indexDirectory + FolderSep + p_opt.m_ssdMappingFile;
            m_blockLimit = max(p_opt.m_postingPageLimit, p_opt.m_searchPostingPageLimit) + p_opt.m_bufferLength + 1;
            m_bufferLimit = 1024;
            m_shutdownCalled = true;

            if (p_opt.m_recovery) {
                std::string recoverpath = p_opt.m_persistentBufferPath + FolderSep + p_opt.m_ssdMappingFile;
                if (fileexists(recoverpath.c_str())) {
                    m_pBlockMapping.Load(recoverpath, 1024 * 1024, MaxSize);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load block mapping successfully from %s!\n", recoverpath.c_str());
                }
                else {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot recover block mapping from %s!\n", recoverpath.c_str());
                    return;
                }
            }
            else {
                if (fileexists(m_mappingPath.c_str())) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load blockmapping successfully from %s\n", m_mappingPath.c_str());
                    Load(m_mappingPath, 1024 * 1024, MaxSize);
                }
                else {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Initialize block mapping successfully!\n");
                    m_pBlockMapping.Initialize(0, 1, 1024 * 1024, MaxSize);
                }
            }
            for (int i = 0; i < m_bufferLimit; i++) {
                m_buffer.push((uintptr_t)(new AddressType[m_blockLimit]));
            }
            m_compactionThreadPool = std::make_shared<Helper::ThreadPool>();
            m_compactionThreadPool->init(1);

            if (!m_pBlockController.Initialize(p_opt)) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to Initialize SPDK!\n");
                return;
            }

            m_shutdownCalled = false;
        }

        ~SPDKIO() {
            ShutDown();
        }

        void ShutDown() override {
            if (m_shutdownCalled) {
                return;
            }
            if (!m_mappingPath.empty()) Save(m_mappingPath);
            for (int i = 0; i < m_pBlockMapping.R(); i++) {
                if (At(i) != 0xffffffffffffffff) delete[]((AddressType*)At(i));
            }
            while (!m_buffer.empty()) {
                uintptr_t ptr = 0xffffffffffffffff;
                if (m_buffer.try_pop(ptr)) delete[]((AddressType*)ptr);
            }
            m_pBlockController.ShutDown();
            m_shutdownCalled = true;
        }

        bool Available() override
        {
            return !m_shutdownCalled;
        }

        inline uintptr_t& At(SizeType key) {
            return *(m_pBlockMapping[key]);
        }

        ErrorCode Get(const SizeType key, std::string* value, const std::chrono::microseconds& timeout, std::vector<Helper::AsyncReadRequest>* reqs) override {
            if (key >= m_pBlockMapping.R()) return ErrorCode::Fail;

            if (m_pBlockController.ReadBlocks((AddressType*)At(key), value, timeout, reqs)) return ErrorCode::Success;
            return ErrorCode::Fail;
        }

        ErrorCode MultiGet(const std::vector<SizeType>& keys, std::vector<std::string>* values, 
            const std::chrono::microseconds &timeout, std::vector<Helper::AsyncReadRequest>* reqs) override {
            std::vector<AddressType*> blocks;
            for (SizeType key : keys) {
                if (key < m_pBlockMapping.R()) blocks.push_back((AddressType*)At(key));
                else {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to read key:%d total key number:%d\n", key, m_pBlockMapping.R());
                }
            }
            if (m_pBlockController.ReadBlocks(blocks, values, timeout, reqs)) return ErrorCode::Success;
            return ErrorCode::Fail; 
        }

        ErrorCode MultiGet(const std::vector<SizeType>& keys, std::vector<Helper::PageBuffer<std::uint8_t>>& values, const std::chrono::microseconds& timeout, std::vector<Helper::AsyncReadRequest>* reqs) override {
            std::vector<AddressType*> blocks;
            for (SizeType key : keys) {
                if (key < m_pBlockMapping.R()) blocks.push_back((AddressType*)At(key));
                else {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to read key:%d total key number:%d\n", key, m_pBlockMapping.R());
                }
            }
            if (m_pBlockController.ReadBlocks(blocks, values, timeout, reqs)) return ErrorCode::Success;
            return ErrorCode::Fail;
        }

        ErrorCode Put(const SizeType key, const std::string& value, const std::chrono::microseconds& timeout, std::vector<Helper::AsyncReadRequest>* reqs) override {
            int blocks = ((value.size() + PageSize - 1) >> PageSizeEx);
            if (blocks >= m_blockLimit) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failt to put key:%d value:%lld since value too long!\n", key, value.size());
                return ErrorCode::Fail;
            }
            int delta = key + 1 - m_pBlockMapping.R();
            if (delta > 0) {
                {
                    std::lock_guard<std::mutex> lock(m_updateMutex);
                    delta = key + 1 - m_pBlockMapping.R();
                    if (delta > 0) {
                        m_pBlockMapping.AddBatch(delta);
                    }
                }
            }
            if (At(key) == 0xffffffffffffffff) {
                if (m_buffer.unsafe_size() > m_bufferLimit) {
                    uintptr_t tmpblocks = 0xffffffffffffffff;
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
                m_pBlockController.WriteBlocks(postingSize + 1, blocks, value, timeout, reqs);
                *postingSize = value.size();
            }
            else {
                uintptr_t tmpblocks = 0xffffffffffffffff;
                while (!m_buffer.try_pop(tmpblocks));
                m_pBlockController.GetBlocks((AddressType*)tmpblocks + 1, blocks);
                m_pBlockController.WriteBlocks((AddressType*)tmpblocks + 1, blocks, value, timeout, reqs);
                *((int64_t*)tmpblocks) = value.size();

                m_pBlockController.ReleaseBlocks(postingSize + 1, (*postingSize + PageSize -1) >> PageSizeEx);
                while (InterlockedCompareExchange(&At(key), tmpblocks, (uintptr_t)postingSize) != (uintptr_t)postingSize) {
                    postingSize = (int64_t*)At(key);
                }
                m_buffer.push((uintptr_t)postingSize);
            }
            return ErrorCode::Success;
        }

        ErrorCode Merge(SizeType key, const std::string &value, const std::chrono::microseconds &timeout,
                        std::vector<Helper::AsyncReadRequest> *reqs,
                        std::function<bool(const void *val, const int size)> checksum) override
        {
            if (key >= m_pBlockMapping.R()) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Key range error: key: %d, mapping size: %d\n", key, m_pBlockMapping.R());
                return ErrorCode::Fail;
            }

            int64_t* postingSize = (int64_t*)At(key);
            auto newSize = *postingSize + value.size();
            int newblocks = ((newSize + PageSize - 1) >> PageSizeEx);
            if (newblocks >= m_blockLimit) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "[Merge] Key %d failed: new size %lld bytes requires %d blocks (limit: %d)\n",
                    key, static_cast<long long>(newSize), newblocks, m_blockLimit);

                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "[Merge] Original size: %lld bytes, Merge size: %lld bytes\n",
                    static_cast<long long>(*postingSize), static_cast<long long>(value.size()));
                return ErrorCode::Fail;
            }

            auto sizeInPage = (*postingSize) % PageSize;
            int oldblocks = (*postingSize >> PageSizeEx);
            int allocblocks = newblocks - oldblocks;
            if (sizeInPage != 0) {
                std::string newValue;
                AddressType readreq[] = { sizeInPage, *(postingSize + 1 + oldblocks) };
                m_pBlockController.ReadBlocks(readreq, &newValue, timeout, reqs);
                newValue += value;

                uintptr_t tmpblocks = 0xffffffffffffffff;
                while (!m_buffer.try_pop(tmpblocks));
                memcpy((AddressType*)tmpblocks, postingSize, sizeof(AddressType) * (oldblocks + 1));
                m_pBlockController.GetBlocks((AddressType*)tmpblocks + 1 + oldblocks, allocblocks);
                m_pBlockController.WriteBlocks((AddressType*)tmpblocks + 1 + oldblocks, allocblocks, newValue, timeout, reqs);
                *((int64_t*)tmpblocks) = newSize;

                m_pBlockController.ReleaseBlocks(postingSize + 1 + oldblocks, 1);
                while (InterlockedCompareExchange(&At(key), tmpblocks, (uintptr_t)postingSize) != (uintptr_t)postingSize) {
                    postingSize = (int64_t*)At(key);
                }
                m_buffer.push((uintptr_t)postingSize);
            }
            else {
                m_pBlockController.GetBlocks(postingSize + 1 + oldblocks, allocblocks);
                m_pBlockController.WriteBlocks(postingSize + 1 + oldblocks, allocblocks, value, timeout, reqs);
                *postingSize = newSize;
            }
            return ErrorCode::Success;
        }

        ErrorCode Delete(SizeType key) override {
            if (key >= m_pBlockMapping.R()) return ErrorCode::Fail;
            int64_t* postingSize = (int64_t*)At(key);
            if (*postingSize < 0) return ErrorCode::Fail;

            int blocks = ((*postingSize + PageSize - 1) >> PageSizeEx);
            m_pBlockController.ReleaseBlocks(postingSize + 1, blocks);
            m_buffer.push((uintptr_t)postingSize);
            At(key) = 0xffffffffffffffff;
            return ErrorCode::Success;
        }

        void ForceCompaction() {
            Save(m_mappingPath);
        }

        void GetStat() {
            int remainBlocks = m_pBlockController.RemainBlocks();
            int remainGB = (long long)remainBlocks << PageSizeEx >> 30;
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Remain %d blocks, totally %d GB\n", remainBlocks, remainGB);
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

        ErrorCode Checkpoint(std::string prefix) override {
            std::string filename = prefix + FolderSep + m_mappingPath.substr(m_mappingPath.find_last_of(FolderSep) + 1);
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPDK: saving block mapping to %s\n", filename.c_str());
            Save(filename);
            return m_pBlockController.Checkpoint(filename + "_postings");
        }

    private:
        SPANN::Options* m_opt;
        std::string m_mappingPath;
        SizeType m_blockLimit;
        COMMON::Dataset<uintptr_t> m_pBlockMapping;
        SizeType m_bufferLimit;
        Helper::Concurrent::ConcurrentQueue<uintptr_t> m_buffer;
        
        std::shared_ptr<Helper::ThreadPool> m_compactionThreadPool;
        BlockController m_pBlockController;

        bool m_shutdownCalled;
        std::mutex m_updateMutex;
    };
}
#endif // _SPTAG_SPANN_EXTRASPDKCONTROLLER_H_
