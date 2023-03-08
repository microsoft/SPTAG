// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_EXTRASPDKCONTROLLER_H_
#define _SPTAG_SPANN_EXTRASPDKCONTROLLER_H_

#include "inc/Helper/KeyValueIO.h"
#include "inc/Core/Common/Dataset.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Helper/ThreadPool.h"
#include <cstdlib>
#include <memory>
#include <atomic>
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_hash_map.h>

namespace SPTAG::SPANN
{
    typedef std::int64_t AddressType;
    class SPDKIO : public Helper::KeyValueIO
    {
        class BlockController {
        private:
            static constexpr const char* kUseMemImplEnv = "SPFRESH_SPDK_USE_MEM_IMPL";
            static constexpr AddressType kMemImplMaxNumBlocks = (1ULL << 30) >> PageSizeEx;
            tbb::concurrent_queue<AddressType> m_blockAddresses;
            bool m_useMemImpl = false;
            static char* m_memBuffer;
        public:
            bool Initialize() {
                const char* kUseMemImplEnvStr = getenv(kUseMemImplEnv);
                m_useMemImpl = kUseMemImplEnvStr && !strcmp(kUseMemImplEnvStr, "1");
                if (m_useMemImpl) {
                    if (m_memBuffer == nullptr) {
                        m_memBuffer = new char[kMemImplMaxNumBlocks * PageSize];
                    }
                    for (AddressType i = 0; i < kMemImplMaxNumBlocks; i++) {
                        m_blockAddresses.push(i);
                    }
                    return true;
                } else {
                    fprintf(stderr, "SPDKIO::BlockController::Initialize failed\n");
                    return false;
                }
            }

            // get p_size blocks from front, and fill in p_data array
            bool GetBlocks(AddressType* p_data, int p_size) {
                AddressType currBlockAddress = 0;
                if (m_useMemImpl) {
                    for (int i = 0; i < p_size; i++) {
                        while (!m_blockAddresses.try_pop(currBlockAddress));
                        p_data[i] = currBlockAddress;
                    }
                    return true;
                } else {
                    fprintf(stderr, "SPDKIO::BlockController::GetBlocks failed\n");
                    return false;
                }
            }

            // release p_size blocks, put them at the end of the queue
            bool ReleaseBlocks(AddressType* p_data, int p_size) {
                if (m_useMemImpl) {
                    for (int i = 0; i < p_size; i++) {
                        m_blockAddresses.push(p_data[i]);
                    }
                    return true;
                } else {
                    fprintf(stderr, "SPDKIO::BlockController::ReleaseBlocks failed\n");
                    return false;
                }
            }

            // read a posting list. p_data[0] is the total data size, 
            // p_data[1], p_data[2], ..., p_data[((p_data[0] + PageSize - 1) >> PageSizeEx)] are the addresses of the blocks
            // concat all the block contents together into p_value string.
            bool ReadBlocks(AddressType* p_data, std::string* p_value) {
                if (m_useMemImpl) {
                    p_value->resize(p_data[0]);
                    AddressType currOffset = 0;
                    AddressType dataIdx = 1;
                    while (currOffset < p_data[0]) {
                        AddressType readSize = (p_data[0] - currOffset) < PageSize ? (p_data[0] - currOffset) : PageSize;
                        memcpy(p_value->data() + currOffset, m_memBuffer + p_data[dataIdx] * PageSize, readSize);
                        currOffset += PageSize;
                        dataIdx++;
                    }
                    return true;
                } else {
                    fprintf(stderr, "SPDKIO::BlockController::ReadBlocks single failed\n");
                    return false;
                }
            }

            // parallel read a list of posting lists.
            bool ReadBlocks(std::vector<AddressType*>& p_data, std::vector<std::string>* p_values) {
                if (m_useMemImpl) {
                    p_values->resize(p_data.size());
                    for (size_t i = 0; i < p_data.size(); i++) {
                        ReadBlocks(p_data[i], &((*p_values)[i]));
                    }
                    return true;
                } else {
                    fprintf(stderr, "SPDKIO::BlockController::ReadBlocks batch failed\n");
                    return false;
                }
            }

            // write p_value into p_size blocks start from p_data
            bool WriteBlocks(AddressType* p_data, int p_size, const std::string& p_value) {
                if (m_useMemImpl) {
                    for (int i = 0; i < p_size; i++) {
                        memcpy(m_memBuffer + p_data[i] * PageSize, p_value.data() + i * PageSize, PageSize);
                    }
                    return true;
                } else {
                    fprintf(stderr, "SPDKIO::BlockController::ReadBlocks single failed\n");
                    return false;
                }
            }

            bool ShutDown() {
                if (m_useMemImpl) {
                    while (!m_blockAddresses.empty()) {
                        AddressType currBlockAddress;
                        m_blockAddresses.try_pop(currBlockAddress);
                    }
                    return true;
                } else {
                    fprintf(stderr, "SPDKIO::BlockController::ShutDown failed\n");
                    return false;
                }
            }
        };

        class CompactionJob : public Helper::ThreadPool::Job
        {
        private:
            SPDKIO* m_spdkIO;

        public:
            CompactionJob(SPDKIO* spdkIO): m_spdkIO(spdkIO) {}

            ~CompactionJob() {}

            inline void exec(IAbortOperation* p_abort) override {
                m_spdkIO->ForceCompaction();
            }
        };

    public:
        SPDKIO(const char* filePath, SizeType blockSize, SizeType capacity, SizeType postingBlocks, SizeType bufferSize = 1024, int compactionThreads = 1)
        {
            m_mappingPath = std::string(filePath);
            m_blockLimit = postingBlocks + 1;
            m_bufferLimit = bufferSize;
            if (fileexists(m_mappingPath.c_str())) {
                Load(m_mappingPath, blockSize, capacity);
            }
            else {
                m_pBlockMapping.Initialize(0, 1, blockSize, capacity);
            }
            for (int i = 0; i < bufferSize; i++) {
                m_buffer.push((uintptr_t)(new AddressType[m_blockLimit]));
            }
            m_compactionThreadPool = std::make_shared<Helper::ThreadPool>();
            m_compactionThreadPool->init(compactionThreads);
            m_pBlockController.Initialize();
            m_shutdownCalled = false;
        }

        ~SPDKIO() {
            ShutDown();
        }

        void ShutDown() override {
            if (m_shutdownCalled) {
                return;
            }
            Save(m_mappingPath);
            for (int i = 0; i < m_pBlockMapping.R(); i++) {
                if (At(i) != 0xffffffffffffffff) delete[]((AddressType*)At(i));
            }
            while (!m_buffer.empty()) {
                uintptr_t ptr;
                if (m_buffer.try_pop(ptr)) delete[]((AddressType*)ptr);
            }
            m_pBlockController.ShutDown();
            m_shutdownCalled = true;
        }

        inline uintptr_t& At(SizeType key) {
            return *(m_pBlockMapping[key]);
        }

        ErrorCode Get(SizeType key, std::string* value) override {
            if (key >= m_pBlockMapping.R()) return ErrorCode::Fail;

            if (m_pBlockController.ReadBlocks((AddressType*)At(key), value)) return ErrorCode::Success;
            return ErrorCode::Fail;
        }

        ErrorCode MultiGet(const std::vector<SizeType>& keys, std::vector<std::string>* values) {
            std::vector<AddressType*> blocks;
            for (SizeType key : keys) {
                if (key < m_pBlockMapping.R()) blocks.push_back((AddressType*)At(key));
                else {
                    LOG(Helper::LogLevel::LL_Error, "Fail to read key:%d total key number:%d\n", key, m_pBlockMapping.R());
                }
            }
            if (m_pBlockController.ReadBlocks(blocks, values)) return ErrorCode::Success;
            return ErrorCode::Fail;
        }

        ErrorCode Put(SizeType key, const std::string& value) override {
            int blocks = ((value.size() + PageSize - 1) >> PageSizeEx);
            if (blocks >= m_blockLimit) {
                LOG(Helper::LogLevel::LL_Error, "Failt to put key:%d value:%lz since value too long!\n", key, value.size());
                return ErrorCode::Fail;
            }
            int delta = key + 1 - m_pBlockMapping.R();
            if (delta > 0) {
                m_pBlockMapping.AddBatch(delta);
            }
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
                *postingSize = value.size();
            }
            else {
                uintptr_t tmpblocks;
                while (!m_buffer.try_pop(tmpblocks));
                m_pBlockController.GetBlocks((AddressType*)tmpblocks + 1, blocks);
                m_pBlockController.WriteBlocks((AddressType*)tmpblocks + 1, blocks, value);
                *((int64_t*)tmpblocks) = value.size();

                m_pBlockController.ReleaseBlocks(postingSize + 1, (*postingSize + PageSize -1) >> PageSizeEx);
                while (InterlockedCompareExchange(&At(key), tmpblocks, (uintptr_t)postingSize) != (uintptr_t)postingSize) {
                    postingSize = (int64_t*)At(key);
                }
                m_buffer.push((uintptr_t)postingSize);
            }
            return ErrorCode::Success;
        }

        ErrorCode Merge(SizeType key, const std::string& value) {
            if (key >= m_pBlockMapping.R()) return ErrorCode::Fail;

            int64_t* postingSize = (int64_t*)At(key);
            auto newSize = *postingSize + value.size();
            int newblocks = ((newSize + PageSize - 1) >> PageSizeEx);
            if (newblocks >= m_blockLimit) {
                LOG(Helper::LogLevel::LL_Error, "Failt to merge key:%d value:%lld since value too long!\n", key, newSize);
                return ErrorCode::Fail;
            }

            auto sizeInPage = (*postingSize) % PageSize;
            int oldblocks = (*postingSize >> PageSizeEx);
            int allocblocks = newblocks - oldblocks;
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

                m_pBlockController.ReleaseBlocks(postingSize + 1 + oldblocks, 1);
                while (InterlockedCompareExchange(&At(key), tmpblocks, (uintptr_t)postingSize) != (uintptr_t)postingSize) {
                    postingSize = (int64_t*)At(key);
                }
                m_buffer.push((uintptr_t)postingSize);
            }
            else {
                m_pBlockController.GetBlocks(postingSize + 1 + oldblocks, allocblocks);
                m_pBlockController.WriteBlocks(postingSize + 1 + oldblocks, allocblocks, value);
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

        }

        ErrorCode Load(std::string path, SizeType blockSize, SizeType capacity) {
            LOG(Helper::LogLevel::LL_Info, "Load mapping From %s\n", path.c_str());
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
            LOG(Helper::LogLevel::LL_Info, "Load mapping (%d,%d) Finish!\n", CR, mycols);
            return ErrorCode::Success;
        }
        
        ErrorCode Save(std::string path) {
            LOG(Helper::LogLevel::LL_Info, "Save mapping To %s\n", path.c_str());
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
            LOG(Helper::LogLevel::LL_Info, "Save mapping (%d,%d) Finish!\n", CR, m_blockLimit);
            return ErrorCode::Success;
        }

    private:
        std::string m_mappingPath;
        SizeType m_blockLimit;
        COMMON::Dataset<uintptr_t> m_pBlockMapping;
        SizeType m_bufferLimit;
        tbb::concurrent_queue<uintptr_t> m_buffer;
        
        //tbb::concurrent_hash_map<SizeType, std::string> *m_pCurrentCache, *m_pNextCache;
        std::shared_ptr<Helper::ThreadPool> m_compactionThreadPool;
        BlockController m_pBlockController;

        bool m_shutdownCalled;
    };
}
#endif // _SPTAG_SPANN_EXTRASPDKCONTROLLER_H_
