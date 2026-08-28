// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _MSC_VER
#include "inc/Core/SPANN/ExtraSPDKController.h"
#include "inc/Helper/AsyncFileReader.h"

namespace SPTAG::SPANN
{
int SPDKIO::BlockController::m_ssdInflight = 0;
int SPDKIO::BlockController::m_ioCompleteCount = 0;
std::unique_ptr<char[]> SPDKIO::BlockController::m_memBuffer;

void SPDKIO::BlockController::SpdkBdevEventCallback(enum spdk_bdev_event_type type, struct spdk_bdev *bdev,
                                                    void *event_ctx)
{
    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SpdkBdevEventCallback: supported bdev event type %d\n", type);
}

void SPDKIO::BlockController::SpdkBdevIoCallback(struct spdk_bdev_io *bdev_io, bool success, void *cb_arg)
{
    Helper::AsyncReadRequest *currSubIo = (Helper::AsyncReadRequest *)cb_arg;
    if (success)
    {
        m_ioCompleteCount++;
        spdk_bdev_free_io(bdev_io);
        reinterpret_cast<Helper::RequestQueue *>(currSubIo->m_extension)->push(currSubIo);
        m_ssdInflight--;
        SpdkIoLoop(currSubIo->m_ctrl);
    }
    else
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SpdkBdevIoCallback: I/O failed %p\n", currSubIo);
        spdk_app_stop(-1);
    }
}

void SPDKIO::BlockController::SpdkStop(void *arg)
{
    SPDKIO::BlockController *ctrl = (SPDKIO::BlockController *)arg;
    // Close I/O channel and bdev
    spdk_put_io_channel(ctrl->m_ssdSpdkBdevIoChannel);
    spdk_bdev_close(ctrl->m_ssdSpdkBdevDesc);
    fprintf(stdout, "SPDKIO::BlockController::SpdkStop: finalized\n");
}

void SPDKIO::BlockController::SpdkIoLoop(void *arg)
{
    SPDKIO::BlockController *ctrl = (SPDKIO::BlockController *)arg;
    int rc = 0;
    Helper::AsyncReadRequest *currSubIo = nullptr;
    while (!ctrl->m_ssdSpdkThreadExiting)
    {
        if (ctrl->m_submittedSubIoRequests.try_pop(currSubIo))
        {
            if ((currSubIo->myiocb).aio_lio_opcode == IOCB_CMD_PREAD)
            {
                rc = spdk_bdev_read(ctrl->m_ssdSpdkBdevDesc, ctrl->m_ssdSpdkBdevIoChannel, currSubIo->m_buffer,
                                    currSubIo->m_offset, PageSize, SpdkBdevIoCallback, currSubIo);
            }
            else
            {
                rc = spdk_bdev_write(ctrl->m_ssdSpdkBdevDesc, ctrl->m_ssdSpdkBdevIoChannel, currSubIo->m_buffer,
                                     currSubIo->m_offset, PageSize, SpdkBdevIoCallback, currSubIo);
            }
            if (rc && rc != -ENOMEM)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                             "SPDKIO::BlockController::SpdkStart %s failed: %d, shutting down, offset: %ld\n",
                             ((currSubIo->myiocb).aio_lio_opcode == IOCB_CMD_PREAD) ? "spdk_bdev_read"
                                                                                    : "spdk_bdev_write",
                             rc, currSubIo->m_offset);
                spdk_app_stop(-1);
                break;
            }
            else
            {
                m_ssdInflight++;
            }
        }
        else if (m_ssdInflight)
        {
            break;
        }
    }
    if (ctrl->m_ssdSpdkThreadExiting)
    {
        SpdkStop(ctrl);
    }
}

void SPDKIO::BlockController::SpdkStart(void *arg)
{
    SPDKIO::BlockController *ctrl = (SPDKIO::BlockController *)arg;

    fprintf(stdout, "SPDKIO::BlockController::SpdkStart: using bdev %s\n", ctrl->m_ssdSpdkBdevName);

    int rc = 0;
    ctrl->m_ssdSpdkBdev = NULL;
    ctrl->m_ssdSpdkBdevDesc = NULL;

    // Open bdev
    rc = spdk_bdev_open_ext(ctrl->m_ssdSpdkBdevName, true, SpdkBdevEventCallback, NULL, &ctrl->m_ssdSpdkBdevDesc);
    if (rc)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SPDKIO::BlockController::SpdkStart: spdk_bdev_open_ext failed, %d\n",
                     rc);
        ctrl->m_ssdSpdkThreadStartFailed = true;
        spdk_app_stop(-1);
        return;
    }
    ctrl->m_ssdSpdkBdev = spdk_bdev_desc_get_bdev(ctrl->m_ssdSpdkBdevDesc);

    // Open I/O channel
    ctrl->m_ssdSpdkBdevIoChannel = spdk_bdev_get_io_channel(ctrl->m_ssdSpdkBdevDesc);
    if (ctrl->m_ssdSpdkBdevIoChannel == NULL)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                     "SPDKIO::BlockController::SpdkStart: spdk_bdev_get_io_channel failed\n");
        spdk_bdev_close(ctrl->m_ssdSpdkBdevDesc);
        ctrl->m_ssdSpdkThreadStartFailed = true;
        spdk_app_stop(-1);
        return;
    }

    ctrl->m_ssdSpdkThreadReady = true;
    m_ssdInflight = 0;

    SpdkIoLoop(ctrl);
}

void *SPDKIO::BlockController::InitializeSpdk(void *arg)
{
    SPDKIO::BlockController *ctrl = (SPDKIO::BlockController *)arg;

    struct spdk_app_opts opts;
    spdk_app_opts_init(&opts, sizeof(opts));
    opts.name = "spfresh";
    const char *spdkConf = getenv(kSpdkConfEnv);
    opts.json_config_file = spdkConf ? spdkConf : "";
    const char *spdkBdevName = getenv(kSpdkBdevNameEnv);
    ctrl->m_ssdSpdkBdevName = spdkBdevName ? spdkBdevName : "";

    int rc;
    rc = spdk_app_start(&opts, &SPTAG::SPANN::SPDKIO::BlockController::SpdkStart, arg);
    if (rc)
    {
        ctrl->m_ssdSpdkThreadStartFailed = true;
    }
    else
    {
        spdk_app_fini();
    }
    pthread_exit(NULL);
}

bool SPDKIO::BlockController::Initialize(SPANN::Options &p_opt)
{
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPDKIO::BlockController::Initialize(%s, %d)\n",
                 p_opt.m_ssdMappingFile.c_str(), p_opt.m_spdkBatchSize);

    const char *useMemImplEnvStr = getenv(kUseMemImplEnv);
    m_useMemImpl = useMemImplEnvStr && !strcmp(useMemImplEnvStr, "1");
    const char *useSsdImplEnvStr = getenv(kUseSsdImplEnv);
    m_useSsdImpl = useSsdImplEnvStr && !strcmp(useSsdImplEnvStr, "1");
    if (m_useMemImpl)
    {
        std::uint64_t memoryBlocks = ((std::uint64_t)p_opt.m_startFileSize) << (30 - PageSizeEx);
        if (m_memBuffer == nullptr)
        {
            m_memBuffer.reset(new char[memoryBlocks >> PageSizeEx]);
        }
        for (AddressType i = 0; i < memoryBlocks; i++)
        {
            m_blockAddresses.push(i);
        }
        return true;
    }
    else if (m_useSsdImpl)
    {
        m_growthThreshold = p_opt.m_growThreshold;
        m_growthBlocks = ((std::uint64_t)p_opt.m_growthFileSize) << (30 - PageSizeEx);
        m_maxBlocks = ((std::uint64_t)p_opt.m_maxFileSize) << (30 - PageSizeEx);
        m_batchSize = p_opt.m_spdkBatchSize;
        m_filePath = p_opt.m_indexDirectory + FolderSep + p_opt.m_ssdMappingFile + "_postings";
        pthread_create(&m_ssdSpdkTid, NULL, &InitializeSpdk, this);
        while (!m_ssdSpdkThreadReady && !m_ssdSpdkThreadStartFailed)
            ;
        if (m_ssdSpdkThreadStartFailed)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SPDKIO::BlockController::Initialize failed\n");
            return false;
        }
        std::string blockpoolPath =
            (p_opt.m_recovery) ? p_opt.m_persistentBufferPath + FolderSep + p_opt.m_ssdMappingFile + "_postings"
                               : m_filePath;
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPDKIO::BlockController:Loading block pool from file:%s\n",
                     blockpoolPath.c_str());
        ErrorCode ret = LoadBlockPool(blockpoolPath, ((std::uint64_t)p_opt.m_startFileSize) << (30 - PageSizeEx),
                                      !p_opt.m_recovery);
        if (ErrorCode::Success != ret)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPDKIO::BlockController:Loading block pool failed!\n");
            return false;
        }
        return true;
    }
    else
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SPDKIO::BlockController::Initialize failed\n");
        return false;
    }
}

bool SPDKIO::BlockController::NeedsExpansion()
{
    // TODO: Replace RemainBlocks() which internally uses tbb::concurrent_queue::unsafe_size().
    // unsafe_size() is *not thread-safe* and may yield inconsistent results under concurrent access.
    size_t available = RemainBlocks();
    size_t total = m_totalAllocatedBlocks.load();
    float ratio = static_cast<float>(available) / static_cast<float>(total);

    return (ratio < m_growthThreshold);
}

bool SPDKIO::BlockController::ExpandFile(AddressType blocksToAdd)
{
    AddressType currentTotal = m_totalAllocatedBlocks.load();

    // Cap the growth so we do not exceed the limit
    AddressType allowedBlocks = min(blocksToAdd, m_maxBlocks - currentTotal);
    if (allowedBlocks <= 0)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                     "[ExpandFile] File is already at max capacity (%llu blocks = %.2f GB). Expansion aborted.\n",
                     static_cast<unsigned long long>(m_maxBlocks),
                     static_cast<float>(m_maxBlocks >> (30 - PageSizeEx)));
    }

    for (AddressType i = 0; i < allowedBlocks; ++i)
    {
        m_blockAddresses.push(currentTotal + i);
    }

    AddressType newTotal = currentTotal + allowedBlocks;
    m_totalAllocatedBlocks.store(newTotal);

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                 "[ExpandFile] Expanded file from %llu to %llu blocks (added %llu blocks = %.2f GB)\n",
                 static_cast<unsigned long long>(currentTotal), static_cast<unsigned long long>(newTotal),
                 static_cast<unsigned long long>(allowedBlocks),
                 static_cast<float>(allowedBlocks >> (30 - PageSizeEx)));

    if (allowedBlocks < blocksToAdd)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                     "[ExpandFile] Requested %llu blocks, but only %llu were allowed due to cap (%llu max)\n",
                     static_cast<unsigned long long>(blocksToAdd), static_cast<unsigned long long>(allowedBlocks),
                     static_cast<unsigned long long>(m_maxBlocks));
    }

    return true;
}

// get p_size blocks from front, and fill in p_data array
bool SPDKIO::BlockController::GetBlocks(AddressType *p_data, int p_size)
{

    if (NeedsExpansion())
    {
        std::unique_lock<std::mutex> lock(
            m_expandLock); // ensure only one thread tries expandingAdd commentMore actions
        if (NeedsExpansion())
        { // recheck inside lock
            AddressType growBy = static_cast<AddressType>(m_growthBlocks);
            AddressType total = m_totalAllocatedBlocks.load();
            if (total + growBy <= m_maxBlocks)
            {
                if (!ExpandFile(growBy))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SPDKIO::BlockController::GetBlocks: expansion failed\n");
                    return false;
                }
            }
            else
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                             "SPDKIO::BlockController::GetBlocks: cannot expand beyond cap (%llu)\n",
                             static_cast<unsigned long long>(m_maxBlocks));
            }
        }
    }

    AddressType currBlockAddress = 0;
    if (m_useMemImpl || m_useSsdImpl)
    {
        for (int i = 0; i < p_size; i++)
        {
            while (!m_blockAddresses.try_pop(currBlockAddress))
                ;
            p_data[i] = currBlockAddress;
        }
        return true;
    }
    else
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SPDKIO::BlockController::GetBlocks failed\n");
        return false;
    }
}

// release p_size blocks, put them at the end of the queue
bool SPDKIO::BlockController::ReleaseBlocks(AddressType *p_data, int p_size)
{
    if (m_useMemImpl || m_useSsdImpl)
    {
        for (int i = 0; i < p_size; i++)
        {
            // m_blockAddresses.push(p_data[i]);
            m_blockAddresses_reserve.push(p_data[i]);
        }
        return true;
    }
    else
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SPDKIO::BlockController::ReleaseBlocks failed\n");
        return false;
    }
}

// read a posting list. p_data[0] is the total data size,
// p_data[1], p_data[2], ..., p_data[((p_data[0] + PageSize - 1) >> PageSizeEx)] are the addresses of the blocks
// concat all the block contents together into p_value string.
bool SPDKIO::BlockController::ReadBlocks(AddressType *p_data, std::string *p_value,
                                         const std::chrono::microseconds &timeout,
                                         std::vector<Helper::AsyncReadRequest> *reqs)
{
    if (m_useMemImpl)
    {
        p_value->resize(p_data[0]);
        AddressType currOffset = 0;
        AddressType dataIdx = 1;
        while (currOffset < p_data[0])
        {
            AddressType readSize = (p_data[0] - currOffset) < PageSize ? (p_data[0] - currOffset) : PageSize;
            memcpy((char *)p_value->data() + currOffset, m_memBuffer.get() + p_data[dataIdx] * PageSize, readSize);
            currOffset += PageSize;
            dataIdx++;
        }
        return true;
    }
    else if (m_useSsdImpl)
    {

        // Clear timeout I/Os
        Helper::AsyncReadRequest *currSubIo;
        Helper::RequestQueue *completeQueue = (Helper::RequestQueue *)(reqs->at(0).m_extension);
        while (completeQueue->try_pop(currSubIo))
            ;

        size_t postingSize = (size_t)p_data[0];
        p_value->resize(postingSize);

        AddressType currOffset = 0;
        AddressType dataIdx = 1;
        int in_flight = 0;
        auto t1 = std::chrono::high_resolution_clock::now();
        // Submit all I/Os
        while (currOffset < postingSize || in_flight)
        {
            auto t2 = std::chrono::high_resolution_clock::now();
            if (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1) > timeout)
            {
                return false;
            }
            // Try submit
            if (currOffset < postingSize)
            {
                Helper::AsyncReadRequest &curr = reqs->at(dataIdx - 1);
                curr.m_readSize = (postingSize - currOffset) < PageSize ? (postingSize - currOffset) : PageSize;
                curr.m_offset = p_data[dataIdx] * PageSize;
                curr.m_payload = (char *)p_value->data() + currOffset;
                curr.m_ctrl = this;
                curr.myiocb.aio_lio_opcode = IOCB_CMD_PREAD;
                m_submittedSubIoRequests.push(&curr);
                currOffset += PageSize;
                dataIdx++;
                in_flight++;
            }
            // Try complete
            if (in_flight && completeQueue->try_pop(currSubIo))
            {
                memcpy((char *)(currSubIo->m_payload), currSubIo->m_buffer, currSubIo->m_readSize);
                in_flight--;
            }
            else
            {
                std::this_thread::sleep_for(std::chrono::microseconds(1)); // Sleep for a short time
            }
        }

        if (in_flight > 0)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Read timeout happens! Skip %d requests...\n", in_flight);
        }
        return true;
    }
    else
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SPDKIO::BlockController::ReadBlocks single failed\n");
        return false;
    }
}

// parallel read a list of posting lists.
bool SPDKIO::BlockController::ReadBlocks(std::vector<AddressType *> &p_data, std::vector<std::string> *p_values,
                                         const std::chrono::microseconds &timeout,
                                         std::vector<Helper::AsyncReadRequest> *reqs)
{
    if (m_useMemImpl)
    {
        p_values->resize(p_data.size());
        for (size_t i = 0; i < p_data.size(); i++)
        {
            ReadBlocks(p_data[i], &((*p_values)[i]), timeout, reqs);
        }
        return true;
    }
    else if (m_useSsdImpl)
    {
        // Temporarily disable timeout

        // Convert request format to SubIoRequests
        auto t1 = std::chrono::high_resolution_clock::now();
        p_values->resize(p_data.size());

        std::uint32_t reqcount = 0;
        for (size_t i = 0; i < p_data.size(); i++)
        {
            AddressType *p_data_i = p_data[i];
            std::string *p_value = &((*p_values)[i]);

            if (p_data_i == nullptr)
            {
                continue;
            }

            std::size_t postingSize = (std::size_t)p_data_i[0];
            p_value->resize(postingSize);
            AddressType currOffset = 0;
            AddressType dataIdx = 1;
            while (currOffset < postingSize)
            {
                Helper::AsyncReadRequest &curr = reqs->at(reqcount);
                curr.m_readSize = (postingSize - currOffset) < PageSize ? (postingSize - currOffset) : PageSize;
                curr.m_offset = p_data_i[dataIdx] * PageSize;
                curr.m_payload = (char *)p_value->data() + currOffset;
                curr.m_ctrl = this;
                curr.myiocb.aio_lio_opcode = IOCB_CMD_PREAD;
                currOffset += PageSize;
                dataIdx++;
                reqcount++;
                if (reqcount >= reqs->size())
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SPDKIO::BlockController::ReadBlocks:  %u >= %u\n",
                                 reqcount, reqs->size());
                    return false;
                }
            }
        }

        // Clear timeout I/Os
        Helper::AsyncReadRequest *currSubIo;
        Helper::RequestQueue *completeQueue = (Helper::RequestQueue *)(reqs->at(0).m_extension);
        while (completeQueue->try_pop(currSubIo))
            ;

        for (int currSubIoStartId = 0; currSubIoStartId < reqcount; currSubIoStartId += m_batchSize)
        {
            int currSubIoEndId =
                (currSubIoStartId + m_batchSize) > reqcount ? reqcount : currSubIoStartId + m_batchSize;
            int currSubIoIdx = currSubIoStartId;

            int in_flight = 0;
            while (currSubIoIdx < currSubIoEndId || in_flight)
            {
                auto t2 = std::chrono::high_resolution_clock::now();
                if (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1) > timeout)
                {
                    break;
                }

                // Try submit
                if (currSubIoIdx < currSubIoEndId)
                {
                    Helper::AsyncReadRequest &curr = reqs->at(currSubIoIdx);
                    m_submittedSubIoRequests.push(&curr);
                    in_flight++;
                    currSubIoIdx++;
                }

                // Try complete
                if (in_flight && completeQueue->try_pop(currSubIo))
                {
                    memcpy((char *)(currSubIo->m_payload), currSubIo->m_buffer, currSubIo->m_readSize);
                    in_flight--;
                }
                else
                {
                    std::this_thread::sleep_for(std::chrono::microseconds(1)); // 休眠一个非常短的时间
                }
            }

            if (in_flight > 0)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Read timeout happens! Skip %d requests...\n",
                             (reqcount - currSubIoEndId + in_flight));
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            if (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1) > timeout)
            {
                break;
            }
        }
        return true;
    }
    else
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SPDKIO::BlockController::ReadBlocks batch failed\n");
        return false;
    }
}

bool SPDKIO::BlockController::ReadBlocks(std::vector<AddressType *> &p_data,
                                         std::vector<Helper::PageBuffer<std::uint8_t>> &p_values,
                                         const std::chrono::microseconds &timeout,
                                         std::vector<Helper::AsyncReadRequest> *reqs)
{
    if (m_useMemImpl)
    {
        for (size_t i = 0; i < p_data.size(); i++)
        {
            std::uint8_t *p_value = p_values[i].GetBuffer();
            auto &p_data_i = p_data[i];
            AddressType currOffset = 0;
            AddressType dataIdx = 1;
            while (currOffset < p_data_i[0])
            {
                AddressType readSize = (p_data_i[0] - currOffset) < PageSize ? (p_data_i[0] - currOffset) : PageSize;
                memcpy((char *)p_value + currOffset, m_memBuffer.get() + p_data_i[dataIdx] * PageSize, readSize);
                currOffset += PageSize;
                dataIdx++;
            }
        }
        return true;
    }
    else if (m_useSsdImpl)
    {
        // Temporarily disable timeout

        // Convert request format to SubIoRequests
        auto t1 = std::chrono::high_resolution_clock::now();
        std::uint32_t reqcount = 0;
        std::uint32_t emptycount = 0;
        for (size_t i = 0; i < p_data.size(); i++)
        {
            AddressType *p_data_i = p_data[i];
            int numPages = (p_values[i].GetPageSize() >> PageSizeEx);

            if (p_data_i == nullptr)
            {
                p_values[i].SetAvailableSize(0);
                for (std::uint32_t r = 0; r < numPages; r++)
                {
                    Helper::AsyncReadRequest &curr = reqs->at(reqcount);
                    curr.m_readSize = 0;
                    reqcount++;
                    emptycount++;
                }
                continue;
            }

            std::size_t postingSize = (std::size_t)p_data_i[0];
            p_values[i].SetAvailableSize(postingSize);
            AddressType currOffset = 0;
            AddressType dataIdx = 1;
            while (currOffset < postingSize)
            {
                if (dataIdx > numPages)
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "SPDKIO::BlockController::ReadBlocks:  block (%d) >= buffer page size (%d)\n",
                                 dataIdx - 1, numPages);
                    return false;
                }

                if (reqcount >= reqs->size())
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "SPDKIO::BlockController::ReadBlocks:  req (%u) >= req array size (%u)\n", reqcount,
                                 reqs->size());
                    return false;
                }

                Helper::AsyncReadRequest &curr = reqs->at(reqcount);
                curr.m_readSize = (postingSize - currOffset) < PageSize ? (postingSize - currOffset) : PageSize;
                curr.m_offset = p_data_i[dataIdx] * PageSize;
                curr.m_ctrl = this;
                curr.myiocb.aio_lio_opcode = IOCB_CMD_PREAD;
                currOffset += PageSize;
                dataIdx++;
                reqcount++;
            }

            while (dataIdx - 1 < numPages)
            {
                if (reqcount >= reqs->size())
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                                 "SPDKIO::BlockController::ReadBlocks:  req (%u) >= req array size (%u)\n", reqcount,
                                 reqs->size());
                    return false;
                }

                Helper::AsyncReadRequest &curr = reqs->at(reqcount);
                curr.m_readSize = 0;
                dataIdx++;
                reqcount++;
                emptycount++;
            }
        }

        // Clear timeout I/Os
        Helper::AsyncReadRequest *currSubIo;
        Helper::RequestQueue *completeQueue = (Helper::RequestQueue *)(reqs->at(0).m_extension);
        while (completeQueue->try_pop(currSubIo))
            ;

        int realcount = reqcount - emptycount;
        int reqIdx = 0;
        for (int currSubIoStartId = 0; currSubIoStartId < realcount; currSubIoStartId += m_batchSize)
        {
            int currSubIoEndId =
                (currSubIoStartId + m_batchSize) > realcount ? realcount : currSubIoStartId + m_batchSize;
            int currSubIoIdx = currSubIoStartId;

            int in_flight = 0;
            while (currSubIoIdx < currSubIoEndId || in_flight)
            {
                auto t2 = std::chrono::high_resolution_clock::now();
                if (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1) > timeout)
                {
                    break;
                }

                // Try submit
                if (currSubIoIdx < currSubIoEndId)
                {
                    while (reqIdx < reqcount && (reqs->at(reqIdx)).m_readSize == 0)
                        reqIdx++;
                    Helper::AsyncReadRequest &curr = reqs->at(reqIdx);
                    m_submittedSubIoRequests.push(&curr);
                    in_flight++;
                    currSubIoIdx++;
                    reqIdx++;
                }

                // Try complete
                if (in_flight && completeQueue->try_pop(currSubIo))
                {
                    in_flight--;
                }
                else
                {
                    std::this_thread::sleep_for(std::chrono::microseconds(1)); // 休眠一个非常短的时间
                }
            }

            if (in_flight > 0)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Read timeout happens! Skip %d requests...\n",
                             (realcount - currSubIoEndId + in_flight));
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            if (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1) > timeout)
            {
                break;
            }
        }
        return true;
    }
    else
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SPDKIO::BlockController::ReadBlocks batch failed\n");
        return false;
    }
}

// write p_value into p_size blocks start from p_data
bool SPDKIO::BlockController::WriteBlocks(AddressType *p_data, int p_size, const std::string &p_value,
                                          const std::chrono::microseconds &timeout,
                                          std::vector<Helper::AsyncReadRequest> *reqs)
{
    if (m_useMemImpl)
    {
        for (int i = 0; i < p_size; i++)
        {
            memcpy(m_memBuffer.get() + p_data[i] * PageSize, p_value.data() + i * PageSize, PageSize);
        }
        return true;
    }
    else if (m_useSsdImpl)
    {
        Helper::AsyncReadRequest *currSubIo;
        Helper::RequestQueue *completeQueue = (Helper::RequestQueue *)(reqs->at(0).m_extension);

        int totalSize = p_value.size();
        AddressType currBlockIdx = 0;
        int inflight = 0;
        // Submit all I/Os
        while (currBlockIdx < p_size || inflight)
        {
            // Try submit
            if (currBlockIdx < p_size)
            {
                Helper::AsyncReadRequest &curr = reqs->at(currBlockIdx);
                int currOffset = currBlockIdx * PageSize;
                curr.m_readSize = (totalSize - currOffset) < PageSize ? (totalSize - currOffset) : PageSize;
                curr.m_offset = p_data[currBlockIdx] * PageSize;
                memcpy(curr.m_buffer, const_cast<char *>(p_value.data()) + currOffset, curr.m_readSize);
                curr.m_ctrl = this;
                curr.myiocb.aio_lio_opcode = IOCB_CMD_PWRITE;
                m_submittedSubIoRequests.push(&curr);
                currBlockIdx++;
                inflight++;
            }

            // Try complete
            if (inflight && completeQueue->try_pop(currSubIo))
            {
                // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPDKIO::BlockController::WriteBlocks: completed %d inflight
                // requests\n", inflight);
                inflight--;
            }
            else
            {
                std::this_thread::sleep_for(std::chrono::microseconds(1)); // 休眠一个非常短的时间
            }
        }
        return true;
    }
    else
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SPDKIO::BlockController::ReadBlocks single failed\n");
        return false;
    }
}

bool SPDKIO::BlockController::IOStatistics()
{
    int currIOCount = m_ioCompleteCount;
    int diffIOCount = currIOCount - m_preIOCompleteCount;
    m_preIOCompleteCount = currIOCount;

    auto currTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(currTime - m_preTime);
    m_preTime = currTime;

    double currIOPS = (double)diffIOCount * 1000 / duration.count();
    double currBandWidth = (double)diffIOCount * PageSize / 1024 * 1000 / 1024 * 1000 / duration.count();

    std::cout << "IOPS: " << currIOPS << "k Bandwidth: " << currBandWidth << "MB/s" << std::endl;

    return true;
}

bool SPDKIO::BlockController::ShutDown()
{
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPDKIO:BlockController:ShutDown!\n");
    if (m_useMemImpl)
    {
        while (!m_blockAddresses.empty())
        {
            AddressType currBlockAddress;
            m_blockAddresses.try_pop(currBlockAddress);
        }
        return true;
    }
    else if (m_useSsdImpl)
    {
        m_ssdSpdkThreadExiting = true;
        spdk_app_start_shutdown();
        pthread_join(m_ssdSpdkTid, NULL);
        Checkpoint(m_filePath);
        while (!m_blockAddresses.empty())
        {
            AddressType currBlockAddress;
            m_blockAddresses.try_pop(currBlockAddress);
        }
        return true;
    }
    else
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SPDKIO::BlockController::ShutDown failed\n");
        return false;
    }
}

} // namespace SPTAG::SPANN
#endif
