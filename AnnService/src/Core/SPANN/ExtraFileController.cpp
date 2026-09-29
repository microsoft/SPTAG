#include "inc/Core/SPANN/ExtraFileController.h"

namespace SPTAG::SPANN
{
extern std::function<std::shared_ptr<Helper::DiskIO>(void)> f_createAsyncIO;

bool FileIO::BlockController::Initialize(SPANN::Options &p_opt)
{
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO::BlockController::Initialize(%s, %d)\n",
                 p_opt.m_ssdMappingFile.c_str(), p_opt.m_spdkBatchSize);

    m_growthThreshold = p_opt.m_growThreshold;
    m_growthBlocks = ((std::uint64_t)p_opt.m_growthFileSize) << (30 - PageSizeEx);
    m_maxBlocks = ((std::uint64_t)p_opt.m_maxFileSize) << (30 - PageSizeEx);
    m_batchSize = p_opt.m_spdkBatchSize;
    strcpy(m_filePath, (p_opt.m_indexDirectory + FolderSep + p_opt.m_ssdMappingFile + "_postings").c_str());
    m_disableCheckpoint = p_opt.m_disableCheckpoint;
    m_startTime = std::chrono::high_resolution_clock::now();

    int numblocks = m_batchSize;
    m_fileHandle = f_createAsyncIO();
    if (m_fileHandle == nullptr ||
        !m_fileHandle->Initialize(
            m_filePath,
#ifndef _MSC_VER
            O_RDWR | O_DIRECT, numblocks, 2, 2,
            max(p_opt.m_ioThreads, (2 * max(p_opt.m_searchThreadNum, p_opt.m_iSSDNumberOfThreads) +
                                    p_opt.m_insertThreadNum + p_opt.m_reassignThreadNum + p_opt.m_appendThreadNum)),
            ((std::uint64_t)p_opt.m_startFileSize) << 30
#else
            GENERIC_READ | GENERIC_WRITE, numblocks, 2, 2,
            (std::uint16_t)(p_opt.m_ioThreads),
            ((std::uint64_t)p_opt.m_startFileSize) << 30
#endif
            ))
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::Initialize failed\n");
        return false;
    }
    std::string blockpoolPath = (p_opt.m_recovery)
                                    ? p_opt.m_persistentBufferPath + FolderSep + p_opt.m_ssdMappingFile + "_postings"
                                    : m_filePath;
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO::BlockController:Loading block pool from file:%s\n",
                 blockpoolPath.c_str());
    ErrorCode ret = LoadBlockPool(blockpoolPath, ((std::uint64_t)p_opt.m_startFileSize) << (30 - PageSizeEx),
                                  !p_opt.m_recovery, p_opt.m_datasetRowsInBlock, p_opt.m_datasetCapacity);
    if (ErrorCode::Success != ret)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO::BlockController:Loading block pool failed!\n");
        return false;
    }
    return true;
}

bool FileIO::BlockController::NeedsExpansion(int p_size)
{
    // TODO: Replace RemainBlocks() which internally uses tbb::concurrent_queue::unsafe_size().
    // unsafe_size() is *not thread-safe* and may yield inconsistent results under concurrent access.
    size_t available = RemainBlocks();
    size_t total = m_totalAllocatedBlocks.load();
    float ratio = static_cast<float>(available) / static_cast<float>(total);

    return (available < p_size) || (ratio < m_growthThreshold);
}

bool FileIO::BlockController::ExpandFile(AddressType blocksToAdd)
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

    if (!m_fileHandle->ExpandFile(allowedBlocks * PageSize))
        return false;

    m_available.AddBatch(allowedBlocks);
    for (AddressType i = 0; i < allowedBlocks; ++i)
    {
        m_blockAddresses.push(currentTotal + i);
        m_available.Insert(currentTotal + i);
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

bool FileIO::BlockController::GetBlocks(AddressType *p_data, int p_size)
{
    // Trigger expansion if we're below threshold
    if (NeedsExpansion(p_size))
    {
        std::unique_lock<std::mutex> lock(
            m_expandLock); // ensure only one thread tries expandingAdd commentMore actions
        if (NeedsExpansion(p_size))
        { // recheck inside lock
            AddressType growBy = max(static_cast<AddressType>(m_growthBlocks), (AddressType)p_size);
            AddressType total = m_totalAllocatedBlocks.load();
            if (total + growBy <= m_maxBlocks)
            {
                if (!ExpandFile(growBy))
                {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::GetBlocks: expansion failed\n");
                    return false;
                }
            }
            else
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
                             "FileIO::BlockController::GetBlocks: cannot expand beyond cap (%llu)\n",
                             static_cast<unsigned long long>(m_maxBlocks));
            }
        }
    }

    for (int i = 0; i < p_size; i++)
    {
        int retry = 0;
        AddressType currBlockAddress = 0xffffffffffffffff;
        while (retry < 3 && !m_blockAddresses.try_pop(currBlockAddress)) retry++;
        if (retry < 3)
        {
            p_data[i] = currBlockAddress;
            if (!m_available.Reset(currBlockAddress))
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "GetBlocks: Allocate a used block id:%lld\n",
                             currBlockAddress);
            }
        }
        else
            return false;
    }
    return true;
}

bool FileIO::BlockController::ReleaseBlocks(AddressType *p_data, int p_size)
{
    for (int i = 0; i < p_size; i++)
    {
        if (!m_available.Insert(p_data[i]))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "ReleaseBlocks: Double release a free block id:%lld\n",
                         p_data[i]);
            continue;
        }
        if (m_disableCheckpoint)
        {
            m_blockAddresses.push(p_data[i]);
        }
        else
        {
            m_blockAddresses_reserve.push(p_data[i]);
        }
    }
    return true;
}

bool FileIO::BlockController::ReadBlocks(AddressType *p_data, std::string *p_value,
                                         const std::chrono::microseconds &timeout,
                                         std::vector<Helper::AsyncReadRequest> *reqs)
{
    if ((uintptr_t)p_data == 0xffffffffffffffff)
    {
        p_value->resize(0);
        return true;
    }

    const int64_t postingSize = (int64_t)(p_data[0]);
    auto blockNum = (postingSize + PageSize - 1) >> PageSizeEx;
    if (blockNum > reqs->size())
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::ReadBlocks: %d > %u\n", (int)blockNum,
                     reqs->size());
        return false;
    }

    AddressType currOffset = 0;
    AddressType dataIdx = 1;
    for (int i = 0; i < blockNum; i++)
    {
        Helper::AsyncReadRequest &curr = reqs->at(i);
        curr.m_readSize = (postingSize - currOffset) < PageSize ? (postingSize - currOffset) : PageSize;
        curr.m_offset = p_data[dataIdx] * PageSize;
        currOffset += PageSize;
        dataIdx++;
    }

    std::uint32_t totalReads = m_fileHandle->BatchReadFile(reqs->data(), blockNum, timeout, m_batchSize);
    read_submit_vec += blockNum;
    read_complete_vec += totalReads;

    if (totalReads < blockNum)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "FileIO::BlockController::ReadBlocks: %u < %u\n", totalReads,
                     blockNum);
        m_batchReadTimeouts++;
        return false;
    }

    p_value->resize(postingSize);
    for (int i = 0; i < blockNum; i++)
    {
        Helper::AsyncReadRequest &curr = reqs->at(i);
        memcpy(p_value->data() + i * PageSize, curr.m_buffer, curr.m_readSize);
    }
    return true;
}

bool FileIO::BlockController::ReadBlocks(
    AddressType *p_data, Helper::PageBuffer<std::uint8_t> &p_value, const std::chrono::microseconds &timeout,
    std::vector<Helper::AsyncReadRequest> *reqs)
{
    if ((uintptr_t)p_data == 0xffffffffffffffff)
    {
        p_value.SetAvailableSize(0);
        return true;
    }

    const int64_t postingSize = (int64_t)(p_data[0]);
    auto blockNum = (postingSize + PageSize - 1) >> PageSizeEx;
    if (blockNum > reqs->size())
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::ReadBlocks: %d > %u\n", (int)blockNum,
                     reqs->size());
        p_value.SetAvailableSize(0);
        return false;
    }

    p_value.SetAvailableSize(postingSize);
    AddressType currOffset = 0;
    AddressType dataIdx = 1;
    for (int i = 0; i < blockNum; i++)
    {
        Helper::AsyncReadRequest &curr = reqs->at(i);
        curr.m_readSize = (postingSize - currOffset) < PageSize ? (postingSize - currOffset) : PageSize;
        curr.m_offset = p_data[dataIdx] * PageSize;
        currOffset += PageSize;
        dataIdx++;
    }

    std::uint32_t totalReads = m_fileHandle->BatchReadFile(reqs->data(), blockNum, timeout, m_batchSize);
    read_submit_vec += blockNum;
    read_complete_vec += totalReads;

    if (totalReads < blockNum)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "FileIO::BlockController::ReadBlocks: %u < %u\n", totalReads,
                     blockNum);
        m_batchReadTimeouts++;
        return false;
    }
    return true;
}
    
bool FileIO::BlockController::ReadBlocks(const std::vector<AddressType *> &p_data, std::vector<std::string> *p_values,
                                         const std::chrono::microseconds &timeout,
                                         std::vector<Helper::AsyncReadRequest> *reqs)
{
    m_batchReadTimes++;

    std::uint32_t reqcount = 0;
    for (size_t i = 0; i < p_data.size(); i++)
    {
        AddressType *p_data_i = p_data[i];

        if (p_data_i == nullptr || (uintptr_t)p_data_i == 0xffffffffffffffff)
        {
            if (p_data_i != nullptr) (*p_values)[i].resize(0);
            continue;
        }

        std::size_t postingSize = (std::size_t)p_data_i[0];
        AddressType currOffset = 0;
        AddressType dataIdx = 1;
        while (currOffset < postingSize)
        {
            if (reqcount >= reqs->size())
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::ReadBlocks:  %u >= %u\n", reqcount,
                             reqs->size());
                return false;
            }

            Helper::AsyncReadRequest &curr = reqs->at(reqcount);
            curr.m_readSize = (postingSize - currOffset) < PageSize ? (postingSize - currOffset) : PageSize;
            curr.m_offset = p_data_i[dataIdx] * PageSize;
            currOffset += PageSize;
            dataIdx++;
            reqcount++;
        }
    }

    std::uint32_t totalReads = m_fileHandle->BatchReadFile(reqs->data(), reqcount, timeout, m_batchSize);
    read_submit_vec += reqcount;
    read_complete_vec += totalReads;

    if (totalReads < reqcount)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "FileIO::BlockController::ReadBlocks: %u < %u\n", totalReads,
                     reqcount);
        m_batchReadTimeouts++;
        return false;
    }

    std::uint32_t reqdone = 0;
    for (size_t i = 0; i < p_data.size(); i++)
    {
        AddressType *p_data_i = p_data[i];

        if (p_data_i == nullptr || (uintptr_t)p_data_i == 0xffffffffffffffff)
        {
            continue;
        }

        std::size_t postingSize = (std::size_t)p_data_i[0];
        ((*p_values)[i]).resize(postingSize);
        AddressType currOffset = 0;
        while (currOffset < postingSize)
        {
            Helper::AsyncReadRequest &curr = reqs->at(reqdone);
            memcpy(((*p_values)[i]).data() + currOffset, curr.m_buffer, curr.m_readSize);
            currOffset += PageSize;
            reqdone++;
        }
    }
    return true;
}

bool FileIO::BlockController::ReadBlocks(const std::vector<AddressType *> &p_data,
                                         std::vector<Helper::PageBuffer<std::uint8_t>> &p_values,
                                         const std::chrono::microseconds &timeout,
                                         std::vector<Helper::AsyncReadRequest> *reqs)
{
    m_batchReadTimes++;
    std::uint32_t reqcount = 0;
    std::uint32_t emptycount = 0;
    for (size_t i = 0; i < p_data.size(); i++)
    {
        AddressType *p_data_i = p_data[i];
        int numPages = (p_values[i].GetPageSize() >> PageSizeEx);

        if (p_data_i == nullptr || (uintptr_t)p_data_i == 0xffffffffffffffff)
        {
            if (p_data_i != nullptr) p_values[i].SetAvailableSize(0);
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
                SPTAGLIB_LOG(
                    Helper::LogLevel::LL_Error,
                    "FileIO::BlockController::ReadBlocks:  block (%d) pos:%llu/%llu >= buffer page size (%d)\n",
                    dataIdx - 1, currOffset, postingSize, numPages);
                break;
            }

            if (reqcount >= reqs->size())
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                             "FileIO::BlockController::ReadBlocks:  req (%u) >= req array size (%u)\n", reqcount,
                             reqs->size());
                return false;
            }

            Helper::AsyncReadRequest &curr = reqs->at(reqcount);
            curr.m_readSize = (postingSize - currOffset) < PageSize ? (postingSize - currOffset) : PageSize;
            curr.m_offset = p_data_i[dataIdx] * PageSize;
            currOffset += PageSize;
            dataIdx++;
            reqcount++;
        }

        while (dataIdx - 1 < numPages)
        {
            if (reqcount >= reqs->size())
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                             "FileIO::BlockController::ReadBlocks:  req (%u) >= req array size (%u)\n", reqcount,
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

    std::uint32_t totalReads = m_fileHandle->BatchReadFile(reqs->data(), reqcount, timeout, m_batchSize);
    read_submit_vec += reqcount - emptycount;
    read_complete_vec += totalReads;
    if (totalReads < reqcount - emptycount)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "FileIO::BlockController::ReadBlocks: %u < %u\n", totalReads,
                     reqcount - emptycount);
        m_batchReadTimeouts++;
        return false;
    }
    return true;
}

bool FileIO::BlockController::WriteBlocks(AddressType *p_data, int p_size, const std::string &p_value,
                                          const std::chrono::microseconds &timeout,
                                          std::vector<Helper::AsyncReadRequest> *reqs)
{
    // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "WriteBlocks to string with blocknum=%d size=%u!\n", p_size,
    // p_value.size());
    AddressType currOffset = 0;
    int totalSize = p_value.size();
    for (int i = 0; i < p_size; i++)
    {
        Helper::AsyncReadRequest &curr = reqs->at(i);
        curr.m_readSize = (totalSize - currOffset) < PageSize ? (totalSize - currOffset) : PageSize;
        curr.m_offset = p_data[i] * PageSize;
        memcpy(curr.m_buffer, p_value.data() + currOffset, curr.m_readSize);
        currOffset += PageSize;
    }

    std::uint32_t totalWrites = m_fileHandle->BatchWriteFile(reqs->data(), p_size, timeout, m_batchSize);
    write_submit_vec += p_size;
    write_complete_vec += totalWrites;
    if (totalWrites < p_size)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "FileIO::BlockController::WriteBlocks: %u < %d\n", totalWrites,
                     p_size);
        return false;
    }
    return true;
}

bool FileIO::BlockController::IOStatistics()
{
    int64_t currReadCount = read_complete_vec;
    int64_t read_submit_count = read_submit_vec;
    int64_t currWriteCount = write_complete_vec;
    int64_t write_submit_count = write_submit_vec;
    int64_t read_blocks_time = read_blocks_time_vec;
    int64_t read_bytes_count = read_bytes_vec;
    int64_t write_bytes_count = write_bytes_vec;

    int currIOCount = currReadCount + currWriteCount;
    int diffIOCount = currIOCount - m_preIOCompleteCount;
    m_preIOCompleteCount = currIOCount;

    int64_t currBytesCount = read_bytes_count + write_bytes_count;
    // int64_t diffBytesCount = currBytesCount - m_preIOBytes;
    m_preIOBytes = currBytesCount;

    auto currTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(currTime - m_preTime);
    m_preTime = currTime;

    double currIOPS = (double)diffIOCount * 1000 / duration.count();
    double currBandWidth = (double)diffIOCount * PageSize / 1024 * 1000 / 1024 * 1000 / duration.count();
    // double currBandWidth = (double)diffBytesCount / 1024 * 1000 / 1024 * 1000 / duration.count();

    std::cout << "Diff IO Count: " << diffIOCount << " Time: " << duration.count() << "us" << std::endl;
    std::cout << "IOPS: " << currIOPS << "k Bandwidth: " << currBandWidth << "MB/s" << std::endl;
    std::cout << "Read Count: " << currReadCount << " Write Count: " << currWriteCount
              << " Read Submit Count: " << read_submit_count << " Write Submit Count: " << write_submit_count
              << std::endl;
    std::cout << "Read Bytes Count: " << read_bytes_count << " Write Bytes Count: " << write_bytes_count << std::endl;
    std::cout << "Read Blocks Time: " << read_blocks_time << "ns" << std::endl;
    std::cout << "Batch Read Times: " << m_batchReadTimes.load()
              << " Batch Read Timeouts: " << m_batchReadTimeouts.load() << std::endl;
    return true;
}

bool FileIO::BlockController::ShutDown()
{
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO:BlockController:ShutDown!\n");
    Checkpoint(m_filePath);
    while (!m_blockAddresses.empty())
    {
        AddressType currBlockAddress;
        m_blockAddresses.try_pop(currBlockAddress);
    }
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO::BlockController::Close file handler\n");
    m_fileHandle->ShutDown();
    return true;
}

} // namespace SPTAG::SPANN
