#include "inc/Core/SPANN/ExtraFileController.h"

namespace SPTAG::SPANN
{
typedef tbb::concurrent_hash_map<int, std::pair<int, uint64_t>> fdmap;
thread_local struct FileIO::BlockController::IoContext FileIO::BlockController::m_currIoContext;
thread_local int FileIO::BlockController::debug_fd = -1;
thread_local int FileIO::id = -1;
thread_local fdmap FileIO::BlockController::fd_to_id_iocp;

std::atomic<int> FileIO::BlockController::m_ioCompleteCount(0);

void FileIO::BlockController::InitializeFileIo() {
    if (m_filePath == nullptr) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "BlockController::InitializeFileIo failed: No filePath\n");
        m_fileIoThreadStartFailed = true;
        fd = -1;
        return;
    }

    struct stat st;
    const off_t fileSize = static_cast<off_t>(kFileIoStartBlocks) << PageSizeEx;

    bool needFallocate = false;

    if (stat(m_filePath, &st) != 0) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "BlockController::Create new file %s\n", m_filePath);
        needFallocate = true;
    } else {
        off_t actualFileSize = st.st_size;
        if (actualFileSize < fileSize) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "BlockController::File exists but is too small (%lld < %lld), expanding.\n",
                static_cast<long long>(actualFileSize), static_cast<long long>(fileSize));
            needFallocate = true;
        } else {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "BlockController::File has sufficient size: %s\n", m_filePath);
        }
    }

    fd = open(m_filePath, O_CREAT | O_RDWR | O_DIRECT, 0666);
    if (fd == -1) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "BlockController::Failed to open file: %s\n", strerror(errno));
        m_fileIoThreadStartFailed = true;
        return;
    }

    if (needFallocate) {
        if (fallocate(fd, 0, 0, fileSize) == -1) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "BlockController::fallocate failed: %s\n", strerror(errno));
            m_fileIoThreadStartFailed = true;
            close(fd);
            fd = -1;
            return;
        } else {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                "BlockController::Allocated file space: %lld bytes for %s\n",
                static_cast<long long>(fileSize), m_filePath);
        }
    }

    // Read tuning parameters from environment
    const char* fileIoDepth = getenv(kFileIoDepth);
    if (fileIoDepth) m_ssdFileIoDepth = atoi(fileIoDepth);

    const char* fileIoThreadNum = getenv(kFileIoThreadNum);
    if (fileIoThreadNum) m_ssdFileIoThreadNum = atoi(fileIoThreadNum);

    const char* fileIoAlignment = getenv(kFileIoAlignment);
    if (fileIoAlignment) m_ssdFileIoAlignment = atoi(fileIoAlignment);

    m_batchReadTimes = 0;
    m_batchReadTimeouts = 0;

    if (!m_fileIoThreadStartFailed) {
        m_fileIoThreadReady = true;
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "BlockController::InitializeFileIo succeeded. fd=%d\n", fd);
    }
}

bool FileIO::BlockController::Initialize(std::string filePath, int batchSize) {
    std::lock_guard<std::mutex> lock(m_initMutex);
    m_numInitCalled++;

    if(m_numInitCalled == 1) {
        m_batchSize = batchSize;
	    strcpy(m_filePath, (filePath + "_postings").c_str());
        m_startTime = std::chrono::high_resolution_clock::now();

	    InitializeFileIo();
        
        while(!m_fileIoThreadReady && !m_fileIoThreadStartFailed);
        if(m_fileIoThreadStartFailed) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::Initialize failed\n");
            return false;
        }
        LoadBlockPool(m_filePath, true);
    }

    int id = 0;
    if (m_idQueue.empty()) {
        id = m_maxId;
        m_maxId++;
    }
    else {
        id = m_idQueue.front();
        m_idQueue.pop();
    }

    if (m_currIoContext.sub_io_requests.size() < m_ssdFileIoDepth) {
        int original = m_currIoContext.sub_io_requests.size();
        m_currIoContext.sub_io_requests.resize(m_ssdFileIoDepth);
        for(int i = original; i < m_ssdFileIoDepth; i++) {
            auto &sr = m_currIoContext.sub_io_requests[i];
            sr.app_buff = nullptr;
            memset(&(sr.myiocb), 0, sizeof(struct iocb));
            auto buf_ptr = BLOCK_ALLOC(m_ssdFileIoAlignment, PageSize);
            if (buf_ptr == nullptr) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::Initialize failed: aligned_alloc failed\n");
                return false;
            }
            sr.myiocb.aio_buf = reinterpret_cast<uint64_t>(buf_ptr);
            sr.myiocb.aio_fildes = fd;
            sr.myiocb.aio_data = reinterpret_cast<uintptr_t>(&sr);
            sr.myiocb.aio_nbytes = PageSize;
            sr.ctrl = this;
            m_currIoContext.free_sub_io_requests.push(&sr);
        }
    }
    uint64_t iocp = 0;
    auto ret = syscall(__NR_io_setup, m_ssdFileIoDepth, &iocp);
    if (ret < 0) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::Initialize io_setup failed: %s\n", strerror(errno));
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "m_ssdFileIoDepth = %d, iocp = %p\n", m_ssdFileIoDepth, &iocp);
        return false;
    }
    
    fdmap::accessor a;
    if (!fd_to_id_iocp.insert(a, fd)) {
        if (!fd_to_id_iocp.find(a, fd)) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::Initialize failed: insert (%d, (%d, %llu)) to fd_to_id_iocp failed\n",
                fd, id, iocp);
        }
    } else {
        a->second = std::make_pair(id, iocp);
    }
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO::BlockController::Initialize(%s, %d) with (%d, (%d, %llu)) num of init calls:%d\n",
        filePath.c_str(), batchSize, fd, id, iocp, m_numInitCalled);

#ifdef USE_FILE_DEBUG
    auto debug_file_name = std::string("/nvme1n1/lml/") + std::to_string(m_numInitCalled) + "_debug.log";
    debug_fd = open(debug_file_name.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
    if (debug_fd == -1) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::Initialize failed: open debug file failed\n");
        return false;
    }
#endif
    return true;
}

bool FileIO::BlockController::NeedsExpansion()
{
    // TODO: Replace RemainBlocks() which internally uses tbb::concurrent_queue::unsafe_size().
    // unsafe_size() is *not thread-safe* and may yield inconsistent results under concurrent access.
    size_t available = RemainBlocks();
    size_t total = m_totalAllocatedBlocks.load();
    float ratio = static_cast<float>(available) / static_cast<float>(total);

    return (ratio < fFileIoThreshold);
}

bool FileIO::BlockController::ExpandFile(AddressType blocksToAdd)
{
    AddressType blockSize = static_cast<AddressType>(1ULL << PageSizeEx);
    AddressType currentTotal = m_totalAllocatedBlocks.load();

    if (currentTotal >= kFileIODefaultMaxBlocks) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
            "[ExpandFile] File is already at max capacity (%llu blocks = %.2f GB). Expansion aborted.\n",
            static_cast<unsigned long long>(kFileIODefaultMaxBlocks),
            (kFileIODefaultMaxBlocks * blockSize) / double(1 << 30));
        return false;
    }

    // Cap the growth so we do not exceed the limit
    AddressType allowedBlocks = std::min(blocksToAdd, kFileIODefaultMaxBlocks - currentTotal);
    off_t startOffset = static_cast<off_t>(currentTotal) * blockSize;
    off_t expandSize = static_cast<off_t>(allowedBlocks) * blockSize;

    if (fallocate(fd, 0, startOffset, expandSize) != 0) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
            "[ExpandFile] fallocate failed at offset %lld for size %lld bytes: %s\n",
            static_cast<long long>(startOffset),
            static_cast<long long>(expandSize),
            strerror(errno));
        return false;
    }

    for (AddressType i = 0; i < allowedBlocks; ++i) {
        m_blockAddresses.push(currentTotal + i);
    }

    AddressType newTotal = currentTotal + allowedBlocks;
    m_totalAllocatedBlocks.store(newTotal);

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
        "[ExpandFile] Expanded file from %llu to %llu blocks (added %llu blocks = %.2f GB)\n",
        static_cast<unsigned long long>(currentTotal),
        static_cast<unsigned long long>(newTotal),
        static_cast<unsigned long long>(allowedBlocks),
        (allowedBlocks * blockSize) / double(1 << 30));

    if (allowedBlocks < blocksToAdd) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Warning,
            "[ExpandFile] Requested %llu blocks, but only %llu were allowed due to cap (%llu max)\n",
            static_cast<unsigned long long>(blocksToAdd),
            static_cast<unsigned long long>(allowedBlocks),
            static_cast<unsigned long long>(kFileIODefaultMaxBlocks));
    }

    return true;
}

bool FileIO::BlockController::GetBlocks(AddressType* p_data, int p_size) {
    AddressType currBlockAddress = 0;

#ifdef USE_FILE_DEBUG
    auto debug_string = std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now() - m_startTime).count()) + " 1";
    auto result = pwrite(debug_fd, debug_string.c_str(), debug_string.size(), 0);
    if (result == -1) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::GetBlocks: %s\n", strerror(errno));
    }
    fsync(debug_fd);
#endif

    // Trigger expansion if we're below threshold
    if (NeedsExpansion()) {
        std::unique_lock<std::mutex> lock(m_expandLock); // ensure only one thread tries expanding
        if (NeedsExpansion()) { // recheck inside lock
            AddressType growBy = static_cast<AddressType>(kFileIoGrowthBlocks);
            AddressType total = m_totalAllocatedBlocks.load();
            if (total + growBy <= kFileIODefaultMaxBlocks) {
                if (!ExpandFile(growBy)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::GetBlocks: expansion failed\n");
                    return false;
                }
            } else {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "FileIO::BlockController::GetBlocks: cannot expand beyond cap (%llu)\n",
                            static_cast<unsigned long long>(kFileIODefaultMaxBlocks));
            }
        }
    }
    for (int i = 0; i < p_size; ++i) {
        while (!m_blockAddresses.try_pop(currBlockAddress)); // Spin-wait until available
        p_data[i] = currBlockAddress;
    }

    return true;
}

bool FileIO::BlockController::ReleaseBlocks(AddressType* p_data, int p_size) {
#ifdef USE_FILE_DEBUG
    auto debug_string = std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - m_startTime).count()) + " 2";
    auto result = pwrite(debug_fd, debug_string.c_str(), debug_string.size(), 0);
    if (result == -1) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::ReleaseBlocks: pwrite failed\n");
    }
    fsync(debug_fd);
#endif
    for(int i = 0; i < p_size; i++) {
        m_blockAddresses_reserve.push(p_data[i]);
    }
    return true;
}

bool FileIO::BlockController::ReadBlocks(AddressType* p_data, std::string* p_value, const std::chrono::microseconds &timeout) {
#ifdef USE_FILE_DEBUG
    auto debug_string = std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - m_startTime).count()) + " 3";
    auto result = pwrite(debug_fd, debug_string.c_str(), debug_string.size(), 0);
    if (result == -1) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::ReadBlocks: pwrite failed\n");
    }
    fsync(debug_fd);
#endif
    p_value->resize(p_data[0]);
    AddressType currOffset = 0;
    AddressType dataIdx = 1;
    auto blockNum = (p_data[0] + PageSize - 1) >> PageSizeEx;
    //read_submit_vec[id] += blockNum;
    if (blockNum > m_ssdFileIoDepth) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::ReadBlocks: blockNum > m_ssdFileIoDepth\n");
        return false;
    }

    fdmap::accessor accessor;
    bool isFound = fd_to_id_iocp.find(accessor, fd);
    if (!isFound) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot find fd=%d in thread local map!\n", fd);
        return false;
    }
    int id = (accessor->second).first;
    uint64_t iocp = (accessor->second).second;
    // clear timeout io
    while (m_currIoContext.free_sub_io_requests.size() < m_ssdFileIoDepth) {
        auto wait = m_ssdFileIoDepth - m_currIoContext.free_sub_io_requests.size();
        std::vector<struct io_event> events(wait);
        struct timespec timeout_ts {0, timeout.count() * 100};   // 用10%的timeout时间等待
        if (timeout.count() == std::chrono::microseconds::max().count()) {
            timeout_ts.tv_sec = 0;
            timeout_ts.tv_nsec = 500 * 1000;
        }
        auto d = syscall(__NR_io_getevents, iocp, wait, wait, events.data(), &timeout_ts);
        for (int i = 0; i < d; i++) {
            auto req = reinterpret_cast<SubIoRequest*>(events[i].data);
            req->app_buff = nullptr;
            m_currIoContext.free_sub_io_requests.push(req);
        }
    }
    if (blockNum > m_currIoContext.free_sub_io_requests.size()) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::ReadBlocks: blockNum > m_currIoContext.free_sub_io_requests.size()\n");
        return false;
    }
    std::vector<struct iocb*> iocbs(blockNum);
    int submitted = 0, done = 0;

    // Create iocb vector
    for (int i = 0; i < blockNum; i++) {
        auto currSubIo = m_currIoContext.free_sub_io_requests.front();
        m_currIoContext.free_sub_io_requests.pop();
        currSubIo->app_buff = (void*)p_value->data() + currOffset;
        currSubIo->real_size = (p_data[0] - currOffset) < PageSize ? (p_data[0] - currOffset) : PageSize;
        currSubIo->myiocb.aio_lio_opcode = IOCB_CMD_PREAD;
        currSubIo->myiocb.aio_offset = p_data[dataIdx] * PageSize;
        currSubIo->myiocb.aio_nbytes = PageSize;
        iocbs[i] = &(currSubIo->myiocb);
        currOffset += PageSize;
        dataIdx++;
    }
    std::vector<struct io_event> events(blockNum);
    int totalDone = 0, totalSubmitted = 0;
    struct timespec timeout_ts {0, timeout.count() * 1000};
    auto t1 = std::chrono::high_resolution_clock::now();
    while(totalDone < blockNum) {
        auto t2 = std::chrono::high_resolution_clock::now();
        if(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1) > timeout) {
            return false;
        }
        // Submit IO
        if(totalSubmitted < blockNum) {
            int s = syscall(__NR_io_submit, iocp, blockNum - totalSubmitted, iocbs.data() + totalSubmitted);
            if(s > 0) {
                totalSubmitted += s;
            }
            else {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::ReadBlocks: io_submit failed\n");
		exit(1);
            }
        }

        // Get IO result
        int wait = totalSubmitted - totalDone;
        auto d = syscall(__NR_io_getevents, iocp, wait, wait, events.data() + totalDone, &timeout_ts);
        for (int i = totalDone; i < totalDone + d; i++) {
            auto req = reinterpret_cast<SubIoRequest*>(events[i].data);
            memcpy(req->app_buff, reinterpret_cast<void *>(req->myiocb.aio_buf), req->real_size);
            req->app_buff = nullptr;
            m_currIoContext.free_sub_io_requests.push(req);
        }
        if (d > 0) {
            totalDone += d;
            //read_complete_vec[id] += d;
        }
    }
    return true;
}

bool FileIO::BlockController::ReadBlocks(AddressType* p_data, ByteArray& p_value, const std::chrono::microseconds &timeout) {
    std::uint8_t* outdata = new std::uint8_t[p_data[0]];    
     p_value.Set(outdata, p_data[0], true);
    AddressType currOffset = 0;
    AddressType dataIdx = 1;
    auto blockNum = (p_data[0] + PageSize - 1) >> PageSizeEx;
    //read_submit_vec[id] += blockNum;
    if (blockNum > m_ssdFileIoDepth) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::ReadBlocks: blockNum > m_ssdFileIoDepth\n");
        return false;
    }

    fdmap::accessor accessor;
    bool isFound = fd_to_id_iocp.find(accessor, fd);
    if (!isFound) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot find fd=%d in thread local map!\n", fd);
        return false;
    }
    int id = (accessor->second).first;
    uint64_t iocp = (accessor->second).second;
    // clear timeout io
    while (m_currIoContext.free_sub_io_requests.size() < m_ssdFileIoDepth) {
        auto wait = m_ssdFileIoDepth - m_currIoContext.free_sub_io_requests.size();
        std::vector<struct io_event> events(wait);
        struct timespec timeout_ts {0, timeout.count() * 100};   // 用10%的timeout时间等待
        if (timeout.count() == std::chrono::microseconds::max().count()) {
            timeout_ts.tv_sec = 0;
            timeout_ts.tv_nsec = 500 * 1000;
        }
        auto d = syscall(__NR_io_getevents, iocp, wait, wait, events.data(), &timeout_ts);
        for (int i = 0; i < d; i++) {
            auto req = reinterpret_cast<SubIoRequest*>(events[i].data);
            req->app_buff = nullptr;
            m_currIoContext.free_sub_io_requests.push(req);
        }
    }
    if (blockNum > m_currIoContext.free_sub_io_requests.size()) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::ReadBlocks: blockNum > m_currIoContext.free_sub_io_requests.size()\n");
        return false;
    }
    std::vector<struct iocb*> iocbs(blockNum);
    int submitted = 0, done = 0;

    // Create iocb vector
    for (int i = 0; i < blockNum; i++) {
        auto currSubIo = m_currIoContext.free_sub_io_requests.front();
        m_currIoContext.free_sub_io_requests.pop();
        currSubIo->app_buff = (void*)p_value.Data() + currOffset;
        currSubIo->real_size = (p_data[0] - currOffset) < PageSize ? (p_data[0] - currOffset) : PageSize;
        // currSubIo->myiocb.aio_nbytes = ((currSubIo->real_size - 1) / 512 + 1) * 512;
        currSubIo->myiocb.aio_nbytes = PageSize;
        //read_bytes_vec[id] += currSubIo->myiocb.aio_nbytes;
        currSubIo->myiocb.aio_lio_opcode = IOCB_CMD_PREAD;
        currSubIo->myiocb.aio_offset = p_data[dataIdx] * PageSize;
        iocbs[i] = &(currSubIo->myiocb);
        currOffset += PageSize;
        dataIdx++;
    }
    std::vector<struct io_event> events(blockNum);
    int totalDone = 0, totalSubmitted = 0;
    struct timespec timeout_ts {0, timeout.count() * 1000};
    auto t1 = std::chrono::high_resolution_clock::now();
    while(totalDone < blockNum) {
        auto t2 = std::chrono::high_resolution_clock::now();
        if(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1) > timeout) {
            return false;
        }
        // Submit IO
        if(totalSubmitted < blockNum) {
            int s = syscall(__NR_io_submit, iocp, blockNum - totalSubmitted, iocbs.data() + totalSubmitted);
            if(s > 0) {
                totalSubmitted += s;
            }
            else {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::ReadBlocks: io_submit failed\n");
		exit(1);
            }
        }
        auto begin = std::chrono::high_resolution_clock::now();

        // Get IO result
        int wait = totalSubmitted - totalDone;
        auto d = syscall(__NR_io_getevents, iocp, wait, wait, events.data() + totalDone, &timeout_ts);
        if (d < 0) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::ReadBlocks: io_getevents failed\n");
        }
        auto end = std::chrono::high_resolution_clock::now();
        //read_blocks_time_vec[id] += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        for (int i = totalDone; i < totalDone + d; i++) {
            auto req = reinterpret_cast<SubIoRequest*>(events[i].data);
            memcpy(req->app_buff, reinterpret_cast<void *>(req->myiocb.aio_buf), req->real_size);
            req->app_buff = nullptr;
            m_currIoContext.free_sub_io_requests.push(req);
        }
        if (d > 0) {
            totalDone += d;
            //read_complete_vec[id] += d;
        }
    }
    return true;
}

bool FileIO::BlockController::ReadBlocks(const std::vector<AddressType*>& p_data, std::vector<std::string>* p_values, const std::chrono::microseconds &timeout) {
#ifdef USE_FILE_DEBUG
    auto debug_string = std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - m_startTime).count()) + " 4";
    auto result = pwrite(debug_fd, debug_string.c_str(), debug_string.size(), 0);
    if (result == -1) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::ReadBlocks: pwrite failed\n");
    }
    fsync(debug_fd);
#endif
    auto t1 = std::chrono::high_resolution_clock::now();
    m_batchReadTimes++;
    p_values->resize(p_data.size());
    const int batch_size = m_batchSize;
    std::vector<struct iocb*> iocbs(batch_size);
    std::vector<struct io_event> events(batch_size);
    std::vector<SubIoRequest> subIoRequests;
    std::vector<int> subIoRequestCount(p_data.size(), 0);
    subIoRequests.reserve(256);
    for(size_t i = 0; i < p_data.size(); i++) {
        AddressType* p_data_i = p_data[i];
        std::string* p_value = &((*p_values)[i]);

        if (p_data_i == nullptr) {
            continue;
        }

        p_value->resize(p_data_i[0]);
        AddressType currOffset = 0;
        AddressType dataIdx = 1;

        while(currOffset < p_data_i[0]) {
            SubIoRequest currSubIo;
            currSubIo.app_buff = (void*)p_value->data() + currOffset;
            currSubIo.real_size = (p_data_i[0] - currOffset) < PageSize ? (p_data_i[0] - currOffset) : PageSize;
            currSubIo.offset = p_data_i[dataIdx] * PageSize;
            currSubIo.posting_id = i;
            subIoRequests.push_back(currSubIo);
            subIoRequestCount[i]++;
            //read_submit_vec[id]++;
            currOffset += PageSize;
            dataIdx++;
        }
    }

    fdmap::accessor accessor;
    bool isFound = fd_to_id_iocp.find(accessor, fd);
    if (!isFound) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot find fd=%d in thread local map!\n", fd);
        return false;
    }
    int id = (accessor->second).first;
    uint64_t iocp = (accessor->second).second;

    while (m_currIoContext.free_sub_io_requests.size() < m_ssdFileIoDepth) {
        int wait = m_ssdFileIoDepth - m_currIoContext.free_sub_io_requests.size();
        std::vector<struct io_event> events(wait);
        struct timespec timeout_ts {0, 0};
        auto d = syscall(__NR_io_getevents, iocp, wait, wait, events.data(), &timeout_ts);
        for (int i = 0; i < d; i++) {
            auto req = reinterpret_cast<SubIoRequest*>(events[i].data);
            req->app_buff = nullptr;
            m_currIoContext.free_sub_io_requests.push(req);
        }
    }

    struct timespec timeout_ts {0, timeout.count() * 1000};
    
    for (int currSubIoStartId = 0; currSubIoStartId < subIoRequests.size(); currSubIoStartId += batch_size) {
        int currSubIoEndId = (currSubIoStartId + batch_size) > subIoRequests.size() ? subIoRequests.size() : currSubIoStartId + batch_size;
        int currSubIoIdx = currSubIoStartId;
        int totalToSubmit = currSubIoEndId - currSubIoStartId;
        int totalSubmitted = 0, totalDone = 0;
        for (int i = 0; i < totalToSubmit; i++) {
            auto currSubIoIdx = currSubIoStartId + i;
            auto currSubIo = m_currIoContext.free_sub_io_requests.front();
            m_currIoContext.free_sub_io_requests.pop();
            currSubIo->app_buff = subIoRequests[currSubIoIdx].app_buff;
            currSubIo->real_size = subIoRequests[currSubIoIdx].real_size;
            currSubIo->posting_id = subIoRequests[currSubIoIdx].posting_id;
            currSubIo->myiocb.aio_lio_opcode = IOCB_CMD_PREAD;
            currSubIo->myiocb.aio_offset = subIoRequests[currSubIoIdx].offset;
            currSubIo->myiocb.aio_nbytes = PageSize;
            iocbs[i] = &(currSubIo->myiocb);
            currSubIoIdx++;
        }
        while (totalDone < totalToSubmit) {
            // Submit all I/Os
            if(totalSubmitted < totalToSubmit) {
                int s = syscall(__NR_io_submit, iocp, totalToSubmit - totalSubmitted, iocbs.data() + totalSubmitted);
                if(s > 0) {
                    totalSubmitted += s;
                } else {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::ReadBlocks: io_submit failed\n");
		    exit(1);
                }
            }
            int wait = totalSubmitted - totalDone;
            auto d = syscall(__NR_io_getevents, iocp, wait, wait, events.data() + totalDone, &timeout_ts);
            for (int i = totalDone; i < totalDone + d; i++) {
                auto req = reinterpret_cast<SubIoRequest*>(events[i].data);
                memcpy(req->app_buff, reinterpret_cast<void *>(req->myiocb.aio_buf), req->real_size);
                subIoRequestCount[req->posting_id]--;
                req->app_buff = nullptr;
                m_currIoContext.free_sub_io_requests.push(req);
            }
            if (d > 0) {
                totalDone += d;
            }
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        if(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1) > timeout) {
            break;
        }
    }

    bool is_timeout = false;
    for (int i = 0; i < subIoRequestCount.size(); i++) {
        if (subIoRequestCount[i] != 0) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::ReadBlocks (batch) : timeout\n");
            (*p_values)[i].clear();
            is_timeout = true;
        }
    }
    if (is_timeout) {
        m_batchReadTimeouts++;
    }
    return true;
}

bool FileIO::BlockController::ReadBlocks(const std::vector<AddressType*>& p_data, std::vector<ByteArray>& p_values, const std::chrono::microseconds& timeout) {
    auto t1 = std::chrono::high_resolution_clock::now();
    p_values.resize(p_data.size());
    const int batch_size = m_batchSize;
    std::vector<struct iocb*> iocbs(batch_size);
    std::vector<struct io_event> events(batch_size);
    std::vector<SubIoRequest> subIoRequests;
    std::vector<int> subIoRequestCount(p_data.size(), 0);
    subIoRequests.reserve(256);
    for(size_t i = 0; i < p_data.size(); i++) {
        AddressType* p_data_i = p_data[i];
        std::uint8_t* out_data = new std::uint8_t[p_data_i[0]];
        p_values[i].Set(out_data, p_data_i[0], true);
        AddressType currOffset = 0;
        AddressType dataIdx = 1;

        while(currOffset < p_data_i[0]) {
            SubIoRequest currSubIo;
            currSubIo.app_buff = (void*)p_values[i].Data() + currOffset;
            currSubIo.real_size = (p_data_i[0] - currOffset) < PageSize ? (p_data_i[0] - currOffset) : PageSize;
            currSubIo.offset = p_data_i[dataIdx] * PageSize;
            currSubIo.posting_id = i;
            subIoRequests.push_back(currSubIo);
            subIoRequestCount[i]++;
            //read_submit_vec[id]++;
            currOffset += PageSize;
            dataIdx++;
        }
    }

    fdmap::accessor accessor;
    bool isFound = fd_to_id_iocp.find(accessor, fd);
    if (!isFound) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot find fd=%d in thread local map!\n", fd);
        return false;
    }
    int id = (accessor->second).first;
    uint64_t iocp = (accessor->second).second;
    // Clear timeout I/Os
    while (m_currIoContext.free_sub_io_requests.size() < m_ssdFileIoDepth) {
        int wait = m_ssdFileIoDepth - m_currIoContext.free_sub_io_requests.size();
        std::vector<struct io_event> events(wait);
        struct timespec timeout_ts {0, 0};
        auto d = syscall(__NR_io_getevents, iocp, wait, wait, events.data(), &timeout_ts);
        for (int i = 0; i < d; i++) {
            auto req = reinterpret_cast<SubIoRequest*>(events[i].data);
            req->app_buff = nullptr;
            m_currIoContext.free_sub_io_requests.push(req);
        }
    }

    struct timespec timeout_ts {0, timeout.count() * 1000};
    
    for (int currSubIoStartId = 0; currSubIoStartId < subIoRequests.size(); currSubIoStartId += batch_size) {
        int currSubIoEndId = (currSubIoStartId + batch_size) > subIoRequests.size() ? subIoRequests.size() : currSubIoStartId + batch_size;
        int currSubIoIdx = currSubIoStartId;
        int totalToSubmit = currSubIoEndId - currSubIoStartId;
        int totalSubmitted = 0, totalDone = 0;
        for (int i = 0; i < totalToSubmit; i++) {
            auto currSubIoIdx = currSubIoStartId + i;
            auto currSubIo = m_currIoContext.free_sub_io_requests.front();
            m_currIoContext.free_sub_io_requests.pop();
            currSubIo->app_buff = subIoRequests[currSubIoIdx].app_buff;
            currSubIo->real_size = subIoRequests[currSubIoIdx].real_size;
            currSubIo->posting_id = subIoRequests[currSubIoIdx].posting_id;
            currSubIo->myiocb.aio_lio_opcode = IOCB_CMD_PREAD;
            currSubIo->myiocb.aio_offset = subIoRequests[currSubIoIdx].offset;
            currSubIo->myiocb.aio_nbytes = PageSize;
            iocbs[i] = &(currSubIo->myiocb);
            currSubIoIdx++;
        }
        while (totalDone < totalToSubmit) {
            // Submit all I/Os
            if(totalSubmitted < totalToSubmit) {
                int s = syscall(__NR_io_submit, iocp, totalToSubmit - totalSubmitted, iocbs.data() + totalSubmitted);
                if(s > 0) {
                    totalSubmitted += s;
                }  else {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::ReadBlocks: io_submit failed\n");
		            exit(1);
                }
            }
            int wait = totalSubmitted - totalDone;
            auto d = syscall(__NR_io_getevents, iocp, wait, wait, events.data() + totalDone, &timeout_ts);
            for (int i = totalDone; i < totalDone + d; i++) {
                auto req = reinterpret_cast<SubIoRequest*>(events[i].data);
                memcpy(req->app_buff, reinterpret_cast<void *>(req->myiocb.aio_buf), req->real_size);
                subIoRequestCount[req->posting_id]--;
                req->app_buff = nullptr;
                m_currIoContext.free_sub_io_requests.push(req);
            }
            if (d > 0) {
                totalDone += d;
                //read_complete_vec[id] += d;
            }
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        if(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1) > timeout) {
            break;
        }
    }

    for (int i = 0; i < subIoRequestCount.size(); i++) {
        if (subIoRequestCount[i] != 0) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::ReadBlocks (batch) : timeout\n");
            delete[] p_values[i].Data();
            p_values[i].Clear();
        }
    }
    return true;
}
bool FileIO::BlockController::WriteBlocks(AddressType* p_data, int p_size, const std::string& p_value) {
#ifdef USE_FILE_DEBUG
    auto debug_string = std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - m_startTime).count()) + " 5";
    auto result = pwrite(debug_fd, debug_string.c_str(), debug_string.size(), 0);
    if (result == -1) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::WriteBlocks: pwrite failed\n");
    }
    fsync(debug_fd);
#endif
    AddressType currBlockIdx = 0;
    SubIoRequest* currSubIo;
    int totalSize = p_value.size();

    fdmap::accessor accessor;
    bool isFound = fd_to_id_iocp.find(accessor, fd);
    if (!isFound) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot find fd=%d in thread local map!\n", fd);
        return false;
    }
    int id = (accessor->second).first;
    uint64_t iocp = (accessor->second).second; 

    while (m_currIoContext.free_sub_io_requests.size() < m_ssdFileIoDepth) {
        int wait = m_ssdFileIoDepth - m_currIoContext.free_sub_io_requests.size();
        std::vector<struct io_event> events(wait);
        struct timespec timeout_ts {0, 0};
        auto d = syscall(__NR_io_getevents, iocp, wait, wait, events.data(), &timeout_ts);
        for (int i = 0; i < d; i++) {
            auto req = reinterpret_cast<SubIoRequest*>(events[i].data);
            req->app_buff = nullptr;
            m_currIoContext.free_sub_io_requests.push(req);
        }
    }

    // Submit all I/Os
    // write_submit_vec[id] += p_size;
    std::vector<struct iocb*> iocbs(p_size);
    for (int i = 0; i < p_size; i++) {
        auto currSubIo = m_currIoContext.free_sub_io_requests.front();
        m_currIoContext.free_sub_io_requests.pop();
        currSubIo->app_buff = (void*)p_value.data() + currBlockIdx * PageSize;
        currSubIo->real_size = (PageSize * (currBlockIdx + 1)) > totalSize ? (totalSize - currBlockIdx * PageSize) : PageSize;
        memcpy(reinterpret_cast<void *>(currSubIo->myiocb.aio_buf), currSubIo->app_buff, currSubIo->real_size);
        currSubIo->myiocb.aio_lio_opcode = IOCB_CMD_PWRITE;
        currSubIo->myiocb.aio_offset = p_data[currBlockIdx] * PageSize;
        currSubIo->myiocb.aio_nbytes = PageSize;
        iocbs[i] = &(currSubIo->myiocb);
        currBlockIdx++;
    }
    std::vector<struct io_event> events(p_size);
    int totalDone = 0, totalSubmitted = 0;
    while(totalDone < p_size) {
        int s = syscall(__NR_io_submit, iocp, p_size - totalSubmitted, iocbs.data() + totalSubmitted);
        if(s > 0) {
            totalSubmitted += s;
        }
        else {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::WriteBlocks: io_submit failed\n");
	    exit(1);
        }
        int wait = totalSubmitted - totalDone;
        auto d = syscall(__NR_io_getevents, iocp, wait, wait, events.data() + totalDone, nullptr);
        for (int i = totalDone; i < totalDone + d; i++) {
            auto req = reinterpret_cast<SubIoRequest*>(events[i].data);
            req->app_buff = nullptr;
            m_currIoContext.free_sub_io_requests.push(req);
        }
        if (d > 0) {
            totalDone += d;
        }
    }
    return true;
}

bool FileIO::BlockController::WriteBlocks(AddressType* p_data, int p_size, const ByteArray& p_value) {
    AddressType currBlockIdx = 0;
    SubIoRequest* currSubIo;
    int totalSize = p_value.Length();

    fdmap::accessor accessor;
    bool isFound = fd_to_id_iocp.find(accessor, fd);
    if (!isFound) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot find fd=%d in thread local map!\n", fd);
        return false;
    }
    int id = (accessor->second).first;
    uint64_t iocp = (accessor->second).second;
    // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "WriteBlocks: %d\n", p_size);
    // Clear timeout I/Os
    while (m_currIoContext.free_sub_io_requests.size() < m_ssdFileIoDepth) {
        int wait = m_ssdFileIoDepth - m_currIoContext.free_sub_io_requests.size();
        std::vector<struct io_event> events(wait);
        struct timespec timeout_ts {0, 0};
        auto d = syscall(__NR_io_getevents, iocp, wait, wait, events.data(), &timeout_ts);
        for (int i = 0; i < d; i++) {
            auto req = reinterpret_cast<SubIoRequest*>(events[i].data);
            req->app_buff = nullptr;
            m_currIoContext.free_sub_io_requests.push(req);
        }
    }

    // Submit all I/Os
    // write_submit_vec[id] += p_size;
    std::vector<struct iocb*> iocbs(p_size);
    for (int i = 0; i < p_size; i++) {
        auto currSubIo = m_currIoContext.free_sub_io_requests.front();
        m_currIoContext.free_sub_io_requests.pop();
        currSubIo->app_buff = (void*)p_value.Data() + currBlockIdx * PageSize;
        currSubIo->real_size = (PageSize * (currBlockIdx + 1)) > totalSize ? (totalSize - currBlockIdx * PageSize) : PageSize;
        currSubIo->myiocb.aio_nbytes = PageSize;
        memcpy(reinterpret_cast<void *>(currSubIo->myiocb.aio_buf), currSubIo->app_buff, currSubIo->real_size);
        currSubIo->myiocb.aio_lio_opcode = IOCB_CMD_PWRITE;
        currSubIo->myiocb.aio_offset = p_data[currBlockIdx] * PageSize;
        iocbs[i] = &(currSubIo->myiocb);
        currBlockIdx++;
    }
    std::vector<struct io_event> events(p_size);
    int totalDone = 0, totalSubmitted = 0;
    while(totalDone < p_size) {
        int s = syscall(__NR_io_submit, iocp, p_size - totalSubmitted, iocbs.data() + totalSubmitted);
        if(s > 0) {
            totalSubmitted += s;
        }
        else {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::WriteBlocks: io_submit failed\n");
	    exit(1);
        }
        int wait = totalSubmitted - totalDone;
        auto d = syscall(__NR_io_getevents, iocp, wait, wait, events.data() + totalDone, nullptr);
        for (int i = totalDone; i < totalDone + d; i++) {
            auto req = reinterpret_cast<SubIoRequest*>(events[i].data);
            req->app_buff = nullptr;
            m_currIoContext.free_sub_io_requests.push(req);
        }
        if (d > 0) {
            totalDone += d;
        }
    }
    return true;
}

bool FileIO::BlockController::IOStatistics() {
    int currReadCount = 0;
    int read_submit_count = 0;
    int currWriteCount = 0;
    int write_submit_count = 0;
    int64_t read_blocks_time = 0;
    int64_t read_bytes_count = 0;
    int64_t write_bytes_count = 0;

    int currIOCount = currReadCount + currWriteCount;
    int diffIOCount = currIOCount - m_preIOCompleteCount;
    m_preIOCompleteCount = currIOCount;

    int64_t currBytesCount = read_bytes_count + write_bytes_count;
    int64_t diffBytesCount = currBytesCount - m_preIOBytes;
    m_preIOBytes = currBytesCount;

    auto currTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(currTime - m_preTime);
    m_preTime = currTime;

    double currIOPS = (double)diffIOCount * 1000 / duration.count();
    double currBandWidth = (double)diffIOCount * PageSize / 1024 * 1000 / 1024 * 1000 / duration.count();
    // double currBandWidth = (double)diffBytesCount / 1024 * 1000 / 1024 * 1000 / duration.count();

    std::cout << "Diff IO Count: " << diffIOCount << " Time: " << duration.count() << "us" << std::endl;
    std::cout << "IOPS: " << currIOPS << "k Bandwidth: " << currBandWidth << "MB/s" << std::endl;
    std::cout << "Read Count: " << currReadCount << " Write Count: " << currWriteCount << " Read Submit Count: " << read_submit_count << " Write Submit Count: " << write_submit_count << std::endl;
    std::cout << "Read Bytes Count: " << read_bytes_count << " Write Bytes Count: " << write_bytes_count << std::endl;
    std::cout << "Remain free IO requests: " << m_currIoContext.free_sub_io_requests.size() << std::endl;
    std::cout << "Read Blocks Time: " << read_blocks_time << "ns" << std::endl;
    std::cout << "Batch Read Times: " << m_batchReadTimes.load() << " Batch Read Timeouts: " << m_batchReadTimeouts.load() << std::endl;
    return true;
}

bool FileIO::BlockController::ShutDown() {
    std::lock_guard<std::mutex> lock(m_initMutex);
    SubIoRequest* currSubIo;
    m_numInitCalled--;
    fdmap::accessor accessor;
    bool isFound = fd_to_id_iocp.find(accessor, fd);
    if (!isFound) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "ShutDown:Cannot find fd=%d in thread local map!\n", fd);
    }
    int id = (accessor->second).first;
    uint64_t iocp = (accessor->second).second;
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO::BlockController::ShutDown (%d, (%d, %llu)) with num of init call:%d\n", fd, id, iocp, m_numInitCalled);
    if (m_numInitCalled == 0) {
        m_fileIoThreadExiting = true;
        //pthread_join(m_fileIoTid, NULL);
        Checkpoint(m_filePath);
        while (!m_blockAddresses.empty()) {
            AddressType currBlockAddress;
            m_blockAddresses.try_pop(currBlockAddress);
        }
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO::BlockController::Close file handler\n");
	    fd_to_id_iocp.erase(accessor);
        close(fd);
    }
    m_idQueue.push(id);
    syscall(__NR_io_destroy, iocp);
    for (auto &sr : m_currIoContext.sub_io_requests) {
        sr.app_buff = nullptr;
        auto buf_ptr = reinterpret_cast<void *>(sr.myiocb.aio_buf);
        if (buf_ptr) BLOCK_FREE(buf_ptr, PageSize);
        sr.myiocb.aio_buf = 0;
    }
    m_currIoContext.sub_io_requests.clear();

    while(m_currIoContext.free_sub_io_requests.size()) {
        m_currIoContext.free_sub_io_requests.pop();
    }
    return true;
}

} // namespace SPTAG::SPANN
