#include "inc/Core/SPANN/ExtraFileController.h"

namespace SPTAG::SPANN
{
thread_local struct FileIO::BlockController::IoContext FileIO::BlockController::m_currIoContext;
thread_local int FileIO::BlockController::debug_fd = -1;
thread_local int FileIO::id = 0;
#ifdef USE_ASYNC_IO
thread_local uint64_t FileIO::BlockController::iocp = 0;
thread_local int FileIO::BlockController::id = 0;
#endif
std::chrono::high_resolution_clock::time_point FileIO::BlockController::m_startTime;
int FileIO::BlockController::m_ssdInflight = 0;
std::atomic<int> FileIO::BlockController::m_ioCompleteCount(0);
int FileIO::BlockController::fd = -1;
char* FileIO::BlockController::filePath = new char[1024];
std::unique_ptr<char[]> FileIO::BlockController::m_memBuffer;

void* FileIO::BlockController::InitializeFileIo(void* args) {
    FileIO::BlockController* ctrl = (FileIO::BlockController *)args;
    struct stat st;
    const char* fileIoPath = getenv(kFileIoPath);
    auto fileSize = kSsdImplMaxNumBlocks << PageSizeEx;
    if(fileIoPath) {
        strcpy(filePath, fileIoPath);
    }
    else {
        fprintf(stderr, "FileIO::BlockController::InitializeFileIo failed: No filePath\n");
        ctrl->m_fileIoThreadStartFailed = true;
        fd = -1;
        // strcpy(filePath, "/home/lml/SPFreshTest/testfile");
    }
    if(stat(filePath, &st) != 0) {
	    fprintf(stderr, "FileIO:create a file\n");
        fd = open(filePath, O_CREAT | O_WRONLY, 0666);
        if(fd == -1) {
            fprintf(stderr, "FileIO::BlockController::InitializeFileIo failed\n");
            // return nullptr;
            ctrl->m_fileIoThreadStartFailed = true;
        }
        else {
            if (fallocate(fd, 0, 0, fileSize) == -1) {
                fprintf(stderr, "FileIO::BlockController::InitializeFileIo failed: fallocate failed\n");
                ctrl->m_fileIoThreadStartFailed = true;
                fd = -1;
            } else {
                close(fd);
                fd = open(filePath, O_RDWR | O_DIRECT);
                if (fd == -1) {
                    auto err_str = strerror(errno);
                    fprintf(stderr, "open failed: %s\n", err_str);
                    ctrl->m_fileIoThreadStartFailed = true;
                } else {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO::BlockController::InitializeFileIo: file %s created, fd=%d\n", filePath, fd);
                }
            }
        }
    }
    else {
	    fprintf(stderr, "FileIO:open a file\n");
        auto actualFileSize = st.st_blocks * st.st_blksize;
        if(actualFileSize < fileSize) {
            fd = open(filePath, O_CREAT | O_WRONLY, 0666);
            if(fd == -1) {
                fprintf(stderr, "FileIO::BlockController::InitializeFileIo failed\n");
                // return nullptr;
                ctrl->m_fileIoThreadStartFailed = true;
            }
            else {
                if (fallocate(fd, 0, 0, fileSize) == -1) {
                    fprintf(stderr, "FileIO::BlockController::InitializeFileIo failed: fallocate failed\n");
                    ctrl->m_fileIoThreadStartFailed = true;
                    fd = -1;
                } else {
                    close(fd);
                    fd = open(filePath, O_RDWR | O_DIRECT);
                    if (fd == -1) {
                        auto err_str = strerror(errno);
                        fprintf(stderr, "open failed: %s\n", err_str);
                        ctrl->m_fileIoThreadStartFailed = true;
                    } else {
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO::BlockController::InitializeFileIo: file %s created, fd=%d\n", filePath, fd);
                    }
                }
            }
        } else {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO::BlockController::InitializeFileIo: file has been created with enough space.\n", filePath, fd);
            fd = open(filePath, O_RDWR | O_DIRECT);
            if (fd == -1) {
                auto err_str = strerror(errno);
                fprintf(stderr, "open failed: %s\n", err_str);
                ctrl->m_fileIoThreadStartFailed = true;
            } else {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO::BlockController::InitializeFileIo: file %s opened, fd=%d\n", filePath, fd);
            }
        }
    }
    const char* fileIoDepth = getenv(kFileIoDepth);
    if (fileIoDepth) ctrl->m_ssdFileIoDepth = atoi(fileIoDepth);
    const char* fileIoThreadNum = getenv(kFileIoThreadNum);
    if (fileIoThreadNum) ctrl->m_ssdFileIoThreadNum = atoi(fileIoThreadNum);
    const char* fileIoAlignment = getenv(kFileIoAlignment);
    if (fileIoAlignment) ctrl->m_ssdFileIoAlignment = atoi(fileIoAlignment);

    ctrl->m_batchReadTimes = 0;
    ctrl->m_batchReadTimeouts = 0;

    if(ctrl->m_fileIoThreadStartFailed == false) {
        ctrl->m_fileIoThreadReady = true;
        // std::lock_guard<std::mutex> lock(ctrl->m_uniqueResourceMutex);
        // m_ssdInflight = 0;
    }
    pthread_exit(NULL);
}

bool FileIO::BlockController::Initialize(int batchSize) {
    std::lock_guard<std::mutex> lock(m_initMutex);
    m_numInitCalled++;

    if(m_numInitCalled == 1) {
        m_batchSize = batchSize;
        m_startTime = std::chrono::high_resolution_clock::now();
        for(AddressType i = 0; i < kSsdImplMaxNumBlocks; i++) {
            m_blockAddresses.push(i);
        }
        pthread_create(&m_fileIoTid, NULL, &InitializeFileIo, this);
        while(!m_fileIoThreadReady && !m_fileIoThreadStartFailed);
        if(m_fileIoThreadStartFailed) {
            fprintf(stderr, "FileIO::BlockController::Initialize failed\n");
            return false;
        }
    }
    if (m_idQueue.empty()) {
        id = m_maxId;
        m_maxId++;
    }
    else {
        id = m_idQueue.front();
        m_idQueue.pop();
    }
    while(read_complete_vec.size() <= id) {
        read_complete_vec.push_back(0);
    }
    while(read_submit_vec.size() <= id) {
        read_submit_vec.push_back(0);
    }
    while(write_complete_vec.size() <= id) {
        write_complete_vec.push_back(0);
    }
    while(write_submit_vec.size() <= id) {
        write_submit_vec.push_back(0);
    }
    while(read_bytes_vec.size() <= id) {
        read_bytes_vec.push_back(0);
    }
    while(write_bytes_vec.size() <= id) {
        write_bytes_vec.push_back(0);
    }
    while(read_blocks_time_vec.size() <= id) {
        read_blocks_time_vec.push_back(0);
    }

    InitializeThreadContext();
#ifdef USE_FILE_DEBUG
    auto debug_file_name = std::string("/nvme1n1/lml/") + std::to_string(m_numInitCalled) + "_debug.log";
    debug_fd = open(debug_file_name.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
    if (debug_fd == -1) {
        fprintf(stderr, "FileIO::BlockController::Initialize failed: open debug file failed\n");
        return false;
    }
#endif
    return true;
}


bool FileIO::BlockController::InitializeThreadContext() {
    if (m_currIoContext.sub_io_requests.size() >= m_ssdFileIoDepth) return true;

    m_currIoContext.sub_io_requests.resize(m_ssdFileIoDepth);
    m_currIoContext.in_flight = 0;
    for(auto &sr : m_currIoContext.sub_io_requests) {
        sr.app_buff = nullptr;
        memset(&(sr.myiocb), 0, sizeof(struct iocb));
        auto buf_ptr = aligned_alloc(m_ssdFileIoAlignment, PageSize);
        if (buf_ptr == nullptr) {
            fprintf(stderr, "FileIO::BlockController::Initialize failed: aligned_alloc failed\n");
            return false;
        }
        sr.myiocb.aio_buf = reinterpret_cast<uint64_t>(buf_ptr);
        sr.myiocb.aio_fildes = fd;
        sr.myiocb.aio_data = reinterpret_cast<uintptr_t>(&sr);
        sr.myiocb.aio_nbytes = PageSize;
        sr.ctrl = this;
        m_currIoContext.free_sub_io_requests.push(&sr);
    }
    iocp = 0;
    auto ret = syscall(__NR_io_setup, m_ssdFileIoDepth, &iocp);
    if (ret < 0) {
        fprintf(stderr, "FileIO::BlockController::Initialize io_setup failed: %s\n", strerror(errno));
        fprintf(stderr, "m_ssdFileIoDepth = %d, iocp = %p\n", m_ssdFileIoDepth, &iocp);
        return false;
    }
    return true;
}

bool FileIO::BlockController::GetBlocks(AddressType* p_data, int p_size) {
    AddressType currBlockAddress = 0;
#ifdef USE_FILE_DEBUG
    auto debug_string = std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - m_startTime).count()) + " 1";
    auto result = pwrite(debug_fd, debug_string.c_str(), debug_string.size(), 0);
    if (result == -1) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::GetBlocks: %s\n", strerror(errno));
    }
    fsync(debug_fd);
#endif
    for(int i = 0; i < p_size; i++) {
        while(!m_blockAddresses.try_pop(currBlockAddress));
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
    //InitializeThreadContext();
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
    read_submit_vec[id] += blockNum;
    if (blockNum > m_ssdFileIoDepth) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::ReadBlocks: blockNum > m_ssdFileIoDepth\n");
        return false;
    }

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
        totalDone += d;
        read_complete_vec[id] += d;
    }
    return true;
}

bool FileIO::BlockController::ReadBlocks(AddressType* p_data, ByteArray& p_value, const std::chrono::microseconds &timeout) {
    //InitializeThreadContext();
    std::uint8_t* outdata = new std::uint8_t[p_data[0]];    
    p_value.Set(outdata, p_data[0], false);
    AddressType currOffset = 0;
    AddressType dataIdx = 1;
    auto blockNum = (p_data[0] + PageSize - 1) >> PageSizeEx;
    read_submit_vec[id] += blockNum;
    if (blockNum > m_ssdFileIoDepth) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::ReadBlocks: blockNum > m_ssdFileIoDepth\n");
        return false;
    }

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
        read_bytes_vec[id] += currSubIo->myiocb.aio_nbytes;
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
            }
        }
        auto begin = std::chrono::high_resolution_clock::now();

        // Get IO result
        int wait = totalSubmitted - totalDone;
        auto d = syscall(__NR_io_getevents, iocp, wait, wait, events.data() + totalDone, &timeout_ts);
        if (d < 0) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::ReadBlocks: io_getevents failed\n");
            return false;
        }
        auto end = std::chrono::high_resolution_clock::now();
        read_blocks_time_vec[id] += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        for (int i = totalDone; i < totalDone + d; i++) {
            auto req = reinterpret_cast<SubIoRequest*>(events[i].data);
            memcpy(req->app_buff, reinterpret_cast<void *>(req->myiocb.aio_buf), req->real_size);
            req->app_buff = nullptr;
            m_currIoContext.free_sub_io_requests.push(req);
        }
        totalDone += d;
        read_complete_vec[id] += d;
    }
    return true;
}

bool FileIO::BlockController::ReadBlocks(const std::vector<AddressType*>& p_data, std::vector<std::string>* p_values, const std::chrono::microseconds &timeout) {
    //InitializeThreadContext();
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
            read_submit_vec[id]++;
            currOffset += PageSize;
            dataIdx++;
        }
    }

       //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO::BlockController::ReadBlocks: sub_io_requests.size:%d free_io_requests.size:%d\n", (int)(m_currIoContext.sub_io_requests.size()), (int)(m_currIoContext.free_sub_io_requests.size()));
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

        //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO::BlockController::ReadBlocks: totalToSubmit:%d\n", totalToSubmit);
        while (totalDone < totalToSubmit) {
            // Submit all I/Os
            if(totalSubmitted < totalToSubmit) {
                int s = syscall(__NR_io_submit, iocp, totalToSubmit - totalSubmitted, iocbs.data() + totalSubmitted);
                if(s > 0) {
                    totalSubmitted += s;
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
            totalDone += d;
            read_complete_vec[id] += d;
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        if(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1) > timeout) {
            break;
        }
    }

       //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO::BlockController::ReadBlocks: totalDone == totalToSubmit\n");
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

       //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO::BlockController::ReadBlocks: finish\n");
    return true;
}

bool FileIO::BlockController::ReadBlocks(const std::vector<AddressType*>& p_data, std::vector<ByteArray>& p_values, const std::chrono::microseconds& timeout) {
    //InitializeThreadContext();
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
        p_values[i].Set(out_data, p_data_i[0], false);
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
            read_submit_vec[id]++;
            currOffset += PageSize;
            dataIdx++;
        }
    }

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
            totalDone += d;
            read_complete_vec[id] += d;
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
    //InitializeThreadContext();
#ifdef USE_FILE_DEBUG
    auto debug_string = std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - m_startTime).count()) + " 5";
    auto result = pwrite(debug_fd, debug_string.c_str(), debug_string.size(), 0);
    if (result == -1) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FileIO::BlockController::WriteBlocks: pwrite failed\n");
    }
    fsync(debug_fd);
#endif
    AddressType currBlockIdx = 0;
    int inflight = 0;
    SubIoRequest* currSubIo;
    int totalSize = p_value.size();
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
    write_submit_vec[id] += p_size;
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
        }
        int wait = totalSubmitted - totalDone;
        auto d = syscall(__NR_io_getevents, iocp, wait, wait, events.data() + totalDone, nullptr);
        for (int i = totalDone; i < totalDone + d; i++) {
            auto req = reinterpret_cast<SubIoRequest*>(events[i].data);
            req->app_buff = nullptr;
            m_currIoContext.free_sub_io_requests.push(req);
        }
        totalDone += d;
        write_complete_vec[id] += d;
    }
    return true;
}

bool FileIO::BlockController::WriteBlocks(AddressType* p_data, int p_size, const ByteArray& p_value) {
    //InitializeThreadContext();
    AddressType currBlockIdx = 0;
    int inflight = 0;
    SubIoRequest* currSubIo;
    int totalSize = p_value.Length();
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
    write_submit_vec[id] += p_size;
    std::vector<struct iocb*> iocbs(p_size);
    for (int i = 0; i < p_size; i++) {
        auto currSubIo = m_currIoContext.free_sub_io_requests.front();
        m_currIoContext.free_sub_io_requests.pop();
        currSubIo->app_buff = (void*)p_value.Data() + currBlockIdx * PageSize;
        currSubIo->real_size = (PageSize * (currBlockIdx + 1)) > totalSize ? (totalSize - currBlockIdx * PageSize) : PageSize;
        // currSubIo->myiocb.aio_nbytes = ((currSubIo->real_size - 1) / 512 + 1) * 512;
        currSubIo->myiocb.aio_nbytes = PageSize;
        write_bytes_vec[id] += currSubIo->myiocb.aio_nbytes;
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
        }
        int wait = totalSubmitted - totalDone;
        auto d = syscall(__NR_io_getevents, iocp, wait, wait, events.data() + totalDone, nullptr);
        for (int i = totalDone; i < totalDone + d; i++) {
            auto req = reinterpret_cast<SubIoRequest*>(events[i].data);
            req->app_buff = nullptr;
            m_currIoContext.free_sub_io_requests.push(req);
        }
        totalDone += d;
        write_complete_vec[id] += d;
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
    for (int i = 0; i < read_complete_vec.size(); i++) {
        currReadCount += read_complete_vec[i];
    }
    for (int i = 0; i < read_submit_vec.size(); i++) {
        read_submit_count += read_submit_vec[i];
    }
    for (int i = 0; i < write_complete_vec.size(); i++) {
        currWriteCount += write_complete_vec[i];
    }
    for (int i = 0; i < write_submit_vec.size(); i++) {
        write_submit_count += write_submit_vec[i];
    }
    for (int i = 0; i < read_bytes_vec.size(); i++) {
        read_bytes_count += read_bytes_vec[i];
    }
    for (int i = 0; i < write_bytes_vec.size(); i++) {
        write_bytes_count += write_bytes_vec[i];
    }
    for (int i = 0; i < read_blocks_time_vec.size(); i++) {
        read_blocks_time += read_blocks_time_vec[i];
    }
    
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
    // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "FileIO::BlockController::ShutDown\n");
    if (m_numInitCalled == 0) {
        m_fileIoThreadExiting = true;
        pthread_join(m_fileIoTid, NULL);
        while (!m_blockAddresses.empty()) {
            AddressType currBlockAddress;
            m_blockAddresses.try_pop(currBlockAddress);
        }
	close(fd);
    }
    m_idQueue.push(id);
    syscall(__NR_io_destroy, iocp);
    for (auto &sr : m_currIoContext.sub_io_requests) {
        sr.app_buff = nullptr;
        auto buf_ptr = reinterpret_cast<void *>(sr.myiocb.aio_buf);
        free(buf_ptr);
        sr.myiocb.aio_buf = 0;
    }
    while(m_currIoContext.free_sub_io_requests.size()) {
        m_currIoContext.free_sub_io_requests.pop();
    }
    return true;
}

} // namespace SPTAG::SPANN
