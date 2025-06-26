#ifndef _SPTAG_PW_FILEIOINTERFACE_H_
#define _SPTAG_PW_FILEIOINTERFACE_H_

#include "inc/Core/SPANN/Options.h"
#include "inc/Core/SPANN/IExtraSearcher.h"
#include "inc/Core/SPANN/ExtraFileController.h"

typedef std::int64_t AddressType;
typedef std::int32_t SizeType;
typedef SPTAG::ErrorCode ErrorCode;
typedef SPTAG::ByteArray ByteArray;
class FileIOInterface {
public:
    FileIOInterface(const char* filePath, SizeType postingBlocks, SizeType bufferSize = 1024, int batchSize = 64) {
        SPTAG::SPANN::Options opt;
        std::string tmp(filePath);
        opt.m_indexDirectory = tmp.substr(0, tmp.find_last_of(FolderSep));
        opt.m_ssdMappingFile = tmp.substr(tmp.find_last_of(FolderSep) + 1);
        opt.m_startFileSize = (postingBlocks >> (30 - SPTAG::PageSizeEx));
        opt.m_bufferLength = bufferSize;
        opt.m_spdkBatchSize = batchSize;

        m_fileIO = std::make_unique<SPTAG::SPANN::FileIO>(opt);
        m_workSpace = std::make_unique<SPTAG::SPANN::ExtraWorkSpace>();
        m_workSpace->Initialize(opt.m_maxCheck, opt.m_hashExp, opt.m_searchInternalResultNum, max(opt.m_postingPageLimit, opt.m_searchPostingPageLimit + 1) << SPTAG::PageSizeEx, true, opt.m_enableDataCompression);
    }

    void ShutDown() {
        m_fileIO->ShutDown();
    }

    bool Get(SizeType key, std::string& value) {
        ErrorCode result = m_fileIO->Get(key, &value, SPTAG::MaxTimeout, &(m_workSpace->m_diskRequests));
        if (result != ErrorCode::Success) {
            value = "";
        }
        return true;
    }

    std::string Get(SizeType key) {
        std::string value;
        ErrorCode result = m_fileIO->Get(key, &value, SPTAG::MaxTimeout, &(m_workSpace->m_diskRequests));
        if (result != ErrorCode::Success) {
            return "";
        }
        return value;
    }

    std::string Get(const std::string& key) {
        std::string value;
        ErrorCode result = m_fileIO->Get(key, &value, SPTAG::MaxTimeout, &(m_workSpace->m_diskRequests));
        if (result != ErrorCode::Success) {
            return "";
        }
        return value;
    }

    bool MultiGet(const std::vector<SizeType>& keys, std::vector<std::string>* values, const std::chrono::microseconds &timeout = std::chrono::microseconds::max()) {
        return m_fileIO->MultiGet(keys, values, timeout, &(m_workSpace->m_diskRequests)) == ErrorCode::Success;
    }

    bool MultiGet(const std::vector<std::string>& keys, std::vector<std::string>* values, const std::chrono::microseconds &timeout = std::chrono::microseconds::max()) {
        return m_fileIO->MultiGet(keys, values, timeout, &(m_workSpace->m_diskRequests)) == ErrorCode::Success;
    }

    bool Put(SizeType key, const std::string& value) {
        return m_fileIO->Put(key, value, SPTAG::MaxTimeout, &(m_workSpace->m_diskRequests)) == ErrorCode::Success;
    }

    bool Put(const std::string& key, const std::string& value) {
        return m_fileIO->Put(key, value, SPTAG::MaxTimeout, &(m_workSpace->m_diskRequests)) == ErrorCode::Success;
    }

    bool Merge(SizeType key, const std::string& value) {
        return m_fileIO->Merge(key, value, SPTAG::MaxTimeout, &(m_workSpace->m_diskRequests)) == ErrorCode::Success;
    }

    bool Merge(const std::string& key, const std::string& value) {
        return m_fileIO->Merge(key, value, SPTAG::MaxTimeout, &(m_workSpace->m_diskRequests)) == ErrorCode::Success;
    }

    bool Delete(SizeType key) {
        return m_fileIO->Delete(key) == ErrorCode::Success;
    }

    bool Delete(const std::string& key) {
        return m_fileIO->Delete(key) == ErrorCode::Success;
    }

    void ForceCompaction() {
        m_fileIO->ForceCompaction();
    }

    void GetStat() {
        m_fileIO->GetStat();
    }

    bool Load(std::string path, SizeType blockSize, SizeType capacity) {
        return m_fileIO->Load(path, blockSize, capacity) == ErrorCode::Success;
    }

    bool Save(std::string path) {
        return m_fileIO->Save(path) == ErrorCode::Success;
    }
    
    bool Checkpoint(std::string prefix) {
        return m_fileIO->Checkpoint(prefix) == ErrorCode::Success;
    }

private:
std::unique_ptr<SPTAG::SPANN::FileIO> m_fileIO;
std::unique_ptr<SPTAG::SPANN::ExtraWorkSpace> m_workSpace;
};

#endif // _SPTAG_PW_FILEIOINTERFACE_H_