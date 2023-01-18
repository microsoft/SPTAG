#ifndef _SPTAG_PW_FILEIOINTERFACE_H_
#define _SPTAG_PW_FILEIOINTERFACE_H_

#include "inc/Core/SPANN/ExtraFileController.h"

typedef std::int64_t AddressType;
typedef std::int32_t SizeType;
typedef SPTAG::ErrorCode ErrorCode;
typedef SPTAG::ByteArray ByteArray;
class FileIOInterface {
public:
    FileIOInterface(const char* filePath, SizeType blockSize, SizeType capacity, SizeType postingBlocks, SizeType bufferSize = 1024, int batchSize = 64, bool recovery = false, int compactionThreads = 1) {
        m_fileIO = std::make_unique<SPTAG::SPANN::FileIO>(filePath, blockSize, capacity, postingBlocks, bufferSize, batchSize, recovery, compactionThreads);
    }

    void ShutDown() {
        m_fileIO->ShutDown();
    }

    ByteArray GetByteArray(SizeType key) {
        ByteArray value;
        ErrorCode result = m_fileIO->Get(key, value);
        if (result != ErrorCode::Success) {
            return ByteArray::c_empty;
        }
        return value;
    }

    bool Get(SizeType key, std::string& value) {
        return true;
    }

    std::string Get(SizeType key) {
        std::string value;
        ErrorCode result = m_fileIO->Get(key, &value);
        if (result != ErrorCode::Success) {
            return "";
        }
        return value;
    }

    std::string Get(const std::string& key) {
        std::string value;
        ErrorCode result = m_fileIO->Get(key, &value);
        if (result != ErrorCode::Success) {
            return "";
        }
        return value;
    }

    std::vector<ByteArray> Scan(const SizeType start_key, const int record_count) {
        std::vector<ByteArray> values;
        auto result = m_fileIO->Scan(start_key, record_count, values);
        if (result != ErrorCode::Success) {
            for (auto ba : values) {
                delete[] ba.Data();
            }
            return std::vector<ByteArray>(0);
        }
        return values;
    }

    bool MultiGet(const std::vector<SizeType>& keys, std::vector<std::string>* values, const std::chrono::microseconds &timeout = std::chrono::microseconds::max()) {
        return m_fileIO->MultiGet(keys, values, timeout) == ErrorCode::Success;
    }

    bool MultiGet(const std::vector<std::string>& keys, std::vector<std::string>* values, const std::chrono::microseconds &timeout = std::chrono::microseconds::max()) {
        return m_fileIO->MultiGet(keys, values, timeout) == ErrorCode::Success;
    }

    bool Put(SizeType key, const std::string& value) {
        return m_fileIO->Put(key, value) == ErrorCode::Success;
    }

    bool Put(SizeType key, ByteArray value) {
        return m_fileIO->Put(key, value) == ErrorCode::Success;
    }

    bool Put(const std::string& key, const std::string& value) {
        return m_fileIO->Put(key, value) == ErrorCode::Success;
    }

    bool Merge(SizeType key, const std::string& value) {
        return m_fileIO->Merge(key, value) == ErrorCode::Success;
    }

    bool Merge(const std::string& key, const std::string& value) {
        return m_fileIO->Merge(key, value) == ErrorCode::Success;
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

    bool Initialize(bool debug = false) {
        return m_fileIO->Initialize(debug);
    }

    bool ExitBlockController(bool debug = false) {
        return m_fileIO->ExitBlockController(debug);
    }
    
    bool Checkpoint(std::string prefix) {
        return m_fileIO->Checkpoint(prefix) == ErrorCode::Success;
    }

private:
std::unique_ptr<SPTAG::SPANN::FileIO> m_fileIO;
};

#endif // _SPTAG_PW_FILEIOINTERFACE_H_