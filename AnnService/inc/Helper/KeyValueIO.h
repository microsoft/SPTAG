#ifndef _SPTAG_HELPER_KEYVALUEIO_H_
#define _SPTAG_HELPER_KEYVALUEIO_H_

#include "inc/Core/Common.h"
#include "inc/Helper/DiskIO.h"
#include <vector>
#include <chrono>

namespace SPTAG
{
    namespace Helper
    {
        class KeyValueIO {
        public:
            KeyValueIO() {}

            virtual ~KeyValueIO() {}

            virtual void ShutDown() = 0;

            virtual ErrorCode Get(const std::string& key, std::string* value, const std::chrono::microseconds& timeout, std::vector<Helper::AsyncReadRequest>* reqs) { return ErrorCode::Undefined; }

            virtual ErrorCode Get(const SizeType key, std::string* value, const std::chrono::microseconds& timeout, std::vector<Helper::AsyncReadRequest>* reqs) = 0;

            virtual ErrorCode Get(const SizeType key, Helper::PageBuffer<std::uint8_t> &value, const std::chrono::microseconds &timeout, std::vector<Helper::AsyncReadRequest> *reqs, bool useCache = true) { return ErrorCode::Undefined; }

            virtual ErrorCode MultiGet(const std::vector<SizeType>& keys, std::vector<SPTAG::Helper::PageBuffer<std::uint8_t>>& values, const std::chrono::microseconds& timeout, std::vector<Helper::AsyncReadRequest>* reqs) { return ErrorCode::Undefined; }

            virtual ErrorCode MultiGet(const std::vector<SizeType>& keys, std::vector<std::string>* values, const std::chrono::microseconds& timeout, std::vector<Helper::AsyncReadRequest>* reqs) = 0;

            virtual ErrorCode MultiGet(const std::vector<std::string>& keys, std::vector<std::string>* values, const std::chrono::microseconds &timeout, std::vector<Helper::AsyncReadRequest>* reqs) { return ErrorCode::Undefined; }            
 
            virtual ErrorCode Put(const std::string& key, const std::string& value, const std::chrono::microseconds& timeout, std::vector<Helper::AsyncReadRequest>* reqs) { return ErrorCode::Undefined; }

            virtual ErrorCode Put(const SizeType key, const std::string& value, const std::chrono::microseconds& timeout, std::vector<Helper::AsyncReadRequest>* reqs) = 0;

            virtual ErrorCode Merge(const SizeType key, const std::string &value,
                                    const std::chrono::microseconds &timeout,
                                    std::vector<Helper::AsyncReadRequest> *reqs,
                                    std::function<bool(const void *val, const int size)> checksum) = 0;

            virtual ErrorCode Delete(SizeType key) = 0;

            virtual ErrorCode DeleteRange(SizeType start, SizeType end) {return ErrorCode::Undefined;}

            virtual void ForceCompaction() {}

            virtual void GetStat() {}

            virtual int64_t GetNumBlocks() { return 0; }
            
            virtual bool Available() { return false; }

            virtual ErrorCode Check(const SizeType key, int size, std::vector<std::uint8_t> *visited)
            {
                return ErrorCode::Undefined;
            }

            virtual int64_t GetApproximateMemoryUsage() const
            {
                return 0;
            }

            virtual ErrorCode Checkpoint(std::string prefix) {return ErrorCode::Undefined;}

            virtual ErrorCode StartToScan(SizeType& key, std::string* value) {return ErrorCode::Undefined;}

            virtual ErrorCode NextToScan(SizeType& key, std::string* value) {return ErrorCode::Undefined;}
        };
    }
}

#endif