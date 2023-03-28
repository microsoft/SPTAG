#ifndef _SPTAG_HELPER_KEYVALUEIO_H_
#define _SPTAG_HELPER_KEYVALUEIO_H_

#include "inc/Core/Common.h"
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

            virtual ErrorCode Get(const std::string& key, std::string* value) { return ErrorCode::Undefined; }

            virtual ErrorCode Get(SizeType key, std::string* value) = 0;

            virtual ErrorCode MultiGet(const std::vector<std::string>& keys, std::vector<std::string>* values, const std::chrono::microseconds &timeout = std::chrono::microseconds::max()) { return ErrorCode::Undefined; }

            virtual ErrorCode MultiGet(const std::vector<SizeType>& keys, std::vector<std::string>* values, const std::chrono::microseconds &timeout = std::chrono::microseconds::max()) = 0;

            virtual ErrorCode Put(const std::string& key, const std::string& value) { return ErrorCode::Undefined; }

            virtual ErrorCode Put(SizeType key, const std::string& value) = 0;

            virtual  ErrorCode Merge(SizeType key, const std::string& value) = 0;

            virtual ErrorCode Delete(SizeType key) = 0;

            virtual void ForceCompaction() {}

            virtual void GetStat() {}

            virtual bool Initialize(bool debug = false) { return false; }

            virtual bool ExitBlockController(bool debug = false) { return false; }
        };
    }
}

#endif