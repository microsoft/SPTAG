#pragma once

#include "inc/SSDServing/VectorSearch/DiskFileReader.h"
#include "inc/SSDServing/VectorSearch/DiskListCommonUtils.h"

#include <memory>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include <thread>
#include <stdint.h>

#include <boost/lockfree/stack.hpp>

#include <Windows.h>

namespace SPTAG {
    namespace SSDServing {
        namespace VectorSearch {

            namespace DiskFileReaderUtil
            {
                struct PrioritizedDiskFileReaderResource;
            }

            class PrioritizedDiskFileReader : public IDiskFileReader
            {
            public:
                PrioritizedDiskFileReader(const char* p_filePath);
                virtual ~PrioritizedDiskFileReader();

                virtual bool IsOpened() const;

                virtual bool ReadFileAsync(const DiskFileReadRequest& p_request);

            private:
                typedef DiskFileReaderUtil::PrioritizedDiskFileReaderResource ResourceType;

                void ListionIOCP();

                ResourceType* GetResource();

                void ReturnResource(ResourceType* p_res);

                void PreAllocQueryContext();

            private:
                std::string m_filePath;

                HandleWrapper m_fileHandle;

                HandleWrapper m_fileIocp;

                std::vector<std::thread> m_fileIocpThreads;

                uint32_t m_diskSectorSize;

                boost::lockfree::stack<ResourceType*> m_resources;
            };
        }
    }
}
