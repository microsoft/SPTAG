#include "inc/SSDServing/VectorSearch/PrioritizedDiskFileReader.h"
#include "inc/SSDServing/VectorSearch/DiskAccessUtils.h"
#include "inc/Core/Common.h"

#include <sstream>
#include <cstdio>

#include <fileapi.h>

namespace SPTAG {
    namespace SSDServing {
        namespace VectorSearch
        {
            namespace DiskFileReaderUtil
            {
                struct CallbackOverLapped : public OVERLAPPED
                {
                    PrioritizedDiskFileReaderResource* const c_registeredResource;

                    const std::function<void()>* m_callback;


                    CallbackOverLapped(PrioritizedDiskFileReaderResource* p_registeredResource)
                        : c_registeredResource(p_registeredResource),
                        m_callback(nullptr)
                    {
                    }
                };


                struct PrioritizedDiskFileReaderResource
                {
                    CallbackOverLapped m_col;

                    PrioritizedDiskFileReaderResource()
                        : m_col(this)
                    {
                    }
                };

            }
        }
    }
}


using namespace SPTAG::SSDServing::VectorSearch;
using namespace SPTAG::SSDServing::VectorSearch::DiskFileReaderUtil;

PrioritizedDiskFileReader::PrioritizedDiskFileReader(const char* p_filePath)
    : m_resources(4096),
    m_diskSectorSize(0)
{
    LOG(Helper::LogLevel::LL_Info, "Start open file handle: %s\n", p_filePath);

    m_filePath = p_filePath;

    m_fileHandle.Reset(::CreateFileA(m_filePath.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,
        NULL));

    if (!m_fileHandle.IsValid())
    {
        LOG(Helper::LogLevel::LL_Error, "Failed to create file handle: %s\n", m_filePath.c_str());
        return;
    }

    int iocpThreads = 4;

    m_fileIocp.Reset(::CreateIoCompletionPort(m_fileHandle.GetHandle(), NULL, NULL, iocpThreads));
    for (int i = 0; i < iocpThreads; ++i)
    {
        m_fileIocpThreads.emplace_back(std::thread(std::bind(&PrioritizedDiskFileReader::ListionIOCP, this)));
    }

    LOG(Helper::LogLevel::LL_Info, "Success open file handle: %s\n", m_filePath.c_str());

    m_diskSectorSize = static_cast<uint32_t>(DiskUtils::GetSectorSize(m_filePath.c_str()));
    LOG(Helper::LogLevel::LL_Info, "DiskSectorSize: %u\n", m_diskSectorSize);

    PreAllocQueryContext();
}


PrioritizedDiskFileReader::~PrioritizedDiskFileReader()
{
    m_fileHandle.Close();
    m_fileIocp.Close();

    for (auto& th : m_fileIocpThreads)
    {
        if (th.joinable())
        {
            th.join();
        }
    }

    ResourceType* res = nullptr;
    while (m_resources.pop(res))
    {
        if (res != nullptr)
        {
            delete res;
        }
    }
}


bool
PrioritizedDiskFileReader::IsOpened() const
{
    return m_fileHandle.IsValid() && m_fileIocp.IsValid();
}


bool
PrioritizedDiskFileReader::ReadFileAsync(const DiskFileReadRequest& p_request)
{
    ResourceType* resource = GetResource();

    CallbackOverLapped& col = resource->m_col;
    LARGE_INTEGER li;
    li.QuadPart = p_request.m_offset;

    col.hEvent = nullptr;
    col.Pointer = nullptr;
    col.Internal = 0;
    col.InternalHigh = 0;

    col.Offset = li.LowPart;
    col.OffsetHigh = li.HighPart;

    col.m_callback = &(p_request.m_callback);

    bool successRead = true;
    if (!::ReadFile(m_fileHandle.GetHandle(),
        p_request.m_buffer,
        static_cast<DWORD>(p_request.m_readSize),
        nullptr,
        &col))
    {
        if (GetLastError() != ERROR_IO_PENDING)
        {
            LOG(Helper::LogLevel::LL_Error, "Failed to read file\n");
            successRead = false;
        }
    }

    if (!successRead)
    {
        ReturnResource(resource);
    }

    return successRead;
}

void
PrioritizedDiskFileReader::ListionIOCP()
{
    DWORD cBytes;
    ULONG_PTR key;
    OVERLAPPED* ol;
    CallbackOverLapped* col;

    while (true)
    {
        BOOL ret = ::GetQueuedCompletionStatus(this->m_fileIocp.GetHandle(),
            &cBytes,
            &key,
            &ol,
            INFINITE);
        if (FALSE == ret || nullptr == ol)
        {
            //We exit the thread
            return;
        }

        col = (CallbackOverLapped*)ol;

        auto callback = col->m_callback;
        ReturnResource(col->c_registeredResource);

        if (nullptr != callback && (*callback))
        {
            (*callback)();
        }
    }
}


PrioritizedDiskFileReader::ResourceType*
PrioritizedDiskFileReader::GetResource()
{
    ResourceType* ret = nullptr;
    if (m_resources.pop(ret))
    {
    }
    else
    {
        ret = new ResourceType();
    }

    return ret;
}


void
PrioritizedDiskFileReader::ReturnResource(PrioritizedDiskFileReader::ResourceType* p_res)
{
    if (p_res != nullptr)
    {
        m_resources.push(p_res);
    }
}

void
PrioritizedDiskFileReader::PreAllocQueryContext()
{
    const size_t num = 64 * 64;
    typedef ResourceType* ResourcePtr;

    ResourcePtr* contextArray = new ResourcePtr[num];

    for (int i = 0; i < num; ++i)
    {
        contextArray[i] = GetResource();
    }

    for (int i = 0; i < num; ++i)
    {
        ReturnResource(contextArray[i]);
        contextArray[i] = nullptr;
    }

    delete[] contextArray;
}