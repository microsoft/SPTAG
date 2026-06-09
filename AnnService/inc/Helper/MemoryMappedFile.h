// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_HELPER_MEMORYMAPPEDFILE_H_
#define _SPTAG_HELPER_MEMORYMAPPEDFILE_H_

#include <cstdint>
#include <string>

#ifdef _MSC_VER
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace SPTAG
{
namespace Helper
{

// RAII read-only whole-file memory mapping. The OS demand-pages the file on access and
// reclaims clean file-backed pages under memory pressure, so resident set stays bounded
// instead of holding the whole file in heap. Header-only so it needs no build-list change.
class MemoryMappedFile
{
public:
    MemoryMappedFile() = default;

    ~MemoryMappedFile() { Close(); }

    // Non-copyable, non-movable: a single owner unmaps exactly once.
    MemoryMappedFile(const MemoryMappedFile&) = delete;
    MemoryMappedFile& operator=(const MemoryMappedFile&) = delete;

    // Maps the entire file read-only. Returns false (and leaves the object closed) on failure.
    bool Open(const std::string& p_path)
    {
        Close();
#ifdef _MSC_VER
        HANDLE file = ::CreateFileA(p_path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE) return false;

        LARGE_INTEGER size;
        if (!::GetFileSizeEx(file, &size) || size.QuadPart <= 0)
        {
            ::CloseHandle(file);
            return false;
        }

        HANDLE mapping = ::CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (mapping == nullptr)
        {
            ::CloseHandle(file);
            return false;
        }

        void* view = ::MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
        if (view == nullptr)
        {
            ::CloseHandle(mapping);
            ::CloseHandle(file);
            return false;
        }

        m_base = reinterpret_cast<std::uint8_t*>(view);
        m_length = static_cast<std::uint64_t>(size.QuadPart);
        m_file = file;
        m_mapping = mapping;
        return true;
#else
        int fd = ::open(p_path.c_str(), O_RDONLY);
        if (fd < 0) return false;

        struct stat st;
        if (::fstat(fd, &st) != 0 || st.st_size <= 0)
        {
            ::close(fd);
            return false;
        }

        void* p = ::mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_SHARED, fd, 0);
        // The mapping keeps its own reference to the file, so the descriptor can be closed now.
        ::close(fd);
        if (p == MAP_FAILED) return false;

#if defined(MADV_RANDOM)
        // BKT build accesses vectors randomly by id; suppress readahead to keep RSS tight.
        ::madvise(p, static_cast<size_t>(st.st_size), MADV_RANDOM);
#endif
        m_base = reinterpret_cast<std::uint8_t*>(p);
        m_length = static_cast<std::uint64_t>(st.st_size);
        return true;
#endif
    }

    void Close()
    {
        if (m_base == nullptr) return;
#ifdef _MSC_VER
        ::UnmapViewOfFile(m_base);
        if (m_mapping != nullptr) ::CloseHandle(m_mapping);
        if (m_file != nullptr) ::CloseHandle(m_file);
        m_mapping = nullptr;
        m_file = nullptr;
#else
        ::munmap(m_base, static_cast<size_t>(m_length));
#endif
        m_base = nullptr;
        m_length = 0;
    }

    const std::uint8_t* Data() const { return m_base; }

    std::uint64_t Length() const { return m_length; }

    bool IsOpen() const { return m_base != nullptr; }

private:
    std::uint8_t* m_base = nullptr;
    std::uint64_t m_length = 0;
#ifdef _MSC_VER
    void* m_file = nullptr;
    void* m_mapping = nullptr;
#endif
};

} // namespace Helper
} // namespace SPTAG

#endif // _SPTAG_HELPER_MEMORYMAPPEDFILE_H_
