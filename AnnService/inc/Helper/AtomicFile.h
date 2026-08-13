// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_HELPER_ATOMICFILE_H_
#define _SPTAG_HELPER_ATOMICFILE_H_

#include <cstdio>
#include <filesystem>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace SPTAG
{
namespace Helper
{

inline bool AtomicReplaceFile(
    const std::string& p_temporary,
    const std::string& p_destination)
{
    std::filesystem::path parent =
        std::filesystem::path(
            p_destination).parent_path();
    if (parent.empty()) parent = ".";
#ifdef _WIN32
    const DWORD parentAttributes =
        GetFileAttributesA(parent.string().c_str());
    if (parentAttributes == INVALID_FILE_ATTRIBUTES ||
        (parentAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0) {
        return false;
    }
    const HANDLE temporaryFile = CreateFileA(
        p_temporary.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE |
            FILE_SHARE_DELETE,
        nullptr, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL, nullptr);
    if (temporaryFile == INVALID_HANDLE_VALUE) {
        return false;
    }
    const bool fileSynced =
        FlushFileBuffers(temporaryFile) != 0;
    const bool fileClosed =
        CloseHandle(temporaryFile) != 0;
    if (!fileSynced || !fileClosed) return false;
    return MoveFileExA(
               p_temporary.c_str(),
               p_destination.c_str(),
               MOVEFILE_REPLACE_EXISTING |
                   MOVEFILE_WRITE_THROUGH) != 0;
#else
    int directoryFlags = O_RDONLY;
#ifdef O_DIRECTORY
    directoryFlags |= O_DIRECTORY;
#endif
    const int directory = open(
        parent.c_str(), directoryFlags);
    if (directory < 0) return false;

    const int temporaryFile = open(
        p_temporary.c_str(), O_RDONLY);
    if (temporaryFile < 0) {
        close(directory);
        return false;
    }
    const bool fileSynced =
        fsync(temporaryFile) == 0;
    const bool fileClosed =
        close(temporaryFile) == 0;
    if (!fileSynced || !fileClosed) {
        close(directory);
        return false;
    }
    if (std::rename(
            p_temporary.c_str(),
            p_destination.c_str()) != 0) {
        close(directory);
        return false;
    }
    fsync(directory);
    close(directory);
    return true;
#endif
}

} // namespace Helper
} // namespace SPTAG

#endif
