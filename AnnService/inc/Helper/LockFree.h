// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_HELPER_LOCKFREE_H_
#define _SPTAG_HELPER_LOCKFREE_H_

#include <cstdint>
#include <vector>
#include "DiskIO.h"

namespace SPTAG
{
    namespace Helper
    {
        namespace LockFree
        {
            template <typename T>
            class LockFreeVector 
            {
            private:
                std::uint64_t size = 0;
                std::uint64_t blockSize;
                std::vector<T*> blocks;
                

            public:
                LockFreeVector() {}

                ~LockFreeVector() 
                {
                    for (T* ptr : blocks) delete ptr;
                    blocks.clear();
                }

                void Initialize(std::uint64_t blocksize, std::uint64_t maxsize) 
                {
                    size = 0;
                    blockSize = blocksize;
                    blocks.reserve((maxsize + blocksize - 1) / blocksize);
                }

                bool Append(const T* data, size_t length)
                {
                    if (size + length > blockSize * blocks.capacity()) return false;

                    std::uint64_t written = 0;
                    while (written < length) {
                        auto currBlock = (size + written) / blockSize;
                        if (currBlock >= blocks.size()) {
                            T* newBlock = new T[blockSize];
                            if (newBlock == nullptr) return false;
                            blocks.push_back(newBlock);
                        }

                        auto curBlockPos = (size + written) % blockSize;
                        auto toWrite = min(blockSize - curBlockPos, length - written);
                        std::memcpy(blocks[currBlock] + curBlockPos, data + written, toWrite * sizeof(T));
                        written += toWrite;
                    }
                    size += written;
                    return true;
                }

                bool Append(const T data)
                {
                    return Append(&data, 1);
                }

                void Clear() { size = 0; }
                std::uint64_t Size() const { return size; }

                T operator[](std::uint64_t offset) const { return *(blocks[offset / blockSize] + offset % blockSize); }

                ByteArray At(std::uint64_t offset, size_t length) const
                {
                    ByteArray b = ByteArray::Alloc(length * sizeof(T));
                    std::uint64_t copy = 0;
                    while (copy < length) {
                        auto blockOffset = (offset + copy) % blockSize;
                        auto toCopy = min(blockSize - blockOffset, length - copy);
                        std::memcpy(b.Data() + copy * sizeof(T), blocks[(offset + copy) / blockSize] + blockOffset, toCopy * sizeof(T));
                        copy += toCopy;
                    }
                    return b;
                }

                bool Save(std::shared_ptr<Helper::DiskPriorityIO> out)
                {
                    auto blockNum = size / blockSize;
                    for (int i = 0; i < blockNum; i++)
                        if (out->WriteBinary(sizeof(T) * blockSize, (char*)blocks[i]) != sizeof(T) * blockSize) return false;

                    auto remainNum = size % blockSize;
                    if (remainNum > 0 && out->WriteBinary(sizeof(T) * remainNum, (char*)blocks[blockNum]) != sizeof(T) * remainNum) return false;
                    return true;
                }

                bool Load(std::shared_ptr<Helper::DiskPriorityIO> in, size_t length)
                {
                    if (size + length > blockSize * blocks.capacity()) return false;

                    std::uint64_t written = 0;
                    while (written < length) {
                        auto currBlock = (size + written) / blockSize;
                        if (currBlock >= blocks.size()) {
                            T* newBlock = new T[blockSize];
                            if (newBlock == nullptr) return false;
                            blocks.push_back(newBlock);
                        }

                        auto curBlockPos = (size + written) % blockSize;
                        auto toWrite = min(blockSize - curBlockPos, length - written);
                        if (in->ReadBinary(sizeof(T) * toWrite, (char*)blocks[currBlock] + curBlockPos) != sizeof(T) * toWrite) return false;
                        written += toWrite;
                    }
                    size += written;
                    return true;
                }
            };
        }
    }
}
#endif // _SPTAG_HELPER_LOCKFREE_H_