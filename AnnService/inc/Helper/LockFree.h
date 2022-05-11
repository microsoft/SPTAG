// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_HELPER_LOCKFREE_H_
#define _SPTAG_HELPER_LOCKFREE_H_

#include <cstdint>
#include <vector>
#include "DiskIO.h"
#include "Concurrent.h"

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
                std::uint64_t m_size = 0;
                std::uint64_t m_maxSize;
                std::uint64_t m_blockSize;
                int m_blockSizeEx;
                std::vector<T*> m_blocks;
                Concurrent::SpinLock m_lock;


            public:
                LockFreeVector() {}

                ~LockFreeVector() 
                {
                    for (T* ptr : m_blocks) delete ptr;
                    m_blocks.clear();
                }

                void reserve(std::uint64_t blocksize, std::uint64_t maxsize = MaxSize) 
                {
                    m_size = 0;
                    m_maxSize = maxsize;
                    m_blockSizeEx = static_cast<int>(ceil(log2(blocksize)));
                    m_blockSize = (1 << m_blockSizeEx) - 1;
                    m_blocks.reserve((maxsize + m_blockSize) >> m_blockSizeEx);
                }

                bool assign(const T* begin, const T* end)
                {
                    size_t length = end - begin;
                    Concurrent::LockGuard<Concurrent::SpinLock> guard(m_lock);
                    if (m_size > m_maxSize - length) return false;

                    std::uint64_t written = 0;
                    while (written < length) {
                        std::uint64_t currBlock = ((m_size + written) >> m_blockSizeEx);
                        if (currBlock >= m_blocks.size()) {
                            T* newBlock = new T[m_blockSize + 1];
                            if (newBlock == nullptr) return false;
                            m_blocks.push_back(newBlock);
                        }

                        auto curBlockPos = ((m_size + written) & m_blockSize);
                        auto toWrite = min(m_blockSize + 1 - curBlockPos, length - written);
                        std::memcpy(m_blocks[currBlock] + curBlockPos, begin + written, toWrite * sizeof(T));
                        written += toWrite;
                    }
                    m_size += written;
                    return true;
                }

                bool push_back(const T data)
                {
                    Concurrent::LockGuard<Concurrent::SpinLock> guard(m_lock);
                    if (m_size > m_maxSize - 1) return false;

                    std::uint64_t currBlock = (m_size >> m_blockSizeEx);
                    if (currBlock >= m_blocks.size()) {
                        T* newBlock = new T[m_blockSize + 1];
                        if (newBlock == nullptr) return false;
                        m_blocks.push_back(newBlock);
                    }

                    *(m_blocks[currBlock] + (m_size & m_blockSize)) = data;
                    m_size++;
                    return true;
                }

                inline const T& back() const { auto idx = m_size - 1; return *(m_blocks[idx >> m_blockSizeEx] + (idx & m_blockSize)); }
                inline void clear() { m_size = 0; }
                inline std::uint64_t size() const { return m_size; }

                inline const T& operator[](std::uint64_t offset) const { return *(m_blocks[offset >> m_blockSizeEx] + (offset & m_blockSize)); }

                ByteArray copy(std::uint64_t offset, size_t length) const
                {
                    ByteArray b = ByteArray::Alloc(length * sizeof(T));
                    std::uint64_t copy = 0;
                    while (copy < length) {
                        auto blockOffset = ((offset + copy) & m_blockSize);
                        auto toCopy = min(m_blockSize + 1 - blockOffset, length - copy);
                        std::memcpy(b.Data() + copy * sizeof(T), m_blocks[(offset + copy) >> m_blockSizeEx] + blockOffset, toCopy * sizeof(T));
                        copy += toCopy;
                    }
                    return b;
                }

                bool save(std::shared_ptr<Helper::DiskIO> out)
                {
                    auto blockNum = (m_size >> m_blockSizeEx);
                    for (int i = 0; i < blockNum; i++)
                        if (out->WriteBinary(sizeof(T) * (m_blockSize + 1), (char*)(m_blocks[i])) != sizeof(T) * (m_blockSize + 1)) return false;

                    auto remainNum = (m_size & m_blockSize);
                    if (remainNum > 0 && out->WriteBinary(sizeof(T) * remainNum, (char*)(m_blocks[blockNum])) != sizeof(T) * remainNum) return false;
                    return true;
                }

                bool load(std::shared_ptr<Helper::DiskIO> in, size_t length)
                {
                    if (m_size + length > (m_blockSize + 1) * m_blocks.capacity()) return false;

                    std::uint64_t written = 0;
                    while (written < length) {
                        auto currBlock = ((m_size + written) >> m_blockSizeEx);
                        if (currBlock >= m_blocks.size()) {
                            T* newBlock = new T[m_blockSize + 1];
                            if (newBlock == nullptr) return false;
                            m_blocks.push_back(newBlock);
                        }

                        auto curBlockPos = ((m_size + written) & m_blockSize);
                        auto toWrite = min(m_blockSize + 1 - curBlockPos, length - written);
                        if (in->ReadBinary(sizeof(T) * toWrite, (char*)(m_blocks[currBlock] + curBlockPos)) != sizeof(T) * toWrite) return false;
                        written += toWrite;
                    }
                    m_size += written;
                    return true;
                }
            };
        }
    }
}
#endif // _SPTAG_HELPER_LOCKFREE_H_