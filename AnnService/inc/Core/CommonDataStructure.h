// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMONDATASTRUCTURE_H_
#define _SPTAG_COMMONDATASTRUCTURE_H_

#include "Common.h"

namespace SPTAG
{

class ByteArray
{
public:
    const static ByteArray c_empty;

    static ByteArray Alloc(std::size_t p_length);

    ByteArray() noexcept;

    ByteArray(ByteArray&& p_right) noexcept;

    ByteArray(std::uint8_t* p_array, std::size_t p_length, bool p_transferOnwership);

    ByteArray(std::uint8_t* p_array, std::size_t p_length, std::shared_ptr<std::uint8_t> p_dataHolder) noexcept;

    ByteArray(const ByteArray& p_right) noexcept;

    ByteArray& operator=(const ByteArray& p_right) noexcept;

    ByteArray& operator=(ByteArray&& p_right) noexcept;

    std::uint8_t* Data() const noexcept 
    {
        return m_data;
    }

    std::size_t Length() const noexcept
    {
        return m_length;
    }

    std::shared_ptr<std::uint8_t> DataHolder() const noexcept
    {
        return m_dataHolder;
    }

    void SetData(std::uint8_t* p_array, std::size_t p_length) noexcept;

    void Clear() noexcept;

private:
    std::uint8_t* m_data;

    std::size_t m_length;

    // Notice this is holding an array. Set correct deleter for this.
    std::shared_ptr<std::uint8_t> m_dataHolder;
};

} // namespace SPTAG

#endif // _SPTAG_COMMONDATASTRUCTURE_H_
