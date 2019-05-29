// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/CommonDataStructure.h"

namespace SPTAG
{ 

const ByteArray ByteArray::c_empty;

ByteArray::ByteArray() noexcept
    : m_data(nullptr),
      m_length(0)
{
}

ByteArray::ByteArray(ByteArray&& p_right) noexcept
    : m_data(p_right.m_data),
      m_length(p_right.m_length),
      m_dataHolder(std::move(p_right.m_dataHolder))
{
    p_right.m_data = nullptr;
    p_right.m_length = 0;
}

ByteArray::ByteArray(std::uint8_t* p_array, std::size_t p_length, bool p_transferOnwership)
    : m_data(p_array),
      m_length(p_length)
{
    if (p_transferOnwership)
    {
        m_dataHolder.reset(m_data, std::default_delete<std::uint8_t[]>());
    }
}

ByteArray::ByteArray(std::uint8_t* p_array, std::size_t p_length, std::shared_ptr<std::uint8_t> p_dataHolder) noexcept
    : m_data(p_array),
      m_length(p_length),
      m_dataHolder(std::move(p_dataHolder))
{
}

ByteArray::ByteArray(const ByteArray& p_right) noexcept
    : m_data(p_right.m_data),
      m_length(p_right.m_length),
      m_dataHolder(p_right.m_dataHolder)
{
}

ByteArray&
ByteArray::operator=(const ByteArray& p_right) noexcept
{
    if (this != std::addressof(p_right))
    {
        m_data = p_right.m_data;
        m_length = p_right.m_length;
        m_dataHolder = p_right.m_dataHolder;
    }

    return *this;
}

ByteArray&
ByteArray::operator=(ByteArray&& p_right) noexcept
{
    if (this != std::addressof(p_right))
    {
        m_data = p_right.m_data;
        m_length = p_right.m_length;
        m_dataHolder = std::move(p_right.m_dataHolder);
    }

    return *this;
}

ByteArray
ByteArray::Alloc(std::size_t p_length)
{
    if (0 == p_length)
    {
        return ByteArray();
    }
    else {
        auto array = new std::uint8_t[p_length];
        return ByteArray(array, p_length, true);
    }
}

void
ByteArray::SetData(std::uint8_t* p_array, std::size_t p_length) noexcept
{
    m_data = p_array;
    m_length = p_length;
    m_dataHolder.reset();
}

void
ByteArray::Clear() noexcept
{
    m_data = nullptr;
    m_dataHolder.reset();
    m_length = 0;
}
}
