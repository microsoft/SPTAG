// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SOCKET_SIMPLESERIALIZATION_H_
#define _SPTAG_SOCKET_SIMPLESERIALIZATION_H_

#include "inc/Core/CommonDataStructure.h"

#include <type_traits>
#include <cstdint>
#include <memory>
#include <cstring>

namespace SPTAG
{
namespace Socket
{
namespace SimpleSerialization
{

    template<typename T>
    inline std::uint8_t*
    SimpleWriteBuffer(const T& p_val, std::uint8_t* p_buffer)
    {
        static_assert(std::is_fundamental<T>::value || std::is_enum<T>::value,
                      "Only applied for fundamental type.");

        *(reinterpret_cast<T*>(p_buffer)) = p_val;
        return p_buffer + sizeof(T);
    }


    template<typename T>
    inline const std::uint8_t*
    SimpleReadBuffer(const std::uint8_t* p_buffer, T& p_val)
    {
        static_assert(std::is_fundamental<T>::value || std::is_enum<T>::value,
                      "Only applied for fundamental type.");

        p_val = *(reinterpret_cast<const T*>(p_buffer));
        return p_buffer + sizeof(T);
    }


    template<typename T>
    inline std::size_t
    EstimateBufferSize(const T& p_val)
    {
        static_assert(std::is_fundamental<T>::value || std::is_enum<T>::value,
                      "Only applied for fundamental type.");

        return sizeof(T);
    }


    template<>
    inline std::uint8_t*
    SimpleWriteBuffer<std::string>(const std::string& p_val, std::uint8_t* p_buffer)
    {
        p_buffer = SimpleWriteBuffer(static_cast<std::uint32_t>(p_val.size()), p_buffer);

        std::memcpy(p_buffer, p_val.c_str(), p_val.size());
        return p_buffer + p_val.size();
    }


    template<>
    inline const std::uint8_t*
    SimpleReadBuffer<std::string>(const std::uint8_t* p_buffer, std::string& p_val)
    {
        p_val.clear();
        std::uint32_t len = 0;
        p_buffer = SimpleReadBuffer(p_buffer, len);

        if (len > 0)
        {
            p_val.reserve(len);
            p_val.assign(reinterpret_cast<const char*>(p_buffer), len);
        }

        return p_buffer + len;
    }


    /// Bounds-checked variants of SimpleReadBuffer.
    /// All return nullptr if a read would overrun [p_buffer, p_bufEnd).
    /// p_buffer is also returned as nullptr (and p_val left unchanged) if it is already nullptr.
    template<typename T>
    inline const std::uint8_t*
    SafeSimpleReadBuffer(const std::uint8_t* p_buffer, const std::uint8_t* p_bufEnd, T& p_val)
    {
        static_assert(std::is_fundamental<T>::value || std::is_enum<T>::value,
                      "Only applied for fundamental type.");

        if (p_buffer == nullptr) return nullptr;
        if (p_bufEnd != nullptr && static_cast<std::size_t>(p_bufEnd - p_buffer) < sizeof(T)) return nullptr;
        p_val = *(reinterpret_cast<const T*>(p_buffer));
        return p_buffer + sizeof(T);
    }


    inline const std::uint8_t*
    SafeSimpleReadBuffer(const std::uint8_t* p_buffer, const std::uint8_t* p_bufEnd, std::string& p_val)
    {
        p_val.clear();
        if (p_buffer == nullptr) return nullptr;
        std::uint32_t len = 0;
        p_buffer = SafeSimpleReadBuffer(p_buffer, p_bufEnd, len);
        if (p_buffer == nullptr) return nullptr;
        if (len > 0)
        {
            if (p_bufEnd != nullptr && static_cast<std::size_t>(p_bufEnd - p_buffer) < len) return nullptr;
            p_val.assign(reinterpret_cast<const char*>(p_buffer), len);
        }
        return p_buffer + len;
    }


    inline const std::uint8_t*
    SafeSimpleReadBuffer(const std::uint8_t* p_buffer, const std::uint8_t* p_bufEnd, ByteArray& p_val)
    {
        p_val.Clear();
        if (p_buffer == nullptr) return nullptr;
        std::uint32_t len = 0;
        p_buffer = SafeSimpleReadBuffer(p_buffer, p_bufEnd, len);
        if (p_buffer == nullptr) return nullptr;
        if (len > 0)
        {
            if (p_bufEnd != nullptr && static_cast<std::size_t>(p_bufEnd - p_buffer) < len) return nullptr;
            p_val = ByteArray::Alloc(len);
            std::memcpy(p_val.Data(), p_buffer, len);
        }
        return p_buffer + len;
    }


    template<>
    inline std::size_t
    EstimateBufferSize<std::string>(const std::string& p_val)
    {
        return sizeof(std::uint32_t) + p_val.size();
    }


    template<>
    inline std::uint8_t*
    SimpleWriteBuffer<ByteArray>(const ByteArray& p_val, std::uint8_t* p_buffer)
    {
        p_buffer = SimpleWriteBuffer(static_cast<std::uint32_t>(p_val.Length()), p_buffer);

        std::memcpy(p_buffer, p_val.Data(), p_val.Length());
        return p_buffer + p_val.Length();
    }


    template<>
    inline const std::uint8_t*
    SimpleReadBuffer<ByteArray>(const std::uint8_t* p_buffer, ByteArray& p_val)
    {
        p_val.Clear();
        std::uint32_t len = 0;
        p_buffer = SimpleReadBuffer(p_buffer, len);

        if (len > 0)
        {
            p_val = ByteArray::Alloc(len);
            std::memcpy(p_val.Data(), p_buffer, len);
        }

        return p_buffer + len;
    }


    template<>
    inline std::size_t
    EstimateBufferSize<ByteArray>(const ByteArray& p_val)
    {
        return sizeof(std::uint32_t) + p_val.Length();
    }


    template<typename T>
    inline std::uint8_t*
    SimpleWriteSharedPtrBuffer(const std::shared_ptr<T>& p_val, std::uint8_t* p_buffer)
    {
        if (nullptr == p_val)
        {
            return SimpleWriteBuffer(false, p_buffer);
        }

        p_buffer = SimpleWriteBuffer(true, p_buffer);
        p_buffer = SimpleWriteBuffer(*p_val, p_buffer);
        return p_buffer;
    }


    template<typename T>
    inline const std::uint8_t*
    SimpleReadSharedPtrBuffer(const std::uint8_t* p_buffer, std::shared_ptr<T>& p_val)
    {
        p_val.reset();
        bool isNotNull = false;
        p_buffer = SimpleReadBuffer(p_buffer, isNotNull);

        if (isNotNull)
        {
            p_val.reset(new T);
            p_buffer = SimpleReadBuffer(p_buffer, *p_val);
        }

        return p_buffer;
    }


    template<typename T>
    inline std::size_t
    EstimateSharedPtrBufferSize(const std::shared_ptr<T>& p_val)
    {
        return sizeof(bool) + (nullptr == p_val ? 0 : EstimateBufferSize(*p_val));
    }

} // namespace SimpleSerialization
} // namespace SPTAG
} // namespace Socket

#endif // _SPTAG_SOCKET_SIMPLESERIALIZATION_H_
