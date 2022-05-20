// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_CORE_COMMONDEFS_H_
#define _SPTAG_CORE_COMMONDEFS_H_

#include <cstdint>
#include <type_traits>
#include <memory>
#include <random>
#include <string>
#include <limits>
#include <vector>
#include <cmath>
#include "inc/Helper/Logging.h"
#include "inc/Helper/DiskIO.h"
#include <tuple>

#ifndef _MSC_VER
#include <stdio.h>
#include <unistd.h>
#include <dirent.h>
#include <cstring>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>

#if defined(__INTEL_COMPILER)
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif

#define FolderSep '/'

inline bool direxists(const char* path) {
    struct stat info;
    return stat(path, &info) == 0 && (info.st_mode & S_IFDIR);
}
inline bool fileexists(const char* path) {
    struct stat info;
    return stat(path, &info) == 0 && (info.st_mode & S_IFDIR) == 0;
}

template <class T>
inline T min(T a, T b) {
    return a < b ? a : b;
}
template <class T>
inline T max(T a, T b) {
    return a > b ? a : b;
}

#ifndef _rotl
#define _rotl(x, n) (((x) << (n)) | ((x) >> (32-(n))))
#endif

#define mkdir(a) mkdir(a, ACCESSPERMS)
#define InterlockedCompareExchange(a,b,c) __sync_val_compare_and_swap(a, c, b)
#define InterlockedExchange8(a,b) __sync_lock_test_and_set(a, b)
#define Sleep(a) usleep(a * 1000)
#define strtok_s(a, b, c) strtok_r(a, b, c)

#else

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif // !WIN32_LEAN_AND_MEAN

#include <Windows.h>
#include <Psapi.h>
#include <malloc.h>

#define FolderSep '\\'

inline bool direxists(const TCHAR* path) {
    auto dwAttr = GetFileAttributes(path);
    return (dwAttr != INVALID_FILE_ATTRIBUTES) && (dwAttr & FILE_ATTRIBUTE_DIRECTORY);
}
inline bool fileexists(const TCHAR* path) {
    auto dwAttr = GetFileAttributes(path);
    return (dwAttr != INVALID_FILE_ATTRIBUTES) && (dwAttr & FILE_ATTRIBUTE_DIRECTORY) == 0;
}
#define mkdir(a) CreateDirectory(a, NULL)
#endif

namespace SPTAG
{
#if (__cplusplus < 201703L)
#define ALIGN_ALLOC(size) _mm_malloc(size, 32)
#define ALIGN_FREE(ptr) _mm_free(ptr)
#define PAGE_ALLOC(size) _mm_malloc(size, 512)
#define PAGE_FREE(ptr) _mm_free(ptr)
#else
#define ALIGN_ALLOC(size) ::operator new(size, (std::align_val_t)32)
#define ALIGN_FREE(ptr) ::operator delete(ptr, (std::align_val_t)32)
#define PAGE_ALLOC(size) ::operator new(size, (std::align_val_t)512)
#define PAGE_FREE(ptr) ::operator delete(ptr, (std::align_val_t)512)
#endif

typedef std::int32_t SizeType;
typedef std::int32_t DimensionType;

const SizeType MaxSize = (std::numeric_limits<SizeType>::max)();
const float MinDist = (std::numeric_limits<float>::min)();
const float MaxDist = (std::numeric_limits<float>::max)() / 10;
const float Epsilon = 0.000001f;
const std::uint16_t PageSize = 4096;
const int PageSizeEx = 12;

extern std::mt19937 rg;

extern std::shared_ptr<Helper::DiskIO>(*f_createIO)();

#define IOBINARY(ptr, func, bytes, ...) if (ptr->func(bytes, __VA_ARGS__) != bytes) return ErrorCode::DiskIOFail
#define IOSTRING(ptr, func, ...) if (ptr->func(__VA_ARGS__) == 0) return ErrorCode::DiskIOFail

extern std::shared_ptr<Helper::Logger> g_pLogger;

#define LOG(l, ...) g_pLogger->Logging("SPTAG", l, __FILE__, __LINE__, __FUNCTION__, __VA_ARGS__)

class MyException : public std::exception 
{
private:
    std::string Exp;
public:
    MyException(std::string e) { Exp = e; }
#ifdef _MSC_VER
    const char* what() const { return Exp.c_str(); }
#else
    const char* what() const noexcept { return Exp.c_str(); }
#endif
};

enum class ErrorCode : std::uint16_t
{
#define DefineErrorCode(Name, Value) Name = Value,
#include "DefinitionList.h"
#undef DefineErrorCode

    Undefined
};
static_assert(static_cast<std::uint16_t>(ErrorCode::Undefined) != 0, "Empty ErrorCode!");


enum class DistCalcMethod : std::uint8_t
{
#define DefineDistCalcMethod(Name) Name,
#include "DefinitionList.h"
#undef DefineDistCalcMethod

    Undefined
};
static_assert(static_cast<std::uint8_t>(DistCalcMethod::Undefined) != 0, "Empty DistCalcMethod!");

enum class VectorValueType : std::uint8_t
{
#define DefineVectorValueType(Name, Type) Name,
#include "DefinitionList.h"
#undef DefineVectorValueType
    Undefined
};
static_assert(static_cast<std::uint8_t>(VectorValueType::Undefined) != 0, "Empty VectorValueType!");

// remove_last is by Vladimir Reshetnikov, https://stackoverflow.com/a/51805324
template<class Tuple>
struct remove_last;

template<>
struct remove_last<std::tuple<>>; // Define as you wish or leave undefined

template<class... Args>
struct remove_last<std::tuple<Args...>>
{
private:
    using Tuple = std::tuple<Args...>;

    template<std::size_t... n>
    static std::tuple<std::tuple_element_t<n, Tuple>...>
        extract(std::index_sequence<n...>);

public:
    using type = decltype(extract(std::make_index_sequence<sizeof...(Args) - 1>()));
};

template<class Tuple>
using remove_last_t = typename remove_last<Tuple>::type;

using VectorValueTypeTuple = remove_last_t<std::tuple<
#define DefineVectorValueType(Name, Type) Type,
#include "DefinitionList.h"
#undef DefineVectorValueType
void>>;

// Dispatcher is based on https://stackoverflow.com/a/34046180
template <typename T, typename F>
std::function<void()> call_with_default(F&& f)
{
    return [f]() {f(T{}); };
}

template <typename F, std::size_t...Is>
void VectorValueTypeDispatch(VectorValueType vectorType, F&& f, std::index_sequence<Is...>)
{
    std::function<void()> fs[] = {
        call_with_default<std::tuple_element_t<Is, VectorValueTypeTuple>>(f)...
    };
    fs[static_cast<int>(vectorType)]();
    
}

template <typename F>
void VectorValueTypeDispatch(VectorValueType vectorType, F f)
{
    constexpr auto VectorCount = std::tuple_size<VectorValueTypeTuple>::value;
    if ((int)vectorType < VectorCount)
    {
        VectorValueTypeDispatch(vectorType, f, std::make_index_sequence<VectorCount>{});
    }
    else
    {
        throw std::exception();
    }
}

enum class IndexAlgoType : std::uint8_t
{
#define DefineIndexAlgo(Name) Name,
#include "DefinitionList.h"
#undef DefineIndexAlgo

    Undefined
};
static_assert(static_cast<std::uint8_t>(IndexAlgoType::Undefined) != 0, "Empty IndexAlgoType!");

enum class VectorFileType : std::uint8_t
{
#define DefineVectorFileType(Name) Name,
#include "DefinitionList.h"
#undef DefineVectorFileType

    Undefined
};
static_assert(static_cast<std::uint8_t>(VectorFileType::Undefined) != 0, "Empty VectorFileType!");

enum class TruthFileType : std::uint8_t
{
#define DefineTruthFileType(Name) Name,
#include "DefinitionList.h"
#undef DefineTruthFileType

    Undefined
};
static_assert(static_cast<std::uint8_t>(TruthFileType::Undefined) != 0, "Empty TruthFileType!");

template<typename T>
constexpr VectorValueType GetEnumValueType()
{
    return VectorValueType::Undefined;
}


#define DefineVectorValueType(Name, Type) \
template<> \
constexpr VectorValueType GetEnumValueType<Type>() \
{ \
    return VectorValueType::Name; \
} \

#include "DefinitionList.h"
#undef DefineVectorValueType


inline std::size_t GetValueTypeSize(VectorValueType p_valueType)
{
    std::size_t out = 0;
    VectorValueTypeDispatch(p_valueType, [&](auto t) { out = sizeof(decltype(t)); });

    return out;
}

enum class QuantizerType : std::uint8_t
{
#define DefineQuantizerType(Name, Type) Name,
#include "DefinitionList.h"
#undef DefineQuantizerType

    Undefined
};
static_assert(static_cast<std::uint8_t>(QuantizerType::Undefined) != 0, "Empty QuantizerType!");

} // namespace SPTAG

#endif // _SPTAG_CORE_COMMONDEFS_H_
