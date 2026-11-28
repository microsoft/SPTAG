// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ResidentMemory.h"
#include <array>
#include <string>
#include <string_view>

namespace SPTAG
{
namespace COMMON
{
// Declare before the strings it owns, so their storage is destroyed first.
// Managed internal writes must go through Assign; public unmanaged Options
// fields remain std::string for source compatibility.
template <size_t Count> class ResidentStringOwners
{
    struct Entry
    {
        std::string *target = nullptr;
        ResidentMemory::Allocation charge;
    };
    std::shared_ptr<ResidentMemory::Account> m_account;
    std::array<Entry, Count> m_entries;

  public:
    explicit ResidentStringOwners(std::shared_ptr<ResidentMemory::Account> account = ResidentMemory::CurrentAccount())
        : m_account(std::move(account))
    {
    }
    ResidentStringOwners(const ResidentStringOwners &) = delete;
    ResidentStringOwners &operator=(const ResidentStringOwners &) = delete;
    void Bind(std::shared_ptr<ResidentMemory::Account> account)
    {
        if (!m_account)
            m_account = std::move(account);
    }
    void Assign(std::string &target, std::string staging)
    {
        if (!m_account)
        {
            target.swap(staging);
            return;
        }
        Entry *entry = nullptr;
        for (auto &candidate : m_entries)
            if (candidate.target == &target)
            {
                if (target == staging)
                    return;
                entry = &candidate;
                break;
            }
        if (!entry)
            for (auto &candidate : m_entries)
                if (!candidate.target)
                {
                    entry = &candidate;
                    break;
                }
        if (!entry)
            throw std::logic_error("Resident string ownership capacity exceeded");
        const auto address = reinterpret_cast<uintptr_t>(staging.data());
        const auto object = reinterpret_cast<uintptr_t>(&staging);
        const bool inObject = address >= object && address - object < sizeof(staging);
        auto charge = ResidentMemory::Charge(m_account, inObject ? 0 : staging.capacity() + 1);
        target.swap(staging);
        auto old = std::move(entry->charge);
        entry->target = &target;
        entry->charge = std::move(charge);
        // The old backing must really be freed before releasing its charge.
        std::string().swap(staging);
        old.Reset();
    }
};

using ResidentString = std::basic_string<char, std::char_traits<char>, ResidentAllocator<char>>;
struct ResidentStringHash
{
    size_t operator()(const ResidentString &value) const noexcept
    {
        return std::hash<std::string_view>()(std::string_view(value.data(), value.size()));
    }
};
inline std::string UnmanagedString(const ResidentString &value)
{
    return std::string(value.data(), value.size());
}

// Keep the standard array allocator and public unique_ptr constructor APIs.
// Declare before the corresponding unique_ptr; replacements free old storage
// before releasing the old token.
class ResidentArrayOwner
{
    std::shared_ptr<ResidentMemory::Account> m_account = ResidentMemory::CurrentAccount();
    ResidentMemory::Allocation m_charge;

  public:
    template <typename T, typename Initialize>
    void Allocate(std::unique_ptr<T[]> &target, size_t count, Initialize initialize)
    {
        auto charge = ResidentMemory::Charge(m_account, ResidentMultiplyBytes(count, sizeof(T)));
        auto staging = std::make_unique<std::remove_const_t<T>[]>(count);
        initialize(staging.get());
        target = std::move(staging);
        m_charge = std::move(charge);
    }
    template <typename T> void Allocate(std::unique_ptr<T[]> &target, size_t count)
    {
        Allocate(target, count, [](auto *) {});
    }
    template <typename T> void Adopt(std::unique_ptr<T[]> &target, std::unique_ptr<T[]> staging, size_t count)
    {
        auto charge = ResidentMemory::Charge(m_account, ResidentMultiplyBytes(count, sizeof(T)));
        target.swap(staging);
        staging.reset();
        m_charge = std::move(charge);
    }
};
} // namespace COMMON
} // namespace SPTAG
