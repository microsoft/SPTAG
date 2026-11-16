// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace SPTAG::SPANN
{
// The publisher withdraws pointers before Advance(), then waits without holding
// locks needed by readers. Guards must precede all reads of published pointers.
class ReadEpoch
{
    struct alignas(64) Record
    {
        std::atomic<uint64_t> epoch{0};
        std::atomic<const void *> owner{nullptr};
        std::atomic<bool> retired{false};
        size_t depth = 0;
    };
    inline static std::atomic<uint64_t> s_nextID{1};
    inline static thread_local std::unordered_map<uint64_t, std::shared_ptr<Record>> s_local;
    const uint64_t m_id = s_nextID.fetch_add(1, std::memory_order_relaxed);
    std::atomic<uint64_t> m_epoch{1};
    mutable std::mutex m_registration;
    std::vector<std::weak_ptr<Record>> m_records;

    Record &Local()
    {
        auto found = s_local.find(m_id);
        if (found != s_local.end())
            return *found->second;
        for (auto it = s_local.begin(); it != s_local.end();)
            if (!it->second->depth && it->second->retired.load(std::memory_order_acquire))
                it = s_local.erase(it);
            else
                ++it;
        if (s_local.size() >= 64)
            for (auto it = s_local.begin(); it != s_local.end(); ++it)
                if (!it->second->depth)
                {
                    s_local.erase(it);
                    break;
                }
        auto record = std::make_shared<Record>();
        {
            std::lock_guard<std::mutex> lock(m_registration);
            m_records.erase(
                std::remove_if(m_records.begin(), m_records.end(), [](const auto &entry) { return entry.expired(); }),
                m_records.end());
            m_records.emplace_back(record);
        }
        return *s_local.emplace(m_id, std::move(record)).first->second;
    }

  public:
    class Guard
    {
        Record &m_record;
        const void *m_previousOwner;

      public:
        explicit Guard(ReadEpoch &domain)
            : m_record(domain.Local()), m_previousOwner(m_record.owner.load(std::memory_order_relaxed))
        {
            if (m_record.depth++ == 0)
            {
                uint64_t epoch;
                do
                {
                    epoch = domain.m_epoch.load();
                    m_record.epoch.store(epoch);
                } while (domain.m_epoch.load() != epoch);
            }
        }
        Guard(const Guard &) = delete;
        Guard &operator=(const Guard &) = delete;
        void SetOwner(const void *owner)
        {
            m_record.owner.store(owner);
        }
        ~Guard()
        {
            m_record.owner.store(m_previousOwner);
            if (--m_record.depth == 0)
                m_record.epoch.store(0);
        }
    };

    ~ReadEpoch()
    {
        for (auto &entry : m_records)
            if (auto record = entry.lock())
                record->retired.store(true, std::memory_order_release);
    }

    uint64_t Advance()
    {
        return m_epoch.fetch_add(1);
    }

    void Wait(uint64_t retired)
    {
        std::vector<std::shared_ptr<Record>> records;
        {
            std::lock_guard<std::mutex> lock(m_registration);
            for (auto &entry : m_records)
                if (auto record = entry.lock())
                    records.push_back(std::move(record));
        }
        for (const auto &record : records)
            for (;;)
            {
                auto epoch = record->epoch.load();
                if (!epoch || epoch > retired)
                    break;
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
    }

    void Synchronize()
    {
        Wait(Advance());
    }

    size_t Active(const void *owner) const
    {
        size_t count = 0;
        std::lock_guard<std::mutex> lock(m_registration);
        for (const auto &entry : m_records)
            if (auto record = entry.lock())
                if (record->epoch.load() && record->owner.load() == owner)
                    ++count;
        return count;
    }
};
} // namespace SPTAG::SPANN
