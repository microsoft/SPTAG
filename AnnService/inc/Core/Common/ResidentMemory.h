// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace SPTAG
{
namespace COMMON
{
// Allocation-capacity accounting, not an allocator or an RSS measurement.
class ResidentMemory
{
  private:
    struct Counters
    {
        uint64_t allocated = 0;
        uint64_t reserved = 0;
        uint64_t peak = 0;
    };

    struct State
    {
        explicit State(uint64_t bytes) : capacity(bytes)
        {
            if (!bytes)
                throw std::invalid_argument("Resident memory capacity must be positive");
        }
        std::mutex mutex;
        std::condition_variable changed;
        const uint64_t capacity;
        Counters counters;
        uint64_t waits = 0;
        uint64_t timeouts = 0;
        uint64_t oversized = 0;
        uint64_t version = 0;
    };

  public:
    struct Stats
    {
        uint64_t allocatedBytes = 0;
        uint64_t reservedBytes = 0;
        uint64_t peakBytes = 0;
        uint64_t capacityBytes = 0;
        uint64_t waits = 0;
        uint64_t timeouts = 0;
        uint64_t oversized = 0;
    };

    enum class Admission
    {
        Success,
        Oversized,
        TimedOut
    };

    class Account;
    class Scope;

  private:
    struct Credit
    {
        explicit Credit(std::shared_ptr<Account> owner) : account(std::move(owner))
        {
        }
        std::shared_ptr<Account> account;
        uint64_t remaining = 0;
        bool open = true;
    };

    struct Lifetime
    {
        explicit Lifetime(std::shared_ptr<Credit> value) : credit(std::move(value))
        {
        }
        ~Lifetime();
        std::shared_ptr<Credit> credit;
    };

  public:
    class Allocation
    {
      public:
        Allocation() = default;
        Allocation(const Allocation &) = delete;
        Allocation &operator=(const Allocation &) = delete;
        Allocation(Allocation &&other) noexcept
        {
            Swap(other);
        }
        Allocation &operator=(Allocation &&other) noexcept
        {
            if (this != &other)
            {
                Reset();
                Swap(other);
            }
            return *this;
        }
        ~Allocation()
        {
            Reset();
        }
        uint64_t Bytes() const noexcept
        {
            return m_bytes;
        }
        // The backing owner must free its memory before calling Reset.
        void Reset() noexcept;

      private:
        friend class Account;
        void Swap(Allocation &other) noexcept
        {
            m_account.swap(other.m_account);
            m_credit.swap(other.m_credit);
            std::swap(m_bytes, other.m_bytes);
        }
        std::shared_ptr<Account> m_account;
        std::shared_ptr<Credit> m_credit;
        uint64_t m_bytes = 0;
    };

    class Reservation
    {
      public:
        Reservation() = default;
        Reservation(const Reservation &) = delete;
        Reservation &operator=(const Reservation &) = delete;
        Reservation(Reservation &&other) noexcept
            : m_credit(std::move(other.m_credit)), m_lifetime(std::move(other.m_lifetime))
        {
        }
        Reservation &operator=(Reservation &&other) noexcept
        {
            if (this != &other)
            {
                Reset();
                m_credit = std::move(other.m_credit);
                m_lifetime = std::move(other.m_lifetime);
            }
            return *this;
        }
        ~Reservation()
        {
            Reset();
        }
        void Reset() noexcept;

      private:
        friend class Account;
        friend class Scope;
        std::shared_ptr<Credit> m_credit;
        std::shared_ptr<Lifetime> m_lifetime;
    };

    class Account : public std::enable_shared_from_this<Account>
    {
      public:
        Admission Reserve(uint64_t bytes, std::chrono::milliseconds timeout, Reservation &result)
        {
            if (result.m_credit)
                throw std::logic_error("Reservation output must be empty");
            auto credit = std::make_shared<Credit>(shared_from_this());
            auto lifetime = std::make_shared<Lifetime>(credit);
            std::unique_lock<std::mutex> lock(m_state->mutex);
            if (bytes > m_state->capacity)
            {
                ++m_state->oversized;
                return Admission::Oversized;
            }
            const auto fits = [&] {
                return bytes <= m_state->capacity - m_state->counters.allocated - m_state->counters.reserved;
            };
            if (!fits())
            {
                if (timeout.count() == 0)
                    return Admission::TimedOut;
                ++m_state->waits;
                if (!m_state->changed.wait_for(lock, timeout, fits))
                {
                    ++m_state->timeouts;
                    return Admission::TimedOut;
                }
            }
            m_counters.reserved += bytes;
            m_state->counters.reserved += bytes;
            credit->remaining = bytes;
            Peak();
            result.m_credit = std::move(credit);
            result.m_lifetime = std::move(lifetime);
            return Admission::Success;
        }

        Allocation Allocate(uint64_t bytes, const std::shared_ptr<Credit> &credit = {})
        {
            Allocation result;
            auto self = shared_from_this();
            std::lock_guard<std::mutex> lock(m_state->mutex);
            const bool prepared = credit && credit->account.get() == this && credit->open;
            if (credit && !prepared)
                throw std::bad_alloc();
            if (prepared)
            {
                // A prepared operation cannot silently spend beyond its preflight.
                if (bytes > credit->remaining)
                    throw std::bad_alloc();
                credit->remaining -= bytes;
                m_counters.reserved -= bytes;
                m_state->counters.reserved -= bytes;
                result.m_credit = credit;
            }
            else if (bytes > m_state->capacity - m_state->counters.allocated - m_state->counters.reserved)
                throw std::bad_alloc();
            m_counters.allocated += bytes;
            m_state->counters.allocated += bytes;
            Peak();
            result.m_account = std::move(self);
            result.m_bytes = bytes;
            return result;
        }

        Stats GetStats() const
        {
            std::lock_guard<std::mutex> lock(m_state->mutex);
            return Snapshot(m_counters, *m_state);
        }

      private:
        friend class ResidentMemory;
        friend class Allocation;
        friend class Reservation;
        friend struct Lifetime;
        explicit Account(std::shared_ptr<State> state) : m_state(std::move(state))
        {
        }
        void Peak()
        {
            m_counters.peak = (std::max)(m_counters.peak, m_counters.allocated + m_counters.reserved);
            auto &global = m_state->counters;
            global.peak = (std::max)(global.peak, global.allocated + global.reserved);
        }
        std::shared_ptr<State> m_state;
        Counters m_counters;
    };

    // Explicit propagation is required on worker threads. Allocations retain the
    // account themselves; freeing on a different thread does not need a scope.
    class Scope
    {
      public:
        struct Context
        {
            std::shared_ptr<Account> account;
            std::shared_ptr<Credit> credit;
            std::shared_ptr<Lifetime> lifetime;
            std::function<Reservation(uint64_t)> reclaim;
        };

        explicit Scope(std::shared_ptr<Account> account, const Reservation *reservation = nullptr)
            : m_previous(std::move(Current()))
        {
            Current().account = std::move(account);
            Current().credit = reservation ? reservation->m_credit : nullptr;
            Current().lifetime = reservation ? reservation->m_lifetime : nullptr;
        }
        explicit Scope(Context context) : m_previous(std::move(Current()))
        {
            Current() = std::move(context);
        }
        Scope(const Scope &) = delete;
        Scope &operator=(const Scope &) = delete;
        ~Scope()
        {
            Current() = std::move(m_previous);
        }
        static Context Capture()
        {
            return Current();
        }

      private:
        friend class ResidentMemory;
        static Context &Current()
        {
            static thread_local Context context;
            return context;
        }
        Context m_previous;
    };

    explicit ResidentMemory(uint64_t capacity) : m_state(std::make_shared<State>(capacity))
    {
    }

    std::shared_ptr<Account> CreateAccount()
    {
        return std::shared_ptr<Account>(new Account(m_state));
    }

    Stats GetStats() const
    {
        std::lock_guard<std::mutex> lock(m_state->mutex);
        return Snapshot(m_state->counters, *m_state);
    }

    uint64_t Version() const
    {
        std::lock_guard<std::mutex> lock(m_state->mutex);
        return m_state->version;
    }

    void Notify()
    {
        {
            std::lock_guard<std::mutex> lock(m_state->mutex);
            ++m_state->version;
        }
        m_state->changed.notify_all();
    }

    void Wait(uint64_t version, std::chrono::microseconds timeout)
    {
        std::unique_lock<std::mutex> lock(m_state->mutex);
        m_state->changed.wait_for(lock, timeout, [&] { return m_state->version != version; });
    }

    static Allocation Charge(uint64_t bytes)
    {
        return Charge(Scope::Current().account, bytes);
    }

    static std::shared_ptr<Account> CurrentAccount()
    {
        return Scope::Current().account;
    }

    static Allocation Charge(const std::shared_ptr<Account> &account, uint64_t bytes)
    {
        if (!account)
            return {};
        auto &context = Scope::Current();
        try
        {
            return account->Allocate(bytes, context.account == account ? context.credit : nullptr);
        }
        catch (const std::bad_alloc &)
        {
            if (context.account != account || context.credit || !context.reclaim)
                throw;
        }
        auto reclaim = context.reclaim;
        auto reservation = reclaim(bytes);
        Scope prepared(account, &reservation);
        return account->Allocate(bytes, Scope::Current().credit);
    }

  private:
    static Stats Snapshot(const Counters &counters, const State &state)
    {
        return {counters.allocated, counters.reserved, counters.peak,  state.capacity,
                state.waits,        state.timeouts,    state.oversized};
    }
    std::shared_ptr<State> m_state;
};

inline void ResidentMemory::Allocation::Reset() noexcept
{
    if (!m_account)
        return;
    auto account = std::move(m_account);
    auto credit = std::move(m_credit);
    auto &state = *account->m_state;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        account->m_counters.allocated -= m_bytes;
        state.counters.allocated -= m_bytes;
        if (credit && credit->open)
        {
            // Reusable preflight headroom remains an explicit reservation until
            // its owner closes it, rather than being reported as live backing.
            credit->remaining += m_bytes;
            account->m_counters.reserved += m_bytes;
            state.counters.reserved += m_bytes;
        }
        ++state.version;
    }
    m_bytes = 0;
    state.changed.notify_all();
}

inline void ResidentMemory::Reservation::Reset() noexcept
{
    m_credit.reset();
    m_lifetime.reset();
}

inline ResidentMemory::Lifetime::~Lifetime()
{
    auto &account = *credit->account;
    auto &state = *account.m_state;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        credit->open = false;
        account.m_counters.reserved -= credit->remaining;
        state.counters.reserved -= credit->remaining;
        credit->remaining = 0;
        ++state.version;
    }
    state.changed.notify_all();
}

// An accounting adapter around the standard allocator. No pool, arena, global
// allocation override, or deferred physical free is introduced.
template <typename T> class ResidentAllocator
{
  private:
    struct Header
    {
        ResidentMemory::Allocation charge;
    };
    static constexpr size_t Alignment = (std::max)(alignof(T), alignof(Header));
    using Unit = typename std::aligned_storage<Alignment, Alignment>::type;
    static constexpr size_t Offset = (sizeof(Header) + Alignment - 1) / Alignment;

  public:
    using value_type = T;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    template <typename U> struct rebind
    {
        using other = ResidentAllocator<U>;
    };

    ResidentAllocator() noexcept : m_account(ResidentMemory::CurrentAccount())
    {
    }
    explicit ResidentAllocator(std::shared_ptr<ResidentMemory::Account> account) noexcept
        : m_account(std::move(account))
    {
    }
    template <typename U> ResidentAllocator(const ResidentAllocator<U> &other) noexcept : m_account(other.Account())
    {
    }

    static uint64_t AllocationBytes(size_t count)
    {
        if (count > ((std::numeric_limits<size_t>::max)() - sizeof(Unit) * Offset - Alignment + 1) / sizeof(T))
            throw std::bad_array_new_length();
        return (Offset + (count * sizeof(T) + Alignment - 1) / Alignment) * sizeof(Unit);
    }

    T *allocate(size_t count)
    {
        if (!m_account)
            return std::allocator<T>().allocate(count);
        if (count > ((std::numeric_limits<size_t>::max)() - sizeof(Unit) * Offset - Alignment + 1) / sizeof(T))
            throw std::bad_array_new_length();
        const auto units = Offset + (count * sizeof(T) + Alignment - 1) / Alignment;
        auto charge = ResidentMemory::Charge(m_account, units * sizeof(Unit));
        auto *storage = std::allocator<Unit>().allocate(units);
        new (storage) Header{std::move(charge)};
        return reinterpret_cast<T *>(storage + Offset);
    }

    void deallocate(T *pointer, size_t count) noexcept
    {
        if (!m_account)
        {
            std::allocator<T>().deallocate(pointer, count);
            return;
        }
        auto *storage = reinterpret_cast<Unit *>(pointer) - Offset;
        auto *header = reinterpret_cast<Header *>(storage);
        auto charge = std::move(header->charge);
        header->~Header();
        const auto units = Offset + (count * sizeof(T) + Alignment - 1) / Alignment;
        std::allocator<Unit>().deallocate(storage, units);
        charge.Reset();
    }

    const std::shared_ptr<ResidentMemory::Account> &Account() const noexcept
    {
        return m_account;
    }
    template <typename U> bool operator==(const ResidentAllocator<U> &other) const noexcept
    {
        return m_account == other.Account();
    }
    template <typename U> bool operator!=(const ResidentAllocator<U> &other) const noexcept
    {
        return !(*this == other);
    }

  private:
    std::shared_ptr<ResidentMemory::Account> m_account;
};

template <typename T> using ResidentVector = std::vector<T, ResidentAllocator<T>>;
template <typename K, typename V>
using ResidentMap = std::unordered_map<K, V, std::hash<K>, std::equal_to<K>, ResidentAllocator<std::pair<const K, V>>>;

inline uint64_t ResidentAddBytes(uint64_t a, uint64_t b)
{
    const auto maximum = (std::numeric_limits<uint64_t>::max)();
    return b > maximum - a ? maximum : a + b;
}

inline uint64_t ResidentMultiplyBytes(uint64_t a, uint64_t b)
{
    const auto maximum = (std::numeric_limits<uint64_t>::max)();
    return b && a > maximum / b ? maximum : a * b;
}

// Measure the actual allocate_shared control-block request with the same
// allocator type. This is cached per type, not a guessed STL metadata charge.
template <typename T> uint64_t ResidentSharedBytes()
{
    static const uint64_t bytes = [] {
        ResidentMemory meter((std::numeric_limits<uint64_t>::max)());
        auto account = meter.CreateAccount();
        ResidentMemory::Scope scope(account);
        auto value = std::allocate_shared<T>(ResidentAllocator<T>(account));
        return meter.GetStats().allocatedBytes;
    }();
    return bytes;
}

template <typename T> class ResidentArray
{
  public:
    explicit ResidentArray(size_t count,
                           std::shared_ptr<ResidentMemory::Account> account = ResidentMemory::CurrentAccount())
        : m_allocator(std::move(account)), m_count(count)
    {
        m_data = m_allocator.allocate(count);
        size_t initialized = 0;
        try
        {
            for (; initialized < count; ++initialized)
                new (m_data + initialized) T();
        }
        catch (...)
        {
            while (initialized)
                m_data[--initialized].~T();
            m_allocator.deallocate(m_data, count);
            throw;
        }
    }
    ResidentArray(const ResidentArray &) = delete;
    ResidentArray &operator=(const ResidentArray &) = delete;
    ~ResidentArray()
    {
        for (size_t i = m_count; i; --i)
            m_data[i - 1].~T();
        m_allocator.deallocate(m_data, m_count);
    }
    T &operator[](size_t index) const noexcept
    {
        return m_data[index];
    }

  private:
    ResidentAllocator<T> m_allocator;
    T *m_data = nullptr;
    size_t m_count;
};
} // namespace COMMON
} // namespace SPTAG
