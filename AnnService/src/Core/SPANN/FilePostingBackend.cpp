// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/SPANN/PostingBackend.h"

#ifdef MULTITENANT
#if defined(__linux__)
#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <condition_variable>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <linux/aio_abi.h>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace SPTAG::SPANN
{
namespace
{
namespace fs = std::filesystem;
constexpr uint64_t Alignment = 4096;
constexpr size_t BatchSize = 128;
constexpr char ImageName[] = "postings.image";
constexpr char ImageMagic[] = "SPTAGFP1";
constexpr uint64_t HashSeed = 14695981039346656037ULL;

struct Descriptor
{
    int fd = -1;
    Descriptor() = default;
    explicit Descriptor(int value) : fd(value)
    {
    }
    Descriptor(const Descriptor &) = delete;
    Descriptor &operator=(const Descriptor &) = delete;
    ~Descriptor()
    {
        if (fd >= 0)
            ::close(fd);
    }
    bool Close()
    {
        int value = fd;
        fd = -1;
        return value < 0 || ::close(value) == 0;
    }
};

std::string Number(uint64_t value)
{
    std::string result(8, '\0');
    for (int i = 7; i >= 0; --i, value >>= 8)
        result[i] = static_cast<char>(value);
    return result;
}

uint64_t Decode(const char *data)
{
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i)
        value = (value << 8) | static_cast<unsigned char>(data[i]);
    return value;
}

std::string Prefix(const std::string &tenant, uint64_t index)
{
    return Number(tenant.size()) + tenant + Number(index);
}

bool ValidKey(const std::string &key)
{
    return key.size() >= 24 && Decode(key.data()) == key.size() - 24 &&
           Decode(key.data() + key.size() - 8) <= static_cast<uint64_t>(std::numeric_limits<SizeType>::max());
}

bool AlignedSize(uint64_t size, uint64_t &aligned)
{
    if (size > static_cast<uint64_t>(std::numeric_limits<ssize_t>::max()) - Alignment)
        return false;
    aligned = (size + Alignment - 1) & ~(Alignment - 1);
    return true;
}

bool Transfer(int fd, void *buffer, uint64_t size, uint64_t offset, bool write)
{
    auto *data = static_cast<char *>(buffer);
    while (size)
    {
        // Linux caps individual reads/writes; this cap is itself page aligned.
        size_t length = static_cast<size_t>(std::min<uint64_t>(size, 0x7ffff000ULL));
        ssize_t done = write ? ::pwrite(fd, data, length, static_cast<off_t>(offset))
                             : ::pread(fd, data, length, static_cast<off_t>(offset));
        if (done < 0 && errno == EINTR)
            continue;
        if (done <= 0)
            return false;
        data += done;
        size -= done;
        offset += done;
    }
    return true;
}

bool Sync(int fd)
{
    int result;
    do
    {
        result = ::fsync(fd);
    } while (result < 0 && errno == EINTR);
    return result == 0;
}

bool SyncDirectory(const fs::path &path)
{
    Descriptor directory(::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
    if (directory.fd < 0 || !Sync(directory.fd))
        return false;
    return directory.Close();
}

uint64_t Hash(uint64_t hash, const void *buffer, size_t size)
{
    auto *data = static_cast<const unsigned char *>(buffer);
    for (size_t i = 0; i < size; ++i)
        hash = (hash ^ data[i]) * 1099511628211ULL;
    return hash;
}

struct Entry
{
    uint64_t offset = 0;
    uint64_t size = 0;
};

// One exclusive lease owns each context. No request survives ReadBatch,
// including failed submissions/completion retrieval. io_destroy synchronously
// cancels/waits before caller-owned buffers can disappear.
class AioContext
{
  public:
    ~AioContext()
    {
        Reset();
    }
    void Reset()
    {
        if (!m_context)
            return;
        long result;
        do
        {
            result = ::syscall(SYS_io_destroy, m_context);
        } while (result < 0 && errno == EINTR);
        m_context = 0;
    }

    ErrorCode ReadBatch(int fd, const std::vector<Entry> &entries, std::vector<Helper::PageBuffer<uint8_t>> &values)
    {
        bool haveReads = false;
        for (const auto &entry : entries)
            haveReads = haveReads || entry.size != 0;
        if (!haveReads)
            return ErrorCode::Success;
        if (!m_context && ::syscall(SYS_io_setup, BatchSize, &m_context) < 0)
            return ErrorCode::DiskIOFail;
        size_t next = 0;
        while (next < entries.size())
        {
            std::array<iocb, BatchSize> controls{};
            std::array<iocb *, BatchSize> pointers{};
            std::array<io_event, BatchSize> events{};
            size_t count = 0;
            while (next < entries.size() && count < BatchSize)
            {
                size_t i = next++;
                if (!entries[i].size)
                    continue;
                uint64_t aligned;
                if (!AlignedSize(entries[i].size, aligned))
                    return ErrorCode::MemoryOverFlow;
                auto &control = controls[count];
                control.aio_lio_opcode = IOCB_CMD_PREAD;
                control.aio_fildes = fd;
                control.aio_buf = reinterpret_cast<uint64_t>(values[i].GetBuffer());
                control.aio_nbytes = aligned;
                control.aio_offset = entries[i].offset;
                pointers[count++] = &control;
            }
            size_t submitted = 0;
            bool failed = false;
            while (submitted < count)
            {
                long result = ::syscall(SYS_io_submit, m_context, count - submitted, pointers.data() + submitted);
                if (result < 0 && errno == EINTR)
                    continue;
                if (result <= 0)
                {
                    failed = true;
                    break;
                }
                submitted += static_cast<size_t>(result);
            }
            size_t completed = 0;
            while (completed < submitted)
            {
                long result = ::syscall(SYS_io_getevents, m_context, 1, submitted - completed, events.data(), nullptr);
                if (result < 0 && errno == EINTR)
                    continue;
                if (result <= 0)
                {
                    Reset();
                    return ErrorCode::DiskIOFail;
                }
                for (long i = 0; i < result; ++i)
                {
                    auto *control = reinterpret_cast<const iocb *>(events[i].obj);
                    if (events[i].res2 != 0 || events[i].res != static_cast<int64_t>(control->aio_nbytes))
                        failed = true;
                }
                completed += static_cast<size_t>(result);
            }
            if (failed)
                return ErrorCode::DiskIOFail;
        }
        return ErrorCode::Success;
    }

  private:
    aio_context_t m_context = 0;
};

class AioPool
{
  public:
    ErrorCode ReadBatch(int fd, const std::vector<Entry> &entries, std::vector<Helper::PageBuffer<uint8_t>> &values)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        size_t selected = 0;
        m_available.wait(lock, [&] {
            for (size_t i = 0; i < m_slots.size(); ++i)
                if (!m_slots[i].busy)
                {
                    selected = i;
                    return true;
                }
            return false;
        });
        m_slots[selected].busy = true;
        lock.unlock();
        Lease lease{*this, selected};
        return m_slots[selected].context.ReadBatch(fd, entries, values);
    }

  private:
    struct Slot
    {
        AioContext context;
        bool busy = false;
    };
    struct Lease
    {
        AioPool &pool;
        size_t slot;
        ~Lease()
        {
            {
                std::lock_guard<std::mutex> lock(pool.m_mutex);
                pool.m_slots[slot].busy = false;
            }
            pool.m_available.notify_one();
        }
    };
    // Kernel contexts are initialized lazily; threads exceeding this bound wait
    // for a drained lease rather than consuming unbounded kernel AIO resources.
    std::array<Slot, 64> m_slots{};
    std::mutex m_mutex;
    std::condition_variable m_available;
};

struct FileOwner
{
    fs::path work;
    Descriptor lock;
    AioPool aio;
    uint64_t nextFile = 0;
};

class FileState
{
  public:
    std::shared_ptr<FileOwner> owner;
    fs::path activePath;
    Descriptor file;
    mutable std::shared_mutex mutex;
    std::map<std::string, Entry> entries;
    uint64_t tail = 0;

    ~FileState()
    {
        if (file.fd >= 0)
            ::unlink(activePath.c_str());
    }

    ErrorCode Read(const Entry &entry, std::string &value) const
    {
        if (!entry.size)
        {
            value.clear();
            return ErrorCode::Success;
        }
        uint64_t aligned;
        if (!AlignedSize(entry.size, aligned))
            return ErrorCode::MemoryOverFlow;
        Helper::PageBuffer<uint8_t> buffer;
        buffer.ReservePageBuffer(aligned);
        if (!buffer.GetBuffer())
            return ErrorCode::MemoryOverFlow;
        if (!Transfer(file.fd, buffer.GetBuffer(), aligned, entry.offset, false))
            return ErrorCode::DiskIOFail;
        value.assign(reinterpret_cast<char *>(buffer.GetBuffer()), entry.size);
        return ErrorCode::Success;
    }

    // Caller holds the mutation lock. Failed/partial writes consume their extent
    // too: no page visible to any earlier reader is ever overwritten.
    ErrorCode Append(const std::string &key, const std::string &value)
    {
        uint64_t aligned;
        if (!AlignedSize(value.size(), aligned) ||
            tail > static_cast<uint64_t>(std::numeric_limits<off_t>::max()) - aligned)
            return ErrorCode::MemoryOverFlow;
        Helper::PageBuffer<uint8_t> buffer;
        if (aligned)
        {
            buffer.ReservePageBuffer(aligned);
            if (!buffer.GetBuffer())
                return ErrorCode::MemoryOverFlow;
            memcpy(buffer.GetBuffer(), value.data(), value.size());
            memset(buffer.GetBuffer() + value.size(), 0, aligned - value.size());
        }
        const uint64_t offset = tail;
        tail += aligned;
        if (aligned && !Transfer(file.fd, buffer.GetBuffer(), aligned, offset, true))
            return ErrorCode::DiskIOFail;
        entries.insert_or_assign(key, Entry{offset, value.size()});
        return ErrorCode::Success;
    }

    ErrorCode Lookup(const std::string &prefix, const std::vector<SizeType> &keys, std::vector<Entry> &found,
                     bool missingAllowed) const
    {
        for (SizeType key : keys)
            if (key < 0)
                return ErrorCode::FailedParseValue;
        found.reserve(keys.size());
        std::shared_lock<std::shared_mutex> lock(mutex);
        for (SizeType key : keys)
        {
            auto it = entries.find(prefix + Number(key));
            if (it == entries.end() && !missingAllowed)
                return ErrorCode::Key_NotFound;
            found.push_back(it == entries.end() ? Entry{} : it->second);
        }
        return ErrorCode::Success;
    }
};

struct StateSlot
{
    std::shared_ptr<FileState> current;
    // Serialize mutations across generations. Reads capture a generation without
    // this gate and keep its append-only file alive until their last I/O drains.
    std::mutex mutations;
};

ErrorCode CreateState(const std::shared_ptr<FileOwner> &owner, std::shared_ptr<FileState> &result)
{
    auto state = std::make_shared<FileState>();
    state->owner = owner;
    while (true)
    {
        if (owner->nextFile == UINT64_MAX)
            return ErrorCode::MemoryOverFlow;
        uint64_t number = owner->nextFile++;
        state->activePath =
            owner->work / (number == 0 ? "postings.active" : "postings.active-" + std::to_string(number));
        state->file.fd =
            ::open(state->activePath.c_str(), O_RDWR | O_CREAT | O_EXCL | O_DIRECT | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (state->file.fd >= 0)
            break;
        if (errno != EEXIST)
            return ErrorCode::FailedCreateFile;
    }
    result = std::move(state);
    return ErrorCode::Success;
}

class FileView : public Helper::KeyValueIO
{
  public:
    FileView(std::shared_ptr<StateSlot> slot, std::string prefix) : m_slot(std::move(slot)), m_prefix(std::move(prefix))
    {
    }

    void ShutDown() override
    {
        std::atomic_store(&m_slot, std::shared_ptr<StateSlot>{});
    }
    bool Available() override
    {
        return std::atomic_load(&m_slot) != nullptr;
    }
    Storage GetStorageType() const override
    {
        return Storage::FILEIO;
    }
    bool CompletesRequestsBeforeReturn() const override
    {
        return true;
    }
    ErrorCode Checkpoint(std::string) override
    {
        return ErrorCode::Success;
    }

    ErrorCode Get(SizeType key, std::string *value, const std::chrono::microseconds &,
                  std::vector<Helper::AsyncReadRequest> *) override
    {
        try
        {
            auto state = Capture();
            if (!state || key < 0 || !value)
                return ErrorCode::FailedParseValue;
            Entry entry;
            {
                std::shared_lock<std::shared_mutex> lock(state->mutex);
                auto it = state->entries.find(m_prefix + Number(key));
                if (it == state->entries.end())
                    return ErrorCode::Key_NotFound;
                entry = it->second;
            }
            return state->Read(entry, *value);
        }
        catch (const std::bad_alloc &)
        {
            return ErrorCode::MemoryOverFlow;
        }
        catch (...)
        {
            return ErrorCode::DiskIOFail;
        }
    }

    ErrorCode Get(SizeType key, Helper::PageBuffer<uint8_t> &value, const std::chrono::microseconds &timeout,
                  std::vector<Helper::AsyncReadRequest> *requests, bool = true) override
    {
        std::string found;
        auto result = Get(key, &found, timeout, requests);
        if (result != ErrorCode::Success)
            return result;
        try
        {
            value.ReservePageBuffer(found.size());
            if (!found.empty() && !value.GetBuffer())
                return ErrorCode::MemoryOverFlow;
            if (!found.empty())
                memcpy(value.GetBuffer(), found.data(), found.size());
            value.SetAvailableSize(found.size());
            return ErrorCode::Success;
        }
        catch (const std::bad_alloc &)
        {
            return ErrorCode::MemoryOverFlow;
        }
        catch (...)
        {
            return ErrorCode::DiskIOFail;
        }
    }

    ErrorCode MultiGet(const std::vector<SizeType> &keys, std::vector<std::string> *values,
                       const std::chrono::microseconds &, std::vector<Helper::AsyncReadRequest> *) override
    {
        try
        {
            auto state = Capture();
            if (!state || !values)
                return ErrorCode::FailedParseValue;
            std::vector<Entry> entries;
            auto result = state->Lookup(m_prefix, keys, entries, false);
            if (result != ErrorCode::Success)
                return result;
            std::vector<Helper::PageBuffer<uint8_t>> buffers(keys.size());
            result = ReadBatch(*state, entries, buffers);
            if (result != ErrorCode::Success)
                return result;
            std::vector<std::string> found(keys.size());
            for (size_t i = 0; i < keys.size(); ++i)
                if (entries[i].size)
                    found[i].assign(reinterpret_cast<char *>(buffers[i].GetBuffer()), entries[i].size);
            *values = std::move(found);
            return ErrorCode::Success;
        }
        catch (const std::bad_alloc &)
        {
            return ErrorCode::MemoryOverFlow;
        }
        catch (...)
        {
            return ErrorCode::DiskIOFail;
        }
    }

    ErrorCode MultiGet(const std::vector<SizeType> &keys, std::vector<Helper::PageBuffer<uint8_t>> &values,
                       const std::chrono::microseconds &, std::vector<Helper::AsyncReadRequest> *) override
    {
        try
        {
            auto state = Capture();
            if (!state || values.size() < keys.size())
                return ErrorCode::FailedParseValue;
            std::vector<Entry> entries;
            auto result = state->Lookup(m_prefix, keys, entries, true);
            if (result != ErrorCode::Success)
                return result;
            result = ReadBatch(*state, entries, values);
            if (result == ErrorCode::Success)
                for (size_t i = 0; i < entries.size(); ++i)
                    values[i].SetAvailableSize(entries[i].size);
            return result;
        }
        catch (const std::bad_alloc &)
        {
            return ErrorCode::MemoryOverFlow;
        }
        catch (...)
        {
            return ErrorCode::DiskIOFail;
        }
    }

    ErrorCode Put(SizeType key, const std::string &value, const std::chrono::microseconds &,
                  std::vector<Helper::AsyncReadRequest> *) override
    {
        return Write(key, value, false, nullptr);
    }
    ErrorCode Merge(SizeType key, const std::string &value, const std::chrono::microseconds &,
                    std::vector<Helper::AsyncReadRequest> *, int &size) override
    {
        return Write(key, value, true, &size);
    }
    ErrorCode MergeWithCAS(SizeType key, const std::string &value, const std::chrono::microseconds &timeout,
                           std::vector<Helper::AsyncReadRequest> *requests, int &size) override
    {
        return Merge(key, value, timeout, requests, size);
    }
    ErrorCode Delete(SizeType key) override
    {
        return Erase(key, key, true);
    }
    ErrorCode DeleteRange(SizeType begin, SizeType end) override
    {
        return Erase(begin, end, false);
    }

  private:
    std::shared_ptr<FileState> Capture() const
    {
        auto slot = std::atomic_load(&m_slot);
        return slot ? std::atomic_load(&slot->current) : nullptr;
    }

    static ErrorCode ReadBatch(FileState &state, const std::vector<Entry> &entries,
                               std::vector<Helper::PageBuffer<uint8_t>> &values)
    {
        // Finish all potentially throwing allocation before the first submission.
        for (size_t i = 0; i < entries.size(); ++i)
        {
            uint64_t aligned;
            if (!AlignedSize(entries[i].size, aligned))
                return ErrorCode::MemoryOverFlow;
            values[i].ReservePageBuffer(aligned);
            if (aligned && !values[i].GetBuffer())
                return ErrorCode::MemoryOverFlow;
        }
        return state.owner->aio.ReadBatch(state.file.fd, entries, values);
    }

    ErrorCode Write(SizeType key, const std::string &value, bool merge, int *size)
    {
        try
        {
            auto slot = std::atomic_load(&m_slot);
            if (!slot || key < 0 || (merge && value.empty()))
                return ErrorCode::FailedParseValue;
            std::lock_guard<std::mutex> mutation(slot->mutations);
            auto state = std::atomic_load(&slot->current);
            std::unique_lock<std::shared_mutex> lock(state->mutex);
            std::string encoded = m_prefix + Number(key);
            if (!merge)
                return state->Append(encoded, value);
            std::string combined;
            auto it = state->entries.find(encoded);
            if (it != state->entries.end())
            {
                auto result = state->Read(it->second, combined);
                if (result != ErrorCode::Success)
                    return result;
            }
            if (combined.size() > INT_MAX || value.size() > INT_MAX - combined.size())
                return ErrorCode::MemoryOverFlow;
            combined.append(value);
            auto result = state->Append(encoded, combined);
            if (result == ErrorCode::Success)
                *size = static_cast<int>(combined.size());
            return result;
        }
        catch (const std::bad_alloc &)
        {
            return ErrorCode::MemoryOverFlow;
        }
        catch (...)
        {
            return ErrorCode::DiskIOFail;
        }
    }

    ErrorCode Erase(SizeType begin, SizeType end, bool single)
    {
        try
        {
            auto slot = std::atomic_load(&m_slot);
            if (!slot || begin < 0 || end < begin)
                return ErrorCode::FailedParseValue;
            std::lock_guard<std::mutex> mutation(slot->mutations);
            auto state = std::atomic_load(&slot->current);
            std::unique_lock<std::shared_mutex> lock(state->mutex);
            if (single)
                state->entries.erase(m_prefix + Number(begin));
            else
                state->entries.erase(state->entries.lower_bound(m_prefix + Number(begin)),
                                     state->entries.lower_bound(m_prefix + Number(end)));
            return ErrorCode::Success;
        }
        catch (const std::bad_alloc &)
        {
            return ErrorCode::MemoryOverFlow;
        }
        catch (...)
        {
            return ErrorCode::DiskIOFail;
        }
    }

    std::shared_ptr<StateSlot> m_slot;
    const std::string m_prefix;
};

// Image format (all integers big-endian): magic[8], entry-count, body-bytes,
// repeated {key-bytes, value-bytes, key, value}, then FNV-1a64 over everything
// preceding the checksum. Keys contain tenant-length, tenant, index, local key.
// Active offsets/padding never enter the image, so it is compact and standalone.
class ImageStream
{
  public:
    explicit ImageStream(int descriptor) : fd(descriptor)
    {
    }
    bool Bytes(void *data, uint64_t size, bool write, bool hash = true)
    {
        if (!Transfer(fd, data, size, offset, write))
            return false;
        offset += size;
        if (hash)
            checksum = Hash(checksum, data, size);
        return true;
    }
    bool Integer(uint64_t &value, bool write, bool hash = true)
    {
        std::string encoded = write ? Number(value) : std::string(8, '\0');
        if (!Bytes(encoded.data(), 8, write, hash))
            return false;
        if (!write)
            value = Decode(encoded.data());
        return true;
    }
    int fd;
    uint64_t offset = 0;
    uint64_t checksum = HashSeed;
};

struct PendingImage
{
    fs::path path;
    bool owned = false;
    ~PendingImage()
    {
        if (owned)
            ::unlink(path.c_str());
    }
};

ErrorCode Restore(FileState &state, const fs::path &imagePath)
{
    Descriptor image(::open(imagePath.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    struct stat status
    {
    };
    if (image.fd < 0 || ::fstat(image.fd, &status) != 0 || !S_ISREG(status.st_mode) || status.st_size < 32)
        return ErrorCode::DiskIOFail;
    ImageStream stream(image.fd);
    std::array<char, 8> magic{};
    uint64_t count = 0, body = 0;
    if (!stream.Bytes(magic.data(), magic.size(), false) || memcmp(magic.data(), ImageMagic, 8) != 0 ||
        !stream.Integer(count, false) || !stream.Integer(body, false) ||
        body != static_cast<uint64_t>(status.st_size) - 32 || count > body / 40)
        return ErrorCode::DiskIOFail;
    const uint64_t end = 24 + body;
    for (uint64_t i = 0; i < count; ++i)
    {
        uint64_t keySize = 0, valueSize = 0;
        if (stream.offset > end || end - stream.offset < 16 || !stream.Integer(keySize, false) ||
            !stream.Integer(valueSize, false) || keySize < 24 || keySize > end - stream.offset ||
            valueSize > end - stream.offset - keySize || keySize > std::numeric_limits<size_t>::max() ||
            valueSize > std::numeric_limits<size_t>::max())
            return ErrorCode::DiskIOFail;
        std::string key(static_cast<size_t>(keySize), '\0');
        std::string value(static_cast<size_t>(valueSize), '\0');
        if (!stream.Bytes(key.data(), key.size(), false) || !ValidKey(key) ||
            state.entries.find(key) != state.entries.end() || !stream.Bytes(value.data(), value.size(), false))
            return ErrorCode::DiskIOFail;
        auto result = state.Append(key, value);
        if (result != ErrorCode::Success)
            return result;
    }
    uint64_t checksum = 0;
    if (stream.offset != end || !stream.Integer(checksum, false, false) || checksum != stream.checksum)
        return ErrorCode::Posting_ChecksumError;
    return image.Close() ? ErrorCode::Success : ErrorCode::DiskIOFail;
}

class FileBackend : public PostingBackend
{
  public:
    explicit FileBackend(std::shared_ptr<StateSlot> slot) : m_slot(std::move(slot))
    {
    }
    std::shared_ptr<Helper::KeyValueIO> View(const std::string &tenant, uint64_t index) override
    {
        return std::make_shared<FileView>(m_slot, Prefix(tenant, index));
    }
    ErrorCode DeleteIndex(const std::string &tenant, uint64_t index) override
    {
        try
        {
            const std::string prefix = Prefix(tenant, index);
            std::lock_guard<std::mutex> mutation(m_slot->mutations);
            auto state = std::atomic_load(&m_slot->current);
            std::unique_lock<std::shared_mutex> lock(state->mutex);
            auto first = state->entries.lower_bound(prefix);
            auto last = first;
            while (last != state->entries.end() && last->first.compare(0, prefix.size(), prefix) == 0)
                ++last;
            state->entries.erase(first, last);
            return ErrorCode::Success;
        }
        catch (const std::bad_alloc &)
        {
            return ErrorCode::MemoryOverFlow;
        }
        catch (...)
        {
            return ErrorCode::DiskIOFail;
        }
    }

    ErrorCode Checkpoint(const std::string &directory) override
    {
        try
        {
            if (directory.empty())
                return ErrorCode::FailedParseValue;
            std::lock_guard<std::mutex> mutation(m_slot->mutations);
            auto state = std::atomic_load(&m_slot->current);
            const fs::path destination = fs::weakly_canonical(directory);
            if (destination == state->owner->work)
                return ErrorCode::FailedParseValue;
            fs::create_directories(destination);
            const fs::path published = destination / ImageName;
            if (fs::exists(published))
                return ErrorCode::FailedCreateFile;
            PendingImage pending{destination / "postings.image.pending", false};
            Descriptor image(::open(pending.path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600));
            if (image.fd < 0)
                return ErrorCode::FailedCreateFile;
            pending.owned = true;
            std::vector<std::pair<std::string, Entry>> entries;
            {
                std::shared_lock<std::shared_mutex> lock(state->mutex);
                entries.assign(state->entries.begin(), state->entries.end());
            }
            uint64_t body = 0;
            for (const auto &item : entries)
            {
                if (item.first.size() > UINT64_MAX - 16 || item.second.size > UINT64_MAX - 16 - item.first.size() ||
                    body > UINT64_MAX - 16 - item.first.size() - item.second.size)
                    return ErrorCode::MemoryOverFlow;
                body += 16 + item.first.size() + item.second.size;
            }
            if (body > static_cast<uint64_t>(std::numeric_limits<off_t>::max()) - 32)
                return ErrorCode::MemoryOverFlow;
            ImageStream stream(image.fd);
            std::array<char, 8> magic{};
            memcpy(magic.data(), ImageMagic, 8);
            uint64_t count = entries.size();
            if (!stream.Bytes(magic.data(), magic.size(), true) || !stream.Integer(count, true) ||
                !stream.Integer(body, true))
                return ErrorCode::DiskIOFail;
            for (auto &item : entries)
            {
                std::string value;
                auto result = state->Read(item.second, value);
                if (result != ErrorCode::Success)
                    return result;
                uint64_t keySize = item.first.size(), valueSize = value.size();
                if (!stream.Integer(keySize, true) || !stream.Integer(valueSize, true) ||
                    !stream.Bytes(item.first.data(), item.first.size(), true) ||
                    !stream.Bytes(value.data(), value.size(), true))
                    return ErrorCode::DiskIOFail;
            }
            uint64_t checksum = stream.checksum;
            if (!stream.Integer(checksum, true, false) || !Sync(image.fd) || !image.Close())
                return ErrorCode::DiskIOFail;
            std::shared_ptr<FileState> compact;
            auto result = CreateState(state->owner, compact);
            if (result != ErrorCode::Success)
                return result;
            result = Restore(*compact, pending.path);
            if (result != ErrorCode::Success)
                return result;
            if (!Sync(compact->file.fd) || !SyncDirectory(state->owner->work))
                return ErrorCode::DiskIOFail;
            // link is an atomic, no-replace publication of a fully synced image.
            if (::link(pending.path.c_str(), published.c_str()) != 0 || ::unlink(pending.path.c_str()) != 0)
                return ErrorCode::DiskIOFail;
            pending.owned = false;
            if (!SyncDirectory(destination) || !SyncDirectory(destination.parent_path()))
                return ErrorCode::DiskIOFail;
            std::atomic_store(&m_slot->current, std::move(compact));
            return ErrorCode::Success;
        }
        catch (const std::bad_alloc &)
        {
            return ErrorCode::MemoryOverFlow;
        }
        catch (...)
        {
            return ErrorCode::DiskIOFail;
        }
    }

  private:
    std::shared_ptr<StateSlot> m_slot;
};
} // namespace

ErrorCode OpenFilePostingBackend(const std::string &workDirectory, const std::string &checkpointDirectory,
                                 std::shared_ptr<PostingBackend> &result)
{
    result.reset();
    try
    {
        if (workDirectory.empty())
            return ErrorCode::FailedParseValue;
        auto owner = std::make_shared<FileOwner>();
        owner->work = fs::weakly_canonical(workDirectory);
        if (!checkpointDirectory.empty())
        {
            auto checkpoint = fs::weakly_canonical(checkpointDirectory);
            auto relative = owner->work.lexically_relative(checkpoint);
            if (relative.empty() || *relative.begin() != "..")
                return ErrorCode::FailedParseValue;
        }
        fs::create_directories(owner->work);
        owner->lock.fd =
            ::open((owner->work / "postings.lock").c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (owner->lock.fd < 0)
            return ErrorCode::FailedCreateFile;
        if (::flock(owner->lock.fd, LOCK_EX | LOCK_NB) != 0)
            return ErrorCode::DiskIOFail;
        std::shared_ptr<FileState> state;
        auto created = CreateState(owner, state);
        if (created != ErrorCode::Success)
            return created;
        if (!checkpointDirectory.empty())
        {
            auto restored = Restore(*state, fs::path(checkpointDirectory) / ImageName);
            if (restored != ErrorCode::Success)
                return restored;
        }
        auto slot = std::make_shared<StateSlot>();
        slot->current = std::move(state);
        result = std::make_shared<FileBackend>(std::move(slot));
        return ErrorCode::Success;
    }
    catch (const std::bad_alloc &)
    {
        return ErrorCode::MemoryOverFlow;
    }
    catch (...)
    {
        return ErrorCode::DiskIOFail;
    }
}
} // namespace SPTAG::SPANN

#else
namespace SPTAG::SPANN
{
ErrorCode OpenFilePostingBackend(const std::string &, const std::string &, std::shared_ptr<PostingBackend> &result)
{
    result.reset();
    return ErrorCode::Fail;
}
} // namespace SPTAG::SPANN
#endif
#endif
