// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/SPANN/PostingBackend.h"

#ifdef MULTITENANT
#ifdef TIKV
#include "kvproto/pdpb.grpc.pb.h"
#include "kvproto/tikvpb.grpc.pb.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <functional>
#include <grpcpp/grpcpp.h>
#include <limits>
#include <mutex>
#include <sstream>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>

namespace SPTAG::SPANN
{
namespace
{
using Deadline = std::chrono::system_clock::time_point;
using Visitor = std::function<ErrorCode(const std::string &, const std::string &)>;
constexpr size_t MaxKey = 1 << 20;
constexpr size_t MaxValue = 64 << 20;
constexpr uint32_t ScanLimit = 1024;
constexpr char ArchiveMagic[] = "SPTAG-TIKV-0001";
constexpr char OwnerValue[] = "SPTAG-managed-rawkv-v1";
constexpr char ArchiveName[] = "tikv-postings.bin";

ErrorCode Failure(const char *message)
{
    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Managed TiKV: %s\n", message);
    return ErrorCode::Fail;
}

std::string Number(uint64_t n)
{
    std::string result(8, '\0');
    for (int i = 7; i >= 0; --i, n >>= 8)
        result[i] = static_cast<char>(n);
    return result;
}

uint64_t ReadNumber(const char *bytes)
{
    uint64_t result = 0;
    for (int i = 0; i < 8; ++i)
        result = (result << 8) | static_cast<unsigned char>(bytes[i]);
    return result;
}

std::string PrefixEnd(std::string prefix)
{
    for (size_t i = prefix.size(); i > 0; --i)
    {
        auto byte = static_cast<unsigned char>(prefix[i - 1]);
        if (byte != 255)
        {
            prefix[i - 1] = static_cast<char>(byte + 1);
            prefix.resize(i);
            return prefix;
        }
    }
    return {};
}

Deadline OperationDeadline(const std::chrono::microseconds &timeout)
{
    auto cap = std::chrono::microseconds(10000000);
    if (timeout.count() > 0)
        cap = std::min(cap, timeout);
    return std::chrono::system_clock::now() + cap;
}

void SetDeadline(grpc::ClientContext &context, Deadline deadline,
                 const std::chrono::microseconds &timeout = std::chrono::microseconds(0))
{
    auto cap = std::chrono::microseconds(2000000);
    if (timeout.count() > 0)
        cap = std::min(cap, timeout);
    context.set_deadline(std::min(deadline, std::chrono::system_clock::now() + cap));
}

void RetryPause(int attempt, Deadline deadline, int baseMilliseconds = 25)
{
    if (attempt > 0)
        std::this_thread::sleep_until(std::min(
            deadline, std::chrono::system_clock::now() + std::chrono::milliseconds(baseMilliseconds << (attempt - 1))));
}

std::string Address(std::string address)
{
    const auto scheme = address.find("://");
    if (scheme != std::string::npos)
        address.erase(0, scheme + 3);
    return address;
}

// This client intentionally does not reuse native TiKVIO's unbounded retry paths.
// Requests own all buffers and have both per-RPC deadlines and bounded attempts.
class ManagedRawClient
{
  public:
    explicit ManagedRawClient(const std::string &addresses)
    {
        if (addresses.size() > 16384)
            return;
        std::istringstream input(addresses);
        std::string address;
        while (std::getline(input, address, ','))
        {
            const auto begin = address.find_first_not_of(" \t\r\n");
            if (begin == std::string::npos)
                continue;
            address = address.substr(begin, address.find_last_not_of(" \t\r\n") - begin + 1);
            if (m_addresses.size() == 16)
            {
                m_addresses.clear();
                return;
            }
            m_addresses.push_back(Address(address));
        }
    }

    void WatchNamespace(std::string prefix)
    {
        m_watchedPrefix = std::move(prefix);
    }
    bool HasUncertainMutation() const
    {
        return m_uncertain.load(std::memory_order_acquire);
    }

    ErrorCode Open()
    {
        auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(8);
        for (const auto &address : m_addresses)
        {
            if (std::chrono::system_clock::now() >= deadline)
                break;
            auto stub = pdpb::PD::NewStub(Channel(address));
            pdpb::GetMembersRequest request;
            pdpb::GetMembersResponse response;
            grpc::ClientContext context;
            SetDeadline(context, deadline);
            auto status = stub->GetMembers(&context, request, &response);
            if (!status.ok() || !response.has_header() || response.header().has_error() ||
                response.header().cluster_id() == 0 || !response.has_leader() ||
                response.leader().client_urls_size() == 0)
                continue;
            m_cluster = response.header().cluster_id();
            m_leader = Address(response.leader().client_urls(0));
            m_pdStubs.push_back(pdpb::PD::NewStub(Channel(m_leader)));
            for (const auto &candidate : m_addresses)
                if (candidate != m_leader)
                    m_pdStubs.push_back(pdpb::PD::NewStub(Channel(candidate)));
            return ErrorCode::Success;
        }
        return Failure("PD discovery failed (empty/invalid endpoints, unavailable cluster, or deadline)");
    }

    ErrorCode Get(const std::string &key, std::string &value, Deadline deadline)
    {
        kvrpcpb::RawGetRequest request;
        kvrpcpb::RawGetResponse response;
        request.set_key(key);
        auto error = Call(key, request, response, deadline, [](auto *stub, auto *ctx, const auto &req, auto *resp) {
            return stub->RawGet(ctx, req, resp);
        });
        if (error != ErrorCode::Success)
            return error;
        if (response.not_found())
            return ErrorCode::Key_NotFound;
        value = response.value();
        return ErrorCode::Success;
    }

    ErrorCode MultiGet(const std::vector<std::string> &keys, std::vector<std::string> &values, Deadline deadline,
                       const std::chrono::microseconds &rpcTimeout = std::chrono::microseconds(0))
    {
        const auto started = std::chrono::steady_clock::now();
        int attempts = 0, transportFailures = 0, regionFailures = 0, invalidResponses = 0;
        int lastTransportCode = 0;
        using Entry = std::pair<size_t, std::string>;
        struct Group
        {
            Route route;
            std::vector<Entry> entries;
            bool succeeded = false;
        };
        std::vector<Entry> pending;
        pending.reserve(keys.size());
        for (size_t i = 0; i < keys.size(); ++i)
        {
            if (keys[i].size() > MaxKey)
                return Failure("batch key exceeds managed format limit");
            pending.emplace_back(i, keys[i]);
        }
        std::vector<std::string> result(keys.size());
        for (int attempt = 0; !pending.empty() && attempt < 4 && std::chrono::system_clock::now() < deadline; ++attempt)
        {
            ++attempts;
            RetryPause(attempt, deadline, 1);
            std::vector<Group> groups;
            std::unordered_map<uint64_t, size_t> byRegion;
            std::vector<Entry> retry;
            for (auto &entry : pending)
            {
                Route route;
                if (!Resolve(entry.second, route, deadline))
                {
                    retry.push_back(std::move(entry));
                    continue;
                }
                auto inserted = byRegion.emplace(route.region.id(), groups.size());
                if (inserted.second)
                    groups.push_back({std::move(route), {}, false});
                groups[inserted.first->second].entries.push_back(std::move(entry));
            }
            struct BatchCall
            {
                Group *group = nullptr;
                kvrpcpb::RawBatchGetRequest request;
                kvrpcpb::RawBatchGetResponse response;
                grpc::ClientContext context;
                grpc::Status status;
                std::unordered_map<std::string, size_t> wanted;
                std::unique_ptr<grpc::ClientAsyncResponseReader<kvrpcpb::RawBatchGetResponse>> reader;
            };
            // Prepare owned requests before submitting any RPC. Their lifetimes
            // extend past the queue guard, including during exception unwinding.
            std::vector<std::unique_ptr<BatchCall>> calls;
            calls.reserve(groups.size());
            for (auto &group : groups)
            {
                auto call = std::make_unique<BatchCall>();
                call->group = &group;
                for (const auto &entry : group.entries)
                    if (call->wanted.emplace(entry.second, entry.first).second)
                        call->request.add_keys(entry.second);
                Context(call->request.mutable_context(), group.route);
                calls.push_back(std::move(call));
            }
            grpc::CompletionQueue queue;
            struct QueueGuard
            {
                grpc::CompletionQueue &queue;
                std::vector<std::unique_ptr<BatchCall>> &calls;
                bool drained = false;
                void Drain()
                {
                    queue.Shutdown();
                    void *tag;
                    bool ok;
                    while (queue.Next(&tag, &ok))
                    {
                    }
                    drained = true;
                }
                ~QueueGuard()
                {
                    if (!drained)
                    {
                        for (auto &call : calls)
                            call->context.TryCancel();
                        Drain();
                    }
                }
            } guard{queue, calls};
            size_t next = 0, inflight = 0;
            auto submit = [&] {
                while (next < calls.size() && inflight < 16 && std::chrono::system_clock::now() < deadline)
                {
                    auto &call = *calls[next++];
                    SetDeadline(call.context, deadline, rpcTimeout);
                    call.reader =
                        call.group->route.store->Next()->AsyncRawBatchGet(&call.context, call.request, &queue);
                    if (!call.reader)
                        continue;
                    call.reader->Finish(&call.response, &call.status, &call);
                    ++inflight;
                }
            };
            submit();
            while (inflight != 0)
            {
                void *tag = nullptr;
                bool ok = false;
                if (!queue.Next(&tag, &ok))
                    return Failure("batch completion queue closed before all RPCs completed");
                --inflight;
                auto &call = *static_cast<BatchCall *>(tag);
                auto &group = *call.group;
                if (!ok || !call.status.ok() || call.response.has_region_error())
                {
                    if (!ok || !call.status.ok())
                    {
                        ++transportFailures;
                        lastTransportCode = static_cast<int>(call.status.error_code());
                    }
                    else
                        ++regionFailures;
                    // A read timeout says nothing about the region's epoch or leader.
                    const auto code = call.status.error_code();
                    if (call.response.has_region_error() ||
                        (code != grpc::StatusCode::OK && code != grpc::StatusCode::DEADLINE_EXCEEDED &&
                         code != grpc::StatusCode::CANCELLED && code != grpc::StatusCode::RESOURCE_EXHAUSTED))
                        Invalidate(group.entries.front().second);
                }
                else
                {
                    std::unordered_map<std::string, const std::string *> found;
                    bool valid = true;
                    for (const auto &pair : call.response.pairs())
                    {
                        if (pair.has_error() || call.wanted.find(pair.key()) == call.wanted.end() ||
                            pair.value().size() > MaxValue || !found.emplace(pair.key(), &pair.value()).second)
                        {
                            valid = false;
                            break;
                        }
                    }
                    if (valid)
                    {
                        for (const auto &entry : group.entries)
                        {
                            const auto it = found.find(entry.second);
                            result[entry.first] = it == found.end() ? std::string() : *it->second;
                        }
                        group.succeeded = true;
                    }
                    else
                        ++invalidResponses;
                }
                submit();
            }
            guard.Drain();
            for (auto &group : groups)
                if (!group.succeeded)
                    for (auto &entry : group.entries)
                        retry.push_back(std::move(entry));
            pending = std::move(retry);
        }
        if (!pending.empty())
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                         "Managed TiKV batch failed: keys=%zu pending=%zu attempts=%d transport=%d last_grpc=%d "
                         "region=%d invalid=%d elapsed_us=%lld\n",
                         keys.size(), pending.size(), attempts, transportFailures, lastTransportCode, regionFailures,
                         invalidResponses,
                         static_cast<long long>(std::chrono::duration_cast<std::chrono::microseconds>(
                                                    std::chrono::steady_clock::now() - started)
                                                    .count()));
            return Failure("RawBatchGet exhausted batch retries/deadline");
        }
        values = std::move(result);
        return ErrorCode::Success;
    }

    ErrorCode Put(const std::string &key, const std::string &value, Deadline deadline)
    {
        if (key.size() > MaxKey || value.size() > MaxValue)
            return Failure("key/value exceeds managed format limit");
        kvrpcpb::RawPutRequest request;
        kvrpcpb::RawPutResponse response;
        request.set_key(key);
        request.set_value(value);
        request.set_for_cas(true);
        return Call(
            key, request, response, deadline,
            [](auto *stub, auto *ctx, const auto &req, auto *resp) { return stub->RawPut(ctx, req, resp); }, false);
    }

    ErrorCode Delete(const std::string &key, Deadline deadline)
    {
        kvrpcpb::RawDeleteRequest request;
        kvrpcpb::RawDeleteResponse response;
        request.set_key(key);
        request.set_for_cas(true);
        return Call(
            key, request, response, deadline,
            [](auto *stub, auto *ctx, const auto &req, auto *resp) { return stub->RawDelete(ctx, req, resp); }, false);
    }

    ErrorCode CAS(const std::string &key, const std::string &value, bool absent, const std::string &previous,
                  bool *swapped, bool *actualAbsent, std::string *actual, Deadline deadline)
    {
        if (!swapped || !actualAbsent || !actual || key.size() > MaxKey || value.size() > MaxValue ||
            previous.size() > MaxValue)
            return Failure("invalid CAS arguments");
        *swapped = false;
        kvrpcpb::RawCASRequest request;
        kvrpcpb::RawCASResponse response;
        request.set_key(key);
        request.set_value(value);
        request.set_previous_not_exist(absent);
        if (!absent)
            request.set_previous_value(previous);
        auto error = Call(
            key, request, response, deadline,
            [](auto *stub, auto *ctx, const auto &req, auto *resp) { return stub->RawCompareAndSwap(ctx, req, resp); },
            false);
        if (error != ErrorCode::Success)
            return error;
        *swapped = response.succeed();
        *actualAbsent = response.previous_not_exist();
        *actual = response.previous_value();
        return ErrorCode::Success;
    }

    // RawScan is region-local. Empty/short pages do NOT mean namespace EOF.
    // Advance through empty regions, and use lastKey + NUL to preserve binary extensions.
    ErrorCode Scan(const std::string &begin, const std::string &end, const Visitor &visitor)
    {
        if (end.empty() || begin >= end)
            return Failure("invalid namespace range");
        std::string cursor = begin;
        while (cursor < end)
        {
            const auto deadline = OperationDeadline(std::chrono::microseconds(10000000));
            bool completed = false;
            uint32_t limit = ScanLimit;
            for (int attempt = 0; attempt < 4 && std::chrono::system_clock::now() < deadline; ++attempt)
            {
                RetryPause(attempt, deadline);
                Route route;
                if (!Resolve(cursor, route, deadline))
                    continue;
                std::string regionEnd = route.region.end_key();
                if (regionEnd.empty() || regionEnd > end)
                    regionEnd = end;
                if (regionEnd <= cursor)
                    return Failure("PD returned non-progressing region boundary");
                kvrpcpb::RawScanRequest request;
                request.set_start_key(cursor);
                request.set_end_key(regionEnd);
                request.set_limit(limit);
                Context(request.mutable_context(), route);
                kvrpcpb::RawScanResponse response;
                grpc::ClientContext context;
                SetDeadline(context, deadline);
                auto status = route.store->Next()->RawScan(&context, request, &response);
                if (!status.ok())
                {
                    Invalidate(cursor);
                    if (status.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED)
                        limit = std::max(uint32_t(1), limit / 16);
                    continue;
                }
                if (response.has_region_error())
                {
                    Invalidate(cursor);
                    continue;
                }
                std::string previous;
                for (const auto &pair : response.kvs())
                {
                    if (pair.has_error() || pair.key() < cursor || pair.key() >= regionEnd ||
                        (!previous.empty() && pair.key() <= previous) || pair.key().size() > MaxKey ||
                        pair.value().size() > MaxValue)
                        return Failure("invalid scan pair or per-key scan error");
                    previous = pair.key();
                }
                for (const auto &pair : response.kvs())
                {
                    auto error = visitor(pair.key(), pair.value());
                    if (error != ErrorCode::Success)
                        return error;
                }
                cursor = response.kvs_size() == 0 ? regionEnd : previous + std::string(1, '\0');
                completed = true;
                break;
            }
            if (!completed)
                return Failure("namespace scan exhausted retries/deadline");
        }
        return ErrorCode::Success;
    }

  private:
    struct StoreConnections
    {
        explicit StoreConnections(const std::string &address)
        {
            for (auto &stub : stubs)
                stub = tikvpb::Tikv::NewStub(Channel(address, true));
        }

        tikvpb::Tikv::Stub *Next()
        {
            return stubs[next.fetch_add(1, std::memory_order_relaxed) % stubs.size()].get();
        }

        std::array<std::unique_ptr<tikvpb::Tikv::Stub>, 16> stubs;
        std::atomic<size_t> next{0};
    };

    struct Route
    {
        metapb::Region region;
        metapb::Peer leader;
        std::shared_ptr<StoreConnections> store;
    };

    static std::shared_ptr<grpc::Channel> Channel(const std::string &address, bool independent = false)
    {
        grpc::ChannelArguments args;
        // A CAS request contains both the previous and replacement values.
        args.SetMaxReceiveMessageSize(132 << 20);
        args.SetMaxSendMessageSize(132 << 20);
        if (independent)
            args.SetInt(GRPC_ARG_USE_LOCAL_SUBCHANNEL_POOL, 1);
        return grpc::CreateCustomChannel(address, grpc::InsecureChannelCredentials(), args);
    }

    bool Resolve(const std::string &key, Route &route, Deadline deadline)
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            for (const auto &cached : m_routes)
                if (Contains(cached, key))
                {
                    route = cached;
                    return true;
                }
        }
        // Stubs are immutable after Open; gRPC supports concurrent calls without reconnect/reset races.
        for (const auto &pd : m_pdStubs)
        {
            if (std::chrono::system_clock::now() >= deadline)
                return false;
            pdpb::GetRegionRequest request;
            request.mutable_header()->set_cluster_id(m_cluster);
            request.set_region_key(key);
            pdpb::GetRegionResponse response;
            grpc::ClientContext context;
            SetDeadline(context, deadline);
            auto status = pd->GetRegion(&context, request, &response);
            if (!status.ok() || !response.has_header() || response.header().has_error() ||
                response.header().cluster_id() != m_cluster || !response.has_region() || !response.has_leader() ||
                response.leader().store_id() == 0 ||
                (!response.region().start_key().empty() && key < response.region().start_key()) ||
                (!response.region().end_key().empty() && key >= response.region().end_key()))
                continue;
            std::string storeAddress;
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                const auto found = m_storeAddresses.find(response.leader().store_id());
                if (found != m_storeAddresses.end())
                    storeAddress = found->second;
            }
            if (storeAddress.empty())
            {
                pdpb::GetStoreRequest storeRequest;
                storeRequest.mutable_header()->set_cluster_id(m_cluster);
                storeRequest.set_store_id(response.leader().store_id());
                pdpb::GetStoreResponse storeResponse;
                grpc::ClientContext storeContext;
                SetDeadline(storeContext, deadline);
                status = pd->GetStore(&storeContext, storeRequest, &storeResponse);
                if (!status.ok() || !storeResponse.has_header() || storeResponse.header().has_error() ||
                    storeResponse.header().cluster_id() != m_cluster || !storeResponse.has_store() ||
                    storeResponse.store().address().empty())
                    continue;
                storeAddress = Address(storeResponse.store().address());
            }
            route.region = response.region();
            route.leader = response.leader();
            std::lock_guard<std::mutex> lock(m_mutex);
            m_storeAddresses[route.leader.store_id()] = storeAddress;
            auto &store = m_stores[storeAddress];
            if (!store)
                store = std::make_shared<StoreConnections>(storeAddress);
            route.store = store;
            m_routes.erase(std::remove_if(m_routes.begin(), m_routes.end(),
                                          [&](const Route &old) {
                                              return (old.region.end_key().empty() ||
                                                      route.region.start_key() < old.region.end_key()) &&
                                                     (route.region.end_key().empty() ||
                                                      old.region.start_key() < route.region.end_key());
                                          }),
                           m_routes.end());
            m_routes.push_back(route);
            return true;
        }
        return false;
    }

    static bool Contains(const Route &route, const std::string &key)
    {
        return (route.region.start_key().empty() || key >= route.region.start_key()) &&
               (route.region.end_key().empty() || key < route.region.end_key());
    }

    void Invalidate(const std::string &key)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (const auto &route : m_routes)
            if (Contains(route, key))
                m_storeAddresses.erase(route.leader.store_id());
        m_routes.erase(
            std::remove_if(m_routes.begin(), m_routes.end(), [&](const Route &route) { return Contains(route, key); }),
            m_routes.end());
    }

    static void Context(kvrpcpb::Context *context, const Route &route)
    {
        context->set_region_id(route.region.id());
        *context->mutable_region_epoch() = route.region.region_epoch();
        *context->mutable_peer() = route.leader;
    }

    template <class Request, class Response, class RPC>
    ErrorCode Call(const std::string &key, Request &request, Response &response, Deadline deadline, RPC rpc,
                   bool retryTransport = true)
    {
        if (key.size() > MaxKey)
            return Failure("key exceeds managed format limit");
        for (int attempt = 0; attempt < 4 && std::chrono::system_clock::now() < deadline; ++attempt)
        {
            RetryPause(attempt, deadline);
            Route route;
            if (!Resolve(key, route, deadline))
                continue;
            Context(request.mutable_context(), route);
            response.Clear();
            grpc::ClientContext context;
            SetDeadline(context, deadline);
            auto status = rpc(route.store->Next(), &context, request, &response);
            if (!status.ok())
            {
                Invalidate(key);
                // A failed mutation may already have committed. Even replaying an
                // idempotent Put could leave an older RPC overwriting a later update.
                if (!retryTransport)
                {
                    if (key.compare(0, m_watchedPrefix.size(), m_watchedPrefix) == 0)
                        m_uncertain.store(true, std::memory_order_release);
                    return Failure("mutation outcome unknown after RPC failure; request was not replayed");
                }
                continue;
            }
            if (response.has_region_error())
            {
                Invalidate(key);
                continue;
            }
            if (!response.error().empty())
                return Failure("RawKV returned an application error");
            return ErrorCode::Success;
        }
        return Failure("RawKV operation exhausted retries/deadline");
    }

    std::vector<std::string> m_addresses;
    std::string m_leader;
    std::string m_watchedPrefix;
    std::atomic<bool> m_uncertain{false};
    uint64_t m_cluster = 0;
    std::mutex m_mutex;
    std::vector<Route> m_routes;
    std::unordered_map<std::string, std::shared_ptr<StoreConnections>> m_stores;
    std::unordered_map<uint64_t, std::string> m_storeAddresses;
    std::vector<std::shared_ptr<pdpb::PD::Stub>> m_pdStubs;
};

uint32_t CRC(uint32_t crc, const char *data, size_t length)
{
    static const auto table = [] {
        std::array<uint32_t, 256> values{};
        for (uint32_t i = 0; i < 256; ++i)
        {
            uint32_t value = i;
            for (int bit = 0; bit < 8; ++bit)
                value = (value >> 1) ^ ((value & 1) ? 0xedb88320U : 0);
            values[i] = value;
        }
        return values;
    }();
    for (size_t i = 0; i < length; ++i)
        crc = table[(crc ^ static_cast<unsigned char>(data[i])) & 255] ^ (crc >> 8);
    return crc;
}

struct File
{
    explicit File(int descriptor) : fd(descriptor)
    {
    }
    ~File()
    {
        if (fd >= 0)
            ::close(fd);
    }
    File(const File &) = delete;
    File &operator=(const File &) = delete;
    int fd;
};

struct StagingCleanup
{
    explicit StagingCleanup(std::filesystem::path name) : path(std::move(name))
    {
    }
    ~StagingCleanup()
    {
        if (owned)
            ::unlink(path.c_str());
    }
    bool Remove()
    {
        if (::unlink(path.c_str()) != 0)
            return false;
        owned = false;
        return true;
    }
    const std::filesystem::path path;
    bool owned = true;
};

bool Write(int fd, const std::string &data, uint32_t &crc)
{
    size_t written = 0;
    while (written < data.size())
    {
        auto n = ::write(fd, data.data() + written, data.size() - written);
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0)
            return false;
        written += static_cast<size_t>(n);
    }
    crc = CRC(crc, data.data(), data.size());
    return true;
}

bool ValidSuffix(const std::string &key, const std::string &value)
{
    if (key == "o")
        return value == OwnerValue;
    if (key.size() < 18 || key[0] != 'v')
        return false;
    auto tenantLength = ReadNumber(key.data() + 1);
    if (tenantLength > key.size() - 18)
        return false;
    const auto offset = static_cast<size_t>(tenantLength) + 17;
    if (key[offset] == '\0')
        return key.size() == offset + 1 && value.empty();
    if (key[offset] == '\1')
        return key.size() == offset + 9 &&
               ReadNumber(key.data() + offset + 1) <= static_cast<uint64_t>((std::numeric_limits<SizeType>::max)());
    return key[offset] == '\2';
}

// Header, sorted R/key-length/value-length/key/value records, then E/count/CRC32.
// Limits are checked before allocation; one complete validation pass precedes import.
ErrorCode ReadArchive(int fd, const Visitor &visitor)
{
    struct stat info
    {
    };
    if (::fstat(fd, &info) != 0 || info.st_size < 0 || ::lseek(fd, 0, SEEK_SET) < 0)
        return Failure("cannot stat/rewind archive");
    uint64_t remaining = static_cast<uint64_t>(info.st_size);
    uint32_t crc = 0xffffffffU;
    auto read = [&](size_t length, std::string &result, bool checksum = true) {
        if (length > remaining)
            return false;
        result.resize(length);
        size_t done = 0;
        while (done < length)
        {
            auto n = ::read(fd, &result[done], length - done);
            if (n < 0 && errno == EINTR)
                continue;
            if (n <= 0)
                return false;
            done += static_cast<size_t>(n);
        }
        remaining -= length;
        if (checksum)
            crc = CRC(crc, result.data(), result.size());
        return true;
    };
    std::string data;
    if (!read(sizeof(ArchiveMagic) - 1, data) || data != ArchiveMagic)
        return Failure("invalid archive header/version");
    std::string previous;
    uint64_t count = 0;
    bool owner = false;
    while (read(1, data))
    {
        if (data == "E")
        {
            if (!read(8, data) || ReadNumber(data.data()) != count || !owner)
                return Failure("invalid archive record count/owner");
            const auto expected = crc ^ 0xffffffffU;
            if (!read(8, data, false) || ReadNumber(data.data()) != expected || remaining != 0)
                return Failure("archive checksum mismatch/trailing bytes");
            return ErrorCode::Success;
        }
        if (data != "R" || !read(16, data))
            return Failure("invalid/truncated archive record");
        const auto keyLength = ReadNumber(data.data());
        const auto valueLength = ReadNumber(data.data() + 8);
        if (keyLength == 0 || keyLength > MaxKey || valueLength > MaxValue || keyLength > remaining ||
            valueLength > remaining - keyLength)
            return Failure("invalid archive field lengths");
        std::string key, value;
        if (!read(static_cast<size_t>(keyLength), key) || !read(static_cast<size_t>(valueLength), value) ||
            !ValidSuffix(key, value) || (!previous.empty() && key <= previous))
            return Failure("invalid archive key/order/payload");
        owner = owner || key == "o";
        previous = key;
        ++count;
        if (visitor)
        {
            auto error = visitor(key, value);
            if (error != ErrorCode::Success)
                return error;
        }
    }
    return Failure("truncated archive footer");
}

class ManagedTiKVView : public Helper::KeyValueIO
{
  public:
    ManagedTiKVView(std::shared_ptr<ManagedRawClient> client, std::string prefix)
        : m_client(std::move(client)), m_prefix(std::move(prefix))
    {
    }
    void ShutDown() override
    {
        std::atomic_store(&m_client, std::shared_ptr<ManagedRawClient>{});
    }
    bool Available() override
    {
        return std::atomic_load(&m_client) != nullptr;
    }
    Storage GetStorageType() const override
    {
        return Storage::TIKVIO;
    }
    // RPCs own their buffers; the batch queue is drained even on exceptions.
    // Caller PageBuffers/strings/AsyncReadRequests never enter an async RPC.
    bool CompletesRequestsBeforeReturn() const override
    {
        return true;
    }
    ErrorCode Get(const std::string &key, std::string *value, const std::chrono::microseconds &timeout,
                  std::vector<Helper::AsyncReadRequest> *) override
    {
        auto client = std::atomic_load(&m_client);
        return client && value ? client->Get(StringKey(key), *value, OperationDeadline(timeout))
                               : Failure("closed view/null output");
    }
    ErrorCode Get(SizeType key, std::string *value, const std::chrono::microseconds &timeout,
                  std::vector<Helper::AsyncReadRequest> *) override
    {
        auto client = std::atomic_load(&m_client);
        return client && value && key >= 0 ? client->Get(Key(key), *value, OperationDeadline(timeout))
                                           : Failure("closed view/invalid key/output");
    }
    ErrorCode Get(SizeType key, Helper::PageBuffer<uint8_t> &value, const std::chrono::microseconds &timeout,
                  std::vector<Helper::AsyncReadRequest> *requests, bool = true) override
    {
        value.SetAvailableSize(0);
        std::string bytes;
        auto error = Get(key, &bytes, timeout, requests);
        if (error == ErrorCode::Success)
            Copy(bytes, value);
        return error;
    }
    ErrorCode Put(const std::string &key, const std::string &value, const std::chrono::microseconds &timeout,
                  std::vector<Helper::AsyncReadRequest> *) override
    {
        auto client = std::atomic_load(&m_client);
        return client ? client->Put(StringKey(key), value, OperationDeadline(timeout)) : Failure("closed view");
    }
    ErrorCode Put(SizeType key, const std::string &value, const std::chrono::microseconds &timeout,
                  std::vector<Helper::AsyncReadRequest> *) override
    {
        auto client = std::atomic_load(&m_client);
        return client && key >= 0 ? client->Put(Key(key), value, OperationDeadline(timeout))
                                  : Failure("closed view/invalid key");
    }
    ErrorCode CompareAndSwap(const std::string &key, const std::string &value, bool absent, const std::string &previous,
                             const std::chrono::microseconds &timeout, std::vector<Helper::AsyncReadRequest> *,
                             bool *swapped, bool *actualAbsent, std::string *actual) override
    {
        auto client = std::atomic_load(&m_client);
        return client ? client->CAS(StringKey(key), value, absent, previous, swapped, actualAbsent, actual,
                                    OperationDeadline(timeout))
                      : Failure("closed view");
    }
    ErrorCode Merge(SizeType key, const std::string &value, const std::chrono::microseconds &timeout,
                    std::vector<Helper::AsyncReadRequest> *requests, int &size) override
    {
        return MergeWithCAS(key, value, timeout, requests, size);
    }
    ErrorCode MergeWithCAS(SizeType key, const std::string &value, const std::chrono::microseconds &timeout,
                           std::vector<Helper::AsyncReadRequest> *, int &size) override
    {
        auto client = std::atomic_load(&m_client);
        if (!client || key < 0 || value.empty())
            return Failure("closed view/invalid append");
        const auto deadline = OperationDeadline(timeout);
        std::string previous;
        auto error = client->Get(Key(key), previous, deadline);
        bool absent = error == ErrorCode::Key_NotFound;
        if (error != ErrorCode::Success && !absent)
            return error;
        for (int attempt = 0; attempt < 16 && std::chrono::system_clock::now() < deadline; ++attempt)
        {
            if (value.size() > MaxValue || previous.size() > MaxValue - value.size())
                return ErrorCode::MemoryOverFlow;
            auto merged = previous + value;
            bool swapped = false, actualAbsent = false;
            std::string actual;
            error = client->CAS(Key(key), merged, absent, previous, &swapped, &actualAbsent, &actual, deadline);
            if (error != ErrorCode::Success)
                return error;
            if (swapped)
            {
                size = static_cast<int>(merged.size());
                return ErrorCode::Success;
            }
            absent = actualAbsent;
            previous = absent ? std::string() : std::move(actual);
        }
        return ErrorCode::Key_Conflict;
    }
    ErrorCode Delete(SizeType key) override
    {
        auto client = std::atomic_load(&m_client);
        return client && key >= 0 ? client->Delete(Key(key), OperationDeadline(std::chrono::microseconds(0)))
                                  : Failure("closed view/invalid key");
    }
    ErrorCode DeleteRange(SizeType begin, SizeType end) override
    {
        auto client = std::atomic_load(&m_client);
        if (!client || begin < 0 || end < begin)
            return Failure("invalid numeric range");
        if (begin == end)
            return ErrorCode::Success;
        return client->Scan(Key(begin), Key(end), [&](const auto &key, const auto &) {
            return client->Delete(key, OperationDeadline(std::chrono::microseconds(0)));
        });
    }
    ErrorCode MultiGet(const std::vector<SizeType> &keys, std::vector<std::string> *values,
                       const std::chrono::microseconds &timeout,
                       std::vector<Helper::AsyncReadRequest> *requests) override
    {
        if (!values)
            return Failure("null MultiGet output");
        auto client = std::atomic_load(&m_client);
        if (!client)
            return Failure("closed view");
        std::vector<std::string> encoded;
        encoded.reserve(keys.size());
        for (auto key : keys)
        {
            if (key < 0)
                return Failure("invalid batch key");
            encoded.push_back(Key(key));
        }
        (void)requests;
        // Native TiKVIO applies timeout per RPC; retain a separate bounded batch/retry budget.
        return client->MultiGet(encoded, *values, OperationDeadline(std::chrono::microseconds(0)), timeout);
    }
    ErrorCode MultiGet(const std::vector<std::string> &keys, std::vector<std::string> *values,
                       const std::chrono::microseconds &timeout, std::vector<Helper::AsyncReadRequest> *) override
    {
        if (!values)
            return Failure("null MultiGet output");
        auto client = std::atomic_load(&m_client);
        if (!client)
            return Failure("closed view");
        std::vector<std::string> encoded;
        encoded.reserve(keys.size());
        for (const auto &key : keys)
            encoded.push_back(StringKey(key));
        return client->MultiGet(encoded, *values, OperationDeadline(std::chrono::microseconds(0)), timeout);
    }
    ErrorCode MultiGet(const std::vector<SizeType> &keys, std::vector<Helper::PageBuffer<uint8_t>> &values,
                       const std::chrono::microseconds &timeout,
                       std::vector<Helper::AsyncReadRequest> *requests) override
    {
        if (values.size() < keys.size())
            return Failure("undersized MultiGet buffers");
        for (size_t i = 0; i < keys.size(); ++i)
            values[i].SetAvailableSize(0);
        std::vector<std::string> bytes;
        auto error = MultiGet(keys, &bytes, timeout, requests);
        if (error != ErrorCode::Success)
            return error;
        for (size_t i = 0; i < keys.size(); ++i)
            Copy(bytes[i], values[i]);
        return ErrorCode::Success;
    }
    ErrorCode Checkpoint(std::string) override
    {
        return ErrorCode::Success;
    }

  private:
    static void Copy(const std::string &bytes, Helper::PageBuffer<uint8_t> &buffer)
    {
        if (!bytes.empty())
        {
            buffer.ReservePageBuffer(bytes.size());
            std::memcpy(buffer.GetBuffer(), bytes.data(), bytes.size());
        }
        buffer.SetAvailableSize(bytes.size());
    }
    std::string Key(SizeType key) const
    {
        return m_prefix + '\1' + Number(static_cast<uint64_t>(key));
    }
    std::string StringKey(const std::string &key) const
    {
        if (key.size() == sizeof(SizeType))
        {
            SizeType posting;
            std::memcpy(&posting, key.data(), sizeof(posting));
            if (posting >= 0)
                return Key(posting);
        }
        return m_prefix + '\2' + key;
    }
    std::shared_ptr<ManagedRawClient> m_client;
    const std::string m_prefix;
};

class ManagedTiKVBackend : public PostingBackend
{
  public:
    ManagedTiKVBackend(std::shared_ptr<ManagedRawClient> client, std::string nodePrefix, uint64_t work)
        : m_client(std::move(client)), m_nodePrefix(std::move(nodePrefix)), m_work(work),
          m_prefix(m_nodePrefix + Number(work))
    {
        m_client->WatchNamespace(m_prefix);
    }

    ErrorCode Initialize(const std::string &checkpoint)
    {
        std::unique_ptr<File> archive;
        if (!checkpoint.empty())
        {
            archive = std::make_unique<File>(
                ::open((std::filesystem::path(checkpoint) / ArchiveName).c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
            if (archive->fd < 0 || ReadArchive(archive->fd, [&](const auto &suffix, const auto &) {
                                       return m_prefix.size() + suffix.size() <= MaxKey
                                                  ? ErrorCode::Success
                                                  : Failure("archive key exceeds target namespace key limit");
                                   }) != ErrorCode::Success)
                return Failure("cannot validate standalone posting archive");
        }
        bool empty = true;
        auto error = m_client->Scan(m_prefix, PrefixEnd(m_prefix), [&](const auto &, const auto &) {
            empty = false;
            return ErrorCode::Fail;
        });
        if (!empty)
            return Failure("new work namespace already exists; refusing to overwrite");
        if (error != ErrorCode::Success)
            return error;
        bool swapped = false, absent = false;
        std::string actual;
        error = m_client->CAS(m_prefix + "o", OwnerValue, true, {}, &swapped, &absent, &actual,
                              OperationDeadline(std::chrono::microseconds(0)));
        if (error != ErrorCode::Success || !swapped)
            return Failure("cannot claim new work namespace");
        if (archive)
        {
            error = ReadArchive(archive->fd, [&](const auto &suffix, const auto &value) {
                if (suffix == "o")
                    return ErrorCode::Success;
                return m_client->Put(m_prefix + suffix, value, OperationDeadline(std::chrono::microseconds(0)));
            });
        }
        return error;
    }

    std::shared_ptr<Helper::KeyValueIO> View(const std::string &tenant, uint64_t index) override
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (tenant.size() > MaxKey || m_prefix.size() + tenant.size() + 32 > MaxKey)
            return {};
        auto prefix = ViewPrefix(tenant, index);
        if (m_client->Put(prefix + '\0', {}, OperationDeadline(std::chrono::microseconds(0))) != ErrorCode::Success)
            return {};
        return std::make_shared<ManagedTiKVView>(m_client, std::move(prefix));
    }

    ErrorCode Checkpoint(const std::string &directory) override
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        try
        {
            if (m_client->HasUncertainMutation())
                return Failure("cannot snapshot after an uncertain mutation; recover into a new work generation");
            if (directory.empty())
                return Failure("empty checkpoint directory");
            std::filesystem::create_directories(directory);
            const auto path = std::filesystem::path(directory) / ArchiveName;
            const auto staging = std::filesystem::path(directory) / (std::string(ArchiveName) + ".incomplete");
            if (std::filesystem::exists(path))
                return Failure("immutable archive already exists");
            File output(::open(staging.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600));
            if (output.fd < 0)
                return Failure("cannot create checkpoint archive");
            StagingCleanup cleanup(staging);
            uint32_t crc = 0xffffffffU;
            uint64_t count = 0;
            bool owner = false;
            bool ok = Write(output.fd, ArchiveMagic, crc);
            auto error = ok ? m_client->Scan(
                                  m_prefix, PrefixEnd(m_prefix),
                                  [&](const std::string &key, const std::string &value) {
                                      const auto suffix = key.substr(m_prefix.size());
                                      if (!ValidSuffix(suffix, value))
                                          return Failure("unrecognized owned namespace key");
                                      owner = owner || suffix == "o";
                                      if (!Write(output.fd, "R" + Number(suffix.size()) + Number(value.size()), crc) ||
                                          !Write(output.fd, suffix, crc) || !Write(output.fd, value, crc))
                                          return ErrorCode::DiskIOFail;
                                      ++count;
                                      return ErrorCode::Success;
                                  })
                            : ErrorCode::DiskIOFail;
            if (error == ErrorCode::Success && !owner)
                error = Failure("work ownership marker missing");
            if (error == ErrorCode::Success)
            {
                ok = Write(output.fd, "E" + Number(count), crc);
                const auto checksum = crc ^ 0xffffffffU;
                ok = ok && Write(output.fd, Number(checksum), crc) && ::fchmod(output.fd, 0444) == 0 &&
                     ::fsync(output.fd) == 0;
                if (!ok)
                    error = ErrorCode::DiskIOFail;
            }
            // link rather than rename: never replace an existing immutable checkpoint.
            if (error == ErrorCode::Success && ::link(staging.c_str(), path.c_str()) != 0)
                error = ErrorCode::DiskIOFail;
            if (!cleanup.Remove() && error == ErrorCode::Success)
                error = ErrorCode::DiskIOFail;
            if (error == ErrorCode::Success)
            {
                File dir(::open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
                if (dir.fd < 0 || ::fsync(dir.fd) != 0)
                    error = ErrorCode::DiskIOFail;
            }
            return error;
        }
        catch (const std::exception &)
        {
            return Failure("checkpoint filesystem/allocation exception");
        }
    }

    ErrorCode DeleteIndex(const std::string &tenant, uint64_t index) override
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (tenant.size() > MaxKey || m_prefix.size() + tenant.size() + 32 > MaxKey)
            return Failure("oversized view namespace");
        auto prefix = ViewPrefix(tenant, index);
        return DeleteNamespace(prefix);
    }

    ErrorCode RemoveWork(uint64_t work) override
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (work >= m_work)
            return Failure("only older registered work may be removed");
        const auto prefix = m_nodePrefix + Number(work);
        std::string owner;
        auto error = m_client->Get(prefix + "o", owner, OperationDeadline(std::chrono::microseconds(0)));
        if (error != ErrorCode::Success && error != ErrorCode::Key_NotFound)
            return error;
        if (error == ErrorCode::Success && owner != OwnerValue)
            return Failure("unexpected work ownership marker");
        // The manager supplies only registered orphan generations. A missing
        // marker must not prevent cleaning partial imports or late old RPCs.
        error = m_client->Scan(prefix, PrefixEnd(prefix), [&](const auto &key, const auto &) {
            if (key == prefix + "o")
                return ErrorCode::Success;
            return m_client->Delete(key, OperationDeadline(std::chrono::microseconds(0)));
        });
        // Keep the marker until all other keys are deleted so partial failures are retryable.
        return error == ErrorCode::Success
                   ? m_client->Delete(prefix + "o", OperationDeadline(std::chrono::microseconds(0)))
                   : error;
    }

  private:
    std::string ViewPrefix(const std::string &tenant, uint64_t index) const
    {
        return m_prefix + "v" + Number(tenant.size()) + tenant + Number(index);
    }
    ErrorCode DeleteNamespace(const std::string &prefix)
    {
        return m_client->Scan(prefix, PrefixEnd(prefix), [&](const auto &key, const auto &) {
            return m_client->Delete(key, OperationDeadline(std::chrono::microseconds(0)));
        });
    }
    std::shared_ptr<ManagedRawClient> m_client;
    const std::string m_nodePrefix;
    const uint64_t m_work;
    const std::string m_prefix;
    std::mutex m_mutex;
};
} // namespace

ErrorCode OpenTiKVPostingBackend(const PostingBackendOptions &options, const std::string &nodeIdentity, uint64_t work,
                                 const std::string &checkpointDirectory, std::shared_ptr<PostingBackend> &result)
{
    result.reset();
    try
    {
        if (nodeIdentity.empty() || nodeIdentity.size() > 4096 || options.tikvKeyPrefix.size() > 4096)
            return Failure("missing/oversized persisted node identity or namespace prefix");
        auto client = std::make_shared<ManagedRawClient>(options.tikvPDAddresses);
        auto error = client->Open();
        if (error != ErrorCode::Success)
            return error;
        const std::string format("\0SPTAG-managed-posting-v1\0", 26);
        auto backend = std::make_shared<ManagedTiKVBackend>(
            client, options.tikvKeyPrefix + format + Number(nodeIdentity.size()) + nodeIdentity, work);
        error = backend->Initialize(checkpointDirectory);
        if (error == ErrorCode::Success)
            result = std::move(backend);
        return error;
    }
    catch (const std::exception &)
    {
        return Failure("backend initialization exception");
    }
}
} // namespace SPTAG::SPANN

#else
namespace SPTAG::SPANN
{
ErrorCode OpenTiKVPostingBackend(const PostingBackendOptions &, const std::string &, uint64_t, const std::string &,
                                 std::shared_ptr<PostingBackend> &result)
{
    result.reset();
    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Managed TiKV backend requires a build with TIKV enabled\n");
    return ErrorCode::Fail;
}
} // namespace SPTAG::SPANN
#endif // TIKV
#endif // ROCKSDB
