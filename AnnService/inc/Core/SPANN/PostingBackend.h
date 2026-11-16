// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef MULTITENANT
#include "inc/Core/SPANN/SharedRocksDB.h"

namespace SPTAG::SPANN
{
struct PostingBackendOptions
{
    Storage storage = Storage::ROCKSDBIO;
    std::string tikvPDAddresses = "127.0.0.1:2379";
    std::string tikvKeyPrefix = "sptag/tenant/";
};

// Only the node owner checkpoints or deletes namespaces. Index views own no engine lifecycle.
class PostingBackend
{
  public:
    virtual ~PostingBackend() = default;
    virtual std::shared_ptr<Helper::KeyValueIO> View(const std::string &tenant, uint64_t index) = 0;
    // Called with managed updates drained. Searches may continue.
    // Produces a standalone immutable image, not a reference to mutable working state.
    virtual ErrorCode Checkpoint(const std::string &directory) = 0;
    virtual ErrorCode DeleteIndex(const std::string &tenant, uint64_t index) = 0;
    virtual ErrorCode RemoveWork(uint64_t work)
    {
        return ErrorCode::Success;
    }
};

ErrorCode OpenFilePostingBackend(const std::string &workDirectory, const std::string &checkpointDirectory,
                                 std::shared_ptr<PostingBackend> &result);
ErrorCode OpenTiKVPostingBackend(const PostingBackendOptions &options, const std::string &nodeIdentity, uint64_t work,
                                 const std::string &checkpointDirectory, std::shared_ptr<PostingBackend> &result);
} // namespace SPTAG::SPANN
#endif
