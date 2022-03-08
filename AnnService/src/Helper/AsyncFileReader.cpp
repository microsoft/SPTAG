// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Helper/AsyncFileReader.h"

namespace SPTAG {
    namespace Helper {
#ifndef _MSC_VER
        struct timespec AIOTimeout {0, 30000};
        int BatchReadFileAsync(std::vector<std::shared_ptr<Helper::DiskPriorityIO>>& handlers, AsyncReadRequest* readRequests, int num)
        {
            std::vector<struct iocb> myiocbs(num);
            std::vector<std::vector<struct iocb*>> iocbs(handlers.size());
            std::vector<int> submitted(handlers.size(), 0);
            std::vector<int> done(handlers.size(), 0);
            int totalToSubmit = 0, channel = 0;

            memset(myiocbs.data(), 0, num * sizeof(struct iocb));
            for (int i = 0; i < num; i++) {
                AsyncReadRequest* readRequest = &(readRequests[i]);
                if (readRequest->m_readSize == 0) continue;

                channel = readRequest->m_status & 0xffff;
                int fileid = (readRequest->m_status >> 16);

                struct iocb* myiocb = &(myiocbs[totalToSubmit++]);
                myiocb->aio_data = reinterpret_cast<uintptr_t>(readRequest);
                myiocb->aio_lio_opcode = IOCB_CMD_PREAD;
                myiocb->aio_fildes = ((AsyncFileIO*)(handlers[fileid].get()))->GetFileHandler();
                myiocb->aio_buf = (std::uint64_t)(readRequest->m_buffer);
                myiocb->aio_nbytes = readRequest->m_readSize;
                myiocb->aio_offset = static_cast<std::int64_t>(readRequest->m_offset);

                iocbs[fileid].emplace_back(myiocb);
            }
            std::vector<struct io_event> events(totalToSubmit);
            int totalDone = 0, totalSubmitted = 0, totalQueued = 0;
            while (totalDone < totalToSubmit) {
                if (totalSubmitted < totalToSubmit) {
                    for (int i = 0; i < handlers.size(); i++) {
                        if (submitted[i] < iocbs[i].size()) {
                            AsyncFileIO* handler = (AsyncFileIO*)(handlers[i].get());
                            int s = syscall(__NR_io_submit, handler->GetIOCP(channel), iocbs[i].size() - submitted[i], iocbs[i].data() + submitted[i]);
                            if (s > 0) {
                                submitted[i] += s;
                                totalSubmitted += s;
                            }
                            else {
                                LOG(Helper::LogLevel::LL_Error, "fid:%d channel %d, to submit:%d, submitted:%s\n", i, channel, iocbs[i].size() - submitted[i], strerror(-s));
                                return totalDone;
                            }
                        }
                    }
                }

                for (int i = totalQueued; i < totalDone; i++) {
                    AsyncReadRequest* req = reinterpret_cast<AsyncReadRequest*>((events[i].data));
                    if (nullptr != req)
                    {
                        req->m_callback(req);
                    }
                }
                totalQueued = totalDone;

                for (int i = 0; i < handlers.size(); i++) {
                    if (done[i] < submitted[i]) {
                        int wait = submitted[i] - done[i];
                        AsyncFileIO* handler = (AsyncFileIO*)(handlers[i].get());
                        auto d = syscall(__NR_io_getevents, handler->GetIOCP(channel), wait, wait, events.data() + totalDone, &AIOTimeout);
                        done[i] += d;
                        totalDone += d;
                    }
                }
            }

            for (int i = totalQueued; i < totalDone; i++) {
                AsyncReadRequest* req = reinterpret_cast<AsyncReadRequest*>((events[i].data));
                if (nullptr != req)
                {
                    req->m_callback(req);
                }
            }
            return totalDone;
        }
#endif
    }
}
