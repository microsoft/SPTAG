// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Helper/AsyncFileReader.h"

namespace SPTAG {
    namespace Helper {
#ifndef _MSC_VER
        void SetThreadAffinity(int threadID, std::thread& thread, char socketStrategy, char idStrategy)
        {
#ifdef NUMA
            int numGroups = numa_num_task_nodes();
            int numCpus = numa_num_task_cpus() / numGroups;

            int group = threadID / numCpus;
            int cpuid = threadID % numCpus;
            if (socketStrategy == 1) {
                group = threadID % numGroups;
                cpuid = (threadID / numGroups) % numCpus;
            }

            struct bitmask* cpumask = numa_allocate_cpumask();
            if (!numa_node_to_cpus(group, cpumask)) {
                unsigned int nodecpu = 0;
                for (unsigned int i = 0; i < cpumask->size; i++) {
                    if (numa_bitmask_isbitset(cpumask, i)) {
                        if (cpuid == nodecpu) {
                            cpu_set_t cpuset;
                            CPU_ZERO(&cpuset);
                            CPU_SET(i, &cpuset);
                            int rc = pthread_setaffinity_np(thread.native_handle(), sizeof(cpu_set_t), &cpuset);
                            if (rc != 0) {
                                LOG(Helper::LogLevel::LL_Error, "Error calling pthread_setaffinity_np for thread %d: %d\n", threadID, rc);
                            }
                            break;
                        }
                        nodecpu++;
                    }
                }
            }
#else
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(threadID, &cpuset);
            int rc = pthread_setaffinity_np(thread.native_handle(), sizeof(cpu_set_t), &cpuset);
            if (rc != 0) {
                LOG(Helper::LogLevel::LL_Error, "Error calling pthread_setaffinity_np for thread %d: %d\n", threadID, rc);
            }
#endif
        }

        struct timespec AIOTimeout {0, 30000};
        void BatchReadFileAsync(std::vector<std::shared_ptr<Helper::DiskIO>>& handlers, AsyncReadRequest* readRequests, int num)
        {
            std::vector<struct iocb> myiocbs(num);
            std::vector<std::vector<struct iocb*>> iocbs(handlers.size());
            std::vector<int> submitted(handlers.size(), 0);
            std::vector<int> done(handlers.size(), 0);
            int totalToSubmit = 0, channel = 0;

            memset(myiocbs.data(), 0, num * sizeof(struct iocb));
            for (int i = 0; i < num; i++) {
                AsyncReadRequest* readRequest = &(readRequests[i]);

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
                            }
                        }
                    }
                }

                for (int i = totalQueued; i < totalDone; i++) {
                    AsyncReadRequest* req = reinterpret_cast<AsyncReadRequest*>((events[i].data));
                    if (nullptr != req)
                    {
                        req->m_callback(true);
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
                    req->m_callback(true);
                }
            }
        }
#else
        ULONGLONG GetCpuMasks(WORD group, DWORD numCpus)
        {
            ULONGLONG masks = 0, mask = 1;
            for (DWORD i = 0; i < numCpus; ++i)
            {
                masks |= mask;
                mask <<= 1;
            }

            return masks;
        }

        void SetThreadAffinity(int threadID, std::thread& thread, char socketStrategy, char idStrategy)
        {
            WORD numGroups = GetActiveProcessorGroupCount();
            DWORD numCpus = GetActiveProcessorCount(0);

            GROUP_AFFINITY ga;
            memset(&ga, 0, sizeof(ga));
            PROCESSOR_NUMBER pn;
            memset(&pn, 0, sizeof(pn));

            WORD group = (WORD)(threadID / numCpus);
            pn.Number = (BYTE)(threadID % numCpus);
            if (socketStrategy == 1) {
                group = (WORD)(threadID % numGroups);
                pn.Number = (BYTE)((threadID / numGroups) % numCpus);
            }

            ga.Group = group;
            ga.Mask = GetCpuMasks(group, numCpus);
            BOOL res = SetThreadGroupAffinity(GetCurrentThread(), &ga, NULL);
            if (!res)
            {
                LOG(Helper::LogLevel::LL_Error, "Failed SetThreadGroupAffinity for group %d and mask %I64x for thread %d.\n", ga.Group, ga.Mask, threadID);
                return;
            }
            pn.Group = group;
            if (idStrategy == 1) {
                pn.Number = (BYTE)(numCpus - 1 - pn.Number);
            }
            res = SetThreadIdealProcessorEx(GetCurrentThread(), &pn, NULL);
            if (!res)
            {
                LOG(Helper::LogLevel::LL_Error, "Unable to set ideal processor for thread %d.\n", threadID);
                return;
            }
            YieldProcessor();
        }

        void BatchReadFileAsync(std::vector<std::shared_ptr<Helper::DiskIO>>& handlers, AsyncReadRequest* readRequests, int num)
        {
            if (handlers.size() == 1) {
                handlers[0]->BatchReadFile(readRequests, num);
            }
            else {
                int currFileId = 0, currReqStart = 0;
                for (int i = 0; i < num; i++) {
                    AsyncReadRequest* readRequest = &(readRequests[i]);

                    int fileid = (readRequest->m_status >> 16);
                    if (fileid != currFileId) {
                        handlers[currFileId]->BatchReadFile(readRequests + currReqStart, i - currReqStart);
                        currFileId = fileid;
                        currReqStart = i;
                    }
                }
                if (currReqStart < num) {
                    handlers[currFileId]->BatchReadFile(readRequests + currReqStart, num - currReqStart);
                }
            }
        }
#endif
    }
}
