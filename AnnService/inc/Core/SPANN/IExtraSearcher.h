// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_SPANN_IEXTRASEARCHER_H_
#define _SPTAG_SPANN_IEXTRASEARCHER_H_

#include "Options.h"

#include "inc/Core/VectorIndex.h"
#include "inc/Core/Common/IVersionMap.h"
#include "inc/Core/Common/VersionLabel.h"
#include "inc/Helper/AsyncFileReader.h"
#include "inc/Helper/VectorSetReader.h"
#include "inc/Helper/ConcurrentSet.h"
#include <memory>
#include <vector>
#include <chrono>
#include <atomic>
#include <array>
#include <set>

namespace SPTAG {
    namespace SPANN {

        struct SearchStats
        {
            SearchStats()
                : m_check(0),
                m_exCheck(0),
                m_totalListElementsCount(0),
                m_diskIOCount(0),
                m_diskAccessCount(0),
                m_totalSearchLatency(0),
                m_totalLatency(0),
                m_exLatency(0),
                m_asyncLatency0(0),
                m_asyncLatency1(0),
                m_asyncLatency2(0),
                m_queueLatency(0),
                m_sleepLatency(0),
                m_compLatency(0),
                m_diskReadLatency(0),
                m_versionMapLatency(0),
                m_exSetUpLatency(0),
                m_threadID(0)
            {
                ResetBreakdown();
            }

            static constexpr int kSearchLatencyBreakdownLayers = 16;

            void ResetBreakdown()
            {
                m_layerSetupLatency.fill(0);
                m_layerPostingReadLatency.fill(0);
                m_layerCompLatency.fill(0);
                m_layerVersionMapLatency.fill(0);
                m_layerTotalLatency.fill(0);
                m_layerPostingCount.fill(0);
                m_layerListElementsCount.fill(0);
                m_layerVersionCheckCount.fill(0);
                m_layerDiskAccessCount.fill(0);
            }

            static bool IsValidBreakdownLayer(int layer)
            {
                return layer >= 0 && layer < kSearchLatencyBreakdownLayers;
            }

            int m_check;

            int m_exCheck;

            int m_totalListElementsCount;

            int m_diskIOCount;

            int m_diskAccessCount;

            double m_totalSearchLatency;

            double m_totalLatency;

            double m_exLatency;

            double m_asyncLatency0;

            double m_asyncLatency1;

            double m_asyncLatency2;

            double m_queueLatency;

            double m_sleepLatency;

            double m_compLatency;

            double m_diskReadLatency;

            double m_versionMapLatency;

            double m_exSetUpLatency;

            std::array<double, kSearchLatencyBreakdownLayers> m_layerSetupLatency;

            std::array<double, kSearchLatencyBreakdownLayers> m_layerPostingReadLatency;

            std::array<double, kSearchLatencyBreakdownLayers> m_layerCompLatency;

            std::array<double, kSearchLatencyBreakdownLayers> m_layerVersionMapLatency;

            std::array<double, kSearchLatencyBreakdownLayers> m_layerTotalLatency;

            std::array<int, kSearchLatencyBreakdownLayers> m_layerPostingCount;

            std::array<int, kSearchLatencyBreakdownLayers> m_layerListElementsCount;

            std::array<int, kSearchLatencyBreakdownLayers> m_layerVersionCheckCount;

            std::array<int, kSearchLatencyBreakdownLayers> m_layerDiskAccessCount;

            std::chrono::steady_clock::time_point m_searchRequestTime;

            int m_threadID;
        };

        struct IndexStats {
            std::atomic_uint32_t m_headMiss{ 0 };
            uint32_t m_appendTaskNum{ 0 };
            uint32_t m_splitNum{ 0 };
            uint32_t m_theSameHeadNum{ 0 };
            uint32_t m_reAssignNum{ 0 };
            uint32_t m_garbageNum{ 0 };
            uint64_t m_reAssignScanNum{ 0 };
            uint32_t m_mergeNum{ 0 };

            //Split
            double m_splitCost{ 0 };
            double m_getCost{ 0 };
            double m_putCost{ 0 };
            double m_clusteringCost{ 0 };
            double m_updateHeadCost{ 0 };
            double m_reassignScanCost{ 0 };
            double m_reassignScanIOCost{ 0 };

            // Append
            double m_appendCost{ 0 };
            double m_appendIOCost{ 0 };

            // reAssign
            double m_reAssignCost{ 0 };
            double m_selectCost{ 0 };
            double m_reAssignAppendCost{ 0 };

            // GC
            double m_garbageCost{ 0 };

            // ----- Diagnostic instrumentation (Append RMW + Split lock) -----
            // Histogram has 22 buckets covering log2 ranges:
            //   bucket i = [2^i, 2^(i+1))   (units: microseconds for time, bytes for size)
            //   i=0    : [1us, 2us)         /  [1B, 2B)
            //   i=10   : [1ms, 2ms)         /  [1KB, 2KB)
            //   i=20   : [1s,  2s)          /  [1MB, 2MB)
            //   i=21   : [>=2s)             /  [>=2MB)   (tail bucket, no upper)
            // Buckets are atomic so any thread can update without locking.
            static constexpr int kHistBuckets = 22;
            std::atomic_uint64_t m_appendLockWaitUs[kHistBuckets]{};   // A. lock contention
            std::atomic_uint64_t m_appendGetUs[kHistBuckets]{};        // B. RMW Get latency (single-chunk)
            std::atomic_uint64_t m_appendPutUs[kHistBuckets]{};        // B+C. RMW Put latency
            std::atomic_uint64_t m_appendPostingBytes[kHistBuckets]{}; // RMW posting size after Get+append
            std::atomic_uint64_t m_splitLockWaitUs[kHistBuckets]{};    // A. Split lock contention
            std::atomic_uint64_t m_appendLockWaitTotalUs{ 0 };
            std::atomic_uint64_t m_appendGetTotalUs{ 0 };
            std::atomic_uint64_t m_appendPutTotalUs{ 0 };
            std::atomic_uint64_t m_appendPostingBytesTotal{ 0 };
            std::atomic_uint64_t m_splitLockWaitTotalUs{ 0 };
            std::atomic_uint64_t m_appendRmwSampleCount{ 0 };          // # of RMWs that recorded above
            std::atomic_uint64_t m_splitLockSampleCount{ 0 };

            // Reassign job latency: one ReassignAsyncJob.exec() = RNGSelection + replicaCount*Append
            // Submitted via MergePostings reassign path (per-vector ReassignAsync).
            // Batched reassigns from CollectReAssign go through Append directly and are
            // already covered by m_appendGetUs / m_appendPutUs above.
            std::atomic_uint64_t m_reassignJobLatencyUs[kHistBuckets]{};
            std::atomic_uint64_t m_reassignJobLatencyTotalUs{ 0 };
            std::atomic_uint64_t m_reassignJobSampleCount{ 0 };

            // Per-source reassign submission counters (cumulative).
            //   from_merge   : ReassignAsync invoked inside MergePostings
            //   from_split   : Append calls inside CollectReAssign batched path (per-vector count)
            std::atomic_uint64_t m_reassignSubmittedFromMerge{ 0 };
            std::atomic_uint64_t m_reassignSubmittedFromSplitBatch{ 0 };

            // Worker thread pool queue depth — sampled (throttled to 1 per 50ms) at AllFinished entry.
            //   m_queueDepth   : total pending jobs in m_splitThreadPool->jobsize()
            //   m_runningJobs  : currently executing jobs (workers busy)
            std::atomic_uint64_t m_queueDepthHist[kHistBuckets]{};
            std::atomic_uint64_t m_runningJobsHist[kHistBuckets]{};
            std::atomic_uint64_t m_queueDepthTotal{ 0 };
            std::atomic_uint64_t m_runningJobsTotal{ 0 };
            std::atomic_uint64_t m_queueDepthSampleCount{ 0 };
            // Last-sample timestamp (us since epoch) for throttling — atomic CAS to avoid herd.
            std::atomic<uint64_t> m_queueDepthLastSampleUs{ 0 };

            // Pre-append posting size: distribution of posting size as observed AT the moment
            // an Append RMW reads it. Together with m_appendPostingBytes (post-append) this
            // reveals how many RMWs are working on near-threshold postings (the "split feedback
            // loop" smoking gun explaining why incremental insert is so much slower than offline
            // build of the same final size).
            std::atomic_uint64_t m_appendPreBytes[kHistBuckets]{};
            std::atomic_uint64_t m_appendPreBytesTotal{ 0 };
            // Cumulative counters: how many RMWs ended up triggering a SplitAsync, and
            // how many were already at >=80% of m_postingSizeLimit when read ("near-threshold").
            std::atomic_uint64_t m_appendTriggeredSplit{ 0 };
            std::atomic_uint64_t m_appendNearThreshold{ 0 };
            // [SAFETY] How many Append RMWs aborted because the TiKV Get returned
            // a real failure (NOT NotFound). Before the fix these were silently
            // turned into empty postings and re-Put, permanently overwriting any
            // existing vectors under that head. Should stay at 0 in healthy runs.
            std::atomic_uint64_t m_appendGetFail{ 0 };

            // Per-batch split/reassign diagnostics. These are reset after each
            // ALL DONE boundary so every log line describes the just-finished
            // insert batch for that layer.
            std::atomic_uint64_t m_splitPostingVectors[kHistBuckets]{};
            std::atomic_uint64_t m_splitNewHeadCount[kHistBuckets]{};
            std::atomic_uint64_t m_splitReassignVectors[kHistBuckets]{};
            std::atomic_uint64_t m_splitReassignRecords[kHistBuckets]{};
            std::atomic_uint64_t m_splitReassignTargetHeads[kHistBuckets]{};
            std::atomic_uint64_t m_splitPostingVectorsTotal{ 0 };
            std::atomic_uint64_t m_splitNewHeadCountTotal{ 0 };
            std::atomic_uint64_t m_splitReassignVectorsTotal{ 0 };
            std::atomic_uint64_t m_splitReassignRecordsTotal{ 0 };
            std::atomic_uint64_t m_splitReassignTargetHeadsTotal{ 0 };
            std::atomic_uint64_t m_splitPostingVectorSampleCount{ 0 };
            std::atomic_uint64_t m_splitNewHeadSampleCount{ 0 };
            std::atomic_uint64_t m_splitReassignSampleCount{ 0 };
            std::atomic_uint64_t m_splitReassignRecordSampleCount{ 0 };
            std::atomic_uint64_t m_splitReassignTargetHeadSampleCount{ 0 };
            std::atomic_uint64_t m_splitSameHeadCount{ 0 };
            std::atomic_uint64_t m_splitExistingHeadMergeCount{ 0 };
            std::atomic_uint64_t m_splitExistingHeadMergeResplitCount{ 0 };
            std::atomic_uint64_t m_splitCreatedNewHeadCount{ 0 };

            // [DIAG-MC] Multi-chunk path histograms (storage=TIKVIO + UseMultiChunkPosting=true).
            // The single-key histograms above (m_appendGetUs/m_appendPutUs/...) stay at 0
            // in MC mode because the MC branch in Append() does NOT do Get+Put. These
            // separate histograms instrument the MC-specific RPCs so we can see which
            // call dominates worker time.
            //   m_mcAppendUs            : AppendChunkAndUpdateCount() end-to-end (lock-held; PutChunkAndCount RPC)
            //   m_mcGetCountMissUs      : GetCachedPostingCount() TiKV fallback latency (cache miss only)
            //   m_mcSplitDelUs          : Split's DeletePosting (DeleteRange over chunk span)
            //   m_mcSplitPutBaseUs      : Split's PutBaseChunk
            //   m_mcSplitSetCountUs     : Split's SetPostingCount
            std::atomic_uint64_t m_mcAppendUs[kHistBuckets]{};
            std::atomic_uint64_t m_mcGetCountMissUs[kHistBuckets]{};
            std::atomic_uint64_t m_mcSplitDelUs[kHistBuckets]{};
            std::atomic_uint64_t m_mcSplitPutBaseUs[kHistBuckets]{};
            std::atomic_uint64_t m_mcSplitSetCountUs[kHistBuckets]{};
            std::atomic_uint64_t m_mcAppendTotalUs{ 0 };
            std::atomic_uint64_t m_mcGetCountMissTotalUs{ 0 };
            std::atomic_uint64_t m_mcSplitDelTotalUs{ 0 };
            std::atomic_uint64_t m_mcSplitPutBaseTotalUs{ 0 };
            std::atomic_uint64_t m_mcSplitSetCountTotalUs{ 0 };
            std::atomic_uint64_t m_mcAppendSampleCount{ 0 };
            std::atomic_uint64_t m_mcGetCountCacheHit{ 0 };
            std::atomic_uint64_t m_mcGetCountCacheMiss{ 0 };
            std::atomic_uint64_t m_mcSplitWriteSampleCount{ 0 };  // # of MC PutPostingToDB invocations

            static int HistBucketOf(uint64_t v) {
                if (v == 0) return 0;
                int b = 0;
                uint64_t x = v;
                while (x > 1) { x >>= 1; ++b; }
                if (b >= kHistBuckets) b = kHistBuckets - 1;
                return b;
            }
            static void HistAdd(std::atomic_uint64_t* hist, uint64_t v) {
                hist[HistBucketOf(v)].fetch_add(1, std::memory_order_relaxed);
            }
            static std::string FormatHist(const char* label, std::atomic_uint64_t* hist,
                                          uint64_t totalSum, uint64_t sampleCount,
                                          const char* unit) {
                char buf[2048];
                int n = snprintf(buf, sizeof(buf), "  %s: count=%lu avg=%.2f%s histo[bucket=count]:",
                                 label, (unsigned long)sampleCount,
                                 sampleCount ? (double)totalSum / sampleCount : 0.0, unit);
                for (int i = 0; i < kHistBuckets && n < (int)sizeof(buf); ++i) {
                    uint64_t c = hist[i].load(std::memory_order_relaxed);
                    if (c == 0) continue;
                    uint64_t lo = (i == 0) ? 0ULL : (1ULL << i);
                    n += snprintf(buf + n, sizeof(buf) - n, " %lu%s+:%lu",
                                  (unsigned long)lo, (i == kHistBuckets - 1) ? ">=" : "", (unsigned long)c);
                }
                return std::string(buf);
            }

            static std::string FormatHistAndReset(const char* label, std::atomic_uint64_t* hist,
                                                  std::atomic_uint64_t& totalSum,
                                                  std::atomic_uint64_t& sampleCount,
                                                  const char* unit) {
                uint64_t counts[kHistBuckets];
                for (int i = 0; i < kHistBuckets; ++i) {
                    counts[i] = hist[i].exchange(0, std::memory_order_relaxed);
                }
                uint64_t total = totalSum.exchange(0, std::memory_order_relaxed);
                uint64_t samples = sampleCount.exchange(0, std::memory_order_relaxed);

                char buf[2048];
                int n = snprintf(buf, sizeof(buf), "  %s: count=%lu avg=%.2f%s histo[bucket=count]:",
                                 label, (unsigned long)samples,
                                 samples ? (double)total / samples : 0.0, unit);
                for (int i = 0; i < kHistBuckets && n < (int)sizeof(buf); ++i) {
                    uint64_t c = counts[i];
                    if (c == 0) continue;
                    uint64_t lo = (i == 0) ? 0ULL : (1ULL << i);
                    n += snprintf(buf + n, sizeof(buf) - n, " %lu%s+:%lu",
                                  (unsigned long)lo, (i == kHistBuckets - 1) ? ">=" : "", (unsigned long)c);
                }
                return std::string(buf);
            }

            void PrintStat(int finishedInsert, bool cost = false, bool reset = false) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "After %d insertion, head vectors split %d times, head missing %d times, same head %d times, reassign %d times, reassign scan %ld times, garbage collection %d times, merge %d times\n",
                    finishedInsert, m_splitNum, m_headMiss.load(), m_theSameHeadNum, m_reAssignNum, m_reAssignScanNum, m_garbageNum, m_mergeNum);

                if (cost) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "AppendTaskNum: %d, TotalCost: %.3lf us, PerCost: %.3lf us\n", m_appendTaskNum, m_appendCost, m_appendCost / m_appendTaskNum);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "AppendTaskNum: %d, AppendIO TotalCost: %.3lf us, PerCost: %.3lf us\n", m_appendTaskNum, m_appendIOCost, m_appendIOCost / m_appendTaskNum);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SplitNum: %d, TotalCost: %.3lf ms, PerCost: %.3lf ms\n", m_splitNum, m_splitCost, m_splitCost / m_splitNum);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SplitNum: %d, Read TotalCost: %.3lf us, PerCost: %.3lf us\n", m_splitNum, m_getCost, m_getCost / m_splitNum);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SplitNum: %d, Clustering TotalCost: %.3lf us, PerCost: %.3lf us\n", m_splitNum, m_clusteringCost, m_clusteringCost / m_splitNum);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SplitNum: %d, UpdateHead TotalCost: %.3lf ms, PerCost: %.3lf ms\n", m_splitNum, m_updateHeadCost, m_updateHeadCost / m_splitNum);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SplitNum: %d, Write TotalCost: %.3lf us, PerCost: %.3lf us\n", m_splitNum, m_putCost, m_putCost / m_splitNum);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SplitNum: %d, ReassignScan TotalCost: %.3lf ms, PerCost: %.3lf ms\n", m_splitNum, m_reassignScanCost, m_reassignScanCost / m_splitNum);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SplitNum: %d, ReassignScanIO TotalCost: %.3lf us, PerCost: %.3lf us\n", m_splitNum, m_reassignScanIOCost, m_reassignScanIOCost / m_splitNum);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "GCNum: %d, TotalCost: %.3lf us, PerCost: %.3lf us\n", m_garbageNum, m_garbageCost, m_garbageCost / m_garbageNum);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "ReassignNum: %d, TotalCost: %.3lf us, PerCost: %.3lf us\n", m_reAssignNum, m_reAssignCost, m_reAssignCost / m_reAssignNum);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "ReassignNum: %d, Select TotalCost: %.3lf us, PerCost: %.3lf us\n", m_reAssignNum, m_selectCost, m_selectCost / m_reAssignNum);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "ReassignNum: %d, ReassignAppend TotalCost: %.3lf us, PerCost: %.3lf us\n", m_reAssignNum, m_reAssignAppendCost, m_reAssignAppendCost / m_reAssignNum);
                }

                // Diagnostic histograms — independent of `cost` so they print every PrintStat
                {
                    uint64_t rmwN = m_appendRmwSampleCount.load();
                    uint64_t splN = m_splitLockSampleCount.load();
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[DIAG] %s\n",
                        FormatHist("AppendLockWait", m_appendLockWaitUs, m_appendLockWaitTotalUs.load(), rmwN, "us").c_str());
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[DIAG] %s\n",
                        FormatHist("AppendGetUs",    m_appendGetUs,      m_appendGetTotalUs.load(),     rmwN, "us").c_str());
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[DIAG] %s\n",
                        FormatHist("AppendPutUs",    m_appendPutUs,      m_appendPutTotalUs.load(),     rmwN, "us").c_str());
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[DIAG] %s\n",
                        FormatHist("AppendPostBytes",m_appendPostingBytes,m_appendPostingBytesTotal.load(), rmwN, "B").c_str());
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[DIAG] %s\n",
                        FormatHist("SplitLockWait",  m_splitLockWaitUs,  m_splitLockWaitTotalUs.load(),  splN, "us").c_str());
                    uint64_t reassignN = m_reassignJobSampleCount.load();
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[DIAG] %s\n",
                        FormatHist("ReassignJobUs",  m_reassignJobLatencyUs, m_reassignJobLatencyTotalUs.load(), reassignN, "us").c_str());
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "[DIAG] ReassignSrc fromMerge=%lu fromSplitBatch=%lu (cumulative)\n",
                        (unsigned long)m_reassignSubmittedFromMerge.load(),
                        (unsigned long)m_reassignSubmittedFromSplitBatch.load());
                    uint64_t qN = m_queueDepthSampleCount.load();
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[DIAG] %s\n",
                        FormatHist("PoolQueueDepth", m_queueDepthHist, m_queueDepthTotal.load(), qN, "jobs").c_str());
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[DIAG] %s\n",
                        FormatHist("PoolRunning",    m_runningJobsHist, m_runningJobsTotal.load(), qN, "wkrs").c_str());
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[DIAG] %s\n",
                        FormatHist("AppendPreBytes", m_appendPreBytes, m_appendPreBytesTotal.load(), rmwN, "B").c_str());
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "[DIAG] AppendOutcome triggeredSplit=%lu nearThreshold(>=80%%)=%lu / RMWs=%lu (cumulative)\n",
                        (unsigned long)m_appendTriggeredSplit.load(),
                        (unsigned long)m_appendNearThreshold.load(),
                        (unsigned long)rmwN);
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                        "[DIAG] AppendGetFail=%lu (cumulative; non-zero means TiKV Get failed and the RMW was correctly aborted to avoid data loss)\n",
                        (unsigned long)m_appendGetFail.load());

                    // [DIAG-MC] Multi-chunk path histograms (only populated when MC is enabled)
                    {
                        uint64_t mcN  = m_mcAppendSampleCount.load();
                        uint64_t mcS  = m_mcSplitWriteSampleCount.load();
                        uint64_t mcGM = m_mcGetCountCacheMiss.load();
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[DIAG-MC] %s\n",
                            FormatHist("MCAppendUs",       m_mcAppendUs,       m_mcAppendTotalUs.load(),       mcN, "us").c_str());
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[DIAG-MC] %s\n",
                            FormatHist("MCGetCountMissUs", m_mcGetCountMissUs, m_mcGetCountMissTotalUs.load(), mcGM,"us").c_str());
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[DIAG-MC] %s\n",
                            FormatHist("MCSplitDelUs",     m_mcSplitDelUs,     m_mcSplitDelTotalUs.load(),     mcS, "us").c_str());
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[DIAG-MC] %s\n",
                            FormatHist("MCSplitPutBaseUs", m_mcSplitPutBaseUs, m_mcSplitPutBaseTotalUs.load(), mcS, "us").c_str());
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "[DIAG-MC] %s\n",
                            FormatHist("MCSplitSetCountUs",m_mcSplitSetCountUs,m_mcSplitSetCountTotalUs.load(),mcS, "us").c_str());
                        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                            "[DIAG-MC] CountCache hit=%lu miss=%lu (miss_ratio=%.4f)\n",
                            (unsigned long)m_mcGetCountCacheHit.load(),
                            (unsigned long)mcGM,
                            (m_mcGetCountCacheHit.load() + mcGM) ?
                                (double)mcGM / (m_mcGetCountCacheHit.load() + mcGM) : 0.0);
                    }
                }

                if (reset) {
                    m_splitNum = 0;
                    m_headMiss = 0;
                    m_theSameHeadNum = 0;
                    m_reAssignNum = 0;
                    m_reAssignScanNum = 0;
                    m_mergeNum = 0;
                    m_garbageNum = 0;
                    m_appendTaskNum = 0;
                    m_splitCost = 0;
                    m_clusteringCost = 0;
                    m_garbageCost = 0;
                    m_updateHeadCost = 0;
                    m_getCost = 0;
                    m_putCost = 0;
                    m_reassignScanCost = 0;
                    m_reassignScanIOCost = 0;
                    m_appendCost = 0;
                    m_appendIOCost = 0;
                    m_reAssignCost = 0;
                    m_selectCost = 0;
                    m_reAssignAppendCost = 0;
                }
            }
        };

        struct ExtraWorkSpace : public SPTAG::COMMON::IWorkSpace
        {
            ExtraWorkSpace() {}

            ~ExtraWorkSpace() {
                if (m_callback) {
                    m_callback();
                }
            }

            ExtraWorkSpace(ExtraWorkSpace& other) {
                Initialize(other.m_deduper.MaxCheck(), other.m_deduper.HashTableExponent(), (int)other.m_pageBuffers.size(), (int)(other.m_pageBuffers[0].GetPageSize()), other.m_blockIO, other.m_enableDataCompression);
            }

            void Initialize(int p_maxCheck, int p_hashExp, int p_internalResultNum, int p_maxPages, bool p_blockIO, bool enableDataCompression) {
                m_deduper.Init(p_maxCheck, p_hashExp);
                Clear(p_internalResultNum, p_maxPages, p_blockIO, enableDataCompression);
                m_relaxedMono = false;
                m_versionReadPolicy = COMMON::VersionReadPolicy::UseCache;
            }

            void Initialize(va_list& arg) {
                int maxCheck = va_arg(arg, int);
                int hashExp = va_arg(arg, int);
                int internalResultNum = va_arg(arg, int);
                int maxPages = va_arg(arg, int);
                bool blockIo = bool(va_arg(arg, int));
                bool enableDataCompression = bool(va_arg(arg, int));
                Initialize(maxCheck, hashExp, internalResultNum, maxPages, blockIo, enableDataCompression);
            }

            void Clear(int p_internalResultNum, int p_maxPages, bool p_blockIO, bool enableDataCompression) {
                if (p_internalResultNum > m_pageBuffers.size() || p_maxPages > m_pageBuffers[0].GetPageSize()) {
                    m_postingIDs.reserve(p_internalResultNum);
                    m_pageBuffers.resize(p_internalResultNum);
                    for (int pi = 0; pi < p_internalResultNum; pi++) {
                        m_pageBuffers[pi].ReservePageBuffer(p_maxPages);
                    }
                    m_blockIO = p_blockIO;
                    if (p_blockIO) {
                        int numPages = (p_maxPages >> PageSizeEx);
                        m_diskRequests.resize(p_internalResultNum * numPages);
			            //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "WorkSpace Init %d*%d reqs\n", p_internalResultNum, numPages);
                        for (int pi = 0; pi < p_internalResultNum; pi++) {
                            for (int pg = 0; pg < numPages; pg++) {
                                int rid = pi * numPages + pg;
                                auto& req = m_diskRequests[rid];

                                req.m_buffer = (char*)(m_pageBuffers[pi].GetBuffer() + ((std::uint64_t)pg << PageSizeEx));
                                req.m_extension = &m_processIocp;
#ifdef _MSC_VER
                                memset(&(req.myres.m_col), 0, sizeof(OVERLAPPED));
                                req.myres.m_col.m_data = (void*)(&req);
#else
                                memset(&(req.myiocb), 0, sizeof(struct iocb));
                                req.myiocb.aio_buf = reinterpret_cast<uint64_t>(req.m_buffer);
                                req.myiocb.aio_data = reinterpret_cast<uintptr_t>(&req);
#endif
                            }
                        }
                    }
                    else {
                        m_diskRequests.resize(p_internalResultNum);
			            //SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "WorkSpace Init %d reqs\n", p_internalResultNum);
                        for (int pi = 0; pi < p_internalResultNum; pi++) {
                            auto& req = m_diskRequests[pi];

                            req.m_buffer = (char*)(m_pageBuffers[pi].GetBuffer());
                            req.m_extension = &m_processIocp;
#ifdef _MSC_VER
                            memset(&(req.myres.m_col), 0, sizeof(OVERLAPPED));
                            req.myres.m_col.m_data = (void*)(&req);
#else
                            memset(&(req.myiocb), 0, sizeof(struct iocb));
                            req.myiocb.aio_buf = reinterpret_cast<uint64_t>(req.m_buffer);
                            req.myiocb.aio_data = reinterpret_cast<uintptr_t>(&req);
#endif
                        }
                    }
                }

                m_enableDataCompression = enableDataCompression;
                if (enableDataCompression) {
                    m_decompressBuffer.ReservePageBuffer(p_maxPages);
                }
                m_versionReadPolicy = COMMON::VersionReadPolicy::UseCache;
            }

            std::vector<SizeType> m_postingIDs;

            COMMON::OptHashPosVector m_deduper;

            COMMON::OptHashPosVector* m_deduperOverride = nullptr;

            COMMON::OptHashPosVector& Deduper()
            {
                return (m_deduperOverride == nullptr) ? m_deduper : *m_deduperOverride;
            }

            Helper::RequestQueue m_processIocp;

            std::vector<Helper::PageBuffer<std::uint8_t>> m_pageBuffers;

            bool m_blockIO = false;

            bool m_enableDataCompression = false;

            Helper::PageBuffer<std::uint8_t> m_decompressBuffer;

            std::vector<Helper::AsyncReadRequest> m_diskRequests;

            int m_ri = 0;

            int m_pi = 0;

            int m_offset = 0;

            bool m_relaxedMono = false;

            COMMON::VersionReadPolicy m_versionReadPolicy = COMMON::VersionReadPolicy::UseCache;

            int m_loadedPostingNum = 0;

            bool m_iteratorHeadInitialized = false;

            struct IteratorLayerState
            {
                std::vector<BasicResult> m_results;
                std::size_t m_cursor = 0;
                std::unique_ptr<COMMON::OptHashPosVector> m_deduper;

                void Reset(int p_maxCheck, int p_hashExp)
                {
                    m_results.clear();
                    m_cursor = 0;
                    if (!m_deduper)
                    {
                        m_deduper.reset(new COMMON::OptHashPosVector());
                        m_deduper->Init(p_maxCheck, p_hashExp);
                    }
                    else
                    {
                        m_deduper->clear();
                    }
                }

                std::size_t Available() const
                {
                    return (m_cursor < m_results.size()) ? (m_results.size() - m_cursor) : 0;
                }

                void CompactConsumed()
                {
                    if (m_cursor == 0)
                        return;

                    if (m_cursor >= m_results.size())
                    {
                        m_results.clear();
                        m_cursor = 0;
                        return;
                    }

                    if (m_cursor > 4096 && m_cursor * 2 > m_results.size())
                    {
                        m_results.erase(m_results.begin(), m_results.begin() + m_cursor);
                        m_cursor = 0;
                    }
                }
            };

            std::vector<IteratorLayerState> m_iteratorLayerStates;

            void ResetIteratorState(int p_layerCount, int p_maxCheck, int p_hashExp)
            {
                m_iteratorLayerStates.resize(p_layerCount);
                for (auto& state : m_iteratorLayerStates)
                {
                    state.Reset(p_maxCheck, p_hashExp);
                }
                m_deduperOverride = nullptr;
                m_iteratorHeadInitialized = false;
                m_loadedPostingNum = 0;
                m_relaxedMono = false;
            }

            std::function<bool(const ByteArray&)> m_filterFunc;

            std::function<void()> m_callback;
        };

        class IExtraSearcher
        {
        public:
            IExtraSearcher()
            {
            }

            ~IExtraSearcher()
            {
            }
            virtual bool Available() = 0;

            virtual bool LoadIndex(Options& p_options) = 0;

            virtual ErrorCode SearchIndex(ExtraWorkSpace* p_exWorkSpace,
                QueryResult& p_queryResults,
                SearchStats* p_stats, std::set<SizeType>* truth = nullptr, std::map<SizeType, std::set<SizeType>>* found = nullptr,
                bool p_checkVersionMap = true) = 0;

            virtual ErrorCode SearchIndexIterativeScan(ExtraWorkSpace* p_exWorkSpace,
                QueryResult& p_queryResults,
                std::vector<BasicResult>& p_results,
                bool p_checkVersionMap = true) = 0;

            virtual ErrorCode SearchIterativeNext(ExtraWorkSpace* p_exWorkSpace, QueryResult& p_headResults,
                QueryResult& p_queryResults) = 0;

            virtual ErrorCode SearchIndexWithoutParsing(ExtraWorkSpace* p_exWorkSpace) = 0;

            virtual ErrorCode SearchNextInPosting(ExtraWorkSpace* p_exWorkSpace, QueryResult& p_headResults,
                QueryResult& p_queryResults) = 0;

            virtual bool BuildIndex(std::shared_ptr<Helper::VectorSetReader>& p_reader, 
                std::shared_ptr<VectorIndex> p_index, 
                Options& p_opt, COMMON::Dataset<SizeType>& p_headtoLocal, Helper::Concurrent::ConcurrentMap<SizeType, SizeType>& p_headGlobaltoLocal, COMMON::Dataset<SizeType>& p_localToGlobal, SizeType upperBound = -1) = 0;
            
            virtual ErrorCode RefineIndex()
            {
                return ErrorCode::Undefined;
            }
            virtual ErrorCode AddIndex(ExtraWorkSpace* p_exWorkSpace, std::shared_ptr<VectorSet>& p_vectorSet,
                SizeType p_begin, bool disableSplit = false) { return ErrorCode::Undefined; }
            virtual ErrorCode DeleteIndex(SizeType p_id) { return ErrorCode::Undefined; }
            virtual ErrorCode ResetIndex(SizeType p_id) { return ErrorCode::Undefined; }
            
            virtual SizeType GetNumSamples() const = 0;

            virtual bool ContainSample(const SizeType idx) const
            {
                return true;
            }

            virtual bool ContainSample(const SizeType idx, COMMON::VersionReadPolicy policy) const
            {
                return ContainSample(idx);
            }

            virtual void ContainSamples(const std::vector<SizeType>& ids, std::vector<uint8_t>& contains, COMMON::VersionReadPolicy policy) const
            {
                contains.resize(ids.size());
                for (size_t i = 0; i < ids.size(); ++i) {
                    contains[i] = ContainSample(ids[i], policy) ? 1 : 0;
                }
            }

            virtual SizeType GetNumDeleted() const
            {
                return 0;
            }

            virtual ErrorCode GetContainedIDs(std::vector<SizeType>& globalIDs) 
            {
                return ErrorCode::Undefined;
            }

            virtual bool AllFinished() { return false; }
            virtual void GetDBStats() { return; }
            virtual int64_t GetNumBlocks() { return 0; }
            virtual void GetIndexStats(int finishedInsert, bool cost, bool reset) { return; }
            virtual void ForceCompaction() { return; }

            virtual ErrorCode CheckPosting(SizeType postingiD, std::vector<std::uint8_t> *visited = nullptr,
                                           ExtraWorkSpace *p_exWorkSpace = nullptr) = 0;
            
            virtual ErrorCode GetWritePosting(ExtraWorkSpace *p_exWorkSpace, SizeType pid, std::string &posting,
                                              bool write = false)
            {
                return ErrorCode::Undefined;
            }

            virtual ErrorCode Checkpoint(std::string prefix) { return ErrorCode::Success; }
        };
    } // SPANN
} // SPTAG

#endif // _SPTAG_SPANN_IEXTRASEARCHER_H_
