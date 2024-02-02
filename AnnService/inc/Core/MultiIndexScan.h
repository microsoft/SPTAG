// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_MULTI_INDEX_SCAN_H
#define _SPTAG_MULTI_INDEX_SCAN_H

#include <memory>

#include <vector>
#include <utility>
#include <unordered_set>
#include <queue>
#include <stack>
#include <chrono>

#include "ResultIterator.h"
#include "VectorIndex.h"
#include <numeric>
namespace SPTAG
{
    class MultiIndexScan
    {
    public:
        MultiIndexScan();
        MultiIndexScan(std::vector<std::shared_ptr<VectorIndex>> vecIndices,
                       std::vector<void*> p_targets,
                       unsigned int k,
                       float (*rankFunction)(std::vector<float>),
                       bool useTimer,
                       int termCondVal,
                       int searchLimit
                       );
        ~MultiIndexScan();
        void Init(std::vector<std::shared_ptr<VectorIndex>> vecIndices,
            std::vector<ByteArray> p_targets,
            std::vector<float> weight,
            unsigned int k,
            bool useTimer,
            int termCondVal,
            int searchLimit);
        bool Next(BasicResult& result);
        void Close();

    private:
        std::vector<std::shared_ptr<ResultIterator>> indexIters;
        std::vector<std::shared_ptr<VectorIndex>> fwdLUTs;
        std::unordered_set<SizeType> seenSet;
        std::vector<SPTAG::ByteArray> p_data_array;
        std::vector<float> weight;

        unsigned int k;


        
        bool useTimer;
        unsigned int termCondVal;
        int searchLimit;
        std::chrono::time_point<std::chrono::high_resolution_clock> t_start;
        
        float (*func)(std::vector<float>);
        
        unsigned int consecutive_drops;
        
        bool terminate;
        using pq_item = std::pair<float, SizeType>;
        class pq_item_compare
        {
        public:
            bool operator()(const pq_item& lhs, const pq_item& rhs)
            {
                return lhs.first < rhs.first;
            }
        };
        std::priority_queue<pq_item, std::vector<pq_item>, pq_item_compare> pq;
        std::stack<pq_item> outputStk;
        float WeightedRankFunc(std::vector<float>);
        
    };
} // namespace SPTAG
#endif
