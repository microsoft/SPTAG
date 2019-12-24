#pragma once
#include "inc/SSDServing/Common/stdafx.h"
#include "inc/SSDServing/SelectHead_BKT/BKTNodeInfo.h"

namespace SPTAG {
    namespace SSDServing {
        namespace SelectHead_BKT {
            void CalcLeafSize(int p_nodeID,
                const shared_ptr<COMMON::BKTree> p_tree,
                unordered_map<int, int>& p_counter);

            void DfsAnalyze(int p_nodeID,
                const shared_ptr<COMMON::BKTree> p_tree,
                const BasicVectorSet& p_vectorSet,
                const Options& p_opts,
                int p_height,
                std::vector<BKTNodeInfo>& p_nodeInfos);
        }
    }
}