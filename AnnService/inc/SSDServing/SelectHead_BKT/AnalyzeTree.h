#pragma once
#include <memory>
#include <unordered_map>

#include "inc/Core/Common.h"
#include "inc/Core/Common/BKTree.h"
#include "inc/Core/VectorSet.h"

#include "inc/SSDServing/SelectHead_BKT/BKTNodeInfo.h"
#include "inc/SSDServing/SelectHead_BKT/Options.h"

namespace SPTAG {
    namespace SSDServing {
        namespace SelectHead_BKT {
            void CalcLeafSize(int p_nodeID,
                const std::shared_ptr<COMMON::BKTree> p_tree,
                std::unordered_map<int, int>& p_counter);

            void DfsAnalyze(int p_nodeID,
                const std::shared_ptr<COMMON::BKTree> p_tree,
                std::shared_ptr<VectorSet> p_vectorSet,
                const Options& p_opts,
                int p_height,
                std::vector<BKTNodeInfo>& p_nodeInfos);
        }
    }
}