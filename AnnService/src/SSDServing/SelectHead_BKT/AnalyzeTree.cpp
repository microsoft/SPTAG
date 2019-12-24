#include "inc/SSDServing/Common/stdafx.h"
#include "inc/SSDServing/SelectHead_BKT/AnalyzeTree.h"

namespace SPTAG {
    namespace SSDServing {
        namespace SelectHead_BKT {

            void CalcLeafSize(int p_nodeID,
                const shared_ptr<COMMON::BKTree> p_tree,
                unordered_map<int, int>& p_counter) {

                SPTAG::COMMON::BKTNode& node = (*p_tree)[p_nodeID];

                p_counter[node.centerid] = 1;

                if (node.childStart < 0)
                {
                    return;
                }

                for (size_t i = node.childStart; i < node.childEnd; i++)
                {
                    CalcLeafSize(i, p_tree, p_counter);
                    p_counter[node.centerid] += p_counter[(*p_tree)[i].centerid];
                }
            }

            void DfsAnalyze(int p_nodeID,
                const shared_ptr<COMMON::BKTree> p_tree,
                const BasicVectorSet& p_vectorSet,
                const Options& p_opts,
                int p_height,
                std::vector<BKTNodeInfo>& p_nodeInfos) {
            
                const auto& node = (*p_tree)[p_nodeID];

                // Leaf.
                if (node.childStart < 0)
                {
                    p_nodeInfos[p_nodeID].leafSize = 1;
                    p_nodeInfos[p_nodeID].minDepth = 0;
                    p_nodeInfos[p_nodeID].maxDepth = 0;

                    return;
                }

                p_nodeInfos[p_nodeID].leafSize = 0;

                int& minDepth = p_nodeInfos[p_nodeID].minDepth;
                int& maxDepth = p_nodeInfos[p_nodeID].maxDepth;

                int sinlgeCount = 0;
                for (int nodeId = node.childStart; nodeId < node.childEnd; ++nodeId)
                {
                    DfsAnalyze(nodeId, p_tree, p_vectorSet, p_opts, p_height + 1, p_nodeInfos);
                    if (minDepth > p_nodeInfos[nodeId].minDepth)
                    {
                        minDepth = p_nodeInfos[nodeId].minDepth;
                    }

                    if (maxDepth < p_nodeInfos[nodeId].maxDepth)
                    {
                        maxDepth = p_nodeInfos[nodeId].maxDepth;
                    }

                    if (p_nodeInfos[nodeId].maxDepth == 1 && p_nodeInfos[nodeId].leafSize == 1)
                    {
                        ++sinlgeCount;
                    }

                    p_nodeInfos[p_nodeID].leafSize += p_nodeInfos[nodeId].leafSize;
                }

                ++minDepth;
                ++maxDepth;

                if (p_height > 5 || sinlgeCount == 0)
                {
                    return;
                }

                fprintf(stdout,
                    "CheckNode: %8d, Height: %3d, MinDepth: %3d, MaxDepth: %3d, Children: %3d, Single: %3d\n",
                    p_nodeID,
                    p_height,
                    minDepth,
                    maxDepth,
                    node.childEnd - node.childStart,
                    sinlgeCount);

                for (int nodeId = node.childStart; nodeId < node.childEnd; ++nodeId)
                {
                    fprintf(stdout,
                        "    ChildNode: %8d, MinDepth: %3d, MaxDepth: %3d, ChildrenCount: %3d, LeafCount: %3d\n",
                        nodeId,
                        p_nodeInfos[nodeId].minDepth,
                        p_nodeInfos[nodeId].maxDepth,
                        (*p_tree)[nodeId].childEnd - (*p_tree)[nodeId].childStart,
                        p_nodeInfos[nodeId].leafSize);
                }
            }
        }
    }
}