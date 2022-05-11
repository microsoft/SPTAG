// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "inc/Core/Common.h"
#include "inc/Core/Common/BKTree.h"
#include "inc/Core/SPANN/Options.h"
#include "inc/Helper/StringConvert.h"
#include "inc/Helper/VectorSetReader.h"
#include "Utils.h"

#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <random>

namespace SPTAG {
	namespace SSDServing {
		namespace SelectHead {
            struct BKTNodeInfo
            {
                BKTNodeInfo() : leafSize(0), minDepth(INT32_MAX), maxDepth(0), parent(-1)
                {
                }

                int leafSize;

                int minDepth;

                int maxDepth;

                int parent;
            };

            struct HeadCandidate
            {
                HeadCandidate(int n, int c, int d) : nodeID(n), childrenSize(c), depth(d)
                {
                }

                int nodeID;

                int childrenSize;

                int depth;
            };

            template<typename T>
            float Std(T* nums, size_t size) {
                float var = 0;

                float mean = 0;
                for (size_t i = 0; i < size; i++)
                {
                    mean += nums[i] / static_cast<float>(size);
                }

                for (size_t i = 0; i < size; i++)
                {
                    float temp = nums[i] - mean;
                    var += (float)pow(temp, 2.0);
                }
                var /= static_cast<float>(size);

                return sqrt(var);
            }

            void CalcLeafSize(int p_nodeID,
                const std::shared_ptr<COMMON::BKTree> p_tree,
                std::unordered_map<int, int>& p_counter) {

                SPTAG::COMMON::BKTNode& node = (*p_tree)[p_nodeID];

                p_counter[node.centerid] = 1;

                if (node.childStart < 0)
                {
                    return;
                }

                for (SPTAG::SizeType i = node.childStart; i < node.childEnd; i++)
                {
                    CalcLeafSize(i, p_tree, p_counter);
                    p_counter[node.centerid] += p_counter[(*p_tree)[i].centerid];
                }
            }

            void DfsAnalyze(int p_nodeID,
                const std::shared_ptr<COMMON::BKTree> p_tree,
                std::shared_ptr<VectorSet> p_vectorSet,
                const SPANN::Options& p_opts,
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

                LOG(Helper::LogLevel::LL_Info,
                    "CheckNode: %8d, Height: %3d, MinDepth: %3d, MaxDepth: %3d, Children: %3d, Single: %3d\n",
                    p_nodeID,
                    p_height,
                    minDepth,
                    maxDepth,
                    node.childEnd - node.childStart,
                    sinlgeCount);

                for (int nodeId = node.childStart; nodeId < node.childEnd; ++nodeId)
                {
                    LOG(Helper::LogLevel::LL_Info,
                        "    ChildNode: %8d, MinDepth: %3d, MaxDepth: %3d, ChildrenCount: %3d, LeafCount: %3d\n",
                        nodeId,
                        p_nodeInfos[nodeId].minDepth,
                        p_nodeInfos[nodeId].maxDepth,
                        (*p_tree)[nodeId].childEnd - (*p_tree)[nodeId].childStart,
                        p_nodeInfos[nodeId].leafSize);
                }
            }

            // Return height of current Node.
            void DfsSelect(int p_nodeID,
                const std::shared_ptr<SPTAG::COMMON::BKTree> p_tree,
                std::vector<HeadCandidate>& p_candidates,
                std::vector<BKTNodeInfo>& p_nodeInfos)
            {
                const auto& node = (*p_tree)[p_nodeID];

                // Leaf.
                if (node.childStart == -1)
                {
                    p_nodeInfos[p_nodeID].leafSize = 1;
                    p_nodeInfos[p_nodeID].minDepth = 0;
                    p_nodeInfos[p_nodeID].maxDepth = 0;

                    return;
                }

                p_nodeInfos[p_nodeID].leafSize = 0;

                int& minDepth = p_nodeInfos[p_nodeID].minDepth;
                int& maxDepth = p_nodeInfos[p_nodeID].maxDepth;

                for (int nodeId = node.childStart; nodeId < node.childEnd; ++nodeId)
                {
                    p_nodeInfos[nodeId].parent = p_nodeID;
                    DfsSelect(nodeId, p_tree, p_candidates, p_nodeInfos);
                    if (minDepth > p_nodeInfos[nodeId].minDepth)
                    {
                        minDepth = p_nodeInfos[nodeId].minDepth;
                    }

                    if (maxDepth < p_nodeInfos[nodeId].maxDepth)
                    {
                        maxDepth = p_nodeInfos[nodeId].maxDepth;
                    }

                    p_nodeInfos[p_nodeID].leafSize += p_nodeInfos[nodeId].leafSize;
                }

                ++minDepth;
                ++maxDepth;

                p_candidates.emplace_back(p_nodeID, p_nodeInfos[p_nodeID].leafSize, minDepth);
            }


            // Return height of current Node.
            int DfsCovered(int p_nodeID,
                const std::shared_ptr<SPTAG::COMMON::BKTree> p_tree,
                const std::unordered_set<int>& p_candidates,
                bool p_covered)
            {
                const auto& node = (*p_tree)[p_nodeID];

                if (p_candidates.count(node.centerid) > 0)
                {
                    p_covered = true;
                }

                // Leaf.
                if (node.childStart == -1)
                {
                    return p_covered ? 1 : 0;
                }

                int ret = 0;
                for (int nodeId = node.childStart; nodeId < node.childEnd; ++nodeId)
                {
                    ret += DfsCovered(nodeId, p_tree, p_candidates, p_covered);
                }

                return ret;
            }


            void SelectHeadStatically(const std::shared_ptr<COMMON::BKTree> p_tree, const int p_vectorCount, const SPANN::Options& p_opts, std::vector<int>& p_selected)
            {
                std::vector<HeadCandidate> candidates;
                candidates.reserve(p_tree->size());

                std::vector<BKTNodeInfo> bktNodeInfos(p_tree->size());

                int selectLimit = static_cast<int>(p_vectorCount * p_opts.m_ratio) + 1;

                if (selectLimit > p_vectorCount)
                {
                    selectLimit = p_vectorCount;
                }

                std::unordered_set<int> overallSelected;
                overallSelected.reserve(static_cast<size_t>(selectLimit) * 2);
                // only use the first tree
                int rootIdOfFirstTree = 0;
                DfsSelect(rootIdOfFirstTree, p_tree, candidates, bktNodeInfos);

                int overallLeafCount = bktNodeInfos[rootIdOfFirstTree].leafSize;
                int bigClusterCount = 0;

                std::sort(candidates.begin(), candidates.end(), [](const HeadCandidate& a, const HeadCandidate& b)
                    {
                        if (a.depth == b.depth)
                        {
                            return a.childrenSize > b.childrenSize;
                        }

                        return a.depth < b.depth;
                    });

                int nextDepth = 1;
                uint32_t lastSelected = 0;
                std::map<int, int> sizeCount;

                std::queue<int> needRevisit;

                for (const auto& c : candidates)
                {
                    if (c.depth > nextDepth)
                    {
                        LOG(Helper::LogLevel::LL_Info,
                            "Iteration of depth %d, selected %u heads, about %.2lf%%\n",
                            nextDepth,
                            static_cast<uint32_t>(overallSelected.size() - lastSelected),
                            overallSelected.size() * 100.0 / p_vectorCount);

                        lastSelected = static_cast<uint32_t>(overallSelected.size());
                        nextDepth = c.depth;

                        if (p_opts.m_recursiveCheckSmallCluster && !needRevisit.empty())
                        {
                            while (!needRevisit.empty())
                            {
                                int nodeId = needRevisit.front();
                                needRevisit.pop();

                                if (nodeId < 0)
                                {
                                    continue;
                                }

                                int vectorId = (*p_tree)[nodeId].centerid;
                                if (overallSelected.count(vectorId) > 0)
                                {
                                    continue;
                                }

                                if (bktNodeInfos[nodeId].leafSize >= p_opts.m_selectThreshold)
                                {
                                    overallSelected.insert(vectorId);
                                }
                                else
                                {
                                    needRevisit.push(bktNodeInfos[nodeId].parent);
                                }
                            }

                            LOG(Helper::LogLevel::LL_Info,
                                "Iteration of Revisit, selected %u heads, about %.2lf%%\n",
                                static_cast<uint32_t>(overallSelected.size() - lastSelected),
                                overallSelected.size() * 100.0 / p_vectorCount);

                            lastSelected = static_cast<uint32_t>(overallSelected.size());
                        }
                    }

                    if (c.depth <= 1)
                    {
                        if (c.childrenSize >= p_opts.m_selectThreshold)
                        {
                            overallSelected.insert((*p_tree)[c.nodeID].centerid);
                        }
                        else if (p_opts.m_recursiveCheckSmallCluster)
                        {
                            needRevisit.push(bktNodeInfos[c.nodeID].parent);
                        }

                        if (c.childrenSize > p_opts.m_splitThreshold)
                        {
                            int selectCount = static_cast<int>(std::ceil(c.childrenSize * 1.0 / p_opts.m_splitFactor) + 0.5);
                            for (int i = 1; i < selectCount; ++i)
                            {
                                int nodeId = (*p_tree)[c.nodeID].childStart + i;
                                overallSelected.insert((*p_tree)[nodeId].centerid);
                                ++bigClusterCount;
                            }
                        }

                        if (sizeCount.count(c.childrenSize) > 0)
                        {
                            ++sizeCount[c.childrenSize];
                        }
                        else
                        {
                            sizeCount[c.childrenSize] = 1;
                        }

                        continue;
                    }

                    if (overallSelected.size() >= selectLimit)
                    {
                        break;
                    }

                    overallSelected.insert((*p_tree)[c.nodeID].centerid);
                }

                overallSelected.erase(p_vectorCount);

                LOG(Helper::LogLevel::LL_Info,
                    "Iteration of depth %d, selected %u heads, about %.2lf%%\n",
                    nextDepth,
                    static_cast<uint32_t>(overallSelected.size() - lastSelected),
                    (overallSelected.size() - lastSelected) * 100.0 / p_vectorCount);

                p_selected.clear();
                p_selected.reserve(overallSelected.size());
                p_selected.assign(overallSelected.begin(), overallSelected.end());

                LOG(Helper::LogLevel::LL_Info,
                    "Finally selected %u heads, about %.2lf%%\n",
                    static_cast<uint32_t>(p_selected.size()),
                    p_selected.size() * 100.0 / p_vectorCount);

                LOG(Helper::LogLevel::LL_Info,
                    "Total leaf count %d, big cluster count %d\n",
                    overallLeafCount,
                    bigClusterCount);

                if (p_opts.m_printSizeCount)
                {
                    LOG(Helper::LogLevel::LL_Info, "Leaf size count:\n");
                    for (const auto& ele : sizeCount)
                    {
                        LOG(Helper::LogLevel::LL_Info, "%3d %10d\n", ele.first, ele.second);
                    }
                }

                int sumGreaterThanLeafSize = 0;
                for (const auto& ele : sizeCount)
                {
                    if (ele.first > p_opts.m_iBKTLeafSize)
                    {
                        sumGreaterThanLeafSize += ele.first * ele.second;
                    }
                }

                LOG(Helper::LogLevel::LL_Info, "Leaf sum count > set leaf size: %d\n", sumGreaterThanLeafSize);

                int covered = DfsCovered(rootIdOfFirstTree, p_tree, overallSelected, false);
                LOG(Helper::LogLevel::LL_Info,
                    "Total covered leaf count %d, about %.2lf\n",
                    covered,
                    covered * 100.0 / p_vectorCount);
            }


            int SelectHeadDynamicallyInternal(const std::shared_ptr<COMMON::BKTree> p_tree,
                int p_nodeID,
                const SPANN::Options& p_opts,
                std::vector<int>& p_selected)
            {
                typedef std::pair<int, int> CSPair;
                std::vector<CSPair> children;
                int childrenSize = 1;
                const auto& node = (*p_tree)[p_nodeID];
                if (node.childStart >= 0)
                {
                    children.reserve(node.childEnd - node.childStart);
                    for (int i = node.childStart; i < node.childEnd; ++i)
                    {
                        int cs = SelectHeadDynamicallyInternal(p_tree, i, p_opts, p_selected);
                        if (cs > 0)
                        {
                            children.emplace_back(i, cs);
                            childrenSize += cs;
                        }
                    }
                }

                if (childrenSize >= p_opts.m_selectThreshold)
                {
                    if (node.centerid < (*p_tree)[0].centerid)
                    {
                        p_selected.push_back(node.centerid);
                    }

                    if (childrenSize > p_opts.m_splitThreshold)
                    {
                        std::sort(children.begin(), children.end(), [](const CSPair& a, const CSPair& b)
                            {
                                return a.second > b.second;
                            });

                        size_t selectCnt = static_cast<size_t>(std::ceil(childrenSize * 1.0 / p_opts.m_splitFactor) + 0.5);
                        //if (selectCnt > 1) selectCnt -= 1;
                        for (size_t i = 0; i < selectCnt && i < children.size(); ++i)
                        {
                            p_selected.push_back((*p_tree)[children[i].first].centerid);
                        }
                    }

                    return 0;
                }

                return childrenSize;
            }

            void SelectHeadDynamically_Old(const std::shared_ptr<COMMON::BKTree> p_tree,
                int p_vectorCount,
                const SPANN::Options& p_opts,
                std::vector<int>& p_selected)
            {
                p_selected.clear();
                p_selected.reserve(p_vectorCount);

                if (static_cast<int>(std::round(p_opts.m_ratio * p_vectorCount)) >= p_vectorCount)
                {
                    for (int i = 0; i < p_vectorCount; ++i)
                    {
                        p_selected.push_back(i);
                    }

                    return;
                }
                SPANN::Options opts = p_opts;

                int selectThreshold = p_opts.m_selectThreshold;
                int splitThreshold = p_opts.m_splitThreshold;

                double minDiff = 100;
                for (int select = 2; select <= p_opts.m_selectThreshold; ++select)
                {
                    opts.m_selectThreshold = select;
                    opts.m_splitThreshold = p_opts.m_splitThreshold;

                    int l = p_opts.m_splitFactor;
                    int r = p_opts.m_splitThreshold;

                    while (l < r - 1)
                    {
                        opts.m_splitThreshold = (l + r) / 2;
                        p_selected.clear();

                        SelectHeadDynamicallyInternal(p_tree, 0, opts, p_selected);
                        std::sort(p_selected.begin(), p_selected.end());
                        p_selected.erase(std::unique(p_selected.begin(), p_selected.end()), p_selected.end());

                        double diff = static_cast<double>(p_selected.size()) / p_vectorCount - p_opts.m_ratio;

                        LOG(Helper::LogLevel::LL_Info,
                            "Select Threshold: %d, Split Threshold: %d, diff: %.2lf%%.\n",
                            opts.m_selectThreshold,
                            opts.m_splitThreshold,
                            diff * 100.0);

                        if (minDiff > fabs(diff))
                        {
                            minDiff = fabs(diff);

                            selectThreshold = opts.m_selectThreshold;
                            splitThreshold = opts.m_splitThreshold;
                        }

                        if (diff > 0)
                        {
                            l = (l + r) / 2;
                        }
                        else
                        {
                            r = (l + r) / 2;
                        }
                    }
                }

                opts.m_selectThreshold = selectThreshold;
                opts.m_splitThreshold = splitThreshold;

                LOG(Helper::LogLevel::LL_Info,
                    "Final Select Threshold: %d, Split Threshold: %d.\n",
                    opts.m_selectThreshold,
                    opts.m_splitThreshold);

                p_selected.clear();
                SelectHeadDynamicallyInternal(p_tree, 0, opts, p_selected);
                std::sort(p_selected.begin(), p_selected.end());
                p_selected.erase(std::unique(p_selected.begin(), p_selected.end()), p_selected.end());
            }

            void SelectHeadDynamically(const std::shared_ptr<COMMON::BKTree> p_tree,
                int p_vectorCount,
                const SPANN::Options& p_opts,
                std::vector<int>& p_selected)
            {
                std::mt19937 random;
                p_selected.clear();
                p_selected.reserve(p_vectorCount);

                if (static_cast<int>(std::round(p_opts.m_ratio * p_vectorCount)) >= p_vectorCount)
                {
                    for (int i = 0; i < p_vectorCount; ++i)
                    {
                        p_selected.push_back(i);
                    }

                    return;
                }

                const double c_minSSRatio = 1.8;
                const double c_maxSSRatio = 2.2;

                SPANN::Options opts = p_opts;

                int selectThreshold = p_opts.m_selectThreshold;
                int splitThreshold = p_opts.m_splitThreshold;

                double minDiff = 100;

                int trySelect = min(p_vectorCount, p_opts.m_selectThreshold);
                int lastTrySelect = trySelect;
                int tryStep = 0;
                int randomTry = 0;

                std::unordered_map<int, int> usedTrySelect;

                while (1 <= trySelect && trySelect <= p_vectorCount)
                {
                    LOG(Helper::LogLevel::LL_Info, "Start try Select Threshold: %d\n", trySelect);

                    opts.m_selectThreshold = trySelect;

                    if (usedTrySelect.count(trySelect) == 0)
                    {
                        int l = p_opts.m_splitFactor;
                        int r = max(p_opts.m_splitFactor + 2, static_cast<int>(trySelect * (c_maxSSRatio + 1)));

                        while (l < r - 1)
                        {
                            opts.m_splitThreshold = (l + r) / 2;
                            p_selected.clear();

                            SelectHeadDynamicallyInternal(p_tree, 0, opts, p_selected);
                            std::sort(p_selected.begin(), p_selected.end());
                            p_selected.erase(std::unique(p_selected.begin(), p_selected.end()), p_selected.end());

                            double diff = static_cast<double>(p_selected.size()) / p_vectorCount - p_opts.m_ratio;

                            LOG(Helper::LogLevel::LL_Info,
                                "    Split Threshold: %d, diff: %.2lf%%.\n",
                                opts.m_splitThreshold,
                                diff * 100.0);

                            if (minDiff > fabs(diff))
                            {
                                minDiff = fabs(diff);

                                selectThreshold = opts.m_selectThreshold;
                                splitThreshold = opts.m_splitThreshold;
                            }

                            if (diff > 0)
                            {
                                l = (l + r) / 2;
                            }
                            else
                            {
                                r = (l + r) / 2;
                            }
                        }
                        usedTrySelect.emplace(trySelect, opts.m_splitThreshold);
                    }
                    else {
                        opts.m_splitThreshold = usedTrySelect[trySelect];
                        randomTry = max(randomTry, 1);
                    }

                    double ssratio = static_cast<double>(opts.m_splitThreshold) / trySelect;
                    double diffOffset = minDiff / p_opts.m_ratio;

                    LOG(Helper::LogLevel::LL_Info, "Best abs(diff): %.3lf%%, target offset %.3lf, SSRatio: %.3lf\n\n", minDiff * 100.0, diffOffset, ssratio);

                    if (randomTry == 0)
                    {
                        if (ssratio < c_minSSRatio)
                        {
                            if (tryStep >= 0)
                            {
                                tryStep = -1;
                            }
                            else if (abs(tryStep) < 128)
                            {
                                tryStep *= 2;
                            }

                            trySelect += tryStep;
                        }
                        else if (ssratio > c_maxSSRatio)
                        {
                            if (tryStep <= 0)
                            {
                                tryStep = 1;
                            }
                            else if (abs(tryStep) < 128)
                            {
                                tryStep *= 2;
                            }

                            trySelect += tryStep;
                        }
                        else if (diffOffset > 0.02)
                        {
                            randomTry = 1;
                        }
                        else
                        {
                            break;
                        }

                        trySelect = min(p_vectorCount, max(1, trySelect));
                    }

                    if (0 < randomTry && randomTry < p_opts.m_maxRandomTryCount)
                    {
                        std::uniform_int_distribution<> distrib(-8, 8);
                        do
                        {
                            tryStep = distrib(random);
                        } while (tryStep == 0);

                        trySelect += tryStep;
                        trySelect = min(p_vectorCount, max(1, trySelect));

                        ++randomTry;
                    }
                    else if (lastTrySelect == trySelect)
                    {
                        break;
                    }


                    lastTrySelect = trySelect;
                }

                opts.m_selectThreshold = selectThreshold;
                opts.m_splitThreshold = splitThreshold;

                LOG(Helper::LogLevel::LL_Info,
                    "Final Select Threshold: %d, Split Threshold: %d.\n",
                    opts.m_selectThreshold,
                    opts.m_splitThreshold);

                p_selected.clear();
                SelectHeadDynamicallyInternal(p_tree, 0, opts, p_selected);
                std::sort(p_selected.begin(), p_selected.end());
                p_selected.erase(std::unique(p_selected.begin(), p_selected.end()), p_selected.end());
            }

            ErrorCode SelectHead(std::shared_ptr<VectorSet> vectorSet, std::shared_ptr<COMMON::BKTree> bkt, SPANN::Options& opts, std::unordered_map<int, int>& counter, std::vector<int>& selected) {
                selected.reserve(vectorSet->Count());

                if (vectorSet->Count() == 1)
                {
                    selected.push_back(0);
                }
                else
                {
                    LOG(Helper::LogLevel::LL_Info, "Start selecting nodes...\n");
                    if (!opts.m_selectDynamically)
                    {
                        LOG(Helper::LogLevel::LL_Info, "Select Head Statically...\n");
                        SelectHeadStatically(bkt, vectorSet->Count(), opts, selected);
                    }
                    else
                    {
                        LOG(Helper::LogLevel::LL_Info, "Select Head Dynamically...\n");
                        SelectHeadDynamically_Old(bkt, vectorSet->Count(), opts, selected);
                    }
                }

                if (selected.empty())
                {
                    LOG(Helper::LogLevel::LL_Error, "Can't select any vector as head with current settings\n");
                    return ErrorCode::Fail;
                }

                if (opts.m_calcStd) {
                    std::vector<int> leafSizes;
                    for (SizeType& item : selected)
                    {
                        if (counter.count(item) <= 0)
                        {
                            LOG(Helper::LogLevel::LL_Error, "selected node: %d can't be used to calculate std.", item);
                            return ErrorCode::Fail;
                        }
                        leafSizes.push_back(counter[item]);
                    }
                    std::map<int, int> leafDict;
                    for (int item : leafSizes) {
                        if (leafDict.count(item) > 0)
                        {
                            leafDict[item]++;
                        }
                        else {
                            leafDict[item] = 1;
                        }
                    }
                    int sum = 0;
                    for (auto& kv : leafDict)
                    {
                        LOG(Helper::LogLevel::LL_Info, "leaf size: %d, nodes: %d. \n", kv.first, kv.second);
                        sum += kv.second;
                    }
                    if (sum != selected.size())
                    {
                        LOG(Helper::LogLevel::LL_Error, "please recheck std computation. \n");
                        return ErrorCode::Fail;
                    }
                    float std = Std(leafSizes.data(), leafSizes.size());
                    LOG(Helper::LogLevel::LL_Info, "standard deviation is %.3f.\n", std);
                }

                return ErrorCode::Success;
            }

            template<typename T>
            std::shared_ptr<COMMON::BKTree> BuildBKT(std::shared_ptr<VectorSet> p_vectorSet, const SPANN::Options& opts) {
                std::shared_ptr<COMMON::BKTree> bkt = std::make_shared<COMMON::BKTree>();
                bkt->m_iBKTKmeansK = opts.m_iBKTKmeansK;
                bkt->m_iBKTLeafSize = opts.m_iBKTLeafSize;
                bkt->m_iSamples = opts.m_iSamples;
                bkt->m_iTreeNumber = opts.m_iTreeNumber;
                bkt->m_fBalanceFactor = opts.m_fBalanceFactor;
                LOG(Helper::LogLevel::LL_Info, "Start invoking BuildTrees.\n");
                LOG(Helper::LogLevel::LL_Info, "BKTKmeansK: %d, BKTLeafSize: %d, Samples: %d, BKTLambdaFactor:%f TreeNumber: %d, ThreadNum: %d.\n",
                    bkt->m_iBKTKmeansK, bkt->m_iBKTLeafSize, bkt->m_iSamples, bkt->m_fBalanceFactor, bkt->m_iTreeNumber, opts.m_iNumberOfThreads);
                VectorSearch::TimeUtils::StopW sw;
                int dataRowsInBlock = opts.m_datasetRowsInBlock;
                int dataCapacity = opts.m_datasetCapacity;
                COMMON::Dataset<T> data(p_vectorSet->Count(), p_vectorSet->Dimension(), dataRowsInBlock, dataCapacity, (T*)(p_vectorSet->GetData()));
                bkt->BuildTrees<T>(data, opts.m_distCalcMethod, opts.m_iNumberOfThreads, nullptr, nullptr, true);
                double elapsedMinutes = sw.getElapsedMin();

                LOG(Helper::LogLevel::LL_Info, "End invoking BuildTrees.\n");
                LOG(Helper::LogLevel::LL_Info, "Invoking BuildTrees used time: %.2lf minutes (about %.2lf hours).\n", elapsedMinutes, elapsedMinutes / 60.0);

                std::stringstream bktFileNameBuilder;
                bktFileNameBuilder << opts.m_vectorPath << ".bkt."
                    << opts.m_iBKTKmeansK << "_"
                    << opts.m_iBKTLeafSize << "_"
                    << opts.m_iTreeNumber << "_"
                    << opts.m_iSamples << "_"
                    << static_cast<int>(opts.m_distCalcMethod) << ".bin";

                std::string bktFileName = bktFileNameBuilder.str();
                if (opts.m_saveBKT) {
                    bkt->SaveTrees(bktFileName);
                }

                return bkt;
            }

            void AdjustOptions(SPANN::Options& p_opts, int p_vectorCount)
            {
                if (p_opts.m_headVectorCount != 0) p_opts.m_ratio = p_opts.m_headVectorCount * 1.0 / p_vectorCount;
                int headCnt = static_cast<int>(std::round(p_opts.m_ratio * p_vectorCount));
                if (headCnt == 0)
                {
                    for (double minCnt = 1; headCnt == 0; minCnt += 0.2)
                    {
                        p_opts.m_ratio = minCnt / p_vectorCount;
                        headCnt = static_cast<int>(std::round(p_opts.m_ratio * p_vectorCount));
                    }

                    LOG(Helper::LogLevel::LL_Info, "Setting requires to select none vectors as head, adjusted it to %d vectors\n", headCnt);
                }

                if (p_opts.m_iBKTKmeansK > headCnt)
                {
                    p_opts.m_iBKTKmeansK = headCnt;
                    LOG(Helper::LogLevel::LL_Info, "Setting of cluster number is less than head count, adjust it to %d\n", headCnt);
                }

                if (p_opts.m_selectThreshold == 0)
                {
                    p_opts.m_selectThreshold = min(p_vectorCount - 1, static_cast<int>(1 / p_opts.m_ratio));
                    LOG(Helper::LogLevel::LL_Info, "Set SelectThreshold to %d\n", p_opts.m_selectThreshold);
                }

                if (p_opts.m_splitThreshold == 0)
                {
                    p_opts.m_splitThreshold = min(p_vectorCount - 1, static_cast<int>(p_opts.m_selectThreshold * 2));
                    LOG(Helper::LogLevel::LL_Info, "Set SplitThreshold to %d\n", p_opts.m_splitThreshold);
                }

                if (p_opts.m_splitFactor == 0)
                {
                    p_opts.m_splitFactor = min(p_vectorCount - 1, static_cast<int>(std::round(1 / p_opts.m_ratio) + 0.5));
                    LOG(Helper::LogLevel::LL_Info, "Set SplitFactor to %d\n", p_opts.m_splitFactor);
                }
            }

            ErrorCode Bootstrap(SPANN::Options& opts) {

                Utils::StopW sw;

                LOG(Helper::LogLevel::LL_Info, "Start loading vector file.\n");
                auto valueType = COMMON::DistanceUtils::Quantizer ? SPTAG::VectorValueType::UInt8 : opts.m_valueType;
                std::shared_ptr<Helper::ReaderOptions> options(new Helper::ReaderOptions(valueType, opts.m_dim, opts.m_vectorType, opts.m_vectorDelimiter));
                auto vectorReader = Helper::VectorSetReader::CreateInstance(options);
                if (ErrorCode::Success != vectorReader->LoadFile(opts.m_vectorPath))
                {
                    LOG(Helper::LogLevel::LL_Error, "Failed to read vector file.\n");
                    exit(1);
                }
                auto vectorSet = vectorReader->GetVectorSet();
                if (opts.m_distCalcMethod == DistCalcMethod::Cosine) vectorSet->Normalize(opts.m_iSelectHeadNumberOfThreads);
                LOG(Helper::LogLevel::LL_Info, "Finish loading vector file.\n");

                AdjustOptions(opts, vectorSet->Count());

                std::vector<int> selected;

                if (Helper::StrUtils::StrEqualIgnoreCase(opts.m_selectType.c_str(), "Random")) {
                    LOG(Helper::LogLevel::LL_Info, "Start generating Random head.\n");
                    selected.resize(vectorSet->Count());
                    for (int i = 0; i < vectorSet->Count(); i++) selected[i] = i;
                    std::shuffle(selected.begin(), selected.end(), rg);

                    int headCnt = static_cast<int>(std::round(opts.m_ratio * vectorSet->Count()));
                    selected.resize(headCnt);

                }
                else if (Helper::StrUtils::StrEqualIgnoreCase(opts.m_selectType.c_str(), "Clustering")) {
                    LOG(Helper::LogLevel::LL_Info, "Start generating Clustering head.\n");
                    int headCnt = static_cast<int>(std::round(opts.m_ratio * vectorSet->Count()));

                    switch (valueType)
                    {
#define DefineVectorValueType(Name, Type) \
    case VectorValueType::Name: \
        Clustering<Type>(vectorSet, opts, selected, headCnt); \
		break;

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType

                    default: break;
                    }

                }
                else if (Helper::StrUtils::StrEqualIgnoreCase(opts.m_selectType.c_str(), "BKT")) {
                    LOG(Helper::LogLevel::LL_Info, "Start generating BKT.\n");
                    std::shared_ptr<COMMON::BKTree> bkt;
                    switch (valueType)
                    {
#define DefineVectorValueType(Name, Type) \
    case VectorValueType::Name: \
        bkt = BuildBKT<Type>(vectorSet, opts); \
		break;

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType

                    default: break;
                    }
                    LOG(Helper::LogLevel::LL_Info, "Finish generating BKT.\n");

                    std::unordered_map<int, int> counter;

                    if (opts.m_calcStd)
                    {
                        CalcLeafSize(0, bkt, counter);
                    }

                    if (opts.m_analyzeOnly)
                    {
                        LOG(Helper::LogLevel::LL_Info, "Analyze Only.\n");

                        std::vector<BKTNodeInfo> bktNodeInfos(bkt->size());

                        // Always use the first tree
                        DfsAnalyze(0, bkt, vectorSet, opts, 0, bktNodeInfos);

                        LOG(Helper::LogLevel::LL_Info, "Analyze Finish.\n");
                    }
                    else {
                        if (SelectHead(vectorSet, bkt, opts, counter, selected) != ErrorCode::Success)
                        {
                            LOG(Helper::LogLevel::LL_Error, "Failed to select head.\n");
                            return ErrorCode::Fail;
                        }
                    }
                }

                LOG(Helper::LogLevel::LL_Info,
                    "Seleted Nodes: %u, about %.2lf%% of total.\n",
                    static_cast<unsigned int>(selected.size()),
                    selected.size() * 100.0 / vectorSet->Count());

                if (!opts.m_noOutput)
                {
                    std::sort(selected.begin(), selected.end());

                    std::shared_ptr<Helper::DiskIO> output = SPTAG::f_createIO(), outputIDs = SPTAG::f_createIO();
                    if (output == nullptr || outputIDs == nullptr ||
                        !output->Initialize(opts.m_headVectorFile.c_str(), std::ios::binary | std::ios::out) ||
                        !outputIDs->Initialize(opts.m_headIDFile.c_str(), std::ios::binary | std::ios::out)) {
                        LOG(Helper::LogLevel::LL_Error, "Failed to create output file:%s %s\n", opts.m_headVectorFile.c_str(), opts.m_headIDFile.c_str());
                        exit(1);
                    }

                    SizeType val = static_cast<SizeType>(selected.size());
                    if (output->WriteBinary(sizeof(val), reinterpret_cast<char*>(&val)) != sizeof(val)) {
                        LOG(Helper::LogLevel::LL_Error, "Failed to write output file!\n");
                        exit(1);
                    }
                    DimensionType dt = vectorSet->Dimension();
                    if (output->WriteBinary(sizeof(dt), reinterpret_cast<char*>(&dt)) != sizeof(dt)) {
                        LOG(Helper::LogLevel::LL_Error, "Failed to write output file!\n");
                        exit(1);
                    }
                    for (auto& ele : selected)
                    {
                        uint64_t vid = static_cast<uint64_t>(ele);
                        if (outputIDs->WriteBinary(sizeof(vid), reinterpret_cast<char*>(&vid)) != sizeof(vid)) {
                            LOG(Helper::LogLevel::LL_Error, "Failed to write output file!\n");
                            exit(1);
                        }
                        if (output->WriteBinary(vectorSet->PerVectorDataSize(), (char*)(vectorSet->GetVector((SizeType)vid))) != vectorSet->PerVectorDataSize()) {
                            LOG(Helper::LogLevel::LL_Error, "Failed to write output file!\n");
                            exit(1);
                        }
                    }
                }

                double elapsedMinutes = sw.getElapsedMin();
                LOG(Helper::LogLevel::LL_Info, "Total used time: %.2lf minutes (about %.2lf hours).\n", elapsedMinutes, elapsedMinutes / 60.0);

                return ErrorCode::Success;
            }
		}
	}
}