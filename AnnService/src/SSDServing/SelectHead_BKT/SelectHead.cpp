#include "inc/SSDServing/Common/stdafx.h"
#include "inc/SSDServing/SelectHead_BKT/SelectHead.h"
#include "inc/SSDServing/SelectHead_BKT/BKTNodeInfo.h"
#include "inc/SSDServing/Common/Utils.h"

namespace SPTAG {
	namespace SSDServing {
		namespace SelectHead_BKT {

            struct HeadCandidate
            {
                HeadCandidate(int n, int c, int d) : nodeID(n), childrenSize(c), depth(d)
                {
                }

                int nodeID;

                int childrenSize;

                int depth;
            };

            // Return height of current Node.
            void DfsSelect(int p_nodeID,
                const shared_ptr<SPTAG::COMMON::BKTree> p_tree,
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
                const shared_ptr<SPTAG::COMMON::BKTree> p_tree,
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


            void SelectHeadStatically(const shared_ptr<COMMON::BKTree> p_tree, const int p_vectorCount, const Options& p_opts, std::vector<int>& p_selected)
            {
                std::vector<HeadCandidate> candidates;
                candidates.reserve(p_tree->size());

                std::vector<BKTNodeInfo> bktNodeInfos(p_tree->size());

                uint32_t selectLimit = static_cast<uint32_t>(p_vectorCount * p_opts.m_ratio) + 1;

                if (selectLimit > p_vectorCount)
                {
                    selectLimit = p_vectorCount;
                }

                std::unordered_set<int> overallSelected;
                overallSelected.reserve(selectLimit * 2);
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
                        fprintf(stderr,
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

                            fprintf(stderr,
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

                fprintf(stderr,
                    "Iteration of depth %d, selected %u heads, about %.2lf%%\n",
                    nextDepth,
                    static_cast<uint32_t>(overallSelected.size() - lastSelected),
                    (overallSelected.size() - lastSelected) * 100.0 / p_vectorCount);

                p_selected.clear();
                p_selected.reserve(overallSelected.size());
                p_selected.assign(overallSelected.begin(), overallSelected.end());

                fprintf(stderr,
                    "Finally selected %u heads, about %.2lf%%\n",
                    static_cast<uint32_t>(p_selected.size()),
                    p_selected.size() * 100.0 / p_vectorCount);

                fprintf(stderr,
                    "Total leaf count %d, big cluster count %d\n",
                    overallLeafCount,
                    bigClusterCount);

                if (p_opts.m_printSizeCount)
                {
                    fprintf(stderr, "Leaf size count:\n");
                    for (const auto& ele : sizeCount)
                    {
                        fprintf(stderr, "%3d %10d\n", ele.first, ele.second);
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

                fprintf(stderr, "Leaf sum count > set leaf size: %d\n", sumGreaterThanLeafSize);

                int covered = DfsCovered(rootIdOfFirstTree, p_tree, overallSelected, false);
                fprintf(stderr,
                    "Total covered leaf count %d, about %.2lf\n",
                    covered,
                    covered * 100.0 / p_vectorCount);
            }


            int SelectHeadDynamicallyInternal(const shared_ptr<COMMON::BKTree> p_tree,
                int p_nodeID,
                const Options& p_opts,
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

                int selectedCenterid;
                if (childrenSize >= p_opts.m_selectThreshold)
                {
                    if (node.centerid < (*p_tree)[0].centerid)
                    {
                        selectedCenterid = node.centerid;
                        p_selected.push_back(selectedCenterid);
                    }

                    if (childrenSize > p_opts.m_splitThreshold)
                    {
                        std::sort(children.begin(), children.end(), [](const CSPair& a, const CSPair& b)
                            {
                                return a.second > b.second;
                            });

                        size_t selectCnt = static_cast<size_t>(std::ceil(childrenSize * 1.0 / p_opts.m_splitFactor) + 0.5);
                        for (size_t i = 0; i < selectCnt && i < children.size(); ++i)
                        {
                            selectedCenterid = (*p_tree)[children[i].first].centerid;
                            p_selected.push_back(selectedCenterid);
                        }
                    }

                    return 0;
                }

                return childrenSize;
            }



            void SelectHeadDynamically(const shared_ptr<COMMON::BKTree> p_tree,
                int p_vectorCount,
                const Options& p_opts,
                std::vector<int>& p_selected)
            {
                p_selected.clear();
                p_selected.reserve(p_vectorCount);

                Options opts = p_opts;

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

                        fprintf(stdout,
                            "Select Threshold: %d, Split Threshold: %d, diff: %.2lf%%.\n",
                            opts.m_selectThreshold,
                            opts.m_splitThreshold,
                            diff * 100.0);

                        if (minDiff > abs(diff))
                        {
                            minDiff = abs(diff);

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

                fprintf(stdout,
                    "Final Select Threshold: %d, Split Threshold: %d.\n",
                    opts.m_selectThreshold,
                    opts.m_splitThreshold);

                p_selected.clear();
                SelectHeadDynamicallyInternal(p_tree, 0, opts, p_selected);
                std::sort(p_selected.begin(), p_selected.end());
                p_selected.erase(std::unique(p_selected.begin(), p_selected.end()), p_selected.end());


            }

			ErrorCode SelectHead(BasicVectorSet vectorSet, shared_ptr<COMMON::BKTree> bkt, Options& opts, unordered_map<int, int>& counter) {
                std::vector<int> selected;
                selected.reserve(vectorSet.Count());

                fprintf(stdout, "Start selecting nodes...\n");
                if (!opts.m_selectDynamically)
                {
                    fprintf(stdout, "Select Head Statically...\n");
                    SelectHeadStatically(bkt, vectorSet.Count(), opts, selected);
                }
                else
                {
                    fprintf(stdout, "Select Head Dynamically...\n");
                    SelectHeadDynamically(bkt, vectorSet.Count(), opts, selected);
                }

                fprintf(stdout,
                    "Seleted Nodes: %u, about %.2lf%% of total.\n",
                    static_cast<unsigned int>(selected.size()),
                    selected.size() * 100.0 / vectorSet.Count());

                if (opts.m_calcStd) {
                    vector<int> leafSizes;
                    for (SizeType& item: selected)
                    {
                        if (counter.count(item) <= 0)
                        {
                            fprintf(stderr, "selected node: %d can't be used to calculate std.", item);
                            return ErrorCode::Fail;
                        }
                        leafSizes.push_back(counter[item]);
                    }
                    map<int, int> leafDict;
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
                        fprintf(stdout, "leaf size: %d, nodes: %d. \n", kv.first, kv.second);
                        sum += kv.second;
                    }
                    if (sum != selected.size())
                    {
                        fprintf(stderr, "please recheck std computation. \n");
                        return ErrorCode::Fail;
                    }
                    float std = Std(leafSizes.data(), leafSizes.size());
                    fprintf(stdout, "standard deviation is %.3f.\n", std);
                }

                if (!opts.m_noOutput)
                {
                    std::sort(selected.begin(), selected.end());

                    std::ofstream output(opts.m_outputVectorFile, std::ios::binary);
                    std::ofstream outputIDs(opts.m_outputIDFile, std::ios::binary);

                    SizeType val = static_cast<SizeType>(selected.size());

                    output.write(reinterpret_cast<char*>(&val), sizeof(val));
                    DimensionType dt = vectorSet.Dimension();
                    output.write(reinterpret_cast<char*>(&dt), sizeof(dt));
                    for (auto& ele : selected)
                    {
                        uint64_t vid = static_cast<uint64_t>(ele);
                        outputIDs.write(reinterpret_cast<char*>(&vid), sizeof(vid));
                        output.write(reinterpret_cast<char*>(vectorSet.GetVector(static_cast<SizeType>(vid))), vectorSet.PerVectorDataSize());
                    }

                    output.close();
                    outputIDs.close();
                }

				return ErrorCode::Success;
			}
		}
	}
}