// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_KNG_H_
#define _SPTAG_COMMON_KNG_H_

#include "NeighborhoodGraph.h"

namespace SPTAG
{
    namespace COMMON
    {
        class KNearestNeighborhoodGraph : public NeighborhoodGraph
        {
        public:
            KNearestNeighborhoodGraph() { m_pNeighborhoodGraph.SetName("NNG"); }

            void RebuildNeighbors(VectorIndex* index, const SizeType node, SizeType* nodes, const BasicResult* queryResults, const int numResults) {
                DimensionType count = 0;
                for (int j = 0; j < numResults && count < m_iNeighborhoodSize; j++) {
                    const BasicResult& item = queryResults[j];
                    if (item.VID < 0) break;
                    if (item.VID == node) continue;
                    nodes[count++] = item.VID;
                }
                for (DimensionType j = count; j < m_iNeighborhoodSize; j++)  nodes[j] = -1;
            }

            void InsertNeighbors(VectorIndex* index, const SizeType node, SizeType insertNode, float insertDist)
            {
                std::lock_guard<std::mutex> lock(m_dataUpdateLock[node]);

                SizeType* nodes = m_pNeighborhoodGraph[node];
                SizeType tmpNode;
                float tmpDist;
                for (DimensionType k = 0; k < m_iNeighborhoodSize; k++)
                {
                    tmpNode = nodes[k];
                    if (tmpNode < -1) break;

                    if (tmpNode < 0 || (tmpDist = index->ComputeDistance(index->GetSample(node), index->GetSample(tmpNode))) > insertDist
                        || (insertDist == tmpDist && insertNode < tmpNode))
                    {
                        nodes[k] = insertNode;
                        while (tmpNode >= 0 && ++k < m_iNeighborhoodSize && nodes[k] >= -1)
                        {
                            std::swap(tmpNode, nodes[k]);
                        }
                        break;
                    }
                }
            }
        };
    }
}
#endif