// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _SPTAG_COMMON_RNG_H_
#define _SPTAG_COMMON_RNG_H_

#include <xmmintrin.h>
#include "NeighborhoodGraph.h"

namespace SPTAG
{
    namespace COMMON
    {
        class RelativeNeighborhoodGraph: public NeighborhoodGraph
        {
        public:
            RelativeNeighborhoodGraph() { m_pNeighborhoodGraph.SetName("RNG"); }

            void RebuildNeighbors(VectorIndex* index, const SizeType node, SizeType* nodes, const BasicResult* queryResults, const int numResults) {
                DimensionType count = 0;
                for (int j = 0; j < numResults && count < m_iNeighborhoodSize; j++) {
                    const BasicResult& item = queryResults[j];
                    if (item.VID < 0) break;
                    if (item.VID == node) continue;

                    bool good = true;
                    for (DimensionType k = 0; k < count; k++) {
                        if (m_fRNGFactor * index->ComputeDistance(index->GetSample(nodes[k]), index->GetSample(item.VID)) < item.Dist) {
                            good = false;
                            break;
                        }
                    }
                    if (good) nodes[count++] = item.VID;
                }
                for (DimensionType j = count; j < m_iNeighborhoodSize; j++)  nodes[j] = -1;
            }

            void InsertNeighbors(VectorIndex* index, const SizeType node, SizeType insertNode, float insertDist)
            {                
                SizeType* nodes = m_pNeighborhoodGraph[node];
                const void* nodeVec = index->GetSample(node);
                const void* insertVec = index->GetSample(insertNode);
                
                std::lock_guard<std::mutex> lock(m_dataUpdateLock[node]);

                _mm_prefetch((const char*)nodes, _MM_HINT_T0);
                _mm_prefetch((const char*)(nodeVec), _MM_HINT_T0);
                _mm_prefetch((const char*)(insertVec), _MM_HINT_T0);
                for (DimensionType i = 0; i < m_iNeighborhoodSize; i++) {
                    _mm_prefetch((const char*)(index->GetSample(nodes[i])), _MM_HINT_T0);
                }

                SizeType tmpNode;
                float tmpDist;
                const void* tmpVec;
                int checkNeighborhoodSize = (nodes[m_iNeighborhoodSize - 1] < -1) ? m_iNeighborhoodSize - 1 : m_iNeighborhoodSize;
                for (DimensionType k = 0; k < checkNeighborhoodSize; k++)
                {
                    tmpNode = nodes[k];
                    if (tmpNode < 0) {
                        nodes[k] = insertNode;
                        break;
                    }

                    tmpVec = index->GetSample(tmpNode);
                    tmpDist = index->ComputeDistance(tmpVec, nodeVec);
                    if (tmpDist > insertDist || (insertDist == tmpDist && insertNode < tmpNode))
                    {
                        nodes[k] = insertNode;
                        while (++k < checkNeighborhoodSize && index->ComputeDistance(tmpVec, nodeVec) <= index->ComputeDistance(tmpVec, insertVec)) {
                            std::swap(tmpNode, nodes[k]);
                            if (tmpNode < 0) return;
                            tmpVec = index->GetSample(tmpNode);
                        }
                        break;
                    }
                    else if (index->ComputeDistance(tmpVec, insertVec) < insertDist) {
                        break;
                    }
                }
            }
        };
    }
}
#endif