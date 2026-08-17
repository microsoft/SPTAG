// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/BKT/Index.h"
#include "inc/Core/ResultIterator.h"
#include <chrono>

#pragma warning(disable : 4242) // '=' : conversion from 'int' to 'short', possible loss of data
#pragma warning(disable : 4244) // '=' : conversion from 'int' to 'short', possible loss of data
#pragma warning(disable : 4127) // conditional expression is constant

namespace SPTAG
{
template <typename T> thread_local std::unique_ptr<T> COMMON::ThreadLocalWorkSpaceFactory<T>::m_workspace;

namespace BKT
{

template <typename T> ErrorCode Index<T>::LoadConfig(Helper::IniReader &p_reader)
{
#define DefineBKTParameter(VarName, VarType, DefaultValue, RepresentStr)                                               \
    SetParameter(RepresentStr, p_reader.GetParameter("Index", RepresentStr, std::string(#DefaultValue)).c_str());

#include "inc/Core/BKT/ParameterDefinitionList.h"
#undef DefineBKTParameter
    return ErrorCode::Success;
}

template <> void Index<std::uint8_t>::SetQuantizer(std::shared_ptr<SPTAG::COMMON::IQuantizer> quantizer)
{
    m_pQuantizer = quantizer;
    m_pTrees.m_pQuantizer = quantizer;
    if (m_pQuantizer)
    {
        m_fComputeDistance = m_pQuantizer->DistanceCalcSelector<std::uint8_t>(m_iDistCalcMethod);
        m_iBaseSquare =
            (m_iDistCalcMethod == DistCalcMethod::Cosine) ? m_pQuantizer->GetBase() * m_pQuantizer->GetBase() : 1;
    }
    else
    {
        m_fComputeDistance = COMMON::DistanceCalcSelector<std::uint8_t>(m_iDistCalcMethod);
        m_iBaseSquare = (m_iDistCalcMethod == DistCalcMethod::Cosine)
                            ? COMMON::Utils::GetBase<std::uint8_t>() * COMMON::Utils::GetBase<std::uint8_t>()
                            : 1;
    }
}

template <typename T> void Index<T>::SetQuantizer(std::shared_ptr<SPTAG::COMMON::IQuantizer> quantizer)
{
    m_pQuantizer = quantizer;
    m_pTrees.m_pQuantizer = quantizer;
    if (quantizer)
    {
        SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Error,
                     "Set non-null quantizer for index with data type other than BYTE");
    }
}

template <typename T> ErrorCode Index<T>::LoadIndexDataFromMemory(const std::vector<ByteArray> &p_indexBlobs)
{
    if (p_indexBlobs.size() < 3)
        return ErrorCode::LackOfInputs;

    if (m_pSamples.Load((char *)p_indexBlobs[0].Data(), m_iDataBlockSize, m_iDataCapacity) != ErrorCode::Success)
        return ErrorCode::FailedParseValue;
    if (m_pTrees.LoadTrees((char *)p_indexBlobs[1].Data()) != ErrorCode::Success)
        return ErrorCode::FailedParseValue;
    if (m_pGraph.LoadGraph((char *)p_indexBlobs[2].Data(), m_iDataBlockSize, m_iDataCapacity) != ErrorCode::Success)
        return ErrorCode::FailedParseValue;
    if (p_indexBlobs.size() <= 3)
        m_deletedID.Initialize(m_pSamples.R(), m_iDataBlockSize, m_iDataCapacity,
                               COMMON::LabelSet::InvalidIDBehavior::AlwaysContains);
    else if (m_deletedID.Load((char *)p_indexBlobs[3].Data(), m_iDataBlockSize, m_iDataCapacity,
                              COMMON::LabelSet::InvalidIDBehavior::AlwaysContains) != ErrorCode::Success)
        return ErrorCode::FailedParseValue;

    if (m_pSamples.R() != m_pGraph.R() || m_pSamples.R() != m_deletedID.R())
    {
        SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Error,
                     "Index data is corrupted, please rebuild the index. Samples: %i, Graph: %i, DeletedID: %i.",
                     m_pSamples.R(), m_pGraph.R(), m_deletedID.R());
        return ErrorCode::FailedParseValue;
    }

    m_pGraph.m_iThreadNum = m_iNumberOfThreads;
    m_threadPool.init();
    return ErrorCode::Success;
}

template <typename T>
ErrorCode Index<T>::LoadIndexData(const std::vector<std::shared_ptr<Helper::DiskIO>> &p_indexStreams)
{
    if (p_indexStreams.size() < 4)
        return ErrorCode::LackOfInputs;

    ErrorCode ret = ErrorCode::Success;
    if (p_indexStreams[0] == nullptr ||
        (ret = m_pSamples.Load(p_indexStreams[0], m_iDataBlockSize, m_iDataCapacity)) != ErrorCode::Success)
        return ret;
    if (p_indexStreams[1] == nullptr || (ret = m_pTrees.LoadTrees(p_indexStreams[1])) != ErrorCode::Success)
        return ret;
    if (p_indexStreams[2] == nullptr ||
        (ret = m_pGraph.LoadGraph(p_indexStreams[2], m_iDataBlockSize, m_iDataCapacity)) != ErrorCode::Success)
        return ret;
    if (p_indexStreams[3] == nullptr)
        m_deletedID.Initialize(m_pSamples.R(), m_iDataBlockSize, m_iDataCapacity,
                               COMMON::LabelSet::InvalidIDBehavior::AlwaysContains);
    else if ((ret = m_deletedID.Load(p_indexStreams[3], m_iDataBlockSize, m_iDataCapacity,
                                     COMMON::LabelSet::InvalidIDBehavior::AlwaysContains)) != ErrorCode::Success)
        return ret;

    if (m_pSamples.R() != m_pGraph.R() || m_pSamples.R() != m_deletedID.R())
    {
        SPTAGLIB_LOG(SPTAG::Helper::LogLevel::LL_Error,
                     "Index data is corrupted, please rebuild the index. Samples: %i, Graph: %i, DeletedID: %i.",
                     m_pSamples.R(), m_pGraph.R(), m_deletedID.R());
        return ErrorCode::FailedParseValue;
    }

    m_pGraph.m_iThreadNum = m_iNumberOfThreads;
    m_threadPool.init();
    return ret;
}

template <typename T> ErrorCode Index<T>::SaveConfig(std::shared_ptr<Helper::DiskIO> p_configOut)
{
    auto workspace = m_workSpaceFactory->GetWorkSpace();
    if (workspace)
    {
        m_iHashTableExp = workspace->HashTableExponent();
    }
    m_workSpaceFactory->ReturnWorkSpace(std::move(workspace));

#define DefineBKTParameter(VarName, VarType, DefaultValue, RepresentStr)                                               \
    IOSTRING(p_configOut, WriteString,                                                                                 \
             (RepresentStr + std::string("=") + GetParameter(RepresentStr) + std::string("\n")).c_str());

#include "inc/Core/BKT/ParameterDefinitionList.h"
#undef DefineBKTParameter

    IOSTRING(p_configOut, WriteString, "\n");
    return ErrorCode::Success;
}

template <typename T>
ErrorCode Index<T>::SaveIndexData(const std::vector<std::shared_ptr<Helper::DiskIO>> &p_indexStreams)
{
    if (p_indexStreams.size() < 4)
        return ErrorCode::LackOfInputs;

    std::lock_guard<std::mutex> lock(m_dataAddLock);
    std::unique_lock<std::shared_timed_mutex> uniquelock(m_dataDeleteLock);

    ErrorCode ret = ErrorCode::Success;
    if ((ret = m_pSamples.Save(p_indexStreams[0])) != ErrorCode::Success)
        return ret;
    if ((ret = m_pTrees.SaveTrees(p_indexStreams[1])) != ErrorCode::Success)
        return ret;
    if ((ret = m_pGraph.SaveGraph(p_indexStreams[2])) != ErrorCode::Success)
        return ret;
    if ((ret = m_deletedID.Save(p_indexStreams[3])) != ErrorCode::Success)
        return ret;
    return ret;
}

#pragma region K - NN search
/*

#define Search(CheckDeleted, CheckDuplicated, CheckFilter) \
                std::shared_lock<std::shared_timed_mutex> lock(*(m_pTrees.m_lock)); \
        m_pTrees.InitSearchTrees(m_pSamples, m_fComputeDistance, p_query, p_space); \
        m_pTrees.SearchTrees(m_pSamples, m_fComputeDistance, p_query, p_space, m_iNumberOfInitialDynamicPivots); \
        const DimensionType checkPos = m_pGraph.m_iNeighborhoodSize - 1; \
        while (!p_space.m_NGQueue.empty()) { \
            NodeDistPair gnode = p_space.m_NGQueue.pop(); \
            SizeType tmpNode = gnode.node; \
            const SizeType *node = m_pGraph[tmpNode]; \
            _mm_prefetch((const char *)node, _MM_HINT_T0); \
            for (DimensionType i = 0; i <= checkPos; i++) { \
                if (node[i] < 0 || node[i] >= m_pSamples.R()) break; \
                _mm_prefetch((const char *)(m_pSamples)[node[i]], _MM_HINT_T0); \
            } \
            if (gnode.distance <= p_query.worstDist()) { \
                SizeType checkNode = node[checkPos]; \
                if (checkNode < -1) { \
                    const COMMON::BKTNode& tnode = m_pTrees[-2 - checkNode]; \
                    SizeType i = -tnode.childStart; \
                    do { \
                        CheckDeleted \
                        { \
                            CheckFilter \
                            { \
                                CheckDuplicated \
                                break; \
                            } \
                        } \
                        tmpNode = m_pTrees[i].centerid; \
                    } while (i++ < tnode.childEnd); \
               } else { \
                   CheckDeleted \
                   { \
                       CheckFilter \
                       { \
                           p_query.AddPoint(tmpNode, gnode.distance); \
                       } \
                   } \
               } \
            } else { \
                CheckDeleted \
                { \
                    if (gnode.distance > p_space.m_Results.worst() || p_space.m_iNumberOfCheckedLeaves >
p_space.m_iMaxCheck) { \
                        p_query.SortResult(); return; \
                    } \
                } \
            } \
            for (DimensionType i = 0; i <= checkPos; i++) { \
                SizeType nn_index = node[i]; \
                if (nn_index < 0) break; \
                if (nn_index >= m_pSamples.R()) continue; \
                if (p_space.CheckAndSet(nn_index)) continue; \
                float distance2leaf = m_fComputeDistance(p_query.GetQuantizedTarget(), (m_pSamples)[nn_index],
GetFeatureDim()); \
                p_space.m_iNumberOfCheckedLeaves++; \
                if (p_space.m_Results.insert(distance2leaf)) { \
                    p_space.m_NGQueue.insert(NodeDistPair(nn_index, distance2leaf)); \
                } \
            } \
            if (p_space.m_NGQueue.Top().distance > p_space.m_SPTQueue.Top().distance) { \
                m_pTrees.SearchTrees(m_pSamples, m_fComputeDistance, p_query, p_space, m_iNumberOfOtherDynamicPivots +
p_space.m_iNumberOfCheckedLeaves); \
            } \
        } \
        p_query.SortResult(); \
*/

/*
#define SearchIterative(CheckDeleted, p_isFirst, batch) \
        if (p_isFirst) { \
            m_pTrees.InitSearchTrees(m_pSamples, m_fComputeDistance, p_query, p_space); \
            m_pTrees.SearchTrees(m_pSamples, m_fComputeDistance, p_query, p_space, m_iNumberOfInitialDynamicPivots); \
        } \
        const DimensionType checkPos = m_pGraph.m_iNeighborhoodSize - 1; \
        while (!p_space.m_NGQueue.empty()) { \
            NodeDistPair gnode = p_space.m_NGQueue.pop(); \
            SizeType tmpNode = gnode.node; \
            const SizeType *node = m_pGraph[tmpNode]; \
            _mm_prefetch((const char *)node, _MM_HINT_T0); \
            for (DimensionType i = 0; i <= checkPos; i++) { \
                _mm_prefetch((const char *)(m_pSamples)[node[i]], _MM_HINT_T0); \
            } \
            CheckDeleted \
                { \
                    p_query.AddPoint(tmpNode, gnode.distance); \
                    count++; \
                    if (gnode.distance > p_space.m_Results.worst() || p_space.m_iNumberOfCheckedLeaves >
p_space.m_iMaxCheck) { \
                        p_space.m_relaxedMono = true; \
                    } \
                } \
            SizeType checkNode = node[checkPos]; \
            if (checkNode < -1) { \
                const COMMON::BKTNode& tnode = m_pTrees[-2 - checkNode]; \
                SizeType i = -tnode.childStart; \
                while (i < tnode.childEnd) { \
                    tmpNode = m_pTrees[i].centerid; \
                    CheckDeleted \
                    { \
                        float distance2leaf = m_fComputeDistance(p_query.GetQuantizedTarget(), (m_pSamples)[tmpNode],
GetFeatureDim()); \
                        if (!p_space.CheckAndSet(tmpNode)) { \
                            p_space.m_NGQueue.insert(NodeDistPair(tmpNode, distance2leaf)); \
                        } \
                    } \
                    i++; \
                }\
            } \
                for (DimensionType i = 0; i <= checkPos; i++) { \
                    SizeType nn_index = node[i]; \
                    if (nn_index < 0) break; \
                    if (p_space.CheckAndSet(nn_index)) continue; \
                    float distance2leaf = m_fComputeDistance(p_query.GetQuantizedTarget(), (m_pSamples)[nn_index],
GetFeatureDim()); \
                    p_space.m_iNumberOfCheckedLeaves++; \
                    p_space.m_NGQueue.insert(NodeDistPair(nn_index, distance2leaf)); \
                    p_space.m_Results.insert(distance2leaf); \
                } \
            if (p_space.m_NGQueue.Top().distance > p_space.m_SPTQueue.Top().distance) { \
                m_pTrees.SearchTrees(m_pSamples, m_fComputeDistance, p_query, p_space, m_iNumberOfOtherDynamicPivots +
p_space.m_iNumberOfCheckedLeaves); \
            } \
            if (count >= batch) {\
                break; \
            } \
        } \
        p_query.SortResult(); \
*/

template <typename T>
template <bool EnableCrossEdges,
          bool (*notDeleted)(const COMMON::LabelSet &, SizeType),
          bool (*isDup)(COMMON::QueryResultSet<T> &, SizeType, float),
          bool (*checkFilter)(const std::shared_ptr<MetadataSet> &, SizeType, std::function<bool(const ByteArray &)>)>
void Index<T>::Search(COMMON::QueryResultSet<T> &p_query, COMMON::WorkSpace &p_space,
                      std::function<bool(const ByteArray &)> filterFunc,
                      const CrossGraphSearchContext* p_crossContext,
                      CrossGraphSearchStats* p_crossStats,
                      std::function<bool(SizeType)> p_resultFilter) const
{
    std::shared_lock<std::shared_timed_mutex> treeLock;
    std::vector<std::shared_lock<std::shared_timed_mutex>> crossTreeLocks;
    if constexpr (EnableCrossEdges)
    {
        crossTreeLocks.reserve(p_crossContext->m_nodes.size());
        for (const auto& nodeContext : p_crossContext->m_nodes)
        {
            if (nodeContext.m_index != nullptr)
            {
                crossTreeLocks.emplace_back(
                    *(nodeContext.m_index->m_pTrees.m_lock));
            }
        }
    }
    else
    {
        treeLock = std::shared_lock<std::shared_timed_mutex>(
            *(m_pTrees.m_lock));
    }
    std::chrono::high_resolution_clock::time_point treeStart;
    if constexpr (EnableCrossEdges)
    {
        treeStart = std::chrono::high_resolution_clock::now();
    }
    m_pTrees.InitSearchTrees(m_pSamples, m_fComputeDistance, p_query, p_space);
    m_pTrees.SearchTrees(m_pSamples, m_fComputeDistance, p_query, p_space, m_iNumberOfInitialDynamicPivots);
    std::chrono::high_resolution_clock::time_point graphStart;

    if constexpr (EnableCrossEdges)
    {
        graphStart = std::chrono::high_resolution_clock::now();
        if (p_crossStats != nullptr)
        {
            p_crossStats->m_seeded = static_cast<int>(p_space.m_NGQueue.size());
            p_crossStats->m_seedChecked = p_space.m_iNumberOfCheckedLeaves;
            p_crossStats->m_treeSearchMs =
                std::chrono::duration<double, std::milli>(
                    graphStart - treeStart)
                    .count();
        }
    }
    const auto routeDistanceForLocal =
        [&](int p_nodeID,
            const Index<T>* p_index,
            SizeType p_localID,
            float p_vectorDistance) {
            if constexpr (EnableCrossEdges)
            {
                if (p_crossContext->m_useHybridDistance &&
                    p_crossContext->m_queryDistance)
                {
                    float best =
                        p_crossContext->m_queryDistance(
                            p_nodeID, p_localID,
                            p_vectorDistance);
                    const DimensionType checkPosition =
                        p_index->m_pGraph
                            .m_iNeighborhoodSize -
                        1;
                    const SizeType checkNode =
                        p_index->m_pGraph[p_localID]
                            [checkPosition];
                    if (checkNode < -1)
                    {
                        const COMMON::BKTNode& treeNode =
                            p_index->m_pTrees[
                                -2 - checkNode];
                        SizeType treeIndex =
                            -treeNode.childStart;
                        SizeType collapsedLocal =
                            p_localID;
                        do
                        {
                            const float collapsedVectorDistance =
                                collapsedLocal == p_localID
                                    ? p_vectorDistance
                                    : p_index->m_fComputeDistance(
                                          p_query.GetQuantizedTarget(),
                                          p_index->m_pSamples[
                                              collapsedLocal],
                                          p_index->GetFeatureDim());
                            best = (std::min)(
                                best,
                                p_crossContext
                                    ->m_queryDistance(
                                        p_nodeID,
                                        collapsedLocal,
                                        collapsedVectorDistance));
                            if (treeIndex <= 0) break;
                            collapsedLocal =
                                p_index->m_pTrees[
                                    treeIndex]
                                    .centerid;
                        } while (
                            treeIndex++ <
                            treeNode.childEnd);
                    }
                    return best;
                }
            }
            return p_vectorDistance;
        };

    if constexpr (EnableCrossEdges)
    {
        if (p_crossContext->m_useHybridDistance)
        {
            std::vector<NodeDistPair> seeds;
            seeds.reserve(static_cast<size_t>(p_space.m_NGQueue.size()));
            while (!p_space.m_NGQueue.empty())
            {
                NodeDistPair seed = p_space.m_NGQueue.pop();
                if (seed.node < 0 ||
                    seed.node >=
                        p_crossContext
                            ->m_nodes[static_cast<size_t>(
                                p_crossContext->m_entryNode)]
                            .m_index->GetNumSamples())
                {
                    continue;
                }
                if (p_crossContext->m_queryDistance)
                {
                    const auto* entryIndex =
                        p_crossContext
                            ->m_nodes[
                                static_cast<size_t>(
                                    p_crossContext
                                        ->m_entryNode)]
                            .m_index;
                    seed.distance =
                        routeDistanceForLocal(
                            p_crossContext
                                ->m_entryNode,
                            entryIndex, seed.node,
                            seed.distance);
                }
                seeds.push_back(seed);
            }
            p_space.m_Results.clear((std::max)(
                p_space.m_iMaxCheck / 16, p_query.GetResultNum()));
            for (const auto& seed : seeds)
            {
                p_space.m_NGQueue.insert(seed);
                p_space.m_Results.insert(seed.distance);
            }
        }
    }

    const auto toQueryNodeCode = [p_crossContext](int p_nodeId) {
        if (p_nodeId == p_crossContext->m_entryNode) return 0;
        if (p_nodeId == 0) return p_crossContext->m_entryNode;
        return p_nodeId;
    };
    const auto fromQueryNodeCode = [p_crossContext](int p_nodeCode) {
        if (p_nodeCode == 0) return p_crossContext->m_entryNode;
        if (p_nodeCode == p_crossContext->m_entryNode) return 0;
        return p_nodeCode;
    };
    const auto encodeQueryLocator =
        [p_crossContext, &toQueryNodeCode](
            int p_nodeId, SizeType p_localId, SizeType& p_locator) -> bool {
            if (p_nodeId < 0 || p_localId < 0 ||
                static_cast<std::uint64_t>(p_localId) >
                    static_cast<std::uint64_t>(p_crossContext->m_locatorLocalMask))
            {
                return false;
            }
            const std::uint64_t encoded =
                (static_cast<std::uint64_t>(toQueryNodeCode(p_nodeId))
                    << p_crossContext->m_locatorLocalBits) |
                static_cast<std::uint64_t>(p_localId);
            if (encoded >= static_cast<std::uint64_t>(MaxSize))
            {
                return false;
            }
            p_locator = static_cast<SizeType>(encoded);
            return true;
        };
    const auto decodeQueryLocator =
        [p_crossContext, &fromQueryNodeCode](
            SizeType p_locator, int& p_nodeId, SizeType& p_localId) -> bool {
            if (p_locator < 0) return false;
            const int nodeCode = static_cast<int>(
                static_cast<std::uint64_t>(p_locator) >>
                p_crossContext->m_locatorLocalBits);
            p_nodeId = fromQueryNodeCode(nodeCode);
            p_localId = static_cast<SizeType>(
                static_cast<std::uint64_t>(p_locator) &
                static_cast<std::uint64_t>(p_crossContext->m_locatorLocalMask));
            if (p_nodeId < 0 ||
                p_nodeId >= static_cast<int>(p_crossContext->m_nodes.size()))
            {
                return false;
            }
            const auto& nodeContext =
                p_crossContext->m_nodes[static_cast<size_t>(p_nodeId)];
            return nodeContext.m_index != nullptr &&
                nodeContext.m_localToGlobal != nullptr &&
                p_localId >= 0 &&
                p_localId < nodeContext.m_index->GetNumSamples() &&
                static_cast<size_t>(p_localId) <
                    nodeContext.m_localToGlobal->size();
        };
    const auto decodeStoredLocator =
        [p_crossContext](
            SizeType p_locator, int& p_nodeId, SizeType& p_localId) -> bool {
            if (p_locator < 0) return false;
            p_nodeId = static_cast<int>(
                static_cast<std::uint64_t>(p_locator) >>
                p_crossContext->m_locatorLocalBits);
            p_localId = static_cast<SizeType>(
                static_cast<std::uint64_t>(p_locator) &
                static_cast<std::uint64_t>(p_crossContext->m_locatorLocalMask));
            if (p_nodeId < 0 ||
                p_nodeId >= static_cast<int>(p_crossContext->m_nodes.size()))
            {
                return false;
            }
            const auto& nodeContext =
                p_crossContext->m_nodes[static_cast<size_t>(p_nodeId)];
            return nodeContext.m_index != nullptr &&
                nodeContext.m_localToGlobal != nullptr &&
                p_localId >= 0 &&
                p_localId < nodeContext.m_index->GetNumSamples() &&
                static_cast<size_t>(p_localId) <
                    nodeContext.m_localToGlobal->size();
        };
    const auto admitFilteredResult =
        [&](SizeType p_key,
            const Index<T>* p_index,
            SizeType p_local,
            SizeType p_result,
            float p_distance) {
            if (!p_resultFilter || p_result < 0 ||
                p_space.CheckResultAndSet(p_key)) {
                return false;
            }
            return p_resultFilter(p_result) &&
                notDeleted(
                    p_index->m_deletedID, p_local) &&
                checkFilter(
                    p_index->m_pMetadata, p_local,
                    filterFunc) &&
                isDup(
                    p_query, p_result, p_distance);
        };
    const auto admitCollapsedResults =
        [&](int p_nodeID,
            const Index<T>* p_index,
            const std::vector<SizeType>*
                p_localToGlobal,
            SizeType p_representative,
            float p_representativeDistance) {
            const DimensionType checkPosition =
                p_index->m_pGraph
                    .m_iNeighborhoodSize -
                1;
            const SizeType checkNode =
                p_index->m_pGraph[
                    p_representative][
                    checkPosition];
            if (checkNode >= -1) return;

            const bool useHybridCollapsed =
                EnableCrossEdges &&
                p_crossContext->m_useHybridDistance;
            const COMMON::BKTNode& treeNode =
                p_index->m_pTrees[
                    -2 - checkNode];
            SizeType treeIndex =
                -treeNode.childStart;
            SizeType collapsedLocal =
                p_representative;
            do
            {
                SizeType collapsedResult =
                    collapsedLocal;
                if constexpr (EnableCrossEdges)
                {
                    if (collapsedLocal < 0 ||
                        p_localToGlobal == nullptr ||
                        static_cast<size_t>(
                            collapsedLocal) >=
                            p_localToGlobal->size())
                    {
                        break;
                    }
                    collapsedResult =
                        (*p_localToGlobal)[
                            static_cast<size_t>(
                                collapsedLocal)];
                }

                float collapsedDistance =
                    p_representativeDistance;
                if constexpr (EnableCrossEdges)
                {
                    if (p_crossContext
                            ->m_useHybridDistance)
                    {
                        const float vectorDistance =
                            p_index
                                ->m_fComputeDistance(
                                    p_query
                                        .GetQuantizedTarget(),
                                    p_index->m_pSamples[
                                        collapsedLocal],
                                    GetFeatureDim());
                        collapsedDistance =
                            p_crossContext
                                ->m_queryDistance(
                                    p_nodeID,
                                    collapsedLocal,
                                    vectorDistance);
                    }
                }

                bool admitted = false;
                if (p_resultFilter)
                {
                    SizeType collapsedKey =
                        collapsedLocal;
                    if constexpr (EnableCrossEdges)
                    {
                        if (!encodeQueryLocator(
                                p_nodeID,
                                collapsedLocal,
                                collapsedKey))
                        {
                            collapsedKey = -1;
                        }
                    }
                    if (collapsedKey >= 0)
                    {
                        admitted =
                            admitFilteredResult(
                                collapsedKey,
                                p_index,
                                collapsedLocal,
                                collapsedResult,
                                collapsedDistance);
                    }
                }
                else if (
                    collapsedResult >= 0 &&
                    notDeleted(
                        p_index->m_deletedID,
                        collapsedLocal) &&
                    checkFilter(
                        p_index->m_pMetadata,
                        collapsedLocal,
                        filterFunc))
                {
                    admitted = isDup(
                        p_query,
                        collapsedResult,
                        collapsedDistance);
                }
                if (admitted &&
                    !useHybridCollapsed)
                {
                    break;
                }
                if (treeIndex <= 0) break;
                collapsedLocal =
                    p_index->m_pTrees[
                        treeIndex].centerid;
            } while (
                treeIndex++ <
                treeNode.childEnd);
        };
    const auto finishSearch = [&]() {
        if constexpr (EnableCrossEdges)
        {
            if (p_crossStats != nullptr)
            {
                p_crossStats->m_checked = p_space.m_iNumberOfCheckedLeaves;
                p_crossStats->m_graphSearchMs =
                    std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - graphStart)
                        .count();
            }
        }
        p_query.SortResult();
    };

    while (!p_space.m_NGQueue.empty())
    {
        NodeDistPair gnode = p_space.m_NGQueue.pop();
        SizeType currentLocal = gnode.node;
        int currentNode = 0;
        const Index<T>* currentIndex = this;
        const std::vector<SizeType>* currentLocalToGlobal = nullptr;
        if constexpr (EnableCrossEdges)
        {
            if (!decodeQueryLocator(gnode.node, currentNode, currentLocal))
            {
                continue;
            }
            const auto& nodeContext =
                p_crossContext->m_nodes[static_cast<size_t>(currentNode)];
            currentIndex = nodeContext.m_index;
            currentLocalToGlobal = nodeContext.m_localToGlobal;
            if (p_crossStats != nullptr) ++p_crossStats->m_expanded;
        }

        SizeType resultNode = currentLocal;
        if constexpr (EnableCrossEdges)
        {
            resultNode =
                (*currentLocalToGlobal)[static_cast<size_t>(currentLocal)];
            if (resultNode < 0) continue;
        }

        const DimensionType localEdgeCount =
            currentIndex->m_pGraph.m_iNeighborhoodSize;
        const DimensionType runtimeCrossEdgeCount =
            EnableCrossEdges
                ? currentIndex->m_pGraph
                      .GetRuntimeEdgeSuffixSize()
                : 0;
        const DimensionType crossEdgeBegin = localEdgeCount;
        const DimensionType edgeCount = crossEdgeBegin +
            runtimeCrossEdgeCount;
        const DimensionType checkPos = localEdgeCount - 1;
        const SizeType* node = currentIndex->m_pGraph[currentLocal];

        if constexpr (!EnableCrossEdges)
        {
            COMMON::PrefetchGraphNeighbors(
                node, localEdgeCount,
                [this](SizeType neighbor) -> const void* {
                    return neighbor >= 0 && neighbor < m_pSamples.R()
                        ? m_pSamples[neighbor]
                        : nullptr;
                });
        }

        SizeType checkNode = node[checkPos];
        const bool hybridCollapsed =
            checkNode < -1 &&
            EnableCrossEdges &&
            p_crossContext->m_useHybridDistance;
        if (gnode.distance <= p_query.worstDist() ||
            hybridCollapsed)
        {
            if (checkNode < -1)
            {
                admitCollapsedResults(
                    currentNode, currentIndex,
                    currentLocalToGlobal,
                    currentLocal,
                    gnode.distance);
            }
            else if (p_resultFilter)
            {
                admitFilteredResult(
                    gnode.node, currentIndex,
                    currentLocal, resultNode,
                    gnode.distance);
            }
            else if (notDeleted(
                         currentIndex->m_deletedID,
                         currentLocal) &&
                     checkFilter(
                         currentIndex->m_pMetadata,
                         currentLocal,
                         filterFunc))
            {
                p_query.AddPoint(resultNode, gnode.distance);
            }
        }
        else if (notDeleted(
                     currentIndex->m_deletedID,
                     currentLocal) &&
                 ((gnode.distance >
                       p_space.m_Results.worst() &&
                   (!p_resultFilter ||
                    p_query.worstDist() < MaxDist)) ||
                  p_space.m_iNumberOfCheckedLeaves >
                      p_space.m_iMaxCheck))
        {
            finishSearch();
            return;
        }
        if (p_resultFilter &&
            p_query.worstDist() == MaxDist &&
            p_space.m_iNumberOfCheckedLeaves >
                p_space.m_iMaxCheck)
        {
            finishSearch();
            return;
        }

        const auto expandEdges =
            [&](const SizeType* p_edges,
                DimensionType p_begin,
                DimensionType p_end,
                bool p_crossEncoded)
        {
            for (DimensionType edge = p_begin;
                 edge < p_end; ++edge)
            {
                const bool isCross =
                    EnableCrossEdges && p_crossEncoded;
                if constexpr (EnableCrossEdges)
                {
                    constexpr DimensionType kPrefetchAhead = 4;
                    const DimensionType futureEdge =
                        edge + kPrefetchAhead;
                    if (futureEdge < p_end)
                    {
                        const bool futureIsCross =
                            p_crossEncoded;
                        const SizeType futureValue =
                            p_edges[futureEdge];
                        if (futureValue >= 0)
                        {
                            int futureNode = currentNode;
                            SizeType futureLocal = futureValue;
                            SizeType futureLocator = -1;
                            const bool valid = futureIsCross
                                ? decodeStoredLocator(
                                      futureValue,
                                      futureNode,
                                      futureLocal) &&
                                    encodeQueryLocator(
                                        futureNode,
                                        futureLocal,
                                        futureLocator)
                                : encodeQueryLocator(
                                    futureNode,
                                    futureLocal,
                                    futureLocator);
                            if (valid)
                            {
                                const auto& futureContext =
                                    p_crossContext->m_nodes[
                                        static_cast<size_t>(
                                            futureNode)];
                                _mm_prefetch(
                                    reinterpret_cast<const char*>(
                                        futureContext.m_index
                                            ->m_pSamples[
                                                futureLocal]),
                                    _MM_HINT_T0);
                            }
                        }
                    }
                }
                const SizeType edgeValue = p_edges[edge];
                if (edgeValue < 0)
                {
                    if (isCross) break;
                    if constexpr (EnableCrossEdges)
                    {
                        continue;
                    }
                    else
                    {
                        break;
                    }
                }

                SizeType targetKey = edgeValue;
                int targetNode = currentNode;
                SizeType targetLocal = edgeValue;
                const Index<T>* targetIndex = currentIndex;
                if constexpr (EnableCrossEdges)
                {
                    const bool valid = isCross
                        ? decodeStoredLocator(
                              edgeValue,
                              targetNode,
                              targetLocal) &&
                            encodeQueryLocator(
                                targetNode,
                                targetLocal,
                                targetKey)
                        : encodeQueryLocator(
                            targetNode,
                            targetLocal,
                            targetKey);
                    if (!valid) continue;
                    targetIndex =
                        p_crossContext->m_nodes[
                            static_cast<size_t>(
                                targetNode)]
                            .m_index;
                    if (isCross &&
                        p_crossStats != nullptr)
                    {
                        ++p_crossStats->m_crossEdges;
                    }
                }

                if (targetLocal < 0 ||
                    targetLocal >=
                        targetIndex->m_pSamples.R() ||
                    p_space.CheckAndSet(targetKey))
                {
                    continue;
                }
                const float distance =
                    m_fComputeDistance(
                        p_query.GetQuantizedTarget(),
                        targetIndex
                            ->m_pSamples[targetLocal],
                        GetFeatureDim());
                const float routeDistance =
                    EnableCrossEdges &&
                        p_crossContext->m_useHybridDistance &&
                        p_crossContext->m_queryDistance
                    ? routeDistanceForLocal(
                          targetNode,
                          targetIndex,
                          targetLocal,
                          distance)
                    : distance;
                ++p_space.m_iNumberOfCheckedLeaves;
                if (p_resultFilter)
                {
                    const DimensionType
                        targetCheckPosition =
                            targetIndex->m_pGraph
                                .m_iNeighborhoodSize -
                            1;
                    const SizeType targetCheckNode =
                        targetIndex->m_pGraph[
                            targetLocal][
                            targetCheckPosition];
                    if (targetCheckNode < -1)
                    {
                        const std::vector<SizeType>*
                            targetLocalToGlobal =
                                nullptr;
                        if constexpr (
                            EnableCrossEdges)
                        {
                            targetLocalToGlobal =
                                p_crossContext
                                    ->m_nodes[
                                        static_cast<
                                            size_t>(
                                            targetNode)]
                                    .m_localToGlobal;
                        }
                        admitCollapsedResults(
                            targetNode,
                            targetIndex,
                            targetLocalToGlobal,
                            targetLocal,
                            routeDistance);
                    }
                    else
                    {
                        SizeType targetResult =
                            targetLocal;
                        if constexpr (
                            EnableCrossEdges)
                        {
                            const auto&
                                targetContext =
                                    p_crossContext
                                        ->m_nodes[
                                            static_cast<
                                                size_t>(
                                                targetNode)];
                            targetResult =
                                (*targetContext
                                      .m_localToGlobal)[
                                    static_cast<
                                        size_t>(
                                        targetLocal)];
                        }
                        admitFilteredResult(
                            targetKey,
                            targetIndex,
                            targetLocal,
                            targetResult,
                            routeDistance);
                    }
                }
                if (p_space.m_Results.insert(
                        routeDistance))
                {
                    p_space.m_NGQueue.insert(
                        NodeDistPair(
                            targetKey,
                            routeDistance));
                }
            }
        };
        expandEdges(node, 0, localEdgeCount, false);
        expandEdges(
            node, crossEdgeBegin, edgeCount, true);
        if (hybridCollapsed)
        {
            const COMMON::BKTNode& treeNode =
                currentIndex->m_pTrees[-2 - checkNode];
            const SizeType begin =
                -treeNode.childStart;
            for (SizeType treeIndex = begin;
                 treeIndex > 0 &&
                 treeIndex < treeNode.childEnd;
                 ++treeIndex)
            {
                const SizeType sibling =
                    currentIndex
                        ->m_pTrees[treeIndex]
                        .centerid;
                if (sibling < 0 ||
                    sibling ==
                        currentLocal ||
                    sibling >=
                        currentIndex
                            ->m_pSamples.R())
                {
                    continue;
                }
                expandEdges(
                    currentIndex->m_pGraph[sibling],
                    crossEdgeBegin, edgeCount, true);
            }
        }
        if (!(EnableCrossEdges &&
              p_crossContext->m_useHybridDistance) &&
            p_space.m_NGQueue.Top().distance >
                p_space.m_SPTQueue.Top().distance)
        {
            m_pTrees.SearchTrees(m_pSamples, m_fComputeDistance, p_query, p_space,
                                 m_iNumberOfOtherDynamicPivots + p_space.m_iNumberOfCheckedLeaves);
        }
    }
    finishSearch();
}

template <typename T>
template <bool (*notDeleted)(const COMMON::LabelSet &, SizeType),
          bool (*isDup)(COMMON::QueryResultSet<T> &, SizeType, float),
          bool (*checkFilter)(const std::shared_ptr<MetadataSet> &, SizeType, std::function<bool(const ByteArray &)>)>
int Index<T>::SearchIterative(COMMON::QueryResultSet<T> &p_query, COMMON::WorkSpace &p_space, bool p_isFirst,
                              int batch) const
{
    std::shared_lock<std::shared_timed_mutex> lock(*(m_pTrees.m_lock));
    if (p_isFirst)
    {
        m_pTrees.InitSearchTrees(m_pSamples, m_fComputeDistance, p_query, p_space);
        m_pTrees.SearchTrees(m_pSamples, m_fComputeDistance, p_query, p_space, m_iNumberOfInitialDynamicPivots);
    }
    int count = 0;
    const DimensionType checkPos = m_pGraph.m_iNeighborhoodSize - 1;
    while (!p_space.m_NGQueue.empty())
    {
        NodeDistPair gnode = p_space.m_NGQueue.pop();
        SizeType tmpNode = gnode.node;
        const SizeType *node = m_pGraph[tmpNode];
        COMMON::PrefetchGraphNeighbors(
            node, m_pGraph.m_iNeighborhoodSize,
            [this](SizeType neighbor) -> const void* {
                return neighbor >= 0 && neighbor < m_pSamples.R() ? m_pSamples[neighbor] : nullptr;
            });
        if (notDeleted(m_deletedID, tmpNode)) 
        {
            if (checkFilter(m_pMetadata, tmpNode, p_space.m_filterFunc))
            {
                p_query.AddPoint(tmpNode, gnode.distance);
                count++;
            }
            if (gnode.distance > p_space.m_Results.worst() || p_space.m_iNumberOfCheckedLeaves > p_space.m_iMaxCheck)
            {
                p_space.m_relaxedMono = true;
            }
        }
        SizeType checkNode = node[checkPos];
        if (checkNode < -1)
        {
            const COMMON::BKTNode &tnode = m_pTrees[-2 - checkNode];
            SizeType i = -tnode.childStart;
            while (i < tnode.childEnd)
            {
                tmpNode = m_pTrees[i].centerid;
                if (notDeleted(m_deletedID, tmpNode))
                {
                    if (!p_space.CheckAndSet(tmpNode))
                    {
                        p_space.m_NGQueue.insert(NodeDistPair(tmpNode, gnode.distance));
                    }
                }
                i++;
            }
        }
        for (DimensionType i = 0; i <= checkPos; i++)
        {
            SizeType nn_index = node[i];
            if (nn_index < 0)
                break;
            if (p_space.CheckAndSet(nn_index))
                continue;
            float distance2leaf =
                m_fComputeDistance(p_query.GetQuantizedTarget(), (m_pSamples)[nn_index], GetFeatureDim());
            p_space.m_iNumberOfCheckedLeaves++;
            p_space.m_NGQueue.insert(NodeDistPair(nn_index, distance2leaf));
            p_space.m_Results.insert(distance2leaf);
        }
        if (p_space.m_NGQueue.Top().distance > p_space.m_SPTQueue.Top().distance)
        {
            m_pTrees.SearchTrees(m_pSamples, m_fComputeDistance, p_query, p_space,
                                 m_iNumberOfOtherDynamicPivots + p_space.m_iNumberOfCheckedLeaves);
        }
        if (count >= batch)
        {
            break;
        }
    }
    p_query.SortResult();
    return count;
}

namespace StaticDispatch
{
template <typename... Args> bool AlwaysTrue(Args...)
{
    return true;
}

bool CheckIfNotDeleted(const COMMON::LabelSet &deletedIDs, SizeType node)
{
    return !deletedIDs.Contains(node);
}

template <typename T> bool CheckDup(COMMON::QueryResultSet<T> &query, SizeType node, float score)
{
    return !query.AddPoint(node, score);
}

template <typename T> bool NeverDup(COMMON::QueryResultSet<T> &query, SizeType node, float score)
{
    query.AddPoint(node, score);
    return true;
}

bool CheckFilter(const std::shared_ptr<MetadataSet> &metadata, SizeType node,
                 std::function<bool(const ByteArray &)> filterFunc)
{
    return filterFunc(metadata->GetMetadata(node));
}

}; // namespace StaticDispatch

template <typename T>
void Index<T>::SearchIndex(COMMON::QueryResultSet<T> &p_query, COMMON::WorkSpace &p_space, bool p_searchDeleted,
                           bool p_searchDuplicated, std::function<bool(const ByteArray &)> filterFunc,
                           std::function<bool(SizeType)> p_resultFilter) const
{
    if (m_pQuantizer && !p_query.HasQuantizedTarget())
    {
        p_query.SetTarget(p_query.GetTarget(), m_pQuantizer);
    }

    // bitflags for which dispatch to take
    uint8_t flags = 0;
    flags += (m_deletedID.Count() == 0 || p_searchDeleted) << 2;
    flags += p_searchDuplicated << 1;
    flags += (filterFunc == nullptr);

    switch (flags)
    {
    case 0b000:
        Search<false, StaticDispatch::CheckIfNotDeleted, StaticDispatch::NeverDup, StaticDispatch::CheckFilter>(
            p_query, p_space, filterFunc, nullptr, nullptr, p_resultFilter);
        break;
    case 0b001:
        Search<false, StaticDispatch::CheckIfNotDeleted, StaticDispatch::NeverDup, StaticDispatch::AlwaysTrue>(
            p_query, p_space, filterFunc, nullptr, nullptr, p_resultFilter);
        break;
    case 0b010:
        Search<false, StaticDispatch::CheckIfNotDeleted, StaticDispatch::CheckDup, StaticDispatch::CheckFilter>(
            p_query, p_space, filterFunc, nullptr, nullptr, p_resultFilter);
        break;
    case 0b011:
        Search<false, StaticDispatch::CheckIfNotDeleted, StaticDispatch::CheckDup, StaticDispatch::AlwaysTrue>(
            p_query, p_space, filterFunc, nullptr, nullptr, p_resultFilter);
        break;
    case 0b100:
        Search<false, StaticDispatch::AlwaysTrue, StaticDispatch::NeverDup, StaticDispatch::CheckFilter>(
            p_query, p_space, filterFunc, nullptr, nullptr, p_resultFilter);
        break;
    case 0b101:
        Search<false, StaticDispatch::AlwaysTrue, StaticDispatch::NeverDup, StaticDispatch::AlwaysTrue>(
            p_query, p_space, filterFunc, nullptr, nullptr, p_resultFilter);
        break;
    case 0b110:
        Search<false, StaticDispatch::AlwaysTrue, StaticDispatch::CheckDup, StaticDispatch::CheckFilter>(
            p_query, p_space, filterFunc, nullptr, nullptr, p_resultFilter);
        break;
    case 0b111:
        Search<false, StaticDispatch::AlwaysTrue, StaticDispatch::CheckDup, StaticDispatch::AlwaysTrue>(
            p_query, p_space, filterFunc, nullptr, nullptr, p_resultFilter);
        break;
    default:
        std::ostringstream oss;
        oss << "Invalid flags in BKT SearchIndex dispatch: " << flags;
        throw std::logic_error(oss.str());
    }

    p_query.SetScanned(p_space.m_iNumberOfCheckedLeaves);
}

template <typename T>
int Index<T>::SearchIndexIterative(COMMON::QueryResultSet<T> &p_query, COMMON::WorkSpace &p_space, bool p_isFirst,
                                   int batch, bool p_searchDeleted, bool p_searchDuplicated) const
{
    int count = 0;
    // bitflags for which dispatch to take
    uint8_t flags = 0;
    flags += (m_deletedID.Count() == 0 || p_searchDeleted) << 2;
    flags += p_searchDuplicated << 1;
    flags += (p_space.m_filterFunc == nullptr);

    switch (flags)
    {
    case 0b000:
        count = SearchIterative<StaticDispatch::CheckIfNotDeleted, StaticDispatch::NeverDup, StaticDispatch::CheckFilter>(
            p_query, p_space, p_isFirst, batch);
        break;
    case 0b001:
        count = SearchIterative<StaticDispatch::CheckIfNotDeleted, StaticDispatch::NeverDup, StaticDispatch::AlwaysTrue>(
            p_query, p_space, p_isFirst, batch);
        break;
    case 0b010:
        count = SearchIterative<StaticDispatch::CheckIfNotDeleted, StaticDispatch::CheckDup, StaticDispatch::CheckFilter>(
            p_query, p_space, p_isFirst, batch);
        break;
    case 0b011:
        count = SearchIterative<StaticDispatch::CheckIfNotDeleted, StaticDispatch::CheckDup, StaticDispatch::AlwaysTrue>(
            p_query, p_space, p_isFirst, batch);
        break;
    case 0b100:
        count = SearchIterative<StaticDispatch::AlwaysTrue, StaticDispatch::NeverDup, StaticDispatch::CheckFilter>(
            p_query, p_space, p_isFirst, batch);
        break;
    case 0b101:
        count = SearchIterative<StaticDispatch::AlwaysTrue, StaticDispatch::NeverDup, StaticDispatch::AlwaysTrue>(
            p_query, p_space, p_isFirst, batch);
        break;
    case 0b110:
        count = SearchIterative<StaticDispatch::AlwaysTrue, StaticDispatch::CheckDup, StaticDispatch::CheckFilter>(
            p_query, p_space, p_isFirst, batch);
        break;
    case 0b111:
        count = SearchIterative<StaticDispatch::AlwaysTrue, StaticDispatch::CheckDup, StaticDispatch::AlwaysTrue>(
            p_query, p_space, p_isFirst, batch);
        break;
    default:
        std::ostringstream oss;
        oss << "Invalid flags in BKT SearchIndex dispatch: " << flags;
        throw std::logic_error(oss.str());
    }

    p_query.SetScanned(p_space.m_iNumberOfCheckedLeaves);
    return count;
}

template <typename T>
bool Index<T>::SearchIndexIterativeFromNeareast(QueryResult &p_query, COMMON::WorkSpace *p_space, bool p_isFirst,
                                                bool p_searchDeleted) const
{
    if (p_isFirst)
    {
        p_space->ResetResult(m_iMaxCheck, p_query.GetResultNum());
        SearchIndex(*((COMMON::QueryResultSet<T> *)&p_query), *p_space, p_searchDeleted, true, p_space->m_filterFunc);
        // make sure other node can be traversed after topk found
        p_space->nodeCheckStatus.clear();
        for (int i = 0; i < p_query.GetResultNum(); ++i)
        {
            SizeType result = p_query.GetResult(i)->VID;
            if (result < 0)
                continue;
            p_space->nodeCheckStatus.CheckAndSet(result);
        }
        
        for (int i = 0; i < p_query.GetResultNum(); ++i)
        {
            SizeType result = p_query.GetResult(i)->VID;
            if (result < 0)
                continue;
            const DimensionType checkPos = m_pGraph.m_iNeighborhoodSize - 1;
            const SizeType *node = m_pGraph[result];
            _mm_prefetch((const char *)node, _MM_HINT_T0);
            for (DimensionType i = 0; i <= checkPos; i++)
            {
                auto futureNode = node[i];
                if (futureNode < 0)
                    break;
                _mm_prefetch((const char *)(m_pSamples)[futureNode], _MM_HINT_T0);
            }
            for (DimensionType i = 0; i <= checkPos; i++)
            {
                SizeType nn_index = node[i];
                if (nn_index < 0)
                    break;
                if (p_space->CheckAndSet(nn_index))
                    continue;
                float distance2leaf = m_fComputeDistance((const T *)p_query.GetQuantizedTarget(),
                                                         (m_pSamples)[nn_index], GetFeatureDim());
                p_space->m_NGQueue.insert(NodeDistPair(nn_index, distance2leaf));
            }
        }
    }
    else
    {
        //p_space->ResetResult(m_iMaxCheck, p_query.GetResultNum());
        SearchIndexIterative(*((COMMON::QueryResultSet<T> *)&p_query), *p_space, p_isFirst, p_query.GetResultNum(),
                             p_searchDeleted, true);
    }
    if (p_query.GetResult(0) == nullptr || p_query.GetResult(0)->VID < 0)
    {
        return false;
    }
    if (p_query.WithMeta() && nullptr != m_pMetadata)
    {
        for (int i = 0; i < p_query.GetResultNum(); ++i)
        {
            SizeType result = p_query.GetResult(i)->VID;
            p_query.SetMetadata(i, (result < 0) ? ByteArray::c_empty : m_pMetadata->GetMetadataCopy(result));
        }
    }
    return true;
}

template <typename T> ErrorCode Index<T>::SearchIndex(QueryResult &p_query, bool p_searchDeleted) const
{
    if (!m_bReady)
        return ErrorCode::EmptyIndex;

    auto workSpace = RentWorkSpace(p_query.GetResultNum(), nullptr, m_iMaxCheck);

    SearchIndex(*((COMMON::QueryResultSet<T> *)&p_query), *workSpace, p_searchDeleted, true);

    m_workSpaceFactory->ReturnWorkSpace(std::move(workSpace));

    if (p_query.WithMeta() && nullptr != m_pMetadata)
    {
        for (int i = 0; i < p_query.GetResultNum(); ++i)
        {
            SizeType result = p_query.GetResult(i)->VID;
            p_query.SetMetadata(i, (result < 0) ? ByteArray::c_empty : m_pMetadata->GetMetadataCopy(result));
        }
    }
    return ErrorCode::Success;
}

template <typename T>
ErrorCode Index<T>::SearchIndexWithFilter(QueryResult &p_query, std::function<bool(const ByteArray &)> filterFunc,
                                          int maxCheck, bool p_searchDeleted) const
{
    if (!m_bReady)
        return ErrorCode::EmptyIndex;

    auto workSpace = RentWorkSpace(p_query.GetResultNum(), filterFunc, maxCheck);

    SearchIndex(*((COMMON::QueryResultSet<T> *)&p_query), *workSpace, p_searchDeleted, true, filterFunc);

    m_workSpaceFactory->ReturnWorkSpace(std::move(workSpace));

    if (p_query.WithMeta() && nullptr != m_pMetadata)
    {
        for (int i = 0; i < p_query.GetResultNum(); ++i)
        {
            SizeType result = p_query.GetResult(i)->VID;
            p_query.SetMetadata(i, (result < 0) ? ByteArray::c_empty : m_pMetadata->GetMetadataCopy(result));
        }
    }
    return ErrorCode::Success;
}

template <typename T>
ErrorCode Index<T>::SearchIndexWithResultFilter(
    QueryResult& p_query,
    std::function<bool(SizeType)> p_resultFilter,
    int p_maxCheck,
    bool p_searchDeleted) const
{
    if (!m_bReady)
        return ErrorCode::EmptyIndex;
    if (!p_resultFilter)
        return SearchIndex(p_query, p_searchDeleted);

    auto workSpace = RentWorkSpace(
        p_query.GetResultNum(), nullptr,
        p_maxCheck > 0 ? p_maxCheck : m_iMaxCheck);
    workSpace->PrepareResultCheckStatus();
    SearchIndex(
        *((COMMON::QueryResultSet<T>*)&p_query), *workSpace,
        p_searchDeleted, true, nullptr, p_resultFilter);
    m_workSpaceFactory->ReturnWorkSpace(std::move(workSpace));

    if (p_query.WithMeta() && nullptr != m_pMetadata)
    {
        for (int i = 0; i < p_query.GetResultNum(); ++i)
        {
            SizeType result = p_query.GetResult(i)->VID;
            p_query.SetMetadata(
                i, (result < 0)
                       ? ByteArray::c_empty
                       : m_pMetadata->GetMetadataCopy(result));
        }
    }
    return ErrorCode::Success;
}

template <typename T>
std::shared_ptr<ResultIterator> Index<T>::GetIterator(const void *p_target, bool p_searchDeleted, std::function<bool(const ByteArray&)> p_filterFunc, int p_maxCheck) const
{
    if (!m_bReady)
        return nullptr;

    std::shared_ptr<ResultIterator> resultIterator =
        std::make_shared<ResultIterator>((const void *)this, p_target, p_searchDeleted, 1, p_filterFunc, p_maxCheck);
    return resultIterator;
}

template <typename T>
ErrorCode Index<T>::SearchIndexIterativeNext(QueryResult &p_query, COMMON::WorkSpace *workSpace, int p_batch,
                                             int &resultCount, bool p_isFirst, bool p_searchDeleted) const
{
    if (!m_bReady)
        return ErrorCode::EmptyIndex;
    if (p_isFirst) workSpace->ResetResult(m_iMaxCheck, p_batch);
    resultCount = SearchIndexIterative(*((COMMON::QueryResultSet<T> *)&p_query), *workSpace, p_isFirst, p_batch,
                                       p_searchDeleted, true);

    if (p_query.WithMeta() && nullptr != m_pMetadata)
    {
        for (int i = 0; i < resultCount; ++i)
        {
            SizeType result = p_query.GetResult(i)->VID;
            p_query.SetMetadata(i, (result < 0) ? ByteArray::c_empty : m_pMetadata->GetMetadataCopy(result));
        }
    }
    return ErrorCode::Success;
}

template <typename T> ErrorCode Index<T>::SearchIndexIterativeEnd(std::unique_ptr<COMMON::WorkSpace> space) const
{
    if (!m_bReady)
        return ErrorCode::EmptyIndex;
    if (nullptr != space)
        m_workSpaceFactory->ReturnWorkSpace(std::move(space));
    return ErrorCode::Success;
}

template <typename T> std::unique_ptr<COMMON::WorkSpace> Index<T>::RentWorkSpace(int batch, std::function<bool(const ByteArray&)> p_filterFunc, int p_maxCheck) const
{
    const int maxCheck = p_maxCheck > 0 ? p_maxCheck : m_iMaxCheck;
    auto workSpace = m_workSpaceFactory->GetWorkSpace();
    if (!workSpace)
    {
        workSpace.reset(new COMMON::WorkSpace());
        workSpace->Initialize(std::max(16, maxCheck), m_iHashTableExp);
    }
    workSpace->Reset(maxCheck, batch);
    workSpace->m_filterFunc = p_filterFunc;
    return std::move(workSpace);
}

template <typename T>
ErrorCode Index<T>::SearchIndexWithCrossEdges(
    QueryResult& p_query,
    const CrossGraphSearchContext& p_context,
    int p_maxCheck,
    CrossGraphSearchStats* p_stats) const
{
    if (!m_bReady)
        return ErrorCode::EmptyIndex;
    if (p_context.m_entryNode < 0 ||
        p_context.m_entryNode >= static_cast<int>(p_context.m_nodes.size()) ||
        p_context.m_locatorLocalBits <= 0 ||
        p_context.m_locatorLocalMask < 0)
    {
        return ErrorCode::Fail;
    }

    if (p_context.m_useHybridDistance &&
        !p_context.m_queryDistance)
    {
        return ErrorCode::Fail;
    }

    const auto& entryContext =
        p_context.m_nodes[
            static_cast<size_t>(
                p_context.m_entryNode)];
    if (entryContext.m_index != this ||
        entryContext.m_localToGlobal == nullptr ||
        entryContext.m_localToGlobal->size() !=
            static_cast<size_t>(GetNumSamples()))
    {
        return ErrorCode::Fail;
    }
    for (const auto& nodeContext : p_context.m_nodes)
    {
        if (nodeContext.m_index == nullptr)
        {
            if (nodeContext.m_localToGlobal != nullptr)
                return ErrorCode::Fail;
            continue;
        }
        if (!nodeContext.m_index->m_bReady ||
            nodeContext.m_localToGlobal == nullptr ||
            nodeContext.m_localToGlobal->size() !=
                static_cast<size_t>(
                    nodeContext.m_index->GetNumSamples()) ||
            nodeContext.m_index->GetFeatureDim() !=
                GetFeatureDim() ||
            (p_context.m_useHybridDistance &&
             nodeContext.m_index->m_pGraph
                     .GetRuntimeEdgeSuffixSize() <=
                 0))
        {
            return ErrorCode::Fail;
        }
    }

    if (p_stats != nullptr)
        *p_stats = CrossGraphSearchStats();

    auto workSpace =
        RentWorkSpace(p_query.GetResultNum(), nullptr, p_maxCheck);
    auto& results = *((COMMON::QueryResultSet<T>*)&p_query);
    if (m_pQuantizer && !results.HasQuantizedTarget())
    {
        results.SetTarget(results.GetTarget(), m_pQuantizer);
    }
    Search<true,
           StaticDispatch::AlwaysTrue,
           StaticDispatch::CheckDup,
           StaticDispatch::AlwaysTrue>(
        results, *workSpace, nullptr,
        &p_context, p_stats);
    results.SetScanned(workSpace->m_iNumberOfCheckedLeaves);
    m_workSpaceFactory->ReturnWorkSpace(std::move(workSpace));
    return ErrorCode::Success;
}

template <typename T> ErrorCode Index<T>::RefineSearchIndex(QueryResult &p_query, bool p_searchDeleted) const
{
    auto workSpace = RentWorkSpace(p_query.GetResultNum(), nullptr, m_pGraph.m_iMaxCheckForRefineGraph);
    SearchIndex(*((COMMON::QueryResultSet<T> *)&p_query), *workSpace, p_searchDeleted, false);
    m_workSpaceFactory->ReturnWorkSpace(std::move(workSpace));
    return ErrorCode::Success;
}

template <typename T> ErrorCode Index<T>::SearchTree(QueryResult &p_query) const
{
    auto workSpace = RentWorkSpace(p_query.GetResultNum(), nullptr, m_pGraph.m_iMaxCheckForRefineGraph);

    COMMON::QueryResultSet<T> *p_results = (COMMON::QueryResultSet<T> *)&p_query;
    m_pTrees.InitSearchTrees(m_pSamples, m_fComputeDistance, *p_results, *workSpace);
    m_pTrees.SearchTrees(
        m_pSamples,
        m_fComputeDistance,
        *p_results,
        *workSpace,
        m_iNumberOfInitialDynamicPivots);
    BasicResult *res = p_query.GetResults();
    for (int i = 0; i < p_query.GetResultNum(); i++)
    {
        auto &cell = workSpace->m_NGQueue.pop();
        res[i].VID = cell.node;
        res[i].Dist = cell.distance;
    }
    m_workSpaceFactory->ReturnWorkSpace(std::move(workSpace));

    return ErrorCode::Success;
}
#pragma endregion

template <typename T>
ErrorCode Index<T>::BuildIndex(const void *p_data, SizeType p_vectorNum, DimensionType p_dimension, bool p_normalized,
                               bool p_shareOwnership)
{
    if (p_data == nullptr || p_vectorNum == 0 || p_dimension == 0)
        return ErrorCode::EmptyData;

    m_pGraph.m_iThreadNum = m_iNumberOfThreads;

    m_pSamples.Initialize(p_vectorNum, p_dimension, m_iDataBlockSize, m_iDataCapacity, (T *)p_data, p_shareOwnership);
    m_deletedID.Initialize(p_vectorNum, m_iDataBlockSize, m_iDataCapacity,
                           COMMON::LabelSet::InvalidIDBehavior::AlwaysContains);

    if (DistCalcMethod::Cosine == m_iDistCalcMethod && !p_normalized)
    {
        int base = m_pQuantizer ? m_pQuantizer->GetBase() : COMMON::Utils::GetBase<T>();
        COMMON::Utils::BatchNormalize(m_pSamples[0], p_vectorNum, p_dimension, base,
                                      m_iNumberOfThreads);
    }

    m_threadPool.init();

    auto t1 = std::chrono::high_resolution_clock::now();
    m_pTrees.BuildTrees<T>(m_pSamples, m_iDistCalcMethod, m_iNumberOfThreads);
    auto t2 = std::chrono::high_resolution_clock::now();
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Build Tree time (s): %lld\n",
                 std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count());

    m_pGraph.BuildGraph<T>(this, &(m_pTrees.GetSampleMap()));

    auto t3 = std::chrono::high_resolution_clock::now();
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Build Graph time (s): %lld\n",
                 std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count());

    m_bReady = true;
    return ErrorCode::Success;
}

template <typename T> ErrorCode Index<T>::RefineIndex(std::shared_ptr<VectorIndex> &p_newIndex)
{
    p_newIndex.reset(new Index<T>());
    Index<T> *ptr = (Index<T> *)p_newIndex.get();

#define DefineBKTParameter(VarName, VarType, DefaultValue, RepresentStr) ptr->VarName = VarName;

#include "inc/Core/BKT/ParameterDefinitionList.h"
#undef DefineBKTParameter

    std::lock_guard<std::mutex> lock(m_dataAddLock);
    std::unique_lock<std::shared_timed_mutex> uniquelock(m_dataDeleteLock);

    SizeType newR = GetNumSamples();

    std::vector<SizeType> indices;
    std::vector<SizeType> reverseIndices(newR);
    for (SizeType i = 0; i < newR; i++)
    {
        if (!m_deletedID.Contains(i))
        {
            indices.push_back(i);
            reverseIndices[i] = i;
        }
        else
        {
            while (m_deletedID.Contains(newR - 1) && newR > i)
                newR--;
            if (newR == i)
                break;
            indices.push_back(newR - 1);
            reverseIndices[newR - 1] = i;
            newR--;
        }
    }

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Refine... from %d -> %d\n", GetNumSamples(), newR);
    if (newR == 0)
        return ErrorCode::EmptyIndex;

    ptr->m_threadPool.init();

    ErrorCode ret = ErrorCode::Success;
    if ((ret = m_pSamples.Refine(indices, ptr->m_pSamples)) != ErrorCode::Success)
        return ret;
    if (nullptr != m_pMetadata &&
        (ret = m_pMetadata->RefineMetadata(indices, ptr->m_pMetadata, m_iDataBlockSize, m_iDataCapacity,
                                           m_iMetaRecordSize)) != ErrorCode::Success)
        return ret;

    ptr->m_deletedID.Initialize(newR, m_iDataBlockSize, m_iDataCapacity,
                                COMMON::LabelSet::InvalidIDBehavior::AlwaysContains);
    COMMON::BKTree *newtree = &(ptr->m_pTrees);
    (*newtree).BuildTrees<T>(ptr->m_pSamples, ptr->m_iDistCalcMethod, m_iNumberOfThreads);
    m_pGraph.RefineGraph<T>(this, indices, reverseIndices, nullptr, &(ptr->m_pGraph), &(ptr->m_pTrees.GetSampleMap()));
    if (HasMetaMapping())
        ptr->BuildMetaMapping(false);
    ptr->m_bReady = true;
    return ret;
}

template <typename T>
ErrorCode Index<T>::RefineIndex(const std::vector<std::shared_ptr<Helper::DiskIO>> &p_indexStreams,
                                IAbortOperation *p_abort, std::vector<SizeType> *p_mapping)
{
    std::lock_guard<std::mutex> lock(m_dataAddLock);
    std::unique_lock<std::shared_timed_mutex> uniquelock(m_dataDeleteLock);

    SizeType newR = GetNumSamples();

    std::vector<SizeType> indices;
    std::vector<SizeType> reverseIndices;
    if (p_mapping == nullptr)
    {
        p_mapping = &reverseIndices;
    }
    p_mapping->resize(newR);
    for (SizeType i = 0; i < newR; i++)
    {
        if (!m_deletedID.Contains(i))
        {
            indices.push_back(i);
            (*p_mapping)[i] = i;
        }
        else
        {
            while (m_deletedID.Contains(newR - 1) && newR > i)
                newR--;
            if (newR == i)
                break;
            indices.push_back(newR - 1);
            (*p_mapping)[newR - 1] = i;
            newR--;
        }
    }

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Refine... from %d -> %d\n", GetNumSamples(), newR);
    if (newR == 0)
        return ErrorCode::EmptyIndex;

    ErrorCode ret = ErrorCode::Success;
    if ((ret = m_pSamples.Refine(indices, p_indexStreams[0])) != ErrorCode::Success)
        return ret;

    if (p_abort != nullptr && p_abort->ShouldAbort())
        return ErrorCode::ExternalAbort;

    COMMON::BKTree newTrees(m_pTrees);
    newTrees.BuildTrees<T>(m_pSamples, m_iDistCalcMethod, m_iNumberOfThreads, &indices, p_mapping);
    if ((ret = newTrees.SaveTrees(p_indexStreams[1])) != ErrorCode::Success)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to Save Refine Tree!!\n");
        return ret;
    }    

    if (p_abort != nullptr && p_abort->ShouldAbort())
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Abort!!\n");
        return ErrorCode::ExternalAbort;
    }
        

    if ((ret = m_pGraph.RefineGraph<T>(this, indices, (*p_mapping), p_indexStreams[2], nullptr,
                                       &(newTrees.GetSampleMap()))) != ErrorCode::Success)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to Save Refine Graph!!\n");
        return ret;
    }
        

    COMMON::LabelSet newDeletedID;
    newDeletedID.Initialize(newR, m_iDataBlockSize, m_iDataCapacity,
                            COMMON::LabelSet::InvalidIDBehavior::AlwaysContains);
    if ((ret = newDeletedID.Save(p_indexStreams[3])) != ErrorCode::Success)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to Save Refine DeletedID!!\n");
        return ret;
    }
        
    if (nullptr != m_pMetadata)
    {
        if (p_indexStreams.size() < 6)
            return ErrorCode::LackOfInputs;
        if ((ret = m_pMetadata->RefineMetadata(indices, p_indexStreams[4], p_indexStreams[5])) != ErrorCode::Success)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Fail to Save Refine Metadata!!\n");
            return ret;
        }          
    }
    return ret;
}

template <typename T> ErrorCode Index<T>::DeleteIndex(const void *p_vectors, SizeType p_vectorNum)
{
    const T *ptr_v = (const T *)p_vectors;
    std::vector<std::thread> mythreads;
    mythreads.reserve(m_iNumberOfThreads);
    std::atomic_size_t sent(0);
    for (int tid = 0; tid < m_iNumberOfThreads; tid++)
    {
        mythreads.emplace_back([&, tid]() {
            size_t i = 0;
            while (true)
            {
                i = sent.fetch_add(1);
                if (i < p_vectorNum)
                {
                    COMMON::QueryResultSet<T> query(ptr_v + i * GetFeatureDim(), m_pGraph.m_iCEF);
                    SearchIndex(query);

                    for (int j = 0; j < m_pGraph.m_iCEF; j++)
                    {
                        if (query.GetResult(j)->Dist < 1e-6)
                        {
                            DeleteIndex(query.GetResult(j)->VID);
                        }
                    }
                }
                else
                {
                    return;
                }
            }
        });
    }
    for (auto &t : mythreads)
    {
        t.join();
    }
    mythreads.clear();
    return ErrorCode::Success;
}

template <typename T> ErrorCode Index<T>::DeleteIndex(const SizeType &p_id)
{
    if (!m_bReady)
        return ErrorCode::EmptyIndex;

    std::shared_lock<std::shared_timed_mutex> sharedlock(m_dataDeleteLock);
    if (m_deletedID.Insert(p_id))
        return ErrorCode::Success;
    return ErrorCode::VectorNotFound;
}

template <typename T>
ErrorCode Index<T>::AddIndex(const void *p_data, SizeType p_vectorNum, DimensionType p_dimension,
                             std::shared_ptr<MetadataSet> p_metadataSet, bool p_withMetaIndex, bool p_normalized)
{
    if (p_data == nullptr || p_vectorNum == 0 || p_dimension == 0)
        return ErrorCode::EmptyData;

    SizeType begin, end;
    ErrorCode ret;
    {
        std::lock_guard<std::mutex> lock(m_dataAddLock);

        begin = GetNumSamples();
        end = begin + p_vectorNum;

        if (begin == 0)
        {
            if (p_metadataSet != nullptr)
            {
                m_pMetadata.reset(new MemMetadataSet(m_iDataBlockSize, m_iDataCapacity, m_iMetaRecordSize));
                m_pMetadata->AddBatch(*p_metadataSet);
                if (p_withMetaIndex)
                    BuildMetaMapping(false);
            }
            if ((ret = BuildIndex(p_data, p_vectorNum, p_dimension, p_normalized)) != ErrorCode::Success)
                return ret;
            return ErrorCode::Success;
        }

        if (p_dimension != GetFeatureDim())
            return ErrorCode::DimensionSizeMismatch;

        if (m_pSamples.AddBatch(p_vectorNum, (const T *)p_data) != ErrorCode::Success ||
            m_pGraph.AddBatch(p_vectorNum) != ErrorCode::Success ||
            m_deletedID.AddBatch(p_vectorNum) != ErrorCode::Success)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Memory Error: Cannot alloc space for vectors!\n");
            m_pSamples.SetR(begin);
            m_pGraph.SetR(begin);
            m_deletedID.SetR(begin);
            return ErrorCode::MemoryOverFlow;
        }

        if (m_pMetadata != nullptr)
        {
            if (p_metadataSet != nullptr)
            {
                m_pMetadata->AddBatch(*p_metadataSet);
                if (HasMetaMapping())
                {
                    for (SizeType i = begin; i < end; i++)
                    {
                        ByteArray meta = m_pMetadata->GetMetadata(i);
                        std::string metastr((char *)meta.Data(), meta.Length());
                        UpdateMetaMapping(metastr, i);
                    }
                }
            }
            else
            {
                for (SizeType i = begin; i < end; i++)
                    m_pMetadata->Add(ByteArray::c_empty);
            }
        }
    }

    if (DistCalcMethod::Cosine == m_iDistCalcMethod && !p_normalized)
    {
        int base = COMMON::Utils::GetBase<T>();
        for (SizeType i = begin; i < end; i++)
        {
            COMMON::Utils::Normalize((T *)m_pSamples[i], GetFeatureDim(), base);
        }
    }

    if (end - m_pTrees.sizePerTree() >= m_addCountForRebuild && m_threadPool.jobsize() == 0)
    {
        m_threadPool.add(new RebuildJob(&m_pSamples, &m_pTrees, &m_pGraph, m_iDistCalcMethod));
    }

    for (SizeType node = begin; node < end; node++)
    {
        m_pGraph.RefineNode<T>(this, node, true, true, m_pGraph.m_iAddCEF);
    }
    return ErrorCode::Success;
}

template <typename T>
ErrorCode Index<T>::AddIndexId(const void *p_data, SizeType p_vectorNum, DimensionType p_dimension, int &beginHead,
                               int &endHead)
{
    if (p_data == nullptr || p_vectorNum == 0 || p_dimension == 0)
        return ErrorCode::EmptyData;

    SizeType begin, end;
    {
        std::lock_guard<std::mutex> lock(m_dataAddLock);

        begin = GetNumSamples();
        end = begin + p_vectorNum;

        if (begin == 0)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Index Error: No vector in Index!\n");
            return ErrorCode::EmptyIndex;
        }

        if (p_dimension != GetFeatureDim())
            return ErrorCode::DimensionSizeMismatch;

        if (m_pSamples.AddBatch(p_vectorNum, (const T *)p_data) != ErrorCode::Success ||
            m_pGraph.AddBatch(p_vectorNum) != ErrorCode::Success ||
            m_deletedID.AddBatch(p_vectorNum) != ErrorCode::Success)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Memory Error: Cannot alloc space for vectors!\n");
            m_pSamples.SetR(begin);
            m_pGraph.SetR(begin);
            m_deletedID.SetR(begin);
            return ErrorCode::MemoryOverFlow;
        }
    }
    beginHead = begin;
    endHead = end;
    return ErrorCode::Success;
}

template <typename T> ErrorCode Index<T>::AddIndexIdx(SizeType begin, SizeType end)
{
    if (end - m_pTrees.sizePerTree() >= m_addCountForRebuild && m_threadPool.jobsize() == 0) {
        m_threadPool.add(new RebuildJob(&m_pSamples, &m_pTrees, &m_pGraph, m_iDistCalcMethod));
    }

    for (SizeType node = begin; node < end; node++)
    {
        m_pGraph.RefineNode<T>(this, node, true, true, m_pGraph.m_iAddCEF);
    }
    return ErrorCode::Success;
}

template <typename T> ErrorCode Index<T>::AddIndexIdxNoBackEdge(SizeType begin, SizeType end)
{
    for (SizeType node = begin; node < end; node++)
    {
        m_pGraph.RefineNode<T>(this, node, false, true, m_pGraph.m_iAddCEF);
    }
    return ErrorCode::Success;
}

template <typename T> ErrorCode Index<T>::UpdateIndex()
{
    m_pGraph.m_iThreadNum = m_iNumberOfThreads;
    return ErrorCode::Success;
}

template <typename T> ErrorCode Index<T>::Check()
{
    std::vector<std::thread> mythreads;
    mythreads.reserve(m_iNumberOfThreads);
    std::atomic_size_t sent(0);
    ErrorCode ret = ErrorCode::Success;
    for (int tid = 0; tid < m_iNumberOfThreads; tid++)
    {
        mythreads.emplace_back([&, tid]() {
            size_t i = 0;
            while (true)
            {
                i = sent.fetch_add(1);
                if (i < m_pSamples.R())
                {
                    if (!m_deletedID.Contains(i))
                    {
                        COMMON::QueryResultSet<T> result(m_pSamples[i], 1);
                        if (SearchIndex(result) != ErrorCode::Success || result.GetResult(0)->VID != i)
                        {
                            ret = ErrorCode::Fail;
                            return;
                        }
                    }
                }
                else
                {
                    return;
                }
            }
        });
    }
    for (auto &t : mythreads)
    {
        t.join();
    }
    mythreads.clear();
    return ret;
}

template <typename T> ErrorCode Index<T>::SetParameter(const char *p_param, const char *p_value, const char *p_section)
{
    if (nullptr == p_param || nullptr == p_value)
        return ErrorCode::Fail;

#define DefineBKTParameter(VarName, VarType, DefaultValue, RepresentStr)                                               \
    else if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, RepresentStr))                                       \
    {                                                                                                                  \
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Setting %s with value %s\n", RepresentStr, p_value);                  \
        VarType tmp;                                                                                                   \
        if (SPTAG::Helper::Convert::ConvertStringTo<VarType>(p_value, tmp))                                            \
        {                                                                                                              \
            VarName = tmp;                                                                                             \
        }                                                                                                              \
    }

#include "inc/Core/BKT/ParameterDefinitionList.h"
#undef DefineBKTParameter

    if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "DistCalcMethod"))
    {
        m_fComputeDistance = m_pQuantizer ? m_pQuantizer->DistanceCalcSelector<T>(m_iDistCalcMethod)
                                          : COMMON::DistanceCalcSelector<T>(m_iDistCalcMethod);
        auto base = m_pQuantizer ? m_pQuantizer->GetBase() : COMMON::Utils::GetBase<T>();
        m_iBaseSquare = (m_iDistCalcMethod == DistCalcMethod::Cosine) ? base * base : 1;
    }
    return ErrorCode::Success;
}

template <typename T> std::string Index<T>::GetParameter(const char *p_param, const char *p_section) const
{
    if (nullptr == p_param)
        return std::string();

#define DefineBKTParameter(VarName, VarType, DefaultValue, RepresentStr)                                               \
    else if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, RepresentStr))                                       \
    {                                                                                                                  \
        return SPTAG::Helper::Convert::ConvertToString(VarName);                                                       \
    }

#include "inc/Core/BKT/ParameterDefinitionList.h"
#undef DefineBKTParameter

    return std::string();
}
} // namespace BKT
} // namespace SPTAG

#define DefineVectorValueType(Name, Type) template class SPTAG::BKT::Index<Type>;

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType
