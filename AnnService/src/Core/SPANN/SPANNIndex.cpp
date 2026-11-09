// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/SPANN/Index.h"
#include "inc/Helper/VectorSetReaders/MemoryReader.h"
#include "inc/Core/SPANN/ExtraDynamicSearcher.h"
#include "inc/Core/SPANN/ExtraStaticSearcher.h"
#include <chrono>
#include <random>
#include <shared_mutex>
#include <filesystem>
#include <numeric>
#include <sstream>
#ifndef _MSC_VER
#include <fcntl.h>
#include <unistd.h>
#endif

#include "inc/Core/ResultIterator.h"
#include "inc/Core/SPANN/SPANNResultIterator.h"

#ifdef SPDK
#include "ExtraSPDKController.h"
#endif

#ifdef ROCKSDB
#include "ExtraRocksDBController.h"
// enable rocksdb io_uring
extern "C" bool RocksDbIOUringEnable() { return true; }
#endif

#ifdef TIKV
#include "inc/Core/SPANN/ExtraTiKVController.h"
#endif

namespace SPTAG
{
template <typename T> thread_local std::unique_ptr<T> COMMON::ThreadLocalWorkSpaceFactory<T>::m_workspace;
namespace SPANN
{
EdgeCompare Selection::g_edgeComparer;

namespace
{
    ErrorCode SyncLayerPath(const std::string& path)
    {
#ifndef _MSC_VER
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot open dynamic layer artifact for fsync: %s.\n", path.c_str());
            return ErrorCode::FailedOpenFile;
        }
        int result = fsync(fd);
        int closeResult = close(fd);
        if (result != 0 || closeResult != 0) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot synchronize dynamic layer artifact: %s.\n", path.c_str());
            return ErrorCode::DiskIOFail;
        }
        return ErrorCode::Success;
#else
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Dynamic layer durable publication currently requires Linux.\n");
        return ErrorCode::Fail;
#endif
    }
}

template <typename T> Index<T>::~Index()
{
    {
        std::lock_guard<std::mutex> lock(m_resizeRequestMutex);
        m_stopResizeThread = true;
    }
    m_resizeRequestCV.notify_all();
    if (m_resizeThread.joinable()) m_resizeThread.join();
    // Workers retain raw references to this index. Drain them before destroying its head and mappings.
    while (!PostingMaintenanceFinished()) std::this_thread::sleep_for(std::chrono::milliseconds(20));
}

template <typename T> bool Index<T>::PostingMaintenanceFinished()
{
    LayerReadLock lock(m_layerMutex, m_layerLockEnabled);
    if (m_options.m_storage == Storage::STATIC) return true;
    bool finished = true;
    for (auto& searcher : m_extraSearchers) {
        if (searcher && !searcher->AllFinished()) finished = false;
    }
    return finished;
}

template <typename T> ErrorCode Index<T>::AcquireSaveGuard(std::unique_ptr<SaveOperationGuard>& guard)
{
    if (!m_layerLockEnabled || LayerReadLock::IsHeld(m_checkPointLock)) return ErrorCode::Success;
    auto operation = std::make_unique<LayerSaveGuard>(*this);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(m_options.m_layerResizeTimeoutSeconds);
    if (!operation->mutations.try_lock_until(deadline)) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Dynamic index save timed out waiting for writes.\n");
        return ErrorCode::Fail;
    }
    operation->mutationContext = std::make_unique<LayerReadLock>(m_checkPointLock, true, true);
    {
        std::unique_lock<LayerMutex> barrier(m_layerMutex, std::defer_lock);
        if (!barrier.try_lock_until(deadline)) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Dynamic index save timed out waiting for queries/iterators.\n");
            return ErrorCode::Fail;
        }
    }
    while (!PostingMaintenanceFinished()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Dynamic index save timed out draining maintenance.\n");
            return ErrorCode::Fail;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (!operation->topology.try_lock_until(deadline)) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Dynamic index save timed out acquiring the topology write lock.\n");
        return ErrorCode::Fail;
    }
    operation->topologyContext = std::make_unique<LayerReadLock>(m_layerMutex, true, true, true);
    guard = std::move(operation);
    return ErrorCode::Success;
}

template <typename T> ErrorCode Index<T>::ValidateDynamicLayers() const
{
    if (!m_options.m_enableDynamicLayers && m_options.m_layerStoragePrefixes.empty()) return ErrorCode::Success;
    if ((m_options.m_storage != Storage::TIKVIO && m_options.m_storage != Storage::FILEIO) ||
        m_pQuantizer || m_options.m_enableWAL ||
        m_options.m_recovery || m_options.m_centeringToZero ||
        (m_options.m_enableDynamicLayers && (m_options.m_headMemoryTarget == 0 ||
            m_options.m_headMemoryTarget >= m_options.m_headMemoryHighWatermark)) ||
        m_options.m_layers < 1 || m_options.m_maxLayers < m_options.m_layers ||
        m_options.m_maxLayers > SearchStats::kSearchLatencyBreakdownLayers ||
        m_options.m_layerResizeTimeoutSeconds <= 0 || m_options.m_replicaCount <= 0 ||
        (m_options.m_enableDynamicLayers && !(m_options.m_ratio > 0 && m_options.m_ratio <= 1)) ||
        (m_options.m_storage == Storage::TIKVIO && m_options.m_tikvKeyPrefix.find('|') != std::string::npos) ||
        m_options.m_internalResultNum < m_options.m_replicaCount ||
        (m_options.m_update && m_options.m_appendThreadNum <= 0)) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
            "Invalid dynamic-layer configuration: requires unquantized FILEIO/TiKV, no WAL/recovery/centering, "
            "0 < HeadMemoryTarget < HeadMemoryHighWatermark, Layers <= MaxLayers <= 16, "
            "positive resize timeout, 0 < Ratio <= 1 and valid replica/worker counts.\n");
        return ErrorCode::FailedParseValue;
    }
#ifndef TIKV
    if (m_options.m_storage == Storage::TIKVIO) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKV dynamic layers require a build with TIKV enabled.\n");
        return ErrorCode::Fail;
    }
#endif
#ifdef _MSC_VER
    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Dynamic layer publication is currently supported only on Linux.\n");
    return ErrorCode::Fail;
#endif
    if (!m_options.m_layerStoragePrefixes.empty()) {
        std::istringstream stream(m_options.m_layerStoragePrefixes);
        std::string prefix;
        int count = 0;
        while (std::getline(stream, prefix, '|')) {
            if (prefix.empty()) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Empty dynamic layer storage prefix.\n");
                return ErrorCode::FailedParseValue;
            }
            if (m_options.m_storage == Storage::FILEIO) {
                const std::filesystem::path path(prefix);
                if (path.is_absolute() || std::any_of(path.begin(), path.end(),
                    [](const std::filesystem::path& part) { return part == ".."; })) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "FILEIO layer storage must be relative to IndexDirectory: %s.\n", prefix.c_str());
                    return ErrorCode::FailedParseValue;
                }
            }
            ++count;
        }
        if (count != m_options.m_layers || m_options.m_layerStoragePrefixes.back() == '|') {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "LayerStoragePrefixes must contain one prefix per posting layer.\n");
            return ErrorCode::FailedParseValue;
        }
    }
    return ErrorCode::Success;
}

template <typename T> std::string Index<T>::LayerStoragePrefix(int layer) const
{
    if (m_options.m_layerStoragePrefixes.empty())
        return m_options.m_storage == Storage::FILEIO ? m_options.m_ssdMappingFile : m_options.m_tikvKeyPrefix;
    std::istringstream stream(m_options.m_layerStoragePrefixes);
    std::string prefix;
    for (int i = 0; i <= layer; ++i) std::getline(stream, prefix, '|');
    return prefix;
}

template <typename T> std::uint64_t Index<T>::GetHeadMemoryUsage() const
{
    LayerReadLock lock(m_layerMutex, m_layerLockEnabled);
    if (!m_topIndex) return 0;
    auto sizes = m_topIndex->BufferSize();
    std::uint64_t bytes = std::accumulate(sizes->begin(), sizes->end(), std::uint64_t{0});
    std::shared_lock<std::shared_timed_mutex> mapping(m_topLocalToGlobalIDLock);
    bytes += m_topLocalToGlobalID.BufferSize();
    bytes += m_topGlobalToLocalID.size() * (2 * sizeof(SizeType) + 2 * sizeof(void*));
    return bytes;
}

template <typename T> void Index<T>::StartLayerResizeWorker()
{
    if (!m_options.m_enableDynamicLayers || m_resizeThread.joinable()) return;
    m_layerLockEnabled = true;
    m_resizeThread = std::thread([this]() {
        std::unique_lock<std::mutex> lock(m_resizeRequestMutex);
        while (true) {
            m_resizeRequestCV.wait(lock, [this]() { return m_stopResizeThread || m_resizeRequested.load(); });
            if (m_stopResizeThread) break;
            m_resizeWorkerBusy = true;
            m_resizeRequested = false;
            lock.unlock();
            ErrorCode ret = ErrorCode::Success;
            bool shouldGrow = false;
            try {
                LayerReadLock topology(m_layerMutex, m_layerLockEnabled);
                if (m_bReady && m_options.m_enableDynamicLayers &&
                    GetHeadMemoryUsage() >= m_options.m_headMemoryHighWatermark) {
                    shouldGrow = true;
                }
            } catch (const std::exception& error) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Dynamic layer threshold check failed: %s.\n", error.what());
                ret = ErrorCode::Fail;
            }
            if (shouldGrow) ret = GrowTopLayer();
            if (ret != ErrorCode::Success) m_layerResizeStatus = ret;
            m_resizeWorkerBusy = false;
            lock.lock();
        }
    });
}

template <typename T> void Index<T>::RequestLayerResize()
{
    if (!m_layerLockEnabled || !m_options.m_enableDynamicLayers || m_resizingLayers) return;
    m_resizeRequested = true;
    m_resizeRequestCV.notify_one();
}

template <typename T> ErrorCode Index<T>::GrowTopLayer()
{
    std::unique_lock<std::mutex> coordinator(m_resizeMutex);
    ErrorCode ret = ValidateDynamicLayers();
    if (ret != ErrorCode::Success) return ret;
    if (!m_options.m_enableDynamicLayers || !m_bReady) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "GrowTopLayer requires a ready index with EnableDynamicLayers=true.\n");
        return ErrorCode::Fail;
    }
    m_resizingLayers = true;
    std::unique_lock<LayerMutex> mutations(m_checkPointLock, std::defer_lock);
    if (!mutations.try_lock_for(std::chrono::seconds(m_options.m_layerResizeTimeoutSeconds))) {
        m_resizingLayers = false;
        m_layerResizeStatus = ErrorCode::Fail;
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Dynamic growth timed out waiting for writes.\n");
        return ErrorCode::Fail;
    }
    try {
        ret = GrowTopLayerInternal();
    } catch (const std::exception& error) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Dynamic layer construction failed: %s.\n", error.what());
        ret = ErrorCode::Fail;
    }
    m_resizingLayers = false;
    m_layerResizeStatus = ret;
    if (ret != ErrorCode::Success) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "GrowTopLayer failed with error %d; existing topology retained unless publication completed.\n",
            static_cast<int>(ret));
    }
    return ret;
}

template <typename T> ErrorCode Index<T>::GrowTopLayerInternal()
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(m_options.m_layerResizeTimeoutSeconds);
    {
        // Finish queries that could have observed the old maintenance-admission flag before draining workers.
        std::unique_lock<LayerMutex> barrier(m_layerMutex, std::defer_lock);
        if (!barrier.try_lock_until(deadline)) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Dynamic layer timed out waiting for active queries/iterators.\n");
            return ErrorCode::Fail;
        }
    }
    while (!PostingMaintenanceFinished()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Dynamic layer timed out draining posting maintenance.\n");
            return ErrorCode::Fail;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    const int layer = static_cast<int>(m_extraSearchers.size());
    if (layer >= m_options.m_maxLayers) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Dynamic layer growth reached MaxLayers=%d.\n", m_options.m_maxLayers);
        return ErrorCode::Fail;
    }
    for (auto& searcher : m_extraSearchers) {
        auto dynamicSearcher = std::dynamic_pointer_cast<ExtraDynamicSearcher<T>>(searcher);
        if (!dynamicSearcher || dynamicSearcher->MaintenanceStatus() != ErrorCode::Success) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Dynamic growth cannot proceed after failed posting maintenance.\n");
            return ErrorCode::Fail;
        }
    }

    std::vector<SizeType> ids;
    ByteArray bytes;
    const auto previousMemory = GetHeadMemoryUsage();
    {
        LayerReadLock topology(m_layerMutex, m_layerLockEnabled);
        for (SizeType i = 0; i < m_topIndex->GetNumSamples(); ++i) {
            if (m_topIndex->ContainSample(i)) ids.push_back(*m_topLocalToGlobalID.At(i));
        }
        if (ids.size() < 2) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Dynamic growth needs at least two live head vectors.\n");
            return ErrorCode::EmptyData;
        }
        const SizeType maxVID = *std::max_element(ids.begin(), ids.end());
        if (m_options.m_maxID <= 0 || std::any_of(ids.begin(), ids.end(), [](SizeType id) { return id < 0; }) || maxVID >= m_options.m_maxID ||
            layer > ((std::numeric_limits<SizeType>::max)() - maxVID) / m_options.m_maxID) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Dynamic layer posting keys exceed MaxID or the configured VID width.\n");
            return ErrorCode::Key_OverFlow;
        }
        bytes = ByteArray::Alloc(ids.size() * GetValueSize());
        size_t offset = 0;
        for (SizeType i = 0; i < m_topIndex->GetNumSamples(); ++i) {
            if (m_topIndex->ContainSample(i)) {
                memcpy(bytes.Data() + offset, m_topIndex->GetSample(i), GetValueSize());
                offset += GetValueSize();
            }
        }
    }
    auto vectors = std::make_shared<BasicVectorSet>(bytes, GetEnumValueType<T>(), m_options.m_dim, static_cast<SizeType>(ids.size()));
    std::random_device random;
    std::ostringstream token;
    token << "_dynamic_layer_" << layer << "_" << std::hex << random() << random();
    const std::string generation = token.str();
    const std::string prefix = m_options.m_storage == Storage::FILEIO
        ? generation + FolderSep + std::filesystem::path(m_options.m_ssdMappingFile).filename().string()
        : m_options.m_tikvKeyPrefix + generation;
    const std::filesystem::path directory = std::filesystem::path(m_options.m_indexDirectory) / generation;
    std::error_code filesystemError;
    if (!std::filesystem::create_directory(directory, filesystemError) || filesystemError) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot create dynamic generation directory: %s.\n", directory.string().c_str());
        return ErrorCode::FailedCreateFile;
    }

    Index<T> candidate;
    candidate.m_options = m_options;
    candidate.m_options.m_enableDynamicLayers = false;
    candidate.m_options.m_layers = layer + 1;
    candidate.m_options.m_headIndexFolder = generation + FolderSep + "HeadIndex";
    candidate.m_options.m_headIDFile = generation + FolderSep + "HeadIDs.bin";
    candidate.m_options.m_headVectorFile = generation + FolderSep + "HeadVectors.bin";
    candidate.m_options.m_noOutput = false;
    candidate.m_options.m_headVectorCount = (std::max)(SizeType{1}, static_cast<SizeType>(
        ids.size() * (std::min)(m_options.m_ratio,
            static_cast<double>(m_options.m_headMemoryTarget) / previousMemory)));
    candidate.m_options.m_headVectorCount = (std::min)(candidate.m_options.m_headVectorCount, static_cast<SizeType>(ids.size() - 1));
    candidate.m_freeWorkSpaceIds = m_freeWorkSpaceIds;
    candidate.m_workspaceCount = m_workspaceCount.load();
    candidate.m_extraSearchers = m_extraSearchers;
    candidate.m_fComputeDistance = m_fComputeDistance;
    candidate.m_iBaseSquare = m_iBaseSquare;
    candidate.m_options.m_layerStoragePrefixes.clear();
    for (int i = 0; i < layer; ++i) {
        if (i > 0) candidate.m_options.m_layerStoragePrefixes += '|';
        candidate.m_options.m_layerStoragePrefixes += LayerStoragePrefix(i);
    }
    candidate.m_options.m_layerStoragePrefixes += "|" + prefix;

    bool published = false;
    const auto construct = [&]() -> ErrorCode {
        auto headConfig = f_createIO();
        const std::string headConfigPath = (directory / "head-options.ini").string();
        if (!headConfig || !headConfig->Initialize(headConfigPath.c_str(), std::ios::out)) return ErrorCode::FailedCreateFile;
        if (headConfig->WriteString("[BuildHead]\n") == 0) return ErrorCode::DiskIOFail;
        auto result = m_topIndex->SaveConfig(headConfig);
        headConfig.reset();
        if (result != ErrorCode::Success) return result;
        Helper::IniReader reader;
        auto configInput = f_createIO();
        if (!configInput || !configInput->Initialize(headConfigPath.c_str(), std::ios::in)) return ErrorCode::FailedOpenFile;
        if ((result = reader.LoadIni(configInput)) != ErrorCode::Success) return result;
        auto readerOptions = std::make_shared<Helper::ReaderOptions>(GetEnumValueType<T>(), m_options.m_dim,
            VectorFileType::DEFAULT, m_options.m_vectorDelimiter, m_options.m_iSSDNumberOfThreads, true);
        std::shared_ptr<Helper::VectorSetReader> vectorReader = std::make_shared<Helper::MemoryVectorReader>(readerOptions, vectors);
        while (true) {
            if (std::chrono::steady_clock::now() >= deadline) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Dynamic layer timed out constructing the replacement head.\n");
                return ErrorCode::Fail;
            }
            candidate.m_topIndex = VectorIndex::CreateInstance(m_topIndex->GetIndexAlgoType(), GetEnumValueType<T>());
            if (!candidate.m_topIndex) return ErrorCode::FailedParseValue;
            for (const auto& parameter : reader.GetParameters("BuildHead")) {
                result = candidate.m_topIndex->SetParameter(parameter.first.c_str(), parameter.second.c_str());
                if (result != ErrorCode::Success) return result;
            }
            // Recompute BKT selection thresholds for each candidate's reduced representative budget.
            candidate.m_options.m_selectThreshold = 0;
            candidate.m_options.m_splitThreshold = 0;
            candidate.m_options.m_splitFactor = 0;
            if (!candidate.SelectHeadInternal<T>(vectorReader)) return ErrorCode::Fail;
            auto headReader = Helper::VectorSetReader::CreateInstance(readerOptions);
            result = headReader->LoadFile(m_options.m_indexDirectory + FolderSep + candidate.m_options.m_headVectorFile);
            if (result != ErrorCode::Success) return result;
            result = candidate.m_topIndex->BuildIndex(headReader->GetVectorSet(), nullptr, false, true, false);
            if (result != ErrorCode::Success) return result;
            result = candidate.m_topLocalToGlobalID.Load(m_options.m_indexDirectory + FolderSep + candidate.m_options.m_headIDFile,
                m_options.m_datasetRowsInBlock, m_options.m_datasetCapacity);
            if (result != ErrorCode::Success) return result;
            candidate.m_topGlobalToLocalID.clear();
            for (SizeType i = 0; i < candidate.m_topLocalToGlobalID.R(); ++i) {
                SizeType local = *candidate.m_topLocalToGlobalID.At(i);
                if (local < 0 || static_cast<size_t>(local) >= ids.size()) return ErrorCode::FailedParseValue;
                SizeType global = ids[local];
                *candidate.m_topLocalToGlobalID.At(i) = global;
                candidate.m_topGlobalToLocalID[global] = i;
            }
            candidate.m_topIndex->SetReady(true);
            auto memory = candidate.GetHeadMemoryUsage();
            if (memory <= m_options.m_headMemoryTarget && memory < previousMemory) break;
            if (candidate.m_options.m_headVectorCount <= 1) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                    "Minimum candidate head exceeds HeadMemoryTarget: estimated %llu bytes, target %llu bytes.\n",
                    (unsigned long long)memory, (unsigned long long)m_options.m_headMemoryTarget);
                return ErrorCode::MemoryOverFlow;
            }
            auto count = candidate.m_options.m_headVectorCount;
            candidate.m_options.m_headVectorCount = (std::max)(SizeType{1}, (std::min)(count - 1,
                static_cast<SizeType>(count * (static_cast<double>(m_options.m_headMemoryTarget) / memory) * 0.8)));
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Rebuilding dynamic head with %lld representatives to meet memory target.\n",
                (std::int64_t)candidate.m_options.m_headVectorCount);
        }
        std::shared_ptr<Helper::KeyValueIO> db;
        if (m_options.m_storage == Storage::FILEIO) candidate.PrepareDB(db, layer);
        else db = CreateTiKVLayerDB(prefix);
        if (!db || !db->Available()) return ErrorCode::Fail;
        auto extra = std::make_shared<ExtraDynamicSearcher<T>>(candidate.m_options, layer, &candidate, db);
        candidate.m_extraSearchers.push_back(extra);
        result = extra->BuildRoutingLayer(vectors, ids, m_options.m_vectorSize);
        if (result != ErrorCode::Success) return result;
        result = extra->MaintenanceStatus();
        if (result != ErrorCode::Success) return result;

        std::filesystem::create_directory(directory / "HeadIndex", filesystemError);
        if (filesystemError) return ErrorCode::FailedCreateFile;
        auto headFiles = candidate.m_topIndex->GetIndexFiles();
        std::vector<std::shared_ptr<Helper::DiskIO>> outputs;
        for (const auto& file : *headFiles) {
            auto output = f_createIO();
            if (!output || !output->Initialize((directory / "HeadIndex" / file).string().c_str(),
                std::ios::binary | std::ios::out)) return ErrorCode::FailedCreateFile;
            outputs.push_back(output);
        }
        result = candidate.m_topIndex->SaveIndexData(outputs);
        outputs.clear();
        if (result != ErrorCode::Success) return result;
        result = candidate.m_topLocalToGlobalID.Save(m_options.m_indexDirectory + FolderSep + candidate.m_options.m_headIDFile);
        if (result != ErrorCode::Success) return result;
        for (auto& searcher : candidate.m_extraSearchers) {
            result = searcher->Checkpoint(m_options.m_indexDirectory);
            if (result != ErrorCode::Success) return result;
        }
        if (m_options.m_shareDB && m_db) {
            result = m_db->Checkpoint(m_options.m_indexDirectory);
            if (result != ErrorCode::Success) return result;
        }

        candidate.m_options.m_enableDynamicLayers = m_options.m_enableDynamicLayers;
        candidate.m_pMetadata = m_pMetadata;
        candidate.m_pMetaToVec = m_pMetaToVec;
        candidate.m_sMetadataFile = m_sMetadataFile;
        candidate.m_sMetadataIndexFile = m_sMetadataIndexFile;
        if (m_pMetadata) {
            auto metadata = f_createIO(), metadataIndex = f_createIO();
            if (!metadata || !metadataIndex ||
                !metadata->Initialize((m_options.m_indexDirectory + FolderSep + m_sMetadataFile).c_str(), std::ios::binary | std::ios::out) ||
                !metadataIndex->Initialize((m_options.m_indexDirectory + FolderSep + m_sMetadataIndexFile).c_str(), std::ios::binary | std::ios::out)) return ErrorCode::FailedCreateFile;
            result = m_pMetadata->SaveMetadata(metadata, metadataIndex);
            metadata.reset();
            metadataIndex.reset();
            if (result != ErrorCode::Success) return result;
            if ((result = SyncLayerPath(m_options.m_indexDirectory + FolderSep + m_sMetadataFile)) != ErrorCode::Success) return result;
            if ((result = SyncLayerPath(m_options.m_indexDirectory + FolderSep + m_sMetadataIndexFile)) != ErrorCode::Success) return result;
        }
        auto configOutput = f_createIO();
        const std::string pending = (directory / "indexloader.ini").string();
        if (!configOutput || !configOutput->Initialize(pending.c_str(), std::ios::out)) return ErrorCode::FailedCreateFile;
        result = candidate.SaveIndexConfig(configOutput);
        configOutput.reset();
        if (result != ErrorCode::Success) return result;
        for (const auto& file : *headFiles) {
            if ((result = SyncLayerPath((directory / "HeadIndex" / file).string())) != ErrorCode::Success) return result;
        }
        if ((result = SyncLayerPath(m_options.m_indexDirectory + FolderSep + candidate.m_options.m_headIDFile)) != ErrorCode::Success) return result;
        if ((result = SyncLayerPath(pending)) != ErrorCode::Success) return result;
        if (m_options.m_storage == Storage::FILEIO || !m_options.m_distributedVersionMap) {
            for (int i = 0; i <= layer; ++i) {
                result = SyncLayerPath(m_options.m_indexDirectory + FolderSep + m_options.m_deleteIDFile + "_" + std::to_string(i));
                if (result != ErrorCode::Success) return result;
            }
            if (m_options.m_storage == Storage::FILEIO) {
                for (int i = 0; i <= layer; ++i) {
                    if (i > 0 && m_options.m_shareDB &&
                        candidate.LayerStoragePrefix(i) == candidate.LayerStoragePrefix(0)) continue;
                    const std::string mapping = m_options.m_indexDirectory + FolderSep +
                        candidate.LayerStoragePrefix(i) + "_" + std::to_string(i);
                    for (const auto& file : {mapping, mapping + "_postings", mapping + "_postings_blockpool"}) {
                        if ((result = SyncLayerPath(file)) != ErrorCode::Success) return result;
                    }
                }
            }
        }
        if ((result = SyncLayerPath((directory / "HeadIndex").string())) != ErrorCode::Success) return result;
        if ((result = SyncLayerPath(directory.string())) != ErrorCode::Success) return result;
        if ((result = SyncLayerPath(m_options.m_indexDirectory)) != ErrorCode::Success) return result;

        const int extraWorkers = m_options.m_insertThreadNum + m_options.m_reassignThreadNum + m_options.m_appendThreadNum;
        for (int i = 0; i < extraWorkers; ++i) m_freeWorkSpaceIds->push(m_workspaceCount.fetch_add(1));
        std::unique_lock<LayerMutex> topology(m_layerMutex, std::defer_lock);
        if (!topology.try_lock_until(std::chrono::steady_clock::now() + std::chrono::seconds(m_options.m_layerResizeTimeoutSeconds))) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Dynamic layer timed out waiting for queries/iterators to release the old topology.\n");
            return ErrorCode::Fail;
        }
        LayerReadLock exclusiveContext(m_layerMutex, true, true, true);
        std::filesystem::rename(pending, std::filesystem::path(m_options.m_indexDirectory) / "indexloader.ini", filesystemError);
        if (filesystemError) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot publish dynamic index manifest: %s.\n", filesystemError.message().c_str());
            return ErrorCode::DiskIOFail;
        }
        published = true;
        m_topIndex.swap(candidate.m_topIndex);
        m_topLocalToGlobalID.Swap(candidate.m_topLocalToGlobalID);
        m_topGlobalToLocalID.swap(candidate.m_topGlobalToLocalID);
        m_extraSearchers.swap(candidate.m_extraSearchers);
        m_options.m_headIndexFolder.swap(candidate.m_options.m_headIndexFolder);
        m_options.m_headIDFile.swap(candidate.m_options.m_headIDFile);
        m_options.m_headVectorFile.swap(candidate.m_options.m_headVectorFile);
        m_options.m_layerStoragePrefixes.swap(candidate.m_options.m_layerStoragePrefixes);
        m_options.m_layers = layer + 1;
        extra->RebindHeadIndex(this, m_options);
        candidate.m_extraSearchers.clear();
        result = SyncLayerPath(m_options.m_indexDirectory);
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
            "[DYNAMIC-LAYER] posting layers %d -> %d, old head %zu vectors/%llu bytes, new head %d vectors/%llu bytes, prefix=%s, base vectors=%lld.\n",
            layer, layer + 1, ids.size(), (unsigned long long)previousMemory,
            m_topIndex->GetNumSamples(), (unsigned long long)GetHeadMemoryUsage(), prefix.c_str(),
            (std::int64_t)m_extraSearchers.front()->GetNumSamples());
        return result;
    };
    ErrorCode result;
    try {
        result = construct();
    } catch (const std::exception& error) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Dynamic candidate construction failed: %s.\n", error.what());
        result = ErrorCode::Fail;
    }
    if (!published) {
        while (!candidate.PostingMaintenanceFinished()) std::this_thread::sleep_for(std::chrono::milliseconds(20));
        candidate.m_extraSearchers.clear();
        std::filesystem::remove_all(directory, filesystemError);
        if (filesystemError) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot clean failed dynamic generation %s: %s.\n",
                directory.string().c_str(), filesystemError.message().c_str());
        }
        // The isolated TiKV prefix is never reused; partial KV data is retained for explicit administrative cleanup.
    }
    return result;
}

std::function<std::shared_ptr<Helper::DiskIO>(void)> f_createAsyncIO = []() -> std::shared_ptr<Helper::DiskIO> {
    return std::shared_ptr<Helper::DiskIO>(new Helper::AsyncFileIO());
};

template <typename T> bool Index<T>::CheckHeadIndexType()
{
    SPTAG::VectorValueType v1 = m_topIndex->GetVectorValueType(), v2 = GetEnumValueType<T>();
    if (v1 != v2)
    {
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Error, "Head index and vectors don't have the same value types, which are %s %s\n",
            SPTAG::Helper::Convert::ConvertToString(v1).c_str(), SPTAG::Helper::Convert::ConvertToString(v2).c_str());
        if (!m_pQuantizer)
            return false;
    }
    return true;
}

template <typename T> void Index<T>::SetQuantizer(std::shared_ptr<SPTAG::COMMON::IQuantizer> quantizer)
{
    LayerReadLock mutations(m_checkPointLock, m_layerLockEnabled);
    LayerReadLock topology(m_layerMutex, m_layerLockEnabled);
    if (m_layerLockEnabled && quantizer && m_bReady) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot attach a quantizer to an active dynamic-layer index.\n");
        return;
    }
    m_pQuantizer = quantizer;
    if (m_pQuantizer)
    {
        m_pQuantizer->SetEnableADC(m_options.m_enableADC);
        m_fComputeDistance = m_pQuantizer->DistanceCalcSelector<T>(m_options.m_distCalcMethod);
        m_iBaseSquare = (m_options.m_distCalcMethod == DistCalcMethod::Cosine)
                            ? m_pQuantizer->GetBase() * m_pQuantizer->GetBase()
                            : 1;
    }
    else
    {
        m_fComputeDistance = COMMON::DistanceCalcSelector<T>(m_options.m_distCalcMethod);
        m_iBaseSquare = (m_options.m_distCalcMethod == DistCalcMethod::Cosine)
                            ? COMMON::Utils::GetBase<std::uint8_t>() * COMMON::Utils::GetBase<std::uint8_t>()
                            : 1;
    }
    if (m_topIndex)
    {
        m_topIndex->SetQuantizer(quantizer);
    }
}

template <typename T> ErrorCode Index<T>::LoadConfig(Helper::IniReader &p_reader)
{
    IndexAlgoType algoType = p_reader.GetParameter("Base", "IndexAlgoType", IndexAlgoType::Undefined);
    VectorValueType valueType = p_reader.GetParameter("Base", "ValueType", VectorValueType::Undefined);
    if ((m_topIndex = CreateInstance(algoType, valueType)) == nullptr)
        return ErrorCode::FailedParseValue;

    std::string sections[] = {"Base", "SelectHead", "BuildHead", "BuildSSDIndex"};
    for (int i = 0; i < 4; i++)
    {
        auto parameters = p_reader.GetParameters(sections[i].c_str());
        for (auto iter = parameters.begin(); iter != parameters.end(); iter++)
        {
            auto status = SetParameter(iter->first.c_str(), iter->second.c_str(), sections[i].c_str());
            if (status != ErrorCode::Success) return status;
        }
    }

    if (m_pQuantizer)
    {
        m_pQuantizer->SetEnableADC(m_options.m_enableADC);
    }

    auto ret = ValidateDynamicLayers();
    if (ret == ErrorCode::Success && (m_options.m_enableDynamicLayers || !m_options.m_layerStoragePrefixes.empty())) {
        m_layerLockEnabled = true;
    }
    return ret;
}

template <typename T> ErrorCode Index<T>::LoadIndexDataFromMemory(const std::vector<ByteArray> &p_indexBlobs)
{
    auto validation = ValidateDynamicLayers();
    if (validation != ErrorCode::Success) return validation;
    /** Need to modify **/
    m_topIndex->SetQuantizer(m_pQuantizer);

    if (!m_options.m_persistentBufferPath.empty() && !direxists(m_options.m_persistentBufferPath.c_str()))
        mkdir(m_options.m_persistentBufferPath.c_str());

    if (m_topIndex->LoadIndexDataFromMemory(p_indexBlobs) != ErrorCode::Success)
        return ErrorCode::Fail;

    m_topIndex->SetParameter("NumberOfThreads", std::to_string(m_options.m_iSSDNumberOfThreads));
    // m_topIndex->SetParameter("MaxCheck", std::to_string(m_options.m_maxCheck));
    // m_topIndex->SetParameter("HashTableExponent", std::to_string(m_options.m_hashExp));
    m_topIndex->UpdateIndex();
    m_topIndex->SetReady(true);

    m_topLocalToGlobalID.Initialize(m_topIndex->GetNumSamples(), 1, m_topIndex->m_iDataBlockSize, m_topIndex->m_iDataCapacity,
                                    p_indexBlobs.back().Data(), false);
    m_topGlobalToLocalID.clear();
    for (int i = 0; i < m_topIndex->GetNumSamples(); i++)
    {
        SizeType globalID = *(m_topLocalToGlobalID[i]);
        m_topGlobalToLocalID[globalID] = i;
    }

    if (m_freeWorkSpaceIds == nullptr) {
        m_freeWorkSpaceIds.reset(new Helper::Concurrent::ConcurrentQueue<int>());
        int maxIOThreads = (m_options.m_storage == Storage::STATIC) ? max(m_options.m_searchThreadNum, m_options.m_iSSDNumberOfThreads) : max(m_options.m_ioThreads, (2 * max(m_options.m_searchThreadNum, m_options.m_iSSDNumberOfThreads) +
                                    (m_options.m_layers + 1) * (m_options.m_insertThreadNum + m_options.m_reassignThreadNum + m_options.m_appendThreadNum)));
        for (int i = 0; i < maxIOThreads; i++) {
            m_freeWorkSpaceIds->push(i);
        }
        m_workspaceCount = maxIOThreads;
    }

    if (m_options.m_shareDB) PrepareDB(m_db);
    m_extraSearchers.resize(m_options.m_layers);
    for (int i = m_options.m_layers - 1; i >= 0; i--) {
        if (m_options.m_storage == Storage::STATIC)
        {
            m_extraSearchers[i] = std::make_shared<ExtraStaticSearcher<T>>(i, this);
        }
        else
        {
            auto db = m_db;
            if (!m_options.m_layerStoragePrefixes.empty() &&
                (!m_options.m_shareDB || LayerStoragePrefix(i) != LayerStoragePrefix(0))) PrepareDB(db, i);
            m_extraSearchers[i] = std::make_shared<ExtraDynamicSearcher<T>>(m_options, i, this, db);
        }

        if (!m_extraSearchers[i]->LoadIndex(m_options))
            return ErrorCode::Fail;
    }
    StartLayerResizeWorker();
    RequestLayerResize();
    return ErrorCode::Success;
}

template <typename T>
ErrorCode Index<T>::LoadIndexData(const std::vector<std::shared_ptr<Helper::DiskIO>> &p_indexStreams)
{
    auto validation = ValidateDynamicLayers();
    if (validation != ErrorCode::Success) return validation;
    m_topIndex->SetQuantizer(m_pQuantizer);

    if (!m_options.m_persistentBufferPath.empty() && !direxists(m_options.m_persistentBufferPath.c_str()))
        mkdir(m_options.m_persistentBufferPath.c_str());

    auto headfiles = m_topIndex->GetIndexFiles();
    if (m_options.m_recovery)
    {
        std::shared_ptr<std::vector<std::string>> files(new std::vector<std::string>);
        auto headfiles = m_topIndex->GetIndexFiles();
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Recovery: Loading another in-memory index\n");
        std::string filename = m_options.m_persistentBufferPath + FolderSep + m_options.m_headIndexFolder;
        for (auto file : *headfiles)
        {
            files->push_back(filename + FolderSep + file);
        }
        std::vector<std::shared_ptr<Helper::DiskIO>> handles;
        for (std::string &f : *files)
        {
            auto ptr = SPTAG::f_createIO();
            if (ptr == nullptr || !ptr->Initialize(f.c_str(), std::ios::binary | std::ios::in))
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot open file %s!\n", f.c_str());
                ptr = nullptr;
            }
            handles.push_back(std::move(ptr));
        }
        if (m_topIndex->LoadIndexData(handles) != ErrorCode::Success)
            return ErrorCode::Fail;
    }
    else if (m_topIndex->LoadIndexData(p_indexStreams) != ErrorCode::Success)
        return ErrorCode::Fail;

    m_topIndex->SetParameter("NumberOfThreads", std::to_string(m_options.m_iSSDNumberOfThreads));
    m_topIndex->SetParameter("MaxCheck", std::to_string(m_options.m_maxCheck));
    m_topIndex->SetParameter("HashTableExponent", std::to_string(m_options.m_hashExp));
    m_topIndex->UpdateIndex();
    m_topIndex->SetReady(true);

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loading headID map\n");
    m_topLocalToGlobalID.Load(p_indexStreams[m_topIndex->GetIndexFiles()->size()], m_topIndex->m_iDataBlockSize,
                              m_topIndex->m_iDataCapacity);
    m_topGlobalToLocalID.clear();
    for (int i = 0; i < m_topIndex->GetNumSamples(); i++)
    {
        SizeType globalID = *(m_topLocalToGlobalID[i]);
        m_topGlobalToLocalID[globalID] = i;
    }

    if (m_freeWorkSpaceIds == nullptr) {
        m_freeWorkSpaceIds.reset(new Helper::Concurrent::ConcurrentQueue<int>());
        int maxIOThreads = (m_options.m_storage == Storage::STATIC) ? max(m_options.m_searchThreadNum, m_options.m_iSSDNumberOfThreads) : max(m_options.m_ioThreads, (2 * max(m_options.m_searchThreadNum, m_options.m_iSSDNumberOfThreads) +
                                    (m_options.m_layers + 1) * (m_options.m_insertThreadNum + m_options.m_reassignThreadNum + m_options.m_appendThreadNum)));
        for (int i = 0; i < maxIOThreads; i++) {
            m_freeWorkSpaceIds->push(i);
        }
        m_workspaceCount = maxIOThreads;
    }

    if (m_options.m_shareDB) PrepareDB(m_db);
    m_extraSearchers.resize(m_options.m_layers);
    for (int i = m_options.m_layers - 1; i >= 0; i--) {
        if (m_options.m_storage == Storage::STATIC)
        {
            m_extraSearchers[i] = std::make_shared<ExtraStaticSearcher<T>>(i, this);
        }
        else
        {
            auto db = m_db;
            if (!m_options.m_layerStoragePrefixes.empty() &&
                (!m_options.m_shareDB || LayerStoragePrefix(i) != LayerStoragePrefix(0))) PrepareDB(db, i);
            m_extraSearchers[i] = std::make_shared<ExtraDynamicSearcher<T>>(m_options, i, this, db);
        }

        if (m_options.m_storage != Storage::STATIC && !m_extraSearchers[i]->Available())
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Extrasearcher is not available and failed to initialize.\n");
            return ErrorCode::Fail;
        }

        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loading storage\n");
        if (!(m_extraSearchers[i]->LoadIndex(m_options)))
            return ErrorCode::Fail;
    }
    StartLayerResizeWorker();
    RequestLayerResize();
    return ErrorCode::Success;
}

template <typename T> ErrorCode Index<T>::SaveConfig(std::shared_ptr<Helper::DiskIO> p_configOut)
{
    LayerReadLock lock(m_layerMutex, m_layerLockEnabled);
    IOSTRING(p_configOut, WriteString, "[Base]\n");
#define DefineBasicParameter(VarName, VarType, DefaultValue, RepresentStr)                                             \
    IOSTRING(p_configOut, WriteString,                                                                                 \
             (RepresentStr + std::string("=") + SPTAG::Helper::Convert::ConvertToString(m_options.VarName) +           \
              std::string("\n"))                                                                                       \
                 .c_str());

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineBasicParameter

    IOSTRING(p_configOut, WriteString, "[SelectHead]\n");
#define DefineSelectHeadParameter(VarName, VarType, DefaultValue, RepresentStr)                                        \
    IOSTRING(p_configOut, WriteString,                                                                                 \
             (RepresentStr + std::string("=") + SPTAG::Helper::Convert::ConvertToString(m_options.VarName) +           \
              std::string("\n"))                                                                                       \
                 .c_str());

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineSelectHeadParameter

    IOSTRING(p_configOut, WriteString, "[BuildHead]\n");
#define DefineBuildHeadParameter(VarName, VarType, DefaultValue, RepresentStr)                                         \
    IOSTRING(p_configOut, WriteString,                                                                                 \
             (RepresentStr + std::string("=") + SPTAG::Helper::Convert::ConvertToString(m_options.VarName) +           \
              std::string("\n"))                                                                                       \
                 .c_str());

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineBuildHeadParameter

    m_topIndex->SaveConfig(p_configOut);

    Helper::Convert::ConvertStringTo<int>(m_topIndex->GetParameter("HashTableExponent").c_str(), m_options.m_hashExp);
    IOSTRING(p_configOut, WriteString, "[BuildSSDIndex]\n");
#define DefineSSDParameter(VarName, VarType, DefaultValue, RepresentStr)                                               \
    IOSTRING(p_configOut, WriteString,                                                                                 \
             (RepresentStr + std::string("=") + SPTAG::Helper::Convert::ConvertToString(m_options.VarName) +           \
              std::string("\n"))                                                                                       \
                 .c_str());

#include "inc/Core/SPANN/ParameterDefinitionList.h"
#undef DefineSSDParameter

    IOSTRING(p_configOut, WriteString, "\n");
    return ErrorCode::Success;
}

template <typename T>
ErrorCode Index<T>::SaveIndexData(const std::vector<std::shared_ptr<Helper::DiskIO>> &p_indexStreams)
{
    if (m_topIndex == nullptr) return ErrorCode::EmptyIndex;

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SaveIndexData: waiting for all background jobs to finish...\n");
    auto waitStart = std::chrono::steady_clock::now();
    int pollCount = 0;
    while (!PostingMaintenanceFinished())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if (++pollCount % 250 == 0) { // every ~5 seconds
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - waitStart).count();
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SaveIndexData: still waiting for background jobs (%lld s elapsed)\n", (long long)elapsed);
        }
    }
    {
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - waitStart).count();
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SaveIndexData: all background jobs finished (waited %lld s)\n", (long long)elapsed);
    }

    LayerReadLock mutations(m_checkPointLock, m_layerLockEnabled);
    LayerReadLock topology(m_layerMutex, m_layerLockEnabled);
    ErrorCode ret;
    if ((ret = m_topIndex->SaveIndexData(p_indexStreams)) != ErrorCode::Success)
        return ret;

    if ((ret = m_topLocalToGlobalID.Save(p_indexStreams[m_topIndex->GetIndexFiles()->size()])) != ErrorCode::Success)
        return ret;

    for (auto& searcher : m_extraSearchers) {
        if ((ret = searcher->Checkpoint(m_options.m_indexDirectory)) != ErrorCode::Success)
            return ret;
    }
    if (m_options.m_shareDB && m_options.m_storage != Storage::STATIC) {
        if (!m_db) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Shared posting DB is unavailable during save.\n");
            return ErrorCode::Fail;
        }
        if ((ret = m_db->Checkpoint(m_options.m_indexDirectory)) != ErrorCode::Success) return ret;
    }
    return ErrorCode::Success;
}

#pragma region K - NN search

template <typename T> ErrorCode Index<T>::SearchIndex(QueryResult &p_query, bool p_searchDeleted) const
{
    return SearchIndex(p_query, nullptr, p_searchDeleted);
}

template <typename T> ErrorCode Index<T>::SearchIndex(QueryResult &p_query, SearchStats* p_stats, bool p_searchDeleted) const
{
    LayerReadLock topology(m_layerMutex, m_layerLockEnabled);
    if (!m_bReady)
        return ErrorCode::EmptyIndex;

    COMMON::QueryResultSet<T> *p_queryResults;
    if (p_query.GetResultNum() >= m_options.m_searchInternalResultNum)
        p_queryResults = (COMMON::QueryResultSet<T> *)&p_query;
    else
    {
        p_queryResults =
            new COMMON::QueryResultSet<T>((const T *)p_query.GetTarget(), m_options.m_searchInternalResultNum, p_query.WithMeta(), p_query.WithVec());
    }

    if (m_pQuantizer && !p_queryResults->HasQuantizedTarget())
    {
        p_queryResults->SetTarget((const T *)p_query.GetTarget(), m_pQuantizer);
    }

    ErrorCode ret;
    auto searchStart = std::chrono::high_resolution_clock::now();
    if ((ret = m_topIndex->SearchIndex(*p_queryResults)) != ErrorCode::Success)
        return ret;
    auto headEnd = std::chrono::high_resolution_clock::now();
    if (p_stats)
    {
        p_stats->m_totalLatency = std::chrono::duration_cast<std::chrono::microseconds>(headEnd - searchStart).count() / 1000.0;
    }

    for (int i = 0; i < p_queryResults->GetResultNum(); ++i)
    {
        auto res = p_queryResults->GetResult(i);
        if (res->VID == -1) break;

        res->VID = static_cast<SizeType>(*(m_topLocalToGlobalID[res->VID]));
    }

    if ((ret = SearchDiskIndex(*p_queryResults, p_stats)) != ErrorCode::Success)
        return ret;
    auto diskEnd = std::chrono::high_resolution_clock::now();
    if (p_stats)
    {
        p_stats->m_exLatency = std::chrono::duration_cast<std::chrono::microseconds>(diskEnd - headEnd).count() / 1000.0;
        p_stats->m_totalLatency = p_stats->m_totalSearchLatency = std::chrono::duration_cast<std::chrono::microseconds>(diskEnd - searchStart).count() / 1000.0;
    }

    if (p_query.GetResultNum() < m_options.m_searchInternalResultNum)
    {
        std::copy(p_queryResults->GetResults(), p_queryResults->GetResults() + p_query.GetResultNum(),
                  p_query.GetResults());
        p_query.SetScanned(p_queryResults->GetScanned());
        delete p_queryResults;
    }

    if (p_query.WithMeta() && nullptr != m_pMetadata)
    {
        for (int i = 0; i < p_query.GetResultNum(); ++i)
        {
            SizeType result = p_query.GetResult(i)->VID;
            // if (result > m_pMetadata->Count()) SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "vid return is beyond the
            // metadata set:(%d > (%d, %d))\n", result, GetNumSamples(), m_pMetadata->Count());
            p_query.SetMetadata(i, (result < 0) ? ByteArray::c_empty : m_pMetadata->GetMetadataCopy(result));
        }
    }
    return ErrorCode::Success;
}

template <typename T>
ErrorCode Index<T>::SearchIndexIterative(QueryResult &p_headQuery, QueryResult &p_query,
                                         COMMON::WorkSpace *p_indexWorkspace, ExtraWorkSpace *p_extraWorkspace,
                                         int p_batch, int &resultCount, bool first) const
{
    LayerReadLock topology(m_layerMutex, m_layerLockEnabled);
    if (!m_bReady)
        return ErrorCode::EmptyIndex;

    COMMON::QueryResultSet<T> *p_headQueryResults = (COMMON::QueryResultSet<T> *)&p_headQuery;
    COMMON::QueryResultSet<T> *p_queryResults = (COMMON::QueryResultSet<T> *)&p_query;
    resultCount = 0;

    const int layerCount = static_cast<int>(m_extraSearchers.size());
    if (layerCount == 0)
        return ErrorCode::EmptyIndex;

    const int internalResultNum = m_options.m_searchInternalResultNum;
    const int refillBatch = max(1, internalResultNum / 2);
    const int headBatch = max(m_options.m_headBatch, internalResultNum);

    const bool iteratorStateSizeMismatch = static_cast<int>(p_extraWorkspace->m_iteratorLayerStates.size()) != layerCount;
    if (!first && iteratorStateSizeMismatch)
        return ErrorCode::Fail;

    if (first)
    {
        p_extraWorkspace->ResetIteratorState(layerCount, m_options.m_maxCheck, m_options.m_hashExp);
        p_extraWorkspace->m_versionReadPolicy = COMMON::VersionReadPolicy::BypassCacheNoFill;
        if (m_pQuantizer)
        {
            p_headQueryResults->SetTarget((const T*)p_headQuery.GetTarget(), m_pQuantizer);
            p_queryResults->SetTarget((const T*)p_query.GetTarget(), m_pQuantizer);
        }
    }

    auto fetchHeadCandidates = [&](int p_count) {
        std::vector<BasicResult> candidates;
        candidates.reserve(p_count);
        while (static_cast<int>(candidates.size()) < p_count)
        {
            p_headQueryResults->SetResultNum(p_extraWorkspace->m_iteratorHeadInitialized ? headBatch : internalResultNum);
            p_headQueryResults->Reset();
            bool found = m_topIndex->SearchIndexIterativeFromNeareast(
                *p_headQueryResults, p_indexWorkspace, !p_extraWorkspace->m_iteratorHeadInitialized);
            p_extraWorkspace->m_iteratorHeadInitialized = true;
            if (!found)
                break;

            for (int i = 0; i < p_headQueryResults->GetResultNum() && static_cast<int>(candidates.size()) < p_count; ++i)
            {
                auto res = p_headQueryResults->GetResult(i);
                if (res == nullptr || res->VID < 0 || m_topLocalToGlobalID.R() == 0)
                    continue;

                SizeType globalVID = static_cast<SizeType>(*(m_topLocalToGlobalID[res->VID]));
                if (globalVID == MaxSize)
                    continue;

                BasicResult candidate = *res;
                candidate.VID = globalVID;
                candidates.emplace_back(std::move(candidate));
            }
        }
        return candidates;
    };

    std::function<ErrorCode(int, std::size_t)> ensureLayerQueueFilled;
    std::function<std::vector<BasicResult>(int, int)> popLayerCandidates;
    ErrorCode refillError = ErrorCode::Success;

    auto appendLayerResults = [&](int p_layer, const std::vector<BasicResult>& p_postings) {
        if (p_postings.empty())
            return ErrorCode::Success;

        if (!m_extraSearchers[p_layer])
            return ErrorCode::Fail;

        auto& state = p_extraWorkspace->m_iteratorLayerStates[p_layer];
        p_extraWorkspace->m_postingIDs.clear();
        p_extraWorkspace->m_postingIDs.reserve(p_postings.size());
        for (const auto& posting : p_postings)
        {
            if (posting.VID >= 0)
                p_extraWorkspace->m_postingIDs.emplace_back(posting.VID);
        }
        if (p_extraWorkspace->m_postingIDs.empty())
            return ErrorCode::Success;

        std::vector<BasicResult> fetched;
        struct DeduperOverrideGuard
        {
            ExtraWorkSpace* m_workspace;

            DeduperOverrideGuard(ExtraWorkSpace* p_workspace, COMMON::OptHashPosVector* p_deduper) :
                m_workspace(p_workspace)
            {
                m_workspace->m_deduperOverride = p_deduper;
            }

            ~DeduperOverrideGuard()
            {
                m_workspace->m_deduperOverride = nullptr;
            }
        } deduperOverrideGuard(p_extraWorkspace, state.m_deduper.get());
        ErrorCode ret = m_extraSearchers[p_layer]->SearchIndexIterativeScan(
            p_extraWorkspace, p_query, fetched, p_layer == 0);
        if (ret != ErrorCode::Success)
            return ret;

        state.CompactConsumed();
        std::size_t sortBegin = state.m_cursor;
        state.m_results.insert(state.m_results.end(), fetched.begin(), fetched.end());
        std::sort(state.m_results.begin() + sortBegin, state.m_results.end(), COMMON::Compare);
        if (p_layer == 0)
            p_extraWorkspace->m_loadedPostingNum += static_cast<int>(p_extraWorkspace->m_postingIDs.size());
        return ErrorCode::Success;
    };

    popLayerCandidates = [&](int p_layer, int p_count) {
        std::vector<BasicResult> candidates;
        if (p_layer < 0 || p_layer >= layerCount || p_count <= 0)
            return candidates;

        ErrorCode ret = ensureLayerQueueFilled(p_layer, p_count);
        if (ret != ErrorCode::Success)
        {
            refillError = ret;
            return candidates;
        }

        auto& state = p_extraWorkspace->m_iteratorLayerStates[p_layer];
        while (static_cast<int>(candidates.size()) < p_count && state.m_cursor < state.m_results.size())
        {
            candidates.emplace_back(state.m_results[state.m_cursor++]);
        }
        return candidates;
    };

    ensureLayerQueueFilled = [&](int p_layer, std::size_t p_desired) {
        if (p_layer < 0 || p_layer >= layerCount)
            return ErrorCode::Fail;

        auto& state = p_extraWorkspace->m_iteratorLayerStates[p_layer];
        while (state.Available() < p_desired)
        {
            const int fetchBatch = (state.m_results.empty() && state.m_cursor == 0) ? internalResultNum : refillBatch;
            refillError = ErrorCode::Success;
            std::vector<BasicResult> postings = (p_layer == layerCount - 1)
                ? fetchHeadCandidates(fetchBatch)
                : popLayerCandidates(p_layer + 1, fetchBatch);
            if (refillError != ErrorCode::Success)
                return refillError;
            if (postings.empty())
                break;

            std::size_t oldSize = state.m_results.size();
            ErrorCode ret = appendLayerResults(p_layer, postings);
            if (ret != ErrorCode::Success)
                return ret;
            if (state.m_results.size() == oldSize && postings.size() < static_cast<std::size_t>(refillBatch))
                break;
        }
        return ErrorCode::Success;
    };

    ErrorCode ret = ensureLayerQueueFilled(0, first ? internalResultNum : 1);
    if (ret != ErrorCode::Success)
        return ret;

    auto& finalState = p_extraWorkspace->m_iteratorLayerStates[0];
    while (resultCount < p_batch)
    {
        if (finalState.Available() <= static_cast<std::size_t>(refillBatch))
        {
            ret = ensureLayerQueueFilled(0, finalState.Available() + refillBatch);
            if (ret != ErrorCode::Success)
                return ret;
        }

        if (finalState.Available() == 0)
            break;

        p_extraWorkspace->m_relaxedMono = p_extraWorkspace->m_loadedPostingNum > internalResultNum;
        auto result = finalState.m_results[finalState.m_cursor++];
        if (p_extraWorkspace->m_filterFunc != nullptr &&
            (m_pMetadata == nullptr || !p_extraWorkspace->m_filterFunc(m_pMetadata->GetMetadata(result.VID))))
        {
            continue;
        }
        if (p_queryResults->AddPoint(result.VID, result.Dist, result.Vec))
            resultCount++;
    }
    p_queryResults->SortResult();

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

template <typename T>
std::shared_ptr<ResultIterator> Index<T>::GetIterator(const void *p_target, bool p_searchDeleted, std::function<bool(const ByteArray&)> p_filterFunc, int p_maxCheck) const
{
    std::shared_lock<LayerMutex> topology(m_layerMutex, std::defer_lock);
    if (m_layerLockEnabled) topology.lock();
    if (!m_bReady)
        return nullptr;
    auto extraWorkspace = m_workSpaceFactory->GetWorkSpace();
    if (!extraWorkspace)
    {
        extraWorkspace.reset(new ExtraWorkSpace());
        InitWorkSpace(extraWorkspace.get(), false);
    }
    else
    {
        InitWorkSpace(extraWorkspace.get(), true);
    }
    extraWorkspace->m_filterFunc = p_filterFunc;
    extraWorkspace->m_versionReadPolicy = COMMON::VersionReadPolicy::BypassCacheNoFill;
    extraWorkspace->m_relaxedMono = false;
    extraWorkspace->m_loadedPostingNum = 0;
    extraWorkspace->m_deduper.clear();
    extraWorkspace->m_postingIDs.clear();
    extraWorkspace->ResetIteratorState(static_cast<int>(m_extraSearchers.size()), m_options.m_maxCheck, m_options.m_hashExp);
    std::shared_ptr<ResultIterator> resultIterator = std::make_shared<SPANNResultIterator<T>>(
        this, m_topIndex.get(), p_target, std::move(extraWorkspace),
        max(m_options.m_headBatch, m_options.m_searchInternalResultNum), p_maxCheck, std::move(topology));
    return resultIterator;
}

template <typename T>
ErrorCode Index<T>::SearchIndexIterativeNext(QueryResult &p_query, COMMON::WorkSpace *workSpace, int p_batch,
                                             int &resultCount, bool p_isFirst, bool p_searchDeleted) const
{
    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "ITERATIVE NOT SUPPORT FOR SPANN");
    return ErrorCode::Undefined;
}

template <typename T> ErrorCode Index<T>::SearchIndexIterativeEnd(std::unique_ptr<COMMON::WorkSpace> space) const
{
    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SearchIndexIterativeEnd NOT SUPPORT FOR SPANN");
    return ErrorCode::Fail;
}

template <typename T>
ErrorCode Index<T>::SearchIndexIterativeEnd(std::unique_ptr<SPANN::ExtraWorkSpace> extraWorkspace) const
{
    if (!m_bReady)
        return ErrorCode::EmptyIndex;

    if (extraWorkspace != nullptr)
        m_workSpaceFactory->ReturnWorkSpace(std::move(extraWorkspace));

    return ErrorCode::Success;
}

template <typename T>
bool Index<T>::SearchIndexIterativeFromNeareast(QueryResult &p_query, COMMON::WorkSpace *p_space, bool p_isFirst,
                                                bool p_searchDeleted) const
{
    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SearchIndexIterativeFromNeareast NOT SUPPORT FOR SPANN");
    return false;
}

template <typename T>
ErrorCode Index<T>::SearchIndexWithFilter(QueryResult &p_query, std::function<bool(const ByteArray &)> filterFunc,
                                          int maxCheck, bool p_searchDeleted) const
{
    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Not Support Filter on SPANN Index!\n");
    return ErrorCode::Fail;
}

template <typename T> 
ErrorCode Index<T>::SearchHeadIndex(QueryResult& p_query, int p_tolayer, ExtraWorkSpace *p_exWorkSpace) const
{
    LayerReadLock topology(m_layerMutex, m_layerLockEnabled);
    COMMON::QueryResultSet<T> *p_queryResults;
    if (p_query.GetResultNum() >= m_options.m_searchInternalResultNum)
        p_queryResults = (COMMON::QueryResultSet<T> *)&p_query;
    else
    {
        p_queryResults =
            new COMMON::QueryResultSet<T>((const T *)p_query.GetTarget(), m_options.m_searchInternalResultNum, p_query.WithMeta(), p_query.WithVec());
    }

    if (m_pQuantizer && !p_queryResults->HasQuantizedTarget())
    {
        p_queryResults->SetTarget((const T *)p_query.GetTarget(), m_pQuantizer);
    }

    ErrorCode ret;
    if ((ret = m_topIndex->SearchIndex(*p_queryResults)) != ErrorCode::Success)
        return ret;

    for (int i = 0; i < p_queryResults->GetResultNum(); ++i)
    {
        auto res = p_queryResults->GetResult(i);
        if (res->VID == -1) break;

        res->VID = static_cast<SizeType>(*(m_topLocalToGlobalID[res->VID]));
    }

    if (p_tolayer < m_extraSearchers.size() && (ret = SearchDiskIndex(*p_queryResults, nullptr, p_tolayer, p_exWorkSpace)) != ErrorCode::Success)
        return ret;

    if (p_query.GetResultNum() < m_options.m_searchInternalResultNum)
    {
        std::copy(p_queryResults->GetResults(), p_queryResults->GetResults() + p_query.GetResultNum(),
                  p_query.GetResults());
        p_query.SetScanned(p_queryResults->GetScanned());
        delete p_queryResults;
    }
    return ErrorCode::Success;
}

template <typename T> 
ErrorCode Index<T>::SearchDiskIndex(QueryResult &p_query, SearchStats *p_stats, int p_tolayer, ExtraWorkSpace *p_exWorkSpace) const
{
    LayerReadLock topology(m_layerMutex, m_layerLockEnabled);
    if (m_extraSearchers.size() == 0) return ErrorCode::EmptyIndex;

    COMMON::QueryResultSet<T>& localResults = *((COMMON::QueryResultSet<T> *)&p_query);
    std::unique_ptr<ExtraWorkSpace> workSpace;
    if (p_exWorkSpace == nullptr) {
        workSpace = m_workSpaceFactory->GetWorkSpace();
        if (!workSpace)
        {
            workSpace.reset(new ExtraWorkSpace());
            InitWorkSpace(workSpace.get(), false);
        }
        else
        {
            InitWorkSpace(workSpace.get(), true);
        }
        p_exWorkSpace = workSpace.get();
        p_exWorkSpace->m_versionReadPolicy = COMMON::VersionReadPolicy::BypassCacheNoFill;
    }

    std::vector<BasicResult> remainResults;
    bool needInternalSort = (localResults.GetResultNum() > m_options.m_searchInternalResultNum);
    if (needInternalSort) {
        remainResults.reserve(localResults.GetResultNum() - m_options.m_searchInternalResultNum);
        for (int i = m_options.m_searchInternalResultNum; i < localResults.GetResultNum(); ++i)
        {
            auto res = localResults.GetResult(i);
            if (res->VID == -1) break;
            remainResults.push_back(*res);
        }
    }

    ErrorCode ret;
    for (int layer = m_extraSearchers.size() - 1; layer >= p_tolayer; layer--) {
        auto& searcher = m_extraSearchers[layer];
        if (!searcher) {
            return ErrorCode::Fail;
        }
        const bool isTargetLayer = (layer == p_tolayer);

        auto setupStart = std::chrono::high_resolution_clock::now();
        p_exWorkSpace->m_deduper.clear();
        p_exWorkSpace->m_postingIDs.clear();

        if (layer < m_extraSearchers.size() - 1 && needInternalSort) localResults.SortResult();
        // Target-layer postings include their posting IDs, so direct candidates are returned through SearchIndex.
        for (int i = 0; i < m_options.m_searchInternalResultNum; ++i)
        {
            auto res = localResults.GetResult(i);
            if (res->VID == -1) continue;

            p_exWorkSpace->m_postingIDs.emplace_back(res->VID);

        }
        auto setupEnd = std::chrono::high_resolution_clock::now();
        double setupLatency = (double)std::chrono::duration_cast<std::chrono::microseconds>(setupEnd - setupStart).count() / 1000;

        localResults.Reset();
        if ((ret = searcher->SearchIndex(p_exWorkSpace, localResults, p_stats, nullptr, nullptr, isTargetLayer)) !=
            ErrorCode::Success)
        {
            return ret;
        }
        if (p_stats)
        {
            p_stats->m_exSetUpLatency += setupLatency;
            if (SearchStats::IsValidBreakdownLayer(layer)) {
                p_stats->m_layerSetupLatency[layer] += setupLatency;
            }
        }
    }

    if (needInternalSort) {
        for (int i = 0; i < remainResults.size(); ++i)
        {
            auto res = &remainResults[i];
            if (res->VID == -1) break;
            if (p_exWorkSpace->m_deduper.CheckAndSet(res->VID)) continue;
            localResults.AddPoint(res->VID, res->Dist, res->Vec);
        }
    }
    localResults.SortResult();
    if (workSpace != nullptr) m_workSpaceFactory->ReturnWorkSpace(std::move(workSpace));
    return ErrorCode::Success;
}

template <typename T>
ErrorCode Index<T>::SearchDiskIndexIterative(QueryResult &p_headQuery, QueryResult &p_query,
                                        ExtraWorkSpace *extraWorkspace) const
{
    /*
    if (extraWorkspace->m_loadPosting)
    {
        COMMON::QueryResultSet<T> *p__headQueryResults = (COMMON::QueryResultSet<T> *)&p_headQuery;
        // std::shared_ptr<ExtraWorkSpace> workSpace = m_workSpacePool->Rent();
        // workSpace->m_deduper.clear();
        extraWorkspace->m_postingIDs.clear();

        // float limitDist = p_queryResults->GetResult(0)->Dist * m_options.m_maxDistRatio;

        for (int i = 0; i < p__headQueryResults->GetResultNum(); ++i)
        {
            auto res = p__headQueryResults->GetResult(i);
            // break or continue
            if (res->VID == -1)
                break;

            extraWorkspace->m_postingIDs.emplace_back(res->VID);

            if (m_topLocalToGlobalID.R() != 0)
                res->VID = static_cast<SizeType>(*(m_topLocalToGlobalID[res->VID]));
            else
            {
                res->VID = -1;
                res->Dist = MaxDist;
            }
            if (res->VID == MaxSize)
            {
                res->VID = -1;
                res->Dist = MaxDist;
            }
        }
        extraWorkspace->m_loadedPostingNum += (int)(extraWorkspace->m_postingIDs.size());
    }

    ErrorCode ret = m_extraSearcher->SearchIterativeNext(extraWorkspace, p_headQuery, p_query, m_topIndex, this);
    if (ret == ErrorCode::VectorNotFound && extraWorkspace->m_loadedPostingNum >= m_options.m_searchInternalResultNum)
        extraWorkspace->m_relaxedMono = true;
    return ret;
    */
   return ErrorCode::Undefined;
}

template <typename T> std::unique_ptr<COMMON::WorkSpace> Index<T>::RentWorkSpace(int batch, std::function<bool(const ByteArray&)> p_filterFunc, int p_maxCheck) const
{
    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "RentWorkSpace NOT SUPPORT FOR SPANN");
    return nullptr;
}

template <typename T>
ErrorCode Index<T>::DebugSearchDiskIndex(QueryResult &p_query, int p_subInternalResultNum, int p_internalResultNum,
                                         SearchStats *p_stats, std::set<SizeType> *truth,
                                         std::map<SizeType, std::set<SizeType>> *found) const
{
    LayerReadLock topology(m_layerMutex, m_layerLockEnabled);
    if (m_extraSearchers.size() == 0)
        return ErrorCode::EmptyIndex;

    COMMON::QueryResultSet<T> newResults(*((COMMON::QueryResultSet<T> *)&p_query));
    for (int i = 0; i < newResults.GetResultNum(); ++i)
    {
        auto res = newResults.GetResult(i);
        if (res->VID == -1) break;

        auto global_VID = -1;
        if (m_topLocalToGlobalID.R() != 0)
            global_VID = static_cast<SizeType>(*(m_topLocalToGlobalID[res->VID]));
        if (truth && truth->count(global_VID))
            (*found)[res->VID].insert(global_VID);
        res->VID = global_VID;
    }
    newResults.Reverse();

    auto workSpace = m_workSpaceFactory->GetWorkSpace();
    if (!workSpace)
    {
        workSpace.reset(new ExtraWorkSpace());
        InitWorkSpace(workSpace.get(), false);
    }
    else
    {
        InitWorkSpace(workSpace.get(), true);
    }
    workSpace->m_deduper.clear();

    int partitions = (p_internalResultNum + p_subInternalResultNum - 1) / p_subInternalResultNum;
    float limitDist = p_query.GetResult(0)->Dist * m_options.m_maxDistRatio;
    for (int p = 0; p < partitions; p++)
    {
        int subInternalResultNum = min(p_subInternalResultNum, p_internalResultNum - p_subInternalResultNum * p);

        workSpace->m_postingIDs.clear();

        for (int i = p * p_subInternalResultNum; i < p * p_subInternalResultNum + subInternalResultNum; i++)
        {
            auto res = p_query.GetResult(i);
            if (res->VID == -1 || (limitDist > 0.1 && res->Dist > limitDist))
                break;

            if (m_topLocalToGlobalID.R() > 0) res->VID = static_cast<SizeType>(*(m_topLocalToGlobalID[res->VID]));
            workSpace->m_postingIDs.emplace_back(res->VID);
        }

        m_extraSearchers.back()->SearchIndex(workSpace.get(), newResults, p_stats, truth, found);
    }
    m_workSpaceFactory->ReturnWorkSpace(std::move(workSpace));
    newResults.SortResult();
    std::copy(newResults.GetResults(), newResults.GetResults() + newResults.GetResultNum(), p_query.GetResults());
    return ErrorCode::Success;
}
#pragma endregion

template <typename T> void Index<T>::SelectHeadAdjustOptions(int p_vectorCount)
{
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Begin Adjust Parameters...\n");

    if (m_options.m_headVectorCount != 0)
        m_options.m_ratio = m_options.m_headVectorCount * 1.0 / p_vectorCount;
    int headCnt = static_cast<int>(std::round(m_options.m_ratio * p_vectorCount));
    if (headCnt == 0)
    {
        for (double minCnt = 1; headCnt == 0; minCnt += 0.2)
        {
            m_options.m_ratio = minCnt / p_vectorCount;
            headCnt = static_cast<int>(std::round(m_options.m_ratio * p_vectorCount));
        }

        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
                     "Setting requires to select none vectors as head, adjusted it to %d vectors\n", headCnt);
    }

    if (m_options.m_iBKTKmeansK > headCnt)
    {
        m_options.m_iBKTKmeansK = headCnt;
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Setting of cluster number is less than head count, adjust it to %d\n",
                     headCnt);
    }

    if (m_options.m_selectThreshold == 0)
    {
        m_options.m_selectThreshold = min(p_vectorCount - 1, static_cast<int>(1 / m_options.m_ratio));
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Set SelectThreshold to %d\n", m_options.m_selectThreshold);
    }

    if (m_options.m_splitThreshold == 0)
    {
        m_options.m_splitThreshold = min(p_vectorCount - 1, static_cast<int>(m_options.m_selectThreshold * 2));
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Set SplitThreshold to %d\n", m_options.m_splitThreshold);
    }

    if (m_options.m_splitFactor == 0)
    {
        m_options.m_splitFactor = min(p_vectorCount - 1, static_cast<int>(std::round(1 / m_options.m_ratio) + 0.5));
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Set SplitFactor to %d\n", m_options.m_splitFactor);
    }
}

template <typename T>
int Index<T>::SelectHeadDynamicallyInternal(const std::shared_ptr<COMMON::BKTree> p_tree, int p_nodeID,
                                            const Options &p_opts, std::vector<int> &p_selected)
{
    typedef std::pair<int, int> CSPair;
    std::vector<CSPair> children;
    int childrenSize = 1;
    const auto &node = (*p_tree)[p_nodeID];
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
            std::sort(children.begin(), children.end(),
                      [](const CSPair &a, const CSPair &b) { return a.second > b.second; });

            size_t selectCnt = static_cast<size_t>(std::ceil(childrenSize * 1.0 / p_opts.m_splitFactor) + 0.5);
            // if (selectCnt > 1) selectCnt -= 1;
            for (size_t i = 0; i < selectCnt && i < children.size(); ++i)
            {
                p_selected.push_back((*p_tree)[children[i].first].centerid);
            }
        }

        return 0;
    }

    return childrenSize;
}

template <typename T>
void Index<T>::SelectHeadDynamically(const std::shared_ptr<COMMON::BKTree> p_tree, int p_vectorCount,
                                     std::vector<int> &p_selected)
{
    p_selected.clear();
    p_selected.reserve(p_vectorCount);

    if (static_cast<int>(std::round(m_options.m_ratio * p_vectorCount)) >= p_vectorCount)
    {
        for (int i = 0; i < p_vectorCount; ++i)
        {
            p_selected.push_back(i);
        }

        return;
    }
    Options opts = m_options;

    int selectThreshold = m_options.m_selectThreshold;
    int splitThreshold = m_options.m_splitThreshold;

    double minDiff = 100;
    for (int select = 2; select <= m_options.m_selectThreshold; ++select)
    {
        opts.m_selectThreshold = select;
        opts.m_splitThreshold = m_options.m_splitThreshold;

        int l = m_options.m_splitFactor;
        int r = m_options.m_splitThreshold;

        while (l < r - 1)
        {
            opts.m_splitThreshold = (l + r) / 2;
            p_selected.clear();

            SelectHeadDynamicallyInternal(p_tree, 0, opts, p_selected);
            std::sort(p_selected.begin(), p_selected.end());
            p_selected.erase(std::unique(p_selected.begin(), p_selected.end()), p_selected.end());

            double diff = static_cast<double>(p_selected.size()) / p_vectorCount - m_options.m_ratio;

            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Select Threshold: %d, Split Threshold: %d, diff: %.2lf%%.\n",
                         opts.m_selectThreshold, opts.m_splitThreshold, diff * 100.0);

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

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Final Select Threshold: %d, Split Threshold: %d.\n",
                 opts.m_selectThreshold, opts.m_splitThreshold);

    p_selected.clear();
    SelectHeadDynamicallyInternal(p_tree, 0, opts, p_selected);
    std::sort(p_selected.begin(), p_selected.end());
    p_selected.erase(std::unique(p_selected.begin(), p_selected.end()), p_selected.end());
}

template <typename T>
template <typename InternalDataType>
bool Index<T>::SelectHeadInternal(std::shared_ptr<Helper::VectorSetReader> &p_reader)
{
    std::shared_ptr<VectorSet> vectorset = p_reader->GetVectorSet();
    if (m_options.m_distCalcMethod == DistCalcMethod::Cosine && !p_reader->IsNormalized())
        vectorset->Normalize(m_options.m_iSelectHeadNumberOfThreads);
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Begin initial data (%d,%d)...\n", vectorset->Count(),
                 vectorset->Dimension());

    COMMON::Dataset<InternalDataType> data(vectorset->Count(), vectorset->Dimension(), vectorset->Count(),
                                           vectorset->Count() + 1, (InternalDataType *)vectorset->GetData());

    auto t1 = std::chrono::high_resolution_clock::now();
    SelectHeadAdjustOptions(data.R());
    std::vector<int> selected;
    if (data.R() == 1)
    {
        selected.push_back(0);
    }
    else if (Helper::StrUtils::StrEqualIgnoreCase(m_options.m_selectType.c_str(), "Random"))
    {
        std::mt19937 rg;
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start generating Random head.\n");
        selected.resize(data.R());
        for (int i = 0; i < data.R(); i++)
            selected[i] = i;
        std::shuffle(selected.begin(), selected.end(), rg);
        int headCnt = static_cast<int>(std::round(m_options.m_ratio * data.R()));
        selected.resize(headCnt);
    }
    else if (Helper::StrUtils::StrEqualIgnoreCase(m_options.m_selectType.c_str(), "BKT"))
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start generating BKT.\n");
        std::shared_ptr<COMMON::BKTree> bkt = std::make_shared<COMMON::BKTree>();
        bkt->m_iBKTKmeansK = m_options.m_iBKTKmeansK;
        bkt->m_iBKTLeafSize = m_options.m_iBKTLeafSize;
        bkt->m_iSamples = m_options.m_iSamples;
        bkt->m_iTreeNumber = m_options.m_iTreeNumber;
        bkt->m_fBalanceFactor = m_options.m_fBalanceFactor;
        bkt->m_parallelBuild = m_options.m_parallelBKTBuild;
        bkt->m_pQuantizer = m_pQuantizer;
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start invoking BuildTrees.\n");
        SPTAGLIB_LOG(
            Helper::LogLevel::LL_Info,
            "BKTKmeansK: %d, BKTLeafSize: %d, Samples: %d, BKTLambdaFactor:%f TreeNumber: %d, ThreadNum: %d, ParallelBuild: %s.\n",
            bkt->m_iBKTKmeansK, bkt->m_iBKTLeafSize, bkt->m_iSamples, bkt->m_fBalanceFactor, bkt->m_iTreeNumber,
            m_options.m_iSelectHeadNumberOfThreads, m_options.m_parallelBKTBuild ? "true" : "false");

        bkt->BuildTrees<InternalDataType>(data, m_options.m_distCalcMethod, m_options.m_iSelectHeadNumberOfThreads,
                                              nullptr, nullptr, true);

        auto t2 = std::chrono::high_resolution_clock::now();
        double elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "End invoking BuildTrees.\n");
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Invoking BuildTrees used time: %.2lf minutes (about %.2lf hours).\n",
                     elapsedSeconds / 60.0, elapsedSeconds / 3600.0);

        if (m_options.m_saveBKT)
        {
            std::stringstream bktFileNameBuilder;
            bktFileNameBuilder << m_options.m_vectorPath << ".bkt." << m_options.m_iBKTKmeansK << "_"
                               << m_options.m_iBKTLeafSize << "_" << m_options.m_iTreeNumber << "_"
                               << m_options.m_iSamples << "_" << static_cast<int>(m_options.m_distCalcMethod) << ".bin";
            bkt->SaveTrees(bktFileNameBuilder.str());
        }
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Finish generating BKT.\n");

        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Start selecting nodes...Select Head Dynamically...\n");
        SelectHeadDynamically(bkt, data.R(), selected);

        if (selected.empty())
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Can't select any vector as head with current settings\n");
            return false;
        }
    }

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Seleted Nodes: %u, about %.2lf%% of total.\n",
                 static_cast<unsigned int>(selected.size()), selected.size() * 100.0 / data.R());

    if (!m_options.m_noOutput)
    {
        std::sort(selected.begin(), selected.end());

        mkdir(m_options.m_indexDirectory.c_str());
        std::shared_ptr<Helper::DiskIO> output = SPTAG::f_createIO(), outputIDs = SPTAG::f_createIO();
        if (output == nullptr || outputIDs == nullptr ||
            !output->Initialize((m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile).c_str(),
                                std::ios::binary | std::ios::out) ||
            !outputIDs->Initialize((m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str(),
                                   std::ios::binary | std::ios::out))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to create output file:%s %s\n",
                         (m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile).c_str(),
                         (m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str());
            return false;
        }

        SizeType val = static_cast<SizeType>(selected.size());
        if (output->WriteBinary(sizeof(val), reinterpret_cast<char *>(&val)) != sizeof(val))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write output file!\n");
            return false;
        }
        if (outputIDs->WriteBinary(sizeof(val), reinterpret_cast<char *>(&val)) != sizeof(val))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write outputID file!\n");
            return false;
        }
        DimensionType dt = data.C();
        if (output->WriteBinary(sizeof(dt), reinterpret_cast<char *>(&dt)) != sizeof(dt))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write output file!\n");
            return false;
        }
        dt = 1;
        if (outputIDs->WriteBinary(sizeof(dt), reinterpret_cast<char *>(&dt)) != sizeof(dt))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write outputID file!\n");
            return false;
        }
        for (int i = 0; i < selected.size(); i++)
        {
            SizeType vid = selected[i];
            if (outputIDs->WriteBinary(sizeof(vid), reinterpret_cast<char *>(&vid)) != sizeof(vid))
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write output file!\n");
                return false;
            }

            if (output->WriteBinary(sizeof(InternalDataType) * data.C(), (char *)(data[vid])) !=
                sizeof(InternalDataType) * data.C())
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to write output file!\n");
                return false;
            }
        }
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(t3 - t1).count();
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Total used time: %.2lf minutes (about %.2lf hours).\n",
                 elapsedSeconds / 60.0, elapsedSeconds / 3600.0);
    return true;
}

template <typename T> ErrorCode Index<T>::BuildIndexInternalLayer(std::shared_ptr<Helper::VectorSetReader> &p_reader)
{
    int currentLayer = static_cast<int>(m_extraSearchers.size());
    COMMON::Dataset<SizeType> localToGlobalID;
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loading headIDFile for layer %d...\n", currentLayer - 1);
        std::string localToGlobalIDPath = (currentLayer == 0)? m_options.m_globalIDPath : m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile;
        std::shared_ptr<Helper::DiskIO> ptr = SPTAG::f_createIO();
        if (ptr == nullptr ||
            !ptr->Initialize(localToGlobalIDPath.c_str(),
                                std::ios::binary | std::ios::in))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "No headIDFile file:%s\n",
                            localToGlobalIDPath.c_str());
        }
        else {
            localToGlobalID.Load(ptr, this->m_iDataBlockSize, this->m_iDataCapacity);
            SizeType vectorCount = p_reader->GetVectorSet()->Count();
            if (localToGlobalID.R() != vectorCount)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "HeadIDFile count %lld doesn't match head vector file count %lld!\n", (int64_t)localToGlobalID.R(), (int64_t)vectorCount);
                localToGlobalID.SetR(0);
            }
        }
    }

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Begin Select Head...\n");
    auto t1 = std::chrono::high_resolution_clock::now();
    if (m_options.m_selectHead && m_topIndex == nullptr)
    {
        bool success = false;
        if (m_pQuantizer)
        {
            success = SelectHeadInternal<std::uint8_t>(p_reader);
        }
        else
        {
            success = SelectHeadInternal<T>(p_reader);
        }
        if (!success)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SelectHead Failed!\n");
            return ErrorCode::Fail;
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double selectHeadTime = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "select head time: %.2lfs\n", selectHeadTime);

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Begin Build Head...\n");
    if (m_options.m_buildHead && m_topIndex == nullptr)
    {
        auto valueType = m_pQuantizer ? SPTAG::VectorValueType::UInt8 : m_options.m_valueType;
        auto dims = m_pQuantizer ? m_pQuantizer->GetNumSubvectors() : m_options.m_dim;

        m_topIndex = SPTAG::VectorIndex::CreateInstance(m_options.m_indexAlgoType, valueType);
        m_topIndex->SetParameter("DistCalcMethod", SPTAG::Helper::Convert::ConvertToString(m_options.m_distCalcMethod));
        m_topIndex->SetParameter("ParallelBKTBuild", m_options.m_parallelBKTBuild ? "true" : "false");
        m_topIndex->SetQuantizer(m_pQuantizer);

        for (const auto &iter : m_topParameters)
        {
            m_topIndex->SetParameter(iter.first.c_str(), iter.second.c_str());
        }

        std::shared_ptr<Helper::ReaderOptions> vectorOptions(
            new Helper::ReaderOptions(valueType, dims, VectorFileType::DEFAULT));
        auto vectorReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
        if (ErrorCode::Success !=
            vectorReader->LoadFile(m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read head vector file.\n");
            return ErrorCode::Fail;
        }
        {
            auto headvectorset = vectorReader->GetVectorSet();
            if (m_topIndex->BuildIndex(headvectorset, nullptr, false, true, true) != ErrorCode::Success)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to build head index.\n");
                return ErrorCode::Fail;
            }
            if (!m_options.m_quantizerFilePath.empty())
                m_topIndex->SetQuantizerFileName(
                    m_options.m_quantizerFilePath.substr(m_options.m_quantizerFilePath.find_last_of("/\\") + 1));
            if (m_topIndex->SaveIndex(m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder) !=
                ErrorCode::Success)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to save head index.\n");
                return ErrorCode::Fail;
            }
        }
        m_topIndex.reset();
        if (LoadIndex(m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder, m_topIndex) !=
            ErrorCode::Success)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot load head index from %s!\n",
                         (m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder).c_str());
        }
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double buildHeadTime = std::chrono::duration_cast<std::chrono::seconds>(t3 - t2).count();
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "select head time: %.2lfs build head time: %.2lfs\n", selectHeadTime,
                 buildHeadTime);

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Begin Build SSDIndex...\n");
    if (m_options.m_enableSSD)
    {
        if (m_topIndex == nullptr && LoadIndex(m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder,
                                            m_topIndex) != ErrorCode::Success)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot load head index from %s!\n",
                         (m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder).c_str());
            return ErrorCode::Fail;
        }
        m_topIndex->SetQuantizer(m_pQuantizer);

        if (!CheckHeadIndexType())
            return ErrorCode::Fail;

        m_topIndex->SetParameter("NumberOfThreads", std::to_string(m_options.m_iSSDNumberOfThreads));
        m_topIndex->SetParameter("MaxCheck", std::to_string(m_options.m_maxCheck));
        m_topIndex->SetParameter("HashTableExponent", std::to_string(m_options.m_hashExp));

        m_topIndex->UpdateIndex();

        if (m_options.m_storage == Storage::STATIC)
        {
            m_extraSearchers.emplace_back(std::make_shared<ExtraStaticSearcher<T>>(m_extraSearchers.size(), this));
        }
        else
        {
            m_extraSearchers.emplace_back(std::make_shared<ExtraDynamicSearcher<T>>(m_options, m_extraSearchers.size(), this, m_db));
        }

        {
            std::shared_ptr<Helper::DiskIO> ptr = SPTAG::f_createIO();
            if (ptr == nullptr ||
                !ptr->Initialize((m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str(),
                                 std::ios::binary | std::ios::in))
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to open headIDFile file:%s\n",
                             (m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile).c_str());
                return ErrorCode::Fail;
            }
            m_topLocalToGlobalID.Load(ptr, m_topIndex->m_iDataBlockSize, m_topIndex->m_iDataCapacity);
            m_topGlobalToLocalID.clear();
            for (int i = 0; i < m_topIndex->GetNumSamples(); i++)
            {
                SizeType globalID = *(m_topLocalToGlobalID[i]);
                m_topGlobalToLocalID[globalID] = i;
            }
        }

        if (m_options.m_buildSsdIndex)
        {
            if (m_options.m_storage != Storage::STATIC && !m_extraSearchers.back()->Available())
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Extrasearcher is not available and failed to initialize.\n");
                return ErrorCode::Fail;
            }
            if (!m_extraSearchers.back()->BuildIndex(p_reader, m_topIndex, m_options, m_topLocalToGlobalID, m_topGlobalToLocalID, localToGlobalID))
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "BuildSSDIndex Failed!\n");
                return ErrorCode::Fail;
            }
        }

        if (!m_extraSearchers.back()->LoadIndex(m_options))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot Load SSDIndex!\n");
            if (m_options.m_buildSsdIndex)
            {
                return ErrorCode::Fail;
            }
            else
            {
                m_extraSearchers.pop_back();
            }
        }
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    double buildSSDTime = std::chrono::duration_cast<std::chrono::seconds>(t4 - t3).count();
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "select head time: %.2lfs build head time: %.2lfs build ssd time: %.2lfs\n",
                 selectHeadTime, buildHeadTime, buildSSDTime);

    return ErrorCode::Success;
}

template <typename T> ErrorCode Index<T>::BuildIndexInternal(std::shared_ptr<Helper::VectorSetReader> &vectorReader, std::shared_ptr<Helper::ReaderOptions> &vectorOptions)
{
    auto validation = ValidateDynamicLayers();
    if (validation != ErrorCode::Success) return validation;
    if (!(m_options.m_indexDirectory.empty()) && !(direxists(m_options.m_indexDirectory.c_str())))
    {
        mkdir(m_options.m_indexDirectory.c_str());
    }
    if (!(m_options.m_persistentBufferPath.empty()) && !(direxists(m_options.m_persistentBufferPath.c_str())))
    {
        mkdir(m_options.m_persistentBufferPath.c_str());
    }

    if (m_freeWorkSpaceIds == nullptr) {
        m_freeWorkSpaceIds.reset(new Helper::Concurrent::ConcurrentQueue<int>());
        int maxIOThreads = (m_options.m_storage == Storage::STATIC) ? max(m_options.m_searchThreadNum, m_options.m_iSSDNumberOfThreads) : max(m_options.m_ioThreads, (2 * max(m_options.m_searchThreadNum, m_options.m_iSSDNumberOfThreads) +
                                    (m_options.m_layers + 1) * (m_options.m_insertThreadNum + m_options.m_reassignThreadNum + m_options.m_appendThreadNum)));
        for (int i = 0; i < maxIOThreads; i++) {
            m_freeWorkSpaceIds->push(i);
        }
        m_workspaceCount = maxIOThreads;
    }

    if (m_db == nullptr && m_options.m_shareDB) PrepareDB(m_db);

    int resumeLayer = m_options.m_resumeLayer;

    if (resumeLayer >= 0) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Resuming build from layer %d (loading layers 0..%d from existing data)\n",
                     resumeLayer, resumeLayer - 1);

        // Save original file path options
        std::string origHeadIDFile = m_options.m_headIDFile;
        std::string origHeadVectorFile = m_options.m_headVectorFile;
        std::string origHeadIndexFolder = m_options.m_headIndexFolder;

        auto setLayerPaths = [&](int layer) {
            if (m_options.m_layers > 1 && layer < m_options.m_layers - 1) {
                std::string suffix = "_layer" + std::to_string(layer);
                m_options.m_headIDFile = origHeadIDFile + suffix;
                m_options.m_headVectorFile = origHeadVectorFile + suffix;
                m_options.m_headIndexFolder = origHeadIndexFolder + suffix;
            } else {
                m_options.m_headIDFile = origHeadIDFile;
                m_options.m_headVectorFile = origHeadVectorFile;
                m_options.m_headIndexFolder = origHeadIndexFolder;
            }
        };

        // For resume, we need the head index that corresponds to the layer ABOVE the resume layer.
        // For 2-layer: resumeLayer=1 means load Layer 1's head index (the last layer, default paths).
        setLayerPaths(resumeLayer);
        std::string headIndexPath = m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder;
        std::string headIDPath = m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile;

        // Load head index from disk
        if (LoadIndex(headIndexPath, m_topIndex) != ErrorCode::Success) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot load head index for resume from %s\n", headIndexPath.c_str());
            m_options.m_headIDFile = origHeadIDFile;
            m_options.m_headVectorFile = origHeadVectorFile;
            m_options.m_headIndexFolder = origHeadIndexFolder;
            return ErrorCode::Fail;
        }
        m_topIndex->SetQuantizer(m_pQuantizer);
        m_topIndex->SetParameter("NumberOfThreads", std::to_string(m_options.m_iSSDNumberOfThreads));
        m_topIndex->SetParameter("MaxCheck", std::to_string(m_options.m_maxCheck));
        m_topIndex->SetParameter("HashTableExponent", std::to_string(m_options.m_hashExp));
        m_topIndex->UpdateIndex();

        // Load head ID mapping
        {
            std::shared_ptr<Helper::DiskIO> ptr = SPTAG::f_createIO();
            if (ptr != nullptr && ptr->Initialize(headIDPath.c_str(),
                    std::ios::binary | std::ios::in)) {
                m_topLocalToGlobalID.Load(ptr, m_topIndex->m_iDataBlockSize, m_topIndex->m_iDataCapacity);
                m_topGlobalToLocalID.clear();
                for (int i = 0; i < m_topIndex->GetNumSamples(); i++) {
                    SizeType globalID = *(m_topLocalToGlobalID[i]);
                    m_topGlobalToLocalID[globalID] = i;
                }
            }
        }

        // Load existing layers below resumeLayer
        for (int i = 0; i < resumeLayer; i++) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Loading existing data for layer %d\n", i);
            if (m_options.m_storage == Storage::STATIC) {
                m_extraSearchers.push_back(std::make_shared<ExtraStaticSearcher<T>>(i, this));
            } else {
                m_extraSearchers.push_back(std::make_shared<ExtraDynamicSearcher<T>>(m_options, i, this, m_db));
            }
            if (!m_extraSearchers.back()->LoadIndex(m_options)) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to load existing index for layer %d\n", i);
                return ErrorCode::Fail;
            }
            SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Layer %d loaded successfully\n", i);
        }

        // For the resume layer and above, build SSD only (skip SelectHead + BuildHead)
        bool origSelectHead = m_options.m_selectHead;
        bool origBuildHead = m_options.m_buildHead;
        m_options.m_selectHead = false;
        m_options.m_buildHead = false;

        for (int layer = resumeLayer; layer < m_options.m_layers; layer++) {
            setLayerPaths(layer);

            if (layer != resumeLayer) m_topIndex = nullptr;
            if (layer == 0) {
                // Layer 0 uses the original base vectors
                auto ret = BuildIndexInternalLayer(vectorReader);
                if (ret != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to resume build for layer 0.\n");
                    m_options.m_selectHead = origSelectHead;
                    m_options.m_buildHead = origBuildHead;
                    m_options.m_headIDFile = origHeadIDFile;
                    m_options.m_headVectorFile = origHeadVectorFile;
                    m_options.m_headIndexFolder = origHeadIndexFolder;
                    return ret;
                }
            } else {
                // Layer N uses the head vectors from the previous layer as input
                setLayerPaths(layer - 1);
                std::string prevHeadVectors = m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile;
                setLayerPaths(layer);

                vectorReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
                if (ErrorCode::Success != vectorReader->LoadFile(prevHeadVectors)) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read headvector file for layer %d from %s.\n", layer, prevHeadVectors.c_str());
                    m_options.m_selectHead = origSelectHead;
                    m_options.m_buildHead = origBuildHead;
                    m_options.m_headIDFile = origHeadIDFile;
                    m_options.m_headVectorFile = origHeadVectorFile;
                    m_options.m_headIndexFolder = origHeadIndexFolder;
                    return ErrorCode::Fail;
                }
                auto ret = BuildIndexInternalLayer(vectorReader);
                if (ret != ErrorCode::Success) {
                    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to resume build for layer %d.\n", layer);
                    m_options.m_selectHead = origSelectHead;
                    m_options.m_buildHead = origBuildHead;
                    m_options.m_headIDFile = origHeadIDFile;
                    m_options.m_headVectorFile = origHeadVectorFile;
                    m_options.m_headIndexFolder = origHeadIndexFolder;
                    return ret;
                }
            }
        }

        m_options.m_selectHead = origSelectHead;
        m_options.m_buildHead = origBuildHead;
        m_options.m_headIDFile = origHeadIDFile;
        m_options.m_headVectorFile = origHeadVectorFile;
        m_options.m_headIndexFolder = origHeadIndexFolder;
    } else {
        // Normal build path
        // Save original file path options for per-layer suffixing
        std::string origHeadIDFile = m_options.m_headIDFile;
        std::string origHeadVectorFile = m_options.m_headVectorFile;
        std::string origHeadIndexFolder = m_options.m_headIndexFolder;

        auto setLayerPaths = [&](int layer) {
            if (m_options.m_layers > 1 && layer < m_options.m_layers - 1) {
                std::string suffix = "_layer" + std::to_string(layer);
                m_options.m_headIDFile = origHeadIDFile + suffix;
                m_options.m_headVectorFile = origHeadVectorFile + suffix;
                m_options.m_headIndexFolder = origHeadIndexFolder + suffix;
            } else {
                m_options.m_headIDFile = origHeadIDFile;
                m_options.m_headVectorFile = origHeadVectorFile;
                m_options.m_headIndexFolder = origHeadIndexFolder;
            }
        };

        // Layer 0
        setLayerPaths(0);
        m_topIndex = nullptr;
        auto ret = BuildIndexInternalLayer(vectorReader);
        if (ret != ErrorCode::Success)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to build index layer 0.\n");
            m_options.m_headIDFile = origHeadIDFile;
            m_options.m_headVectorFile = origHeadVectorFile;
            m_options.m_headIndexFolder = origHeadIndexFolder;
            return ret;
        }
        for (int layer = 1; layer < m_options.m_layers; layer++)
        {
            // Load previous layer's head vectors as input for this layer
            std::string prevHeadVectors = m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile;
            std::string prevHeadIDFile = m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile;
            setLayerPaths(layer);

            // Copy previous layer's headIDFile to current layer's path so that
            // BuildIndexInternalLayer can load it as localToGlobalID (mapping from
            // the previous layer's local head IDs to original global vector IDs).
            // SelectHead will overwrite this file later with the current layer's head IDs.
            std::string curHeadIDFile = m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile;
            std::error_code ec;
            std::filesystem::copy_file(prevHeadIDFile, curHeadIDFile, std::filesystem::copy_options::overwrite_existing, ec);
            if (ec) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Warning, "Could not copy previous layer headIDFile %s to %s: %s\n",
                             prevHeadIDFile.c_str(), curHeadIDFile.c_str(), ec.message().c_str());
            } else {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Copied localToGlobalID from %s to %s for layer %d\n",
                             prevHeadIDFile.c_str(), curHeadIDFile.c_str(), layer);
            }

            vectorReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
            if (ErrorCode::Success != vectorReader->LoadFile(prevHeadVectors))
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read headvector file for layer %d from %s.\n", layer, prevHeadVectors.c_str());
                m_options.m_headIDFile = origHeadIDFile;
                m_options.m_headVectorFile = origHeadVectorFile;
                m_options.m_headIndexFolder = origHeadIndexFolder;
                return ErrorCode::Fail;
            }
            m_topIndex = nullptr;
            auto ret = BuildIndexInternalLayer(vectorReader);
            if (ret != ErrorCode::Success)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to build index layer %d.\n", layer);
                m_options.m_headIDFile = origHeadIDFile;
                m_options.m_headVectorFile = origHeadVectorFile;
                m_options.m_headIndexFolder = origHeadIndexFolder;
                return ret;
            }
        }

        // Restore original options
        m_options.m_headIDFile = origHeadIDFile;
        m_options.m_headVectorFile = origHeadVectorFile;
        m_options.m_headIndexFolder = origHeadIndexFolder;
    } // end of normal vs resume path

    // Clean up last layer's head vector file (not needed at runtime)
    if (fileexists((m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile).c_str())) 
    {
        if (remove((m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile).c_str()) != 0)
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to delete head vector file: %s\n",
                         (m_options.m_indexDirectory + FolderSep + m_options.m_headVectorFile).c_str());
        }
    }
    m_bReady = true;
    StartLayerResizeWorker();
    RequestLayerResize();
    return ErrorCode::Success;
}

template <typename T> ErrorCode Index<T>::BuildIndex(bool p_normalized)
{
    SPTAG::VectorValueType valueType = m_pQuantizer ? SPTAG::VectorValueType::UInt8 : m_options.m_valueType;
    SizeType dim = m_pQuantizer ? m_pQuantizer->GetNumSubvectors() : m_options.m_dim;
    std::shared_ptr<Helper::ReaderOptions> vectorOptions(
        new Helper::ReaderOptions(valueType, dim, m_options.m_vectorType, m_options.m_vectorDelimiter,
                                  m_options.m_iSSDNumberOfThreads, p_normalized));
    auto vectorReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
    if (m_options.m_vectorPath.empty())
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Vector file is empty. Skipping loading.\n");
    }
    else
    {
        if (ErrorCode::Success != vectorReader->LoadFile(m_options.m_vectorPath))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read vector file.\n");
            return ErrorCode::Fail;
        }
        m_options.m_vectorSize = vectorReader->GetVectorSet()->Count();
    }

    return BuildIndexInternal(vectorReader, vectorOptions);
}

template <typename T>
ErrorCode Index<T>::BuildIndex(const void *p_data, SizeType p_vectorNum, DimensionType p_dimension, bool p_normalized,
                               bool p_shareOwnership)
{
    if (p_data == nullptr || p_vectorNum == 0 || p_dimension == 0)
        return ErrorCode::EmptyData;

    std::shared_ptr<VectorSet> vectorSet;
    if (p_shareOwnership)
    {
        vectorSet.reset(
            new BasicVectorSet(ByteArray((std::uint8_t *)p_data, sizeof(T) * p_vectorNum * p_dimension, false),
                               GetEnumValueType<T>(), p_dimension, p_vectorNum));
    }
    else
    {
        ByteArray arr = ByteArray::Alloc(sizeof(T) * p_vectorNum * p_dimension);
        memcpy(arr.Data(), p_data, sizeof(T) * p_vectorNum * p_dimension);
        vectorSet.reset(new BasicVectorSet(arr, GetEnumValueType<T>(), p_dimension, p_vectorNum));
    }

    if (m_options.m_distCalcMethod == DistCalcMethod::Cosine && !p_normalized)
    {
        vectorSet->Normalize(m_options.m_iSSDNumberOfThreads);
    }
    SPTAG::VectorValueType valueType = m_pQuantizer ? SPTAG::VectorValueType::UInt8 : m_options.m_valueType;
    std::shared_ptr<Helper::ReaderOptions> vectorOptions(
        new Helper::ReaderOptions(valueType, p_dimension, VectorFileType::DEFAULT, m_options.m_vectorDelimiter,
                                  m_options.m_iSSDNumberOfThreads, true));
    
    std::shared_ptr<Helper::VectorSetReader> vectorReader(new Helper::MemoryVectorReader(
        vectorOptions,
        vectorSet));

    m_options.m_valueType = GetEnumValueType<T>();
    m_options.m_dim = p_dimension;
    m_options.m_vectorSize = p_vectorNum;

    return BuildIndexInternal(vectorReader, vectorOptions);
}

template <typename T> ErrorCode Index<T>::UpdateIndex()
{
    LayerReadLock mutations(m_checkPointLock, m_layerLockEnabled);
    LayerReadLock topology(m_layerMutex, m_layerLockEnabled);
    m_topIndex->SetParameter("NumberOfThreads", std::to_string(m_options.m_iSSDNumberOfThreads));
    // m_topIndex->SetParameter("MaxCheck", std::to_string(m_options.m_maxCheck));
    // m_topIndex->SetParameter("HashTableExponent", std::to_string(m_options.m_hashExp));
    m_topIndex->UpdateIndex();
    return ErrorCode::Success;
}

template <typename T>
ErrorCode Index<T>::RefineIndex(const std::vector<std::shared_ptr<Helper::DiskIO>> &p_indexStreams,
                                IAbortOperation *p_abort, std::vector<SizeType> *p_mapping)
{
    if (m_topIndex == nullptr)
        return ErrorCode::EmptyIndex;

    while (!PostingMaintenanceFinished())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    LayerReadLock mutations(m_checkPointLock, m_layerLockEnabled);
    LayerReadLock topology(m_layerMutex, m_layerLockEnabled);
    std::lock_guard<std::mutex> lock(m_dataAddLock);
    std::unique_lock<std::shared_timed_mutex> uniquelock(m_dataDeleteLock);
    ErrorCode ret;
    /*
    for (int layer = 0; layer < m_extraSearchers.size(); layer++)
    {
        if ((ret = m_extraSearchers[layer]->RefineIndex()) != ErrorCode::Success)
            return ret;
    }
    */

    std::vector<SizeType> headOldtoNew;

    if ((ret = m_topIndex->RefineIndex(p_indexStreams, nullptr, &headOldtoNew)) != ErrorCode::Success)
        return ret;

    COMMON::Dataset<SizeType> new_topLocalToGlobalID(m_topIndex->GetNumSamples() - m_topIndex->GetNumDeleted(), 1,
                                                          m_topIndex->m_iDataBlockSize, m_topIndex->m_iDataCapacity);
    for (int i = 0; i < m_topLocalToGlobalID.R(); i++)
    {
        if (m_topIndex->ContainSample(i))
        {
            *(new_topLocalToGlobalID[headOldtoNew[i]]) = *(m_topLocalToGlobalID[i]);
        }
    }
    new_topLocalToGlobalID.Save(p_indexStreams[m_topIndex->GetIndexFiles()->size()]);
    
    if ((ret = VectorIndex::LoadIndex(m_options.m_indexDirectory + FolderSep + m_options.m_headIndexFolder,
                                                m_topIndex)) != ErrorCode::Success)
        return ret;

    if ((ret = m_topLocalToGlobalID.Load(
                             m_options.m_indexDirectory + FolderSep + m_options.m_headIDFile, m_topIndex->m_iDataBlockSize,
                             m_topIndex->m_iDataCapacity)) != ErrorCode::Success)
                        return ret;
    m_topGlobalToLocalID.clear();
    for (int i = 0; i < m_topLocalToGlobalID.R(); i++)
    {    
        SizeType globalID = *(m_topLocalToGlobalID[i]);
        m_topGlobalToLocalID[globalID] = i;
    }

    for (int i = 0; i < p_indexStreams.size(); i++)
    {
        p_indexStreams[i]->ShutDown();
    }
    return ret;
}

template <typename T> ErrorCode Index<T>::SetParameter(const char *p_param, const char *p_value, const char *p_section)
{
    LayerReadLock mutations(m_checkPointLock, m_layerLockEnabled);
    std::unique_lock<LayerMutex> topology(m_layerMutex, std::defer_lock);
    if (m_layerLockEnabled && !LayerReadLock::IsWriteHeld(m_layerMutex)) topology.lock();
    LayerReadLock exclusiveContext(m_layerMutex, m_layerLockEnabled, true, true);
    if (m_bReady && (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "EnableDynamicLayers") ||
        SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "LayerStoragePrefixes") ||
        (m_layerLockEnabled && (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "MaxID") ||
          SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "Layers") ||
          SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "Storage") ||
          SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "TiKVKeyPrefix") ||
          SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "DistributedVersionMap")))) &&
        GetParameter(p_param, p_section) != p_value) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "%s must be configured before building/loading the index.\n", p_param);
        return ErrorCode::Fail;
    }
    if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_section, "BuildHead") &&
        !SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "isExecute"))
    {
        if (m_topIndex != nullptr)
            return m_topIndex->SetParameter(p_param, p_value);
        else
            m_topParameters[p_param] = p_value;
    }
    else
    {
        auto status = m_options.SetParameter(p_section, p_param, p_value);
        if (status != ErrorCode::Success) return status;
    }
    if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "DistCalcMethod"))
    {
        if (m_pQuantizer)
        {
            m_fComputeDistance = m_pQuantizer->DistanceCalcSelector<T>(m_options.m_distCalcMethod);
            m_iBaseSquare = (m_options.m_distCalcMethod == DistCalcMethod::Cosine)
                                ? m_pQuantizer->GetBase() * m_pQuantizer->GetBase()
                                : 1;
        }
        else
        {
            m_fComputeDistance = COMMON::DistanceCalcSelector<T>(m_options.m_distCalcMethod);
            m_iBaseSquare = (m_options.m_distCalcMethod == DistCalcMethod::Cosine)
                                ? COMMON::Utils::GetBase<T>() * COMMON::Utils::GetBase<T>()
                                : 1;
        }
    }
    if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "EnableADC") && m_pQuantizer)
    {
        m_pQuantizer->SetEnableADC(m_options.m_enableADC);
    }
    return ErrorCode::Success;
}

template <typename T> std::string Index<T>::GetParameter(const char *p_param, const char *p_section) const
{
    LayerReadLock topology(m_layerMutex, m_layerLockEnabled);
    if (SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_section, "BuildHead") &&
        !SPTAG::Helper::StrUtils::StrEqualIgnoreCase(p_param, "isExecute"))
    {
        if (m_topIndex != nullptr)
            return m_topIndex->GetParameter(p_param);
        else
        {
            auto iter = m_topParameters.find(p_param);
            if (iter != m_topParameters.end())
                return iter->second;
            return "Undefined!";
        }
    }
    else
    {
        return m_options.GetParameter(p_section, p_param);
    }
}

// Add insert entry to persistent buffer
template <typename T>
ErrorCode Index<T>::AddIndex(const void *p_data, SizeType p_vectorNum, DimensionType p_dimension,
                             std::shared_ptr<MetadataSet> p_metadataSet, bool p_withMetaIndex, bool p_normalized, SizeType* VID)
{
    LayerReadLock mutationGate(m_checkPointLock, m_layerLockEnabled);
    LayerReadLock topology(m_layerMutex, m_layerLockEnabled);
    if ((m_options.m_storage == Storage::STATIC) || m_extraSearchers.size() == 0)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Only Support KV Extra Update\n");
        return ErrorCode::Fail;
    }
    if (p_data == nullptr || p_vectorNum == 0 || p_dimension == 0)
        return ErrorCode::EmptyData;
    if (p_dimension != GetFeatureDim())
        return ErrorCode::DimensionSizeMismatch;

    LayerReadLock lock(m_checkPointLock, true);

    SizeType begin, end;
    std::uint64_t addIndexLockWaitUs = 0;
    std::uint64_t addIndexLockHoldUs = 0;
    std::uint64_t addIndexAddCapacityLayer0Us = 0;
    std::uint64_t addIndexAddCapacityUpperUs = 0;
    std::uint64_t addIndexMetadataUs = 0;
    {
        auto lockWaitBegin = std::chrono::high_resolution_clock::now();
        m_dataAddLock.lock();
        auto lockWaitEnd = std::chrono::high_resolution_clock::now();
        std::lock_guard<std::mutex> dataAddLock(m_dataAddLock, std::adopt_lock);
        addIndexLockWaitUs = std::chrono::duration_cast<std::chrono::microseconds>(lockWaitEnd - lockWaitBegin).count();
        auto lockHoldBegin = lockWaitEnd;

        begin = m_options.m_vectorSize;
        if (m_layerLockEnabled && (p_vectorNum < 0 || p_vectorNum > m_options.m_maxID ||
            begin > m_options.m_maxID - p_vectorNum)) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Dynamic insert exceeds the configured global VID namespace.\n");
            return ErrorCode::Key_OverFlow;
        }
        end = begin + p_vectorNum;

        if (begin == 0)
        {
            return ErrorCode::EmptyIndex;
        }

        auto metadataBegin = std::chrono::high_resolution_clock::now();
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
        auto metadataEnd = std::chrono::high_resolution_clock::now();
        addIndexMetadataUs = std::chrono::duration_cast<std::chrono::microseconds>(metadataEnd - metadataBegin).count();
        m_options.m_vectorSize = end;
        auto lockHoldEnd = std::chrono::high_resolution_clock::now();
        addIndexLockHoldUs = std::chrono::duration_cast<std::chrono::microseconds>(lockHoldEnd - lockHoldBegin).count();
    }

    auto updateMax = [](std::atomic_uint64_t& target, std::uint64_t value) {
        std::uint64_t current = target.load(std::memory_order_relaxed);
        while (current < value && !target.compare_exchange_weak(current, value, std::memory_order_relaxed)) {}
    };
    updateMax(m_addIndexLockWaitMaxUs, addIndexLockWaitUs);
    updateMax(m_addIndexLockHoldMaxUs, addIndexLockHoldUs);

    m_addIndexDiagVectors.fetch_add(p_vectorNum, std::memory_order_relaxed);
    m_addIndexLockWaitUs.fetch_add(addIndexLockWaitUs, std::memory_order_relaxed);
    m_addIndexLockHoldUs.fetch_add(addIndexLockHoldUs, std::memory_order_relaxed);
    m_addIndexAddCapacityLayer0Us.fetch_add(addIndexAddCapacityLayer0Us, std::memory_order_relaxed);
    m_addIndexAddCapacityUpperUs.fetch_add(addIndexAddCapacityUpperUs, std::memory_order_relaxed);
    m_addIndexMetadataUs.fetch_add(addIndexMetadataUs, std::memory_order_relaxed);
    std::uint64_t addIndexCalls = m_addIndexDiagCalls.fetch_add(1, std::memory_order_relaxed) + 1;
    if (addIndexCalls <= 16 || addIndexCalls % 50000 == 0)
    {
        double calls = static_cast<double>(addIndexCalls);
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info,
            "[DIAG-ADDINDEX] calls=%llu vectors=%llu lockWaitAvg=%.2fus lockWaitMax=%lluus lockHoldAvg=%.2fus lockHoldMax=%lluus addCapL0Avg=%.2fus addCapUpperAvg=%.2fus metadataAvg=%.2fus\n",
            static_cast<unsigned long long>(addIndexCalls),
            static_cast<unsigned long long>(m_addIndexDiagVectors.load(std::memory_order_relaxed)),
            m_addIndexLockWaitUs.load(std::memory_order_relaxed) / calls,
            static_cast<unsigned long long>(m_addIndexLockWaitMaxUs.load(std::memory_order_relaxed)),
            m_addIndexLockHoldUs.load(std::memory_order_relaxed) / calls,
            static_cast<unsigned long long>(m_addIndexLockHoldMaxUs.load(std::memory_order_relaxed)),
            m_addIndexAddCapacityLayer0Us.load(std::memory_order_relaxed) / calls,
            m_addIndexAddCapacityUpperUs.load(std::memory_order_relaxed) / calls,
            m_addIndexMetadataUs.load(std::memory_order_relaxed) / calls);
    }

    if (VID != nullptr)
    {
        for (int i = 0; i < p_vectorNum; i++) VID[i] = begin + i;
    }

    std::shared_ptr<VectorSet> vectorSet;
    if (m_options.m_distCalcMethod == DistCalcMethod::Cosine && !p_normalized)
    {
        ByteArray arr = ByteArray::Alloc(sizeof(T) * p_vectorNum * p_dimension);
        memcpy(arr.Data(), p_data, sizeof(T) * p_vectorNum * p_dimension);
        vectorSet.reset(new BasicVectorSet(arr, GetEnumValueType<T>(), p_dimension, p_vectorNum));
        int base = COMMON::Utils::GetBase<T>();
        for (SizeType i = 0; i < p_vectorNum; i++)
        {
            COMMON::Utils::Normalize((T *)(vectorSet->GetVector(i)), p_dimension, base);
        }
    }
    else
    {
        vectorSet.reset(
            new BasicVectorSet(ByteArray((std::uint8_t *)p_data, sizeof(T) * p_vectorNum * p_dimension, false),
                               GetEnumValueType<T>(), p_dimension, p_vectorNum));
    }

    auto workSpace = m_workSpaceFactory->GetWorkSpace();
    if (!workSpace)
    {
        workSpace.reset(new ExtraWorkSpace());
        InitWorkSpace(workSpace.get(), false);
    }
    else
    {
        InitWorkSpace(workSpace.get(), true);
    }
    workSpace->m_deduper.clear();
    workSpace->m_postingIDs.clear();
    return m_extraSearchers[0]->AddIndex(workSpace.get(), vectorSet, begin);
}

template <typename T>
ErrorCode Index<T>::Check()
{
    std::atomic<ErrorCode> ret = ErrorCode::Success;
    while (!AllFinished())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    LayerReadLock mutations(m_checkPointLock, m_layerLockEnabled);
    LayerReadLock topology(m_layerMutex, m_layerLockEnabled);
    //if ((ret = m_topIndex->Check()) != ErrorCode::Success)
    //    return ret;

    std::vector<std::thread> mythreads;
    mythreads.reserve(m_options.m_iSSDNumberOfThreads);
    for (int layer = m_extraSearchers.size() - 1; layer >= 0; layer--)
    {
        std::vector<SizeType> globalIDs;
        GetHeadIndexMapping(layer + 1, globalIDs);
        SizeType currentPostingNum = globalIDs.size();
        std::atomic_size_t sent(0);
        std::vector<std::uint8_t> checked(m_extraSearchers[layer]->GetNumBlocks(), false);
        for (int tid = 0; tid < m_options.m_iSSDNumberOfThreads; tid++)
        {
            mythreads.emplace_back([&, tid]() {
                auto workSpace = m_workSpaceFactory->GetWorkSpace();
                if (!workSpace)
                {
                    workSpace.reset(new ExtraWorkSpace());
                    InitWorkSpace(workSpace.get(), false);
                }
                else
                {
                    InitWorkSpace(workSpace.get(), true);
                }
                size_t i = 0;
                while (true)
                {
                    i = sent.fetch_add(1);
                    if (i < currentPostingNum)
                    {
                        if (m_extraSearchers[layer]->CheckPosting(globalIDs[i], &checked, workSpace.get()) != ErrorCode::Success)
                        {
                            ret = ErrorCode::Fail;
                            return;
                        }
                    }
                    else
                    {
                        return;
                    }
                }
                m_workSpaceFactory->ReturnWorkSpace(std::move(workSpace));
            });
        }
        for (auto &t : mythreads)
        {
            t.join();
        }
        mythreads.clear();
    }
    return ret.load();
}

template <typename T> ErrorCode Index<T>::DeleteIndex(const SizeType &p_id, int tolayer)
{
    LayerReadLock topology(m_layerMutex, m_layerLockEnabled);
    std::shared_lock<std::shared_timed_mutex> sharedlock(m_dataDeleteLock);
    if (tolayer >= m_extraSearchers.size()) {
        auto it = m_topGlobalToLocalID.find(p_id);
        if (it == m_topGlobalToLocalID.end())
        {
            return ErrorCode::VectorNotFound;
        }
        return m_topIndex->DeleteIndex(it->second);
    }
    return m_extraSearchers[tolayer]->DeleteIndex(p_id);
}

template <typename T> ErrorCode Index<T>::ResetIndex(const SizeType& p_id, int tolayer)
{
    LayerReadLock topology(m_layerMutex, m_layerLockEnabled);
    std::shared_lock<std::shared_timed_mutex> sharedlock(m_dataDeleteLock);
    if (tolayer >= m_extraSearchers.size()) {
        auto it = m_topGlobalToLocalID.find(p_id);
        if (it != m_topGlobalToLocalID.end())
        {
            m_topIndex->ResetIndex(it->second);
        } else {
            return ErrorCode::VectorNotFound;
        }
        return ErrorCode::Success;
    }
    return m_extraSearchers[tolayer]->ResetIndex(p_id);
}

template <typename T> ErrorCode Index<T>::DeleteIndex(const void *p_vectors, SizeType p_vectorNum)
{
    LayerReadLock mutations(m_checkPointLock, m_layerLockEnabled);
    LayerReadLock topology(m_layerMutex, m_layerLockEnabled);
    // TODO: Support batch delete
    DimensionType p_dimension = GetFeatureDim();
    std::shared_ptr<VectorSet> vectorSet;
    if (m_options.m_distCalcMethod == DistCalcMethod::Cosine)
    {
        ByteArray arr = ByteArray::Alloc(sizeof(T) * p_vectorNum * p_dimension);
        memcpy(arr.Data(), p_vectors, sizeof(T) * p_vectorNum * p_dimension);
        vectorSet.reset(new BasicVectorSet(arr, GetEnumValueType<T>(), p_dimension, p_vectorNum));
        int base = COMMON::Utils::GetBase<T>();
        for (SizeType i = 0; i < p_vectorNum; i++)
        {
            COMMON::Utils::Normalize((T *)(vectorSet->GetVector(i)), p_dimension, base);
        }
    }
    else
    {
        vectorSet.reset(new BasicVectorSet(ByteArray((std::uint8_t *)p_vectors, sizeof(T) * p_vectorNum * p_dimension, false),
                                           GetEnumValueType<T>(), p_dimension, p_vectorNum));
    }

    for (int i = 0; i < p_vectorNum; i++) {
        QueryResult queryResults(vectorSet->GetVector(i), 1, false);
        SearchHeadIndex(queryResults, 0);
        if (queryResults.GetResult(0)->Dist < Epsilon)
        {
            DeleteIndex(queryResults.GetResult(0)->VID);
        }
        else {
            return ErrorCode::VectorNotFound;
        }
    }
    return ErrorCode::Success;
}

template <typename T> void Index<T>::PrepareDB(std::shared_ptr<Helper::KeyValueIO>& db, int layer)
{
    if(m_options.m_storage == Storage::FILEIO) {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPANNIndex:UseFileIO\n");
        auto options = m_options;
        options.m_ssdMappingFile = LayerStoragePrefix(layer);
        db = std::make_shared<FileIO>(options, layer);
    }
    else if (m_options.m_storage == Storage::SPDKIO) {
#ifdef SPDK
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPANNIndex:UseSPDK\n");
        db.reset(new SPDKIO(m_options));
#else
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SPANNIndex:SPDK unsupport! Use -DSPDK to enable SPDK when doing cmake.\n");
        return;
#endif
    } 
    else if (m_options.m_storage == Storage::ROCKSDBIO) {
#ifdef ROCKSDB
        std::string indexDir = (m_options.m_recovery)? m_options.m_persistentBufferPath + FolderSep: m_options.m_indexDirectory + FolderSep;
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPANNIndex:UseKV\n");
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPANNIndex:dbPath:%s\n", (indexDir + m_options.m_KVFile + "_" + std::to_string(layer)).c_str());
        db.reset(new RocksDBIO((indexDir + m_options.m_KVFile + "_" + std::to_string(layer)).c_str(), m_options.m_useDirectIO, m_options.m_enableWAL, m_options.m_recovery));
#else
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SPANNIndex:RocksDB unsupport! Use -DROCKSDB to enable RocksDB when doing cmake.\n");
        return;
#endif
    }
    else if (m_options.m_storage == Storage::TIKVIO) {
#ifdef TIKV
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPANNIndex:UseTiKV\n");
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "SPANNIndex:PD addresses:%s, prefix:%s, useMultiChunkPosting:%s\n",
                     m_options.m_tikvPDAddresses.c_str(), m_options.m_tikvKeyPrefix.c_str(), m_options.m_useMultiChunkPosting ? "true" : "false");
        db = CreateTiKVLayerDB(LayerStoragePrefix(layer));
#else
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "SPANNIndex:TiKV unsupport! Use -DTIKV to enable TiKV when doing cmake.\n");
        return;
#endif
    }

}

template <typename T> std::shared_ptr<Helper::KeyValueIO> Index<T>::CreateTiKVLayerDB(const std::string& prefix)
{
#ifdef TIKV
    return std::make_shared<TiKVIO>(m_options.m_tikvPDAddresses, prefix, m_options.m_useMultiChunkPosting,
        m_options.m_postingCountCacheCapacity, m_options.m_asyncRpcMaxInflight);
#else
    SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "TiKV layer DB requires TIKV support.\n");
    return nullptr;
#endif
}
} // namespace SPANN
} // namespace SPTAG

#define DefineVectorValueType(Name, Type) template class SPTAG::SPANN::Index<Type>;

#include "inc/Core/DefinitionList.h"
#undef DefineVectorValueType
