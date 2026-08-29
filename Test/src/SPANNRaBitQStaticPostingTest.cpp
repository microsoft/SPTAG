// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Test.h"

#include "inc/Core/Common/RaBitQQuantizer.h"
#include "inc/Core/SPANN/Index.h"
#include "inc/Core/SPANN/ExtraStaticSearcher.h"
#include "inc/Core/VectorIndex.h"

#include <cmath>
#include <array>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace SPTAG;

namespace
{

constexpr DimensionType kDimension = 64;
constexpr SizeType kVectorCount = 72;
constexpr int kResultCount = 5;
constexpr int kRaBitQBits = 5;
constexpr std::uint32_t kStaticPostingHeaderMagic = 0x32545053U;
constexpr std::uint32_t kStaticPostingHeaderVersion = 2U;
constexpr std::uint32_t kStaticPostingHeaderRaBitQSplit = 1U;
constexpr std::uint32_t kStaticBatchPostingHeaderVersion = 3U;
constexpr std::uint32_t kStaticPostingHeaderRaBitQBatch = 2U;
constexpr std::uint32_t kBatchSidecarMagic = 0x53425152U;
constexpr const char* kPostingQuantizerIndexFile = "SPTAGPostingRaBitQQuantizer.bin";

struct StaticPostingHeader
{
    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    std::uint32_t format = 0;
    std::uint32_t reserved = 0;
    SizeType listCount = 0;
    SizeType totalDocumentCount = 0;
    int dataDimension = 0;
    SizeType listPageOffset = 0;
    int vectorInfoSize = 0;
    std::uint64_t postingQuantizerFingerprint = 0;
};

struct StaticBatchPostingHeader
{
    StaticPostingHeader base;
    std::uint32_t batchSize = 0;
    std::uint32_t batchRecordBytes = 0;
    std::uint32_t extendedBytesPerVector = 0;
    std::uint32_t flags = 0;
};

struct BatchRecordHeader
{
    std::uint32_t validCount = 0;
    std::uint32_t reserved = 0;
};

struct BatchSidecarHeader
{
    std::uint32_t magic = 0;
    std::uint32_t version = 0;
    std::uint32_t kind = 0;
    std::uint32_t layer = 0;
    std::uint32_t fileID = 0;
    std::uint32_t listCount = 0;
    std::uint32_t dimension = 0;
    std::uint32_t batchSize = 0;
    std::uint64_t postingQuantizerFingerprint = 0;
    std::uint64_t dataBytesPerRecord = 0;
};

struct BatchSidecarListInfo
{
    std::uint64_t vectorCount = 0;
    std::uint64_t batchCount = 0;
    std::uint64_t centroidOffset = 0;
    std::uint64_t dataOffset = 0;
    std::uint64_t dataBytes = 0;
};

class ScopedCleanup
{
public:
    explicit ScopedCleanup(std::vector<std::string> paths) : m_paths(std::move(paths)) {}

    ~ScopedCleanup()
    {
        for (const auto& path : m_paths)
        {
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
            std::filesystem::remove(path, ec);
        }
    }

private:
    std::vector<std::string> m_paths;
};

std::shared_ptr<VectorSet> MakeVectors(SizeType vectorCount = kVectorCount)
{
    ByteArray bytes = ByteArray::Alloc(
        sizeof(float) * static_cast<std::size_t>(vectorCount) * static_cast<std::size_t>(kDimension));
    auto* values = reinterpret_cast<float*>(bytes.Data());
    for (SizeType vector = 0; vector < vectorCount; ++vector)
    {
        for (DimensionType dim = 0; dim < kDimension; ++dim)
        {
            values[static_cast<std::size_t>(vector) * static_cast<std::size_t>(kDimension) +
                   static_cast<std::size_t>(dim)] =
                static_cast<float>(vector) * 0.2F +
                static_cast<float>(dim) * 0.01F +
                static_cast<float>((vector + dim) % 7) * 0.001F;
        }
    }
    return std::make_shared<BasicVectorSet>(
        bytes, VectorValueType::Float, kDimension, vectorCount);
}

std::shared_ptr<VectorSet> MakeShiftedVectors(float shift)
{
    ByteArray bytes = ByteArray::Alloc(
        sizeof(float) * static_cast<std::size_t>(kVectorCount) * static_cast<std::size_t>(kDimension));
    auto* values = reinterpret_cast<float*>(bytes.Data());
    for (SizeType vector = 0; vector < kVectorCount; ++vector)
    {
        for (DimensionType dim = 0; dim < kDimension; ++dim)
        {
            values[static_cast<std::size_t>(vector) * static_cast<std::size_t>(kDimension) +
                   static_cast<std::size_t>(dim)] =
                static_cast<float>(vector) * (0.19F + shift * 0.01F) +
                static_cast<float>(dim) * (0.012F + shift * 0.001F) +
                static_cast<float>((vector * 3 + dim * 5) % 11) * 0.002F +
                shift;
        }
    }
    return std::make_shared<BasicVectorSet>(bytes, VectorValueType::Float, kDimension, kVectorCount);
}

std::shared_ptr<VectorSet> MakeSubsetVectors(const std::shared_ptr<VectorSet>& source,
                                             const std::vector<SizeType>& ids)
{
    ByteArray bytes = ByteArray::Alloc(
        sizeof(float) * ids.size() * static_cast<std::size_t>(kDimension));
    auto* values = reinterpret_cast<float*>(bytes.Data());
    for (std::size_t i = 0; i < ids.size(); ++i)
    {
        std::memcpy(values + i * static_cast<std::size_t>(kDimension),
                    source->GetVector(ids[i]),
                    sizeof(float) * static_cast<std::size_t>(kDimension));
    }
    return std::make_shared<BasicVectorSet>(
        bytes, VectorValueType::Float, kDimension, static_cast<SizeType>(ids.size()));
}

std::shared_ptr<COMMON::RaBitQQuantizer> TrainAndSaveRaBitQModel(
    const std::shared_ptr<VectorSet>& vectors,
    const std::string& modelPath,
    DistCalcMethod metric)
{
    auto quantizer = std::make_shared<COMMON::RaBitQQuantizer>(
        kDimension,
        kRaBitQBits,
        false,
        metric,
        COMMON::RaBitQQuantizer::QuantizationMode::Exact);
    BOOST_REQUIRE(quantizer->Train(vectors) == ErrorCode::Success);

    auto output = f_createIO();
    BOOST_REQUIRE(output != nullptr);
    BOOST_REQUIRE(output->Initialize(modelPath.c_str(), std::ios::binary | std::ios::out));
    BOOST_REQUIRE(quantizer->SaveQuantizer(output) == ErrorCode::Success);
    output->ShutDown();
    return quantizer;
}

std::shared_ptr<VectorIndex> MakeStaticSpannIndex(const std::string& indexDir,
                                                  const std::string& modelPath,
                                                  DistCalcMethod metric,
                                                  bool enableDeltaEncoding = false,
                                                  bool enablePostingRearrange = false,
                                                  int rerank = 0,
                                                  int layers = 1,
                                                  bool useBatchPosting = false,
                                                  int postingRaBitQRerank = 0,
                                                  int ssdIndexFileNum = 1)
{
    auto index = VectorIndex::CreateInstance(IndexAlgoType::SPANN, VectorValueType::Float);
    BOOST_REQUIRE(index != nullptr);

    index->SetParameter("IndexDirectory", indexDir, "Base");
    index->SetParameter("IndexAlgoType", "BKT", "Base");
    index->SetParameter("DistCalcMethod", Helper::Convert::ConvertToString(metric), "Base");
    index->SetParameter(
        "SSDIndexFileNum", std::to_string(ssdIndexFileNum), "Base");

    index->SetParameter("isExecute", "true", "SelectHead");
    index->SetParameter("NumberOfThreads", "1", "SelectHead");
    index->SetParameter("TreeNumber", "1", "SelectHead");
    index->SetParameter("BKTKmeansK", "4", "SelectHead");
    index->SetParameter("BKTLeafSize", "4", "SelectHead");
    index->SetParameter("SamplesNumber", "16", "SelectHead");
    index->SetParameter("SelectThreshold", "2", "SelectHead");
    index->SetParameter("SplitFactor", "2", "SelectHead");
    index->SetParameter("SplitThreshold", "8", "SelectHead");
    index->SetParameter("Ratio", "0.2", "SelectHead");
    index->SetParameter("SelectHeadType", "BKT", "SelectHead");

    index->SetParameter("isExecute", "true", "BuildHead");
    index->SetParameter("NumberOfThreads", "1", "BuildHead");
    index->SetParameter("NeighborhoodSize", "8", "BuildHead");
    index->SetParameter("TPTNumber", "8", "BuildHead");
    index->SetParameter("TPTLeafSize", "32", "BuildHead");
    index->SetParameter("MaxCheck", "128", "BuildHead");
    index->SetParameter("MaxCheckForRefineGraph", "128", "BuildHead");
    index->SetParameter("RefineIterations", "1", "BuildHead");

    index->SetParameter("isExecute", "true", "BuildSSDIndex");
    index->SetParameter("BuildSsdIndex", "true", "BuildSSDIndex");
    index->SetParameter("Storage", "STATIC", "BuildSSDIndex");
    index->SetParameter("NumberOfThreads", "1", "BuildSSDIndex");
    index->SetParameter("SearchThreadNum", "1", "BuildSSDIndex");
    index->SetParameter("PostingPageLimit", "4", "BuildSSDIndex");
    index->SetParameter("SearchPostingPageLimit", "4", "BuildSSDIndex");
    index->SetParameter("InternalResultNum", "16", "BuildSSDIndex");
    index->SetParameter("SearchInternalResultNum", "16", "BuildSSDIndex");
    index->SetParameter("MaxCheck", "256", "BuildSSDIndex");
    index->SetParameter("HashTableExponent", "4", "BuildSSDIndex");
    index->SetParameter("ReplicaCount", "4", "BuildSSDIndex");
    index->SetParameter("EnableDeltaEncoding", enableDeltaEncoding ? "true" : "false", "BuildSSDIndex");
    index->SetParameter("EnablePostingListRearrange", enablePostingRearrange ? "true" : "false", "BuildSSDIndex");
    index->SetParameter("EnableDataCompression", "false", "BuildSSDIndex");
    index->SetParameter(
        "PostingQuantizer",
        useBatchPosting ? "RaBitQBatch" : "RaBitQ",
        "BuildSSDIndex");
    index->SetParameter("PostingQuantizerFile", modelPath, "BuildSSDIndex");
    index->SetParameter("Rerank", std::to_string(rerank), "BuildSSDIndex");
    index->SetParameter(
        "PostingRaBitQRerank",
        std::to_string(postingRaBitQRerank),
        "BuildSSDIndex");
    index->SetParameter("Layers", std::to_string(layers), "BuildSSDIndex");

    return index;
}

std::vector<BasicResult> CollectValidResults(const QueryResult& result)
{
    std::vector<BasicResult> collected;
    for (int i = 0; i < result.GetResultNum(); ++i)
    {
        const BasicResult* current = result.GetResult(i);
        if (current == nullptr || current->VID < 0) break;
        collected.push_back(*current);
    }
    return collected;
}

bool ContainsVID(const std::vector<BasicResult>& results, SizeType vid)
{
    for (const auto& result : results)
    {
        if (result.VID == vid) return true;
    }
    return false;
}

std::vector<BasicResult> SearchOnce(const std::shared_ptr<VectorIndex>& index,
                                    const std::shared_ptr<VectorSet>& vectors,
                                    SizeType queryID)
{
    QueryResult result(vectors->GetVector(queryID), kResultCount, false, false);
    BOOST_REQUIRE(index->SearchIndex(result) == ErrorCode::Success);
    auto collected = CollectValidResults(result);
    BOOST_REQUIRE(!collected.empty());
    for (const auto& entry : collected)
    {
        BOOST_CHECK(std::isfinite(entry.Dist));
    }
    return collected;
}

StaticPostingHeader ReadPostingHeader(const std::string& postingPath)
{
    std::ifstream input(postingPath, std::ios::binary);
    BOOST_REQUIRE(input.good());
    StaticPostingHeader header;
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    BOOST_REQUIRE(input.good());
    return header;
}

template <typename T>
T ReadObject(const std::string& path, std::uint64_t offset = 0)
{
    std::ifstream input(path, std::ios::binary);
    BOOST_REQUIRE(input.good());
    input.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    T value;
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    BOOST_REQUIRE(input.good());
    return value;
}

std::vector<float> ReadFloats(
    const std::string& path, std::uint64_t offset, std::size_t count)
{
    std::ifstream input(path, std::ios::binary);
    BOOST_REQUIRE(input.good());
    input.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    std::vector<float> values(count);
    input.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    BOOST_REQUIRE(input.good());
    return values;
}

std::vector<char> ReadBinaryFile(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    BOOST_REQUIRE(input.good());
    input.seekg(0, std::ios::end);
    const auto size = static_cast<std::size_t>(input.tellg());
    input.seekg(0, std::ios::beg);
    std::vector<char> bytes(size);
    input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    BOOST_REQUIRE(input.good() || input.eof());
    BOOST_REQUIRE_EQUAL(static_cast<std::size_t>(input.gcount()), bytes.size());
    return bytes;
}

void WriteBinaryFile(const std::string& path, const std::vector<char>& bytes)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    BOOST_REQUIRE(output.good());
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    BOOST_REQUIRE(output.good());
}

} // namespace

BOOST_AUTO_TEST_SUITE(SPANNRaBitQStaticPostingTest)

BOOST_AUTO_TEST_CASE(StaticSpannRaBitQPostingBuildsSearchesAndReloads)
{
    const std::string indexDir = "spann_rabitq_static_posting_build_search";
    const std::string relocatedDir = "spann_rabitq_static_posting_build_search_relocated";
    const std::string modelPath = "spann_rabitq_static_posting_build_search.bin";
    ScopedCleanup cleanup({indexDir, relocatedDir, modelPath});

    const auto vectors = MakeVectors();
    TrainAndSaveRaBitQModel(vectors, modelPath, DistCalcMethod::L2);

    auto index = MakeStaticSpannIndex(indexDir, modelPath, DistCalcMethod::L2);
    BOOST_REQUIRE(index->BuildIndex(vectors, nullptr) == ErrorCode::Success);

    const std::vector<SizeType> queryIDs = {3, 19, 51};
    for (SizeType queryID : queryIDs)
    {
        auto results = SearchOnce(index, vectors, queryID);
        BOOST_CHECK(ContainsVID(results, queryID));
    }

    BOOST_REQUIRE(index->SaveIndex(relocatedDir) == ErrorCode::Success);
    BOOST_REQUIRE(std::filesystem::exists(
        relocatedDir + FolderSep + kPostingQuantizerIndexFile));

    std::filesystem::remove(modelPath);
    std::filesystem::remove_all(indexDir);

    std::shared_ptr<VectorIndex> loaded;
    BOOST_REQUIRE(VectorIndex::LoadIndex(relocatedDir, loaded) == ErrorCode::Success);
    BOOST_REQUIRE(loaded != nullptr);

    for (std::size_t i = 0; i < queryIDs.size(); ++i)
    {
        auto afterReload = SearchOnce(loaded, vectors, queryIDs[i]);
        BOOST_CHECK(ContainsVID(afterReload, queryIDs[i]));
    }
}

BOOST_AUTO_TEST_CASE(StaticSpannRaBitQPostingStoresSplitCodesWithHeadCentroids)
{
    const std::string indexDir = "spann_rabitq_static_posting_centroid";
    const std::string modelPath = "spann_rabitq_static_posting_centroid.bin";
    ScopedCleanup cleanup({indexDir, modelPath});

    const auto vectors = MakeVectors();
    auto quantizer = TrainAndSaveRaBitQModel(vectors, modelPath, DistCalcMethod::L2);
    const auto layout = quantizer->GetSplitCodeLayout();
    const int expectedRecordBytes = static_cast<int>(sizeof(SizeType) + layout.totalBytes);

    auto index = MakeStaticSpannIndex(indexDir, modelPath, DistCalcMethod::L2);
    BOOST_REQUIRE(index->BuildIndex(vectors, nullptr) == ErrorCode::Success);
    BOOST_REQUIRE(index->SaveIndex(indexDir) == ErrorCode::Success);

    const StaticPostingHeader header = ReadPostingHeader(indexDir + FolderSep + "SPTAGFullList.bin_0");
    BOOST_CHECK_EQUAL(header.magic, kStaticPostingHeaderMagic);
    BOOST_CHECK_EQUAL(header.version, kStaticPostingHeaderVersion);
    BOOST_CHECK_EQUAL(header.format, kStaticPostingHeaderRaBitQSplit);
    BOOST_CHECK_EQUAL(header.dataDimension, kDimension);
    BOOST_CHECK_EQUAL(header.vectorInfoSize, expectedRecordBytes);
    BOOST_CHECK_NE(header.postingQuantizerFingerprint, 0ULL);

    std::shared_ptr<VectorIndex> loaded;
    BOOST_REQUIRE(VectorIndex::LoadIndex(indexDir, loaded) == ErrorCode::Success);
    auto spann = std::dynamic_pointer_cast<SPANN::Index<float>>(loaded);
    BOOST_REQUIRE(spann != nullptr);

    std::vector<SizeType> headIDs;
    BOOST_REQUIRE(spann->GetHeadIndexMapping(1, headIDs) == ErrorCode::Success);
    BOOST_REQUIRE(!headIDs.empty());

    SizeType chosenLocalHead = -1;
    std::string posting;
    for (std::size_t head = 0; head < headIDs.size(); ++head)
    {
        posting.clear();
        if (spann->GetDiskIndex(0)->GetWritePosting(nullptr, headIDs[head], posting) == ErrorCode::Success &&
            !posting.empty())
        {
            chosenLocalHead = static_cast<SizeType>(head);
            break;
        }
    }

    BOOST_REQUIRE(chosenLocalHead >= 0);
    BOOST_REQUIRE_EQUAL(posting.size() % static_cast<std::size_t>(expectedRecordBytes), 0U);

    const auto* centroid = reinterpret_cast<const float*>(vectors->GetVector(headIDs[chosenLocalHead]));
    BOOST_REQUIRE(centroid != nullptr);

    std::vector<std::uint8_t> expectedBinary(layout.binaryBytes, 0);
    std::vector<std::uint8_t> expectedExtended(layout.extendedBytes, 0);
    std::vector<std::uint8_t> wrongBinary(layout.binaryBytes, 0);
    std::vector<std::uint8_t> wrongExtended(layout.extendedBytes, 0);
    const float* wrongCentroid = centroid;
    if (headIDs.size() > 1)
    {
        const std::size_t wrongHead =
            (static_cast<std::size_t>(chosenLocalHead) + 1) % headIDs.size();
        wrongCentroid = reinterpret_cast<const float*>(vectors->GetVector(headIDs[wrongHead]));
    }

    bool matchedStoredCode = false;
    bool rejectedWrongCentroid = false;
    for (std::size_t offset = 0; offset < posting.size(); offset += static_cast<std::size_t>(expectedRecordBytes))
    {
        SizeType vectorID = -1;
        std::memcpy(&vectorID, posting.data() + offset, sizeof(SizeType));

        const auto* storedBinary = reinterpret_cast<const std::uint8_t*>(
            posting.data() + offset + sizeof(SizeType));
        const auto* storedExtended = (layout.extendedBytes == 0)
            ? nullptr
            : storedBinary + layout.binaryBytes;

        BOOST_REQUIRE(quantizer->QuantizeSplitVector(
                          vectors->GetVector(vectorID),
                          centroid,
                          expectedBinary.data(),
                          layout.extendedBytes == 0 ? nullptr : expectedExtended.data()) ==
                      ErrorCode::Success);
        BOOST_CHECK_EQUAL(
            std::memcmp(expectedBinary.data(), storedBinary, layout.binaryBytes),
            0);
        if (layout.extendedBytes > 0)
        {
            BOOST_CHECK_EQUAL(
                std::memcmp(expectedExtended.data(), storedExtended, layout.extendedBytes),
                0);
        }

        matchedStoredCode = true;

        if (wrongCentroid != centroid)
        {
            BOOST_REQUIRE(quantizer->QuantizeSplitVector(
                              vectors->GetVector(vectorID),
                              wrongCentroid,
                              wrongBinary.data(),
                              layout.extendedBytes == 0 ? nullptr : wrongExtended.data()) ==
                          ErrorCode::Success);
            const bool binaryMismatch =
                std::memcmp(wrongBinary.data(), storedBinary, layout.binaryBytes) != 0;
            const bool extendedMismatch =
                layout.extendedBytes == 0 ||
                std::memcmp(wrongExtended.data(), storedExtended, layout.extendedBytes) != 0;
            if (binaryMismatch || extendedMismatch)
            {
                rejectedWrongCentroid = true;
                break;
            }
        }
    }

    BOOST_CHECK(matchedStoredCode);
    BOOST_CHECK(rejectedWrongCentroid);
}

BOOST_AUTO_TEST_CASE(StaticSpannRaBitQPostingRejectsReplacedSameShapeModel)
{
    const std::string indexDir = "spann_rabitq_static_posting_fingerprint";
    const std::string modelPath = "spann_rabitq_static_posting_fingerprint.bin";
    const std::string replacementModelPath = "spann_rabitq_static_posting_fingerprint_replacement.bin";
    ScopedCleanup cleanup({indexDir, modelPath, replacementModelPath});

    const auto vectors = MakeVectors();
    TrainAndSaveRaBitQModel(vectors, modelPath, DistCalcMethod::L2);

    auto index = MakeStaticSpannIndex(indexDir, modelPath, DistCalcMethod::L2);
    BOOST_REQUIRE(index->BuildIndex(vectors, nullptr) == ErrorCode::Success);
    BOOST_REQUIRE(index->SaveIndex(indexDir) == ErrorCode::Success);

    const auto replacementVectors = MakeShiftedVectors(0.75F);
    TrainAndSaveRaBitQModel(replacementVectors, replacementModelPath, DistCalcMethod::L2);

    const auto replacementBytes = ReadBinaryFile(replacementModelPath);
    WriteBinaryFile(indexDir + FolderSep + kPostingQuantizerIndexFile, replacementBytes);

    std::shared_ptr<VectorIndex> loaded;
    BOOST_CHECK(VectorIndex::LoadIndex(indexDir, loaded) != ErrorCode::Success);
}

BOOST_AUTO_TEST_CASE(StaticSpannRaBitQPostingRejectsTruncatedModel)
{
    const std::string indexDir = "spann_rabitq_static_posting_truncated_model";
    const std::string modelPath = "spann_rabitq_static_posting_truncated_model.bin";
    ScopedCleanup cleanup({indexDir, modelPath});

    const auto vectors = MakeVectors();
    TrainAndSaveRaBitQModel(vectors, modelPath, DistCalcMethod::L2);

    auto index = MakeStaticSpannIndex(
        indexDir, modelPath, DistCalcMethod::L2);
    BOOST_REQUIRE(
        index->BuildIndex(vectors, nullptr) == ErrorCode::Success);
    BOOST_REQUIRE(index->SaveIndex(indexDir) == ErrorCode::Success);

    const std::string persistedModel =
        indexDir + FolderSep + kPostingQuantizerIndexFile;
    auto modelBytes = ReadBinaryFile(persistedModel);
    BOOST_REQUIRE(modelBytes.size() > 1);
    modelBytes.resize(modelBytes.size() - 1);
    WriteBinaryFile(persistedModel, modelBytes);

    std::shared_ptr<VectorIndex> loaded;
    BOOST_CHECK(
        VectorIndex::LoadIndex(indexDir, loaded) != ErrorCode::Success);
}

BOOST_AUTO_TEST_CASE(StaticSpannRaBitQPostingRejectsCorruptMainMetadata)
{
    const std::string indexDir = "spann_rabitq_static_posting_corrupt_main";
    const std::string modelPath = "spann_rabitq_static_posting_corrupt_main.bin";
    ScopedCleanup cleanup({indexDir, modelPath});

    const auto vectors = MakeVectors();
    TrainAndSaveRaBitQModel(vectors, modelPath, DistCalcMethod::L2);

    auto index = MakeStaticSpannIndex(
        indexDir, modelPath, DistCalcMethod::L2);
    BOOST_REQUIRE(
        index->BuildIndex(vectors, nullptr) == ErrorCode::Success);
    BOOST_REQUIRE(index->SaveIndex(indexDir) == ErrorCode::Success);

    const std::string postingPath =
        indexDir + FolderSep + "SPTAGFullList.bin_0";
    const auto originalBytes = ReadBinaryFile(postingPath);
    BOOST_REQUIRE(
        originalBytes.size() >=
        sizeof(StaticPostingHeader) + sizeof(int));

    auto expectRejected =
        [&](const std::vector<char>& corruptBytes)
        {
            WriteBinaryFile(postingPath, corruptBytes);
            std::shared_ptr<VectorIndex> loaded;
            BOOST_CHECK(
                VectorIndex::LoadIndex(indexDir, loaded) !=
                ErrorCode::Success);
        };

    {
        auto corruptBytes = originalBytes;
        StaticPostingHeader header;
        std::memcpy(&header, corruptBytes.data(), sizeof(header));
        header.listCount = std::numeric_limits<SizeType>::max();
        std::memcpy(corruptBytes.data(), &header, sizeof(header));
        expectRejected(corruptBytes);
    }

    {
        auto corruptBytes = originalBytes;
        StaticPostingHeader header;
        std::memcpy(&header, corruptBytes.data(), sizeof(header));
        header.listPageOffset = std::numeric_limits<SizeType>::max();
        std::memcpy(corruptBytes.data(), &header, sizeof(header));
        expectRejected(corruptBytes);
    }

    {
        auto corruptBytes = originalBytes;
        const int invalidPageNumber = std::numeric_limits<int>::max();
        std::memcpy(
            corruptBytes.data() + sizeof(StaticPostingHeader),
            &invalidPageNumber,
            sizeof(invalidPageNumber));
        expectRejected(corruptBytes);
    }

    WriteBinaryFile(postingPath, originalBytes);
    std::shared_ptr<VectorIndex> loaded;
    BOOST_REQUIRE(
        VectorIndex::LoadIndex(indexDir, loaded) == ErrorCode::Success);
}

BOOST_AUTO_TEST_CASE(StaticSpannRaBitQPostingPrefersIndexLocalModelOverCwdShadow)
{
    const std::string indexDir = "spann_rabitq_static_posting_shadow";
    const std::string modelPath = "spann_rabitq_static_posting_shadow.bin";
    const std::string replacementModelPath = "spann_rabitq_static_posting_shadow_replacement.bin";
    const std::string shadowModelPath = kPostingQuantizerIndexFile;
    ScopedCleanup cleanup({indexDir, modelPath, replacementModelPath, shadowModelPath});

    const auto vectors = MakeVectors();
    TrainAndSaveRaBitQModel(vectors, modelPath, DistCalcMethod::L2);

    auto index = MakeStaticSpannIndex(indexDir, modelPath, DistCalcMethod::L2);
    BOOST_REQUIRE(index->BuildIndex(vectors, nullptr) == ErrorCode::Success);
    BOOST_REQUIRE(index->SaveIndex(indexDir) == ErrorCode::Success);
    BOOST_REQUIRE(std::filesystem::exists(indexDir + FolderSep + shadowModelPath));

    const auto replacementVectors = MakeShiftedVectors(0.5F);
    TrainAndSaveRaBitQModel(replacementVectors, replacementModelPath, DistCalcMethod::L2);
    WriteBinaryFile(shadowModelPath, ReadBinaryFile(replacementModelPath));

    std::shared_ptr<VectorIndex> loaded;
    BOOST_REQUIRE(VectorIndex::LoadIndex(indexDir, loaded) == ErrorCode::Success);
    BOOST_REQUIRE(loaded != nullptr);

    auto results = SearchOnce(loaded, vectors, 7);
    BOOST_CHECK(ContainsVID(results, 7));
}

BOOST_AUTO_TEST_CASE(StaticSpannRaBitQPostingPrunesOnlyAgainstRetainedUpperBounds)
{
    const auto vectors = MakeVectors();
    COMMON::QueryResultSet<float> retained(
        reinterpret_cast<const float*>(vectors->GetVector(0)), 2, false, false);
    BOOST_REQUIRE(retained.AddPoint(20, 2.0F, ByteArray::c_empty));
    BOOST_REQUIRE(retained.AddPoint(10, 1.0F, ByteArray::c_empty));
    BOOST_CHECK_EQUAL(retained.worstDist(), 2.0F);

    SPANN::ExtraWorkSpace workspace;
    workspace.ResetRaBitQPruning(2);
    workspace.RecordRaBitQCandidate(10, 1.0F, 1.0F);
    workspace.RecordRaBitQCandidate(20, 2.0F, 5.0F);

    const float candidateLowerBound = 3.0F;
    BOOST_CHECK(candidateLowerBound > retained.worstDist());
    BOOST_CHECK(!workspace.CanPruneRaBitQCandidate(candidateLowerBound));
    BOOST_CHECK(workspace.CanPruneRaBitQCandidate(5.1F));
}

BOOST_AUTO_TEST_CASE(StaticSpannRaBitQPostingRejectsWithVecSearch)
{
    const std::string indexDir = "spann_rabitq_static_posting_with_vec";
    const std::string modelPath = "spann_rabitq_static_posting_with_vec.bin";
    ScopedCleanup cleanup({indexDir, modelPath});

    const auto vectors = MakeVectors();
    TrainAndSaveRaBitQModel(vectors, modelPath, DistCalcMethod::L2);

    auto index = MakeStaticSpannIndex(indexDir, modelPath, DistCalcMethod::L2);
    BOOST_REQUIRE(index->BuildIndex(vectors, nullptr) == ErrorCode::Success);

    QueryResult result(vectors->GetVector(0), kResultCount, false, true);
    BOOST_CHECK(index->SearchIndex(result) == ErrorCode::Undefined);
}

BOOST_AUTO_TEST_CASE(StaticSpannRaBitQPostingUsesLayerLocalVectorsWithRemappedStoredIDs)
{
    const std::string indexDir = "spann_rabitq_static_posting_direct_mapping";
    const std::string modelPath = "spann_rabitq_static_posting_direct_mapping.bin";
    ScopedCleanup cleanup({indexDir, modelPath});

    const auto vectors = MakeVectors();
    auto quantizer = TrainAndSaveRaBitQModel(vectors, modelPath, DistCalcMethod::L2);
    const auto layout = quantizer->GetSplitCodeLayout();
    const int recordBytes = static_cast<int>(sizeof(SizeType) + layout.totalBytes);

    auto index = MakeStaticSpannIndex(indexDir, modelPath, DistCalcMethod::L2);
    BOOST_REQUIRE(index->BuildIndex(vectors, nullptr) == ErrorCode::Success);

    auto spann = std::dynamic_pointer_cast<SPANN::Index<float>>(index);
    BOOST_REQUIRE(spann != nullptr);
    auto searcher = std::dynamic_pointer_cast<SPANN::ExtraStaticSearcher<float>>(spann->GetDiskIndex(0));
    BOOST_REQUIRE(searcher != nullptr);

    const std::vector<SizeType> syntheticIDs = {4, 17, 29};
    const auto syntheticVectors = MakeSubsetVectors(vectors, syntheticIDs);
    COMMON::Dataset<SizeType> localToGlobal(3, 1, 4, 3);
    const std::array<SizeType, 3> remappedIDs = {2, 0, 1};
    for (std::size_t i = 0; i < remappedIDs.size(); ++i)
    {
        *localToGlobal[i] = remappedIDs[i];
    }

    SPANN::Selection selections(remappedIDs.size(), indexDir);
    for (std::size_t i = 0; i < remappedIDs.size(); ++i)
    {
        selections.m_selections[i].node = 0;
        selections.m_selections[i].distance = static_cast<float>(i);
        selections.m_selections[i].tonode = static_cast<SizeType>(i);
    }

    const auto* centroid =
        reinterpret_cast<const float*>(spann->GetMemoryIndex()->GetSample(0));
    BOOST_REQUIRE(centroid != nullptr);

    const std::string posting = searcher->GetPostingListFullData(
        0,
        remappedIDs.size(),
        selections,
        syntheticVectors,
        localToGlobal,
        false,
        false,
        nullptr);
    BOOST_REQUIRE_EQUAL(
        posting.size(),
        remappedIDs.size() * static_cast<std::size_t>(recordBytes));

    bool observedNonIdentityMapping = false;
    std::vector<std::uint8_t> expectedBinary(layout.binaryBytes, 0);
    std::vector<std::uint8_t> expectedExtended(layout.extendedBytes, 0);
    std::vector<std::uint8_t> wrongBinary(layout.binaryBytes, 0);
    std::vector<std::uint8_t> wrongExtended(layout.extendedBytes, 0);
    for (std::size_t localVID = 0; localVID < remappedIDs.size(); ++localVID)
    {
        const char* record = posting.data() + localVID * static_cast<std::size_t>(recordBytes);
        SizeType storedVID = -1;
        std::memcpy(&storedVID, record, sizeof(SizeType));
        BOOST_CHECK_EQUAL(storedVID, remappedIDs[localVID]);

        const auto* storedBinary =
            reinterpret_cast<const std::uint8_t*>(record + sizeof(SizeType));
        const auto* storedExtended =
            layout.extendedBytes == 0 ? nullptr : storedBinary + layout.binaryBytes;

        BOOST_REQUIRE(quantizer->QuantizeSplitVector(
                          syntheticVectors->GetVector(static_cast<SizeType>(localVID)),
                          centroid,
                          expectedBinary.data(),
                          layout.extendedBytes == 0 ? nullptr : expectedExtended.data()) ==
                      ErrorCode::Success);
        BOOST_CHECK_EQUAL(
            std::memcmp(expectedBinary.data(), storedBinary, layout.binaryBytes),
            0);
        if (layout.extendedBytes > 0)
        {
            BOOST_CHECK_EQUAL(
                std::memcmp(expectedExtended.data(), storedExtended, layout.extendedBytes),
                0);
        }

        if (storedVID != static_cast<SizeType>(localVID))
        {
            BOOST_REQUIRE(quantizer->QuantizeSplitVector(
                              syntheticVectors->GetVector(storedVID),
                              centroid,
                              wrongBinary.data(),
                              layout.extendedBytes == 0 ? nullptr : wrongExtended.data()) ==
                          ErrorCode::Success);
            const bool wrongBinaryMismatch =
                std::memcmp(wrongBinary.data(), storedBinary, layout.binaryBytes) != 0;
            const bool wrongExtendedMismatch =
                layout.extendedBytes == 0 ||
                std::memcmp(wrongExtended.data(), storedExtended, layout.extendedBytes) != 0;
            if (wrongBinaryMismatch || wrongExtendedMismatch)
            {
                observedNonIdentityMapping = true;
            }
        }
    }

    BOOST_CHECK(observedNonIdentityMapping);
}

BOOST_AUTO_TEST_CASE(StaticSpannRaBitQBatchPersistsOfficialSplitLayoutAndMeanCentroid)
{
    const std::string indexDir = "spann_rabitq_batch_layout";
    const std::string relocatedDir = "spann_rabitq_batch_layout_relocated";
    const std::string modelPath = "spann_rabitq_batch_layout.bin";
    ScopedCleanup cleanup({indexDir, relocatedDir, modelPath});

    const auto vectors = MakeVectors();
    auto quantizer = TrainAndSaveRaBitQModel(
        vectors, modelPath, DistCalcMethod::L2);
    const auto layout = quantizer->GetSplitBatchLayout();
    const std::size_t recordBytes = sizeof(BatchRecordHeader) +
        layout.batchSize * sizeof(SizeType) + layout.binaryBytes;

    auto index = MakeStaticSpannIndex(
        indexDir,
        modelPath,
        DistCalcMethod::L2,
        false,
        false,
        0,
        1,
        true);
    BOOST_REQUIRE(index->BuildIndex(vectors, nullptr) == ErrorCode::Success);

    const std::string postingPath =
        indexDir + FolderSep + "SPTAGFullList.bin_0";
    const std::string extendedPath = postingPath + ".rabitq.ext";
    const auto header = ReadObject<StaticBatchPostingHeader>(postingPath);
    BOOST_CHECK_EQUAL(header.base.magic, kStaticPostingHeaderMagic);
    BOOST_CHECK_EQUAL(
        header.base.version, kStaticBatchPostingHeaderVersion);
    BOOST_CHECK_EQUAL(
        header.base.format, kStaticPostingHeaderRaBitQBatch);
    BOOST_CHECK_EQUAL(header.batchSize, layout.batchSize);
    BOOST_CHECK_EQUAL(header.batchRecordBytes, recordBytes);
    BOOST_CHECK_EQUAL(
        header.extendedBytesPerVector, layout.extendedBytesPerVector);
    BOOST_CHECK_EQUAL(header.flags, 0U);

    const auto sidecarHeader = ReadObject<BatchSidecarHeader>(extendedPath);
    BOOST_CHECK_EQUAL(sidecarHeader.magic, kBatchSidecarMagic);
    BOOST_CHECK_EQUAL(sidecarHeader.kind, 1U);
    BOOST_CHECK_EQUAL(sidecarHeader.batchSize, layout.batchSize);
    BOOST_CHECK_EQUAL(
        sidecarHeader.postingQuantizerFingerprint,
        header.base.postingQuantizerFingerprint);

    auto spann = std::dynamic_pointer_cast<SPANN::Index<float>>(index);
    BOOST_REQUIRE(spann != nullptr);
    std::vector<SizeType> headIDs;
    BOOST_REQUIRE(
        spann->GetHeadIndexMapping(1, headIDs) == ErrorCode::Success);

    std::size_t chosenHead = headIDs.size();
    std::string posting;
    for (std::size_t head = 0; head < headIDs.size(); ++head)
    {
        posting.clear();
        if (spann->GetDiskIndex(0)->GetWritePosting(
                nullptr, headIDs[head], posting) == ErrorCode::Success &&
            !posting.empty())
        {
            const auto* lastHeader = reinterpret_cast<const BatchRecordHeader*>(
                posting.data() +
                (posting.size() / recordBytes - 1) * recordBytes);
            if (lastHeader->validCount < layout.batchSize)
            {
                chosenHead = head;
                break;
            }
        }
    }
    BOOST_REQUIRE(chosenHead < headIDs.size());
    BOOST_REQUIRE_EQUAL(posting.size() % recordBytes, 0U);

    const auto sidecarInfo = ReadObject<BatchSidecarListInfo>(
        extendedPath,
        sizeof(BatchSidecarHeader) +
            chosenHead * sizeof(BatchSidecarListInfo));
    BOOST_CHECK_EQUAL(
        sidecarInfo.batchCount, posting.size() / recordBytes);
    BOOST_CHECK_EQUAL(
        sidecarInfo.dataBytes,
        sidecarInfo.batchCount * layout.extendedBytes);
    BOOST_CHECK_EQUAL(
        posting.size(), sidecarInfo.batchCount * recordBytes);
    BOOST_CHECK(sidecarInfo.dataBytes > 0U);

    const auto centroid = ReadFloats(
        extendedPath, sidecarInfo.centroidOffset, kDimension);
    std::vector<float> expectedCentroid(kDimension, 0.0F);
    std::vector<SizeType> postingIDs;
    for (std::size_t batch = 0; batch < sidecarInfo.batchCount; ++batch)
    {
        const char* record = posting.data() + batch * recordBytes;
        BatchRecordHeader recordHeader;
        std::memcpy(&recordHeader, record, sizeof(recordHeader));
        const std::size_t expectedValid = std::min<std::size_t>(
            layout.batchSize,
            sidecarInfo.vectorCount - batch * layout.batchSize);
        BOOST_CHECK_EQUAL(recordHeader.validCount, expectedValid);
        BOOST_CHECK_EQUAL(recordHeader.reserved, 0U);
        for (std::size_t i = 0; i < recordHeader.validCount; ++i)
        {
            SizeType id = -1;
            std::memcpy(
                &id,
                record + sizeof(BatchRecordHeader) + i * sizeof(SizeType),
                sizeof(id));
            postingIDs.push_back(id);
            const auto* vector = reinterpret_cast<const float*>(
                vectors->GetVector(id));
            for (DimensionType dim = 0; dim < kDimension; ++dim)
            {
                expectedCentroid[dim] += vector[dim];
            }
        }
    }
    BOOST_REQUIRE_EQUAL(postingIDs.size(), sidecarInfo.vectorCount);
    for (DimensionType dim = 0; dim < kDimension; ++dim)
    {
        expectedCentroid[dim] /= static_cast<float>(postingIDs.size());
        BOOST_CHECK_SMALL(
            centroid[dim] - expectedCentroid[dim], 1.0e-5F);
    }

    const auto* headVector = reinterpret_cast<const float*>(
        vectors->GetVector(headIDs[chosenHead]));
    bool differsFromHead = false;
    for (DimensionType dim = 0; dim < kDimension; ++dim)
    {
        if (std::fabs(centroid[dim] - headVector[dim]) > 1.0e-5F)
        {
            differsFromHead = true;
            break;
        }
    }
    BOOST_CHECK(differsFromHead);

    const auto firstHeader = reinterpret_cast<const BatchRecordHeader*>(
        posting.data());
    std::vector<float> batchVectors(
        firstHeader->validCount * static_cast<std::size_t>(kDimension));
    for (std::size_t i = 0; i < firstHeader->validCount; ++i)
    {
        std::memcpy(
            batchVectors.data() + i * static_cast<std::size_t>(kDimension),
            vectors->GetVector(postingIDs[i]),
            sizeof(float) * static_cast<std::size_t>(kDimension));
    }
    std::vector<std::uint8_t> expectedBinary(layout.binaryBytes);
    std::vector<std::uint8_t> expectedExtended(layout.extendedBytes);
    std::size_t validCount = 0;
    BOOST_REQUIRE(
        quantizer->QuantizeSplitBatch(
            batchVectors.data(),
            firstHeader->validCount,
            centroid.data(),
            expectedBinary.data(),
            expectedExtended.data(),
            validCount) == ErrorCode::Success);
    BOOST_CHECK_EQUAL(validCount, firstHeader->validCount);
    const char* storedBinary = posting.data() + sizeof(BatchRecordHeader) +
        layout.batchSize * sizeof(SizeType);
    BOOST_CHECK_EQUAL(
        std::memcmp(
            expectedBinary.data(), storedBinary, expectedBinary.size()),
        0);
    const auto extendedBytes = ReadBinaryFile(extendedPath);
    BOOST_REQUIRE(
        sidecarInfo.dataOffset + expectedExtended.size() <=
        extendedBytes.size());
    BOOST_CHECK_EQUAL(
        std::memcmp(
            expectedExtended.data(),
            extendedBytes.data() + sidecarInfo.dataOffset,
            expectedExtended.size()),
        0);

    BOOST_REQUIRE(index->SaveIndex(relocatedDir) == ErrorCode::Success);
    BOOST_CHECK(std::filesystem::exists(
        relocatedDir + FolderSep + "SPTAGFullList.bin_0.rabitq.ext"));
    std::filesystem::remove_all(indexDir);
    std::filesystem::remove(modelPath);
    std::shared_ptr<VectorIndex> loaded;
    BOOST_REQUIRE(
        VectorIndex::LoadIndex(relocatedDir, loaded) == ErrorCode::Success);
    auto results = SearchOnce(loaded, vectors, postingIDs.front());
    BOOST_CHECK(ContainsVID(results, postingIDs.front()));
}

BOOST_AUTO_TEST_CASE(StaticSpannRaBitQBatchRejectsMissingOrCorruptSidecar)
{
    const std::string indexDir = "spann_rabitq_batch_sidecar_corruption";
    const std::string modelPath = "spann_rabitq_batch_sidecar_corruption.bin";
    ScopedCleanup cleanup({indexDir, modelPath});

    const auto vectors = MakeVectors();
    TrainAndSaveRaBitQModel(vectors, modelPath, DistCalcMethod::L2);
    auto index = MakeStaticSpannIndex(
        indexDir,
        modelPath,
        DistCalcMethod::L2,
        false,
        false,
        0,
        1,
        true);
    BOOST_REQUIRE(index->BuildIndex(vectors, nullptr) == ErrorCode::Success);
    BOOST_REQUIRE(index->SaveIndex(indexDir) == ErrorCode::Success);

    const std::string sidecar =
        indexDir + FolderSep + "SPTAGFullList.bin_0.rabitq.ext";
    const auto bytes = ReadBinaryFile(sidecar);
    std::filesystem::remove(sidecar);
    std::shared_ptr<VectorIndex> loaded;
    BOOST_CHECK(VectorIndex::LoadIndex(indexDir, loaded) != ErrorCode::Success);

    WriteBinaryFile(sidecar, bytes);
    auto truncated = bytes;
    truncated.resize(sizeof(BatchSidecarHeader));
    WriteBinaryFile(sidecar, truncated);
    loaded.reset();
    BOOST_CHECK(VectorIndex::LoadIndex(indexDir, loaded) != ErrorCode::Success);

    auto metadataCorruption = bytes;
    const auto* sidecarHeader =
        reinterpret_cast<const BatchSidecarHeader*>(metadataCorruption.data());
    auto* sidecarInfos = reinterpret_cast<BatchSidecarListInfo*>(
        metadataCorruption.data() + sizeof(BatchSidecarHeader));
    bool corruptedPayloadOffset = false;
    for (std::uint64_t i = 0; i < sidecarHeader->listCount; ++i)
    {
        if (sidecarInfos[i].vectorCount == 0)
        {
            continue;
        }
        sidecarInfos[i].centroidOffset = 0;
        sidecarInfos[i].dataOffset =
            static_cast<std::uint64_t>(kDimension) * sizeof(float);
        corruptedPayloadOffset = true;
        break;
    }
    BOOST_REQUIRE(corruptedPayloadOffset);
    WriteBinaryFile(sidecar, metadataCorruption);
    loaded.reset();
    BOOST_CHECK(VectorIndex::LoadIndex(indexDir, loaded) != ErrorCode::Success);
}

BOOST_AUTO_TEST_CASE(StaticSpannRaBitQBatchRequiresEveryConfiguredPartAndSidecar)
{
    const std::string indexDir = "spann_rabitq_batch_missing_part";
    const std::string modelPath = "spann_rabitq_batch_missing_part.bin";
    ScopedCleanup cleanup({indexDir, modelPath});

    const auto vectors = MakeVectors();
    TrainAndSaveRaBitQModel(vectors, modelPath, DistCalcMethod::L2);
    auto index = MakeStaticSpannIndex(
        indexDir,
        modelPath,
        DistCalcMethod::L2,
        false,
        false,
        0,
        1,
        true,
        0,
        3);
    BOOST_REQUIRE(index->BuildIndex(vectors, nullptr) == ErrorCode::Success);
    BOOST_REQUIRE(index->SaveIndex(indexDir) == ErrorCode::Success);

    const std::string part1 =
        indexDir + FolderSep + "SPTAGFullList.bin_0.part1";
    const std::string part2 =
        indexDir + FolderSep + "SPTAGFullList.bin_0.part2";
    const std::string part1Sidecar = part1 + ".rabitq.ext";
    BOOST_REQUIRE(std::filesystem::exists(part1));
    BOOST_REQUIRE(std::filesystem::exists(part2));
    BOOST_REQUIRE(std::filesystem::exists(part1Sidecar));

    const auto part2Bytes = ReadBinaryFile(part2);
    std::filesystem::remove(part2);
    std::shared_ptr<VectorIndex> loaded;
    BOOST_CHECK(VectorIndex::LoadIndex(indexDir, loaded) != ErrorCode::Success);

    WriteBinaryFile(part2, part2Bytes);
    const auto sidecarBytes = ReadBinaryFile(part1Sidecar);
    std::filesystem::remove(part1Sidecar);
    loaded.reset();
    BOOST_CHECK(VectorIndex::LoadIndex(indexDir, loaded) != ErrorCode::Success);

    WriteBinaryFile(part1Sidecar, sidecarBytes);
    loaded.reset();
    BOOST_REQUIRE(VectorIndex::LoadIndex(indexDir, loaded) == ErrorCode::Success);
}

BOOST_AUTO_TEST_CASE(StaticSpannRaBitQBatchReportsOnDemandSidecarIO)
{
    const std::string indexDir = "spann_rabitq_batch_sidecar_stats";
    const std::string modelPath = "spann_rabitq_batch_sidecar_stats.bin";
    ScopedCleanup cleanup({indexDir, modelPath});

    const auto vectors = MakeVectors();
    TrainAndSaveRaBitQModel(vectors, modelPath, DistCalcMethod::L2);
    auto index = MakeStaticSpannIndex(
        indexDir,
        modelPath,
        DistCalcMethod::L2,
        false,
        false,
        0,
        1,
        true);
    BOOST_REQUIRE(index->BuildIndex(vectors, nullptr) == ErrorCode::Success);

    auto spann = std::dynamic_pointer_cast<SPANN::Index<float>>(index);
    BOOST_REQUIRE(spann != nullptr);
    QueryResult headResults(vectors->GetVector(13), 16, false, false);
    BOOST_REQUIRE(
        spann->GetMemoryIndex()->SearchIndex(headResults) ==
        ErrorCode::Success);
    const auto mainReadUpperBound = CollectValidResults(headResults).size();
    BOOST_REQUIRE_GT(mainReadUpperBound, 0U);

    SPANN::SearchStats stats;
    QueryResult result(vectors->GetVector(13), kResultCount, false, false);
    BOOST_REQUIRE(
        spann->SearchIndex(result, &stats) == ErrorCode::Success);
    BOOST_CHECK_GT(
        stats.m_diskIOCount, static_cast<int>(mainReadUpperBound));
    BOOST_CHECK_GE(stats.m_diskAccessCount, stats.m_diskIOCount);

    stats.m_diskIOCount = 1000000;
    stats.m_diskAccessCount = 1000000;
    QueryResult repeated(vectors->GetVector(13), kResultCount, false, false);
    BOOST_REQUIRE(
        spann->SearchIndex(repeated, &stats) == ErrorCode::Success);
    BOOST_CHECK_GT(stats.m_diskIOCount, 0);
    BOOST_CHECK_LT(stats.m_diskIOCount, 1000000);
    BOOST_CHECK_GE(stats.m_diskAccessCount, stats.m_diskIOCount);
    BOOST_CHECK_LT(stats.m_diskAccessCount, 1000000);
}

BOOST_AUTO_TEST_CASE(StaticSpannRaBitQBatchRawRerankReturnsExactVectors)
{
    const std::string approximateDir = "spann_rabitq_batch_approximate";
    const std::string rerankDir = "spann_rabitq_batch_rerank";
    const std::string modelPath = "spann_rabitq_batch_rerank.bin";
    ScopedCleanup cleanup({approximateDir, rerankDir, modelPath});

    const auto vectors = MakeShiftedVectors(0.35F);
    TrainAndSaveRaBitQModel(vectors, modelPath, DistCalcMethod::L2);
    auto approximate = MakeStaticSpannIndex(
        approximateDir,
        modelPath,
        DistCalcMethod::L2,
        false,
        false,
        0,
        1,
        true);
    BOOST_REQUIRE(
        approximate->BuildIndex(vectors, nullptr) == ErrorCode::Success);

    QueryResult rejected(vectors->GetVector(11), kResultCount, false, true);
    BOOST_CHECK(
        approximate->SearchIndex(rejected) == ErrorCode::Undefined);
    auto approximateSpann =
        std::dynamic_pointer_cast<SPANN::Index<float>>(approximate);
    BOOST_REQUIRE(approximateSpann != nullptr);
    SPANN::SearchStats approximateStats;
    QueryResult approximateResult(
        vectors->GetVector(11), kResultCount, false, false);
    BOOST_REQUIRE(
        approximateSpann->SearchIndex(
            approximateResult, &approximateStats) == ErrorCode::Success);

    auto reranked = MakeStaticSpannIndex(
        rerankDir,
        modelPath,
        DistCalcMethod::L2,
        false,
        false,
        0,
        1,
        true,
        16);
    BOOST_REQUIRE(reranked->BuildIndex(vectors, nullptr) == ErrorCode::Success);
    BOOST_CHECK(std::filesystem::exists(
        rerankDir + FolderSep + "SPTAGFullList.bin_0.rabitq.raw"));

    auto rerankedSpann =
        std::dynamic_pointer_cast<SPANN::Index<float>>(reranked);
    BOOST_REQUIRE(rerankedSpann != nullptr);
    SPANN::SearchStats rerankedStats;
    QueryResult result(vectors->GetVector(11), kResultCount, false, true);
    BOOST_REQUIRE(
        rerankedSpann->SearchIndex(result, &rerankedStats) ==
        ErrorCode::Success);
    BOOST_CHECK_GT(
        rerankedStats.m_diskIOCount, approximateStats.m_diskIOCount);
    BOOST_CHECK_GT(
        rerankedStats.m_diskAccessCount,
        approximateStats.m_diskAccessCount);
    const auto* query = reinterpret_cast<const float*>(vectors->GetVector(11));
    for (int i = 0; i < result.GetResultNum(); ++i)
    {
        const BasicResult* entry = result.GetResult(i);
        if (entry == nullptr || entry->VID < 0) break;
        const auto* expected = reinterpret_cast<const float*>(
            vectors->GetVector(entry->VID));
        const float exact = COMMON::DistanceUtils::ComputeDistance(
            query, expected, kDimension, DistCalcMethod::L2);
        BOOST_CHECK_SMALL(entry->Dist - exact, 1.0e-4F);
        BOOST_REQUIRE_EQUAL(
            entry->Vec.Length(),
            sizeof(float) * static_cast<std::size_t>(kDimension));
        BOOST_CHECK_EQUAL(
            std::memcmp(
                entry->Vec.Data(),
                expected,
                sizeof(float) * static_cast<std::size_t>(kDimension)),
            0);
    }
    BOOST_REQUIRE(reranked->SaveIndex(rerankDir) == ErrorCode::Success);
    std::filesystem::remove(
        rerankDir + FolderSep + "SPTAGFullList.bin_0.rabitq.raw");
    std::shared_ptr<VectorIndex> missingRaw;
    BOOST_CHECK(
        VectorIndex::LoadIndex(rerankDir, missingRaw) != ErrorCode::Success);
}

BOOST_AUTO_TEST_CASE(StaticSpannRaBitQBatchSupportsInnerProductReload)
{
    const std::string indexDir = "spann_rabitq_batch_ip";
    const std::string relocatedDir = "spann_rabitq_batch_ip_relocated";
    const std::string modelPath = "spann_rabitq_batch_ip.bin";
    ScopedCleanup cleanup({indexDir, relocatedDir, modelPath});

    const auto vectors = MakeVectors();
    vectors->Normalize(1);
    TrainAndSaveRaBitQModel(
        vectors, modelPath, DistCalcMethod::InnerProduct);
    auto index = MakeStaticSpannIndex(
        indexDir,
        modelPath,
        DistCalcMethod::InnerProduct,
        false,
        false,
        0,
        1,
        true,
        16);
    BOOST_REQUIRE(index->BuildIndex(vectors, nullptr) == ErrorCode::Success);

    QueryResult result(vectors->GetVector(5), kResultCount, false, true);
    BOOST_REQUIRE(index->SearchIndex(result) == ErrorCode::Success);
    const auto* query = reinterpret_cast<const float*>(vectors->GetVector(5));
    for (int i = 0; i < result.GetResultNum(); ++i)
    {
        const BasicResult* entry = result.GetResult(i);
        if (entry == nullptr || entry->VID < 0) break;
        const float exact = COMMON::DistanceUtils::ComputeDistance(
            query,
            reinterpret_cast<const float*>(vectors->GetVector(entry->VID)),
            kDimension,
            DistCalcMethod::InnerProduct);
        BOOST_CHECK_SMALL(entry->Dist - exact, 1.0e-3F);
    }

    BOOST_REQUIRE(index->SaveIndex(relocatedDir) == ErrorCode::Success);
    std::filesystem::remove_all(indexDir);
    std::filesystem::remove(modelPath);
    std::shared_ptr<VectorIndex> loaded;
    BOOST_REQUIRE(
        VectorIndex::LoadIndex(relocatedDir, loaded) == ErrorCode::Success);
    QueryResult afterReload(
        vectors->GetVector(5), kResultCount, false, true);
    BOOST_REQUIRE(
        loaded->SearchIndex(afterReload) == ErrorCode::Success);
}

BOOST_AUTO_TEST_CASE(StaticSpannRaBitQBatchRawRerankSupportsCosine)
{
    const std::string indexDir = "spann_rabitq_batch_cosine";
    const std::string modelPath = "spann_rabitq_batch_cosine.bin";
    ScopedCleanup cleanup({indexDir, modelPath});

    const auto vectors = MakeShiftedVectors(0.2F);
    vectors->Normalize(1);
    TrainAndSaveRaBitQModel(
        vectors, modelPath, DistCalcMethod::Cosine);
    auto index = MakeStaticSpannIndex(
        indexDir,
        modelPath,
        DistCalcMethod::Cosine,
        false,
        false,
        0,
        1,
        true,
        16);
    BOOST_REQUIRE(index->BuildIndex(vectors, nullptr) == ErrorCode::Success);

    QueryResult result(vectors->GetVector(23), kResultCount, false, true);
    BOOST_REQUIRE(index->SearchIndex(result) == ErrorCode::Success);
    const auto* query = reinterpret_cast<const float*>(
        vectors->GetVector(23));
    for (int i = 0; i < result.GetResultNum(); ++i)
    {
        const BasicResult* entry = result.GetResult(i);
        if (entry == nullptr || entry->VID < 0) break;
        const float exact = COMMON::DistanceUtils::ComputeDistance(
            query,
            reinterpret_cast<const float*>(
                vectors->GetVector(entry->VID)),
            kDimension,
            DistCalcMethod::Cosine);
        BOOST_CHECK_SMALL(entry->Dist - exact, 1.0e-4F);
        BOOST_REQUIRE_EQUAL(
            entry->Vec.Length(),
            sizeof(float) * static_cast<std::size_t>(kDimension));
    }
}

BOOST_AUTO_TEST_CASE(StaticSpannRaBitQBatchUsesLayerLocalVectorsAndGlobalIDs)
{
    const std::string indexDir = "spann_rabitq_batch_local_global";
    const std::string modelPath = "spann_rabitq_batch_local_global.bin";
    ScopedCleanup cleanup({indexDir, modelPath});

    const auto vectors = MakeVectors();
    const auto quantizer = TrainAndSaveRaBitQModel(
        vectors, modelPath, DistCalcMethod::L2);
    auto index = MakeStaticSpannIndex(
        indexDir,
        modelPath,
        DistCalcMethod::L2,
        false,
        false,
        0,
        1,
        true);
    BOOST_REQUIRE(index->BuildIndex(vectors, nullptr) == ErrorCode::Success);

    auto spann = std::dynamic_pointer_cast<SPANN::Index<float>>(index);
    BOOST_REQUIRE(spann != nullptr);
    auto searcher = std::dynamic_pointer_cast<
        SPANN::ExtraStaticSearcher<float>>(spann->GetDiskIndex(0));
    BOOST_REQUIRE(searcher != nullptr);

    const std::vector<SizeType> sourceIDs = {4, 17, 29};
    const auto layerVectors = MakeSubsetVectors(vectors, sourceIDs);
    COMMON::Dataset<SizeType> localToGlobal(3, 1, 4, 3);
    const std::array<SizeType, 3> globalIDs = {101, 205, 309};
    for (std::size_t i = 0; i < globalIDs.size(); ++i)
    {
        *localToGlobal[i] = globalIDs[i];
    }

    SPANN::Selection selections(globalIDs.size(), indexDir);
    for (std::size_t i = 0; i < globalIDs.size(); ++i)
    {
        selections.m_selections[i].node = 0;
        selections.m_selections[i].distance = static_cast<float>(i);
        selections.m_selections[i].tonode = static_cast<SizeType>(i);
    }
    const std::string posting = searcher->GetPostingListFullData(
        0,
        globalIDs.size(),
        selections,
        layerVectors,
        localToGlobal);
    const auto layout = quantizer->GetSplitBatchLayout();
    BOOST_REQUIRE_EQUAL(
        posting.size(),
        sizeof(BatchRecordHeader) +
            layout.batchSize * sizeof(SizeType) + layout.binaryBytes);

    BatchRecordHeader header;
    std::memcpy(&header, posting.data(), sizeof(header));
    BOOST_CHECK_EQUAL(header.validCount, globalIDs.size());
    std::vector<float> centroid(kDimension, 0.0F);
    std::vector<float> contiguous(globalIDs.size() * kDimension);
    for (std::size_t i = 0; i < globalIDs.size(); ++i)
    {
        SizeType storedID = -1;
        std::memcpy(
            &storedID,
            posting.data() + sizeof(header) + i * sizeof(SizeType),
            sizeof(storedID));
        BOOST_CHECK_EQUAL(storedID, globalIDs[i]);
        const auto* source = reinterpret_cast<const float*>(
            layerVectors->GetVector(static_cast<SizeType>(i)));
        std::copy(
            source,
            source + kDimension,
            contiguous.begin() + i * kDimension);
        for (DimensionType dim = 0; dim < kDimension; ++dim)
        {
            centroid[dim] += source[dim];
        }
    }
    const float centroidScale =
        1.0F / static_cast<float>(globalIDs.size());
    for (float& value : centroid)
    {
        value *= centroidScale;
    }
    std::vector<std::uint8_t> expectedBinary(layout.binaryBytes);
    std::vector<std::uint8_t> expectedExtended(layout.extendedBytes);
    std::size_t validCount = 0;
    BOOST_REQUIRE(
        quantizer->QuantizeSplitBatch(
            contiguous.data(),
            globalIDs.size(),
            centroid.data(),
            expectedBinary.data(),
            expectedExtended.data(),
            validCount) == ErrorCode::Success);
    BOOST_CHECK_EQUAL(validCount, globalIDs.size());
    BOOST_CHECK_EQUAL(
        std::memcmp(
            expectedBinary.data(),
            posting.data() + sizeof(header) +
                layout.batchSize * sizeof(SizeType),
            expectedBinary.size()),
        0);
}

BOOST_AUTO_TEST_CASE(StaticSpannRaBitQBatchTruthInstrumentationReadsIDLanes)
{
    const std::string indexDir = "spann_rabitq_batch_truth_lanes";
    const std::string modelPath = "spann_rabitq_batch_truth_lanes.bin";
    ScopedCleanup cleanup({indexDir, modelPath});

    const auto vectors = MakeVectors();
    TrainAndSaveRaBitQModel(vectors, modelPath, DistCalcMethod::L2);
    auto index = MakeStaticSpannIndex(
        indexDir,
        modelPath,
        DistCalcMethod::L2,
        false,
        false,
        0,
        1,
        true);
    BOOST_REQUIRE(
        index->BuildIndex(vectors, nullptr) == ErrorCode::Success);

    auto spann = std::dynamic_pointer_cast<SPANN::Index<float>>(index);
    BOOST_REQUIRE(spann != nullptr);
    std::vector<SizeType> headIDs;
    BOOST_REQUIRE(
        spann->GetHeadIndexMapping(1, headIDs) == ErrorCode::Success);
    const std::set<SizeType> headIDSet(headIDs.begin(), headIDs.end());
    std::set<SizeType> truth;
    for (SizeType id = 0; id < vectors->Count(); ++id)
    {
        if (headIDSet.count(id) == 0)
        {
            truth.insert(id);
        }
    }
    BOOST_REQUIRE(!truth.empty());

    QueryResult headResults(vectors->GetVector(9), 16, false, false);
    BOOST_REQUIRE(
        spann->GetMemoryIndex()->SearchIndex(headResults) ==
        ErrorCode::Success);
    std::map<SizeType, std::set<SizeType>> found;
    BOOST_REQUIRE(
        spann->DebugSearchDiskIndex(
            headResults, 16, 16, nullptr, &truth, &found) ==
        ErrorCode::Success);

    size_t foundTruthCount = 0;
    for (const auto& posting : found)
    {
        for (SizeType id : posting.second)
        {
            BOOST_CHECK(truth.count(id) != 0);
            ++foundTruthCount;
        }
    }
    BOOST_CHECK_GT(foundTruthCount, 0U);
}

BOOST_AUTO_TEST_CASE(StaticSpannRaBitQBatchBuildsAndReloadsTwoLayers)
{
    const std::string indexDir = "spann_rabitq_batch_two_layers";
    const std::string modelPath = "spann_rabitq_batch_two_layers.bin";
    ScopedCleanup cleanup({indexDir, modelPath});

    const auto vectors = MakeVectors();
    TrainAndSaveRaBitQModel(vectors, modelPath, DistCalcMethod::L2);
    auto index = MakeStaticSpannIndex(
        indexDir,
        modelPath,
        DistCalcMethod::L2,
        false,
        false,
        0,
        2,
        true);
    BOOST_REQUIRE(index->BuildIndex(vectors, nullptr) == ErrorCode::Success);
    BOOST_REQUIRE(index->SaveIndex(indexDir) == ErrorCode::Success);

    std::shared_ptr<VectorIndex> loaded;
    BOOST_REQUIRE(VectorIndex::LoadIndex(indexDir, loaded) == ErrorCode::Success);
    const auto results = SearchOnce(loaded, vectors, 11);
    BOOST_CHECK(ContainsVID(results, 11));
}

BOOST_AUTO_TEST_CASE(StaticSpannRaBitQPostingRejectsIncompatibleConfigs)
{
    const auto vectors = MakeVectors();

    {
        const std::string indexDir = "spann_rabitq_static_posting_reject_metric";
        const std::string modelPath = "spann_rabitq_static_posting_reject_metric.bin";
        ScopedCleanup cleanup({indexDir, modelPath});
        TrainAndSaveRaBitQModel(vectors, modelPath, DistCalcMethod::Cosine);
        auto index = MakeStaticSpannIndex(indexDir, modelPath, DistCalcMethod::L2);
        BOOST_CHECK(index->BuildIndex(vectors, nullptr) == ErrorCode::Fail);
    }

    {
        const std::string indexDir = "spann_rabitq_static_posting_reject_delta";
        const std::string modelPath = "spann_rabitq_static_posting_reject_delta.bin";
        ScopedCleanup cleanup({indexDir, modelPath});
        TrainAndSaveRaBitQModel(vectors, modelPath, DistCalcMethod::L2);
        auto index = MakeStaticSpannIndex(indexDir, modelPath, DistCalcMethod::L2, true, false, 0);
        BOOST_CHECK(index->BuildIndex(vectors, nullptr) == ErrorCode::Fail);
    }

    {
        const std::string indexDir = "spann_rabitq_static_posting_reject_global";
        const std::string modelPath = "spann_rabitq_static_posting_reject_global.bin";
        ScopedCleanup cleanup({indexDir, modelPath});
        TrainAndSaveRaBitQModel(vectors, modelPath, DistCalcMethod::L2);
        auto index = MakeStaticSpannIndex(indexDir, modelPath, DistCalcMethod::L2);
        index->SetParameter("QuantizerFilePath", modelPath, "Base");
        BOOST_CHECK(index->BuildIndex(vectors, nullptr) == ErrorCode::Fail);
    }

    {
        const std::string indexDir = "spann_rabitq_static_posting_reject_rerank";
        const std::string modelPath = "spann_rabitq_static_posting_reject_rerank.bin";
        ScopedCleanup cleanup({indexDir, modelPath});
        TrainAndSaveRaBitQModel(vectors, modelPath, DistCalcMethod::L2);
        auto index = MakeStaticSpannIndex(indexDir, modelPath, DistCalcMethod::L2, false, false, 4);
        BOOST_CHECK(index->BuildIndex(vectors, nullptr) == ErrorCode::Fail);
    }

    {
        const std::string indexDir = "spann_rabitq_batch_reject_compression";
        const std::string modelPath =
            "spann_rabitq_batch_reject_compression.bin";
        ScopedCleanup cleanup({indexDir, modelPath});
        TrainAndSaveRaBitQModel(vectors, modelPath, DistCalcMethod::L2);
        auto index = MakeStaticSpannIndex(
            indexDir,
            modelPath,
            DistCalcMethod::L2,
            false,
            false,
            0,
            1,
            true);
        index->SetParameter(
            "EnableDataCompression", "true", "BuildSSDIndex");
        BOOST_CHECK(index->BuildIndex(vectors, nullptr) == ErrorCode::Fail);
    }

    {
        const std::string indexDir = "spann_rabitq_batch_reject_rearrange";
        const std::string modelPath =
            "spann_rabitq_batch_reject_rearrange.bin";
        ScopedCleanup cleanup({indexDir, modelPath});
        TrainAndSaveRaBitQModel(vectors, modelPath, DistCalcMethod::L2);
        auto index = MakeStaticSpannIndex(
            indexDir,
            modelPath,
            DistCalcMethod::L2,
            false,
            true,
            0,
            1,
            true);
        BOOST_CHECK(index->BuildIndex(vectors, nullptr) == ErrorCode::Fail);
    }

    {
        const std::string indexDir = "spann_rabitq_batch_reject_short_rerank";
        const std::string modelPath =
            "spann_rabitq_batch_reject_short_rerank.bin";
        ScopedCleanup cleanup({indexDir, modelPath});
        TrainAndSaveRaBitQModel(vectors, modelPath, DistCalcMethod::L2);
        auto index = MakeStaticSpannIndex(
            indexDir,
            modelPath,
            DistCalcMethod::L2,
            false,
            false,
            0,
            1,
            true,
            4);
        BOOST_CHECK(index->BuildIndex(vectors, nullptr) == ErrorCode::Fail);
    }
}

BOOST_AUTO_TEST_SUITE_END()
