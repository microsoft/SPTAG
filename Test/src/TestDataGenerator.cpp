#include "inc/TestDataGenerator.h"
#include <cstring>
#include <iostream>
#include <sstream>

using namespace SPTAG;

namespace TestUtils
{

template <typename T>
TestDataGenerator<T>::TestDataGenerator(int n, int q, int m, int k, std::string distMethod, int a, bool isRandom,
                                        std::string vectorPath, std::string queryPath)
    : m_n(n), m_a((a == 0)? n : a), m_q(q), m_m(m), m_k(k), m_distMethod(std::move(distMethod)), m_isRandom(isRandom),
      m_vectorPath(vectorPath), m_queryPath(queryPath)
{
}

template <typename T>
void TestDataGenerator<T>::Run(std::shared_ptr<VectorSet> &vecset, std::shared_ptr<MetadataSet> &metaset,
                               std::shared_ptr<VectorSet> &queryset, std::shared_ptr<VectorSet> &truth,
                               std::shared_ptr<VectorSet> &addvecset, std::shared_ptr<MetadataSet> &addmetaset,
                               std::shared_ptr<VectorSet> &addtruth)
{
    LoadOrGenerateBase(vecset, metaset);
    LoadOrGenerateQuery(queryset);
    LoadOrGenerateAdd(addvecset, addmetaset);
    LoadOrGenerateTruth("perftest_truth." + m_distMethod, vecset, queryset, truth, true);
    LoadOrGenerateTruth("perftest_addtruth." + m_distMethod, CombineVectorSets(vecset, addvecset), queryset, addtruth,
                        true);
}

template <typename T>
void TestDataGenerator<T>::RunBatches(std::shared_ptr<SPTAG::VectorSet>& vecset,
    std::shared_ptr<SPTAG::MetadataSet>& metaset,
    std::shared_ptr<SPTAG::VectorSet>& addvecset, std::shared_ptr<SPTAG::MetadataSet>& addmetaset,
    std::shared_ptr<SPTAG::VectorSet>& queryset, int base, int batchinsert, int batchdelete, int batches,
    std::shared_ptr<SPTAG::VectorSet>& truths)
{
    LoadOrGenerateBase(vecset, metaset);
    LoadOrGenerateQuery(queryset);
    LoadOrGenerateAdd(addvecset, addmetaset);
    LoadOrGenerateBatchTruth("perftest_batchtruth." + m_distMethod, CombineVectorSets(vecset, addvecset), queryset,
                             truths, base, batchinsert, batchdelete, batches, true);
}

template <typename T>
void TestDataGenerator<T>::RunLargeBatches(std::string &vecset, std::string &metaset, std::string &metaidx,
                                           std::string &addset, std::string &addmetaset, std::string &addmetaidx,
                                           std::string &queryset, int base, int batchinsert, int batchdelete,
                                           int batches, std::string &truth, bool generateTruth)
{
    vecset = "perftest_vector.bin";
    metaset = "perftest_meta.bin";
    metaidx = "perftest_metaidx.bin";
    addset = "perftest_addvector.bin";
    addmetaset = "perftest_addmeta.bin";
    addmetaidx = "perftest_addmetaidx.bin";
    queryset = "perftest_query.bin";
    truth = "perftest_batchtruth." + m_distMethod;
    std::string empty;

    GenerateVectorSet(vecset, metaset, metaidx, m_vectorPath, 0, m_n);
    GenerateVectorSet(queryset, empty, empty, m_queryPath, 0, m_q);
    GenerateVectorSet(addset, addmetaset, addmetaidx, m_vectorPath, m_n, m_a);
    if (generateTruth)
    {
        GenerateBatchTruth(truth, vecset, addset, queryset, base, batchinsert, batchdelete, batches, true);
    }
}

template<typename T>
void TestDataGenerator<T>::GenerateVectorSet(std::string & pvecset, std::string & pmetaset, std::string & pmetaidx, std::string& pvecPath, SPTAG::SizeType start, int count)
{
    if (!fileexists(pvecset.c_str()))
    {
        std::shared_ptr<SPTAG::VectorSet> vecset;
        if (m_isRandom)
        {
            vecset = GenerateRandomVectorSet(count, m_m);
        }
        else
        {
            vecset = GenerateLoadVectorSet(count, m_m, pvecPath, start);
        }
        vecset->Save(pvecset);
    }

    if (pmetaset.empty() || pmetaidx.empty())
        return;

    if (!fileexists(pmetaset.c_str()) || !fileexists(pmetaidx.c_str()))
    {
        auto metaset = GenerateMetadataSet(count, start);
        metaset->SaveMetadata(pmetaset, pmetaidx);
    }
}

template <typename T>
void TestDataGenerator<T>::GenerateBatchTruth(const std::string &filename, std::string &pvecset, std::string &paddset, std::string &pqueryset, int base,
                                                    int batchinsert, int batchdelete, int batches, bool normalize)
{
    if (fileexists(filename.c_str()))
        return;

    auto vectorOptions = std::shared_ptr<Helper::ReaderOptions>(new Helper::ReaderOptions(GetEnumValueType<T>(), m_m, VectorFileType::DEFAULT));
    auto vectorReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
    if (!fileexists(pvecset.c_str()) || ErrorCode::Success != vectorReader->LoadFile(pvecset))
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot find or load %s. Using random generation!\n", pvecset.c_str());
        return;
    }
    auto vecset = vectorReader->GetVectorSet();

    auto queryReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
    if (!fileexists(pqueryset.c_str()) || ErrorCode::Success != queryReader->LoadFile(pqueryset))
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot find or load %s. Using random generation!\n",
                     pqueryset.c_str());
        return;
    }
    auto queryset = queryReader->GetVectorSet();

    auto addReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
    if (!fileexists(paddset.c_str()) || ErrorCode::Success != addReader->LoadFile(paddset))
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot find or load %s. Using random generation!\n",
                     paddset.c_str());
        return;
    }
    auto addset = addReader->GetVectorSet();


    DistCalcMethod distMethod;
    Helper::Convert::ConvertStringTo(m_distMethod.c_str(), distMethod);
    if (normalize && distMethod == DistCalcMethod::Cosine)
    {
        COMMON::Utils::BatchNormalize((T *)vecset->GetData(), vecset->Count(), vecset->Dimension(),
                                      COMMON::Utils::GetBase<T>(), 5);
        COMMON::Utils::BatchNormalize((T *)addset->GetData(), addset->Count(), addset->Dimension(),
                                      COMMON::Utils::GetBase<T>(), 5);
    }

    ByteArray tru = ByteArray::Alloc(2 * sizeof(float) * (batches + 1) * queryset->Count() * m_k);
    int distbase = (batches + 1) * queryset->Count() * m_k;
    int start = 0;
    int end = base;
    int maxthreads = std::thread::hardware_concurrency();
    for (int iter = 0; iter < batches + 1; iter++)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Generating groundtruth for batch %d\n", iter);
        std::vector<std::thread> mythreads;
        mythreads.reserve(maxthreads);
        std::atomic_size_t sent(0);
        for (int tid = 0; tid < maxthreads; tid++)
        {
            mythreads.emplace_back([&, tid]() {
                size_t i = 0;
                while (true)
                {
                    i = sent.fetch_add(1);
                    if (i < queryset->Count())
                    {
                        SizeType *neighbors = ((SizeType *)tru.Data()) + iter * (queryset->Count() * m_k) + i * m_k;
                        float *dists = ((float *)tru.Data()) + distbase + iter * (queryset->Count() * m_k) + i * m_k;
                        COMMON::QueryResultSet<T> res((const T *)queryset->GetVector(i), m_k);
                        for (SizeType j = start; j < end; ++j)
                        {
                            float dist = MaxDist;
                            if (j < vecset->Count())
                                dist = COMMON::DistanceUtils::ComputeDistance(res.GetTarget(), 
                                    reinterpret_cast<T *>(vecset->GetVector(j)), m_m, distMethod);
                            else
                                dist = COMMON::DistanceUtils::ComputeDistance(res.GetTarget(), 
                                    reinterpret_cast<T *>(addset->GetVector(j - vecset->Count())), m_m, distMethod);

                            res.AddPoint(j, dist);
                        }
                        res.SortResult();
                        for (int j = 0; j < m_k; ++j)
                        {
                            neighbors[j] = res.GetResult(j)->VID;
                            dists[j] = res.GetResult(j)->Dist;
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
        start += batchdelete;
        end += batchinsert;
    }
    auto truths = std::make_shared<BasicVectorSet>(tru, GetEnumValueType<float>(), m_k, 2 * (batches + 1) * queryset->Count());
    truths->Save(filename);
}

template <typename T>
void TestDataGenerator<T>::LoadOrGenerateBase(std::shared_ptr<VectorSet> &vecset, std::shared_ptr<MetadataSet> &metaset)
{
    if (fileexists("perftest_vector.bin") && fileexists("perftest_meta.bin") && fileexists("perftest_metaidx.bin"))
    {
        auto reader = LoadReader("perftest_vector.bin");
        vecset = reader->GetVectorSet();
        metaset.reset(
            new MemMetadataSet("perftest_meta.bin", "perftest_metaidx.bin", vecset->Count() * 2, MaxSize, 10));
    }
    else
    {
        if (m_isRandom)
        {
            vecset = GenerateRandomVectorSet(m_n, m_m);
        }
        else
        {
            vecset = GenerateLoadVectorSet(m_n, m_m, m_vectorPath, 0);
        }
        vecset->Save("perftest_vector.bin");

        metaset = GenerateMetadataSet(m_n, 0);
        metaset->SaveMetadata("perftest_meta.bin", "perftest_metaidx.bin");
    }
}

template <typename T> void TestDataGenerator<T>::LoadOrGenerateQuery(std::shared_ptr<VectorSet> &queryset)
{
    if (fileexists("perftest_query.bin"))
    {
        auto reader = LoadReader("perftest_query.bin");
        queryset = reader->GetVectorSet();
    }
    else
    {
        if (m_isRandom)
        {
            queryset = GenerateRandomVectorSet(m_q, m_m);
        }
        else
        {
            queryset = GenerateLoadVectorSet(m_q, m_m, m_queryPath, 0);
        }
        queryset->Save("perftest_query.bin");
    }
}

template <typename T>
void TestDataGenerator<T>::LoadOrGenerateAdd(std::shared_ptr<VectorSet> &addvecset,
                                             std::shared_ptr<MetadataSet> &addmetaset)
{
    if (fileexists("perftest_addvector.bin") && fileexists("perftest_addmeta.bin") &&
        fileexists("perftest_addmetaidx.bin"))
    {
        auto reader = LoadReader("perftest_addvector.bin");
        addvecset = reader->GetVectorSet();
        addmetaset.reset(
            new MemMetadataSet("perftest_addmeta.bin", "perftest_addmetaidx.bin", addvecset->Count() * 2, MaxSize, 10));
    }
    else
    {
        if (m_isRandom)
        {
            addvecset = GenerateRandomVectorSet(m_a, m_m);
        }
        else
        {
            addvecset = GenerateLoadVectorSet(m_a, m_m, m_vectorPath, m_n);
        }
        addvecset->Save("perftest_addvector.bin");

        addmetaset = GenerateMetadataSet(m_a, m_n);
        addmetaset->SaveMetadata("perftest_addmeta.bin", "perftest_addmetaidx.bin");
    }
}

template <typename T>
void TestDataGenerator<T>::LoadOrGenerateTruth(const std::string &filename, std::shared_ptr<VectorSet> vecset,
                                               std::shared_ptr<VectorSet> queryset, std::shared_ptr<VectorSet> &truth,
                                               bool normalize)
{
    if (fileexists(filename.c_str()))
    {
        auto opts = std::make_shared<Helper::ReaderOptions>(GetEnumValueType<float>(), m_m, VectorFileType::DEFAULT);
        auto reader = Helper::VectorSetReader::CreateInstance(opts);
        if (ErrorCode::Success != reader->LoadFile(filename))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read file %s\n", filename.c_str());
            exit(1);
        }
        truth = reader->GetVectorSet();
        return;
    }

    DistCalcMethod distMethod;
    Helper::Convert::ConvertStringTo(m_distMethod.c_str(), distMethod);
    if (normalize && distMethod == DistCalcMethod::Cosine)
    {
        COMMON::Utils::BatchNormalize((T *)vecset->GetData(), vecset->Count(), vecset->Dimension(),
                                      COMMON::Utils::GetBase<T>(), 5);
    }

    ByteArray tru = ByteArray::Alloc(sizeof(float) * queryset->Count() * m_k);
    
    std::vector<std::thread> mythreads;
    int maxthreads = std::thread::hardware_concurrency();
    mythreads.reserve(maxthreads);
    std::atomic_size_t sent(0);
    for (int tid = 0; tid < maxthreads; tid++)
    {
        mythreads.emplace_back([&, tid]() {
            size_t i = 0;
            while (true)
            {
                i = sent.fetch_add(1);
                if (i < queryset->Count())
                {
                    SizeType *neighbors = ((SizeType *)tru.Data()) + i * m_k;
                    COMMON::QueryResultSet<T> res((const T *)queryset->GetVector(i), m_k);
                    for (SizeType j = 0; j < vecset->Count(); ++j)
                    {
                        float dist = COMMON::DistanceUtils::ComputeDistance(
                            res.GetTarget(), reinterpret_cast<T *>(vecset->GetVector(j)), m_m, distMethod);
                        res.AddPoint(j, dist);
                    }
                    res.SortResult();
                    for (int j = 0; j < m_k; ++j)
                        neighbors[j] = res.GetResult(j)->VID;
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

    truth = std::make_shared<BasicVectorSet>(tru, GetEnumValueType<float>(), m_k, queryset->Count());
    truth->Save(filename);
}

template <typename T>
void TestDataGenerator<T>::LoadOrGenerateBatchTruth(const std::string &filename,
                                                    std::shared_ptr<SPTAG::VectorSet> vecset,
                                                    std::shared_ptr<SPTAG::VectorSet> queryset,
                                                    std::shared_ptr<SPTAG::VectorSet> &truths, int base,
                                                    int batchinsert, int batchdelete, int batches, bool normalize)
{
    if (fileexists(filename.c_str()))
    {
        auto opts = std::make_shared<Helper::ReaderOptions>(GetEnumValueType<float>(), m_m, VectorFileType::DEFAULT);
        auto reader = Helper::VectorSetReader::CreateInstance(opts);
        if (ErrorCode::Success != reader->LoadFile(filename))
        {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read file %s\n", filename.c_str());
            exit(1);
        }
        truths = reader->GetVectorSet();
        return;
    }

    DistCalcMethod distMethod;
    Helper::Convert::ConvertStringTo(m_distMethod.c_str(), distMethod);
    if (normalize && distMethod == DistCalcMethod::Cosine)
    {
        COMMON::Utils::BatchNormalize((T *)vecset->GetData(), vecset->Count(), vecset->Dimension(),
                                      COMMON::Utils::GetBase<T>(), 5);
    }

    ByteArray tru = ByteArray::Alloc(sizeof(float) * (batches + 1) * queryset->Count() * m_k);
    int start = 0;
    int end = base;
    int maxthreads = std::thread::hardware_concurrency();
    for (int iter = 0; iter < batches + 1; iter++)
    {
        std::vector<std::thread> mythreads;
        mythreads.reserve(maxthreads);
        std::atomic_size_t sent(0);
        for (int tid = 0; tid < maxthreads; tid++)
        {
            mythreads.emplace_back([&, tid]() {
                size_t i = 0;
                while (true)
                {
                    i = sent.fetch_add(1);
                    if (i < queryset->Count())
                    {
                        SizeType *neighbors = ((SizeType *)tru.Data()) + iter * (queryset->Count() * m_k) + i * m_k;
                        COMMON::QueryResultSet<T> res((const T *)queryset->GetVector(i), m_k);
                        for (SizeType j = start; j < end; ++j)
                        {
                            float dist = COMMON::DistanceUtils::ComputeDistance(
                                res.GetTarget(), reinterpret_cast<T *>(vecset->GetVector(j)), m_m, distMethod);
                            res.AddPoint(j, dist);
                        }
                        res.SortResult();
                        for (int j = 0; j < m_k; ++j)
                            neighbors[j] = res.GetResult(j)->VID;
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
        start += batchdelete;
        end += batchinsert;
    }
    truths = std::make_shared<BasicVectorSet>(tru, GetEnumValueType<float>(), m_k, (batches + 1) * queryset->Count());
    truths->Save(filename);
}

template <typename T>
std::shared_ptr<VectorSet> TestDataGenerator<T>::GenerateRandomVectorSet(SizeType count, DimensionType dim)
{
    ByteArray vec = ByteArray::Alloc(sizeof(T) * count * dim);
    for (SizeType i = 0; i < count * dim; ++i)
    {
        ((T *)vec.Data())[i] = (T)COMMON::Utils::rand(127, -127);
    }
    return std::make_shared<BasicVectorSet>(vec, GetEnumValueType<T>(), dim, count);
}

template <typename T>
std::shared_ptr<MetadataSet> TestDataGenerator<T>::GenerateMetadataSet(SizeType count, SizeType offsetBase)
{
    ByteArray meta = ByteArray::Alloc(count * 10);
    ByteArray metaoffset = ByteArray::Alloc((count + 1) * sizeof(std::uint64_t));
    std::uint64_t offset = 0;
    for (SizeType i = 0; i < count; i++)
    {
        ((std::uint64_t *)metaoffset.Data())[i] = offset;
        std::string id = std::to_string(i + offsetBase);
        std::memcpy(meta.Data() + offset, id.c_str(), id.length());
        offset += id.length();
    }
    ((std::uint64_t *)metaoffset.Data())[count] = offset;
    return std::make_shared<MemMetadataSet>(meta, metaoffset, count, count * 2, MaxSize, 10);
}

template <typename T>
std::shared_ptr<SPTAG::VectorSet> TestDataGenerator<T>::GenerateLoadVectorSet(SPTAG::SizeType count,
                                                                              SPTAG::DimensionType dim,
                                                                              std::string path, SPTAG::SizeType start)
{
    VectorFileType fileType = VectorFileType::DEFAULT;
    if (path.find(".fvecs") != std::string::npos || path.find(".ivecs") != std::string::npos)
    {
        fileType = VectorFileType::XVEC;
    }
    auto vectorOptions =
        std::shared_ptr<Helper::ReaderOptions>(new Helper::ReaderOptions(GetEnumValueType<T>(), dim, fileType));
    auto vectorReader = Helper::VectorSetReader::CreateInstance(vectorOptions);

    if (!fileexists(path.c_str()) || ErrorCode::Success != vectorReader->LoadFile(path))
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Cannot find or load %s. Using random generation!\n", path.c_str());
        return GenerateRandomVectorSet(count, dim);
    }

    auto allVectors = vectorReader->GetVectorSet(start, start + count);
    if (allVectors->Count() < count)
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error,
                        "Cannot get %d vectors start from %d. Using random generation!\n", count, start);
        return GenerateRandomVectorSet(count, dim);
    }

    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Load %d vectors start from %d\n", count, start);
    return allVectors;
}

template <typename T>
std::shared_ptr<VectorSet> TestDataGenerator<T>::CombineVectorSets(std::shared_ptr<VectorSet> base,
                                                                   std::shared_ptr<VectorSet> add)
{
    ByteArray vec = ByteArray::Alloc(sizeof(T) * (base->Count() + add->Count()) * m_m);
    memcpy(vec.Data(), base->GetData(), sizeof(T) * (base->Count()) * m_m);
    memcpy(vec.Data() + sizeof(T) * (base->Count()) * m_m, add->GetData(), sizeof(T) * (add->Count()) * m_m);
    return std::make_shared<BasicVectorSet>(vec, GetEnumValueType<T>(), m_m, base->Count() + add->Count());
}

template <typename T>
std::shared_ptr<Helper::VectorSetReader> TestDataGenerator<T>::LoadReader(const std::string &filename)
{
    auto opts = std::make_shared<Helper::ReaderOptions>(GetEnumValueType<T>(), m_m, VectorFileType::DEFAULT);
    auto reader = Helper::VectorSetReader::CreateInstance(opts);
    if (ErrorCode::Success != reader->LoadFile(filename))
    {
        SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read file %s\n", filename.c_str());
        exit(1);
    }
    return reader;
}

// Explicit instantiation
template class TestDataGenerator<int8_t>;
template class TestDataGenerator<uint8_t>;
template class TestDataGenerator<int16_t>;
template class TestDataGenerator<float>;
} // namespace TestUtils
