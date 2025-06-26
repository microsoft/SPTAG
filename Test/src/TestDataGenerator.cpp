#include "inc/TestDataGenerator.h"
#include <cstring>
#include <sstream>
#include <iostream>

using namespace SPTAG;

namespace TestUtils {

    template <typename T>
    TestDataGenerator<T>::TestDataGenerator(int n, int q, int m, int k, std::string distMethod)
        : m_n(n), m_q(q), m_m(m), m_k(k), m_distMethod(std::move(distMethod)) {
    }

    template <typename T>
    void TestDataGenerator<T>::Run(std::shared_ptr<VectorSet>& vecset,
        std::shared_ptr<MetadataSet>& metaset,
        std::shared_ptr<VectorSet>& queryset,
        std::shared_ptr<VectorSet>& truth,
        std::shared_ptr<VectorSet>& addvecset,
        std::shared_ptr<MetadataSet>& addmetaset,
        std::shared_ptr<VectorSet>& addtruth) {
        LoadOrGenerateBase(vecset, metaset);
        LoadOrGenerateQuery(queryset);
        LoadOrGenerateAdd(addvecset, addmetaset);
        LoadOrGenerateTruth("perftest_truth." + m_distMethod, vecset, queryset, truth, true);
        LoadOrGenerateTruth("perftest_addtruth." + m_distMethod, CombineVectorSets(vecset, addvecset), queryset, addtruth, true);
    }

    template <typename T>
    void TestDataGenerator<T>::LoadOrGenerateBase(std::shared_ptr<VectorSet>& vecset,
        std::shared_ptr<MetadataSet>& metaset) {
        if (fileexists("perftest_vector.bin") && fileexists("perftest_meta.bin") && fileexists("perftest_metaidx.bin")) {
            auto reader = LoadReader("perftest_vector.bin");
            vecset = reader->GetVectorSet();
            metaset.reset(new MemMetadataSet("perftest_meta.bin", "perftest_metaidx.bin", vecset->Count() * 2, MaxSize, 10));
        }
        else {
            vecset = GenerateRandomVectorSet(m_n, m_m);
            vecset->Save("perftest_vector.bin");

            metaset = GenerateMetadataSet(m_n, 0);
            metaset->SaveMetadata("perftest_meta.bin", "perftest_metaidx.bin");
        }
    }

    template <typename T>
    void TestDataGenerator<T>::LoadOrGenerateQuery(std::shared_ptr<VectorSet>& queryset) {
        if (fileexists("perftest_query.bin")) {
            auto reader = LoadReader("perftest_query.bin");
            queryset = reader->GetVectorSet();
        }
        else {
            queryset = GenerateRandomVectorSet(m_q, m_m);
            queryset->Save("perftest_query.bin");
        }
    }

    template <typename T>
    void TestDataGenerator<T>::LoadOrGenerateAdd(std::shared_ptr<VectorSet>& addvecset,
        std::shared_ptr<MetadataSet>& addmetaset) {
        if (fileexists("perftest_addvector.bin") && fileexists("perftest_addmeta.bin") && fileexists("perftest_addmetaidx.bin")) {
            auto reader = LoadReader("perftest_addvector.bin");
            addvecset = reader->GetVectorSet();
            addmetaset.reset(new MemMetadataSet("perftest_addmeta.bin", "perftest_addmetaidx.bin", addvecset->Count() * 2, MaxSize, 10));
        }
        else {
            addvecset = GenerateRandomVectorSet(m_n, m_m);
            addvecset->Save("perftest_addvector.bin");

            addmetaset = GenerateMetadataSet(m_n, m_n);
            addmetaset->SaveMetadata("perftest_addmeta.bin", "perftest_addmetaidx.bin");
        }
    }

    template <typename T>
    void TestDataGenerator<T>::LoadOrGenerateTruth(const std::string& filename,
        std::shared_ptr<VectorSet> vecset,
        std::shared_ptr<VectorSet> queryset,
        std::shared_ptr<VectorSet>& truth,
        bool normalize) {
        if (fileexists(filename.c_str())) {
            auto opts = std::make_shared<Helper::ReaderOptions>(GetEnumValueType<float>(), m_m, VectorFileType::DEFAULT);
            auto reader = Helper::VectorSetReader::CreateInstance(opts);
            if (ErrorCode::Success != reader->LoadFile(filename)) {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read file %s\n", filename.c_str());
                exit(1);
            }
            truth = reader->GetVectorSet();
            return;
        }

        DistCalcMethod distMethod;
        Helper::Convert::ConvertStringTo(m_distMethod.c_str(), distMethod);
        if (normalize && distMethod == DistCalcMethod::Cosine) {
            COMMON::Utils::BatchNormalize((T*)vecset->GetData(), vecset->Count(), vecset->Dimension(), COMMON::Utils::GetBase<T>(), 5);
        }

        ByteArray tru = ByteArray::Alloc(sizeof(float) * queryset->Count() * m_k);
#pragma omp parallel for
        for (SizeType i = 0; i < queryset->Count(); ++i) {
            SizeType* neighbors = ((SizeType*)tru.Data()) + i * m_k;
            COMMON::QueryResultSet<T> res((const T*)queryset->GetVector(i), m_k);
            for (SizeType j = 0; j < vecset->Count(); ++j) {
                float dist = COMMON::DistanceUtils::ComputeDistance(res.GetTarget(), reinterpret_cast<T*>(vecset->GetVector(j)), m_m, distMethod);
                res.AddPoint(j, dist);
            }
            res.SortResult();
            for (int j = 0; j < m_k; ++j) neighbors[j] = res.GetResult(j)->VID;
        }
        truth = std::make_shared<BasicVectorSet>(tru, GetEnumValueType<float>(), m_k, queryset->Count());
        truth->Save(filename);
    }

    template <typename T>
    std::shared_ptr<VectorSet> TestDataGenerator<T>::GenerateRandomVectorSet(SizeType count, DimensionType dim) {
        ByteArray vec = ByteArray::Alloc(sizeof(T) * count * dim);
        for (SizeType i = 0; i < count * dim; ++i) {
            ((T*)vec.Data())[i] = (T)COMMON::Utils::rand(127, -127);
        }
        return std::make_shared<BasicVectorSet>(vec, GetEnumValueType<T>(), dim, count);
    }

    template <typename T>
    std::shared_ptr<MetadataSet> TestDataGenerator<T>::GenerateMetadataSet(SizeType count, SizeType offsetBase) {
        ByteArray meta = ByteArray::Alloc(count * 6);
        ByteArray metaoffset = ByteArray::Alloc((count + 1) * sizeof(std::uint64_t));
        std::uint64_t offset = 0;
        for (SizeType i = 0; i < count; i++) {
            ((std::uint64_t*)metaoffset.Data())[i] = offset;
            std::string id = std::to_string(i + offsetBase);
            std::memcpy(meta.Data() + offset, id.c_str(), id.length());
            offset += id.length();
        }
        ((std::uint64_t*)metaoffset.Data())[count] = offset;
        return std::make_shared<MemMetadataSet>(meta, metaoffset, count, count * 2, MaxSize, 10);
    }

    template <typename T>
    std::shared_ptr<VectorSet> TestDataGenerator<T>::CombineVectorSets(std::shared_ptr<VectorSet> base,
        std::shared_ptr<VectorSet> add) {
        return base;
    }

    template <typename T>
    std::shared_ptr<Helper::VectorSetReader> TestDataGenerator<T>::LoadReader(const std::string& filename) {
        auto opts = std::make_shared<Helper::ReaderOptions>(GetEnumValueType<T>(), m_m, VectorFileType::DEFAULT);
        auto reader = Helper::VectorSetReader::CreateInstance(opts);
        if (ErrorCode::Success != reader->LoadFile(filename)) {
            SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Failed to read file %s\n", filename.c_str());
            exit(1);
        }
        return reader;
    }

    // Explicit instantiation
    template class TestDataGenerator<int8_t>;
    template class TestDataGenerator<uint8_t>;
    template class TestDataGenerator<float>;
}
