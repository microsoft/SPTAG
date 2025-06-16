#pragma once

#include <memory>
#include <string>
#include "inc/Core/VectorIndex.h"
#include "inc/Core/SPANN/Index.h"
#include "inc/Core/Common/CommonUtils.h"
#include "inc/Core/Common/QueryResultSet.h"
#include "inc/Core/Common/DistanceUtils.h"
#include "inc/Helper/VectorSetReader.h"
#include "inc/Helper/DiskIO.h"

namespace TestUtils {

    template <typename T>
    class TestDataGenerator {
    public:
        TestDataGenerator(int n, int q, int m, int k, std::string distMethod);

        void Run(std::shared_ptr<SPTAG::VectorSet>& vecset,
                 std::shared_ptr<SPTAG::MetadataSet>& metaset,
                 std::shared_ptr<SPTAG::VectorSet>& queryset,
                 std::shared_ptr<SPTAG::VectorSet>& truth,
                 std::shared_ptr<SPTAG::VectorSet>& addvecset,
                 std::shared_ptr<SPTAG::MetadataSet>& addmetaset,
                 std::shared_ptr<SPTAG::VectorSet>& addtruth);

    private:
        int m_n, m_q, m_m, m_k;
        std::string m_distMethod;

        bool FileExists(const std::string& filename);

        std::shared_ptr<SPTAG::Helper::VectorSetReader> LoadReader(const std::string& filename);

        std::shared_ptr<SPTAG::VectorSet> GenerateRandomVectorSet(SPTAG::SizeType count, SPTAG::DimensionType dim);

        std::shared_ptr<SPTAG::MetadataSet> GenerateMetadataSet(SPTAG::SizeType count, SPTAG::SizeType offsetStart);

        void LoadOrGenerateBase(std::shared_ptr<SPTAG::VectorSet>& vecset, std::shared_ptr<SPTAG::MetadataSet>& metaset);

        void LoadOrGenerateQuery(std::shared_ptr<SPTAG::VectorSet>& queryset);

        void LoadOrGenerateAdd(std::shared_ptr<SPTAG::VectorSet>& addvecset, std::shared_ptr<SPTAG::MetadataSet>& addmetaset);

        void LoadOrGenerateTruth(const std::string& filename,
                                 std::shared_ptr<SPTAG::VectorSet> vecset,
                                 std::shared_ptr<SPTAG::VectorSet> queryset,
                                 std::shared_ptr<SPTAG::VectorSet>& truth,
                                 bool normalize);

        std::shared_ptr<SPTAG::VectorSet> CombineVectorSets(std::shared_ptr<SPTAG::VectorSet> base,
                                                            std::shared_ptr<SPTAG::VectorSet> additional);
    };

} // namespace TestUtils