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
        TestDataGenerator(int n, int q, int m, int k, std::string distMethod, int a = 0, bool isRandom = true,
                        std::string vectorPath = "", std::string queryPath = "");

        void Run(std::shared_ptr<SPTAG::VectorSet>& vecset,
            std::shared_ptr<SPTAG::MetadataSet>& metaset,
            std::shared_ptr<SPTAG::VectorSet>& queryset,
            std::shared_ptr<SPTAG::VectorSet>& truth,
            std::shared_ptr<SPTAG::VectorSet>& addvecset,
            std::shared_ptr<SPTAG::MetadataSet>& addmetaset,
            std::shared_ptr<SPTAG::VectorSet>& addtruth);

        static std::shared_ptr<SPTAG::VectorSet> GenerateRandomVectorSet(SPTAG::SizeType count, SPTAG::DimensionType dim);

        static std::shared_ptr<SPTAG::MetadataSet> GenerateMetadataSet(SPTAG::SizeType count, SPTAG::SizeType offsetStart);

        static std::shared_ptr<SPTAG::VectorSet> GenerateLoadVectorSet(SPTAG::SizeType count, SPTAG::DimensionType dim,
                                                                       std::string path, SPTAG::SizeType start = 0);

        void RunBatches(std::shared_ptr<SPTAG::VectorSet> &vecset, std::shared_ptr<SPTAG::MetadataSet> &metaset,
                        std::shared_ptr<SPTAG::VectorSet> &addvecset, std::shared_ptr<SPTAG::MetadataSet> &addmetaset,
                        std::shared_ptr<SPTAG::VectorSet> &queryset, int base, int batchinsert, int batchdelete, int batches, 
                        std::shared_ptr<SPTAG::VectorSet> &truths);

        void RunLargeBatches(std::string &vecset, std::string &metaset, std::string &metaidx, 
                     std::string &addset, std::string &addmetaset, std::string &addmetaidx,
                     std::string &queryset, int bash, int batchinsert, int batchdelete, int batches,
                     std::string &truth, bool generateTruth = true);

    private:
        int m_n, m_a, m_q, m_m, m_k;
        std::string m_distMethod;
        bool m_isRandom;
        std::string m_vectorPath, m_queryPath;

        std::shared_ptr<SPTAG::Helper::VectorSetReader> LoadReader(const std::string& filename);

        void LoadOrGenerateBase(std::shared_ptr<SPTAG::VectorSet>& vecset, std::shared_ptr<SPTAG::MetadataSet>& metaset);

        void LoadOrGenerateQuery(std::shared_ptr<SPTAG::VectorSet>& queryset);

        void LoadOrGenerateAdd(std::shared_ptr<SPTAG::VectorSet>& addvecset, std::shared_ptr<SPTAG::MetadataSet>& addmetaset);

        void LoadOrGenerateTruth(const std::string& filename,
            std::shared_ptr<SPTAG::VectorSet> vecset,
            std::shared_ptr<SPTAG::VectorSet> queryset,
            std::shared_ptr<SPTAG::VectorSet>& truth,
            bool normalize);

        void LoadOrGenerateBatchTruth(const std::string &filename, std::shared_ptr<SPTAG::VectorSet> vecset,
                                      std::shared_ptr<SPTAG::VectorSet> queryset,
                                      std::shared_ptr<SPTAG::VectorSet> &truths, int base, int batchinsert,
                                      int batchdelete, int batches, bool normalize);

        std::shared_ptr<SPTAG::VectorSet> CombineVectorSets(std::shared_ptr<SPTAG::VectorSet> base,
            std::shared_ptr<SPTAG::VectorSet> additional);

        void GenerateVectorSet(std::string &vecset, std::string &metaset, std::string &metaidx, std::string &vecPath,
                               SPTAG::SizeType start, int count);

        void GenerateBatchTruth(const std::string &filename, std::string &pvecset, std::string &paddset, std::string &pqueryset,
                                int base, int batchinsert, int batchdelete, int batches,
                                bool normalize);
    };

} // namespace TestUtils