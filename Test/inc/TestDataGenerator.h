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

        static std::shared_ptr<SPTAG::VectorSet> GenerateRandomVectorSet(SPTAG::SizeType count, SPTAG::DimensionType dim);

        static std::shared_ptr<SPTAG::MetadataSet> GenerateMetadataSet(SPTAG::SizeType count, SPTAG::SizeType offsetStart);

        static std::shared_ptr<SPTAG::VectorSet> GenerateLoadVectorSet(SPTAG::SizeType count, SPTAG::DimensionType dim,
                                                                       std::string path, SPTAG::SizeType start = 0);

        static std::shared_ptr<SPTAG::VectorSet> LoadVectorSet(const std::string pvecset, SPTAG::DimensionType dim, SPTAG::SizeType start = 0, SPTAG::SizeType count = -1);

        static std::shared_ptr<SPTAG::MetadataSet> LoadMetadataSet(const std::string pmetaset, const std::string pmetaidx, SPTAG::SizeType start = 0, SPTAG::SizeType count = -1);

        // Compute recall against truth file.
        //
        // Distributed (per-node) recall: when each node only owns a SUBSET of
        // the global query set, pass the global query count and this node's
        // query offset so the truth row indexing is computed in global terms.
        // The truth file is laid out as:
        //   [iter=0 VIDs for queries 0..Q-1] [iter=1 VIDs ...] ...
        //   [iter=0 dists for queries 0..Q-1] [iter=1 dists ...] ...
        // where Q is the GLOBAL query count, NOT res.size(). With the legacy
        // res.size()-based formula, distributed batches > 0 read the wrong
        // rows (off by Q-myCount), giving near-random recall that's noise.
        // totalQueries=-1 (default) preserves the legacy single-node formula.
        static float EvaluateRecall(const std::vector<SPTAG::QueryResult> &res, std::shared_ptr<SPTAG::VectorSet> &truth, int recallK, int k, int batch, int totalbatches,
                                    int totalQueries = -1, int queryOffset = 0);

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

        void GenerateVectorSet(std::string &pvecset, std::string &pmetaset, std::string &pmetaidx, std::string &vecPath,
                               SPTAG::SizeType start, int count);

        void GenerateBatchTruth(const std::string &filename, std::string &pvecset, std::string &paddset, std::string &pqueryset,
                                int base, int batchinsert, int batchdelete, int batches,
                                bool normalize);
    };

} // namespace TestUtils