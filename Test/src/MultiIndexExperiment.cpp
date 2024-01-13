// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Test.h"
#include "inc/Helper/SimpleIniReader.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Core/MultiIndexScan.h"
#include "inc/Core/Common/CommonUtils.h"

#include <chrono>
#include <numeric>
#include <set>
std::vector<int> VID2rid;


// top number to query 
int topk = 50;
//vector dimension
SPTAG::DimensionType m = 1024;
std::string pathEmbeddings = "/embeddings/";
std::string outputResult = "/result/";
std::vector<int> testSearchLimit = {256, 512, 1024};


struct FinalData{
    double Dist;
    int rid, vid;
    FinalData(double x, int y, int z): Dist(x), rid(y), vid(z){}
    bool operator < (const FinalData x) const {
        return Dist < x.Dist;
    }
};

static std::string indexName(unsigned int id){
    return "testindices" + std::to_string(id);
}

template <typename T>
void BuildIndex(SPTAG::IndexAlgoType algo, std::string distCalcMethod, std::shared_ptr<SPTAG::VectorSet>& vec, std::shared_ptr<SPTAG::MetadataSet>& meta, const std::string out)
{

    std::shared_ptr<SPTAG::VectorIndex> vecIndex = SPTAG::VectorIndex::CreateInstance(algo, SPTAG::GetEnumValueType<T>());
    BOOST_CHECK(nullptr != vecIndex);

    vecIndex->SetParameter("DistCalcMethod", distCalcMethod);
    vecIndex->SetParameter("NumberOfThreads", "16");

    BOOST_CHECK(SPTAG::ErrorCode::Success == vecIndex->BuildIndex(vec, meta));
    BOOST_CHECK(SPTAG::ErrorCode::Success == vecIndex->SaveIndex(out));
}

float sumFunc(std::vector<float> in){
    return std::accumulate(in.begin(), in.end(), (float)0.0);
}

template <typename T>
void GenerateVectorDataSet(SPTAG::IndexAlgoType algo, std::string distCalcMethod, unsigned int id, std::vector<T> &query, std::vector<T> &query_id, std::string File_name){
    SPTAG::SizeType n = 0, q = 0;

    std::string fileName;
    fileName = pathEmbeddings + File_name + "_embeds_collection.tsv";
	std::ifstream in1(fileName.c_str());
    if (!in1)
	{
		std::cout << "Fail to read " << fileName << "." << std::endl;
		return;
	}
    std::vector<T> vec;
    std::string line;
    std::vector<std::uint64_t> meta;

    for (; getline(in1, line); n++)
	{
        std::uint64_t number = 0;
        std::string a = line.substr(0, line.find("\t"));
        for (size_t j = 0; j < a.length(); j++)
            number = number * 10 + a[j] - '0'; 
        VID2rid.push_back((int)number);
        meta.push_back(number);

        int l = (int)vec.size();
		line.erase(0, line.find("[") + 1);
        for (SPTAG::DimensionType j = 0; j < m - 1; j++) {
            vec.push_back((T)std::stof(line.substr(0, line.find(","))));
            line.erase(0, line.find(",") + 2);
        }
        vec.push_back((T)std::stof(line.substr(0, line.find("]"))));
        //normalization;
        int r = (int)vec.size();
        double sum = 0.0;
        for (int i = l; i < r; i++)
            sum += vec[i] * vec[i];
        sum = sqrt(sum);
        for (int i = l; i < r; i++)
            vec[i] = (T)(vec[i] / sum);

    }
    fileName = pathEmbeddings + File_name + "_embeds_query.tsv";
	std::ifstream in2(fileName.c_str());
    if (!in2)
	{
		std::cout << "Fail to read " << fileName << "." << std::endl;
		return;
	}
    for (; getline(in2, line); q++)
	{
        query_id.push_back((T)std::stof(line.substr(0, line.find("\t"))));

        int l = (int)query.size();
		line.erase(0, line.find("[") + 1);
        for (SPTAG::DimensionType j = 0; j < m - 1; j++) {
            query.push_back((T)std::stof(line.substr(0, line.find(","))));
            line.erase(0, line.find(",") + 2);
        }
        query.push_back((T)std::stof(line.substr(0, line.find("]"))));
        //normalization;
        int r = (int)query.size();
        double sum = 0.0;
        for (int i = l; i < r; i++)
            sum += query[i] * query[i];
        sum = sqrt(sum);
        for (int i = l; i < r; i++)
            query[i] = (T)(query[i] / sum);
    }

    std::shared_ptr<SPTAG::VectorSet> vecset(new SPTAG::BasicVectorSet(
        SPTAG::ByteArray((std::uint8_t*)vec.data(), sizeof(T) * n * m, false),
        SPTAG::GetEnumValueType<T>(), m, n));

    std::shared_ptr<SPTAG::MetadataSet> metaset (new SPTAG::MemMetadataSet(
            n * sizeof(std::uint64_t),
            n * sizeof(std::uint64_t),
            n));
    for (auto x : meta){
        std::uint8_t result[sizeof(x)];
        std::memcpy(result, &x, sizeof(x));
        metaset->Add(SPTAG::ByteArray(result, 8, false));
    }

    BuildIndex<T>(algo, distCalcMethod, vecset, metaset, indexName(id).c_str());
}

template <typename T>
void TestMultiIndexScanN(SPTAG::IndexAlgoType algo, std::string distCalcMethod, unsigned int n)
{
    
    std::vector<std::vector<T>> queries(n, std::vector<T>());
    std::vector<std::vector<T>> query_id(n, std::vector<T>());

    GenerateVectorDataSet<T>(algo, distCalcMethod, 0, queries[0], query_id[0], "img");
    GenerateVectorDataSet<T>(algo, distCalcMethod, 1, queries[1], query_id[1], "rec");

    for (auto KK: testSearchLimit)
    {
        std::string output_result_path = outputResult + std::to_string(KK) + "_qrels.txt";
        std::string output_lantency_path = outputResult + std::to_string(KK) + "_latency.txt";

        std::ofstream out1;
        out1.open(output_result_path);

        if (!out1.is_open())
        {
            std::cout << "Cannot open file out1" << std::endl;
            return ;
        }
        
        std::ofstream out2;
        out2.open(output_lantency_path);
        if (!out2.is_open())
        {
            std::cout << "Cannot open file out2" << std::endl;
            return ;
        }

        for ( int i = 0; i < query_id[0].size(); i++ ){
            std::vector<std::vector<T>> query_current(n, std::vector<T>());
            for ( unsigned int j = 0; j < n ; j++ ){
                for ( int k = i * m; k <  i * m + m; k++ ){
                    query_current[j].push_back(queries[j][k]);
                }
            }
            int idquery = (int)query_id[0][i];
            std::vector<std::shared_ptr<SPTAG::VectorIndex>> vecIndices;
            std::vector<void*> p_targets;
            for ( unsigned int i = 0; i < n; i++ ) {
                std::shared_ptr<SPTAG::VectorIndex> vecIndex;
                BOOST_CHECK(SPTAG::ErrorCode::Success == SPTAG::VectorIndex::LoadIndex(indexName(i).c_str(), vecIndex));
                BOOST_CHECK(nullptr != vecIndex);
                vecIndices.push_back(vecIndex);
                p_targets.push_back(query_current[i].data());
            }

            double Begin_time = clock();
            SPTAG::MultiIndexScan scan(vecIndices, p_targets, topk, &sumFunc, false, 10000000, KK);
            SPTAG::BasicResult result;

            std::set<FinalData> FinalResult;
            
            for (int i = 0; i < topk; i++) {
                bool hasResult = scan.Next(result);
                if (!hasResult) break;

                FinalResult.insert(FinalData(result.Dist, VID2rid[result.VID], result.VID));
                auto Finaltmp = FinalResult.end();
                Finaltmp--;
                if (FinalResult.size() > topk) FinalResult.erase(Finaltmp);
            }

            double End_time = clock();
            out2 << idquery << "\t" << (End_time-Begin_time)/CLOCKS_PER_SEC << std::endl;
            int Rank = 0;
            for (auto i = FinalResult.begin(); i != FinalResult.end(); i++)
                out1 << idquery << " " << i->rid << " " << ++Rank << " " << i->vid << " " << i->Dist << std::endl;

            for (int i = (int)FinalResult.size(); i < topk; i++)
                out1 << idquery << " " << -1 << " " << ++Rank << " " << -1 << " " << -1 << std::endl;

            scan.Close();
        }
    }
}

BOOST_AUTO_TEST_SUITE(MultiIndexExperiment)

BOOST_AUTO_TEST_CASE(BKTTest)
{
    TestMultiIndexScanN<float>(SPTAG::IndexAlgoType::BKT, "InnerProduct", 2);
}

BOOST_AUTO_TEST_SUITE_END()
