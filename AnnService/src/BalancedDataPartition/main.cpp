// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <mpi.h>
#include <thread>
#include <cstdlib>
#include <algorithm>
#include "inc/Core/Common/DistanceUtils.h"
#include "inc/Core/Common/Dataset.h"
#include "inc/Core/Common/BKTree.h"
#include "inc/Helper/VectorSetReader.h"
#include "inc/Helper/CommonHelper.h"

using namespace SPTAG;

#define CHECKIO(ptr, func, bytes, ...) if (ptr->func(bytes, __VA_ARGS__) != bytes) { \
    LOG(Helper::LogLevel::LL_Error, "DiskError: Cannot read or write %d bytes.\n", (int)(bytes)); \
    exit(1); \
}

typedef short LabelType;

class PartitionOptions : public Helper::ReaderOptions
{
public:
    PartitionOptions():Helper::ReaderOptions(VectorValueType::Float, 0, VectorFileType::TXT, "|", 32)
    {
        AddRequiredOption(m_inputFiles, "-i", "--input", "Input raw data.");
        AddRequiredOption(m_clusterNum, "-c", "--numclusters", "Number of clusters.");
        AddOptionalOption(m_stopDifference, "-d", "--diff", "Clustering stop center difference.");
        AddOptionalOption(m_maxIter, "-r", "--iters", "Max clustering iterations.");
        AddOptionalOption(m_localSamples, "-s", "--samples", "Number of samples for fast clustering.");
        AddOptionalOption(m_lambda, "-l", "--lambda", "lambda for balanced size level.");
        AddOptionalOption(m_distMethod, "-m", "--dist", "Distance method (L2 or Cosine).");
        AddOptionalOption(m_outdir, "-o", "--outdir", "Output directory.");
        AddOptionalOption(m_weightfile, "-w", "--weight", "vector weight file.");
        AddOptionalOption(m_wlambda, "-lw", "--wlambda", "lambda for balanced weight level.");
        AddOptionalOption(m_seed, "-e", "--seed", "Random seed.");
        AddOptionalOption(m_initIter, "-x", "--init", "Number of iterations for initialization.");
        AddOptionalOption(m_clusterassign, "-a", "--assign", "Number of clusters to be assigned.");
        AddOptionalOption(m_vectorfactor, "-vf", "--vectorscale", "Max vector number scale factor.");
        AddOptionalOption(m_closurefactor, "-cf", "--closurescale", "Max closure factor");
        AddOptionalOption(m_stage, "-g", "--stage", "Running function (Clustering or LocalPartition)");
        AddOptionalOption(m_centers, "-ct", "--centers", "File path to store centers.");
        AddOptionalOption(m_labels, "-lb", "--labels", "File path to store labels.");
        AddOptionalOption(m_status, "-st", "--status", "Cosmos path to store intermediate centers.");
        AddOptionalOption(m_totalparts, "-tp", "--parts", "Total partitions.");
        AddOptionalOption(m_syncscript, "-ss", "--script", "Run sync script.");
        AddOptionalOption(m_recoveriter, "-ri", "--recover", "Recover iteration.");
        AddOptionalOption(m_newp, "-np", "--newpenalty", "old penalty: 0, new penalty: 1");
        AddOptionalOption(m_hardcut, "-hc", "--hard", "soft: 0, hard: 1");
    }

    ~PartitionOptions() {}

    std::string m_inputFiles;
    int m_clusterNum;

    float m_stopDifference = 0.000001f;
    int m_maxIter = 100;
    int m_localSamples = 1000;
    float m_lambda = 0.00000f;
    float m_wlambda = 0.00000f;
    int m_seed = -1;
    int m_initIter = 3;
    int m_clusterassign = 1;
    int m_totalparts = 1;
    int m_recoveriter = -1;
    float m_vectorfactor = 2.0f;
    float m_closurefactor = 1.2f;
    int m_newp = 0;
    int m_hardcut = 0;
    DistCalcMethod m_distMethod = DistCalcMethod::L2;

    std::string m_labels = "labels.bin";
    std::string m_centers = "centers.bin";
    std::string m_outdir = "-";
    std::string m_outfile = "vectors.bin";
    std::string m_outmetafile = "meta.bin";
    std::string m_outmetaindexfile = "metaindex.bin";
    std::string m_weightfile = "-";
    std::string m_stage = "Clustering";
    std::string m_status = ".";
    std::string m_syncscript = "";
} options;

EdgeCompare g_edgeComparer;

template <typename T>
bool LoadCenters(T* centers, SizeType row, DimensionType col, const std::string& centerpath, float* lambda = nullptr, float* diff = nullptr, float* mindist = nullptr, int* noimprovement = nullptr) {
    if (fileexists(centerpath.c_str())) {
        auto ptr = f_createIO();
        if (ptr == nullptr || !ptr->Initialize(centerpath.c_str(), std::ios::binary | std::ios::in)) {
            LOG(Helper::LogLevel::LL_Error, "Failed to read center file %s.\n", centerpath.c_str());
            return false;
        }

        SizeType r;
        DimensionType c;
        float f;
        int i;
        if (ptr->ReadBinary(sizeof(SizeType), (char*)&r) != sizeof(SizeType)) return false;
        if (ptr->ReadBinary(sizeof(DimensionType), (char*)&c) != sizeof(DimensionType)) return false;

        if (r != row || c != col) {
            LOG(Helper::LogLevel::LL_Error, "Row(%d,%d) or Col(%d,%d) cannot match.\n", r, row, c, col);
            return false;
        }

        if (ptr->ReadBinary(sizeof(T) * row * col, (char*)centers) != sizeof(T) * row * col) return false;

        if (lambda) {
            if (ptr->ReadBinary(sizeof(float), (char*)&f) == sizeof(float)) *lambda = f;
        }
        if (diff) {
            if (ptr->ReadBinary(sizeof(float), (char*)&f) == sizeof(float)) *diff = f;
        }
        if (mindist) {
            if (ptr->ReadBinary(sizeof(float), (char*)&f) == sizeof(float)) *mindist = f;
        }
        if (noimprovement) {
            if (ptr->ReadBinary(sizeof(int), (char*)&i) == sizeof(int)) *noimprovement = i;
        }
        LOG(Helper::LogLevel::LL_Info, "Load centers(%d,%d) from file %s.\n", row, col, centerpath.c_str());
        return true;
    }
    return false;
}

template <typename T>
void SaveCenters(T* centers, SizeType row, DimensionType col, const std::string& centerpath, float lambda = 0.0, float diff = 0.0, float mindist = 0.0, int noimprovement = 0) {
    auto ptr = f_createIO();
    if (ptr == nullptr || !ptr->Initialize(centerpath.c_str(), std::ios::binary | std::ios::out)) {
        LOG(Helper::LogLevel::LL_Error, "Failed to open center file %s to write.\n", centerpath.c_str());
        exit(1);
    }

    CHECKIO(ptr, WriteBinary, sizeof(SizeType), (char*)&row);
    CHECKIO(ptr, WriteBinary, sizeof(DimensionType), (char*)&col);
    CHECKIO(ptr, WriteBinary, sizeof(T) * row * col, (char*)centers);
    CHECKIO(ptr, WriteBinary, sizeof(float), (char*)&lambda);
    CHECKIO(ptr, WriteBinary, sizeof(float), (char*)&diff);
    CHECKIO(ptr, WriteBinary, sizeof(float), (char*)&mindist);
    CHECKIO(ptr, WriteBinary, sizeof(int), (char*)&noimprovement);
    LOG(Helper::LogLevel::LL_Info, "Save centers(%d,%d) to file %s.\n", row, col, centerpath.c_str());
}

template <typename T>
inline float MultipleClustersAssign(const COMMON::Dataset<T>& data,
    std::vector<SizeType>& indices,
    const SizeType first, const SizeType last, COMMON::KmeansArgs<T>& args, COMMON::Dataset<LabelType>& label, bool updateCenters, float lambda, std::vector<float>& weights, float wlambda) {
    float currDist = 0;
    SizeType subsize = (last - first - 1) / args._T + 1;

    std::uint64_t avgCount = 0;
    for (int k = 0; k < args._K; k++) avgCount += args.counts[k];
    avgCount /= args._K;

    auto func = [&](int tid)
    {
        SizeType istart = first + tid * subsize;
        SizeType iend = min(first + (tid + 1) * subsize, last);
        SizeType* inewCounts = args.newCounts + tid * args._K;
        float* inewWeightedCounts = args.newWeightedCounts + tid * args._K;
        float* inewCenters = args.newCenters + tid * args._K * args._D;
        SizeType* iclusterIdx = args.clusterIdx + tid * args._K;
        float* iclusterDist = args.clusterDist + tid * args._K;
        float idist = 0;
        std::vector<SPTAG::NodeDistPair> centerDist(args._K, SPTAG::NodeDistPair());
        for (SizeType i = istart; i < iend; i++) {
            for (int k = 0; k < args._K; k++) {
                float penalty = lambda * (((options.m_newp == 1) && (args.counts[k] < avgCount)) ? avgCount : args.counts[k]) + wlambda * args.weightedCounts[k];
                float dist = args.fComputeDistance(data[indices[i]], args.centers + k * args._D, args._D) + penalty;
                centerDist[k].node = k;
                centerDist[k].distance = dist;
            }
            std::sort(centerDist.begin(), centerDist.end(), [](const SPTAG::NodeDistPair& a, const SPTAG::NodeDistPair& b) {
                return (a.distance < b.distance) || (a.distance == b.distance && a.node < b.node);
                });

            for (int k = 0; k < label.C(); k++) {
                if (centerDist[k].distance <= centerDist[0].distance * options.m_closurefactor) {
                    label[i][k] = (LabelType)(centerDist[k].node);
                    inewCounts[centerDist[k].node]++;
                    inewWeightedCounts[centerDist[k].node] += weights[indices[i]];
                    idist += centerDist[k].distance;

                    if (updateCenters) {
                        const T* v = (const T*)data[indices[i]];
                        float* center = inewCenters + centerDist[k].node * args._D;
                        for (DimensionType j = 0; j < args._D; j++) center[j] += v[j];
                        if (centerDist[k].distance > iclusterDist[centerDist[k].node]) {
                            iclusterDist[centerDist[k].node] = centerDist[k].distance;
                            iclusterIdx[centerDist[k].node] = indices[i];
                        }
                    }
                    else {
                        if (centerDist[k].distance <= iclusterDist[centerDist[k].node]) {
                            iclusterDist[centerDist[k].node] = centerDist[k].distance;
                            iclusterIdx[centerDist[k].node] = indices[i];
                        }
                    }
                }
                else {
                    label[i][k] = (std::numeric_limits<LabelType>::max)();
                }
            }
        }
        SPTAG::COMMON::Utils::atomic_float_add(&currDist, idist);
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < args._T; i++) { threads.emplace_back(func, i); }
    for (auto& thread : threads) { thread.join(); }

    for (int i = 1; i < args._T; i++) {
        for (int k = 0; k < args._K; k++) {
            args.newCounts[k] += args.newCounts[i*args._K + k];
            args.newWeightedCounts[k] += args.newWeightedCounts[i*args._K + k];
        }
    }

    if (updateCenters) {
        for (int i = 1; i < args._T; i++) {
            float* currCenter = args.newCenters + i*args._K*args._D;
            for (size_t j = 0; j < ((size_t)args._K) * args._D; j++) args.newCenters[j] += currCenter[j];

            for (int k = 0; k < args._K; k++) {
                if (args.clusterIdx[i*args._K + k] != -1 && args.clusterDist[i*args._K + k] > args.clusterDist[k]) {
                    args.clusterDist[k] = args.clusterDist[i*args._K + k];
                    args.clusterIdx[k] = args.clusterIdx[i*args._K + k];
                }
            }
        }
    }
    else {
        for (int i = 1; i < args._T; i++) {
            for (int k = 0; k < args._K; k++) {
                if (args.clusterIdx[i*args._K + k] != -1 && args.clusterDist[i*args._K + k] <= args.clusterDist[k]) {
                    args.clusterDist[k] = args.clusterDist[i*args._K + k];
                    args.clusterIdx[k] = args.clusterIdx[i*args._K + k];
                }
            }
        }
    }
    return currDist;
}

template <typename T>
inline float HardMultipleClustersAssign(const COMMON::Dataset<T>& data,
    std::vector<SizeType>& indices,
    const SizeType first, const SizeType last, COMMON::KmeansArgs<T>& args, COMMON::Dataset<LabelType>& label, SizeType* mylimit, std::vector<float>& weights,
    const int clusternum, const bool fill) {
    float currDist = 0;
    SizeType subsize = (last - first - 1) / args._T + 1;

    SPTAG::Edge* items = new SPTAG::Edge[last - first];

    auto func1 = [&](int tid)
    {
        SizeType istart = first + tid * subsize;
        SizeType iend = min(first + (tid + 1) * subsize, last);
        float* iclusterDist = args.clusterDist + tid * args._K;
        std::vector<SPTAG::NodeDistPair> centerDist(args._K, SPTAG::NodeDistPair());
        for (SizeType i = istart; i < iend; i++) {
            for (int k = 0; k < args._K; k++) {
                float dist = args.fComputeDistance(data[indices[i]], args.centers + k * args._D, args._D);
                centerDist[k].node = k;
                centerDist[k].distance = dist;
            }
            std::sort(centerDist.begin(), centerDist.end(), [](const SPTAG::NodeDistPair& a, const SPTAG::NodeDistPair& b) {
                return (a.distance < b.distance) || (a.distance == b.distance && a.node < b.node);
                });

            if (centerDist[clusternum].distance <= centerDist[0].distance * options.m_closurefactor) {
                items[i - first].node = centerDist[clusternum].node;
                items[i - first].distance = centerDist[clusternum].distance;
                items[i - first].tonode = i;
                iclusterDist[centerDist[clusternum].node] += centerDist[clusternum].distance;
            }
            else {
                items[i - first].node = MaxSize;
                items[i - first].distance = MaxDist;
                items[i - first].tonode = -i-1;
            }
        }
    };

    {
        std::vector<std::thread> threads;
        for (int i = 0; i < args._T; i++) { threads.emplace_back(func1, i); }
        for (auto& thread : threads) { thread.join(); }
    }

    std::sort(items, items + last - first, g_edgeComparer);

    for (int i = 0; i < args._T; i++) {
        for (int k = 0; k < args._K; k++) {
            mylimit[k] -= args.newCounts[i * args._K + k];
            if (i > 0) args.clusterDist[k] += args.clusterDist[i * args._K + k];
        }
    }
    std::size_t startIdx = 0;
    for (int i = 0; i < args._K; ++i)
    {
        std::size_t endIdx = std::lower_bound(items, items + last - first, i + 1, g_edgeComparer) - items;
        LOG(Helper::LogLevel::LL_Info, "cluster %d: avgdist:%f limit:%d, drop:%zu - %zu\n", items[startIdx].node, args.clusterDist[i] / (endIdx - startIdx), mylimit[i], startIdx + mylimit[i], endIdx);
        for (size_t dropID = startIdx + mylimit[i]; dropID < endIdx; ++dropID)
        {
            if (items[dropID].tonode >= 0) items[dropID].tonode = -items[dropID].tonode - 1;
        }
        startIdx = endIdx;
    }

    auto func2 = [&, subsize](int tid)
    {
        SizeType istart = tid * subsize;
        SizeType iend = min((tid + 1) * subsize, last - first);
        SizeType* inewCounts = args.newCounts + tid * args._K;
        float* inewWeightedCounts = args.newWeightedCounts + tid * args._K;
        float idist = 0;
        for (SizeType i = istart; i < iend; i++) {
            if (items[i].tonode >= 0) {
                label[items[i].tonode][clusternum] = (LabelType)(items[i].node);
                inewCounts[items[i].node]++;
                inewWeightedCounts[items[i].node] += weights[indices[items[i].tonode]];
                idist += items[i].distance;
            }
            else {
                items[i].tonode = -items[i].tonode - 1;
                label[items[i].tonode][clusternum] = (std::numeric_limits<LabelType>::max)();
            }

            if (fill) {
                for (int k = clusternum + 1; k < label.C(); k++) {
                    label[items[i].tonode][k] = (std::numeric_limits<LabelType>::max)();
                }
            }
        }
        SPTAG::COMMON::Utils::atomic_float_add(&currDist, idist);
    };

    {
        std::vector<std::thread> threads2;
        for (int i = 0; i < args._T; i++) { threads2.emplace_back(func2, i); }
        for (auto& thread : threads2) { thread.join(); }
    }
    delete[] items;

    std::memset(args.counts, 0, sizeof(SizeType) * args._K);
    std::memset(args.weightedCounts, 0, sizeof(float) * args._K);
    for (int i = 0; i < args._T; i++) {
        for (int k = 0; k < args._K; k++) {
            args.counts[k] += args.newCounts[i*args._K + k];
            args.weightedCounts[k] += args.newWeightedCounts[i*args._K + k];
        }
    }
    return currDist;
}

template <typename T>
void Process(MPI_Datatype type) {
    int rank, size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto vectorReader = Helper::VectorSetReader::CreateInstance(std::make_shared<Helper::ReaderOptions>(options));
    options.m_inputFiles = Helper::StrUtils::ReplaceAll(options.m_inputFiles, "*", std::to_string(rank));
    if (ErrorCode::Success != vectorReader->LoadFile(options.m_inputFiles))
    {
        LOG(Helper::LogLevel::LL_Error, "Failed to read input file.\n");
        exit(1);
    }
    std::shared_ptr<VectorSet> vectors = vectorReader->GetVectorSet();
    std::shared_ptr<MetadataSet> metas = vectorReader->GetMetadataSet();
    if (options.m_distMethod == DistCalcMethod::Cosine) vectors->Normalize(options.m_threadNum);

    std::vector<float> weights(vectors->Count(), 0.0f);
    if (options.m_weightfile.compare("-") != 0) {
        options.m_weightfile = Helper::StrUtils::ReplaceAll(options.m_weightfile, "*", std::to_string(rank));
        std::ifstream win(options.m_weightfile, std::ifstream::binary);
        if (!win.is_open()) {
            LOG(Helper::LogLevel::LL_Error, "Rank %d failed to read weight file %s.\n", rank, options.m_weightfile.c_str());
            exit(1);
        }
        SizeType rows;
        win.read((char*)&rows, sizeof(SizeType));
        if (rows != vectors->Count()) {
            win.close();
            LOG(Helper::LogLevel::LL_Error, "Number of weights (%d) is not equal to number of vectors (%d).\n", rows, vectors->Count());
            exit(1);
        }
        win.read((char*)weights.data(), sizeof(float)*rows);
        win.close();
    }
    COMMON::Dataset<T> data(vectors->Count(), vectors->Dimension(), 1024*1024, vectors->Count() + 1, (T*)vectors->GetData());
    COMMON::KmeansArgs<T> args(options.m_clusterNum, vectors->Dimension(), vectors->Count(), options.m_threadNum, options.m_distMethod);
    COMMON::Dataset<LabelType> label(vectors->Count(), options.m_clusterassign, vectors->Count(), vectors->Count());

    std::vector<SizeType> localindices(data.R(), 0);
    for (SizeType i = 0; i < data.R(); i++) localindices[i] = i;
    unsigned long long localCount = data.R(), totalCount;
    MPI_Allreduce(&localCount, &totalCount, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    totalCount = static_cast<unsigned long long>(totalCount * 1.0 / args._K * options.m_vectorfactor);

    if (rank == 0 && options.m_maxIter > 0 && options.m_lambda < -1e-6f) {
        float fBalanceFactor = COMMON::DynamicFactorSelect<T>(data, localindices, 0, data.R(), args, data.R());
        options.m_lambda = COMMON::Utils::GetBase<T>() * COMMON::Utils::GetBase<T>() / fBalanceFactor / data.R();
    }
    MPI_Bcast(&(options.m_lambda), 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    LOG(Helper::LogLevel::LL_Info, "rank %d  data:(%d,%d) machines:%d clusters:%d type:%d threads:%d lambda:%f samples:%d maxcountperpartition:%d\n",
        rank, data.R(), data.C(), size, options.m_clusterNum, ((int)options.m_inputValueType), options.m_threadNum, options.m_lambda, options.m_localSamples, totalCount);

    if (rank == 0) {
        LOG(Helper::LogLevel::LL_Info, "rank 0 init centers\n");
        if (!LoadCenters(args.newTCenters, args._K, args._D, options.m_centers, &(options.m_lambda))) {
            if (options.m_seed >= 0) std::srand(options.m_seed);
            COMMON::InitCenters<T, T>(data, localindices, 0, data.R(), args, options.m_localSamples, options.m_initIter);
        }
    }

    float currDiff = 1.0, d, currDist, minClusterDist = MaxDist;
    int iteration = 0;
    int noImprovement = 0;
    while (currDiff > options.m_stopDifference && iteration < options.m_maxIter) {
        if (rank == 0) {
            std::memcpy(args.centers, args.newTCenters, sizeof(T)*args._K*args._D);
        }
        MPI_Bcast(args.centers, args._K*args._D, type, 0, MPI_COMM_WORLD);

        args.ClearCenters();
        args.ClearCounts();
        args.ClearDists(-MaxDist);
        d = MultipleClustersAssign<T>(data, localindices, 0, data.R(), args, label, true, (iteration == 0) ? 0.0f : options.m_lambda, weights, (iteration == 0) ? 0.0f : options.m_wlambda);
        MPI_Allreduce(args.newCounts, args.counts, args._K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(args.newWeightedCounts, args.weightedCounts, args._K, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&d, &currDist, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        if (currDist < minClusterDist) {
            noImprovement = 0;
            minClusterDist = currDist;
        }
        else {
            noImprovement++;
        }
        if (noImprovement >= 10) break;

        if (rank == 0) {
            MPI_Reduce(MPI_IN_PLACE, args.newCenters, args._K * args._D, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
            currDiff = COMMON::RefineCenters<T, T>(data, args);
            LOG(Helper::LogLevel::LL_Info, "iter %d dist:%f diff:%f\n", iteration, currDist, currDiff);
        } else
            MPI_Reduce(args.newCenters, args.newCenters, args._K * args._D, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

        iteration++;
        MPI_Bcast(&currDiff, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    if (options.m_maxIter == 0) {
        if (rank == 0) {
            std::memcpy(args.centers, args.newTCenters, sizeof(T)*args._K*args._D);
        }
        MPI_Bcast(args.centers, args._K*args._D, type, 0, MPI_COMM_WORLD);
    }
    else {
        if (rank == 0) {
            for (int i = 0; i < args._K; i++)
                LOG(Helper::LogLevel::LL_Info, "cluster %d contains vectors:%d weights:%f\n", i, args.counts[i], args.weightedCounts[i]);
        }
    }
    d = 0;
    for (SizeType i = 0; i < data.R(); i++) localindices[i] = i;
    std::vector<SizeType> myLimit(args._K, (options.m_hardcut == 0) ? data.R() : (SizeType)(options.m_hardcut * totalCount / size));
    std::memset(args.counts, 0, sizeof(SizeType)*args._K);
    args.ClearCounts();
    args.ClearDists(0);
    for (int i = 0; i < options.m_clusterassign - 1; i++) {
        d += HardMultipleClustersAssign<T>(data, localindices, 0, data.R(), args, label, myLimit.data(), weights, i, false);
        std::memcpy(myLimit.data(), args.counts, sizeof(SizeType)*args._K);
        MPI_Allreduce(MPI_IN_PLACE, args.counts, args._K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, args.weightedCounts, args._K, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        if (rank == 0) {
            LOG(Helper::LogLevel::LL_Info, "assign %d....................d:%f\n", i, d);
            for (int i = 0; i < args._K; i++)
                LOG(Helper::LogLevel::LL_Info, "cluster %d contains vectors:%d weights:%f\n", i, args.counts[i], args.weightedCounts[i]);
        }
        for (int k = 0; k < args._K; k++)
            if (totalCount > args.counts[k])
                myLimit[k] += (SizeType)((totalCount - args.counts[k]) / size);
    }
    d += HardMultipleClustersAssign<T>(data, localindices, 0, data.R(), args, label, myLimit.data(), weights, options.m_clusterassign - 1, true);
    std::memcpy(args.newCounts, args.counts, sizeof(SizeType)*args._K);
    MPI_Allreduce(args.newCounts, args.counts, args._K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, args.weightedCounts, args._K, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&d, &currDist, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    if (label.Save(options.m_labels + "." + std::to_string(rank)) != ErrorCode::Success) {
        LOG(Helper::LogLevel::LL_Error, "Failed to save labels.\n");
        exit(1);
    }
    if (rank == 0) {
        SaveCenters(args.centers, args._K, args._D, options.m_centers, options.m_lambda);
        LOG(Helper::LogLevel::LL_Info, "final dist:%f\n", currDist);
        for (int i = 0; i < args._K; i++)
            LOG(Helper::LogLevel::LL_Status, "cluster %d contains vectors:%d weights:%f\n", i, args.counts[i], args.weightedCounts[i]);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (options.m_outdir.compare("-") != 0) {
        for (int i = 0; i < args._K; i++) {
            if (i % size == rank) {
                LOG(Helper::LogLevel::LL_Info, "Cluster %d start ......\n", i);
            }
            noImprovement = 0;
            std::string vecfile = options.m_outdir + "/" + options.m_outfile + "." + std::to_string(i + 1);
            if (fileexists(vecfile.c_str())) noImprovement = 1;
            MPI_Allreduce(MPI_IN_PLACE, &noImprovement, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            if (noImprovement) continue;

            if (i % size == rank) {
                std::string vecfile = options.m_outdir + "/" + options.m_outfile + "." + std::to_string(i);
                std::string metafile = options.m_outdir + "/" + options.m_outmetafile + "." + std::to_string(i);
                std::string metaindexfile = options.m_outdir + "/" + options.m_outmetaindexfile + "." + std::to_string(i);
                std::shared_ptr<Helper::DiskIO> out = f_createIO(), metaout = f_createIO(), metaindexout = f_createIO();
                if (out == nullptr || !out->Initialize(vecfile.c_str(), std::ios::binary | std::ios::out)) {
                    LOG(Helper::LogLevel::LL_Error, "Cannot open %s to write.\n", vecfile.c_str());
                    exit(1);
                }
                if (metaout == nullptr || !metaout->Initialize(metafile.c_str(), std::ios::binary | std::ios::out)) {
                    LOG(Helper::LogLevel::LL_Error, "Cannot open %s to write.\n", metafile.c_str());
                    exit(1);
                }
                if (metaindexout == nullptr || !metaindexout->Initialize(metaindexfile.c_str(), std::ios::binary | std::ios::out)) {
                    LOG(Helper::LogLevel::LL_Error, "Cannot open %s to write.\n", metaindexfile.c_str());
                    exit(1);
                }

                CHECKIO(out, WriteBinary, sizeof(int), (char*)(&args.counts[i]));
                CHECKIO(out, WriteBinary, sizeof(int), (char*)(&args._D));
                if (metas != nullptr) CHECKIO(metaindexout, WriteBinary, sizeof(int), (char*)(&args.counts[i]));

                std::uint64_t offset = 0;
                T* recvbuf = args.newTCenters;
                int recvmetabuflen = 200;
                char* recvmetabuf = new char [recvmetabuflen];
                for (int j = 0; j < size; j++) {
                    uint64_t offset_before = offset;
                    if (j != rank) {
                        int recv = 0;
                        MPI_Recv(&recv, 1, MPI_INT, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        for (int k = 0; k < recv; k++) {
                            MPI_Recv(recvbuf, args._D, type, j, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            CHECKIO(out, WriteBinary, sizeof(T)* args._D, (char*)recvbuf);

                            if (metas != nullptr) {
                                int len;
                                MPI_Recv(&len, 1, MPI_INT, j, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                                if (len > recvmetabuflen) {
                                    LOG(Helper::LogLevel::LL_Info, "enlarge recv meta buf to %d\n", len);
                                    delete[] recvmetabuf;
                                    recvmetabuflen = len;
                                    recvmetabuf = new char[recvmetabuflen];
                                }
                                MPI_Recv(recvmetabuf, len, MPI_CHAR, j, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                                CHECKIO(metaout, WriteBinary, len, recvmetabuf);
                                CHECKIO(metaindexout, WriteBinary, sizeof(std::uint64_t), (char*)(&offset));
                                offset += len;
                            }
                        }
                        LOG(Helper::LogLevel::LL_Info, "rank %d <- rank %d: %d vectors, %llu bytes meta\n", rank, j, recv, (offset - offset_before));
                    }
                    else {
                        size_t total_rec = 0;
                        for (int k = 0; k < data.R(); k++) {
                            for (int kk = 0; kk < label.C(); kk++) {
                                if (label[k][kk] == (LabelType)i) {
                                    CHECKIO(out, WriteBinary, sizeof(T) * args._D, (char*)(data[localindices[k]]));
                                    if (metas != nullptr) {
                                        ByteArray meta = metas->GetMetadata(localindices[k]);
                                        CHECKIO(metaout, WriteBinary, meta.Length(), (const char*)meta.Data());
                                        CHECKIO(metaindexout, WriteBinary, sizeof(std::uint64_t), (char*)(&offset));
                                        offset += meta.Length();
                                    }
                                    total_rec++;
                                }
                            }
                        }
                        LOG(Helper::LogLevel::LL_Info, "rank %d <- rank %d: %d(%d) vectors, %llu bytes meta\n", rank, j, args.newCounts[i], total_rec, (offset - offset_before));
                    }
                }
                delete[] recvmetabuf;
                if (metas != nullptr) CHECKIO(metaindexout, WriteBinary, sizeof(std::uint64_t), (char*)(&offset));
                out->ShutDown();
                metaout->ShutDown();
                metaindexout->ShutDown();
            }
            else {
                int dest = i % size;
                MPI_Send(&args.newCounts[i], 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
                size_t total_len = 0;
                size_t total_rec = 0;
                for (int j = 0; j < data.R(); j++) {
                    for (int kk = 0; kk < label.C(); kk++) {
                        if (label[j][kk] == (LabelType)i) {
                            MPI_Send(data[localindices[j]], args._D, type, dest, 1, MPI_COMM_WORLD);
                            if (metas != nullptr) {
                                ByteArray meta = metas->GetMetadata(localindices[j]);
                                int len = (int)meta.Length();
                                MPI_Send(&len, 1, MPI_INT, dest, 2, MPI_COMM_WORLD);
                                MPI_Send(meta.Data(), len, MPI_CHAR, dest, 3, MPI_COMM_WORLD);
                                total_len += len;
                            }
                            total_rec++;
                        }
                    }
                }
                LOG(Helper::LogLevel::LL_Info, "rank %d -> rank %d: %d(%d) vectors, %llu bytes meta\n", rank, dest, args.newCounts[i], total_rec, total_len);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    MPI_Finalize();
}

template <typename T>
ErrorCode SyncSaveCenter(COMMON::KmeansArgs<T> &args, int rank, int iteration, unsigned long long localCount, float localDist, float lambda, float diff, float mindist, int noimprovement, int savecenters, bool assign = false)
{
    if (!direxists(options.m_status.c_str())) mkdir(options.m_status.c_str());
    std::string folder = options.m_status + FolderSep + std::to_string(iteration);
    if (!direxists(folder.c_str())) mkdir(folder.c_str());

    if (!direxists(folder.c_str())) {
        LOG(Helper::LogLevel::LL_Error, "Cannot create the folder %s.\n", folder.c_str());
        exit(1);
    }

    if (rank == 0 && savecenters > 0) {
        SaveCenters(args.newTCenters, args._K, args._D, folder + FolderSep + "centers.bin", lambda, diff, mindist, noimprovement);
    }

    std::string savePath = folder + FolderSep + "status." + std::to_string(iteration) + "." + std::to_string(rank);
    auto out = f_createIO();
    if (out == nullptr || !out->Initialize(savePath.c_str(), std::ios::binary | std::ios::out)) {
        LOG(Helper::LogLevel::LL_Error, "Cannot open %s to write status.\n", savePath.c_str());
        exit(1);
    }

    CHECKIO(out, WriteBinary, sizeof(unsigned long long), (const char*)&localCount);
    CHECKIO(out, WriteBinary, sizeof(float), (const char*)&localDist);
    CHECKIO(out, WriteBinary, sizeof(float) * args._K * args._D, (const char*)args.newCenters);
    if (assign) {
        CHECKIO(out, WriteBinary, sizeof(int) * args._K, (const char*)args.counts);
        CHECKIO(out, WriteBinary, sizeof(float) * args._K, (const char*)args.weightedCounts);
    }
    else {
        CHECKIO(out, WriteBinary, sizeof(int) * args._K, (const char*)args.newCounts);
        CHECKIO(out, WriteBinary, sizeof(float) * args._K, (const char*)args.newWeightedCounts);
    }
    out->ShutDown();

    if (!options.m_syncscript.empty()) {
        system((options.m_syncscript + " upload " + folder + " " + std::to_string(options.m_totalparts) + " " + std::to_string(savecenters)).c_str());
    }
    else {
        LOG(Helper::LogLevel::LL_Error, "Error: Sync script is empty.\n");
    }
    return ErrorCode::Success;
}

template <typename T>
ErrorCode SyncLoadCenter(COMMON::KmeansArgs<T>& args, int rank, int iteration, unsigned long long &totalCount, float &currDist, float &lambda, float &diff, float &mindist, int &noimprovement, bool loadcenters)
{
    std::string folder = options.m_status + FolderSep + std::to_string(iteration);

    //TODO download
    if (!options.m_syncscript.empty()) {
        system((options.m_syncscript + " download " + folder + " " + std::to_string(options.m_totalparts) + " " + std::to_string(loadcenters)).c_str());
    }
    else {
        LOG(Helper::LogLevel::LL_Error, "Error: Sync script is empty.\n");
    }

    if (loadcenters) {
        if (!LoadCenters(args.newTCenters, args._K, args._D, folder + FolderSep + "centers.bin", &lambda, &diff, &mindist, &noimprovement)) {
            LOG(Helper::LogLevel::LL_Error, "Cannot load centers.\n");
            exit(1);
        }
    }

    memset(args.newCenters, 0, sizeof(float) * args._K * args._D);
    memset(args.counts, 0, sizeof(int) * args._K);
    memset(args.weightedCounts, 0, sizeof(float) * args._K);
    std::unique_ptr<char[]> buf(new char[sizeof(float) * args._K * args._D]);
    unsigned long long localCount;
    float localDist;

    totalCount = 0;
    currDist = 0;
    for (int part = 0; part < options.m_totalparts; part++) {
        std::string loadPath = folder + FolderSep + "status." + std::to_string(iteration) + "." + std::to_string(part);
        auto input = f_createIO();
        if (input == nullptr || !input->Initialize(loadPath.c_str(), std::ios::binary | std::ios::in)) {
            LOG(Helper::LogLevel::LL_Error, "Cannot open %s to read status.", loadPath.c_str());
            exit(1);
        }

        CHECKIO(input, ReadBinary, sizeof(unsigned long long), (char*)&localCount);
        totalCount += localCount;

        CHECKIO(input, ReadBinary, sizeof(float), (char*)&localDist);
        currDist += localDist;

        CHECKIO(input, ReadBinary, sizeof(float) * args._K * args._D, buf.get());
        for (int i = 0; i < args._K * args._D; i++) args.newCenters[i] += *((float*)(buf.get()) + i);

        CHECKIO(input, ReadBinary, sizeof(int) * args._K, buf.get());
        for (int i = 0; i < args._K; i++) {
            int partsize = *((int*)(buf.get()) + i);
            if (partsize >= 0 && args.counts[i] <= MaxSize - partsize) args.counts[i] += partsize;
            else {
                LOG(Helper::LogLevel::LL_Error, "Cluster %d counts overflow:%d + %d(%d)! Set it to MaxSize.\n", i, args.counts[i], partsize, part);
                args.counts[i] = MaxSize;
            }
        }

        CHECKIO(input, ReadBinary, sizeof(float) * args._K, buf.get());
        for (int i = 0; i < args._K; i++) args.weightedCounts[i] += *((float*)(buf.get()) + i);
    }
    return ErrorCode::Success;
}

template <typename T>
void ProcessWithoutMPI() {
    std::string rankstr = options.m_labels.substr(options.m_labels.rfind(".") + 1);
    int rank = std::stoi(rankstr);
    LOG(Helper::LogLevel::LL_Info, "DEBUG:rank--%d labels--%s\n", rank, options.m_labels.c_str());

    auto vectorReader = Helper::VectorSetReader::CreateInstance(std::make_shared<Helper::ReaderOptions>(options));
    options.m_inputFiles = Helper::StrUtils::ReplaceAll(options.m_inputFiles, "*", std::to_string(rank));
    if (ErrorCode::Success != vectorReader->LoadFile(options.m_inputFiles))
    {
        LOG(Helper::LogLevel::LL_Error, "Failed to read input file.\n");
        exit(1);
    }
    std::shared_ptr<VectorSet> vectors = vectorReader->GetVectorSet();
    std::shared_ptr<MetadataSet> metas = vectorReader->GetMetadataSet();
    if (options.m_distMethod == DistCalcMethod::Cosine) vectors->Normalize(options.m_threadNum);

    std::vector<float> weights(vectors->Count(), 0.0f);
    if (options.m_weightfile.compare("-") != 0) {
        options.m_weightfile = Helper::StrUtils::ReplaceAll(options.m_weightfile, "*", std::to_string(rank));
        std::ifstream win(options.m_weightfile, std::ifstream::binary);
        if (!win.is_open()) {
            LOG(Helper::LogLevel::LL_Error, "Rank %d failed to read weight file %s.\n", rank, options.m_weightfile.c_str());
            exit(1);
        }
        SizeType rows;
        win.read((char*)&rows, sizeof(SizeType));
        if (rows != vectors->Count()) {
            win.close();
            LOG(Helper::LogLevel::LL_Error, "Number of weights (%d) is not equal to number of vectors (%d).\n", rows, vectors->Count());
            exit(1);
        }
        win.read((char*)weights.data(), sizeof(float) * rows);
        win.close();
    }
    COMMON::Dataset<T> data(vectors->Count(), vectors->Dimension(), 1024*1024, vectors->Count() + 1, (T*)vectors->GetData());
    COMMON::KmeansArgs<T> args(options.m_clusterNum, vectors->Dimension(), vectors->Count(), options.m_threadNum, options.m_distMethod);
    COMMON::Dataset<LabelType> label(vectors->Count(), options.m_clusterassign, vectors->Count(), vectors->Count());
    std::vector<SizeType> localindices(data.R(), 0);
    for (SizeType i = 0; i < data.R(); i++) localindices[i] = i;
    args.ClearCounts();

    unsigned long long totalCount;
    float currDiff = 1.0, d = 0.0, currDist, minClusterDist = MaxDist;
    int iteration = options.m_recoveriter;
    int noImprovement = 0;

    if (rank == 0 && iteration < 0) {
        LOG(Helper::LogLevel::LL_Info, "rank 0 init centers\n");
        if (!LoadCenters(args.newTCenters, args._K, args._D, options.m_centers, &(options.m_lambda))) {
            if (options.m_seed >= 0) std::srand(options.m_seed);
            if (options.m_maxIter > 0 && options.m_lambda < -1e-6f) {
                float fBalanceFactor = COMMON::DynamicFactorSelect<T>(data, localindices, 0, data.R(), args, data.R());
                options.m_lambda = COMMON::Utils::GetBase<T>() * COMMON::Utils::GetBase<T>() / fBalanceFactor / data.R();
            }
            COMMON::InitCenters<T, T>(data, localindices, 0, data.R(), args, options.m_localSamples, options.m_initIter);
        }
    }
    if (iteration < 0) {
        iteration = 0;
        SyncSaveCenter(args, rank, iteration, data.R(), d, options.m_lambda, currDiff, minClusterDist, noImprovement, 2);
    }
    else {
        LOG(Helper::LogLevel::LL_Info, "recover from iteration:%d\n", iteration);
    }
    SyncLoadCenter(args, rank, iteration, totalCount, currDist, options.m_lambda, currDiff, minClusterDist, noImprovement, true);

    LOG(Helper::LogLevel::LL_Info, "rank %d  data:(%d,%d) machines:%d clusters:%d type:%d threads:%d lambda:%f samples:%d maxcountperpartition:%d\n",
        rank, data.R(), data.C(), options.m_totalparts, options.m_clusterNum, ((int)options.m_inputValueType), options.m_threadNum, options.m_lambda, options.m_localSamples, static_cast<unsigned long long>(totalCount * 1.0 / args._K * options.m_vectorfactor));

    while (noImprovement < 10 && currDiff > options.m_stopDifference && iteration < options.m_maxIter) {
        std::memcpy(args.centers, args.newTCenters, sizeof(T) * args._K * args._D);

        args.ClearCenters();
        args.ClearCounts();
        args.ClearDists(-MaxDist);
        d = MultipleClustersAssign<T>(data, localindices, 0, data.R(), args, label, true, (iteration == 0) ? 0.0f : options.m_lambda, weights, (iteration == 0) ? 0.0f : options.m_wlambda);

        SyncSaveCenter(args, rank, iteration + 1, data.R(), d, options.m_lambda, currDiff, minClusterDist, noImprovement, 0);
        if (rank == 0) {
            SyncLoadCenter(args, rank, iteration + 1, totalCount, currDist, options.m_lambda, currDiff, minClusterDist, noImprovement, false);
            currDiff = COMMON::RefineCenters<T, T>(data, args);
            if (currDist < minClusterDist) {
                noImprovement = 0;
                minClusterDist = currDist;
            }
            else {
                noImprovement++;
            }
            SyncSaveCenter(args, rank, iteration + 1, data.R(), d, options.m_lambda, currDiff, minClusterDist, noImprovement, 1);
        }
        else {
            SyncLoadCenter(args, rank, iteration + 1, totalCount, currDist, options.m_lambda, currDiff, minClusterDist, noImprovement, true);
        }
        iteration++;

        LOG(Helper::LogLevel::LL_Info, "iter %d dist:%f diff:%f\n", iteration, currDist, currDiff);
    }

    if (options.m_maxIter == 0) {
        std::memcpy(args.centers, args.newTCenters, sizeof(T) * args._K * args._D);
    }
    else {
        if (rank == 0) {
            for (int i = 0; i < args._K; i++)
                LOG(Helper::LogLevel::LL_Info, "cluster %d contains vectors:%d weights:%f\n", i, args.counts[i], args.weightedCounts[i]);
        }
    }
    d = 0;
    totalCount = static_cast<unsigned long long>(totalCount * 1.0 / args._K * options.m_vectorfactor);
    unsigned long long tmpTotalCount;
    for (SizeType i = 0; i < data.R(); i++) localindices[i] = i;
    std::vector<SizeType> myLimit(args._K, (options.m_hardcut == 0)? data.R() : (SizeType)(options.m_hardcut * totalCount / options.m_totalparts));
    std::memset(args.counts, 0, sizeof(SizeType) * args._K);
    args.ClearCounts();
    args.ClearDists(0);
    for (int i = 0; i < options.m_clusterassign - 1; i++) {
        d += HardMultipleClustersAssign<T>(data, localindices, 0, data.R(), args, label, myLimit.data(), weights, i, false);
        std::memcpy(myLimit.data(), args.counts, sizeof(SizeType) * args._K);
        SyncSaveCenter(args, rank, 10000 + iteration + 1 + i, data.R(), d, options.m_lambda, currDiff, minClusterDist, noImprovement, 0, true);
        SyncLoadCenter(args, rank, 10000 + iteration + 1 + i, tmpTotalCount, currDist, options.m_lambda, currDiff, minClusterDist, noImprovement, false);
        if (rank == 0) {
            LOG(Helper::LogLevel::LL_Info, "assign %d....................d:%f\n", i, d);
            for (int i = 0; i < args._K; i++)
                LOG(Helper::LogLevel::LL_Info, "cluster %d contains vectors:%d weights:%f\n", i, args.counts[i], args.weightedCounts[i]);
        }
        for (int k = 0; k < args._K; k++)
            if (totalCount > args.counts[k])
                myLimit[k] += (SizeType)((totalCount - args.counts[k]) / options.m_totalparts);
    }
    d += HardMultipleClustersAssign<T>(data, localindices, 0, data.R(), args, label, myLimit.data(), weights, options.m_clusterassign - 1, true);
    std::memcpy(args.newCounts, args.counts, sizeof(SizeType) * args._K);
    SyncSaveCenter(args, rank, 10000 + iteration + options.m_clusterassign, data.R(), d, options.m_lambda, currDiff, minClusterDist, noImprovement, 0, true);
    SyncLoadCenter(args, rank, 10000 + iteration + options.m_clusterassign, tmpTotalCount, currDist, options.m_lambda, currDiff, minClusterDist, noImprovement, false);

    if (label.Save(options.m_labels) != ErrorCode::Success) {
        LOG(Helper::LogLevel::LL_Error, "Failed to save labels.\n");
        exit(1);
    }
    if (rank == 0) {
        SaveCenters(args.centers, args._K, args._D, options.m_centers, options.m_lambda);
        LOG(Helper::LogLevel::LL_Info, "final dist:%f\n", currDist);
        for (int i = 0; i < args._K; i++)
            LOG(Helper::LogLevel::LL_Status, "cluster %d contains vectors:%d weights:%f\n", i, args.counts[i], args.weightedCounts[i]);
    }
}

template <typename T>
void Partition() {
    if (options.m_outdir.compare("-") == 0) return;

    auto vectorReader = Helper::VectorSetReader::CreateInstance(std::make_shared<Helper::ReaderOptions>(options));
    if (ErrorCode::Success != vectorReader->LoadFile(options.m_inputFiles))
    {
        LOG(Helper::LogLevel::LL_Error, "Failed to read input file.\n");
        exit(1);
    }
    std::shared_ptr<VectorSet> vectors = vectorReader->GetVectorSet();
    std::shared_ptr<MetadataSet> metas = vectorReader->GetMetadataSet();
    if (options.m_distMethod == DistCalcMethod::Cosine) vectors->Normalize(options.m_threadNum);

    COMMON::Dataset<T> data(vectors->Count(), vectors->Dimension(), 1024*1024, vectors->Count() + 1, (T*)vectors->GetData());

    COMMON::Dataset<LabelType> label;
    if (label.Load(options.m_labels, vectors->Count(), vectors->Count()) != ErrorCode::Success) {
        LOG(Helper::LogLevel::LL_Error, "Failed to read labels.\n");
        exit(1);
    }

    std::string taskId = options.m_labels.substr(options.m_labels.rfind(".") + 1);
    for (int i = 0; i < options.m_clusterNum; i++) {
        std::string vecfile = options.m_outdir + "/" + options.m_outfile + "." + taskId + "." + std::to_string(i);
        std::string metafile = options.m_outdir + "/" + options.m_outmetafile + "." + taskId + "." + std::to_string(i);
        std::string metaindexfile = options.m_outdir + "/" + options.m_outmetaindexfile + "." + taskId + "." + std::to_string(i);
        std::shared_ptr<Helper::DiskIO> out = f_createIO(), metaout = f_createIO(), metaindexout = f_createIO();
        if (out == nullptr || !out->Initialize(vecfile.c_str(), std::ios::binary | std::ios::out)) {
            LOG(Helper::LogLevel::LL_Error, "Cannot open %s to write.\n", vecfile.c_str());
            exit(1);
        }
        if (metaout == nullptr || !metaout->Initialize(metafile.c_str(), std::ios::binary | std::ios::out)) {
            LOG(Helper::LogLevel::LL_Error, "Cannot open %s to write.\n", metafile.c_str());
            exit(1);
        }
        if (metaindexout == nullptr || !metaindexout->Initialize(metaindexfile.c_str(), std::ios::binary | std::ios::out)) {
            LOG(Helper::LogLevel::LL_Error, "Cannot open %s to write.\n", metaindexfile.c_str());
            exit(1);
        }

        int rows = data.R(), cols = data.C();
        CHECKIO(out, WriteBinary, sizeof(int), (char*)(&rows));
        CHECKIO(out, WriteBinary, sizeof(int), (char*)(&cols));
        if (metas != nullptr) CHECKIO(metaindexout, WriteBinary, sizeof(int), (char*)(&rows));

        std::uint64_t offset = 0;
        int records = 0;
        for (int k = 0; k < data.R(); k++) {
            for (int kk = 0; kk < label.C(); kk++) {
                if (label[k][kk] == (LabelType)i) {
                    CHECKIO(out, WriteBinary, sizeof(T) * cols, (char*)(data[k]));
                    if (metas != nullptr) {
                        ByteArray meta = metas->GetMetadata(k);
                        CHECKIO(metaout, WriteBinary, meta.Length(), (const char*)meta.Data());
                        CHECKIO(metaindexout, WriteBinary, sizeof(std::uint64_t), (char*)(&offset));
                        offset += meta.Length();
                    }
                    records++;
                }
            }
        }
        LOG(Helper::LogLevel::LL_Info, "part %s cluster %d: %d vectors, %llu bytes meta.\n", taskId.c_str(), i, records, offset);

        if (metas != nullptr) CHECKIO(metaindexout, WriteBinary, sizeof(std::uint64_t), (char*)(&offset));
        CHECKIO(out, WriteBinary, sizeof(int), (char*)(&records), 0);
        CHECKIO(metaindexout, WriteBinary, sizeof(int), (char*)(&records), 0);

        out->ShutDown();
        metaout->ShutDown();
        metaindexout->ShutDown();
    }
}

int main(int argc, char* argv[]) {
    if (!options.Parse(argc - 1, argv + 1))
    {
        exit(1);
    }

    if (options.m_stage.compare("Clustering") == 0) {
        switch (options.m_inputValueType) {
        case SPTAG::VectorValueType::Float:
            Process<float>(MPI_FLOAT);
            break;
        case SPTAG::VectorValueType::Int16:
            Process<std::int16_t>(MPI_SHORT);
            break;
        case SPTAG::VectorValueType::Int8:
            Process<std::int8_t>(MPI_CHAR);
            break;
        case SPTAG::VectorValueType::UInt8:
            Process<std::uint8_t>(MPI_CHAR);
            break;
        default:
            LOG(Helper::LogLevel::LL_Error, "Error data type!\n");
        }
    }
    else if (options.m_stage.compare("ClusteringWithoutMPI") == 0) {
        switch (options.m_inputValueType) {
        case SPTAG::VectorValueType::Float:
            ProcessWithoutMPI<float>();
            break;
        case SPTAG::VectorValueType::Int16:
            ProcessWithoutMPI<std::int16_t>();
            break;
        case SPTAG::VectorValueType::Int8:
            ProcessWithoutMPI<std::int8_t>();
            break;
        case SPTAG::VectorValueType::UInt8:
            ProcessWithoutMPI<std::uint8_t>();
            break;
        default:
            LOG(Helper::LogLevel::LL_Error, "Error data type!\n");
        }
    }
    else if (options.m_stage.compare("LocalPartition") == 0) {
        switch (options.m_inputValueType) {
        case SPTAG::VectorValueType::Float:
            Partition<float>();
            break;
        case SPTAG::VectorValueType::Int16:
            Partition<std::int16_t>();
            break;
        case SPTAG::VectorValueType::Int8:
            Partition<std::int8_t>();
            break;
        case SPTAG::VectorValueType::UInt8:
            Partition<std::uint8_t>();
            break;
        default:
            LOG(Helper::LogLevel::LL_Error, "Error data type!\n");
        }
    }
    return 0;
}
