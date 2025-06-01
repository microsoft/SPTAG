#!/bin/bash -x

if [ "$1" == "init_env" ]; then
sudo apt install -y cmake gcc-9 g++-9 libjemalloc-dev libsnappy-dev libgflags-dev pkg-config swig libboost-all-dev libtbb-dev libisal-dev gnuplot
git clone https://github.com/SPFresh/SPFresh
cd SPFresh
git submodule update --init --recursive
cd ThirdParty/spdk
sudo bash ./scripts/pkgdep.sh
CC=gcc-9 ./configure
CC=gcc-9 make -j
cd ../isal-l_crypto
./autogen.sh
./configure
make -j
cd ..
git clone https://github.com/PtilopsisL/rocksdb
cd rocksdb
mkdir build && cd build
cmake -DUSE_RTTI=1 -DWITH_JEMALLOC=1 -DWITH_SNAPPY=1 -DCMAKE_C_COMPILER=gcc-9 -DCMAKE_CXX_COMPILER=g++-9 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-fPIC" ..
make -j
sudo make install
cd ../../..
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DGPU=OFF ..
# add #include <mutex> in SPFresh/AnnService/inc/Helper/ConcurrentSet.h and SPFresh/AnnService/inc/Helper/Logging.h
make -j
cd ..

cp Script_AE/bdev.json .
sudo nvme format /dev/nvme0n1
sudo ./ThirdParty/spdk/scripts/setup.sh
# it will print out:
#1462:00:00.0 (1414 b111): nvme -> uio_pci_generic
#01ca:00:00.0 (1414 b111): nvme -> uio_pci_generic
# fill the 1462:00:00.0 into bdev.json and use the traddr in the PCI_ALLOWED=1462.00.00.0 in the run_update commend
else
echo "go to SPFresh Directory"
#cd SPFresh
fi

cd Release
dataset="sift"
datatype='UInt8'
dim=128
basefile="base.1B.u8bin"
queryfile="query.public.10K.u8bin"

testscale="1m"
updateto="2m"
testscale_number=1000000
updateto_number=2000000
query_number=10000
batch_size=10000

if [ "$1" == "create_dataset" ]; then
mkdir -p ${dataset}1b
cd ${dataset}1b
if [ "$dataset" == "sift" ]; then
    echo "begin download $dataset..."

    if [ ! -f "$basefile" ]; then
        wget https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/$basefile
    fi
    if [ ! -f "$queryfile" ]; then
        wget https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/$queryfile
    fi
else
    #TODO: download spacev dataset
    echo "not support $dataset..."
fi

pip3 install numpy
echo 'import numpy as np
import argparse
import struct

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="The input file (.fvecs)")
    parser.add_argument("--dst", help="The output file (.fvecs)")
    parser.add_argument("--topk", type=int, help="The number of element to pick up")
    return parser.parse_args()


if __name__ == "__main__":
    args = process_args()

    # Read topk vector one by one
    vecs = ""
    row_bin = "";
    dim_bin = "";
    with open(args.src, "rb") as f:

        row_bin = f.read(4)
        assert row_bin != b""
        row, = struct.unpack("i", row_bin)

        dim_bin = f.read(4)
        assert dim_bin != b""
        dim, = struct.unpack("i", dim_bin)

        vecs = f.read(args.topk * dim)

    with open(args.dst, "wb") as f:
        f.write(struct.pack("i", args.topk))
        f.write(dim_bin)
        f.write(vecs)
' > generate_dataset.py
echo "python3 generate_dataset.py --src $basefile --dst $dataset.$testscale.bin --topk $testscale_number"
python3 generate_dataset.py --src $basefile --dst $dataset.$testscale.bin --topk $testscale_number
echo "python3 generate_dataset.py --src $basefile --dst $dataset.$updateto.bin --topk $updateto_number"
python3 generate_dataset.py --src $basefile --dst $dataset.$updateto.bin --topk $updateto_number

toolpath=..
setname="6c VectorPath=${dataset}${testscale}_update_set"
truthname="18c TruthPath=${dataset}${testscale}_update_truth"
deletesetname="${dataset}${testscale}_update_set"
reservesetname="${dataset}${testscale}_update_reserve"
currentsetname="${dataset}${testscale}_update_current"
echo "[Base]
ValueType=$datatype
DistCalcMethod=L2
IndexAlgoType=BKT
Dim=$dim
VectorPath=$dataset.$testscale.bin
VectorType=DEFAULT
VectorSize=$testscale_number
VectorDelimiter=
QueryPath=$queryfile
QueryType=DEFAULT
QuerySize=$query_number
QueryDelimiter=
WarmupPath=
WarmupType=DEFAULT
WarmupSize=$query_number
WarmupDelimiter=
TruthPath=${dataset}${testscale}_truth
TruthType=DEFAULT
GenerateTruth=true

[SearchSSDIndex]
ResultNum=100
NumberOfThreads=16
" > genTruth.ini
$toolpath/ssdserving genTruth.ini

echo "[Base]
ValueType=$datatype
DistCalcMethod=L2
IndexAlgoType=BKT
Dim=$dim
VectorPath=${dataset}${testscale}_update_set88
VectorType=DEFAULT
VectorSize=$testscale_number
VectorDelimiter=
QueryPath=$queryfile
QueryType=DEFAULT
QuerySize=$query_number
QueryDelimiter=
WarmupPath=
WarmupType=DEFAULT
WarmupSize=$query_number
WarmupDelimiter=
TruthPath=${dataset}${testscale}_update_truth88
TruthType=DEFAULT
GenerateTruth=true

[SearchSSDIndex]
ResultNum=100
NumberOfThreads=16" > genTruth.ini
for i in {0..99}
do
    echo "start batch $i..."
    $toolpath/usefultool --GenTrace true --vectortype $datatype --VectorPath $dataset.$updateto.bin --filetype DEFAULT --UpdateSize $batch_size --BaseNum $testscale_number --ReserveNum $testscale_number --CurrentListFileName ${dataset}${testscale}_update_current --ReserveListFileName ${dataset}${testscale}_update_reserve --TraceFileName ${dataset}${testscale}_update_trace --NewDataSetFileName ${dataset}${testscale}_update_set -d $dim --Batch $i -f DEFAULT
    newsetname=$setname$i
    newtruthname=$truthname$i
    newdeletesetname=$deletesetname$i
    newreservesetname=$reservesetname$i
    newcurrentsetname=$currentsetname$i
    sed -i "$newsetname" genTruth.ini
    sed -i "$newtruthname" genTruth.ini
    $toolpath/ssdserving genTruth.ini
    $toolpath/usefultool --ConvertTruth true --vectortype $datatype --VectorPath $dataset.$updateto.bin --filetype DEFAULT --UpdateSize $batch_size --BaseNum $testscale_number --ReserveNum $testscale_number --CurrentListFileName ${dataset}${testscale}_update_current --ReserveListFileName ${dataset}${testscale}_update_reserve --TraceFileName ${dataset}${testscale}_update_trace --NewDataSetFileName ${dataset}${testscale}_update_set -d $dim --Batch $i -f DEFAULT --truthPath ${dataset}${testscale}_update_truth --truthType DEFAULT --querySize $query_number --resultNum 100
done
cd ..
fi

if [ "$1" == "build_index" ]; then
echo "[Base]
ValueType=$datatype
DistCalcMethod=L2
IndexAlgoType=BKT
Dim=$dim
VectorPath=${dataset}1b/$dataset.$testscale.bin
VectorType=DEFAULT
VectorSize=$testscale_number
VectorDelimiter=
QueryPath=${dataset}1b/$queryfile
QueryType=DEFAULT
QuerySize=$query_number
QueryDelimiter=
WarmupPath=
WarmupType=DEFAULT
WarmupSize=$query_number
WarmupDelimiter=
TruthPath=${dataset}1b/
TruthType=DEFAULT
GenerateTruth=false
HeadVectorIDs=head_vectors_ID_$datatype\_L2_base_DEFUALT.bin
HeadVectors=head_vectors_$datatype\_L2_base_DEFUALT.bin
IndexDirectory=store_${dataset}${testscale}/
HeadIndexFolder=head_index

[SelectHead]
isExecute=true
TreeNumber=1
BKTKmeansK=32
BKTLeafSize=8
SamplesNumber=1000
NumberOfThreads=16
SaveBKT=false
AnalyzeOnly=false
CalcStd=true
SelectDynamically=true
NoOutput=false
SelectThreshold=12
SplitFactor=9
SplitThreshold=18
Ratio=0.1
RecursiveCheckSmallCluster=true
PrintSizeCount=true

[BuildHead]
isExecute=true
NumberOfThreads=16

[BuildSSDIndex]
isExecute=true
BuildSsdIndex=true
InternalResultNum=64
NumberOfThreads=16
ReplicaCount=8
PostingPageLimit=4
OutputEmptyReplicaID=1
TmpDir=store_${dataset}${testscale}/tmpdir" > build_SPANN_store_${dataset}${testscale}.ini
./ssdserving build_SPANN_store_${dataset}${testscale}.ini
echo "[Index]
IndexAlgoType=SPANN
ValueType=$datatype

[Base]
ValueType=$datatype
DistCalcMethod=L2
IndexAlgoType=BKT
Dim=$dim
VectorPath=${dataset}1b/$dataset.$testscale.bin
VectorType=DEFAULT
VectorSize=$testscale_number
VectorDelimiter=
QueryPath=${dataset}1b/$queryfile
QueryType=DEFAULT
QuerySize=$query_number
QueryDelimiter=
WarmupPath=
WarmupType=DEFAULT
WarmupSize=$query_number
WarmupDelimiter=
TruthPath=${dataset}1b/
TruthType=DEFAULT
GenerateTruth=false
HeadVectorIDs=head_vectors_ID_$datatype\_L2_base_DEFUALT.bin
HeadVectors=head_vectors_$datatype\_L2_base_DEFUALT.bin
IndexDirectory=store_${dataset}${testscale}/
HeadIndexFolder=head_index


[SelectHead]
isExecute=false
TreeNumber=1
BKTKmeansK=32
BKTLeafSize=8
SamplesNumber=1000
NumberOfThreads=16
SaveBKT=false
AnalyzeOnly=false
CalcStd=true
SelectDynamically=true
NoOutput=false
SelectThreshold=12
SplitFactor=9
SplitThreshold=18
Ratio=0.15
RecursiveCheckSmallCluster=true
PrintSizeCount=true

[BuildHead]
isExecute=false
TreeFilePath=tree.bin
GraphFilePath=graph.bin
VectorFilePath=vectors.bin
DeleteVectorFilePath=deletes.bin
EnableBfs=0
BKTNumber=1
BKTKmeansK=32
BKTLeafSize=8
Samples=1000
BKTLambdaFactor=100.000000
TPTNumber=32
TPTLeafSize=2000
NumTopDimensionTpTreeSplit=5
NeighborhoodSize=32
GraphNeighborhoodScale=2.000000
GraphCEFScale=2.000000
RefineIterations=2
EnableRebuild=0
CEF=1000
AddCEF=500
MaxCheckForRefineGraph=8192
RNGFactor=1.000000
GPUGraphType=2
GPURefineSteps=0
GPURefineDepth=30
GPULeafSize=500
HeadNumGPUs=1
TPTBalanceFactor=2
NumberOfThreads=16
DistCalcMethod=L2
DeletePercentageForRefine=0.400000
AddCountForRebuild=1000
MaxCheck=4096
ThresholdOfNumberOfContinuousNoBetterPropagation=3
NumberOfInitialDynamicPivots=50
NumberOfOtherDynamicPivots=4
HashTableExponent=2
DataBlockSize=1048576
DataCapacity=2147483647
MetaRecordSize=10

[BuildSSDIndex]
isExecute=true
BuildSsdIndex=false
NumberOfThreads=16
InternalResultNum=64
ReplicaCount=8
PostingPageLimit=4
OutputEmptyReplicaID=1
TmpDir=store_${dataset}${testscale}/tmpdir
UseSPDK=true
ExcludeHead=false
UseDirectIO=true
ResultNum=10
SearchInternalResultNum=64
SearchThreadNum=2
SearchTimes=1
Update=true
SteadyState=true
Days=100
InsertThreadNum=1
AppendThreadNum=1
ReassignThreadNum=0
TruthFilePrefix=${dataset}1b/
FullVectorPath=${dataset}1b/$dataset.$updateto.bin
DisableReassign=false
ReassignK=64
LatencyLimit=20.0
CalTruth=true
SearchPostingPageLimit=4
MaxDistRatio=1000000
SearchDuringUpdate=true
MergeThreshold=10
UpdateFilePrefix=${dataset}1b/${dataset}${testscale}_update_trace
DeleteQPS=800
ShowUpdateProgress=false
Sampling=4
BufferLength=6
InPlace=true
SearchResult=${dataset}1b/result_spfresh_balance
EndVectorNum=2000000" > store_${dataset}${testscale}/indexloader.ini
fi

if [ "$1" == "run_update" ]; then
#SPDK version
#PCI_ALLOWED="1462:00:00.0" SPFRESH_SPDK_USE_SSD_IMPL=1 SPFRESH_SPDK_CONF=../bdev.json SPFRESH_SPDK_BDEV=Nvme0n1 sudo -E ./spfresh store_${dataset}${testscale} |tee log_spfresh.log

#FileIO version
echo "[Index]
IndexAlgoType=SPANN
ValueType=$datatype

[Base]
ValueType=$datatype
DistCalcMethod=L2
IndexAlgoType=BKT
Dim=$dim
VectorPath=${dataset}1b/$dataset.$testscale.bin
VectorType=DEFAULT
VectorSize=$testscale_number
VectorDelimiter=
QueryPath=${dataset}1b/$queryfile
QueryType=DEFAULT
QuerySize=$query_number
QueryDelimiter=
WarmupPath=
WarmupType=DEFAULT
WarmupSize=$query_number
WarmupDelimiter=
TruthPath=${dataset}1b/
TruthType=DEFAULT
GenerateTruth=false
HeadVectorIDs=head_vectors_ID_$datatype\_L2_base_DEFUALT.bin
HeadVectors=head_vectors_$datatype\_L2_base_DEFUALT.bin
IndexDirectory=store_${dataset}${testscale}/
HeadIndexFolder=head_index


[SelectHead]
isExecute=false
TreeNumber=1
BKTKmeansK=32
BKTLeafSize=8
SamplesNumber=1000
NumberOfThreads=16
SaveBKT=false
AnalyzeOnly=false
CalcStd=true
SelectDynamically=true
NoOutput=false
SelectThreshold=12
SplitFactor=9
SplitThreshold=18
Ratio=0.15
RecursiveCheckSmallCluster=true
PrintSizeCount=true

[BuildHead]
isExecute=false
TreeFilePath=tree.bin
GraphFilePath=graph.bin
VectorFilePath=vectors.bin
DeleteVectorFilePath=deletes.bin
EnableBfs=0
BKTNumber=1
BKTKmeansK=32
BKTLeafSize=8
Samples=1000
BKTLambdaFactor=100.000000
TPTNumber=32
TPTLeafSize=2000
NumTopDimensionTpTreeSplit=5
NeighborhoodSize=32
GraphNeighborhoodScale=2.000000
GraphCEFScale=2.000000
RefineIterations=2
EnableRebuild=0
CEF=1000
AddCEF=500
MaxCheckForRefineGraph=8192
RNGFactor=1.000000
GPUGraphType=2
GPURefineSteps=0
GPURefineDepth=30
GPULeafSize=500
HeadNumGPUs=1
TPTBalanceFactor=2
NumberOfThreads=16
DistCalcMethod=L2
DeletePercentageForRefine=0.400000
AddCountForRebuild=1000
MaxCheck=4096
ThresholdOfNumberOfContinuousNoBetterPropagation=3
NumberOfInitialDynamicPivots=50
NumberOfOtherDynamicPivots=4
HashTableExponent=2
DataBlockSize=1048576
DataCapacity=2147483647
MetaRecordSize=10

[BuildSSDIndex]
isExecute=true
BuildSsdIndex=false
NumberOfThreads=16
InternalResultNum=64
ReplicaCount=8
PostingPageLimit=4
OutputEmptyReplicaID=1
TmpDir=store_${dataset}${testscale}/tmpdir
UseSPDK=false
UseKV=false
UseFileIO=true
SpdkBatchSize=64
ExcludeHead=false
UseDirectIO=false
ResultNum=10
SearchInternalResultNum=64
SearchThreadNum=2
SearchTimes=1
Update=true
SteadyState=true
Days=100
InsertThreadNum=1
AppendThreadNum=1
ReassignThreadNum=0
TruthFilePrefix=${dataset}1b/${dataset}${testscale}_update_truth_after
FullVectorPath=${dataset}1b/$dataset.$updateto.bin
DisableReassign=false
ReassignK=64
LatencyLimit=50.0
CalTruth=true
SearchPostingPageLimit=4
MaxDistRatio=1000000
SearchDuringUpdate=true
MergeThreshold=10
UpdateFilePrefix=${dataset}1b/${dataset}${testscale}_update_trace
DeleteQPS=800
ShowUpdateProgress=false
Sampling=4
BufferLength=6
InPlace=true
SearchResult=${dataset}1b/result_spfresh_balance
LoadAllVectors=true
PersistentBufferPath=test_pbfile
SsdInfoFile=test_ssdinfo
SpdkMappingPath=test_spdkmapping
EndVectorNum=2000000" > store_${dataset}${testscale}/indexloader.ini

PCI_ALLOWED="1462:00:00.0" SPFRESH_SPDK_USE_SSD_IMPL=1 SPFRESH_SPDK_CONF=../bdev.json SPFRESH_SPDK_BDEV=Nvme0n1 SPFRESH_FILE_IO_PATH=testfile SPFRESH_FILE_IO_USE_CACHE=False SPFRESH_FILE_IO_THREAD_NUM=16 SPFRESH_FILE_IO_USE_LOCK=False SPFRESH_FILE_IO_LOCK_SIZE=262144 sudo -E ./spfresh store_${dataset}${testscale} |tee log_spfresh.log
fi

if [ "$1" == "plot_result" ]; then
cp ../Script_AE/Figure6/process_spfresh.py .
python3 process_spfresh.py log_spfresh.log overall_performance_${dataset}_spfresh_result.csv

mkdir -p spfresh_result
cp -rf ${dataset}1b/result_spfresh_balance* spfresh_result

resultnamePrefix=/spfresh_result/
i=-1
for FILE in `ls -v1 ./spfresh_result/`
do
    if [ $i -eq -1 ];
    then
        ./usefultool --CallRecall true --resultNum 10 --queryPath ${dataset}1b/$queryfile --searchResult $PWD$resultnamePrefix$FILE --truthType DEFAULT --truthPath ${dataset}1b/${dataset}${testscale}_truth --VectorPath ${dataset}1b/$dataset.$updateto.bin --vectortype $datatype -d $dim -f DEFAULT |tee log_spfresh_$i
    else
        ./usefultool --CallRecall true --resultNum 10 --queryPath ${dataset}1b/$queryfile --searchResult $PWD$resultnamePrefix$FILE --truthType DEFAULT --truthPath ${dataset}1b/${dataset}${testscale}_update_truth_after$i --VectorPath ${dataset}1b/$dataset.$updateto.bin --vectortype $datatype -d $dim -f DEFAULT |tee log_spfresh_$i
    fi
    let "i=i+1"
done
cp ../Script_AE/Figure6/OverallPerformance_merge_result.py .
cp ../Script_AE/Figure6/overall_performance_spacev_new.p .
python3 OverallPerformance_merge_result.py log_spfresh_ log_spfresh_ log_spfresh_ overall_performance_${dataset}_spfresh_result.csv overall_performance_${dataset}_spfresh_result.csv overall_performance_${dataset}_spfresh_result.csv
gnuplot overall_performance_spacev_new.p
fi
