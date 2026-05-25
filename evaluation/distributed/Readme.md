## 0. each machine compile code with LARGEVID
```bash
python3 -m venv ~/.local
wget https://bootstrap.pypa.io/get-pip.py
~/.local/bin/python3 get-pip.py
~/.local/bin/pip3 install numpy openmpi mpi4py
cmake .. -DTIKV=ON -DTBB=ON -DCMAKE_BUILD_TYPE=Release -DLARGEVID=ON
make -j
```

## 1. Prepare data: split into #machine partitions and copy data to different machines

```bash
python3 split_and_convert_data.py --data_file=/mnt/md0/qi/distributed_build/perftest_vector.bin.UInt8_999000000_128 --meta_file=/mnt/md0/qi/distributed_build/perftest_meta.bin.0_999000000 --metaidx_file=/mnt/md0/qi/distributed_build/perftest_metaidx.bin.0_999000000 --data_type=uint8 --vid_intype=int32 --batch_size=1000000 --partitions=3 --vid_outtype=int64 --output_dir=/mnt/md0/qi/distributed_build/
scp /mnt/md0/qi/distributed_build/*.bin.1 <hostname1>:/mnt/md0/qi/distributed_build/
scp /mnt/md0/qi/distributed_build/*.bin.2 <hostname2>:/mnt/md0/qi/distributed_build/
```

## 2. distributed clustering
create a myhosts file:
```bash
172.27.0.4 slots=1
172.27.0.6 slots=1
172.27.0.5 slots=1
```
run distributed clustering example:
```bash
#mpirun -np 3 ./balanceddatapartition 
#-d 128          # Dimension
#--filetype DEFAULT # Input file format (bin)
#--vectortype UInt8 # Data type
#-i              /mnt/md0/qi/distributed_build/vectors.bin.\*,/mnt/md0/qi/distributed_build/meta.bin.\*,/mnt/md0/qi/distributed_build/metaidx.bin.\* # Input data pattern
#-c 3            # Number of centers
#-t 48            # Number of threads
#-l 0.0000005    # Balanced clustering parameter (higher = more balanced)
#-s 10000        # Initial clustering samples
#-m L2           # Similarity metric
#-e 0            # Random seed
#-x 3            # Init iterations
#-r 100          # K-means iterations
#-a 4            # Assign to the 4 closest shards
#--closurescale 1.03 # Assignment limit: only assign to shards within 1.3x distance
#--vectorscale 1.2   # Shard capacity limit: max 1.2x original size
#--hard 0        # Hard limit for vectorscale (0 = ensure at least one replica per vector)
#--gid          /mnt/md0/qi/distributed_build/vid.bin.\*
#--centers      /mnt/md0/qi/distributed_build/centersNew.bin # Output center file
#--labels       /mnt/md0/qi/distributed_build/labelsNew.bin   # Output label file
#--stage Clustering #Clustering/LocalPartition, default Clustering

# generate centroids and labels
(mpirun --mca btl_tcp_if_include eth0 --hostfile myhosts -np 3 ./balanceddatapartition -d 128 --filetype DEFAULT --vectortype UInt8 -i /mnt/md0/qi/distributed_build/vectors.bin.\*,/mnt/md0/qi/distributed_build/meta.bin.\*,/mnt/md0/qi/distributed_build/metaidx.bin.\* -c 6 -t 32 -l 0.00000005 -s 10000 -m L2 -e 0 -x 3 -r 100 -a 4 --closurescale 1.03 --vectorscale 1.2 --hard 0 --gid /mnt/md0/qi/distributed_build/vid.bin.\* --centers /mnt/md0/qi/distributed_build/centersNew.bin --labels /mnt/md0/qi/distributed_build/labelsNew.bin &> test_log &)

mpirun --mca btl_tcp_if_include eth0 --hostfile myhosts -np 3 /bin/bash -c "rm -rf /mnt/md0/qi/distributed_build/clustered/*"

#generate clustered data
(mpirun --mca btl_tcp_if_include eth0 --hostfile myhosts -np 3 ./balanceddatapartition -d 128 --filetype DEFAULT --vectortype UInt8 -i /mnt/md0/qi/distributed_build/vectors.bin.\*,/mnt/md0/qi/distributed_build/meta.bin.\*,/mnt/md0/qi/distributed_build/metaidx.bin.\* -c 6 -t 32 -l 0.00000005 -s 10000 -m L2 -e 0 -x 3 -r 100 -a 4 --closurescale 1.03 --vectorscale 1.2 --hard 0 --gid /mnt/md0/qi/distributed_build/vid.bin.\* --centers /mnt/md0/qi/distributed_build/centersNew.bin --labels /mnt/md0/qi/distributed_build/labelsNew.bin.\* --stage LocalPartition --outdir /mnt/md0/qi/distributed_build/clustered &> test_log &)
```

## 3. shuffle data and build index in parallel

prepare a configure file for index build: configure.template
```bash
[Base]
DistCalcMethod=L2
IndexAlgoType=BKT
VectorPath=/mnt/md0/qi/distributed_build/merged/vectors.bin.*
GlobalIDPath=/mnt/md0/qi/distributed_build/merged/vid.bin.*
ValueType=UInt8
Dim=128
IndexDirectory=/mnt/md0/qi/distributed_build/SPANN

[SelectHead]
isExecute=true
NumberOfThreads=32
SelectHeadType=BKT
SelectThreshold=0
SplitFactor=0
SplitThreshold=0
Ratio=0.2
ParallelBKTBuild=true

[BuildHead]
isExecute=true
AddCountForRebuild=10000
NumberOfThreads=32

[BuildSSDIndex]
isExecute=true
BuildSsdIndex=true
InternalResultNum=64
SearchInternalResultNum=64
NumberOfThreads=32
PostingPageLimit=4
SearchPostingPageLimit=4
TmpDir=tmpdir
SpdkBatchSize=64
ExcludeHead=false
ResultNum=10
SearchThreadNum=4
Update=true
SteadyState=true
InsertThreadNum=1
AppendThreadNum=4
ReassignThreadNum=0
DisableReassign=false
ReassignK=64
SearchDuringUpdate=true
MergeThreshold=10
Sampling=4
BufferLength=4
InPlace=true
StartFileSizeGB=1
OneClusterCutMax=true
ConsistencyCheck=false
ChecksumCheck=false
ChecksumInRead=false
AsyncMergeInSearch=true
DeletePercentageForRefine=0.4
AsyncAppendQueueSize=0
AllowZeroReplica=false
ShareDB=true            
Layers=2
LatencyLimit=100
MaxCheck=8192
UseMultiChunkPosting=false
VersionCacheMaxChunks=100000
MaxID=2000000000
Storage=TIKVIO
TiKVPDAddresses=annservicFX071Y:2379,annservicKFECPH:2379,annservicP92MC8:2379
TiKVKeyPrefix=qi_1b_l2
```

```bash
mpirun --mca btl_tcp_if_include eth0 --hostfile myhosts -np 3 ~/.local/bin/python3 ~/cheqi/SPTAG/evaluation/distributed/shuffle_data.py /mnt/md0/qi/distributed_build/clustered /mnt/md0/qi/distributed_build/merged 6 myhosts q configure.template
mpirun --mca btl_tcp_if_include eth0 --hostfile myhosts -np 3 ./indexbuilder --dimension 128 --vectortype UInt8 --filetype DEFAULT --outputfolder /mnt/md0/qi/distributed_build/SPANN --algo SPANN --thread 32 --config configure.template.ini
```

## 4. merge head index
```bash
python3 merge_head.py /mnt/md0/qi/distributed_build/SPANN /mnt/md0/qi/distributed_build/mergedhead myhosts q
./indexbuilder --dimension 128 --vectortype UInt8 --filetype DEFAULT --outputfolder /mnt/md0/qi/distributed_build/mergedhead/HeadIndex --algo BKT --thread 32 --input /mnt/md0/qi/distributed_build/mergedhead/vectors.bin Index.AddCountForRebuild=10000 Index.ParallelBKTBuild=true
# replace HeadIndex and SPTAGHeadVectorIDs.bin in SPANN index folder
```

## 5. Use SPFreshTest to test recall
prepare a configure file benchmark.ini
```bash
[Benchmark]
VectorPath=sift1b/base.1B.u8bin
QueryPath=sift1b/query.public.10K.u8bin
TruthPath=truth
IndexPath=/mnt/md0/qi/distributed_build/SPANN
ValueType=UInt8
Dimension=128
BaseVectorCount=999000000
InsertVectorCount=1000000
DeleteVectorCount=0
BatchNum=10
TopK=5
NumSearchThreads=4
NumInsertThreads=16
AppendThreadNum=48
NumSearchDuringInsertThreads=1
NumQueries=200
DistMethod=L2
Rebuild=false
Resume=-1
Layers=2


[SelectHead]
ParallelBKTBuild=true

[BuildSSDIndex]
LatencyLimit=100
MaxCheck=8192
SearchInternalResultNum=64
UseMultiChunkPosting=false
ReassignK=64
AsyncMergeInSearch=true
VersionCacheMaxChunks=100000
MaxID=2000000000
Storage=TIKVIO
TiKVPDAddresses=annservicFX071Y:2379,annservicKFECPH:2379,annservicP92MC8:2379
TiKVKeyPrefix=qi_1b_l2
```
Test Recall:
```bash
export BENCHMARK_CONFIG=benchmark.ini
SPTAGTest --run_test=SPFreshTest/RunBenchmarkFromConfig
```