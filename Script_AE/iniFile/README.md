## **If want to generate these data**

> In our paper, we use SIFT and SPACEV dataset and synthetic benchmark.
these dataset can be download here：
```
http://big-ann-benchmarks.com/neurips21.html
```
> Download Dataset
```bash
wget https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin
wget https://comp21storage.blob.core.windows.net/publiccontainer/comp21/spacev1b/spacev1b_base.i8bin
```
> Download Query
```bash
wget https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin
wget https://comp21storage.blob.core.windows.net/publiccontainer/comp21/spacev1b/query.i8bin
```

> To generate dataset for Overall Performance
```bash
# generate dataset for index build
python generate_dataset.py --src source_file_path --dst output_file_path --topk 100000000
# generate dataset for update
python generate_dataset.py --src source_file_path --dst output_file_path --topk 200000000
```
> To generate trace for Overall Performance & generate truth for Overall Performance

This will takes hundreds of hours to generate groundTruth (the nearest K vectors of all queries) without GPU support

```bash
bash generateOverallPerformanceTraceAndTruth.sh
```

> To generate trace for Stress Test

```bash
/home/sosp/SPFresh/Release/usefultool -GenStress true --vectortype UInt8 --VectorPath /home/sosp/data/sift_data/base.1B.u8bin --filetype DEFAULT --UpdateSize 10000000 --BaseNum 1000000000 --TraceFileName bigann1b_update_trace -d 128 --Batch 20 -f DEFAULT
```

> To generate SPANN Index for Overall Peformance

build base SPANN Index

This will takes 1 days to generate SPANN Index of SPACEV100M with a 160 threads machine (we build it offline)

```bash
/home/sosp/SPFresh/Release/ssdserving build_SPANN_spacev100m.ini
mv iniFile/store_spacev100m/*.ini /home/sosp/data/store_spacev100m
```

> To generate DiskANN Index for Overall Performance

This will takes 0.5 days to generate DiskANN Index of SPACEV100M with a 160 threads machine (we build it offline)

```bash
mkdir /home/yuming/data/store_diskann_100m
/home/sosp/DiskANN_Baseline/build/tests/build_disk_index int8 /home/sosp/data/spacev_data/spacev100m_base.i8bin /home/yuming/data/store_diskann_100m/diskann_spacev_100m_ 64 75 128 128 16 l2 0
```

> To generate SPANN Index for Stress Test

This will takes 5 days to generate SPANN Index of SPACEV100M with a 160 threads machine (we build it offline)

```bash
/home/sosp/SPFresh/Release/ssdserving build_SPANN_sift1b.ini
mv iniFile/store_sift1b/indexloader_stress.ini /home/sosp/data/store_sift_cluster/indexloader.ini
```

> To generate data for figure 1,9,10
```bash
# using sift
python generate_dataset.py --src /home/sosp/data/sift_data/base.1B.u8bin --dst /home/sosp/data/sift_data/bigann2m_base.u8bin --topk 2000000
# this command require numpy and sklearn
python data_clustering_sift.py --src /home/sosp/data/sift_data/bigann2m_base.u8bin --dst /home/sosp/data/sift_data/bigann1m_update_clustering
mv /home/sosp/data/sift_data/bigann1m_update_clustering0 /home/sosp/data/sift_data/bigann1m_update_clustering
mv /home/sosp/data/sift_data/bigann1m_update_clustering1 /home/sosp/data/sift_data/bigann1m_update_clustering_trace0
mv /home/sosp/data/sift_data/bigann1m_update_clustering2 /home/sosp/data/sift_data/bigann2m_update_clustering
#generate truth
/home/sosp/SPFresh/Release/ssdserving genTruth_clustering.ini
mv /home/sosp/data/sift_data/bigann2m_update_clustering_origin_truth0 /home/sosp/data/sift_data/bigann2m_update_clustering_origin_truth

#build index
/home/sosp/SPFresh/Release/ssdserving build_clustering_1m.ini
mv iniFile/store_sift_cluster/*.ini /home/sosp/data/store_sift_cluster/
/home/sosp/SPFresh/Release/ssdserving build_clustering_2m.ini
mv iniFile/store_sift_cluster_2m/indexloader_clustering_2m.ini /home/sosp/data/store_sift_cluster/indexloader.ini

```

> To generate data for figure 11
```bash
python generate_dataset.py --src /home/sosp/data/sift_data/base.1B.u8bin --dst /home/sosp/data/sift_data/bigann1m_base.u8bin --topk 1000000
# build index
/home/sosp/SPFresh/Release/ssdserving build_sift1m.ini
mv iniFile/store_sift1m/indexloader_sift1m.ini /home/sosp/data/store_sift1m/indexloader.ini
```