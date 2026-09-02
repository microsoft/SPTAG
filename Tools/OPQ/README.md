# OPQ gpu training and inference tool

## Package Requirements (tbd)

1. Python>=3.7 
2. numpy>=1.18.1 
3. faiss>=1.7.0


## Parameter Sample
--data_file [input_path]\vectors.bin.0 --query_file [model_path]\query.bin --output_truth [output_path] --output_dir [output_path]\5474\cluster_unzip --task 0  --data_type float32 --k 5 --dim 1024 --B 1000000 --Q 1000 --D L2 --data_format DEFAULT --T 18 --train_samples 1000000 --quan_type opq --quan_dim 1024 --output_quantizer quantizer.bin --output_quan_vector_file dssm_vectors.bin --output_rec_vector_file vectors.bin --quan_test 1 --data_normalize 0 --query_normalize 0

## Example
python3 OPQ_gpu_train_infer.py --data_file perftest_vector.bin --query_file perftest_query.bin --task 0 --data_type float32 --k 5 --dim 64 --B 1000000 --Q 1000 --D L2 --data_format DEFAULT --T 20 --train_samples 1000000 --quan_type opq --quan_dim 32 --output_quantizer quantizer.bin

## RaBitQ storage-bit auto tuning

`--rabitq_auto_tune` runs before vector encoding or index construction. It evaluates
RaBitQ bit counts in ascending order and selects the first (therefore minimum)
count whose reranking Recall reaches `--rabitq_target_recall`. Every candidate
uses the centroid streamed over the complete base file, the same configured
`--Q` queries, and the full pre-generated ground-truth candidate pool. In INI mode,
`[SearchSSDIndex] ResultNum` supplies K: the candidate pool is reranked by
RaBitQ distance and its first K IDs are compared with the exact first K.

The command fails instead of silently choosing a bit count when the configured
query/ground-truth count is unavailable or no candidate in
`[--rabitq_min_bits, --rabitq_max_bits]` reaches the target. The selected storage
bit count, actual SPTAG byte width (including padded dimensions and five Float
factors), measured recalls, and sample counts are written atomically to
`<output_dir>/rabitq_auto_tuning.json`. Use `native_quantizer_qd` as `-qd` for the
native SPTAG `quantizer` command that generates the quantizer and vectors consumed
by the subsequent index build. The tuner deliberately does not emit Faiss codes:
their persisted layout is not the SPTAG global RaBitQ format.

The same parameters can be supplied exclusively through the
`[RaBitQAutoTune]` section of an INI:

```bash
python3 Tools/OPQ/OPQ_gpu_train_infer.py \
  --config Script_AE/iniFile/rabitq_auto_tune_sift1m.ini
```

In INI mode no additional CLI parameters are accepted. `QueryCount` defaults to
`[SearchSSDIndex] QueryCountLimit`, keeping the configured query count in one
place. Faiss/OpenMP reuses `[BuildSSDIndex] NumberOfThreads`. Global RaBitQ
tuning requires `[Base] ValueType=Float` and `DistCalcMethod=L2`; there are no
separate data/query normalization settings.