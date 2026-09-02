# OPQ gpu training and inference tool

## Package Requirements (tbd)

1. Python>=3.7 
2. numpy>=1.18.1 
3. faiss>=1.7.0


## Parameter Sample
--data_file [input_path]\vectors.bin.0 --query_file [model_path]\query.bin --output_truth [output_path] --output_dir [output_path]\5474\cluster_unzip --task 0  --data_type float32 --k 5 --dim 1024 --B 1000000 --Q 1000 --D L2 --data_format DEFAULT --T 18 --train_samples 1000000 --quan_type opq --quan_dim 1024 --output_quantizer quantizer.bin --output_quan_vector_file dssm_vectors.bin --output_rec_vector_file vectors.bin --quan_test 1 --data_normalize 0 --query_normalize 0

## Example
python3 OPQ_gpu_train_infer.py --data_file perftest_vector.bin --query_file perftest_query.bin --task 0 --data_type float32 --k 5 --dim 64 --B 1000000 --Q 1000 --D L2 --data_format DEFAULT --T 20 --train_samples 1000000 --quan_type opq --quan_dim 32 --output_quantizer quantizer.bin
python3 OPQ_gpu_train_infer.py --data_file laion_5M.bin --query_file laion_test.bin --output_truth truth.txt --output_dir rabitq_tuned --data_type float32 --target_type float32 --k 100 --dim 768 --B 1000000 --Q 10000 --D L2 --train_samples 1000000 --quan_type rabitq --quan_test 1 --rabitq_auto_tune --rabitq_target_recall 0.95 --rabitq_min_bits 1 --rabitq_max_bits 8
python3 OPQ_gpu_train_infer.py --data_file openai_5M.bin --query_file openai_test.bin --output_truth truth.txt --output_dir rabitq_tuned --data_type float32 --target_type float32 --k 100 --dim 1536 --B 1000000 --Q 10000 --D Cosine --train_samples 1000000 --quan_type rabitq --quan_test 1 --rabitq_auto_tune --rabitq_target_recall 0.95 --data_normalize 1 --query_normalize 1

## RaBitQ storage-bit auto tuning

`--rabitq_auto_tune` runs before vector encoding or index construction. It evaluates
RaBitQ bit counts in ascending order and selects the first (therefore minimum)
count whose exhaustive `Recall@k` reaches `--rabitq_target_recall`. Every candidate
uses the same first `--train_samples` vectors, the same configured `--Q` queries,
and the pre-generated `--output_truth` top-`k` neighbors.

The command fails instead of silently choosing a bit count when the configured
query/ground-truth count is unavailable or no candidate in
`[--rabitq_min_bits, --rabitq_max_bits]` reaches the target. The selected storage
bit count, actual SPTAG byte width (including padded dimensions and five Float
factors), measured recalls, and sample counts are written atomically to
`<output_dir>/rabitq_auto_tuning.json`. Use `native_quantizer_qd` as `-qd` for the
native SPTAG `quantizer` command that generates the quantizer and vectors consumed
by the subsequent index build. The tuner deliberately does not emit Faiss codes:
their persisted layout is not the SPTAG global RaBitQ format.