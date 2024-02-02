# OPQ gpu training and inference tool

## Package Requirements (tbd)

1. Python>=3.7 
2. numpy>=1.18.1 
3. faiss>=1.7.0
4. LibVQ


## Parameter Sample
--data_file [input_path]\vectors.bin.0 --query_file [model_path]\query.bin --output_truth [output_path] --output_dir [output_path]\5474\cluster_unzip --task 0  --data_type float32 --k 5 --dim 1024 --B 1000000 --Q 1000 --D L2 --data_format DEFAULT --T 18 --train_samples 1000000 --quan_type opq --quan_dim 1024 --output_quantizer quantizer.bin --output_quan_vector_file dssm_vectors.bin --output_rec_vector_file vectors.bin --quan_test 1 --data_normalize 0 --query_normalize 0
