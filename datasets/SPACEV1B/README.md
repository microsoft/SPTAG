# SPACEV1B: A billion-Scale vector dataset for text descriptors
This is a dataset released by Microsoft Bing for large scale vector search related research usage. It consists of more than one billion document vectors 
and 29K+ query vectors encoded by Microsoft SpaceV model. This model is trained to capture generic intent representation for both documents and queries. 
The goal is to match the query vector to the closest document vectors in order to achieve topk relevant documents for each query.

## Introduction

This dataset contains:

 * [vectors.bin](vectors.bin): It contains 1,402,020,720 100-dimensional int8-type document descriptors.
 * [query.bin](query.bin):  It contains 29,316 100-dimensional int8-type query descriptors.
 * [truth.bin](truth.bin): It contains 100 nearest ground truth（include vector ids and distances) of 29,316 queries according to L2 distance.


## How to read the vectors, queries, and truth

```python
import struct
import numpy as np

fvec = open('SPACEV1B.bin', 'rb')
vec_count = struct.unpack('i', fvec.read(4))[0]
vec_dimension = struct.unpack('i', fvec.read(4))[0]
X = np.frombuffer(fvec.read(vec_count * vec_dimension), dtype=np.int8).reshape((vec_count, vec_dimension))

fq = open('query.bin', 'rb')
q_count = struct.unpack('i', fq.read(4))[0]
q_dimension = struct.unpack('i', fq.read(4))[0]
queries = np.frombuffer(fq.read(q_count * q_dimension), dtype=np.int8).reshape((q_count, q_dimension))

ftruth = open('truth.bin', 'rb')
t_count = struct.unpack('i', ftruth.read(4))[0]
topk = struct.unpack('i', ftruth.read(4))[0]
truth_vids = np.frombuffer(ftruth.read(t_count * topk * 4), dtype=np.int32).reshape((t_count, topk))
truth_distances = np.frombuffer(ftruth.read(t_count * topk * 4), dtype=np.float).reshape((t_count, topk))
```

## License

The entire dataset is under [O-UDA license](LICENSE)