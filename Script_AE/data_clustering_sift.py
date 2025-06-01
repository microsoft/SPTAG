import numpy as np
import argparse
import struct
from sklearn.cluster import KMeans
import sklearn.metrics


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="The input file (.i8bin)")
    parser.add_argument("--dst", help="The output file prefix (.i8bin)")
    return parser.parse_args()


if __name__ == "__main__":
    clusters = 2
    args = process_args()

    # Read topk vector one by one
    vecs = []
    row_bin = "";
    dim_bin = ""; 
    with open(args.src, "rb") as f:

        row_bin = f.read(4)
        assert row_bin != b''
        row, = struct.unpack('i', row_bin)

        dim_bin = f.read(4)
        assert dim_bin != b''
        dim, = struct.unpack('i', dim_bin)

        i = 0
        while 1:

            # The next dim byte is for a vector for spacev
            vec = struct.unpack('b' * dim, f.read(dim))
            
            # Store it
            vecs.append(vec)
            i += 1
            if i == row:
                break
    
    # clustering vectors
            
    vecs = np.array(vecs, dtype=np.int8)
    assert vecs.shape[0] == row
    print("vecs.shape:", vecs.shape)
    estimator = KMeans(n_clusters=clusters)
    estimator.fit(vecs)
    label_pred = estimator.labels_ 

    print("cluster finished")
    #print(sklearn.metrics.silhouette_score(vecs, label_pred, metric='euclidean'))

    # generate result
    vec_list = []
    vec_num_list = []
    for i in range(0, clusters):
        vec_num_list.append(0)

    for i in range(0, clusters):
        with open(args.src, "rb") as f:

            row_bin = f.read(4)
            assert row_bin != b''
            row, = struct.unpack('i', row_bin)

            dim_bin = f.read(4)
            assert dim_bin != b''
            dim, = struct.unpack('i', dim_bin)

            j = 0

            vecs = ""

            while 1:

                
                # The next dim byte is for a vector for spacev

                if label_pred[j] == i:
                    vecs += f.read(dim)
                    vec_num_list[label_pred[j]] += 1
                else:
                    f.read(dim)

                j += 1

                if j % 100000 == 0:
                    print(j)

                if j == row:
                    break
            vec_list.append(vecs)

    print("cluster_result: ", vec_num_list)

    for i in range(0, clusters):
        with open(args.dst + str(i), "wb") as f:
            f.write(struct.pack('i', vec_num_list[i]))
            f.write(dim_bin)
            f.write(vec_list[i])

    with open(args.dst + str(clusters), "wb") as f:
        f.write(struct.pack('i', row))
        f.write(dim_bin)
        for i in range(0, clusers):
            f.write(vec_list[i])
    

