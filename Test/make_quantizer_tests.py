"""
Owner: isst
Please implement your own ModelImp
"""

import numpy as np
from struct import pack, unpack, calcsize
import math
import heapq
import argparse
import copy
import time
import random
import os

DIM = 10
KS = 256
BATCHSIZE = 100
M = 5

def distance(data, query, D = 'Cosine'):
    if D == 'Cosine':
        return (1.0 - np.dot(query, data.T))
    else:
        return np.linalg.norm(query-data)**2


def WriteBin(out_file, xb):
    with open(out_file, 'wb') as fl:
        fl.write(pack('i', xb.shape[0]))
        fl.write(pack('i', xb.shape[1]))
        xb.tofile(fl)

def WriteText(out_file, data, metadata):
    with open(out_file, 'w') as fl:
        for i in range(data.shape[0]):
            fl.write(metadata[i])
            fl.write('\t')
            fl.write("|".join([str(elem) for elem in data[i]]))
            fl.write('\n')

def WriteCodebook(out_file, xcb):
    with open(out_file, 'wb') as fl:
        fl.write(pack('i', len(xcb)))
        fl.write(pack('i', len(xcb[0])))
        fl.write(pack('i', len(xcb[0][0])))
        for codebook in xcb:
            for entry in codebook:
                for value in entry:
                    fl.write(pack('f', value))

def main():
    xcb = []
    for i in range(M):
        cb = []
        for j in range(KS):
            vec = []
            for k in range(int(DIM/M)):
                vec.append(random.random())
            cb.append(vec)
        xcb.append(cb)
    xcb = np.array(xcb)
    print(xcb.shape)

    vecs = []
    for i in range(BATCHSIZE):
        vec = []
        for j in range(M):
            vec.append(random.randrange(KS))
        vecs.append(vec)
    vecs = np.array(vecs, dtype=np.uint8)
    print(vecs.shape)

    binout = "res" + os.path.sep + "testvectors-quantized.bin"
    txtout = "res" + os.path.sep + "testvectors-quantized.txt"
    cbout = "res" + os.path.sep + "test-quantizer.bin"
    print(binout)
    metadata = [str(i) for i in range(BATCHSIZE)]
    WriteBin(binout, vecs)
    WriteText(txtout, vecs, metadata)
    WriteCodebook(cbout, xcb)

    with open("res" + os.path.sep + "vector-distances-quantized.txt", 'w') as outfl:
        base_vec = vecs[0]
        for i in range(BATCHSIZE):
            vec = vecs[i]
            l2_dist = 0
            cosine_dist = 0
            for j in range(M):
                bcomp = np.array(xcb[j][base_vec[j]])
                qcomp = np.array(xcb[j][vec[j]])
                l2_dist += distance(bcomp, qcomp, D = 'L2')
                cosine_dist += distance(bcomp, qcomp, D = 'Cosine')
            outfl.write(str(l2_dist) + '\n')
            outfl.write(str(cosine_dist) + '\n')
            print(cosine_dist)

main()
                
