# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import h5py
import time

import numpy as np
import os
from model import BruteForceBLAS
from dataset import DataReader, HDF5Reader
import argparse
import json


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--train_file',
        help='the data file to load training points from, '
        'could be text file, binary flie or ann-benchmark format hdf5 file',
        default='glove-100-angular.hdf5')
    parser.add_argument(
        '--query_file',
        help='the data file to load query points from, if you use '
        'ann-benchmark format hdf5 file in train_file, this should be None',
        default=None)
    parser.add_argument(
        '--output_dir',
        help=
        'the dir to save sampled train set and pre-calculated ground truth',
        default='./sampled')
    parser.add_argument(
        "--num_sample",
        default=1000000,
        type=int,
        help="the size of sampled train set. -1 means don't do sample")
    parser.add_argument("--k",
                        default=32,
                        type=int,
                        help="the number of near neighbours to search for")
    parser.add_argument("--distance",
                        default='angular',
                        help="the type of distance for searching")
    parser.add_argument("--dim",
                        default=100,
                        type=int,
                        help="the dimention of training vectors")
    parser.add_argument("--input_type",
                        default="float32",
                        help="the data type of input vectors")
    parser.add_argument("--data_type",
                        default="float32",
                        help="the data type for building and search in SPTAG ")
    args = parser.parse_args()

    def tostring(a):
        return str(a)

    if args.train_file.endswith(".hdf5"):
        # ann-benchmark format hdf5 file got all we want, so args like distance are ignored
        data_reader = HDF5Reader(args.train_file, args.data_type)
        X_train, X_test = data_reader.readallbatches()
        distance = data_reader.distance
        dimension = data_reader.featuredim

    else:
        X_train = DataReader(args.train_file,
                             args.dim,
                             batchsize=-1,
                             datatype=args.input_type,
                             targettype=args.data_type).readbatch()[1]
        X_test = DataReader(args.query_file,
                            args.dim,
                            batchsize=-1,
                            datatype=args.input_type,
                            targettype=args.data_type).readbatch()[1]
        distance = args.distance
        dimension = args.dim

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # -1 means don't do sample
    if args.num_sample != -1:
        np.random.shuffle(X_train)
        X_train = X_train[:args.num_sample]

        print('sampled train set to size (%d)' % (X_train.shape[0]))
        prefix = args.train_file.replace(".bin", "")
        target_path = os.path.join(
            args.output_dir,
            os.path.splitext(os.path.split(prefix)[1])[0] + '-sampled-' +
            str(args.num_sample) + '.txt')
        with open(target_path, 'w') as f:
            for i in range(len(X_train)):
                f.write('\t')
                f.write('|'.join(list(map(tostring, X_train[i]))))
                f.write('\n')

        print('write sampled train set to', target_path, "complete")

    label_index = []

    # we calculate groundtruth index with brute force
    bf = BruteForceBLAS(distance)
    bf.fit(X_train)
    for i, x in enumerate(X_test):
        if i % 1000 == 0:
            print('%d/%d...' % (i, len(X_test)))
        res = list(bf.query_with_distances(x, args.k))
        res.sort(key=lambda t: t[-1])
        label_index.append([n for n, _ in res])

    print('generated a ground truth index set of size (%d)' %
          (len(label_index)))

    with open(os.path.join(args.output_dir, 'ground_truth.txt'), 'w') as f:
        for i in range(len(label_index)):
            f.write(' '.join(list(map(tostring, label_index[i]))))
            f.write('\n')

    print('write ground truth to',
          os.path.join(args.output_dir, 'ground_truth.txt'), "complete")


if __name__ == '__main__':
    main()
