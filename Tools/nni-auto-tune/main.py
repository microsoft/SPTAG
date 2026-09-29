# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import nni
import h5py
import time

import numpy as np
import os
from model import Sptag, BruteForceBLAS
from runner import run_individual_query
from dataset import DataReader, HDF5Reader
import argparse
import json
import shutil
import itertools
from multiprocess import Pool, Process
import multiprocess


def knn_threshold(data, k, epsilon):
    return data[k - 1] + epsilon


def get_recall_from_distance(dataset_distances,
                             run_distances,
                             k,
                             epsilon=1e-3):
    recalls = np.zeros(len(run_distances))
    for i in range(len(run_distances)):
        t = knn_threshold(dataset_distances[i], k, epsilon)
        actual = 0
        for d in run_distances[i][:k]:
            if d <= t:
                actual += 1
        recalls[i] = actual

    return (np.mean(recalls) / float(k), np.std(recalls) / float(k), recalls)


def get_recall_from_index(dataset_index, run_index, k):
    recalls = np.zeros(len(run_index))
    for i in range(len(run_index)):
        actual = 0
        for d in run_index[i][:k]:
            # need to conver to string because default loaded label are strings
            if str(d) in dataset_index[i][:k]:
                actual += 1
        recalls[i] = actual

    return (np.mean(recalls) / float(k), np.std(recalls) / float(k), recalls)


def queries_per_second(attrs):
    return 1.0 / attrs["best_search_time"]


def compute_metrics(groundtruth, attrs, results, k, from_index=False):
    if from_index:
        mean, std, recalls = get_recall_from_index(groundtruth, results, k)
    else:
        mean, std, recalls = get_recall_from_distance(groundtruth, results, k)
    qps = queries_per_second(attrs)
    print('mean: %12.3f,std: %12.3f, qps: %12.3f' % (mean, std, qps))
    return mean, qps


def grid_search(params):
    param_num = len(params)
    params = list(params.items())
    max_param_choices = max([len(p[1]) for p in params])
    temp = []
    for i in range(max_param_choices):
        temp += [i for _ in range(param_num)]
    for c in set(itertools.permutations(temp, param_num)):
        res = {}
        for i in range(len(c)):
            if c[i] >= len(params[i][1]):
                break
            else:
                res[params[i][0]] = params[i][1][c[i]]
        if len(res) == param_num:
            yield res


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
        '--label_file',
        help=
        'the data file to load groundtruth index from, only support text file',
        default=None)
    parser.add_argument('--algorithm',
                        help='the name of SPTAG algorithm',
                        default="BKT")
    parser.add_argument("--k",
                        default=10,
                        type=int,
                        help="the number of near neighbours to search for")
    parser.add_argument("--distance",
                        default='angular',
                        help="the type of distance for searching")
    parser.add_argument(
        "--max_build_time",
        default=-1,
        type=int,
        help="the limit of index build time in seconds. -1 means no limit")
    parser.add_argument(
        "--max_memory",
        default=-1,
        type=int,
        help=
        "the limit of memory use during searching in bytes. -1 means no limit")
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

    if args.train_file.endswith(".hdf5"):
        # ann-benchmark format hdf5 file got all we want, so args like distance are ignored
        data_reader = HDF5Reader(args.train_file, args.data_type)
        X_train, X_test = data_reader.readallbatches()
        distance = data_reader.distance
        dimension = data_reader.featuredim
        label = data_reader.label
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
        label = []
        if args.label_file is None:
            # if the groundtruth is not provided
            # we calculate groundtruth distances with brute force
            bf = BruteForceBLAS(distance)
            bf.fit(X_train)
            for i, x in enumerate(X_test):
                if i % 1000 == 0:
                    print('%d/%d...' % (i, len(X_test)))
                res = list(bf.query_with_distances(x, args.k))
                res.sort(key=lambda t: t[-1])
                label.append([d for _, d in res])
        else:
            label = []
            # we assume the groundtruth index are split by space
            with open(args.label_file, 'r') as f:
                for line in f:
                    label.append(line.strip().split())

    print('got a train set of size (%d * %d)' % (X_train.shape[0], dimension))
    print('got %d queries' % len(X_test))

    para = nni.get_next_parameter()
    algo = Sptag(args.algorithm, distance)

    t0 = time.time()
    if args.max_build_time > 0:
        pool = Pool(1)
        results = pool.apply_async(algo.fit,
                                   kwds=dict(X=X_train,
                                             para=para,
                                             data_type=args.data_type,
                                             save_index=True))
        try:
            results.get(args.max_build_time
                        )  # Wait timeout seconds for func to complete.
            algo.load('index')
            shutil.rmtree("index")
            pool.close()
            pool.join()
        except multiprocess.TimeoutError:  # kill subprocess if timeout
            print("Aborting due to timeout", args.max_build_time)
            pool.terminate()
            nni.report_final_result({
                'default': -1,
                "recall": 0,
                "qps": 0,
                "build_time": args.max_build_time
            })
            return
    else:
        algo.fit(X=X_train, para=para, data_type=args.data_type)

    build_time = time.time() - t0

    print('Built index in', build_time)

    search_param_choices = {
        "NumberOfInitialDynamicPivots": [1, 2, 4, 8, 16, 32, 50],
        "MaxCheck": [512, 3200, 5120, 8192, 12800, 16400, 19600],
        "NumberOfOtherDynamicPivots": [1, 2, 4, 8, 10]
    }

    best_metric = -1
    best_res = {}
    for i, search_params in enumerate(grid_search(search_param_choices)):
        algo.set_query_arguments(search_params)
        try:
            attrs, results = run_individual_query(algo,
                                                  X_train,
                                                  X_test,
                                                  distance,
                                                  args.k,
                                                  max_mem=args.max_memory)
        except MemoryError:
            print("Aborting due to exceed memory limit")
            nni.report_final_result({
                'default': -1,
                "recall": 0,
                "qps": 0,
                "build_time": args.max_build_time
            })
            return

        neighbors = [0 for _ in results]
        distances = [0 for _ in results]

        for idx, (t, ds) in enumerate(results):
            neighbors[idx] = [n for n, d in ds] + [-1] * (args.k - len(ds))
            distances[idx] = [d for n, d in ds
                              ] + [float('inf')] * (args.k - len(ds))
        if args.label_file is None:
            recalls_mean, qps = compute_metrics(label, attrs, distances,
                                                args.k)
        else:
            recalls_mean, qps = compute_metrics(label,
                                                attrs,
                                                neighbors,
                                                args.k,
                                                from_index=True)

        combined_metric = -1 * np.log10(1 - recalls_mean) + 0.1 * np.log10(qps)

        res = {
            "default": combined_metric,
            "recall": recalls_mean,
            "qps": qps,
            "build_time": build_time
        }

        if combined_metric > best_metric:
            best_metric = combined_metric
            best_res = res.copy()

        res["build_params"] = para
        res["search_params"] = search_params

        experiment_id = nni.get_experiment_id()
        result_dir = os.path.join('results', args.train_file.split('.')[0])
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        trial_id = nni.get_trial_id()
        with open(
                os.path.join(
                    result_dir,
                    "result_" + str(trial_id) + ' ' + str(i) + ".json"),
                "w") as f:
            json.dump(res, f)

    nni.report_final_result(best_res)


if __name__ == '__main__':
    main()
