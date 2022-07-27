# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import nni
import h5py
import time

import numpy as np
import os
from model import Sptag
from runner import run_individual_query
from dataset import DATASETS, dataset_transform
import argparse
import json
import itertools


def knn_threshold(data, k, epsilon):
    return data[k - 1] + epsilon


def get_recall_values(dataset_distances, run_distances, k, epsilon=1e-3):
    recalls = np.zeros(len(run_distances))
    for i in range(len(run_distances)):
        t = knn_threshold(dataset_distances[i], k, epsilon)
        actual = 0
        for d in run_distances[i][:k]:
            if d <= t:
                actual += 1
        recalls[i] = actual

    return (np.mean(recalls) / float(k), np.std(recalls) / float(k), recalls)

def sigmoid(x):
    return 1/(1+(np.exp((-x))))

def queries_per_second(attrs):
    return 1.0 / attrs["best_search_time"]


def compute_metrics(true_nn_distances, attrs, run_distances, k):
    mean, std, recalls = get_recall_values(true_nn_distances, run_distances, k)
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
    for c in set(itertools.permutations(temp,param_num)):
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
    parser.add_argument('--dataset',
                        metavar='NAME',
                        help='the dataset to load training points from',
                        default='glove-100-angular')
    parser.add_argument('--algorithm',
                        metavar='NAME',
                        help='run only the named algorithm',
                        default="BKT")
    parser.add_argument("--k",
                        default=10,
                        help="the number of near neighbours to search for")
    args = parser.parse_args()

    D, dimension = DATASETS[args.dataset]()
    X_train = np.array(D['train'])
    X_test = np.array(D['test'])
    distance = D.attrs['distance']
    print('got a train set of size (%d * %d)' % (X_train.shape[0], dimension))
    print('got %d queries' % len(X_test))

    X_train, X_test = dataset_transform(D)

    t0 = time.time()

    para = nni.get_next_parameter()
    algo = Sptag(args.algorithm, distance)
    algo.fit(X_train,para)
    build_time = time.time() - t0

    print('Built index in', build_time)

    search_param_choices = {"NumberOfInitialDynamicPivots":[1,2,4,8,16,32,50],
                            "MaxCheck":[512,640,896,1408,2432,4408,8192],
                            "NumberOfOtherDynamicPivots":[1,2,4,8,10]}

    best_metric = -1
    best_res = {}
    for i, search_params in enumerate(grid_search(search_param_choices)):
        algo.set_query_arguments(search_params)
        attrs, results = run_individual_query(algo, X_train, X_test, distance,
                                            args.k, 1)

        neighbors = [0 for _ in results]
        distances = [0 for _ in results]

        for idx, (t, ds) in enumerate(results):
            neighbors[idx] = [n for n, d in ds] + [-1] * (args.k - len(ds))
            distances[idx] = [d for n, d in ds] + [float('inf')] * (args.k - len(ds))

        recalls_mean, qps = compute_metrics(np.array(D["distances"]), attrs,
                                            distances, args.k)

        combined_metric = recalls_mean * np.log(qps)

        res = {"default": combined_metric, "recall": recalls_mean, "qps": qps, "build_time": build_time}

        if combined_metric > best_metric:
            best_metric = combined_metric
            best_res = res.copy()

        res["build_params"] = para
        res["search_params"] = search_params
        
        experiment_id = nni.get_experiment_id()
        result_dir = os.path.join('results',args.dataset)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        trial_id = nni.get_trial_id()
        with open(os.path.join(result_dir,"result_"+ str(trial_id)+ ' ' + str(i) +".json"),"w") as f:
            json.dump(res,f)

    nni.report_final_result(best_res)

if __name__ == '__main__':
    main()
