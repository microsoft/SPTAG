# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from model import metrics
import time
import os
import psutil


def run_individual_query(algo,
                         X_train,
                         X_test,
                         distance,
                         k,
                         run_count=1,
                         max_mem=-1):

    best_search_time = float('inf')
    for i in range(run_count):
        print('Run %d/%d...' % (i + 1, run_count))
        # a bit dumb but can't be a scalar since of Python's scoping rules
        n_items_processed = [0]

        def single_query(v):
            start = time.time()
            candidates = algo.query(v, k)
            if max_mem > 0 and psutil.Process(
                    os.getpid()).memory_info().rss > max_mem:
                raise MemoryError
            total = (time.time() - start)

            candidates = [
                (int(idx),
                 float(metrics[distance]['distance'](v, X_train[idx])))  # noqa
                for idx in candidates
            ]
            n_items_processed[0] += 1
            if n_items_processed[0] % 1000 == 0:
                print('Processed %d/%d queries...' %
                      (n_items_processed[0], len(X_test)))
            if len(candidates) > k:
                print('warning: algorithm returned %d results, but k'
                      ' is only %d)' % (len(candidates), k))
            return (total, candidates)

        results = [single_query(x) for x in X_test]

        total_time = sum(t for t, _ in results)
        total_candidates = sum(len(candidates) for _, candidates in results)
        search_time = total_time / len(X_test)
        avg_candidates = total_candidates / len(X_test)
        best_search_time = min(best_search_time, search_time)

    attrs = {
        "best_search_time": best_search_time,
        "candidates": avg_candidates,
        "name": 'BKT',
        "run_count": run_count,
        "distance": distance,
        "count": int(k)
    }

    return (attrs, results)
