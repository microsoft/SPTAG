from scipy.spatial.distance import pdist as scipy_pdist
import time

metric_dic = {'angular': 'cosine', 'euclidean': 'euclidean'}


def run_individual_query(algo, X_train, X_test, distance, k, run_count):

    best_search_time = float('inf')
    for i in range(run_count):
        print('Run %d/%d...' % (i + 1, run_count))
        # a bit dumb but can't be a scalar since of Python's scoping rules
        n_items_processed = [0]

        def single_query(v):
            start = time.time()
            candidates = algo.query(v, k)
            total = (time.time() - start)

            candidates = [
                (int(idx),
                 float(
                     scipy_pdist([v, X_train[idx]],
                                 metric=metric_dic[distance])))  # noqa
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
        "batch_mode": False,
        "best_search_time": best_search_time,
        "candidates": avg_candidates,
        "expect_extra": False,
        "name": 'BKT',
        "run_count": run_count,
        "distance": distance,
        "count": int(k)
    }

    return (attrs, results)
