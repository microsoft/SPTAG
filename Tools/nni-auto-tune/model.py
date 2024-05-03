# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from sptag import SPTAG as SPTAG
from scipy.spatial.distance import pdist as scipy_pdist
import numpy
""" We use the implementation of metric and brute force algorithm from ann-banchmark"""


def pdist(a, b, metric):
    return scipy_pdist([a, b], metric=metric)[0]


def jaccard(a, b):
    if len(a) == 0 or len(b) == 0:
        return 0
    intersect = len(set(a) & set(b))
    return intersect / (float)(len(a) + len(b) - intersect)


metrics = {
    'hamming': {
        'distance': lambda a, b: pdist(a, b, "hamming"),
        'distance_valid': lambda a: True
    },
    # return 1 - jaccard similarity, because smaller distances are better.
    # modified to use pdist jaccard given new data format (boolean ndarray)
    'jaccard': {
        'distance': lambda a, b: 1 - jaccard(a, b),  #pdist(a, b, "jaccard"), 
        'distance_valid': lambda a: a < 1 - 1e-5
    },
    'euclidean': {
        'distance': lambda a, b: pdist(a, b, "euclidean"),
        'distance_valid': lambda a: True
    },
    'angular': {
        'distance': lambda a, b: pdist(a, b, "cosine"),
        'distance_valid': lambda a: True
    }
}


class Sptag:

    def __init__(self, algo, metric):
        self._algo = str(algo)
        self._para = {}
        self._metric = {'angular': 'Cosine', 'euclidean': 'L2'}[metric]

    def fit(self, X, para=None, data_type='float32', save_index=False):
        self._data_type = {
            'float32': 'Float',
            'int8': 'Int8',
            'int16': 'Int16'
        }[data_type]
        self._sptag = SPTAG.AnnIndex(self._algo, self._data_type, X.shape[1])
        self._sptag.SetBuildParam("NumberOfThreads", '32', "Index")
        self._sptag.SetBuildParam("DistCalcMethod", self._metric, "Index")

        if para:
            self._para = para
            for k, v in para.items():
                self._sptag.SetBuildParam(k, str(v), "Index")

        self._sptag.Build(X, X.shape[0], False)
        # temp save for auto tune, please use self.save() for intended saving
        if save_index:
            self.save("index")

    def set_query_arguments(self, s_para=None):
        if s_para:
            self.s_para = s_para
            for k, v in s_para.items():
                self._sptag.SetSearchParam(k, str(v), "Index")

    def query(self, v, k):
        return self._sptag.Search(v, k)[0]

    def save(self, fn):
        self._sptag.Save(fn)

    def load(self, fn):
        self._sptag = SPTAG.AnnIndex.Load(fn)

    def __str__(self):
        s = ''
        if self._para:
            s += ", " + ", ".join(
                [k + "=" + str(v) for k, v in self._para.items()])
        if self.s_para:
            s += ", " + ", ".join(
                [k + "_s" + "=" + str(v) for k, v in self.s_para.items()])
        return 'Sptag(metric=%s, algo=%s' % (self._metric,
                                             self._algo) + s + ')'


class BruteForceBLAS:
    """kNN search that uses a linear scan = brute force."""

    def __init__(self, metric, precision=numpy.float32):
        if metric not in ('angular', 'euclidean', 'hamming', 'jaccard'):
            raise NotImplementedError(
                "BruteForceBLAS doesn't support metric %s" % metric)
        elif metric == 'hamming' and precision != numpy.bool:
            raise NotImplementedError(
                "BruteForceBLAS doesn't support precision"
                " %s with Hamming distances" % precision)
        self._metric = metric
        self._precision = precision
        self.name = 'BruteForceBLAS()'

    def fit(self, X):
        """Initialize the search index."""
        X = X.astype(self._precision)
        if self._metric == 'angular':
            # precompute (squared) length of each vector
            lens = (X**2).sum(-1)
            # normalize index vectors to unit length
            X /= numpy.sqrt(lens)[..., numpy.newaxis]
            self.index = numpy.ascontiguousarray(X, dtype=self._precision)
        elif self._metric == 'hamming':
            # Regarding bitvectors as vectors in l_2 is faster for blas
            X = X.astype(numpy.float32)
            # precompute (squared) length of each vector
            lens = (X**2).sum(-1)
            self.index = numpy.ascontiguousarray(X, dtype=numpy.float32)
            self.lengths = numpy.ascontiguousarray(lens, dtype=numpy.float32)
        elif self._metric == 'euclidean':
            # precompute (squared) length of each vector
            self.index = numpy.ascontiguousarray(X, dtype=self._precision)
            lens = (self.index**2).sum(-1)
            self.lengths = numpy.ascontiguousarray(lens, dtype=self._precision)
        elif self._metric == 'jaccard':
            self.index = X
        else:
            # shouldn't get past the constructor!
            assert False, "invalid metric"

    def query(self, v, n):
        return [index for index, _ in self.query_with_distances(v, n)]

    def query_with_distances(self, v, n):
        """Find indices of `n` most similar vectors from the index to query
        vector `v`."""
        v = v.astype(self._precision)
        if self._metric != 'jaccard':
            # use same precision for query as for index
            v = numpy.ascontiguousarray(v, dtype=self.index.dtype)

        # HACK we ignore query length as that's a constant
        # not affecting the final ordering
        if self._metric == 'angular':
            # argmax_a cossim(a, b) = argmax_a dot(a, b) / |a||b| = argmin_a -dot(a, b)  # noqa
            dists = -numpy.dot(self.index, v)
        elif self._metric == 'euclidean':
            # argmin_a (a - b)^2 = argmin_a a^2 - 2ab + b^2 = argmin_a a^2 - 2ab  # noqa
            dists = self.lengths - 2 * numpy.dot(self.index, v)
        elif self._metric == 'hamming':
            # Just compute hamming distance using euclidean distance
            dists = self.lengths - 2 * numpy.dot(self.index, v)
        elif self._metric == 'jaccard':
            dists = [
                metrics[self._metric]['distance'](v, e) for e in self.index
            ]
        else:
            # shouldn't get past the constructor!
            assert False, "invalid metric"
        # partition-sort by distance, get `n` closest
        nearest_indices = numpy.argpartition(dists, n)[:n]
        indices = [
            idx for idx in nearest_indices
            if metrics[self._metric]["distance_valid"](dists[idx])
        ]

        def fix(index):
            ep = self.index[index]
            ev = v
            return (index, metrics[self._metric]['distance'](ep, ev))

        return map(fix, indices)
