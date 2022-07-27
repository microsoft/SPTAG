from sptag import SPTAG as SPTAG


class Sptag:

    def __init__(self, algo, metric):
        self._algo = str(algo)
        self._para = {}
        self._metric = {'angular': 'Cosine', 'euclidean': 'L2'}[metric]

    def fit(self, X, para=None):
        self._sptag = SPTAG.AnnIndex(self._algo, 'Float', X.shape[1])
        self._sptag.SetBuildParam("NumberOfThreads", '32', "Index")
        self._sptag.SetBuildParam("DistCalcMethod", self._metric, "Index")

        if para:
            self._para = para
            for k, v in para.items():
                self._sptag.SetBuildParam(k, str(v), "Index")

        self._sptag.Build(X, X.shape[0], False)

    def set_query_arguments(self, s_para=None):
        if s_para:
            self.s_para = s_para
            for k, v in s_para.items():
                self._sptag.SetSearchParam(k, str(v), "Index")

    def query(self, v, k):
        return self._sptag.Search(v, k)[0]

    def save(self, fn):
        self._sptag.Save(fn)

    def __str__(self):
        s = ''
        if self._para:
            s += ", " + ", ".join(
                [k + "=" + str(v) for k, v in self._para.items()])
        if self.s_para:
            s += ", " + ", ".join(
                [k+"_s" + "=" + str(v) for k, v in self.s_para.items()])
        return 'Sptag(metric=%s, algo=%s' % (
            self._metric, self._algo) + s + ')'
