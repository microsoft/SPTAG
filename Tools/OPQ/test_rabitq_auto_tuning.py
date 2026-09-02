import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


MODULE_PATH = Path(__file__).with_name('OPQ_gpu_train_infer.py')
SPEC = importlib.util.spec_from_file_location('opq_gpu_train_infer', MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeIndex:
    def __init__(self, results):
        self.results = np.asarray(results, dtype=np.int64)

    def search(self, queries, topk):
        count = len(queries)
        results = self.results[:count, :topk]
        self.results = self.results[count:]
        return np.zeros(results.shape, dtype=np.float32), results


class RaBitQAutoTuningTest(unittest.TestCase):
    def test_data_reader_accepts_fbin(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'vectors.fbin'
            with path.open('wb') as output:
                np.asarray([2, 3], dtype=np.int32).tofile(output)
                np.arange(6, dtype=np.float32).tofile(output)
            reader = MODULE.DataReader(str(path), 3, 2, 0, 'float32')
            count, vectors = reader.readbatch()
            reader.close()
            self.assertEqual(2, count)
            np.testing.assert_array_equal(vectors, np.arange(6, dtype=np.float32).reshape(2, 3))

    def test_binary_reader_accepts_numbered_shards(self):
        self.assertTrue(MODULE.is_binary_vector_file('vectors.bin.0'))
        self.assertTrue(MODULE.is_binary_vector_file('vectors.fbin.12'))
        self.assertFalse(MODULE.is_binary_vector_file('vectors.txt.0'))

    def test_sptag_storage_bytes_include_padding_and_factors(self):
        self.assertEqual(68, MODULE.sptag_rabitq_storage_bytes(128, 3))
        self.assertEqual(68, MODULE.sptag_rabitq_storage_bytes(127, 3))
        self.assertEqual(92, MODULE.sptag_rabitq_storage_bytes(129, 3))

    def test_rejects_invalid_storage_bits(self):
        with self.assertRaises(ValueError):
            MODULE.sptag_rabitq_storage_bytes(128, 0)

    def test_load_ground_truth_honors_query_count_and_topk(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'truth.txt'
            path.write_text('1 2 3\n4 5 6\n', encoding='ascii')
            self.assertEqual([{1, 2}, {4, 5}], MODULE.load_ground_truth(path, 2, 2))
            with self.assertRaises(ValueError):
                MODULE.load_ground_truth(path, 3, 2)

    def test_recall_is_query_equal_weighted(self):
        index = FakeIndex([[1, 9], [4, 5]])
        queries = np.zeros((2, 4), dtype=np.float32)
        recall = MODULE.recall_at_k(index, queries, [{1, 2}, {3, 4}], 2, batch_size=2)
        self.assertEqual(0.5, recall)


if __name__ == '__main__':
    unittest.main()
