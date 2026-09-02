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

    def test_ini_is_authoritative_and_reuses_query_count_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'build.ini'
            path.write_text(
                '[Base]\n'
                'ValueType=Float\n'
                'DistCalcMethod=L2\n'
                'Dim=128\n'
                'VectorPath=base.bin\n'
                'QueryPath=query.bin\n'
                'TruthPath=truth.txt\n'
                '\n'
                '[RaBitQAutoTune]\n'
                'isExecute=true\n'
                'OutputDir=tuning\n'
                'RecallAt=1000\n'
                'TargetRecall=0.97\n'
                'MinBits=2\n'
                'MaxBits=7\n'
                '\n'
                '[SearchSSDIndex]\n'
                'QueryCountLimit=10000\n',
                encoding='ascii')
            args = MODULE.get_config(['--config', str(path)])
            self.assertTrue(args.rabitq_auto_tune)
            self.assertEqual(10000, args.Q)
            self.assertEqual(1000, args.k)
            self.assertEqual(0.97, args.rabitq_target_recall)
            self.assertEqual((2, 7), (args.rabitq_min_bits, args.rabitq_max_bits))
            self.assertEqual('base.bin', args.data_file)
            self.assertEqual('query.bin', args.query_file)
            self.assertEqual('truth.txt', args.output_truth)

            with self.assertRaises(ValueError):
                MODULE.get_config(['--config', str(path), '--Q', '1'])

    def test_ini_rejects_unknown_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'build.ini'
            path.write_text(
                '[Base]\n'
                'ValueType=Float\n'
                'DistCalcMethod=L2\n'
                'Dim=128\n'
                'VectorPath=base.bin\n'
                'QueryPath=query.bin\n'
                'TruthPath=truth.txt\n'
                '\n'
                '[RaBitQAutoTune]\n'
                'isExecute=true\n'
                'OutputDir=tuning\n'
                'RecallAt=1000\n'
                'QueryCount=10000\n'
                'TypoTargetRecal=0.95\n',
                encoding='ascii')
            with self.assertRaises(ValueError):
                MODULE.get_config(['--config', str(path)])

    def test_ini_rejects_inherited_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'build.ini'
            path.write_text(
                '[DEFAULT]\n'
                'QueryCount=1\n'
                '\n'
                '[Base]\n'
                'ValueType=Float\n'
                'DistCalcMethod=L2\n'
                'Dim=128\n'
                'VectorPath=base.bin\n'
                'QueryPath=query.bin\n'
                'TruthPath=truth.txt\n'
                '\n'
                '[RaBitQAutoTune]\n'
                'isExecute=true\n'
                'OutputDir=tuning\n'
                'RecallAt=1000\n'
                '\n'
                '[SearchSSDIndex]\n'
                'QueryCountLimit=10000\n',
                encoding='ascii')
            with self.assertRaises(ValueError):
                MODULE.get_config(['--config', str(path)])


if __name__ == '__main__':
    unittest.main()
