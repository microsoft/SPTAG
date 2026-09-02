import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


MODULE_PATH = Path(__file__).with_name('OPQ_gpu_train_infer.py')
SPEC = importlib.util.spec_from_file_location('opq_gpu_train_infer', MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


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
            truths = MODULE.load_ground_truth(path, 2)
            np.testing.assert_array_equal(truths, [[1, 2, 3], [4, 5, 6]])
            with self.assertRaises(ValueError):
                MODULE.load_ground_truth(path, 3)

    def test_inferred_ground_truth_topk_requires_uniform_unique_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'truth.txt'
            path.write_text('1 2\n3 4 5\n', encoding='ascii')
            with self.assertRaises(ValueError):
                MODULE.load_ground_truth(path, 2)
            path.write_text('1 2\n3 3\n', encoding='ascii')
            with self.assertRaises(ValueError):
                MODULE.load_ground_truth(path, 2)

    def test_reranking_recall_uses_result_num_with_deeper_candidates(self):
        import faiss
        base = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        queries = np.asarray([[0.1]], dtype=np.float32)
        index = faiss.IndexFlatL2(1)
        index.add(base)
        candidates = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
        recall = MODULE.reranking_recall_at_k(
            faiss, index, queries, candidates, 2)
        self.assertEqual(1.0, recall)
        with self.assertRaises(ValueError):
            MODULE.reranking_recall_at_k(
                faiss, index, queries, candidates[:, :2], 2)

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
                'TargetRecall=0.97\n'
                'MinBits=2\n'
                'MaxBits=7\n'
                '\n'
                '[BuildSSDIndex]\n'
                'NumberOfThreads=46\n'
                '\n'
                '[SearchSSDIndex]\n'
                'QueryCountLimit=10000\n'
                'ResultNum=100\n',
                encoding='ascii')
            args = MODULE.get_config(['--config', str(path)])
            self.assertTrue(args.rabitq_auto_tune)
            self.assertEqual(10000, args.Q)
            self.assertEqual(100, args.k)
            self.assertEqual(46, args.T)
            self.assertEqual('float32', args.target_type)
            self.assertIsNone(args.train_samples)
            self.assertEqual(0.97, args.rabitq_target_recall)
            self.assertEqual((2, 7), (args.rabitq_min_bits, args.rabitq_max_bits))
            self.assertEqual('base.bin', args.data_file)
            self.assertEqual('query.bin', args.query_file)
            self.assertEqual('truth.txt', args.output_truth)

            with self.assertRaises(ValueError):
                MODULE.get_config(['--config', str(path), '--Q', '1'])

    def test_streaming_centroid_uses_all_base_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'vectors.bin'
            vectors = np.arange(20, dtype=np.float32).reshape(5, 4)
            with path.open('wb') as output:
                np.asarray([5, 4], dtype=np.int32).tofile(output)
                vectors.tofile(output)
            args = type('Args', (), {
                'data_file': str(path),
                'dim': 4,
                'data_normalize': 0,
                'data_type': 'float32',
                'target_type': 'float32',
            })()
            centroid, count = MODULE.compute_streaming_centroid(args)
            self.assertEqual(5, count)
            np.testing.assert_array_equal(centroid, vectors.mean(axis=0))

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
                'QueryCount=10000\n'
                'TypoTargetRecal=0.95\n',
                encoding='ascii')
            with self.assertRaises(ValueError):
                MODULE.get_config(['--config', str(path)])

    def test_ini_rejects_removed_execution_parameters(self):
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
                'Threads=46\n'
                '\n'
                '[SearchSSDIndex]\n'
                'QueryCountLimit=10000\n'
                'ResultNum=100\n',
                encoding='ascii')
            with self.assertRaises(ValueError):
                MODULE.get_config(['--config', str(path)])

    def test_ini_rejects_normalization_and_non_l2_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'build.ini'
            config = (
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
                'DataNormalize=1\n'
                '\n'
                '[BuildSSDIndex]\n'
                'NumberOfThreads=46\n'
                '\n'
                '[SearchSSDIndex]\n'
                'QueryCountLimit=10000\n'
                'ResultNum=100\n')
            path.write_text(config, encoding='ascii')
            with self.assertRaises(ValueError):
                MODULE.get_config(['--config', str(path)])
            path.write_text(
                config.replace('DistCalcMethod=L2', 'DistCalcMethod=Cosine')
                      .replace('DataNormalize=1\n', ''),
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
                '\n'
                '[SearchSSDIndex]\n'
                'QueryCountLimit=10000\n'
                'ResultNum=100\n',
                encoding='ascii')
            with self.assertRaises(ValueError):
                MODULE.get_config(['--config', str(path)])


if __name__ == '__main__':
    unittest.main()
