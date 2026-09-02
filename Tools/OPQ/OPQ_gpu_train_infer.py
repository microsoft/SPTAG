import numpy as np
import math
import tqdm
import time
from struct import pack, unpack, calcsize
from typing import Dict, List
import heapq
import argparse
import copy
import configparser
import json
from operator import itemgetter
import os
import subprocess
import sys

RABITQ_BATCH_SIZE = 1000000

def get_cli_parser():
    parser = argparse.ArgumentParser(description ='implementation of nnsearch.')
    parser.add_argument('--config', type = str, help='native INI containing [RaBitQAutoTune]')
    parser.add_argument('--data_file', default = 'traindata', type = str, help = 'binary data file')
    parser.add_argument('--query_file', default = 'query.tsv', type= str, help='query tsv file')
    parser.add_argument('--data_normalize', default = 0, type = int, help='normalize data vectors')
    parser.add_argument('--query_normalize', default = 0, type = int, help='normalize query vectors')
    parser.add_argument('--data_type', default = 'float32', type = str, help = 'data type for binary file: float32, int8, int16')
    parser.add_argument('--target_type', default = 'float32', type = str, help = 'GPU data type')
    parser.add_argument('--k', type= int, default = 32, help='knn')
    parser.add_argument('--dim', type= int, default = 100, help='vector dimensions')
    parser.add_argument('--B', type= int, default = -1, help='batch data size')
    parser.add_argument('--Q', type= int, default = 10000, help='batch query size')
    parser.add_argument('--S', type= int, default = 1000, help='data split size')
    parser.add_argument('--D', type= str, default = "L2", help='distance type')
    parser.add_argument('--output_truth', type = str, default = "truth.txt", help='output truth file')
    parser.add_argument('--data_format', type = str, default = "DEFAULT", help='data format')
    parser.add_argument('--task', type = int, default = 0, help='task id')
    parser.add_argument('-log_dir', type = str, default = "", help='debug log dir in cosmos')

    parser.add_argument('--T', type = int, default = 32, help="thread number")
    parser.add_argument('--train_samples', type = int, default = 1000000, help='OPQ, PQ training samples')
    parser.add_argument('--quan_type', type = str, default = 'none', help='quantizer type')
    parser.add_argument('--quan_dim', type = int, default = -1, help='quantized vector dimensions')
    parser.add_argument('--output_dir', type = str, default = 'quan_tmp', help='output dir')
    parser.add_argument('--output_quantizer', type = str, default = "quantizer.bin", help='output quantizer file')
    parser.add_argument('--output_quan_vector_file', type = str, default = "", help='quantized vectors')
    parser.add_argument('--output_rec_vector_file', type = str, default = "", help = "reconstruct vectors")
    parser.add_argument('--quan_test', type = int, default = 0, help='compare with ground truth')
    parser.add_argument('--rabitq_auto_tune', action = 'store_true', help='select the minimum RaBitQ storage bits before encoding')
    parser.add_argument('--rabitq_target_recall', type = float, default = 0.95, help='minimum Recall@k for RaBitQ auto tuning')
    parser.add_argument('--rabitq_min_bits', type = int, default = 1, help='minimum RaBitQ storage bits to evaluate')
    parser.add_argument('--rabitq_max_bits', type = int, default = 8, help='maximum RaBitQ storage bits to evaluate')
    parser.add_argument('--rabitq_tuning_result', type = str, default = 'rabitq_auto_tuning.json', help='auto-tuning result file under output_dir')
    return parser

def load_rabitq_auto_tune_ini(path):
    config = configparser.ConfigParser(interpolation=None)
    if not config.read(path):
        raise ValueError(f'cannot read INI file: {path}')
    if config.defaults():
        raise ValueError('[DEFAULT] parameters are forbidden in strict RaBitQ INI mode')
    section_name = 'RaBitQAutoTune'
    if not config.has_section(section_name):
        raise ValueError(f'INI file is missing [{section_name}]')
    if not config.has_section('Base'):
        raise ValueError('INI file is missing [Base]')
    section = config[section_name]
    allowed_keys = {
        'isexecute', 'outputdir', 'querycount', 'targetrecall',
        'minbits', 'maxbits', 'tuningresult', 'dataformat', 'task',
    }
    unknown_keys = set(section.keys()) - allowed_keys
    if unknown_keys:
        raise ValueError(
            f'unknown [{section_name}] parameter(s): {", ".join(sorted(unknown_keys))}')
    required_keys = ('OutputDir',)
    missing_keys = [key for key in required_keys if not section.get(key)]
    if missing_keys:
        raise ValueError(
            f'[{section_name}] is missing required parameter(s): {", ".join(missing_keys)}')
    if not section.getboolean('isExecute', fallback=False):
        raise ValueError(f'[{section_name}] isExecute must be true')

    base = config['Base']
    base_required_keys = ('VectorPath', 'QueryPath', 'TruthPath', 'Dim', 'ValueType', 'DistCalcMethod')
    missing_base_keys = [key for key in base_required_keys if not base.get(key)]
    if missing_base_keys:
        raise ValueError(
            f'[Base] is missing required parameter(s): {", ".join(missing_base_keys)}')
    value_types = {
        'float': 'float32',
        'int8': 'int8',
        'uint8': 'uint8',
        'int16': 'int16',
    }
    value_type = base['ValueType'].lower()
    if value_type not in value_types:
        raise ValueError(f'unsupported [Base] ValueType for RaBitQ tuning: {base["ValueType"]}')
    if value_type != 'float':
        raise ValueError('RaBitQ tuning requires [Base] ValueType=Float')
    if base['DistCalcMethod'].lower() != 'l2':
        raise ValueError('RaBitQ tuning requires [Base] DistCalcMethod=L2')

    query_count = section.getint('QueryCount', fallback=None)
    if query_count is None:
        if not config.has_option('SearchSSDIndex', 'QueryCountLimit'):
            raise ValueError(
                f'query count must be set by [{section_name}] QueryCount or '
                '[SearchSSDIndex] QueryCountLimit')
        query_count = config.getint('SearchSSDIndex', 'QueryCountLimit')
    if not config.has_option('SearchSSDIndex', 'ResultNum'):
        raise ValueError('[SearchSSDIndex] ResultNum is required for RaBitQ tuning')
    if not config.has_option('BuildSSDIndex', 'NumberOfThreads'):
        raise ValueError('[BuildSSDIndex] NumberOfThreads is required for RaBitQ tuning')

    return argparse.Namespace(
        config=path,
        data_file=base['VectorPath'],
        query_file=base['QueryPath'],
        data_normalize=0,
        query_normalize=0,
        data_type=value_types[value_type],
        target_type=value_types[value_type],
        k=config.getint('SearchSSDIndex', 'ResultNum'),
        dim=base.getint('Dim'),
        B=RABITQ_BATCH_SIZE,
        Q=query_count,
        S=1000,
        D=base['DistCalcMethod'],
        output_truth=base['TruthPath'],
        data_format=section.get('DataFormat', fallback='DEFAULT'),
        task=section.getint('Task', fallback=0),
        log_dir='',
        T=config.getint('BuildSSDIndex', 'NumberOfThreads'),
        train_samples=None,
        quan_type='rabitq',
        quan_dim=-1,
        output_dir=section['OutputDir'],
        output_quantizer='quantizer.bin',
        output_quan_vector_file='',
        output_rec_vector_file='',
        quan_test=1,
        rabitq_auto_tune=True,
        rabitq_target_recall=section.getfloat('TargetRecall', fallback=0.95),
        rabitq_min_bits=section.getint('MinBits', fallback=1),
        rabitq_max_bits=section.getint('MaxBits', fallback=8),
        rabitq_tuning_result=section.get(
            'TuningResult', fallback='rabitq_auto_tuning.json'),
    )

def get_config(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument('--config')
    config_args, remaining = config_probe.parse_known_args(argv)
    if config_args.config is not None:
        if remaining:
            raise ValueError(
                'INI mode accepts only --config; command-line parameter overrides are forbidden')
        return load_rabitq_auto_tune_ini(config_args.config)
    return get_cli_parser().parse_args(argv)

def is_binary_vector_file(filename):
    suffixes = ('.bin', '.fbin', '.u8bin', '.i8bin')
    if filename.endswith(suffixes):
        return True
    stem, separator, shard = filename.rpartition('.')
    return separator != '' and shard.isdigit() and stem.endswith(suffixes)

class DataReader:
    def __init__(self, filename, featuredim, batchsize, normalize, datatype, targettype='float32'):
        self.mytype = targettype
        if is_binary_vector_file(filename):
            self.fin = open(filename, 'rb')
            R = unpack('i', self.fin.read(4))[0]
            self.featuredim = unpack('i', self.fin.read(4))[0]
            self.isbinary = True
            self.type = datatype
            print ('Open Binary DataReader for data(%d,%d)...' % (R, self.featuredim))
        else:
            with open(filename) as f:
                R = sum(1 for _ in f)
            self.fin = open(filename, 'r')
            self.featuredim = featuredim
            self.isbinary = False
            self.type = self.mytype

        if batchsize <= 0:
            batchsize = R
        else:
            batchsize = min(batchsize, R)
        self.query = np.zeros([batchsize, self.featuredim], dtype=self.mytype)
        self.normalize = normalize

    def norm(self, data):
        square = np.sqrt(np.sum(np.square(data), axis=1))
        data[square < 1e-6] = 1e-6 / math.sqrt(float(self.featuredim))
        square[square < 1e-6] = 1e-6
        data = data / square.reshape([-1, 1])
        return data

    def readbatch(self):
        numQuerys = self.query.shape[0]
        i = 0
        if self.isbinary:
            while i < numQuerys:
                vec = self.fin.read((np.dtype(self.type).itemsize)*self.featuredim)
                if len(vec) == 0: break
                if len(vec) != (np.dtype(self.type).itemsize)*self.featuredim:
                    print ("%d vector cannot be read correctly: require %d bytes but only read %d bytes" % (i, (np.dtype(self.type).itemsize)*self.featuredim, len(vec)))
                    continue
                self.query[i] = np.frombuffer(vec, dtype=self.type).astype(self.mytype)
                i += 1
        else:
             while i < numQuerys:
                 line = self.fin.readline()
                 if len(line) == 0: break

                 index = line.rfind("\t")
                 if index < 0: continue

                 items = line[index+1:].split("|")
                 if len(items) < self.featuredim: continue

                 for j in range(self.featuredim): self.query[i, j] = float(items[j])
                 i += 1
        print ('Load batch query size:%r' % (i))
        if self.normalize != 0: return i, self.norm(self.query[0:i])
        return i, self.query[0:i]

    def readallbatches(self):
        numQuerys = self.query.shape[0]
        data = []
        R = 0
        while True:
            i, q = self.readbatch()
            if i == numQuerys:
                data.append(copy.deepcopy(q))
                R += i
            else:
                if i > 0:
                    data.append(copy.deepcopy(q[0:i]))
                    R += i
                break
        return R, data

    def close(self):
        self.fin.close()

def gpusearch(args):
    import faiss
    ngpus = faiss.get_num_gpus()
    print ('number of GPUs:', ngpus)

    gpu_resources = []
    for i in range(ngpus):
        res = faiss.StandardGpuResources()
        gpu_resources.append(res)

    datareader = DataReader(args.data_file, args.dim, args.B, args.data_normalize, args.data_type, args.target_type)
    queryreader = DataReader(args.query_file, args.dim, args.Q, args.query_normalize, args.data_type, args.target_type)
    RQ, dataQ = queryreader.readallbatches()

    batch = 0
    totaldata = 0
    while True:
        numData, data = datareader.readbatch()
        if numData == 0:
            datareader.close()
            break

        totaldata += numData
        batch += 1
        print ("Begin batch %d" % batch)

        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = False if args.target_type == 'float32' else True
        co.useFloat16CoarseQuantizer = False
        if args.D != 'Cosine':
            cpu_index = faiss.IndexFlatL2(args.dim)
        else:
            cpu_index = faiss.IndexFlatIP(args.dim)

        gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co, ngpu=ngpus)
        gpu_index.add(data)

        fout = open('truth.txt.%d' % batch, 'w')
        foutd = open('dist.bin.%d' % batch, 'wb')

        foutd.write(pack('i', RQ))
        foutd.write(pack('i', args.k))

        for query in dataQ:
            D, I = gpu_index.search(query, args.k)
            foutd.write(D.tobytes())
            for i in range(I.shape[0]):
                for j in range(I.shape[1]):
                    fout.write(str(I[i][j]) + " ")
                fout.write('\n')

        fout.close()
        foutd.close()

    if args.B <= 0 or args.B >= totaldata: args.B = totaldata

    truth = [[] for j in range(RQ)]
    for i in range(1, batch + 1):
        f = open('truth.txt.%d' % i, 'r')
        fd = open('dist.bin.%d' % i, 'rb')
        r = unpack('i', fd.read(4))[0]
        c = unpack('i', fd.read(4))[0]
        print ('batch %d: r:%d c:%d RQ:%d k:%d' % (i, r, c, RQ, args.k))
        currdist = np.frombuffer(fd.read(4 * RQ * args.k), dtype=np.float32).reshape((RQ, args.k))
        fd.close()

        for j in range(RQ):
            items = f.readline()[0:-1].split()
            truth[j].extend([(int(items[k]) + args.B * (i-1), currdist[j][k]) for k in range(args.k)])
            truth[j].sort(key=itemgetter(1, 0))
            truth[j] = truth[j][0:args.k]
        f.close()

    if not os.path.exists(args.output_truth + '.dist'):
        os.mkdir(args.output_truth + '.dist')

    fout = open(args.output_truth, 'w')
    foutd = open(args.output_truth + '.dist\\dist.bin.' + str(args.task), 'wb')
    foutd.write(pack('i', RQ))
    foutd.write(pack('i', args.k))
    for i in range(RQ):
        for j in range(args.k):
            fout.write(str(truth[i][j][0]) + " ")
            foutd.write(pack('i', truth[i][j][0]))
            foutd.write(pack('f', truth[i][j][1]))
        fout.write('\n')
    fout.close()
    foutd.close()

def search(faiss_index,
           query_embeddings: np.ndarray,
           topk: int = 1000,
           nprobe: int = None,
           batch_size: int = 64):
    import faiss
    if nprobe is not None:
        if isinstance(faiss_index, faiss.IndexPreTransform):
            ivf_index = faiss.downcast_index(faiss_index.index)
            ivf_index.nprobe = nprobe
        else:
            faiss_index.nprobe = nprobe

    start_time = time.time()
    if batch_size:
        batch_num = math.ceil(len(query_embeddings) / batch_size)
        all_scores = []
        all_search_results = []
        for step in tqdm.tqdm(range(batch_num)):
            start = batch_size * step
            end = min(batch_size * (step + 1), len(query_embeddings))
            batch_emb = np.array(query_embeddings[start:end])
            score, batch_results = faiss_index.search(batch_emb, topk)
            all_search_results.extend([list(x) for x in batch_results])
            all_scores.extend([list(x) for x in score])
    else:
        all_scores, all_search_results = faiss_index.search(query_embeddings, topk)
    search_time = time.time() - start_time
    print(
        f'number of query:{len(query_embeddings)},  searching time per query: {search_time / len(query_embeddings)}')
    return all_scores, all_search_results

def evaluate(retrieve_results: List[List[int]],
             ground_truths: Dict[int, List[int]],
             MRR_cutoffs: List[int] = [10],
             Recall_cutoffs: List[int] = [5, 10, 50],
             qids: List[int] = None):
    """
    calculate MRR and Recall
    """
    MRR = [0.0] * len(MRR_cutoffs)
    Recall = [0.0] * len(Recall_cutoffs)
    ranking = []
    finalk = min(Recall_cutoffs)
    if qids is None:
        qids = list(range(len(retrieve_results)))
    for qid, candidate_pid in zip(qids, retrieve_results):
        if qid in ground_truths:
            target_pid = ground_truths[qid]
            ranking.append(-1)

            for i in range(0, max(MRR_cutoffs)):
                if candidate_pid[i] in target_pid:
                    ranking.pop()
                    ranking.append(i + 1)
                    for inx, cutoff in enumerate(MRR_cutoffs):
                        if i <= cutoff - 1:
                            MRR[inx] += 1 / (i + 1)
                    break

            for i, k in enumerate(Recall_cutoffs):
                Recall[i] += (len(set.intersection(set(target_pid), set(candidate_pid[:k]))) / len(set(target_pid)))

    if len(ranking) == 0:
        raise IOError("No matching QIDs found. Are you sure you are scoring the evaluation set?")

    print(f"{len(ranking)} matching queries found")
    MRR = [x / len(ranking) for x in MRR]
    for i, k in enumerate(MRR_cutoffs):
        print(f'MRR{finalk}@{k}:{MRR[i]}')

    Recall = [x / len(ranking) for x in Recall]
    for i, k in enumerate(Recall_cutoffs):
        print(f'Recall{finalk}@{k}:{Recall[i]}')

    return MRR, Recall

def sptag_rabitq_storage_bytes(dim, bits):
    if dim <= 0:
        raise ValueError('RaBitQ dimension must be positive')
    if bits < 1 or bits > 8:
        raise ValueError(f'RaBitQ bits must be in [1, 8], got {bits}')
    padded_dimension = ((dim + 63) // 64) * 64
    return padded_dimension * bits // 8 + 5 * np.dtype(np.float32).itemsize

def rabitq_bits_from_quantized_dimension(dim, quan_dim):
    total_bits = quan_dim * 8
    if dim <= 0 or quan_dim <= 0 or total_bits % dim != 0:
        raise ValueError(f'quan_dim={quan_dim} does not represent an integral RaBitQ bit count for dimension {dim}')
    bits = total_bits // dim
    if bits < 1 or bits > 8:
        raise ValueError(f'RaBitQ bits must be in [1, 8], got {bits}')
    return bits

def load_ground_truth(path, query_count):
    rows = []
    candidate_count = None
    with open(path, 'r') as truth_file:
        for query_id in range(query_count):
            line = truth_file.readline()
            if not line:
                raise ValueError(f'ground truth contains only {query_id} queries, expected {query_count}')
            neighbors = line.strip().split()
            if candidate_count is None:
                candidate_count = len(neighbors)
                if candidate_count == 0:
                    raise ValueError('ground truth query 0 contains no neighbors')
            if len(neighbors) != candidate_count:
                raise ValueError(
                    f'ground truth query {query_id} contains {len(neighbors)} neighbors, '
                    f'expected exactly {candidate_count}')
            row = [int(neighbor) for neighbor in neighbors]
            if len(set(row)) != candidate_count:
                raise ValueError(f'ground truth query {query_id} contains duplicate neighbor IDs')
            rows.append(row)
    return np.asarray(rows, dtype=np.int64)

def reranking_recall_at_k(faiss, faiss_index, queries, candidates, topk):
    if len(queries) != len(candidates):
        raise ValueError('query and ground-truth counts differ')
    if len(queries) == 0:
        raise ValueError('at least one query is required')
    if topk <= 0:
        raise ValueError('ResultNum must be positive')
    candidate_count = candidates.shape[1]
    if candidate_count <= topk:
        raise ValueError(
            f'ground-truth candidate depth must exceed ResultNum: {candidate_count} <= {topk}')
    if np.any(candidates < 0) or np.any(candidates >= faiss_index.ntotal):
        raise ValueError('ground truth contains a vector ID outside the base data')

    recall_sum = 0.0
    for query_id in tqdm.tqdm(range(len(queries))):
        candidate_ids = np.ascontiguousarray(candidates[query_id])
        parameters = faiss.SearchParameters()
        parameters.sel = faiss.IDSelectorArray(
            candidate_count, faiss.swig_ptr(candidate_ids))
        _, results = faiss_index.search(
            np.ascontiguousarray(queries[query_id:query_id + 1]), topk,
            params=parameters)
        expected = set(int(candidate) for candidate in candidate_ids[:topk])
        recall_sum += len(expected.intersection(int(candidate) for candidate in results[0])) / topk
    return recall_sum / len(queries)

def create_rabitq_index(faiss, dim, bits, centroid):
    faiss_index = faiss.index_factory(dim, f"RaBitQ{bits}", faiss.METRIC_L2)
    faiss_index.train(np.ascontiguousarray(centroid.reshape(1, dim), dtype=np.float32))
    return faiss_index

def compute_streaming_centroid(args):
    datareader = DataReader(
        args.data_file, args.dim, RABITQ_BATCH_SIZE,
        args.data_normalize, args.data_type, args.target_type)
    accumulator = np.zeros(args.dim, dtype=np.float64)
    total = 0
    while True:
        num_data, data = datareader.readbatch()
        if num_data == 0:
            break
        accumulator += np.sum(data, axis=0, dtype=np.float64)
        total += num_data
    datareader.close()
    if total == 0:
        raise ValueError('RaBitQ input data is empty')
    return (accumulator / total).astype(np.float32), total

def add_rabitq_data(args, faiss_index):
    datareader = DataReader(
        args.data_file, args.dim, args.B, args.data_normalize, args.data_type, args.target_type)
    total = 0
    while True:
        num_data, data = datareader.readbatch()
        if num_data == 0:
            break
        faiss_index.add(data)
        total += num_data
    datareader.close()
    if total == 0:
        raise ValueError('RaBitQ input data is empty')
    return total

def tune_rabitq_bits(args, faiss, centroid, centroid_vector_count, queries, ground_truth_candidates):
    if not 0.0 < args.rabitq_target_recall <= 1.0:
        raise ValueError('rabitq_target_recall must be in (0, 1]')
    if args.rabitq_min_bits < 1 or args.rabitq_max_bits > 8 or args.rabitq_min_bits > args.rabitq_max_bits:
        raise ValueError('RaBitQ tuning range must satisfy 1 <= min_bits <= max_bits <= 8')

    trials = []
    for bits in range(args.rabitq_min_bits, args.rabitq_max_bits + 1):
        print(f'Auto tuning RaBitQ{bits} for Recall@{args.k} >= {args.rabitq_target_recall:.6f}')
        candidate = create_rabitq_index(faiss, args.dim, bits, centroid)
        data_count = add_rabitq_data(args, candidate)
        if data_count != centroid_vector_count:
            raise RuntimeError(
                f'base vector count changed during tuning: {centroid_vector_count} != {data_count}')
        recall = reranking_recall_at_k(
            faiss, candidate, queries, ground_truth_candidates, args.k)
        trials.append({'bits': bits, 'recall': recall})
        print(f'RaBitQ{bits} Recall@{args.k}: {recall:.6f}')
        if recall >= args.rabitq_target_recall:
            return bits, data_count, trials

    measured = ', '.join(f'{trial["bits"]}-bit={trial["recall"]:.6f}' for trial in trials)
    raise RuntimeError(
        f'No RaBitQ bit count in [{args.rabitq_min_bits}, {args.rabitq_max_bits}] '
        f'reached Recall@{args.k} >= {args.rabitq_target_recall:.6f}; {measured}')

def train_rabitq(args):
    import faiss

    output_dir = args.output_dir

    if args.D != 'L2':
        raise ValueError('SPTAG global RaBitQ tuning requires [Base] DistCalcMethod=L2')
    if args.Q <= 0 or args.k <= 0:
        raise ValueError('Q and k must be positive')
    if len(args.output_quan_vector_file) > 0 or len(args.output_rec_vector_file) > 0:
        raise ValueError(
            'RaBitQ vectors must be generated by the native SPTAG quantizer after tuning; '
            'do not use output_quan_vector_file or output_rec_vector_file')
    centroid, centroid_vector_count = compute_streaming_centroid(args)
    print(f'train RaBitQ using the centroid of all {centroid_vector_count} base vectors ...')
    if args.T <= 0:
        raise ValueError('NumberOfThreads must be positive')
    faiss.omp_set_num_threads(args.T)
    if args.rabitq_auto_tune:
        if args.quan_test <= 0:
            raise ValueError('rabitq_auto_tune requires quan_test > 0 and a pre-generated ground truth')
        queryreader = DataReader(
            args.query_file, args.dim, args.Q, args.query_normalize, args.data_type, args.target_type)
        num_query, queries = queryreader.readbatch()
        queryreader.close()
        if num_query != args.Q:
            raise ValueError(f'query file contains {num_query} queries, but configured Q is {args.Q}')
        ground_truth_candidates = load_ground_truth(
            args.output_truth, num_query)
        bits, data_count, trials = tune_rabitq_bits(
            args, faiss, centroid, centroid_vector_count, queries, ground_truth_candidates)
        result = {
            'selected_bits': bits,
            'native_quantizer_qd': bits,
            'storage_bytes_per_vector': sptag_rabitq_storage_bytes(args.dim, bits),
            'target_recall': args.rabitq_target_recall,
            'recall_at': args.k,
            'rerank_candidate_count': ground_truth_candidates.shape[1],
            'query_count': num_query,
            'data_count': data_count,
            'centroid_vector_count': centroid_vector_count,
            'trials': trials,
        }
        result_path = os.path.join(output_dir, args.rabitq_tuning_result)
        temporary_result_path = result_path + '.tmp'
        with open(temporary_result_path, 'w') as result_file:
            json.dump(result, result_file, indent=2)
            result_file.write('\n')
        os.replace(temporary_result_path, result_path)
        print(f'Selected RaBitQ storage bits: {bits}; result: {result_path}')
    else:
        bits = rabitq_bits_from_quantized_dimension(args.dim, args.quan_dim)
        faiss_index = create_rabitq_index(faiss, args.dim, bits, centroid)
        if args.quan_test > 0:
            add_rabitq_data(args, faiss_index)

    if args.quan_test > 0 and not args.rabitq_auto_tune:
        queryreader = DataReader(args.query_file, args.dim, args.Q, args.query_normalize, args.data_type, args.target_type)
        numQuery, query = queryreader.readbatch()

        qid2ground_truths = {}
        f = open(args.output_truth, 'r')
        for i in range(numQuery):
            items = f.readline()[0:-1].strip().split(' ')
            qid2ground_truths[i] = set([int(gt) for gt in items])
        f.close()

        # Test the performance
        scores, retrieve_results = search(faiss_index, query, topk=args.k * 10, batch_size=64)
        evaluate(retrieve_results, qid2ground_truths, MRR_cutoffs=[args.k, args.k*2, args.k*4, args.k*8], Recall_cutoffs=[args.k, args.k*2, args.k*4, args.k*8], qids=None)

def train_pq(args):
    import faiss

    output_dir = args.output_dir

    if args.train_samples > args.B: args.train_samples = args.B

    datareader = DataReader(args.data_file, args.dim, args.B, args.data_normalize, args.data_type, args.target_type)

    print (f'train PQ using {args.train_samples} samples...')

    subvector_num = args.quan_dim
    subvector_bits = 8
    numData, data = datareader.readbatch()

    faiss.omp_set_num_threads(args.T)

    faiss_index = faiss.index_factory(len(data[0]), f"PQ{subvector_num}x{subvector_bits}", faiss.METRIC_L2)
    print('Training the index with doc embeddings')

    t1 = time.perf_counter()
    faiss_index.train(data[0:args.train_samples])
    t2 = time.perf_counter()
    elapsed_time = t2 - t1
    print (f"Train time: {elapsed_time:.6f} seconds")


    rtype = np.uint8(0)
    if args.data_type == 'uint8':
        rtype = np.uint8(1)
    elif args.data_type == 'int16':
        rtype = np.uint8(2)
    elif args.data_type == 'float32':
        rtype = np.uint8(3)

    ivf_index = faiss.downcast_index(faiss_index)
    centroid_embedings = faiss.vector_to_array(ivf_index.pq.centroids)
    codebooks = centroid_embedings.reshape(ivf_index.pq.M, ivf_index.pq.ksub, ivf_index.pq.dsub)
    print ('codebooks shape:')
    print (codebooks.shape)

    codebooks = codebooks.astype(np.float32)
    with open(os.path.join(output_dir, args.output_quantizer + '.' + str(args.task)),'wb') as f:
        f.write(pack('B', 1))
        f.write(pack('B', rtype))
        f.write(pack('i', codebooks.shape[0]))
        f.write(pack('i', codebooks.shape[1]))
        f.write(pack('i', codebooks.shape[2]))
        f.write(codebooks.tobytes())

    if args.quan_test == 0 and len(args.output_quan_vector_file) == 0 and len(args.output_rec_vector_file) == 0:
        if os.path.exists(args.output_truth):
            os.rename(args.output_truth, os.path.join(output_dir, 'truth.txt' + '.' + str(args.task)))
        return

    if len(args.output_quan_vector_file) > 0:
        fquan = open(os.path.join(output_dir, args.output_quan_vector_file + '.' + str(args.task) + '.tmp'), 'wb')
        fquan.write(pack('i', 0))
        fquan.write(pack('i', args.quan_dim))

    if len(args.output_rec_vector_file) > 0:
        frec = open(os.path.join(output_dir, args.output_rec_vector_file + '.' + str(args.task) + '.tmp'), 'wb')
        frec.write(pack('i', 0))
        frec.write(pack('i', data.shape[1]))

    writeitems = 0
    while numData > 0:
        if args.quan_test > 0: faiss_index.add(data)

        codes = ivf_index.pq.compute_codes(data)

        print ('codes shape:')
        print (codes.shape)

        if len(args.output_quan_vector_file) > 0:
            fquan.write(codes.tobytes())

        if len(args.output_rec_vector_file) > 0:
            reconstructed = ivf_index.pq.decode(codes).astype(args.data_type)
            frec.write(reconstructed.tobytes())

        writeitems += numData
        numData, data = datareader.readbatch()

    datareader.close()

    if len(args.output_quan_vector_file) > 0:
        p = fquan.tell()
        fquan.seek(0)
        fquan.write(pack('i', writeitems))
        fquan.seek(p)
        fquan.close()
        if os.path.exists(os.path.join(output_dir, args.output_quan_vector_file + '.' + str(args.task))):
            os.remove(os.path.join(output_dir, args.output_quan_vector_file + '.' + str(args.task)))
        os.rename(os.path.join(output_dir, args.output_quan_vector_file + '.' + str(args.task) + '.tmp'), os.path.join(output_dir, args.output_quan_vector_file + '.' + str(args.task)))
    if len(args.output_rec_vector_file) > 0:
        p = frec.tell()
        frec.seek(0)
        frec.write(pack('i', writeitems))
        frec.seek(p)
        frec.close()
        if os.path.exists(os.path.join(output_dir, args.output_rec_vector_file + '.' + str(args.task))):
            os.remove(os.path.join(output_dir, args.output_rec_vector_file + '.' + str(args.task)))
        os.rename(os.path.join(output_dir, args.output_rec_vector_file + '.' + str(args.task) + '.tmp'), os.path.join(output_dir, args.output_rec_vector_file + '.' + str(args.task)))

    os.rename(args.output_truth, os.path.join(output_dir, 'truth.txt' + '.' + str(args.task)))

    if args.quan_test > 0:
        queryreader = DataReader(args.query_file, args.dim, args.Q, args.query_normalize, args.data_type, args.target_type)
        numQuery, query = queryreader.readbatch()

        qid2ground_truths = {}
        f = open(os.path.join(output_dir, 'truth.txt.' + str(args.task)), 'r')
        for i in range(numQuery):
            items = f.readline()[0:-1].strip().split(' ')
            qid2ground_truths[i] = set([int(gt) for gt in items])
        f.close()

        # Test the performance
        scores, retrieve_results = search(faiss_index, query, topk=args.k * 10, batch_size=64)
        evaluate(retrieve_results, qid2ground_truths, MRR_cutoffs=[args.k, args.k*2, args.k*4, args.k*8], Recall_cutoffs=[args.k, args.k*2, args.k*4, args.k*8], qids=None)

def train_opq(args):
    import faiss

    output_dir = args.output_dir

    if args.train_samples > args.B: args.train_samples = args.B

    datareader = DataReader(args.data_file, args.dim, args.B, args.data_normalize, args.data_type, args.target_type)

    print (f'train OPQ using {args.train_samples} samples...')

    index_method = 'opq'
    ivf_centers_num = -1
    subvector_num = args.quan_dim
    subvector_bits = 8
    numData, data = datareader.readbatch()
    data = data.astype(np.float32)
    args.data_type = 'float32'
    faiss.omp_set_num_threads(args.T)
    faiss_index = faiss.index_factory(len(data[0]), f"OPQ{subvector_num},PQ{subvector_num}x{subvector_bits}", faiss.METRIC_L2)

    print('Training the index with doc embeddings')
    t1 = time.perf_counter()
    faiss_index.train(data[0:args.train_samples])
    t2 = time.perf_counter()
    elapsed_time = t2 - t1
    print (f"Train time: {elapsed_time:.6f} seconds")

    rtype = np.uint8(0)
    if args.data_type == 'uint8':
        rtype = np.uint8(1)
    elif args.data_type == 'int16':
        rtype = np.uint8(2)
    elif args.data_type == 'float32':
        rtype = np.uint8(3)

    if isinstance(faiss_index, faiss.IndexPreTransform):
        vt = faiss.downcast_VectorTransform(faiss_index.chain.at(0))
        assert isinstance(vt, faiss.LinearTransform)
        rotate = faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in)
        rotate_matrix = rotate.T
        print ('rotate shape:')
        print (rotate.shape)

        ivf_index = faiss.downcast_index(faiss_index.index)
        centroid_embedings = faiss.vector_to_array(ivf_index.pq.centroids)
        codebooks = centroid_embedings.reshape(ivf_index.pq.M, ivf_index.pq.ksub, ivf_index.pq.dsub)
        print ('codebooks shape:')
        print (codebooks.shape)

        codebooks = codebooks.astype(np.float32)
        rotate = rotate.astype(np.float32)
        with open(os.path.join(output_dir, args.output_quantizer + '.' + str(args.task)), 'wb') as f:
            f.write(pack('B', 2))
            f.write(pack('B', rtype))
            f.write(pack('i', codebooks.shape[0]))
            f.write(pack('i', codebooks.shape[1]))
            f.write(pack('i', codebooks.shape[2]))
            f.write(codebooks.tobytes())
            f.write(rotate_matrix.tobytes())

    if args.quan_test == 0 and len(args.output_quan_vector_file) == 0 and len(args.output_rec_vector_file) == 0:
        if os.path.exists(args.output_truth):
            os.rename(args.output_truth, os.path.join(output_dir, 'truth.txt' + '.' + str(args.task)))
        return

    if len(args.output_quan_vector_file) > 0:
        fquan = open(os.path.join(output_dir, args.output_quan_vector_file + '.' + str(args.task) + '.tmp'), 'wb')
        fquan.write(pack('i', 0))
        fquan.write(pack('i', args.quan_dim))

    if len(args.output_rec_vector_file) > 0:
        frec = open(os.path.join(output_dir, args.output_rec_vector_file + '.' + str(args.task) + '.tmp'), 'wb')
        frec.write(pack('i', 0))
        frec.write(pack('i', data.shape[1]))

    writeitems = 0
    while numData > 0:
        if args.quan_test > 0: faiss_index.add(data)

        rdata = np.matmul(data, rotate.T)
        codes = ivf_index.pq.compute_codes(rdata)

        print ('codes shape:')
        print (codes.shape)
        if len(args.output_quan_vector_file) > 0:
            fquan.write(codes.tobytes())

        if len(args.output_rec_vector_file) > 0:
            Y = ivf_index.pq.decode(codes)
            reconstructed = np.matmul(Y, rotate).astype(args.data_type)
            frec.write(reconstructed.tobytes())

        writeitems += numData
        numData, data = datareader.readbatch()
        data = data.astype(np.float32)

    datareader.close()

    if len(args.output_quan_vector_file) > 0:
        p = fquan.tell()
        fquan.seek(0)
        fquan.write(pack('i', writeitems))
        fquan.seek(p)
        fquan.close()
        if os.path.exists(os.path.join(output_dir, args.output_quan_vector_file + '.' + str(args.task))):
            os.remove(os.path.join(output_dir, args.output_quan_vector_file + '.' + str(args.task)))
        os.rename(os.path.join(output_dir, args.output_quan_vector_file + '.' + str(args.task) + '.tmp'), os.path.join(output_dir, args.output_quan_vector_file + '.' + str(args.task)))

    if len(args.output_rec_vector_file) > 0:
        p = frec.tell()
        frec.seek(0)
        frec.write(pack('i', writeitems))
        frec.seek(p)
        frec.close()
        if os.path.exists(os.path.join(output_dir, args.output_rec_vector_file + '.' + str(args.task))):
            os.remove(os.path.join(output_dir, args.output_rec_vector_file + '.' + str(args.task)))
        os.rename(os.path.join(output_dir, args.output_rec_vector_file + '.' + str(args.task) + '.tmp'), os.path.join(output_dir, args.output_rec_vector_file + '.' + str(args.task)))

    os.rename(args.output_truth, os.path.join(output_dir, 'truth.txt' + '.' + str(args.task)))

    if args.quan_test > 0:
        queryreader = DataReader(args.query_file, args.dim, args.Q, args.query_normalize, args.data_type, args.target_type)
        numQuery, query = queryreader.readbatch()

        qid2ground_truths = {}
        f = open(os.path.join(output_dir, 'truth.txt.' + str(args.task)), 'r')
        for i in range(numQuery):
            items = f.readline()[0:-1].strip().split(' ')
            qid2ground_truths[i] = set([int(gt) for gt in items])
        f.close()

        # Test the performance
        scores, retrieve_results = search(faiss_index, query, topk=args.k * 10, batch_size=64)
        evaluate(retrieve_results, qid2ground_truths, MRR_cutoffs=[args.k, args.k*2, args.k*4, args.k*8], Recall_cutoffs=[args.k, args.k*2, args.k*4, args.k*8], qids=None)

def quan_reconstruct_vectors(args):
    import faiss

    output_dir = args.output_dir

    datareader = DataReader(args.data_file, args.dim, args.train_samples, args.data_normalize, args.data_type, args.target_type)
    numData, data = datareader.readbatch()

    print ('Quantize and Reconstruct Vectors...')

    quantizer_path = os.path.join(os.path.dirname(args.query_file), args.output_quantizer)
    f = open(quantizer_path, 'rb')
    pqtype = unpack('B', f.read(1))[0]
    rectype = unpack('B', f.read(1))[0]

    d0 = unpack('i', f.read(4))[0]
    d1 = unpack('i', f.read(4))[0]
    d2 = unpack('i', f.read(4))[0]

    codebooks = np.frombuffer(f.read(d0*d1*d2*4), dtype=np.float32).reshape((d0,d1,d2))
    if pqtype == 2:
        rotate_matrix = np.frombuffer(f.read(data.shape[1]*data.shape[1]*4), dtype = np.float32).reshape((data.shape[1], data.shape[1]))
        rotate = np.transpose(rotate_matrix.copy())
        print (rotate_matrix.shape)
    f.close()

    with open(os.path.join(output_dir, args.output_quantizer + '.' + str(args.task)), 'wb') as f:
        f.write(pack('B', pqtype))
        f.write(pack('B', rectype))
        f.write(pack('i', codebooks.shape[0]))
        f.write(pack('i', codebooks.shape[1]))
        f.write(pack('i', codebooks.shape[2]))
        f.write(codebooks.tobytes())
        if pqtype == 2: f.write(rotate_matrix.tobytes())
        f.close()

    if len(args.output_quan_vector_file) == 0 and len(args.output_rec_vector_file) == 0:
        os.rename(args.output_truth, os.path.join(output_dir, 'truth.txt' + '.' + str(args.task)))
        ret = subprocess.run(['ZipKDTree.exe', output_dir, args.output_truth])
        print (ret)
        return

    if len(args.output_quan_vector_file) > 0:
        fquan = open(os.path.join(output_dir, args.output_quan_vector_file + '.' + str(args.task) + '.tmp'), 'wb')
        fquan.write(pack('i', 0))
        fquan.write(pack('i', args.quan_dim))

    if len(args.output_rec_vector_file) > 0:
        frec = open(os.path.join(output_dir, args.output_rec_vector_file + '.' + str(args.task) + '.tmp'), 'wb')
        frec.write(pack('i', 0))
        frec.write(pack('i', data.shape[1]))

    def fourcc(x):
        h = np.uint32(0)
        h = h | ord(x[0]) | ord(x[1]) << 8 | ord(x[2]) << 16 | ord(x[3]) << 24
        return h

    with open('tmp_faiss_index', 'wb') as f:
        h = fourcc('IxPq')
        d = np.uint64(data.shape[1])
        M = np.uint64(codebooks.shape[0])
        nbits = np.uint64(math.log2(codebooks.shape[1]))
        codesize = np.uint64(d0 * d1 * d2)
        totalitems = np.uint64(0)
        print ('h:%u d:%u M:%u nbits:%u codesize:%u lencode:%u' % (h, d, M, nbits, codesize, len(codebooks.tobytes())))
        f.write(pack('I', h))
        f.write(pack('i', d))
        f.write(pack('q', np.int64(0)))
        dummy = np.int64(1048576)
        f.write(pack('q', dummy))
        f.write(pack('q', dummy))
        f.write(pack('B', np.int8(1))) # is_trained
        f.write(pack('i', np.int32(1))) # metric_type
        f.write(pack('Q', d)) # size_t
        f.write(pack('Q', M)) # size_t
        f.write(pack('Q', nbits)) # size_t
        f.write(pack('Q', codesize))
        f.write(codebooks.tobytes())
        f.write(pack('Q', totalitems)) # size_t

        f.write(pack('i', np.int32(0))) # search_type
        f.write(pack('B', np.int8(0)))
        f.write(pack('i', np.int32(nbits * M + 1)))
        f.close()

    ivf_index =  faiss.read_index('tmp_faiss_index')
    if not ivf_index:
        print ('Error: faiss index cannot be loaded!')
        exit (1)
    print ('ksubs:%d dsub:%d code_size:%d nbits:%d M:%d d:%d polysemous_ht:%d' % (ivf_index.pq.ksub, ivf_index.pq.dsub, ivf_index.pq.code_size, ivf_index.pq.nbits, ivf_index.pq.M, ivf_index.pq.d, ivf_index.polysemous_ht))

    writeitems = 0
    while numData > 0:
        print (data[0])
        if pqtype == 2:
            data = np.matmul(data, rotate_matrix)
            print ('rotate:')
            print (data[0])

        codes = ivf_index.pq.compute_codes(data)
        print ('encode:')
        print (codes[0])

        if len(args.output_quan_vector_file) > 0:
            fquan.write(codes.tobytes())

        if len(args.output_rec_vector_file) > 0:
            recY = ivf_index.pq.decode(codes)
            print ('decode:')
            print (recY[0])
            if pqtype == 2:
                recY = np.matmul(recY, rotate).astype(args.data_type)
                print ('rotateback:')
                print (recY[0])
            frec.write(recY.tobytes())

        writeitems += numData
        numData, data = datareader.readbatch()

    datareader.close()

    if len(args.output_quan_vector_file) > 0:
        p = fquan.tell()
        fquan.seek(0)
        fquan.write(pack('i', writeitems))
        fquan.seek(p)
        fquan.close()
        if os.path.exists(os.path.join(output_dir, args.output_quan_vector_file + '.' + str(args.task))):
            os.remove(os.path.join(output_dir, args.output_quan_vector_file + '.' + str(args.task)))
        os.rename(os.path.join(output_dir, args.output_quan_vector_file + '.' + str(args.task) + '.tmp'), os.path.join(output_dir, args.output_quan_vector_file + '.' + str(args.task)))

    if len(args.output_rec_vector_file) > 0:
        p = frec.tell()
        frec.seek(0)
        frec.write(pack('i', writeitems))
        frec.seek(p)
        frec.close()
        if os.path.exists(os.path.join(output_dir, args.output_rec_vector_file + '.' + str(args.task))):
            os.remove(os.path.join(output_dir, args.output_rec_vector_file + '.' + str(args.task)))
        os.rename(os.path.join(output_dir, args.output_rec_vector_file + '.' + str(args.task) + '.tmp'), os.path.join(output_dir, args.output_rec_vector_file + '.' + str(args.task)))

    if os.path.exists(os.path.join(output_dir, 'truth.txt' + '.' + str(args.task))):
        os.remove(os.path.join(output_dir, 'truth.txt' + '.' + str(args.task)))
    os.rename(args.output_truth, os.path.join(output_dir, 'truth.txt' + '.' + str(args.task)))

if __name__ == '__main__':
    args = get_config()
    print ('log_dir:%s' % args.log_dir)
    print ('output_dir:%s' % args.output_dir)

    if not os.path.exists(args.output_dir): os.mkdir(args.output_dir)


    #gpusearch(args)

    if args.quan_type != 'none':
        if args.quan_type == 'pq':
            train_pq(args)
        elif args.quan_type == 'opq':
            train_opq(args)
        elif args.quan_type == 'rabitq':
            train_rabitq(args)
        elif args.quan_type == 'quan_reconstruct':
            quan_reconstruct_vectors(args)

    if args.log_dir != '':
        localpath = args.output_truth + '.dist//dist.bin.' + str(args.task)
        # upload to cloud storage
