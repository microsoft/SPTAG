import numpy as np
from struct import pack, unpack, calcsize
from struct import pack, unpack, calcsize
import heapq
import argparse
import copy
from operator import itemgetter
import os
import subprocess

def get_config():
    parser = argparse.ArgumentParser(description ='implementation of nnsearch.')
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
    args = parser.parse_args()
    return args

class DataReader:
    def __init__(self, filename, featuredim, batchsize, normalize, datatype, targettype='float32'):
        self.mytype = targettype
        if filename.find('.bin') >= 0:
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

        if batchsize <= 0: batchsize = R
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

def train_pq(args):
    import faiss
    from LibVQ.base_index import FaissIndex

    output_dir = args.output_dir

    datareader = DataReader(args.data_file, args.dim, args.train_samples, args.data_normalize, args.data_type, args.target_type)
    
    print ('train PQ...')
    
    index_method = 'pq'
    ivf_centers_num = -1
    subvector_num = args.quan_dim
    subvector_bits = 8
    numData, data = datareader.readbatch()
    
    faiss.omp_set_num_threads(args.T)
    index = FaissIndex(index_method=index_method,
                       emb_size=len(data[0]),
                       ivf_centers_num=ivf_centers_num,
                       subvector_num=subvector_num,
                       subvector_bits=subvector_bits,
                       dist_mode='l2')

    print('Training the index with doc embeddings')

    index.fit(data)
    
    rtype = np.uint8(0)
    if args.data_type == 'uint8':
        rtype = np.uint8(1)
    elif args.data_type == 'int16':
        rtype = np.uint8(2)
    elif args.data_type == 'float32':
        rtype = np.uint8(3)
     
    faiss_index = index.index
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

    writeitems = 0
    while numData > 0:
        if args.quan_test > 0: index.add(data)
        
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
    ret = subprocess.run(['ZipKDTree.exe', output_dir, args.output_truth])
    print (ret)

    if args.quan_test > 0:
        queryreader = DataReader(args.query_file, args.dim, -1, args.query_normalize, args.data_type, args.target_type)
        numQuery, query = queryreader.readbatch()

        qid2ground_truths = {}
        f = open(os.path.join(output_dir, 'truth.txt.' + str(args.task)), 'r')
        for i in range(numQuery):
            items = f.readline()[0:-1].strip().split(' ')
            qid2ground_truths[i] = set([int(gt) for gt in items])
        f.close()

        # Test the performance
        index.test(query, qid2ground_truths, topk=args.k, batch_size=64,
                   MRR_cutoffs=[5, 10], Recall_cutoffs=[5, 10, 20, 40])

def train_opq(args):
    import faiss
    from LibVQ.base_index import FaissIndex

    output_dir = args.output_dir

    datareader = DataReader(args.data_file, args.dim, args.train_samples, args.data_normalize, args.data_type, args.target_type)
    
    print ('train OPQ...')
    
    index_method = 'opq'
    ivf_centers_num = -1
    subvector_num = args.quan_dim
    subvector_bits = 8
    numData, data = datareader.readbatch()
    
    faiss.omp_set_num_threads(args.T)
    index = FaissIndex(index_method=index_method,
                       emb_size=len(data[0]),
                       ivf_centers_num=ivf_centers_num,
                       subvector_num=subvector_num,
                       subvector_bits=subvector_bits,
                       dist_mode='l2')

    print('Training the index with doc embeddings')

    index.fit(data)
    
    rtype = np.uint8(0)
    if args.data_type == 'uint8':
        rtype = np.uint8(1)
    elif args.data_type == 'int16':
        rtype = np.uint8(2)
    elif args.data_type == 'float32':
        rtype = np.uint8(3)
     
    faiss_index = index.index
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

    writeitems = 0
    while numData > 0:
        if args.quan_test > 0: index.add(data)
        
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
    ret = subprocess.run(['ZipKDTree.exe', output_dir, args.output_truth])
    print (ret)

    if args.quan_test > 0:
        queryreader = DataReader(args.query_file, args.dim, -1, args.query_normalize, args.data_type, args.target_type)
        numQuery, query = queryreader.readbatch()

        qid2ground_truths = {}
        f = open(os.path.join(output_dir, 'truth.txt.' + str(args.task)), 'r')
        for i in range(numQuery):
            items = f.readline()[0:-1].strip().split(' ')
            qid2ground_truths[i] = set([int(gt) for gt in items])
        f.close()

        # Test the performance
        index.test(query, qid2ground_truths, topk=args.k, batch_size=64,
                   MRR_cutoffs=[5, 10], Recall_cutoffs=[5, 10, 20, 40])

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
    ret = subprocess.run(['ZipKDTree.exe', output_dir, args.output_truth])
    print (ret)

if __name__ == '__main__':
    args = get_config()
    print ('log_dir:%s' % args.log_dir)
    print ('output_dir:%s' % args.output_dir)

    if not os.path.exists(args.output_dir): os.mkdir(args.output_dir)

    if args.data_format != 'DEFAULT':
        target = 'PreprocessData.exe' if args.data_format == 'BOND' else 'ProcessData.exe'
        casttype = 'BYTE'
        if args.data_type == 'uint8': casttype = 'UBYTE'
        elif args.data_type == 'int16': casttype = 'SHORT'
        elif args.data_type == 'float32': casttype = 'FLOAT'
        ret = subprocess.run([target, args.data_file, os.path.join(args.output_dir, 'vectors.bin.%d' % args.task), os.path.join(args.output_dir, 'meta.bin.%d' % args.task), os.path.join(args.output_dir, 'metaindex.bin.%d' % args.task), str(args.dim), casttype, '0'])
        args.data_file = os.path.join(args.output_dir, 'vectors.bin.%d' % args.task)
        print(ret)
        
    if args.query_file[-4:] != '.bin':
        casttype = 'BYTE'
        if args.data_type == 'uint8': casttype = 'UBYTE'
        elif args.data_type == 'int16': casttype = 'SHORT'
        elif args.data_type == 'float32': casttype = 'FLOAT'
        ret = subprocess.run(['SearchPreprocess.exe', '-q', args.query_file, '-o', args.query_file + '_queryVector.bin', '-v', args.query_file + '_validQuery.bin', '-d', str(args.dim), '-t', casttype, '-n', '0'])
        args.query_file = args.query_file + '_queryVector.bin'
        print(ret)

    gpusearch(args)
    
    if args.quan_type != 'none':
        if args.quan_type == 'pq':
            train_pq(args)
        elif args.quan_type == 'opq':
            train_opq(args)
        elif args.quan_type == 'quan_reconstruct':
            quan_reconstruct_vectors(args)

    if args.log_dir != '':
        localpath = args.output_truth + '.dist\\dist.bin.' + str(args.task)
        ret = subprocess.run(['CosmosFolderTransfer.exe', 'uploadStream', localpath.replace('\\', '/'), args.log_dir + '/dist/', 'ap'])
        print (ret)
