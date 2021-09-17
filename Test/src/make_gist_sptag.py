import sys
import numpy as np
import faiss

def DEFAULT_read(fname, np_type):
    buf = np.fromfile(fname, dtype='int32')
    n = buf[0]
    d = buf[1]
    print(buf.shape)
    print(n)
    print(d)
    #assert((n*d) % (buf.shape[0] - 2) == 0)
    return buf[2:].view(np_type).reshape(n, d).copy()

def DEFAULT_write(fname, m):
    with open(fname, 'wb') as f:
        np.array(m.shape[0], dtype=np.int32).tofile(f)
        np.array(m.shape[1], dtype=np.int32).tofile(f)
        m.tofile(f)

def codebooks_write(fname, m):
    with open(fname, 'wb') as f:
        np.array(m.shape[0], dtype=np.int32).tofile(f)
        np.array(m.shape[1], dtype=np.int32).tofile(f)
        np.array(m.shape[2], dtype=np.int32).tofile(f)
        m.tofile(f)

def ivecs_write(fname, m):
    n, d = m.shape
    m1 = np.empty((n, d + 1), dtype='int32')
    m1[:, 0] = d
    m1[:, 1:] = m
    m1.tofile(fname)


def fvecs_write(fname, m):
    m = m.astype('float32')
    ivecs_write(fname, m.view('int32'))

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def sanitize(x):
    return np.ascontiguousarray(x, dtype='float32')

def main():
    xd = fvecs_read("D:\data\gist\gist_base.fvecs")
    xq = fvecs_read("D:\data\gist\gist_query.fvecs")
    xt = fvecs_read("D:\data\gist\gist_learn.fvecs")


    index = faiss.IndexFlatL2(xd.shape[1])
    index.add(sanitize(xd))

    pqidx = faiss.IndexPQ(xd.shape[1], int(xd.shape[1] / 2), 8)
    pqidx.train(sanitize(xt))
    centroids = faiss.vector_to_array(pqidx.pq.centroids)
    print(centroids.shape)
    centroids = centroids.reshape(int(xd.shape[1] / 2), 256, 2)

    DEFAULT_write("gist_vector.bin", xd)
    DEFAULT_write("gist_query.bin", xq)
    codebooks_write("gist_codebook.bin", centroids)

main()