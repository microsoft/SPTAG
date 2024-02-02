import h5py
import numpy as np
import os
import random
from struct import pack, unpack, calcsize
import math
import argparse
import copy


class DataReader:

    def __init__(self,
                 filename,
                 featuredim,
                 batchsize,
                 datatype='float32',
                 normalize=False,
                 targettype='float32'):
        self.mytype = targettype
        if filename.find('.bin') >= 0:
            self.fin = open(filename, 'rb')
            R = unpack('i', self.fin.read(4))[0]
            self.featuredim = unpack('i', self.fin.read(4))[0]
            self.isbinary = True
            self.type = datatype
            print('Open Binary DataReader for data(%d,%d)...' %
                  (R, self.featuredim))
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
                vec = self.fin.read(
                    (np.dtype(self.type).itemsize) * self.featuredim)
                if len(vec) == 0: break
                if len(vec) != (np.dtype(
                        self.type).itemsize) * self.featuredim:
                    print(
                        "%d vector cannot be read correctly: require %d bytes but only read %d bytes"
                        % (i, (np.dtype(self.type).itemsize) * self.featuredim,
                           len(vec)))
                    continue
                self.query[i] = np.frombuffer(vec, dtype=self.type).astype(
                    self.mytype)
                i += 1
        else:
            while i < numQuerys:
                line = self.fin.readline()
                if len(line) == 0: break

                index = line.rfind("\t")
                if index < 0: continue

                items = line[index + 1:].split("|")
                if len(items) < self.featuredim: continue

                for j in range(self.featuredim):
                    self.query[i, j] = float(items[j])
                i += 1
        print('Load batch query size:%r' % (i))
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
        return R, np.array(data)

    def close(self):
        self.fin.close()


def dataset_transform(dataset):
    if dataset.attrs.get('type', 'dense') != 'sparse':
        return np.array(dataset['train']), np.array(dataset['test'])
    return sparse_to_lists(dataset['train'],
                           dataset['size_train']), sparse_to_lists(
                               dataset['test'], dataset['size_test'])


class HDF5Reader:

    def __init__(self, filename, data_type='float32'):
        self.data = h5py.File(filename, 'r')
        self._data_type = data_type
        self.featuredim = int(self.data.attrs['dimension']
                              ) if 'dimension' in self.data.attrs else len(
                                  self.data['train'][0])
        self.train, self.test = dataset_transform(self.data)
        self.distance = self.data.attrs['distance']
        self.label = np.array(self.data['distances'])

    def norm(self, data):
        square = np.sqrt(np.sum(np.square(data), axis=1))
        data[square < 1e-6] = 1e-6 / math.sqrt(float(self.featuredim))
        square[square < 1e-6] = 1e-6
        data = data / square.reshape([-1, 1])
        return data

    def readallbatches(self):
        return np.array(self.train,
                        dtype=self._data_type), np.array(self.test,
                                                         dtype=self._data_type)

    def close(self):
        pass


def sparse_to_lists(data, lengths):
    X = []
    index = 0
    for l in lengths:
        X.append(data[index:index + l])
        index += l
    return X
