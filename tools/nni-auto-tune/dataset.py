import h5py
import numpy as np
import os
import random


def sift():
    hdf5_f = h5py.File('sift-128-euclidean.hdf5', 'r')
    dimension = int(
        hdf5_f.attrs['dimension']) if 'dimension' in hdf5_f.attrs else len(
            hdf5_f['train'][0])
    return hdf5_f, dimension


def glove100():
    hdf5_f = h5py.File('glove-100-angular.hdf5', 'r')

    dimension = int(
        hdf5_f.attrs['dimension']) if 'dimension' in hdf5_f.attrs else len(
            hdf5_f['train'][0])

    return hdf5_f, dimension


def sparse_to_lists(data, lengths):
    X = []
    index = 0
    for l in lengths:
        X.append(data[index:index + l])
        index += l
    return X


def dataset_transform(dataset):
    if dataset.attrs.get('type', 'dense') != 'sparse':
        return np.array(dataset['train']), np.array(dataset['test'])
    return sparse_to_lists(dataset['train'],
                           dataset['size_train']), sparse_to_lists(
                               dataset['test'], dataset['size_test'])


DATASETS = {
    'glove-100-angular': glove100,
    'sift-128-euclidean': sift,
}
