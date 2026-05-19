import numpy as np
import math
import time
from struct import pack, unpack, calcsize
from struct import pack, unpack, calcsize
from typing import Dict, List
import argparse
import copy
from operator import itemgetter
import os
import subprocess

def get_config():
    parser = argparse.ArgumentParser(description ='implementation of split and convert.')
    parser.add_argument('--data_file', default = 'vectors.bin', type = str, help = 'data file')
    parser.add_argument('--meta_file', default = 'meta.bin', type= str, help='meta file')
    parser.add_argument('--metaidx_file', default = 'metaidx.bin', type= str, help='meta index file') 
    parser.add_argument('--data_type', default = 'uint8', type = str, help = 'data type for binary file: float32, int8, int16')
    parser.add_argument('--vid_intype', default = 'int32', type = str, help = 'VID input data type')
    parser.add_argument('--batch_size', default = 1000000, type = int, help = 'batch size for loading data')
    parser.add_argument('--partitions', default = 3, type = int, help = 'number of data partitions')
    parser.add_argument('--vid_outtype', default = 'int64', type = str, help = 'VID output data type')
    parser.add_argument('--output_dir', type = str, default = '.', help='output dir')
    args = parser.parse_args()
    return args

def run(args):
    if args.vid_intype == 'int32':
        inpacktype = 'i'
        inpacksize = 4
    elif args.vid_intype == 'int64':
        inpacktype = 'q'
        inpacksize = 8
    if args.vid_outtype == 'int32':
        outpacktype = 'i'
        outpacksize = 4
    elif args.vid_outtype == 'int64':
        outpacktype = 'q'
        outpacksize = 8
    
    fin = open(args.data_file, 'rb')
    R = unpack(inpacktype, fin.read(inpacksize))[0]
    C = unpack('i', fin.read(4))[0]
    partition_size = (R + args.partitions - 1) // args.partitions
    print (f'R:{R}, C:{C}, partition_size:{partition_size}')
    fmetain = open(args.meta_file, 'rb')
    fmetaidxin = open(args.metaidx_file, 'rb')
    M = unpack(inpacktype, fmetaidxin.read(inpacksize))[0]
    if M != R:
        print (f'Meta index file record number {M} does not match data file record number {R}')
        exit (1)
    offset = np.frombuffer(fmetaidxin.read((R + 1) * np.dtype('uint64').itemsize), dtype=np.uint64)

    VC = 1
    for i in range(args.partitions):
        fout = open(os.path.join(args.output_dir, f'vectors.bin.{i}'), 'wb')
        fmetaout = open(os.path.join(args.output_dir, f'meta.bin.{i}'), 'wb')
        fmetaidxout = open(os.path.join(args.output_dir, f'metaidx.bin.{i}'), 'wb')
        fvidout = open(os.path.join(args.output_dir, f'vid.bin.{i}'), 'wb')

        global_start = i * partition_size
        global_offset_start = offset[global_start]

        part_size = min(partition_size, R - i * partition_size)
        part_size = np.int64(part_size).astype(args.vid_outtype)
        print (f'Partition {i}, start: {global_start}, size: {part_size}')

        fout.write(pack(outpacktype, part_size))
        fout.write(pack('i', C))
        fvidout.write(pack(outpacktype, part_size))
        fvidout.write(pack('i', VC))

        fmetaidxout.write(pack(outpacktype, part_size))
        part_offset = offset[global_start: global_start + part_size + 1] - global_offset_start
        fmetaidxout.write(part_offset.tobytes())

        start = 0
        while start < partition_size:
            readsize = min(args.batch_size, partition_size - start)
            fout.write(fin.read(readsize * C * np.dtype(args.data_type).itemsize))
            fmetaout.write(fmetain.read(part_offset[start + readsize] - part_offset[start]))
            vids = np.zeros((readsize,), dtype=args.vid_outtype)
            for j in range(readsize): vids[j] = global_start + start + j
            fvidout.write(vids.tobytes())
            start += readsize
        fout.close()
        fmetaout.close()
        fmetaidxout.close()
        fvidout.close()
    fin.close()
    fmetain.close()
    fmetaidxin.close()      

if __name__ == '__main__':
    args = get_config()
    print ('output_dir:%s' % args.output_dir)
    run(args)
