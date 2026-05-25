from mpi4py import MPI
import numpy as np
import os, sys
import subprocess
import struct

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def pack(inputfolder, clusternum):
    vfiles = [f for f in os.listdir(inputfolder) if f.startswith('vectors.bin')]
    count = np.zeros(clusternum, dtype=np.uint64)
    for f in vfiles:
        fid = f[f.rfind('.') + 1:]
        count[int(fid)] = os.path.getsize(os.path.join(inputfolder, f))
    print (f'localcount: {count}')
    totalcount = np.zeros(clusternum, dtype=np.uint64)
    comm.Reduce([count, clusternum, MPI.UNSIGNED_LONG_LONG], [totalcount, clusternum, MPI.UNSIGNED_LONG_LONG], op=MPI.SUM, root=0)
    if rank == 0:        
        print (f'totalcount: {totalcount}')
        allcount = np.sum(totalcount)
        perPartition = (allcount + size - 1) // size
        print (f"Total size: {allcount} perPartition: {perPartition}")
        x = np.argsort(totalcount)
        print (f"Sorted indices: {x}")
        assign = [0] * size
        partitions = [[] for _ in range(size)]
        for j in range(clusternum -1, -1, -1):
            for current in range(0, size):
                if assign[current] > perPartition: continue

                if assign[current] + totalcount[x[j]] > perPartition:
                    if len(partitions[current]) == 0:
                        partitions[current].append(x[j])
                        assign[current] += totalcount[x[j]]
                        break
                else:
                    partitions[current].append(x[j])
                    assign[current] += totalcount[x[j]]
                    break
        print (f'Partitions: {partitions}')
        revert = {}
        for i in range(size):
            for j in partitions[i]:
                revert[j] = i
    else:
        revert = {}
    revert = comm.bcast(revert, root=0)
    print (f'Revert: {revert}')
    return revert

def shuffle(inputfolder, outputfolder, revert, hosts):
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
    else:
        for f in os.listdir(outputfolder):
            os.remove(os.path.join(outputfolder, f))
    f = open(hosts, 'r')
    hs = []
    for line in f:
        hs.append(line.split()[0])
    f.close()
    print (hs)
    for f in os.listdir(inputfolder):
        h = hs[revert[int(f[f.rfind('.') + 1:])]]
        ret = subprocess.run(["scp", os.path.join(inputfolder, f), h + ":" + os.path.join(outputfolder, f)])
        print (f"Copying {f} to {h}:{outputfolder}")
        if ret.returncode != 0:
            print(f"Error: Failed to copy {f} to {h}:{outputfolder}")
            exit(1)
    return

def merge(inputfolder, taskid, outputfolder, packtype = 'i'):
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
    else:
        for f in os.listdir(outputfolder):
            os.remove(os.path.join(outputfolder, f))

    vfiles = [f for f in os.listdir(inputfolder) if f.startswith('vectors.bin')]
    mfiles = [f for f in os.listdir(inputfolder) if f.startswith('meta.bin')]
    vidfiles = [f for f in os.listdir(inputfolder) if f.startswith('vid.bin')]

    if packtype == 'i':
        total = np.int32(0)
        packsize = 4
    else:
        total = np.int64(0)
        packsize = 8
    col = int(0)
    offset = np.uint64(0)
    
    fvec = open(os.path.join(outputfolder, 'vectors.bin.' + taskid), 'wb')
    fvec.write(struct.pack(packtype, 0))
    fvec.write(struct.pack('i', 0))

    if len(mfiles) > 0:
        fmeta = open(os.path.join(outputfolder, 'meta.bin.' + taskid), 'wb')
        fmetaindex = open(os.path.join(outputfolder, 'metaindex.bin.' + taskid), 'wb')
        fmetaindex.write(struct.pack(packtype, 0))
        fmetaindex.write(offset.tobytes()) 
    
    if len(vidfiles) > 0:
        fvid = open(os.path.join(outputfolder, 'vid.bin.' + taskid), 'wb')
        fvid.write(struct.pack(packtype, 0))
        fvid.write(struct.pack('i', 1))

    for f in vfiles:
        fid = f[f.find('bin') + 4:]
        print ('%s id:%s' % (f, fid))
        vecsize = os.path.getsize(os.path.join(inputfolder, 'vectors.bin.' + fid)) - 4 - packsize
        vecin = open(os.path.join(inputfolder, 'vectors.bin.' + fid), 'rb')

        numv = struct.unpack(packtype, vecin.read(packsize))[0]
        colv = struct.unpack('i', vecin.read(4))[0]
        if col == 0: col = colv
        elif col != colv:
            print ("Error: %s col not match %d vs %d!" % (f, col, colv))
            exit(1)
        
        if vecsize < numv * colv:
            print ("Error: %s vector size %d is not match (%d, %d)" % (f, vecsize, numv, colv))
            exit(1)

        fvec.write(vecin.read(vecsize))
        vecin.close()
        #os.remove(os.path.join(inputfolder, 'vectors.bin.' + fid))
        total += numv
        
        if os.path.exists(os.path.join(inputfolder, 'meta.bin.' + fid)):        
            metasize = os.path.getsize(os.path.join(inputfolder, 'meta.bin.' + fid))
            metain = open(os.path.join(inputfolder, 'meta.bin.' + fid), 'rb')
            fmeta.write(metain.read(metasize))
            metain.close()
        
            metaindexin = open(os.path.join(inputfolder, 'metaindex.bin.' + fid), 'rb')
            numo = struct.unpack(packtype, metaindexin.read(packsize))[0]
            if numo != numv:
                print ("Error: numv:%d not match numo:%d in metaindex\n" % (numv, numo))
                exit(1)

            arro = np.frombuffer(metaindexin.read(8*(numo+1)), dtype=np.uint64)[1:]
            arro = arro + offset
            for i in range(numo):
                fmetaindex.write(arro[i].tobytes())
            metaindexin.close()
            if numo > 0: offset = arro[numo-1]
            
            #os.remove(os.path.join(inputfolder, 'meta.bin.' + fid))
            #os.remove(os.path.join(inputfolder, 'metaindex.bin.' + fid))   

        if os.path.exists(os.path.join(inputfolder, 'vid.bin.' + fid)):
            vidin = open(os.path.join(inputfolder, 'vid.bin.' + fid), 'rb')
            numo = struct.unpack(packtype, vidin.read(packsize))[0]
            if numo != numv:
                print ("Error: numv:%d not match numo:%d in vid\n" % (numv, numo))
                exit(1)
            _ = struct.unpack('i', vidin.read(4))[0]
            fvid.write(vidin.read(packsize*numo))
            vidin.close()
            #os.remove(os.path.join(inputfolder, 'vid.bin.' + fid))
    fvec.seek(0)
    fvec.write(struct.pack(packtype, total))
    fvec.write(struct.pack('i', col))
    fvec.close()
    
    if len(mfiles) > 0:
        fmeta.close()
    
        fmetaindex.seek(0)
        fmetaindex.write(struct.pack(packtype, total))
        fmetaindex.close()
    
    if len(vidfiles) > 0:
        fvid.seek(0)
        fvid.write(struct.pack(packtype, total))
        fvid.close()

    print ("Successfully merge vectors (%d,%d)" % (total, col)) 

def prepare(configure_file):
    if rank == 0:
        f = open(configure_file, 'r')
        s = f.read()
    else:
        s = ""
    s = comm.bcast(s, root=0)
    s = s.replace('*', str(rank))
    print (s)
    f = open(configure_file + '.ini', 'w')
    f.write(s)
    f.close()

if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("Usage: python3 shuffle_data.py <inputfolder> <outputfolder> <cluster number> <hosts> <packtype> <configure.ini>")
        exit(1)

    inputfolder = sys.argv[1]
    outputfolder = sys.argv[2]
    clusternumber = int(sys.argv[3])
    hostfile = sys.argv[4]
    packtype = sys.argv[5]
    configure_file = sys.argv[6]

    tmpfolder = inputfolder + "_tmp"

    
    partitions = pack(inputfolder, clusternumber)
    shuffle(inputfolder, tmpfolder, partitions, hostfile)
    comm.Barrier()
    merge(tmpfolder, str(rank), outputfolder, packtype)
    prepare(configure_file)
