import numpy as np
import struct
import sys
import os


def copy_head(inputfolder, outputfolder, hostfile, packtype):
    if os.path.exists(outputfolder):
        for f in os.listdir(outputfolder):
            os.remove(os.path.join(outputfolder, f))
    else:
        os.makedirs(outputfolder)
    f = open(hostfile, 'r')
    hs = []
    for line in f:
        hs.append(line.split()[0])
    f.close()
    print (hs)
    if packtype == 'i':
        packsize = 4
        nptype = np.int32
    else:
        packsize = 8
        nptype = np.int64
    f = open(os.path.join(outputfolder, 'vectors.bin'), 'wb')
    fid = open(os.path.join(outputfolder, 'SPTAGHeadVectorIDs.bin'), 'wb')
    f.write(struct.pack(packtype, 0))
    f.write(struct.pack('i', 0))
    fid.write(struct.pack(packtype, 0))
    fid.write(struct.pack('i', 1))
    unique_vids = {}
    cols = 0
    total = nptype(0)
    for i, h in enumerate(hs):
        ret = subprocess.run(["scp", h + ":" + os.path.join(os.path.join(inputfolder, "HeadIndex"), "vectors.bin"), os.path.join(outputfolder, "vectors.bin." + str(i))])
        print (f"Copying {"vectors.bin"} from {h}:{os.path.join(inputfolder, "HeadIndex")} to {outputfolder}")
        if ret.returncode != 0:
            print(f"Error: Failed to copy {h}:{os.path.join(os.path.join(inputfolder, "HeadIndex"), "vectors.bin")}")
            exit(1)
        ret = subprocess.run(["scp", h + ":" + os.path.join(os.path.join(inputfolder, "HeadIndex"), "SPTAGHeadVectorIDs.bin"), os.path.join(outputfolder, "SPTAGHeadVectorIDs.bin." + str(i))])
        print (f"Copying {"SPTAGHeadVectorIDs.bin"} from {h}:{os.path.join(inputfolder, "HeadIndex")} to {outputfolder}")
        if ret.returncode != 0:
            print(f"Error: Failed to copy {h}:{os.path.join(os.path.join(inputfolder, "HeadIndex"), "SPTAGHeadVectorIDs.bin")}")
            exit(1)

        if len(unique_vids) == 0:
            fin = open(os.path.join(outputfolder, "vectors.bin." + str(i)), "rb")
            r = struct.unpack(packtype, fin.read(packsize))[0]
            c = struct.unpack('i', fin.read(4))[0]
            cols = c
            f.write(fin.read())
            fin.close()

            fidin = open(os.path.join(outputfolder, "SPTAGHeadVectorIDs.bin." + str(i)), "rb")
            idr = struct.unpack(packtype, fidin.read(packsize))[0]
            idc = struct.unpack('i', fidin.read(4))[0]
            if idr != r:
                print(f"Error: Mismatched vector IDs between {"vectors.bin." + str(i)} and {"SPTAGHeadVectorIDs.bin." + str(i)}")
                exit(1)
            vids = np.frombuffer(fidin.read(idr * packsize), dtype=nptype)
            fid.write(vids.tobytes())
            fidin.close()
            for i in range(idr):
                unique_vids.add(vids[i])
            total += r
        else:
            vecsize = os.path.getsize(os.path.join(outputfolder, 'vectors.bin.' + str(i))) - 4 - packsize
            fin = open(os.path.join(outputfolder, "vectors.bin." + str(i)), "rb")
            r = struct.unpack(packtype, fin.read(packsize))[0]
            c = struct.unpack('i', fin.read(4))[0]
            if c != cols:
                print(f"Error: Mismatched column count between {"vectors.bin." + str(i)} and previous files")
                exit(1)
            vecsize = int(vecsize // r)
            fidin = open(os.path.join(outputfolder, "SPTAGHeadVectorIDs.bin." + str(i)), "rb")
            idr = struct.unpack(packtype, fidin.read(packsize))[0]
            idc = struct.unpack('i', fidin.read(4))[0]
            if idr != r:
                print(f"Error: Mismatched vector IDs between {"vectors.bin." + str(i)} and {"SPTAGHeadVectorIDs.bin." + str(i)}")
                exit(1)
            vids = np.frombuffer(fidin.read(idr * packsize), dtype=nptype)
            for i in range(idr):
                if vids[i] in unique_vids:
                    fin.read(vecsize)
                else:
                    f.write(fin.read(vecsize))
                    fid.write(vids[i].tobytes())
                    total += 1
                    unique_vids.add(vids[i])
            fin.close()
            fidin.close()
        os.remove(os.path.join(outputfolder, 'vectors.bin.' + str(i)))
        os.remove(os.path.join(outputfolder, 'SPTAGHeadVectorIDs.bin.' + str(i)))
    f.seek(0)
    f.write(struct.pack(packtype, total))
    f.write(struct.pack('i', cols))
    f.close()
    fid.seek(0)
    fid.write(struct.pack(packtype, total))
    fid.close()

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python3 merge_head.py <inputfolder> <outputfolder> <hosts> <packtype>")
        exit(1)
    
    inputfolder = sys.argv[1]
    outputfolder = sys.argv[2]
    hostfile = sys.argv[3]
    packtype = sys.argv[4]
    copy_head(inputfolder, outputfolder, hostfile, packtype)

