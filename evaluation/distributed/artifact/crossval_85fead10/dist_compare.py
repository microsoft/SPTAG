#!/usr/bin/env python3
# Distance-level cross-validation: sorted top-K distances baseline vs concat(node0,node1).
import struct, sys

def load(p):
    d = open(p, 'rb').read()
    nq = struct.unpack_from('<q', d, 0)[0]
    tk = struct.unpack_from('<i', d, 8)[0]
    off = 12
    rows = []
    for q in range(nq):
        r = []
        for k in range(tk):
            vid = struct.unpack_from('<q', d, off)[0]; off += 8
            dist = struct.unpack_from('<f', d, off)[0]; off += 4
            r.append((vid, dist))
        rows.append(r)
    return rows

b = load(sys.argv[1])
d = load(sys.argv[2]) + load(sys.argv[3])
n = min(len(b), len(d))
exact = close = 0
sd = mx = 0.0
slots = 0
for i in range(n):
    bd = [x[1] for x in b[i]]
    dd = [x[1] for x in d[i]]
    if bd == dd:
        exact += 1
    ok = True
    for x, y in zip(bd, dd):
        dev = abs(x - y); mx = max(mx, dev); sd += dev; slots += 1
        if dev > 1e-3 * max(1.0, abs(x)):
            ok = False
    if ok:
        close += 1
print(f"exact distance-vector match : {exact}/{n} ({100.0*exact/n:.1f}%)")
print(f"match within 0.1%           : {close}/{n}")
print(f"mean abs distance deviation : {sd/slots:.4f}")
print(f"max abs distance deviation  : {mx:.4f}")
