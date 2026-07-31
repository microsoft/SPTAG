#!/usr/bin/env python3
# Compare single-node baseline SearchResult dump vs concatenated distributed per-node dumps.
# Format: [int64 numQueries][int32 topK][numQueries * topK * (int64 VID + float Dist)]
import struct, sys

def load(path):
    with open(path, 'rb') as f:
        data = f.read()
    nq = struct.unpack_from('<q', data, 0)[0]
    tk = struct.unpack_from('<i', data, 8)[0]
    off = 12
    rows = []
    for q in range(nq):
        ids = []
        for k in range(tk):
            vid = struct.unpack_from('<q', data, off)[0]; off += 8
            dist = struct.unpack_from('<f', data, off)[0]; off += 4
            ids.append((vid, dist))
        rows.append(ids)
    return nq, tk, rows

def main():
    baseline = sys.argv[1]
    node_files = sys.argv[2:]
    bnq, btk, brows = load(baseline)
    print(f"baseline {baseline}: nq={bnq} topK={btk}")
    drows = []
    for nf in node_files:
        nq, tk, rows = load(nf)
        print(f"  {nf}: nq={nq} topK={tk}")
        drows.extend(rows)
    print(f"distributed concat: nq={len(drows)} topK={btk}")
    n = min(len(brows), len(drows))
    if len(brows) != len(drows):
        print(f"WARN: length mismatch baseline={len(brows)} dist={len(drows)}; comparing first {n}")

    exact_id = 0          # identical id set AND order
    setmatch = 0          # same id set (order-insensitive)
    total_id_overlap = 0  # sum of per-query top-K id intersection
    total_slots = 0
    per_query_diffs = []
    for i in range(n):
        b = brows[i]; d = drows[i]
        bids = [x[0] for x in b]
        dids = [x[0] for x in d]
        if bids == dids:
            exact_id += 1
        bs, ds = set(bids), set(dids)
        if bs == ds:
            setmatch += 1
        overlap = len(bs & ds)
        total_id_overlap += overlap
        total_slots += len(bids)
        if bs != ds:
            per_query_diffs.append((i, bids, dids))

    print("\n=== Cross-validation result ===")
    print(f"queries compared      : {n}")
    print(f"exact match (id+order): {exact_id}/{n} ({100.0*exact_id/n:.2f}%)")
    print(f"set match  (id unord) : {setmatch}/{n} ({100.0*setmatch/n:.2f}%)")
    print(f"top-K id overlap      : {total_id_overlap}/{total_slots} ({100.0*total_id_overlap/total_slots:.3f}%)")
    print(f"queries with any diff : {len(per_query_diffs)}")
    for i, bids, dids in per_query_diffs[:10]:
        print(f"  q{i}: baseline={bids} dist={dids}")

if __name__ == '__main__':
    main()
