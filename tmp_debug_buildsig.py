import numpy as np
from sptag import SPTAG

DATA_DIR = '/tmp/test_data_10k'
IDX_DIR = '/tmp/sptag_inline_tag_10k'
DIM = 128

t0_tags = np.load(f'{DATA_DIR}/t0_tags.npy')

mgr = SPTAG.CreateTenantIndexManager(DIM, 'SPANN', 'Float')
mgr.LoadAll(IDX_DIR)

s2i = {}
with open(f'{IDX_DIR}/manifest.txt') as f:
    for line in f:
        parts = line.strip().split()
        if parts and parts[0] == 'tenant_mapping':
            s2i[parts[2]] = int(parts[1])

iid = s2i['0']
print('before BuildSignatures', flush=True)
mgr.BuildSignatures(iid, t0_tags.tobytes(), len(t0_tags), 4)
print('after BuildSignatures', flush=True)
