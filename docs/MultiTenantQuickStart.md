# Multi-tenant SPANN quick start

Run a local server, create two isolated tenants, insert/search/delete vectors,
and reopen the persisted node. This small example needs no external dataset,
NumPy code or running TiKV cluster. It uses RocksDB postings by default.

Use a checkout containing the multi-tenant feature/PR, and run all commands from
the repository root. The example targets Ubuntu 24.04 x86-64 with an
AVX2-capable CPU. It binds only to localhost: tenant names are namespaces, not
authentication or authorization. The server uses SPTAG's socket protocol, not HTTP.

## 1. Build

If the build dependencies are not already installed:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build git swig python3-dev python3-numpy \
  libboost-system-dev libboost-thread-dev libboost-serialization-dev \
  libboost-regex-dev libboost-filesystem-dev libboost-test-dev \
  libtbb-dev libnuma-dev librocksdb-dev
```

Build the server, Python client and offline administration tool:

```bash
git submodule update --init ThirdParty/zstd
cmake -S . -B build-multitenant -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DFOLLY=OFF -DTBB=ON -DSPDK=OFF -DGPU=OFF -DRABITQ=OFF \
  -DMULTITENANT=ON -DTIKV=OFF
cmake --build build-multitenant --target server _SPTAGClient tenanttool --parallel 2
```

`MULTITENANT=ON` automatically enables the RocksDB build dependency, including
when postings use FILEIO or TiKV. It does not install RocksDB or start a service.
Binaries and `SPTAGClient.py` are written to `Release/`, not the build directory.
Different build configurations share this output directory; use separate
checkouts instead of building them concurrently in one source tree.

## 2. Start the server in terminal 1

This creates a unique, disposable demo directory and selects an available local
port without touching other services. The node root must initially be empty.

```bash
export SPTAG_DEMO="$(mktemp -d /tmp/sptag-multitenant.XXXXXX)"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
python3 - <<'PY'
import os
from pathlib import Path
import shlex
import socket

work = Path(os.environ["SPTAG_DEMO"])
assert not (work / "node").exists()
with socket.socket() as reserve:
    reserve.bind(("127.0.0.1", 0))
    port = reserve.getsockname()[1]
(work / "port").write_text(str(port))
(work / "server.ini").write_text(
    f"[Service]\nListenAddr=127.0.0.1\nListenPort={port}\n"
    "ThreadNumber=2\nSocketThreadNumber=1\n"
    f"[MultiTenant]\nRoot={work / 'node'}\nStorage=ROCKSDBIO\n"
    "MaxTenants=8\nMaxResidentTenants=2\n"
    "AutoCheckpoint=true\nCheckpointLogBytes=1073741824\n"
    "CheckpointPressureBytes=4294967296\nCheckpointMinIntervalSeconds=300\n"
)
print("Run these two commands in terminal 2:")
print("cd " + shlex.quote(str(Path.cwd())))
print("export SPTAG_DEMO=" + shlex.quote(str(work)))
print(f"Server endpoint: 127.0.0.1:{port}")
PY
Release/server -m socket -c "$SPTAG_DEMO/server.ini"
```

Leave this terminal running. If another process claims the port before startup,
repeat this step to get a new directory/port; do not stop an unrelated service.

## 3. Create, insert, search and delete from terminal 2

First run the `cd` and `export SPTAG_DEMO=...` commands printed by terminal 1.
Then copy this block. It waits for the client connection and checks every result.

```bash
export PYTHONPATH="$PWD/Release${PYTHONPATH:+:$PYTHONPATH}"
python3 - <<'PY'
import os
from pathlib import Path
import struct
import time
import SPTAGClient

work = Path(os.environ["SPTAG_DEMO"])
client = SPTAGClient.AnnClient("127.0.0.1", (work / "port").read_text().strip())
deadline = time.monotonic() + 15
while not client.IsConnected():
    if time.monotonic() >= deadline:
        raise RuntimeError("Server did not connect; check terminal 1 and server.ini")
    time.sleep(0.05)
client.SetTimeoutMilliseconds(30000)

def checked(operation, result):
    if result["status_name"] != "Success":
        raise RuntimeError(f"{operation}: {result}")
    print(operation, result)
    return result

def seed(base):
    values = [base + i * 0.1 + j * 0.01 for i in range(128) for j in range(4)]
    return struct.pack("<512f", *values)

# BKT/KDT select each tenant's SPANN head index.
checked("create alice", client.CreateTenant("alice", seed(0), 128, 4, "Float", "L2", "KDT"))
checked("create bob", client.CreateTenant("bob", seed(1000), 128, 4, "Float", "L2", "BKT"))
query = struct.pack("<4f", 500.0, 500.01, 500.02, 500.03)
inserted = checked("insert alice", client.InsertTenant("alice", query, 1, "Float"))
assert inserted["first_id"] == 128 and inserted["count"] == 1, inserted
alice = checked("search alice", client.SearchTenant("alice", query, 1, "Float"))
bob = checked("search bob", client.SearchTenant("bob", query, 1, "Float"))
assert alice["ids"] == [128] and alice["distances"] == [0.0], alice
assert bob["ids"] and 128 not in bob["ids"], bob
checked("delete alice vector 0", client.DeleteTenantVector("alice", 0))

# Only CLI files need the count/dimension header; RPC buffers contain packed values.
(work / "query.bin").write_bytes(struct.pack("<ii", 1, 4) + query)
print("PASS: two tenants, insert/search isolation, and vector deletion")
PY
```

Expected: all operations report `Success`, Alice's inserted vector has ID `128`
and distance `0`, Bob cannot see that vector, and the final `PASS` line appears.
Vector IDs are tenant-local, not globally unique. RPC `Float` buffers contain
packed float32 values without a file header; creation supplies count/dimension
separately.

Run this creation block only once per demo root. Repeating it against the same
root fails because the tenants already exist. Start a new demo for another
creation run; do not overwrite a node containing data you need.

## 4. Stop the server and verify persistence offline

Press **Ctrl+C in terminal 1** and wait for the server process to exit.
Do not run `tenanttool` against a live server: the node has a single process owner.
In terminal 2:

```bash
export SPTAG_POSTING_STORAGE=ROCKSDBIO
Release/tenanttool "$SPTAG_DEMO/node" list
Release/tenanttool "$SPTAG_DEMO/node" stats
Release/tenanttool "$SPTAG_DEMO/node" search alice "$SPTAG_DEMO/query.bin" 1
Release/tenanttool "$SPTAG_DEMO/node" checkpoint
```

The list contains `alice` and `bob`; the search returns ID `128` with distance
`0`. These separate CLI processes reopen the node, demonstrating that the
acknowledged insertion survived the server exit. The checkpoint command publishes
a node checkpoint; successful CLI commands also close/checkpoint their node.

The tiny demo does not reach the default 1 GiB automatic-checkpoint threshold.
The default policy is 1 GiB of retained logical recovery records, a 300-second
minimum interval, and a 4 GiB pressure trigger that bypasses the interval and holds
new insert/delete admission. These are not physical disk quotas.

To serve the existing tenants again, run the same command in terminal 1:

```bash
Release/server -m socket -c "$SPTAG_DEMO/server.ini"
```

Skip the creation calls when reopening. Keep the demo directory to inspect its
recovery/state, postings and snapshots. Removing that directory removes its data.

## Optional: FILEIO or TiKV postings

Choose the backend **before the first server start, using a fresh demo root**.
An existing root must reopen with its recorded backend; changing `Storage` is
not a migration.

| Backend | Changes to the generated `server.ini` | Additional requirements |
| --- | --- | --- |
| RocksDB | Keep `Storage=ROCKSDBIO` | No external service |
| FILEIO | Set `Storage=FILEIO` | Same multi-tenant build; local RocksDB still stores recovery/state |
| TiKV | Set `Storage=TIKVIO`, `TiKVPDAddresses=host:port`, `TiKVKeyPrefix=your-unique-owned-prefix` | Reconfigure/build with `TIKV=ON`, protobuf/gRPC/Abseil dependencies, and a running TiKV cluster |

The Python API is unchanged. For offline tools set `SPTAG_POSTING_STORAGE` to
the same backend. TiKV additionally needs matching `SPTAG_TIKV_PD` and
`SPTAG_TIKV_PREFIX`; never reuse another node's prefix. See the
[TiKV integration guide](TiKV-Integration.md) for cluster/dependency setup.
New TiKV tenants default to `BuildSSDIndex.DistributedVersionMap=true`. Their
per-layer counts and VID versions live in the same tenant/index/work-generation
namespace as their postings; the posting archive restores both into a fresh
generation. `DistributedVersionMap=false` uses local version files in the head
snapshot instead. Existing managed TiKV checkpoints retain their recorded
`false` mode; reopening does not migrate them. Do not edit the mode in a
checkpoint or mix head snapshots and posting archives from different checkpoints.
TiKV distributes storage, not tenant ownership or failover.

## Troubleshooting and further reading

- `import SPTAGClient` fails: build `_SPTAGClient` and set `PYTHONPATH` to this checkout's `Release/`.
- Connection times out: inspect terminal 1; verify both terminals use the same demo directory and the server was built with `MULTITENANT=ON`.
- Opening the node fails: stop its current owner and verify the backend configuration matches the recorded node.
- A call returns `FailedParseValue`: inspect its result dictionary; check buffer length, `Float`/dimension/count, and whether the tenant already exists.

[Configuration, recovery guarantees and limitations](MultiTenantDesign.md)
describe the complete settings and trusted-caller boundary. For the automated
CLI/client/socket/crash-recovery smoke test, run
`python3 Test/tenant_smoke.py --bin-dir Release` using the binaries built above.
