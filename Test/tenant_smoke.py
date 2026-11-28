# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Exercise the tenant CLI, Python client and actual socket protocol using disposable data."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time


def fake_peer(port_pipe):
    """Return unsupported/malformed/late replies to exercise client-only failure paths."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port_pipe.send(listener.getsockname()[1])
        port_pipe.close()
        peer, _ = listener.accept()
        with peer:
            peer.settimeout(10)

            def read(size):
                result = b""
                while len(result) < size:
                    chunk = peer.recv(size - len(result))
                    assert chunk, "Client closed before completing the failure scenario"
                    result += chunk
                return result

            operation = 0
            while operation < 5:
                kind, _, size, _, resource = struct.unpack("<BBIII2x", read(16))
                read(size)
                if kind in (1, 2):
                    peer.sendall(struct.pack("<BBIII2x", kind | 0x80, 0, 0, 1, resource))
                    continue
                assert kind == 4
                operation += 1
                if operation == 5:
                    break
                status = 0
                body = struct.pack("<HHiiI", 1, 0x5001, -1, 0, 0)
                if operation == 1:
                    status, body = 2, b""
                elif operation == 2:
                    body = b"\x01"
                elif operation == 3:
                    time.sleep(0.2)
                peer.sendall(struct.pack("<BBIII2x", 0x84, status, len(body), 1, resource) + body)


def client_failure_paths(client_module):
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=fake_peer, args=(send,))
    process.start()
    send.close()
    client = None
    try:
        assert receive.poll(10), "Fake peer did not start"
        client = client_module.AnnClient("127.0.0.1", str(receive.recv()))
        for _ in range(200):
            if client.IsConnected():
                break
            time.sleep(0.01)
        assert client.IsConnected()
        client.SetTimeoutMilliseconds(1000)
        assert client.DeleteTenant("A")["status_name"] == "Socket_UnsupportedRequest"
        assert client.DeleteTenant("A")["status_name"] == "Socket_InvalidMessage"
        client.SetTimeoutMilliseconds(30)
        start = time.monotonic()
        assert client.DeleteTenant("A")["status_name"] == "Socket_Timeout"
        assert time.monotonic() - start < 0.8
        client.SetTimeoutMilliseconds(1000)
        assert client.DeleteTenant("A")["status_name"] == "Key_NotFound"
        assert client.DeleteTenant("A")["status_name"] in ("Socket_Timeout", "Socket_FailedNetwork")
        for _ in range(200):
            if not client.IsConnected():
                break
            time.sleep(0.01)
        assert client.DeleteTenant("A")["status_name"] == "Socket_FailedNetwork"
    finally:
        del client
        receive.close()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join()
    assert process.exitcode == 0


def concurrent_peer(port_pipe):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port_pipe.send(listener.getsockname()[1])
        port_pipe.close()
        peer, _ = listener.accept()
        with peer:
            peer.settimeout(10)

            def read(size):
                result = b""
                while len(result) < size:
                    chunk = peer.recv(size - len(result))
                    assert chunk
                    result += chunk
                return result

            requests = []
            while len(requests) < 4:
                kind, _, size, _, resource = struct.unpack("<BBIII2x", read(16))
                body = read(size)
                if kind in (1, 2):
                    peer.sendall(struct.pack("<BBIII2x", kind | 0x80, 0, 0, 1, resource))
                    continue
                assert kind == 4
                # Deliberately wait for every caller before responding: this requires GIL release.
                requests.append((resource, int(struct.unpack_from("<f", body, len(body) - 16)[0])))
            for resource, value in reversed(requests):
                body = struct.pack("<HHiiIif", 1, 0, -1, 1, 1, value, 0.0)
                peer.sendall(struct.pack("<BBIII2x", 0x84, 0, len(body), 1, resource) + body)
            # Keep the connection alive until the client has consumed all responses.
            while peer.recv(4096):
                pass


def concurrent_client_calls(client_module):
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=concurrent_peer, args=(send,))
    process.start()
    send.close()
    client = None
    try:
        assert receive.poll(10)
        client = client_module.AnnClient("127.0.0.1", str(receive.recv()))
        for _ in range(200):
            if client.IsConnected():
                break
            time.sleep(0.01)
        assert client.IsConnected()
        client.SetTimeoutMilliseconds(5000)
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(
                lambda i: client.SearchTenant("A", struct.pack("<4f", i, 0, 0, 0), 1, "Float"), range(4)))
        for i, result in enumerate(results):
            assert result["status"] == 0 and result["ids"] == [i], result
    finally:
        del client
        receive.close()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join()
    assert process.exitcode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, default=Path("Release"))
    args = parser.parse_args()
    binaries = args.bin_dir.resolve()
    sys.path.insert(0, str(binaries))
    import SPTAGClient

    client_failure_paths(SPTAGClient)
    concurrent_client_calls(SPTAGClient)
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")

    with tempfile.TemporaryDirectory(prefix="sptag-tenant-smoke-") as directory:
        work = Path(directory)
        root = work / "node"

        def vectors(name, count, base):
            path = work / name
            values = [base + i * 0.1 + j * 0.01 for i in range(count) for j in range(4)]
            path.write_bytes(struct.pack("<ii", count, 4) + struct.pack(f"<{count * 4}f", *values))
            return str(path)

        def tool(*command, expected=0, overrides=None):
            result = subprocess.run(
                [str(binaries / "tenanttool"), str(root), *command],
                env=dict(env, **(overrides or {})), capture_output=True, text=True, timeout=30,
            )
            assert result.returncode == expected, (command, result.stdout[-2000:], result.stderr[-2000:])
            return result.stdout

        tool("create", "alice", "Float", "L2", vectors("alice.bin", 128, 0))
        tool("create", "bob", "Float", "L2", vectors("bob.bin", 128, 1000), "BKT")
        stats = tool("stats", overrides={
            "SPTAG_AUTO_CHECKPOINT": "0",
            "SPTAG_CHECKPOINT_LOG_BYTES": "1",
            "SPTAG_CHECKPOINT_PRESSURE_BYTES": "1",
            "SPTAG_CHECKPOINT_MIN_INTERVAL_SECONDS": "0",
        })
        assert "auto_checkpoint=0" in stats
        tool("stats", expected=2, overrides={"SPTAG_AUTO_CHECKPOINT": "2"})
        tool("stats", expected=2, overrides={"SPTAG_CHECKPOINT_LOG_BYTES": "-1"})
        tool("stats", expected=1, overrides={"SPTAG_CHECKPOINT_LOG_BYTES": "0"})
        tool("create", "bad-head", "Float", "L2", str(work / "bob.bin"), "SPANN", expected=2)
        query = vectors("query.bin", 1, 500)
        tool("insert", "alice", query)
        tool("search", "alice", query, "1")
        with socket.socket() as reserve:
            reserve.bind(("127.0.0.1", 0))
            port = reserve.getsockname()[1]
        config = work / "server.ini"
        config.write_text(
            f"[Service]\nListenAddr=127.0.0.1\nListenPort={port}\n"
            f"ThreadNumber=2\nSocketThreadNumber=1\n[MultiTenant]\nRoot={root}\n"
        )

        with tempfile.TemporaryFile(mode="w+") as log:
            server = subprocess.Popen(
                [str(binaries / "server"), "-m", "socket", "-c", str(config)],
                env=env, stdout=log, stderr=subprocess.STDOUT,
            )
            connection = None
            client = None
            try:
                for _ in range(200):
                    assert server.poll() is None, "Server exited before readiness"
                    try:
                        connection = socket.create_connection(("127.0.0.1", port), timeout=0.2)
                        break
                    except ConnectionRefusedError:
                        time.sleep(0.05)
                assert connection is not None, "Server did not become ready"
                connection.settimeout(5)

                def read(size):
                    result = b""
                    while len(result) < size:
                        chunk = connection.recv(size - len(result))
                        assert chunk, "Unexpected socket EOF"
                        result += chunk
                    return result

                # Keep the readiness connection open and complete the existing registration handshake.
                kind, status, size, _, resource = struct.unpack("<BBIII2x", read(16))
                assert (kind, status, size) == (2, 0, 0)
                connection.sendall(struct.pack("<BBIII2x", 0x82, 0, 0, 1, resource))
                tool("list", expected=1)
                assert server.poll() is None

                def search(tenant, expected_status=0):
                    text = f"$tenantid:{tenant} $resultnum:1 500|500.01|500.02|500.03".encode()
                    body = struct.pack("<HHBI", 1, 0, 0, len(text)) + text
                    connection.sendall(struct.pack("<BBIII2x", 3, 0, len(body), 0, 17) + body)
                    kind, status, size, _, _ = struct.unpack("<BBIII2x", read(16))
                    assert (kind, status) == (0x83, expected_status), (kind, status)
                    result = read(size)
                    if expected_status:
                        return None
                    major, _, status, count = struct.unpack_from("<HHBI", result)
                    assert (major, status, count) == (1, 0, 1)
                    length = struct.unpack_from("<I", result, 9)[0]
                    assert result[13:13 + length].decode() == tenant
                    count, metadata, vector_id, _ = struct.unpack_from("<IBif", result, 13 + length)
                    assert count == 1 and not metadata
                    return vector_id

                assert search("alice") == 128
                assert search("bob") != 128
                search("missing", expected_status=3)

                # Malformed tenant packets must fail explicitly without mutating state.
                for body in (b"", b"\x02\x00", b"\x01\x00\xff"):
                    connection.sendall(struct.pack("<BBIII2x", 4, 0, len(body), 0, 18) + body)
                    kind, status, size, _, resource = struct.unpack("<BBIII2x", read(16))
                    assert (kind, status, resource) == (0x84, 0, 18)
                    reply = read(size)
                    assert struct.unpack_from("<HH", reply) == (1, 0x11)

                client = SPTAGClient.AnnClient("127.0.0.1", str(port))
                for _ in range(200):
                    if client.IsConnected():
                        break
                    time.sleep(0.05)
                assert client.IsConnected()
                client.SetTimeoutMilliseconds(30000)

                def check(result, status="Success"):
                    assert result["status_name"] == status, result
                    assert (result["status"] == 0) == (status == "Success"), result
                    assert set(result) == {"status", "status_name", "first_id", "count", "ids", "distances"}
                    return result

                seed = Path(vectors("carol.bin", 128, -1000)).read_bytes()[8:]
                target = Path(query).read_bytes()[8:]
                tenant = "carol / binary id"
                check(client.CreateTenant(tenant, seed, 128, 4, "Float", "L2"))
                check(client.CreateTenant(tenant, seed, 128, 4, "Float", "L2"), "FailedParseValue")
                check(client.CreateTenant("bad", seed, 128, 5, "Float", "L2"), "FailedParseValue")
                check(client.CreateTenant("empty", b"", 0, 4, "Float", "L2"), "FailedParseValue")
                for head in ("SPANN", "unknown", "Undefined"):
                    check(client.CreateTenant("bkt", seed, 128, 4, "Float", "L2", head), "FailedParseValue")
                check(client.CreateTenant("bkt", seed, 128, 4, "Float", "L2", "BKT"))
                assert check(client.InsertTenant("bkt", target, 1, "Float"))["first_id"] == 128
                check(client.DeleteTenantVector("bkt", 0))
                assert check(client.SearchTenant("bkt", target, 1, "Float"))["ids"] == [128]
                integer_seed = bytes((i * 13 + i // 7) % 251 for i in range(128 * 7))
                check(client.CreateTenant("integer", integer_seed, 128, 7, "UInt8", "L2", "KDT"))
                found = check(client.SearchTenant("integer", integer_seed[:7], 1, "UInt8"))
                assert found["ids"] == [0] and found["distances"] == [0.0], found
                check(client.InsertTenant(tenant, target, 1, "UInt8"), "FailedParseValue")
                check(client.SearchTenant(tenant, target, 1, "UInt8"), "FailedParseValue")
                check(client.InsertTenant(tenant, target[:-4], 1, "Float"), "FailedParseValue")
                check(client.SearchTenant(tenant, target, 1001, "Float"), "FailedParseValue")
                check(client.InsertTenant(tenant, struct.pack("<4f", float("nan"), 0, 0, 0), 1, "Float"),
                      "FailedParseValue")
                check(client.DeleteTenant("missing"), "Key_NotFound")
                check(client.DeleteTenantVector(tenant, 9999), "VectorNotFound")
                check(client.DeleteTenant("integer"))
                check(client.SearchTenant("integer", integer_seed[:7], 1, "UInt8"), "Key_NotFound")
                check(client.CreateTenant("integer", integer_seed, 128, 7, "UInt8", "L2", "BKT"))
                check(client.DeleteTenant("integer"))

                # The old tuple-returning Search remains usable with tenantid.
                client.SetSearchParam("tenantid", "alice")
                legacy = client.Search(target, 1, "Float", False)
                assert len(legacy) == 4 and legacy[0] == [128], legacy
                # New RPCs use their explicit tenant, not mutable legacy search parameters.
                inserted = check(client.InsertTenant(tenant, target, 1, "Float"))
                assert (inserted["first_id"], inserted["count"]) == (128, 1), inserted
                found = check(client.SearchTenant(tenant, target, 1, "Float"))
                assert found["ids"] == [128] and found["count"] == 1, found
                gate = threading.Barrier(6)

                def concurrent_operation(worker):
                    gate.wait()
                    if worker < 4:
                        values = [3000 + worker * 100 + i * 0.1 + j * 0.01
                                  for i in range(64) for j in range(4)]
                        return check(client.InsertTenant(tenant, struct.pack("<256f", *values), 64, "Float"))
                    for _ in range(50):
                        result = check(client.SearchTenant(tenant, target, 1, "Float"))
                        assert result["ids"] == [128], result
                    return None

                with ThreadPoolExecutor(max_workers=6) as executor:
                    parallel_results = list(executor.map(concurrent_operation, range(6)))
                assert sorted(result["first_id"] for result in parallel_results[:4]) == [129, 193, 257, 321]
                check(client.DeleteTenantVector(tenant, 0))
                check(client.DeleteTenantVector(tenant, 0), "VectorNotFound")
                assert 0 not in check(client.SearchTenant(tenant, seed[:16], 3, "Float"))["ids"]
                assert check(client.SearchTenant("bob", target, 1, "Float"))["ids"] != [128]
                assert server.poll() is None
                # No graceful checkpoint: online acknowledged mutations must survive process failure.
                server.kill()
                server.wait(timeout=10)
            finally:
                del client
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
                if connection is not None:
                    connection.close()

        assert "128 " in tool("search", "carol / binary id", query, "1")
        assert "128 " in tool("search", "bkt", query, "1")
        tool("delete-vector", "bkt", "0", expected=1)
        tool("delete-tenant", "bkt")
        tool("delete-vector", "carol / binary id", "0", expected=1)
        tool("delete-tenant", "carol / binary id")
        tool("delete-vector", "alice", "128")
        tool("delete-tenant", "bob")
        tool("list")
    print("PASS: CLI/Python CRUD, concurrent RPC/GIL release, legacy search, isolation, "
          "timeout/late replies, errors, ownership and crash recovery")


if __name__ == "__main__":
    main()
