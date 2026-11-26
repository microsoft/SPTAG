import hashlib
import json
import pathlib
import shutil
import struct
import time
import urllib.request


ROOT = pathlib.Path(__file__).resolve().parent
SOURCE = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/"
COUNT = 11_000_000
DIMENSION = 128


def download(name, destination, count, source_count, ranged):
    size = 8 + count * DIMENSION
    if destination.exists():
        with destination.open("rb") as stream:
            header = stream.read(8)
        if destination.stat().st_size != size or header != struct.pack("<ii", count, DIMENSION):
            raise ValueError(f"Existing dataset is invalid: {destination}")
        return
    request = urllib.request.Request(SOURCE + name)
    if ranged:
        request.add_header("Range", f"bytes=0-{size - 1}")
    partial = destination.with_suffix(".partial")
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
        if ranged and (response.status != 206 or
                       response.headers.get("Content-Range") != f"bytes 0-{size - 1}/{8 + source_count * DIMENSION}"):
            raise RuntimeError("The source did not honor the exact bounded range request")
        if response.read(8) != struct.pack("<ii", source_count, DIMENSION):
            raise ValueError("Unexpected SIFT source header")
        output.write(struct.pack("<ii", count, DIMENSION))
        remaining = size - 8
        while remaining:
            if time.monotonic() - started > 3600:
                raise TimeoutError("Dataset download exceeded one hour")
            data = response.read(min(4 * 1024 * 1024, remaining))
            if not data:
                raise EOFError("Incomplete SIFT download")
            output.write(data)
            remaining -= len(data)
        if response.read(1):
            raise ValueError("Source returned more data than requested")
    partial.rename(destination)
    print(f"Downloaded {destination}: {size} bytes", flush=True)


def main():
    if shutil.disk_usage(ROOT).free < 35 * 1024**3:
        raise RuntimeError("At least 35 GiB free is required before preparing this experiment")
    data = ROOT / "input"
    data.mkdir(exist_ok=True)
    download("base.1B.u8bin", data / "sift-first-11m.u8bin", COUNT, 1_000_000_000, True)
    download("query.public.10K.u8bin", data / "query.10k.u8bin", 10_000, 10_000, False)
    result = {"source": SOURCE, "vectors": COUNT, "dimension": DIMENSION, "files": {}}
    for path in sorted(data.glob("*.u8bin")):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(4 * 1024 * 1024):
                digest.update(block)
        result["files"][path.name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    (ROOT / "data.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
