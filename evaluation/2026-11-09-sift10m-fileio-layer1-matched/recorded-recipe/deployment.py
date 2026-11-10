import json
from pathlib import Path
import shutil
import socket
import subprocess
import time
import urllib.request


PD_IMAGE = "pingcap/pd@sha256:2a79836fcfe4948103b9eed9f69cc664208c3e6d5bae5e12d0c01e29c3d1598e"
TIKV_IMAGE = "pingcap/tikv@sha256:b6893e84e1914cff9fd262daac479ab24ef07ad99b15fd2b33457d02cb9ee38a"
ENDPOINTS = "127.0.0.1:33991,127.0.0.1:33992,127.0.0.1:33993"
PORTS = [base + offset for base in (33990, 34000, 30360, 30380) for offset in (1, 2, 3)]


class Cluster:
    def __init__(self, root, prefix):
        self.root = Path(root)
        self.prefix = prefix
        self.owned = []

    def docker(self, *args, **kwargs):
        return subprocess.run(["sudo", "-n", "docker", *args], check=True, **kwargs)

    def wait(self, endpoint, filename, predicate):
        started = time.monotonic()
        while True:
            try:
                with urllib.request.urlopen(
                        "http://127.0.0.1:33991/pd/api/v1/" + endpoint, timeout=5) as response:
                    data = json.load(response)
                (self.root / filename).write_text(json.dumps(data, indent=2) + "\n")
                if predicate(data):
                    return
                print("Waiting for cluster readiness:", endpoint, flush=True)
            except (OSError, ValueError) as error:
                print("Cluster readiness retry:", endpoint, error, flush=True)
            if time.monotonic() - started > 240:
                raise TimeoutError("Cluster readiness deadline: " + endpoint)
            time.sleep(3)

    def start_container(self, name, image, args, mounts):
        command = ["run", "-d", "--name", name, "--label", "sptag.run=" + self.prefix,
                   "--network", "host"]
        for mount in mounts:
            command.extend(["-v", mount])
        self.docker(*command, image, *args)
        self.owned.append(name)

    def start(self):
        self.docker("info", stdout=subprocess.DEVNULL)
        for port in PORTS:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", port))
        with (self.root / "images.json").open("x") as stream:
            self.docker("image", "inspect", PD_IMAGE, TIKV_IMAGE, stdout=stream)
        cluster = self.root / "cluster"
        cluster.mkdir()
        shutil.copy2(self.root / "tikv.toml", cluster / "tikv.toml")
        initial = ",".join(f"sift10m-pd{i}=http://127.0.0.1:{34000+i}" for i in (1, 2, 3))
        for i in (1, 2, 3):
            (cluster / f"pd{i}").mkdir()
            self.start_container(f"{self.prefix}-pd{i}", PD_IMAGE, [
                f"--name=sift10m-pd{i}", f"--client-urls=http://127.0.0.1:{33990+i}",
                f"--advertise-client-urls=http://127.0.0.1:{33990+i}",
                f"--peer-urls=http://127.0.0.1:{34000+i}",
                f"--advertise-peer-urls=http://127.0.0.1:{34000+i}",
                "--initial-cluster=" + initial, "--data-dir=/data/pd", "--log-file=/data/pd.log",
            ], [f"{cluster}/pd{i}:/data"])
        self.wait("health", "pd-health.json",
                  lambda data: len(data) == 3 and all(item["health"] for item in data))
        for i in (1, 2, 3):
            (cluster / f"tikv{i}").mkdir()
            self.start_container(f"{self.prefix}-tikv{i}", TIKV_IMAGE, [
                "--pd-endpoints=" + ENDPOINTS, f"--addr=127.0.0.1:{30360+i}",
                f"--advertise-addr=127.0.0.1:{30360+i}", f"--status-addr=127.0.0.1:{30380+i}",
                "--data-dir=/data/tikv", "--log-file=/data/tikv.log", "--config=/config.toml",
            ], [f"{cluster}/tikv{i}:/data", f"{cluster}/tikv.toml:/config.toml:ro"])
        self.wait("stores", "stores-before.json", lambda data: data["count"] == 3 and
                  all(item["store"]["state_name"] == "Up" for item in data["stores"]))
        self.wait("config/replicate", "replication.json", lambda data: data["max-replicas"] == 3)

    def close(self):
        if not self.owned:
            return
        result = self.docker("inspect", *self.owned, capture_output=True, text=True)
        if any(item["Config"]["Labels"].get("sptag.run") != self.prefix
               for item in json.loads(result.stdout)):
            raise RuntimeError("Refusing to stop containers with a different ownership label")
        for role in ("tikv", "pd"):
            names = [name for name in self.owned if name.rsplit("-", 1)[-1].startswith(role)]
            if names:
                self.docker("stop", "-t", "120", *names)
        result = self.docker("inspect", *self.owned, capture_output=True, text=True)
        (self.root / "cluster-stopped.json").write_text(result.stdout)
        if any(item["State"]["Running"] or item["State"]["OOMKilled"] or item["State"]["ExitCode"]
               for item in json.loads(result.stdout)):
            raise RuntimeError("Owned containers failed clean shutdown; see cluster-stopped.json")
