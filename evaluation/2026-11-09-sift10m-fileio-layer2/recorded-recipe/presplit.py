import bisect
import json
import math
from pathlib import Path
import re
import time
import urllib.request


def plan(prefix, layers, vectors, buckets):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", prefix):
        raise ValueError("Invalid owned run prefix")
    if layers not in (1, 2) or not 3 <= buckets <= 256 or not buckets <= vectors <= 2147483647:
        raise ValueError("Expected 1/2 layers, 3..256 buckets and a valid vector count")
    lower = (prefix + "_").encode("ascii")
    upper = lower[:-1] + b"`"
    keys = {lower, upper}
    keys.update(lower + bytes([i * 256 // buckets, 0, 0, 0]) for i in range(1, buckets))
    samples = {"posting": [
        lower + bytes([i * 256 // buckets, 0, 0, 0]) for i in range(buckets)]}
    for layer in range(layers):
        stem = lower + f"v:{layer}:".encode("ascii")
        keys.update((stem, stem[:-1] + b";"))
        keys.update(stem + f"{i * vectors // buckets:010d}".encode("ascii")
                    for i in range(buckets + 1))
        samples[f"version{layer}"] = [
            stem + f"{(2 * i + 1) * vectors // (2 * buckets):010d}".encode("ascii")
            for i in range(buckets)]
    return lower, upper, sorted(keys), samples


def replication_issues(regions, stores):
    return [region["id"] for region in regions
            if region.get("pending_peers") or region.get("down_peers") or
            len(region["peers"]) != len(stores) or
            {peer["store_id"] for peer in region["peers"]} != stores or
            any(peer.get("role_name", "Voter") != "Voter" or peer.get("role", 0) != 0
                for peer in region["peers"]) or
            (region.get("leader") or {}).get("store_id") not in stores]


def inspect_plan(regions, stores, lower, upper, keys, samples):
    ordered = sorted(regions, key=lambda row: bytes.fromhex(row["start_key"]))
    starts = [bytes.fromhex(row["start_key"]) for row in ordered]
    missing = [key.hex() for key in keys if key not in set(starts)]
    owned = [row for row, start in zip(ordered, starts) if lower <= start < upper]
    if any(not row["end_key"] or bytes.fromhex(row["end_key"]) > upper for row in owned):
        raise RuntimeError("Run keyspace is not isolated by its boundary regions")
    groups, counts = {}, {}
    for group, points in samples.items():
        ids = set()
        leaders = {str(store): 0 for store in sorted(stores)}
        for key in points:
            position = bisect.bisect_right(starts, key) - 1
            if position < 0:
                raise RuntimeError("No region covers planned sample key")
            row = ordered[position]
            if row["end_key"] and key >= bytes.fromhex(row["end_key"]):
                raise RuntimeError("Region coverage has a gap")
            ids.add(row["id"])
            leader = (row.get("leader") or {}).get("store_id")
            if leader in stores:
                leaders[str(leader)] += 1
        groups[group] = sorted(ids)
        counts[group] = leaders
    return {
        "missing_boundaries": missing,
        "owned_region_ids": [row["id"] for row in owned],
        "unready_region_ids": replication_issues(owned, stores),
        "groups": groups, "leader_counts": counts,
        "balanced": all(len(groups[name]) == len(points) and
                        sum(counts[name].values()) == len(points) and
                        max(counts[name].values()) - min(counts[name].values()) <=
                        max(3, math.ceil(len(points) * 0.20))
                        for name, points in samples.items()),
    }


def prepare_cluster(root, prefix, layers, vectors, buckets):
    root = Path(root)
    lower, upper, keys, samples = plan(prefix, layers, vectors, buckets)
    evidence = {"prefix": prefix, "layers": layers, "vectors": vectors,
                "buckets_per_key_family": buckets, "split_keys": [key.hex() for key in keys],
                "split_responses": [], "scatter_responses": {}}

    def save(name, value):
        (root / name).write_text(json.dumps(value, indent=2) + "\n")

    def api(endpoint, payload=None):
        request = urllib.request.Request(
            "http://127.0.0.1:33991/pd/api/v1/" + endpoint,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.load(response)

    def regions():
        data = api("regions")
        if data["count"] != len(data["regions"]):
            raise RuntimeError("PD returned an incomplete region listing")
        return data["regions"]

    store_data = api("stores")
    if store_data["count"] != 3 or any(
            row["store"]["state_name"] != "Up" for row in store_data["stores"]):
        raise RuntimeError("Presplit requires three Up stores in the owned fresh cluster")
    stores = {row["store"]["id"] for row in store_data["stores"]}
    evidence["stores_before"] = store_data

    def wait(stage, predicate, stable=1):
        deadline = time.monotonic() + 600
        consecutive = 0
        while True:
            rows = regions()
            ready, detail = predicate(rows)
            save("presplit-progress.json", {"phase": stage, "detail": detail})
            save("status.json", {"phase": stage, "detail": detail})
            consecutive = consecutive + 1 if ready else 0
            if consecutive >= stable:
                return rows, detail
            if time.monotonic() >= deadline:
                raise TimeoutError("Presplit readiness deadline: " + stage + " " + str(detail))
            print("Presplit readiness:", stage, detail, flush=True)
            time.sleep(3)

    wait("waiting_initial_replication", lambda rows: (
        not replication_issues(rows, stores),
        {"unready_region_ids": replication_issues(rows, stores)}))
    for offset in range(0, len(keys), 64):
        print("Splitting boundary keys:", offset, "/", len(keys), flush=True)
        response = api("regions/split", {
            "split_keys": [key.hex() for key in keys[offset:offset + 64]], "retry_limit": 16})
        evidence["split_responses"].append(response)
        save("presplit-validation.json", evidence)
        if response["processed-percentage"] != 100:
            raise RuntimeError("PD could not split every requested boundary")

    def boundary_check(rows):
        starts = {bytes.fromhex(row["start_key"]) for row in rows}
        missing = [key.hex() for key in keys if key not in starts]
        unready = replication_issues(rows, stores)
        return not missing and not unready, {
            "missing_boundaries": len(missing), "unready_region_ids": unready}

    rows, _ = wait("waiting_split_replication", boundary_check)
    state = inspect_plan(rows, stores, lower, upper, keys, samples)
    groups = dict(state["groups"])
    covered = {region for ids in groups.values() for region in ids}
    groups["other"] = sorted(set(state["owned_region_ids"]) - covered)
    for group, ids in groups.items():
        if ids:
            response = api("regions/scatter", {
                "regions_id": ids, "group": prefix + "-" + group, "retry_limit": 16})
            evidence["scatter_responses"][group] = response
            save("presplit-validation.json", evidence)
            if response["processed-percentage"] != 100:
                raise RuntimeError("PD scatter was incomplete for " + group)

    def balance_check(rows):
        state = inspect_plan(rows, stores, lower, upper, keys, samples)
        operators = api("operators")
        ready = (not state["missing_boundaries"] and not state["unready_region_ids"] and
                 state["balanced"] and not operators)
        return ready, {"leader_counts": state["leader_counts"],
                       "unready_region_ids": state["unready_region_ids"],
                       "missing_boundaries": len(state["missing_boundaries"]),
                       "active_operators": len(operators)}

    rows, _ = wait("waiting_presplit_balance", balance_check, stable=3)
    evidence.update(inspect_plan(rows, stores, lower, upper, keys, samples))
    evidence.update(status="verified", verified_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    maximum_leader_count_difference=math.ceil(buckets * 0.20),
                    stores_after=api("stores"))
    save("presplit-validation.json", evidence)
    save("regions-before-workload.json", {"count": len(rows), "regions": rows})
    print("VERIFIED presplit, three caught-up replicas and balanced leaders:",
          evidence["leader_counts"], flush=True)
