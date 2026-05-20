#!/bin/bash
# Multi-machine distributed benchmark orchestrator for SPTAG.
#
# Usage:
#   ./run_distributed.sh deploy     <cluster.conf>                Deploy binary + data to all nodes
#   ./run_distributed.sh setup-bins <cluster.conf>                Download tikv-server / pd-server to every node
#   ./run_distributed.sh start-tikv <cluster.conf> [node_count]   Start independent TiKV/PD instances
#   ./run_distributed.sh stop-tikv  <cluster.conf> [node_count]   Stop TiKV/PD instances
#   ./run_distributed.sh run        <cluster.conf> <scale> <node_count>  Run benchmark
#   ./run_distributed.sh bench      <cluster.conf> <scale> [scale...]    Run 1-node + N-node for each scale
#   ./run_distributed.sh cleanup    <cluster.conf>                Remove deployed files from remote nodes
#
# Environment variables:
#   NOCACHE=1          Disable all caches (TiKV block cache, OS page cache, VersionCache)
#   BUILD_WITH_CACHE=1 (only with NOCACHE=1) Use cached TiKV+VersionCache during the
#                      build phase, then restart TiKV with nocache config and drop all
#                      OS caches before the search/insert phase. Useful for large scales
#                      (e.g. 100M) where building under nocache is impractical.
#   SKIP_TIKV_SWAP=1   (only with BUILD_WITH_CACHE=1) Skip the TiKV container restart.
#                      Drop OS caches and rely on VersionCache=0 INI overrides for "nocache"
#                      semantics. Avoids docker rm -f corruption that has destroyed recall
#                      at 100M scale; TiKV block cache stays warm but contains mostly recent
#                      build writes (random search reads largely miss it anyway).
#   SKIP_SAVE_LOAD=1   (only with NOCACHE=1) Bypass the post-build SaveIndex / per-batch
#                      LoadIndex / Clone / SaveIndex cycles. For 1-node, build+search+insert
#                      run in a single SPTAGTest process, dropping OS pagecache after build.
#                      For 2-node, the build phase skips the broken final SaveIndex (relies
#                      on the index files written during BuildLargeIndex). Required at 100M
#                      scale where SaveIndex's "wait for all background jobs to finish" loop
#                      never terminates and risks a gRPC SEGFAULT after several hours.
#                      VersionCache cannot be reset mid-process so it stays warm from build.
#   SKIP_HEAD_BUILD=1  Reuse existing HeadIndex if present (RebuildSSDOnly). Falls back to
#                      full build if HeadIndex is missing.
#
# Prerequisites:
#   - Passwordless SSH from driver to all nodes (configure ssh_key in cluster.conf)
#   - Docker installed on all nodes (for TiKV)
#   - cluster.conf configured (see cluster.conf.example)
#
# The driver (first node in [nodes]) orchestrates everything.
# Compute nodes share a single TiKV raft cluster: all PDs join one raft group,
# all TiKVs point to all PDs, max-replicas=1 (no replication, each region on
# exactly one store). With 2 nodes this gives 2 PDs + 2 TiKV stores in one
# cluster; any compute can read any posting via PD-routed TiKV calls, so the
# distributed routing layer no longer needs to forward reads between computes.

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="$(cd "$SCRIPT_DIR/../.." && pwd)/benchmark_logs"
mkdir -p "$LOGDIR"

# ─── Config Parsing ───

declare -a NODE_HOSTS NODE_ROUTER_PORTS
declare -a TIKV_HOSTS TIKV_PD_CLIENT_PORTS TIKV_PD_PEER_PORTS TIKV_PORTS
declare SSH_USER SPTAG_DIR DATA_DIR TIKV_VERSION PD_VERSION SSH_KEY
declare TIKV_IMAGE PD_IMAGE HELPER_IMAGE BIN_DIR MIRROR
TOTAL_NODES=0

parse_config() {
    local CONF="$1"
    if [ ! -f "$CONF" ]; then
        echo "ERROR: Config file not found: $CONF"
        exit 1
    fi

    local SECTION=""

    while IFS= read -r line || [ -n "$line" ]; do
        # Strip comments and whitespace
        line="${line%%#*}"
        line="$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        [ -z "$line" ] && continue

        # Section header
        if [[ "$line" =~ ^\[(.+)\]$ ]]; then
            SECTION="${BASH_REMATCH[1]}"
            continue
        fi

        case "$SECTION" in
            cluster)
                local key="${line%%=*}"
                local val="${line#*=}"
                case "$key" in
                    ssh_user)     SSH_USER="$val" ;;
                    sptag_dir)    SPTAG_DIR="$val" ;;
                    data_dir)     DATA_DIR="$val" ;;
                    tikv_version) TIKV_VERSION="$val" ;;
                    pd_version)   PD_VERSION="$val" ;;
                    tikv_image)   TIKV_IMAGE="$val" ;;
                    pd_image)     PD_IMAGE="$val" ;;
                    helper_image) HELPER_IMAGE="$val" ;;
                    bin_dir)      BIN_DIR="$val" ;;
                    mirror)       MIRROR="$val" ;;
                    ssh_key)      SSH_KEY="$val" ;;
                esac
                ;;
            nodes)
                read -r host rport <<< "$line"
                NODE_HOSTS+=("$host")
                NODE_ROUTER_PORTS+=("$rport")
                ;;
            tikv)
                read -r host pd_client pd_peer tikv_port <<< "$line"
                TIKV_HOSTS+=("$host")
                TIKV_PD_CLIENT_PORTS+=("$pd_client")
                TIKV_PD_PEER_PORTS+=("$pd_peer")
                TIKV_PORTS+=("$tikv_port")
                ;;
        esac
    done < "$CONF"

    # Defaults
    SSH_USER="${SSH_USER:-$(whoami)}"
    TIKV_VERSION="${TIKV_VERSION:-v8.5.1}"
    PD_VERSION="${PD_VERSION:-v8.5.1}"
    # Single image used for ALL containers (PD, TiKV, helper). Stock MCR
    # ubuntu:22.04 — never modified, never layered, so security scanners see
    # only the MCR base image. TiKV / PD binaries are downloaded to the host
    # at $BIN_DIR by `setup-bins` and bind-mounted into the container.
    HELPER_IMAGE="${HELPER_IMAGE:-mcr.microsoft.com/mirror/docker/library/ubuntu:22.04}"
    TIKV_IMAGE="${TIKV_IMAGE:-${HELPER_IMAGE}}"
    PD_IMAGE="${PD_IMAGE:-${HELPER_IMAGE}}"
    # Host path on every node where tikv-server / pd-server live. Populated
    # by `setup-bins`. Mounted read-only into containers as /sptag-bin.
    BIN_DIR="${BIN_DIR:-${SPTAG_DIR}/evaluation/distributed/bin}"
    MIRROR="${MIRROR:-https://tiup-mirrors.pingcap.com}"

    # Expand ~ in ssh_key path
    if [ -n "$SSH_KEY" ]; then
        SSH_KEY="${SSH_KEY/#\~/$HOME}"
    fi

    TOTAL_NODES=${#NODE_HOSTS[@]}

    if [ "$TOTAL_NODES" -lt 1 ]; then
        echo "ERROR: No compute nodes defined in [nodes]"
        exit 1
    fi
    if [ ${#TIKV_HOSTS[@]} -lt 1 ]; then
        echo "ERROR: No TiKV instances defined in [tikv]"
        exit 1
    fi

    echo "Cluster config loaded:"
    echo "  Compute nodes: $TOTAL_NODES (driver: ${NODE_HOSTS[0]})"
    echo "  TiKV instances: ${#TIKV_HOSTS[@]}"
    echo "  SSH user: $SSH_USER"
    echo "  SSH key: ${SSH_KEY:-(none)}"
    echo "  SPTAG dir: $SPTAG_DIR"
    echo "  Data dir: $DATA_DIR"
}

# ─── SSH Helpers ───

# Build SSH options string (key + host checking)
_ssh_opts() {
    local opts="-o StrictHostKeyChecking=no -o ConnectTimeout=10"
    if [ -n "$SSH_KEY" ]; then
        opts+=" -i $SSH_KEY"
    fi
    echo "$opts"
}

# Run command on remote host (or locally if it's the driver)
remote_exec() {
    local host="$1"; shift
    if [ "$host" = "${NODE_HOSTS[0]}" ] || [ "$host" = "localhost" ] || [ "$host" = "127.0.0.1" ]; then
        eval "$@"
    else
        ssh $(_ssh_opts) "$SSH_USER@$host" "$@"
    fi
}

# rsync files to remote host
remote_sync() {
    local host="$1"
    local src="$2"
    local dst="$3"
    if [ "$host" = "${NODE_HOSTS[0]}" ] || [ "$host" = "localhost" ]; then
        # Local copy — skip if same path
        if [ "$(realpath "$src")" != "$(realpath "$dst")" ]; then
            rsync -az --progress "$src" "$dst"
        fi
    else
        rsync -az --progress -e "ssh $(_ssh_opts)" "$src" "$SSH_USER@$host:$dst"
    fi
}

# ─── Deploy ───

cmd_deploy() {
    echo ""
    echo "=== Deploying SPTAG to ${#NODE_HOSTS[@]} nodes ==="
    echo ""

    # Validate SSH connectivity
    for host in "${NODE_HOSTS[@]}"; do
        if [ "$host" = "${NODE_HOSTS[0]}" ]; then continue; fi
        echo -n "  Checking SSH to $host... "
        if remote_exec "$host" "echo ok" >/dev/null 2>&1; then
            echo "OK"
        else
            echo "FAILED"
            echo "ERROR: Cannot SSH to $SSH_USER@$host"
            exit 1
        fi
    done

    # Deploy binary to all remote nodes
    echo ""
    echo "Deploying binary..."
    local BINARY="$SPTAG_DIR/Release/SPTAGTest"
    if [ ! -f "$BINARY" ]; then
        echo "ERROR: Binary not found: $BINARY (run cmake build first)"
        exit 1
    fi

    for host in "${NODE_HOSTS[@]}"; do
        if [ "$host" = "${NODE_HOSTS[0]}" ]; then continue; fi
        echo "  → $host:$SPTAG_DIR/Release/"
        remote_exec "$host" "mkdir -p $SPTAG_DIR/Release"
        remote_sync "$host" "$BINARY" "$SPTAG_DIR/Release/SPTAGTest"
        # Also deploy any shared libraries
        if ls "$SPTAG_DIR/Release/"*.so 2>/dev/null; then
            remote_sync "$host" "$SPTAG_DIR/Release/*.so" "$SPTAG_DIR/Release/"
        fi
        # Deploy bundled runtime libs (boost 1.73 / abseil / tbb / libstdc++)
        # used by SPTAGTest. Not committed; produced locally on the driver.
        if [ -d "$SPTAG_DIR/Release/runtime_libs" ]; then
            remote_exec "$host" "mkdir -p $SPTAG_DIR/Release/runtime_libs"
            rsync -az -e "ssh $(_ssh_opts)" \
                "$SPTAG_DIR/Release/runtime_libs/" \
                "$SSH_USER@$host:$SPTAG_DIR/Release/runtime_libs/"
        fi
    done

    # Deploy data files (perftest_* vectors, queries)
    echo ""
    echo "Deploying data files..."
    for host in "${NODE_HOSTS[@]}"; do
        if [ "$host" = "${NODE_HOSTS[0]}" ]; then continue; fi
        echo "  → $host:$SPTAG_DIR/ (perftest_* files)"
        remote_exec "$host" "mkdir -p $SPTAG_DIR"
        rsync -az --progress \
            --include='perftest_*' --exclude='*' \
            -e "ssh $(_ssh_opts)" \
            "$SPTAG_DIR/" "$SSH_USER@$host:$SPTAG_DIR/"
    done

    echo ""
    echo "Deploy complete."
}

# ─── TiKV/PD Binary Setup ───

setup_bins_one_host() {
    # Ensure tikv-server / pd-server are present at $BIN_DIR on $1.
    # Downloads from $MIRROR if missing or version mismatch. Idempotent.
    local host="$1"
    local cmd
    # shellcheck disable=SC2016
    cmd='set -e
        mkdir -p "'"$BIN_DIR"'"
        cd "'"$BIN_DIR"'"
        need_tikv=1
        if [ -x tikv-server ] && ./tikv-server --version 2>/dev/null | grep -q "Release Version:[[:space:]]*'"${TIKV_VERSION#v}"'"; then
            need_tikv=0
        fi
        if [ "$need_tikv" = "1" ]; then
            echo "  Downloading tikv-'"${TIKV_VERSION}"'..."
            curl -fsSL "'"${MIRROR}"'/tikv-'"${TIKV_VERSION}"'-linux-amd64.tar.gz" | tar -xz
            chmod +x tikv-server
        else
            echo "  tikv-'"${TIKV_VERSION}"' already present"
        fi
        need_pd=1
        if [ -x pd-server ] && ./pd-server --version 2>/dev/null | grep -q "Release Version:[[:space:]]*'"${PD_VERSION}"'"; then
            need_pd=0
        fi
        if [ "$need_pd" = "1" ]; then
            echo "  Downloading pd-'"${PD_VERSION}"'..."
            curl -fsSL "'"${MIRROR}"'/pd-'"${PD_VERSION}"'-linux-amd64.tar.gz" | tar -xz
            chmod +x pd-server pd-ctl pd-recover 2>/dev/null || true
        else
            echo "  pd-'"${PD_VERSION}"' already present"
        fi'

    if [ "$host" = "${NODE_HOSTS[0]}" ] || [ "$host" = "localhost" ] || [ "$host" = "127.0.0.1" ]; then
        bash -c "$cmd"
    else
        remote_exec "$host" "$cmd"
    fi
}

cmd_setup_bins() {
    # Download tikv-server + pd-server to ${BIN_DIR} on every distinct host
    # used by the cluster (compute nodes ∪ tikv nodes). Idempotent.
    echo ""
    echo "=== Setting up TiKV/PD binaries ==="
    echo "  BIN_DIR : $BIN_DIR"
    echo "  TIKV    : $TIKV_VERSION"
    echo "  PD      : $PD_VERSION"
    echo "  MIRROR  : $MIRROR"

    declare -A seen
    local -a hosts=()
    local h
    for h in "${NODE_HOSTS[@]}" "${TIKV_HOSTS[@]}"; do
        if [ -z "${seen[$h]:-}" ]; then
            seen[$h]=1
            hosts+=("$h")
        fi
    done

    for h in "${hosts[@]}"; do
        echo ""
        echo "→ $h"
        setup_bins_one_host "$h"
    done

    echo ""
    echo "Binary setup complete."
}

# ─── TiKV Management (Independent Mode) ───


tikv_start() {
    # Start the first <node_count> PD+TiKV pairs.
    #
    # node_count == 1: standalone PD + TiKV (1-node benchmarks).
    # node_count >= 2: SHARED raft cluster — all PDs join one raft group,
    #                  all TiKVs point to all PDs. max-replicas=1 so each
    #                  region lives on exactly one store; PD routes reads
    #                  to whichever store has the region.
    local node_count="${1:-${#TIKV_HOSTS[@]}}"
    echo ""
    if [ "$node_count" -le 1 ]; then
        echo "=== Starting 1 standalone TiKV instance ==="
    else
        echo "=== Starting $node_count-node SHARED TiKV raft cluster ==="
    fi

    # Ensure binaries are present on every host that will run a container.
    # Cheap if already there (version-grep, no download).
    local i_host
    for (( i_host=0; i_host<node_count; i_host++ )); do
        local h="${TIKV_HOSTS[$i_host]}"
        # quick presence check; only call full setup if missing
        local present
        if [ "$h" = "${NODE_HOSTS[0]}" ] || [ "$h" = "localhost" ] || [ "$h" = "127.0.0.1" ]; then
            present=$([ -x "$BIN_DIR/tikv-server" ] && [ -x "$BIN_DIR/pd-server" ] && echo yes || echo no)
        else
            present=$(remote_exec "$h" "[ -x $BIN_DIR/tikv-server ] && [ -x $BIN_DIR/pd-server ] && echo yes || echo no" 2>/dev/null | tr -d '[:space:]')
        fi
        if [ "$present" != "yes" ]; then
            echo "  → $h: binaries missing, running setup-bins"
            setup_bins_one_host "$h"
        fi
    done

    # Build the initial-cluster string used by every PD.
    # For 1-node it's a single-member raft; for N>=2 every PD lists all members.
    local initial_cluster=""
    for (( i=0; i<node_count; i++ )); do
        local host="${TIKV_HOSTS[$i]}"
        local peer_port="${TIKV_PD_PEER_PORTS[$i]}"
        local pd_name="pd${i}"
        [ -n "$initial_cluster" ] && initial_cluster+=","
        initial_cluster+="${pd_name}=http://${host}:${peer_port}"
    done

    # Build the comma-separated pd-endpoints list for TiKV --pd-endpoints.
    # For shared mode, every TiKV connects to every PD so PD-raft failover works.
    local pd_endpoints=""
    for (( i=0; i<node_count; i++ )); do
        local host="${TIKV_HOSTS[$i]}"
        local pd_port="${TIKV_PD_CLIENT_PORTS[$i]}"
        [ -n "$pd_endpoints" ] && pd_endpoints+=","
        pd_endpoints+="http://${host}:${pd_port}"
    done

    # Start PD instances. With node_count >= 2 they form a raft group.
    echo "Starting PD instances (initial-cluster=${initial_cluster})..."
    for (( i=0; i<node_count; i++ )); do
        local host="${TIKV_HOSTS[$i]}"
        local client_port="${TIKV_PD_CLIENT_PORTS[$i]}"
        local peer_port="${TIKV_PD_PEER_PORTS[$i]}"
        local pd_name="pd${i}"
        echo "  PD $i on $host:$client_port"

        remote_exec "$host" "docker rm -f sptag-pd-$i 2>/dev/null; \
            docker run -d --name sptag-pd-$i --net host \
            -v $DATA_DIR/tikv-data/pd-$i:/data \
            -v ${BIN_DIR}:/sptag-bin:ro \
            --entrypoint /sptag-bin/pd-server \
            ${PD_IMAGE} \
            --name=${pd_name} \
            --data-dir=/data \
            --client-urls=http://0.0.0.0:${client_port} \
            --advertise-client-urls=http://${host}:${client_port} \
            --peer-urls=http://0.0.0.0:${peer_port} \
            --advertise-peer-urls=http://${host}:${peer_port} \
            --initial-cluster=${initial_cluster}"
    done

    echo "Waiting for PD raft to form..."
    sleep 5

    # Wait until every PD reports the expected member count (raft quorum up).
    for (( i=0; i<node_count; i++ )); do
        local host="${TIKV_HOSTS[$i]}"
        local pd_port="${TIKV_PD_CLIENT_PORTS[$i]}"
        for attempt in $(seq 1 60); do
            local members
            members=$(curl -sf "http://${host}:${pd_port}/pd/api/v1/members" 2>/dev/null \
                | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('members',[])))" 2>/dev/null || echo 0)
            if [ "$members" -ge "$node_count" ]; then
                echo "  PD $i ($host:$pd_port) healthy (members=${members})"
                break
            fi
            if [ "$attempt" -eq 60 ]; then
                echo "  ERROR: PD $i ($host:$pd_port) only sees ${members}/${node_count} members after 60s"
                return 1
            fi
            sleep 1
        done
    done

    # NOTE: max-replicas is configured AFTER TiKV starts (see below). Setting
    # placement rules requires cluster bootstrap, which only happens once a
    # TiKV store joins. Before bootstrap, /pd/api/v1/config/rule returns 500
    # ErrNotBootstrapped. We rely on the fact that no data is written until
    # SPTAGTest connects (which happens after this function returns), so the
    # brief window where bootstrap uses default max-replicas=3 is harmless.

    # Start TiKV instances pointing at the shared PD endpoints.
    echo "Starting TiKV instances (pd-endpoints=${pd_endpoints})..."
    for (( i=0; i<node_count; i++ )); do
        local host="${TIKV_HOSTS[$i]}"
        local tikv_port="${TIKV_PORTS[$i]}"
        echo "  TiKV $i on $host:$tikv_port → shared PD cluster"

        # Deploy tikv.toml to remote host.
        # When BUILD_WITH_CACHE=1 we always start with the cached config; the search
        # phase will swap to tikv_nocache.toml via tikv_switch_to_nocache().
        local TIKV_TOML="$SCRIPT_DIR/configs/tikv.toml"
        if [[ "${NOCACHE:-0}" == "1" && "${BUILD_WITH_CACHE:-0}" != "1" \
              && -f "$SCRIPT_DIR/configs/tikv_nocache.toml" ]]; then
            TIKV_TOML="$SCRIPT_DIR/configs/tikv_nocache.toml"
            echo "  [NOCACHE] Using tikv_nocache.toml (block cache = 1MB)"
        elif [[ "${NOCACHE:-0}" == "1" && "${BUILD_WITH_CACHE:-0}" == "1" ]]; then
            echo "  [NOCACHE+BUILD_WITH_CACHE] Starting with cached tikv.toml (will swap before run phase)"
        fi
        if [[ -f "$TIKV_TOML" ]]; then
            remote_exec "$host" "docker run --rm -v $DATA_DIR/tikv-data:/data ${HELPER_IMAGE} mkdir -p /data/conf"
            if [ "$host" = "${NODE_HOSTS[0]}" ] || [ "$host" = "localhost" ] || [ "$host" = "127.0.0.1" ]; then
                docker run --rm -v $DATA_DIR/tikv-data/conf:/conf -v $(realpath "$TIKV_TOML"):/src/tikv.toml:ro ${HELPER_IMAGE} cp /src/tikv.toml /conf/tikv.toml
            else
                scp $(_ssh_opts) "$TIKV_TOML" "${SSH_USER}@${host}:${SPTAG_DIR}/tikv.toml"
                remote_exec "$host" "docker run --rm -v $DATA_DIR/tikv-data/conf:/conf -v ${SPTAG_DIR}/tikv.toml:/src/tikv.toml:ro ${HELPER_IMAGE} cp /src/tikv.toml /conf/tikv.toml"
            fi
        fi

        remote_exec "$host" "docker rm -f sptag-tikv-$i 2>/dev/null; \
            docker run -d --name sptag-tikv-$i --net host \
            --ulimit nofile=1048576:1048576 \
            -v $DATA_DIR/tikv-data/tikv-$i:/data \
            -v $DATA_DIR/tikv-data/conf:/conf \
            -v ${BIN_DIR}:/sptag-bin:ro \
            --entrypoint /sptag-bin/tikv-server \
            ${TIKV_IMAGE} \
            --config=/conf/tikv.toml \
            --addr=0.0.0.0:${tikv_port} \
            --advertise-addr=${host}:${tikv_port} \
            --data-dir=/data \
            --pd-endpoints=${pd_endpoints}"
    done

    echo "Waiting for TiKV stores to register..."
    sleep 5

    # All stores show up in PD's store list (any PD works — they share state).
    local pd_host="${TIKV_HOSTS[0]}"
    local pd_port_first="${TIKV_PD_CLIENT_PORTS[0]}"
    for attempt in $(seq 1 60); do
        local store_count
        store_count=$(curl -sf "http://${pd_host}:${pd_port_first}/pd/api/v1/stores" 2>/dev/null \
            | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('count',0))" 2>/dev/null || echo 0)
        if [ "$store_count" -ge "$node_count" ]; then
            echo "  All ${store_count} TiKV stores registered"
            break
        fi
        if [ "$attempt" -eq 60 ]; then
            echo "  WARNING: only ${store_count}/${node_count} TiKV stores registered after 60s"
        fi
        sleep 1
    done

    # Set max-replicas=1 on the shared cluster, NOW that cluster is bootstrapped.
    #
    # PD v6+ defaults to enable-placement-rules=true. The authoritative source
    # for replica count is then the default placement rule, NOT the legacy
    # max-replicas config. /config POST auto-syncs to the rule but is racy;
    # we explicitly POST the rule too. Both endpoints require bootstrap.
    # Bug seen v45: skipping this caused 30%+ of a 1-node run to execute with
    # max-replicas=3 → PD endlessly tried to schedule replicas onto 1 store
    # → constant region state changes → gRPC Deadline / region_error storm.
    echo "Setting max-replicas=1 (default placement rule)..."
    local target_replicas=1
    local mr_ok=0
    for attempt in $(seq 1 30); do
        curl -sf "http://${pd_host}:${pd_port_first}/pd/api/v1/config" \
            -X POST -d "{\"max-replicas\": ${target_replicas}}" >/dev/null 2>&1 || true
        curl -sf "http://${pd_host}:${pd_port_first}/pd/api/v1/config/rule" \
            -X POST -d "{\"group_id\":\"pd\",\"id\":\"default\",\"start_key\":\"\",\"end_key\":\"\",\"role\":\"voter\",\"count\":${target_replicas}}" \
            >/dev/null 2>&1 || true
        sleep 1
        local got_cfg
        got_cfg=$(curl -sf "http://${pd_host}:${pd_port_first}/pd/api/v1/config/replicate" 2>/dev/null \
            | python3 -c 'import sys,json;print(json.load(sys.stdin).get("max-replicas"))' 2>/dev/null)
        local got_rule
        got_rule=$(curl -sf "http://${pd_host}:${pd_port_first}/pd/api/v1/config/rule/pd/default" 2>/dev/null \
            | python3 -c 'import sys,json;print(json.load(sys.stdin).get("count"))' 2>/dev/null)
        if [ "$got_cfg" = "$target_replicas" ] && [ "$got_rule" = "$target_replicas" ]; then
            echo "  max-replicas=${target_replicas} set (attempt $attempt, config & rule verified)"
            mr_ok=1
            break
        fi
        sleep 1
    done
    if [ "$mr_ok" != "1" ]; then
        echo "  ERROR: Failed to set max-replicas=${target_replicas} after 30 attempts. Aborting." >&2
        return 1
    fi

    echo "TiKV cluster started ($node_count node(s))."
}

tikv_stop() {
    # Stop the first <node_count> TiKV+PD instances.
    local node_count="${1:-${#TIKV_HOSTS[@]}}"
    echo ""
    echo "=== Stopping $node_count TiKV instances ==="

    for (( i=0; i<node_count; i++ )); do
        local host="${TIKV_HOSTS[$i]}"
        echo "  Stopping TiKV $i and PD $i on $host..."
        remote_exec "$host" "docker rm -f sptag-tikv-$i sptag-pd-$i 2>/dev/null || true"
    done

    echo "TiKV instances stopped."
}

tikv_switch_to_nocache() {
    # Restart TiKV containers (NOT PD) with the nocache config, so that the search
    # and insert phases use cold block cache. Data on disk is preserved because we
    # reuse the same data-dir; PD keeps the cluster metadata.
    local node_count="${1:-${#TIKV_HOSTS[@]}}"
    if [[ ! -f "$SCRIPT_DIR/configs/tikv_nocache.toml" ]]; then
        echo "  ERROR: configs/tikv_nocache.toml not found; cannot switch to nocache"
        return 1
    fi
    echo ""
    echo "=== Restarting $node_count TiKV instances with tikv_nocache.toml ==="

    # Reconstruct the shared pd-endpoints list (same as tikv_start).
    local pd_endpoints=""
    for (( i=0; i<node_count; i++ )); do
        local h="${TIKV_HOSTS[$i]}"
        local pp="${TIKV_PD_CLIENT_PORTS[$i]}"
        [ -n "$pd_endpoints" ] && pd_endpoints+=","
        pd_endpoints+="http://${h}:${pp}"
    done

    for (( i=0; i<node_count; i++ )); do
        local host="${TIKV_HOSTS[$i]}"
        local tikv_port="${TIKV_PORTS[$i]}"
        local TIKV_TOML="$SCRIPT_DIR/configs/tikv_nocache.toml"
        echo "  TiKV $i on $host:$tikv_port → swapping config"

        remote_exec "$host" "docker run --rm -v $DATA_DIR/tikv-data:/data ${HELPER_IMAGE} mkdir -p /data/conf"
        if [ "$host" = "${NODE_HOSTS[0]}" ] || [ "$host" = "localhost" ] || [ "$host" = "127.0.0.1" ]; then
            docker run --rm -v $DATA_DIR/tikv-data/conf:/conf -v $(realpath "$TIKV_TOML"):/src/tikv.toml:ro ${HELPER_IMAGE} cp /src/tikv.toml /conf/tikv.toml
        else
            scp $(_ssh_opts) "$TIKV_TOML" "${SSH_USER}@${host}:${SPTAG_DIR}/tikv.toml"
            remote_exec "$host" "docker run --rm -v $DATA_DIR/tikv-data/conf:/conf -v ${SPTAG_DIR}/tikv.toml:/src/tikv.toml:ro ${HELPER_IMAGE} cp /src/tikv.toml /conf/tikv.toml"
        fi

        remote_exec "$host" "docker stop -t 120 sptag-tikv-$i 2>/dev/null; \
            docker rm -f sptag-tikv-$i 2>/dev/null; \
            docker run -d --name sptag-tikv-$i --net host \
            --ulimit nofile=1048576:1048576 \
            -v $DATA_DIR/tikv-data/tikv-$i:/data \
            -v $DATA_DIR/tikv-data/conf:/conf \
            -v ${BIN_DIR}:/sptag-bin:ro \
            --entrypoint /sptag-bin/tikv-server \
            ${TIKV_IMAGE} \
            --config=/conf/tikv.toml \
            --addr=0.0.0.0:${tikv_port} \
            --advertise-addr=${host}:${tikv_port} \
            --data-dir=/data \
            --pd-endpoints=${pd_endpoints}"
    done

    echo "Waiting for TiKV stores to re-register..."
    sleep 5
    local pd_host_first="${TIKV_HOSTS[0]}"
    local pd_port_first="${TIKV_PD_CLIENT_PORTS[0]}"
    for attempt in $(seq 1 60); do
        local store_count
        store_count=$(curl -sf "http://${pd_host_first}:${pd_port_first}/pd/api/v1/stores" 2>/dev/null \
            | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('count',0))" 2>/dev/null || echo 0)
        if [ "$store_count" -ge "$node_count" ]; then
            echo "  All ${store_count} TiKV stores re-registered"
            break
        fi
        if [ "$attempt" -eq 60 ]; then
            echo "  WARNING: only ${store_count}/${node_count} stores re-registered after 60s"
        fi
        sleep 1
    done
    echo "TiKV switched to nocache mode."
}

tikv_clean() {
    # Clean TiKV data for the first <node_count> instances.
    local node_count="${1:-${#TIKV_HOSTS[@]}}"
    echo ""
    echo "=== Cleaning TiKV data ($node_count instances) ==="

    for (( i=0; i<node_count; i++ )); do
        local host="${TIKV_HOSTS[$i]}"
        echo "  Cleaning TiKV data on $host..."
        remote_exec "$host" "docker run --rm -v $DATA_DIR/tikv-data:/data ${HELPER_IMAGE} \
            rm -rf /data/tikv-$i /data/pd-$i 2>/dev/null || true"
    done
}

# Legacy wrappers for the main case block
cmd_start_tikv() { tikv_start "${1:-${#TIKV_HOSTS[@]}}"; }
cmd_stop_tikv()  { tikv_stop  "${1:-${#TIKV_HOSTS[@]}}"; }

# ─── Cache Management ───

drop_all_caches() {
    # Drop OS page cache + dentries/inodes on the first <node_count> nodes.
    # This may take 30-60s per node if there are many dirty pages.
    local node_count="${1:-1}"
    if [[ "${SKIP_DROP_CACHES:-0}" == "1" ]]; then
        echo "[SKIP_DROP_CACHES=1] skipping OS page-cache drop on $node_count node(s)"
        return 0
    fi
    echo "Dropping OS page cache on $node_count node(s) (timeout 10s per node)..."
    for (( i=0; i<node_count; i++ )); do
        local host="${NODE_HOSTS[$i]}"
        echo -n "  $host: "
        remote_exec "$host" "timeout 10 sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'" && echo "done" || echo "timeout/failed (non-fatal)"
    done
    echo "Cache drop complete."
}

# ─── INI Generation ───

generate_ini() {
    # Generate a benchmark INI from a template, filling in [Distributed] fields.
    # Usage: generate_ini <scale> <node_count> [overrides...]
    local SCALE="$1"
    local NODE_COUNT="$2"
    shift 2

    local IDX_PATH="$DATA_DIR/proidx_${SCALE}_${NODE_COUNT}node/spann_index"
    local KEY_PREFIX="bench${SCALE}_${NODE_COUNT}node"

    # Build comma-separated address lists from the first node_count entries
    local dispatcher_addr="${NODE_HOSTS[0]}:30001"
    local worker_addrs="" store_addrs="" pd_addrs=""
    for (( i=0; i<NODE_COUNT; i++ )); do
        [ -n "$worker_addrs" ] && worker_addrs+=","
        worker_addrs+="${NODE_HOSTS[$i]}:${NODE_ROUTER_PORTS[$i]}"
        [ -n "$store_addrs" ] && store_addrs+=","
        store_addrs+="${TIKV_HOSTS[$i]}:${TIKV_PORTS[$i]}"
        [ -n "$pd_addrs" ] && pd_addrs+=","
        pd_addrs+="${TIKV_HOSTS[$i]}:${TIKV_PD_CLIENT_PORTS[$i]}"
    done

    # Load the base INI template
    local BASE_INI="$SCRIPT_DIR/configs/benchmark_${SCALE}_template.ini"
    if [ ! -f "$BASE_INI" ]; then
        echo "ERROR: Template INI not found: $BASE_INI" >&2
        return 1
    fi

    local OUT="$SCRIPT_DIR/configs/benchmark_${SCALE}_${NODE_COUNT}node.ini"
    cp "$BASE_INI" "$OUT"

    # Fill in placeholder fields
    sed -i "s|^IndexPath=.*|IndexPath=${IDX_PATH}|" "$OUT"
    sed -i "s|^TiKVKeyPrefix=.*|TiKVKeyPrefix=${KEY_PREFIX}|" "$OUT"
    sed -i "s|^DispatcherAddr=.*|DispatcherAddr=${dispatcher_addr}|" "$OUT"
    sed -i "s|^WorkerAddrs=.*|WorkerAddrs=${worker_addrs}|" "$OUT"
    sed -i "s|^StoreAddrs=.*|StoreAddrs=${store_addrs}|" "$OUT"
    sed -i "s|^PDAddrs=.*|PDAddrs=${pd_addrs}|" "$OUT"

    # Apply extra overrides (key=value pairs)
    for override in "$@"; do
        local key="${override%%=*}"
        local val="${override#*=}"
        if grep -q "^${key}=" "$OUT"; then
            sed -i "s|^${key}=.*|${key}=${val}|" "$OUT"
        else
            # Append to [Benchmark] section
            sed -i "/^\[Benchmark\]/a ${key}=${val}" "$OUT"
        fi
    done

    echo "$OUT"
}

# ─── Worker Management ───

WORKER_SSH_PIDS=()

start_remote_worker() {
    # Start a worker on a remote node. Returns immediately; worker runs in background.
    local NODE_IDX="$1"
    local INI="$2"
    local SCALE="$3"
    local NODE_COUNT="$4"
    local host="${NODE_HOSTS[$NODE_IDX]}"
    local LOG="$LOGDIR/benchmark_${SCALE}_${NODE_COUNT}node_worker${NODE_IDX}.log"

    # Copy INI + binary to remote
    remote_sync "$host" "$INI" "$SPTAG_DIR/worker_n${NODE_IDX}.ini"

    # Start worker via SSH (foreground on remote, background locally).
    # Use `ssh -n` to redirect stdin from /dev/null so SSH doesn't try to
    # acquire a TTY when the parent script runs under `nohup`. Without -n,
    # the SSH client sometimes silently re-points fd1 → /dev/null and fd2
    # → a deleted /tmp file, dropping the worker log.
    ssh -n $(_ssh_opts) "$SSH_USER@$host" \
        "cd $SPTAG_DIR && LD_LIBRARY_PATH=$SPTAG_DIR/Release/runtime_libs:/usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH:-} \
         WORKER_INDEX=${NODE_IDX} BENCHMARK_CONFIG=worker_n${NODE_IDX}.ini \
         ./Release/SPTAGTest --run_test=SPFreshTest/BenchmarkFromConfig 2>&1" \
        </dev/null > "$LOG" 2>&1 &
    local ssh_pid=$!
    WORKER_SSH_PIDS+=($ssh_pid)
    echo "  Worker n${NODE_IDX} on $host (SSH PID: $ssh_pid, log: $LOG)"
}

wait_workers_ready() {
    local SCALE="$1"
    local NODE_COUNT="$2"
    local TIMEOUT=120

    echo "Waiting for ${#WORKER_SSH_PIDS[@]} workers to be ready..."
    for attempt in $(seq 1 $TIMEOUT); do
        local all_ready=true
        for i in $(seq 1 $((NODE_COUNT - 1))); do
            local LOG="$LOGDIR/benchmark_${SCALE}_${NODE_COUNT}node_worker${i}.log"
            if ! grep -q "Worker.*[Rr]eady\|Waiting for dispatch" "$LOG" 2>/dev/null; then
                all_ready=false
            fi
        done
        if $all_ready; then
            echo "  All workers ready (${attempt}s)"
            return 0
        fi
        # Check if any worker SSH process died
        for idx in "${!WORKER_SSH_PIDS[@]}"; do
            if ! kill -0 "${WORKER_SSH_PIDS[$idx]}" 2>/dev/null; then
                echo "  ERROR: Worker SSH PID ${WORKER_SSH_PIDS[$idx]} exited prematurely"
                return 1
            fi
        done
        sleep 1
    done
    echo "  WARNING: Not all workers ready after ${TIMEOUT}s"
    return 1
}

stop_remote_workers() {
    # Wait for workers to self-exit (driver sends TCP Stop), then force-kill.
    local TIMEOUT=${1:-30}
    if [ ${#WORKER_SSH_PIDS[@]} -eq 0 ]; then return; fi

    echo "Waiting for ${#WORKER_SSH_PIDS[@]} remote workers to exit (${TIMEOUT}s timeout)..."
    for pid in "${WORKER_SSH_PIDS[@]}"; do
        local elapsed=0
        while kill -0 "$pid" 2>/dev/null && [ $elapsed -lt $TIMEOUT ]; do
            sleep 1
            elapsed=$((elapsed + 1))
        done
        if kill -0 "$pid" 2>/dev/null; then
            echo "  WARNING: SSH PID $pid still alive, force killing"
            kill -9 "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        else
            echo "  Worker (SSH PID $pid) exited gracefully"
        fi
    done
    WORKER_SSH_PIDS=()
}

# Watchdog: detect driver death (segfault, OOM, SIGKILL by oom_killer, ...)
# and tear down remote workers so they don't linger forever.
# The C++ heartbeat watchdog inside the worker is the primary defense (bounded
# at HeartbeatTimeoutSec, default 180s). This shell watchdog is a faster
# secondary path: as soon as the driver PID is gone we (a) kill the local SSH
# wrappers and (b) `pkill` the remote SPTAGTest processes.
DRIVER_WATCHDOG_PID=""

start_driver_watchdog() {
    local DRIVER_PID="$1"
    local NODE_COUNT="$2"
    if [ "$NODE_COUNT" -lt 2 ]; then return; fi
    if [ ${#WORKER_SSH_PIDS[@]} -eq 0 ]; then return; fi

    # Snapshot what we need before backgrounding (subshell forks current env).
    local _ssh_pids="${WORKER_SSH_PIDS[*]}"
    local _hosts=()
    for (( i=1; i<NODE_COUNT; i++ )); do _hosts+=("${NODE_HOSTS[$i]}"); done
    local _hosts_str="${_hosts[*]}"
    local _ssh_user="$SSH_USER"
    local _ssh_opts_str="$(_ssh_opts)"

    (
        while kill -0 "$DRIVER_PID" 2>/dev/null; do
            sleep 5
        done
        echo "[watchdog] Driver PID $DRIVER_PID is gone; tearing down remote workers" >&2
        for pid in $_ssh_pids; do
            kill -TERM "$pid" 2>/dev/null || true
        done
        for host in $_hosts_str; do
            ssh -n $_ssh_opts_str "$_ssh_user@$host" \
                "pkill -TERM -f 'SPTAGTest.*BenchmarkFromConfig' 2>/dev/null; \
                 sleep 5; \
                 pkill -KILL -f 'SPTAGTest.*BenchmarkFromConfig' 2>/dev/null; true" \
                </dev/null >/dev/null 2>&1 || true
        done
        for pid in $_ssh_pids; do
            kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
        done
    ) &
    DRIVER_WATCHDOG_PID=$!
    echo "  Driver watchdog started (PID: $DRIVER_WATCHDOG_PID, monitoring driver $DRIVER_PID)"
}

stop_driver_watchdog() {
    if [ -n "$DRIVER_WATCHDOG_PID" ] && kill -0 "$DRIVER_WATCHDOG_PID" 2>/dev/null; then
        kill -TERM "$DRIVER_WATCHDOG_PID" 2>/dev/null || true
        wait "$DRIVER_WATCHDOG_PID" 2>/dev/null || true
    fi
    DRIVER_WATCHDOG_PID=""
}

# ─── Benchmark Run ───

distribute_head_index() {
    # Copy the head index from driver to all worker nodes.
    local SCALE="$1"
    local NODE_COUNT="$2"
    local SRC="$DATA_DIR/proidx_${SCALE}_${NODE_COUNT}node/spann_index"

    echo "Distributing head index to $((NODE_COUNT - 1)) workers..."
    for (( i=1; i<NODE_COUNT; i++ )); do
        local host="${NODE_HOSTS[$i]}"
        local DST="$DATA_DIR/proidx_${SCALE}_${NODE_COUNT}node/spann_index"
        echo "  → n${i} ($host)"
        remote_exec "$host" "mkdir -p $DST"
        remote_sync "$host" "$SRC/" "$DST/"
    done
}

distribute_perftest_files() {
    # rsync generated perftest_* files from driver to workers.
    local NODE_COUNT="$1"
    echo "Distributing perftest_* data files to workers..."
    for (( i=1; i<NODE_COUNT; i++ )); do
        local host="${NODE_HOSTS[$i]}"
        echo "  → $host"
        rsync -az --progress \
            --include='perftest_*' --exclude='*' \
            -e "ssh $(_ssh_opts)" \
            "$SPTAG_DIR/" "$SSH_USER@$host:$SPTAG_DIR/"
    done
}

# Determine build mode: full rebuild or SSD-only (reuse HeadIndex).
# Sets BUILD_MODE_OVERRIDES array for generate_ini.
# Usage: resolve_build_mode <scale> <node_count>
resolve_build_mode() {
    local SCALE="$1" NODE_COUNT="$2"
    local IDX_DIR="$DATA_DIR/proidx_${SCALE}_${NODE_COUNT}node/spann_index"
    local HEAD_DIR="$IDX_DIR/HeadIndex"

    BUILD_MODE_OVERRIDES=()
    if [[ "${SKIP_HEAD_BUILD:-0}" == "1" ]] && [ -d "$HEAD_DIR" ] && [ -n "$(ls -A "$HEAD_DIR" 2>/dev/null)" ]; then
        echo "HeadIndex found at $HEAD_DIR — using RebuildSSDOnly (skip SelectHead+BuildHead)"
        BUILD_MODE_OVERRIDES=("RebuildSSDOnly=true")
    else
        if [[ "${SKIP_HEAD_BUILD:-0}" == "1" ]]; then
            echo "SKIP_HEAD_BUILD=1 but HeadIndex not found at $HEAD_DIR — falling back to full build"
        fi
        BUILD_MODE_OVERRIDES=("Rebuild=true")
    fi
}

cmd_run() {
    local SCALE="$1"
    local NODE_COUNT="$2"
    if [ -z "$SCALE" ] || [ -z "$NODE_COUNT" ]; then
        echo "Usage: $0 run <cluster.conf> <scale> <node_count>"
        exit 1
    fi

    local BINARY="$SPTAG_DIR/Release/SPTAGTest"

    echo ""
    echo "═══════════════════════════════════════════════════"
    echo "  ${SCALE}: ${NODE_COUNT}-node benchmark${NOCACHE:+ [NOCACHE]}"
    echo "  Start: $(date)"
    echo "═══════════════════════════════════════════════════"

    if [ "$NODE_COUNT" -eq 1 ]; then
        # ─── Single-node flow ───
        echo ""
        echo "--- Phase 0: Prepare TiKV (1 instance) ---"
        tikv_stop 1
        tikv_clean 1
        if ! tikv_start 1; then
            echo "ERROR: tikv_start failed; aborting benchmark." >&2
            return 1
        fi

        # Resolve build mode before cleaning (SKIP_HEAD_BUILD needs existing dir)
        resolve_build_mode "$SCALE" "$NODE_COUNT"

        if [[ " ${BUILD_MODE_OVERRIDES[*]} " != *"RebuildSSDOnly=true"* ]]; then
            # Full build: clean old index dir
            rm -rf "$DATA_DIR/proidx_${SCALE}_1node"
        fi
        mkdir -p "$DATA_DIR/proidx_${SCALE}_1node"

        if [[ "${NOCACHE:-0}" == "1" ]]; then
            # NOCACHE: Split into build + cache-drop + search
            local BUILD_VERSIONCACHE_OVERRIDES=("VersionCacheTTLMs=0" "VersionCacheMaxChunks=0")
            if [[ "${BUILD_WITH_CACHE:-0}" == "1" ]]; then
                # Build phase keeps caches enabled; the run phase below switches to nocache
                BUILD_VERSIONCACHE_OVERRIDES=()
                echo ""
                echo "--- Phase 1: Build only (BUILD_WITH_CACHE=1, caches enabled) ---"
            else
                echo ""
                echo "--- Phase 1: Build only (NOCACHE) ---"
            fi

            if [[ "${SKIP_SAVE_LOAD:-0}" == "1" ]]; then
                # Single-process flow: build + search + insert in one SPTAGTest invocation.
                # SkipSaveLoadCycles=true bypasses the broken post-build SaveIndex and per-batch
                # Load/Clone/Save. SPTAGTest itself drops OS pagecache after build, before query.
                echo "[SKIP_SAVE_LOAD=1] running build + search + insert in a single SPTAGTest process"
                local SINGLE_INI
                SINGLE_INI=$(generate_ini "$SCALE" 1 "${BUILD_MODE_OVERRIDES[@]}" \
                    "SkipSaveLoadCycles=true" "${BUILD_VERSIONCACHE_OVERRIDES[@]}") || exit 1

                ( cd "$SPTAG_DIR" && LD_LIBRARY_PATH="$SPTAG_DIR/Release/runtime_libs:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" BENCHMARK_CONFIG="$SINGLE_INI" \
                  BENCHMARK_OUTPUT="output_${SCALE}_1node.json" \
                  "$BINARY" --run_test=SPFreshTest/BenchmarkFromConfig 2>&1 ) \
                    | tee "$LOGDIR/benchmark_${SCALE}_1node_driver.log"

                echo "Done: $(date)"
                tikv_stop 1
                return 0
            fi

            local BUILD_INI
            BUILD_INI=$(generate_ini "$SCALE" 1 "${BUILD_MODE_OVERRIDES[@]}" "BuildOnly=true" "${BUILD_VERSIONCACHE_OVERRIDES[@]}") || exit 1

            ( cd "$SPTAG_DIR" && LD_LIBRARY_PATH="$SPTAG_DIR/Release/runtime_libs:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" BENCHMARK_CONFIG="$BUILD_INI" \
              BENCHMARK_OUTPUT="output_${SCALE}_1node_build.json" \
              "$BINARY" --run_test=SPFreshTest/BenchmarkFromConfig 2>&1 ) \
                | tee "$LOGDIR/benchmark_${SCALE}_1node_build.log"

            echo "Build done: $(date)"

            if [[ "${BUILD_WITH_CACHE:-0}" == "1" && "${SKIP_TIKV_SWAP:-0}" != "1" ]]; then
                echo ""
                echo "--- Phase 1.4: Switch TiKV to nocache config ---"
                tikv_switch_to_nocache 1
            elif [[ "${SKIP_TIKV_SWAP:-0}" == "1" ]]; then
                echo "[SKIP_TIKV_SWAP=1] keeping TiKV containers running; relying on drop_caches + VersionCache=0"
            fi

            echo ""
            echo "--- Phase 1.5: Drop all caches (NOCACHE) ---"
            drop_all_caches 1

            echo ""
            echo "--- Phase 2: Search+Insert (cold cache) ---"
            local RUN_INI
            RUN_INI=$(generate_ini "$SCALE" 1 "Rebuild=false" "VersionCacheTTLMs=0" "VersionCacheMaxChunks=0") || exit 1

            ( cd "$SPTAG_DIR" && LD_LIBRARY_PATH="$SPTAG_DIR/Release/runtime_libs:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" BENCHMARK_CONFIG="$RUN_INI" \
              BENCHMARK_OUTPUT="output_${SCALE}_1node.json" \
              "$BINARY" --run_test=SPFreshTest/BenchmarkFromConfig 2>&1 ) \
                | tee "$LOGDIR/benchmark_${SCALE}_1node_driver.log"
        else
            echo ""
            echo "--- Phase 1: Single-node run ---"
            local INI
            INI=$(generate_ini "$SCALE" 1 "${BUILD_MODE_OVERRIDES[@]}") || exit 1

            echo "Starting driver on ${NODE_HOSTS[0]}..."
            ( cd "$SPTAG_DIR" && LD_LIBRARY_PATH="$SPTAG_DIR/Release/runtime_libs:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" BENCHMARK_CONFIG="$INI" \
              BENCHMARK_OUTPUT="output_${SCALE}_1node.json" \
              "$BINARY" --run_test=SPFreshTest/BenchmarkFromConfig 2>&1 ) \
                | tee "$LOGDIR/benchmark_${SCALE}_1node_driver.log"
        fi

        echo "Done: $(date)"
        tikv_stop 1
    else
        # ─── Multi-node flow ───
        echo ""
        echo "--- Phase 0: Prepare TiKV ($NODE_COUNT instances) ---"
        tikv_stop "$NODE_COUNT"
        tikv_clean "$NODE_COUNT"
        if ! tikv_start "$NODE_COUNT"; then
            echo "ERROR: tikv_start failed; aborting benchmark." >&2
            return 1
        fi

        # --- Phase 1: Build index on driver ---
        echo ""
        echo "--- Phase 1: Build index on driver ---"
        local BUILD_INI
        local NOCACHE_OVERRIDES=()
        local BUILD_NOCACHE_OVERRIDES=()
        if [[ "${NOCACHE:-0}" == "1" ]]; then
            NOCACHE_OVERRIDES=("VersionCacheTTLMs=0" "VersionCacheMaxChunks=0" "WorkerTimeout=14400")
            if [[ "${BUILD_WITH_CACHE:-0}" == "1" ]]; then
                # Build with cache, only run phase is nocache
                BUILD_NOCACHE_OVERRIDES=()
                echo "[BUILD_WITH_CACHE=1] build phase keeps caches; will switch before run phase"
            else
                BUILD_NOCACHE_OVERRIDES=("${NOCACHE_OVERRIDES[@]}")
            fi
        fi

        # Resolve build mode before cleaning (SKIP_HEAD_BUILD needs existing dir)
        resolve_build_mode "$SCALE" "$NODE_COUNT"

        if [[ " ${BUILD_MODE_OVERRIDES[*]} " != *"RebuildSSDOnly=true"* ]]; then
            # Full build: clean old index dirs on all nodes
            for (( i=0; i<NODE_COUNT; i++ )); do
                local host="${NODE_HOSTS[$i]}"
                remote_exec "$host" "rm -rf $DATA_DIR/proidx_${SCALE}_${NODE_COUNT}node"
            done
        fi
        mkdir -p "$DATA_DIR/proidx_${SCALE}_${NODE_COUNT}node"

        local SKIP_SAVE_LOAD_OVERRIDES=()
        if [[ "${SKIP_SAVE_LOAD:-0}" == "1" ]]; then
            # In multi-node, the build phase still needs to persist files to disk so
            # workers can LoadIndex them. SkipSaveLoadCycles=true skips ONLY the redundant
            # post-build final SaveIndex (which truncates SPTAGHeadVectorIDs.bin and then
            # blocks forever in the SaveIndexData drain at 100M scale). Files written by
            # BuildLargeIndex during BuildHead remain valid on disk for the run phase.
            SKIP_SAVE_LOAD_OVERRIDES=("SkipSaveLoadCycles=true")
            echo "[SKIP_SAVE_LOAD=1] build phase will skip post-build SaveIndex"
        fi

        BUILD_INI=$(generate_ini "$SCALE" "$NODE_COUNT" "${BUILD_MODE_OVERRIDES[@]}" "BuildOnly=true" "${BUILD_NOCACHE_OVERRIDES[@]}" "${SKIP_SAVE_LOAD_OVERRIDES[@]}") || exit 1

        # Build runs on the driver only — shared TiKV cluster routes each
        # key to the owning store via PD, so the driver writes all postings
        # straight to TiKV without any per-node dispatch. Workers are not
        # launched during the build phase; they come up in Phase 3 (run).
        local BUILD_LOG="$LOGDIR/benchmark_${SCALE}_${NODE_COUNT}node_build.log"
        echo "Starting driver build on ${NODE_HOSTS[0]}..."
        ( cd "$SPTAG_DIR" && LD_LIBRARY_PATH="$SPTAG_DIR/Release/runtime_libs:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" BENCHMARK_CONFIG="$BUILD_INI" \
          BENCHMARK_OUTPUT="output_${SCALE}_${NODE_COUNT}node_build.json" \
          "$BINARY" --run_test=SPFreshTest/BenchmarkFromConfig ) \
            > "$BUILD_LOG" 2>&1 &
        local BUILD_PID=$!
        echo "  Driver build PID: $BUILD_PID"

        # Shell-side watchdog: if the driver dies unexpectedly (segfault, OOM,
        # SIGKILL) we want a fast failure path rather than hanging forever.
        WORKER_SSH_PIDS=()
        start_driver_watchdog "$BUILD_PID" "$NODE_COUNT"

        # Wait for the driver build to finish
        echo "  Waiting for driver build to complete..."
        wait "$BUILD_PID"
        local BUILD_RC=$?
        echo "Driver build done (exit=$BUILD_RC): $(date)"
        stop_driver_watchdog

        if [[ $BUILD_RC -ne 0 ]] || grep -q "===== SEGFAULT" "$BUILD_LOG"; then
            echo ""
            echo "ERROR: Build phase failed (exit=$BUILD_RC, segfault=$(grep -c '===== SEGFAULT' "$BUILD_LOG"))"
            echo "Refusing to proceed to run phase with broken build state."
            echo "Tail of build log:"
            tail -30 "$BUILD_LOG"
            tikv_stop "$NODE_COUNT"
            exit 1
        fi

        echo "Build done: $(date)"

        # --- Phase 2: Distribute data ---
        echo ""
        echo "--- Phase 2: Distribute head index + data ---"
        rm -f "$DATA_DIR/proidx_${SCALE}_${NODE_COUNT}node/spann_index/checkpoint.txt"

        distribute_head_index "$SCALE" "$NODE_COUNT"
        distribute_perftest_files "$NODE_COUNT"

        # Sync SPTAGTest binary + bundled runtime libs to all workers so
        # they pick up the latest compiled changes. (cmd_deploy is a separate
        # subcommand; without this step a stale binary on the worker silently
        # diverges from the driver.)
        echo ""
        echo "Syncing SPTAGTest binary + runtime_libs to workers..."
        for host in "${NODE_HOSTS[@]}"; do
            if [ "$host" = "${NODE_HOSTS[0]}" ]; then continue; fi
            remote_exec "$host" "mkdir -p $SPTAG_DIR/Release"
            remote_sync "$host" "$SPTAG_DIR/Release/SPTAGTest" "$SPTAG_DIR/Release/SPTAGTest"
            if [ -d "$SPTAG_DIR/Release/runtime_libs" ]; then
                remote_exec "$host" "mkdir -p $SPTAG_DIR/Release/runtime_libs"
                rsync -az -e "ssh $(_ssh_opts)" \
                    "$SPTAG_DIR/Release/runtime_libs/" \
                    "$SSH_USER@$host:$SPTAG_DIR/Release/runtime_libs/"
            fi
        done

        # Binary already pushed; nothing else to do here.

        # --- Phase 3: Start driver first (contains dispatcher), then workers ---
        echo ""

        # Drop caches if NOCACHE mode
        if [[ "${NOCACHE:-0}" == "1" ]]; then
            if [[ "${BUILD_WITH_CACHE:-0}" == "1" && "${SKIP_TIKV_SWAP:-0}" != "1" ]]; then
                echo "--- Phase 2.4: Switch TiKV to nocache config ---"
                tikv_switch_to_nocache "$NODE_COUNT"
            elif [[ "${SKIP_TIKV_SWAP:-0}" == "1" ]]; then
                echo "[SKIP_TIKV_SWAP=1] keeping TiKV containers running; relying on drop_caches + VersionCache=0"
            fi
            echo "--- Phase 2.5: Drop all caches (NOCACHE) ---"
            drop_all_caches "$NODE_COUNT"
        fi

        echo "--- Phase 3: Distributed run ---"

        local RUN_INI
        RUN_INI=$(generate_ini "$SCALE" "$NODE_COUNT" "Rebuild=false" "${NOCACHE_OVERRIDES[@]}") || exit 1

        # Start driver in background first — it contains the dispatcher that
        # workers need to connect to for ring registration.
        local DRIVER_LOG="$LOGDIR/benchmark_${SCALE}_${NODE_COUNT}node_driver.log"
        echo "Starting driver (dispatcher+worker0) on ${NODE_HOSTS[0]}..."
        ( cd "$SPTAG_DIR" && LD_LIBRARY_PATH="$SPTAG_DIR/Release/runtime_libs:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" BENCHMARK_CONFIG="$RUN_INI" \
          BENCHMARK_OUTPUT="output_${SCALE}_${NODE_COUNT}node.json" \
          "$BINARY" --run_test=SPFreshTest/BenchmarkFromConfig ) \
            > "$DRIVER_LOG" 2>&1 &
        local DRIVER_PID=$!
        echo "  Driver PID: $DRIVER_PID"

        # Wait for dispatcher to start listening before launching workers
        local DISP_PORT=30001
        echo "  Waiting for dispatcher to listen on port $DISP_PORT..."
        for attempt in $(seq 1 60); do
            if ss -tlnp 2>/dev/null | grep -q ":${DISP_PORT} " || \
               netstat -tlnp 2>/dev/null | grep -q ":${DISP_PORT} "; then
                echo "  Dispatcher listening (${attempt}s)"
                break
            fi
            if ! kill -0 "$DRIVER_PID" 2>/dev/null; then
                echo "  ERROR: Driver exited prematurely"
                cat "$DRIVER_LOG"
                return 1
            fi
            if [ "$attempt" -eq 60 ]; then
                echo "  WARNING: Dispatcher not detected on port $DISP_PORT after 60s, proceeding anyway"
            fi
            sleep 1
        done

        # Now start remote workers — they can connect to the dispatcher
        WORKER_SSH_PIDS=()
        for (( i=1; i<NODE_COUNT; i++ )); do
            start_remote_worker "$i" "$RUN_INI" "$SCALE" "$NODE_COUNT"
        done

        # Shell-side watchdog (see comment in build phase).
        start_driver_watchdog "$DRIVER_PID" "$NODE_COUNT"

        # Wait for driver to complete (it runs the full benchmark)
        echo "  Waiting for driver to complete..."
        wait "$DRIVER_PID"
        local DRIVER_EXIT=$?
        echo "Driver done (exit=$DRIVER_EXIT): $(date)"
        stop_driver_watchdog
        # Show driver output
        tail -20 "$DRIVER_LOG"

        # Driver sends TCP Stop to workers; wait for graceful exit
        stop_remote_workers 60

        # Collect remote logs
        echo "Collecting remote logs..."
        for (( i=1; i<NODE_COUNT; i++ )); do
            local host="${NODE_HOSTS[$i]}"
            local REMOTE_LOG="$SPTAG_DIR/worker_n${i}.log"
            scp $(_ssh_opts) "$SSH_USER@$host:$REMOTE_LOG" \
                "$LOGDIR/benchmark_${SCALE}_${NODE_COUNT}node_worker${i}_remote.log" 2>/dev/null || true
        done

        tikv_stop "$NODE_COUNT"
    fi

    echo ""
    echo "═══════════════════════════════════════════════════"
    echo "  ${SCALE} ${NODE_COUNT}-node done: $(date)"
    echo "  Results: output_${SCALE}_${NODE_COUNT}node.json"
    echo "  Logs:    $LOGDIR/benchmark_${SCALE}_${NODE_COUNT}node_*.log"
    echo "═══════════════════════════════════════════════════"
}

cmd_bench() {
    # Run 1-node baseline + N-node distributed for each specified scale.
    # Usage: cmd_bench <scale> [scale...]
    # Special scale "all" expands to all scales with templates in configs/.
    local scales=()
    for arg in "$@"; do
        if [ "$arg" = "all" ]; then
            for tmpl in "$SCRIPT_DIR"/configs/benchmark_*_template.ini; do
                local name
                name="$(basename "$tmpl")"
                name="${name#benchmark_}"
                name="${name%_template.ini}"
                scales+=("$name")
            done
        else
            scales+=("$arg")
        fi
    done

    if [ ${#scales[@]} -eq 0 ]; then
        echo "Usage: $0 bench <cluster.conf> <scale> [scale...] | all"
        echo "Available scales:"
        for tmpl in "$SCRIPT_DIR"/configs/benchmark_*_template.ini; do
            local name
            name="$(basename "$tmpl")"
            name="${name#benchmark_}"
            name="${name%_template.ini}"
            echo "  $name"
        done
        exit 1
    fi

    echo ""
    echo "═══════════════════════════════════════════════════"
    echo "  Benchmark suite: ${scales[*]}"
    echo "  Cluster: $TOTAL_NODES nodes"
    echo "  Start: $(date)"
    echo "═══════════════════════════════════════════════════"

    for scale in "${scales[@]}"; do
        echo ""
        echo "▶▶▶ Scale: $scale — 1-node baseline"
        cmd_run "$scale" 1

        if [ "$TOTAL_NODES" -gt 1 ]; then
            echo ""
            echo "▶▶▶ Scale: $scale — ${TOTAL_NODES}-node distributed"
            cmd_run "$scale" "$TOTAL_NODES"
        else
            echo "  (Skipping multi-node: cluster has only 1 node)"
        fi
    done

    echo ""
    echo "═══════════════════════════════════════════════════"
    echo "  Benchmark suite complete: $(date)"
    echo "═══════════════════════════════════════════════════"
}

# ─── Cleanup ───

cmd_cleanup() {
    echo ""
    echo "=== Cleaning up remote nodes ==="

    for i in $(seq 1 $((${#NODE_HOSTS[@]} - 1))); do
        local host="${NODE_HOSTS[$i]}"
        echo "  Cleaning $host..."
        remote_exec "$host" "rm -rf $SPTAG_DIR/Release/SPTAGTest $SPTAG_DIR/perftest_* $SPTAG_DIR/worker_*.ini"
        # Clean index directories
        remote_exec "$host" "rm -rf $DATA_DIR/proidx_*"
    done
    echo "Cleanup complete."
}

# ─── Main ───

CMD="$1"
CONF="$2"

if [ -z "$CMD" ] || [ -z "$CONF" ]; then
    echo "Usage: $0 <command> <cluster.conf> [args...]"
    echo ""
    echo "Commands:"
    echo "  deploy      Deploy binary and data to all nodes"
    echo "  start-tikv  Start independent TiKV/PD instances"
    echo "  stop-tikv   Stop TiKV/PD instances"
    echo "  run         Run benchmark: $0 run cluster.conf <scale> <node_count>"
    echo "  bench       Run full benchmark suite: $0 bench cluster.conf <scale> [scale...] | all"
    echo "  cleanup     Remove deployed files from remote nodes"
    exit 1
fi

parse_config "$CONF"

# Trap for cleanup on interrupt
trap 'echo ""; echo "Interrupted!"; stop_driver_watchdog; stop_remote_workers 5; cmd_stop_tikv; exit 1' INT TERM

case "$CMD" in
    deploy)
        cmd_deploy
        ;;
    setup-bins)
        cmd_setup_bins
        ;;
    start-tikv)
        cmd_start_tikv "${3:-}"
        ;;
    stop-tikv)
        cmd_stop_tikv "${3:-}"
        ;;
    run)
        cmd_run "$3" "$4"
        ;;
    bench)
        shift 2  # skip cmd and conf
        cmd_bench "$@"
        ;;
    cleanup)
        cmd_cleanup
        ;;
    *)
        echo "Unknown command: $CMD"
        echo "Valid commands: deploy, setup-bins, start-tikv, stop-tikv, run, bench, cleanup"
        exit 1
        ;;
esac
