#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

DATA_ROOT=${DATA_ROOT:-/mnt/md0/qiazh/tikv}
PD_IMAGE=${PD_IMAGE:-pingcap/pd:nightly}
TIKV_IMAGE=${TIKV_IMAGE:-pingcap/tikv:nightly}
REMOTE_DIR=${REMOTE_DIR:-/mnt/md0/qiazh/SPTAG/evaluation/2026-04-23}
REMOTE_HOSTS=${REMOTE_HOSTS:-"azureuser@annservicFX071Y azureuser@annservicKFECPH azureuser@annservicP92MC8"}
PRIVATE_IPS=${PRIVATE_IPS:-"annservicFX071Y annservicKFECPH annservicP92MC8"}
DOCKER_CMD=${DOCKER_CMD:-docker}

read -r -a PRIVATE_IP_ARRAY <<< "$PRIVATE_IPS"
read -r -a REMOTE_HOST_ARRAY <<< "$REMOTE_HOSTS"

usage() {
  cat <<'USAGE'
Usage: deploy_tikv_3hosts.sh <command>

Commands:
  prepare     Create data/log directories, copy tikv.toml, pull images.
  start-pd    Start the local PD container for this host.
  start-tikv  Start the local TiKV container for this host.
  configure   Set max-replicas=1 through PD.
  status      Show local containers and PD store status.
  stop        Remove local PD/TiKV containers.
  wipe        Remove local PD/TiKV containers and wipe local data/log dirs.
  remote      Copy this script/config to REMOTE_HOSTS and deploy all hosts.

Environment overrides:
  DATA_ROOT=/mnt/md0/qiazh/tikv
  PRIVATE_IPS="annservicFX071Y annservicKFECPH annservicP92MC8"
  REMOTE_HOSTS="azureuser@annservicFX071Y azureuser@annservicKFECPH azureuser@annservicP92MC8"
  DOCKER_CMD="sudo -n docker" # optional, if azureuser cannot access docker directly
  NODE_IP=annservicFX071Y  # optional, otherwise auto-detected from hostname -I
  NODE_INDEX=1             # optional fallback if NODE_IP cannot be detected
USAGE
}

run_docker() {
  read -r -a docker_cmd <<< "$DOCKER_CMD"
  "${docker_cmd[@]}" "$@"
}

wait_all() {
  local status=0
  local pid
  for pid in "$@"; do
    wait "$pid" || status=1
  done
  return "$status"
}

require_three_nodes() {
  if [[ ${#PRIVATE_IP_ARRAY[@]} -ne 3 ]]; then
    echo "PRIVATE_IPS must contain exactly 3 node addresses" >&2
    exit 1
  fi
}

node_ip() {
  if [[ -n "${NODE_IP:-}" ]]; then
    echo "$NODE_IP"
    return
  fi

  local host_ip candidate
  for host_ip in $(hostname -I); do
    for candidate in "${PRIVATE_IP_ARRAY[@]}"; do
      if [[ "$host_ip" == "$candidate" ]]; then
        echo "$candidate"
        return
      fi
    done
  done

  if [[ -n "${NODE_INDEX:-}" ]]; then
    echo "${PRIVATE_IP_ARRAY[$((NODE_INDEX - 1))]}"
    return
  fi

  echo "Cannot detect this host's TiKV advertise address. Set NODE_IP or NODE_INDEX." >&2
  exit 1
}

node_index() {
  local ip=$1
  local idx
  for idx in "${!PRIVATE_IP_ARRAY[@]}"; do
    if [[ "${PRIVATE_IP_ARRAY[$idx]}" == "$ip" ]]; then
      echo $((idx + 1))
      return
    fi
  done
  echo "Node address $ip is not in PRIVATE_IPS=$PRIVATE_IPS" >&2
  exit 1
}

pd_initial_cluster() {
  printf 'pd1=http://%s:2380,pd2=http://%s:2380,pd3=http://%s:2380' \
    "${PRIVATE_IP_ARRAY[0]}" "${PRIVATE_IP_ARRAY[1]}" "${PRIVATE_IP_ARRAY[2]}"
}

pd_endpoints() {
  printf '%s:2379,%s:2379,%s:2379' \
    "${PRIVATE_IP_ARRAY[0]}" "${PRIVATE_IP_ARRAY[1]}" "${PRIVATE_IP_ARRAY[2]}"
}

pd_url() {
  printf 'http://%s:2379' "${PRIVATE_IP_ARRAY[0]}"
}

prepare() {
  mkdir -p "$DATA_ROOT"/{pd-data,pd-logs,tikv-data,tikv-logs}
  cp "$SCRIPT_DIR/tikv.toml" "$DATA_ROOT/tikv.toml"
  run_docker pull "$PD_IMAGE"
  run_docker pull "$TIKV_IMAGE"
}

stop_local() {
  run_docker rm -f tikv-pd tikv-server >/dev/null 2>&1 || true
}

wipe_local() {
  stop_local
  sudo rm -rf "$DATA_ROOT"/{pd-data,pd-logs,tikv-data,tikv-logs}
  mkdir -p "$DATA_ROOT"/{pd-data,pd-logs,tikv-data,tikv-logs}
  cp "$SCRIPT_DIR/tikv.toml" "$DATA_ROOT/tikv.toml"
}

start_pd() {
  require_three_nodes
  prepare
  local ip idx
  ip=$(node_ip)
  idx=$(node_index "$ip")
  run_docker rm -f tikv-pd >/dev/null 2>&1 || true
  run_docker run -d --name tikv-pd --restart unless-stopped --network host \
    -v "$DATA_ROOT/pd-data:/data" \
    -v "$DATA_ROOT/pd-logs:/logs" \
    "$PD_IMAGE" \
    --name="pd$idx" \
    --client-urls=http://0.0.0.0:2379 \
    --peer-urls=http://0.0.0.0:2380 \
    --advertise-client-urls="http://$ip:2379" \
    --advertise-peer-urls="http://$ip:2380" \
    --initial-cluster="$(pd_initial_cluster)" \
    --data-dir=/data/pd \
    --log-file=/logs/pd.log
}

start_tikv() {
  require_three_nodes
  prepare
  local ip
  ip=$(node_ip)
  run_docker rm -f tikv-server >/dev/null 2>&1 || true
  run_docker run -d --name tikv-server --restart unless-stopped --network host \
    -v "$DATA_ROOT/tikv-data:/data" \
    -v "$DATA_ROOT/tikv-logs:/logs" \
    -v "$DATA_ROOT/tikv.toml:/opt/tikv.toml:ro" \
    "$TIKV_IMAGE" \
    --pd-endpoints="$(pd_endpoints)" \
    --addr=0.0.0.0:20160 \
    --advertise-addr="$ip:20160" \
    --status-addr=0.0.0.0:20180 \
    --data-dir=/data/tikv \
    --log-file=/logs/tikv.log \
    --config=/opt/tikv.toml
}

configure_cluster() {
  run_docker run --rm --network host --entrypoint /pd-ctl "$PD_IMAGE" \
    -u "$(pd_url)" config set max-replicas 1
  run_docker run --rm --network host --entrypoint /pd-ctl "$PD_IMAGE" \
    -u "$(pd_url)" config show | grep -E 'max-replicas|location-labels|strictly-match-label'
}

status_local() {
  run_docker ps --filter name='tikv-' --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'
  curl -fsS "$(pd_url)/pd/api/v1/stores" | python3 -c "
import json
import sys

stores = json.load(sys.stdin).get('stores', [])
for item in stores:
    store = item.get('store', {})
    status = item.get('status', {})
    print(store.get('id'), store.get('address'), store.get('state_name'), 'leaders', status.get('leader_count'))
"
}

remote_deploy() {
  require_three_nodes
  if [[ ${#REMOTE_HOST_ARRAY[@]} -ne 3 ]]; then
    echo "REMOTE_HOSTS must contain exactly 3 hosts" >&2
    exit 1
  fi

  for host in "${REMOTE_HOST_ARRAY[@]}"; do
    ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$host" "mkdir -p '$REMOTE_DIR'"
    scp "$SCRIPT_DIR/tikv.toml" "$SCRIPT_DIR/$(basename "$0")" "$host:$REMOTE_DIR/"
  done

  local pids=()
  for idx in "${!REMOTE_HOST_ARRAY[@]}"; do
    host=${REMOTE_HOST_ARRAY[$idx]}
    ssh -o BatchMode=yes "$host" "DOCKER_CMD='$DOCKER_CMD' bash '$REMOTE_DIR/$(basename "$0")' prepare" &
    pids+=("$!")
  done
  wait_all "${pids[@]}"

  pids=()
  for idx in "${!REMOTE_HOST_ARRAY[@]}"; do
    host=${REMOTE_HOST_ARRAY[$idx]}
    node_addr=${PRIVATE_IP_ARRAY[$idx]}
    ssh -o BatchMode=yes "$host" "NODE_IP='$node_addr' DOCKER_CMD='$DOCKER_CMD' bash '$REMOTE_DIR/$(basename "$0")' start-pd" &
    pids+=("$!")
  done
  wait_all "${pids[@]}"

  pids=()
  for idx in "${!REMOTE_HOST_ARRAY[@]}"; do
    host=${REMOTE_HOST_ARRAY[$idx]}
    node_addr=${PRIVATE_IP_ARRAY[$idx]}
    ssh -o BatchMode=yes "$host" "NODE_IP='$node_addr' DOCKER_CMD='$DOCKER_CMD' bash '$REMOTE_DIR/$(basename "$0")' start-tikv" &
    pids+=("$!")
  done
  wait_all "${pids[@]}"

  ssh -o BatchMode=yes "${REMOTE_HOST_ARRAY[0]}" "DOCKER_CMD='$DOCKER_CMD' bash '$REMOTE_DIR/$(basename "$0")' configure"
  ssh -o BatchMode=yes "${REMOTE_HOST_ARRAY[0]}" "DOCKER_CMD='$DOCKER_CMD' bash '$REMOTE_DIR/$(basename "$0")' status"
}

command=${1:-}
case "$command" in
  prepare) prepare ;;
  start-pd) start_pd ;;
  start-tikv) start_tikv ;;
  configure) configure_cluster ;;
  status) status_local ;;
  stop) stop_local ;;
  wipe) wipe_local ;;
  remote) remote_deploy ;;
  -h|--help|help) usage ;;
  *) usage; exit 1 ;;
esac