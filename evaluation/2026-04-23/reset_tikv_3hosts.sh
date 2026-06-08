#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

DATA_ROOT=${DATA_ROOT:-/mnt/md0/qiazh/tikv}
REMOTE_HOSTS=${REMOTE_HOSTS:-"azureuser@annservicFX071Y azureuser@annservicKFECPH azureuser@annservicP92MC8"}
PRIVATE_IPS=${PRIVATE_IPS:-"annservicFX071Y annservicKFECPH annservicP92MC8"}
REMOTE_DIR=${REMOTE_DIR:-/mnt/md0/qiazh/SPTAG/evaluation/2026-04-23}
PD_IMAGE=${PD_IMAGE:-pingcap/pd:nightly}
TIKV_KEY_PREFIX=${TIKV_KEY_PREFIX:-spfresh_sift1b}
REGION_COUNT=${REGION_COUNT:-128}
WAIT_TIMEOUT_SECONDS=${WAIT_TIMEOUT_SECONDS:-180}
DOCKER_CMD=${DOCKER_CMD:-"sudo -n docker"}

CONFIRM=0
SKIP_PRESPLIT=0

usage() {
  cat <<'USAGE'
Usage: reset_tikv_3hosts.sh --yes [options]

Stops the current 3-node TiKV/PD cluster, deletes each node's TiKV/PD data and
log directories, restarts the cluster, then pre-splits and scatters regions.

Required:
  --yes                      Confirm destructive reset.

Options:
  --hosts "h1 h2 h3"          Override remote SSH hosts.
  --private-ips "ip1 ip2 ip3" Override TiKV/PD advertise addresses.
  --data-root PATH            Override TiKV data root. Default: /mnt/md0/qiazh/tikv
  --remote-dir PATH           Remote directory containing deploy scripts.
  --pd-image IMAGE            PD image used for pd-ctl. Default: pingcap/pd:nightly
  --key-prefix PREFIX         TiKVKeyPrefix used for split keys. Default: spfresh_sift1b
  --regions N                 Target region count after pre-split. Default: 128.
  --wait-timeout SECONDS      Wait for stores after restart. Default: 180
  --skip-presplit             Restart only; do not pre-split/scatter regions.
  -h, --help                  Show this help.

Environment overrides:
  DATA_ROOT, REMOTE_HOSTS, PRIVATE_IPS, REMOTE_DIR, PD_IMAGE,
  TIKV_KEY_PREFIX, REGION_COUNT, WAIT_TIMEOUT_SECONDS, DOCKER_CMD
USAGE
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

quote() {
  printf '%q' "$1"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --yes|-y)
        CONFIRM=1
        shift
        ;;
      --hosts)
        [[ $# -ge 2 ]] || die "--hosts requires a value"
        REMOTE_HOSTS=$2
        shift 2
        ;;
      --private-ips)
        [[ $# -ge 2 ]] || die "--private-ips requires a value"
        PRIVATE_IPS=$2
        shift 2
        ;;
      --data-root)
        [[ $# -ge 2 ]] || die "--data-root requires a value"
        DATA_ROOT=$2
        shift 2
        ;;
      --remote-dir)
        [[ $# -ge 2 ]] || die "--remote-dir requires a value"
        REMOTE_DIR=$2
        shift 2
        ;;
      --pd-image)
        [[ $# -ge 2 ]] || die "--pd-image requires a value"
        PD_IMAGE=$2
        shift 2
        ;;
      --key-prefix)
        [[ $# -ge 2 ]] || die "--key-prefix requires a value"
        TIKV_KEY_PREFIX=$2
        shift 2
        ;;
      --regions)
        [[ $# -ge 2 ]] || die "--regions requires a value"
        REGION_COUNT=$2
        shift 2
        ;;
      --wait-timeout)
        [[ $# -ge 2 ]] || die "--wait-timeout requires a value"
        WAIT_TIMEOUT_SECONDS=$2
        shift 2
        ;;
      --skip-presplit)
        SKIP_PRESPLIT=1
        shift
        ;;
      -h|--help|help)
        usage
        exit 0
        ;;
      *)
        die "unknown argument: $1"
        ;;
    esac
  done
}

wait_for_stores() {
  local pd_host=$1
  local expected=$2
  local deadline=$((SECONDS + WAIT_TIMEOUT_SECONDS))
  local pd_url="http://$pd_host:2379"

  echo "== waiting for $expected TiKV stores at $pd_url =="
  while (( SECONDS < deadline )); do
    if ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "${REMOTE_HOST_ARRAY[0]}" \
      "PD_URL=$(quote "$pd_url") EXPECTED=$(quote "$expected") python3 -" <<'PY'
import json
import os
import sys
import urllib.request

pd_url = os.environ['PD_URL']
expected = int(os.environ['EXPECTED'])
with urllib.request.urlopen(pd_url + '/pd/api/v1/stores', timeout=5) as response:
    stores = json.load(response).get('stores', [])
up = [item for item in stores if item.get('store', {}).get('state_name') == 'Up']
print(f'up_stores={len(up)} total_stores={len(stores)}')
sys.exit(0 if len(up) >= expected else 1)
PY
    then
      return
    fi
    sleep 5
  done

  die "timed out waiting for TiKV stores"
}

presplit_regions() {
  local controller_host=${REMOTE_HOST_ARRAY[0]}
  local pd_host=${PRIVATE_IP_ARRAY[0]}
  local pd_url="http://$pd_host:2379"

  echo "== presplitting regions for prefix '$TIKV_KEY_PREFIX' via $pd_url =="
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$controller_host" \
    "PD_URL=$(quote "$pd_url") PD_IMAGE=$(quote "$PD_IMAGE") TIKV_KEY_PREFIX=$(quote "$TIKV_KEY_PREFIX") REGION_COUNT=$(quote "$REGION_COUNT") DOCKER_CMD=$(quote "$DOCKER_CMD") python3 -" <<'PY'
import json
import os
import subprocess
import sys
import time

pd_url = os.environ['PD_URL']
pd_image = os.environ['PD_IMAGE']
prefix = (os.environ['TIKV_KEY_PREFIX'] + '_').encode('utf-8')
region_count = int(os.environ['REGION_COUNT'])
docker_cmd = os.environ['DOCKER_CMD'].split()

if region_count < 1 or region_count > 256:
    print('REGION_COUNT must be in 1..256', file=sys.stderr)
    sys.exit(2)

pdctl = docker_cmd + ['run', '--rm', '--network', 'host', '--entrypoint', '/pd-ctl', pd_image, '-u', pd_url]

def run(args):
    return subprocess.check_output(pdctl + args, text=True, stderr=subprocess.STDOUT)

def region_for_key(hex_key):
    return json.loads(run(['region', 'key', '--format=hex', hex_key]))['id']

split_ok = 0
split_keys = [(index * 256) // region_count for index in range(1, region_count)]
for index, boundary in enumerate(split_keys, 1):
    key = (prefix + bytes([boundary, 0, 0, 0])).hex()
    region_id = region_for_key(key)
    print(f'split {index}/{len(split_keys)} region={region_id} key={key}', flush=True)
    out = run(['operator', 'add', 'split-region', str(region_id), '--policy=usekey', '--keys', key])
    if 'Success' in out:
        split_ok += 1
    time.sleep(0.05)

regions = json.loads(run(['region', 'scan']))['regions']
print(f'split_ok={split_ok} region_count={len(regions)}', flush=True)
for region in regions:
    region_id = region['id']
    print(f'scatter region={region_id}', flush=True)
    run(['operator', 'add', 'scatter-region', str(region_id)])

count = json.loads(run(['region', 'scan']))['count']
print(f'final_region_count={count}', flush=True)
PY
}

main() {
  parse_args "$@"
  [[ "$CONFIRM" == "1" ]] || die "refusing to reset TiKV without --yes"

  read -r -a REMOTE_HOST_ARRAY <<< "$REMOTE_HOSTS"
  read -r -a PRIVATE_IP_ARRAY <<< "$PRIVATE_IPS"
  [[ ${#REMOTE_HOST_ARRAY[@]} -eq 3 ]] || die "REMOTE_HOSTS must contain exactly 3 hosts"
  [[ ${#PRIVATE_IP_ARRAY[@]} -eq 3 ]] || die "PRIVATE_IPS must contain exactly 3 addresses"
  [[ -x "$SCRIPT_DIR/wipe_tikv_3hosts.sh" ]] || die "missing executable $SCRIPT_DIR/wipe_tikv_3hosts.sh"
  [[ -x "$SCRIPT_DIR/deploy_tikv_3hosts.sh" ]] || die "missing executable $SCRIPT_DIR/deploy_tikv_3hosts.sh"

  echo "DATA_ROOT=$DATA_ROOT"
  echo "REMOTE_HOSTS=${REMOTE_HOST_ARRAY[*]}"
  echo "PRIVATE_IPS=${PRIVATE_IP_ARRAY[*]}"
  echo "TIKV_KEY_PREFIX=$TIKV_KEY_PREFIX"
  echo "REGION_COUNT=$REGION_COUNT"

  echo "== stopping cluster and deleting data/log directories =="
  DATA_ROOT="$DATA_ROOT" REMOTE_HOSTS="$REMOTE_HOSTS" DOCKER_CMD="$DOCKER_CMD" \
    "$SCRIPT_DIR/wipe_tikv_3hosts.sh" --yes

  echo "== restarting cluster =="
  DATA_ROOT="$DATA_ROOT" REMOTE_HOSTS="$REMOTE_HOSTS" PRIVATE_IPS="$PRIVATE_IPS" REMOTE_DIR="$REMOTE_DIR" PD_IMAGE="$PD_IMAGE" DOCKER_CMD="$DOCKER_CMD" \
    "$SCRIPT_DIR/deploy_tikv_3hosts.sh" remote

  wait_for_stores "${PRIVATE_IP_ARRAY[0]}" "${#PRIVATE_IP_ARRAY[@]}"

  if [[ "$SKIP_PRESPLIT" == "0" ]]; then
    presplit_regions
  fi

  echo "== reset complete =="
}

main "$@"