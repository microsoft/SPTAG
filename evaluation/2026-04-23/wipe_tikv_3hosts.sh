#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/mnt/md0/qiazh/tikv}
REMOTE_HOSTS=${REMOTE_HOSTS:-"azureuser@annservicFX071Y azureuser@annservicKFECPH azureuser@annservicP92MC8"}
DOCKER_CMD=${DOCKER_CMD:-"sudo -n docker"}
WIPE_LOGS=${WIPE_LOGS:-1}

CONFIRM=0
LOCAL_ONLY=0

usage() {
  cat <<'USAGE'
Usage: wipe_tikv_3hosts.sh --yes [options]

Stops TiKV/PD containers and removes TiKV/PD data directories.
By default this runs on the three benchmark hosts.

Required:
  --yes                 Confirm destructive wipe.

Options:
  --local               Wipe only the current host.
  --hosts "h1 h2 h3"    Override remote SSH hosts.
  --data-root PATH      Override TiKV data root. Default: /mnt/md0/qiazh/tikv
  --docker-cmd CMD      Override docker command. Default: sudo -n docker
  --keep-logs           Wipe data directories only; keep logs.
  -h, --help            Show this help.

Environment overrides:
  DATA_ROOT=/mnt/md0/qiazh/tikv
  REMOTE_HOSTS="azureuser@annservicFX071Y azureuser@annservicKFECPH azureuser@annservicP92MC8"
  DOCKER_CMD="sudo -n docker"
  WIPE_LOGS=1

This removes:
  $DATA_ROOT/pd-data
  $DATA_ROOT/tikv-data
  $DATA_ROOT/pd-logs    when WIPE_LOGS=1
  $DATA_ROOT/tikv-logs  when WIPE_LOGS=1
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
      --local)
        LOCAL_ONLY=1
        shift
        ;;
      --hosts)
        [[ $# -ge 2 ]] || die "--hosts requires a value"
        REMOTE_HOSTS=$2
        shift 2
        ;;
      --data-root)
        [[ $# -ge 2 ]] || die "--data-root requires a value"
        DATA_ROOT=$2
        shift 2
        ;;
      --docker-cmd)
        [[ $# -ge 2 ]] || die "--docker-cmd requires a value"
        DOCKER_CMD=$2
        shift 2
        ;;
      --keep-logs)
        WIPE_LOGS=0
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

wipe_current_host() {
  echo "== $(hostname -f 2>/dev/null || hostname): stopping TiKV/PD containers =="
  read -r -a docker_cmd <<< "$DOCKER_CMD"
  "${docker_cmd[@]}" rm -f tikv-pd tikv-server >/dev/null 2>&1 || true

  echo "== $(hostname -f 2>/dev/null || hostname): wiping data under $DATA_ROOT =="
  sudo -n rm -rf "$DATA_ROOT/pd-data" "$DATA_ROOT/tikv-data"
  if [[ "$WIPE_LOGS" == "1" ]]; then
    sudo -n rm -rf "$DATA_ROOT/pd-logs" "$DATA_ROOT/tikv-logs"
  fi

  mkdir -p "$DATA_ROOT/pd-data" "$DATA_ROOT/tikv-data" "$DATA_ROOT/pd-logs" "$DATA_ROOT/tikv-logs"
  echo "== $(hostname -f 2>/dev/null || hostname): wipe complete =="
}

wipe_remote_host() {
  local host=$1
  echo "== $host: wiping TiKV/PD data =="
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$host" \
    "DATA_ROOT=$(quote "$DATA_ROOT") DOCKER_CMD=$(quote "$DOCKER_CMD") WIPE_LOGS=$(quote "$WIPE_LOGS") bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail

echo "== $(hostname -f 2>/dev/null || hostname): stopping TiKV/PD containers =="
read -r -a docker_cmd <<< "$DOCKER_CMD"
"${docker_cmd[@]}" rm -f tikv-pd tikv-server >/dev/null 2>&1 || true

echo "== $(hostname -f 2>/dev/null || hostname): wiping data under $DATA_ROOT =="
sudo -n rm -rf "$DATA_ROOT/pd-data" "$DATA_ROOT/tikv-data"
if [[ "$WIPE_LOGS" == "1" ]]; then
  sudo -n rm -rf "$DATA_ROOT/pd-logs" "$DATA_ROOT/tikv-logs"
fi

mkdir -p "$DATA_ROOT/pd-data" "$DATA_ROOT/tikv-data" "$DATA_ROOT/pd-logs" "$DATA_ROOT/tikv-logs"
echo "== $(hostname -f 2>/dev/null || hostname): wipe complete =="
REMOTE_SCRIPT
}

main() {
  parse_args "$@"
  [[ "$CONFIRM" == "1" ]] || die "refusing to wipe TiKV data without --yes"

  echo "DATA_ROOT=$DATA_ROOT"
  echo "DOCKER_CMD=$DOCKER_CMD"
  echo "WIPE_LOGS=$WIPE_LOGS"

  if [[ "$LOCAL_ONLY" == "1" ]]; then
    wipe_current_host
    return
  fi

  read -r -a hosts <<< "$REMOTE_HOSTS"
  [[ ${#hosts[@]} -gt 0 ]] || die "REMOTE_HOSTS is empty"
  echo "REMOTE_HOSTS=${hosts[*]}"

  local host
  for host in "${hosts[@]}"; do
    wipe_remote_host "$host"
  done
}

main "$@"