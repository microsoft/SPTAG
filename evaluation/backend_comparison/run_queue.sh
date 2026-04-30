#!/bin/bash
# Sequential benchmark queue runner.
# Usage: run_queue.sh <queue_file>
# Each line in queue_file is an INI filename (relative to backend_comparison/).
# Runs them sequentially; cleans /mnt_ssd/data/proidx_<scale>_<backend>_<L?> between runs.
# Use_sg_docker=1 wraps invocation with `sg docker -c` (needed on 0.7 for TiKV).

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QUEUE_FILE="${1:?Usage: $0 <queue_file>}"
USE_SG_DOCKER="${USE_SG_DOCKER:-0}"
LOG_DIR="$SCRIPT_DIR/results"
QUEUE_LOG="$LOG_DIR/queue.log"

mkdir -p "$LOG_DIR"
echo "[$(date -Is)] queue runner starting; queue=$QUEUE_FILE; sg_docker=$USE_SG_DOCKER" >> "$QUEUE_LOG"

while IFS= read -r ini; do
  [[ -z "$ini" || "$ini" =~ ^# ]] && continue
  name="${ini%.ini}"
  echo "[$(date -Is)] === RUN $ini ===" >> "$QUEUE_LOG"

  # Derive index dir from the INI's IndexPath
  idxpath=$(grep -E '^IndexPath=' "$SCRIPT_DIR/$ini" | head -1 | cut -d= -f2)
  idxdir=$(dirname "$idxpath")
  if [[ -d "$idxdir" ]]; then
    echo "[$(date -Is)] cleaning $idxdir" >> "$QUEUE_LOG"
    rm -rf "$idxdir"
  fi

  if [[ "$USE_SG_DOCKER" = "1" ]]; then
    sg docker -c "bash $SCRIPT_DIR/run_benchmark.sh $ini" >> "$LOG_DIR/${name}.runner.log" 2>&1
  else
    bash "$SCRIPT_DIR/run_benchmark.sh" "$ini" >> "$LOG_DIR/${name}.runner.log" 2>&1
  fi
  rc=$?
  echo "[$(date -Is)] $ini exited rc=$rc" >> "$QUEUE_LOG"
  # Continue regardless of rc — record and proceed
done < "$QUEUE_FILE"

echo "[$(date -Is)] queue runner done" >> "$QUEUE_LOG"
