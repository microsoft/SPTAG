#!/bin/bash
# Wait until SPTAGTest is no longer running, then run queue.
# Usage: wait_then_queue.sh <queue_file>
# Polls every 60s.

set -u
QUEUE="${1:?Usage: $0 <queue_file>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$SCRIPT_DIR/results/queue.log"

mkdir -p "$SCRIPT_DIR/results"
echo "[$(date -Is)] wait_then_queue: waiting for SPTAGTest to exit before running $QUEUE" >> "$LOG"

while pgrep -f "Release/SPTAGTest" > /dev/null; do
  sleep 60
done

echo "[$(date -Is)] wait_then_queue: SPTAGTest gone; starting queue $QUEUE" >> "$LOG"
exec bash "$SCRIPT_DIR/run_queue.sh" "$QUEUE"
