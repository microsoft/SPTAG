#!/bin/bash
# Single-node backend comparison benchmark runner.
#
# Usage:
#   ./run_benchmark.sh <ini_file> [output_file]
#
# Example:
#   ./run_benchmark.sh benchmark_1b_rocksdb_L2.ini results/rocksdb_L2.json
#
# Prerequisites:
#   - SPTAGTest binary at ../../Release/SPTAGTest (relative to this script)
#     or set SPTAG_BINARY env var to override
#   - SIFT1B data at /mnt_ssd/data/sift1b/ (or as specified in INI)
#   - For TiKV tests: PD + TiKV running (see setup_tikv.sh section below)
#   - For RocksDB tests: RocksDB 8.11.3 + Folly + liburing 2.6 linked in binary
#
# Output:
#   - <output_file>.json  — structured benchmark results
#   - <output_file>.log   — full console log
#
# The script automatically creates the IndexPath directory and perftest working dir.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ─── Arguments ───

INI_FILE="${1:?Usage: $0 <ini_file> [output_file]}"
if [[ ! -f "$SCRIPT_DIR/$INI_FILE" && ! -f "$INI_FILE" ]]; then
    echo "ERROR: INI file not found: $INI_FILE"
    exit 1
fi
# Resolve to absolute path
if [[ -f "$SCRIPT_DIR/$INI_FILE" ]]; then
    INI_FILE="$SCRIPT_DIR/$INI_FILE"
else
    INI_FILE="$(cd "$(dirname "$INI_FILE")" && pwd)/$(basename "$INI_FILE")"
fi

# Output file: default to results/<ini_basename>.json
INI_BASENAME="$(basename "$INI_FILE" .ini)"
OUTPUT_FILE="${2:-$SCRIPT_DIR/results/${INI_BASENAME}.json}"
# Resolve to absolute path (needed because we cd to WORK_DIR later)
if [[ "$OUTPUT_FILE" != /* ]]; then
    OUTPUT_FILE="$(pwd)/$OUTPUT_FILE"
fi
LOG_FILE="${OUTPUT_FILE%.json}.log"

# ─── Binary ───

SPTAG_BINARY="${SPTAG_BINARY:-$SCRIPT_DIR/../../Release/SPTAGTest}"
if [[ ! -x "$SPTAG_BINARY" ]]; then
    echo "ERROR: SPTAGTest binary not found at $SPTAG_BINARY"
    echo "  Set SPTAG_BINARY env var or build with: cd Release && cmake -DROCKSDB=ON -DTIKV=ON -DURING=ON .. && make -j SPTAGTest"
    exit 1
fi

# ─── Parse key paths from INI ───

get_ini_val() {
    grep -m1 "^$1=" "$INI_FILE" | cut -d= -f2- | tr -d '[:space:]'
}

INDEX_PATH="$(get_ini_val IndexPath)"
STORAGE="$(get_ini_val Storage)"
LAYERS="$(get_ini_val Layers)"
BASE_COUNT="$(get_ini_val BaseVectorCount)"
INSERT_COUNT="$(get_ini_val InsertVectorCount)"

echo "══════════════════════════════════════════════════════"
echo "  Backend Comparison Benchmark"
echo "══════════════════════════════════════════════════════"
echo "  Config:    $INI_FILE"
echo "  Storage:   $STORAGE"
echo "  Layers:    $LAYERS"
echo "  Scale:     ${BASE_COUNT}+${INSERT_COUNT}"
echo "  Output:    $OUTPUT_FILE"
echo "  Log:       $LOG_FILE"
echo "══════════════════════════════════════════════════════"

# ─── Prepare directories ───

mkdir -p "$(dirname "$OUTPUT_FILE")"
mkdir -p "$INDEX_PATH" 2>/dev/null || true

# Working directory: where perftest_* files are generated
# Must be writable and have enough space for intermediate data
WORK_DIR="$(dirname "$INDEX_PATH")"
mkdir -p "$WORK_DIR"

# ─── TiKV auto-deploy ───

if [[ "$STORAGE" == "TIKVIO" ]]; then
    PD_ADDR="$(get_ini_val TiKVPDAddresses)"
    PD_HOST="${PD_ADDR%%:*}"
    PD_PORT="${PD_ADDR##*:}"
    TIKV_VERSION="${TIKV_VERSION:-v8.5.1}"
    TIKV_DATA_DIR="${TIKV_DATA_DIR:-/mnt_ssd/data/tikv_standalone}"

    echo "Checking TiKV PD at $PD_ADDR ..."
    if curl -s --connect-timeout 3 "http://$PD_ADDR/pd/api/v1/stores" > /dev/null 2>&1; then
        echo "  PD already running"
    else
        echo "  PD not running — starting TiKV stack (${TIKV_VERSION}) ..."

        # Stop stale containers if any
        docker rm -f pd-standalone tikv-standalone 2>/dev/null || true

        # Start PD
        echo "  Starting PD ..."
        docker run -d --name pd-standalone --net=host \
            -v "${TIKV_DATA_DIR}/pd:/data" \
            "pingcap/pd:${TIKV_VERSION}" \
            --name=pd-standalone \
            --data-dir=/data \
            --client-urls="http://0.0.0.0:${PD_PORT}" \
            --advertise-client-urls="http://127.0.0.1:${PD_PORT}" \
            --peer-urls="http://0.0.0.0:2380" \
            --advertise-peer-urls="http://127.0.0.1:2380"

        # Wait for PD to be ready
        echo -n "  Waiting for PD "
        for i in $(seq 1 30); do
            if curl -s --connect-timeout 2 "http://127.0.0.1:${PD_PORT}/pd/api/v1/members" > /dev/null 2>&1; then
                echo " ready"
                break
            fi
            echo -n "."
            sleep 1
        done

        # Start TiKV (with tuning config if available)
        echo "  Starting TiKV ..."
        TIKV_DOCKER_ARGS=(-d --name tikv-standalone --net=host -v "${TIKV_DATA_DIR}/tikv:/data")
        TIKV_CMD_ARGS=(--addr="0.0.0.0:20160" --advertise-addr="127.0.0.1:20160" --status-addr="0.0.0.0:20180" --pd-endpoints="http://127.0.0.1:${PD_PORT}" --data-dir=/data)
        TIKV_TOML="${SCRIPT_DIR}/tikv.toml"
        if [[ -f "$TIKV_TOML" ]]; then
            echo "  Using TiKV config: $TIKV_TOML"
            TIKV_DOCKER_ARGS+=(-v "${TIKV_TOML}:/tikv.toml:ro")
            TIKV_CMD_ARGS+=(--config=/tikv.toml)
        fi
        docker run "${TIKV_DOCKER_ARGS[@]}" "pingcap/tikv:${TIKV_VERSION}" "${TIKV_CMD_ARGS[@]}"

        # Wait for TiKV store to register
        echo -n "  Waiting for TiKV store "
        for i in $(seq 1 60); do
            STORE_COUNT=$(curl -s "http://127.0.0.1:${PD_PORT}/pd/api/v1/stores" 2>/dev/null | grep -c '"state_name"' || true)
            if [[ "$STORE_COUNT" -ge 1 ]]; then
                echo " registered (${STORE_COUNT} store(s))"
                break
            fi
            echo -n "."
            sleep 2
        done

        # Set max-replicas=1 for single-node
        echo "  Setting max-replicas=1 ..."
        curl -s "http://127.0.0.1:${PD_PORT}/pd/api/v1/config/replicate" \
            -X POST -d '{"max-replicas": 1}' > /dev/null
        echo "  TiKV stack ready"
    fi
    echo ""
fi

# ─── Run benchmark ───

echo ""
echo "Starting benchmark at $(date '+%Y-%m-%d %H:%M:%S') ..."
echo ""

cd "$WORK_DIR"

BENCHMARK_CONFIG="$INI_FILE" \
BENCHMARK_OUTPUT="$OUTPUT_FILE" \
"$SPTAG_BINARY" --run_test=SPFreshTest/BenchmarkFromConfig --log_level=message 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "══════════════════════════════════════════════════════"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "  Benchmark completed successfully"
else
    echo "  Benchmark FAILED (exit code: $EXIT_CODE)"
fi
echo "  Output:  $OUTPUT_FILE"
echo "  Log:     $LOG_FILE"
echo "  Finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "══════════════════════════════════════════════════════"

exit $EXIT_CODE
