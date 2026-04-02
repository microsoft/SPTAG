#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
PY310_PYTHON="/home/v-mochengli/anaconda3/envs/py310/bin/python"

if [[ ! -d "$BUILD_DIR" ]]; then
    echo "[ERROR] build directory not found: $BUILD_DIR"
    echo "Run: mkdir -p build && cd build && cmake .."
    exit 1
fi

if [[ ! -x "$PY310_PYTHON" ]]; then
    echo "[ERROR] py310 python not found: $PY310_PYTHON"
    exit 1
fi

echo "[1/2] Build in $BUILD_DIR"
cd "$BUILD_DIR"
make -j

echo "[2/2] Editable install in $PROJECT_ROOT (py310)"
cd "$PROJECT_ROOT"
"$PY310_PYTHON" -m pip install -e .

echo "[DONE] Build + editable install completed."
