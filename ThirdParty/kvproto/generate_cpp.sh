#!/bin/bash
# Generate C++ gRPC stubs from kvproto proto files for SPANN-TiKV integration.
#
# Prerequisites:
#   - protoc (Protocol Buffer compiler)
#   - grpc_cpp_plugin (gRPC C++ plugin)
#
# Usage:
#   cd SPTAG/ThirdParty/kvproto
#   ./generate_cpp.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_DIR="${SCRIPT_DIR}/proto"
OUT_DIR="${SCRIPT_DIR}/generated"

# Find grpc_cpp_plugin
GRPC_CPP_PLUGIN=$(which grpc_cpp_plugin 2>/dev/null || echo "")
if [ -z "$GRPC_CPP_PLUGIN" ]; then
    # Try common locations
    for path in /usr/local/bin/grpc_cpp_plugin /usr/bin/grpc_cpp_plugin; do
        if [ -x "$path" ]; then
            GRPC_CPP_PLUGIN="$path"
            break
        fi
    done
fi

if [ -z "$GRPC_CPP_PLUGIN" ]; then
    echo "Error: grpc_cpp_plugin not found. Install gRPC development tools."
    exit 1
fi

echo "Using grpc_cpp_plugin: $GRPC_CPP_PLUGIN"
echo "Proto dir: $PROTO_DIR"
echo "Output dir: $OUT_DIR"

mkdir -p "$OUT_DIR/kvproto"

# Generate protobuf C++ files
for proto in metapb errorpb kvrpcpb tikvpb pdpb; do
    echo "Generating ${proto}.pb.cc/h ..."
    protoc \
        --proto_path="$PROTO_DIR" \
        --cpp_out="$OUT_DIR/kvproto" \
        "$PROTO_DIR/${proto}.proto"
done

# Regenerate even empty stubs to avoid mixing gRPC generator versions.
for proto in metapb errorpb kvrpcpb tikvpb pdpb; do
    echo "Generating ${proto}.grpc.pb.cc/h ..."
    protoc \
        --proto_path="$PROTO_DIR" \
        --grpc_out="$OUT_DIR/kvproto" \
        --plugin=protoc-gen-grpc="$GRPC_CPP_PLUGIN" \
        "$PROTO_DIR/${proto}.proto"
done

echo ""
echo "Done! Generated files in: $OUT_DIR/kvproto/"
ls -la "$OUT_DIR/kvproto/"
echo ""
echo "Add the following include path to your build:"
echo "  -I${OUT_DIR}"
