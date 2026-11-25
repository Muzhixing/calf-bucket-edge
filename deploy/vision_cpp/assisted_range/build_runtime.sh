#!/bin/sh
set -eu
# Directory must contain official rknn_api.h, rknn_custom_op.h and librknnrt.so.
# Keeps system /usr/lib and the detector's runtime unchanged.
RUNTIME_DIR=$(cd "$1" && pwd)
SOURCE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
g++ -O3 -std=c++17 -fPIC -shared -fopenmp -Wall -Wextra \
  -I"$RUNTIME_DIR" "$SOURCE_DIR/fast_acv_runtime.cpp" \
  -L"$RUNTIME_DIR" -Wl,-rpath,'$ORIGIN' -lrknnrt \
  -o "$RUNTIME_DIR/libfastacv.so"
