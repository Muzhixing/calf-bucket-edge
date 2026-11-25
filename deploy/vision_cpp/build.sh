#!/bin/sh
# Standard CMake configure/build/install; this script keeps the legacy output paths.
set -eu
BASE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BUILD=${BUCKET_NATIVE_BUILD_DIR:-$BASE/build}
CMAKE_BIN=${BUCKET_CMAKE:-cmake}
"$CMAKE_BIN" -S "$BASE" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DOPENCV_PREFIX="${OPENCV_PREFIX:-$BASE/deps/install}" \
  -DRKNN_SDK_DIR="${RKNN_SDK_DIR:-$BASE/deps/rknn}" \
  -DGST_INCLUDE_DIR="${GST_INCLUDE_DIR:-$BASE/deps/headers/usr/include/gstreamer-1.0}"
"$CMAKE_BIN" --build "$BUILD" --parallel "${BUCKET_BUILD_JOBS:-1}"
"$CMAKE_BIN" --install "$BUILD" --prefix "$BUILD/package"
mkdir -p "$BASE/lib"
for library in "$BUILD/package/lib/"*.so.*; do
  name=$(basename "$library")
  cp "$library" "$BASE/lib/$name.new"
  mv -f "$BASE/lib/$name.new" "$BASE/lib/$name"
done
cp "$BUILD/package/vision" "$BASE/.vision.new"
mv -f "$BASE/.vision.new" "$BASE/vision"
