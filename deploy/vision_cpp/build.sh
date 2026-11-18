#!/bin/sh
set -eu
cd "$(dirname "$0")"
BASE=$(pwd)
SDK=${RKNN_SDK_DIR:-/home/elf/research/rk3588_regression_crossseed_20261029/package}
CV=${OPENCV_PREFIX:-/home/elf/experiments/cpp_vision/deps/install}
GST_HEADERS=${GST_INCLUDE_DIR:-$BASE/deps/headers/usr/include/gstreamer-1.0}
mkdir -p lib
# Link to local copies and give the runtime a deployment-local library search path.
for name in core imgproc imgcodecs calib3d features2d flann videoio; do
    cp -L "$CV/lib/libopencv_$name.so.4.10.0" "lib/libopencv_$name.so.410.new"
    mv -f "lib/libopencv_$name.so.410.new" "lib/libopencv_$name.so.410"
done
g++ -O3 -std=c++17 -Wall -Wextra -Wno-deprecated-declarations -pthread main.cpp \
  -I"$SDK" -I"$CV/include/opencv4" -I"$GST_HEADERS" \
  $(pkg-config --cflags glib-2.0 gobject-2.0 libdrm) \
  -L"$CV/lib" -Wl,-rpath,'$ORIGIN/lib' \
  -lopencv_calib3d -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -lopencv_videoio \
  -L/usr/lib -lrknnrt -lcrypto \
  -l:libgstapp-1.0.so.0 -l:libgstreamer-1.0.so.0 \
  $(pkg-config --libs glib-2.0 gobject-2.0 libdrm) -o .vision.new
mv -f .vision.new vision
