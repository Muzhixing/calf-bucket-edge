#!/bin/sh
set -eu
cd "$(dirname "$0")"
BASE=$(pwd)
SDK=${RKNN_SDK_DIR:-/home/elf/research/rk3588_regression_crossseed_20261029/package}
g++ -O3 -std=c++17 -Wall -Wextra -pthread vision.cpp -I"$SDK" \
  -I"$BASE/deps/install/include/opencv4" -L"$BASE/deps/install/lib" \
  -Wl,-rpath,"$BASE/deps/install/lib" -lopencv_calib3d -lopencv_imgcodecs -lopencv_imgproc -lopencv_core \
  -L/usr/lib -lrknnrt -o vision
