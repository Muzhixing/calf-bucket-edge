#!/bin/sh
# Build only; no hardware devices, host networking or board access.
set -eu
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 /path/to/rknn-sdk /path/to/build-output" >&2
  exit 2
fi
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
SDK=$(CDPATH= cd -- "$1" && pwd)
test -f "$SDK/rknn_api.h"
test -f "$SDK/librknnrt.so"
mkdir -p "$2"
OUTPUT=$(CDPATH= cd -- "$2" && pwd)
docker run --rm --platform linux/arm64 --network none \
  --mount "type=bind,src=$ROOT,dst=/src,readonly" \
  --mount "type=bind,src=$SDK,dst=/sdk,readonly" \
  --mount "type=bind,src=$OUTPUT,dst=/build" \
  bucket-arm64-build:opencv-4.10 sh -eu -c '
    cmake -S /src/deploy/vision_cpp -B /build/native -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE=/src/tools/build/linux-aarch64.cmake \
      -DOPENCV_PREFIX=/opt/opencv -DRKNN_SDK_DIR=/sdk \
      -DCMAKE_LIBRARY_PATH=/sdk -DGST_INCLUDE_DIR=/usr/include/gstreamer-1.0
    cmake --build /build/native --parallel 2
    cmake --install /build/native --prefix /build/package
    file /build/package/vision
    readelf -h /build/package/vision
    readelf -d /build/package/vision
    readelf --version-info /build/package/vision
    dpkg-query -W > /build/build-packages.tsv
  '
