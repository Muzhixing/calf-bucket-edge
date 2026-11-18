#!/bin/sh
# Install the native runtime beside its calibration and capture support package.
# Reuse the board-local dependencies without changing system packages.
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
V2_HOST=${BUCKET_V2_HOST:-elf@192.168.31.175}
SOCK=${BUCKET_SSH_SOCKET:-/tmp/bucket-v1-20261116.sock}
ssh -S "$SOCK" -O check "$V2_HOST" 2>/dev/null || ssh -M -S "$SOCK" -o ControlPersist=2h -Nf "$V2_HOST"
ssh -S "$SOCK" "$V2_HOST" 'mkdir -p /home/elf/deploy/vision_cpp'
# Explicit source allowlist; never transfer recordings, test files or binaries.
rsync -a -e "ssh -S $SOCK" "$HERE"/*.hpp "$HERE/main.cpp" "$HERE/prepare.py" \
  "$HERE/build.sh" "$HERE/build_opencv.sh" "$HERE/run.sh" "$HERE/README.md" "$V2_HOST:/home/elf/deploy/vision_cpp/"
ssh -S "$SOCK" "$V2_HOST" 'set -eu
cd /home/elf/deploy/vision_cpp
if [ ! -f /usr/include/gstreamer-1.0/gst/gst.h ] && [ ! -f deps/headers/usr/include/gstreamer-1.0/gst/gst.h ]; then
  mkdir -p deps
  (cd deps && apt-get download libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev && for p in *.deb; do dpkg-deb -x "$p" headers; done)
fi
if [ -f /usr/include/gstreamer-1.0/gst/gst.h ]; then export GST_INCLUDE_DIR=/usr/include/gstreamer-1.0; fi
python3 prepare.py
./build.sh'
