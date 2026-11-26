#!/bin/sh
set -eu
DEBUG_ROOT=/home/elf/deploy/chassis_debug
export LD_LIBRARY_PATH="$DEBUG_ROOT/runtime/usr/lib/aarch64-linux-gnu:$DEBUG_ROOT/runtime/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$DEBUG_ROOT/runtime/usr/bin/openocd" \
  -s "$DEBUG_ROOT/runtime/usr/share/openocd/scripts" \
  -f "$DEBUG_ROOT/attach_readonly.cfg"
