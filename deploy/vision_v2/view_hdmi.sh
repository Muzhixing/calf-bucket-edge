#!/bin/sh
set -eu
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec python3 -u tools/live_viewer.py --hdmi --id "hdmi_$(date +%Y%m%d_%H%M%S)_$$" "$@"
