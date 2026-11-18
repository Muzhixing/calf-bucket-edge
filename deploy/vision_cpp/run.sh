#!/bin/sh
set -eu
cd "$(dirname "$0")"
BASE=$(pwd)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="$BASE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Pass user options last; the native parser validates ids, scale and duration.
ID="live_$(date +%Y%m%d_%H%M%S)_$$"
# Match the output folder to a user-supplied --id without evaluating shell text.
next_id=false
for arg in "$@"; do
    if "$next_id"; then ID=$arg; next_id=false; elif [ "$arg" = --id ]; then next_id=true; fi
done
case "$ID" in ''|*[!A-Za-z0-9_.-]*|.|..) echo 'Invalid run id' >&2; exit 2;; esac
exec "$BASE/vision" --model "$BASE/assets/seed2_REG_FP16.rknn" \
  --calibration "$BASE/assets/calibration.yml" --calibration-sha "$(cat calibration.sha256)" \
  --assets "$BASE/assets" --output "$BASE/runs/$ID" --id "$ID" "$@"
