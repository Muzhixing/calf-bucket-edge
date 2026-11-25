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
# Optional CSRT uses an isolated dependency bundle; ordinary runs retain their
# existing Python/OpenCV environment. The native detector libraries are unchanged.
for arg in "$@"; do
  if [ "$arg" = --csrt ] && [ -d "$BASE/python_deps/cv2" ]; then
    export PYTHONPATH="$BASE/python_deps${PYTHONPATH:+:$PYTHONPATH}"
    break
  fi
done
exec python3 -m bucket_robot track --native "$BASE/vision" --model "$BASE/assets/B1_C0_REG_FP16.rknn" \
  --calibration "$BASE/assets/calibration.yml" --calibration-sha "$(cat calibration.sha256)" \
  --cores 0_1_2 --assets "$BASE/assets" --output "$BASE/runs/$ID" --id "$ID" "$@"
