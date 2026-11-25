#!/bin/sh
# Compatibility launcher; a complete release supplies code and all assets.
set -eu
ROOT=${BUCKET_RELEASE_ROOT:?Set BUCKET_RELEASE_ROOT to a verified release directory}
export PYTHONPATH="$ROOT/python" PYTHONDONTWRITEBYTECODE=1
export BUCKET_ACV_BRIDGE="$ROOT/assisted_runtime/libfastacv.so" BUCKET_ACV_NPU_CORE=2
export LD_LIBRARY_PATH="$ROOT/assisted_runtime" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
exec python3 -m bucket_robot forward --recover-loss --track-b --post-approach-plan \
  --reference "$ROOT/python/bucket_robot/resources/arm_reference.json" \
  --left-reference "$ROOT/python/bucket_robot/resources/left_view_reference.json" \
  --calibration "$ROOT/assets/calibration.yml" \
  --approach-model "$ROOT/assets/fast_acv.rknn" \
  --approach-calibration "$ROOT/assets/calibration.json" \
  --approach-surface-band .35 .78 .65 .95 "$@"
