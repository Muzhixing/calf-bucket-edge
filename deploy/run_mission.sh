#!/bin/sh
set -eu
HERE=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
exec python3 -m bucket_robot mission "$@"
