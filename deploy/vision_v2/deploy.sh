#!/bin/sh
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
V2_HOST=${BUCKET_V2_HOST:-elf@192.168.31.175}
SOCK=${BUCKET_SSH_SOCKET:-/tmp/bucket-v1-20261116.sock}
ssh -S "$SOCK" -O check "$V2_HOST" 2>/dev/null || ssh -M -S "$SOCK" -o ControlPersist=2h -Nf "$V2_HOST"
ssh -S "$SOCK" "$V2_HOST" 'mkdir -p /home/elf/deploy/vision_v2/captures /home/elf/deploy/vision_v2/results /home/elf/deploy/vision_v2/logs /home/elf/deploy/vision_v2/trial_records'
rsync -a --exclude __pycache__ --exclude '*.pyc' --exclude .DS_Store --exclude captures --exclude results --exclude logs --exclude trial_records \
 -e "ssh -S $SOCK" "$HERE/" "$V2_HOST:/home/elf/deploy/vision_v2/"
ssh -S "$SOCK" "$V2_HOST" 'cd /home/elf/deploy/vision_v2 && python3 tools/verify.py'

# Build the native live backend beside the preserved Python runtime.
"$HERE/../vision_cpp/deploy.sh"
