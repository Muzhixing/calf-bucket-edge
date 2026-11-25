#!/bin/sh
# Deploy a complete, verified release; never update partial source lists.
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$HERE/../../tools/release/deploy_release.py" "$@"
