#!/bin/sh
set -eu
cd "$(dirname "$0")"
exec ../vision_cpp/run.sh --hdmi --id "hdmi_$(date +%Y%m%d_%H%M%S)_$$" "$@"
