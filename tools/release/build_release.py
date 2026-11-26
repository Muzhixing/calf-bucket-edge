#!/usr/bin/env python3
"""Assemble an immutable release from explicit inputs; never start hardware."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def build(args):
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    provenance = {}

    def copy(source, relative):
        source = Path(source)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target, follow_symlinks=True)
        provenance[str(relative)] = str(source.resolve())

    with zipfile.ZipFile(args.wheel) as wheel:
        for item in wheel.infolist():
            rel = Path(item.filename)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("invalid_wheel_path")
            if not item.is_dir():
                target = output / "python" / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(wheel.read(item))
    copy(args.native_dir / "vision", "native/vision")
    for p in sorted((args.native_dir / "lib").glob("*.so*")):
        if p.is_file():
            copy(p, Path("native/lib") / p.name)
    for p in sorted(args.assets_dir.iterdir()):
        if p.is_file() and p.suffix in (".png", ".jpg", ".html", ".yml"):
            copy(p, Path("assets") / p.name)
    copy(args.detector, "assets/detector.rknn")
    copy(args.range_model, "assets/fast_acv.rknn")
    field_tools = getattr(args, "field_tools_dir", None)
    if field_tools is not None:
        for name in ("start_full_mission.py", "initialize_origins.py", "startup_policy.json"):
            copy(field_tools / name, Path("field") / name)
    copy(args.calibration, "assets/calibration.json")
    for name in ("libfastacv.so", "librknnrt.so"):
        copy(args.range_runtime / name, Path("assisted_runtime") / name)
    source_dir = getattr(args, "native_source_dir", None) or args.native_dir
    for name in (
        "main.cpp",
        "core.hpp",
        "display.hpp",
        "score_order.hpp",
        "source.hpp",
        "worker_channel.hpp",
        "detector_process.hpp",
        "build.sh",
        "CMakeLists.txt",
    ):
        if (source_dir / name).exists():
            copy(source_dir / name, Path("native/source") / name)
    (output / "build_provenance.json").write_text(
        json.dumps(
            {
                "source_wheel": args.wheel.name,
                "wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
                "inputs": provenance,
                "native_source_relation": args.native_source_relation,
            },
            indent=2,
        )
        + "\n"
    )
    common = """#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="$ROOT/python" PYTHONDONTWRITEBYTECODE=1
"""
    (output / "run_mission.sh").write_text(
        common
        + """exec python3 -m bucket_robot mission --release-root "$ROOT" "$@"
"""
    )
    (output / "run_track.sh").write_text(
        common
        + """export LD_LIBRARY_PATH="$ROOT/native/lib"
ID="live_$(date +%Y%m%d_%H%M%S)_$$"
RUN_ROOT=${BUCKET_RUN_ROOT:-${XDG_STATE_HOME:-$HOME/.local/state}/bucket/runs}
mkdir -p "$RUN_ROOT"
exec python3 -m bucket_robot track --release-root "$ROOT" --native "$ROOT/native/vision" --model "$ROOT/assets/detector.rknn" --calibration "$ROOT/assets/calibration.yml" --calibration-sha "$(sha256sum "$ROOT/assets/calibration.yml" 2>/dev/null | cut -d ' ' -f 1)" --assets "$ROOT/assets" --output "$RUN_ROOT/$ID" --id "$ID" --seconds 3600 --cores 0_1_2 "$@"
"""
    )
    for script in ("run_mission.sh", "run_track.sh"):
        (output / script).chmod(0o755)
    entries = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            data = path.read_bytes()
            entries.append(
                dict(
                    path=str(path.relative_to(output)),
                    bytes=len(data),
                    sha256=hashlib.sha256(data).hexdigest(),
                )
            )
    (output / "manifest.json").write_text(
        json.dumps(dict(schema_version=1, version=args.version, files=entries), indent=2) + "\n"
    )
    print(json.dumps(dict(output=str(output), files=len(entries), version=args.version)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "wheel",
        "native-dir",
        "assets-dir",
        "detector",
        "range-model",
        "calibration",
        "range-runtime",
        "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--native-source-dir", type=Path)
    parser.add_argument("--field-tools-dir", type=Path)
    parser.add_argument(
        "--native-source-relation",
        choices=("built_from_supplied_source", "binary_only"),
        default="binary_only",
    )
    build(parser.parse_args())


if __name__ == "__main__":
    main()
