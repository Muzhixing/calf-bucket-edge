"""Standard module entry points. Help and release verification perform no motion."""

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(prog="python -m bucket_robot")
    parser.add_argument(
        "command", choices=("track", "mission", "forward", "post", "telemetry", "verify")
    )
    if len(sys.argv) == 1 or sys.argv[1] in ("-h", "--help"):
        parser.print_help()
        return 0
    args = parser.parse_args(sys.argv[1:2])
    rest = sys.argv[2:]
    if args.command == "verify":
        from bucket_robot.runtime.release import ReleaseLayout

        verify = argparse.ArgumentParser(prog="bucket_robot verify")
        verify.add_argument("release_root", type=Path)
        manifest = ReleaseLayout(verify.parse_args(rest).release_root).verify()
        print(
            json.dumps(dict(version=manifest["version"], files=len(manifest["files"]), valid=True))
        )
        return 0
    # Lazy loading keeps verify/help independent of OpenCV and device imports.
    import importlib

    modules = {
        "track": "bucket_robot.runtime.application",
        "mission": "bucket_robot.runtime.application",
        "forward": "bucket_robot.missions.forward",
        "post": "bucket_robot.missions.post_execute",
        "telemetry": "bucket_robot.transport.telemetry",
    }
    sys.argv = [sys.argv[0] + " " + args.command, *rest]
    module = importlib.import_module(modules[args.command])
    return (
        (module.mission_main() if args.command == "mission" else module.track_main())
        if args.command in ("track", "mission")
        else module.main()
    )


if __name__ == "__main__":
    raise SystemExit(main())
