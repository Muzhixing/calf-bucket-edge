#!/usr/bin/env python3
"""Upload a verified release into a new directory, without activating it."""

import argparse
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import tempfile

from bucket_robot.runtime.release import ReleaseLayout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.@-]+", args.host):
        parser.error("use an SSH hostname or configured alias")
    if not args.destination.is_absolute():
        parser.error("destination must be an absolute new directory")
    ReleaseLayout(args.release_dir).verify()
    target = shlex.quote(str(args.destination))
    # Destination creation deliberately fails if it already exists.
    subprocess.run(
        [
            "ssh",
            args.host,
            "mkdir -p " + shlex.quote(str(args.destination.parent)) + " && mkdir " + target,
        ],
        check=True,
    )
    with tempfile.TemporaryFile() as stream:
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            for path in sorted(args.release_dir.iterdir()):
                archive.add(path, arcname=path.name)
        stream.seek(0)
        subprocess.run(["ssh", args.host, "tar -xzf - -C " + target], stdin=stream, check=True)
    verify = (
        "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="
        + shlex.quote(str(args.destination / "python"))
        + " python3 -m bucket_robot verify "
        + target
    )
    subprocess.run(["ssh", args.host, verify], check=True)
    print("Verified release at " + str(args.destination) + "; existing entry points are unchanged.")


if __name__ == "__main__":
    main()
