"""Collect a reviewed, bounded sequence; stop on any incomplete stereo board."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--route", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--isolated-chassis-pump", action="store_true", required=True)
    args = p.parse_args()
    route = json.loads(args.route.read_text())
    base = Path(__file__).resolve().parent
    from arm_step import CalibrationLimits
    from bucket_robot.resources import default_reference

    limits = CalibrationLimits(default_reference())
    current = route["start"]
    limits.check(current)
    for item in route["poses"]:
        target = item["pose"]
        limits.check(target)
        if sum(a != b for a, b in zip(current, target)) != 1:
            raise ValueError("Each waypoint must change exactly one axis")
        if item["role"] not in ("fit", "holdout", "pilot"):
            raise ValueError("Invalid role")
        current = target
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "route.json").write_text(json.dumps(route, indent=2) + "\n")
    (args.out / "source_sha256.json").write_text(
        json.dumps(
            {
                s: hashlib.sha256((base / s).read_bytes()).hexdigest()
                for s in ("arm_step.py", "capture_pair.py", "collect_route.py")
            },
            indent=2,
        )
        + "\n"
    )
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    current = route["start"]
    step = 0
    for index, item in enumerate(route["poses"]):
        target = item["pose"]
        while current != target:
            axis = next(i for i in range(3) if current[i] != target[i])
            limit = (2, 10, 5)[axis]
            next_pose = current.copy()
            next_pose[axis] += max(-limit, min(limit, target[axis] - current[axis]))
            move_dir = args.out / f"move_{step:03d}"
            subprocess.run(
                [
                    sys.executable,
                    str(base / "arm_step.py"),
                    "--isolated-chassis-pump",
                    "--expect",
                    *map(str, current),
                    "--target",
                    *map(str, next_pose),
                    "--out",
                    str(move_dir),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            current = next_pose
            step += 1
        pose_dir = (
            args.out / f"{index:03d}_{item['role']}_A{current[0]}_B{current[1]}_C{current[2]}"
        )
        subprocess.run(
            [
                sys.executable,
                str(base / "capture_pair.py"),
                "--out",
                str(pose_dir),
                "--role",
                item["role"],
                "--pairs",
                str(item.get("pairs", 1)),
            ],
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        manifest = json.loads((pose_dir / "capture.json").read_text())
        manifest["arm_telemetry"] = json.loads((move_dir / "result.json").read_text())["telemetry"]
        manifest["board_orientation_group"] = route["board_orientation_group"]
        (pose_dir / "capture.json").write_text(json.dumps(manifest, indent=2) + "\n")
        good = all(f["eyes"][e]["found"] for f in manifest["frames"] for e in ("left", "right"))
        print(
            json.dumps(dict(pose=current, both_54=good, role=item["role"], saved=str(pose_dir))),
            flush=True,
        )
        if not good:
            raise RuntimeError("Incomplete checkerboard; route stopped with arm disabled")


if __name__ == "__main__":
    main()
