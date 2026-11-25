from bucket_robot.resources import default_reference

"""One requested clearance probe: A <=2 degrees or B <=10 mm.

Use only with the main program stopped or in observe-only mode. Every invocation
requires a fresh expected pose and a new evidence directory. It does not change
the production operating envelope or infer physical clearance from telemetry.
"""
import argparse
import json
from pathlib import Path
import time

from bucket_robot.motion.arm import ArmPort, Limits, pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-pose", nargs=3, type=float, required=True)
    targets = parser.add_mutually_exclusive_group(required=True)
    targets.add_argument("--target-a", type=int)
    targets.add_argument("--target-b", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    start = args.expect_pose
    target = list(start)
    if args.target_a is not None:
        target[0] = args.target_a
        if not (
            0 <= start[0] <= 180 and 0 <= target[0] <= 180 and 0 < abs(target[0] - start[0]) <= 2
        ):
            parser.error("A probe must change A by at most 2 protocol degrees")
    else:
        target[1] = args.target_b
        if not (0 <= start[0] <= 180 and 0 < abs(target[1] - start[1]) <= 10):
            parser.error("B probe must change B by at most 10 mm at a valid A angle")
    if any(v != round(v) for v in start):
        parser.error("expected pose must use whole units; normalize B separately")
    limits = Limits(default_reference())
    # This ephemeral range authorizes only the single requested test interval.
    limits.a_min = min(start[0], target[0])
    limits.a_max = max(start[0], target[0])
    limits.check(start)
    limits.check(target)
    args.out.mkdir(parents=True, exist_ok=False)

    def log(event, data):
        with (args.out / "events.jsonl").open("a") as f:
            f.write(
                json.dumps({"timestamp": time.monotonic(), "event": event, "data": data}) + "\n"
            )

    port = None
    result = {}
    try:
        port = ArmPort(limits, log)
        if pose(port.state) != start:
            raise RuntimeError("live_pose_differs_from_expected")
        port.move([int(v) for v in target])
        until = time.monotonic() + 8 + abs(target[1] - start[1]) / 5
        while port.phase != "idle" and time.monotonic() < until:
            port.tick()
            time.sleep(0.01)
        if port.phase != "idle" or port.state["enabled"] or pose(port.state) != target:
            raise RuntimeError("probe_completion_not_confirmed")
        result = {
            "requested": target,
            "telemetry": port.state,
            "physical_clearance": "requires_operator_observation",
        }
    finally:
        if port is not None:
            error = port.shutdown()
            if error:
                result["stop_error"] = error
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        if result:
            print(json.dumps(result), flush=True)
    if result.get("stop_error"):
        raise RuntimeError(result["stop_error"])


if __name__ == "__main__":
    main()
