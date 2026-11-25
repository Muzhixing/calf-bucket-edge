"""Explicit field-only chassis probe."""

import argparse, json, math, signal, time
from pathlib import Path
from bucket_robot.motion.chassis import ChassisPort, HeldArmLimits
from bucket_robot.resources import default_reference


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--direction", choices=["forward", "backward", "left", "right"], required=True)
    ap.add_argument("--speed", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=1.2)
    ap.add_argument(
        "--continuous",
        action="store_true",
        help="Run until STOP file, signal, or fault; requires active operator observation",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if (
        not 1 <= args.speed <= 20
        or not math.isfinite(args.seconds)
        or not 0.6 <= args.seconds <= 10
    ):
        ap.error("speed 1..20; seconds 0.6..10")
    args.out.mkdir(parents=True, exist_ok=False)

    def interrupted(signum, frame):
        raise KeyboardInterrupt("signal_" + str(signum))

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    port = None
    result = {"error": None, "stop_error": None, "drive_observed": False}
    with (args.out / "events.jsonl").open("x", buffering=1) as events:

        def log(kind, data):
            events.write(
                json.dumps(
                    {
                        "utc_epoch": time.time(),
                        "monotonic": time.monotonic(),
                        "kind": kind,
                        "data": data,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        try:
            port = ChassisPort(HeldArmLimits(default_reference()), log)
            result["initial_state"] = port.state
            port.start_pulse(args.direction, args.speed, args.seconds, continuous=args.continuous)
            announced = False
            while port.phase != "idle":
                if (args.out / "STOP").exists():
                    port.stop()
                port.tick()
                if port.drive_observed and not announced:
                    print("DRIVING_CONFIRMED", flush=True)
                    announced = True
                time.sleep(0.01)
            result["drive_observed"] = port.drive_observed
        except BaseException as exc:
            result["error"] = str(exc)
        finally:
            if port is not None:
                result["stop_error"] = port.shutdown()
                result["final_state"] = port.state
            (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["drive_observed"] and not result["error"] and not result["stop_error"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
