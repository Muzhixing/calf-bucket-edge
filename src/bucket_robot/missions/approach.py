from bucket_robot.missions.approach_policy import ApproachGate as ApproachGate
from bucket_robot.perception.observer import StereoObserver as StereoObserver

"""Bounded left approach after confirmed A132 alignment; default is observation.

Classical-first neural fallback runs only while disabled. Every 0.6 s left pulse
is followed by acknowledged disable and a new measurement. No zero registration,
arm movement, rightward correction, dispensing, or automatic restart.
"""
import argparse
import json
from pathlib import Path
import signal
import threading
import time

from bucket_robot.motion.chassis import HeldArmLimits, SidePort
from bucket_robot.motion.arm import pose


def run_approach(
    *,
    out,
    reference,
    calibration,
    model,
    seed_image,
    seed_box,
    native_port=8899,
    execute=False,
    bearing_tracker=False,
    recover_loss=False,
    seed_timestamp=None,
    verified_surface_band=None,
    target_session=None,
):
    if recover_loss:
        from bucket_robot.missions.recovering_approach import run_recovering_approach

        return run_recovering_approach(
            out=out,
            reference=reference,
            calibration=calibration,
            model=model,
            seed_image=seed_image,
            seed_box=seed_box,
            native_port=native_port,
            execute=execute,
            bearing_tracker=bearing_tracker,
            seed_timestamp=seed_timestamp,
            verified_surface_band=verified_surface_band,
            target_session=target_session,
        )
    import cv2
    from bucket_robot.perception.assisted_range.stereo import AssistedRange
    import numpy as np
    from bucket_robot.perception.target import iou, weighted_ncc

    cv2.setNumThreads(1)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    gate = ApproachGate()
    result = dict(
        mode="bounded_motion" if execute else "observe_only",
        error=None,
        stop_error=None,
        target_m=0.20,
        stop_margin_m=0.01,
        pulses=0,
    )
    p = observer = estimator = worker = None
    started = time.monotonic()
    with (out / "events.jsonl").open("x", buffering=1) as stream:

        def log(kind, data):
            stream.write(
                json.dumps(dict(monotonic=time.monotonic(), kind=kind, data=data), allow_nan=False)
                + "\n"
            )

        try:
            estimator = AssistedRange(calibration, model)
            observer = StereoObserver(
                native_port, estimator.geometry, seed_image, seed_box, bearing_tracker
            )
            observer.thread.start()
            if execute:
                p = SidePort(HeldArmLimits(reference), log)
                result["initial_state"] = dict(p.state)

            def check():
                if (out / "STOP").exists():
                    raise RuntimeError("operator_stop")
                if time.monotonic() - started > 90:
                    raise RuntimeError("approach_time_budget")
                if p:
                    p.tick()
                obs = observer.get()
                if obs is not None:
                    gate.target_check(time.monotonic(), obs, pose(p.state)[0] if p else 132)
                return obs

            while check() is None:
                if time.monotonic() - started > 10:
                    raise RuntimeError("no_target")
                time.sleep(0.01)
            while True:
                # The chassis is disabled throughout inference, including CPU fallback.
                obs = check()
                if p and (p.phase != "idle" or p.state["enabled"]):
                    raise RuntimeError("measurement_requires_disabled_output")
                completed = threading.Event()
                box = obs["target"]["box"]
                snapshot = obs
                answer = {}
                x1, y1, x2, y2 = map(int, box)
                original_patch = cv2.cvtColor(snapshot["left"][y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                original_patch = cv2.resize(original_patch, (96, 96), interpolation=cv2.INTER_AREA)
                weights = np.ones((96, 96), np.float32)

                def check_static(current):
                    if current is None or iou(current["target"]["box"], box) < 0.97:
                        raise RuntimeError("scene_changed_during_measurement")
                    patch = cv2.cvtColor(current["left"][y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                    patch = cv2.resize(patch, (96, 96), interpolation=cv2.INTER_AREA)
                    if weighted_ncc(original_patch, patch, weights) < 0.97:
                        raise RuntimeError("scene_changed_during_measurement")
                    if p and (p.state["enabled"] or p.phase != "idle"):
                        raise RuntimeError("motion_during_measurement")

                def measure():
                    try:
                        right = cv2.remap(
                            snapshot["raw"][:, 1280:], *estimator.geometry.maps[1], cv2.INTER_LINEAR
                        )
                        answer.update(estimator.measure([snapshot["left"], right], box))
                    except BaseException as exc:
                        answer.update(valid=False, error=repr(exc))
                    finally:
                        completed.set()

                worker = threading.Thread(target=measure, daemon=True)
                worker.start()
                deadline = time.monotonic() + 8
                while not completed.wait(0.01):
                    current = check()
                    check_static(current)
                    if time.monotonic() > deadline:
                        raise RuntimeError("measurement_timeout")
                current = check()
                check_static(current)
                answer.update(
                    frame_id=obs["frame_id"],
                    capture_time=obs["metadata"]["arrival"],
                    stationary_verified=p is not None and not p.state["enabled"],
                )
                decision = gate.decide(time.monotonic(), answer)
                log("range_decision", dict(measurement=answer, decision=decision))
                result["last_measurement"] = answer
                if not execute or decision != "left_pulse":
                    result["reason"] = decision if execute else "observation_complete"
                    break
                p.start_pulse("left", 8, 0.6)
                gate.drive_seconds += 0.6
                result["pulses"] += 1
                while p.phase != "idle":
                    check()
                    time.sleep(0.005)
                if not p.drive_observed:
                    raise RuntimeError("drive_not_observed")
                settle = time.monotonic() + 0.3
                while time.monotonic() < settle:
                    check()
                    time.sleep(0.01)
                # Reject a camera that has not advanced since the last stop.
                if check()["metadata"]["arrival"] < settle - 0.3:
                    raise RuntimeError("no_post_stop_image")
        except BaseException as exc:
            result["error"] = repr(exc)
        finally:
            if p is not None:
                result["stop_error"] = p.shutdown()
                result["final_state"] = dict(p.state)
            if observer is not None:
                observer.end.set()
                observer.thread.join(timeout=1)
            if estimator is not None and (worker is None or not worker.is_alive()):
                estimator.close()
            result["commanded_drive_seconds"] = gate.drive_seconds
            (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("out", "reference", "calibration", "model", "seed-image"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--seed-box", type=int, nargs=4, required=True)
    ap.add_argument("--native-port", type=int, default=8899)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--bearing-tracker", action="store_true")
    ap.add_argument("--recover-loss", action="store_true")
    args = ap.parse_args()

    def interrupt(signum, frame):
        if args.recover_loss and signum == signal.SIGHUP:
            from bucket_robot.missions.lifecycle import TaskPause

            raise TaskPause("operator_connection_lost")
        raise KeyboardInterrupt("signal_" + str(signum))

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupt)
    result = run_approach(**vars(args))
    print(json.dumps(result))
    if (
        args.recover_loss
        and (result["error"] or result["stop_error"])
        and not result.get("operator_stopped")
    ):
        from bucket_robot.missions.lifecycle import park_fault

        park_fault(args.out, result)
    return 2 if result["error"] or result["stop_error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
