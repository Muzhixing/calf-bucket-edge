"""Persistent, stopped-measurement approach with bounded loss deceleration."""

import json
import math
from pathlib import Path
import threading
import time

from bucket_robot.motion.combined import LateralPort
from bucket_robot.motion.chassis import HeldArmLimits
from bucket_robot.motion.arm import pose


def static_surface_matches(image, current_box, box, patch, weights):
    """Box association tolerates detector jitter; fixed pixels prove stasis."""
    import cv2
    from bucket_robot.perception.target import iou, weighted_ncc

    if iou(current_box, box) < 0.90:
        return False
    x1, y1, x2, y2 = map(int, box)
    current = cv2.resize(cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY), (96, 96))
    return weighted_ncc(patch, current, weights) >= 0.97


class LateralPulse:
    """Healthy pulse stays 0.6 s. Loss adds at most 0.8 s command ramp-down.

    Recovery during this pulse never accelerates it: stop and obtain a fresh
    distance before authorizing the next lateral pulse.
    """

    def __init__(self, started):
        self.started = started
        self.lost_at = None

    def step(self, now, visible):
        if self.lost_at is None and now - self.started >= 0.6:
            return 0
        if not visible and self.lost_at is None:
            self.lost_at = now
        if self.lost_at is None:
            return 8
        return max(0, int(math.floor(8 - 10 * (now - self.lost_at) + 1e-8)))


def run_recovering_approach(
    *,
    out,
    reference,
    calibration,
    model,
    seed_image,
    seed_box,
    native_port=8899,
    execute=False,
    bearing_tracker=True,
    seed_timestamp=None,
    verified_surface_band=None,
    target_session=None,
):
    import cv2
    import numpy as np
    from bucket_robot.missions.approach_policy import ApproachGate
    from bucket_robot.perception.observer import StereoObserver
    from bucket_robot.perception.assisted_range.stereo import AssistedRange

    # Optional scene-specific band, visually verified on the stopped seed frame.
    # Never infer an exterior wall from a clipped detection's geometric centre.
    if verified_surface_band is not None:
        u1, v1, u2, v2 = verified_surface_band
        if not (0 <= u1 < u2 <= 1 and 0 <= v1 < v2 <= 1):
            raise ValueError("invalid_surface_band")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        mode="recovering_approach" if execute else "observe_only",
        error=None,
        stop_error=None,
        target_m=0.20,
        pulses=0,
        commanded_drive_seconds=0.0,
        drive_observed=False,
    )
    result["verified_surface_band"] = verified_surface_band
    gate = ApproachGate()
    p = observer = estimator = worker = None
    last_status = None
    last_status_at = 0.0
    with (out / "events.jsonl").open("x", buffering=1) as stream:

        def log(kind, data):
            stream.write(
                json.dumps(dict(monotonic=time.monotonic(), kind=kind, data=data), allow_nan=False)
                + "\n"
            )

        def status(phase, reason=None):
            nonlocal last_status, last_status_at
            now = time.monotonic()
            if (phase, reason) != last_status or now - last_status_at >= 1:
                item = dict(
                    phase=phase,
                    reason=reason,
                    updated_monotonic=now,
                    commanded_drive_seconds=result["commanded_drive_seconds"],
                )
                tmp = out / "task_status.json.new"
                tmp.write_text(json.dumps(item, indent=2))
                tmp.replace(out / "task_status.json")
                log("task_state", item)
                last_status = (phase, reason)
                last_status_at = now

        try:
            estimator = AssistedRange(calibration, model)
            observer = StereoObserver(
                native_port,
                estimator.geometry,
                seed_image,
                seed_box,
                bearing_tracker,
                True,
                seed_timestamp,
                **({"target_session": target_session} if target_session is not None else {}),
            )
            observer.thread.start()
            if execute:
                p = LateralPort(HeldArmLimits(reference), log)
                result["initial_state"] = dict(p.state)
                p.rearm()

            def check():
                if (out / "STOP").exists() or (out.parent / "STOP").exists():
                    raise KeyboardInterrupt("operator_stop")
                if p:
                    p.tick()
                obs = observer.get()
                if obs is None:
                    return None
                try:
                    gate.target_check(time.monotonic(), obs, pose(p.state)[0] if p else 132)
                except (RuntimeError, KeyError, TypeError):
                    return None
                return obs

            def stopped():
                if p and p.phase != "stopping" and p.state["enabled"]:
                    p.stop()
                while p and p.phase == "stopping":
                    check()
                    time.sleep(0.005)
                if p and (p.state["enabled"] or p.phase != "idle"):
                    raise RuntimeError("disable_not_confirmed")

            while True:
                stopped()
                obs = check()
                while obs is None:
                    status("waiting_target", "verifying_same_bucket")
                    time.sleep(0.02)
                    obs = check()
                snapshot = obs
                box = obs["target"]["box"]
                x1, y1, x2, y2 = map(int, box)
                surface_roi = None
                if verified_surface_band is not None:
                    u1, v1, u2, v2 = verified_surface_band
                    w = x2 - x1
                    h = y2 - y1
                    surface_roi = [
                        int((x1 + u1 * w) / 2),
                        int((y1 + v1 * h) / 2),
                        int((x1 + u2 * w) / 2),
                        int((y1 + v2 * h) / 2),
                    ]
                patch = cv2.resize(
                    cv2.cvtColor(obs["left"][y1:y2, x1:x2], cv2.COLOR_BGR2GRAY), (96, 96)
                )
                weights = np.ones((96, 96), np.float32)
                completed = threading.Event()
                answer = {}
                scene_valid = True

                def still_static(current):
                    if current is None:
                        return False
                    return static_surface_matches(
                        current["left"], current["target"]["box"], box, patch, weights
                    )

                def measure():
                    try:
                        right = cv2.remap(
                            snapshot["raw"][:, 1280:], *estimator.geometry.maps[1], cv2.INTER_LINEAR
                        )
                        if surface_roi is None:
                            answer.update(estimator.measure([snapshot["left"], right], box))
                        else:
                            answer.update(
                                estimator.measure(
                                    [snapshot["left"], right], box, point_roi=surface_roi
                                )
                            )
                    except Exception as exc:
                        answer.update(valid=False, error=repr(exc))
                    finally:
                        completed.set()

                worker = threading.Thread(target=measure, daemon=True)
                worker.start()
                began = time.monotonic()
                status("measuring")
                while not completed.wait(0.01):
                    current = check()
                    scene_valid = still_static(current) and scene_valid
                    if p and (p.state["enabled"] or p.phase != "idle"):
                        raise RuntimeError("motion_during_measurement")
                    if time.monotonic() - began > 8:
                        status("waiting_model", "inference_slow_outputs_disabled")
                current = check()
                scene_valid = still_static(current) and scene_valid
                if not scene_valid:
                    answer.update(valid=False, error="scene_changed_during_measurement")
                answer.update(
                    frame_id=snapshot["frame_id"],
                    capture_time=snapshot["metadata"]["arrival"],
                    stationary_verified=p is not None and not p.state["enabled"],
                )
                try:
                    decision = gate.decide(time.monotonic(), answer)
                except RuntimeError as exc:
                    decision = str(exc)
                result["last_measurement"] = answer
                log("range_decision", dict(measurement=answer, decision=decision))
                if not execute:
                    result["reason"] = "observation_complete"
                    break
                if decision in ("target_distance_reached", "distance_stop_margin"):
                    result["reason"] = decision
                    status("complete", decision)
                    break
                if decision == "drive_budget_exhausted":
                    while True:
                        status("budget_paused", decision)
                        check()
                        time.sleep(0.05)
                if decision != "left_pulse":
                    status("waiting_distance", decision)
                    until = time.monotonic() + 0.25
                    while time.monotonic() < until:
                        check()
                        time.sleep(0.01)
                    continue
                # A new verified range authorizes one pulse only.
                while not p.can_send():
                    check()
                    time.sleep(0.005)
                if check() is None:
                    continue
                p.request(p.fixed_a, 8)
                pulse = LateralPulse(time.monotonic())
                result["pulses"] += 1
                last_tick = time.monotonic()
                velocity = 8
                status("left_pulse")
                pulse_observed = False
                while True:
                    current = check()
                    now = time.monotonic()
                    pulse_observed |= p.state["state"] == 3
                    if velocity > 0:
                        result["commanded_drive_seconds"] += now - last_tick
                    last_tick = now
                    demand = pulse.step(now, current is not None)
                    if result["commanded_drive_seconds"] >= gate.max_drive:
                        demand = 0
                    if demand == 0:
                        stopped()
                        break
                    if demand != velocity and p.can_send():
                        p.request(p.fixed_a, demand)
                        velocity = demand
                        log(
                            "deceleration_command",
                            dict(speed=demand, target_visible=current is not None),
                        )
                    if pulse.lost_at is not None:
                        status("decelerating", "target_lost_remeasure_after_stop")
                    time.sleep(0.005)
                result["drive_observed"] |= pulse_observed
                if not pulse_observed:
                    raise RuntimeError("drive_not_observed")
                gate.drive_seconds = result["commanded_drive_seconds"]
                settle = time.monotonic() + 0.3
                while time.monotonic() < settle:
                    check()
                    time.sleep(0.01)
        except BaseException as exc:
            result["error"] = repr(exc)
            result["operator_stopped"] = isinstance(exc, KeyboardInterrupt)
        finally:
            if p is not None:
                result["stop_error"] = p.shutdown()
                result["final_state"] = dict(p.state)
            if observer is not None:
                observer.end.set()
                observer.thread.join(timeout=1)
            if estimator is not None and (worker is None or not worker.is_alive()):
                estimator.close()
            (out / "summary.json").write_text(json.dumps(result, indent=2))
    return result
