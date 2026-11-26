"""Continuous lateral approach using the same delayed-range and recovery policies."""

import json
from pathlib import Path
import time

from bucket_robot.missions.approach_policy import ApproachGate
from bucket_robot.missions.forward_range import ForwardRange
from bucket_robot.missions.recovery import RecoveryGate
from bucket_robot.motion.arm import pose
from bucket_robot.motion.chassis import HeldArmLimits
from bucket_robot.motion.combined import LateralPort
from bucket_robot.perception.observer import StereoObserver


class StereoRangeSource:
    def __init__(self, observer):
        self.observer = observer

    def range_snapshot(self):
        obs = self.observer.get()
        return (obs, obs.get("left"), obs.get("raw"), None) if obs else (None, None, None, None)


def run_continuous_approach(
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
    existing_port=None,
):
    from bucket_robot.perception.assisted_range.stereo import Geometry

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        mode="continuous_approach",
        error=None,
        stop_error=None,
        target_m=0.20,
        drive_observed=False,
        commanded_drive_seconds=0.0,
        pulses=0,
    )
    p = observer = ranger = None
    gate = RecoveryGate(20, left=999)
    alignment = ApproachGate()
    with (out / "events.jsonl").open("x", buffering=1) as stream:

        def log(kind, data):
            stream.write(json.dumps(dict(monotonic=time.monotonic(), kind=kind, data=data)) + "\n")

        def status(phase, reason=None):
            state = dict(
                phase=phase,
                reason=reason,
                updated_monotonic=time.monotonic(),
                commanded_drive_seconds=gate.drive_seconds,
                **{"ranging": plan or {}},
            )
            tmp = out / "task_status.json.new"
            tmp.write_text(json.dumps(state, indent=2))
            tmp.replace(out / "task_status.json")

        plan = None
        try:
            if not execute:
                raise ValueError("continuous_approach_requires_explicit_execute")
            p = existing_port
            if p is not None:
                if (
                    p.state["enabled"]
                    or not 131 <= pose(p.state)[0] <= 133
                    or pose(p.state)[2] != 0
                ):
                    raise RuntimeError("invalid_stopped_port_handoff")
                p.lateral = True
                p.fixed_a = pose(p.state)[0]
            else:
                p = LateralPort(HeldArmLimits(reference), log)
                p.rearm()
            p.log = log
            result["initial_state"] = dict(p.state)
            observer = StereoObserver(
                native_port,
                Geometry(calibration),
                seed_image,
                seed_box,
                bearing_tracker,
                True,
                seed_timestamp,
                target_session=target_session,
            )
            observer.thread.start()
            ranger = ForwardRange(
                StereoRangeSource(observer), calibration, model, verified_surface_band
            )
            ranger.thread.start()
            last_status = 0.0
            last_range = -1
            while True:
                if (out / "STOP").exists() or (out.parent / "STOP").exists():
                    raise KeyboardInterrupt("operator_stop")
                p.tick()
                obs = observer.get()
                now = time.monotonic()
                usable = obs
                try:
                    alignment.target_check(now, obs, pose(p.state)[0])
                except (RuntimeError, KeyError, TypeError):
                    usable = None
                plan = ranger.evaluate(
                    now,
                    usable,
                    obs.get("left") if obs else None,
                    p.phase == "idle" and not p.state["enabled"],
                )
                if plan.get("measurement"):
                    m = plan["measurement"]
                    if m["frame_id"] != last_range:
                        log("range_decision", plan)
                        last_range = m["frame_id"]
                if plan["near"]:
                    result["last_measurement"] = plan["measurement"]
                    result["reason"] = "target_distance_reached"
                    break
                velocity, _ = gate.evaluate(
                    now,
                    usable["target"] if usable else None,
                    usable["identity"] if usable else 1,
                    pose(p.state)[0],
                    True,
                    motion_permitted=plan["permit"],
                    speed_cap=plan.get("speed_cap", 8),
                )
                if gate.latched:
                    raise RuntimeError(gate.latched)
                if velocity == 0 and (gate.disable or not plan["permit"]):
                    if p.phase != "stopping" and p.state["enabled"]:
                        p.stop()
                elif velocity > 0 and p.can_send():
                    p.request(p.fixed_a, velocity)
                result["drive_observed"] |= p.state["state"] == 3
                if now - last_status >= 0.5:
                    status(gate.phase, gate.reason if usable is None else plan["reason"])
                    last_status = now
                time.sleep(0.005)
        except BaseException as exc:
            result["error"] = repr(exc)
            result["operator_stopped"] = isinstance(exc, KeyboardInterrupt)
        finally:
            from bucket_robot.missions.lifecycle import stop_and_drain

            cleanup_error = stop_and_drain(p, [w for w in (ranger, observer) if w is not None])
            if cleanup_error and not result["error"]:
                result["error"] = cleanup_error
            if p is not None:
                result["stop_error"] = p.shutdown()
                result["final_state"] = dict(p.state)
            result["commanded_drive_seconds"] = gate.drive_seconds
            result["recovery"] = gate.status()
            final = result.get("final_state", {})
            result["disabled_confirmed"] = (
                bool(final) and not final.get("enabled", True) and not result["stop_error"]
            )
            status(
                "stopped" if result["error"] else "complete",
                result.get("reason") or result["error"],
            )
            (out / "summary.json").write_text(json.dumps(result, indent=2))
    return result
