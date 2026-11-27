"""Continuous lateral approach using the same delayed-range and recovery policies."""

import json
from pathlib import Path
import time

from bucket_robot.missions.approach_policy import ApproachGate
from bucket_robot.missions.forward_range import ForwardRange
from bucket_robot.missions.recovery import RecoveryGate, RecoverySettings
from bucket_robot.motion.arm import Limits, pose
from bucket_robot.motion.chassis import HeldArmLimits
from bucket_robot.motion.combined import LateralPort
from bucket_robot.perception.observer import StereoObserver


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
    align_upper_after_stop=False,
    cruise_speed=8,
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
    if type(cruise_speed) is not int or cruise_speed not in (8, 20):
        raise ValueError("invalid_lateral_approach_speed")
    speed = 20 if align_upper_after_stop else cruise_speed
    gate = RecoveryGate(20, left=999, settings=RecoverySettings(cruise=speed))
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
            if phase == "heading_correcting":
                state["heading"] = heading_recovery.status()
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
                p = LateralPort(
                    Limits(reference) if align_upper_after_stop else HeldArmLimits(reference), log
                )
                p.rearm()
            p.max_velocity = speed
            p.log = log
            result["initial_state"] = dict(p.state)
            from bucket_robot.motion.heading import HeadingRecovery

            heading_recovery = getattr(p, "heading_recovery", None)
            if heading_recovery is None:
                heading_recovery = p.heading_recovery = HeadingRecovery(p.fixed_yaw, p.heading)
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
            from bucket_robot.perception.observation import TaskObservationPublisher

            observer.publisher = TaskObservationPublisher(out, task_root=out.parent)
            observer.thread.start()
            ranger = ForwardRange(observer, calibration, model, verified_surface_band, cruise=speed)
            ranger.thread.start()
            last_status = 0.0
            last_range = -1
            upper_aligned_at = None
            while True:
                if (out / "STOP").exists() or (out.parent / "STOP").exists():
                    raise KeyboardInterrupt("operator_stop")
                p.tick()
                obs = observer.get()
                now = time.monotonic()
                was_correcting = heading_recovery.active
                correcting = heading_recovery.advance(p, now)
                if was_correcting or correcting:
                    gate.hold_for_heading(now)
                    if correcting:
                        if now - last_status >= 0.25:
                            status("heading_correcting", heading_recovery.phase)
                            last_status = now
                        time.sleep(0.005)
                        continue
                    ranger.invalidate_before(now)
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
                if plan["near"] and align_upper_after_stop and upper_aligned_at is None:
                    from bucket_robot.missions.vertical import align_parked
                    from bucket_robot.tracking.controller import TrackingController
                    from bucket_robot.tracking.policy import FollowPolicy

                    matrix = observer.geometry.c["P1"]
                    controller = TrackingController(
                        FollowPolicy(
                            Limits(reference),
                            matrix[0, 0],
                            matrix[1, 1],
                            matrix[0, 2],
                            matrix[1, 2],
                        )
                    )
                    result["upper_alignment"] = align_parked(observer, p, controller, out, log)
                    upper_aligned_at = result["upper_alignment"]["settled_at"]
                    if result["upper_alignment"]["moved"]:
                        ranger.invalidate_before(upper_aligned_at)
                        continue
                    upper_aligned_at = plan["measurement"]["capture_time"]
                if upper_aligned_at is not None:
                    if time.monotonic() - upper_aligned_at > 25:
                        raise RuntimeError("waiting_range_after_upper_alignment_timeout")
                    m = plan.get("measurement") or {}
                    if (
                        not m.get("valid")
                        or m.get("capture_time", 0) < upper_aligned_at
                        or not plan.get("static_verified")
                        or not usable
                    ):
                        time.sleep(0.005)
                        continue
                    plan = dict(plan, near=True)
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
                    speed_cap=heading_recovery.speed_cap(plan.get("speed_cap", 8)),
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
                if getattr(p, "heading_recovery", None):
                    result["heading_recovery"] = p.heading_recovery.status()
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
