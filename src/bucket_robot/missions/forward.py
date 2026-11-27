"""Compose shared target tracking, range gates and acknowledged mission stages.

Native vision runs without serial ownership. Optional post-actions are handed
to completion.py only after tracking has stopped and workers have drained.
"""

from bucket_robot.missions.lifecycle import TaskPause as TaskPause, park_fault as park_fault
import json, math, signal, time
from bucket_robot.motion.arm import Limits, pose
from bucket_robot.motion.combined import CombinedPort
from bucket_robot.tracking.controller import TrackingController


class ForwardGate:
    def __init__(self, seconds, left=132, until_left=False):
        maximum = 60 if until_left else 10
        if not math.isfinite(seconds) or not 0.6 <= seconds <= maximum:
            raise ValueError("drive_duration_0.6_to_" + str(maximum))
        self.seconds = seconds
        self.left = left
        self.phase = "centering"
        self.stable = None
        self.started = None
        self.identity = None

    def evaluate(self, now, target, identity, a, settled):
        if target is None:
            raise RuntimeError("target_lost")
        if not 0 <= now - target["timestamp"] <= 0.35:
            raise RuntimeError("stale_target")
        if self.identity is None:
            self.identity = identity
        if identity != self.identity:
            raise RuntimeError("target_identity_changed")
        dx = target["offset_px"][0]
        if not math.isfinite(dx):
            raise RuntimeError("invalid_target")
        from bucket_robot.perception.bearing import measured_horizontal

        bearing = measured_horizontal(target)
        if target.get("partial_visible", False) and not bearing:
            if (
                self.phase == "centering"
                and target.get("horizontal_clipped") is True
                and target.get("bearing_evidence") == "current_detection_and_visible_appearance"
            ):
                self.stable = None
                return 0, None
            raise RuntimeError("invalid_target")
        if self.started is not None and now - self.started >= self.seconds:
            return 0, "time_limit"
        if a >= self.left - 1 and abs(dx) <= 20:
            return 0, "left_reference_reached"
        if a >= self.left:
            return 0, "left_boundary_reached"
        if self.phase == "centering":
            if abs(dx) <= 20 and settled:
                if self.stable is None:
                    self.stable = now
                if now - self.stable >= 0.7:
                    self.phase = "driving"
                    self.started = now
            else:
                self.stable = None
        if self.phase == "driving" and abs(dx) > 45:
            raise RuntimeError("horizontal_tracking_error")
        return (8 if self.phase == "driving" else 0), None


# Historical import remains compatible; implementation belongs to perception.
from bucket_robot.perception.observer import TargetObserver as Observer


def main():
    from bucket_robot.missions.options import parse_options
    from bucket_robot.missions.profile import MissionProfile
    from bucket_robot.missions.stages import MissionStages, Stage

    args = parse_options()
    profile = MissionProfile.from_options(args)
    from bucket_robot.missions.recovery import RecoveryGate, RecoverySettings
    from bucket_robot.tracking.aim import horizontal_aligned

    gate = (
        RecoveryGate(
            args.seconds,
            settings=profile.recovery_settings(),
        )
        if args.recover_loss
        else ForwardGate(args.seconds, until_left=args.until_left)
    )
    ref = json.loads(args.left_reference.read_text())
    if ref["left_A_command"] != 132 or ref["forward_A_command"] != 73:
        raise RuntimeError("reference_mismatch")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "mission_profile.json").write_text(json.dumps(profile.as_dict(), indent=2))
    stages = MissionStages(args.out)
    from bucket_robot.perception.observation import TaskObservationPublisher

    observer = Observer(
        args.native_port,
        args.out,
        args.seed_image,
        args.seed_box,
        args.bearing_tracker,
        args.recover_loss,
        **({"range_calibration": args.approach_calibration} if args.distance_first else {}),
    )

    observer.publisher = TaskObservationPublisher(args.out)

    def interrupted(signum, frame):
        if args.recover_loss and signum == signal.SIGHUP:
            raise TaskPause("operator_connection_lost")
        raise KeyboardInterrupt("signal_" + str(signum))

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    p = None
    ranger = None
    near_measurement = None
    post_plan = None
    post_aligning = False
    handoff = None
    heading_recovery = None
    result = {
        "mode": "observe_only"
        if args.observe_only
        else "A_only_centering"
        if args.center_only
        else "recovering_mission"
        if args.recover_loss
        else "continuous_left_alignment"
        if args.until_left
        else "bounded_motion",
        "drive_timeout_s": args.seconds,
        "drive_observed": False,
        "error": None,
        "stop_error": None,
    }
    with (args.out / "serial.jsonl").open("x", buffering=1) as log:

        def record(kind, data):
            log.write(
                json.dumps({"monotonic": time.monotonic(), "kind": kind, "data": data}) + "\n"
            )

        try:
            stages.check()
            observer.thread.start()
            if not args.observe_only:
                limits = Limits(args.reference)
                limits.a_min = 73
                limits.a_max = 132
                if args.track_b:
                    from bucket_robot.missions.vertical import ABCombinedPort, VerticalSupervisor

                    p = ABCombinedPort(limits, record)
                else:
                    p = CombinedPort(limits, record)
                if pose(p.state) != [args.expect_a, args.expect_b, 0]:
                    raise RuntimeError("initial_pose_mismatch")
                result["initial_state"] = dict(p.state)
                p.max_velocity = args.cruise_speed
                p.rearm()
            if args.distance_first:
                from bucket_robot.missions.forward_range import ForwardRange

                ranger = ForwardRange(
                    observer,
                    args.approach_calibration,
                    args.approach_model,
                    args.approach_surface_band,
                    **({"cruise": args.cruise_speed} if args.cruise_speed != 8 else {}),
                )
                ranger.thread.start()
            started = time.monotonic()
            last_initial_status = 0.0
            while observer.snapshot()[0] is None or (
                args.recover_loss and observer.snapshot()[0]["target"] is None
            ):
                if p is not None:
                    p.tick()
                if observer.snapshot()[2]:
                    raise RuntimeError(observer.snapshot()[2])
                if (args.out / "STOP").exists():
                    raise KeyboardInterrupt("operator_stop")
                if (not args.recover_loss or args.observe_only) and time.monotonic() - started > 8:
                    raise RuntimeError("no_fresh_target")
                if args.recover_loss and time.monotonic() - last_initial_status >= 1:
                    waiting = observer.snapshot()[0] or {}
                    (args.out / "task_status.json").write_text(
                        json.dumps(
                            dict(
                                phase="waiting_initial_target",
                                reason=waiting.get("reason", "initial_acquisition"),
                                serial_opened=p is not None,
                            ),
                            indent=2,
                        )
                    )
                    last_initial_status = time.monotonic()
                time.sleep(0.02)
            if args.observe_only:
                while time.monotonic() - started < 5:
                    obs, _, error = observer.snapshot()
                    if error:
                        raise RuntimeError(error)
                    if (
                        not args.recover_loss
                        and time.monotonic() - obs["target"]["timestamp"] > 0.35
                    ):
                        raise RuntimeError("stale_target")
                    time.sleep(0.01)
                result["reason"] = "observation_complete"
                return 0
            import cv2
            from bucket_robot.tracking.policy import FollowPolicy

            fs = cv2.FileStorage(str(args.calibration), cv2.FILE_STORAGE_READ)
            P = fs.getNode("P1").mat()
            fs.release()
            policy = FollowPolicy(
                limits, float(P[0, 0]), float(P[1, 1]), float(P[0, 2]), float(P[1, 2])
            )
            vertical = VerticalSupervisor(policy) if args.track_b else None
            controller = (
                vertical.controller if vertical else TrackingController(policy, target_max_age=0.35)
            )
            stages.transition(Stage.TRACK)
            began = time.monotonic()
            last_decision = 0.0
            last_frame = -1
            announced = False
            last_recovery = 0
            last_status = None
            last_status_at = 0.0
            last_range_frame = -1
            upper_aligned_at = None
            visibility_wait = False
            visibility_limited = False
            heading_recovery = None
            if args.recover_loss:
                from bucket_robot.motion.heading import HeadingRecovery

                heading_recovery = p.heading_recovery = HeadingRecovery(p.fixed_yaw, p.heading)
            while True:
                stages.check()
                now = time.monotonic()
                p.tick()
                if (args.out / "STOP").exists():
                    result["reason"] = "operator_stop"
                    result["operator_stopped"] = True
                    break
                if not args.recover_loss and now - began > (
                    args.seconds + 10 if args.until_left else 20
                ):
                    result["reason"] = "overall_timeout"
                    break
                if (
                    not args.recover_loss
                    and args.until_left
                    and gate.started is None
                    and now - began > 10
                ):
                    result["reason"] = "centering_timeout"
                    break
                obs, current_image, error = observer.snapshot()
                if error:
                    raise RuntimeError(error)
                # Read the clock after UART polling and the observer snapshot:
                # a newly arrived frame must not look as though it is future-dated.
                now = time.monotonic()
                target = controller.target(obs["target"])
                if heading_recovery:
                    was_correcting = heading_recovery.active
                    correcting = heading_recovery.advance(p, now)
                    if was_correcting or correcting:
                        gate.hold_for_heading(now)
                        policy.reset()
                        if vertical:
                            if vertical.stage == "moving":
                                vertical.stage = "stopping"
                            vertical.on_disabled(p)
                        if correcting:
                            if now - last_status_at >= 0.25:
                                status = dict(
                                    phase="heading_correcting",
                                    reason=heading_recovery.phase,
                                    velocity=0,
                                    heading=heading_recovery.status(),
                                    disabled_confirmed=p.phase == "idle" and not p.state["enabled"],
                                )
                                (args.out / "task_status.json").write_text(json.dumps(status))
                                record("recovery_state", status)
                                last_status_at = now
                            time.sleep(0.005)
                            continue
                        if ranger:
                            ranger.invalidate_before(now)
                policy.set_a_rate_limit(
                    profile.a_rate_limit(
                        departed=gate.started is not None, post_aligning=post_aligning
                    )
                )
                current = pose(p.state)[0]
                settled = abs(current - (p.target or [current])[0]) <= 1
                vertical_plan = (
                    dict(ready=True, goal=None)
                    if vertical and args.a_start_only and not post_aligning
                    else vertical.plan(
                        now,
                        target,
                        pose(p.state)[1],
                        obs.get("metadata", {}),
                        pose(p.state),
                        p.target,
                    )
                    if vertical
                    else None
                )
                if args.a_start_only and vertical and not post_aligning:
                    vertical_plan = dict(ready=True, goal=None)
                    vertical.reason = "B_held_during_travel"
                    region = (target or {}).get("aim_region_px")
                    if region and region[1] >= 64 and region[3] <= 560:
                        visibility_limited = False
                    if region and not visibility_limited and (region[1] < 24 or region[3] > 600):
                        visibility_wait = True
                range_plan = (
                    ranger.evaluate(
                        now, obs, current_image, p.phase == "idle" and not p.state["enabled"]
                    )
                    if ranger and not post_aligning
                    else None
                )
                if range_plan and range_plan.get("measurement"):
                    measurement = range_plan["measurement"]
                    if measurement["frame_id"] != last_range_frame:
                        record("forward_range", range_plan)
                        last_range_frame = measurement["frame_id"]
                if visibility_wait and not upper_aligned_at:
                    if p.phase == "idle" and not p.state["enabled"]:
                        from bucket_robot.missions.vertical import align_parked

                        result["view_recovery"] = align_parked(
                            observer, p, controller, args.out, record, visibility_only=True
                        )
                        if result["view_recovery"]["moved"]:
                            ranger.invalidate_before(result["view_recovery"]["settled_at"])
                        visibility_wait = False
                        visibility_limited = result["view_recovery"]["constrained"]
                        gate.speed = gate.velocity = 0
                        continue
                    if range_plan:
                        range_plan = dict(range_plan, permit=False, speed_cap=0)
                if range_plan and visibility_limited:
                    range_plan = dict(range_plan, speed_cap=min(20, range_plan.get("speed_cap", 0)))
                if range_plan and (range_plan["near"] or upper_aligned_at is not None):
                    if args.a_start_only and upper_aligned_at is None:
                        from bucket_robot.missions.vertical import align_parked

                        result["upper_alignment"] = align_parked(
                            observer, p, controller, args.out, record
                        )
                        upper_aligned_at = result["upper_alignment"]["settled_at"]
                        if result["upper_alignment"]["moved"]:
                            ranger.invalidate_before(upper_aligned_at)
                            continue
                        upper_aligned_at = range_plan["measurement"]["capture_time"]
                    if upper_aligned_at is not None:
                        m = range_plan.get("measurement") or {}
                        if time.monotonic() - upper_aligned_at > 25:
                            raise RuntimeError("waiting_range_after_upper_alignment_timeout")
                        if (
                            not target
                            or not m.get("valid")
                            or m.get("capture_time", 0) < upper_aligned_at
                            or not range_plan.get("static_verified")
                            or range_plan.get("reason") == "waiting_valid_same_target_range"
                        ):
                            time.sleep(0.005)
                            continue
                    near_measurement = range_plan["measurement"]
                    if args.execute_post_geometry:
                        from bucket_robot.missions.post_geometry import alignment_extension_plan
                        from bucket_robot.missions.post_execute import lateral_timing

                        if p.phase != "idle" or p.state["enabled"]:
                            raise RuntimeError("triangle_requires_stopped_range_confirmation")
                        post_plan = alignment_extension_plan(
                            near_measurement["distance_m"], current
                        )
                        lateral_timing(post_plan["sequence"][2]["diagnostic_half_left_mm"])
                        record("stopped_triangle_plan", post_plan)
                        (args.out / "post_approach.json").write_text(
                            json.dumps(post_plan, indent=2)
                        )
                        stages.transition(Stage.POST_ALIGN)
                        post_aligning = True
                        # The range gate has already verified a newly captured,
                        # disabled scene. Keep the exact same Observer/session.
                        previous_drive = gate.drive_seconds
                        if previous_drive >= args.seconds:
                            raise RuntimeError("drive_budget_exhausted_before_post_alignment")
                        from bucket_robot.missions.post_alignment import PostAlignmentGate

                        gate = PostAlignmentGate(
                            max(0.6, args.seconds - previous_drive),
                            settings=RecoverySettings(
                                cruise=20,
                                acceleration_per_s=20,
                                deceleration_per_s=40,
                                initial_confirm_s=0.1,
                            )
                            if args.a_start_only or args.cruise_speed > 20
                            else None,
                        )
                        gate.identity = obs["identity"]
                        vertical = None  # Preserve the geometry measurement height.
                        policy.reset()
                        range_plan = None
                        continue
                    result["reason"] = "target_distance_reached"
                    break
                if args.recover_loss:
                    velocity, reason = gate.evaluate(
                        now,
                        target,
                        obs["identity"],
                        current,
                        settled,
                        commanded_a=(p.target or [current])[0],
                        speed_cap=(
                            heading_recovery.speed_cap(range_plan.get("speed_cap", 8))
                            if heading_recovery and range_plan
                            else range_plan.get("speed_cap", 8)
                        )
                        if range_plan
                        else 20
                        if post_aligning and (args.a_start_only or args.cruise_speed > 20)
                        else 8,
                        chassis_stopped=p.phase == "idle" and not p.state["enabled"],
                        motion_permitted=(
                            vertical_plan["ready"] and vertical.stage == "idle"
                            if vertical
                            else True
                        )
                        and (range_plan["permit"] if range_plan else True),
                    )
                else:
                    velocity, reason = gate.evaluate(now, target, obs["identity"], current, settled)
                if (
                    range_plan
                    and not range_plan["permit"]
                    and gate.reason == "vertical_adjustment"
                    and (vertical_plan is None or vertical_plan["ready"])
                ):
                    gate.reason = range_plan["reason"]
                if reason:
                    result["reason"] = (
                        "post_left_aligned"
                        if post_aligning and reason == "left_reference_reached"
                        else reason
                    )
                    if reason == "left_reference_reached":
                        handoff = (obs, current_image, error)
                    break
                if args.center_only and velocity > 0:
                    result["reason"] = "horizontal_centered"
                    break
                result["drive_observed"] |= p.state["state"] == 3
                if args.recover_loss:
                    status = gate.status()
                    if post_aligning:
                        status["phase"] = "post_forward_align_" + gate.phase
                    if vertical:
                        status["vertical"] = vertical.status()
                    if range_plan:
                        status["ranging"] = {
                            k: v for k, v in range_plan.items() if k != "measurement"
                        }
                    if gate.phase != last_status or now - last_status_at >= 1:
                        status.update(
                            updated_monotonic=now,
                            tracking_reason=obs.get("reason"),
                            recovery_count_visual=obs.get("recovery_count", 0),
                        )
                        temp = args.out / "task_status.json.new"
                        temp.write_text(json.dumps(status, indent=2))
                        temp.replace(args.out / "task_status.json")
                        record("recovery_state", status)
                        last_status = gate.phase
                        last_status_at = now
                    if gate.disable:
                        if p.phase != "stopping" and p.state["enabled"]:
                            p.stop()
                        if vertical and vertical.stage == "moving":
                            vertical.stage = "stopping"
                        if vertical:
                            vertical.on_disabled(p)
                        policy.reset()
                        time.sleep(0.005)
                        continue
                    if obs.get("recovery_count", 0) != last_recovery:
                        policy.reset()
                        last_recovery = obs.get("recovery_count", 0)
                if vertical and vertical.advance(
                    p, vertical_plan, velocity, target, obs.get("metadata", {}), now
                ):
                    time.sleep(0.005)
                    continue
                if (
                    range_plan
                    and not range_plan["permit"]
                    and velocity == 0
                    and target
                    and horizontal_aligned(target)
                    and settled
                ):
                    if p.phase != "stopping" and p.state["enabled"]:
                        p.stop()
                    time.sleep(0.005)
                    continue
                if (
                    ranger
                    and velocity == 0
                    and target
                    and settled
                    and horizontal_aligned(target)
                    and p.phase == "idle"
                    and not p.state["enabled"]
                ):
                    # Keep the scene physically stopped during range/identity
                    # confirmation, rather than enabling a zero-speed hold.
                    time.sleep(0.005)
                    continue
                if gate.started is not None and not announced:
                    print(
                        "FORWARD_PHASE_STARTED speed="
                        + str(args.cruise_speed)
                        + " max_seconds="
                        + str(args.seconds),
                        flush=True,
                    )
                    announced = True
                fresh_frame = obs["frame_id"] != last_frame
                if (
                    now - last_decision >= 0.05
                    and (fresh_frame or args.recover_loss and velocity != p.velocity)
                    and p.can_send()
                ):
                    usable = not args.recover_loss or gate.allow_a
                    a = (
                        controller.horizontal_goal(
                            p, target, now, centering=gate.phase == "centering"
                        )
                        if usable and fresh_frame
                        else current
                    )
                    p.request(a, velocity)
                    record(
                        "decision",
                        {
                            "A": a,
                            "speed": velocity,
                            "error_x": target["offset_px"][0] if target else None,
                            "phase": gate.phase,
                            "frame": obs["frame_id"],
                            "horizontal_bearing_valid": (target or {}).get(
                                "horizontal_bearing_valid"
                            ),
                            "vertical_clipped": (target or {}).get("vertical_clipped"),
                        },
                    )
                    last_decision = now
                    last_frame = obs["frame_id"]
                time.sleep(0.005)
        except BaseException as e:
            result["error"] = repr(e)
            result["operator_stopped"] = isinstance(e, KeyboardInterrupt)
            if args.recover_loss and not isinstance(e, KeyboardInterrupt):
                gate.trip(repr(e))
        finally:
            stages.transition(Stage.DRAIN)
            if heading_recovery:
                result["heading_recovery"] = heading_recovery.status()
            result["drive_phase_elapsed_s"] = (
                time.monotonic() - gate.started if gate.started is not None else 0
            )
            if args.recover_loss:
                result["recovery"] = gate.status()
            from bucket_robot.missions.lifecycle import stop_and_drain

            cleanup_error = stop_and_drain(p, [w for w in (ranger, observer) if w is not None])
            if cleanup_error and not result["error"]:
                result["error"] = cleanup_error
            keep_serial = bool(
                args.distance_first
                and result.get("reason") == "left_reference_reached"
                and not result["error"]
            )
            if p is not None:
                result["stop_error"] = p.shutdown(close_port=False) if keep_serial else p.shutdown()
                result["final_state"] = dict(p.state)
            if stages.error and not result["error"]:
                result["error"] = "stage_journal_failed:" + stages.error
            if args.recover_loss:
                status = gate.status()
                status.update(
                    phase="stopped"
                    if result.get("reason") == "operator_stop" or result.get("operator_stopped")
                    else gate.phase,
                    reason=result.get("error") or result.get("reason") or gate.reason,
                    disabled_confirmed=p is not None
                    and not p.state["enabled"]
                    and result["stop_error"] is None,
                )
                if status["disabled_confirmed"]:
                    status["velocity"] = 0
                (args.out / "task_status.json").write_text(json.dumps(status, indent=2))
            (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
    from bucket_robot.missions.completion import complete_mission

    return complete_mission(args, result, near_measurement, post_plan, observer, handoff, p, stages)


if __name__ == "__main__":
    raise SystemExit(main())
