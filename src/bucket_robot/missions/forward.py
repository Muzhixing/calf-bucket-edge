from bucket_robot.missions.lifecycle import TaskPause as TaskPause, park_fault as park_fault

"""Bounded single-bucket A tracking + forward probe. One serial owner, B/C held.

Native vision must already run without a serial interface. No startup sweep,
zero registration, lateral motion, C extension or dispensing is performed.
"""
import argparse, json, math, signal, threading, time
from pathlib import Path
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


class Observer:
    def __init__(
        self,
        port,
        folder,
        seed_image,
        box,
        bearing_tracker=False,
        recover_loss=False,
        range_calibration=None,
    ):
        self.port = port
        self.range_calibration = range_calibration
        self.latest_raw = None
        self.folder = folder
        self.seed_image = seed_image
        self.box = box
        self.bearing_tracker = bearing_tracker
        self.recover_loss = recover_loss
        self.latest = None
        self.latest_image = None
        self.error = None
        self.end = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        import cv2, numpy as np
        from bucket_robot.perception.target import NativeCapture, TargetLock, weighted_ncc
        from bucket_robot.perception.session import new_target_session

        cv2.setNumThreads(1)
        geometry = None
        if self.range_calibration:
            from bucket_robot.perception.assisted_range.stereo import Geometry

            geometry = Geometry(self.range_calibration)
        cap = NativeCapture(self.port, **({"stereo": True} if geometry is not None else {}))
        manager = new_target_session(same_target=self.recover_loss, bearing=self.bearing_tracker)
        self.target_session = manager
        first = True
        count = 0
        last_frame = -1
        failures = 0
        last_saved_loss = None
        automatic = TargetLock("auto") if self.seed_image is None else None
        try:
            with (self.folder / "vision.jsonl").open("x", buffering=1) as log:
                while not self.end.is_set():
                    try:
                        image, meta = cap.read()
                        raw = image if geometry is not None else None
                        if geometry is not None:
                            image = cv2.remap(raw[:, :1280], *geometry.maps[0], cv2.INTER_LINEAR)
                        last_frame = meta["frame_id"]
                        if first:
                            if automatic is not None:
                                acquired = automatic.update(image, meta)
                                if acquired is None:
                                    obs = dict(
                                        target=None,
                                        identity=None,
                                        frame_id=last_frame,
                                        metadata=meta,
                                        tracking_state="searching",
                                        reason="automatic_acquisition",
                                        recovery_count=0,
                                    )
                                    with self.lock:
                                        self.latest = obs
                                        self.latest_image = image
                                    log.write(json.dumps(obs) + "\n")
                                    continue
                                self.box = acquired.get("raw_box", acquired["box"])
                                cv2.imwrite(str(self.folder / "automatic_target.png"), image)
                                (self.folder / "automatic_target.json").write_text(
                                    json.dumps(
                                        dict(
                                            box=self.box,
                                            frame_id=last_frame,
                                            source="original_TargetLock_auto",
                                        )
                                    )
                                )
                            else:
                                seed = cv2.imread(self.seed_image)
                                x1, y1, x2, y2 = self.box
                                if seed is None:
                                    raise RuntimeError("missing_seed_image")
                                a = cv2.cvtColor(seed[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                                b = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                                if (
                                    a.shape != b.shape
                                    or weighted_ncc(a, b, np.ones(a.shape, np.float32)) < 0.85
                                ):
                                    raise RuntimeError("scene_changed_since_seed")
                            manager.select(image, self.box, meta["arrival"])
                            first = False
                        target = manager.update(image, meta)
                        failures = 0
                    except Exception as exc:
                        if not self.recover_loss:
                            raise
                        manager.missing(repr(exc))
                        failures += 1
                        obs = {
                            "target": None,
                            "identity": 1,
                            "frame_id": last_frame,
                            "metadata": {},
                            "tracking_state": "recovering",
                            "reason": repr(exc),
                            "recovery_count": manager.reacquired,
                        }
                        with self.lock:
                            self.latest = obs
                        log.write(json.dumps(obs) + "\n")
                        cap = NativeCapture(
                            self.port, **({"stereo": True} if geometry is not None else {})
                        )
                        self.end.wait(min(1.0, 0.05 * failures))
                        continue
                    if (
                        self.recover_loss
                        and manager.state == "recovering"
                        and manager.reacquired != last_saved_loss
                        and manager.previous_image is not None
                        and manager.latest is not None
                    ):
                        suffix = str(manager.reacquired)
                        cv2.imwrite(
                            str(self.folder / ("recovery_reference_" + suffix + ".png")),
                            manager.previous_image,
                        )
                        cv2.imwrite(str(self.folder / ("recovery_frame_" + suffix + ".png")), image)
                        (self.folder / ("recovery_reference_" + suffix + ".json")).write_text(
                            json.dumps(dict(reference=manager.latest, current=meta), indent=2)
                        )
                        last_saved_loss = manager.reacquired
                    if manager.state != "tracking" or target is None:
                        if not self.recover_loss:
                            cv2.imwrite(str(self.folder / "failure.jpg"), image)
                            if self.bearing_tracker and manager.previous_image is not None:
                                cv2.imwrite(
                                    str(self.folder / "failure_previous.png"),
                                    manager.previous_image,
                                )
                            (self.folder / "failure.json").write_text(
                                json.dumps({"reason": manager.reason, "metadata": meta}, indent=2)
                            )
                            raise RuntimeError("target_lost_" + manager.reason)
                    if manager.generation != 1 or (manager.reacquired and not self.recover_loss):
                        raise RuntimeError("target_identity_changed")
                    obs = {
                        "target": target,
                        "identity": manager.generation,
                        "frame_id": meta["frame_id"],
                        "metadata": meta,
                        "tracking_state": manager.state,
                        "reason": manager.reason,
                        "recovery_count": manager.reacquired,
                    }
                    with self.lock:
                        self.latest = obs
                        self.latest_image = image
                        self.latest_raw = raw
                    log.write(json.dumps(obs, allow_nan=False) + "\n")
                    if count % 15 == 0:
                        cv2.imwrite(str(self.folder / "latest.jpg"), image)
                    count += 1
        except BaseException as e:
            with self.lock:
                self.error = repr(e)
        finally:
            if hasattr(manager, "close"):
                manager.close()

    def get(self):
        with self.lock:
            return self.latest, self.error

    def snapshot(self):
        with self.lock:
            return self.latest, self.latest_image, self.error

    def range_snapshot(self):
        with self.lock:
            return self.latest, self.latest_image, self.latest_raw, self.error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--left-reference", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--seed-image")
    ap.add_argument("--seed-box", nargs=4, type=int)
    ap.add_argument("--native-port", type=int, default=8899)
    ap.add_argument(
        "--auto-target",
        action="store_true",
        help="Acquire with original automatic tracker; retain that identity for this mission",
    )
    ap.add_argument("--expect-a", type=int, default=97)
    ap.add_argument("--seconds", type=float, default=5)
    ap.add_argument("--observe-only", action="store_true")
    ap.add_argument(
        "--until-left",
        action="store_true",
        help="Continuous alignment; drive budget maximum 60 s legacy / 120 s recovery mission",
    )
    ap.add_argument(
        "--bearing-tracker",
        action="store_true",
        help="Require current associated detector evidence for visible bucket bearing, including vertically clipped views",
    )
    ap.add_argument(
        "--center-only",
        action="store_true",
        help="Verify A-only acquisition and stop before any chassis motion",
    )
    ap.add_argument(
        "--recover-loss",
        action="store_true",
        help="Keep task alive, ramp down on visual loss and resume only a verified same target",
    )
    axes = ap.add_mutually_exclusive_group()
    axes.add_argument(
        "--a-only",
        action="store_true",
        help="Explicit A-only diagnostic; ordinary missions use --track-b",
    )
    axes.add_argument(
        "--track-b",
        action="store_true",
        help="Pause the chassis and track vertical error with B before resuming",
    )
    ap.add_argument("--expect-b", type=int, default=0)
    ap.add_argument(
        "--post-approach-plan",
        action="store_true",
        help="Record C150 then left100mm plan; wait for a validated displacement source",
    )
    ap.add_argument(
        "--approach-surface-band",
        nargs=4,
        type=float,
        help="Scene-verified exterior wall region as relative box coordinates",
    )
    ap.add_argument(
        "--approach-model",
        type=Path,
        help="After confirmed left alignment, run bounded classical/neural left approach",
    )
    ap.add_argument(
        "--approach-calibration", type=Path, help="Hash-bound original JSON stereo calibration"
    )
    ap.add_argument(
        "--distance-first",
        action="store_true",
        help="Range from lock; stop near 20cm at any A angle",
    )
    ap.add_argument("--cruise-speed", type=int, default=8, choices=range(1, 81))
    ap.add_argument("--a-start-only", action="store_true")
    ap.add_argument("--execute-post-geometry", action="store_true")
    args = ap.parse_args()
    if args.a_start_only and not args.track_b:
        ap.error("A-priority departure requires --track-b for parked upper alignment")
    if args.cruise_speed > 20 and not (args.distance_first or args.observe_only):
        ap.error("high cruise speed requires distance-first assisted ranging")
    if args.execute_post_geometry and not (args.distance_first and args.post_approach_plan):
        ap.error("post geometry execution requires distance-first and post-approach-plan")
    if not (args.track_b or args.a_only or args.observe_only or args.center_only):
        ap.error(
            "select --track-b for shared A/B tracking, or explicitly --a-only for a diagnostic"
        )
    if args.auto_target:
        if args.seed_image or args.seed_box:
            ap.error("automatic acquisition cannot use a manual seed")
    elif not args.seed_image or not args.seed_box:
        ap.error("provide --auto-target or both seed arguments")
    if args.track_b and not args.recover_loss:
        ap.error("--track-b requires --recover-loss")
    if args.distance_first and (
        not args.recover_loss or not args.approach_model or args.observe_only
    ):
        ap.error("distance-first requires recovering motion and assisted ranging")
    if args.post_approach_plan and not args.approach_model:
        ap.error("post-approach plan requires range approach")
    from bucket_robot.missions.recovery import RecoveryGate, RecoverySettings
    from bucket_robot.tracking.aim import horizontal_aligned

    gate = (
        RecoveryGate(
            args.seconds,
            settings=RecoverySettings(
                cruise=args.cruise_speed,
                acceleration_per_s=40 if args.cruise_speed > 20 else 8,
                deceleration_per_s=80 if args.cruise_speed > 20 else 10,
                initial_confirm_s=0 if args.a_start_only else 0.7,
                initial_confirm_frames=1 if args.a_start_only else 3,
            ),
        )
        if args.recover_loss
        else ForwardGate(args.seconds, until_left=args.until_left)
    )
    if bool(args.approach_model) != bool(args.approach_calibration):
        ap.error("approach model and calibration must be supplied together")
    ref = json.loads(args.left_reference.read_text())
    if ref["left_A_command"] != 132 or ref["forward_A_command"] != 73:
        raise RuntimeError("reference_mismatch")
    args.out.mkdir(parents=True, exist_ok=False)
    observer = Observer(
        args.native_port,
        args.out,
        args.seed_image,
        args.seed_box,
        args.bearing_tracker,
        args.recover_loss,
        **({"range_calibration": args.approach_calibration} if args.distance_first else {}),
    )

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
            while observer.get()[0] is None or (
                args.recover_loss and observer.get()[0]["target"] is None
            ):
                if p is not None:
                    p.tick()
                if observer.get()[1]:
                    raise RuntimeError(observer.get()[1])
                if (args.out / "STOP").exists():
                    raise KeyboardInterrupt("operator_stop")
                if (not args.recover_loss or args.observe_only) and time.monotonic() - started > 8:
                    raise RuntimeError("no_fresh_target")
                if args.recover_loss and time.monotonic() - last_initial_status >= 1:
                    waiting = observer.get()[0] or {}
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
                    obs, error = observer.get()
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
            while True:
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
                policy.set_a_rate_limit(
                    30
                    if args.a_start_only
                    else 10
                    if gate.started is None and not post_aligning
                    else 30
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
                        lateral_timing(post_plan["sequence"][2]["relative_target_mm"])
                        record("stopped_triangle_plan", post_plan)
                        (args.out / "post_approach.json").write_text(
                            json.dumps(post_plan, indent=2)
                        )
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
                            if args.a_start_only
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
                        speed_cap=range_plan.get("speed_cap", 8)
                        if range_plan
                        else 20
                        if post_aligning and args.a_start_only
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
    if near_measurement is not None and not result["error"] and not result["stop_error"]:
        result["approach"] = dict(
            mode="distance_first",
            reason="target_distance_reached",
            error=None,
            stop_error=None,
            final_state=result["final_state"],
            last_measurement=near_measurement,
        )
        if args.post_approach_plan:
            from bucket_robot.missions.post_approach import pending_extension_plan

            result["post_approach"] = post_plan or pending_extension_plan(result["approach"])
            (args.out / "post_approach.json").write_text(
                json.dumps(result["post_approach"], indent=2)
            )
        (args.out / "task_status.json").write_text(
            json.dumps(
                dict(
                    phase="waiting_displacement_measurement"
                    if args.post_approach_plan
                    else "complete",
                    reason="target_distance_reached",
                    disabled_confirmed=True,
                ),
                indent=2,
            )
        )
        (args.out / "summary.json").write_text(json.dumps(result, indent=2))
    if (
        args.approach_model
        and not args.observe_only
        and not args.center_only
        and not result["error"]
        and not result["stop_error"]
        and result.get("reason") == "left_reference_reached"
    ):
        try:
            latest = observer.snapshot()
            if (
                latest[0]
                and latest[0].get("target") is not None
                and latest[1] is not None
                and not latest[2]
            ):
                obs, image, error = latest
            elif handoff is not None:
                obs, image, error = handoff
            else:
                raise RuntimeError("invalid_approach_handoff")
            if error or image is None or obs is None or obs.get("target") is None:
                raise RuntimeError("invalid_approach_handoff")
            import cv2
            from bucket_robot.missions.approach import run_approach

            seed = args.out / "left_handoff.png"
            if not cv2.imwrite(str(seed), image):
                raise RuntimeError("handoff_image_write_failed")
            if args.recover_loss:
                (args.out / "task_status.json").write_text(
                    json.dumps(
                        dict(phase="left_approach", status_file="approach/task_status.json"),
                        indent=2,
                    )
                )
            if hasattr(observer.thread, "is_alive") and observer.thread.is_alive():
                raise RuntimeError("tracking_observer_handoff_not_stopped")
            result["approach"] = run_approach(
                out=args.out / "approach",
                reference=args.reference,
                calibration=args.approach_calibration,
                model=args.approach_model,
                seed_image=seed,
                seed_box=obs["target"].get("raw_box", obs["target"]["box"]),
                native_port=args.native_port,
                execute=True,
                bearing_tracker=args.bearing_tracker,
                recover_loss=args.recover_loss,
                seed_timestamp=obs["target"]["timestamp"],
                verified_surface_band=args.approach_surface_band,
                continuous=args.distance_first,
                existing_port=p if args.distance_first else None,
                target_session=getattr(observer, "target_session", None)
                if args.recover_loss
                else None,
                align_upper_after_stop=args.a_start_only,
            )
            result["operator_stopped"] = result["approach"].get("operator_stopped", False)
            if result["approach"]["error"] or result["approach"]["stop_error"]:
                result["error"] = "approach_failed"
            if args.post_approach_plan and not result["error"] and not result["operator_stopped"]:
                from bucket_robot.missions.post_approach import pending_extension_plan

                result["post_approach"] = pending_extension_plan(result["approach"])
                (args.out / "post_approach.json").write_text(
                    json.dumps(result["post_approach"], indent=2)
                )
            if args.recover_loss:
                final = result["approach"].get("final_state") or {}
                (args.out / "task_status.json").write_text(
                    json.dumps(
                        dict(
                            phase="stopped"
                            if result["operator_stopped"]
                            else "fault_paused"
                            if result["error"]
                            else "waiting_displacement_measurement"
                            if args.post_approach_plan
                            else "complete",
                            reason=result["approach"].get("reason") or result["error"],
                            disabled_confirmed=bool(final)
                            and not final.get("enabled", True)
                            and not result["approach"]["stop_error"],
                        ),
                        indent=2,
                    )
                )
        except BaseException as exc:
            result["error"] = repr(exc)
            result["operator_stopped"] = isinstance(exc, KeyboardInterrupt)
        finally:
            if p is not None and p.fd is not None:
                result["stop_error"] = p.shutdown() or result["stop_error"]
        (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    if (
        args.execute_post_geometry
        and (result.get("reason") == "post_left_aligned" or result.get("approach"))
        and not result.get("operator_stopped")
        and not result["error"]
        and not result["stop_error"]
    ):
        from bucket_robot.missions.post_execute import execute_post

        result["post_execution"] = execute_post(
            post_plan or result["post_approach"], args.reference, args.out
        )
        result["error"] = result["post_execution"]["error"]
        result["stop_error"] = result["post_execution"]["stop_error"]
        result["final_state"] = result["post_execution"].get("final_state")
        result["operator_stopped"] = result["post_execution"].get("operator_stopped", False)
        (args.out / "summary.json").write_text(json.dumps(result, indent=2))
        print("POST_EXECUTION " + json.dumps(result["post_execution"]), flush=True)
    if (
        args.recover_loss
        and not args.observe_only
        and (result["error"] or result["stop_error"])
        and not result.get("operator_stopped")
    ):
        park_fault(args.out, result)
    return 0 if not result["error"] and not result["stop_error"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
