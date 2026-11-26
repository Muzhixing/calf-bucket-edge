"""Opt-in field execution after stopped same-bucket geometry planning.

Lateral travel uses the operator's single-run timing reference, not odometry.
Every mechanism/chassis handoff requires an acknowledged disabled state.
"""

import json
import math
from pathlib import Path
import time

from bucket_robot.motion.arm import ArmPort, Limits, pose
from bucket_robot.motion.chassis import ChassisPort, HeldArmLimits


def lateral_timing(target_mm):
    if type(target_mm) not in (int, float) or not math.isfinite(target_mm):
        raise ValueError("invalid_lateral_target")
    # Restrict this first execution to the neighborhood of the observed 3s run.
    seconds = target_mm / 32.5
    if not 0.6 <= seconds <= 4.0:
        raise ValueError("lateral_target_outside_timing_trial_range")
    return dict(
        target_mm=target_mm,
        speed_percent=20,
        seconds=seconds,
        estimated_travel_mm_interval=[seconds * 95 / 3, seconds * 100 / 3],
        source="operator_20percent_left_3s_95_to_100mm_20261126",
        metric_feedback=False,
    )


class ExtensionPort(ArmPort):
    """Shared bounded arm transaction; C only, stationary chassis."""

    def extend(self):
        if self.phase != "idle" or self.state["enabled"]:
            raise RuntimeError("extension_requires_disabled_chassis")
        self.check_state(self.state)
        current = pose(self.state)
        if not 131 <= current[0] <= 132 or current[2] != 0:
            raise RuntimeError("extension_pose_not_ready")
        q = [current[0], current[1], 150]
        self.limits.check(q)
        if time.monotonic() - self.rx_at > 0.35:
            raise RuntimeError("stale_telemetry")
        self.origin, self.target = current, q
        self.arrived_at = None
        self.move_counter = (self.expected + 1) % 256
        self.phase, self.attempted = "moving", True
        self.started = time.monotonic()
        self.deadline = self.started + 150 / self.limits.c_max_speed + 4
        if not self.gateway.write_control_frame(self.frame(q, True), repeats=1):
            raise RuntimeError("extension_write_failed_no_retry")


def execute_post(
    plan, reference, folder, *, chassis_factory=ChassisPort, extension_factory=ExtensionPort
):
    """Forward same-target alignment was completed by the original mission loop."""
    timing = lateral_timing(plan["sequence"][2]["relative_target_mm"])
    result = dict(
        error=None,
        stop_error=None,
        timing=timing,
        C_extension_sent=False,
        chassis_command_sent=False,
        lateral_completion="not_started",
    )
    port = None

    def status(phase):
        (folder / "task_status.json").write_text(
            json.dumps(
                dict(
                    phase=phase,
                    reason="post_geometry_trial",
                    timing=timing,
                    disabled_confirmed=port is not None
                    and not port.state["enabled"]
                    and port.phase == "idle",
                    velocity=0 if phase == "complete" else None,
                ),
                indent=2,
            )
        )

    with (folder / "serial.jsonl").open("a", buffering=1) as log:

        def record(kind, data):
            log.write(json.dumps(dict(monotonic=time.monotonic(), kind=kind, data=data)) + "\n")

        def wait():
            yaw = port.state["yaw_deg"]
            while port.phase != "idle":
                if (folder / "STOP").exists():
                    raise KeyboardInterrupt("operator_stop")
                port.tick()
                if (
                    port.phase != "stopping"
                    and abs((port.state["yaw_deg"] - yaw + 180) % 360 - 180) > 3
                ):
                    raise RuntimeError("heading_drift")
                time.sleep(0.005)

        try:
            if (folder / "STOP").exists():
                raise KeyboardInterrupt("operator_stop")
            port = chassis_factory(HeldArmLimits(reference), record)
            if not 131 <= pose(port.state)[0] <= 132 or pose(port.state)[2] not in (0, 150):
                raise RuntimeError("post_left_alignment_not_confirmed")
            result["initial_state"] = dict(port.state)
            already_extended = pose(port.state)[2] == 150
            if not already_extended:
                status("post_left_translate_timed")
                port.start_pulse("left", 20, timing["seconds"])
                result["chassis_command_sent"] = True
                wait()
                if not port.drive_observed:
                    raise RuntimeError("lateral_drive_not_observed")
                result["lateral_completion"] = "timed_estimate_no_odometry"
            else:
                result["lateral_completion"] = "skipped_C_already_extended_no_repeated_motion"
            result["lateral_state"] = dict(port.state)
            result["stop_error"] = port.shutdown()
            if result["stop_error"]:
                raise RuntimeError("lateral_stop_unconfirmed")
            port = None
            if (folder / "STOP").exists():
                raise KeyboardInterrupt("operator_stop")
            port = extension_factory(Limits(reference), record)
            if pose(port.state) != pose(result["lateral_state"]):
                raise RuntimeError("pose_changed_between_post_stages")
            status("post_raising_B_tracking_paused")
            result["tracking_paused"] = True
            while pose(port.state)[1] > 0:
                if (folder / "STOP").exists():
                    raise KeyboardInterrupt("operator_stop")
                while time.monotonic() < port.ready_at:
                    port.tick()
                    time.sleep(0.005)
                q = pose(port.state)
                port.move([q[0], max(0, q[1] - 20), q[2]])
                wait()
            result["B_top_confirmed"] = pose(port.state)[1] == 0
            if not already_extended:
                status("post_extending_C")
                port.extend()
                result["C_extension_sent"] = True
                wait()
            if pose(port.state)[2] != 150 or port.state["enabled"]:
                raise RuntimeError("C150_disabled_not_confirmed")
            result["phase"] = "complete"
        except BaseException as exc:
            result["error"] = repr(exc)
            result["operator_stopped"] = isinstance(exc, KeyboardInterrupt)
            result["phase"] = "stopped" if result["operator_stopped"] else "fault_paused"
        finally:
            if port is not None:
                result["stop_error"] = port.shutdown() or result["stop_error"]
                result["final_state"] = dict(port.state)
            else:
                result["final_state"] = result.get("lateral_state")
            if result["stop_error"]:
                result["phase"] = "fault_paused"
            status(result["phase"])
            (folder / "post_execution.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    import argparse
    import signal
    from bucket_robot.missions.post_geometry import alignment_extension_plan

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= time.time() - args.plan.stat().st_mtime <= 900:
        raise RuntimeError("resume_plan_expired")
    saved = json.loads(args.plan.read_text())
    geometry = saved["geometry"]
    # Recompute from the saved measured inputs instead of trusting a command list.
    plan = alignment_extension_plan(geometry["range_m"], geometry["A_command"])
    args.out.mkdir(parents=True, exist_ok=False)

    def interrupted(signum, frame):
        raise KeyboardInterrupt("signal_" + str(signum))

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    result = execute_post(plan, args.reference, args.out)
    result["resumed_plan"] = str(args.plan)
    (args.out / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return 2 if result["error"] or result["stop_error"] else 0
