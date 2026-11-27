"""Opt-in field execution after stopped same-bucket geometry planning.

Lateral travel uses the operator's single-run timing reference, not odometry.
Every mechanism/chassis handoff requires an acknowledged disabled state.
"""

import json
from pathlib import Path
import time

from bucket_robot.motion.arm import Limits, pose
from bucket_robot.missions.timed_translate import TimedTranslate, check_geometry_admission
from bucket_robot.motion.extension import ExtensionPort as ExtensionPort
from bucket_robot.motion.chassis import ChassisPort, HeldArmLimits


def lateral_timing(target_mm):
    check_geometry_admission(target_mm)
    action = TimedTranslate()
    seconds = action.duration_s
    return dict(
        diagnostic_half_left_mm=target_mm,
        speed_percent=action.speed_percent,
        seconds=seconds,
        estimated_travel_mm_interval=[seconds * 95 / 3, seconds * 100 / 3],
        source="operator_20percent_left_3s_95_to_100mm_20261126",
        metric_feedback=False,
        timing_policy=action.policy,
        execution=action.as_dict(),
    )


def execute_post(
    plan, reference, folder, *, chassis_factory=ChassisPort, extension_factory=ExtensionPort
):
    """Forward same-target alignment was completed by the original mission loop."""
    timing = lateral_timing(plan["sequence"][2]["diagnostic_half_left_mm"])
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
                    execution_progress={
                        key: result[key]
                        for key in ("initial_state", "lateral_completion", "B_top_confirmed")
                        if key in result
                    },
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
                port.start_pulse(
                    timing["execution"]["direction"], timing["speed_percent"], timing["seconds"]
                )
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
