"""Exercise the actual mission loop + UART packets, with no physical devices."""

import json, sys, tempfile, threading, time, unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import bucket_robot.missions.forward as runner
from tests.test_track_forward_probe import MCU, Sender


class FakeThread:
    def __init__(self, owner):
        self.owner = owner

    def start(self):
        self.owner.started = time.monotonic()

    def join(self, timeout=None):
        pass


class ScriptedObserver:
    def __init__(self, *args, **kwargs):
        self.end = threading.Event()
        self.thread = FakeThread(self)
        self.started = None

    def get(self):
        now = time.monotonic()
        elapsed = now - self.started
        fid = int(elapsed * 30)
        stamp = self.started + fid / 30
        target = None if 3.5 <= elapsed < 4.7 else dict(timestamp=stamp, offset_px=[0, 0])
        return dict(
            target=target, identity=1, frame_id=fid, metadata={}, recovery_count=int(elapsed >= 4.7)
        ), None

    def snapshot(self):
        return self.get()[0], None, None


class FixedPolicy:
    def __init__(self, *args):
        pass

    def reset(self):
        pass

    def set_a_rate_limit(self, value):
        pass

    def frame_angle(self, p, stamp):
        return p.state["servo_software_deg"]

    def next_a(self, target, frame, current, now):
        return current


class RecoverySessionTests(unittest.TestCase):
    def run_session(
        self,
        inject_fault=False,
        initial_missing=False,
        distance_first=False,
        post=False,
        left_first=False,
        left_offset=0,
        left_a=131,
        cruise=8,
        a_start_only=False,
    ):
        m = MCU()
        if left_first:
            m.a = left_a
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        out = root / "run"
        ref = root / "ref.json"
        ref.write_text(
            json.dumps(
                {
                    "operating_envelope": {
                        "A_command_degrees": [73, 132],
                        "B_command_mm": [0, 96],
                        "C_trial_hold_mm": 0,
                    },
                    "control_policy": {"original_sender_module_on_rk3588": "unused"},
                }
            )
        )
        left = root / "left.json"
        left.write_text(json.dumps({"left_A_command": 132, "forward_A_command": 73}))
        calibration = root / "calibration.yml"
        fs = cv2.FileStorage(str(calibration), cv2.FILE_STORAGE_WRITE)
        fs.write("P1", np.array([[700.0, 0, 640, 0], [0, 700, 360, 0], [0, 0, 1, 0]]))
        fs.release()
        args = [
            "test",
            "--a-only",
            "--out",
            str(out),
            "--reference",
            str(ref),
            "--left-reference",
            str(left),
            "--calibration",
            str(calibration),
            "--seed-image",
            "unused",
            "--seed-box",
            "400",
            "200",
            "800",
            "600",
            "--expect-a",
            str(left_a) if left_first else "97",
            "--seconds",
            "10",
            "--recover-loss",
        ]
        if distance_first:
            args += [
                "--distance-first",
                "--approach-model",
                "unused",
                "--approach-calibration",
                "unused",
            ]
        if post:
            args += ["--post-approach-plan", "--execute-post-geometry"]
        if cruise != 8:
            args += ["--cruise-speed", str(cruise)]
        if a_start_only:
            args.remove("--a-only")
            args += ["--track-b", "--a-start-only"]

        class Range:
            def __init__(self, *args, **kwargs):
                if not m.commands or m.commands[-1][8] or m.enabled:
                    raise AssertionError("model initialized before acknowledged UART disable")
                self.thread = type("Thread", (), {"start": lambda self: None})()
                self.started = time.monotonic()

            def evaluate(self, now, obs, image, stopped):
                reached = now - self.started > 3 and (cruise == 8 or post)
                near = reached and (stopped or not post)
                if post and near:
                    observed["near_confirmed"] = True
                return dict(
                    permit=not reached,
                    near=near,
                    static_verified=stopped,
                    reason="range_ready",
                    speed_cap=cruise,
                    measurement=dict(
                        valid=True,
                        frame_id=obs["frame_id"],
                        distance_m=0.2 if reached else 1.0,
                        lower_m=0.195 if reached else 0.995,
                        capture_time=now,
                        distance_kind="stereo_optical_midpoint_to_bucket_surface",
                    ),
                )

            def start_pulse(self, now):
                return True

            def close(self):
                pass

        actual = runner.CombinedPort
        observed = {"paused": False}

        class PostPolicy(FixedPolicy):
            def next_a(self, target, frame, current, now):
                return min(131, current + 1) if observed.get("near_confirmed") else current

        def post_execute(plan, reference, folder):
            observed["post_calls"] = observed.get("post_calls", 0) + 1
            observed["post_plan"] = plan
            if m.enabled or not 131 <= m.a <= 132:
                raise AssertionError("Post execution before disabled same-target left alignment")
            return dict(
                error=None,
                stop_error=None,
                final_state=dict(
                    enabled=False,
                    state=0,
                    system_fault=0,
                    motor_fault=0,
                    servo_software_deg=131,
                    lift_position_mm=0,
                    slide_position_mm=150,
                ),
            )

        def left_approach(**kwargs):
            observed["left_approach_called"] = True
            return dict(
                error=None,
                stop_error=None,
                reason="target_distance_reached",
                final_state=dict(
                    enabled=False,
                    state=0,
                    system_fault=0,
                    motor_fault=0,
                    servo_software_deg=131,
                    lift_position_mm=0,
                    slide_position_mm=0,
                ),
                last_measurement=dict(
                    valid=True,
                    distance_m=0.2,
                    distance_kind="stereo_optical_midpoint_to_bucket_surface",
                ),
            )

        def parked_alignment(*args, **kwargs):
            observed["upper_calls"] = observed.get("upper_calls", 0) + 1
            observed["upper_command_index"] = len(m.commands)
            return dict(moved=False, settled_at=time.monotonic(), constrained=False)

        class NoTargetObserver(ScriptedObserver):
            def get(self):
                obs, error = super().get()
                obs["target"] = None
                return obs, error

        class LeftObserver(ScriptedObserver):
            def snapshot(self):
                obs, error = self.get()
                if obs["target"]:
                    obs["target"]["box"] = [10, 10, 50, 50]
                    obs["target"]["offset_px"] = [left_offset, 0]
                return obs, np.zeros((80, 80, 3), np.uint8), error

        def port(limits, log):
            return actual(limits, log, port=m.path, sender_factory=Sender, check_holders=False)

        end = threading.Event()

        def controller():
            start = time.monotonic()
            while not end.wait(0.02):
                elapsed = time.monotonic() - start
                if inject_fault and elapsed > 2.7:
                    m.yaw = 40
                state = out / "task_status.json"
                if inject_fault and state.exists():
                    try:
                        s = json.loads(state.read_text())
                    except (ValueError, OSError):
                        continue
                    if s["phase"] == "fault_paused":
                        observed["paused"] = s.get("disabled_confirmed", False)
                        time.sleep(0.15)
                        (out / "STOP").touch()
                        return
                if elapsed > (12 if post else 5 if inject_fault else 10 if cruise > 20 else 7):
                    out.mkdir(exist_ok=True)
                    (out / "STOP").touch()
                    return

        thread = threading.Thread(target=controller, daemon=True)
        thread.start()
        try:
            with (
                patch.object(sys, "argv", args),
                patch.object(
                    runner,
                    "Observer",
                    NoTargetObserver
                    if initial_missing
                    else LeftObserver
                    if left_first
                    else ScriptedObserver,
                ),
                patch.object(runner, "CombinedPort", port),
                patch("bucket_robot.missions.vertical.ABCombinedPort", port),
                patch.object(runner.signal, "signal"),
                patch("bucket_robot.missions.forward_range.ForwardRange", Range),
                patch(
                    "bucket_robot.tracking.policy.FollowPolicy", PostPolicy if post else FixedPolicy
                ),
                patch("bucket_robot.missions.post_execute.execute_post", post_execute),
                patch("bucket_robot.missions.approach.run_approach", left_approach),
                patch("bucket_robot.missions.vertical.align_parked", parked_alignment),
            ):
                runner.main()
            summary = json.loads((out / "summary.json").read_text())
            states = [
                json.loads(x)["data"]["phase"]
                for x in (out / "serial.jsonl").read_text().splitlines()
                if json.loads(x)["kind"] == "recovery_state"
            ]
            observed["rx_count"] = sum(
                json.loads(x)["kind"] == "serial_rx"
                for x in (out / "serial.jsonl").read_text().splitlines()
            )
            return summary, states, list(m.commands), observed
        finally:
            end.set()
            thread.join(timeout=1)
            m.close()
            temp.cleanup()

    def test_distance_first_stops_before_left_angle_without_lateral_stage(self):
        summary, states, commands, _ = self.run_session(distance_first=True)
        self.assertIsNone(summary["error"])
        self.assertIsNone(summary["stop_error"])
        self.assertTrue(summary["drive_observed"])
        self.assertEqual(summary["reason"], "target_distance_reached")
        self.assertEqual(summary["approach"]["mode"], "distance_first")
        self.assertEqual(summary["final_state"]["servo_software_deg"], 97)
        self.assertFalse(summary["final_state"]["enabled"])
        self.assertTrue(all(f[2] == 0 for f in commands))

    def test80_continuous_main_sends80_holds_B_and_recovers_visual_loss(self):
        summary, states, commands, _ = self.run_session(
            distance_first=True, cruise=80, a_start_only=True
        )
        self.assertIsNone(summary["error"], summary)
        self.assertIsNone(summary["stop_error"])
        self.assertIn(80, [f[1] for f in commands if f[8] & 1])
        self.assertTrue(all(f[2] == 0 and f[5:8] == bytes([0, 0, 0]) for f in commands))
        self.assertIn("waiting_target", states)
        self.assertFalse(summary["final_state"]["enabled"])

    def test80_main_parks_upper_aim_then_uses20_for_final_forward_alignment(self):
        summary, _, commands, observed = self.run_session(
            distance_first=True, post=True, cruise=80, a_start_only=True
        )
        self.assertIsNone(summary["error"], summary)
        self.assertEqual(observed.get("upper_calls"), 1)
        self.assertEqual(observed.get("post_calls"), 1)
        self.assertTrue(all(f[1] <= 20 for f in commands[observed["upper_command_index"] :]))
        self.assertEqual(summary["final_state"]["slide_position_mm"], 150)

    def test_post_phase_preserves_observer_and_aligns_before_sideways_extension(self):
        summary, states, commands, observed = self.run_session(distance_first=True, post=True)
        self.assertIsNone(summary["error"], summary)
        self.assertEqual(summary["reason"], "post_left_aligned")
        self.assertEqual(observed.get("post_calls"), 1)
        self.assertEqual(observed["post_plan"]["geometry"]["A_command"], 97)
        self.assertEqual(summary["final_state"]["slide_position_mm"], 150)
        self.assertTrue(any(s.startswith("post_forward_align_") for s in states))
        self.assertFalse(summary["final_state"]["enabled"])

    def test_left_angle_first_still_executes_post_phase_after_range_approach(self):
        summary, _, _, observed = self.run_session(distance_first=True, post=True, left_first=True)
        self.assertIsNone(summary["error"], summary)
        self.assertTrue(observed.get("left_approach_called"))
        self.assertEqual(observed.get("post_calls"), 1)
        self.assertEqual(observed["post_plan"]["geometry"]["A_command"], 131)

    def test_main_left_boundary_offcenter_far_bucket_enters_lateral_then_extension(self):
        summary, _, commands, observed = self.run_session(
            distance_first=True,
            post=True,
            left_first=True,
            left_a=132,
            left_offset=-59,
        )
        self.assertIsNone(summary["error"], summary)
        self.assertIsNone(summary["stop_error"])
        self.assertEqual(summary["reason"], "left_reference_reached")
        self.assertTrue(observed.get("left_approach_called"))
        self.assertEqual(observed.get("post_calls"), 1)
        self.assertEqual(summary["final_state"]["slide_position_mm"], 150)
        self.assertTrue(all(f[1] == 0 for f in commands))

    def test_visual_loss_decelerates_parks_and_resumes_without_restart(self):
        summary, states, commands, _ = self.run_session()
        self.assertIsNone(summary["error"])
        self.assertIsNone(summary["stop_error"])
        self.assertIn("decelerating", states)
        self.assertIn("waiting_target", states)
        self.assertGreaterEqual(states.count("resuming"), 2)
        values = [f[1] if f[8] else 0 for f in commands]
        first_peak = values.index(8)
        stop = values.index(0, first_peak + 1)
        self.assertTrue(any(0 < v < 8 for v in values[first_peak + 1 : stop]))
        self.assertIn(8, values[stop + 1 :])
        self.assertTrue(all(f[2] == 0 and f[5:8] == bytes([0, 0, 0]) for f in commands))
        self.assertFalse(summary["final_state"]["enabled"])

    def test_initial_target_absence_keeps_telemetry_live_without_enabling_motion(self):
        summary, _, commands, observed = self.run_session(initial_missing=True)
        self.assertGreater(observed["rx_count"], 10)
        self.assertTrue(commands)
        self.assertTrue(all(f[8] == 0 for f in commands))
        self.assertFalse(summary["final_state"]["enabled"])

    def test_hard_fault_confirms_disable_and_leaves_task_alive_paused(self):
        summary, states, commands, observed = self.run_session(True)
        self.assertIn("heading_drift", summary["error"])
        self.assertIsNone(summary["stop_error"])
        self.assertTrue(observed["paused"])
        self.assertFalse(summary["final_state"]["enabled"])
        self.assertEqual(commands[-1][1:3], bytes([0, 0]))
        self.assertEqual(commands[-1][8], 0)


if __name__ == "__main__":
    unittest.main()
