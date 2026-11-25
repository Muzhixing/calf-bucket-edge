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
    def __init__(self, *args):
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

    def frame_angle(self, p, stamp):
        return p.state["servo_software_deg"]

    def next_a(self, target, frame, current, now):
        return current


class RecoverySessionTests(unittest.TestCase):
    def run_session(self, inject_fault=False):
        m = MCU()
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
            "97",
            "--seconds",
            "10",
            "--recover-loss",
        ]
        actual = runner.CombinedPort
        observed = {"paused": False}

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
                if elapsed > (5 if inject_fault else 7):
                    out.mkdir(exist_ok=True)
                    (out / "STOP").touch()
                    return

        thread = threading.Thread(target=controller, daemon=True)
        thread.start()
        try:
            with (
                patch.object(sys, "argv", args),
                patch.object(runner, "Observer", ScriptedObserver),
                patch.object(runner, "CombinedPort", port),
                patch.object(runner.signal, "signal"),
                patch("bucket_robot.tracking.policy.FollowPolicy", FixedPolicy),
            ):
                runner.main()
            summary = json.loads((out / "summary.json").read_text())
            states = [
                json.loads(x)["data"]["phase"]
                for x in (out / "serial.jsonl").read_text().splitlines()
                if json.loads(x)["kind"] == "recovery_state"
            ]
            return summary, states, list(m.commands), observed
        finally:
            end.set()
            thread.join(timeout=1)
            m.close()
            temp.cleanup()

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
