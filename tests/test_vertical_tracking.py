import json, sys, tempfile, threading, time, unittest, types
from pathlib import Path
from unittest.mock import patch
import cv2, numpy as np
import bucket_robot.missions.forward as runner
from bucket_robot.missions.vertical import ABCombinedPort, VerticalSupervisor
from bucket_robot.missions.recovery import RecoveryGate
from tests.test_track_forward_probe import MCU, Sender
from tests.test_recovery_session import FakeThread
from bucket_robot.tracking.policy import FollowPolicy


class Policy(FollowPolicy):
    def __init__(self, limits, *args):
        super().__init__(limits, *(args or (700, 700, 640, 360)))


class VerticalTests(unittest.TestCase):
    def supervisor(self):
        return VerticalSupervisor(
            Policy(
                types.SimpleNamespace(
                    b_min=0,
                    b_max=96,
                    a_min=73,
                    a_max=132,
                    b_far_gain=1,
                    b_fast_error=12.5,
                    b_max_step=24,
                    b_schedule_interval=0.12,
                )
            )
        )

    def test_clipped_bucket_reuses_original_B_gain_and_step_without_five_mm_rule(self):
        v = self.supervisor()
        t = dict(
            timestamp=1,
            offset_px=[0, 200],
            box=[300, 450, 900, 712],
            raw_box=[300, 450, 900, 719],
            vertical_clipped=True,
        )
        expected = v.policy.next_b(t, 0, {})
        self.assertEqual(expected, 24)
        self.assertEqual(v.plan(1, t, 0, {})["goal"], expected)
        t["timestamp"] = 1.1
        self.assertEqual(
            v.plan(1.1, t, 96, {}, [100, 96, 0], [100, 96, 0]), dict(ready=False, goal=None)
        )
        self.assertEqual(v.reason, "B_range_limit")

    def test_B_limit_allows_measured_horizontal_tracking_only_after_B_stops(self):
        for clipped in (False, True):
            v = self.supervisor()
            target = dict(
                timestamp=1,
                offset_px=[0, 120],
                vertical_clipped=clipped,
                box=[400, 450, 800, 712],
                horizontal_bearing_valid=True,
                horizontal_clipped=False,
            )
            plan = v.plan(1, target, 96, {}, [100, 96, 0], [100, 96, 0])
            self.assertEqual(plan, dict(ready=True, goal=None))
            self.assertEqual(v.reason, "B_range_holding")
            self.assertFalse(v.plan(1, target, 96, {}, [100, 96, 0], [100, 95, 0])["ready"])
            self.assertFalse(v.plan(2, target, 96, {}, [100, 96, 0], [100, 96, 0])["ready"])
            target["horizontal_clipped"] = True
            self.assertFalse(v.plan(1, target, 96, {}, [100, 96, 0], [100, 96, 0])["ready"])

    def test_fresh_full_target_controls_height_and_stale_target_cannot(self):
        v = self.supervisor()
        t = dict(timestamp=1, offset_px=[0, 24], box=[500, 300, 600, 400])
        self.assertEqual(v.plan(1, t, 0, {})["goal"], 24)
        self.assertIsNone(v.plan(2, t, 0, {})["goal"])
        t.update(timestamp=2, offset_px=[0, 0])
        self.assertTrue(v.plan(2, t, 24, {})["ready"])

    def test_vertical_pause_ramps_down_and_prevents_early_left_handoff(self):
        g = RecoveryGate(60)
        for i in range(41):
            g.evaluate(i * 0.05, dict(timestamp=i * 0.05, offset_px=[0, 0]), 1, 100, True)
        values = []
        for i in range(1, 21):
            t = 2 + i * 0.05
            speed, reason = g.evaluate(
                t,
                dict(timestamp=t, offset_px=[0, 100]),
                1,
                131,
                True,
                commanded_a=131,
                motion_permitted=False,
            )
            self.assertIsNone(reason)
            values.append(speed)
        self.assertEqual(values[-1], 0)
        self.assertEqual(values, sorted(values, reverse=True))


class VerticalSessionTests(unittest.TestCase):
    def test_real_uart_loop_lowers_B_before_forward_and_preserves_height(self):
        self.run_session()

    def test_loss_during_B_move_parks_then_recovers_without_chassis_motion(self):
        self.run_session(True)

    def run_session(self, inject_loss=False):
        class BMcu(MCU):
            def __init__(self):
                self.b = 0.0
                super().__init__()

            def packet(self):
                goal = self.commands[-1][5] if self.commands else 0
                if self.enabled:
                    self.b += max(-1.0, min(1.0, goal - self.b))
                f = bytearray(super().packet())
                f[6:8] = int(self.b * 10).to_bytes(2, "little")
                if self.enabled and abs(goal - self.b) > 0.01:
                    f[4] = 2
                f[18] = 0
                for v in f[1:18]:
                    f[18] ^= v
                return f

        m = BMcu()
        end = threading.Event()
        lost_at = [None]

        class Observer:
            def __init__(self, *args):
                self.end = threading.Event()
                self.thread = FakeThread(self)

            def get(self):
                now = time.monotonic()
                fid = int(now * 30)
                stamp = fid / 30
                if inject_loss and m.b >= 5 and lost_at[0] is None:
                    lost_at[0] = now
                missing = lost_at[0] is not None and now - lost_at[0] < 0.25
                return dict(
                    target=None
                    if missing
                    else dict(
                        timestamp=stamp,
                        offset_px=[40 if 0 < m.b < 24 else 0, 24 - m.b],
                        box=[500, 300, 600, 400],
                    ),
                    frame_id=fid,
                    identity=1,
                    metadata={},
                    recovery_count=0,
                ), None

            def snapshot(self):
                return self.get()[0], None, None

        try:
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                out = root / "out"
                ref = root / "ref.json"
                left = root / "left.json"
                cal = root / "cal.yml"
                ref.write_text(
                    json.dumps(
                        {
                            "operating_envelope": {
                                "A_command_degrees": [73, 132],
                                "B_command_mm": [0, 96],
                                "C_trial_hold_mm": 0,
                                "C_command_mm": [0, 150],
                            },
                            "control_policy": {"original_sender_module_on_rk3588": "unused"},
                            "tracking_motion": {"b_far_position_gain": 1},
                        }
                    )
                )
                left.write_text(json.dumps(dict(left_A_command=132, forward_A_command=73)))
                fs = cv2.FileStorage(str(cal), cv2.FILE_STORAGE_WRITE)
                fs.write("P1", np.array([[700.0, 0, 640, 0], [0, 700, 360, 0], [0, 0, 1, 0]]))
                fs.release()
                args = [
                    "test",
                    "--out",
                    str(out),
                    "--reference",
                    str(ref),
                    "--left-reference",
                    str(left),
                    "--calibration",
                    str(cal),
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
                    "--track-b",
                ]

                def stopper():
                    deadline = time.monotonic() + 8
                    while not end.wait(0.02):
                        if (
                            any(f[8] and f[1] >= 8 for f in m.commands)
                            or time.monotonic() > deadline
                        ):
                            out.mkdir(exist_ok=True)
                            (out / "STOP").touch()
                            return

                th = threading.Thread(target=stopper)
                th.start()
                actual = ABCombinedPort

                def port(limits, log):
                    return actual(
                        limits, log, port=m.path, sender_factory=Sender, check_holders=False
                    )

                with (
                    patch.object(sys, "argv", args),
                    patch.object(runner, "Observer", Observer),
                    patch("bucket_robot.tracking.policy.FollowPolicy", Policy),
                    patch("bucket_robot.missions.vertical.ABCombinedPort", port),
                    patch.object(runner.signal, "signal"),
                ):
                    runner.main()
                result = json.loads((out / "summary.json").read_text())
                self.assertIsNone(result["error"])
                self.assertIsNone(result["stop_error"])
                if inject_loss:
                    self.assertGreaterEqual(result["recovery"]["loss_count"], 1)
                commands = list(m.commands)
                b_moves = [i for i, f in enumerate(commands) if f[8] and f[5] == 24 and f[1] == 0]
                drives = [i for i, f in enumerate(commands) if f[8] and f[1] > 0]
                self.assertTrue(b_moves)
                self.assertTrue(drives)
                self.assertLess(b_moves[0], drives[0])
                self.assertTrue(any(not f[8] for f in commands[b_moves[0] + 1 : drives[0]]))
                self.assertTrue(
                    any(
                        f[8] and f[4] != 97 and f[5] == 24 and f[1] == 0
                        for f in commands[: drives[0]]
                    ),
                    "A must keep tracking while B descends",
                )
                self.assertTrue(all(commands[i][5] == 24 and commands[i][6] == 0 for i in drives))
                self.assertEqual(result["final_state"]["lift_position_mm"], 24)
                self.assertFalse(result["final_state"]["enabled"])
                end.set()
                th.join()
        finally:
            end.set()
            m.close()


if __name__ == "__main__":
    unittest.main()
