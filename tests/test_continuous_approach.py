import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import bucket_robot.missions.continuous_approach as task
from tests.test_track_forward_probe import MCU, Sender


class ContinuousApproachTests(unittest.TestCase):
    def test_real_uart_continuous_drive_slows_then_stops_without_periodic_pulses(self):
        self.run_task()

    def test_stopped_forward_port_hands_over_to_lateral_without_reopening_serial(self):
        self.run_task(reuse=True)

    def run_task(self, reuse=False):
        class SideMCU(MCU):
            def packet(self):
                f = bytearray(super().packet())
                if self.enabled and self.commands and self.commands[-1][2]:
                    f[4] = 3
                f[18] = 0
                for v in f[1:18]:
                    f[18] ^= v
                return bytes(f)

        m = SideMCU()
        m.a = 132
        clock = [None]

        class Observer:
            def __init__(self, *a, **kw):
                self.end = threading.Event()
                self.thread = SimpleNamespace(start=lambda: None, join=lambda timeout=None: None)

            def get(self):
                now = time.monotonic()
                return dict(
                    identity=1,
                    frame_id=int(now * 30),
                    target=dict(timestamp=now, offset_px=[0, 0], box=[400, 200, 800, 600]),
                    left=None,
                )

        class Ranger:
            def __init__(self, *a):
                clock[0] = time.monotonic()
                self.thread = SimpleNamespace(start=lambda: None)

            def evaluate(self, now, *args):
                elapsed = now - clock[0]
                return dict(
                    permit=elapsed < 5.5,
                    near=elapsed >= 5.5,
                    speed_cap=8 if elapsed < 4 else 4,
                    reason="continuous_range",
                    measurement=dict(frame_id=int(now * 30), distance_m=0.2),
                )

            def close(self):
                pass

        actual = task.LateralPort

        def port(limits, log):
            return actual(limits, log, port=m.path, sender_factory=Sender, check_holders=False)

        try:
            with tempfile.TemporaryDirectory() as folder:
                ref = Path(folder) / "ref.json"
                ref.write_text(
                    json.dumps(dict(control_policy=dict(original_sender_module_on_rk3588="unused")))
                )
                existing = None
                if reuse:
                    from bucket_robot.motion.arm import Limits
                    from bucket_robot.motion.combined import ABCombinedPort

                    config = json.loads(ref.read_text())
                    config["operating_envelope"] = dict(
                        A_command_degrees=[73, 132], B_command_mm=[0, 96], C_trial_hold_mm=0
                    )
                    ref.write_text(json.dumps(config))
                    existing = ABCombinedPort(
                        Limits(ref),
                        lambda *a: None,
                        port=m.path,
                        sender_factory=Sender,
                        check_holders=False,
                    )
                    existing.rearm()
                with (
                    patch.object(task, "StereoObserver", Observer),
                    patch.object(task, "ForwardRange", Ranger),
                    patch.object(task, "LateralPort", port),
                    patch(
                        "bucket_robot.perception.assisted_range.stereo.Geometry", return_value=None
                    ),
                ):
                    result = task.run_continuous_approach(
                        out=Path(folder) / "run",
                        reference=ref,
                        calibration="unused",
                        model="unused",
                        seed_image="unused",
                        seed_box=[400, 200, 800, 600],
                        execute=True,
                        existing_port=existing,
                    )
                self.assertIsNone(result["error"])
                self.assertIsNone(result["stop_error"])
                self.assertTrue(result["disabled_confirmed"])
                speeds = [f[2] if f[8] else 0 for f in m.commands]
                peak = speeds.index(max(speeds))
                self.assertEqual(max(speeds), 8)
                self.assertTrue(all(v > 0 for v in speeds[peak:-1]), speeds)
                self.assertIn(4, speeds[peak:])
                self.assertEqual(speeds[-1], 0)
                self.assertTrue(all(f[1] == 0 and f[4] == 132 for f in m.commands))
                self.assertEqual(result["pulses"], 0)
        finally:
            m.close()
