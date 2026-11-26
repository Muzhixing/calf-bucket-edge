import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from bucket_robot.missions.lifecycle import stop_and_drain
from bucket_robot.motion.arm import Limits
from bucket_robot.motion.combined import CombinedPort
from tests.test_track_forward_probe import MCU, Sender


class WorkerDrainTests(unittest.TestCase):
    def test_disabled_uart_remains_fresh_while_slow_model_exits(self):
        m = MCU()
        records = []
        p = None
        try:
            with tempfile.TemporaryDirectory() as folder:
                ref = Path(folder) / "ref.json"
                ref.write_text(
                    json.dumps(
                        dict(
                            operating_envelope=dict(
                                A_command_degrees=[73, 132], B_command_mm=[0, 96], C_trial_hold_mm=0
                            ),
                            control_policy=dict(original_sender_module_on_rk3588="unused"),
                        )
                    )
                )
                p = CombinedPort(
                    Limits(ref),
                    lambda kind, data: records.append((time.monotonic(), kind, data)),
                    port=m.path,
                    sender_factory=Sender,
                    check_holders=False,
                )
                p.rearm()
                p.request(97, 8)
                until = time.monotonic() + 0.4
                while time.monotonic() < until:
                    p.tick()
                    time.sleep(0.01)
                worker = SimpleNamespace(
                    end=threading.Event(), thread=threading.Thread(target=lambda: time.sleep(1.3))
                )
                worker.thread.start()
                start = time.monotonic()
                self.assertIsNone(stop_and_drain(p, [worker]))
                rx = [t for t, k, d in records if k == "serial_rx" and t >= start]
                self.assertGreater(rx[-1] - start, 1.0)
                self.assertTrue(all(b - a < 0.4 for a, b in zip(rx, rx[1:])))
                self.assertFalse(p.state["enabled"])
                self.assertIsNotNone(p.fd)
                self.assertIsNone(p.shutdown(close_port=False))
                self.assertIsNotNone(p.fd)
                self.assertIsNone(p.shutdown())
                self.assertIsNone(p.fd)
        finally:
            if p is not None and p.fd is not None:
                p.shutdown()
            m.close()
