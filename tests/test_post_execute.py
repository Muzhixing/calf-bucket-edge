import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bucket_robot.missions.post_execute import execute_post, ExtensionPort, lateral_timing
from bucket_robot.missions.post_geometry import alignment_extension_plan
from bucket_robot.motion.chassis import ChassisPort
from bucket_robot.resources import default_reference
from tests.test_track_forward_probe import MCU, Sender


class PostMCU(MCU):
    def __init__(self, b=0, c=0):
        self.initial_b, self.initial_c = b, c
        super().__init__()

    def packet(self):
        f = bytearray(super().packet())
        active = [c for c in self.commands if c[8]]
        c = active[-1][6] if active else self.initial_c
        b = active[-1][5] if active else self.initial_b
        f[6:8] = int(b * 10).to_bytes(2, "little")
        f[8:10] = int(c * 10).to_bytes(2, "little")
        if self.enabled and self.commands[-1][2]:
            f[4] = 3
        f[18] = 0
        for value in f[1:18]:
            f[18] ^= value
        return f


class PostExecuteTests(unittest.TestCase):
    def test_calibration_is_timed_estimate_and_bounded(self):
        t = lateral_timing(97.5)
        self.assertEqual(t["seconds"], 3)
        self.assertEqual(t["estimated_travel_mm_interval"], [95, 100])
        self.assertFalse(t["metric_feedback"])
        for target in [0, -1, 10, 131, float("nan"), True]:
            with self.assertRaises(ValueError):
                lateral_timing(target)

    def test_actual_uart_left_disable_then_C150_disable(self):
        m = PostMCU()
        m.a = 131

        def chassis(limits, record):
            return ChassisPort(
                limits, record, port=m.path, sender_factory=Sender, check_holders=False
            )

        def extension(limits, record):
            return ExtensionPort(
                limits, record, port=m.path, sender_factory=Sender, check_holders=False
            )

        try:
            with tempfile.TemporaryDirectory() as root:
                result = execute_post(
                    alignment_extension_plan(0.2, 131),
                    default_reference(),
                    Path(root),
                    chassis_factory=chassis,
                    extension_factory=extension,
                )
                self.assertIsNone(result["error"], result)
                self.assertIsNone(result["stop_error"])
                self.assertEqual(result["final_state"]["slide_position_mm"], 150)
                self.assertFalse(result["final_state"]["enabled"])
                commands = list(m.commands)
                extend = next(i for i, c in enumerate(commands) if c[8] and c[6] == 150)
                self.assertTrue(any(c[8] and c[2] == 20 for c in commands[:extend]))
                self.assertEqual(commands[extend - 1][8], 0)
                self.assertTrue(all(c[1] == 0 for c in commands))
                self.assertTrue(all(c[2] == 0 for c in commands[extend:]))
                self.assertEqual(commands[-1][8], 0)
                self.assertEqual(
                    json.loads((Path(root) / "task_status.json").read_text())["phase"], "complete"
                )
        finally:
            m.close()

    def test_stop_file_never_constructs_a_motion_port(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            (folder / "STOP").touch()
            with patch("bucket_robot.missions.post_execute.time.sleep"):

                def forbidden(*args):
                    self.fail("Motion after operator STOP")

                result = execute_post(
                    alignment_extension_plan(0.2, 131),
                    default_reference(),
                    folder,
                    chassis_factory=forbidden,
                    extension_factory=forbidden,
                )
            self.assertTrue(result["operator_stopped"])
            self.assertFalse(result["C_extension_sent"])

    def test_existing_C150_only_raises_B_without_chassis_or_repeat_extension(self):
        m = PostMCU(b=96, c=150)
        m.a = 132

        def chassis(limits, record):
            return ChassisPort(
                limits, record, port=m.path, sender_factory=Sender, check_holders=False
            )

        def extension(limits, record):
            return ExtensionPort(
                limits, record, port=m.path, sender_factory=Sender, check_holders=False
            )

        try:
            with tempfile.TemporaryDirectory() as root:
                result = execute_post(
                    alignment_extension_plan(0.2, 125),
                    default_reference(),
                    Path(root),
                    chassis_factory=chassis,
                    extension_factory=extension,
                )
                self.assertIsNone(result["error"], result)
                self.assertTrue(result["B_top_confirmed"])
                self.assertFalse(result["chassis_command_sent"])
                self.assertFalse(result["C_extension_sent"])
                self.assertEqual(result["final_state"]["slide_position_mm"], 150)
                self.assertTrue(all(c[1:3] == bytes([0, 0]) and c[6] == 150 for c in m.commands))
        finally:
            m.close()
