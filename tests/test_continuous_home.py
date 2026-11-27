from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bucket_robot.motion.arm import ArmPort, Limits
from bucket_robot.resources import default_reference


class ContinuousHomeTests(unittest.TestCase):
    def make_port(self):
        p = ArmPort.__new__(ArmPort)
        p.limits = Limits(default_reference())
        p.phase, p.ready_at, p.rx_at, p.expected = "idle", 0, 10, 10
        p.state = dict(
            servo_software_deg=131,
            lift_position_mm=96.0,
            slide_position_mm=0.0,
            enabled=False,
            state=0,
            result=0,
            command_count_mod256=10,
            system_fault=0,
            motor_fault=0,
            lift_zero_confirmed=True,
            slide_zero_confirmed=True,
            angle_valid=True,
            heading_reference_established=True,
            yaw_deg=2,
        )
        p.attempted = False
        p.target = None
        p.log = lambda *args: None
        p.commands = []
        p.gateway = SimpleNamespace(
            write_control_frame=lambda f, repeats: p.commands.append(f) or True
        )
        return p

    def feed(self, p, now, b, enabled=True, state=2, counter=11):
        frame = dict(
            p.state, lift_position_mm=b, enabled=enabled, state=state, command_count_mod256=counter
        )

        def poll():
            p.state = frame
            p.rx_at = now
            return [frame]

        p.poll = poll
        with patch("bucket_robot.motion.arm.time.monotonic", return_value=now):
            p.tick()

    def test_full_B_return_stays_enabled_through_intermediate_positions_then_stops_once(self):
        p = self.make_port()
        with patch("bucket_robot.motion.arm.time.monotonic", return_value=10):
            self.assertTrue(p.return_axis_home(1))
        self.assertEqual(list(p.commands[0][1:3]), [0, 0])
        self.assertEqual(list(p.commands[0][4:7]), [131, 0, 0])
        self.assertGreater(p.deadline, 18)
        for now, b in [(12, 76), (14, 56), (16, 36), (18, 16)]:
            self.feed(p, now, b)
            self.assertEqual(len(p.commands), 1)
            self.assertEqual(p.phase, "moving")
        self.feed(p, 19, 0, state=1)
        self.assertEqual(len(p.commands), 1)
        self.feed(p, 19.2, 0, state=1)
        self.assertEqual(p.phase, "stopping")
        self.assertEqual([c[8] for c in p.commands], [1, 0])
        self.feed(p, 19.3, 0, enabled=False, state=0, counter=12)
        self.assertEqual(p.phase, "idle")
        self.assertFalse(p.state["enabled"])

    def test_stationary_freshness_and_axis_order_are_required_before_write(self):
        for case in [
            "enabled",
            "busy",
            "stale",
            "not_ready",
            "fault",
            "preserved_fraction",
            "A_order",
        ]:
            p = self.make_port()
            axis = 1
            if case == "enabled":
                p.state["enabled"] = True
            elif case == "busy":
                p.phase = "moving"
            elif case == "stale":
                p.rx_at = 9
            elif case == "not_ready":
                p.ready_at = 11
            elif case == "fault":
                p.state["system_fault"] = 3
            elif case == "preserved_fraction":
                p.state["slide_position_mm"] = 0.1
            elif case == "A_order":
                axis = 0
            with patch("bucket_robot.motion.arm.time.monotonic", return_value=10):
                with self.assertRaises(RuntimeError, msg=case):
                    p.return_axis_home(axis)
            self.assertEqual(p.commands, [], case)

    def test_full_home_does_not_relax_tracking_steps_and_ignores_already_home(self):
        p = self.make_port()
        with patch("bucket_robot.motion.arm.time.monotonic", return_value=10):
            with self.assertRaisesRegex(RuntimeError, "invalid_tracking_step"):
                p.move([131, 0, 0])
            p.state["lift_position_mm"] = 0
            self.assertFalse(p.return_axis_home(1))
        self.assertEqual(p.commands, [])

    def test_preserved_A_C_and_selected_endpoint_only(self):
        for axis, current, target in [
            (1, [131, 96, 150], [131, 0, 150]),
            (2, [131, 96, 150], [131, 96, 0]),
            (0, [131, 0, 0], [73, 0, 0]),
        ]:
            p = self.make_port()
            p.state.update(
                servo_software_deg=current[0],
                lift_position_mm=current[1],
                slide_position_mm=current[2],
            )
            with patch("bucket_robot.motion.arm.time.monotonic", return_value=10):
                p.return_axis_home(axis)
            self.assertEqual(list(p.commands[0][4:7]), target)
            self.assertEqual(p.commands[0][7:9], bytes([0, 1]))

    def test_stop_during_return_preserves_final_target_and_accepts_intermediate_disabled_pose(self):
        p = self.make_port()
        with patch("bucket_robot.motion.arm.time.monotonic", return_value=10):
            p.return_axis_home(1)
        self.feed(p, 12, 76)
        with patch("bucket_robot.motion.arm.time.monotonic", return_value=12):
            p.stop()
        self.assertEqual(list(p.commands[-1][4:7]), [131, 0, 0])
        self.feed(p, 12.1, 75.5, enabled=False, state=0, counter=12)
        self.assertEqual(p.phase, "idle")
        self.assertEqual(len(p.commands), 2)

    def test_missing_ack_or_timeout_cannot_cause_another_home_command(self):
        p = self.make_port()
        with patch("bucket_robot.motion.arm.time.monotonic", return_value=10):
            p.return_axis_home(1)
        with patch("bucket_robot.motion.arm.time.monotonic", return_value=11):
            with self.assertRaises(RuntimeError):
                p.return_axis_home(1)
        with self.assertRaisesRegex(RuntimeError, "movement_timeout"):
            self.feed(p, 24, 95)
        self.assertEqual(len(p.commands), 1)
