import unittest
from unittest.mock import patch

from bucket_robot.motion.chassis import ChassisPort, HeldArmLimits, pulse_frame


def state():
    return dict(
        servo_software_deg=90,
        lift_position_mm=0.0,
        slide_position_mm=0.0,
        yaw_deg=359.5,
        system_fault=0,
        motor_fault=0,
        angle_valid=True,
        heading_reference_established=True,
        lift_zero_confirmed=True,
        slide_zero_confirmed=True,
        enabled=True,
        result=0,
        state=3,
        command_count_mod256=11,
        uart_error_count_mod256=0,
    )


class ChassisTests(unittest.TestCase):
    @patch("bucket_robot.motion.chassis.time.monotonic", return_value=100.0)
    def test_ten_second_pulse_keeps_local_deadline_and_stops(self, clock):
        from unittest.mock import Mock

        p = self.port()
        p.phase = "idle"
        p.state = state()
        p.state["enabled"] = False
        p.limits = Mock()
        p.gateway = Mock()
        p.log = Mock()
        p.start_pulse("left", 8, 10)
        self.assertEqual(p.deadline, 110.0)
        p.poll = lambda: [state()]
        clock.return_value = 109.99
        p.rx_at = 109.99
        p.tick()
        self.assertEqual(p.phase, "driving")
        clock.return_value = 110.0
        p.rx_at = 110.0
        p.tick()
        self.assertEqual(p.phase, "stopping")
        frame = p.gateway.write_control_frame.call_args.args[0]
        self.assertEqual(frame[1:3], bytes([0, 0]))
        self.assertEqual(frame[8], 0)
        for invalid in (10.01, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                p.start_pulse("forward", 8, invalid)

    @patch("bucket_robot.motion.chassis.time.monotonic", return_value=100.0)
    def test_continuous_mode_keeps_deadline_open_but_stop_frame_disables(self, clock):
        from unittest.mock import Mock

        p = self.port()
        p.phase = "idle"
        p.state = state()
        p.state["enabled"] = False
        p.limits = Mock()
        p.gateway = Mock()
        p.log = Mock()
        p.start_pulse("backward", 8, 2, continuous=True)
        self.assertEqual(p.deadline, float("inf"))
        p.stop()
        self.assertEqual(p.phase, "stopping")
        frame = p.gateway.write_control_frame.call_args.args[0]
        self.assertEqual(frame[1:3], bytes([0, 0]))
        self.assertEqual(frame[8], 0)

    def test_existing_pose_locked_including_retracted_C(self):
        limits = object.__new__(HeldArmLimits)
        limits.held = None
        limits.check([90, 0, 0])
        limits.check([90, 0, 0])
        with self.assertRaisesRegex(RuntimeError, "unchanged"):
            limits.check([90, 0, 10])

    def test_directions_preserve_arm_and_wrap_heading(self):
        for direction, xy in [
            ("forward", (8, 0)),
            ("backward", (248, 0)),
            ("left", (0, 8)),
            ("right", (0, 248)),
        ]:
            self.assertEqual(
                pulse_frame(state(), direction, 8), bytes([179, *xy, 0, 90, 0, 0, 0, 1, 180])
            )

    def test_fractional_axis_refused_instead_of_moving_arm(self):
        s = state()
        s["lift_position_mm"] = 0.1
        with self.assertRaisesRegex(RuntimeError, "fractional"):
            pulse_frame(s, "forward", 8)

    def test_limits(self):
        self.assertEqual(pulse_frame(state(), "forward", 20)[1], 20)
        self.assertEqual(pulse_frame(state(), "backward", 20)[1], 236)
        for speed in (0, 21, -1, 1.5, True):
            with self.assertRaises(ValueError):
                pulse_frame(state(), "forward", speed)

    def port(self, frame=None):
        p = object.__new__(ChassisPort)
        p.phase = "driving"
        p.rx_at = 100.0
        p.started = 100.0
        p.deadline = 101.2
        p.origin = [90, 0.0, 0.0]
        p.uart_errors = 0
        p.expected = 10
        p.move_counter = 11
        p.acknowledged = p.drive_observed = False
        p.poll = lambda: [state() if frame is None else frame]
        return p

    @patch("bucket_robot.motion.chassis.time.monotonic", return_value=100.2)
    def test_drive_confirmation_and_arm_guard(self, clock):
        p = self.port()
        p.tick()
        self.assertTrue(p.acknowledged and p.drive_observed)
        s = state()
        s["servo_software_deg"] = 91
        with self.assertRaisesRegex(RuntimeError, "arm_motion"):
            self.port(s).tick()

    @patch("bucket_robot.motion.chassis.time.monotonic", return_value=100.01)
    def test_immediate_enable_report_precedes_control_tick(self, clock):
        s = state()
        s["state"] = 0
        p = self.port(s)
        p.tick()
        self.assertTrue(p.acknowledged)
        self.assertFalse(p.drive_observed)
        clock.return_value = 100.2
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            p.tick()

    @patch("bucket_robot.motion.chassis.time.monotonic", return_value=101.2)
    def test_deadline_sends_stop(self, clock):
        p = self.port()
        p.rx_at = 101.2
        with patch.object(p, "stop") as stop:
            p.tick()
            stop.assert_called_once()

    @patch("bucket_robot.motion.chassis.time.monotonic", return_value=100.7)
    def test_timeout_or_motor_fault_aborts(self, clock):
        with self.assertRaisesRegex(RuntimeError, "telemetry_timeout"):
            self.port().tick()
        s = state()
        s["motor_fault"] = 2
        p = self.port(s)
        p.rx_at = 100.7
        with self.assertRaisesRegex(RuntimeError, "firmware_fault"):
            p.tick()


if __name__ == "__main__":
    unittest.main()
