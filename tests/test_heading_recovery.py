import unittest
from bucket_robot.motion.heading import HeadingRecovery
from bucket_robot.missions.recovery import RecoveryGate


class Port:
    def __init__(self):
        self.state = dict(
            yaw_deg=356.9,
            servo_software_deg=100,
            lift_position_mm=96,
            slide_position_mm=0,
            enabled=True,
        )
        self.phase = "following"
        self.rx_at = 1.0
        self.pending = []
        self.commands = []
        self.events = []

    def log(self, kind, data):
        self.events.append((kind, data.copy()))

    def stop(self):
        self.commands.append(("disable",))
        self.phase = "stopping"

    def disabled_ack(self, now):
        self.phase = "idle"
        self.state["enabled"] = False
        self.rx_at = now

    def can_send(self):
        return self.phase == "idle"

    def request(self, a, v):
        self.commands.append(("request", a, v, self.state["lift_position_mm"]))
        self.phase = "following"
        self.state["enabled"] = True
        return True


class HeadingRecoveryTests(unittest.TestCase):
    def test_pauses_translation_corrects_original_heading_and_waits_final_disable(self):
        p = Port()
        h = HeadingRecovery(0, 0)
        self.assertTrue(h.advance(p, 1))
        self.assertEqual(p.commands, [("disable",)])
        # No correction can be armed before the actual disabled acknowledgment.
        self.assertTrue(h.advance(p, 1.1))
        self.assertFalse(any(c[0] == "request" for c in p.commands))
        p.disabled_ack(1.2)
        self.assertTrue(h.advance(p, 1.2))
        self.assertEqual(p.commands[-1], ("request", 100, 0, 96))
        for now, yaw in [(1.4, 358), (1.6, 359), (1.9, 359.2), (2.2, 359.4)]:
            p.rx_at = now
            p.state["yaw_deg"] = yaw
            self.assertTrue(h.advance(p, now))
        self.assertEqual(h.phase, "confirming_disabled")
        self.assertEqual(p.commands[-1], ("disable",))
        p.disabled_ack(2.3)
        self.assertFalse(h.advance(p, 2.3))
        self.assertEqual((h.completed, h.attempts), (1, 1))
        self.assertEqual(h.reference, 0)
        self.assertEqual(h.target, 0)

    def test_old_samples_never_confirm_recovery(self):
        p = Port()
        h = HeadingRecovery(0, 0)
        h.advance(p, 1)
        p.disabled_ack(1.1)
        h.advance(p, 1.1)
        p.rx_at = 1.2
        p.state["yaw_deg"] = 0.5
        for now in [1.2, 1.25, 1.3, 1.4]:
            self.assertTrue(h.advance(p, now))
        self.assertEqual(h.stable_frames, 1)
        self.assertEqual(h.phase, "correcting")
        with self.assertRaisesRegex(RuntimeError, "stale_telemetry"):
            h.advance(p, 1.6)

    def test_large_drift_invalid_heading_or_jump_remain_hard_faults(self):
        for yaw in [8, 352, float("nan"), None]:
            with self.assertRaises(RuntimeError):
                HeadingRecovery(0, 0).check(yaw, 1)
        h = HeadingRecovery(0, 0)
        h.check(3, 1)
        with self.assertRaisesRegex(RuntimeError, "discontinuity"):
            h.check(357, 1.25)
        # Wrap-around itself is not a discontinuity.
        h = HeadingRecovery(359.5, 0)
        h.check(359.5, 1)
        h.check(0.5, 1.25)
        self.assertEqual(h.error, 1)

    def test_correction_is_bounded_and_does_not_reissue_command(self):
        p = Port()
        h = HeadingRecovery(0, 0)
        h.advance(p, 1)
        p.disabled_ack(1.1)
        h.advance(p, 1.1)
        for now in [2, 3, 4, 5, 6, 7]:
            p.rx_at = now
            self.assertTrue(h.advance(p, now))
        p.rx_at = 7.2
        with self.assertRaisesRegex(RuntimeError, "correction_timeout"):
            h.advance(p, 7.2)
        self.assertEqual(sum(c[0] == "request" for c in p.commands), 1)
        p = Port()
        h = HeadingRecovery(0, 0)
        h.attempts = 3
        with self.assertRaisesRegex(RuntimeError, "attempts_exhausted"):
            h.advance(p, 1)

    def test_fractional_B_or_extended_C_cannot_be_moved_by_heading_recovery(self):
        for key, value in [("lift_position_mm", 20.1), ("slide_position_mm", 150)]:
            p = Port()
            h = HeadingRecovery(0, 0)
            h.advance(p, 1)
            p.disabled_ack(1.1)
            p.state[key] = value
            with self.assertRaisesRegex(RuntimeError, "settled_retracted_pose"):
                h.advance(p, 1.1)
            self.assertFalse(any(c[0] == "request" for c in p.commands))

    def test_warning_slows_travel_and_pause_preserves_budget_identity_and_boundary(self):
        h = HeadingRecovery(0, 0)
        h.check(2.5, 1)
        self.assertEqual(h.speed_cap(80), 20)
        self.assertEqual(h.speed_cap(8), 8)
        g = RecoveryGate(60)
        g.last_now, g.velocity, g.speed = 1, 8, 8
        g.drive_seconds, g.identity, g.started, g.left_latched = 10, 1, 0.5, True
        g.hold_for_heading(1.1)
        g.hold_for_heading(5)
        self.assertAlmostEqual(g.drive_seconds, 10.1)
        self.assertEqual((g.identity, g.started, g.left_latched), (1, 0.5, True))
        self.assertEqual((g.velocity, g.speed, g.last_now), (0, 0, 5))
