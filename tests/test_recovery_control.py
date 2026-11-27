import unittest
from bucket_robot.missions.recovery import RecoveryGate


def target(t, dx=0, **kw):
    return dict(timestamp=t, offset_px=[dx, 0], **kw)


class RecoveryControlTests(unittest.TestCase):
    def test_whole_mission_budget_is_bounded_and_never_resets_on_reacquisition(self):
        g = RecoveryGate(120)
        self.cruise(g)
        g.drive_seconds = 119.95
        g.evaluate(2.1, target(2.1), 1, 100, True)
        self.assertTrue(g.disable)
        self.assertEqual(g.phase, "budget_paused")
        for i in range(20):
            self.assertEqual(g.evaluate(2.2 + i * 0.05, target(2.2 + i * 0.05), 1, 100, True)[0], 0)
        for v in (121, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                RecoveryGate(v)

    def cruise(self, g):
        for i in range(41):
            g.evaluate(i * 0.05, target(i * 0.05), 1, 100, True)
        self.assertEqual(g.velocity, 8)

    def test_loss_ramps_down_then_waits_without_ending(self):
        g = RecoveryGate(60)
        self.cruise(g)
        speeds = []
        for i in range(1, 25):
            v, reason = g.evaluate(2 + i * 0.05, None, 1, 100, True)
            speeds.append(v)
            self.assertIsNone(reason)
        self.assertGreater(speeds[0], 0)
        self.assertEqual(sorted(speeds, reverse=True), speeds)
        self.assertTrue(all(a - b <= 1 for a, b in zip([8] + speeds, speeds)))
        self.assertEqual(speeds[-1], 0)
        self.assertTrue(g.disable)
        self.assertEqual(g.phase, "waiting_target")

    def test_same_target_recovers_after_fresh_confirmation_and_ramps_up(self):
        g = RecoveryGate(60)
        self.cruise(g)
        for i in range(1, 21):
            g.evaluate(2 + i * 0.05, None, 1, 100, True)
        observed = []
        for i in range(1, 31):
            t = 3 + i * 0.05
            v, _ = g.evaluate(t, target(t), 1, 100, True)
            observed.append(v)
        self.assertEqual(observed[:4], [0] * 4)
        self.assertEqual(observed[-1], 8)
        self.assertTrue(all(b - a <= 1 for a, b in zip(observed, observed[1:])))
        self.assertEqual(g.loss_count, 1)
        self.assertEqual(g.recovery_count, 1)

    def test_other_identity_or_repeated_old_frame_cannot_restart(self):
        for case in ("identity", "repeated"):
            g = RecoveryGate(60)
            self.cruise(g)
            for i in range(1, 21):
                g.evaluate(2 + i * 0.05, None, 1, 100, True)
            for i in range(1, 21):
                t = 3 + i * 0.05
                v, _ = g.evaluate(
                    t,
                    target(t if case == "identity" else 3.05),
                    2 if case == "identity" else 1,
                    100,
                    True,
                )
                self.assertEqual(v, 0)

    def test_stale_camera_and_far_off_center_decelerate(self):
        for case in ("stale", "offset"):
            g = RecoveryGate(60)
            self.cruise(g)
            for i in range(1, 21):
                t = 2 + i * 0.05
                g.evaluate(
                    t,
                    target(1 if case == "stale" else t, 100 if case == "offset" else 0),
                    1,
                    100,
                    True,
                )
            self.assertEqual(g.velocity, 0)

    def test_hard_fault_and_control_stall_do_not_ramp_or_auto_resume(self):
        for case in ("fault", "stall"):
            g = RecoveryGate(60)
            self.cruise(g)
            if case == "fault":
                g.trip("serial_lost")
            else:
                g.evaluate(2.4, target(2.4), 1, 100, True)
            self.assertEqual(g.velocity, 0)
            self.assertTrue(g.disable)
            for i in range(20):
                self.assertEqual(
                    g.evaluate(2.5 + i * 0.05, target(2.5 + i * 0.05), 1, 100, True)[0], 0
                )

    def test_time_budget_pauses_and_left_alignment_completes(self):
        g = RecoveryGate(1)
        for i in range(80):
            g.evaluate(i * 0.05, target(i * 0.05), 1, 100, True)
        self.assertEqual(g.phase, "budget_paused")
        self.assertTrue(g.disable)
        g = RecoveryGate(60)
        for i in range(8):
            t = 1 + i * 0.05
            answer = g.evaluate(t, target(t), 1, 131, True, chassis_stopped=True)
        self.assertEqual(answer, (0, "left_reference_reached"))

    def test_left_clipping_can_only_steer_a_at_zero_speed(self):
        g = RecoveryGate(60)
        for i in range(40):
            t = i * 0.05
            v, _ = g.evaluate(
                t,
                target(
                    t,
                    -400,
                    partial_visible=True,
                    horizontal_clipped=True,
                    bearing_evidence="current_detection_and_visible_appearance",
                ),
                1,
                85,
                True,
            )
            self.assertEqual(v, 0)
            self.assertTrue(g.allow_a)

    def test_handoff_requires_disable_preserved_command_in_left_pose(self):
        g = RecoveryGate(60)
        self.assertIsNone(g.evaluate(1, target(1), 1, 131, True, commanded_a=130)[1])
        for i in range(8):
            t = 1.05 + i * 0.05
            answer = g.evaluate(t, target(t), 1, 131, True, commanded_a=131, chassis_stopped=True)
        self.assertEqual(answer, (0, "left_reference_reached"))

    def test_moving_frame_or_one_degree_A_lag_cannot_handoff(self):
        for case in ("moving", "A_lag", "repeated"):
            g = RecoveryGate(60)
            for i in range(16):
                t = 1 + i * 0.04
                _, reason = g.evaluate(
                    t,
                    target(1 if case == "repeated" else t),
                    1,
                    131,
                    True,
                    commanded_a=132 if case == "A_lag" else 131,
                    chassis_stopped=case != "moving",
                )
                self.assertIsNone(reason, case)

    def test_left_boundary_handoff_ignores_pixel_offset_after_disable_ack(self):
        g = RecoveryGate(60)
        for i in range(8):
            now = 1 + i * 0.05
            v, reason = g.evaluate(
                now,
                target(now, -59),
                1,
                132,
                True,
                commanded_a=132,
                chassis_stopped=False,
            )
            self.assertEqual(v, 0)
            self.assertTrue(g.disable)
            self.assertIsNone(reason)
        for i in range(8):
            now = 1.5 + i * 0.05
            answer = g.evaluate(
                now,
                target(now, -59),
                1,
                132,
                True,
                commanded_a=132,
                chassis_stopped=True,
            )
        self.assertEqual(answer, (0, "left_reference_reached"))

    def test_left_boundary_never_resumes_forward_on_loss_identity_or_stale_frame(self):
        for case in ("lost", "identity", "horizontal_clipped", "stale"):
            g = RecoveryGate(60)
            g.evaluate(1, target(1, -59), 1, 132, True, commanded_a=132)
            for i in range(8):
                now = 1.05 + i * 0.05
                seen = target(now, 0)
                if case == "lost":
                    seen = None
                elif case == "horizontal_clipped":
                    seen.update(partial_visible=True, horizontal_clipped=True)
                elif case == "stale":
                    seen["timestamp"] = 0
                v, reason = g.evaluate(
                    now,
                    seen,
                    2 if case == "identity" else 1,
                    132,
                    True,
                    commanded_a=132,
                    chassis_stopped=True,
                )
                self.assertEqual(v, 0, case)
                self.assertTrue(g.disable, case)
                self.assertIsNone(reason, case)
            for i in range(8):
                now = 1.5 + i * 0.05
                answer = g.evaluate(
                    now,
                    target(now, -59),
                    1,
                    132,
                    True,
                    commanded_a=132,
                    chassis_stopped=True,
                )
            self.assertEqual(answer, (0, "left_reference_reached"), case)

    def test_forward_motion_gate_does_not_block_stopped_left_handoff(self):
        # B/range may veto forward travel; the next lateral stage obtains its
        # own range permission. Handoff requires the actual disabled ACK.
        g = RecoveryGate(60)
        for i in range(8):
            now = 1 + i * 0.05
            answer = g.evaluate(
                now,
                target(now, -59),
                1,
                132,
                True,
                commanded_a=132,
                chassis_stopped=True,
                motion_permitted=False,
            )
        self.assertEqual(answer, (0, "left_reference_reached"))


if __name__ == "__main__":
    unittest.main()
