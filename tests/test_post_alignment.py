import unittest

from bucket_robot.missions.post_alignment import PostAlignmentGate


class PostAlignmentTests(unittest.TestCase):
    def test_left_pose_latches_stop_despite_offcenter_and_waits_disable(self):
        g = PostAlignmentGate(20)
        for now in [1, 1.2, 1.4]:
            v, reason = g.evaluate(
                now,
                dict(timestamp=now, offset_px=[-66, 116]),
                1,
                132,
                True,
                commanded_a=132,
                chassis_stopped=False,
            )
            self.assertEqual(v, 0)
            self.assertIsNone(reason)
            self.assertTrue(g.disable)
        for now in [1.5, 1.7, 1.9]:
            v, reason = g.evaluate(
                now,
                dict(timestamp=now, offset_px=[-66, 116]),
                1,
                132,
                True,
                commanded_a=132,
                chassis_stopped=True,
            )
        self.assertEqual(reason, "left_reference_reached")

    def test_loss_or_other_identity_cannot_unlock_lateral_but_never_restarts_forward(self):
        g = PostAlignmentGate(20)
        g.evaluate(
            1,
            dict(timestamp=1, offset_px=[-60, 0]),
            1,
            131,
            True,
            commanded_a=131,
            chassis_stopped=False,
        )
        for now, target, identity in [
            (1.1, None, 1),
            (1.2, dict(timestamp=1.2, offset_px=[0, 0]), 2),
        ]:
            v, reason = g.evaluate(
                now, target, identity, 131, True, commanded_a=131, chassis_stopped=True
            )
            self.assertEqual(v, 0)
            self.assertIsNone(reason)
            self.assertTrue(g.disable)
