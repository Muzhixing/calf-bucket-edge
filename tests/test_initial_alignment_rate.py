import unittest

from bucket_robot.motion.arm import Limits
from bucket_robot.resources import default_reference
from bucket_robot.tracking.policy import FollowPolicy
from bucket_robot.missions.profile import MissionProfile


class InitialAlignmentRateTests(unittest.TestCase):
    def test_initial_camera_rate_is_independent_of_high_chassis_cruise(self):
        profile = MissionProfile("production-center", 80, True, False, True, True, True)
        self.assertEqual(profile.a_rate_limit(departed=False), 10)
        self.assertEqual(profile.a_rate_limit(departed=True), 30)
        self.assertEqual(profile.a_rate_limit(departed=False, post_aligning=True), 30)

    def trace(self, rate):
        p = FollowPolicy(Limits(default_reference()), 700, 700, 640, 360)
        p.set_a_rate_limit(rate)
        current = 73
        commands = [current]
        for i in range(20):
            now = 1 + i * 0.05
            current = p.next_a(
                dict(timestamp=now, offset_px=[-500, 0], quality=1), current, current, now
            )
            self.assertLessEqual(abs(p.rate), rate)
            commands.append(current)
        return commands, p

    def test_initial_alignment_has_bounded_rate_without_changing_default_following(self):
        initial, p = self.trace(10)
        normal, _ = self.trace(30)
        self.assertLessEqual(initial[-1] - initial[0], 11)
        self.assertGreater(normal[-1] - normal[0], initial[-1] - initial[0])
        p.set_a_rate_limit(30)
        self.assertEqual(p.a_rate_limit, 30)
        p.reset()
        self.assertEqual(p.a_rate_limit, 30)
        for bad in (0, 31, float("nan"), True):
            with self.assertRaises(ValueError):
                p.set_a_rate_limit(bad)
