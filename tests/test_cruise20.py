import unittest
from bucket_robot.missions.range_prediction import RangePrediction
from bucket_robot.missions.recovery import RecoveryGate, RecoverySettings
from tests.test_range_prediction import measurement


class Cruise20Tests(unittest.TestCase):
    def test_requested_distance_curve(self):
        for lower, cap in [
            (0.50, 20),
            (0.40, 20),
            (0.35, 16),
            (0.30, 13),
            (0.25, 10),
            (0.24, 8),
            (0.23, 6),
            (0.22, 4),
            (0.21, 2),
            (0.20, 0),
        ]:
            p = RangePrediction(cruise=20)
            p.add(1, measurement(1, 1, lower + 0.005))
            result = p.estimate(1)
            self.assertEqual(result["speed_cap"], cap, (lower, result))
        self.assertTrue(result["verify_stop"])

    def test_cruise_reaches20_and_loss_or_near_cap_ramps_down(self):
        g = RecoveryGate(120, settings=RecoverySettings(cruise=20))
        for i in range(81):
            t = i * 0.05
            g.evaluate(t, dict(timestamp=t, offset_px=[0, 0]), 1, 100, True, speed_cap=20)
        self.assertEqual(g.velocity, 20)
        speeds = []
        for i in range(1, 31):
            t = 4 + i * 0.05
            v, _ = g.evaluate(t, dict(timestamp=t, offset_px=[0, 0]), 1, 100, True, speed_cap=6)
            speeds.append(v)
        self.assertEqual(speeds[-1], 6)
        self.assertEqual(speeds, sorted(speeds, reverse=True))
        for i in range(1, 20):
            g.evaluate(5.5 + i * 0.05, None, 1, 100, True, speed_cap=20)
        self.assertEqual(g.velocity, 0)
