import math
import unittest

from bucket_robot.missions.range_prediction import RangePrediction
from bucket_robot.missions.recovery import RecoveryGate, RecoverySettings
from bucket_robot.tracking.aim import aimed_target
from bucket_robot.tracking.policy import AlignmentGate
from tests.test_range_prediction import measurement


class Cruise80AimTests(unittest.TestCase):
    def test_top_region_is_ten_percent_area_and_preserves_detection(self):
        target = dict(timestamp=1, box=[440, 300, 840, 700], offset_px=[0, 140])
        answer = aimed_target(target, dict(placement="top_center", area_fraction=0.1))
        l, t, r, b = answer["aim_region_px"]
        self.assertAlmostEqual((r - l) * (b - t), 0.1 * 400 * 400)
        self.assertAlmostEqual((r + l) / 2, 640)
        self.assertEqual(t, 300)
        self.assertAlmostEqual(b, 300 + math.sqrt(0.1) * 400)
        self.assertEqual(answer["box"], target["box"])
        self.assertEqual(target["offset_px"], [0, 140])
        self.assertTrue(answer["vertical_aim_aligned"])
        self.assertTrue(AlignmentGate().update(answer))

    def test_static80_distance_curve_and_final_stop(self):
        for lower, expected in [
            (0.5, 80),
            (0.4, 80),
            (0.35, 56),
            (0.3, 33),
            (0.25, 10),
            (0.22, 4),
            (0.2, 0),
        ]:
            p = RangePrediction(cruise=80)
            p.add(1, dict(measurement(1, 1, lower + 0.005), elapsed_s=0.05))
            self.assertEqual(p.estimate(1)["speed_cap"], expected)

    def test_faster_measured_closing_and_latency_slow_before_raw40cm(self):
        p = RangePrediction(cruise=80)
        p.add(1, measurement(1, 1, 1.2))
        p.add(1, measurement(2, 4, 0.81))
        answer = p.estimate(4.5)
        self.assertTrue(answer["permit"])
        self.assertAlmostEqual(answer["closing_mps"], 0.13)
        self.assertGreater(answer["predicted_distance_m"], 0.4)
        self.assertLess(answer["speed_cap"], 80)
        self.assertLess(answer["control_lower_m"], answer["predicted_lower_m"])

    def test_a_settled_can_depart_without_vertical_center_or_extra_dwell(self):
        gate = RecoveryGate(
            120,
            settings=RecoverySettings(
                cruise=80,
                acceleration_per_s=40,
                deceleration_per_s=80,
                initial_confirm_s=0,
                initial_confirm_frames=1,
            ),
        )
        for i in range(10):
            stamp = i * 0.05
            gate.evaluate(
                stamp, dict(timestamp=stamp, offset_px=[0, 200]), 1, 100, False, speed_cap=80
            )
        v, _ = gate.evaluate(
            0.5, dict(timestamp=0.5, offset_px=[0, 200]), 1, 100, True, speed_cap=80
        )
        self.assertGreater(v, 0)
        for i in range(11, 61):
            stamp = i * 0.05
            gate.evaluate(
                stamp, dict(timestamp=stamp, offset_px=[0, 200]), 1, 100, True, speed_cap=80
            )
        self.assertEqual(gate.velocity, 80)
        values = []
        for i in range(1, 22):
            v, _ = gate.evaluate(3 + i * 0.05, None, 1, 100, True, speed_cap=80)
            values.append(v)
        self.assertTrue(any(0 < v < 80 for v in values))
        self.assertEqual(values[-1], 0)
        self.assertIsNone(gate.latched)
