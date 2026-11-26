import unittest
from bucket_robot.missions.range_prediction import RangePrediction


def measurement(frame, stamp, distance):
    return dict(
        frame_id=frame,
        capture_time=stamp,
        valid=True,
        distance_m=distance,
        lower_m=distance - 0.005,
        elapsed_s=3,
        distance_kind="stereo_optical_midpoint_to_bucket_surface",
    )


class RangePredictionTests(unittest.TestCase):
    def test_delayed_result_is_extrapolated_from_capture_time(self):
        p = RangePrediction()
        for i in range(4):
            p.add(1, measurement(i, i * 3, 0.6 - i * 0.03))
        result = p.estimate(12)
        self.assertAlmostEqual(result["closing_mps"], 0.01)
        self.assertAlmostEqual(result["predicted_distance_m"], 0.48)
        self.assertEqual(result["speed_cap"], 8)
        self.assertEqual(result["measurement_age_s"], 3)

    def test_continuous_far_drive_does_not_require_per_result_stop(self):
        p = RangePrediction()
        for i in range(3):
            p.add(1, measurement(i, i * 3, 0.8 - i * 0.03))
        for i in range(61):
            result = p.estimate(6 + i * 0.1)
            self.assertTrue(result["permit"])
            self.assertEqual(result["speed_cap"], 8)
            self.assertFalse(result["verify_stop"])
        self.assertFalse(p.estimate(15)["permit"])

    def test_speed_reduces_with_closing_distance_and_prediction_only_requests_verification(self):
        caps = []
        for distance in (0.5, 0.35, 0.30, 0.25, 0.22):
            p = RangePrediction()
            p.add(1, measurement(1, 1, distance))
            r = p.estimate(4)
            caps.append(r["speed_cap"])
            self.assertFalse(r["near"])
        self.assertEqual(caps, sorted(caps, reverse=True))
        p = RangePrediction()
        p.add(1, measurement(1, 1, 0.2))
        self.assertTrue(p.estimate(4)["verify_stop"])
        self.assertFalse(p.estimate(4)["near"])

    def test_duplicate_wrong_basis_nonfinite_and_expired_data_cannot_grant_motion(self):
        p = RangePrediction()
        self.assertFalse(p.add(1, dict(measurement(1, 1, 0.4), distance_m=float("nan"))))
        self.assertFalse(p.add(1, dict(measurement(1, 1, 0.4), distance_kind="Z")))
        self.assertTrue(p.add(1, measurement(1, 1, 0.4)))
        self.assertFalse(p.add(1, measurement(1, 1, 0.2)))
        self.assertFalse(p.estimate(10)["permit"])
        p.add(2, measurement(2, 2, 0.5))
        self.assertEqual(len(p.samples), 1)
        self.assertFalse(p.estimate(3, 0.4)["permit"])
