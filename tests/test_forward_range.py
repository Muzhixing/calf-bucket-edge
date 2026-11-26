"""Distance-first decisions cannot reuse slow moving or wrong-target measurements."""

from unittest.mock import patch
import unittest

from bucket_robot.missions.forward_range import ForwardRange


class ForwardRangeTests(unittest.TestCase):
    def ranger(self, capture=1, distance=0.3):
        r = ForwardRange(None, None, None)
        r.answer = dict(
            identity=1,
            box=[400, 200, 800, 600],
            patch=None,
            weights=None,
            measurement=dict(
                valid=True,
                frame_id=4,
                capture_time=capture,
                distance_m=distance,
                lower_m=distance - 0.005,
                distance_kind="stereo_optical_midpoint_to_bucket_surface",
            ),
        )
        return r

    def obs(self, now, identity=1):
        return dict(identity=identity, target=dict(timestamp=now, box=[400, 200, 800, 600]))

    def test_near_distance_ignores_left_reference_but_requires_fresh_same_target(self):
        for distance in (0.18, 0.2):
            r = self.ranger(distance=distance)
            r.static_since = 0.5
            with patch(
                "bucket_robot.missions.forward_range.static_surface_matches", return_value=True
            ):
                self.assertTrue(r.evaluate(1.1, self.obs(1.1), None, True)["near"])
        for case in ("old", "identity", "lost", "invalid"):
            r = self.ranger(distance=0.19)
            if case == "invalid":
                r.answer["measurement"]["valid"] = False
            now = 10 if case == "old" else 1.1
            obs = None if case == "lost" else self.obs(now, 2 if case == "identity" else 1)
            answer = r.evaluate(now, obs, None, False)
            self.assertFalse(answer["near"], case)
            self.assertFalse(answer["permit"], case)

    def test_slow_model_requires_disabled_static_scene_throughout_capture(self):
        for case in ("static", "changed", "moved", "late_stop"):
            r = self.ranger(capture=1.1, distance=0.19)
            r.static_since = 1 if case != "late_stop" else 1.5
            with patch(
                "bucket_robot.missions.forward_range.static_surface_matches",
                return_value=case != "changed",
            ):
                result = r.evaluate(3, self.obs(3), None, case != "moved")
            self.assertEqual(result["near"], case == "static", case)

    def test_far_result_is_compensated_until_deadline_and_loss_cancels_it(self):
        r = self.ranger()
        self.assertTrue(r.evaluate(1.1, self.obs(1.1), None, False)["permit"])
        self.assertFalse(r.evaluate(1.15, self.obs(1.15, 2), None, False)["permit"])
        self.assertTrue(r.evaluate(1.5, self.obs(1.5), None, False)["permit"])
        self.assertFalse(r.evaluate(10, self.obs(10), None, False)["permit"])
        self.assertFalse(r.evaluate(1.6, None, None, False)["permit"])

    def test_worker_failure_pauses_without_claiming_completion(self):
        r = self.ranger()
        r.error = "model unavailable"
        answer = r.evaluate(1.1, self.obs(1.1), None, True)
        self.assertFalse(answer["near"])
        self.assertFalse(answer["permit"])
        self.assertEqual(answer["reason"], "range_worker_unavailable")

    def test_arm_movement_rejects_even_matching_old_measurement_until_post_settle_capture(self):
        r = self.ranger(capture=1, distance=0.19)
        r.static_since = 0.5
        r.invalidate_before(2)
        with patch("bucket_robot.missions.forward_range.static_surface_matches", return_value=True):
            self.assertFalse(r.evaluate(2.1, self.obs(2.1), None, True)["permit"])
            r.answer["measurement"].update(capture_time=2.2, frame_id=5)
            answer = r.evaluate(2.3, self.obs(2.3), None, True)
            self.assertTrue(answer["static_verified"])
            self.assertTrue(answer["near"])
