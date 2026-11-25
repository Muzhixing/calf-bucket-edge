import unittest
from unittest.mock import patch
import numpy as np
from tests import test_tracking
from bucket_robot.perception.target import TargetTracker


class RecentViewTests(unittest.TestCase):
    def tracker(self):
        image = test_tracking.TrackingTests.scene()
        t = TargetTracker(image, [480, 250, 720, 450], 1.0)
        t.detector_support = [1.01, 1.05]
        return t

    def accept_changed_view(self, t, stamp, score=None, pair=0.94):
        d = [] if score is None else [dict(left=480, top=250, right=720, bottom=450, score=score)]
        aligned = t.anchor.copy()
        with (
            patch.object(t, "alignment", return_value=(t.box.copy(), aligned, 0.35, 0.55)),
            patch("bucket_robot.perception.target.weighted_ncc", side_effect=[pair, 0.35]),
            patch(
                "bucket_robot.perception.target.coarse_ncc",
                side_effect=[0.40] * len(t.references) + [0.97],
            ),
        ):
            return t.accept(t.gray, np.eye(3), stamp, "flow", d)

    def test_recent_continuity_has_nonrenewing_deadline(self):
        t = self.tracker()
        self.assertEqual(self.accept_changed_view(t, 1.10)["method"], "flow_recent")
        self.accept_changed_view(t, 1.19)
        with self.assertRaises(RuntimeError):
            self.accept_changed_view(t, 1.21)
        self.assertEqual(t.last_verified_view, 1.0)

    def test_only_confirmed_strong_detection_renews_recent_view(self):
        for score, renew in [(None, False), (0.5, False), (0.61, False), (0.7, True)]:
            with self.subTest(score=score):
                t = self.tracker()
                self.accept_changed_view(t, 1.15, score)
                self.assertEqual(t.last_verified_view, 1.15 if renew else 1.0)
                if renew:
                    self.accept_changed_view(t, 1.25)
                else:
                    with self.assertRaises(RuntimeError):
                        self.accept_changed_view(t, 1.25)

    def test_changed_pair_or_single_detection_does_not_bypass_identity(self):
        t = self.tracker()
        with self.assertRaises(RuntimeError):
            self.accept_changed_view(t, 1.15, 0.9, pair=0.4)
        t = self.tracker()
        t.detector_support = [1.24]
        with self.assertRaises(RuntimeError):
            self.accept_changed_view(t, 1.25, 0.9)

    def test_confirmed_changed_view_reaches_reference_bank(self):
        t = self.tracker()
        anchor = t.anchor.copy()
        for stamp in (1.50, 1.60, 1.70):
            t.detector_support = [stamp - 0.08, stamp - 0.03]
            result = self.accept_changed_view(t, stamp, 0.70)
            self.assertEqual(result["method"], "flow_supported")
        self.assertEqual(len(t.references), 2)
        np.testing.assert_array_equal(t.references[0], anchor)

    def test_reference_update_requires_strong_current_pair_and_multiple_votes(self):
        for score, pair in [(0.61, 0.94), (0.70, 0.86)]:
            t = self.tracker()
            for stamp in (1.50, 1.60, 1.70):
                t.detector_support = [stamp - 0.08, stamp - 0.03]
                self.accept_changed_view(t, stamp, score, pair)
            self.assertEqual(len(t.references), 1)
        t = self.tracker()
        t.detector_support = [1.42, 1.47]
        self.accept_changed_view(t, 1.50, 0.70)
        self.assertEqual(len(t.references), 1)


if __name__ == "__main__":
    unittest.main()
