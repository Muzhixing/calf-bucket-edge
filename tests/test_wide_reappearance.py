import unittest
from unittest.mock import patch
import cv2
import numpy as np
from tests import test_tracking
from bucket_robot.perception.target import TargetLock
from tests.test_visual_robust import detection


class WideReappearanceTests(unittest.TestCase):
    def scene(self):
        original = test_tracking.TrackingTests.scene()
        m = TargetLock()
        m.select(original, [480, 250, 720, 450], 1.0)
        blank = np.zeros_like(original) + 25
        self.assertIsNone(m.update(blank, {"arrival": 1.05, "detections": []}))
        changed = blank.copy()
        box = [60, 70, 492, 430]
        changed[70:430, 60:492] = cv2.resize(original[250:450, 480:720][::-1, ::-1], (432, 360))
        return m, changed, blank, box

    def test_far_changed_scale_and_view_confirms_without_five_second_wait(self):
        m, image, blank, box = self.scene()
        result = None
        for i in range(8):
            result = m.update(
                image, {"arrival": 1.2 + i * 0.04, "detections": [detection(box, 0.95)]}
            )
            if i < 3:
                self.assertIsNone(result)
            if result is not None:
                break
        self.assertIsNotNone(result)
        self.assertEqual(result["method"], "detector_wide_relock")
        self.assertLessEqual(result["candidate_verify_s"], 0.161)
        self.assertEqual(m.generation, 2)
        self.assertEqual(result["box"], box)
        self.assertIsNone(m.update(blank, {"arrival": 1.6, "detections": []}))

    def test_false_boxes_over_full_occlusion_never_confirm(self):
        m, image, blank, box = self.scene()
        for i in range(12):
            self.assertIsNone(
                m.update(blank, {"arrival": 1.2 + i * 0.04, "detections": [detection(box, 0.95)]})
            )
        self.assertEqual(m.generation, 1)

    def test_single_weak_ambiguous_or_inconsistent_observations_do_not_confirm(self):
        for case in ("single", "weak", "ambiguous", "different_view", "duplicate"):
            with self.subTest(case=case):
                m, image, blank, box = self.scene()
                result = None
                other = image.copy()
                other[70:430, 60:492] = 255 - other[70:430, 60:492]
                for i in range(8):
                    ds = [detection(box, 0.7 if case == "weak" else 0.95)]
                    frame = other if case == "different_view" and i % 2 else image
                    if case == "single" and i:
                        ds = []
                    if case == "ambiguous":
                        ds.append(detection([800, 250, 1050, 500], 0.95))
                    stamp = 1.2 if case == "duplicate" else 1.2 + i * 0.04
                    result = m.update(frame, {"arrival": stamp, "detections": ds})
                    self.assertIsNone(result)
                self.assertEqual(m.generation, 1)

    def test_gap_and_missing_detection_reset_confirmation(self):
        m, image, blank, box = self.scene()
        d = detection(box, 0.95)
        for stamp in (1.2, 1.24):
            self.assertIsNone(m.update(image, {"arrival": stamp, "detections": [d]}))
        self.assertIsNone(m.update(image, {"arrival": 1.5, "detections": [d]}))
        self.assertEqual(len(m.wide_views), 1)
        self.assertIsNone(m.update(image, {"arrival": 1.54, "detections": []}))
        self.assertEqual(len(m.wide_views), 0)
        self.assertIsNone(m.update(image, {"arrival": 1.58, "detections": [d]}))

    def test_manual_selection_never_uses_global_automatic_acquisition(self):
        m, image, blank, box = self.scene()
        m.mode = "manual"
        for i in range(8):
            self.assertIsNone(
                m.update(image, {"arrival": 1.2 + i * 0.04, "detections": [detection(box, 0.95)]})
            )
        self.assertEqual(m.generation, 1)

    def test_moderate_view_change_needs_longer_all_strong_sequence(self):
        for weak in (False, True):
            m, image, blank, box = self.scene()
            # Isolate a moderate adjacent-view score; all geometry, image
            # texture and candidate initialization still use real image data.
            with patch("bucket_robot.perception.target.weighted_ncc", return_value=0.45):
                for i in range(4):
                    score = 0.7 if weak and i == 2 else 0.95
                    result = m.update(
                        image, {"arrival": 1.2 + i * 0.08, "detections": [detection(box, score)]}
                    )
                    if i < 3 or weak:
                        self.assertIsNone(result)
                if not weak:
                    self.assertEqual(result["method"], "detector_wide_relock")
                    self.assertEqual(result["candidate_evidence"], "strong_detection_sequence")

    def test_relock_warmup_needs_current_pixels_and_cannot_renew_itself(self):
        m, image, blank, box = self.scene()
        result = None
        for i in range(8):
            result = m.update(
                image, {"arrival": 1.2 + i * 0.04, "detections": [detection(box, 0.95)]}
            )
            if result:
                break
        t = m.tracker
        deadline = t.measurement_warmup_until
        weak = detection(box, 0.7)
        for stamp in (deadline - 0.04, deadline + 0.04):
            t.detector_support = [stamp - 0.08, stamp - 0.04]
            t.detector_view = (t.measured_view(image, box), box, stamp - 0.04)
            self.assertEqual(t.measured_continuity(image, weak, stamp), stamp < deadline)
            self.assertFalse(t.measured_continuity(image, None, stamp))
            self.assertFalse(t.measured_continuity(blank, weak, stamp))
            t.remember_detection(image, weak, stamp, result)
            self.assertEqual(t.measurement_warmup_until, deadline)


if __name__ == "__main__":
    unittest.main()
