import unittest
import cv2
import numpy as np
from tests import test_tracking
from tests.test_visual_robust import detection
from bucket_robot.perception.target import TargetLock, EdgeTracker, valid_box, visible_detection_box


class EdgeReappearanceTests(unittest.TestCase):
    def setup_scene(self, box=(0, 120, 310, 700)):
        original = test_tracking.TrackingTests.scene()
        m = TargetLock()
        m.select(original, [480, 250, 720, 450], 1.0)
        blank = np.zeros_like(original) + 25
        self.assertIsNone(m.update(blank, {"arrival": 1.05, "detections": []}))
        image = blank.copy()
        x, y, r, b = box
        image[y:b, x:r] = cv2.resize(original[250:450, 480:720], (r - x, b - y))
        return m, image, blank, list(box)

    def acquire(self, m, image, box, start=1.2):
        for i in range(10):
            stamp = start + i * 0.04
            result = m.update(image, {"arrival": stamp, "detections": [detection(box, 0.95)]})
            if result is not None:
                return stamp, result
        self.fail("edge not acquired")

    def test_each_edge_and_corner_reacquire_with_current_evidence(self):
        for box in (
            [0, 120, 310, 700],
            [970, 120, 1280, 700],
            [440, 0, 750, 300],
            [440, 420, 750, 720],
            [0, 0, 310, 580],
        ):
            with self.subTest(box=box):
                m, image, blank, box = self.setup_scene(box)
                stamp, result = self.acquire(m, image, box)
                self.assertEqual(result["method"], "detector_edge_relock")
                self.assertTrue(result["partial_visible"])
                self.assertTrue(valid_box(result["box"]))
                self.assertLess(stamp, 1.41)
                self.assertIsInstance(m.tracker, EdgeTracker)
                result = m.update(
                    image, {"arrival": stamp + 0.04, "detections": [detection(box, 0.95)]}
                )
                self.assertEqual(result["method"], "edge_detector")
                self.assertIsNone(m.update(blank, {"arrival": stamp + 0.08, "detections": []}))
                # The old partial image cannot authorize appearance-only recovery.
                for i in range(3):
                    self.assertIsNone(
                        m.update(image, {"arrival": stamp + 0.12 + i * 0.04, "detections": []})
                    )

    def test_full_box_handover_keeps_current_target_then_occlusion_stops(self):
        m, image, blank, box = self.setup_scene()
        stamp, _ = self.acquire(m, image, box)
        full = [12, 120, 322, 700]
        moved = blank.copy()
        moved[120:700, 12:322] = image[120:700, 0:310]
        results = []
        for i in range(6):
            result = m.update(
                moved, {"arrival": stamp + 0.04 * (i + 1), "detections": [detection(full, 0.95)]}
            )
            self.assertIsNotNone(result)
            results.append(result)
        self.assertTrue(any(r["method"] == "edge_full_lock" for r in results))
        self.assertNotIsInstance(m.tracker, EdgeTracker)
        self.assertIsNone(m.update(blank, {"arrival": stamp + 0.28, "detections": []}))

    def test_missing_weak_ambiguous_duplicate_and_blank_cannot_acquire(self):
        for case in ("missing", "weak", "ambiguous", "duplicate", "blank"):
            m, image, blank, box = self.setup_scene()
            for i in range(10):
                ds = [detection(box, 0.7 if case == "weak" else 0.95)]
                if case == "missing" and i % 2:
                    ds = []
                if case == "ambiguous":
                    ds.append(detection([800, 250, 1050, 500], 0.95))
                r = m.update(
                    blank if case == "blank" else image,
                    {"arrival": 1.2 if case == "duplicate" else 1.2 + i * 0.04, "detections": ds},
                )
                self.assertIsNone(r, case)
            self.assertEqual(m.generation, 1)

    def test_edge_continuation_rejects_blank_ambiguous_and_long_gap(self):
        for case in ("blank", "ambiguous", "gap"):
            m, image, blank, box = self.setup_scene()
            stamp, _ = self.acquire(m, image, box)
            ds = [detection(box, 0.95)]
            if case == "ambiguous":
                ds.append(detection([800, 250, 1050, 500], 0.95))
            self.assertIsNone(
                m.update(
                    blank if case == "blank" else image,
                    {"arrival": stamp + (0.3 if case == "gap" else 0.04), "detections": ds},
                )
            )

    def test_geometry_guard_is_specific_and_manual_mode_unchanged(self):
        for box in (
            [-1, 20, 300, 400],
            [0, 20, 50, 400],
            [0, 0, 1280, 720],
            [0, 20, float("nan"), 400],
            [0, 20, 1300, 400],
        ):
            self.assertIsNone(visible_detection_box(box))
        self.assertFalse(valid_box([0, 120, 310, 700]))
        m, image, blank, box = self.setup_scene()
        m.mode = "manual"
        for i in range(8):
            self.assertIsNone(
                m.update(image, {"arrival": 1.2 + i * 0.04, "detections": [detection(box, 0.95)]})
            )

    def test_reappearance_after_search_timeout_still_uses_edge_evidence(self):
        m, image, blank, box = self.setup_scene()
        self.assertIsNone(m.update(blank, {"arrival": 6.2, "detections": []}))
        self.assertIsNone(m.tracker)
        _, result = self.acquire(m, image, box, 6.3)
        self.assertEqual(result["method"], "detector_edge_relock")

    def test_visible_crop_shift_does_not_create_predictive_velocity(self):
        from bucket_robot.tracking.policy import FollowPolicy

        ref = test_tracking.Reference()
        self.addCleanup(ref.directory.cleanup)
        policy = FollowPolicy(ref.limits, 800, 800, 640, 360)
        for i in range(8):
            target = {
                "timestamp": 1 + i * 0.04,
                "offset_px": [-500 + i * 25, 0],
                "quality": 0.95,
                "method": "edge_detector",
                "partial_visible": True,
            }
            policy.observe(target, 80)
            self.assertEqual(policy.velocity, 0.0)
            self.assertAlmostEqual(policy.filtered_world, policy.world)

    def test_template_fallback_cannot_renew_flow_deadline_without_model_evidence(self):
        m, image, blank, box = self.setup_scene()
        tracker = m.tracker
        gray = tracker.gray
        tracker.accept(gray, tracker.transform, 1.1, "appearance", [])
        self.assertEqual(tracker.last_flow_at, 1.0)
        for ds in ([], [detection(tracker.box, 0.6)]):
            with self.assertRaisesRegex(RuntimeError, "appearance_requires_current_evidence"):
                tracker.accept(gray, tracker.transform, 1.2, "appearance", ds)
        # A current independent detection still permits this fallback.
        result = tracker.accept(
            gray, tracker.transform, 1.2, "appearance", [detection(tracker.box, 0.95)]
        )
        self.assertEqual(result["method"], "appearance")


if __name__ == "__main__":
    unittest.main()
