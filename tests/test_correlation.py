import sys, unittest, importlib.util
from pathlib import Path
from unittest.mock import patch
import cv2, numpy as np
from bucket_robot.perception import target

ROOT = Path(target.__file__).parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from tests import test_tracking
from bucket_robot.perception.correlation import install


@unittest.skipUnless(hasattr(cv2, "TrackerCSRT_create"), "optional CSRT dependency unavailable")
class CsrtTests(unittest.TestCase):
    def setup_tracker(self):
        spec = importlib.util.spec_from_file_location(
            "tested_csrt", ROOT / "bucket_robot.perception.target.py"
        )
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        install(m)
        image = test_tracking.TrackingTests.scene()
        image = cv2.GaussianBlur(image, (7, 7), 1.5)
        return m, m.TargetTracker(image, [480, 250, 720, 450], 1.0), image

    def unavailable_base(self, t):
        t.flow_transform = lambda *a: (_ for _ in ()).throw(RuntimeError("forced_flow_failure"))
        t.local_search = lambda *a: (_ for _ in ()).throw(RuntimeError("forced_local_failure"))

    def test_real_csrt_can_propose_fresh_matching_box(self):
        m, t, image = self.setup_tracker()
        self.unavailable_base(t)
        moved = cv2.warpAffine(image, np.float32([[1, 0, 12], [0, 1, 8]]), (1280, 720))
        result = t.update(moved, 1.05)
        self.assertEqual(result["method"], "csrt")
        self.assertLess(abs(result["box"][0] - 492), 6)
        self.assertEqual(len(t.references), 1)

    def test_occlusion_cannot_use_even_a_confident_csrt_box(self):
        m, t, image = self.setup_tracker()
        self.unavailable_base(t)

        class Fake:
            def update(self, image):
                return True, (240, 125, 120, 100)

        t.csrt = Fake()
        with self.assertRaises(RuntimeError):
            t.update(np.zeros_like(image) + 25, 1.05)
        self.assertEqual(t.stamp, 1.0)
        self.assertIsNone(t.csrt)

    def test_wrong_appearance_rejected_and_reference_immutable(self):
        m, t, image = self.setup_tracker()
        self.unavailable_base(t)
        wrong = image.copy()
        wrong[250:450, 480:720] = 255 - wrong[250:450, 480:720]

        class Fake:
            def update(self, image):
                return True, (240, 125, 120, 100)

        t.csrt = Fake()
        anchor = t.anchor.copy()
        with self.assertRaises(RuntimeError):
            t.update(wrong, 1.05)
        np.testing.assert_array_equal(t.references[0], anchor)
        self.assertEqual(len(t.references), 1)

    def test_old_frame_cannot_activate_csrt(self):
        m, t, image = self.setup_tracker()
        self.unavailable_base(t)
        with self.assertRaises(RuntimeError):
            t.update(image, 1.3)
        self.assertEqual(t.csrt_attempts, 0)

    def test_provisional_flow_failure_never_runs_appearance_fallback(self):
        m, t, image = self.setup_tracker()
        manager = m.TargetLock()
        manager.provisional = dict(
            tracker=t, started=1.0, frames=1, score=0.9, known=True, votes=[(1.0, 0.9)]
        )
        t.flow_transform = lambda *a: (_ for _ in ()).throw(RuntimeError("forced_flow_failure"))
        with (
            patch.object(t, "local_search", side_effect=AssertionError("unusable local search")),
            patch.object(cv2, "TrackerCSRT_create", side_effect=AssertionError("unusable CSRT")),
        ):
            self.assertIsNone(manager.verified_reappearance(image, 1.04, []))
        self.assertIsNone(manager.provisional)
        self.assertEqual(t.stamp, 1.0)
        self.assertEqual(t.csrt_attempts, 0)

    def test_recovery_refreshes_frame_before_next_csrt_proposal(self):
        m, t, image = self.setup_tracker()
        moved = cv2.warpAffine(image, np.float32([[1, 0, 96], [0, 1, 0]]), (1280, 720))
        d = dict(left=576, top=250, right=816, bottom=450, score=0.9)
        self.assertIsNone(t.recover(moved, 1.3, [d]))
        self.assertIsNone(t.recover(moved, 1.4, [d]))
        self.assertIsNotNone(t.recover(moved, 1.5, [d]))
        self.unavailable_base(t)
        next_frame = cv2.warpAffine(image, np.float32([[1, 0, 108], [0, 1, 8]]), (1280, 720))
        result = t.update(next_frame, 1.55)
        self.assertEqual(result["method"], "csrt")
        self.assertLess(abs(result["box"][0] - 588), 6)


if __name__ == "__main__":
    unittest.main()
