import os, sys
from pathlib import Path
import unittest
import cv2
import numpy as np
from bucket_robot.perception.session import RecoveringTarget


class TargetRecoveryTests(unittest.TestCase):
    def scene(self, seed=15, box=(480, 300, 780, 600)):
        rng = np.random.default_rng(seed)
        im = np.full((720, 1280, 3), 25, np.uint8)
        x, y, r, b = box
        patch = rng.integers(0, 256, (b - y, r - x, 3), dtype=np.uint8)
        im[y:b, x:r] = cv2.GaussianBlur(patch, (5, 5), 1)
        return im, list(box)

    def detection(self, b):
        return dict(zip(("left", "top", "right", "bottom", "score"), [*b, 0.95]))

    def lost(self, box=(480, 300, 780, 600)):
        im, b = self.scene(box=box)
        m = RecoveringTarget(True)
        m.select(im, b, 1.0)
        self.assertIsNotNone(m.update(im, dict(arrival=1.0, detections=[self.detection(b)])))
        self.assertIsNone(m.update(np.zeros_like(im) + 25, dict(arrival=1.04, detections=[])))
        self.assertEqual(m.state, "recovering")
        return m, im, b

    def test_same_target_reappears_and_task_identity_survives(self):
        for box in ((480, 300, 780, 600), (480, 450, 780, 720)):
            m, im, b = self.lost(box)
            answer = None
            for i in range(15):
                answer = m.update(im, dict(arrival=1.2 + i * 0.04, detections=[self.detection(b)]))
                if answer is not None:
                    break
            self.assertIsNotNone(answer, m.reason)
            self.assertEqual(answer["task_target_id"], 1)
            self.assertEqual(m.reacquired, 1)

    def test_wrong_appearance_ambiguous_or_duplicate_frames_cannot_resume(self):
        for case in ("wrong", "ambiguous", "duplicate"):
            m, im, b = self.lost()
            if case == "wrong":
                im, _ = self.scene(seed=111)
            for i in range(12):
                ds = [self.detection(b)]
                if case == "ambiguous":
                    ds.append(self.detection([485, 300, 785, 600]))
                answer = m.update(
                    im, dict(arrival=1.2 if case == "duplicate" else 1.2 + i * 0.04, detections=ds)
                )
                self.assertIsNone(answer, case)
            self.assertEqual(m.reacquired, 0)

    def test_camera_exception_preserves_reference_for_recovery(self):
        im, b = self.scene()
        m = RecoveringTarget(True)
        m.select(im, b, 1.0)
        m.update(im, dict(arrival=1.0, detections=[self.detection(b)]))
        m.missing("camera_timeout")
        self.assertEqual(m.state, "recovering")
        self.assertIsNotNone(m.frozen)
        for i in range(10):
            t = m.update(im, dict(arrival=2 + i * 0.04, detections=[self.detection(b)]))
            if t is not None:
                break
        self.assertIsNotNone(t)

    def test_stage_handoff_reverifies_shifted_same_bucket(self):
        seed, b = self.scene()
        m = RecoveringTarget(True)
        m.load_verified_reference(seed, b, 1.0)
        image, moved = self.scene(box=(500, 300, 800, 600))
        answer = None
        for i in range(12):
            answer = m.update(image, dict(arrival=2 + i * 0.04, detections=[self.detection(moved)]))
            if answer is not None:
                break
        self.assertIsNotNone(answer, m.reason)
        self.assertEqual(answer["task_target_id"], 1)
        self.assertEqual(m.reacquired, 1)


if __name__ == "__main__":
    unittest.main()
