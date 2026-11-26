"""Slow identity proof must never block fresh observations or authorize stale motion."""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from bucket_robot.perception.session import RecoveringTarget


class RecoveryDeadlineTests(unittest.TestCase):
    def scene(self, clipped=False):
        image = np.full((720, 1280, 3), 25, np.uint8)
        box = [450, 320, 880, 718 if clipped else 690]
        image[box[1] : box[3], box[0] : box[2]] = cv2.GaussianBlur(
            np.random.default_rng(312).integers(
                0, 256, (box[3] - box[1], box[2] - box[0], 3), dtype=np.uint8
            ),
            (5, 5),
            1,
        )
        detection = dict(zip(("left", "top", "right", "bottom", "score"), [*box, 0.95]))
        manager = RecoveringTarget(True)
        manager.select(image, box, 1)
        manager.update(image, dict(arrival=1, detections=[detection]))
        manager.missing("camera_gap")
        return manager, image, box, detection

    def test_clipped_raw_box_uses_fast_verifier_without_slow_search(self):
        manager, image, box, detection = self.scene(clipped=True)
        with (
            patch("bucket_robot.perception.relocation.known_feature_relocation") as slow,
            patch("cv2.grabCut", side_effect=AssertionError("slow clipped verification")),
        ):
            target = None
            for i in range(8):
                target = manager.update(image, dict(arrival=1.2 + i * 0.04, detections=[detection]))
                if target:
                    break
            self.assertIsNotNone(target, manager.reason)
            slow.assert_not_called()
        self.assertEqual(target["raw_box"], box)
        self.assertTrue(target["vertical_clipped"])
        self.assertTrue(target["horizontal_bearing_valid"])
        self.assertEqual(target["task_target_id"], 1)

    def test_slow_proof_keeps_one_job_and_requires_fresh_frames_after_completion(self):
        manager, image, _, detection = self.scene()
        entered, release = threading.Event(), threading.Event()

        def expensive(*args):
            entered.set()
            if not release.wait(2):
                raise AssertionError("test cleanup failed")
            return {"test_known_geometry": True}

        try:
            with patch("bucket_robot.perception.relocation.known_feature_relocation", expensive):
                with patch.object(
                    manager.known_recovery, "verified_reappearance", return_value=None
                ):
                    completed = threading.Event()

                    def observe():
                        manager.update(image, dict(arrival=1.2, detections=[detection]))
                        completed.set()

                    caller = threading.Thread(target=observe, daemon=True)
                    caller.start()
                    self.assertTrue(entered.wait(1))
                    self.assertTrue(completed.wait(0.5), "feature proof blocked observation")
                    job = manager.feature_job
                    for stamp in (1.24, 1.28):
                        self.assertIsNone(
                            manager.update(image, dict(arrival=stamp, detections=[detection]))
                        )
                        self.assertIs(manager.feature_job, job)
                    release.set()
                    self.assertTrue(job.done.wait(1))
                self.assertIsNone(manager.update(image, dict(arrival=1.28, detections=[detection])))
                self.assertIsNone(manager.update(image, dict(arrival=1.3, detections=[])))
                target = None
                for stamp in (1.32, 1.36, 1.40, 1.44):
                    target = manager.update(image, dict(arrival=stamp, detections=[detection]))
                    if target:
                        break
                self.assertIsNotNone(target, manager.reason)
                self.assertEqual(target["task_target_id"], 1)
                self.assertGreaterEqual(
                    target["appearance_recovery_evidence"]["candidate_detection_votes"], 2
                )
        finally:
            release.set()

    def test_proof_from_previous_recovery_epoch_cannot_install_a_target(self):
        manager, image, _, _ = self.scene()
        done = threading.Event()
        done.set()
        manager.feature_job = SimpleNamespace(
            epoch=manager.relocation_epoch - 1,
            done=done,
            result=(object(), {"old": True}),
        )
        with patch.object(manager.known_recovery, "verified_reappearance", return_value=None):
            self.assertIsNone(manager.update(image, dict(arrival=1.2, detections=[])))
        self.assertIsNone(manager.feature_reference)
        self.assertEqual(manager.reacquired, 0)

    def test_owner_close_cancels_and_drains_pending_opencv_work(self):
        manager, image, _, detection = self.scene()
        entered, release = threading.Event(), threading.Event()

        def expensive(*args):
            entered.set()
            release.wait(2)
            return {"geometry": True}

        try:
            with patch("bucket_robot.perception.relocation.known_feature_relocation", expensive):
                with patch.object(
                    manager.known_recovery, "verified_reappearance", return_value=None
                ):
                    manager.update(image, dict(arrival=1.2, detections=[detection]))
                job = manager.feature_job
                self.assertTrue(entered.wait(1))
                closer = threading.Thread(target=manager.close)
                closer.start()
                self.assertTrue(job.cancel.wait(1))
                release.set()
                closer.join(1)
                self.assertFalse(closer.is_alive())
                self.assertFalse(job.thread.is_alive())
                self.assertIsNone(job.result)
                self.assertEqual(manager.reacquired, 0)
        finally:
            release.set()
