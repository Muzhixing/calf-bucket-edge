"""Near buckets retain measured bearing without expanding whole-object limits."""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from bucket_robot.perception.bearing import BearingLock
from bucket_robot.perception.session import RecoveringTarget
from bucket_robot.perception.target import valid_box, visible_detection_box


def detection(box):
    return dict(zip(("left", "top", "right", "bottom", "score"), [*box, 0.95]))


class CloseTargetContinuityTests(unittest.TestCase):
    def setUp(self):
        self.image = cv2.GaussianBlur(
            np.random.default_rng(593).integers(0, 256, (720, 1280, 3), dtype=np.uint8),
            (7, 7),
            1.5,
        )

    def test_large_visible_bucket_uses_inner_pixels_and_raw_horizontal_bearing(self):
        old = [120, 120, 1180, 716]
        near = [60, 70, 1220, 716]
        self.assertFalse(valid_box(near))
        roi = visible_detection_box(near)
        self.assertIsNotNone(roi)
        self.assertTrue(valid_box(roi))
        self.assertEqual(roi[::2], near[::2])
        self.assertGreaterEqual(roi[1], near[1])
        self.assertLessEqual(roi[3], near[3])
        manager = BearingLock()
        manager.select(self.image, old, 1)
        for i in range(1, 31):
            # Detector height jitter must not redefine the physical visible pixels.
            box = [60, 70 + i % 3 * 10, 1220, 716]
            target = manager.update(
                self.image, dict(arrival=1 + i * 0.04, detections=[detection(box)])
            )
            self.assertIsNotNone(target, manager.reason)
            self.assertEqual(target["raw_box"], box)
            self.assertTrue(target["horizontal_bearing_valid"])
            self.assertEqual(target["offset_px"][0], 0)
        self.assertEqual(manager.generation, 1)

    def test_edge_install_does_not_run_segmentation_inside_frame_deadline(self):
        manager = RecoveringTarget(True)
        with patch("cv2.grabCut", side_effect=AssertionError("blocking segmentation")):
            manager.select(self.image, [60, 70, 1220, 716], 1)
            target = manager.update(
                self.image,
                dict(arrival=1, detections=[detection([60, 70, 1220, 716])]),
            )
            self.assertIsNotNone(target)
            manager.missing("camera_timeout")
        self.assertIsNotNone(manager.frozen)

    def test_measured_large_bearing_completes_left_gate_only_after_pose_and_stop(self):
        from bucket_robot.missions.post_alignment import PostAlignmentGate

        manager = BearingLock()
        box = [60, 70, 1220, 716]
        manager.select(self.image, box, 1)
        gate = PostAlignmentGate(30)
        reason = None
        for i in range(1, 31):
            stamp = 1 + i * 0.05
            target = manager.update(self.image, dict(arrival=stamp, detections=[detection(box)]))
            self.assertIsNotNone(target, manager.reason)
            angle = 126 if i < 10 else 131
            speed, reason = gate.evaluate(
                stamp,
                target,
                1,
                angle,
                True,
                commanded_a=angle,
                chassis_stopped=i >= 12,
            )
            if i < 12:
                self.assertIsNone(reason)
            if angle == 131:
                self.assertEqual(speed, 0)
        self.assertEqual(reason, "left_reference_reached")

    def test_large_view_still_rejects_missing_ambiguous_wrong_and_delayed_evidence(self):
        near = [60, 70, 1220, 716]
        for case in ("missing", "ambiguous", "wrong", "delayed"):
            with self.subTest(case=case):
                manager = BearingLock()
                manager.select(self.image, near, 1)
                image = self.image if case != "wrong" else np.full_like(self.image, 25)
                ds = [] if case == "missing" else [detection(near)]
                if case == "ambiguous":
                    ds.append(detection([62, 72, 1218, 716]))
                target = manager.update(
                    image, dict(arrival=1.3 if case == "delayed" else 1.04, detections=ds)
                )
                self.assertIsNone(target)

    def test_full_frame_and_invalid_regions_remain_rejected(self):
        for invalid in (
            [0, 0, 1280, 720],
            [-1, 0, 1280, 720],
            [0, 0, 20, 720],
            [0, 0, float("nan"), 720],
        ):
            self.assertIsNone(visible_detection_box(invalid))
