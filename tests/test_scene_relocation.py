"""Scene geometry proposes a correspondence; observed target pixels must prove it."""

import unittest
import cv2
import numpy as np

from bucket_robot.perception.relocation import (
    FeatureRelocationJob,
    known_feature_relocation,
    known_scene_relocation,
    scene_projection,
)


class SceneRelocationTests(unittest.TestCase):
    def pair(self):
        rng = np.random.default_rng(2711)
        image = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
        box = [460, 300, 700, 550]
        y, x = np.mgrid[:250, :240]
        body = np.clip(90 + x * 0.35 + 25 * np.sin(y / 18), 0, 255).astype(np.uint8)
        image[300:550, 460:700] = body[:, :, None]
        matrix = np.array([[0.97, 0.015, 80], [-0.01, 1.01, -15], [0.00002, -0.00001, 1]])
        current = cv2.warpPerspective(image, matrix, (1280, 720))
        corners = np.float32([[460, 300], [700, 300], [700, 550], [460, 550]])
        corners = cv2.perspectiveTransform(corners[None], matrix)[0]
        candidate = np.r_[corners.min(0), corners.max(0)].round().astype(int).tolist()
        return image, box, current, candidate

    def test_low_texture_target_can_use_distributed_scene_and_common_pixels(self):
        image, old, current, box = self.pair()
        self.assertIsNone(known_feature_relocation(image, old, current, box))
        projection = scene_projection(image, current)
        self.assertIsNotNone(projection)
        proof = known_scene_relocation(image, old, current, box, projection)
        self.assertIsNotNone(proof)
        self.assertGreaterEqual(proof["common_pixel_ncc"], 0.85)
        self.assertGreaterEqual(proof["common_fraction"], 0.75)

    def test_matching_scene_does_not_authorize_replaced_or_displaced_object(self):
        image, old, current, box = self.pair()
        projection = scene_projection(image, current)
        self.assertIsNone(
            known_scene_relocation(image, old, current, [30, 50, 200, 300], projection)
        )
        changed = current.copy()
        x, y, r, b = box
        changed[y:b, x:r] = np.random.default_rng(92).integers(
            0, 255, (b - y, r - x, 3), dtype=np.uint8
        )
        projection = scene_projection(image, changed)
        self.assertIsNotNone(projection)
        self.assertIsNone(known_scene_relocation(image, old, changed, box, projection))

    def test_missing_geometry_or_concentrated_features_cannot_prove_target(self):
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.assertIsNone(scene_projection(blank, blank))
        corner = blank.copy()
        corner[:180, :200] = np.random.default_rng(1).integers(
            0, 255, (180, 200, 3), dtype=np.uint8
        )
        self.assertIsNone(scene_projection(corner, corner))

    def test_background_job_preserves_unique_detection_requirement(self):
        image, old, current, box = self.pair()
        detection = dict(zip(("left", "top", "right", "bottom"), box), score=0.94)
        for detections, accepted in [
            ([detection], True),
            ([], False),
            ([detection, detection], False),
        ]:
            job = FeatureRelocationJob(
                1, [(image, old)], current, dict(arrival=2, detections=detections)
            )
            try:
                job.thread.join(3)
                self.assertFalse(job.thread.is_alive())
                self.assertIsNone(job.error)
                self.assertEqual(job.result is not None, accepted)
            finally:
                job.close()
