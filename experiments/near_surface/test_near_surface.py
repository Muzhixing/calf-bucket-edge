"""Offline checks; needs numpy and OpenCV, never opens device interfaces."""
import unittest

import cv2
import numpy as np

from near_surface import NearSurface


class NearSurfaceTest(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1)
        self.model = NearSurface(43.71358314333768)
        self.box = [420, 140, 1070, 720]
        rng = np.random.default_rng(41)
        self.left = rng.integers(0, 256, (720, 1280), dtype=np.uint8)
        self.left = cv2.GaussianBlur(self.left, (3, 3), 0)

    def shifted(self, disparity):
        right = np.zeros_like(self.left)
        right[:, :-disparity] = self.left[:, disparity:]
        return right

    def test_known_near_plane(self):
        # Correspondence constructed independently of the depth estimator.
        result = self.model.measure(self.left, self.shifted(180), self.box)
        self.assertTrue(result['valid'], result)
        self.assertAlmostEqual(result['distance_m'], self.model.factor / 180, delta=.003)

    def test_uniform_pair_has_no_distance(self):
        flat = np.full_like(self.left, 128)
        result = self.model.measure(flat, flat, self.box)
        self.assertFalse(result['valid'])
        self.assertIsNone(result['distance_m'])

    def test_missing_right_image_has_no_distance(self):
        result = self.model.measure(self.left, np.zeros_like(self.left), self.box)
        self.assertFalse(result['valid'])
        self.assertIsNone(result['distance_m'])

    def test_zero_disparity_has_no_distance(self):
        result = self.model.measure(self.left, self.left, self.box)
        self.assertFalse(result['valid'])

    def test_beyond_search_range_has_no_distance(self):
        result = self.model.measure(self.left, self.shifted(300), self.box)
        self.assertFalse(result['valid'])

    def test_invalid_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            NearSurface(float('nan'))
        with self.assertRaises(ValueError):
            self.model.measure(self.left, self.left, [0, 0, 1400, 720])


if __name__ == '__main__':
    unittest.main()
