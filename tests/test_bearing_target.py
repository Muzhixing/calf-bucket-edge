import os, sys
from pathlib import Path
import unittest
import cv2
import numpy as np
from bucket_robot.perception.bearing import BearingLock
from bucket_robot.missions.forward import ForwardGate


def detection(box, score=0.95):
    return dict(zip(("left", "top", "right", "bottom", "score"), [*box, score]))


class BearingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        self.texture = cv2.GaussianBlur(
            rng.integers(0, 256, (320, 300, 3), dtype=np.uint8), (15, 15), 3
        )
        for i in range(20):
            cv2.circle(
                self.texture, tuple(map(int, rng.integers(20, 280, 2))), 10, (10, 40, 200), -1
            )

    def frame(self, x=480, y=430):
        im = np.full((720, 1280, 3), 25, np.uint8)
        l, t, r, b = max(0, x), max(0, y), min(1280, x + 300), min(720, y + 320)
        im[t:b, l:r] = self.texture[t - y : b - y, l - x : r - x]
        return im, [l, t, r, b]

    def test_bottom_crop_continues_and_horizontal_bearing_remains_measured(self):
        im, b = self.frame(y=390)
        m = BearingLock()
        m.select(im, b, 1)
        for i in range(1, 31):
            im, b = self.frame(y=390 + i * 2)
            t = m.update(
                im,
                dict(
                    arrival=1 + i * 0.04,
                    detections=[detection(b), detection([900, 100, 1200, 350], 0.99)],
                ),
            )
            self.assertIsNotNone(t, m.reason)
            self.assertTrue(t["horizontal_bearing_valid"])
            self.assertEqual(t["offset_px"][0], -10)
        self.assertTrue(t["vertical_clipped"])
        self.assertEqual(m.generation, 1)

    def test_left_crop_to_full_keeps_identity_without_assuming_hidden_center(self):
        im, b = self.frame(x=-30)
        m = BearingLock()
        m.select(im, b, 1)
        for i in range(1, 26):
            im, b = self.frame(x=-30 + i * 2)
            t = m.update(im, dict(arrival=1 + i * 0.04, detections=[detection(b)]))
            self.assertIsNotNone(t, m.reason)
            self.assertEqual(t["horizontal_bearing_valid"], b[0] >= 8)
        self.assertEqual(m.generation, 1)

    def test_clipped_detector_crop_jitter_requires_same_pixel_geometry(self):
        im, b = self.frame()
        m = BearingLock()
        m.select(im, b, 1)
        shifted = [b[0], b[1] + 25, b[2], b[3]]
        t = m.update(im, dict(arrival=1.04, detections=[detection(shifted)]))
        self.assertIsNotNone(t, m.reason)
        self.assertEqual(t["method"], "edge_detector_common_flow")
        self.assertGreater(t["common_region_ncc"], 0.98)
        self.assertEqual(m.generation, 1)

    def test_brief_weak_detection_needs_current_geometry_and_cannot_renew_deadline(self):
        im, b = self.frame()
        m = BearingLock()
        m.select(im, b, 1)
        for stamp in (1.04, 1.08, 1.12):
            t = m.update(im, dict(arrival=stamp, detections=[detection(b, 0.622)]))
            self.assertIsNotNone(t, m.reason)
            self.assertEqual(t["method"], "edge_detector_weak_common_flow")
            self.assertGreater(t["common_region_ncc"], 0.98)
        self.assertIsNone(m.update(im, dict(arrival=1.16, detections=[detection(b, 0.622)])))

    def test_missing_ambiguous_changed_appearance_or_stale_cannot_continue(self):
        for case in ("missing", "ambiguous", "appearance", "stale"):
            im, b = self.frame()
            m = BearingLock()
            m.select(im, b, 1)
            ds = [detection(b)]
            if case == "missing":
                ds = []
            if case == "ambiguous":
                ds.append(detection([b[0] + 2, b[1], b[2] + 2, b[3]]))
            if case == "appearance":
                im[:] = 40
            t = m.update(im, dict(arrival=1 + (0.3 if case == "stale" else 0.04), detections=ds))
            self.assertIsNone(t, case)
            self.assertIsNone(m.update(im, dict(arrival=1.4, detections=[detection(b)])))

    def test_only_horizontal_evidence_permits_forward_and_alignment(self):
        def t(stamp, horizontal):
            return dict(
                timestamp=stamp,
                offset_px=[0, 220],
                partial_visible=True,
                horizontal_clipped=not horizontal,
                horizontal_bearing_valid=horizontal,
                bearing_evidence="current_detection_and_visible_appearance",
            )

        g = ForwardGate(60, until_left=True)
        self.assertEqual(g.evaluate(1, t(1, False), 1, 132, True), (0, None))
        g.evaluate(2, t(2, True), 1, 97, True)
        self.assertEqual(g.evaluate(2.8, t(2.8, True), 1, 97, True), (8, None))
        with self.assertRaises(RuntimeError):
            g.evaluate(3, t(3, False), 1, 100, True)
        self.assertEqual(g.evaluate(4, t(4, True), 1, 131, True), (0, "left_reference_reached"))
        legacy = dict(timestamp=1, offset_px=[0, 220], partial_visible=True)
        with self.assertRaises(RuntimeError):
            ForwardGate(5).evaluate(1, legacy, 1, 97, True)


if __name__ == "__main__":
    unittest.main()
