import unittest
import cv2, numpy as np
from bucket_robot.perception.matching import weighted_response, SpectrumCache


def original(search, template, weights):
    s = search.astype(np.float32)
    t = template.astype(np.float32)
    w = weights.astype(np.float32)
    total = float(w.sum())
    center = t - float((t * w).sum() / total)
    variance = float((center * center * w).sum())
    n = cv2.matchTemplate(s, center * w, cv2.TM_CCORR)
    a = cv2.matchTemplate(s, w, cv2.TM_CCORR)
    b = cv2.matchTemplate(s * s, w, cv2.TM_CCORR)
    d = np.sqrt(np.maximum(b - a * a / total, 0) * max(variance, 1e-6))
    r = np.full(n.shape, -1.0, np.float32)
    np.divide(n, d, out=r, where=d > 1.0)
    return np.clip(r, -1, 1)


class MatchingTests(unittest.TestCase):
    def test_exact_grid_matches_original_for_texture_exposure_and_boundaries(self):
        rng = np.random.default_rng(8)
        for h, w, gap in [
            (64, 64, 0),
            (65, 67, 1),
            (150, 210, 8),
            (400, 418, 16),
            (440, 476, 16),
            (45, 60, 50),
        ]:
            s = rng.integers(30, 210, (h + gap, w + gap), dtype=np.uint8)
            t = s[gap // 2 : gap // 2 + h, gap // 2 : gap // 2 + w]
            weights = rng.uniform(0.001, 1, (h, w)).astype(np.float32)
            for image in (s, (s * 0.8 + 25).astype(np.uint8)):
                a = original(image, t, weights)
                b = weighted_response(image, t, weights)
                np.testing.assert_allclose(b, a, atol=2e-5, rtol=2e-5)
                self.assertEqual(np.argmax(a), np.argmax(b))

    def test_flat_and_wrong_content_cannot_become_strong_match(self):
        rng = np.random.default_rng(91)
        w = np.ones((128, 160), np.float32)
        for template in (
            np.full(w.shape, 95, np.uint8),
            rng.integers(0, 255, w.shape, dtype=np.uint8),
        ):
            response = weighted_response(np.full((144, 176), 95, np.uint8), template, w)
            self.assertTrue(np.isfinite(response).all())
            self.assertLess(float(response.max()), 0.4)

    def test_cache_reuses_only_reference_spectra_and_is_invalidated(self):
        rng = np.random.default_rng(29)
        s = rng.integers(0, 255, (140, 172), dtype=np.uint8)
        t = s[6:134, 6:166].copy()
        w = rng.uniform(0.01, 1, t.shape).astype(np.float32)
        c = SpectrumCache()
        c.bind(t, w)
        a = weighted_response(s, t, w, cache=c)
        self.assertGreater(c.bytes, 0)
        b = weighted_response(s[::-1].copy(), t, w, cache=c)
        np.testing.assert_allclose(b, weighted_response(s[::-1].copy(), t, w), atol=1e-6)
        self.assertGreater(float(a.max()), float(b.max()))
        changed = 255 - t
        c.bind(changed, w)
        self.assertEqual(c.bytes, 0)
        np.testing.assert_allclose(
            weighted_response(s, changed, w, cache=c), original(s, changed, w), atol=2e-5
        )
        c.bind(changed, w.copy())
        self.assertEqual(c.bytes, 0)

    def test_cache_memory_is_bounded(self):
        c = SpectrumCache(max_bytes=40000)
        for i in range(20):
            a = np.zeros((64, 64), np.float32)
            c.put(i, (1, 1, a, a.copy()))
            self.assertLessEqual(c.bytes, c.max_bytes)
        self.assertEqual(len(c.entries), 1)
        c.put(
            "too_large", (1, 1, np.zeros((200, 200), np.float32), np.zeros((200, 200), np.float32))
        )
        self.assertLessEqual(c.bytes, c.max_bytes)


if __name__ == "__main__":
    unittest.main()
