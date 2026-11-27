import unittest
from unittest.mock import patch
import cv2
import numpy as np
from bucket_robot.perception.session import RecoveringTarget


class TargetRecoveryTests(unittest.TestCase):
    def target_session(self):
        manager = RecoveringTarget(True)

        def close():
            result = manager.close()
            self.assertTrue(result.stopped, result.pending)

        self.addCleanup(close)
        return manager

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
        m = self.target_session()
        m.select(im, b, 1.0)
        self.assertIsNotNone(m.update(im, dict(arrival=1.0, detections=[self.detection(b)])))
        self.assertIsNone(m.update(np.zeros_like(im) + 25, dict(arrival=1.04, detections=[])))
        self.assertEqual(m.state, "recovering")
        return m, im, b

    def test_clipped_automatic_acquisition_handoff_keeps_original_extent(self):
        from bucket_robot.perception.target import TargetLock

        image, box = self.scene(box=(20, 456, 398, 720))
        for x in range(40, 380, 45):
            cv2.circle(image, (x, 530), 14, (10, 30, 220), -1)
            cv2.rectangle(image, (x, 620), (x + 16, 650), (200, 10, 40), -1)
        auto = TargetLock("auto")
        for i in range(12):
            meta = dict(arrival=1 + i * 0.04, detections=[self.detection(box)])
            acquired = auto.update(image, meta)
            if acquired is not None:
                break
        self.assertIsNotNone(acquired, auto.reason)
        self.assertEqual(acquired["raw_box"], box)
        mission = self.target_session()
        mission.select(image, acquired["raw_box"], meta["arrival"])
        target = mission.update(image, meta)
        self.assertIsNotNone(target, mission.reason)
        self.assertTrue(target["vertical_clipped"])
        self.assertTrue(target["horizontal_bearing_valid"])
        self.assertEqual(target["task_target_id"], 1)

    def test_first_update_loss_recovers_selected_identity_without_new_selection(self):
        for case in ("same", "wrong", "ambiguous", "duplicate"):
            image, box = self.scene(box=(20, 456, 398, 720))
            mission = self.target_session()
            mission.select(image, box, 1)
            self.assertIsNone(mission.update(image, dict(arrival=1.04, detections=[])))
            self.assertIsNotNone(mission.frozen)
            if case == "wrong":
                image, _ = self.scene(seed=111, box=box)
            targets = []
            for i in range(20):
                ds = [self.detection(box)]
                if case == "ambiguous":
                    ds.append(self.detection([22, 456, 400, 720]))
                target = mission.update(
                    image,
                    dict(arrival=1.2 if case == "duplicate" else 1.2 + i * 0.04, detections=ds),
                )
                if target is not None:
                    targets.append(target)
            if case == "same":
                self.assertTrue(targets, mission.reason)
                self.assertTrue(all(t["task_target_id"] == 1 for t in targets))
                self.assertEqual(mission.reacquired, 1)
            else:
                self.assertFalse(targets, case)

    def test_camera_rotation_relocation_requires_known_geometry_and_fresh_votes(self):
        for case in ("same", "wrong", "ambiguous", "duplicate"):
            seed, box = self.scene(box=(40, 300, 340, 600))
            mission = self.target_session()
            mission.select(seed, box, 1)
            mission.update(seed, dict(arrival=1, detections=[self.detection(box)]))
            mission.missing("camera_rotation_gap")
            image, moved = self.scene(seed=111 if case == "wrong" else 15, box=(500, 300, 800, 600))
            targets = []
            for i in range(16):
                ds = [self.detection(moved)]
                if case == "ambiguous":
                    ds.append(self.detection([502, 300, 802, 600]))
                target = mission.update(
                    image,
                    dict(arrival=1.2 if case == "duplicate" else 1.2 + i * 0.04, detections=ds),
                )
                if target is not None:
                    targets.append(target)
            if case == "same":
                self.assertTrue(targets, mission.reason)
                self.assertEqual(mission.reacquired, 1)
                self.assertTrue(all(t["task_target_id"] == 1 for t in targets))
                self.assertTrue(
                    targets[0].get("relocation_evidence")
                    or targets[0].get("appearance_recovery_evidence")
                )
            else:
                self.assertFalse(targets, case)

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
        m = self.target_session()
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

    def test_known_appearance_recovers_far_from_old_box_without_ORB_or_new_identity(self):
        previous, old = self.scene(box=(40, 300, 340, 600))
        manager = self.target_session()
        manager.select(previous, old, 1)
        manager.update(previous, dict(arrival=1, detections=[self.detection(old)]))
        trusted = manager.manager.tracker
        manager.missing("initial_camera_turn")
        self.assertIs(manager.frozen, trusted)
        current = np.full_like(previous, 25)
        box = [600, 320, 840, 560]
        current[320:560, 600:840] = cv2.resize(previous[300:600, 40:340], (240, 240))
        answer = None
        with patch(
            "bucket_robot.perception.relocation.known_feature_relocation", return_value=None
        ) as slow_search:
            for i in range(12):
                answer = manager.update(
                    current, dict(arrival=1.2 + i * 0.04, detections=[self.detection(box)])
                )
                if answer is not None:
                    break
            slow_search.assert_not_called()
        self.assertIsNotNone(answer, manager.reason)
        self.assertEqual(answer["task_target_id"], 1)
        proof = answer["appearance_recovery_evidence"]
        self.assertTrue(proof["known_appearance_verified"])
        self.assertGreaterEqual(proof["candidate_detection_votes"], 2)
        self.assertLessEqual(proof["candidate_verify_s"], 0.4)

    def test_known_only_verifier_never_seeds_unknown_nearby_bucket(self):
        from bucket_robot.perception.target import TargetLock, TargetTracker

        old, box = self.scene()
        other, _ = self.scene(seed=921)
        verifier = TargetLock("manual")
        reference = TargetTracker(old, box, 1)
        for i in range(10):
            result = verifier.verified_reappearance(
                other,
                1.2 + i * 0.04,
                [self.detection(box)],
                reference=reference,
                require_known=True,
            )
            self.assertIsNone(result)
            self.assertIsNone(verifier.provisional)

    def test_stage_handoff_reverifies_shifted_same_bucket(self):
        seed, b = self.scene()
        m = self.target_session()
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
