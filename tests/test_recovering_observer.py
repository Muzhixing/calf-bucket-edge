import json, tempfile, time, unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from bucket_robot.missions.forward import Observer


class RecoveringObserverTests(unittest.TestCase):
    def test_automatic_acquisition_waits_then_locks_without_manual_seed(self):
        image = np.zeros((720, 1280, 3), np.uint8) + 25
        image[300:600, 480:780] = cv2.GaussianBlur(
            np.random.default_rng(25).integers(0, 256, (300, 300, 3), dtype=np.uint8), (5, 5), 1
        )
        d = dict(left=480, top=300, right=780, bottom=600, score=0.95)

        class Capture:
            count = 0

            def __init__(self, port):
                pass

            def read(self):
                time.sleep(0.025)
                Capture.count += 1
                return image, dict(
                    arrival=time.monotonic(),
                    frame_id=Capture.count,
                    detections=[] if Capture.count < 8 else [d],
                )

        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            observer = Observer(8899, out, None, None, True, True)
            try:
                with patch("bucket_robot.perception.target.NativeCapture", Capture):
                    observer.thread.start()
                    until = time.monotonic() + 4
                    while time.monotonic() < until:
                        obs, error = observer.get()
                        if error or obs and obs["target"] is not None:
                            break
                        time.sleep(0.01)
                    self.assertIsNone(error)
                    self.assertIsNotNone(obs["target"])
                    self.assertEqual(obs["identity"], 1)
                    self.assertTrue((out / "automatic_target.json").exists())
                    observer.end.set()
                    observer.thread.join(timeout=1)
                rows = [json.loads(s) for s in (out / "vision.jsonl").read_text().splitlines()]
                self.assertTrue(any(r["identity"] is None and r["target"] is None for r in rows))
            finally:
                observer.end.set()
                observer.thread.join(timeout=1)

    def test_camera_timeout_and_visual_occlusion_do_not_kill_observer(self):
        image = np.zeros((720, 1280, 3), np.uint8) + 25
        box = [480, 300, 780, 600]
        texture = np.random.default_rng(25).integers(0, 256, (300, 300, 3), dtype=np.uint8)
        image[300:600, 480:780] = cv2.GaussianBlur(texture, (5, 5), 1)
        detection = dict(left=480, top=300, right=780, bottom=600, score=0.95)
        for case in ("camera_timeout", "occlusion"):

            class Capture:
                count = 0

                def __init__(self, port):
                    pass

                def read(self):
                    time.sleep(0.02)
                    Capture.count += 1
                    n = Capture.count
                    if case == "camera_timeout" and n == 10:
                        raise OSError("simulated_camera_disconnect")
                    lost = case == "occlusion" and 10 <= n <= 14
                    return (np.zeros_like(image) + 25 if lost else image), dict(
                        arrival=time.monotonic(), frame_id=n, detections=[] if lost else [detection]
                    )

            with tempfile.TemporaryDirectory() as folder:
                out = Path(folder)
                seed = out / "seed.png"
                cv2.imwrite(str(seed), image)
                observer = Observer(8899, out, str(seed), box, True, True)
                try:
                    with patch("bucket_robot.perception.target.NativeCapture", Capture):
                        observer.thread.start()
                        until = time.monotonic() + 4
                        while time.monotonic() < until:
                            obs, error = observer.get()
                            if (
                                error
                                or obs
                                and obs.get("recovery_count", 0) >= 1
                                and obs["target"] is not None
                            ):
                                break
                            time.sleep(0.01)
                        self.assertIsNone(error, case)
                        self.assertIsNotNone(obs["target"], case)
                        self.assertEqual(obs["identity"], 1)
                        self.assertGreaterEqual(obs["recovery_count"], 1)
                        self.assertTrue(observer.thread.is_alive())
                        observer.end.set()
                        observer.thread.join(timeout=1)
                    rows = [json.loads(x) for x in (out / "vision.jsonl").read_text().splitlines()]
                    self.assertTrue(any(r["target"] is None for r in rows))
                finally:
                    observer.end.set()
                    observer.thread.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
