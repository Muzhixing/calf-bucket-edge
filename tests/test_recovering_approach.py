import json, sys, tempfile, threading, time, types, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import bucket_robot.missions.recovering_approach as task
from bucket_robot.missions.recovering_approach import LateralPulse
from tests.test_track_forward_probe import MCU, Sender


class LateralPulseTests(unittest.TestCase):
    def test_static_pixels_tolerate_box_jitter_but_reject_scene_motion(self):
        import cv2

        image = np.random.default_rng(2).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
        box = [346, 450, 934, 712]
        x1, y1, x2, y2 = box
        patch = cv2.resize(cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY), (96, 96))
        weights = np.ones((96, 96), np.float32)
        self.assertTrue(
            task.static_surface_matches(image, [355, 455, 930, 710], box, patch, weights)
        )
        self.assertFalse(
            task.static_surface_matches(np.roll(image, 8, axis=1), box, box, patch, weights)
        )
        self.assertFalse(
            task.static_surface_matches(image, [600, 450, 1000, 712], box, patch, weights)
        )

    def test_nominal_pulse_is_bounded(self):
        p = LateralPulse(1)
        self.assertEqual(p.step(1, True), 8)
        self.assertEqual(p.step(1.59, True), 8)
        self.assertEqual(p.step(1.61, True), 0)

    def test_loss_ramps_down_even_if_target_returns_mid_pulse(self):
        p = LateralPulse(1)
        self.assertEqual(p.step(1.2, False), 8)
        speeds = [p.step(1.2 + i * 0.1, True) for i in range(10)]
        self.assertEqual(speeds[:9], [8, 7, 6, 5, 4, 3, 2, 1, 0])
        self.assertEqual(speeds[-1], 0)

    def test_loss_at_normal_deadline_does_not_extend_motion(self):
        self.assertEqual(LateralPulse(1).step(1.7, False), 0)


class ApproachRecoverySessionTests(unittest.TestCase):
    def test_invalid_distance_retries_and_loss_ramps_down_before_remeasurement(self):
        class SideMCU(MCU):
            def packet(self):
                f = bytearray(super().packet())
                f[6:8] = (380).to_bytes(2, "little")
                if self.enabled and self.commands and self.commands[-1][2]:
                    f[4] = 3
                f[18] = 0
                for v in f[1:18]:
                    f[18] ^= v
                return bytes(f)

        m = SideMCU()
        m.a = 132
        started = time.monotonic()
        first_drive = [None]
        measurements = []
        y, x = np.mgrid[:720, :1280].astype(np.float32)
        image = np.random.default_rng(4).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
        raw = np.concatenate([image, image], axis=1)
        end = threading.Event()

        class FakeThread:
            def start(self):
                pass

            def join(self, timeout=None):
                pass

        class Observer:
            def __init__(self, *args):
                self.end = threading.Event()
                self.thread = FakeThread()

            def get(self):
                now = time.monotonic()
                if first_drive[0] is None and any(f[8] and f[2] > 0 for f in m.commands):
                    first_drive[0] = now
                lost = first_drive[0] is not None and 0.10 < now - first_drive[0] < 0.40
                fid = int((now - started) * 30)
                stamp = started + fid / 30
                return dict(
                    target=None
                    if lost
                    else dict(box=[480, 300, 780, 600], offset_px=[0, 0], timestamp=stamp),
                    identity=1,
                    frame_id=fid,
                    metadata=dict(arrival=stamp),
                    left=image,
                    raw=raw,
                )

        class Estimator:
            def __init__(self, *args):
                self.geometry = types.SimpleNamespace(maps=[(x, y), (x, y)])

            def measure(self, pair, box, point_roi=None):
                if m.enabled:
                    raise AssertionError("measurement while outputs enabled")
                if point_roi != [292, 267, 337, 292]:
                    raise AssertionError("verified wall band not applied")
                measurements.append(time.monotonic())
                time.sleep(0.02)
                if len(measurements) == 1:
                    return dict(valid=False)
                distance = 0.30 if len(measurements) == 2 else 0.20
                return dict(
                    valid=True,
                    distance_m=distance,
                    lower_m=distance - 0.005,
                    distance_kind="stereo_optical_midpoint_to_bucket_surface",
                )

            def close(self):
                pass

        actual = task.LateralPort

        def port(limits, log):
            return actual(limits, log, port=m.path, sender_factory=Sender, check_holders=False)

        fake_module = types.ModuleType("bucket_robot.perception.assisted_range.stereo")
        fake_module.AssistedRange = Estimator
        try:
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                out = root / "approach"
                ref = root / "ref.json"
                ref.write_text(
                    json.dumps({"control_policy": {"original_sender_module_on_rk3588": "unused"}})
                )

                def timeout():
                    if not end.wait(6):
                        out.mkdir(exist_ok=True)
                        (out / "STOP").touch()

                watchdog = threading.Thread(target=timeout, daemon=True)
                watchdog.start()
                with (
                    patch.dict(
                        sys.modules, {"bucket_robot.perception.assisted_range.stereo": fake_module}
                    ),
                    patch("bucket_robot.perception.observer.StereoObserver", Observer),
                    patch.object(task, "LateralPort", port),
                ):
                    result = task.run_recovering_approach(
                        out=out,
                        reference=ref,
                        calibration="unused",
                        model="unused",
                        seed_image="unused",
                        seed_box=[480, 300, 780, 600],
                        execute=True,
                        verified_surface_band=[0.35, 0.78, 0.65, 0.95],
                    )
                self.assertIsNone(result["error"])
                self.assertIsNone(result["stop_error"])
                self.assertEqual(result["reason"], "target_distance_reached")
                self.assertEqual(result["pulses"], 1)
                self.assertGreaterEqual(len(measurements), 3)
                speeds = [f[2] if f[8] else 0 for f in m.commands]
                self.assertTrue(any(0 < v < 8 for v in speeds))
                self.assertEqual(speeds[-1], 0)
                self.assertTrue(
                    all(f[1] == 0 and f[4:8] == bytes([132, 38, 0, 0]) for f in m.commands)
                )
                self.assertFalse(result["final_state"]["enabled"])
                end.set()
                watchdog.join(timeout=1)
        finally:
            end.set()
            m.close()


if __name__ == "__main__":
    unittest.main()
