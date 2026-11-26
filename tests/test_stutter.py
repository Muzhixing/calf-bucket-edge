import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from tests import test_tracking
from tests.test_tracking import MCU, Sender, Reference
from bucket_robot.perception.target import TargetTracker
from bucket_robot.runtime.journal import AsyncJournal
from bucket_robot.runtime.stationary import Session
from bucket_robot.motion.arm import ContinuousArmPort


class ContinuityTests(unittest.TestCase):
    def warm(self):
        im = test_tracking.TrackingTests.scene()
        box = [480, 250, 720, 450]
        d = dict(zip(("left", "top", "right", "bottom", "score"), [*box, 0.95]))
        t = TargetTracker(im, box, 1.0)
        t.update(im, 1.05, [d])
        t.update(im, 1.10, [d])
        return t, im, d

    def test_fresh_detection_can_bridge_old_template_rejection(self):
        t, im, d = self.warm()
        anchor = t.anchor.copy()
        # Keep real pixels and real successive detector-view checks; only the
        # old alignment scores are forced low, as observed under view changes.
        original = t.alignment

        def changed(*args):
            box, aligned, _, _ = original(*args)
            return box, aligned, 0.3, 0.3

        with (
            patch.object(t, "flow_transform", side_effect=RuntimeError("target_lost_geometry")),
            patch.object(t, "alignment", side_effect=changed),
        ):
            result = t.update(im, 1.15, [d])
        self.assertEqual(result["method"], "detector_continuity")
        self.assertEqual(result["timestamp"], 1.15)
        np.testing.assert_array_equal(t.anchor, anchor)
        self.assertEqual(len(t.references), 1)

    def test_continuity_rejects_occlusion_wrong_view_and_ambiguous_boxes(self):
        for kind in ("blank", "wrong", "ambiguous", "weak", "missing", "stale"):
            with self.subTest(kind=kind):
                t, im, d = self.warm()
                ds = [d]
                stamp = 1.15
                if kind == "blank":
                    im = np.zeros_like(im) + 25
                if kind == "wrong":
                    im[250:450, 480:720] = np.random.default_rng(81).integers(
                        0, 255, (200, 240, 3), dtype=np.uint8
                    )
                if kind == "ambiguous":
                    ds.append({**d, "left": 500, "right": 740})
                if kind == "weak":
                    ds = [{**d, "score": 0.7}]
                if kind == "missing":
                    ds = []
                if kind == "stale":
                    stamp = 1.25
                detection = t.associated_detection(ds, t.box)
                self.assertFalse(t.measured_continuity(im, detection, stamp))
                if kind in ("blank", "wrong", "ambiguous", "missing"):
                    with (
                        patch.object(t, "flow_transform", side_effect=RuntimeError("lost")),
                        patch.object(t, "local_search", side_effect=RuntimeError("lost")),
                    ):
                        with self.assertRaises(RuntimeError):
                            t.update(im, stamp, ds)
                    self.assertEqual(t.stamp, 1.10)


class JournalTests(unittest.TestCase):
    def test_slow_writer_is_bounded_and_preserves_critical_events(self):
        with tempfile.TemporaryDirectory() as folder:
            entered = threading.Event()
            release = threading.Event()
            lines = []

            def writer(line):
                entered.set()
                release.wait(3)
                lines.append(json.loads(line))

            j = AsyncJournal(folder, sample_limit=2, limit=5, writer=writer)
            try:
                j.append("serial_rx", [])
                self.assertTrue(entered.wait(1))
                start = time.monotonic()
                for _ in range(100):
                    j.append("tracking_sample", {})
                j.append("disabled_confirmed", {})
                self.assertLess(time.monotonic() - start, 0.1)
                self.assertGreater(j.dropped_samples, 0)
                self.assertLessEqual(len(j.jobs), 5)
            finally:
                release.set()
                self.assertTrue(j.close())
            self.assertIn("disabled_confirmed", [x["event"] for x in lines])

    def test_blocked_disk_does_not_block_serial_or_occlusion_stop(self):
        ref = Reference()
        mcu = MCU()
        release = threading.Event()
        entered = threading.Event()

        def writer(line):
            entered.set()
            release.wait(5)

        j = AsyncJournal(ref.directory.name, writer=writer)

        def factory(limits, log):
            return ContinuousArmPort(
                limits, log, port=mcu.path, sender_factory=Sender, check_holders=False
            )

        s = Session(
            ref.directory.name,
            ref.limits,
            smooth=True,
            automatic=True,
            intrinsics=dict(fx=700.0, fy=700.0, cx=640.0, cy=360.0),
            port_factory=factory,
        )
        s.journal = j
        worker = threading.Thread(target=s.arm_loop)
        worker.start()

        def feed(seconds, visible):
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                with s.lock:
                    s.metadata = {"arrival": time.monotonic()}
                    s.target_id = 1
                    s.vision_state = "tracking" if visible else "reacquiring"
                    s.tracking = (
                        {
                            "box": [400, 200, 650, 450],
                            "offset_px": [60, 0],
                            "timestamp": time.monotonic(),
                            "method": "flow",
                        }
                        if visible
                        else None
                    )
                time.sleep(0.02)

        try:
            feed(2.3, True)
            self.assertTrue(entered.is_set())
            self.assertTrue(s.active, s.reason)
            self.assertTrue(any(f[8] for f in mcu.commands))
            self.assertFalse(release.is_set())
            feed(0.5, False)
            self.assertFalse(mcu.enabled)
            self.assertTrue(s.active, s.reason)
            self.assertEqual(s.arm["telemetry"]["system_fault"], 0)
        finally:
            s.stop_event.set()
            worker.join(4)
            release.set()
            j.close()
            mcu.close()
            ref.directory.cleanup()


if __name__ == "__main__":
    unittest.main()
