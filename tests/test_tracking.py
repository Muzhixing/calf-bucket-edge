import json
import os
from pathlib import Path
import pty
import select
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from bucket_robot.motion.arm import ArmPort, Limits, plan, pose, should_interrupt
from bucket_robot.runtime.stationary import Session
from bucket_robot.perception.target import Acquirer, TargetTracker, TargetLock, valid_box


class Reference:
    def __init__(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "reference.json"
        self.path.write_text(
            json.dumps(
                {
                    "operating_envelope": {"B_command_mm": [0, 96], "C_trial_hold_mm": 10},
                    "control_policy": {"original_sender_module_on_rk3588": "unused"},
                }
            )
        )
        self.limits = Limits(self.path)


class Sender:
    def __init__(self, port, serial_factory, command_repeats):
        assert command_repeats == 1
        self.factory = serial_factory
        self.port = port

    def _open_serial(self):
        self.serial = self.factory(self.port, 115200)

    def write_control_frame(self, frame, repeats):
        assert repeats == 1
        return self.serial.write(frame) == 10


class MCU:
    def __init__(self):
        self.master, self.slave = pty.openpty()
        self.path = os.ttyname(self.slave)
        os.set_blocking(self.master, False)
        self.commands = []
        self.count = 0
        self.enabled = False
        self.b = 40.0
        self.target = 40.0
        self.a = 78
        self.c = 10
        self.drop = False
        self.speed = 100
        self.done = threading.Event()
        self.buffer = b""
        self.thread = threading.Thread(target=self.loop)
        self.thread.start()

    def frame(self):
        f = bytearray(
            [
                0xB5,
                1,
                self.count,
                0,
                2 if self.enabled and self.b != self.target else 1 if self.enabled else 0,
                46 | int(self.enabled),
            ]
        )
        f += int(round(self.b * 10)).to_bytes(2, "little") + int(self.c * 10).to_bytes(2, "little")
        f += (52).to_bytes(2, "little") + (65535).to_bytes(2, "little") + bytes([self.a, 0, 0, 0])
        checksum = 0
        for v in f[1:18]:
            checksum ^= v
        return f + bytes([checksum, 0xB6])

    def loop(self):
        last = time.monotonic()
        sent = 0
        while not self.done.wait(0.005):
            now = time.monotonic()
            dt = now - last
            last = now
            if select.select([self.master], [], [], 0)[0]:
                try:
                    self.buffer += os.read(self.master, 4096)
                except OSError:
                    continue
            while len(self.buffer) >= 10:
                if self.buffer[0] != 0xB3:
                    self.buffer = self.buffer[1:]
                    continue
                f = self.buffer[:10]
                self.buffer = self.buffer[10:]
                if f[-1] != 0xB4:
                    continue
                self.commands.append(f)
                self.count = (self.count + 1) % 256
                self.enabled = bool(f[8] & 1)
                if self.enabled:
                    self.a, self.target, self.c = f[4], float(f[5]), f[6]
            if self.enabled:
                change = max(-dt * self.speed, min(dt * self.speed, self.target - self.b))
                self.b += change
                if abs(self.target - self.b) < 0.001:
                    self.b = self.target
            if not self.drop and now - sent >= 0.1:
                try:
                    os.write(self.master, self.frame())
                except OSError:
                    pass
                sent = now

    def close(self):
        self.done.set()
        self.thread.join()
        os.close(self.master)
        os.close(self.slave)


class TrackingTests(unittest.TestCase):
    def test_stable_detection_and_identity(self):
        a = Acquirer()
        d = {"left": 400, "top": 250, "right": 600, "bottom": 500, "score": 0.8}
        self.assertIsNone(a.update([d], 0))
        self.assertIsNone(a.update([d], 0.2))
        self.assertEqual(a.update([d], 0.4), (400, 250, 600, 500))
        self.assertEqual(a.update([d], 0.6), (400, 250, 600, 500))
        self.assertIsNone(a.update([{**d, "left": 800, "right": 1000}], 0.7))
        self.assertFalse(valid_box([0, 20, 150, 200]))

    def test_intermittent_detection_window_and_isolated_false_positive(self):
        d = {"left": 400, "top": 250, "right": 600, "bottom": 500, "score": 0.61}
        a = Acquirer()
        for i in range(9):
            result = a.update([d] if i % 2 == 0 else [], i * 0.1)
            if i < 8:
                self.assertIsNone(result)
        self.assertEqual(result, (400, 250, 600, 500))
        b = Acquirer()
        self.assertIsNone(b.update([{**d, "score": 0.95}], 0))
        for i in range(1, 20):
            self.assertIsNone(b.update([], i * 0.1))
        self.assertIsNone(b.update([d], 2))

    def test_background_change_does_not_move_foreground_target(self):
        image = self.scene()
        tracker = TargetTracker(image, [450, 220, 750, 480], 1.0)
        changed = image.copy()
        changed[220:480, 450:750] = 170
        changed[250:450, 480:720] = image[250:450, 480:720]
        result = tracker.update(changed, 1.1)
        self.assertLess(abs(result["offset_px"][0] + 40), 3)
        self.assertLess(abs(result["offset_px"][1] + 10), 3)

    def test_detector_dropout_then_occlusion_reacquires_same_identity(self):
        image = self.scene()
        manager = TargetLock()
        manager.select(image, [480, 250, 720, 450], 1.0)
        generation = manager.generation
        for i in range(1, 6):
            moved = cv2.warpAffine(image, np.float32([[1, 0, i * 3], [0, 1, 0]]), (1280, 720))
            self.assertIsNotNone(manager.update(moved, {"arrival": 1 + i * 0.1, "detections": []}))
        self.assertIsNone(
            manager.update(np.zeros_like(image) + 25, {"arrival": 1.6, "detections": []})
        )
        self.assertEqual(manager.state, "reacquiring")
        moved = cv2.warpAffine(image, np.float32([[1, 0, 30], [0, 1, 0]]), (1280, 720))
        self.assertIsNone(manager.update(moved, {"arrival": 1.7, "detections": []}))
        self.assertIsNone(manager.update(moved, {"arrival": 1.86, "detections": []}))
        result = manager.update(moved, {"arrival": 2.02, "detections": []})
        self.assertIsNotNone(result)
        self.assertEqual(manager.generation, generation)
        self.assertEqual(manager.reacquired, 1)
        self.assertLess(abs(result["offset_px"][0] + 10), 3)

    def test_recovery_rejects_different_appearance_and_expires(self):
        image = self.scene()
        manager = TargetLock("manual")
        manager.select(image, [480, 250, 720, 450], 1.0)
        blank = np.zeros_like(image) + 25
        manager.update(blank, {"arrival": 1.1, "detections": []})
        different = image.copy()
        different[250:450, 480:720] = np.random.default_rng(3).integers(
            0, 255, (200, 240, 3), dtype=np.uint8
        )
        detections = [{"left": 480, "top": 250, "right": 720, "bottom": 450, "score": 0.99}]
        for stamp in (1.2, 1.3, 1.4, 2.0, 3.0):
            self.assertIsNone(
                manager.update(different, {"arrival": stamp, "detections": detections})
            )
        manager.update(blank, {"arrival": 6.2, "detections": []})
        self.assertTrue(manager.expired)
        self.assertEqual(manager.reason, "target_reacquire_timeout")

    def test_reference_bank_requires_independent_detection_support(self):
        image = self.scene()
        unverified = TargetTracker(image, [480, 250, 720, 450], 1.0)
        verified = TargetTracker(image, [480, 250, 720, 450], 1.0)
        for i in range(1, 14):
            dx = i * 2
            moved = cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, 0]]), (1280, 720))
            d = [{"left": 480 + dx, "top": 250, "right": 720 + dx, "bottom": 450, "score": 0.9}]
            unverified.update(moved, 1 + i * 0.1)
            verified.update(moved, 1 + i * 0.1, d)
        self.assertEqual(len(unverified.references), 1)
        self.assertGreater(len(verified.references), 1)
        np.testing.assert_array_equal(verified.references[0], verified.anchor)
        self.assertLessEqual(len(verified.references), 4)

    def test_horizontal_priority_at_vertical_limit(self):
        ref = Reference()
        self.addCleanup(ref.directory.cleanup)
        self.assertEqual(plan([77, 96, 10], [40, 150], ref.limits), [75, 96, 10])
        self.assertIsNone(plan([77, 96, 10], [10, 150], ref.limits))
        self.assertTrue(should_interrupt([77, 70, 10], [77, 80, 10], [80, 120], [77, 60, 10]))

    @staticmethod
    def scene():
        rng = np.random.default_rng(91)
        image = np.zeros((720, 1280, 3), np.uint8) + 25
        image[250:450, 480:720] = cv2.GaussianBlur(
            rng.integers(20, 240, (200, 240, 3), dtype=np.uint8), (3, 3), 0
        )
        return image

    def test_tracking_translation_without_detector(self):
        image = self.scene()
        tracker = TargetTracker(image, [480, 250, 720, 450], 1.0)
        for i in range(1, 9):
            moved = cv2.warpAffine(image, np.float32([[1, 0, 4 * i], [0, 1, 2 * i]]), (1280, 720))
            result = tracker.update(moved, 1.0 + i * 0.1)
            self.assertLess(abs(result["offset_px"][0] - (-40 + 4 * i)), 2)
            self.assertLess(abs(result["offset_px"][1] - (-10 + 2 * i)), 2)

    def test_local_appearance_bridges_flow_failure_but_rejects_loss(self):
        image = self.scene()
        tracker = TargetTracker(image, [480, 250, 720, 450], 1.0)

        def lost_flow(gray):
            raise RuntimeError("target_lost_geometry")

        tracker.flow_transform = lost_flow
        moved = cv2.warpAffine(image, np.float32([[1, 0, 24], [0, 1, 8]]), (1280, 720))
        result = tracker.update(moved, 1.1)
        self.assertEqual(result["method"], "appearance")
        self.assertLess(abs(result["offset_px"][0] + 16), 1)
        with self.assertRaises(RuntimeError):
            tracker.update(np.zeros_like(image) + 25, 1.2)

    def test_loss_and_distractor_do_not_switch_target(self):
        image = self.scene()
        for distractor in (False, True):
            tracker = TargetTracker(image, [480, 250, 720, 450], 1.0)
            lost = np.zeros_like(image) + 25
            if distractor:
                lost[250:450, 900:1140] = image[250:450, 480:720]
            with self.assertRaises(RuntimeError):
                tracker.update(lost, 1.1)

    def test_stale_frame_rejected(self):
        image = self.scene()
        tracker = TargetTracker(image, [480, 250, 720, 450], 1.0)
        with self.assertRaises(RuntimeError):
            tracker.update(image, 2.0)

    def test_detection_acquires_while_target_is_moving(self):
        acquirer = Acquirer()
        result = None
        for i in range(8):
            result = acquirer.update(
                [
                    {
                        "left": 400 + i * 8,
                        "top": 250,
                        "right": 600 + i * 8,
                        "bottom": 450,
                        "score": 0.9,
                    }
                ],
                i * 0.1,
            )
        self.assertEqual(result, (456, 250, 656, 450))

    def test_high_confidence_jitter_locks_without_settling_or_duplicate_votes(self):
        a = Acquirer()
        locked = None
        for i in range(8):
            dx = (i % 3 - 1) * 9 + i * 3
            dy = (i % 2) * 7
            d = {
                "left": 400 + dx,
                "top": 250 + dy,
                "right": 600 + dx,
                "bottom": 450 + dy,
                "score": 0.9,
            }
            result = a.update([d], 1 + i / 30)
            self.assertIsNone(a.update([d], 1 + i / 30))
            if result is not None and locked is None:
                locked = i / 30
        self.assertIsNotNone(locked)
        self.assertLessEqual(locked, 0.17)

    def test_auto_detector_relock_tracks_changing_view_before_old_timeout(self):
        image = self.scene()
        manager = TargetLock()
        manager.select(image, [480, 250, 720, 450], 1.0)
        manager.update(np.zeros_like(image) + 25, {"arrival": 1.1, "detections": []})
        changed = image.copy()
        changed[250:450, 480:720] = image[250:450, 480:720][::-1, ::-1]
        result = None
        for i in range(8):
            dx = i * 6
            dy = (i % 2) * 4
            moved = cv2.warpAffine(changed, np.float32([[1, 0, dx], [0, 1, dy]]), (1280, 720))
            d = {
                "left": 480 + dx,
                "top": 250 + dy,
                "right": 720 + dx,
                "bottom": 450 + dy,
                "score": 0.9,
            }
            result = manager.update(moved, {"arrival": 1.16 + i * 0.04, "detections": [d]})
            if result is not None:
                break
        self.assertIsNotNone(result)
        self.assertEqual(result["method"], "detector_flow_relock")
        self.assertEqual(manager.generation, 2)
        self.assertEqual(manager.reacquired, 0)
        self.assertLess(result["timestamp"] - 1.1, 0.5)

    def test_fast_relock_rejects_ambiguous_far_and_single_frame_detections(self):
        image = self.scene()
        blank = np.zeros_like(image) + 25
        near = {"left": 480, "top": 250, "right": 720, "bottom": 450, "score": 0.95}
        for kind in ("far", "ambiguous", "single"):
            manager = TargetLock()
            manager.select(image, [480, 250, 720, 450], 1.0)
            manager.update(blank, {"arrival": 1.1, "detections": []})
            for i in range(8):
                ds = (
                    [{**near, "left": 950, "right": 1190}]
                    if kind == "far"
                    else [near, {**near, "left": 550, "right": 790}]
                    if kind == "ambiguous"
                    else [near]
                    if i == 0
                    else []
                )
                self.assertIsNone(
                    manager.update(blank, {"arrival": 1.2 + i * 0.04, "detections": ds}), kind
                )
            self.assertEqual(manager.generation, 1)

    def test_medium_confidence_changed_appearance_relock_requires_five_votes(self):
        image = self.scene()
        manager = TargetLock()
        manager.select(image, [480, 250, 720, 450], 1.0)
        manager.update(np.zeros_like(image) + 25, {"arrival": 1.1, "detections": []})
        # No saved-appearance support: the detector-only path still needs five votes.
        image = image.copy()
        image[250:450, 480:720] = image[250:450, 480:720][::-1, ::-1]
        for i in range(5):
            d = {"left": 480, "top": 250, "right": 720, "bottom": 450, "score": 0.72}
            # Geometry cannot independently confirm this candidate, so exercise
            # the unchanged detector-only fallback.
            with patch.object(
                TargetTracker, "flow_transform", side_effect=RuntimeError("target_lost_flow")
            ):
                result = manager.update(image, {"arrival": 1.2 + i * 0.08, "detections": [d]})
            if i < 4:
                self.assertIsNone(result)
        self.assertIsNotNone(result)
        self.assertEqual(result["method"], "detector_relock")

    def test_replan_uses_live_error_not_error_growth(self):
        self.assertFalse(should_interrupt([78, 45, 10], [78, 60, 10], [0, 90]))
        self.assertTrue(should_interrupt([78, 45, 10], [78, 60, 10], [0, -50]))
        self.assertTrue(should_interrupt([78, 45, 10], [78, 60, 10], [0, 0]))
        self.assertFalse(should_interrupt([76, 79.1, 10], [76, 79, 10], [-31, 140], [76, 79.1, 10]))

    def test_policy_limits_and_fixed_C(self):
        ref = Reference()
        limits = ref.limits
        rng = np.random.default_rng(21)
        self.addCleanup(ref.directory.cleanup)
        for _ in range(400):
            q = [int(rng.integers(70, 101)), int(rng.integers(0, 97)), 10]
            e = rng.uniform([-400, -250], [400, 250]).tolist()
            target = plan(q, e, limits)
            if target is not None:
                limits.check(target)
                self.assertEqual(target[2], 10)
                self.assertLessEqual(abs(target[0] - q[0]), 4)
                self.assertLessEqual(abs(target[1] - q[1]), 20)
                self.assertEqual(sum(a != b for a, b in zip(q, target)), 1)
        self.assertIsNone(plan([78, 96, 10], [7, -5], limits))
        with self.assertRaises(RuntimeError):
            limits.check([78, 97, 10])


class SerialTests(unittest.TestCase):
    def setUp(self):
        self.ref = Reference()
        self.mcu = MCU()
        self.events = []
        self.port = ArmPort(
            self.ref.limits,
            lambda k, v: self.events.append((k, v)),
            port=self.mcu.path,
            sender_factory=Sender,
            check_holders=False,
        )

    def tearDown(self):
        if self.port.fd is not None:
            self.port.shutdown()
        self.mcu.close()
        self.ref.directory.cleanup()

    def test_move_then_confirm_disable(self):
        self.port.move([78, 60, 10])
        until = time.monotonic() + 5
        while self.port.phase != "idle" and time.monotonic() < until:
            self.port.tick()
            time.sleep(0.01)
        self.assertEqual(self.port.phase, "idle")
        self.assertEqual(pose(self.port.state), [78, 60, 10])
        self.assertFalse(self.port.state["enabled"])
        self.assertEqual([f[8] for f in self.mcu.commands], [1, 0])
        self.assertTrue(any(e[0] == "disabled_confirmed" for e in self.events))

    def test_outside_range_and_C_change_never_transmit(self):
        for q in ([78, 97, 10], [78, 40, 11], [78, 70, 10]):
            with self.assertRaises(RuntimeError):
                self.port.move(q)
        self.assertEqual(self.mcu.commands, [])

    def test_telemetry_loss_attempts_stop_but_does_not_claim_confirmation(self):
        self.port.move([78, 60, 10])
        self.mcu.drop = True
        time.sleep(0.8)
        with self.assertRaises(RuntimeError):
            self.port.tick()
        error = self.port.shutdown()
        self.assertIsNotNone(error)
        self.assertEqual(self.mcu.commands[-1][8], 0)
        self.assertFalse(any(k == "disabled_confirmed" for k, v in self.events))


class SessionTests(unittest.TestCase):
    def test_short_visual_loss_stops_then_same_identity_resumes(self):
        ref = Reference()
        mcu = MCU()
        mcu.speed = 10

        def factory(limits, log):
            return ArmPort(limits, log, port=mcu.path, sender_factory=Sender, check_holders=False)

        s = Session(ref.directory.name, ref.limits, port_factory=factory)
        s.active = True
        s.lease = time.monotonic()
        s.locked_at = 0
        s.target_id = 1
        s.vision_state = "tracking"
        worker = threading.Thread(target=s.arm_loop)
        worker.start()
        try:
            until = time.monotonic() + 5
            while not mcu.commands and time.monotonic() < until:
                with s.lock:
                    s.lease = time.monotonic()
                    s.tracking = {"offset_px": [0, 60], "timestamp": time.monotonic()}
                time.sleep(0.02)
            self.assertTrue(mcu.commands)
            with s.lock:
                s.vision_state = "reacquiring"
                s.tracking = None
            until = time.monotonic() + 2
            while s.arm["phase"] != "reacquiring" and time.monotonic() < until:
                with s.lock:
                    s.lease = time.monotonic()
                time.sleep(0.02)
            self.assertTrue(s.active, s.reason)
            self.assertFalse(mcu.enabled)
            self.assertEqual(s.arm["phase"], "reacquiring")
            count = len(mcu.commands)
            time.sleep(0.15)
            self.assertEqual(len(mcu.commands), count)
            until = time.monotonic() + 4
            while time.monotonic() < until:
                with s.lock:
                    s.vision_state = "tracking"
                    s.lease = time.monotonic()
                    s.tracking = {"offset_px": [0, -60], "timestamp": time.monotonic()}
                if any(f[8] == 1 and f[5] < 40 for f in mcu.commands):
                    break
                time.sleep(0.02)
            self.assertTrue(any(f[8] == 1 and f[5] < 40 for f in mcu.commands), mcu.commands)
            self.assertTrue(s.active, s.reason)
            self.assertEqual(s.target_id, 1)
            s.command("stop", {})
            until = time.monotonic() + 2
            while s.arm["phase"] != "disabled" and time.monotonic() < until:
                time.sleep(0.02)
            self.assertFalse(s.active)
            self.assertFalse(mcu.enabled)
        finally:
            s.stop_event.set()
            worker.join(timeout=4)
            mcu.close()
            ref.directory.cleanup()

    def test_slow_recovery_waits_for_fresh_frame_before_publication(self):
        ref = Reference()
        self.addCleanup(ref.directory.cleanup)
        s = Session(ref.directory.name, ref.limits)
        image = TrackingTests.scene()
        owner = self
        s.active = True
        s.vision_state = "reacquiring"

        class Manager:
            def __init__(self, *args):
                self.reacquired = 1
                self.generation = 0
                self.expired = False
                self.state = "tracking"
                self.reason = ""
                self.tracker = None
                self.provisional = None

            def update(self, image, metadata):
                return {
                    "box": [480, 250, 720, 450],
                    "offset_px": [-40, -10],
                    "timestamp": metadata["arrival"],
                }

        class Capture:
            count = 0

            def read(self):
                self.count += 1
                if self.count == 2:
                    owner.assertIsNone(s.tracking)
                    owner.assertEqual(s.vision_state, "reacquiring")
                    owner.assertTrue(s.active)
                    s.stop_event.set()
                return image, {
                    "arrival": time.monotonic() - (0.5 if self.count == 1 else 0),
                    "frame_id": self.count,
                }

        with patch("bucket_robot.runtime.stationary.TargetLock", Manager):
            s.vision_loop(Capture())
        self.assertIsNotNone(s.tracking)
        self.assertTrue(s.active)
        self.assertEqual(s.reacquired_count, 1)
        self.assertLess(time.monotonic() - s.tracking["timestamp"], 0.35)

    def test_capture_startup_failure_recovers_without_arming(self):
        ref = Reference()
        self.addCleanup(ref.directory.cleanup)
        s = Session(ref.directory.name, ref.limits)
        image = TrackingTests.scene()

        class Capture:
            count = 0

            def read(self):
                self.count += 1
                if self.count <= 2:
                    raise OSError("camera_starting")
                if self.count >= 12:
                    s.stop_event.set()
                return image, {
                    "arrival": time.monotonic(),
                    "frame_id": self.count,
                    "detections": [
                        {"left": 480, "top": 250, "right": 720, "bottom": 450, "score": 0.9}
                    ],
                }

        capture = Capture()
        worker = threading.Thread(target=s.vision_loop, args=(capture,))
        worker.start()
        worker.join(timeout=5)
        s.stop_event.set()
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertGreaterEqual(capture.count, 12)
        self.assertIsNotNone(s.tracking)
        self.assertFalse(s.active)
        self.assertEqual(s.reason, "")

    def test_fractional_position_normalizes_then_advances(self):
        ref = Reference()
        mcu = MCU()
        mcu.b = mcu.target = 79.1

        def factory(limits, log):
            return ArmPort(limits, log, port=mcu.path, sender_factory=Sender, check_holders=False)

        s = Session(ref.directory.name, ref.limits, port_factory=factory)
        s.active = True
        s.lease = time.monotonic()
        s.locked_at = 0
        s.target_id = 1
        worker = threading.Thread(target=s.arm_loop)
        worker.start()
        try:
            until = time.monotonic() + 7
            while time.monotonic() < until:
                with s.lock:
                    s.lease = time.monotonic()
                    s.tracking = {"offset_px": [0, 140], "timestamp": time.monotonic()}
                if any(f[8] == 1 and f[5] > 80 for f in mcu.commands):
                    break
                time.sleep(0.02)
            enabled = [f for f in mcu.commands if f[8] == 1]
            self.assertGreaterEqual(len(enabled), 2, mcu.commands)
            self.assertEqual(enabled[0][5], 79)
            self.assertGreater(enabled[1][5], 80)
            self.assertTrue(s.active, s.reason)
        finally:
            s.stop_event.set()
            worker.join(timeout=4)
            mcu.close()
            ref.directory.cleanup()

    def test_moving_target_reversal_replans_without_disarming(self):
        ref = Reference()
        mcu = MCU()
        mcu.speed = 10

        def factory(limits, log):
            return ArmPort(limits, log, port=mcu.path, sender_factory=Sender, check_holders=False)

        s = Session(ref.directory.name, ref.limits, port_factory=factory)
        s.active = True
        s.lease = time.monotonic()
        s.locked_at = 0
        s.target_id = 1
        worker = threading.Thread(target=s.arm_loop)
        worker.start()
        try:
            reversed_target = False
            until = time.monotonic() + 8
            while time.monotonic() < until:
                if mcu.commands:
                    reversed_target = True
                with s.lock:
                    s.lease = time.monotonic()
                    s.tracking = {
                        "offset_px": [0, -60 if reversed_target else 60],
                        "timestamp": time.monotonic(),
                    }
                if any(f[8] == 1 and f[5] < 40 for f in mcu.commands):
                    break
                time.sleep(0.02)
            self.assertTrue(reversed_target)
            self.assertTrue(any(f[8] == 0 for f in mcu.commands))
            self.assertTrue(any(f[8] == 1 and f[5] < 40 for f in mcu.commands), mcu.commands)
            self.assertTrue(s.active, s.reason)
            self.assertEqual(s.reason, "")
        finally:
            s.stop_event.set()
            worker.join(timeout=4)
            mcu.close()
            ref.directory.cleanup()

    def test_readonly_and_confirmation_gate(self):
        ref = Reference()
        s = Session(ref.directory.name, ref.limits, motion_allowed=False)
        self.addCleanup(ref.directory.cleanup)
        with self.assertRaises(ValueError):
            s.command("start", {"chassis_isolated": True})
        s.motion_allowed = True
        with self.assertRaises(ValueError):
            s.command("start", {"chassis_isolated": False})
        self.assertFalse(s.active)

    def test_operator_lease_expiry_prevents_serial_open(self):
        ref = Reference()
        self.addCleanup(ref.directory.cleanup)
        opened = []

        def factory(*args):
            opened.append(True)
            raise AssertionError("must not open")

        s = Session(ref.directory.name, ref.limits, port_factory=factory)
        s.active = True
        s.lease = time.monotonic() - 4
        worker = threading.Thread(target=s.arm_loop)
        worker.start()
        try:
            until = time.monotonic() + 1
            while s.active and time.monotonic() < until:
                time.sleep(0.01)
            self.assertFalse(s.active)
            self.assertEqual(opened, [])
            self.assertEqual(s.reason, "operator_connection_lost")
        finally:
            s.stop_event.set()
            worker.join(timeout=2)

    def test_target_loss_disables_and_does_not_rearm(self):
        ref = Reference()
        mcu = MCU()

        def factory(limits, log):
            return ArmPort(limits, log, port=mcu.path, sender_factory=Sender, check_holders=False)

        s = Session(ref.directory.name, ref.limits, port_factory=factory)
        s.active = True
        s.lease = time.monotonic()
        s.locked_at = 0
        s.target_id = 1
        worker = threading.Thread(target=s.arm_loop)
        worker.start()
        try:
            until = time.monotonic() + 5
            while not mcu.commands and time.monotonic() < until:
                with s.lock:
                    s.lease = time.monotonic()
                    s.tracking = {"offset_px": [0, 60], "timestamp": time.monotonic()}
                time.sleep(0.02)
            self.assertTrue(mcu.commands)
            with s.lock:
                s.tracking = None
            until = time.monotonic() + 3
            while (s.active or s.arm["phase"] != "disabled") and time.monotonic() < until:
                time.sleep(0.02)
            self.assertFalse(s.active)
            self.assertFalse(mcu.enabled)
            self.assertEqual([f[8] for f in mcu.commands], [1, 0])
        finally:
            s.stop_event.set()
            worker.join(timeout=4)
            mcu.close()
            ref.directory.cleanup()


if __name__ == "__main__":
    unittest.main()
