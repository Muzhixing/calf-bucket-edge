"""Wire-level and ownership contracts for the production serial adapter."""

import unittest
from bucket_robot.transport.writer import FrameWriter
from bucket_robot.motion.arm import ContinuousArmPort
from tests.test_tracking import Reference, MCU
import time


class FrameWriterTests(unittest.TestCase):
    def test_production_port_uses_packaged_writer_and_confirms_disable(self):
        reference = Reference()
        mcu = MCU()
        port = None
        try:
            port = ContinuousArmPort(
                reference.limits, lambda *args: None, port=mcu.path, check_holders=False
            )
            self.assertIsInstance(port.gateway, FrameWriter)
            self.assertTrue(port.follow([80, 40, 10]))
            until = time.monotonic() + 1
            while port.pending and time.monotonic() < until:
                port.tick()
                time.sleep(0.01)
            self.assertFalse(port.pending)
            self.assertIsNone(port.shutdown())
            self.assertFalse(port.state["enabled"])
        finally:
            if port is not None:
                port.shutdown()
            mcu.close()
            reference.directory.cleanup()

    def writer(self, result=10):
        class FD:
            def __init__(self):
                self.frames = []

            def write(self, data):
                self.frames.append(data)
                return result

        fd = FD()
        writer = FrameWriter("simulated", lambda *args: fd)
        writer._open_serial()
        return writer, fd

    def test_exact_original_frame_is_written_once(self):
        writer, fd = self.writer()
        frame = bytes.fromhex("B3 08 00 00 61 18 00 00 01 B4")
        self.assertTrue(writer.write_control_frame(frame))
        self.assertEqual(fd.frames, [frame])

    def test_partial_write_is_failure_without_retry(self):
        writer, fd = self.writer(4)
        self.assertFalse(writer.write_control_frame(bytes.fromhex("B3 00 00 00 61 18 00 00 00 B4")))
        self.assertEqual(len(fd.frames), 1)

    def test_invalid_frame_and_repeat_request_never_write(self):
        writer, fd = self.writer()
        for frame, repeats in [
            (b"bad", 1),
            (bytes(10), 1),
            (bytes.fromhex("B3 00 00 00 61 18 00 00 00 B4"), 2),
        ]:
            with self.assertRaises(ValueError):
                writer.write_control_frame(frame, repeats)
        self.assertEqual(fd.frames, [])
