"""One-shot probe regression on a fresh PTY; never open physical serial."""
import contextlib
import io
import os
from pathlib import Path
import select
import tempfile
import threading
import time
import unittest

from disable_once import DISABLE, probe
from test_receive import frame


class DisableOnceTest(unittest.TestCase):
    def test_only_exact_disable_and_parse_response(self):
        import pty
        master, slave = pty.openpty()
        port = os.ttyname(slave)
        # Retain the PTY slave descriptor so the simulated MCU does not get
        # EIO before the receiver opens. This mock is never used for real ports.
        received, errors = bytearray(), []
        done = threading.Event()

        def simulated_mcu():
            try:
                while not done.is_set():
                    if select.select([master], [], [], 0.05)[0]:
                        received.extend(os.read(master, 1024))
                        if len(received) == 10:
                            os.write(master, frame(**{"2": 1, "5": 0}))
            except Exception as exc:
                errors.append(repr(exc))

        def check(path, allow_partial):
            self.assertEqual(path, port)
            self.assertTrue(path.startswith("/dev/pts/") or path.startswith("/dev/ttys"))
            return {"test_only": "fresh PTY"}

        worker = threading.Thread(target=simulated_mcu)
        worker.start()
        try:
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                result = probe(port, Path(tmp) / "run", "synthetic PTY only", before=0.15, after=0.3, occupancy_check=check)
            self.assertIsNone(result["error"])
            self.assertEqual(bytes(received), DISABLE)
            self.assertEqual(result["tx"]["write_calls"], 1)
            self.assertEqual(result["tx"]["bytes_accepted_by_host"], 10)
            self.assertEqual(result["valid_frames_after"], 1)
            self.assertFalse(errors)
        finally:
            done.set(); worker.join(timeout=2)
            os.close(slave); os.close(master)


if __name__ == "__main__":
    unittest.main()
