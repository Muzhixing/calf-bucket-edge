"""Exercise Linux capture on /dev/pts only; never use ttyUSB/ttyACM here."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import select
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import receive
from test_receive import frame


@unittest.skipUnless(sys.platform == "linux", "Linux PTY test")
class LinuxCaptureTests(unittest.TestCase):
    def run_pty(self, payload):
        import pty
        master, slave = pty.openpty()
        port = os.ttyname(slave)
        self.assertTrue(port.startswith("/dev/pts/"))
        os.close(slave)
        errors = []

        def producer():
            try:
                time.sleep(0.15)  # receiver has configured raw mode
                os.write(master, payload[:7])
                time.sleep(0.01)
                os.write(master, payload[7:])
                time.sleep(0.25)
                os.write(master, payload * 2)
            except OSError as exc:
                errors.append(repr(exc))

        def synthetic_holders(device):
            # Other users' /proc descriptors may be hidden. Bypass only for this
            # newly created, controlled PTY, never for a physical device.
            self.assertEqual(str(device), port)
            self.assertTrue(str(device).startswith("/dev/pts/"))

        try:
            with tempfile.TemporaryDirectory() as temporary:
                target = Path(temporary) / "run"
                args = argparse.Namespace(port=port, seconds=0.8, out=str(target),
                                          site_confirmation="synthetic PTY only; no hardware")
                writer = threading.Thread(target=producer)
                writer.start()
                real_open, calls = receive.os.open, []

                def tracked_open(path, flags, *rest, **kwargs):
                    calls.append((str(path), flags))
                    return real_open(path, flags, *rest, **kwargs)

                with patch.object(receive, "holders", synthetic_holders), patch.object(receive.os, "open", tracked_open), contextlib.redirect_stdout(io.StringIO()):
                    code = receive.capture(args)
                writer.join(timeout=2)
                result = json.loads((target / "summary.json").read_text())
                raw = (target / "raw.bin").read_bytes()
                self.assertTrue(any(p == port and flags & os.O_ACCMODE == os.O_RDONLY for p, flags in calls))
                self.assertFalse(errors)
                # Master input receives only bytes sent by the slave. EIO here
                # means slave closed with no queued output, not transmitted data.
                outgoing = b""
                if select.select([master], [], [], 0)[0]:
                    try:
                        outgoing = os.read(master, 4096)
                    except OSError as exc:
                        if exc.errno != 5:
                            raise
                self.assertEqual(outgoing, b"")
                return code, result, raw
        finally:
            os.close(master)

    def test_linux_receive_only_end_to_end(self):
        code, result, raw = self.run_pty(frame())
        self.assertEqual(code, 0)
        self.assertEqual(raw, frame() * 3)
        self.assertEqual(result["valid_frames"], 3)
        self.assertEqual(result["coalesced_intervals"], 1)
        self.assertEqual(result["serial_write_calls"], 0)

    def test_enabled_report_stops_without_sending(self):
        code, result, raw = self.run_pty(frame(**{"5": 1}))
        self.assertEqual(code, 2)
        self.assertEqual(result["enabled_frames"], 1)
        self.assertEqual(result["stop_reason"], "enabled_or_motion_reported_site_check_required")
        self.assertEqual(raw, frame(**{"5": 1}))


if __name__ == "__main__":
    unittest.main()
