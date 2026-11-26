"""Single-write B3/B4 adapter; ArmPort owns the exclusive file descriptor."""

import threading


class FrameWriter:
    """Write once. Partial writes fail and are never retried automatically."""

    def __init__(self, port, serial_factory, command_repeats=1):
        if command_repeats != 1:
            raise ValueError("single_write_required")
        self.port = port
        self.serial_factory = serial_factory
        self._serial = None
        self._write_lock = threading.Lock()

    def _open_serial(self):
        self._serial = self.serial_factory(self.port, 115200)

    def write_control_frame(self, frame, repeats=1):
        frame = bytes(frame)
        if len(frame) != 10 or frame[0] != 0xB3 or frame[-1] != 0xB4:
            raise ValueError("invalid_control_frame")
        if repeats != 1:
            raise ValueError("single_write_required")
        if self._serial is None:
            return False
        try:
            with self._write_lock:
                return self._serial.write(frame) == len(frame)
        except OSError:
            return False
