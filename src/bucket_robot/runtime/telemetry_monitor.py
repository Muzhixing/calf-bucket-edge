"""Passive telemetry only. Never sends a control/zero/test-motion frame."""

import fcntl
import json
import os
from pathlib import Path
import select
import termios
import time
from bucket_robot.transport.telemetry import Parser
from bucket_robot.runtime.supervision import Backoff


class TelemetryMonitor:
    def __init__(self, port, health, clock=time.monotonic):
        self.port, self.health, self.clock = port, health, clock
        self.fd = None
        self.parser = Parser()
        self.backoff = Backoff()
        self.received = None

    def tick(self):
        if self.clock() < self.backoff.next_at:
            return
        try:
            if self.fd is None:
                self.fd = os.open(self.port, os.O_RDONLY | os.O_NOCTTY | os.O_NONBLOCK)
                fcntl.ioctl(self.fd, termios.TIOCEXCL)
                settings = termios.tcgetattr(self.fd)
                settings[0] = settings[1] = settings[3] = 0
                settings[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
                settings[4] = settings[5] = termios.B115200
                settings[6][termios.VMIN] = settings[6][termios.VTIME] = 0
                termios.tcsetattr(self.fd, termios.TCSANOW, settings)
                self.received = self.clock()
                self.parser = Parser()
            while select.select([self.fd], [], [], 0)[0]:
                data = os.read(self.fd, 4096)
                if not data:
                    raise OSError("serial disconnected")
                frames = self.parser.feed(data)
                if frames:
                    self.received = self.clock()
                    self.health.telemetry(frames)
                    self.backoff.reset()
            if self.clock() - self.received > 0.7:
                raise OSError("no fresh telemetry")
        except (OSError, ValueError, termios.error):
            self.close()
            self.health.disconnected()
            self.backoff.failed(self.clock())

    def close(self):
        if self.fd is not None:
            try:
                fcntl.ioctl(self.fd, termios.TIOCNXCL)
            finally:
                os.close(self.fd)
                self.fd = None


class TaskTelemetryReader:
    def __init__(self, path, health):
        self.path, self.health = Path(path), health
        self.offset = 0
        self.partial = b""

    def tick(self):
        try:
            with self.path.open("rb") as stream:
                stream.seek(self.offset)
                data = stream.read(1_048_576)
                self.offset = stream.tell()
            lines = (self.partial + data).split(b"\n")
            self.partial = lines.pop()[-65536:]
            for line in lines:
                try:
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        continue
                    if (
                        item.get("kind") == "serial_rx"
                        and 0 <= time.monotonic() - item.get("monotonic", 0) <= 0.7
                        and isinstance(item.get("data"), list)
                        and all(isinstance(frame, dict) for frame in item["data"])
                    ):
                        self.health.telemetry(item["data"])
                    if item.get("kind") == "stop_unconfirmed":
                        self.health.fault("STOP_UNCONFIRMED", "停止未确认")
                except (ValueError, KeyError, TypeError):
                    continue
        except OSError:
            pass
