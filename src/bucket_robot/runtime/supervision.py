"""Bounded restart policy for visual workers. Never restart motion implicitly."""

import subprocess
import time


class Backoff:
    def __init__(self, initial=0.5, maximum=8):
        self.initial, self.maximum = initial, maximum
        self.attempt = 0
        self.next_at = 0

    def failed(self, now):
        self.next_at = now + min(self.maximum, self.initial * 2 ** min(self.attempt, 8))
        self.attempt += 1

    def reset(self):
        self.attempt = 0
        self.next_at = 0


class VisualSupervisor:
    def __init__(
        self,
        command,
        health,
        stop_event,
        *,
        popen=subprocess.Popen,
        clock=time.monotonic,
        restart_after=lambda: None,
    ):
        self.command, self.health, self.stop_event = command, health, stop_event
        self.popen, self.clock = popen, clock
        self.process = None
        self.backoff = Backoff()
        self.last_frame = None
        self.started = 0
        self.attempt = 0
        self.restart_after = restart_after

    def observed_frame(self):
        self.last_frame = self.clock()

    def tick(self):
        now = self.clock()
        if self.process is not None:
            exited = self.process.poll() is not None
            hung = now - (self.last_frame if self.last_frame is not None else self.started) > 10
            limit = self.restart_after()
            recycle = limit is not None and now - self.started > limit
            if not exited and not hung and not recycle:
                return
            if hung or recycle:
                self.close()
            self.process = None
            self.health.report(
                "backend",
                "recovering",
                "BACKEND_EXITED" if exited else "BACKEND_STALLED",
                "视觉后端退出，正在重启" if exited else "视觉后端长时间无画面，正在重启",
            )
            self.health.fault("VISION_INTERRUPTED", "视觉后端中断，运动保持暂停")
            self.backoff.failed(now)
        if self.stop_event.is_set() or now < self.backoff.next_at:
            return
        try:
            self.attempt += 1
            self.process = self.popen(self.command(self.attempt), start_new_session=True)
            self.started = now
            self.last_frame = None
            self.health.report("backend", "starting", "BACKEND_STARTING", "视觉后端启动中")
        except Exception as exc:
            self.health.report("backend", "unavailable", "BACKEND_START_FAILED", str(exc))
            self.backoff.failed(now)

    def run(self):
        try:
            while not self.stop_event.is_set():
                self.tick()
                self.stop_event.wait(0.1)
        finally:
            self.close()

    def close(self):
        if self.process is None:
            return
        process = self.process
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        self.process = None
