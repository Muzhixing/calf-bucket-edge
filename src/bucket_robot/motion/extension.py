"""Bounded stationary C transaction using the shared arm acknowledgment state machine."""

import time

from bucket_robot.motion.arm import ArmPort, pose


class ExtensionPort(ArmPort):
    """Stationary B return and C extension using the shared arm transaction."""

    def extend(self):
        if self.phase != "idle" or self.state["enabled"]:
            raise RuntimeError("extension_requires_disabled_chassis")
        self.check_state(self.state)
        current = pose(self.state)
        if not 131 <= current[0] <= 132 or current[2] != 0:
            raise RuntimeError("extension_pose_not_ready")
        q = [current[0], current[1], 150]
        self.limits.check(q)
        if time.monotonic() - self.rx_at > 0.35:
            raise RuntimeError("stale_telemetry")
        self.origin, self.target = current, q
        self.arrived_at = None
        self.move_counter = (self.expected + 1) % 256
        self.phase, self.attempted = "moving", True
        self.started = time.monotonic()
        self.deadline = self.started + 150 / self.limits.c_max_speed + 4
        if not self.gateway.write_control_frame(self.frame(q, True), repeats=1):
            raise RuntimeError("extension_write_failed_no_retry")
