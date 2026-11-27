"""Bounded heading recovery through the existing port and firmware yaw loop."""

import math

from bucket_robot.motion.arm import pose


def angle_delta(a, b):
    return (a - b + 180) % 360 - 180


class HeadingRecovery:
    """Pause translation, hold the original yaw target, then confirm disabled.

    No raw writer, new yaw baseline, arm movement, or fault clearing is used.
    The port still owns freshness, packet acknowledgments and all hard faults.
    """

    def __init__(self, reference, heading):
        self.reference = reference
        self.target = heading * 2
        self.phase = "idle"
        self.attempts = self.completed = 0
        self.error = 0.0
        self.last_sample = self.last_yaw = None
        self.started = self.correcting_since = None
        self.stable_since = self.stable_sample = None
        self.stable_frames = 0
        self.held_pose = None

    @property
    def active(self):
        return self.phase != "idle"

    def check(self, yaw, sample):
        if yaw is None or not math.isfinite(yaw):
            raise RuntimeError("heading_invalid")
        self.error = angle_delta(yaw, self.reference)
        if abs(self.error) >= 8:
            raise RuntimeError("heading_drift_hard_limit")
        if sample != self.last_sample:
            if self.last_yaw is not None and abs(angle_delta(yaw, self.last_yaw)) > 5:
                raise RuntimeError("heading_discontinuity")
            self.last_sample, self.last_yaw = sample, yaw

    def speed_cap(self, cap):
        return min(cap, 20) if abs(self.error) > 2 else cap

    def status(self):
        return dict(
            phase=self.phase,
            error_deg=self.error,
            reference_deg=self.reference,
            command_heading_deg=self.target,
            attempts=self.attempts,
            completed=self.completed,
        )

    def advance(self, port, now):
        """True owns this iteration; outer loop must not send normal motion."""
        self.check(port.state["yaw_deg"], port.rx_at)
        if not self.active:
            if abs(self.error) <= 3:
                return False
            if self.attempts >= 3:
                raise RuntimeError("heading_recovery_attempts_exhausted")
            self.attempts += 1
            self.started = now
            self.phase = "stopping_translation"
            port.log("heading_recovery_started", self.status())
        if now - self.started > 10:
            raise RuntimeError("heading_recovery_timeout")
        if now - port.rx_at > 0.35:
            raise RuntimeError("heading_recovery_stale_telemetry")
        if self.phase == "stopping_translation":
            if port.phase != "idle" or port.state["enabled"]:
                port.stop()
                return True
            q = pose(port.state)
            if q[2] != 0 or any(abs(v - round(v)) > 0.05 for v in q):
                raise RuntimeError("heading_recovery_requires_settled_retracted_pose")
            if not port.can_send():
                return True
            self.held_pose = q
            # vx=vy=0; A/B/C stay fixed. The original firmware performs yaw
            # correction even at zero translation, with the same heading byte.
            if not port.request(q[0], 0):
                return True
            self.correcting_since = now
            self.stable_since = self.stable_sample = None
            self.stable_frames = 0
            self.phase = "correcting"
            port.log("heading_correction_command", self.status())
        elif self.phase == "correcting":
            if pose(port.state) != self.held_pose:
                raise RuntimeError("heading_recovery_unexpected_arm_motion")
            if now - self.correcting_since > 6:
                raise RuntimeError("heading_correction_timeout")
            sample = port.rx_at
            within = (
                abs(self.error) <= 1.5
                and abs(angle_delta(port.state["yaw_deg"], self.target)) <= 1.5
                and port.state["enabled"]
                and not port.pending
            )
            if within:
                if self.stable_since is None:
                    self.stable_since = sample
                if sample != self.stable_sample:
                    self.stable_sample = sample
                    self.stable_frames += 1
                if self.stable_frames >= 3 and sample - self.stable_since >= 0.5:
                    port.stop()
                    self.phase = "confirming_disabled"
            else:
                self.stable_since = self.stable_sample = None
                self.stable_frames = 0
        elif self.phase == "confirming_disabled":
            if port.phase == "idle" and not port.state["enabled"]:
                if abs(self.error) > 1.5:
                    raise RuntimeError("heading_unstable_after_correction")
                self.completed += 1
                self.phase = "idle"
                port.log("heading_recovery_completed", self.status())
                return False
        return True
