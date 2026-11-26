"""Task-level recovery policy. Pure logic: no camera, UART or hardware access."""

from dataclasses import dataclass
import math
from bucket_robot.tracking.aim import horizontal_aligned


@dataclass(frozen=True)
class RecoverySettings:
    cruise: int = 8
    deceleration_per_s: float = 10.0
    acceleration_per_s: float = 8.0
    initial_confirm_s: float = 0.7
    initial_confirm_frames: int = 3
    recovery_confirm_s: float = 0.25
    target_max_age_s: float = 0.35
    control_gap_s: float = 0.25


class RecoveryGate:
    """Loss reduces speed while observation continues; it never ends the task.

    Rates describe speed *commands*, not calibrated physical acceleration.
    Faults and operator stop remain immediate disable requests. Budget and hard
    fault pauses are latched; new images alone cannot release them.
    """

    def __init__(self, seconds, left=132, settings=None):
        if not math.isfinite(seconds) or not 0.6 <= seconds <= 120:
            raise ValueError("drive_budget_0.6_to_120")
        self.settings = settings or RecoverySettings()
        self.seconds, self.left = seconds, left
        self.phase = "centering"
        self.reason = "initial_acquisition"
        self.identity = None
        self.started = None
        self.last_now = None
        self.speed = 0.0
        self.velocity = 0
        self.drive_seconds = 0.0
        self.stable_since = None
        self.stable_stamp = None
        self.stable_frames = 0
        self.allow_a = False
        self.disable = True
        self.latched = None
        self.loss_count = 0
        self.recovery_count = 0
        self.in_loss = False
        self.align_since = None
        self.align_stamp = None
        self.align_frames = 0

    def trip(self, reason):
        self.latched = reason
        self.phase = "fault_paused"
        self.reason = reason
        self.speed = 0.0
        self.velocity = 0
        self.allow_a = False
        self.disable = True

    def _unstable(self):
        self.stable_since = None
        self.stable_stamp = None
        self.stable_frames = 0

    def _quality(self, now, target, identity):
        if target is None:
            return False, False, "target_unobserved"
        if identity is None:
            return False, False, "target_identity_not_confirmed"
        stamp = target.get("timestamp")
        if (
            not isinstance(stamp, (int, float))
            or not math.isfinite(stamp)
            or not 0 <= now - stamp <= self.settings.target_max_age_s
        ):
            return False, False, "camera_or_target_stale"
        if self.identity is None:
            self.identity = identity
        if identity != self.identity:
            return False, False, "target_identity_not_confirmed"
        dx = target.get("offset_px", [math.nan])[0]
        if not isinstance(dx, (int, float)) or not math.isfinite(dx):
            return False, False, "invalid_bearing"
        measured = (
            target.get("horizontal_bearing_valid") is True
            and target.get("bearing_evidence") == "current_detection_and_visible_appearance"
        )
        if target.get("partial_visible", False) and not measured:
            inward = (
                target.get("horizontal_clipped") is True
                and target.get("bearing_evidence") == "current_detection_and_visible_appearance"
            )
            return inward, False, "horizontal_extent_not_visible"
        return True, True, None

    def evaluate(
        self,
        now,
        target,
        identity,
        a,
        settled,
        commanded_a=None,
        motion_permitted=True,
        chassis_stopped=False,
        speed_cap=8,
    ):
        if not math.isfinite(now):
            self.trip("invalid_control_clock")
            return 0, None
        dt = 0.0 if self.last_now is None else now - self.last_now
        if dt < 0 or (self.velocity > 0 and dt > self.settings.control_gap_s):
            self.trip("control_loop_gap")
        if self.velocity > 0:
            self.drive_seconds += max(0.0, dt)
        self.last_now = now
        if self.latched:
            return 0, None
        if self.drive_seconds >= self.seconds:
            self.trip("drive_budget_exhausted")
            self.phase = "budget_paused"
            return 0, None
        self.allow_a, horizontal, problem = self._quality(now, target, identity)
        if not self.allow_a:
            if not self.in_loss:
                self.loss_count += 1
            self.in_loss = True
        # Disable preserves the last absolute A command. A one-degree feedback
        # tolerance must not permit handoff while that command is still 130.
        command_aligned = commanded_a is None or commanded_a >= self.left - 1
        if (
            motion_permitted
            and self.allow_a
            and horizontal
            and a >= self.left - 1
            and command_aligned
            and horizontal_aligned(target)
            and settled
            and (commanded_a is None or abs(a - commanded_a) < 0.1)
        ):
            self.speed = 0.0
            self.velocity = 0
            self.disable = True
            # Confirm fresh alignment after disable ACK and physical settling.
            # The old moving-frame observation cannot authorize a stage handoff.
            if chassis_stopped:
                stamp = target["timestamp"]
                if self.align_since is None:
                    self.align_since = stamp
                if stamp != self.align_stamp:
                    self.align_stamp = stamp
                    self.align_frames += 1
                if self.align_frames >= 3 and stamp - self.align_since >= 0.3:
                    self.phase = "aligned"
                    self.reason = "left_reference_reached"
                    return 0, "left_reference_reached"
            else:
                self.align_since = None
                self.align_stamp = None
                self.align_frames = 0
            self.phase = "verifying_left_alignment"
            self.reason = "waiting_stopped_alignment"
            return 0, None
        self.align_since = None
        self.align_stamp = None
        self.align_frames = 0
        centered = (
            motion_permitted
            and self.allow_a
            and horizontal
            and horizontal_aligned(target, 40 if self.started is not None else 20)
            and (settled or self.started is not None)
        )
        confirmed = False
        if centered:
            stamp = target["timestamp"]
            if self.stable_since is None:
                self.stable_since = stamp
            if stamp != self.stable_stamp:
                self.stable_frames += 1
                self.stable_stamp = stamp
            span = (
                self.settings.initial_confirm_s
                if self.started is None
                else self.settings.recovery_confirm_s
            )
            frames = self.settings.initial_confirm_frames if self.started is None else 3
            confirmed = self.stable_frames >= frames and stamp - self.stable_since >= span
        else:
            self._unstable()
        if confirmed:
            if self.in_loss:
                self.recovery_count += 1
            self.in_loss = False
            cap = min(float(self.settings.cruise), float(speed_cap))
            self.speed = (
                max(cap, self.speed - self.settings.deceleration_per_s * dt)
                if self.speed > cap
                else min(cap, self.speed + self.settings.acceleration_per_s * dt)
            )
            self.phase = "driving" if self.speed >= self.settings.cruise else "resuming"
            self.reason = None
        else:
            self.speed = max(0.0, self.speed - self.settings.deceleration_per_s * dt)
            if self.speed >= 1.0:
                self.phase = "decelerating"
            elif not self.allow_a:
                self.phase = "waiting_target"
            else:
                self.phase = "recentering" if self.started is not None else "centering"
            self.reason = problem or (
                "vertical_adjustment"
                if not motion_permitted
                else "confirming_same_target"
                if centered
                else "horizontal_recentering"
            )
        self.velocity = int(math.floor(self.speed + 1e-8))
        self.disable = self.velocity == 0 and not self.allow_a
        if self.velocity > 0 and self.started is None:
            self.started = now
        return self.velocity, None

    def status(self):
        return dict(
            phase=self.phase,
            reason=self.reason,
            velocity=self.velocity,
            allow_a=self.allow_a,
            disable=self.disable,
            identity=self.identity,
            drive_seconds=self.drive_seconds,
            drive_budget_s=self.seconds,
            loss_count=self.loss_count,
            recovery_count=self.recovery_count,
            hard_pause=self.latched,
        )
