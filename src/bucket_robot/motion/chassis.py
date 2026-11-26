#!/usr/bin/env python3
"""Suspended-chassis observation, preserving the complete arm pose.

Run on RK3588 with the tracking process stopped. No zero registration or
firmware reset. The original firmware has no communication-loss watchdog.
"""

import json
import math
from pathlib import Path
import time

from bucket_robot.motion.arm import ArmPort, pose


class HeldArmLimits:
    """Latch a valid existing pose; this controller cannot request an arm move."""

    def __init__(self, reference):
        config = json.loads(Path(reference).read_text())
        self.sender = config["control_policy"]["original_sender_module_on_rk3588"]
        self.held = None

    def check(self, q):
        if (
            len(q) != 3
            or any(not math.isfinite(v) or v != round(v) for v in q)
            or not (0 <= q[0] <= 180 and 0 <= q[1] <= 175 and 0 <= q[2] <= 175)
        ):
            raise RuntimeError("invalid_existing_arm_pose")
        if self.held is None:
            self.held = tuple(q)
        if tuple(q) != self.held:
            raise RuntimeError("arm_pose_must_remain_unchanged")


def pulse_frame(state, direction, speed):
    if direction not in ("forward", "backward", "left", "right"):
        raise ValueError("invalid_direction")
    if type(speed) is not int or not 1 <= speed <= 20:
        raise ValueError("speed_must_be_1_to_20")
    q = pose(state)
    if any(not math.isfinite(v) or v != round(v) for v in q):
        raise RuntimeError("fractional_arm_position_requires_separate_normalization")
    vx, vy = {
        "forward": (speed, 0),
        "backward": (-speed, 0),
        "left": (0, speed),
        "right": (0, -speed),
    }[direction]
    yaw = state["yaw_deg"]
    if yaw is None or not math.isfinite(yaw):
        raise RuntimeError("heading_invalid")
    return bytes([0xB3, vx & 255, vy & 255, round(yaw / 2) % 180, *map(int, q), 0, 1, 0xB4])


class ChassisPort(ArmPort):
    def start_pulse(self, direction, speed, seconds, continuous=False):
        if not math.isfinite(seconds) or not 0.6 <= seconds <= 10:
            raise ValueError("duration_must_be_0.6_to_10_seconds")
        if self.phase != "idle":
            raise RuntimeError("controller_busy")
        self.check_state(self.state)
        if time.monotonic() - self.rx_at > 0.35:
            raise RuntimeError("stale_telemetry")
        frame = pulse_frame(self.state, direction, speed)
        self.origin = self.target = pose(self.state)
        self.limits.check(self.target)
        self.uart_errors = self.state["uart_error_count_mod256"]
        self.move_counter = (self.expected + 1) % 256
        self.started = time.monotonic()
        self.deadline = math.inf if continuous else self.started + seconds
        self.phase = "driving"
        self.attempted = True
        self.acknowledged = False
        self.drive_observed = False
        self.log(
            "continuous" if continuous else "pulse",
            {
                "direction": direction,
                "speed_percent": speed,
                "duration_s": None if continuous else seconds,
                "preserved_pose": self.target,
            },
        )
        if not self.gateway.write_control_frame(frame, repeats=1):
            raise RuntimeError("drive_write_failed")

    def tick(self):
        if self.phase != "driving":
            return super().tick()
        frames = self.poll()
        now = time.monotonic()
        if now - self.rx_at > 0.6:
            raise RuntimeError("telemetry_timeout")
        for s in frames:
            self.check_state(s)
            if pose(s) != self.origin:
                raise RuntimeError("unexpected_arm_motion")
            if s["uart_error_count_mod256"] != self.uart_errors:
                raise RuntimeError("uart_error")
            counter = s["command_count_mod256"]
            if counter not in (self.expected, self.move_counter):
                raise RuntimeError("unexpected_external_control")
            if counter == self.move_counter:
                # Apply() sets enabled before the next 10 ms control update
                # changes state from disabled to driving. Its immediate report
                # may therefore contain enabled=True, state=0.
                transient = s["state"] == 0 and now - self.started <= 0.1
                if (
                    s["result"]
                    or not s["enabled"]
                    or (s["state"] not in (1, 3, 5) and not transient)
                ):
                    raise RuntimeError("drive_rejected_or_interrupted")
                self.acknowledged = True
                self.drive_observed |= s["state"] == 3
        if not self.acknowledged and now - self.started > 0.6:
            raise RuntimeError("command_ack_timeout")
        if now >= self.deadline:
            self.stop()


class SidePort(ChassisPort):
    def preflight_pose(self, q):
        super().preflight_pose(q)
        if q[1:] != [0, 0] or not 131 <= q[0] <= 133:
            raise RuntimeError("not_in_left_travel_pose")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.limits.boot_pose = [73, 0, 0]
        self.watch_firmware_reset = True
        self.fixed_yaw = self.state["yaw_deg"]
        self.initial_errors = self.state["uart_error_count_mod256"]

    def tick(self):
        stopping = self.phase == "stopping"
        super().tick()
        if stopping:
            return
        if self.state["uart_error_count_mod256"] != self.initial_errors:
            raise RuntimeError("new_uart_error")
        yaw = self.state["yaw_deg"]
        if yaw is None or abs((yaw - self.fixed_yaw + 180) % 360 - 180) > 3:
            raise RuntimeError("heading_drift")
