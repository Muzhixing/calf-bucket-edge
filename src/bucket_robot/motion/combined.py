"""Chassis/arm adapters. All command transactions belong to ContinuousArmPort."""

import time
from bucket_robot.motion.arm import ContinuousArmPort, pose


class CombinedPort(ContinuousArmPort):
    def __init__(self, *args, **kwargs):
        self.velocity = 0
        super().__init__(*args, **kwargs)
        if pose(self.state)[1:] != [0, 0]:
            raise RuntimeError("B_C_must_be_retracted")
        self.fixed_yaw = self.state["yaw_deg"]
        self.heading = round(self.fixed_yaw / 2) % 180
        self.errors = self.state["uart_error_count_mod256"]

    def frame(self, q, enable):
        return bytes(
            [
                0xB3,
                self.velocity if enable else 0,
                0,
                self.heading,
                *map(int, q),
                0,
                int(enable),
                0xB4,
            ]
        )

    def can_send(self):
        return super().can_send() and not self.pending

    def request(self, a, velocity):
        q = [a, 0, 0]
        self.limits.check(q)
        if type(velocity) is not int or not 0 <= velocity <= 8:
            raise RuntimeError("invalid_speed")
        if not self.can_send():
            return False
        self.check_state(self.state)
        if time.monotonic() - self.rx_at > 0.35:
            raise RuntimeError("stale_telemetry")
        current = pose(self.state)
        old = self.target or current
        if current[1:] != [0, 0] or abs(a - old[0]) > 4 or abs(a - current[0]) > 4:
            raise RuntimeError("invalid_A_only_step")
        if self.target == q and self.velocity == velocity and self.state["enabled"]:
            return False
        self.velocity = velocity
        self.b_origin = 0.0
        self.b_started = time.monotonic()
        return self._send_target(q, current)

    def tick(self):
        stopping = self.phase == "stopping"
        super().tick()
        # Movement faults trigger disable, but must not interrupt its ACK wait.
        # The base stopping path still verifies counter, accepted result and
        # disabled output, with its unchanged feedback timeout.
        if stopping:
            return
        if pose(self.state)[1:] != [0, 0]:
            raise RuntimeError("unexpected_B_C_motion")
        if self.state["uart_error_count_mod256"] != self.errors:
            raise RuntimeError("new_uart_error")
        if abs((self.state["yaw_deg"] - self.fixed_yaw + 180) % 360 - 180) > 3:
            raise RuntimeError("heading_drift")

    def rearm(self):
        self.rearm_pose = pose(self.state)
        expected = (self.expected + 1) % 256
        if not self.gateway.write_control_frame(self.frame(pose(self.state), False), repeats=1):
            raise RuntimeError("disable_write_failed")
        until = time.monotonic() + 1.5
        while time.monotonic() < until:
            for s in self.poll():
                self.check_state(s)
                if s["command_count_mod256"] not in (self.expected, expected):
                    raise RuntimeError("external_controller")
                if s["enabled"] or pose(s) != self.rearm_pose:
                    raise RuntimeError("unexpected_rearm_pose")
                if s["command_count_mod256"] == expected:
                    if s["result"] or s["state"] != 0:
                        raise RuntimeError("disable_rejected")
                    self.expected = expected
                    return
            time.sleep(0.01)
        raise RuntimeError("disable_ack_timeout")


class ABCombinedPort(CombinedPort):
    def __init__(self, *args, **kwargs):
        self.velocity = 0
        ContinuousArmPort.__init__(self, *args, **kwargs)
        if pose(self.state)[2] != 0:
            raise RuntimeError("C_must_be_retracted")
        self.fixed_yaw = self.state["yaw_deg"]
        self.heading = round(self.fixed_yaw / 2) % 180
        self.errors = self.state["uart_error_count_mod256"]

    def tick(self):
        stopping = self.phase == "stopping"
        ContinuousArmPort.tick(self)
        if stopping:
            return
        if pose(self.state)[2] != 0:
            raise RuntimeError("unexpected_C_motion")
        if self.state["uart_error_count_mod256"] != self.errors:
            raise RuntimeError("new_uart_error")
        if abs((self.state["yaw_deg"] - self.fixed_yaw + 180) % 360 - 180) > 3:
            raise RuntimeError("heading_drift")

    def request(self, a, velocity):
        current = pose(self.state)
        if abs(current[1] - round(current[1])) > 0.05:
            raise RuntimeError("B_not_settled_for_chassis")
        return self._request([a, int(round(current[1])), 0], velocity, False)

    def follow(self, q):
        if self.velocity or self.state["state"] == 3:
            raise RuntimeError("B_requires_stopped_chassis")
        if q[2] != 0:
            raise RuntimeError("C_must_be_retracted")
        return ContinuousArmPort.follow(self, q)

    def _request(self, q, velocity, b_move):
        self.limits.check(q)
        if type(velocity) is not int or not 0 <= velocity <= 8:
            raise RuntimeError("invalid_speed")
        if not self.can_send():
            return False
        self.check_state(self.state)
        if time.monotonic() - self.rx_at > 0.35:
            raise RuntimeError("stale_telemetry")
        current = pose(self.state)
        old = self.target or current
        if current[2] != 0 or q[2] != 0:
            raise RuntimeError("C_must_be_retracted")
        if b_move:
            if (
                velocity
                or q[0] != current[0]
                or abs(q[1] - current[1]) > getattr(self.limits, "b_max_step", 24)
            ):
                raise RuntimeError("invalid_B_step")
        elif q[1] != current[1] or abs(q[0] - current[0]) > 4 or abs(q[0] - old[0]) > 4:
            raise RuntimeError("invalid_A_only_step")
        if self.target == q and self.velocity == velocity and self.state["enabled"]:
            return False
        self.velocity = velocity
        self.b_origin = current[1]
        self.b_started = time.monotonic()
        return self._send_target(q, current)


class LateralPort(ABCombinedPort):
    def preflight_pose(self, q):
        super().preflight_pose(q)
        if not 131 <= q[0] <= 133 or not 0 <= q[1] <= 96 or q[2] != 0:
            raise RuntimeError("not_in_left_travel_pose")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.limits.boot_pose = [73, 0, 0]
        self.fixed_a = pose(self.state)[0]

    def frame(self, q, enable):
        return bytes(
            [
                0xB3,
                0,
                self.velocity if enable else 0,
                self.heading,
                *map(int, q),
                0,
                int(enable),
                0xB4,
            ]
        )

    def request(self, a, velocity):
        if a != self.fixed_a:
            raise RuntimeError("approach_arm_must_stay_fixed")
        return super().request(a, velocity)
