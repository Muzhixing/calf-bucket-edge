"""One bounded calibration action using the shared transport and arm controller.

Requires chassis/pump isolation. Zero registration never physically homes axes.
Production tracking configuration is not changed. Each move ends disabled.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import termios
import time

from bucket_robot.motion.arm import ArmPort, Limits, pose
from bucket_robot.transport.writer import FrameWriter
from bucket_robot.resources import default_reference
from bucket_robot.transport.telemetry import Parser, DEFAULT_PORT, holders


class CalibrationLimits(Limits):
    def __init__(self, reference):
        super().__init__(reference)
        self.b_max = min(self.b_max, 60)

    def check(self, q):
        if len(q) != 3 or not all(isinstance(v, (int, float)) and v == round(v) for v in q):
            raise RuntimeError("invalid_calibration_pose")
        if not (
            self.a_min <= q[0] <= self.a_max
            and self.b_min <= q[1] <= self.b_max
            and 0 <= q[2] <= 20
        ):
            raise RuntimeError("outside_confirmed_calibration_range")


class CalibrationArm(ArmPort):
    def move(self, q):
        # A/B use the unchanged production implementation. C uses the same
        # telemetry/disable state machine, limited to one previously tested 5 mm step.
        if q[2] == pose(self.state)[2]:
            return super().move(q)
        if self.phase != "idle" or time.monotonic() < self.ready_at:
            raise RuntimeError("arm_busy")
        self.check_state(self.state)
        self.limits.check(q)
        start = pose(self.state)
        self.limits.check(start)
        if time.monotonic() - self.rx_at > 0.7:
            raise RuntimeError("stale_telemetry")
        if q[:2] != start[:2] or not 0 < abs(q[2] - start[2]) <= 5:
            raise RuntimeError("invalid_C_step")
        self.origin, self.target = start, q
        self.arrived_at = None
        self.move_counter = (self.expected + 1) % 256
        self.phase, self.attempted = "moving", True
        self.started = time.monotonic()
        self.deadline = self.started + 6 + abs(q[2] - start[2]) / 5
        if not self.gateway.write_control_frame(self.frame(q, True), repeats=1):
            raise RuntimeError("move_write_failed")


def register_zero(log):
    holders(DEFAULT_PORT, allow_partial=True)
    fd = os.open(DEFAULT_PORT, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    decoder = Parser()

    def receive(seconds):
        rows = []
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if select.select([fd], [], [], 0.03)[0]:
                data = os.read(fd, 4096)
                if not data:
                    raise RuntimeError("serial_disconnected")
                rows.extend(decoder.feed(data))
        log("rx", rows)
        return rows

    try:
        fcntl.ioctl(fd, termios.TIOCEXCL)
        a = termios.tcgetattr(fd)
        a[0] = a[1] = a[3] = 0
        a[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        a[4] = a[5] = termios.B115200
        a[6][termios.VMIN] = a[6][termios.VTIME] = 0
        termios.tcsetattr(fd, termios.TCSANOW, a)
        receive(0.4)
        rows = receive(1.2)
        if len(rows) < 3:
            raise RuntimeError("no_fresh_telemetry")
        last = rows[-1]
        for s in rows:
            if (
                s["enabled"]
                or s["state"] != 0
                or s["system_fault"]
                or s["motor_fault"]
                or not s["angle_valid"]
                or not s["heading_reference_established"]
                or pose(s) != pose(last)
                or s["command_count_mod256"] != last["command_count_mod256"]
            ):
                raise RuntimeError("not_stable_disabled_and_ready")
        if any(last[k] for k in ("lift_zero_confirmed", "slide_zero_confirmed")):
            raise RuntimeError("refuse_rezero_existing_origins")
        if pose(last)[1:] != [0, 0]:
            raise RuntimeError("unexpected_startup_positions")
        factory = FrameWriter

        class ExistingFD:
            def write(self, data):
                n = os.write(fd, data)
                log("tx", dict(hex=data.hex(" "), host_bytes=n))
                if n != len(data):
                    raise RuntimeError("partial_write_no_retry")
                return n

        gateway = factory(
            port=DEFAULT_PORT, serial_factory=lambda port, baud: ExistingFD(), command_repeats=1
        )
        gateway._open_serial()
        # Disabled, flags 0b110; preserves A; no actuator enable is sent.
        frame = bytes(
            [
                0xB3,
                0,
                0,
                round(last["yaw_deg"] / 2) % 180,
                last["servo_software_deg"],
                0,
                0,
                0,
                6,
                0xB4,
            ]
        )
        if not gateway.write_control_frame(frame, repeats=1):
            raise RuntimeError("zero_registration_write_failed")
        rows = receive(1.3)
        valid = [
            s for s in rows if s["command_count_mod256"] == (last["command_count_mod256"] + 1) % 256
        ]
        if len(valid) < 3:
            raise RuntimeError("zero_registration_not_confirmed")
        for s in valid:
            ArmPort.check_state(s)
            if s["enabled"] or s["state"] != 0 or s["result"] or pose(s) != pose(last):
                raise RuntimeError("unexpected_registration_result")
        return valid[-1]
    finally:
        os.close(fd)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--isolated-chassis-pump", action="store_true", required=True)
    p.add_argument("--register-confirmed-home", action="store_true")
    p.add_argument("--expect", type=int, nargs=3)
    p.add_argument("--target", type=int, nargs=3)
    args = p.parse_args()
    if args.register_confirmed_home:
        if args.expect or args.target:
            p.error("registration and motion must be separate")
    else:
        if args.expect is None or args.target is None:
            p.error("motion requires expected and target poses")
        delta = [abs(a - b) for a, b in zip(args.expect, args.target)]
        if sum(d > 0 for d in delta) != 1 or delta[0] > 2 or delta[1] > 10 or delta[2] > 5:
            p.error("one A <=2 deg, B <=10 mm or C <=5 mm step")
    args.out.mkdir(parents=True, exist_ok=False)

    def log(event, data):
        with (args.out / "events.jsonl").open("a") as f:
            f.write(json.dumps(dict(monotonic=time.monotonic(), event=event, data=data)) + "\n")

    def interrupted(signum, frame):
        raise RuntimeError(f"interrupted_{signum}")

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    result, port = {}, None
    try:
        if args.register_confirmed_home:
            result["telemetry"] = register_zero(log)
        else:
            limits = CalibrationLimits(default_reference())
            limits.check(args.expect)
            limits.check(args.target)
            port = CalibrationArm(limits, log)
            if pose(port.state) != args.expect:
                raise RuntimeError("live_pose_differs_from_expected")
            port.move(args.target)
            while port.phase != "idle":
                port.tick()
                time.sleep(0.01)
            if port.state["enabled"] or pose(port.state) != args.target:
                raise RuntimeError("final_pose_mismatch")
            result["telemetry"] = port.state
        result["disabled_confirmed"] = True
    except BaseException as exc:
        result["error"] = str(exc)
        raise
    finally:
        if port is not None:
            error = port.shutdown()
            if error:
                result["stop_error"] = error
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    if result.get("stop_error"):
        raise RuntimeError(result["stop_error"])


if __name__ == "__main__":
    main()
