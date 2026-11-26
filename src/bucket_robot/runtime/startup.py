"""Explicit operator-confirmed MCU/motor restart preparation; original protocol."""

import fcntl
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
import uuid
from collections import deque

from bucket_robot.motion.arm import ArmPort, pose, reset_telemetry, close_serial


def c_selftest_target(limits):
    value = limits.startup_policy.get("c_selftest_max_mm", limits.c_bounds[1])
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value != round(value)
        or not 0 < value <= limits.c_bounds[1]
    ):
        raise ValueError("invalid_C_selftest_target")
    return int(value)


def derive_wait_profile(limits):
    """Explicit fixed wait takes precedence over legacy range-scaled estimate."""
    policy = limits.startup_policy
    if (
        len(limits.c_bounds) != 2
        or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in limits.c_bounds)
        or limits.c_bounds[0] != 0
        or not limits.c <= limits.c_bounds[1] <= 175
    ):
        raise ValueError("invalid_startup_C_range")
    distances = {
        "A_deg": max(
            abs(limits.a_min - limits.boot_pose[0]), abs(limits.a_max - limits.boot_pose[0])
        ),
        "B_mm": max(abs(limits.b_min), abs(limits.b_max)),
        "C_mm": max(abs(v) for v in limits.c_bounds),
    }
    reference = policy.get("reference_return_distances", {"A_deg": 30, "B_mm": 96, "C_mm": 30})
    observed = policy.get("observed_home_max_s", 5)
    margin = policy.get("settle_margin_s", 3)
    if any(
        not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0
        for v in [observed, margin, *[reference.get(k) for k in distances]]
    ):
        raise ValueError("invalid_observed_startup_wait_profile")
    scale = max(1.0, *[distances[k] / reference[k] for k in distances])
    fixed = policy.get("fixed_wait_seconds")
    if fixed is not None and (
        isinstance(fixed, bool) or not isinstance(fixed, (int, float)) or not math.isfinite(fixed)
    ):
        raise ValueError("invalid_fixed_startup_wait")
    seconds = fixed if fixed is not None else math.ceil(observed * scale + margin)
    if not 5 <= seconds <= 300:
        raise ValueError("startup_wait_outside_5_to_300s")
    evidence = policy.get(
        "evidence", "User observed whole-machine autonomous homing finish within about 5 seconds."
    )
    basis = json.dumps(
        {
            "method": "explicit_fixed_wait"
            if fixed is not None
            else "ceil(observed_seconds*max(1,range_ratio)+settle_margin)",
            "fixed_wait_seconds": fixed,
            "range_scaling_applied": fixed is None,
            "observed_seconds": observed,
            "margin_s": margin,
            "range_scale": scale,
            "distances": distances,
            "reference_distances": reference,
            "evidence": evidence,
            "expanded_range_validated": False,
            "hardware_complete_signal": False,
        },
        ensure_ascii=False,
    )
    return {
        "wait_seconds": seconds,
        "wait_basis": basis,
        "distances": distances,
        "range_scale": scale,
    }


class NativeWarmup:
    """Ten advancing completed native inference frames; no target required."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.last_id = -1
        self.last_arrival = 0.0

    def observe(self, metadata, now=None):
        now = time.monotonic() if now is None else now
        try:
            frame_id = metadata["frame_id"]
            arrival = float(metadata["arrival"])
            elapsed = float(metadata["detector"]["npu_run_ms"])
            valid = (
                metadata["backend"] == "cpp"
                and isinstance(metadata["detections"], list)
                and math.isfinite(elapsed)
                and elapsed > 0
                and 0 <= now - arrival <= 0.4
            )
        except (KeyError, ValueError, TypeError):
            valid = False
        if not valid:
            self.reset()
            return
        if frame_id == self.last_id:
            return
        if (
            frame_id < self.last_id
            or arrival < self.last_arrival
            or arrival - self.last_arrival > 0.5
        ):
            self.count = 0
        self.count = min(10, self.count + 1)
        self.last_id = frame_id
        self.last_arrival = arrival

    def status(self, now=None):
        now = time.monotonic() if now is None else now
        return {
            "ready": self.count >= 10 and 0 <= now - self.last_arrival <= 0.4,
            "completed_inference_frames": self.count,
        }


def boot_identity():
    path = Path("/proc/sys/kernel/random/boot_id")
    return path.read_text().strip() if path.is_file() else "unavailable"


class PowerCycleStore:
    """Durable at-most-once zero attempt per operator-confirmed cycle.

    A boot id is only an invalidator, never evidence of mechanical homing.
    Interrupted initialization is never resumed, including before the write.
    """

    def __init__(self, path, wait_seconds=None, wait_basis="", boot_id=None):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.wait_seconds = wait_seconds
        self.wait_basis = wait_basis
        if wait_seconds is not None and (
            isinstance(wait_seconds, bool)
            or not isinstance(wait_seconds, (int, float))
            or not math.isfinite(wait_seconds)
            or not 5 <= wait_seconds <= 300
            or not wait_basis.strip()
        ):
            raise ValueError("invalid_power_cycle_wait_profile")
        self.boot_id = boot_id or boot_identity()
        self.lock = threading.RLock()
        self.fd = os.open(str(self.path) + ".lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(self.fd)
            self.fd = None
            raise RuntimeError("startup_state_owned_by_another_process")
        self.error = ""
        self.data = {
            "version": 1,
            "phase": "unconfirmed",
            "seen_requests": [],
            "zero_attempted": False,
        }
        if self.path.exists():
            try:
                d = json.loads(self.path.read_text())
                if (
                    d.get("version") != 1
                    or not isinstance(d.get("seen_requests"), list)
                    or d.get("phase")
                    not in (
                        "unconfirmed",
                        "waiting",
                        "preflight",
                        "disable_pending",
                        "zero_pending",
                        "registered",
                        "a_check_left",
                        "a_check_right",
                        "a_moving",
                        "bc_check_out",
                        "bc_check_return",
                        "bc_check_b_out",
                        "bc_check_b_return",
                        "bc_check_c_out",
                        "bc_check_c_return",
                        "c_moving",
                        "ready",
                        "uncertain",
                    )
                ):
                    raise ValueError("invalid_state")
                if not isinstance(d.get("zero_attempted"), bool):
                    raise ValueError("invalid_zero_attempt_marker")
                reused = (
                    d.get("reuse_existing_origins") is True
                    and d.get("origins_observed") is True
                    and d.get("registered_telemetry", {}).get("lift_zero_confirmed") is True
                    and d.get("registered_telemetry", {}).get("slide_zero_confirmed") is True
                )
                if d["phase"] == "ready" and (
                    not (d["zero_attempted"] or reused)
                    or not d.get("cycle_id")
                    or not d.get("ready_telemetry")
                ):
                    raise ValueError("invalid_ready_record")
                self.data = d
                if d["phase"] not in ("unconfirmed", "uncertain") and (
                    d["phase"] != "ready" or d.get("boot_id") != self.boot_id
                ):
                    self.invalidate("process_restart_or_host_boot_changed")
            except Exception as exc:
                self.error = "startup_state_unreadable"
                self.data = {
                    "version": 1,
                    "phase": "uncertain",
                    "seen_requests": [],
                    "zero_attempted": True,
                    "reason": str(exc),
                }

    def _save(self):
        # File + directory fsync before returning, especially before flags=6.
        if self.error:
            raise RuntimeError(self.error)
        temp = None
        try:
            fd, temp = tempfile.mkstemp(prefix=self.path.name + ".", dir=self.path.parent)
            with os.fdopen(fd, "w") as stream:
                json.dump(self.data, stream, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, self.path)
            temp = None
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except BaseException:
            self.error = "startup_state_write_failed"
            raise
        finally:
            if temp is not None:
                try:
                    os.unlink(temp)
                except FileNotFoundError:
                    pass

    def status(self):
        with self.lock:
            return {
                **self.data,
                "seen_requests_count": len(self.data["seen_requests"]),
                "seen_requests": None,
                "wait_seconds": self.wait_seconds,
                "wait_basis": self.wait_basis,
                "wait_configured": self.wait_seconds is not None,
                "error": self.error,
            }

    @property
    def ready(self):
        with self.lock:
            return not self.error and self.data["phase"] == "ready"

    def confirm(self, request_id, event="whole_machine_power_cycle"):
        if event not in (
            "whole_machine_power_cycle",
            "confirmed_mcu_motor_restart",
            "automatic_mcu_restart",
        ):
            raise ValueError("invalid_restart_event")
        try:
            request_id = str(uuid.UUID(request_id))
        except (ValueError, TypeError, AttributeError):
            raise ValueError("invalid_power_cycle_request_id")
        with self.lock:
            if self.error:
                raise ValueError(self.error)
            if request_id in self.data["seen_requests"]:
                return False
            if self.wait_seconds is None:
                raise ValueError("power_cycle_wait_not_calibrated")
            if self.data["phase"] not in ("unconfirmed", "uncertain", "ready"):
                raise ValueError("power_cycle_already_in_progress")
            seen = self.data["seen_requests"] + [request_id]
            self.data = {
                "version": 1,
                "phase": "waiting",
                "seen_requests": seen,
                "cycle_id": request_id,
                "boot_id": self.boot_id,
                "confirmed_at": time.time(),
                "zero_attempted": False,
                "operator_paused": False,
                "wait_seconds_used": self.wait_seconds,
                "wait_basis_used": self.wait_basis,
                "restart_event": event,
                "origin_evidence": (
                    "stable_firmware_boot_signature_then_timed_assumption"
                    if event == "automatic_mcu_restart"
                    else "operator_confirmed_" + event + "_then_timed_assumption"
                ),
            }
            self._save()
            return True

    def phase(self, value, **fields):
        with self.lock:
            self.data.update(phase=value, **fields)
            self._save()

    def adopt_registered(self, telemetry, home):
        """Use freshly observed firmware origins; this is not a new homing event."""
        with self.lock:
            if self.error:
                raise ValueError(self.error)
            if self.data.get("operator_paused"):
                raise ValueError("operator_paused")
            if self.data["phase"] not in ("unconfirmed", "uncertain"):
                raise ValueError("preparation_already_in_progress")
            if not registered_home(telemetry, home):
                raise ValueError("registered_home_not_ready")
            previous_reason = self.data.get("reason")
            self.data = {
                "version": 1,
                "phase": "waiting",
                "seen_requests": self.data["seen_requests"],
                "cycle_id": str(uuid.uuid4()),
                "boot_id": self.boot_id,
                "confirmed_at": time.time(),
                "zero_attempted": False,
                "operator_paused": False,
                "reuse_existing_origins": True,
                "origins_observed": True,
                "registered_telemetry": dict(telemetry),
                "wait_seconds_used": 0,
                "origin_evidence": "fresh_stable_registered_home",
                "previous_reason": previous_reason,
            }
            self._save()

    def before_zero(self):
        with self.lock:
            if (
                self.data["phase"] != "disable_pending"
                or self.data["zero_attempted"]
                or self.data.get("reuse_existing_origins")
            ):
                raise RuntimeError("zero_registration_already_attempted")
            self.data.update(phase="zero_pending", zero_attempted=True, zero_attempt_at=time.time())
            self._save()

    def invalidate(self, reason):
        with self.lock:
            self.phase("uncertain", reason=reason)

    def close(self):
        with self.lock:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None


class RestartEvidence:
    """Require several fresh batches of the combined MCU boot signature."""

    def __init__(self, boot_pose=(90, 0, 0)):
        self.boot_pose = tuple(boot_pose)
        self.reset()

    def reset(self):
        self.since = None
        self.last = None
        self.samples = 0

    def observe(self, frames, now):
        if not frames:
            return False
        if self.last is not None and now - self.last > 0.7:
            self.reset()
        if not all(reset_telemetry(s, self.boot_pose) for s in frames):
            self.reset()
            return False
        if self.since is None:
            self.since = now
        # Coalesced frames count as one observation, not many independent samples.
        if self.last is None or now > self.last:
            self.samples += 1
        self.last = now
        return self.samples >= 4 and now - self.since >= 0.75


def registered_home(s, home):
    if not s:
        return False
    try:
        return (
            not s["enabled"]
            and s["state"] == 0
            and s["result"] == 0
            and s["lift_zero_confirmed"]
            and s["slide_zero_confirmed"]
            and not s["system_fault"]
            and not s["motor_fault"]
            and s["angle_valid"]
            and s["heading_reference_established"]
            and s["yaw_deg"] is not None
            and math.isfinite(s["yaw_deg"])
            and pose(s) == [home, 0, 0]
        )
    except (KeyError, TypeError, ValueError):
        return False


class RegisteredHomeEvidence(RestartEvidence):
    """Fresh stable disabled state, including ownership/error counters."""

    def __init__(self, home):
        self.home = home
        self.reset()

    def reset(self):
        super().reset()
        self.counters = None

    def observe(self, frames, now):
        if not frames:
            return False
        if self.last is not None and now - self.last > 0.7:
            self.reset()
        counters = (
            frames[-1].get("command_count_mod256"),
            frames[-1].get("uart_error_count_mod256"),
        )
        if None in counters or not all(
            registered_home(s, self.home)
            and (s.get("command_count_mod256"), s.get("uart_error_count_mod256")) == counters
            for s in frames
        ):
            self.reset()
            return False
        if self.counters is not None and self.counters != counters:
            self.reset()
        self.counters = counters
        if self.since is None:
            self.since = now
        if self.last is None or now > self.last:
            self.samples += 1
        self.last = now
        return self.samples >= 4 and now - self.since >= 0.75


class RestartWatchPort(ArmPort):
    """Sole serial owner when idle: read only, including on close/failure."""

    def __init__(self, limits, *args, **kwargs):
        kwargs["defer_preflight"] = True
        self.opened_at = time.monotonic()
        self.evidence = RestartEvidence(limits.boot_pose)
        self.registered_evidence = RegisteredHomeEvidence(limits.a_home)
        self.registered_ready = False
        super().__init__(limits, *args, **kwargs)

    def reset_ready(self):
        frames = self.poll()
        now = time.monotonic()
        if now - self.opened_at < 0.4:
            return False
        self.registered_ready = self.registered_evidence.observe(frames, now)
        return self.evidence.observe(frames, now)

    def shutdown(self):
        if self.fd is not None:
            close_serial(self.fd)
            self.fd = None
        return None


def can_auto_prepare(store):
    # After an ambiguous write, a remaining boot signature cannot authorize retry.
    # A later registered/ready state followed by lost flags is a different epoch.
    d = store.data
    return (
        not store.error
        and not d.get("operator_paused", False)
        and (
            not d.get("zero_attempted", False)
            or d.get("origins_observed", False)
            or bool(d.get("ready_telemetry"))
        )
    )


class StartupPort(ArmPort):
    """Original-protocol port, bounded startup only, no CAN/new firmware fields."""

    def __init__(self, *args, **kwargs):
        self.registered = False
        self.ack_flags = None
        self.reuse_existing_origins = False
        self.opened_at = time.monotonic()
        self.preflight_history = deque(maxlen=128)
        kwargs["defer_preflight"] = True
        super().__init__(*args, **kwargs)

    def poll(self):
        frames = super().poll()
        now = time.monotonic()
        if now - self.opened_at >= 0.4:
            self.preflight_history.extend((now, s) for s in frames)
        while self.preflight_history and now - self.preflight_history[0][0] > 1.3:
            self.preflight_history.popleft()
        return frames

    def preflight_ready(self):
        now = time.monotonic()
        frames = list(self.preflight_history)
        if (
            now - self.opened_at < 1.6
            or len(frames) < 3
            or frames[-1][0] - frames[0][0] < 0.7
            or now - self.rx_at > 0.4
        ):
            return False
        try:
            self.validate_preflight([s for _, s in frames])
        except RuntimeError as exc:
            if str(exc) in ("no_fresh_telemetry", "heading_not_ready"):
                return False
            raise
        return True

    @staticmethod
    def healthy(s, allow_comms=False):
        if s["motor_fault"] or s["system_fault"] not in ((0, 1) if allow_comms else (0,)):
            raise RuntimeError("firmware_fault")
        if (
            not s["angle_valid"]
            or not s["heading_reference_established"]
            or s["yaw_deg"] is None
            or not math.isfinite(s["yaw_deg"])
        ):
            raise RuntimeError("heading_not_ready")

    def preflight_state(self, s):
        if self.reuse_existing_origins:
            if not registered_home(s, self.limits.a_home):
                raise RuntimeError("registered_home_not_ready")
            return
        self.healthy(s, allow_comms=True)
        if s["lift_zero_confirmed"] or s["slide_zero_confirmed"]:
            raise RuntimeError("power_cycle_origins_already_registered")
        if pose(s) != self.limits.boot_pose:
            raise RuntimeError("power_cycle_start_pose_mismatch")

    def preflight_disabled(self, s):
        return not s["enabled"] and s["state"] == (4 if s["system_fault"] == 1 else 0)

    def preflight_pose(self, q):
        if getattr(self, "bc_sweeping", False):
            self.limits.check(q)
            if q[0] != self.limits.a_home:
                raise RuntimeError("unexpected_A_during_BC_selftest")
            return
        low, high = (
            (self.limits.a_min, self.limits.a_max)
            if getattr(self, "sweeping", False)
            else sorted((self.limits.a_home, self.limits.boot_pose[0]))
        )
        if not (low <= q[0] <= high and q[1] == 0 and 0 <= q[2] <= self.limits.c):
            raise RuntimeError("unexpected_startup_pose")
        self.limits.check([q[0], 0, self.limits.c])

    def check_state(self, s):
        self.healthy(s)
        if self.registered and not (s["lift_zero_confirmed"] and s["slide_zero_confirmed"]):
            raise RuntimeError("startup_origins_lost")

    def disabled_command(self, flags):
        if self.phase != "idle" or flags not in (0, 6):
            raise RuntimeError("invalid_startup_command")
        if self.reuse_existing_origins and flags == 6:
            raise RuntimeError("existing_origins_must_not_be_registered_again")
        if time.monotonic() - self.rx_at > 0.7:
            raise RuntimeError("stale_telemetry")
        self.ack_flags = flags
        self.move_counter = (self.expected + 1) % 256
        self.phase = "startup_ack"
        self.deadline = time.monotonic() + 1.5
        target = (
            [self.limits.a_home, 0, 0] if self.reuse_existing_origins else self.limits.boot_pose
        )
        if not self.gateway.write_control_frame(self.frame(target, flags), repeats=1):
            raise RuntimeError("startup_write_failed")

    def startup_move(self, q, sweep=False, bc_sweep=False):
        if bc_sweep:
            self.bc_sweeping = True
        if sweep:
            self.sweeping = True
        current = pose(self.state)
        self.check_state(self.state)
        self.preflight_pose(current)
        self.preflight_pose(q)
        home = self.limits.a_home
        allowed = (
            q == [home, 0, 0]
            and current == self.limits.boot_pose
            or q == [home, 0, self.limits.c]
            and current == [home, 0, 0]
        )
        if sweep:
            allowed = (
                q[0] in (self.limits.a_min, self.limits.a_max, home)
                and q[1:] == [0, 0]
                and current[1:] == [0, 0]
            )
        if bc_sweep:
            home_pose = [home, 0, 0]
            endpoints = ([home, self.limits.b_max, 0], [home, 0, c_selftest_target(self.limits)])
            allowed = (
                self.limits.parallel_bc
                and self.limits.startup_policy.get("bc_startup_selftest")
                and any((current, q) in ((home_pose, end), (end, home_pose)) for end in endpoints)
            )
        if not self.registered or not allowed or self.phase != "idle":
            raise RuntimeError("invalid_startup_move")
        if time.monotonic() - self.rx_at > 0.7:
            raise RuntimeError("stale_telemetry")
        self.origin = current
        self.target = q
        self.arrived_at = None
        self.move_counter = (self.expected + 1) % 256
        self.phase = "moving"
        self.attempted = True
        self.started = time.monotonic()
        duration = max(
            abs(q[1] - current[1]) / self.limits.b_max_speed
            + self.limits.b_max_speed / self.limits.b_accel,
            abs(q[2] - current[2]) / self.limits.c_max_speed
            + self.limits.c_max_speed / self.limits.c_accel,
        )
        self.deadline = self.started + max(8, duration + 3)
        if not self.gateway.write_control_frame(self.frame(q, True), repeats=1):
            raise RuntimeError("move_write_failed")

    def tick(self):
        if self.phase != "startup_ack":
            return super().tick()
        now = time.monotonic()
        for s in self.poll():
            target = (
                [self.limits.a_home, 0, 0] if self.reuse_existing_origins else self.limits.boot_pose
            )
            if s["enabled"] or pose(s) != target:
                raise RuntimeError("unexpected_mechanism_motion")
            if s["command_count_mod256"] not in (self.expected, self.move_counter):
                raise RuntimeError("unexpected_command_counter")
            if s["command_count_mod256"] == self.move_counter:
                self.healthy(s)
                if s["result"] != 0 or s["state"] != 0:
                    raise RuntimeError("startup_command_rejected")
                zeros = s["lift_zero_confirmed"] and s["slide_zero_confirmed"]
                if self.ack_flags == 6 and not zeros:
                    raise RuntimeError("zero_registration_not_acknowledged")
                if self.ack_flags == 0:
                    if self.reuse_existing_origins and not zeros:
                        raise RuntimeError("startup_origins_lost")
                    if not self.reuse_existing_origins and (
                        s["lift_zero_confirmed"] or s["slide_zero_confirmed"]
                    ):
                        raise RuntimeError("unexpected_origin_registration")
                self.registered = bool(zeros)
                self.expected = self.move_counter
                self.phase = "idle"
                self.log("startup_disabled_ack", {"flags": self.ack_flags, "telemetry": s})
                return
        if now - self.rx_at > 0.7 or now > self.deadline:
            raise RuntimeError("startup_ack_timeout_no_retry")


class StartupSequence:
    def __init__(
        self, store, limits, log, port_factory=StartupPort, ready_check=None, started_at=None
    ):
        self.store, self.limits, self.log, self.port_factory = store, limits, log, port_factory
        self.port = None
        self.started = time.monotonic() if started_at is None else started_at
        self.stage = "waiting"
        self.done = False
        self.ready_check = ready_check or (lambda: {"ready": True})
        self.last_check = None
        self.reuse_existing_origins = store.data.get("reuse_existing_origins") is True
        self.wait_seconds = (
            0
            if self.reuse_existing_origins and not limits.startup_policy.get("bc_startup_selftest")
            else store.wait_seconds
        )
        self.log(
            "registered_preparation_started"
            if self.reuse_existing_origins
            else "restart_preparation_wait",
            {**store.status(), "window_started_at": self.started},
        )

    def tick(self):
        checks = self.ready_check()
        if checks != self.last_check:
            self.last_check = checks
            self.log("startup_self_check", checks)
        if self.stage == "waiting":
            # Open and read only; UART/heading preflight overlaps homing and model warmup.
            if self.port is None:
                self.port = self.port_factory(self.limits, self.log)
                self.port.reuse_existing_origins = self.reuse_existing_origins
            self.port.poll()
            if time.monotonic() - self.started < self.wait_seconds:
                return
            if not checks.get("ready") or not self.port.preflight_ready():
                return
            self.store.phase("preflight", wait_seconds_used=self.wait_seconds)
            self.stage = "connect"
            self.log(
                "registered_preflight_complete"
                if self.reuse_existing_origins
                else "homing_assumed_after_wait",
                {
                    "wait_seconds": self.wait_seconds,
                    "elapsed_seconds": time.monotonic() - self.started,
                    "basis": self.store.wait_basis,
                    "hardware_complete_signal": False,
                    "self_checks": checks,
                },
            )
            return
        if not checks.get("ready"):
            if self.port.attempted or self.port.state and self.port.state["enabled"]:
                raise RuntimeError("startup_self_check_lost")
            self.port.poll()
            return
        if self.stage == "connect":
            self.store.phase("disable_pending")
            self.port.disabled_command(0)
            self.stage = "disable"
            return
        self.port.tick()
        if self.port.phase != "idle":
            return
        if self.stage == "disable":
            if self.reuse_existing_origins:
                self.begin_axis_preparation()
                return
            self.store.before_zero()  # Durable write-ahead; never retry flags6.
            self.log("origin_registration_attempt", self.store.status())
            self.port.disabled_command(6)
            self.stage = "zero"
        elif self.stage == "zero":
            self.store.phase("registered", origins_observed=True)
            self.log("origin_registered_by_startup_assumption", self.port.state)
            self.begin_axis_preparation()
        elif self.stage == "a_left":
            self.store.phase("a_check_right")
            self.log("startup_a_check", {"stage": "right", "target_deg": self.limits.a_max})
            self.port.startup_move([self.limits.a_max, 0, 0], sweep=True)
            self.stage = "a_right"
        elif self.stage == "a_right":
            self.store.phase("a_moving")
            self.log("startup_a_check", {"stage": "return", "target_deg": self.limits.a_home})
            self.port.startup_move([self.limits.a_home, 0, 0], sweep=True)
            self.stage = "a"
        elif self.stage == "a":
            if self.limits.startup_policy.get("bc_startup_selftest"):
                self.bc_move("b", "out", [self.limits.a_home, self.limits.b_max, 0])
            elif self.limits.c == 0 or self.limits.startup_policy.get("a_startup_selftest"):
                self.complete()  # Already at C0: no C command or extra enable cycle.
            else:
                self.log("startup_c_preparing", {"target_mm": self.limits.c})
                self.store.phase("c_moving")
                self.port.startup_move([self.limits.a_home, 0, self.limits.c])
                self.stage = "c"
        elif self.stage == "bc_b_out":
            self.bc_move("b", "return", [self.limits.a_home, 0, 0])
        elif self.stage == "bc_b_return":
            self.bc_move("c", "out", [self.limits.a_home, 0, c_selftest_target(self.limits)])
        elif self.stage == "bc_c_out":
            self.bc_move("c", "return", [self.limits.a_home, 0, 0])
        elif self.stage == "bc_c_return":
            self.complete()
        elif self.stage == "c":
            self.complete()

    def bc_move(self, axis, direction, target):
        self.store.phase("bc_check_" + axis + "_" + direction)
        self.log(
            "startup_bc_check",
            {
                "axis": axis.upper(),
                "stage": direction,
                "target": target,
                "max_speed_mm_s": [self.limits.b_max_speed, self.limits.c_max_speed],
            },
        )
        self.port.startup_move(target, bc_sweep=True)
        self.stage = "bc_" + axis + "_" + direction

    def begin_axis_preparation(self):
        self.port.check_state(self.port.state)
        if not self.port.registered:
            raise RuntimeError("startup_origins_lost")
        if self.limits.startup_policy.get("a_startup_selftest"):
            self.store.phase("a_check_left")
            self.log("startup_a_check", {"stage": "left", "target_deg": self.limits.a_min})
            self.port.startup_move([self.limits.a_min, 0, 0], sweep=True)
            self.stage = "a_left"
        elif pose(self.port.state) == [self.limits.a_home, 0, 0]:
            # Already home: avoid a redundant enable command.
            self.store.phase("a_moving")
            self.stage = "a"
        else:
            self.store.phase("a_moving")
            self.port.startup_move([self.limits.a_home, 0, 0])
            self.stage = "a"

    def complete(self):
        self.port.check_state(self.port.state)
        self.limits.check(pose(self.port.state))
        self.store.phase("ready", ready_at=time.time(), ready_telemetry=self.port.state)
        self.log("power_cycle_preparation_complete", self.port.state)
        self.done = True

    def close(self, reason=None):
        # Stop motion even if the state disk has failed; preserve stop uncertainty.
        state_error = None
        try:
            if reason:
                self.store.invalidate(reason)
        except Exception as exc:
            state_error = str(exc)
        error = None
        if self.port:
            error = self.port.shutdown()
            self.port = None
        return error or state_error
