"""UART origin initialization and optional acknowledged return to registered home."""

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
import time
import uuid

from bucket_robot.motion.arm import ArmPort, pose, reset_telemetry
from bucket_robot.runtime.startup import (
    PowerCycleStore,
    StartupPort,
    can_auto_prepare,
    derive_wait_profile,
)


def read_policy(path):
    policy = json.loads(Path(path).read_text())
    if not (
        policy.get("schema_version") == 1
        and policy.get("B_at_top_by_default") is True
        and policy.get("C_fully_retracted_by_default") is True
        and policy.get("require_repeated_position_confirmation") is False
        and policy.get("new_registration_requires_fresh_firmware_boot_signature") is True
        and policy.get("reuse_valid_registered_origins") is True
    ):
        raise RuntimeError("invalid_startup_assumption_policy")
    return policy


@dataclass(frozen=True)
class StartupResult:
    ready: bool
    stop_confirmed: bool
    error: str | None = None
    stop_error: str | None = None

    def as_dict(self):
        return asdict(self)


class StartupFailure(RuntimeError):
    def __init__(self, result):
        self.result = result
        super().__init__(result.stop_error or result.error or "startup_failed")


class StartupService:
    """Own startup preparation; UART transactions remain in StartupPort."""

    def prepare(self, limits, ledger, log, **kwargs):
        return _initialize(limits, ledger, log, **kwargs)


def initialize(limits, ledger, log, **kwargs):
    """Compatibility facade for installed and field entry points."""
    return StartupService().prepare(limits, ledger, log, **kwargs)


def _initialize(
    limits,
    ledger,
    log,
    *,
    port_factory=StartupPort,
    wait_seconds=None,
    timeout=18,
    return_home=False,
):
    profile = derive_wait_profile(limits)
    wait = profile["wait_seconds"] if wait_seconds is None else wait_seconds
    store = PowerCycleStore(ledger, profile["wait_seconds"], profile["wait_basis"])
    port = None
    try:
        if store.error:
            raise RuntimeError(store.error)
        if store.data.get("stop_error"):
            raise RuntimeError("STOP_UNCONFIRMED_prior_startup")
        if store.data.get("operator_paused"):
            raise RuntimeError("operator_paused")
        port = port_factory(limits, log)
        port.return_registered_home = return_home
        began = time.monotonic()
        while True:
            port.poll()
            state = port.state
            if state is not None:
                flags = [state["lift_zero_confirmed"], state["slide_zero_confirmed"]]
                if flags[0] != flags[1]:
                    raise RuntimeError("partial_origin_registration_requires_diagnosis")
                reused = all(flags)
                port.reuse_existing_origins = reused
                if (reused or time.monotonic() - began >= wait) and port.preflight_ready():
                    break
            if time.monotonic() - began >= timeout:
                raise RuntimeError("startup_preflight_timeout")
            time.sleep(0.01)
        initial = dict(port.state)
        if reused:
            port.registered = True
            port.watch_firmware_reset = True
            if return_home and pose(initial) != [limits.a_home, 0, 0]:
                log("registered_home_return_started", initial)
                until = time.monotonic() + 90
                while pose(port.state) != [limits.a_home, 0, 0]:
                    if time.monotonic() > until:
                        raise RuntimeError("registered_home_return_timeout")
                    while time.monotonic() < port.ready_at:
                        port.tick()
                        time.sleep(0.01)
                    a, b, c = pose(port.state)
                    if c:
                        goal = [a, b, 0]
                    elif b:
                        goal = [a, max(0, math.ceil(b - 20)), 0]
                    else:
                        goal = [a + max(-4, min(4, limits.a_home - a)), 0, 0]
                    port.registered_return_step(goal)
                    while port.phase != "idle":
                        if time.monotonic() > until:
                            raise RuntimeError("registered_home_return_timeout")
                        port.tick()
                        time.sleep(0.01)
                log("registered_home_return_completed", port.state)
            final = dict(port.state)
            if not store.ready:
                store.invalidate("adopt_fresh_registered_home")
                store.adopt_registered(final, limits.a_home)
                store.phase("ready", ready_at=time.time(), ready_telemetry=final)
            log("valid_origins_reused_without_registration", final)
            result = dict(
                initial=initial,
                final_state=final,
                zero_registered_now=False,
                origins_reused=True,
                returned_home=pose(initial) != pose(final),
                output_enabled=False,
            )
            return result
        if not reset_telemetry(initial, limits.boot_pose):
            raise RuntimeError("new_origin_registration_requires_firmware_boot_signature")
        if not can_auto_prepare(store):
            raise RuntimeError("prior_origin_registration_unconfirmed_no_retry")
        store.confirm(str(uuid.uuid4()), event="automatic_mcu_restart")
        store.phase("disable_pending", physical_assumption="standing_user_B_top_C_retracted")

        def wait_ack():
            while port.phase != "idle":
                port.tick()
                time.sleep(0.01)

        port.disabled_command(0)
        wait_ack()
        store.before_zero()  # Persistent write-ahead marker; never blind retry flags6.
        log("origin_registration_under_standing_user_assumption", store.status())
        port.disabled_command(6)
        wait_ack()
        final = dict(port.state)
        ArmPort.check_state(final)
        if final["enabled"] or pose(final) != [limits.a_home, 0, 0]:
            raise RuntimeError("initialization_not_disabled_home")
        store.phase("ready", origins_observed=True, ready_at=time.time(), ready_telemetry=final)
        result = dict(
            initial=initial,
            final_state=final,
            zero_registered_now=True,
            origins_reused=False,
            output_enabled=False,
        )
        return result
    except Exception as exc:
        if isinstance(exc, StartupFailure):
            raise
        raise StartupFailure(StartupResult(False, port is None, error=str(exc))) from exc
    finally:
        pending = sys.exc_info()[1]
        stop_error = store.data.get("stop_error")
        try:
            if port is not None:
                try:
                    stop_error = port.shutdown()
                except BaseException as exc:
                    stop_error = repr(exc)
                if stop_error:
                    if not store.error:
                        try:
                            store.phase("uncertain", stop_error=stop_error)
                        except Exception:
                            pass  # The returned failure also crosses the process boundary.
                    try:
                        log("initialization_shutdown_error", stop_error)
                    except Exception:
                        pass  # Broken logging cannot discard the stop-failure result.
            if isinstance(pending, StartupFailure):
                pending.result = StartupResult(
                    False, not bool(stop_error), pending.result.error, stop_error
                )
            if stop_error:
                raise StartupFailure(
                    StartupResult(False, False, str(pending) if pending else None, stop_error)
                ) from pending
            if pending is None:
                result.update(StartupResult(True, True).as_dict())
        finally:
            store.close()
