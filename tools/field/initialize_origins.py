"""UART origin initialization and optional acknowledged return to registered home."""

import json
import math
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


def initialize(
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
            return dict(
                initial=initial,
                final_state=final,
                zero_registered_now=False,
                origins_reused=True,
                returned_home=pose(initial) != pose(final),
                output_enabled=False,
            )
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
        return dict(
            initial=initial,
            final_state=final,
            zero_registered_now=True,
            origins_reused=False,
            output_enabled=False,
        )
    finally:
        if port is not None:
            error = port.shutdown()
            if error:
                log("initialization_shutdown_error", error)
        store.close()
