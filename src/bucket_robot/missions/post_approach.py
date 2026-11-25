"""Requested sequential post-approach actions; no time-based displacement claim.

No physical displacement adapter has been validated. This module deliberately
emits a pending plan, not UART commands, until that prerequisite is implemented.
"""

import math


def pending_extension_plan(approach):
    state = approach.get("final_state") or {}
    if (
        approach.get("error")
        or approach.get("stop_error")
        or state.get("enabled", True)
        or state.get("state") != 0
        or state.get("system_fault")
        or state.get("motor_fault")
    ):
        raise ValueError("approach_not_safely_stopped")
    if approach.get("reason") not in ("target_distance_reached", "distance_stop_margin"):
        raise ValueError("near_bucket_phase_not_completed")
    if (
        not 131 <= state.get("servo_software_deg", -1) <= 133
        or not 0 <= state.get("lift_position_mm", -1) <= 96
        or state.get("slide_position_mm") != 0
    ):
        raise ValueError("post_approach_pose_not_ready")
    measurement = approach.get("last_measurement") or {}
    distance = measurement.get("distance_m")
    if (
        not measurement.get("valid")
        or not isinstance(distance, (float, int))
        or not math.isfinite(distance)
        or measurement.get("distance_kind") != "stereo_optical_midpoint_to_bucket_surface"
    ):
        raise ValueError("no_valid_approach_measurement")
    return dict(
        phase="waiting_displacement_measurement",
        output_enabled=False,
        trigger="completed camera-to-bucket-surface approach near 20 cm",
        sequence=[
            dict(action="extend_C", absolute_target_mm=150, require_disabled_chassis=True),
            dict(action="left_translate", relative_target_mm=100, require_C_settled=True),
        ],
        simultaneous=False,
        firmware_interlock_preserved=True,
        blocked_by="validated lateral displacement feedback not connected",
        displacement_source=None,
        elapsed_time_is_not_displacement=True,
        C_extension_sent=False,
        chassis_command_sent=False,
    )
