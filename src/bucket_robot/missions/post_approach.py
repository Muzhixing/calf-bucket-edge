"""Build a stopped post-approach plan; execution requires explicit opt-in.

Geometry supplies diagnostics and admission checks. The approved lateral
execution is timed and does not claim measured metric displacement.
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
        not (
            73 <= state.get("servo_software_deg", -1) <= 132
            if approach.get("mode") == "distance_first"
            else 131 <= state.get("servo_software_deg", -1) <= 133
        )
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
    from bucket_robot.missions.post_geometry import alignment_extension_plan

    plan = alignment_extension_plan(distance, state["servo_software_deg"])
    plan.update(
        approach_A_deg=state["servo_software_deg"],
        approach_mode=approach.get("mode", "left_approach"),
        trigger="completed camera-to-bucket-surface approach near 20 cm",
    )
    return plan
