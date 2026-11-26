"""First-order overhead plan. No UART and no inferred metric motion completion."""

import math


def camera_triangle(distance_m, a_command, *, forward_command=73, left_command=131):
    values = (distance_m, a_command, forward_command, left_command)
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("invalid_geometry")
    if not 0 < distance_m <= 1.2 or not forward_command < left_command:
        raise ValueError("invalid_geometry")
    if not forward_command <= a_command <= left_command + 1:
        raise ValueError("target_outside_left_front_sector")
    # Both references are operator observations, not an encoder calibration.
    # One command degree must not silently be interpreted as one physical degree.
    alpha_deg = 90 * (left_command - a_command) / (left_command - forward_command)
    alpha = math.radians(alpha_deg)
    return dict(
        range_m=distance_m,
        A_command=a_command,
        forward_A_command=forward_command,
        left_A_command=left_command,
        alpha_estimate_deg=alpha_deg,
        a_forward_mm=distance_m * math.sin(alpha) * 1000,
        b_left_mm=distance_m * math.cos(alpha) * 1000,
        angle_model="two_reference_linear_approximation",
        physical_angle_calibrated=False,
        range_projection_assumption="supplied range approximates overhead camera-to-surface range",
        target_basis="measured_bucket_surface_point_not_bucket_mouth_center",
    )


def alignment_extension_plan(distance_m, a_command, *, c_max_mm=150):
    if c_max_mm != 150:
        raise ValueError("C_working_limit_is_150mm")
    geometry = camera_triangle(distance_m, a_command)
    # 131 and 132 were both accepted within the approximate left-pose tolerance.
    advance = 0.0 if a_command >= 131 else geometry["a_forward_mm"]
    sideways = geometry["b_left_mm"] / 2
    return dict(
        phase="waiting_displacement_measurement",
        output_enabled=False,
        geometry=geometry,
        sequence=[
            dict(
                action="forward_align",
                relative_target_mm=advance,
                visual_stop_A_command=131,
                keep_tracking_same_bucket=True,
            ),
            dict(action="confirm_disabled"),
            dict(
                action="left_translate",
                relative_target_mm=sideways,
                require_C_retracted=True,
                require_metric_displacement_feedback=True,
            ),
            dict(action="confirm_disabled"),
            dict(
                action="extend_C",
                absolute_target_mm=c_max_mm,
                require_disabled_chassis=True,
                require_bucket_mouth_height_ready=True,
            ),
        ],
        camera_outlet_coincident_assumption=True,
        residual_camera_to_same_surface_point_mm=sideways,
        simultaneous=False,
        firmware_interlock_preserved=True,
        blocked_by="validated metric displacement feedback not connected",
        displacement_source=None,
        elapsed_time_is_not_displacement=True,
        C_extension_sent=False,
        chassis_command_sent=False,
    )
