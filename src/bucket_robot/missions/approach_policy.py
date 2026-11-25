"""Pure distance and alignment gates for a stopped-measurement approach."""

import math


class ApproachGate:
    def __init__(self, target=0.20, margin=0.01, max_drive=10.0):
        if target != 0.20 or not 0.005 <= margin <= 0.02 or not 0.6 <= max_drive <= 10:
            raise ValueError("invalid_approach_limits")
        self.target, self.margin, self.max_drive = target, margin, max_drive
        self.identity = None
        self.last_frame = -1
        self.drive_seconds = 0.0
        self.last_distance = None
        self.last_lower = None

    def target_check(self, now, obs, a):
        t = obs["target"]
        from bucket_robot.perception.bearing import measured_horizontal

        if t is None or t.get("partial_visible", False) and not measured_horizontal(t):
            raise RuntimeError("target_lost_or_partial")
        if not 0 <= now - t["timestamp"] <= 0.5:
            raise RuntimeError("stale_target")
        if self.identity is None:
            self.identity = obs["identity"]
        if obs["identity"] != self.identity:
            raise RuntimeError("target_identity_changed")
        dx = t["offset_px"][0]
        if not math.isfinite(dx) or abs(dx) > 25 or not 131 <= a <= 133:
            raise RuntimeError("left_alignment_lost")

    def decide(self, now, measurement):
        if measurement["frame_id"] <= self.last_frame:
            raise RuntimeError("reused_depth_frame")
        self.last_frame = measurement["frame_id"]
        # A slow neural result is usable only with a continuously checked static
        # scene and disabled chassis. Never extend the age budget while moving.
        age_limit = 5.0 if measurement.get("stationary_verified") is True else 0.5
        if not 0 <= now - measurement["capture_time"] <= age_limit:
            return "measurement_too_old"
        if not measurement.get("valid"):
            return "distance_unavailable"
        if measurement.get("distance_kind") != "stereo_optical_midpoint_to_bucket_surface":
            return "distance_basis_mismatch"
        distance, lower = measurement.get("distance_m"), measurement.get("lower_m")
        if any(
            type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (distance, lower)
        ):
            return "invalid_distance"
        if lower > distance:
            return "invalid_distance_bounds"
        # Stop on a near estimate; never require several near frames while driving.
        if distance <= self.target:
            return "target_distance_reached"
        if lower <= self.target + self.margin:
            return "distance_stop_margin"
        if self.last_distance is not None:
            if distance > self.last_distance + 0.02 or abs(distance - self.last_distance) > 0.04:
                return "unexpected_distance_change"
        self.last_distance, self.last_lower = distance, lower
        if self.drive_seconds + 0.6 > self.max_drive + 1e-9:
            return "drive_budget_exhausted"
        return "left_pulse"
