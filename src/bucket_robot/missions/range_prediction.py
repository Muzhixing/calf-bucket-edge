"""Capture-time range estimates and continuous speed caps, without odometry claims."""

from collections import deque
import math
import statistics


class RangePrediction:
    def __init__(self, cruise=8):
        if type(cruise) is not int or not 1 <= cruise <= 20:
            raise ValueError("invalid_cruise_speed")
        self.cruise = cruise
        self.samples = deque(maxlen=5)
        self.identity = None
        self.last_frame = -1

    def reset(self):
        self.samples.clear()
        self.last_frame = -1

    def add(self, identity, measurement):
        m = measurement
        if identity != self.identity:
            self.reset()
            self.identity = identity
        if (
            not m.get("valid")
            or m.get("distance_kind") != "stereo_optical_midpoint_to_bucket_surface"
        ):
            return False
        if any(
            type(m.get(k)) not in (int, float) or not math.isfinite(m[k])
            for k in ("capture_time", "distance_m", "lower_m")
        ):
            return False
        if not 0 < m["lower_m"] <= m["distance_m"] or m["frame_id"] <= self.last_frame:
            return False
        if self.samples and m["capture_time"] <= self.samples[-1]["capture_time"]:
            return False
        self.samples.append(dict(m))
        self.last_frame = m["frame_id"]
        return True

    def estimate(self, now, width_ratio=1.0):
        if not self.samples:
            return dict(permit=False, near=False, speed_cap=0, reason="waiting_range")
        m = self.samples[-1]
        age = now - m["capture_time"]
        if not 0 <= age <= 8:
            return dict(permit=False, near=False, speed_cap=0, reason="range_expired")
        # Rates are measured distances / capture-time differences, never UART
        # percentages converted into metres. Median slopes suppress isolated noise.
        rates = [
            (b["distance_m"] - a["distance_m"]) / (b["capture_time"] - a["capture_time"])
            for a, b in zip(self.samples, list(self.samples)[1:])
            if 0.3 <= b["capture_time"] - a["capture_time"] <= 8
        ]
        rate = statistics.median(rates) if rates else 0.0
        if abs(rate) > 0.08 or not 0.6 <= width_ratio <= 1.5:
            return dict(permit=False, near=False, speed_cap=0, reason="range_motion_inconsistent")
        temporal = m["distance_m"] + min(0.0, rate) * age
        # Expansion provides a second, current-image cue while the next model
        # result is pending. It only reduces distance; shrinking cannot grant
        # extra clearance. The model remains the absolute distance reference.
        scale = m["distance_m"] * min(1.0, width_ratio)
        predicted = max(0.0, min(temporal, scale))
        closing = max(0.0, -rate)
        spread = statistics.median(abs(v - rate) for v in rates) if rates else 0.0
        uncertainty = (m["distance_m"] - m["lower_m"]) + spread * age
        lower = max(0.0, predicted - uncertainty)
        latency = min(4.0, max(0.1, m.get("elapsed_s", age)))
        lookahead = lower - closing * latency
        # Keep the near command above the unvalidated low-speed stiction range.
        # 6% produced measured approach on the loaded chassis during acceptance.
        cap = 8 if lookahead >= 0.35 else 7 if lookahead >= 0.30 else 6
        if self.cruise > 8:
            if lower >= 0.40:
                cap = self.cruise
            elif lower >= 0.25:
                cap = int(math.floor(10 + (self.cruise - 10) * (lower - 0.25) / 0.15 + 1e-8))
            else:
                cap = max(0, int(math.floor(10 * (lower - 0.20) / 0.05 + 1e-8)))
        cap = min(self.cruise, cap)
        return dict(
            permit=True,
            near=False,
            speed_cap=cap,
            reason="continuous_range",
            measurement_age_s=age,
            closing_mps=closing,
            predicted_distance_m=predicted,
            predicted_lower_m=lower,
            uncertainty_m=uncertainty,
            capture_distance_m=m["distance_m"],
            capture_time=m["capture_time"],
            prediction_time=now,
            verify_stop=lower <= 0.205,
        )
