"""The operator-approved lateral action, separate from geometric diagnostics."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class TimedTranslate:
    speed_percent: int = 40
    duration_s: float = 2.0
    direction: str = "left"
    completion_basis: str = "elapsed_time_then_acknowledged_disable"
    policy: str = "operator_40percent_left_2s_20261127"

    def as_dict(self):
        return asdict(self)


def check_geometry_admission(diagnostic_half_left_mm):
    value = diagnostic_half_left_mm
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("invalid_lateral_target")
    if not 19.5 <= value <= 130:
        raise ValueError("lateral_target_outside_timing_trial_range")
