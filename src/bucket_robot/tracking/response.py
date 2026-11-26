"""Bounded effective A response model; software telemetry is not an encoder."""

import json
import math
from pathlib import Path
import numpy as np


def response_angles(commands, times, delay, tau):
    """Piecewise-constant command, pure delay, optional first-order response."""
    if len(commands) == 0:
        raise ValueError("empty_command_history")
    values = []
    i = 0
    demand = float(commands[0][1])
    angle = demand
    cursor = float(commands[0][0]) + delay
    for stamp in times:
        stamp = float(stamp)
        if stamp < cursor:
            values.append(angle)
            continue
        while i + 1 < len(commands) and commands[i + 1][0] + delay <= stamp:
            event = float(commands[i + 1][0]) + delay
            angle = (
                demand
                if tau == 0
                else demand + (angle - demand) * math.exp(-(event - cursor) / tau)
            )
            cursor = event
            i += 1
            demand = float(commands[i][1])
        angle = (
            demand if tau == 0 else demand + (angle - demand) * math.exp(-(stamp - cursor) / tau)
        )
        cursor = stamp
        values.append(angle)
    return np.asarray(values)


class ResponseProfile:
    def __init__(self, gain=20.0, delay=0.14, tau=0.0, scope=None):
        if not all(math.isfinite(v) for v in (gain, delay, tau)) or not (
            10 <= gain <= 35 and 0.04 <= delay <= 0.25 and 0 <= tau <= 0.15
        ):
            raise ValueError("response_parameters_out_of_bounds")
        self.gain, self.delay, self.tau, self.scope = float(gain), float(delay), float(tau), scope

    @classmethod
    def load(cls, path, calibration_sha):
        if (
            not isinstance(calibration_sha, str)
            or len(calibration_sha) != 64
            or any(c not in "0123456789abcdef" for c in calibration_sha)
        ):
            raise ValueError("invalid_calibration_digest")
        data = json.loads(Path(path).read_text())
        if (
            data.get("schema") != 1
            or data.get("source") != "field_log"
            or data.get("calibration_sha256") != calibration_sha
            or data.get("validation", {}).get("accepted") is not True
        ):
            raise ValueError("unverified_or_mismatched_response_profile")
        scope = data["scope"]
        for key, lo, hi in [("B_mm", 0, 96), ("distance_m", 0.15, 3.0)]:
            v = scope[key]
            if len(v) != 2 or not all(math.isfinite(x) for x in v) or not lo <= v[0] < v[1] <= hi:
                raise ValueError("invalid_response_scope")
        if scope.get("C_mm") != 10:
            raise ValueError("invalid_response_C_scope")
        return cls(data["gain_px_per_degree"], data["delay_s"], data["tau_s"], scope)

    def eligible(self, state, metadata):
        if self.scope is None:
            return True  # Direct construction for offline tests.
        z = metadata.get("distance_m")
        age = metadata.get("distance_age_ms")
        b = state.get("lift_position_mm")
        c = state.get("slide_position_mm")
        return (
            isinstance(b, (int, float))
            and self.scope["B_mm"][0] <= b <= self.scope["B_mm"][1]
            and c == self.scope["C_mm"]
            and isinstance(z, (int, float))
            and self.scope["distance_m"][0] <= z <= self.scope["distance_m"][1]
            and isinstance(age, (int, float))
            and 0 <= age <= 350
        )


def fit_response(data):
    """Offline identification, with held-out time segment and excitation gates.

    Caller must provide a recorded static target/base calibration, not arbitrary
    moving-bucket logs. No values are learned or activated by the live controller.
    """
    if data.get("base_stationary") is not True or data.get("target_stationary") is not True:
        raise ValueError("stationary_calibration_required")
    commands = np.asarray(data["commands"], float)
    obs = np.asarray(data["observations"], float)
    if commands.ndim != 2 or commands.shape[1] != 2 or obs.ndim != 2 or obs.shape[1] != 2:
        raise ValueError("expected_timestamp_value_pairs")
    if (
        len(commands) < 12
        or len(obs) < 90
        or not np.isfinite(commands).all()
        or not np.isfinite(obs).all()
    ):
        raise ValueError("insufficient_or_invalid_samples")
    if np.any(np.diff(commands[:, 0]) <= 0) or np.any(np.diff(obs[:, 0]) <= 0):
        raise ValueError("unordered_samples")
    if np.any((commands[:, 1] < 60) | (commands[:, 1] > 100)):
        raise ValueError("commands_out_of_envelope")
    changes = np.diff(commands[:, 1])
    sign = np.sign(changes[np.abs(changes) > 0.1])
    if np.ptp(commands[:, 1]) < 4 or np.sum(sign[1:] != sign[:-1]) < 3 or np.ptp(obs[:, 0]) < 4:
        raise ValueError("insufficient_bidirectional_excitation")
    # Drop history warmup, preserve a contiguous held-out final segment.
    obs = obs[(obs[:, 0] >= commands[0, 0] + 0.7) & (obs[:, 0] <= commands[-1, 0])]
    cut = int(0.7 * len(obs))
    if cut < 40 or len(obs) - cut < 20:
        raise ValueError("insufficient_validation_samples")
    train = obs[:cut]
    validation = obs[cut:]
    candidates = []
    for delay in np.arange(0.04, 0.251, 0.01):
        for tau in np.arange(0, 0.151, 0.025):
            a = response_angles(commands, obs[:, 0], float(delay), float(tau))
            if np.ptp(a[:cut]) < 3 or np.ptp(a[cut:]) < 2:
                continue
            design = np.c_[a[:cut], np.ones(cut)]
            gain, offset = np.linalg.lstsq(design, train[:, 1], rcond=None)[0]
            if not 10 <= gain <= 35:
                continue
            error = obs[:, 1] - (a * gain + offset)
            # Rank by training only. Validation is a gate, not a tuning set.
            rmse = float(np.sqrt(np.mean(error[:cut] ** 2)))
            candidates.append(
                (
                    rmse,
                    round(float(delay), 3),
                    round(float(tau), 3),
                    float(gain),
                    float(offset),
                    float(np.sqrt(np.mean(error[cut:] ** 2))),
                )
            )
    if not candidates:
        raise ValueError("unidentifiable_response")
    candidates.sort()
    rmse, delay, tau, gain, offset, held = candidates[0]
    baseline = response_angles(commands, obs[:, 0], 0.14, 0.0) * 20
    baseline += np.mean(train[:, 1] - baseline[:cut])
    old = float(np.sqrt(np.mean((validation[:, 1] - baseline[cut:]) ** 2)))
    ambiguous = any(
        r[0] <= rmse + 0.15
        and (abs(r[1] + r[2] - delay - tau) > 0.05 or abs(r[3] / gain - 1) > 0.15)
        for r in candidates[1:]
    )
    accepted = rmse <= 2.5 and held <= 3.0 and held < 0.8 * old and not ambiguous
    return {
        "gain_px_per_degree": gain,
        "delay_s": delay,
        "tau_s": tau,
        "validation": {
            "accepted": bool(accepted),
            "training_rmse_px": rmse,
            "heldout_rmse_px": held,
            "default_heldout_rmse_px": old,
            "ambiguous": ambiguous,
            "training_samples": cut,
            "heldout_samples": len(obs) - cut,
        },
    }
