#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure bucket-following control logic for STM32 frame generation."""

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.domain.events import RangingUpdateEvent

FRAME_HEAD = 0xB3
FRAME_TAIL = 0xB4
NEUTRAL_RUDDER_DEG = 90
MAX_ROD_MM = 180
TRACKING_ACTIONS = {"feed", "track_bucket", "bucket_track"}


def is_tracking_action(action) -> bool:
    return str(action or "").strip().lower() in TRACKING_ACTIONS


def is_tracking_point(point) -> bool:
    return isinstance(point, dict) and is_tracking_action(point.get("action"))


def _clamp(value, low, high):
    return max(low, min(high, value))


def _to_float(value, default=None):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_byte(value, default=0):
    number = _to_float(value, default)
    if number is None:
        number = default
    return int(round(_clamp(number, 0, 255)))


def _best_detection(event: Optional[RangingUpdateEvent]):
    if event is None or not event.detected:
        return None
    detections = list(event.detections or [])
    if not detections:
        return None
    return max(detections, key=lambda item: _to_float(item.get("score"), 0.0) or 0.0)


@dataclass(frozen=True)
class BucketTrackingConfig:
    enabled: bool = True
    center_tolerance: float = 0.08
    target_distance_m: float = 0.45
    distance_tolerance_m: float = 0.05
    track_timeout_s: float = 20.0
    lost_timeout_s: float = 1.0
    forward_speed: int = 30
    rudder_max_delta_deg: float = 45.0
    arm_extend_mm: int = 180
    arm_hold_s: float = 1.0


@dataclass(frozen=True)
class BucketTrackingResult:
    frame: bytes
    state: str
    detail: str
    completed: bool = False
    failed: bool = False


class BucketTrackingController:
    """Stateful controller that turns bucket detections into STM32 frames."""

    def __init__(self, config: Optional[BucketTrackingConfig] = None):
        self.config = config or BucketTrackingConfig()
        self._command_key = None
        self._started_at = None
        self._arm_started_at = None
        self._last_seen_at = None
        self._last_result: Optional[BucketTrackingResult] = None

    def reset(self):
        self._command_key = None
        self._started_at = None
        self._arm_started_at = None
        self._last_seen_at = None
        self._last_result = None

    def get_status(self) -> Dict[str, Any]:
        result = self._last_result
        return {
            "tracking_state": result.state if result is not None else None,
            "tracking_detail": result.detail if result is not None else None,
            "tracking_completed": (
                bool(result.completed) if result is not None else False
            ),
            "tracking_failed": bool(result.failed) if result is not None else False,
        }

    def update(
        self,
        command_key,
        task_point: Dict[str, Any],
        event: Optional[RangingUpdateEvent],
        now=None,
    ):
        now = time.monotonic() if now is None else float(now)
        if command_key != self._command_key:
            self._command_key = command_key
            self._started_at = now
            self._arm_started_at = None
            self._last_seen_at = None
            self._last_result = None

        timeout_s = max(
            0.1,
            _to_float(
                task_point.get("tracking_timeout_s"), self.config.track_timeout_s
            ),
        )
        if self._started_at is not None and now - self._started_at > timeout_s:
            result = BucketTrackingResult(
                frame=self._stop_frame(),
                state="FAILED",
                detail=f"tracking timeout after {timeout_s:.1f}s",
                failed=True,
            )
            self._last_result = result
            return result

        detection = _best_detection(event)
        if detection is None:
            if self._last_seen_at is None:
                detail = "bucket not detected"
            else:
                detail = f"bucket lost for {now - self._last_seen_at:.2f}s"
            result = BucketTrackingResult(
                frame=self._stop_frame(),
                state="SEARCHING",
                detail=detail,
            )
            self._last_result = result
            return result

        self._last_seen_at = now
        image_width = int(event.image_width or 1280)
        left = _to_float(detection.get("left"), 0.0) or 0.0
        right = _to_float(detection.get("right"), left) or left
        center_x = (left + right) / 2.0
        image_center_x = image_width / 2.0
        half_width = max(1.0, image_width / 2.0)
        center_error = _clamp((center_x - image_center_x) / half_width, -1.0, 1.0)

        center_tolerance = max(
            0.0,
            _to_float(task_point.get("center_tolerance"), self.config.center_tolerance),
        )
        target_distance_m = max(
            0.0,
            _to_float(
                task_point.get("target_distance_m"), self.config.target_distance_m
            ),
        )
        distance_tolerance_m = max(
            0.0,
            _to_float(
                task_point.get("distance_tolerance_m"), self.config.distance_tolerance_m
            ),
        )
        distance_m = _to_float(event.distance_m)
        centered = abs(center_error) <= center_tolerance
        close_enough = (
            distance_m is not None
            and distance_m <= target_distance_m + distance_tolerance_m
        )

        rudder = self._rudder_from_error(center_error, task_point)
        if not centered:
            result = BucketTrackingResult(
                frame=self._frame(
                    task_point,
                    forward_speed=self._tracking_speed(task_point),
                    rudder_angle=rudder,
                ),
                state="CENTERING",
                detail=f"center_error={center_error:.3f}",
            )
            self._last_result = result
            return result

        if not close_enough:
            if distance_m is None:
                detail = "centered; distance unavailable"
            else:
                detail = f"centered; distance={distance_m:.3f}m target={target_distance_m:.3f}m"
            result = BucketTrackingResult(
                frame=self._frame(
                    task_point,
                    forward_speed=self._tracking_speed(task_point),
                    rudder_angle=rudder,
                ),
                state="APPROACHING",
                detail=detail,
            )
            self._last_result = result
            return result

        if self._arm_started_at is None:
            self._arm_started_at = now
        hold_s = max(
            0.0, _to_float(task_point.get("arm_hold_s"), self.config.arm_hold_s)
        )
        completed = now - self._arm_started_at >= hold_s
        result = BucketTrackingResult(
            frame=self._frame(
                task_point,
                forward_speed=0,
                rudder_angle=rudder,
                lift_rod=_to_byte(task_point.get("lift_rod"), 0),
                horizontal_rod=self._arm_extend(task_point),
            ),
            state="COMPLETED" if completed else "ARM_EXTENDING",
            detail=f"arm_extend_mm={self._arm_extend(task_point)} hold={now - self._arm_started_at:.2f}/{hold_s:.2f}s",
            completed=completed,
        )
        self._last_result = result
        return result

    def _tracking_speed(self, task_point):
        return _to_byte(
            task_point.get("tracking_forward_speed"), self.config.forward_speed
        )

    def _rudder_from_error(self, center_error, task_point):
        max_delta = max(
            0.0,
            _to_float(
                task_point.get("rudder_max_delta_deg"), self.config.rudder_max_delta_deg
            ),
        )
        return int(round(_clamp(NEUTRAL_RUDDER_DEG + center_error * max_delta, 0, 180)))

    def _arm_extend(self, task_point):
        return _to_byte(
            task_point.get("horizontal_rod"),
            task_point.get("arm_extend_mm", self.config.arm_extend_mm),
        )

    def _frame(
        self, task_point, forward_speed, rudder_angle, lift_rod=0, horizontal_rod=0
    ):
        return bytes(
            [
                FRAME_HEAD,
                _to_byte(forward_speed, 0),
                _to_byte(task_point.get("horizontal_speed"), 0),
                _to_byte(
                    task_point.get("target_angle"),
                    _to_float(task_point.get("target_angle_deg"), 0.0) or 0,
                ),
                _to_byte(task_point.get("rudder_angle"), rudder_angle),
                _to_byte(lift_rod, 0),
                _to_byte(horizontal_rod, 0),
                _to_byte(task_point.get("switch_one"), 0),
                _to_byte(task_point.get("switch_two"), 0),
                FRAME_TAIL,
            ]
        )

    def _stop_frame(self):
        return bytes([FRAME_HEAD, 0x10, 0, 0, 0, 0, 0, 0, 0, FRAME_TAIL])
