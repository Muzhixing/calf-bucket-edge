#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STM32 serial control frame output."""

import json
import math
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import serial
except Exception:  # pragma: no cover - pyserial is optional in unit tests
    serial = None

from app.domain.events import RangingUpdateEvent
from app.infrastructure.bucket_tracking import (
    BucketTrackingConfig,
    BucketTrackingController,
    is_tracking_point,
)

FRAME_HEAD = 0xB3
FRAME_TAIL = 0xB4
FRAME_SIZE = 10
TELEMETRY_HEAD = 0xB5
TELEMETRY_TAIL = 0xB6
TELEMETRY_SIZE = 10
TELEMETRY_STATUS_DISTANCE_VALID = 0x01
TELEMETRY_STATUS_CALIBRATED = 0x02
TELEMETRY_STATUS_PUMP_REQUESTED = 0x04
TELEMETRY_STATUS_FORCED_OFF = 0x08
NEUTRAL_RUDDER_DEG = 90
MAX_ROD_MM = 180
# STM32 field test verified this as the stop command on 2026-07-07.
STOP_FORWARD_SPEED = 0x10
STOP_FRAME = bytes(
    [
        FRAME_HEAD,
        STOP_FORWARD_SPEED,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        FRAME_TAIL,
    ]
)
# Field-verified motion frame. The STM32 firmware accepted this exact frame
# from both PC and board USB-UART tests.
KNOWN_GOOD_MOTION_FRAME = bytes.fromhex("B3 1E 14 2D 78 32 1E 00 00 B4")
DEFAULT_INTER_FRAME_DELAY_S = 5.0


@dataclass(frozen=True)
class Stm32Telemetry:
    version: int
    raw_distance_mm: int
    distance_mm: Optional[int]
    level_raw: int
    level_percent: Optional[int]
    pump_on: bool
    status: int
    distance_valid: bool
    calibrated: bool
    pump_requested: bool
    forced_off: bool
    sequence: int
    reserved: int
    raw: bytes

    def as_dict(self):
        return {
            "version": self.version,
            "raw_distance_mm": self.raw_distance_mm,
            "distance_mm": self.distance_mm,
            "level_raw": self.level_raw,
            "level_percent": self.level_percent,
            "pump_on": self.pump_on,
            "status": self.status,
            "distance_valid": self.distance_valid,
            "calibrated": self.calibrated,
            "pump_requested": self.pump_requested,
            "forced_off": self.forced_off,
            "sequence": self.sequence,
            "reserved": self.reserved,
            "raw": self.raw,
        }


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


def frame_to_hex(frame) -> str:
    return bytes(frame or b"").hex(" ").upper()


def parse_control_frame(value):
    if isinstance(value, bytes):
        frame = value
    elif isinstance(value, bytearray):
        frame = bytes(value)
    elif isinstance(value, (list, tuple)):
        try:
            frame = bytes(int(item) & 0xFF for item in value)
        except (TypeError, ValueError) as exc:
            raise ValueError("control frame list must contain byte values") from exc
    elif isinstance(value, str):
        compact = re.sub(r"[^0-9a-fA-F]", "", value.replace("0x", "").replace("0X", ""))
        if len(compact) % 2:
            raise ValueError("control frame hex string must contain whole bytes")
        frame = bytes.fromhex(compact)
    else:
        raise ValueError("control frame must be bytes, list, tuple, or hex string")

    if len(frame) != FRAME_SIZE:
        raise ValueError(f"control frame must be {FRAME_SIZE} bytes, got {len(frame)}")
    if frame[0] != FRAME_HEAD or frame[-1] != FRAME_TAIL:
        raise ValueError("control frame must start with B3 and end with B4")
    return frame


def _target_angle_to_byte(angle_deg):
    # STM32 stores target_angle directly as an 8-bit value.
    angle = _to_float(angle_deg, 0.0)
    return int(round(_clamp(angle, 0.0, 255.0)))


def _find_best_detection(event: RangingUpdateEvent):
    detections = list(event.detections or [])
    if not detections:
        return None
    return max(detections, key=lambda item: _to_float(item.get("score"), 0.0) or 0.0)


def parse_telemetry_frame(frame):
    data = bytes(frame or b"")
    if len(data) != TELEMETRY_SIZE:
        return None
    if data[0] != TELEMETRY_HEAD or data[-1] != TELEMETRY_TAIL:
        return None

    raw_distance_mm = (data[2] << 8) | data[3]
    status = data[6]
    distance_valid = bool(status & TELEMETRY_STATUS_DISTANCE_VALID)
    level_raw = data[4]

    return Stm32Telemetry(
        version=data[1],
        raw_distance_mm=raw_distance_mm,
        distance_mm=raw_distance_mm if distance_valid else None,
        level_raw=level_raw,
        level_percent=None if level_raw == 0xFF else level_raw,
        pump_on=bool(data[5]),
        status=status,
        distance_valid=distance_valid,
        calibrated=bool(status & TELEMETRY_STATUS_CALIBRATED),
        pump_requested=bool(status & TELEMETRY_STATUS_PUMP_REQUESTED),
        forced_off=bool(status & TELEMETRY_STATUS_FORCED_OFF),
        sequence=data[7],
        reserved=data[8],
        raw=data,
    )


def extract_telemetry_frames(stream):
    data = bytes(stream or b"")
    frames = []
    index = 0
    while index <= len(data) - TELEMETRY_SIZE:
        packet = parse_telemetry_frame(data[index : index + TELEMETRY_SIZE])
        if packet is not None:
            frames.append(packet)
            index += TELEMETRY_SIZE
        else:
            index += 1
    return frames


class SerialPortGateway:
    """Encodes route and vision state into the STM32 10-byte control frame."""

    def __init__(
        self,
        port="/dev/myserial",
        baudrate=115200,
        enabled=False,
        send_hz=10.0,
        rudder_max_delta_deg=45.0,
        serial_factory=None,
        dry_run=False,
        log_frames=False,
        command_repeats=1,
        inter_frame_delay_s=DEFAULT_INTER_FRAME_DELAY_S,
        stop_repeats=1,
        bucket_tracking_config=None,
    ):
        self.port = port or "/dev/myserial"
        self.baudrate = int(baudrate)
        self.enabled = bool(enabled)
        self.send_hz = max(1.0, float(send_hz))
        self.rudder_max_delta_deg = max(0.0, float(rudder_max_delta_deg))
        self.serial_factory = serial_factory
        self.dry_run = bool(dry_run)
        self.log_frames = bool(log_frames)
        self.command_repeats = max(1, int(command_repeats))
        self.inter_frame_delay_s = max(0.0, float(inter_frame_delay_s))
        self.stop_repeats = max(1, int(stop_repeats))
        if bucket_tracking_config is None:
            bucket_tracking_config = BucketTrackingConfig()
        self.bucket_tracking_config = bucket_tracking_config
        self.bucket_tracker = BucketTrackingController(bucket_tracking_config)

        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._task_snapshot: Optional[Dict[str, Any]] = None
        self._latest_event: Optional[RangingUpdateEvent] = None
        self._serial = None
        self._thread = None
        self._stop = threading.Event()
        self._last_frame_sent_at = None
        self._last_frame_hex = None
        self._sent_frame_count = 0
        self._last_active_frame = False
        self._task_generation = 0
        self._sent_generation = 0
        self._command_key = None

    def start(self):
        if not self.enabled or self._thread is not None:
            return
        if not self.dry_run:
            self._open_serial()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="stm32-serial", daemon=True
        )
        self._thread.start()
        print(
            f"STM32 串口发送已启用: {self.port} @ {self.baudrate}, {self.send_hz:.1f}Hz"
        )

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._send_stop_frames()
        if self._serial is not None:
            try:
                self._serial.close()
            finally:
                self._serial = None

    def handle_ranging_update(self, event: RangingUpdateEvent):
        with self._lock:
            self._latest_event = event

    def handle_task_update(self, task_snapshot):
        if not isinstance(task_snapshot, dict):
            return
        command_key = self._snapshot_command_key(task_snapshot)
        with self._lock:
            if command_key == self._command_key:
                return
            self._task_snapshot = dict(task_snapshot)
            self._command_key = command_key
            self._task_generation += 1

    def clear_task(self):
        with self._lock:
            self._task_snapshot = None
            self._command_key = None
            self._task_generation += 1
        self.bucket_tracker.reset()

    def get_sent_stats(self):
        with self._lock:
            return {
                "last_frame_sent_at": self._last_frame_sent_at,
                "last_frame_hex": self._last_frame_hex,
                "sent_frame_count": self._sent_frame_count,
                **self.bucket_tracker.get_status(),
            }

    def build_current_frame(self):
        with self._lock:
            snapshot = (
                dict(self._task_snapshot) if self._task_snapshot is not None else None
            )
            event = self._latest_event
        if snapshot is None:
            return None
        if event is None:
            event = self._default_event()
        point = snapshot.get("currentPoint")
        if not isinstance(point, dict):
            return None
        return self.build_frame(point, event)

    def _build_pending_frame(self):
        with self._lock:
            if self._task_snapshot is None:
                return None, None
            snapshot = dict(self._task_snapshot)
            event = self._latest_event
            generation = self._task_generation
            command_key = self._command_key
        if event is None:
            event = self._default_event()
        point = snapshot.get("currentPoint")
        if not isinstance(point, dict):
            return None, None
        if self.bucket_tracking_config.enabled and is_tracking_point(point):
            result = self.bucket_tracker.update(command_key, point, event)
            return result.frame, generation
        if generation == self._sent_generation:
            return None, None
        return self.build_frame(point, event), generation

    def build_frame(self, task_point: Dict[str, Any], event: RangingUpdateEvent):
        direct_frame = self._direct_frame_from_task(task_point)
        if direct_frame is not None:
            return direct_frame

        best = (
            _find_best_detection(event)
            if event is not None and event.detected
            else None
        )
        vision = (
            self._vision_control_fields(best, event)
            if best is not None and event is not None
            else self._default_vision_fields()
        )
        frame = bytes(
            [
                FRAME_HEAD,
                self._task_forward_speed(task_point),
                self._task_horizontal_speed(task_point),
                self._task_target_angle(task_point),
                _to_byte(task_point.get("rudder_angle"), vision["rudder_angle"]),
                _to_byte(task_point.get("lift_rod"), vision["lift_rod"]),
                _to_byte(task_point.get("horizontal_rod"), vision["horizontal_rod"]),
                _to_byte(task_point.get("switch_one"), 0),
                _to_byte(task_point.get("switch_two"), 0),
                FRAME_TAIL,
            ]
        )
        return frame

    def _run(self):
        interval = 1.0 / self.send_hz
        while not self._stop.is_set():
            started = time.monotonic()
            frame, generation = self._build_pending_frame()
            if frame is not None:
                if self._write_frame_repeated(frame, self.command_repeats):
                    if self._should_mark_generation_sent(generation):
                        self._mark_generation_sent(generation)
                    self._last_active_frame = True
            elif self._last_active_frame:
                with self._lock:
                    has_active_task = self._task_snapshot is not None
                if not has_active_task:
                    self._send_stop_frames()
                    self._last_active_frame = False
            elapsed = time.monotonic() - started
            self._stop.wait(max(0.0, interval - elapsed))

    def _open_serial(self):
        if self.serial_factory is not None:
            self._serial = self.serial_factory(self.port, self.baudrate)
            return
        if serial is None:
            print("pyserial 未安装，STM32 串口发送不可用。")
            return
        try:
            self._serial = serial.Serial(
                self.port,
                self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False,
                timeout=0,
                write_timeout=0.2,
            )
        except Exception as exc:
            print(f"打开 STM32 串口失败: {self.port} @ {self.baudrate}: {exc}")
            self._serial = None

    def _write_frame(self, frame):
        frame = parse_control_frame(frame)
        if self.log_frames:
            print("[stm32 frame]", frame_to_hex(frame))
        if self.dry_run:
            self._record_sent_frame(frame)
            return True
        if self._serial is None:
            return False
        try:
            with self._write_lock:
                self._serial.write(frame)
            self._record_sent_frame(frame)
            return True
        except Exception as exc:
            print(f"STM32 串口写入失败: {exc}")
            return False

    def write_control_frame(self, frame, repeats=1, inter_frame_delay_s=None):
        return self._write_frame_repeated(frame, repeats, inter_frame_delay_s)

    def send_known_good_motion(self, repeats=1, inter_frame_delay_s=None):
        return self.write_control_frame(
            KNOWN_GOOD_MOTION_FRAME, repeats, inter_frame_delay_s
        )

    def send_verified_stop(self, repeats=None, inter_frame_delay_s=None):
        return self._send_stop_frames(
            repeats=repeats, inter_frame_delay_s=inter_frame_delay_s
        )

    def _write_frame_repeated(self, frame, repeats=1, inter_frame_delay_s=None):
        frame = parse_control_frame(frame)
        count = max(1, int(repeats or 1))
        delay = (
            self.inter_frame_delay_s
            if inter_frame_delay_s is None
            else max(0.0, float(inter_frame_delay_s))
        )
        ok = True
        for index in range(count):
            ok = self._write_frame(frame) and ok
            if index + 1 < count and delay > 0:
                time.sleep(delay)
        return ok

    def _send_stop_frames(self, repeats=None, inter_frame_delay_s=None):
        count = self.stop_repeats if repeats is None else max(1, int(repeats))
        return self._write_frame_repeated(STOP_FRAME, count, inter_frame_delay_s)

    def _record_sent_frame(self, frame):
        with self._lock:
            self._last_frame_sent_at = time.monotonic()
            self._last_frame_hex = frame_to_hex(frame)
            self._sent_frame_count += 1

    def _mark_generation_sent(self, generation):
        with self._lock:
            if generation is not None and generation > self._sent_generation:
                self._sent_generation = generation

    def _should_mark_generation_sent(self, generation):
        with self._lock:
            if generation is None or self._task_snapshot is None:
                return False
            point = self._task_snapshot.get("currentPoint")
        if self.bucket_tracking_config.enabled and is_tracking_point(point):
            status = self.bucket_tracker.get_status()
            return bool(
                status.get("tracking_completed") or status.get("tracking_failed")
            )
        return True

    def _snapshot_command_key(self, snapshot):
        current_point = (
            snapshot.get("currentPoint") if isinstance(snapshot, dict) else None
        )
        return json.dumps(
            {
                "taskID": (
                    snapshot.get("taskID") if isinstance(snapshot, dict) else None
                ),
                "currentSeq": (
                    snapshot.get("currentSeq") if isinstance(snapshot, dict) else None
                ),
                "currentPoint": (
                    current_point if isinstance(current_point, dict) else None
                ),
            },
            ensure_ascii=True,
            sort_keys=True,
            default=str,
        )

    def _direct_frame_from_task(self, point):
        if not isinstance(point, dict):
            return None
        action = str(point.get("action", "") or "").strip().lower()
        if action in {"stop", "halt", "brake"}:
            return STOP_FRAME
        if action in {
            "known_good_motion",
            "known-good-motion",
            "motion_test",
            "test_motion",
        }:
            return KNOWN_GOOD_MOTION_FRAME
        for key in ("frame_hex", "control_frame", "raw_frame"):
            value = point.get(key)
            if value is not None and value != "":
                return parse_control_frame(value)
        return None

    def _task_forward_speed(self, point):
        return _to_byte(point.get("forward_speed"), 0)

    def _task_horizontal_speed(self, point):
        if "horizontal_speed" in point:
            return _to_byte(point.get("horizontal_speed"), 0)
        horizontal_m = _to_float(point.get("horizontal_m"), 0.0) or 0.0
        if math.isclose(horizontal_m, 0.0, abs_tol=1e-9):
            return 0
        return 255 if horizontal_m > 0 else 0

    def _task_target_angle(self, point):
        if "target_angle" in point:
            return _to_byte(point.get("target_angle"), 0)
        return _target_angle_to_byte(point.get("target_angle_deg"))

    def _vision_control_fields(self, detection, event: RangingUpdateEvent):
        image_width = int(event.image_width or 1280)
        image_height = int(event.image_height or 720)
        left = _to_float(detection.get("left"), 0.0) or 0.0
        right = _to_float(detection.get("right"), left) or left
        bottom = _to_float(detection.get("bottom"), image_height) or image_height
        center_x = (left + right) / 2.0
        image_center_x = image_width / 2.0
        half_width = max(1.0, image_width / 2.0)
        center_error = _clamp((center_x - image_center_x) / half_width, -1.0, 1.0)

        rudder = NEUTRAL_RUDDER_DEG + center_error * self.rudder_max_delta_deg
        lift_gap_ratio = _clamp(
            (image_height - bottom) / max(1.0, image_height), 0.0, 1.0
        )
        lift_rod = lift_gap_ratio * MAX_ROD_MM
        horizontal_rod = self._horizontal_rod_from_offset(
            center_x, image_width, event.distance_m
        )

        return {
            "rudder_angle": int(round(_clamp(rudder, 0, 180))),
            "lift_rod": int(round(_clamp(lift_rod, 0, MAX_ROD_MM))),
            "horizontal_rod": horizontal_rod,
        }

    def _horizontal_rod_from_offset(self, center_x, image_width, distance_m):
        distance = _to_float(distance_m)
        if distance is None or distance <= 0:
            return 0
        fx, cx = self._camera_fx_cx(image_width)
        offset_mm = (float(center_x) - cx) * distance * 1000.0 / max(1.0, fx)
        return int(round(_clamp(abs(offset_mm), 0, MAX_ROD_MM)))

    def _camera_fx_cx(self, image_width):
        try:
            from app.infrastructure import camera_config

            return float(camera_config.P1[0, 0]), float(camera_config.P1[0, 2])
        except Exception:
            return 698.2342, float(image_width) / 2.0

    def _default_event(self):
        return RangingUpdateEvent(
            frame_ts=time.time(),
            detected=False,
            distance_m=None,
            detections=[],
            image_width=1280,
            image_height=720,
        )

    def _default_vision_fields(self):
        return {
            "rudder_angle": NEUTRAL_RUDDER_DEG,
            "lift_rod": 0,
            "horizontal_rod": 0,
        }
