#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thread-safe task state for manually planned frontend routes."""

import json
import threading
import time
from typing import Any, Dict, List, Optional


class TaskStateStore:
    """Holds the latest route task received from the frontend server."""

    def __init__(self, device_id: str):
        self.device_id = device_id
        self._lock = threading.Lock()
        self._task_id = None
        self._path: List[Dict[str, Any]] = []
        self._current_index = 0
        self._task_state = "idle"
        self._put_status = False
        self._last_update_ts = None
        self._signature = None
        self._last_error = None

    def apply_task(self, payload: Dict[str, Any]) -> bool:
        """Apply a task payload and return True when it changed local state."""
        task = self._extract_task(payload)
        if not task:
            return False

        task_device_id = task.get("deviceID") or task.get("deviceId")
        if task_device_id and str(task_device_id) != str(self.device_id):
            return False

        path = (
            task.get("movementSteps")
            or task.get("robotPath")
            or task.get("robotpath")
            or []
        )
        if not isinstance(path, list) or not path:
            return False

        normalized_path = self._normalize_path(path)
        if not normalized_path:
            return False

        task_id = (
            task.get("taskID")
            or task.get("taskId")
            or task.get("id")
            or f"task-{int(time.time())}"
        )
        signature = json.dumps(
            {"taskID": task_id, "path": normalized_path},
            ensure_ascii=True,
            sort_keys=True,
        )

        with self._lock:
            if signature == self._signature and self._task_state in {
                "accepted",
                "running",
                "completed",
            }:
                return False
            self._task_id = str(task_id)
            self._path = normalized_path
            self._current_index = 0
            self._task_state = "accepted"
            self._put_status = False
            self._last_error = None
            self._last_update_ts = time.time()
            self._signature = signature
            return True

    def mark_running(self):
        with self._lock:
            if self._task_state in {"accepted", "idle"}:
                self._task_state = "running"
                self._last_update_ts = time.time()

    def mark_failed(self, message: str):
        with self._lock:
            self._task_state = "failed"
            self._last_error = message
            self._last_update_ts = time.time()

    def mark_completed(self):
        with self._lock:
            self._task_state = "completed"
            self._put_status = any(p.get("action") == "feed" for p in self._path)
            self._last_update_ts = time.time()

    def advance_to_next(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._current_index + 1 >= len(self._path):
                self._task_state = "completed"
                self._put_status = any(p.get("action") == "feed" for p in self._path)
                self._last_update_ts = time.time()
                return None
            self._current_index += 1
            self._task_state = "running"
            self._last_update_ts = time.time()
            return dict(self._path[self._current_index])

    def get_current_point(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if not self._path or self._current_index >= len(self._path):
                return None
            return dict(self._path[self._current_index])

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            current = None
            if self._path and self._current_index < len(self._path):
                current = dict(self._path[self._current_index])
            return {
                "taskID": self._task_id,
                "taskState": self._task_state,
                "currentSeq": current.get("seq") if current else None,
                "currentPoint": current,
                "pathLength": len(self._path),
                "put_status": self._put_status,
                "lastTaskUpdateTs": self._last_update_ts,
                "lastTaskError": self._last_error,
            }

    def _extract_task(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return None
        data = payload.get("data")
        if isinstance(data, dict):
            return data
        return payload

    def _normalize_path(self, path: List[Any]) -> List[Dict[str, Any]]:
        points = []
        for idx, raw in enumerate(path, start=1):
            if not isinstance(raw, dict):
                continue
            point = dict(raw)
            try:
                point["seq"] = int(point.get("seq", idx))
            except (TypeError, ValueError):
                point["seq"] = idx
            point["action"] = str(point.get("action", "pass")).lower()
            if "x" not in point and "toX" in point:
                point["x"] = point.get("toX")
            if "y" not in point and "toY" in point:
                point["y"] = point.get("toY")
            for key in ("row", "col", "fromSeq", "toSeq"):
                if key in point and point[key] is not None:
                    try:
                        point[key] = int(point[key])
                    except (TypeError, ValueError):
                        pass
            for key in (
                "x",
                "y",
                "fromX",
                "fromY",
                "toX",
                "toY",
                "feedAmount",
                "turn_deg",
                "forward_m",
                "horizontal_m",
                "target_angle_deg",
                "duration_s",
                "target_distance_m",
                "center_tolerance",
                "tracking_timeout_s",
                "distance_tolerance_m",
                "arm_hold_s",
            ):
                if key in point and point[key] is not None:
                    try:
                        point[key] = float(point[key])
                    except (TypeError, ValueError):
                        pass
            for key in (
                "forward_speed",
                "horizontal_speed",
                "target_angle",
                "rudder_angle",
                "lift_rod",
                "horizontal_rod",
                "arm_extend_mm",
                "tracking_forward_speed",
                "switch_one",
                "switch_two",
            ):
                if key in point and point[key] is not None:
                    try:
                        point[key] = int(point[key])
                    except (TypeError, ValueError):
                        pass
            points.append(point)
        points.sort(key=lambda item: item.get("seq", 0))
        return points
