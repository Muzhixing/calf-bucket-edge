#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Periodic HTTP status reporting to the frontend server."""

import json
import threading
import time
import urllib.request
from datetime import datetime, timezone


class StatusReporter:
    """Posts latest board, ranging, and route-task state as JSON."""

    def __init__(
        self,
        status_url,
        board_config,
        state_store,
        task_state=None,
        interval=2.0,
        timeout=2.0,
    ):
        self.status_url = status_url
        self.board_config = board_config
        self.state_store = state_store
        self.task_state = task_state
        self.interval = max(0.5, float(interval))
        self.timeout = float(timeout)
        self._stop = threading.Event()
        self._thread = None
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def start(self):
        if not self.status_url or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="status-reporter", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def build_payload(self):
        task = self.task_state.snapshot() if self.task_state is not None else {}
        current_point = task.get("currentPoint") or {}
        payload = {
            "deviceID": self.board_config.device_id,
            "deviceName": self.board_config.device_name,
            "farmID": self.board_config.farm_id,
            "status": True,
            "put_status": bool(task.get("put_status", False)),
            "lastOnlineTime": datetime.now(timezone.utc)
            .astimezone()
            .isoformat(timespec="seconds"),
            "location_x": self.board_config.location_x,
            "location_y": self.board_config.location_y,
            "detected": self.state_store.get_latest_detected(),
            "distance_m": self.state_store.get_latest_distance(),
            "detections": self._format_detections(
                self.state_store.get_latest_detections()
            ),
            "frame_id": self.state_store.get_latest_frame_id(),
            "frame_ts": self.state_store.get_latest_frame_ts(),
            "image": {
                "width": self.state_store.get_latest_image_width(),
                "height": self.state_store.get_latest_image_height(),
                "source": "left_rectified",
            },
            "taskID": task.get("taskID"),
            "taskState": task.get("taskState", "idle"),
            "currentSeq": task.get("currentSeq"),
            "pathLength": task.get("pathLength", 0),
            "target_x": current_point.get("x"),
            "target_y": current_point.get("y"),
            "targetAction": current_point.get("action"),
            "stm32State": "not_connected",
        }
        return payload

    def _run(self):
        print(f"状态上报已启用: {self.status_url}")
        while not self._stop.is_set():
            try:
                self._post_json(self.build_payload())
            except Exception:
                pass
            self._stop.wait(self.interval)

    def _post_json(self, payload):
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        req = urllib.request.Request(
            self.status_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._opener.open(req, timeout=self.timeout) as resp:
            resp.read()

    def _format_detections(self, detections):
        formatted = []
        for det in detections:
            formatted.append(
                {
                    "label": det.get("label"),
                    "score": det.get("score"),
                    "class_id": det.get("class_id"),
                    "bbox": {
                        "left": det.get("left"),
                        "top": det.get("top"),
                        "right": det.get("right"),
                        "bottom": det.get("bottom"),
                    },
                    "left": det.get("left"),
                    "top": det.get("top"),
                    "right": det.get("right"),
                    "bottom": det.get("bottom"),
                }
            )
        return formatted
