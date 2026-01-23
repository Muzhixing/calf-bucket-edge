#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read model for presentation layer."""

import threading

from app.domain.events import RangingUpdateEvent


class RangingStateStore:
    """Thread-safe latest-state cache updated by domain events."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame_ts = None
        self._display_frame = None
        self._left_frame = None
        self._distance_m = None
        self._detected = False
        self._detections = []

    def handle_ranging_update(self, event: RangingUpdateEvent):
        with self._lock:
            self._frame_ts = event.frame_ts
            self._display_frame = event.display_frame
            self._left_frame = event.left_frame
            self._distance_m = event.distance_m
            self._detected = event.detected
            self._detections = list(event.detections) if event.detections is not None else []

    def _copy_if_possible(self, value):
        if value is None:
            return None
        copy_fn = getattr(value, "copy", None)
        return copy_fn() if callable(copy_fn) else value

    def get_latest_display_frame(self):
        with self._lock:
            return self._copy_if_possible(self._display_frame)

    def get_latest_left_frame(self):
        with self._lock:
            return self._copy_if_possible(self._left_frame)

    def get_latest_distance(self):
        with self._lock:
            return self._distance_m

    def get_latest_detected(self):
        with self._lock:
            return self._detected

    def get_latest_detections(self):
        with self._lock:
            return list(self._detections)

    def get_latest_frame_ts(self):
        with self._lock:
            return self._frame_ts
