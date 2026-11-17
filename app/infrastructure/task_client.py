#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task sources and timed route-step dispatch."""

import json
import os
import threading
import time
import urllib.parse
import urllib.request

from app.infrastructure.bucket_tracking import is_tracking_point


class TaskPullClient:
    """Polls TASK_PULL_URL and dispatches route steps to the serial gateway."""

    def __init__(
        self,
        task_pull_url,
        task_state,
        serial_gateway=None,
        device_id="robot001",
        poll_interval=2.0,
        timeout=2.0,
        forward_mps=0.2,
        horizontal_mps=0.1,
        default_step_duration=2.0,
    ):
        self.task_pull_url = task_pull_url
        self.task_state = task_state
        self.serial_gateway = serial_gateway
        self.device_id = device_id
        self.poll_interval = max(0.5, float(poll_interval))
        self.timeout = float(timeout)
        self.forward_mps = max(0.001, float(forward_mps))
        self.horizontal_mps = max(0.001, float(horizontal_mps))
        self.default_step_duration = max(0.1, float(default_step_duration))
        self._stop = threading.Event()
        self._thread = None
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self._step_started_at = None
        self._step_start_sent_count = None

    def start(self):
        if not self.task_pull_url or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="task-pull", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self):
        print(f"路径任务拉取已启用: {self.task_pull_url}")
        while not self._stop.is_set():
            try:
                payload = self._fetch_task()
                if payload is not None and self.task_state.apply_task(payload):
                    self._dispatch_next_sendable_point()
                self._advance_step_if_needed()
            except Exception as exc:
                print(f"路径任务拉取失败: {exc}")
            self._stop.wait(self.poll_interval)

    def _fetch_task(self):
        url = self.task_pull_url.replace(
            "{deviceID}",
            urllib.parse.quote(str(self.device_id), safe=""),
        )
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "X-Device-ID": str(self.device_id),
            },
            method="GET",
        )
        with self._opener.open(req, timeout=self.timeout) as resp:
            body = resp.read()
        if not body:
            return None
        return json.loads(body.decode("utf-8"))

    def _dispatch_current_point(self):
        snapshot = self.task_state.snapshot()
        try:
            if self.serial_gateway is not None:
                self.serial_gateway.handle_task_update(snapshot)
            self.task_state.mark_running()
            if self.serial_gateway is None:
                self._step_started_at = time.monotonic()
                self._step_start_sent_count = None
            else:
                self._step_started_at = None
                self._step_start_sent_count = self._serial_sent_count()
        except Exception as exc:
            self.task_state.mark_failed(str(exc))
            raise

    def _dispatch_next_sendable_point(self):
        while self._skip_current_point_if_noop():
            pass
        if self.task_state.snapshot().get("taskState") == "completed":
            if self.serial_gateway is not None:
                self.serial_gateway.clear_task()
            return
        self._dispatch_current_point()

    def _advance_step_if_needed(self):
        snapshot = self.task_state.snapshot()
        if snapshot.get("taskState") != "running":
            return
        current_point = snapshot.get("currentPoint")
        if not isinstance(current_point, dict):
            return
        if is_tracking_point(current_point):
            self._advance_tracking_step_if_needed(snapshot)
            return
        if self._step_started_at is None:
            current_sent_count = self._serial_sent_count()
            if current_sent_count is None:
                self._step_started_at = time.monotonic()
            elif (
                self._step_start_sent_count is not None
                and current_sent_count > self._step_start_sent_count
            ):
                self._step_started_at = time.monotonic()
            return

        duration = self._step_duration_seconds(current_point)
        if time.monotonic() - self._step_started_at < duration:
            return

        next_point = self.task_state.advance_to_next()
        if next_point is None:
            if self.serial_gateway is not None:
                self.serial_gateway.clear_task()
            print(f"路径任务完成: {snapshot.get('taskID')}")
            self._step_started_at = None
            self._step_start_sent_count = None
            return
        self._dispatch_next_sendable_point()

    def _advance_tracking_step_if_needed(self, snapshot):
        stats = self._serial_stats()
        if not stats:
            return
        if stats.get("tracking_failed"):
            message = stats.get("tracking_detail") or "bucket tracking failed"
            self.task_state.mark_failed(message)
            if self.serial_gateway is not None:
                self.serial_gateway.clear_task()
            print(f"路径任务失败: {snapshot.get('taskID')}: {message}")
            self._step_started_at = None
            self._step_start_sent_count = None
            return
        if not stats.get("tracking_completed"):
            return

        next_point = self.task_state.advance_to_next()
        if next_point is None:
            if self.serial_gateway is not None:
                self.serial_gateway.clear_task()
            print(f"路径任务完成: {snapshot.get('taskID')}")
            self._step_started_at = None
            self._step_start_sent_count = None
            return
        self._dispatch_next_sendable_point()

    def _step_duration_seconds(self, point):
        explicit = _to_float(point.get("duration_s"))
        if explicit is not None and explicit >= 0:
            return explicit

        durations = []
        forward_m = _to_float(point.get("forward_m"))
        if forward_m:
            durations.append(abs(forward_m) / self.forward_mps)
        horizontal_m = _to_float(point.get("horizontal_m"))
        if horizontal_m:
            durations.append(abs(horizontal_m) / self.horizontal_mps)
        if durations:
            return max(durations)
        return self.default_step_duration

    def _serial_sent_count(self):
        stats = self._serial_stats()
        if not stats:
            return None
        value = stats.get("sent_frame_count")
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _serial_stats(self):
        if self.serial_gateway is None:
            return None
        stats_fn = getattr(self.serial_gateway, "get_sent_stats", None)
        if not callable(stats_fn):
            return None
        stats = stats_fn()
        if not isinstance(stats, dict):
            return None
        return stats

    def _skip_current_point_if_noop(self):
        point = self.task_state.get_current_point()
        if not isinstance(point, dict) or not self._is_noop_step(point):
            return False
        next_point = self.task_state.advance_to_next()
        if next_point is None:
            print(f"路径任务完成: {self.task_state.snapshot().get('taskID')}")
        return True

    def _is_noop_step(self, point):
        duration = _to_float(point.get("duration_s"))
        forward_m = abs(_to_float(point.get("forward_m")) or 0.0)
        horizontal_m = abs(_to_float(point.get("horizontal_m")) or 0.0)
        if duration is None or duration > 0:
            return False
        if forward_m > 1e-9 or horizontal_m > 1e-9:
            return False
        return point.get("action") in {None, "", "start", "pass"}


class MockTaskFileClient(TaskPullClient):
    """Loads a local JSON task file once per file modification."""

    def __init__(
        self,
        mock_task_file,
        task_state,
        serial_gateway=None,
        device_id="robot001",
        poll_interval=1.0,
        forward_mps=0.2,
        horizontal_mps=0.1,
        default_step_duration=2.0,
    ):
        super().__init__(
            task_pull_url=mock_task_file,
            task_state=task_state,
            serial_gateway=serial_gateway,
            device_id=device_id,
            poll_interval=poll_interval,
            timeout=0,
            forward_mps=forward_mps,
            horizontal_mps=horizontal_mps,
            default_step_duration=default_step_duration,
        )
        self.mock_task_file = mock_task_file
        self._last_mtime = None

    def _run(self):
        print(f"本地模拟任务已启用: {self.mock_task_file}")
        while not self._stop.is_set():
            try:
                payload = self._fetch_task()
                if payload is not None and self.task_state.apply_task(payload):
                    self._dispatch_next_sendable_point()
                self._advance_step_if_needed()
            except Exception as exc:
                print(f"本地模拟任务处理失败: {exc}")
            self._stop.wait(self.poll_interval)

    def _fetch_task(self):
        try:
            stat = os.stat(self.mock_task_file)
        except OSError:
            return None
        if self._last_mtime is not None and stat.st_mtime == self._last_mtime:
            return None
        with open(self.mock_task_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self._last_mtime = stat.st_mtime
        return payload


def _to_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
