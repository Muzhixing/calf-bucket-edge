"""Integrated viewer/tracker/controller; native detector remains a child process."""

from collections import deque
from bucket_robot.runtime.http import handler_for as handler_for
import json
import math
import os
import platform
from pathlib import Path
import secrets
import threading
import time
import uuid

import cv2
import numpy as np

from bucket_robot.motion.arm import ArmPort, ContinuousArmPort, plan, pose, should_interrupt
from bucket_robot.tracking.policy import FollowPolicy, alignment_radius
from bucket_robot.perception.target import TargetLock, valid_box
from bucket_robot.perception.ego import BackgroundMotion
from bucket_robot.runtime.diagnostics import StartupDiagnostics
from bucket_robot.runtime.startup import (
    StartupSequence,
    StartupPort,
    NativeWarmup,
    RestartWatchPort,
    can_auto_prepare,
)
from bucket_robot.tracking.controller import TrackingController
from bucket_robot.perception.session import new_target_session


class Session:
    def __init__(
        self,
        folder,
        limits,
        motion_allowed=True,
        port_factory=None,
        smooth=False,
        automatic=False,
        intrinsics=None,
        motion_compensation=False,
        response_profile=None,
        startup_store=None,
        startup_port_factory=StartupPort,
        watch_port_factory=RestartWatchPort,
        health=None,
    ):
        self.folder, self.limits, self.motion_allowed = Path(folder), limits, motion_allowed
        self.health = health
        self.on_frame = None
        self.startup_store = startup_store
        self.startup_port_factory = startup_port_factory
        self.startup_request = None
        self.startup_sequence = None
        self.startup_window_started = None
        self.watch_port_factory = watch_port_factory
        self.restart_watch = None
        self.watch_retry_at = 0.0
        self.watch_error = None
        self.registered_startup_attempted = False
        self.auto_prepare = bool(
            motion_allowed
            and smooth
            and startup_store is not None
            and limits.startup_policy.get("automatic_restart_detection")
            and limits.startup_policy.get("mcu_restart_returns_all_axes") is True
        )
        self.operator_paused = bool(startup_store and startup_store.data.get("operator_paused"))
        self.ui_events = deque(maxlen=50)
        self.ui_lock = threading.Lock()
        self.ui_sequence = 0
        self.last_countdown = None
        self.native_warmup = NativeWarmup()
        self.native_process = None
        self.startup_diagnostics = None
        self.diagnostics_logged = False
        self.expected_binding = {}
        self.follow_after = 0.0
        self.smooth = smooth
        self.auto_follow = (
            automatic and motion_allowed and (startup_store is None or startup_store.ready)
        )
        if self.operator_paused:
            self.auto_follow = False
        self.intrinsics = intrinsics
        self.motion_compensation = motion_compensation
        self.response_profile = response_profile
        self.port_factory = port_factory or (ContinuousArmPort if smooth else ArmPort)
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.log_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.active = False
        self.lease = 0.0
        self.revision = 0
        self.pending = None
        self.mode = "auto"
        self.target_id = 0
        self.tracking = None
        self.locked_at = 0.0
        self.reason = ""
        if startup_store is not None and not startup_store.ready:
            self.reason = startup_store.error or "power_cycle_confirmation_required"
        self.vision_state = "searching"
        self.reacquired_count = 0
        self.arm = {"phase": "disabled", "telemetry": None, "timestamp": 0.0}
        self.image = None
        self.metadata = {}
        self.jpeg = None
        self.frame_id = 0
        self.snapshots = deque(maxlen=2)
        self.video_id = 0
        self.video_at = 0.0
        self.video_arrival = 0.0
        self.video_fps = 0.0
        self.track_fps = 0.0
        self.preview_image = None
        self.preview_metadata = {}
        self.remote_jpeg = None
        self.remote_video_id = 0
        self.remote_arrival = 0.0
        self.remote_encoded_at = 0.0
        self.token = secrets.token_urlsafe(32)
        self.journal = None

    def diagnostic_call(self, method, *args, **kwargs):
        diagnostic = self.startup_diagnostics
        if diagnostic is None:
            return None
        try:
            return getattr(diagnostic, method)(*args, **kwargs)
        except Exception:
            diagnostic.error("diagnostic_internal_error:" + method)
            return None

    def ui_log(self, message, level="info"):
        with self.ui_lock:
            if self.ui_events and self.ui_events[-1]["message"] == message:
                return
            self.ui_sequence += 1
            self.ui_events.append(
                {"id": self.ui_sequence, "at": time.time(), "message": message, "level": level}
            )

    def log(self, event, data):
        if self.health is not None:
            if event == "serial_rx":
                self.health.telemetry(data)
            if event in ("control_error", "startup_failed"):
                self.health.fault(
                    "STOP_UNCONFIRMED" if data.get("stop_error") else "CONTROL_FAULT",
                    "停止未确认" if data.get("stop_error") else "控制异常，运动已暂停",
                )
                if data.get("stop_error"):
                    self.health.report(
                        "arm",
                        "unavailable",
                        "STOP_UNCONFIRMED",
                        "停止未确认",
                        recovery="现场确认停止并检查通信",
                    )
        try:
            self._log(event, data)
        except Exception as exc:
            if self.health is not None:
                self.health.report(
                    "journal",
                    "unavailable",
                    "JOURNAL_FAILED",
                    str(exc),
                    recovery="检查存储与日志配置",
                )
                if self.active:
                    self.health.fault("JOURNAL_FAILED", "日志写入失败，运动暂停")
                    self.active = self.auto_follow = False

    def _log(self, event, data):
        messages = {
            "session_started": "系统启动，相机和模型正在准备",
            "restart_watch_connected": "已连接 STM32，正在检查复位状态",
            "automatic_restart_detected": "检测到 STM32 复位，自动进入准备流程",
            "restart_preparation_wait": "开始回零等待，模型预热和自检并行",
            "registered_preparation_started": "STM32 已就绪，沿用 B/C 零点，开始自检",
            "registered_preflight_complete": "实时状态与视觉自检通过，开始 A 自检",
            "homing_assumed_after_wait": "等待结束，必要自检通过",
            "origin_registration_attempt": "登记 B/C 起点",
            "origin_registered_by_startup_assumption": "B/C 起点登记已确认，开始 A 准备",
            "startup_c_preparing": "A%d° 已确认，正在准备 C" % self.limits.a_home,
            "power_cycle_preparation_complete": "机械臂准备完成，自动追踪已开启",
            "automatic_follow_ready": "自动追踪已恢复",
            "startup_failed": "准备中断，自动动作已取消",
            "control_error": "控制暂停，后台继续检查设备",
        }
        if event == "startup_bc_check":
            self.ui_log(
                data["axis"]
                + " 自检 · "
                + ("最高设定速度向外" if data["stage"] == "out" else "最高设定速度返回")
                + " · B%s / C%s mm" % (data["target"][1], data["target"][2])
            )
        if event == "startup_a_check":
            action = {"left": "左侧", "right": "右侧", "return": "回正"}[data["stage"]]
            self.ui_log("A 自检 · %s %s°（B/C 保持）" % (action, data["target_deg"]))
        if event == "power_cycle_preparation_complete":
            messages[event] = "A%d° / C%s mm 准备完成，自动追踪已开启" % (
                self.limits.a_home,
                data.get("slide_position_mm", self.limits.c),
            )
        if event in ("startup_failed", "control_error") and data.get("stop_error"):
            messages[event] = "停车未确认，请切断动力并检查连接"
        if event in messages:
            self.ui_log(
                messages[event], "error" if event in ("startup_failed", "control_error") else "info"
            )
        if event == "startup_self_check":
            if data.get("ready"):
                self.ui_log("相机、推理预热和图像处理已就绪")
        if event == "startup_disabled_ack" and data.get("flags") == 0:
            self.ui_log("串口命令确认，启动通信故障已解除")
        if event == "startup_diagnostics_result":
            ms = data.get("metrics", {}).get("inference", {}).get("p95_ms")
            self.ui_log(
                "启动采样完成 · "
                + ("推理 P95 %.0f ms" % ms if ms is not None else "推理样本不足")
                + " · "
                + data.get("status", "样本不足"),
                "warning" if data.get("issues") else "info",
            )
        if event == "operator_command" and data.get("action") in ("stop", "start"):
            self.ui_log("已手动暂停自动流程" if data["action"] == "stop" else "已恢复自动流程")

        if event == "serial_observation" and self.startup_diagnostics is not None:
            self.diagnostic_call("serial", data["frames"], data["parser"])
        if self.journal is not None:
            self.journal.append(event, data)
            return
        record = {"timestamp": time.monotonic(), "event": event, "data": data}
        with self.log_lock:
            if self.folder.is_dir():
                with (self.folder / "tracking_events.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def diagnostic_image(self, name, image):
        if self.journal is not None:
            self.journal.image(name, image)
        else:
            cv2.imwrite(str(self.folder / name), image)

    def disarm(self, reason):
        with self.lock:
            was_active = self.active
            self.active = False
            self.auto_follow = False
            self.reason = reason
            self.startup_request = None
        if was_active:
            self.log("disarmed", {"reason": reason})

    def command(self, action, body):
        with self.lock:
            self._command(action, body)

    def _command(self, action, body):
        with self.lock:
            if self.health is not None and action in (
                "start",
                "confirm_power_cycle",
                "confirm_mcu_restart",
            ):
                snapshot = self.health.snapshot()
                if (snapshot["motion_latch"] or {}).get("code") == "STOP_UNCONFIRMED" or self.arm[
                    "phase"
                ] == "stop_unconfirmed":
                    raise ValueError("STOP_UNCONFIRMED: 停止尚未确认，禁止恢复运动")
                if not self.motion_allowed or any(
                    snapshot["components"][name]["state"] != "ready"
                    for name in (
                        "camera",
                        "detector",
                        "calibration",
                        "motion_config",
                        "journal",
                        "serial",
                    )
                ):
                    raise ValueError("CAPABILITY_BLOCKED: 设备或视觉尚未就绪，请查看组件状态")
            if action == "report_mcu_reset":
                if self.startup_store is None:
                    raise ValueError("startup_not_available")
                self.disarm("mcu_reset_reported")
                self.startup_store.invalidate("operator_reported_mcu_motor_restart")
                self.tracking = None
                self.pending = None
                self.revision += 1
            elif action in ("confirm_power_cycle", "confirm_mcu_restart"):
                if not self.motion_allowed or not self.smooth or self.startup_store is None:
                    raise ValueError("startup_not_available")
                event = (
                    "confirmed_mcu_motor_restart"
                    if action == "confirm_mcu_restart"
                    else "whole_machine_power_cycle"
                )
                if action == "confirm_mcu_restart":
                    if body.get("mcu_restarted") is not True:
                        raise ValueError("mcu_restart_confirmation_required")
                    if self.limits.startup_policy.get("mcu_restart_returns_all_axes") is not True:
                        raise ValueError("mcu_homing_convention_not_confirmed")
                elif body.get("whole_machine_power_cycled") is not True:
                    raise ValueError("whole_machine_confirmation_required")
                if self.active or self.arm["phase"] not in (
                    "disabled",
                    "startup_waiting",
                    "startup_preparing",
                    "startup_checking",
                ):
                    raise ValueError("stop_before_power_cycle_confirmation")
                self.begin_preparation(body.get("request_id"), event)
            elif action == "stop":
                cancel_pending = self.startup_request is not None and self.startup_sequence is None
                self.operator_paused = True
                self.startup_request = None
                self.active = False
                self.auto_follow = False
                self.reason = "operator_stop"
                try:
                    if self.auto_prepare:
                        self.startup_store.phase(
                            self.startup_store.data["phase"], operator_paused=True
                        )
                    if cancel_pending:
                        self.startup_store.invalidate("cancelled_before_sequence_started")
                except Exception as exc:
                    self.ui_log("已停止；暂停状态保存失败，需检查存储", "error")
                    self.log("pause_persistence_failed", {"reason": str(exc)})
            elif action == "heartbeat":
                self.lease = time.monotonic()
            elif action == "start":
                if self.auto_prepare:
                    self.startup_store.phase(
                        self.startup_store.data["phase"], operator_paused=False
                    )
                    self.operator_paused = False
                    self.active = False
                    self.auto_follow = self.startup_store.ready
                    self.reason = "" if self.auto_follow else "waiting_for_mcu_restart"
                    self.ui_log(
                        "自动流程已恢复，等待最新画面"
                        if self.auto_follow
                        else "自动检测已恢复，等待设备就绪"
                    )
                    if self.health is not None:
                        self.health.clear_motion_latch()
                    return
                if self.startup_store is not None and not self.startup_store.ready:
                    raise ValueError("power_cycle_confirmation_required")
                if not self.motion_allowed:
                    raise ValueError("当前为只读观察，不能启动机构")
                if not self.smooth and body.get("chassis_isolated") is not True:
                    raise ValueError("先确认底盘行走动力已断开")
                if self.image is None or time.monotonic() - self.metadata.get("arrival", 0) > 0.4:
                    raise ValueError("画面尚未就绪")
                if self.mode == "manual" and self.tracking is None:
                    raise ValueError("请重新框选目标")
                self.active = True
                self.auto_follow = self.smooth
                self.lease = time.monotonic()
                self.reason = ""
            elif action == "auto":
                self.active = False
                self.auto_follow = (
                    self.smooth
                    and self.motion_allowed
                    and not self.operator_paused
                    and (self.startup_store is None or self.startup_store.ready)
                )
                self.mode = "auto"
                self.pending = None
                self.revision += 1
                self.tracking = None
                self.reason = ""
            elif action == "select":
                if self.active or self.arm["phase"] not in ("disabled", "idle"):
                    raise ValueError("请先停止机构，等待禁用确认")
                box = body.get("box", [])
                if not valid_box(box):
                    raise ValueError("请框住完整容器，并留出画面边缘")
                snapshot = next((s for s in self.snapshots if s[0] == body.get("frame_id")), None)
                if snapshot is None or time.monotonic() - snapshot[1] > 0.4:
                    raise ValueError("框选画面已过期，请在实时画面重新选框")
                self.pending = (snapshot[2], list(map(int, box)), snapshot[1])
                self.mode = "manual"
                self.auto_follow = False
                self.revision += 1
                self.tracking = None
                self.reason = ""
            else:
                raise ValueError("未知操作")
            if self.health is not None and action in (
                "start",
                "confirm_power_cycle",
                "confirm_mcu_restart",
            ):
                self.health.clear_motion_latch()
        self.log("operator_command", {"action": action, "revision": self.revision})

    def status(self):
        with self.lock:
            with self.ui_lock:
                events = list(self.ui_events)
            data = dict(self.metadata)
            data["frame_age_s"] = max(0, time.monotonic() - data.get("arrival", 0)) if data else 999
            data["preview"] = {
                "frame_id": self.video_id,
                "fps": self.video_fps,
                "tracking_fps": self.track_fps,
                "frame_age_s": max(0, time.monotonic() - self.video_arrival)
                if self.video_arrival
                else 999,
            }
            track = dict(self.tracking) if self.tracking else None
            data["tracking"] = {
                "mode": self.mode,
                "target": track,
                "active": self.active,
                "automatic": self.auto_follow,
                "continuous": self.smooth,
                "target_id": self.target_id,
                "reason": self.reason,
                "vision_state": self.vision_state,
                "reacquired_count": self.reacquired_count,
                "auto_prepare": self.auto_prepare,
                "operator_paused": self.operator_paused,
                "events": events,
                "motion_available": self.motion_allowed,
                "limits": {
                    "A": [self.limits.a_min, self.limits.a_max],
                    "B": [self.limits.b_min, self.limits.b_max],
                    "C": self.limits.c_bounds,
                },
                "arm": dict(self.arm),
                "startup": (
                    {
                        **self.startup_store.status(),
                        "self_check": self.startup_checks(),
                        "report": self.diagnostic_call("finish")
                        if self.startup_diagnostics
                        else None,
                    }
                    if self.startup_store
                    else None
                ),
            }
            data["tracking"]["target_age_s"] = (
                max(0, time.monotonic() - track["timestamp"]) if track else None
            )
            if self.journal is not None:
                data["diagnostics"] = self.journal.status()
                if self.health is not None and self.journal.error:
                    self.health.report(
                        "journal",
                        "unavailable",
                        "JOURNAL_FAILED",
                        self.journal.error,
                        recovery="检查存储后重新检查组件",
                    )
                    self.health.fault("JOURNAL_FAILED", "日志异常，运动暂停")
            if self.health is not None:
                if self.arm["phase"] == "stop_unconfirmed":
                    self.health.fault("STOP_UNCONFIRMED", "停止未确认")
                data["health"] = self.health.snapshot()
                data["tracking"]["motion_available"] = (
                    self.motion_allowed
                    and data["health"]["capabilities"]["arm_tracking"]["available"]
                )
            return data

    def waiting_for_startup_checks(self):
        with self.lock:
            return self.startup_request is not None and (
                self.startup_sequence is None or self.startup_sequence.stage == "waiting"
            )

    def capture_loop(self, capture, hub):
        while not self.stop_event.is_set():
            try:
                fetch_started = time.monotonic()
                image, metadata = capture.read()
                received = time.monotonic()
                metadata = {**metadata, "_host_received_at": received}
                if self.health is not None:
                    self.health.ready("camera", "画面正常", ttl=0.5)
                    self.health.ready("backend", ttl=1)
                    components = metadata.get("components", {})
                    for name in ("detector", "calibration", "ranging"):
                        if name in getattr(self, "invalid_components", set()):
                            continue
                        state = components.get(
                            name, {"state": "ready", "code": "OK", "message": "正常"}
                        )
                        self.health.report(
                            name,
                            state["state"],
                            state.get("code", "OK"),
                            state.get("message", ""),
                            ttl=0.5,
                        )
                    if not self.health.allows("detection"):
                        self.tracking = None
                        if self.active:
                            self.health.fault("VISION_DEGRADED", "视觉能力不可用，运动暂停")
                    if self.on_frame:
                        self.on_frame()
                if self.startup_diagnostics is not None:
                    self.diagnostic_call(
                        "metric", "capture_fetch", (received - fetch_started) * 1000
                    )
                    self.diagnostic_call("frame", metadata, image)
                hub.publish(image, metadata)
                with self.lock:
                    self.preview_image = image
                    self.preview_metadata = metadata
                    self.native_warmup.observe(metadata)
            except Exception as exc:
                if self.startup_diagnostics is not None:
                    self.diagnostic_call("error", "capture:" + str(exc))
                if str(exc) == "frame_not_advanced":
                    continue
                hub.fail(exc)
                if self.health is not None:
                    self.health.report(
                        "camera",
                        "unavailable",
                        "CAMERA_UNAVAILABLE",
                        "摄像头画面不可用，正在重连",
                        ttl=0.5,
                    )
                    capture.last_id = -1
                    if self.active:
                        self.health.fault("CAMERA_LOST", "画面中断，运动暂停")
                with self.lock:
                    self.native_warmup.reset()
                    self.tracking = None
                    if str(exc) == "stale_camera":
                        self.vision_state = "waiting_frame"
                        if self.active:
                            self.reason = "waiting_for_target_frame"
                    else:
                        self.vision_state = "unavailable"
                        if (
                            self.active or self.preview_metadata
                        ) and not self.waiting_for_startup_checks():
                            self.disarm(str(exc))
                self.stop_event.wait(0.05)

    def preview_loop(self, hub):
        if platform.machine() == "aarch64" and hasattr(os, "sched_setaffinity"):
            available = {5, 7} & os.sched_getaffinity(0)
            if available:
                os.sched_setaffinity(0, available)
        previous = -1
        times = deque(maxlen=90)
        last_log = 0.0
        while not self.stop_event.is_set():
            try:
                image, metadata = hub.next(previous)
                previous = metadata["frame_id"]
                begin = time.monotonic()
                with self.lock:
                    target = dict(self.tracking) if self.tracking else None
                    mode, active, phase, state = (
                        self.mode,
                        self.active,
                        self.arm["phase"],
                        self.vision_state,
                    )
                    control = dict(self.arm.get("control", {}))
                # Fresh video must not display an old lock as a current observation.
                if (
                    target is not None
                    and not 0 <= metadata["arrival"] - target["timestamp"] <= 0.15
                ):
                    target = None
                annotated = image.copy()
                cv2.drawMarker(annotated, (640, 360), (240, 245, 242), cv2.MARKER_CROSS, 26, 2)
                if target:
                    b = target["box"]
                    x, y = target["offset_px"]
                    radius = alignment_radius(target)
                    color = (
                        (100, 235, 130)
                        if control.get("alignment_hold") or math.hypot(x, y) <= radius
                        else (0, 215, 255)
                    )
                    cv2.rectangle(annotated, tuple(b[:2]), tuple(b[2:]), color, 2)
                    cv2.drawMarker(
                        annotated, (round(x + 640), round(y + 360)), color, cv2.MARKER_CROSS, 14, 2
                    )
                    cv2.putText(
                        annotated,
                        f"{mode.upper()}  dx {x:+.1f}  dy {y:+.1f} px",
                        (18, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        color,
                        2,
                    )
                else:
                    for d in metadata.get("detections", []):
                        cv2.rectangle(
                            annotated,
                            (d["left"], d["top"]),
                            (d["right"], d["bottom"]),
                            (170, 150, 80),
                            1,
                        )
                label = (
                    ("reacquiring" if state == "reacquiring" else phase)
                    if active
                    else ("observing" if target else state)
                )
                cv2.putText(
                    annotated,
                    f"TRACK {label} | A/B follow",
                    (18, 695),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 215, 255),
                    1,
                )
                ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    raise RuntimeError("JPEG编码失败")
                # Refresh compact preview every frame without changing tracker input.
                ok, compact = cv2.imencode(
                    ".jpg", cv2.resize(annotated, (640, 360)), [cv2.IMWRITE_JPEG_QUALITY, 60]
                )
                if not ok:
                    raise RuntimeError("预览JPEG编码失败")
                remote = compact.tobytes()
                now = time.monotonic()
                times.append(now)
                if self.startup_diagnostics is not None:
                    self.diagnostic_call("metric", "preview", (now - begin) * 1000)
                    if "_host_received_at" in metadata:
                        self.diagnostic_call(
                            "metric",
                            "host_to_preview",
                            (now - metadata["_host_received_at"]) * 1000,
                        )
                fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0
                with self.condition:
                    # JPEG bytes and their sequence are published atomically.
                    self.remote_jpeg = remote
                    self.remote_video_id = previous
                    self.remote_arrival = metadata["arrival"]
                    self.remote_encoded_at = now
                    self.jpeg = encoded.tobytes()
                    self.video_id = previous
                    self.video_at = now
                    self.video_arrival = metadata["arrival"]
                    self.video_fps = fps
                    self.condition.notify_all()
                overlay = {
                    "timestamp": target["timestamp"] if target else now,
                    "state": label,
                    "box": target["box"] if target else [],
                }
                if self.health is not None:
                    self.health.ready("preview", "画面输出正常", ttl=0.5)
                try:
                    temporary = self.folder / "tracking_overlay.json.tmp"
                    temporary.write_text(json.dumps(overlay))
                    temporary.replace(self.folder / "tracking_overlay.json")
                except OSError as exc:
                    if self.health is not None:
                        self.health.report(
                            "journal",
                            "unavailable",
                            "OVERLAY_WRITE_FAILED",
                            str(exc),
                            recovery="检查存储后重新检查组件",
                        )
                        self.health.fault("JOURNAL_FAILED", "状态记录异常，运动暂停")
                if now - last_log >= 1:
                    self.log(
                        "preview_sample",
                        {
                            "frame_id": previous,
                            "fps": fps,
                            "frame_age_ms": (now - metadata["arrival"]) * 1000,
                            "render_ms": (now - begin) * 1000,
                            "jpeg_bytes": len(encoded),
                        },
                    )
                    last_log = now
            except Exception as exc:
                if str(exc) != "frame_not_advanced":
                    self.diagnostic_call("error", "preview:" + str(exc))
                    if self.health is not None:
                        self.health.report(
                            "preview",
                            "unavailable",
                            "PREVIEW_FAILED",
                            "画面输出失败：" + str(exc),
                            ttl=0.5,
                        )
                    self.stop_event.wait(0.05)

    def vision_loop(self, capture):
        if self.smooth and platform.machine() == "aarch64" and hasattr(os, "sched_setaffinity"):
            requested = {int(v) for v in os.environ.get("BUCKET_TRACK_CPUS", "4,6").split(",")}
            available = requested & os.sched_getaffinity(0)
            if available:
                os.sched_setaffinity(0, available)
            cv2.setNumThreads(int(os.environ.get("BUCKET_TRACK_THREADS", "2")))
            self.log(
                "tracking_compute",
                {"cpus": sorted(os.sched_getaffinity(0)), "opencv_threads": cv2.getNumThreads()},
            )
        mode = self.mode
        manager = new_target_session(mode, manager_factory=TargetLock)
        revision = -1
        recoverable_reason = None
        last_state = "searching"
        generation = 0
        recoveries = 0
        background = BackgroundMotion() if self.motion_compensation else None
        times = deque(maxlen=90)
        while not self.stop_event.is_set():
            begin = time.monotonic()
            try:
                image, metadata = capture.read()
                if self.health is not None and not self.health.allows("detection"):
                    with self.lock:
                        self.tracking = None
                        self.metadata = metadata
                        self.image = image
                        self.vision_state = "unavailable"
                    self.health.report(
                        "tracker",
                        "unavailable",
                        "DETECTION_UNAVAILABLE",
                        "检测不可用，视觉追踪暂停",
                        ttl=0.5,
                    )
                    self.stop_event.wait(0.02)
                    continue
                stamp = float(metadata["arrival"])
                tracking_started = time.monotonic()
                with self.lock:
                    rev, mode, pending = self.revision, self.mode, self.pending
                if rev != revision:
                    manager = new_target_session(mode, manager_factory=TargetLock)
                    revision = rev
                    generation = 0
                    recoveries = 0
                    last_state = "searching"
                if mode == "manual" and pending is not None:
                    reference, box, reference_stamp = pending
                    if stamp < reference_stamp:
                        continue
                    manager.select(reference, box, reference_stamp)
                    with self.lock:
                        if revision == self.revision:
                            self.pending = None
                tracking = manager.update(image, metadata)
                if self.health is not None:
                    self.health.ready("tracker", "视觉追踪处理正常", ttl=0.5)
                if background is not None:
                    if (
                        not tracking
                        or manager.generation != generation
                        or manager.reacquired != recoveries
                    ):
                        background.reset()
                    if tracking:
                        boxes = [tracking["box"]] + [
                            [d[k] for k in ("left", "top", "right", "bottom")]
                            for d in metadata.get("detections", [])
                        ]
                        tracking["ego_motion"] = background.update(image, stamp, boxes)
                if (
                    tracking
                    and manager.reacquired != recoveries
                    and time.monotonic() - tracking["timestamp"] > 0.35
                ):
                    # A costly search can identify the object in an old frame.
                    # Stay paused until ordinary tracking verifies a fresh frame.
                    self.log("recovery_wait_fresh", {"frame_id": metadata["frame_id"]})
                    continue
                if manager.expired and not self.auto_follow and self.startup_request is None:
                    self.disarm("target_reacquire_timeout")
                save_loss = manager.state != last_state and manager.state == "reacquiring"
                if manager.state != last_state:
                    self.log(
                        "visual_state",
                        {
                            "from": last_state,
                            "to": manager.state,
                            "reason": manager.reason,
                            "frame_id": metadata["frame_id"],
                        },
                    )
                    last_state = manager.state
                with self.lock:
                    if revision != self.revision:
                        continue
                    if manager.generation != generation:
                        generation = manager.generation
                        self.target_id += 1
                        self.locked_at = time.monotonic()
                        self.log(
                            "target_locked",
                            {
                                "source": mode,
                                "box": tracking["box"] if tracking else None,
                                "frame_id": metadata["frame_id"],
                            },
                        )
                    if manager.reacquired != recoveries:
                        recoveries = manager.reacquired
                        self.reacquired_count += 1
                        self.locked_at = time.monotonic()
                        self.log(
                            "same_target_reacquired",
                            {
                                "target_id": self.target_id,
                                "box": tracking["box"],
                                "frame_id": metadata["frame_id"],
                            },
                        )
                    if recoverable_reason is not None and self.reason == recoverable_reason:
                        self.reason = ""
                    recoverable_reason = None
                    if manager.state == "reacquiring" and self.active:
                        self.reason = "target_reacquiring"
                    elif self.reason == "target_reacquiring":
                        self.reason = ""
                    self.vision_state = manager.state
                    self.tracking = tracking
                    self.image = image
                    self.metadata = metadata
                    times.append(time.monotonic())
                    self.track_fps = (
                        (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0
                    )
                    self.frame_id = metadata["frame_id"]
                    active = self.active
                if self.startup_diagnostics is not None:
                    self.diagnostic_call(
                        "metric", "tracking", (time.monotonic() - tracking_started) * 1000
                    )
                    if "_host_received_at" in metadata:
                        self.diagnostic_call(
                            "metric",
                            "host_to_tracking",
                            (time.monotonic() - metadata["_host_received_at"]) * 1000,
                        )
                if save_loss and manager.tracker is not None:
                    try:
                        tag = str(metadata["frame_id"])
                        self.diagnostic_image("loss_" + tag + "_current.jpg", image)
                        self.diagnostic_image("loss_" + tag + "_previous.jpg", manager.tracker.gray)
                        self.diagnostic_image("loss_" + tag + "_anchor.jpg", manager.tracker.anchor)
                        self.diagnostic_image(
                            "loss_" + tag + "_weights.png",
                            (manager.tracker.weights * 255).astype(np.uint8),
                        )
                    except Exception:
                        pass
                self.log(
                    "tracking_sample",
                    {
                        "frame_id": metadata["frame_id"],
                        "vision_state": manager.state,
                        "target_id": self.target_id,
                        "detected": bool(metadata.get("detections")),
                        "target": tracking,
                        "active": active,
                        "rejection": (
                            {
                                "stages": getattr(manager.tracker, "last_rejections", []),
                                "appearance": getattr(manager.tracker, "last_appearance", {}),
                            }
                            if manager.state == "reacquiring" and manager.tracker is not None
                            else None
                        ),
                        "reappearance_candidate": (
                            {
                                k: manager.provisional[k]
                                for k in ("started", "frames", "score", "known", "votes")
                            }
                            if manager.provisional
                            else None
                        ),
                        "processing_ms": (time.monotonic() - begin) * 1000,
                    },
                )
            except Exception as exc:
                if str(exc) != "frame_not_advanced":
                    self.diagnostic_call("error", "tracking:" + str(exc))
                if background is not None and str(exc) != "frame_not_advanced":
                    background.reset()
                if str(exc) == "frame_not_advanced":
                    self.stop_event.wait(0.005)
                    continue
                if str(exc) == "stale_camera" and self.smooth:
                    # Temporary frame delay keeps tracking intent and appearance.
                    # The arm pauses until fresh observations are available.
                    with self.lock:
                        self.tracking = None
                        self.vision_state = "waiting_frame"
                        if self.active:
                            self.reason = "waiting_for_target_frame"
                    self.log("camera_frame_wait", {})
                    self.stop_event.wait(0.05)
                    continue
                if self.health is not None:
                    self.health.report(
                        "tracker",
                        "unavailable",
                        "TRACKING_FAILED",
                        "视觉追踪异常：" + str(exc),
                        ttl=0.5,
                    )
                    if self.active:
                        self.health.fault("TRACKING_FAILED", "视觉追踪异常，运动暂停")
                # A capture failure is not a detector miss. Do not resume motion
                # automatically across camera failure, stale data or bad input.
                with self.lock:
                    recoverable_reason = (
                        str(exc)
                        if manager.tracker is None and not self.active and self.mode == "auto"
                        else None
                    )
                if (self.active or self.frame_id) and not self.waiting_for_startup_checks():
                    self.disarm(str(exc))
                manager = new_target_session(mode, manager_factory=TargetLock)
                generation = 0
                recoveries = 0
                last_state = "unavailable"
                with self.lock:
                    self.tracking = None
                    self.pending = None
                    self.vision_state = "unavailable"
                self.log("tracking_unavailable", {"reason": str(exc)})
                self.stop_event.wait(0.15)
            self.stop_event.wait(
                max(0, (0.001 if self.smooth else 0.08) - (time.monotonic() - begin))
            )

    def arm_loop(self):
        if self.smooth:
            return self.continuous_arm_loop()
        port = None
        held = False
        move_identity = None
        try:
            while not self.stop_event.is_set():
                try:
                    if self.health is not None:
                        health = self.health.snapshot()
                        if (
                            health["motion_latch"]
                            or not health["capabilities"]["detection"]["available"]
                        ):
                            self.active = self.auto_follow = False
                    with self.lock:
                        active = self.active
                        lease = self.lease
                        tracking = dict(self.tracking) if self.tracking else None
                        identity = self.target_id
                        locked_at = self.locked_at
                        recovering = self.vision_state == "reacquiring"
                    now = time.monotonic()
                    if active and now - lease > 3:
                        self.disarm("operator_connection_lost")
                        active = False
                    if active and port is None:
                        port = self.port_factory(self.limits, self.log)
                        held = False
                        continue  # Reread intent after serial preflight.
                    if port is None:
                        self.stop_event.wait(0.02)
                        continue
                    port.tick()
                    if active and tracking is None:
                        # Searching is allowed only while stationary.
                        if port.phase != "idle":
                            if recovering:
                                port.stop()
                            else:
                                self.disarm("target_lost")
                                active = False
                    if active and tracking and not 0 <= now - tracking["timestamp"] <= 0.4:
                        self.disarm("stale_target")
                        active = False
                    if not active:
                        port.stop()
                        if port.phase == "idle":
                            last = port.state
                            error = port.shutdown()
                            port = None
                            with self.lock:
                                self.arm = {
                                    "phase": "disabled" if not error else "stop_unconfirmed",
                                    "telemetry": last,
                                    "timestamp": now,
                                }
                            if error:
                                self.disarm(error)
                            self.stop_event.wait(0.02)
                            continue
                    elif tracking:
                        error = tracking["offset_px"]
                        if port.phase == "moving":
                            if identity != move_identity:
                                self.disarm("target_identity_changed")
                                port.stop()
                            elif should_interrupt(
                                pose(port.state), port.target, error, port.origin
                            ):
                                self.log(
                                    "tracking_replan",
                                    {"offset_px": error, "remaining_target": port.target},
                                )
                                port.stop()
                        elif (
                            port.phase == "idle" and now >= port.ready_at and now - locked_at >= 0.2
                        ):
                            if held and (abs(error[0]) > 30 or abs(error[1]) > 24):
                                held = False
                            if abs(error[0]) <= 20 and abs(error[1]) <= 15:
                                held = True
                            if not held:
                                target = plan(pose(port.state), error, self.limits)
                                if target is not None:
                                    move_identity = identity
                                    with self.lock:
                                        if self.reason == "working_range_limit":
                                            self.reason = ""
                                    with self.lock:
                                        if (
                                            not self.active
                                            or self.stop_event.is_set()
                                            or (
                                                self.health is not None
                                                and not self.health.allows("arm_tracking")
                                            )
                                        ):
                                            continue
                                        port.move(target)
                                    self.log(
                                        "tracking_move",
                                        {
                                            "target": target,
                                            "offset_px": error,
                                            "target_id": identity,
                                        },
                                    )
                                else:
                                    with self.lock:
                                        self.reason = "working_range_limit"
                    with self.lock:
                        self.arm = {
                            "phase": (
                                "reacquiring"
                                if recovering and port.phase == "idle"
                                else "holding"
                                if held and port.phase == "idle"
                                else port.phase
                            ),
                            "telemetry": port.state,
                            "timestamp": port.rx_at,
                        }
                except Exception as exc:
                    self.disarm(str(exc))
                    held = False
                    if port is not None:
                        last = port.state
                        error = port.shutdown()
                        port = None
                    else:
                        last = None
                        error = None
                    with self.lock:
                        self.arm = {
                            "phase": "stop_unconfirmed" if error else "disabled",
                            "telemetry": last,
                            "timestamp": time.monotonic(),
                        }
                    if error:
                        self.disarm(error)
                    self.log("control_error", {"reason": str(exc), "stop_error": error})
                self.stop_event.wait(0.02)
        finally:
            if port is not None:
                error = port.shutdown()
                with self.lock:
                    self.arm = {
                        "phase": "stop_unconfirmed" if error else "disabled",
                        "telemetry": port.state,
                        "timestamp": time.monotonic(),
                    }

    def startup_checks(self):
        with self.lock:
            now = time.monotonic()
            warmup = self.native_warmup.status(now)
            camera = (
                bool(self.preview_metadata)
                and 0 <= now - self.preview_metadata.get("arrival", 0) <= 0.4
            )
            tracker = bool(self.metadata) and 0 <= now - self.metadata.get("arrival", 0) <= 0.4
            preview = self.jpeg is not None and 0 <= now - self.video_arrival <= 0.4
            calibration = (
                bool(self.intrinsics)
                and all(math.isfinite(v) for v in self.intrinsics.values())
                and self.intrinsics.get("fx", 0) > 0
                and self.intrinsics.get("fy", 0) > 0
            )
            journal = self.journal is None or not self.journal.error
            process = self.native_process is None or self.native_process.poll() is None
            return {
                "ready": all(
                    (warmup["ready"], camera, tracker, preview, calibration, journal, process)
                ),
                "inference_warmup": warmup,
                "camera_fresh": camera,
                "tracker_fresh": tracker,
                "preview_fresh": preview,
                "calibration_valid": calibration,
                "journal_healthy": journal,
                "native_running": process,
            }

    def begin_preparation(self, request_id, event, observed_at=None):
        if self.startup_store.confirm(request_id, event):
            self.schedule_preparation(observed_at)

    def schedule_preparation(self, observed_at=None):
        if self.restart_watch is not None:
            self.restart_watch.shutdown()
            self.restart_watch = None
        self.operator_paused = False
        self.startup_request = self.startup_store.data["cycle_id"]
        self.active = False
        self.auto_follow = False
        self.reason = "power_cycle_preparing"
        # Diagnostics only collect after this point. A boot signature observed
        # while paused must not consume the new eight-second sampling window.
        self.startup_window_started = time.monotonic()
        self.startup_diagnostics = StartupDiagnostics(
            self.startup_store.wait_seconds, now=self.startup_window_started
        )
        self.diagnostics_logged = False
        self.last_countdown = None
        self.diagnostic_call("probe_async", self.folder, self.limits, self.expected_binding)

    def advance_restart_watch(self):
        if not self.auto_prepare:
            return False
        if self.startup_request is not None or self.startup_sequence is not None:
            if self.restart_watch is not None:
                self.restart_watch.shutdown()
                self.restart_watch = None
            return False
        # An already prepared normal startup uses the ordinary live preflight.
        if self.active or self.auto_follow:
            if self.restart_watch is not None:
                self.restart_watch.shutdown()
                self.restart_watch = None
            return False
        now = time.monotonic()
        if now < self.watch_retry_at:
            return True
        try:
            if self.restart_watch is None:
                self.restart_watch = self.watch_port_factory(self.limits, self.log)
                self.log("restart_watch_connected", {})
            ready = self.restart_watch.reset_ready()
            q = self.restart_watch.state
            if self.health is not None:
                health = self.health.snapshot()
                if health["motion_latch"] or not health["capabilities"]["detection"]["available"]:
                    self.auto_follow = False
                    return True
            with self.lock:
                self.arm = {
                    "phase": "disabled" if self.operator_paused else "detecting_restart",
                    "telemetry": q if now - self.restart_watch.rx_at < 0.7 else None,
                    "timestamp": self.restart_watch.rx_at,
                }
                if self.operator_paused:
                    return True
                if ready:
                    if can_auto_prepare(self.startup_store):
                        self.log(
                            "automatic_restart_detected",
                            {"telemetry": q, "evidence": "stable_combined_boot_signature"},
                        )
                        if self.startup_store.ready:
                            self.startup_store.invalidate("mcu_restart_detected")
                        self.begin_preparation(
                            str(uuid.uuid4()),
                            "automatic_mcu_restart",
                            self.restart_watch.evidence.since,
                        )
                    else:
                        self.reason = "startup_registration_uncertain"
                        self.ui_log("上次登记回传未确认，保持停止，未重复登记", "warning")
                elif (
                    getattr(self.restart_watch, "registered_ready", False)
                    and not self.registered_startup_attempted
                    and not self.startup_store.error
                    and self.startup_store.data["phase"] in ("unconfirmed", "uncertain")
                ):
                    # A live registered controller needs no new reset or zero command.
                    # One attempt per process prevents repeated sweeps after a failed self-test.
                    self.startup_store.adopt_registered(q, self.limits.a_home)
                    self.registered_startup_attempted = True
                    self.schedule_preparation()
            self.watch_error = None
        except Exception as exc:
            if self.health is not None:
                self.health.disconnected("运动控制器未连接，正在等待有效回传")
            if self.restart_watch is not None:
                self.restart_watch.shutdown()
                self.restart_watch = None
            self.watch_retry_at = now + 1
            with self.lock:
                self.arm = {"phase": "waiting_device", "telemetry": None, "timestamp": now}
            if self.watch_error != str(exc):
                self.ui_log("等待 STM32 连接", "warning")
                self.watch_error = str(exc)
        return True

    def advance_startup(self):
        if self.startup_diagnostics is not None and not self.diagnostics_logged:
            report = self.diagnostic_call("finish")
            if report is not None:
                self.diagnostics_logged = True
                self.log("startup_diagnostics_result", report)
        if self.startup_store is None:
            return False
        with self.lock:
            request = self.startup_request
            if request is not None and self.startup_sequence is None:
                self.startup_sequence = StartupSequence(
                    self.startup_store,
                    self.limits,
                    self.log,
                    self.startup_port_factory,
                    self.startup_checks,
                    started_at=self.startup_window_started,
                )
            sequence = self.startup_sequence
        if sequence is None:
            return False
        try:
            if self.journal is not None and self.journal.error:
                raise RuntimeError(self.journal.error)
            if request is None or self.stop_event.is_set():
                error = sequence.close("initialization_cancelled")
                with self.lock:
                    self.startup_sequence = None
                    self.arm = {
                        "phase": "stop_unconfirmed" if error else "disabled",
                        "telemetry": None,
                        "timestamp": time.monotonic(),
                    }
                    if error:
                        self.disarm(error)
                return True
            # Waiting polls read-only while model/camera threads warm up in parallel.
            if sequence.stage == "waiting":
                sequence.tick()
            else:
                with self.lock:
                    if self.startup_request is not None and not self.stop_event.is_set():
                        sequence.tick()
            with self.lock:
                remaining = max(0, sequence.wait_seconds - (time.monotonic() - sequence.started))
                second = math.ceil(remaining)
                if sequence.stage == "waiting" and second != self.last_countdown:
                    self.last_countdown = second
                    self.ui_log("回零等待 · 剩余 %d 秒" % second if second else "等待自检就绪")
                self.arm = {
                    "phase": ("startup_waiting" if remaining > 0 else "startup_checking")
                    if sequence.stage == "waiting"
                    else "startup_preparing",
                    "telemetry": sequence.port.state if sequence.port else None,
                    "timestamp": time.monotonic(),
                    "startup_wait_remaining_s": remaining,
                }
                if sequence.done:
                    error = sequence.close()
                    self.startup_sequence = None
                    self.startup_request = None
                    if error:
                        raise RuntimeError(error)
                    self.active = False
                    self.auto_follow = True
                    self.reason = ""
                    self.follow_after = time.monotonic()
                    self.tracking = None
                    self.pending = None
                    self.revision += 1
                    self.arm["phase"] = "disabled"
            return True
        except BaseException as exc:
            try:
                error = sequence.close(str(exc))
            except Exception as cleanup_error:
                error = str(cleanup_error)
            with self.lock:
                self.startup_sequence = None
                self.startup_request = None
            self.disarm(error or str(exc))
            self.log("startup_failed", {"reason": str(exc), "stop_error": error})
            raise RuntimeError(error or str(exc)) from exc

    def continuous_arm_loop(self):
        port = None
        policy = FollowPolicy(
            self.limits, **self.intrinsics, response_profile=self.response_profile
        )
        identity = None
        controller = TrackingController(policy)
        alignment = controller.alignment
        if platform.machine() == "aarch64" and hasattr(os, "sched_setaffinity"):
            available = {0, 2} & os.sched_getaffinity(0)
            if available:
                os.sched_setaffinity(0, available)
        last_tick = time.monotonic()
        try:
            while not self.stop_event.is_set():
                try:
                    if self.health is not None:
                        health = self.health.snapshot()
                        if (
                            health["motion_latch"]
                            or not health["capabilities"]["detection"]["available"]
                        ):
                            self.active = self.auto_follow = False
                            self.startup_request = None
                            if self.startup_sequence is not None:
                                self.advance_startup()
                            if port is not None:
                                port.stop()
                                port.tick()
                                if port.phase == "idle":
                                    error = port.shutdown()
                                    self.arm = {
                                        "phase": "stop_unconfirmed" if error else "disabled",
                                        "telemetry": port.state,
                                        "timestamp": time.monotonic(),
                                    }
                                    if error:
                                        self.health.fault("STOP_UNCONFIRMED", "停止未确认")
                                    port = None
                                self.stop_event.wait(0.02)
                                continue
                    if port is None and self.advance_restart_watch():
                        last_tick = time.monotonic()
                        self.stop_event.wait(0.01)
                        continue
                    if self.advance_startup():
                        last_tick = time.monotonic()
                        self.stop_event.wait(0.01)
                        continue
                    now = time.monotonic()
                    if now - last_tick > 0.08:
                        self.log("control_loop_gap", {"gap_ms": 1000 * (now - last_tick)})
                    last_tick = now
                    if self.journal is not None and self.journal.error and self.active:
                        raise RuntimeError(self.journal.error)
                    with self.lock:
                        if (
                            self.auto_follow
                            and not self.active
                            and self.metadata
                            and now - self.metadata.get("arrival", 0) < 0.3
                            and self.metadata.get(
                                "_host_received_at", self.metadata.get("arrival", 0)
                            )
                            > self.follow_after
                        ):
                            self.active = True
                            self.reason = ""
                            self.log("automatic_follow_ready", {})
                        active = self.active
                        target = dict(self.tracking) if self.tracking else None
                        if (
                            self.metadata.get("_host_received_at", self.metadata.get("arrival", 0))
                            <= self.follow_after
                        ):
                            target = None
                        metadata = dict(self.metadata)
                        target_id = self.target_id
                    if active and port is None:
                        port = self.port_factory(self.limits, self.log)
                        policy.reset()
                        identity = None
                        last_tick = (
                            time.monotonic()
                        )  # Initial stopped preflight is intentionally blocking.
                        continue  # Read intent again after preflight, including a user stop.
                    if port is None:
                        self.stop_event.wait(0.01)
                        continue
                    port.tick()
                    if not active:
                        alignment.reset()
                        port.stop()
                        if port.phase == "idle":
                            last = port.state
                            error = port.shutdown()
                            port = None
                            with self.lock:
                                self.arm = {
                                    "phase": "stop_unconfirmed" if error else "disabled",
                                    "telemetry": last,
                                    "timestamp": now,
                                }
                            if error:
                                self.disarm(error)
                    elif target is None or not 0 <= now - target["timestamp"] <= 0.4:
                        port.stop()
                        controller.reset()
                        identity = None
                        with self.lock:
                            waiting = target is not None or self.vision_state == "waiting_frame"
                            if waiting and self.active:
                                if self.reason != "waiting_for_target_frame":
                                    self.log("target_frame_wait", {})
                                self.reason = "waiting_for_target_frame"
                            self.arm = {
                                "phase": "waiting_frame"
                                if waiting
                                else "reacquiring"
                                if self.vision_state == "reacquiring"
                                else "searching",
                                "telemetry": port.state,
                                "timestamp": port.rx_at,
                            }
                    else:
                        age = now - target["timestamp"]
                        with self.lock:
                            if self.reason == "waiting_for_target_frame":
                                self.reason = ""
                        was_holding = alignment.holding
                        decision = controller.step(port, target, metadata, now, target_id)
                        identity = target_id
                        holding = decision.holding
                        if holding != was_holding:
                            policy.reset()
                            self.log(
                                "alignment_hold" if holding else "alignment_resume",
                                {
                                    "target_id": identity,
                                    "offset_px": target["offset_px"],
                                    **alignment.status(),
                                },
                            )
                        if holding:
                            # Disabled holds A PWM and cancels finite B motion. Keep
                            # automatic intent, but send no enabled micro-corrections.
                            port.stop()
                            with self.lock:
                                if alignment.constrained:
                                    self.reason = "working_range_limit"
                                elif self.reason == "working_range_limit":
                                    self.reason = ""
                                self.arm = {
                                    "phase": "stopping"
                                    if port.phase == "stopping"
                                    else "range_holding"
                                    if alignment.constrained
                                    else "holding",
                                    "telemetry": port.state,
                                    "timestamp": port.rx_at,
                                    "control": alignment.status(),
                                }
                            self.stop_event.wait(0.01)
                            continue
                        q = list(decision.goal)
                        normalizing_b = decision.normalizing_b
                        if decision.stop:
                            port.stop()
                            policy.reset()
                        elif port.can_send():
                            with self.lock:
                                sent = (
                                    self.active
                                    and not self.stop_event.is_set()
                                    and (self.health is None or self.health.allows("arm_tracking"))
                                    and port.follow(q)
                                )
                            if sent:
                                self.log(
                                    "continuous_target",
                                    {
                                        "target": q,
                                        "target_id": identity,
                                        "offset_px": target["offset_px"],
                                        "frame_age_s": age,
                                        "normalizing_lift": normalizing_b,
                                        **policy.diagnostics,
                                    },
                                )
                        at_limit = (
                            q[0] == self.limits.a_min
                            and target["offset_px"][0] > 15
                            or q[0] == self.limits.a_max
                            and target["offset_px"][0] < -15
                            or q[1] == self.limits.b_min
                            and target["offset_px"][1] < -15
                            or q[1] == self.limits.b_max
                            and target["offset_px"][1] > 15
                        )
                        with self.lock:
                            if at_limit:
                                self.reason = "working_range_limit"
                            elif self.reason == "working_range_limit":
                                self.reason = ""
                            self.arm = {
                                "phase": "stopping" if port.phase == "stopping" else "following",
                                "telemetry": port.state,
                                "timestamp": port.rx_at,
                                "control": {**policy.diagnostics, **alignment.status()},
                            }
                except Exception as exc:
                    self.disarm(str(exc))
                    controller.reset()
                    if self.startup_store is not None and self.startup_store.ready:
                        try:
                            self.startup_store.invalidate("tracking_state_uncertain:" + str(exc))
                        except Exception as state_error:
                            self.log("startup_state_error", {"reason": str(state_error)})
                    error = (
                        port.shutdown()
                        if port
                        else (str(exc) if str(exc).startswith("STOP_UNCONFIRMED") else None)
                    )
                    last = port.state if port else None
                    port = None
                    with self.lock:
                        self.arm = {
                            "phase": "stop_unconfirmed" if error else "disabled",
                            "telemetry": last,
                            "timestamp": time.monotonic(),
                        }
                    if error:
                        self.disarm(error)
                    self.log("control_error", {"reason": str(exc), "stop_error": error})
                self.stop_event.wait(0.01)
        finally:
            if self.restart_watch is not None:
                self.restart_watch.shutdown()
                self.restart_watch = None
            if self.startup_sequence is not None:
                self.startup_sequence.close("program_exit_during_initialization")
                self.startup_sequence = None
            if port is not None:
                error = port.shutdown()
                with self.lock:
                    self.arm = {
                        "phase": "stop_unconfirmed" if error else "disabled",
                        "telemetry": port.state,
                        "timestamp": time.monotonic(),
                    }


def main():
    from bucket_robot.runtime.application import track_main

    return track_main()


if __name__ == "__main__":
    raise SystemExit(main())
