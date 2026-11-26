"""Always-on UI shell. Hardware initialization is supervised behind the HTTP API."""

import argparse
from collections import deque
from functools import partial
import hashlib
from http.server import ThreadingHTTPServer
import importlib
import json
import logging
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

from bucket_robot.resources import default_reference
from bucket_robot.runtime.health import HealthRegistry
from bucket_robot.runtime.http import handler_for
from bucket_robot.runtime.supervision import VisualSupervisor
from bucket_robot.runtime.thermal import ThermalMonitor


class WaitingView:
    """Display-only state while optional dependencies are not available."""

    def __init__(self, health):
        self.health = health
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.stop_event = threading.Event()
        self.token = secrets.token_urlsafe(32)
        self.preview_image = self.image = None
        self.preview_metadata = {}
        self.metadata = {}
        self.jpeg = self.remote_jpeg = None
        self.video_id = self.remote_video_id = 0
        self.video_arrival = self.remote_arrival = 0.0
        self.snapshots = deque(maxlen=2)

    def status(self):
        return {
            "health": self.health.snapshot(),
            "preview": {"fps": 0, "tracking_fps": 0, "frame_age_s": 999},
            "tracking": {"active": False, "motion_available": False, "arm": {}, "events": []},
        }

    def command(self, action, body):
        if action not in ("stop", "heartbeat"):
            raise ValueError("组件初始化中，请查看健康状态")


def unavailable_limits():
    # Display-only unknown ranges. A Session with these limits never gets an arm worker.
    return SimpleNamespace(
        a_min=None,
        a_max=None,
        b_min=None,
        b_max=None,
        c_bounds=None,
        c=0,
        a_home=0,
        startup_policy={},
    )


class Application:
    def __init__(self, args, native_args, *, session_factory=None, process_factory=None):
        self.args, self.native_args = args, list(native_args)
        self.health = HealthRegistry(motion_enabled=not args.observe_only)
        self.thermal = ThermalMonitor()
        self.session = WaitingView(self.health)
        self.token = self.session.token
        self.stop_event = threading.Event()
        self.threads = []
        self.supervisor = None
        self.session_factory = session_factory
        self.process_factory = process_factory
        self.server = ThreadingHTTPServer((args.host, args.port), handler_for(self))
        self.server.daemon_threads = True
        self.task = None
        self.task_requested = args.mission and not args.observe_only
        self.task_attempted = False
        self.task_status = None
        self.serial_monitor = None
        self.task_telemetry = None
        self.approach_telemetry = None
        self.task_started = 0
        self.task_stop_requested = None
        self.invalid_assets = {}
        self.failure_fallback = False
        self.recheck_requested = False
        self.rechecking = False
        self.stop_file_session = None
        self.run_id = str(time.time_ns())
        self.health.ready("application", "界面与状态服务已启动")

    def __getattr__(self, name):
        return getattr(self.session, name)

    def value(self, key, default=None):
        values = [self.native_args[i + 1] for i, v in enumerate(self.native_args[:-1]) if v == key]
        return values[-1] if values else default

    def command(self, action, body):
        if action == "stop":
            # The operator's stop must also survive a not-yet-published Session.
            self.health.fault("OPERATOR_STOP", "已手动暂停，恢复前需明确确认")
        if (
            action in ("start", "confirm_power_cycle", "confirm_mcu_restart")
            and (self.args.output / "STOP").is_file()
        ):
            raise ValueError("STOP文件仍存在，请先处理停止请求")
        if (self.recheck_requested or self.rechecking) and action not in ("stop", "heartbeat"):
            raise ValueError("正在重新检查组件，请等待检查完成")
        if action == "retry":
            if (
                (self.health.snapshot()["motion_latch"] or {}).get("code") == "STOP_UNCONFIRMED"
                or self.task is not None
                and self.task.poll() is None
                or self.session.status()["tracking"].get("active")
                or self.session.status()["tracking"].get("arm", {}).get("phase")
                == "stop_unconfirmed"
            ):
                raise ValueError("先停止并确认运动状态，再重新检查组件")
            self.health.report("application", "recovering", "RECHECK_REQUESTED", "正在重新检查资源")
            self.health.fault("RECHECK", "重新检查后需明确恢复运动")
            self.recheck_requested = True
            self.rechecking = True
            return
        if self.args.mission and action in ("start", "stop"):
            if action == "stop":
                self.task_requested = False
                if self.task and self.task.poll() is None and self.task_stop_requested is None:
                    self.task.send_signal(signal.SIGTERM)
                    self.task_stop_requested = time.monotonic()
            else:
                if self.task is not None and self.task.poll() is None:
                    raise ValueError("上一任务尚未退出，请先等待停止确认")
                snapshot = self.health.snapshot()
                if (snapshot["motion_latch"] or {}).get("code") == "STOP_UNCONFIRMED":
                    raise ValueError("停止尚未确认，禁止恢复运动")
                if not self.health.motion_enabled or any(
                    snapshot["components"][key]["state"] != "ready"
                    for key in self.health.dependencies["mission"]
                ):
                    raise ValueError("任务能力未就绪，请查看组件状态")
                self.health.clear_motion_latch()
                self.task_requested = True
                self.task_attempted = False
            return
        if action == "start" and self.args.step_control and self.serial_monitor:
            self.serial_monitor.close()
        return self.session.command(action, body)

    def status(self):
        data = self.session.status()
        data["health"] = self.health.snapshot()
        data["thermal"] = self.thermal.snapshot()
        if self.args.mission:
            data["task"] = self.task_status
            data["tracking"]["motion_available"] = data["health"]["capabilities"]["mission"][
                "available"
            ]
            data["tracking"]["active"] = (
                self.task is not None
                and self.task.poll() is None
                and not data["health"]["motion_latch"]
            )
        return data

    def inspect_stop_file(self):
        if (self.args.output / "STOP").is_file():
            self.health.fault("STOP_FILE", "已收到STOP文件停止请求，运动保持暂停")
            if self.stop_file_session is not self.session or self.task_requested:
                self.command("stop", {})
                self.stop_file_session = self.session
        else:
            self.stop_file_session = None

    def bootstrap(self):
        store = None
        try:
            if self.args.release_root:
                from bucket_robot.runtime.release import ReleaseLayout

                self.invalid_assets = ReleaseLayout(self.args.release_root).component_issues()
                for name, issues in self.invalid_assets.items():
                    if issues:
                        self.health.report(
                            name,
                            "unavailable",
                            "ASSET_INVALID",
                            "资源缺失或校验失败：" + ", ".join(issues),
                            recovery="修复资源后重新检查",
                        )
            module = importlib.import_module("bucket_robot.runtime.stationary")
            import cv2
            import numpy as np
            from bucket_robot.motion.arm import Limits, ArmPort, ContinuousArmPort
            from bucket_robot.runtime.preview import LatestFrames
            from bucket_robot.perception.capture import NativeCapture
            from bucket_robot.runtime.startup import (
                PowerCycleStore,
                derive_wait_profile,
                StartupPort,
                RestartWatchPort,
            )
            from bucket_robot.runtime.journal import AsyncJournal

            if self.args.csrt:
                try:
                    from bucket_robot.perception import target
                    from bucket_robot.perception.correlation import install

                    install(target)
                except Exception:
                    self.health.report(
                        "tracker", "degraded", "CSRT_UNAVAILABLE", "可选CSRT不可用，保留光流追踪"
                    )
            valid_config = valid_calibration = True
            try:
                if self.invalid_assets.get("motion_config") or self.invalid_assets.get(
                    "application"
                ):
                    raise ValueError("asset_integrity")
                limits = Limits(self.args.reference)
                self.health.boot_pose = limits.boot_pose
                self.health.ready("motion_config")
            except Exception:
                valid_config = False
                limits = unavailable_limits()
                self.health.report(
                    "motion_config",
                    "unavailable",
                    "MOTION_CONFIG_INVALID",
                    "机构配置不可用",
                    recovery="修复配置后点击重新检查",
                )
            intrinsics = None
            try:
                if self.invalid_assets.get("calibration"):
                    raise ValueError("asset_integrity")
                path = Path(self.value("--calibration", ""))
                if hashlib.sha256(path.read_bytes()).hexdigest() != self.value("--calibration-sha"):
                    raise ValueError("calibration_hash_mismatch")
                storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
                try:
                    projection = storage.getNode("P1").mat()
                finally:
                    storage.release()
                if (
                    projection is None
                    or projection.shape != (3, 4)
                    or not np.isfinite(projection).all()
                    or min(projection[0, 0], projection[1, 1]) <= 0
                ):
                    raise ValueError("invalid_intrinsics")
                intrinsics = dict(
                    fx=float(projection[0, 0]),
                    fy=float(projection[1, 1]),
                    cx=float(projection[0, 2]),
                    cy=float(projection[1, 2]),
                )
                self.health.ready("calibration")
            except Exception:
                valid_calibration = False
                self.health.report(
                    "calibration",
                    "unavailable",
                    "CALIBRATION_INVALID",
                    "标定不可用，仅显示原始画面",
                    recovery="修复标定后点击重新检查",
                )
            motion = (
                valid_config
                and valid_calibration
                and not self.args.observe_only
                and "--replay" not in self.native_args
                and not self.args.mission
            )
            response = None
            if self.args.response_profile:
                try:
                    from bucket_robot.tracking.response import ResponseProfile

                    response = ResponseProfile.load(
                        self.args.response_profile, self.value("--calibration-sha")
                    )
                except Exception:
                    motion = False
                    self.health.report(
                        "motion_config",
                        "unavailable",
                        "RESPONSE_INVALID",
                        "响应参数无效，运动不可用",
                    )
            store = None
            if motion:
                try:
                    wait = derive_wait_profile(limits)
                    store = PowerCycleStore(
                        self.args.startup_state, wait["wait_seconds"], wait["wait_basis"]
                    )
                except Exception:
                    motion = False
                    self.health.report(
                        "motion_config",
                        "unavailable",
                        "STARTUP_LEDGER_UNAVAILABLE",
                        "启动账本不可用，禁止运动",
                        recovery="检查账本占用或存储后重新检查",
                    )
            session = (self.session_factory or module.Session)(
                self.args.output,
                limits,
                motion_allowed=motion,
                smooth=not self.args.step_control,
                automatic=motion,
                intrinsics=intrinsics,
                startup_store=store,
                response_profile=response,
                motion_compensation=self.args.motion_compensation,
                health=self.health,
                port_factory=partial(
                    ArmPort if self.args.step_control else ContinuousArmPort,
                    port=self.args.serial_port,
                ),
                startup_port_factory=partial(StartupPort, port=self.args.serial_port),
                watch_port_factory=partial(RestartWatchPort, port=self.args.serial_port),
            )
            session.token = self.token
            session.invalid_components = set()
            if not valid_calibration:
                session.invalid_components.add("calibration")
            session.expected_binding = {
                "calibration_sha256": self.value("--calibration-sha"),
                "backend": "cpp",
                "model_sha256": "4a59b48b634c25ca79d843cc746d8e062ca1161ec5ef7c0c0c96533c78d3fce7",
            }
            try:
                self.args.output.mkdir(parents=True, exist_ok=True)
                session.journal = AsyncJournal(self.args.output)
                self.health.ready("journal")
            except Exception:
                self.health.report(
                    "journal", "unavailable", "JOURNAL_FAILED", "日志不可写，界面保持运行"
                )
                session.motion_allowed = session.auto_follow = False
                session.auto_prepare = False
                motion = False
            self.session = session
            env = os.environ.copy()
            if self.args.release_root:
                env["LD_LIBRARY_PATH"] = str(self.args.release_root / "native/lib")
            kwargs = {"popen": self.process_factory or partial(subprocess.Popen, env=env)}
            self.supervisor = VisualSupervisor(
                self.native_command,
                self.health,
                self.stop_event,
                restart_after=lambda: 30 if self.failure_fallback else None,
                **kwargs,
            )
            session.on_frame = self.supervisor.observed_frame
            hub = LatestFrames()
            for name, target, args in [
                ("capture", session.capture_loop, (NativeCapture(self.args.port + 1), hub)),
                ("preview", session.preview_loop, (hub,)),
                ("tracking", session.vision_loop, (hub.reader(),)),
            ]:
                self.start_worker(name, target, args)
            if self.args.mission:
                root = self.args.release_root
                assets = [
                    root / "assets/fast_acv.rknn",
                    root / "assets/calibration.json",
                    root / "assisted_runtime/libfastacv.so",
                    root / "assisted_runtime/librknnrt.so",
                ]
                if all(path.is_file() for path in assets) and not self.invalid_assets.get(
                    "approach"
                ):
                    self.health.ready("approach")
            if motion:
                self.start_worker("motion", session.arm_loop, ())
            if not motion or self.args.step_control:
                from bucket_robot.runtime.telemetry_monitor import TelemetryMonitor

                self.serial_monitor = TelemetryMonitor(self.args.serial_port, self.health)
            if not self.invalid_assets.get("application"):
                self.health.ready("application", "界面运行中，设备后台检查")
        except Exception:
            logging.exception("application component initialization failed")
            if store is not None and getattr(self.session, "startup_store", None) is not store:
                store.close()
            self.health.fault("INITIALIZATION_FAILED", "组件初始化异常，运动暂停")
            self.health.report(
                "application",
                "degraded",
                "OPTIONAL_DEPENDENCY_FAILED",
                "视觉或控制依赖加载失败，状态页面保持运行",
                recovery="修复环境后点击重新检查",
            )

    def native_command(self, attempt):
        self.failure_fallback = attempt > 3 and attempt % 4 == 0
        if (
            not self.args.native
            or not Path(self.args.native).is_file()
            or self.invalid_assets.get("backend")
            or self.failure_fallback
        ):
            self.health.report(
                "detector", "unavailable", "NATIVE_UNAVAILABLE", "视觉程序不可用，使用原始画面预览"
            )
            return [
                sys.executable,
                "-m",
                "bucket_robot.runtime.raw_camera",
                "--port",
                str(self.args.port + 1),
                "--device",
                self.value(
                    "--device", "/dev/v4l/by-id/usb-USB_Camera_USB_Camera_01.00.00-video-index0"
                ),
            ]
        flags = []
        if self.invalid_assets.get("detector"):
            flags.append("--disable-model")
        if self.invalid_assets.get("calibration"):
            flags.append("--raw-preview")
        return [
            self.args.native,
            *self.native_args,
            *flags,
            "--output",
            str(self.args.output / ("native_%s_%03d" % (self.run_id, attempt))),
            "--tracking-overlay",
            str(self.args.output / "tracking_overlay.json"),
            "--port",
            str(self.args.port + 1),
        ]

    def start_worker(self, name, target, args):
        def run():
            try:
                target(*args)
            except Exception:
                logging.exception("%s worker stopped", name)
                self.health.report(
                    "tracker" if name == "tracking" else name,
                    "unavailable",
                    "WORKER_FAILED",
                    name + " 工作线程异常",
                )
                self.health.fault("WORKER_FAILED", "处理线程异常，运动暂停")

        thread = threading.Thread(target=run, name="bucket-" + name, daemon=True)
        thread.start()
        self.threads.append(thread)

    def tick_task(self):
        if self.task is not None:
            if self.task_telemetry:
                self.task_telemetry.tick()
            if (
                time.monotonic() - self.task_started >= self.args.wall_budget
                and self.task.poll() is None
            ):
                self.health.fault("TASK_TIME_LIMIT", "任务总时长到限，等待停止确认")
            file = self.args.output / "mission/task_status.json"
            try:
                self.task_status = json.loads(file.read_text())
                if self.task_status.get("phase") == "left_approach":
                    from bucket_robot.runtime.telemetry_monitor import TaskTelemetryReader

                    if self.approach_telemetry is None:
                        self.approach_telemetry = TaskTelemetryReader(
                            self.args.output / "mission/approach/events.jsonl", self.health
                        )
                    self.approach_telemetry.tick()
                    child_status = self.args.output / "mission/approach/task_status.json"
                    if child_status.is_file():
                        self.task_status = json.loads(child_status.read_text())
                if self.task_status.get("phase") in ("fault_paused", "stop_unconfirmed"):
                    self.health.fault(
                        "STOP_UNCONFIRMED"
                        if self.task_status["phase"] == "stop_unconfirmed"
                        else "TASK_FAULT",
                        self.task_status.get("reason") or "任务异常，保持停止",
                    )
            except (OSError, ValueError):
                pass
            if self.task.poll() is None:
                health = self.health.snapshot()
                handoff = time.monotonic() - self.task_started <= 2
                blocked = [
                    name
                    for name in self.health.dependencies["mission"]
                    if health["components"][name]["state"] != "ready"
                    and not (handoff and name in ("serial", "arm", "chassis"))
                ]
                if blocked:
                    self.health.fault(
                        "MISSION_CAPABILITY_LOST",
                        "任务依赖不可用：" + "、".join(self.health.names[name] for name in blocked),
                    )
                if (
                    time.monotonic() - self.task_started > 2
                    and self.health.snapshot()["components"]["serial"]["state"] != "ready"
                ):
                    self.health.fault("SERIAL_LOST", "任务控制器回传中断，等待停止确认")
                if self.health.snapshot()["motion_latch"] and self.task_stop_requested is None:
                    self.task.send_signal(signal.SIGTERM)
                    self.task_stop_requested = time.monotonic()
                if (
                    self.task_stop_requested is not None
                    and time.monotonic() - self.task_stop_requested > 8
                ):
                    self.health.fault("STOP_UNCONFIRMED", "任务未确认停止，禁止恢复运动")
                return
            if self.task.returncode:
                self.health.fault("TASK_FAILED", "任务已停止，请查看故障后明确恢复")
            if not self.task_status or self.task_status.get("disabled_confirmed") is not True:
                self.health.fault("STOP_UNCONFIRMED", "任务退出但未取得禁用确认，禁止恢复运动")
            self.task = None
            self.task_requested = False
        if not self.task_requested or self.task_attempted or not self.health.allows("mission"):
            return
        from bucket_robot.missions.coordinator import command
        from bucket_robot.runtime.release import ReleaseLayout

        layout = ReleaseLayout(self.args.release_root)
        if self.serial_monitor:
            self.serial_monitor.close()
        env = os.environ.copy()
        env.update(
            BUCKET_SERIAL_PORT=self.args.serial_port,
            BUCKET_ACV_BRIDGE=str(layout.range_runtime / "libfastacv.so"),
            BUCKET_ACV_NPU_CORE="2",
            LD_LIBRARY_PATH=str(layout.range_runtime),
        )
        folder = self.args.output / "mission"
        if folder.exists():
            self.health.fault("TASK_OUTPUT_EXISTS", "任务输出目录已存在，请使用新的任务运行目录")
            return
        # One control process; the UI shell retains the existing vision backend.
        self.task = subprocess.Popen(
            command(
                layout,
                self.args.output,
                self.args.port + 1,
                self.args.drive_budget,
                execute_post_geometry=getattr(self.args, "execute_post_geometry", False),
                resume_post_plan=getattr(self.args, "resume_post_plan", None),
                cruise_speed=getattr(self.args, "cruise_speed", 80),
            ),
            env=env,
            start_new_session=True,
        )
        from bucket_robot.runtime.telemetry_monitor import TaskTelemetryReader

        self.task_telemetry = TaskTelemetryReader(folder / "serial.jsonl", self.health)
        self.approach_telemetry = None
        self.task_started = time.monotonic()
        self.task_stop_requested = None
        self.health.report("serial", "starting", "SERIAL_HANDOFF", "控制器正在接管串口", ttl=0)
        self.task_attempted = True

    def run(self):
        web = threading.Thread(target=self.server.serve_forever, name="bucket-http", daemon=True)
        web.start()
        thermal = threading.Thread(
            target=self.thermal.run, args=(self.stop_event,), name="bucket-thermal", daemon=True
        )
        thermal.start()
        self.start_worker("bootstrap", self.bootstrap, ())
        began = time.monotonic()
        try:
            while not self.stop_event.wait(0.1):
                if time.monotonic() - began >= self.args.seconds:
                    break
                try:
                    self.inspect_stop_file()
                    if self.supervisor:
                        self.supervisor.tick()
                        self.session.native_process = self.supervisor.process
                    serial_free = not self.args.step_control or (
                        not getattr(self.session, "active", False)
                        and getattr(self.session, "arm", {}).get("phase") in ("disabled", "idle")
                    )
                    if self.serial_monitor and not self.task and serial_free:
                        self.serial_monitor.tick()
                    journal = getattr(self.session, "journal", None)
                    if journal is not None and journal.error:
                        self.health.report(
                            "journal",
                            "unavailable",
                            "JOURNAL_FAILED",
                            journal.error,
                            recovery="检查存储后重新检查组件",
                        )
                        self.health.fault("JOURNAL_FAILED", "日志异常，运动暂停")
                    if self.args.mission:
                        self.tick_task()
                except Exception:
                    logging.exception("component supervisor error")
                    self.health.report(
                        "application",
                        "degraded",
                        "SUPERVISION_FAILED",
                        "组件检查异常，界面继续运行",
                    )
                    self.health.fault("SUPERVISION_FAILED", "组件检查异常，运动暂停")
                if self.recheck_requested:
                    self.recheck_requested = False
                    if not self.close_components():
                        continue
                    self.health.fault("RECHECK", "重新检查后需明确恢复运动")
                    self.bootstrap()
                    self.rechecking = False
        finally:
            self.stop_event.set()
            closed = self.close_components()
            # The process is exiting even if a motion stop could not be proved.
            # Retain the failure result, but do not orphan camera/NPU ownership.
            if self.supervisor is not None:
                try:
                    self.supervisor.close()
                except Exception:
                    logging.exception("visual worker cleanup failed during exit")
                    closed = False
                self.supervisor = None
            thermal.join(2)
            self.server.shutdown()
            self.server.server_close()
            web.join(2)
        return 0 if closed else 2

    def close_components(self):
        self.session.stop_event.set()
        try:
            return self._close_components()
        except Exception:
            logging.exception("component cleanup failed")
            self.health.report(
                "application",
                "degraded",
                "CLEANUP_FAILED",
                "组件退出异常，界面保持运行",
                recovery="确认停止并检查组件",
            )
            self.health.fault("STOP_UNCONFIRMED", "组件退出未确认，禁止重新初始化")
            return False

    def _close_components(self):
        if self.task and self.task.poll() is None:
            self.task.send_signal(signal.SIGTERM)
            try:
                self.task.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.health.fault("STOP_UNCONFIRMED", "任务未确认停止，禁止自动重启")
                return False
        if self.task is not None:
            try:
                final = json.loads((self.args.output / "mission/task_status.json").read_text())
            except (OSError, ValueError):
                final = {}
            if not isinstance(final, dict) or final.get("disabled_confirmed") is not True:
                self.health.fault("STOP_UNCONFIRMED", "任务退出但未取得禁用确认，禁止重新初始化")
                return False
            self.task_status = final
            self.task = None
        self.session.stop_event.set()
        if self.supervisor:
            self.supervisor.close()
            self.supervisor = None
        if self.serial_monitor:
            self.serial_monitor.close()
            self.serial_monitor = None
        for thread in self.threads:
            if thread is not threading.current_thread():
                thread.join(4)
        if any(thread.is_alive() for thread in self.threads):
            self.health.fault("STOP_UNCONFIRMED", "组件未确认退出，禁止重新初始化")
            return False
        if self.session.status()["tracking"].get("arm", {}).get("phase") == "stop_unconfirmed":
            self.health.fault("STOP_UNCONFIRMED", "机构未确认停止，禁止重新初始化")
            return False
        self.threads = []
        store = getattr(self.session, "startup_store", None)
        if store:
            store.close()
        journal = getattr(self.session, "journal", None)
        if journal:
            journal.close()
        return True


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--native")
    ap.add_argument("--reference", type=Path, default=default_reference())
    ap.add_argument("--observe-only", action="store_true")
    ap.add_argument("--step-control", action="store_true")
    ap.add_argument("--csrt", action="store_true")
    ap.add_argument("--motion-compensation", action="store_true")
    ap.add_argument("--response-profile")
    ap.add_argument("--port", type=int, default=8788)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--native-port", type=int, help="兼容任务入口；界面使用前一个端口")
    ap.add_argument("--seconds", type=float, default=3600)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path.home() / ".local/state/bucket/runs" / time.strftime("%Y%m%d_%H%M%S"),
    )
    ap.add_argument(
        "--startup-state", default=str(Path.home() / ".local/state/bucket-arm/power-cycle.json")
    )
    ap.add_argument(
        "--serial-port",
        default=os.environ.get(
            "BUCKET_SERIAL_PORT", "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"
        ),
    )
    ap.add_argument("--release-root", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--drive-budget", type=float, default=120)
    ap.add_argument("--wall-budget", type=float, default=300)
    ap.add_argument("--execute-post-geometry", action="store_true")
    ap.add_argument("--cruise-speed", type=int, choices=range(1, 81), default=80)
    ap.add_argument("--resume-post-plan", type=Path)
    return ap


def track_main(mission=False):
    args, native = parser().parse_known_args()
    args.mission = mission
    if args.native_port is not None:
        args.port = args.native_port - 1
    if not 1024 <= args.port < 65535 or not 0 < args.seconds < float("inf"):
        raise ValueError("invalid port/duration")
    if mission:
        if not 0.6 <= args.drive_budget <= 120 or not 30 <= args.wall_budget <= 600:
            raise ValueError("invalid bounded mission budget")
        if args.release_root is None:
            args.release_root = (
                Path(os.environ["BUCKET_RELEASE_ROOT"])
                if "BUCKET_RELEASE_ROOT" in os.environ
                else None
            )
        if args.release_root is None:
            raise ValueError("--release-root is required")
        root = args.release_root
        args.output = args.out or args.output
        args.native = str(root / "native/vision")
        args.reference = root / "python/bucket_robot/resources/arm_reference.json"
        calibration = root / "assets/calibration.yml"
        try:
            digest = hashlib.sha256(calibration.read_bytes()).hexdigest()
        except OSError:
            digest = ""
        native += [
            "--model",
            str(root / "assets/detector.rknn"),
            "--calibration",
            str(calibration),
            "--calibration-sha",
            digest,
            "--assets",
            str(root / "assets"),
            "--cores",
            "0",
        ]
    if args.release_root and args.output.resolve().is_relative_to(args.release_root.resolve()):
        raise ValueError("run output must be outside the immutable release")
    native += ["--seconds", str(args.seconds)]
    app = Application(args, native)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda *unused: app.stop_event.set())
    return app.run()


def mission_main():
    return track_main(mission=True)
