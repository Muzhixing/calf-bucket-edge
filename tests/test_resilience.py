"""Local fault injection; never opens a physical camera, UART or NPU."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import os
from pathlib import Path
import socket
import select
import tempfile
import threading
import time
import unittest
import urllib.request
import urllib.error
from unittest.mock import patch

import cv2
import numpy as np
from bucket_robot.runtime.health import HealthRegistry
from bucket_robot.runtime.application import Application, parser
from bucket_robot.runtime.supervision import VisualSupervisor
from bucket_robot.runtime.telemetry_monitor import TelemetryMonitor, TaskTelemetryReader
from tests.test_receive import frame
from bucket_robot.transport.telemetry import decode


class HealthTests(unittest.TestCase):
    def test_open_idle_uart_without_controller_reports_offline_and_recovers_on_real_bytes(self):
        from bucket_robot.runtime.stationary import Session
        from bucket_robot.runtime.startup import PowerCycleStore, RestartWatchPort
        from tests.test_tracking import Reference, Sender

        reference = Reference()
        master, slave = os.openpty()
        health = HealthRegistry()
        for name in ("camera", "calibration", "detector"):
            health.ready(name)
        reference.limits.startup_policy.update(
            automatic_restart_detection=True, mcu_restart_returns_all_axes=True
        )
        store = PowerCycleStore(Path(reference.directory.name) / "state.json", 8, "test")
        session = Session(
            reference.directory.name,
            reference.limits,
            smooth=True,
            startup_store=store,
            health=health,
            watch_port_factory=lambda limits, log: RestartWatchPort(
                limits, log, port=os.ttyname(slave), sender_factory=Sender, check_holders=False
            ),
        )
        session.operator_paused = True
        try:
            session.advance_restart_watch()
            session.restart_watch.opened_at -= 1
            session.advance_restart_watch()
            self.assertEqual(health.snapshot()["components"]["serial"]["code"], "STM32_OFFLINE")
            self.assertEqual(health.snapshot()["components"]["arm"]["code"], "DEVICE_STATE_UNKNOWN")
            os.write(master, frame())
            for _ in range(20):
                session.advance_restart_watch()
                if health.snapshot()["components"]["serial"]["state"] == "ready":
                    break
                time.sleep(0.005)
            self.assertEqual(health.snapshot()["components"]["serial"]["state"], "ready")
            self.assertFalse(select.select([master], [], [], 0.05)[0])
        finally:
            if session.restart_watch:
                session.restart_watch.shutdown()
            store.close()
            os.close(slave)
            os.close(master)
            reference.directory.cleanup()

    def test_camera_and_detection_work_without_motors_and_no_false_presence_claim(self):
        health = HealthRegistry()
        for component in ("camera", "preview", "detector", "calibration", "tracker"):
            health.ready(component)
        health.disconnected()
        snapshot = health.snapshot()
        self.assertTrue(snapshot["capabilities"]["detection"]["available"])
        self.assertTrue(snapshot["capabilities"]["preview"]["available"])
        self.assertFalse(snapshot["capabilities"]["arm_tracking"]["available"])
        self.assertEqual(snapshot["components"]["arm"]["code"], "DEVICE_STATE_UNKNOWN")

    def test_stale_telemetry_latches_motion_and_reconnect_does_not_resume(self):
        now = [1.0]
        health = HealthRegistry(clock=lambda: now[0])
        for component in (
            "camera",
            "preview",
            "detector",
            "calibration",
            "tracker",
            "motion_config",
            "journal",
            "arm",
        ):
            health.ready(component)
        health.ready("serial", ttl=0.7)
        self.assertTrue(health.allows("arm_tracking"))
        now[0] = 2
        self.assertFalse(health.allows("arm_tracking"))
        health.ready("serial", ttl=0.7)
        self.assertFalse(health.allows("arm_tracking"))
        health.clear_motion_latch()
        self.assertTrue(health.allows("arm_tracking"))

    def test_repeated_error_is_one_component_record_and_stop_unknown_wins(self):
        health = HealthRegistry()
        for _ in range(10):
            health.report("detector", "unavailable", "MODEL_LOAD_FAILED", "加载失败")
        self.assertEqual(health.snapshot()["components"]["detector"]["count"], 10)
        health.fault("SERIAL_LOST", "通信丢失")
        health.fault("STOP_UNCONFIRMED", "停止未确认")
        self.assertEqual(health.snapshot()["motion_latch"]["code"], "STOP_UNCONFIRMED")

    def test_stop_unconfirmed_cannot_be_cleared_by_start_or_recheck_api(self):
        from bucket_robot.runtime.stationary import Session
        from bucket_robot.motion.arm import Limits
        from bucket_robot.resources import default_reference

        health = HealthRegistry()
        for name in health.components:
            health.ready(name)
        health.fault("STOP_UNCONFIRMED", "停止未确认")
        session = Session("unused", Limits(default_reference()), health=health)
        with self.assertRaisesRegex(ValueError, "STOP_UNCONFIRMED"):
            session.command("start", {})
        self.assertFalse(session.active)
        self.assertIsNotNone(health.snapshot()["motion_latch"])

    def test_passive_uart_monitor_reads_valid_frames_without_transmitting(self):
        master, slave = os.openpty()
        monitor = TelemetryMonitor(os.ttyname(slave), HealthRegistry())
        try:
            monitor.tick()
            os.write(master, frame())
            for _ in range(20):
                monitor.tick()
                if monitor.health.snapshot()["components"]["serial"]["state"] == "ready":
                    break
                time.sleep(0.005)
            self.assertEqual(monitor.health.snapshot()["components"]["serial"]["state"], "ready")
            self.assertFalse(select.select([master], [], [], 0.05)[0], "monitor must send no bytes")
        finally:
            monitor.close()
            os.close(slave)
            os.close(master)

    def test_visual_crash_and_hang_retry_with_backoff_without_unlocking_motion(self):
        class Process:
            returncode = None

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = 0

            def wait(self, timeout=None):
                return self.returncode

        now = [1.0]
        started = []

        def popen(*args, **kwargs):
            process = Process()
            started.append(process)
            return process

        health = HealthRegistry(clock=lambda: now[0])
        supervisor = VisualSupervisor(
            lambda attempt: ["fake"], health, threading.Event(), popen=popen, clock=lambda: now[0]
        )
        supervisor.tick()
        started[-1].returncode = 1
        supervisor.tick()
        self.assertEqual(len(started), 1)
        now[0] += 0.5
        supervisor.tick()
        self.assertEqual(len(started), 2)
        now[0] += 11
        supervisor.tick()
        self.assertIsNone(supervisor.process)
        now[0] += 1
        supervisor.tick()
        self.assertEqual(len(started), 3)
        supervisor.observed_frame()
        self.assertIsNotNone(health.snapshot()["motion_latch"])
        supervisor.close()

    def test_replaying_old_task_logs_cannot_refresh_telemetry(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "events.jsonl"
            health = HealthRegistry()
            path.write_text(
                '[]\n{"kind":"serial_rx","data":false}\n'
                + json.dumps(dict(monotonic=0, kind="serial_rx", data=[decode(frame())]))
                + "\n"
            )
            reader = TaskTelemetryReader(path, health)
            reader.tick()
            self.assertEqual(health.snapshot()["components"]["serial"]["state"], "unknown")
            with path.open("a") as stream:
                stream.write(
                    json.dumps(
                        dict(monotonic=time.monotonic(), kind="serial_rx", data=[decode(frame())])
                    )
                    + "\n"
                )
            reader.tick()
            self.assertEqual(health.snapshot()["components"]["serial"]["state"], "ready")


class MissionShellTests(unittest.TestCase):
    def test_mission_handoff_reads_approach_telemetry_and_fault_requests_stop_once(self):
        class Process:
            returncode = None

            def __init__(self):
                self.signals = []

            def poll(self):
                return None

            def send_signal(self, value):
                self.signals.append(value)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args, _ = parser().parse_known_args(
                ["--observe-only", "--port", str(available_pair()), "--output", str(root)]
            )
            args.mission = True
            app = Application(args, [])
            for name in app.health.components:
                app.health.ready(name)
            process = app.task = Process()
            try:
                app.task_started = time.monotonic()
                (root / "mission/approach").mkdir(parents=True)
                (root / "mission/task_status.json").write_text(
                    json.dumps(dict(phase="left_approach"))
                )
                (root / "mission/approach/events.jsonl").write_text(
                    json.dumps(
                        dict(monotonic=time.monotonic(), kind="serial_rx", data=[decode(frame())])
                    )
                    + "\n"
                )
                app.tick_task()
                self.assertEqual(app.health.snapshot()["components"]["serial"]["state"], "ready")
                self.assertFalse(process.signals)
                app.health.report("detector", "unavailable", "MODEL_FAILED", "模型故障")
                app.tick_task()
                app.tick_task()
                self.assertEqual(len(process.signals), 1)
                app.task_stop_requested -= 9
                app.tick_task()
                self.assertEqual(app.health.snapshot()["motion_latch"]["code"], "STOP_UNCONFIRMED")
                with self.assertRaises(ValueError):
                    app.command("start", {})
                with self.assertRaises(ValueError):
                    app.command("retry", {})
                app.health.clear_motion_latch()
                app.health.ready("detector")
                app.health.motion_enabled = True
                with self.assertRaisesRegex(ValueError, "上一任务"):
                    app.command("start", {})
            finally:
                app.task = None
                app.server.server_close()

    def test_root_stop_file_survives_session_initialization_and_blocks_resume(self):
        from bucket_robot.runtime.application import WaitingView

        with tempfile.TemporaryDirectory() as folder:
            args, _ = parser().parse_known_args(
                ["--observe-only", "--port", str(available_pair()), "--output", folder]
            )
            args.mission = True
            app = Application(args, [])
            try:
                app.command("stop", {})
                self.assertEqual(app.health.snapshot()["motion_latch"]["code"], "OPERATOR_STOP")
                (Path(folder) / "STOP").touch()
                app.task_requested = True
                app.inspect_stop_file()
                self.assertFalse(app.task_requested)
                app.session = WaitingView(app.health)
                app.inspect_stop_file()
                self.assertIs(app.stop_file_session, app.session)
                self.assertIsNotNone(app.health.snapshot()["motion_latch"])
                with self.assertRaisesRegex(ValueError, "STOP文件"):
                    app.command("start", {})
                (Path(folder) / "STOP").unlink()
                app.inspect_stop_file()
                self.assertIsNone(app.stop_file_session)
                self.assertIsNotNone(app.health.snapshot()["motion_latch"])
            finally:
                app.server.server_close()

    def test_visual_cleanup_timeout_keeps_page_online_and_motion_locked(self):
        from types import SimpleNamespace
        import subprocess

        with tempfile.TemporaryDirectory() as folder:
            args, _ = parser().parse_known_args(
                ["--observe-only", "--port", str(available_pair()), "--output", folder]
            )
            args.mission = False
            app = Application(args, [])

            def failed_close():
                raise subprocess.TimeoutExpired("fake-visual", 2)

            app.supervisor = SimpleNamespace(close=failed_close)
            try:
                with self.assertLogs(level="ERROR"):
                    self.assertFalse(app.close_components())
                self.assertTrue(app.session.stop_event.is_set())
                self.assertEqual(app.status()["health"]["motion_latch"]["code"], "STOP_UNCONFIRMED")
                web = threading.Thread(target=app.server.serve_forever)
                web.start()
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{args.port}/health", timeout=2
                    ) as response:
                        self.assertEqual(response.status, 200)
                finally:
                    app.server.shutdown()
                    web.join(2)
            finally:
                app.server.server_close()


def available_pair():
    for _ in range(50):
        with socket.socket() as first:
            first.bind(("127.0.0.1", 0))
            port = first.getsockname()[1]
            try:
                with socket.socket() as second:
                    second.bind(("127.0.0.1", port + 1))
                return port
            except OSError:
                continue
    raise RuntimeError("no port pair")


class FakeVision:
    """A real local HTTP frame protocol, synthetic image and fake process lifecycle."""

    def __init__(self, command, **kwargs):
        self.returncode = None
        self.seq = 0
        self.failure = None
        port = int(command[command.index("--port") + 1])
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                time.sleep(0.035)
                if owner.failure == "camera":
                    self.send_response(503)
                    self.end_headers()
                    return
                owner.seq += 1
                now = time.monotonic()
                ready = owner.failure != "model"
                meta = dict(
                    frame_id=owner.seq,
                    arrival=now,
                    detector={"npu_run_ms": 3},
                    detections=[dict(left=480, top=250, right=720, bottom=450, score=0.9, anchor=1)]
                    if ready
                    else [],
                    components={
                        "detector": dict(
                            state="ready" if ready else "unavailable",
                            code="OK" if ready else "MODEL_FAILED",
                            message="正常" if ready else "模型失败",
                        ),
                        "calibration": dict(state="ready", code="OK", message="正常"),
                        "ranging": dict(state="ready", code="OK", message="正常"),
                    },
                )
                data = np.full((720, 1280, 3), 80, np.uint8).tobytes()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("X-Frame-Id", str(owner.seq))
                self.send_header(
                    "X-Observation",
                    json.dumps(meta, ensure_ascii=False).encode("utf-8").decode("latin-1"),
                )
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        self.worker = threading.Thread(target=self.server.serve_forever)
        self.worker.start()

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0
        self.server.shutdown()
        self.server.server_close()
        self.worker.join(2)

    def kill(self):
        self.terminate()

    def wait(self, timeout=None):
        return self.returncode


class ApplicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cal = self.root / "calibration.yml"
        storage = cv2.FileStorage(str(self.cal), cv2.FILE_STORAGE_WRITE)
        storage.write("P1", np.array([[700.0, 0, 640, 0], [0, 700, 360, 0], [0, 0, 1, 0]]))
        storage.release()
        self.binary = self.root / "fake-native"
        self.binary.touch()
        self.processes = []
        self.app = self.worker = None

    def tearDown(self):
        if self.app:
            self.app.stop_event.set()
        if self.worker:
            self.worker.join(8)
            self.assertFalse(self.worker.is_alive())
        self.temp.cleanup()

    def start(self, missing_calibration=False):
        args, native = parser().parse_known_args(
            [
                "--observe-only",
                "--native",
                str(self.binary),
                "--serial-port",
                str(self.root / "missing-uart"),
                "--port",
                str(available_pair()),
                "--output",
                str(self.root / "run"),
            ]
        )
        args.mission = False
        native += [
            "--calibration",
            str(self.cal if not missing_calibration else self.root / "missing-calibration"),
            "--calibration-sha",
            hashlib.sha256(self.cal.read_bytes()).hexdigest(),
        ]

        def process(command, **kwargs):
            instance = FakeVision(command, **kwargs)
            self.processes.append(instance)
            return instance

        self.app = Application(args, native, process_factory=process)
        self.worker = threading.Thread(target=self.app.run)
        self.worker.start()
        return self.app

    def get(self, path="/status"):
        with urllib.request.urlopen(
            f"http://127.0.0.1:{self.app.args.port}" + path, timeout=2
        ) as reply:
            return reply.read()

    def until(self, predicate, seconds=5):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.05)
        self.fail("condition not reached")

    def test_only_camera_missing_controller_keeps_real_http_video_and_detection(self):
        self.start()
        self.until(
            lambda: json.loads(self.get())["health"]["capabilities"]["detection"]["available"]
        )
        self.until(lambda: self.app.session.remote_jpeg is not None)
        status = json.loads(self.get())
        self.assertEqual(status["health"]["components"]["detector"]["message"], "正常")
        self.assertFalse(status["tracking"]["motion_available"])
        self.assertEqual(status["health"]["components"]["arm"]["code"], "DEVICE_STATE_UNKNOWN")
        self.assertIn(b"<html", self.get("/"))
        self.assertGreater(len(self.get("/frame.jpg?remote=1")), 100)
        native_command = self.app.native_command(1)
        self.assertEqual(
            native_command[native_command.index("--tracking-overlay") + 1],
            str(self.app.args.output / "tracking_overlay.json"),
        )

    def test_model_fault_keeps_video_and_auto_recovers_detection(self):
        self.start()
        self.until(lambda: bool(self.processes) and self.app.session.remote_jpeg is not None)
        self.processes[-1].failure = "model"
        self.until(
            lambda: not json.loads(self.get())["health"]["capabilities"]["detection"]["available"]
        )
        self.assertGreater(len(self.get("/frame.jpg?remote=1")), 100)
        self.processes[-1].failure = None
        self.until(
            lambda: json.loads(self.get())["health"]["capabilities"]["detection"]["available"]
        )

    def test_preview_encoder_failure_reports_cause_while_page_and_detection_survive(self):
        with patch("cv2.imencode", side_effect=RuntimeError("synthetic encoder failure")):
            self.start()
            self.until(
                lambda: (
                    json.loads(self.get())["health"]["components"]["preview"]["code"]
                    == "PREVIEW_FAILED"
                )
            )
            self.until(
                lambda: json.loads(self.get())["health"]["capabilities"]["detection"]["available"]
            )
            self.assertIn(b"<html", self.get("/"))
            self.assertTrue(
                json.loads(self.get())["health"]["capabilities"]["detection"]["available"]
            )
        self.until(lambda: json.loads(self.get())["health"]["capabilities"]["preview"]["available"])

    def test_camera_loss_page_stays_online_old_frame_rejected_and_recovers(self):
        self.start()
        self.until(lambda: bool(self.processes) and self.app.session.remote_jpeg is not None)
        self.processes[-1].failure = "camera"
        time.sleep(0.7)
        self.assertEqual(
            json.loads(self.get("/health"))["components"]["camera"]["state"], "unavailable"
        )
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.get("/frame.jpg?remote=1")
        self.assertEqual(error.exception.code, 503)
        self.processes[-1].failure = None
        self.until(lambda: json.loads(self.get())["health"]["capabilities"]["preview"]["available"])

    def test_bad_configuration_does_not_prevent_page(self):
        self.start(missing_calibration=True)
        self.until(lambda: bool(self.processes))
        self.assertIn(b"<html", self.get("/"))
        self.assertTrue(self.worker.is_alive())
        self.until(lambda: self.app.session.remote_jpeg is not None)
        self.assertEqual(
            json.loads(self.get())["health"]["components"]["calibration"]["code"],
            "CALIBRATION_INVALID",
        )
        self.assertFalse(json.loads(self.get())["health"]["capabilities"]["detection"]["available"])

    def test_api_auth_and_server_side_capability_gate(self):
        self.start()
        self.until(lambda: bool(self.processes))
        url = f"http://127.0.0.1:{self.app.args.port}/api/start"
        for token, status in (("bad-token", 403), (self.app.token, 400)):
            request = urllib.request.Request(url, b"{}", headers={"X-Control-Token": token})
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request, timeout=2)
            self.assertEqual(error.exception.code, status)

    def test_backend_start_failure_and_missing_optional_python_dependency_leave_ui_up(self):
        args, _ = parser().parse_known_args(
            ["--observe-only", "--port", str(available_pair()), "--output", str(self.root / "run")]
        )
        args.mission = False
        self.app = Application(args, [])
        with patch(
            "bucket_robot.runtime.application.importlib.import_module",
            side_effect=ImportError("cv2 unavailable"),
        ):
            self.worker = threading.Thread(target=self.app.run)
            self.worker.start()
            self.until(
                lambda: (
                    json.loads(self.get())["health"]["components"]["application"]["code"]
                    == "OPTIONAL_DEPENDENCY_FAILED"
                )
            )
            self.assertIn(b"<html", self.get("/"))
