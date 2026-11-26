"""Local contracts for startup, ownership, cleanup and explicit mission policy."""

import ast
import json
import re
from pathlib import Path
import tempfile
import threading
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch

from bucket_robot.motion.arm import Limits
from bucket_robot.resources import default_reference
from bucket_robot.runtime.preparation import StartupFailure, initialize
from bucket_robot.runtime.startup import PowerCycleStore, can_auto_prepare, default_startup_state
from bucket_robot.runtime.application import Application, parser
from bucket_robot.perception.observation import TaskObservationPublisher, TaskObservationReader
from bucket_robot.perception.relocation import FeatureRelocationJob
from bucket_robot.perception.session import RecoveringTarget
from bucket_robot.missions.lifecycle import stop_and_drain
from bucket_robot.missions.recovery import RecoveryGate, RecoverySettings
from bucket_robot.missions.options import parse_options
from bucket_robot.missions.profile import MissionProfile
from bucket_robot.missions.stages import MissionStages, Stage


class StartupContracts(unittest.TestCase):
    def fake_port(self, stop_error=None, poll_error=None):
        state = dict(
            lift_zero_confirmed=True,
            slide_zero_confirmed=True,
            servo_software_deg=73,
            lift_position_mm=0,
            slide_position_mm=0,
            enabled=False,
            state=0,
            angle_valid=True,
            heading_reference_established=True,
            system_fault=0,
            motor_fault=0,
            yaw_deg=0,
        )

        def poll():
            if poll_error:
                raise RuntimeError(poll_error)

        return SimpleNamespace(
            state=state, poll=poll, preflight_ready=lambda: True, shutdown=lambda: stop_error
        )

    def test_shutdown_failure_cannot_return_ready_and_survives_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = Path(folder) / "power-cycle.json"
            with self.assertRaises(StartupFailure) as error:
                initialize(
                    Limits(default_reference()),
                    ledger,
                    lambda *e: None,
                    port_factory=lambda *a: self.fake_port("missing_disable_ack"),
                )
            result = error.exception.result
            self.assertFalse(result.ready)
            self.assertFalse(result.stop_confirmed)
            self.assertEqual(result.stop_error, "missing_disable_ack")
            with self.assertRaises(StartupFailure) as retry:
                initialize(
                    Limits(default_reference()),
                    ledger,
                    lambda *e: None,
                    port_factory=lambda *a: self.fail("must not reopen serial"),
                )
            self.assertFalse(retry.exception.result.stop_confirmed)
            self.assertEqual(json.loads(ledger.read_text())["stop_error"], "missing_disable_ack")

    def test_original_failure_and_cleanup_failure_are_both_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(StartupFailure) as error:
                initialize(
                    Limits(default_reference()),
                    Path(folder) / "state.json",
                    lambda *e: None,
                    port_factory=lambda *a: self.fake_port("missing_ack", "home_timeout"),
                )
            self.assertIn("home_timeout", error.exception.result.error)
            self.assertEqual(error.exception.result.stop_error, "missing_ack")

    def test_legacy_uncertain_write_blocks_both_entry_paths_and_keeps_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            legacy = Path(folder) / "launcher-startup.json"
            store = PowerCycleStore(legacy, 8, "test")
            store.confirm(str(uuid.uuid4()))
            store.phase("disable_pending")
            store.before_zero()
            store.close()
            original = legacy.read_bytes()
            canonical = Path(folder) / "power-cycle.json"
            for _ in range(2):
                store = PowerCycleStore(canonical, 8, "test")
                try:
                    self.assertFalse(can_auto_prepare(store))
                    self.assertTrue(store.data["legacy_startup_imported"]["zero_attempted"])
                    with self.assertRaisesRegex(RuntimeError, "another_process"):
                        PowerCycleStore(legacy, 8, "test")
                finally:
                    store.close()
            self.assertEqual(legacy.read_bytes(), original)
            self.assertEqual(Path(parser().get_default("startup_state")), default_startup_state())
            # A deliberately confirmed new epoch retains the migration marker.
            store = PowerCycleStore(canonical, 8, "test")
            store.confirm(str(uuid.uuid4()))
            self.assertEqual(store.data["legacy_startup_imported"]["phase"], "zero_pending")
            store.close()

    def test_canonical_owner_also_excludes_an_old_launcher_before_first_migration(self):
        with tempfile.TemporaryDirectory() as folder:
            canonical = Path(folder) / "power-cycle.json"
            owner = PowerCycleStore(canonical, 8, "test")
            try:
                with self.assertRaisesRegex(RuntimeError, "another_process"):
                    PowerCycleStore(canonical.with_name("launcher-startup.json"), 8, "test")
            finally:
                owner.close()

    def test_preview_retains_stop_fault_and_every_resume_command_is_blocked(self):
        with tempfile.TemporaryDirectory() as folder:
            result = Path(folder) / "result.json"
            result.write_text(
                json.dumps(dict(ready=False, stop_confirmed=False, stop_error="missing_ack"))
            )
            args = parser().parse_args(
                [
                    "--port",
                    "0",
                    "--observe-only",
                    "--startup-result",
                    str(result),
                    "--startup-state",
                    str(Path(folder) / "state.json"),
                    "--output",
                    folder,
                ]
            )
            args.mission = True
            app = Application(args, [])
            try:
                self.assertEqual(app.status()["health"]["motion_latch"]["code"], "STOP_UNCONFIRMED")
                for action in ("start", "retry", "confirm_power_cycle", "confirm_mcu_restart"):
                    with self.assertRaises(ValueError, msg=action):
                        app.command(action, {})
            finally:
                app.server.server_close()

    def test_malformed_startup_result_keeps_status_service_available_and_locked(self):
        with tempfile.TemporaryDirectory() as folder:
            result = Path(folder) / "result.json"
            result.write_text("[]")
            args = parser().parse_args(
                [
                    "--port",
                    "0",
                    "--observe-only",
                    "--startup-result",
                    str(result),
                    "--startup-state",
                    str(Path(folder) / "state.json"),
                    "--output",
                    folder,
                ]
            )
            args.mission = False
            app = Application(args, [])
            try:
                self.assertEqual(app.status()["health"]["motion_latch"]["code"], "STOP_UNCONFIRMED")
            finally:
                app.server.server_close()

    def test_ledger_stat_permission_error_keeps_status_service_available(self):
        with tempfile.TemporaryDirectory() as folder:
            args = parser().parse_args(["--port", "0", "--output", folder])
            args.mission = False
            with patch.object(Path, "is_file", side_effect=PermissionError("denied")):
                app = Application(args, [])
            try:
                self.assertEqual(app.status()["health"]["motion_latch"]["code"], "STOP_UNCONFIRMED")
            finally:
                app.server.server_close()


class ObservationContracts(unittest.TestCase):
    def test_two_buckets_loss_and_staleness_never_substitute_preview_target(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "mission"
            root.mkdir()
            (root / "task_status.json").write_text("{}")
            now = [10.0]
            publisher = TaskObservationPublisher(root, clock=lambda: now[0])
            reader = TaskObservationReader(root, clock=lambda: now[0])
            target = dict(timestamp=10, offset_px=[1, 2], box=[100, 100, 300, 400])
            publisher.publish(dict(target=target, identity=1, frame_id=50))
            self.assertEqual(reader.read()["target"], target)
            publisher.publish(
                dict(target=None, identity=1, frame_id=51, reason="original_occluded")
            )
            self.assertIsNone(reader.read()["target"])
            self.assertEqual(reader.read()["target_id"], 1)
            now[0] = 10.2
            publisher.publish(dict(target=dict(target, timestamp=10.2), identity=1, frame_id=52))
            now[0] = 11
            self.assertIsNone(reader.read()["target"])
            self.assertEqual(reader.read()["reason"], "task_observation_stale")

    def test_ui_and_child_handoff_use_task_identity_not_independent_preview(self):
        with tempfile.TemporaryDirectory() as folder:
            args = parser().parse_args(
                [
                    "--port",
                    "0",
                    "--output",
                    folder,
                    "--startup-state",
                    str(Path(folder) / "state.json"),
                ]
            )
            args.mission = True
            root = Path(folder) / "mission"
            root.mkdir()
            child = root / "approach"
            child.mkdir()
            (root / "task_status.json").write_text('{"phase":"left_approach"}')
            publisher = TaskObservationPublisher(child, task_root=root)
            import time

            now = time.monotonic()
            publisher.publish(
                dict(
                    target=None,
                    identity=1,
                    frame_id=1,
                    reason="original_occluded",
                    metadata=dict(arrival=now),
                )
            )
            app = Application(args, [])
            app.task_attempted = True
            app.session.status = lambda: dict(
                tracking=dict(target=dict(box=[1, 2, 3, 4]), target_id=99)
            )
            try:
                status = app.status()
                self.assertIsNone(status["tracking"]["target"])
                self.assertEqual(status["tracking"]["target_id"], 1)
                self.assertEqual(status["tracking"]["reason"], "original_occluded")
                self.assertIsNone(status["distance_m"])
            finally:
                app.server.server_close()

    def test_actual_preview_draws_only_the_task_target_and_clears_on_loss(self):
        import time
        import numpy as np
        from bucket_robot.runtime.stationary import Session
        from tests.test_tracking import Reference

        reference = Reference()
        self.addCleanup(reference.directory.cleanup)
        session = Session(reference.directory.name, reference.limits, motion_allowed=False)
        other_box = [800, 300, 1000, 600]
        task_box = [100, 200, 300, 500]
        for visible in (True, False):
            session.stop_event.clear()
            now = time.monotonic()
            session.tracking = dict(timestamp=now, box=other_box, offset_px=[260, 90])
            target = dict(timestamp=now, box=task_box, offset_px=[-440, -10]) if visible else None
            session.task_observation = lambda: dict(target=target, tracking_state="tracking")

            class OneFrame:
                def next(self, previous):
                    session.stop_event.set()
                    return np.zeros((720, 1280, 3), np.uint8), dict(
                        frame_id=1, arrival=now, detections=[]
                    )

            with patch("bucket_robot.runtime.stationary.cv2.rectangle") as draw:
                session.preview_loop(OneFrame())
                boxes = [(call.args[1], call.args[2]) for call in draw.call_args_list]
                self.assertNotIn((tuple(other_box[:2]), tuple(other_box[2:])), boxes)
                self.assertEqual((tuple(task_box[:2]), tuple(task_box[2:])) in boxes, visible)
                self.assertIsNotNone(session.jpeg)


class CleanupContracts(unittest.TestCase):
    def test_nested_blocked_job_is_not_reported_stopped_or_handoff_ready(self):
        release = threading.Event()
        job = object.__new__(FeatureRelocationJob)
        job.cancel = threading.Event()
        job.thread = threading.Thread(target=release.wait, name="blocked-proof")
        job.thread.start()
        manager = RecoveringTarget()
        manager.feature_job = job
        worker = SimpleNamespace(
            cleanup_children=lambda: (manager,), request_stop=manager.request_stop
        )
        calls = []
        port = SimpleNamespace(
            phase="idle",
            state=dict(enabled=False),
            stop=lambda: calls.append("stop"),
            poll=lambda: [],
        )
        try:
            job.request_stop()
            result = job.await_stopped(0.01)
            self.assertFalse(result.stopped)
            self.assertIn("blocked-proof", result.pending)
            error = stop_and_drain(port, [worker], timeout=0.02)
            self.assertIn("worker_cleanup_timeout", error)
            self.assertEqual(calls, ["stop"])
        finally:
            release.set()
            job.thread.join(1)
        self.assertTrue(manager.await_stopped().stopped)


class PolicyAndEntryContracts(unittest.TestCase):
    def test_exact_low_speed_cap_converges_with_deterministic_clock(self):
        gate = RecoveryGate(30, left=999, settings=RecoverySettings(cruise=20))
        speeds = []
        for i in range(141):
            now = i * 0.05
            cap = 20 if now < 4 else 4
            v, _ = gate.evaluate(
                now, dict(timestamp=now, offset_px=[0, 0]), 1, 100, True, speed_cap=cap
            )
            speeds.append(v)
        self.assertEqual(max(speeds), 20)
        ramp = speeds[80:]
        self.assertTrue(all(b <= a for a, b in zip(ramp, ramp[1:])))
        self.assertTrue(all(0 <= a - b <= 1 for a, b in zip(ramp, ramp[1:])))
        self.assertEqual(speeds[-10:], [4] * 10)
        for i in range(141, 191):
            now = i * 0.05
            gate.evaluate(now, None, 1, 100, True)
        self.assertEqual(gate.velocity, 0)
        self.assertTrue(gate.disable)

    def test_production_profile_and_stage_order_are_explicit(self):
        args = parse_options(
            [
                "--profile",
                "production-center",
                "--out",
                "unused",
                "--reference",
                "unused",
                "--left-reference",
                "unused",
                "--calibration",
                "unused",
                "--approach-model",
                "unused",
                "--approach-calibration",
                "unused",
            ]
        )
        effective = MissionProfile.from_options(args).as_dict()
        self.assertEqual(effective["cruise_speed"], 80)
        self.assertTrue(effective["track_b"])
        self.assertFalse(effective["a_priority_departure"])
        self.assertEqual(effective["post_translate"]["duration_s"], 4)
        with tempfile.TemporaryDirectory() as folder:
            stage = MissionStages(Path(folder))
            with self.assertRaisesRegex(RuntimeError, "invalid_mission_transition"):
                stage.transition(Stage.POST_ACTION)
            for value in (
                Stage.TRACK,
                Stage.POST_ALIGN,
                Stage.DRAIN,
                Stage.POST_ACTION,
                Stage.COMPLETE,
            ):
                stage.transition(value)

    def test_stage_journal_failure_is_explicit_and_does_not_interrupt_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            stage = MissionStages(Path(folder))
            with patch.object(Path, "open", side_effect=OSError("disk_full")):
                stage.transition(Stage.DRAIN)
            with self.assertRaisesRegex(RuntimeError, "stage_journal_failed"):
                stage.check()
            self.assertEqual(stage.stage, Stage.DRAIN)

    def test_forward_and_stereo_adapters_share_implementation_and_frame_contract(self):
        from bucket_robot.perception.observer import TargetObserver, StereoObserver, FrameBundle
        from bucket_robot.missions.forward import Observer

        self.assertIs(Observer, TargetObserver)
        self.assertIs(StereoObserver.run, TargetObserver.run)
        target = dict(timestamp=1, offset_px=[0, 0], box=[0, 0, 10, 10])
        for observer in (
            Observer(1, None, "unused", [0, 0, 10, 10]),
            StereoObserver(1, None, "unused", [0, 0, 10, 10]),
        ):
            observer.latest = dict(target=target, identity=1, frame_id=1)
            self.assertIsInstance(observer.frame(), FrameBundle)
            self.assertEqual(observer.get()["identity"], 1)
            self.assertIs(observer.range_snapshot()[0]["target"], target)

    def test_manifest_covers_installed_and_field_entries_without_uart_forks(self):
        root = Path(__file__).parents[1]
        manifest = json.loads(
            (root / "src/bucket_robot/resources/production_entries.json").read_text()
        )
        project = (root / "pyproject.toml").read_text()
        script_section = project.split("[project.scripts]", 1)[1].split("[", 1)[0]
        actual = dict(re.findall(r'^([\w-]+)\s*=\s*"([^"]+)"', script_section, re.M))
        self.assertEqual(actual, manifest["console_scripts"])
        for name, function in manifest["console_scripts"].items():
            self.assertIn(f'{name} = "{function}"', project)
        for name in manifest["field_entries"]:
            source = (root / "tools/field" / name).read_text()
            ast.parse(source)
            self.assertIn("bucket_robot.runtime.preparation", source)
            self.assertNotIn("launcher-startup.json", source)
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Attribute):
                    self.assertNotIn(
                        node.attr, ("disabled_command", "before_zero", "write_control_frame")
                    )
        self.assertEqual(
            Path(parser().get_default("startup_state")).name, manifest["startup_ledger"]
        )
