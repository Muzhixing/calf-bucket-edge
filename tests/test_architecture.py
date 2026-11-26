"""Contracts that prevent task code from silently forking mature capabilities."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import contextlib
import io
import sys
from unittest.mock import patch

import bucket_robot
from bucket_robot.motion.arm import Limits
from bucket_robot.missions.vertical import VerticalSupervisor
from bucket_robot.resources import default_reference
from bucket_robot.tracking.controller import TrackingController
from bucket_robot.tracking.policy import FollowPolicy


class TrackingContractTests(unittest.TestCase):
    def test_omitting_B_requires_explicit_diagnostic_selection_before_any_IO(self):
        from bucket_robot.missions import forward

        args = [
            "test",
            "--out",
            "unused",
            "--reference",
            "unused",
            "--left-reference",
            "unused",
            "--calibration",
            "unused",
            "--auto-target",
            "--recover-loss",
        ]
        with (
            patch.object(sys, "argv", args),
            patch.object(forward, "Observer", side_effect=AssertionError("no IO")),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as error:
                forward.main()
        self.assertEqual(error.exception.code, 2)

    def test_both_contexts_match_180_pre_refactor_decisions(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/ab_control_trace.json").read_text())

        def policy():
            limits = Limits(default_reference())
            self.assertIsNone(limits.aim_area)  # Production restores the recorded centre aim.
            return FollowPolicy(limits, *fixture["intrinsics"])

        controllers = [TrackingController(policy()), VerticalSupervisor(policy()).controller]
        for row in fixture["rows"]:
            port = SimpleNamespace(
                state=row["state"], target=row["target"], b_origin=row["b_origin"]
            )
            port.can_send = lambda: True
            port.angle_at = lambda timestamp: port.state["servo_software_deg"]
            for controller in controllers:
                result = controller.step(
                    port, row["target_observation"], {}, row["now"], row["identity"]
                )
                actual = dict(
                    goal=list(result.goal) if result.goal is not None else None,
                    stop=result.stop,
                    holding=result.holding,
                    normalizing_b=result.normalizing_b,
                )
                self.assertEqual(actual, row["expected"], row["now"])

    def test_stale_target_returns_stop_without_advancing_policy(self):
        policy = FollowPolicy(Limits(default_reference()), 700, 700, 640, 360)
        controller = TrackingController(policy)
        result = controller.step(None, {"timestamp": 1}, {}, 2)
        self.assertTrue(result.stop)
        self.assertIsNone(result.goal)
        self.assertIsNone(policy.last_send)


class DependencyBoundaryTests(unittest.TestCase):
    def test_dependency_direction_matches_documented_layers(self):
        root = Path(bucket_robot.__file__).parent
        allowed = {
            "domain": {"domain"},
            "resources": {"resources"},
            "transport": {"domain", "transport"},
            "perception": {"domain", "perception"},
            "tracking": {"domain", "perception", "tracking"},
            "motion": {"domain", "transport", "motion"},
            "missions": {"domain", "resources", "perception", "tracking", "motion", "missions"},
        }
        for file in root.rglob("*.py"):
            layer = file.relative_to(root).parts[0]
            if layer not in allowed:
                continue
            for node in ast.walk(ast.parse(file.read_text())):
                if isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                elif isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                else:
                    continue
                for name in names:
                    if name == "bucket_robot.runtime.release" and layer == "missions":
                        continue
                    parts = name.split(".")
                    if len(parts) > 1 and parts[0] == "bucket_robot":
                        self.assertIn(parts[1], allowed[layer], str(file) + ": " + name)

    def test_installed_core_has_no_legacy_or_tool_imports(self):
        root = Path(bucket_robot.__file__).parent
        legacy = {
            "arm",
            "control",
            "vision",
            "receive",
            "chassis_probe",
            "track_forward_probe",
            "vertical_tracking",
            "target_recovery",
            "side_approach",
            "runtime",
        }
        for file in root.rglob("*.py"):
            tree = ast.parse(file.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                elif isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                else:
                    continue
                for name in names:
                    self.assertNotIn(name.split(".")[0], legacy, str(file))
                    self.assertFalse(
                        name.startswith(("tools.", "tests.", "experiments.")), str(file)
                    )
            self.assertNotIn("sys.path", file.read_text(), str(file))

    def test_missions_cannot_implement_axis_policy_or_ack_transactions(self):
        root = Path(bucket_robot.__file__).parent / "missions"
        for file in root.glob("*.py"):
            tree = ast.parse(file.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertNotIn(
                        node.func.attr,
                        {"next_a", "next_b", "choose_axis", "write_control_frame"},
                        str(file),
                    )
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Attribute):
                            self.assertNotIn(
                                target.attr,
                                {"pending_counter", "move_counter", "last_tx"},
                                str(file),
                            )
