"""Read-only mission display, including partial writes and live post-action evidence."""

import json
from pathlib import Path
import tempfile
import unittest
from bucket_robot.runtime.task_progress import TaskProgressReader


class TaskProgressTests(unittest.TestCase):
    def test_child_stage_and_actual_post_evidence_are_used(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reader = TaskProgressReader(root)
            self.assertIsNone(reader.snapshot(None, observe_only=True)["phase"])
            (root / "task_lifecycle.jsonl").write_text(
                '{"stage":"acquire"}\n{"stage":"track"}\n{broken'
            )
            (root / "mission_profile.json").write_text('{"execute_post":true}')
            (root / "task_status.json").write_text('{"phase":"left_approach"}')
            (root / "approach").mkdir()
            (root / "approach/task_status.json").write_text('{"phase":"waiting_target"}')
            result = reader.snapshot(None, running=True)
            self.assertEqual(result["phase"], "waiting_target")
            self.assertEqual(result["stages"], ["acquire", "track"])
            (root / "task_status.json").write_text(
                json.dumps(
                    dict(
                        phase="post_raising_B_tracking_paused",
                        execution_progress=dict(lateral_completion="timed_estimate_no_odometry"),
                    )
                )
            )
            result = reader.snapshot(None, running=True)
            self.assertEqual(result["post"]["lateral_completion"], "timed_estimate_no_odometry")

    def test_partial_result_never_retains_a_previous_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reader = TaskProgressReader(root)
            result = root / "post_execution.json"
            result.write_text('{"phase":"complete"}')
            self.assertEqual(reader.snapshot(None)["post"]["phase"], "complete")
            result.write_text("{")
            self.assertEqual(reader.snapshot(None)["post"], {})
            result.unlink()
            self.assertEqual(reader.snapshot(None)["post"], {})
