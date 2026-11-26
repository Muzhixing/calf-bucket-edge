"""Presentation contracts; no browser or hardware is needed."""

from pathlib import Path
import shutil
import subprocess
import unittest


class ConsoleViewTests(unittest.TestCase):
    def test_reading_freshness_and_real_component_states(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js required for presentation-only tests")
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                node,
                "--test",
                str(root / "tests/ui/console_view.test.cjs"),
                str(root / "tests/ui/video_stream.test.cjs"),
            ],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
