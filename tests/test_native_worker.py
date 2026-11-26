"""Compile and fault-inject the platform-independent native isolation boundary."""

from pathlib import Path
import resource
import shutil
import subprocess
import tempfile
import unittest


class WorkerChannelTests(unittest.TestCase):
    def test_native_worker_exceptions_crash_hang_and_restart(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        if compiler is None:
            self.skipTest("C++17 compiler unavailable")
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            binary = str(Path(folder) / "worker-test")
            subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-pthread",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-I",
                    str(root / "deploy/vision_cpp"),
                    str(root / "tests/native/worker_channel_test.cpp"),
                    "-o",
                    binary,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            def no_core():
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

            result = subprocess.run(
                [binary], check=True, capture_output=True, text=True, timeout=12, preexec_fn=no_core
            )
            self.assertIn("bounded shutdown passed", result.stdout)
