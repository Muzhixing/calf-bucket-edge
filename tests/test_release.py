import contextlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import zipfile

from bucket_robot.missions.coordinator import command
from bucket_robot.runtime.release import ReleaseLayout


class ReleaseTests(unittest.TestCase):
    def make_release(self, root):
        builder = Path(__file__).resolve().parents[1] / "tools/release/build_release.py"
        spec = importlib.util.spec_from_file_location("release_builder", builder)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in ["native/lib", "assets", "range"]:
            (root / name).mkdir(parents=True)
        for name in [
            "native/vision",
            "assets/calibration.yml",
            "assets/calibration.json",
            "assets/detector.rknn",
            "assets/fast_acv.rknn",
            "range/libfastacv.so",
            "range/librknnrt.so",
        ]:
            (root / name).write_bytes(b"fixture")
        wheel = root / "fixture.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            for name in [
                "__main__.py",
                "resources/arm_reference.json",
                "resources/left_view_reference.json",
                "runtime/index.html",
            ]:
                archive.writestr("bucket_robot/" + name, "fixture")
        args = SimpleNamespace(
            out=root / "release",
            version="test",
            wheel=wheel,
            native_dir=root / "native",
            assets_dir=root / "assets",
            detector=root / "assets/detector.rknn",
            range_model=root / "assets/fast_acv.rknn",
            calibration=root / "assets/calibration.json",
            range_runtime=root / "range",
            native_source_relation="binary_only",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            module.build(args)
        return ReleaseLayout(args.out)

    def test_complete_manifest_covers_entry_points_and_all_assets(self):
        with tempfile.TemporaryDirectory() as folder:
            layout = self.make_release(Path(folder))
            manifest = layout.verify()
            paths = {f["path"] for f in manifest["files"]}
            self.assertIn("run_mission.sh", paths)
            self.assertIn("run_track.sh", paths)
            self.assertIn("python/bucket_robot/__main__.py", paths)
            args = command(layout, Path("/tmp/output"), 8899, 120)
            self.assertIn("--track-b", args)
            self.assertIn("--recover-loss", args)
            self.assertNotIn("sh", args)

    def test_changed_missing_or_extra_file_is_rejected(self):
        for kind in ["changed", "missing", "extra"]:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                layout = self.make_release(Path(folder))
                file = layout.assets / "detector.rknn"
                if kind == "changed":
                    file.write_bytes(b"modified")
                elif kind == "missing":
                    file.unlink()
                else:
                    (layout.root / "unlisted.py").write_text("extra")
                with self.assertRaises(ValueError):
                    layout.verify()

    def test_runtime_integrity_scopes_missing_detector_to_detection(self):
        with tempfile.TemporaryDirectory() as folder:
            layout = self.make_release(Path(folder))
            self.assertFalse(any(layout.component_issues().values()))
            (layout.assets / "detector.rknn").unlink()
            issues = layout.component_issues()
            self.assertEqual(issues["detector"], ["assets/detector.rknn"])
            self.assertFalse(issues["backend"])
            self.assertFalse(issues["motion_config"])

    def test_runtime_integrity_rejects_omitted_required_asset(self):
        import json

        with tempfile.TemporaryDirectory() as folder:
            layout = self.make_release(Path(folder))
            path = layout.root / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["files"] = [
                row for row in manifest["files"] if row["path"] != "assets/detector.rknn"
            ]
            path.write_text(json.dumps(manifest))
            self.assertEqual(layout.component_issues()["detector"], ["assets/detector.rknn"])
