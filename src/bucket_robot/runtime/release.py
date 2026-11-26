"""Immutable release layout and offline integrity verification."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class ReleaseLayout:
    root: Path

    @property
    def assets(self) -> Path:
        return self.root / "assets"

    @property
    def native(self) -> Path:
        return self.root / "native" / "vision"

    @property
    def python(self) -> Path:
        return self.root / "python"

    @property
    def range_runtime(self) -> Path:
        return self.root / "assisted_runtime"

    @property
    def config(self) -> Path:
        return self.python / "bucket_robot" / "resources"

    def verify(self) -> dict:
        """Reject missing, modified, undeclared, escaping or symlinked files."""
        manifest = json.loads((self.root / "manifest.json").read_text())
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported_release_manifest")
        declared = set()
        for item in manifest["files"]:
            rel = Path(item["path"])
            if rel.is_absolute() or ".." in rel.parts or str(rel) in declared:
                raise ValueError("invalid_manifest_path:" + str(rel))
            declared.add(str(rel))
            path = self.root / rel
            if (
                path.is_symlink()
                or not path.is_file()
                or not path.resolve().is_relative_to(self.root.resolve())
            ):
                raise ValueError("missing_release_file:" + str(rel))
            data = path.read_bytes()
            if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("release_hash_mismatch:" + str(rel))
        actual = {str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file()}
        actual.discard("manifest.json")
        if actual != declared:
            raise ValueError("undeclared_release_files:" + repr(sorted(actual ^ declared)))
        required = {
            "native/vision",
            "assets/detector.rknn",
            "assets/fast_acv.rknn",
            "assets/calibration.yml",
            "assets/calibration.json",
            "assisted_runtime/libfastacv.so",
            "assisted_runtime/librknnrt.so",
            "python/bucket_robot/__main__.py",
            "python/bucket_robot/resources/arm_reference.json",
            "python/bucket_robot/resources/left_view_reference.json",
            "python/bucket_robot/runtime/index.html",
        }
        if not required <= declared:
            raise ValueError("incomplete_release:" + repr(sorted(required - declared)))
        return manifest

    def component_issues(self) -> dict:
        """Runtime asset checks are scoped; bad optional assets do not block UI."""
        issues = {
            name: []
            for name in (
                "backend",
                "detector",
                "calibration",
                "motion_config",
                "approach",
                "application",
            )
        }

        def group(path):
            if path == "assets/detector.rknn":
                return "detector"
            if path in ("assets/calibration.yml",):
                return "calibration"
            if "resources/arm_reference.json" in path:
                return "motion_config"
            if path.startswith("assisted_runtime/") or path in (
                "assets/fast_acv.rknn",
                "assets/calibration.json",
                "python/bucket_robot/resources/left_view_reference.json",
            ):
                return "approach"
            if path.startswith("native/") and not path.startswith("native/source/"):
                return "backend"
            return "application"

        try:
            manifest = json.loads((self.root / "manifest.json").read_text())
            if manifest.get("schema_version") != 1:
                raise ValueError("unsupported_release_manifest")
            declared = set()
            for row in manifest["files"]:
                rel = Path(row["path"])
                path = self.root / rel
                if rel.is_absolute() or ".." in rel.parts or str(rel) in declared:
                    raise ValueError("invalid_manifest_path")
                declared.add(str(rel))
                if (
                    not path.is_file()
                    or path.is_symlink()
                    or not path.resolve().is_relative_to(self.root.resolve())
                ):
                    issues[group(str(rel))].append(str(rel))
                    continue
                data = path.read_bytes()
                if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                    issues[group(str(rel))].append(str(rel))
            for required in (
                "native/vision",
                "assets/detector.rknn",
                "assets/calibration.yml",
                "assets/fast_acv.rknn",
                "assets/calibration.json",
                "assisted_runtime/libfastacv.so",
                "assisted_runtime/librknnrt.so",
                "python/bucket_robot/__main__.py",
                "python/bucket_robot/runtime/index.html",
                "python/bucket_robot/resources/arm_reference.json",
                "python/bucket_robot/resources/left_view_reference.json",
            ):
                if required not in declared:
                    issues[group(required)].append(required)
        except (OSError, ValueError, KeyError, TypeError):
            for key in ("backend", "motion_config", "approach"):
                issues[key].append("manifest.json")
        return issues
