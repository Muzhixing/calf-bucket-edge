"""Packaged defaults; site overrides are explicit and never depend on a cwd."""

from importlib.resources import files
from pathlib import Path
import os


def resource(name: str) -> Path:
    directory = os.environ.get("BUCKET_CONFIG_DIR")
    return Path(directory) / name if directory else Path(str(files(__package__).joinpath(name)))


def default_reference() -> Path:
    return resource("arm_reference.json")
