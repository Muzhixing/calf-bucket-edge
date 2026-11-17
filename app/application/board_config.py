#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Board identity and deployment configuration."""

import os
from dataclasses import dataclass
from typing import Optional


def _get_float_env(name: str, default: Optional[float] = None) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def split_csv_env(name: str):
    raw = os.getenv(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def env_enabled(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class BoardConfig:
    """Stable identity fields included in board-to-server payloads."""

    device_id: str = "robot001"
    device_name: str = "calf-feeder-001"
    farm_id: int = 1
    location_x: Optional[float] = 0.0
    location_y: Optional[float] = 0.0

    @classmethod
    def from_env(cls):
        return cls(
            device_id=os.getenv("DEVICE_ID", "robot001"),
            device_name=os.getenv("DEVICE_NAME", "calf-feeder-001"),
            farm_id=_get_int_env("FARM_ID", 1),
            location_x=_get_float_env("LOCATION_X", 0.0),
            location_y=_get_float_env("LOCATION_Y", 0.0),
        )
