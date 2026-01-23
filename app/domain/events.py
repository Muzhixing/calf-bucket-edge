#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Domain events for the ranging system."""

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Mapping


@dataclass(frozen=True)
class RangingUpdateEvent:
    """Represents a single ranging cycle update."""

    frame_ts: float
    detected: bool
    distance_m: Optional[float]
    detections: Sequence[Mapping[str, Any]]
    display_frame: Any = None
    left_frame: Any = None
