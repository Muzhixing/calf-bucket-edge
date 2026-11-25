"""Structural interfaces accepted by the pure tracking controller."""

from typing import Any, Mapping, Optional, Protocol, Sequence


class TrackingPort(Protocol):
    state: Mapping[str, Any]
    target: Optional[Sequence[float]]
    b_origin: float

    def can_send(self) -> bool: ...

    def angle_at(self, timestamp: float) -> float: ...
