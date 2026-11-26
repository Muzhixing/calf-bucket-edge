"""Shared tracking decisions for stationary and chassis missions.

This component does not send commands or own threads. The application supplies
an observation and a telemetry view; motion adapters apply the returned goal.
The existing calibrated policy remains the only A/B control algorithm.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

from bucket_robot.domain.ports import TrackingPort
from bucket_robot.domain.state import pose
from bucket_robot.tracking.policy import AlignmentGate, next_ab_tracking_goal


@dataclass(frozen=True)
class TrackingDecision:
    goal: Optional[Tuple[float, float, float]] = None
    stop: bool = False
    holding: bool = False
    normalizing_b: bool = False
    reason: str = ""


class TrackingController:
    def __init__(self, policy, target_max_age=0.4):
        self.policy = policy
        self.target_max_age = target_max_age
        self.alignment = AlignmentGate()
        self.identity = None
        self.vertical_reason = None

    def reset(self):
        self.policy.reset()
        self.alignment.reset()
        self.identity = None

    def step(self, port: TrackingPort, target, metadata, now, identity=None):
        """Apply identity, freshness, hold and the shared A/B scheduler once."""
        if target is None or not 0 <= now - target["timestamp"] <= self.target_max_age:
            self.reset()
            return TrackingDecision(stop=True, reason="waiting_for_target_frame")
        if identity != self.identity:
            self.reset()
            self.identity = identity
        was_holding = self.alignment.holding
        holding = self.alignment.update(target, pose(port.state), self.policy.limits, port.target)
        if holding != was_holding:
            self.policy.reset()
        if holding:
            return TrackingDecision(
                stop=True,
                holding=True,
                reason="working_range_limit" if self.alignment.constrained else "",
            )
        result = next_ab_tracking_goal(self.policy, port, target, metadata, now)
        return TrackingDecision(
            goal=tuple(result["goal"]), stop=result["stop"], normalizing_b=result["normalizing_b"]
        )

    def horizontal_goal(self, port: TrackingPort, target, now, centering=False):
        """A-only operation of the same policy while a task owns the chassis."""
        current = pose(port.state)[0]
        goal = self.policy.next_a(
            target, self.policy.frame_angle(port, target["timestamp"]), current, now
        )
        if centering and abs(target["offset_px"][0]) <= 15:
            self.policy.reset()
            return current
        return goal

    def vertical_plan(self, now, target, b, metadata, position=None, goal=None):
        """Determine B readiness using the same alignment and vertical policy."""
        self.vertical_reason = None
        if (
            target is None
            or not 0 <= now - target.get("timestamp", -math.inf) <= self.target_max_age
        ):
            self.vertical_reason = "waiting_vertical_target"
            return dict(ready=False, goal=None)
        dy = target.get("offset_px", [0, math.nan])[1]
        if not math.isfinite(dy):
            return dict(ready=False, goal=None)
        limits = self.policy.limits
        holding = self.alignment.update(target, position, limits, goal)
        allowed = self.policy.allow_b(target, now)
        requested = self.policy.next_b(target, b, metadata) if allowed and not holding else round(b)
        requested = int(max(limits.b_min, min(limits.b_max, requested)))
        moving = goal is not None and abs(goal[1] - b) > 0.11
        # A physical B limit makes vertical centering unreachable. Like the
        # stationary tracker, accept that constraint while retaining a measured
        # horizontal bearing. B must have stopped before chassis motion resumes.
        constrained = (
            self.alignment.blocked[1]
            and target.get("horizontal_bearing_valid") is True
            and not target.get("horizontal_clipped", False)
        )
        ready = (
            (constrained or holding or not allowed)
            and not moving
            and abs(b - round(b)) <= 0.05
            and (
                constrained or not self.alignment.blocked[1] and not target.get("vertical_clipped")
            )
        )
        if not ready and abs(requested - b) <= 0.05:
            self.vertical_reason = (
                "B_range_limit" if self.alignment.blocked[1] else "vertical_alignment_pending"
            )
        else:
            self.vertical_reason = (
                "B_range_holding"
                if ready and constrained
                else "vertical_aligned"
                if ready
                else "B_adjustment"
            )
        return dict(ready=ready, goal=requested if abs(requested - b) > 0.05 else None)
