"""Opt-in B tracking with an acknowledged chassis stop before every B move."""

from bucket_robot.motion.combined import ABCombinedPort as ABCombinedPort
from bucket_robot.tracking.controller import TrackingController


class VerticalSupervisor:
    def __init__(self, policy):
        self.policy = policy
        self.goal = None
        self.stage = "idle"
        self.reason = None
        self.controller = TrackingController(policy, target_max_age=0.35)
        self.alignment = self.controller.alignment

    def plan(self, now, target, b, metadata, position=None, goal=None):
        result = self.controller.vertical_plan(now, target, b, metadata, position, goal)
        self.reason = self.controller.vertical_reason
        return result

    def advance(self, p, plan, velocity, target, metadata, now):
        """True consumes this control iteration; never move B with chassis on."""
        if self.stage == "moving":
            if self.reason == "waiting_vertical_target":
                p.stop()
                self.stage = "stopping"
                return True
            if self.alignment.holding or plan["ready"]:
                p.stop()
                self.stage = "stopping"
                return True
            decision = self.controller.step(p, target, metadata, now)
            if decision.stop:
                p.stop()
                self.stage = "stopping"
                self.policy.reset()
            elif p.can_send():
                p.follow(list(decision.goal))
                self.goal = list(decision.goal)[1]
            return True
        if self.stage == "stopping":
            if p.phase == "idle" and not p.state["enabled"]:
                self.stage = "idle"
                self.goal = None
                self.policy.reset()
            return True
        if plan["ready"]:
            return False
        if velocity > 0:
            return False  # Existing recovery ramp first reduces vx.
        if p.phase == "stopping":
            return True
        if plan["goal"] is None:
            return False  # A may still correct horizontal error at vx=0.
        if p.state["enabled"]:
            p.stop()
            return True
        if p.can_send():
            p.velocity = 0
            self.stage = "moving"
            decision = self.controller.step(p, target, metadata, now)
            if not decision.stop:
                p.follow(list(decision.goal))
                self.goal = list(decision.goal)[1]
        return True

    def status(self):
        return dict(stage=self.stage, reason=self.reason, goal_mm=self.goal)
