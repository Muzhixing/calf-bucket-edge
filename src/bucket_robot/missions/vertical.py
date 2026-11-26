"""Opt-in B tracking with an acknowledged chassis stop before every B move."""

from bucket_robot.motion.combined import ABCombinedPort as ABCombinedPort
from bucket_robot.tracking.controller import TrackingController


def align_parked(observer, port, controller, folder, record, *, visibility_only=False, timeout=60):
    """Use the shared A/B policy with A held and the chassis acknowledged off."""
    import json
    import time
    from bucket_robot.motion.arm import pose

    if port.phase != "idle" or port.state["enabled"] or pose(port.state)[2] != 0:
        raise RuntimeError("parked_alignment_requires_disabled_chassis")
    limits = controller.policy.limits
    bounds = (limits.a_min, limits.a_max)
    fixed_a = pose(port.state)[0]
    initial_pose = pose(port.state)
    limits.a_min = limits.a_max = fixed_a
    port.velocity = 0
    controller.reset()
    began = time.monotonic()
    phase = "restoring_upper_view" if visibility_only else "parked_upper_alignment"
    last_status = 0
    normalizing = None
    try:
        while time.monotonic() - began < timeout:
            if (folder / "STOP").exists() or (folder.parent / "STOP").exists():
                raise KeyboardInterrupt("operator_stop")
            port.tick()
            value = observer.get()
            obs = value[0] if isinstance(value, tuple) else value
            now = time.monotonic()
            target = controller.target(obs.get("target")) if obs else None
            fresh = (
                target is not None
                and obs.get("identity") == 1
                and 0 <= now - target["timestamp"] <= 0.35
            )
            if normalizing is not None:
                if abs(pose(port.state)[1] - normalizing) <= 0.05 and not port.pending:
                    if port.state["enabled"] and port.phase != "stopping":
                        port.stop()
                    elif port.phase == "idle":
                        normalizing = None
                time.sleep(0.005)
                continue
            if obs and obs.get("identity") not in (None, 1):
                raise RuntimeError("parked_alignment_identity_changed")
            if now - last_status >= 0.25:
                (folder / "task_status.json").write_text(
                    json.dumps(
                        dict(phase=phase, reason="B_adjustment_with_chassis_stopped", velocity=0)
                    )
                )
                last_status = now
            visible = fresh and target.get("aim_region_visible", False)
            region = target.get("aim_region_px") if target else None
            view_ready = visible and region is not None and region[1] >= 64 and region[3] <= 560
            decision = controller.step(port, target, (obs or {}).get("metadata", {}), now, 1)
            ready = fresh and (
                view_ready or controller.alignment.constrained and visible
                if visibility_only
                else decision.holding
            )
            if not fresh or ready or decision.stop:
                if port.phase != "stopping" and port.state["enabled"]:
                    port.stop()
                if ready and port.phase == "idle" and not port.state["enabled"]:
                    a, b, c = pose(port.state)
                    if abs(b - round(b)) > 0.05:
                        if port.can_send():
                            port.follow([a, round(b), c])
                            normalizing = round(b)
                    else:
                        result = dict(
                            final_state=dict(port.state),
                            settled_at=now,
                            elapsed_s=now - began,
                            constrained=controller.alignment.constrained,
                            moved=pose(port.state) != initial_pose,
                        )
                        record("parked_aim_completed", result)
                        return result
            elif port.can_send():
                goal = list(decision.goal)
                goal[0] = fixed_a
                port.follow(goal)
            time.sleep(0.005)
        raise RuntimeError("parked_upper_alignment_timeout")
    finally:
        limits.a_min, limits.a_max = bounds
        controller.reset()


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
