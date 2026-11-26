"""Preserve a task's target across visual loss using the existing recoverer."""

import numpy as np
from bucket_robot.perception.target import TargetLock, TargetTracker, visible_detection_box, iou
from bucket_robot.perception.bearing import BearingLock


def new_target_session(mode="manual", *, same_target=False, bearing=False, manager_factory=None):
    """Select the identity policy once; all applications use this factory.

    Ordinary observation may acquire a new identity. A motion mission must keep
    its previously verified target, even when another detection scores higher.
    Both policies reuse the existing optical tracking and appearance recovery.
    """
    if same_target:
        return RecoveringTarget(bearing)
    if bearing:
        return BearingLock()
    return (manager_factory or TargetLock)(mode)


class RecoveringTarget:
    def __init__(self, bearing=True):
        self.bearing = bearing
        self.manager = BearingLock() if bearing else TargetLock("manual")
        self.state = "searching"
        self.reason = "initial_acquisition"
        self.generation = 1
        self.reacquired = 0
        self.latest = None
        self.previous_image = None
        self.frozen = None
        self.recovery_stamp = -float("inf")

    def select(self, image, box, stamp):
        self.manager.select(image, box, stamp)
        self.state = "tracking"
        self.reason = ""

    def load_verified_reference(self, image, box, stamp):
        """Carry a previously verified frame across a stopped stage handoff."""
        self.latest = dict(box=list(box), timestamp=stamp)
        self.previous_image = image
        self.state = "tracking"
        self.missing("verifying_stage_handoff")

    def missing(self, reason):
        if self.state == "tracking":
            self.state = "recovering"
            self.recovery_stamp = -float("inf")
            if self.latest is not None and self.previous_image is not None:
                box = visible_detection_box(self.latest["box"], self.previous_image.shape)
                try:
                    self.frozen = TargetTracker(self.previous_image, box, self.latest["timestamp"])
                except (RuntimeError, TypeError, ValueError):
                    self.frozen = None
        self.reason = reason

    def update(self, image, metadata):
        if self.state == "tracking":
            target = self.manager.update(image, metadata)
            if target is not None and self.manager.state == "tracking":
                self.latest = target
                self.previous_image = image
                return self._decorate(target)
            self.missing(self.manager.reason or "target_unobserved")
            return None
        if self.frozen is None:
            return None
        if metadata["arrival"] <= self.recovery_stamp:
            return None
        self.recovery_stamp = metadata["arrival"]
        # Recovery belongs to the old target, never to the highest-scoring box.
        candidates = []
        old = self.latest["box"]
        old_center = (np.asarray(old[:2]) + old[2:]) / 2
        for d in metadata.get("detections", []):
            if d["score"] < 0.65:
                continue
            raw = [d[k] for k in ("left", "top", "right", "bottom")]
            box = visible_detection_box(raw, image.shape)
            if box is None:
                continue
            center = (np.asarray(box[:2]) + box[2:]) / 2
            if iou(box, old) < 0.15 or np.linalg.norm(center - old_center) > 160:
                continue
            candidates.append((d, raw, box))
        if len(candidates) != 1:
            self.frozen.recovery_count = 0
            self.frozen.recovery_candidate = None
            self.reason = "recovery_missing_or_ambiguous"
            return None
        d, raw, box = candidates[0]
        normalized = dict(d, **dict(zip(("left", "top", "right", "bottom"), box)))
        result = self.frozen.recover(image, metadata["arrival"], [normalized])
        if result is None:
            self.reason = "verifying_previous_appearance"
            return None
        if iou(result["box"], box) < 0.55:
            self.reason = "recovery_geometry_disagree"
            return None
        manager = BearingLock() if self.bearing else TargetLock("manual")
        manager.select(image, raw if self.bearing else box, metadata["arrival"])
        target = manager.update(image, metadata)
        if target is None:
            self.reason = "recovery_install_not_verified"
            return None
        self.manager = manager
        self.reacquired += 1
        self.state = "tracking"
        self.reason = ""
        self.latest = target
        self.previous_image = image
        self.frozen = None
        return self._decorate(target)

    def _decorate(self, target):
        return dict(target, task_target_id=1, recovery_count=self.reacquired)
