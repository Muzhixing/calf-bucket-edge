"""Preserve a task's target across visual loss using the existing recoverer."""

import numpy as np
from bucket_robot.perception.target import (
    TargetLock,
    TargetTracker,
    EdgeTracker,
    visible_detection_box,
    iou,
    weighted_ncc,
)
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
        self.relocation_votes = []
        self.selection_image = None
        self.selection_box = None
        self.relocation_probe = None
        self.known_recovery = None

    def select(self, image, box, stamp):
        self.manager.select(image, box, stamp)
        self.selection_image, self.selection_box = image, list(box)
        self.state = "tracking"
        self.reason = ""
        # Selection has already been verified by acquisition/seed validation.
        # Keep it as recovery evidence even if the first update is missing.
        # It is not exposed as a fresh motion observation.
        self.latest = dict(box=list(box), timestamp=stamp)
        self.previous_image = image
        self.frozen = None
        self.known_recovery = None

    def load_verified_reference(self, image, box, stamp):
        """Carry a previously verified frame across a stopped stage handoff."""
        self.latest = dict(box=list(box), timestamp=stamp)
        self.previous_image = image
        self.selection_image, self.selection_box = image, list(box)
        self.state = "tracking"
        self.missing("verifying_stage_handoff")

    def missing(self, reason):
        if self.state == "tracking":
            self.state = "recovering"
            self.recovery_stamp = -float("inf")
            self.relocation_votes = []
            self.relocation_probe = None
            self.known_recovery = TargetLock("manual")
            self.known_recovery.loss_since = self.latest["timestamp"] if self.latest else None
            if self.latest is not None and self.previous_image is not None:
                box = visible_detection_box(self.latest["box"], self.previous_image.shape)
                try:
                    trusted = getattr(self.manager, "tracker", None)
                    self.frozen = (
                        trusted
                        if isinstance(trusted, TargetTracker)
                        and not isinstance(trusted, EdgeTracker)
                        and trusted.stamp == self.latest["timestamp"]
                        else TargetTracker(self.previous_image, box, self.latest["timestamp"])
                    )
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
        # Reuse the mature known-appearance + fresh-flow verifier. A camera
        # turn may move this exact bucket outside the old position gate.
        # require_known excludes its automatic new-identity fallback entirely.
        if self.known_recovery is not None:
            known = self.known_recovery.verified_reappearance(
                image,
                metadata["arrival"],
                metadata.get("detections", []),
                reference=self.frozen,
                require_known=True,
            )
            if known is not None:
                evidence = {
                    k: known[k]
                    for k in (
                        "known_appearance_verified",
                        "candidate_verify_s",
                        "candidate_seed_score",
                        "candidate_detection_votes",
                    )
                }
                matches = [
                    d
                    for d in metadata.get("detections", [])
                    if d["score"] >= 0.536
                    and iou(known["box"], [d[k] for k in ("left", "top", "right", "bottom")])
                    >= 0.55
                ]
                if len(matches) == 1:
                    raw = [matches[0][k] for k in ("left", "top", "right", "bottom")]
                    return self._install_recovered(
                        image,
                        metadata,
                        raw if self.bearing else known["box"],
                        appearance_recovery_evidence=evidence,
                    )
            if self.known_recovery.provisional is not None:
                # Keep this fresh-frame path responsive. Running slow global
                # feature relocation here can itself exceed the verifier's
                # frame-gap bound and restart confirmation indefinitely.
                self.reason = "verifying_known_appearance"
                return None
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
            nearby = iou(box, old) >= 0.15 and np.linalg.norm(center - old_center) <= 160
            evidence = None
            if d["score"] >= 0.8 and self.previous_image is not None:
                from bucket_robot.perception.relocation import known_feature_relocation

                probe = self.relocation_probe
                if (
                    probe is not None
                    and 0 < metadata["arrival"] - probe[2] <= 0.5
                    and iou(box, probe[1]) >= 0.55
                ):
                    view = TargetTracker.measured_view(image, box)
                    prior = TargetTracker.measured_view(probe[0], probe[1])
                    if weighted_ncc(view, prior, np.ones(view.shape, np.float32)) >= 0.9:
                        evidence = probe[3]
                if evidence is None:
                    evidence = known_feature_relocation(self.previous_image, old, image, box)
                    if evidence is None and self.selection_image is not None:
                        evidence = known_feature_relocation(
                            self.selection_image, self.selection_box, image, box
                        )
                    if evidence is not None:
                        self.relocation_probe = (image, box, metadata["arrival"], evidence)
            if nearby or evidence is not None:
                candidates.append((d, raw, box, evidence))
        if len(candidates) != 1:
            self.frozen.recovery_count = 0
            self.frozen.recovery_candidate = None
            self.relocation_votes = []
            self.reason = "recovery_missing_or_ambiguous"
            return None
        d, raw, box, evidence = candidates[0]
        normalized = dict(d, **dict(zip(("left", "top", "right", "bottom"), box)))
        result = (
            None
            if evidence is not None
            else self.frozen.recover(image, metadata["arrival"], [normalized])
        )
        if result is None and evidence is not None:
            stamp = metadata["arrival"]
            if self.relocation_votes and iou(self.relocation_votes[-1][1], box) < 0.55:
                self.relocation_votes = []
            self.relocation_votes = [(t, b) for t, b in self.relocation_votes if stamp - t <= 0.5]
            self.relocation_votes.append((stamp, box))
            if len(self.relocation_votes) >= 3 and stamp - self.relocation_votes[0][0] >= 0.18:
                result = dict(box=box, relocation_evidence=evidence)
        elif evidence is None:
            self.relocation_votes = []
        if result is None:
            self.reason = "verifying_previous_appearance"
            return None
        if iou(result["box"], box) < 0.55:
            self.reason = "recovery_geometry_disagree"
            return None
        return self._install_recovered(
            image,
            metadata,
            raw if self.bearing else box,
            **(
                {"relocation_evidence": result["relocation_evidence"]}
                if result.get("relocation_evidence")
                else {}
            ),
        )

    def _install_recovered(self, image, metadata, box, **evidence):
        manager = BearingLock() if self.bearing else TargetLock("manual")
        manager.select(image, box, metadata["arrival"])
        target = manager.update(image, metadata)
        if target is None:
            self.reason = "recovery_install_not_verified"
            return None
        target.update(evidence)
        self.manager = manager
        self.reacquired += 1
        self.state = "tracking"
        self.reason = ""
        self.latest = target
        self.previous_image = image
        self.frozen = None
        self.known_recovery = None
        return self._decorate(target)

    def _decorate(self, target):
        return dict(target, task_target_id=1, recovery_count=self.reacquired)
