"""Preserve a task's target across visual loss using the existing recoverer."""

from bucket_robot.perception.target import (
    TargetLock,
    TargetTracker,
    EdgeTracker,
    visible_detection_box,
    valid_box,
    iou,
)
from bucket_robot.perception.bearing import BearingLock
from bucket_robot.perception.relocation import FeatureRelocationJob


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
        self.selection_image = None
        self.selection_box = None
        self.known_recovery = None
        self.feature_job = None
        self.feature_reference = None
        self.relocation_epoch = 0
        self.feature_attempt_at = -float("inf")

    def select(self, image, box, stamp):
        self.relocation_epoch += 1
        self.feature_reference = None
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
            self.relocation_epoch += 1
            self.feature_reference = None
            self.feature_attempt_at = -float("inf")
            self.recovery_stamp = -float("inf")
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
                        else TargetTracker(
                            self.previous_image,
                            box,
                            self.latest["timestamp"],
                            segment_foreground=not isinstance(trusted, EdgeTracker),
                        )
                    )
                except (RuntimeError, TypeError, ValueError):
                    self.frozen = None
        self.reason = reason

    def update(self, image, metadata):
        if self.feature_job is not None and self.feature_job.done.is_set():
            if self.feature_job.epoch == self.relocation_epoch:
                self.feature_reference = self.feature_job.result
            self.feature_job = None
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
            # The whole-object verifier rejects raw edge boxes. Match and verify
            # their measured visible pixels, then install the original raw box
            # below so BearingLock retains the clipping/bearing restrictions.
            visible_detections = []
            for detection in metadata.get("detections", []):
                raw = [detection[k] for k in ("left", "top", "right", "bottom")]
                visible = visible_detection_box(raw, image.shape) if self.bearing else raw
                if visible is not None:
                    visible_detections.append(
                        dict(
                            detection,
                            **dict(zip(("left", "top", "right", "bottom"), visible)),
                            visible_only=not valid_box(raw, image.shape),
                        )
                    )
            known = self.known_recovery.verified_reappearance(
                image,
                metadata["arrival"],
                visible_detections,
                reference=self.feature_reference[0] if self.feature_reference else self.frozen,
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
                if self.feature_reference:
                    evidence["relocation_evidence"] = self.feature_reference[1]
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
        # A slow proof may take longer than a frame-continuity deadline. Run
        # at most one job; its verified template still needs the current-frame
        # verifier above. Old coordinates never become a recovered observation.
        if self.feature_job is None and metadata["arrival"] - self.feature_attempt_at >= 0.3:
            self.feature_attempt_at = metadata["arrival"]
            self.feature_job = FeatureRelocationJob(
                self.relocation_epoch,
                [
                    (self.previous_image, self.latest["box"]),
                    (self.selection_image, self.selection_box),
                ],
                image,
                metadata,
            )
        self.reason = "waiting_known_feature_proof"
        return None

    def _install_recovered(self, image, metadata, box, **evidence):
        manager = BearingLock() if self.bearing else TargetLock("manual")
        manager.select(image, box, metadata["arrival"])
        target = manager.update(image, metadata)
        if target is None:
            self.reason = "recovery_install_not_verified"
            return None
        target.update(evidence)
        self.manager = manager
        self.relocation_epoch += 1
        self.feature_reference = None
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

    def close(self):
        if self.feature_job is not None:
            self.feature_job.close()
