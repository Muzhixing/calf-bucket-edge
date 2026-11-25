"""Measured visible-bucket bearing for chassis alignment, never a hidden center.

Uses the existing whole-object visual tracker and detector-backed edge tracker.
Crossing the edge requires adjacent matching visible evidence and preserves ID.
Clipped views require a current unique detection; there is no blind reacquisition.
"""

import cv2
import numpy as np
from bucket_robot.perception.target import (
    EdgeTracker,
    TargetTracker,
    valid_box,
    visible_detection_box,
    iou,
    weighted_ncc,
)


class VisibleEdgeTracker(EdgeTracker):
    """Detector crop jitter must not resize different physical pixels together.

    The ordinary edge check remains first. An alternate acceptance requires current
    forward/backward optical flow, RANSAC geometry and a strong match of the same
    visible pixels, masked to the current detection. It cannot bridge missing boxes.
    """

    def __init__(self, image, box, stamp):
        super().__init__(image, box, stamp)
        self.last_strong_detection = stamp

    def update(self, image, stamp, detections=(), flow_only=False):
        weak = len(detections) == 1 and 0.536 <= detections[0]["score"] < 0.65
        try:
            result = super().update(image, stamp, detections, flow_only)
        except RuntimeError as exc:
            if str(exc) != "edge_appearance_lost" and not (
                str(exc) == "edge_requires_unique_detection"
                and weak
                and 0 < stamp - self.last_strong_detection <= 0.15
            ):
                raise
            d = detections[0]
            raw = [d[k] for k in ("left", "top", "right", "bottom")]
            b = np.asarray(visible_detection_box(raw, image.shape), float)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.flow_dt = stamp - self.stamp
            matrix = self.flow_transform(gray)[:2]
            x1, y1, x2, y2 = np.rint(self.box).astype(int)
            corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            projected = cv2.transform(corners[None], matrix)[0]
            projected_box = np.r_[projected.min(axis=0), projected.max(axis=0)]
            if iou(projected_box, b) < 0.55:
                raise RuntimeError("edge_flow_detection_disagree")
            mask = np.zeros(gray.shape, np.uint8)
            l, t, r, bot = np.rint(b).astype(int)
            mask[t + 4 : bot - 4, l + 4 : r - 4] = 1
            aligned = cv2.warpAffine(
                gray, matrix, (1280, 720), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
            )
            common = cv2.warpAffine(
                mask, matrix, (1280, 720), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
            )[y1:y2, x1:x2]
            if float(common.mean()) < 0.65:
                raise RuntimeError("edge_common_region_too_small")
            weights = cv2.resize(self.weights, (x2 - x1, y2 - y1)) * common
            score = weighted_ncc(self.gray[y1:y2, x1:x2], aligned[y1:y2, x1:x2], weights)
            if score < (0.90 if weak else 0.85):
                raise RuntimeError("edge_common_appearance_lost")
            if valid_box(raw, image.shape) and d["score"] >= 0.8:
                self.full_votes = [t for t in self.full_votes if stamp - t <= 0.3] + [stamp]
            else:
                self.full_votes = []
            self.ready_for_full = len(self.full_votes) >= 3 and stamp - self.full_votes[0] >= 0.12
            self.gray = gray
            self.box = b
            self.stamp = stamp
            self.quality = min(score, d["score"])
            self.method = "edge_detector_weak_common_flow" if weak else "edge_detector_common_flow"
            self.detector_view = (self.measured_view(image, b), b.tolist(), stamp)
            self.detector_support = [t for t in self.detector_support if stamp - t <= 0.3] + [stamp]
            result = self.result()
            result["common_region_ncc"] = score
        # EdgeTracker only needed resized views. The physical-pixel alternative
        # also needs the actual immediately preceding image/features every time.
        self.gray_small = cv2.resize(self.gray, (640, 360), interpolation=cv2.INTER_AREA)
        self.points = self.features(self.gray, self.box)
        self.transform = np.eye(3)
        if not weak:
            self.last_strong_detection = stamp
        return result


class BearingLock:
    def __init__(self):
        self.tracker = None
        self.state = "searching"
        self.generation = 0
        self.reacquired = 0
        self.reason = ""
        self.raw = None
        self.previous_image = None

    def select(self, image, box, stamp):
        visible = visible_detection_box(box, image.shape)
        if visible is None:
            raise RuntimeError("invalid_visible_bucket")
        self.tracker = (TargetTracker if valid_box(box, image.shape) else VisibleEdgeTracker)(
            image, visible, stamp
        )
        self.tracker.detector_view = (self.tracker.measured_view(image, visible), visible, stamp)
        self.raw = list(box)
        self.previous_image = image
        self.state = "tracking"
        self.generation += 1

    def update(self, image, metadata):
        if self.state != "tracking":
            return None
        stamp = metadata["arrival"]
        old = self.tracker.box
        matches = []
        for d in metadata.get("detections", []):
            if d["score"] < 0.536:
                continue
            raw = [d[k] for k in ("left", "top", "right", "bottom")]
            visible = visible_detection_box(raw, image.shape)
            if visible is None:
                continue
            b = np.asarray(visible, float)
            ratio = (b[2:] - b[:2]) / (old[2:] - old[:2])
            if (
                iou(b, old) >= 0.4
                and np.all((ratio >= 0.65) & (ratio <= 1.5))
                and np.linalg.norm((b[:2] + b[2:] - old[:2] - old[2:]) / 2) <= 120
            ):
                matches.append(d)
        try:
            if len(matches) > 1:
                raise RuntimeError("bearing_detection_ambiguous")
            d = matches[0] if matches else None
            raw = [d[k] for k in ("left", "top", "right", "bottom")] if d else None
            if isinstance(self.tracker, EdgeTracker):
                if d is None:
                    raise RuntimeError("bearing_detection_missing")
                if stamp == self.tracker.stamp:
                    result = self.tracker.result()
                else:
                    result = self.tracker.update(image, stamp, matches)
                self.raw = raw
                if self.tracker.ready_for_full:
                    previous = self.tracker
                    self.tracker = TargetTracker(image, raw, stamp)
                    self.tracker.detector_support = previous.detector_support.copy()
                    self.tracker.detector_view = previous.detector_view
                    self.tracker.measurement_warmup_until = stamp + 0.35
                    result = self.tracker.result()
                    result["method"] = "edge_full_continuity"
            elif raw is not None and not valid_box(raw, image.shape):
                # Verify the same adjacent visible view before changing tracker
                # representation. This is neither reacquisition nor a new ID.
                if d["score"] < 0.65:
                    raise RuntimeError("edge_requires_strong_detection")
                previous = self.tracker
                edge = VisibleEdgeTracker(
                    self.previous_image, previous.box.astype(int).tolist(), previous.stamp
                )
                edge.detector_view = (
                    edge.measured_view(self.previous_image, previous.box),
                    previous.box.tolist(),
                    previous.stamp,
                )
                result = edge.update(image, stamp, matches)
                self.tracker = edge
                self.raw = raw
            else:
                if stamp == self.tracker.stamp:
                    result = self.tracker.result()
                else:
                    result = self.tracker.update(image, stamp, metadata.get("detections", []))
                self.raw = result["box"]
            x1, y1, x2, y2 = self.raw
            h, w = image.shape[:2]
            horizontal = 8 <= x1 < x2 <= w - 8
            vertical = 8 <= y1 < y2 <= h - 8
            result.update(
                raw_box=self.raw,
                partial_visible=not (horizontal and vertical),
                horizontal_bearing_valid=horizontal,
                bearing_evidence="current_detection_and_visible_appearance"
                if isinstance(self.tracker, EdgeTracker)
                else "whole_target_visual_tracking",
                horizontal_clipped=not horizontal,
                vertical_clipped=not vertical,
            )
            # For a whole horizontal extent, use the measured left/right edges.
            # A clipped horizontal extent is only an inward centering cue.
            result["offset_px"][0] = (x1 + x2) / 2 - w / 2
            self.previous_image = image
            return result
        except RuntimeError as exc:
            self.state = "lost"
            self.reason = str(exc)
            return None


def measured_horizontal(target):
    return (
        target.get("horizontal_bearing_valid") is True
        and target.get("bearing_evidence") == "current_detection_and_visible_appearance"
    )
