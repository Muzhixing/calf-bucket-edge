"""Geometric identity evidence for a measured bucket relocated by camera motion."""

import cv2
import numpy as np
import threading
from bucket_robot.domain.lifecycle import CleanupResult
from bucket_robot.perception.target import (
    TargetTracker,
    iou,
    valid_box,
    visible_detection_box,
    weighted_ncc,
)


class FeatureRelocationJob:
    """Compute identity evidence off the fresh-frame observation path.

    The result is an appearance reference, never a motion observation. The
    caller must verify new current frames before exposing any recovered target.
    """

    def __init__(self, epoch, references, image, metadata):
        self.epoch = epoch
        self.done = threading.Event()
        self.cancel = threading.Event()
        self.result = None
        self.error = None
        references = [(im.copy(), list(box)) for im, box in references if im is not None]
        image = image.copy()
        detections = [dict(d) for d in metadata.get("detections", [])]
        self.thread = threading.Thread(
            target=self.run,
            args=(references, image, detections, metadata["arrival"]),
            daemon=False,
            name="bucket-identity-proof",
        )
        self.thread.start()

    def run(self, references, image, detections, stamp):
        try:
            matches = []
            scene_proofs = {}
            for d in detections:
                if self.cancel.is_set():
                    return
                raw = [d[k] for k in ("left", "top", "right", "bottom")]
                box = visible_detection_box(raw, image.shape)
                if d["score"] < 0.8 or box is None:
                    continue
                for index, (reference, old) in enumerate(references):
                    if self.cancel.is_set():
                        return
                    evidence = known_feature_relocation(reference, old, image, box)
                    if evidence is None:
                        if index not in scene_proofs:
                            scene_proofs[index] = scene_projection(reference, image)
                        if scene_proofs[index] is not None:
                            evidence = known_scene_relocation(
                                reference, old, image, box, scene_proofs[index]
                            )
                    if evidence is not None:
                        matches.append((raw, box, evidence))
                        break
            if len(matches) == 1 and not self.cancel.is_set():
                raw, box, evidence = matches[0]
                tracker = TargetTracker(
                    image, box, stamp, segment_foreground=valid_box(raw, image.shape)
                )
                self.result = (tracker, evidence)
        except Exception as exc:
            self.error = repr(exc)
        finally:
            self.done.set()

    def request_stop(self):
        self.cancel.set()

    def await_stopped(self, timeout=2):
        self.thread.join(timeout=timeout)
        alive = self.thread.is_alive()
        return CleanupResult(not alive, (self.thread.name,) if alive else ())

    def close(self):
        self.request_stop()
        return self.await_stopped()


def scene_projection(reference, current):
    """Geometry evidence only; a scene match cannot establish target visibility."""
    if reference.shape != current.shape:
        return None
    orb = cv2.ORB_create(nfeatures=4000, edgeThreshold=8, fastThreshold=8)
    a, da = orb.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), None)
    b, db = orb.detectAndCompute(cv2.cvtColor(current, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None or min(len(a), len(b)) < 16:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    forward = [
        m
        for pair in matcher.knnMatch(da, db, k=2)
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < 0.75 * n.distance
    ]
    reverse = {
        m.queryIdx: m.trainIdx
        for pair in matcher.knnMatch(db, da, k=2)
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < 0.75 * n.distance
    }
    matches = [m for m in forward if reverse.get(m.trainIdx) == m.queryIdx]
    if len(matches) < 16:
        return None
    src = np.float32([a[m.queryIdx].pt for m in matches])
    dst = np.float32([b[m.trainIdx].pt for m in matches])
    # Locally optimized hypotheses handle blurred, non-uniform scene support;
    # the independent distribution, residual and foreground gates remain below.
    matrix, mask = cv2.findHomography(
        src, dst, cv2.USAC_ACCURATE, 3, maxIters=10000, confidence=0.999
    )
    if matrix is None or mask is None or not np.isfinite(matrix).all():
        return None
    keep = mask.ravel().astype(bool)
    if keep.sum() < 16 or keep.mean() < 0.45:
        return None
    h, w = reference.shape[:2]
    for points in (src[keep], dst[keep]):
        span = np.ptp(points, axis=0) / [w, h]
        cells = set((min(3, int(x * 4 / w)), min(2, int(y * 3 / h))) for x, y in points)
        if np.any(span < 0.3) or len(cells) < 3:
            return None
    predicted = cv2.perspectiveTransform(src[keep][None], matrix)[0]
    residual = float(np.median(np.linalg.norm(predicted - dst[keep], axis=1)))
    if not np.isfinite(residual) or residual > 2:
        return None
    return matrix, dict(
        scene_inliers=int(keep.sum()),
        scene_mutual_matches=len(matches),
        scene_inlier_fraction=float(keep.mean()),
        scene_residual_px=residual,
    )


def known_scene_relocation(reference, old_box, current, box, projection):
    """Verify actual common target pixels after an independently measured warp.

    The projection is never exposed as a target coordinate. Current detection,
    unique-candidate selection and the original fresh-frame verifier still apply.
    """
    matrix, evidence = projection
    x, y, r, bottom = map(int, old_box)
    corners = np.float32([[x, y], [r, y], [r, bottom], [x, bottom]])
    homogeneous = np.c_[corners, np.ones(4)] @ matrix.T
    if np.any(homogeneous[:, 2] <= 0):
        return None
    projected = cv2.perspectiveTransform(corners[None], matrix)[0]
    if not np.isfinite(projected).all() or not cv2.isContourConvex(projected):
        return None
    if cv2.contourArea(projected, oriented=True) <= 0:
        return None
    projected_box = np.r_[projected.min(axis=0), projected.max(axis=0)]
    ratio = (projected_box[2:] - projected_box[:2]) / [r - x, bottom - y]
    overlap = iou(projected_box, box)
    if overlap < 0.65 or np.any((ratio < 0.5) | (ratio > 1.8)):
        return None
    h, w = reference.shape[:2]
    gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    warped = cv2.warpPerspective(
        gray, matrix, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    )
    mask = np.zeros(gray.shape, np.uint8)
    l, t, right, bot = map(int, box)
    mask[t + 5 : bot - 5, l + 5 : right - 5] = 1
    common = cv2.warpPerspective(
        mask, matrix, (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
    )
    region = (slice(y + 5, bottom - 5), slice(x + 5, r - 5))
    weights = common[region].astype(np.float32)
    if weights.size == 0 or weights.mean() < 0.75:
        return None
    original = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)[region]
    score = weighted_ncc(original, warped[region], weights)
    if not np.isfinite(score) or score < 0.85:
        return None
    return dict(
        evidence,
        source="scene_projection_and_common_target_pixels",
        common_pixel_ncc=float(score),
        common_fraction=float(weights.mean()),
        projected_iou=float(overlap),
    )


def known_feature_relocation(reference, old_box, current, box):
    orb = cv2.ORB_create(nfeatures=1600, edgeThreshold=8, patchSize=15, fastThreshold=8)
    features = []
    for image, region in ((reference, old_box), (current, box)):
        mask = np.zeros(image.shape[:2], np.uint8)
        x1, y1, x2, y2 = map(int, region)
        mask[y1 + 5 : y2 - 5, x1 + 5 : x2 - 5] = 255
        features.append(orb.detectAndCompute(image, mask))
    (old, a), (new, b) = features
    if a is None or b is None or min(len(a), len(b)) < 12:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    forward = [
        m
        for pair in matcher.knnMatch(a, b, k=2)
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < 0.75 * n.distance
    ]
    reverse = {
        m.queryIdx: m.trainIdx
        for pair in matcher.knnMatch(b, a, k=2)
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < 0.75 * n.distance
    }
    matches = [m for m in forward if reverse.get(m.trainIdx) == m.queryIdx]
    if len(matches) < 12:
        return None
    src = np.float32([old[m.queryIdx].pt for m in matches])
    dst = np.float32([new[m.trainIdx].pt for m in matches])
    matrix, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3)
    if matrix is None or mask is None:
        return None
    inliers = mask.ravel().astype(bool)
    scale = float(np.linalg.norm(matrix[:, 0]))
    if inliers.sum() < 8 or inliers.mean() < 0.35 or not 0.5 <= scale <= 1.8:
        return None
    size = np.array(old_box[2:]) - old_box[:2]
    span = np.ptp(src[inliers], axis=0) / size
    if np.any(span < 0.20):
        return None
    x, y, r, bottom = old_box
    corners = np.float32([[x, y], [r, y], [r, bottom], [x, bottom]])
    projected = cv2.transform(corners[None], matrix)[0]
    projected_box = np.r_[projected.min(axis=0), projected.max(axis=0)]
    overlap = iou(projected_box, box)
    if overlap < 0.5:
        return None
    # Sparse matching alone is insufficient on a plain bucket. Verify the
    # measured common physical pixels independently after geometric alignment.
    gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    warped = cv2.warpAffine(
        gray, matrix, (1280, 720), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    )
    mask = np.zeros(gray.shape, np.uint8)
    l, t, r, bot = map(int, box)
    mask[t + 5 : bot - 5, l + 5 : r - 5] = 1
    common = cv2.warpAffine(
        mask, matrix, (1280, 720), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
    )
    l, t, r, bot = map(int, old_box)
    region = (slice(t + 5, bot - 5), slice(l + 5, r - 5))
    weights = common[region].astype(np.float32)
    if weights.mean() < 0.65:
        return None
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)[region]
    score = weighted_ncc(reference_gray, warped[region], weights)
    if score < 0.82:
        return None
    return dict(
        common_pixel_ncc=float(score),
        inliers=int(inliers.sum()),
        mutual_matches=len(matches),
        inlier_fraction=float(inliers.mean()),
        reference_span=span.tolist(),
        projected_iou=float(overlap),
        scale=scale,
    )
