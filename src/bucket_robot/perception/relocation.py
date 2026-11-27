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
            for d in detections:
                if self.cancel.is_set():
                    return
                raw = [d[k] for k in ("left", "top", "right", "bottom")]
                box = visible_detection_box(raw, image.shape)
                if d["score"] < 0.8 or box is None:
                    continue
                for reference, old in references:
                    if self.cancel.is_set():
                        return
                    evidence = known_feature_relocation(reference, old, image, box)
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
