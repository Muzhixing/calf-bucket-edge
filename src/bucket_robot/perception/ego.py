"""Background motion evidence using OpenCV LK and RANSAC, never target visibility."""

import math
import cv2
import numpy as np


class BackgroundMotion:
    def __init__(self):
        self.reset()

    def reset(self):
        self.previous = None
        self.mask = None
        self.stamp = None

    @staticmethod
    def exclusion(boxes):
        mask = np.zeros((180, 320), np.uint8)
        mask[4:-4, 4:-4] = 255
        for box in boxes:
            a = np.asarray(box, dtype=float)
            if a.shape != (4,) or not np.isfinite(a).all():
                continue
            margin = np.maximum(12.0, 0.20 * (a[2:] - a[:2]))
            x1, y1 = np.floor((a[:2] - margin) / 4).astype(int)
            x2, y2 = np.ceil((a[2:] + margin) / 4).astype(int)
            if x2 <= 0 or y2 <= 0 or x1 >= 320 or y1 >= 180:
                continue
            mask[max(0, y1) : min(180, y2), max(0, x1) : min(320, x2)] = 0
        return mask

    def update(self, image, stamp, boxes):
        # No visible target means there is no compensable target observation.
        if not boxes:
            self.reset()
            return None
        if self.stamp is not None and stamp <= self.stamp:
            return None
        small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (320, 180))
        mask = self.exclusion(boxes)
        previous, oldmask, oldstamp = self.previous, self.mask, self.stamp
        self.previous, self.mask, self.stamp = small, mask, stamp
        result = {
            "valid": False,
            "timestamp": stamp,
            "previous_timestamp": oldstamp,
            "reason": "warmup",
        }
        if previous is None:
            return result
        dt = stamp - oldstamp
        if not 0.015 <= dt <= 0.20:
            result["reason"] = "frame_gap"
            return result
        points = cv2.goodFeaturesToTrack(previous, 160, 0.015, 8, mask=oldmask, blockSize=5)
        if points is None or len(points) < 20:
            result["reason"] = "few_background_features"
            return result
        # At 320x180, level 3 windows cover most of the scene and can be
        # dominated by an excluded foreground object despite masked corners.
        settings = dict(
            winSize=(19, 19),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.02),
        )
        try:
            forward, ok, _ = cv2.calcOpticalFlowPyrLK(previous, small, points, None, **settings)
            if forward is None:
                result["reason"] = "flow_failed"
                return result
            backward, back_ok, _ = cv2.calcOpticalFlowPyrLK(
                small, previous, forward, None, **settings
            )
            if backward is None:
                result["reason"] = "backward_failed"
                return result
            src = points.reshape(-1, 2)
            dst = forward.reshape(-1, 2)
            good = (
                (ok.ravel() == 1)
                & (back_ok.ravel() == 1)
                & (np.linalg.norm(backward - points, axis=2).ravel() < 0.6)
            )
            inside = (
                np.isfinite(dst).all(axis=1)
                & (dst[:, 0] >= 0)
                & (dst[:, 0] < 320)
                & (dst[:, 1] >= 0)
                & (dst[:, 1] < 180)
            )
            ix = np.clip(np.nan_to_num(dst[:, 0]), 0, 319).astype(int)
            iy = np.clip(np.nan_to_num(dst[:, 1]), 0, 179).astype(int)
            good &= inside & (mask[iy, ix] > 0)
            src, dst = src[good], dst[good]
            if len(src) < 20:
                result["reason"] = "few_consistent_features"
                return result
            matrix, inliers = cv2.estimateAffinePartial2D(
                src,
                dst,
                method=cv2.RANSAC,
                ransacReprojThreshold=0.8,
                maxIters=1000,
                confidence=0.99,
                refineIters=10,
            )
            if matrix is None or not np.isfinite(matrix).all():
                result["reason"] = "geometry"
                return result
            keep = inliers.ravel().astype(bool)
            support = src[keep]
            cells = set((min(3, int(x / 80)), min(2, int(y / 60))) for x, y in support)
            fraction = float(keep.mean())
            if (
                len(support) < 20
                or fraction < 0.70
                or len(cells) < 6
                or np.ptp(support[:, 0]) < 160
                or np.ptp(support[:, 1]) < 72
            ):
                result["reason"] = "background_not_distributed"
                return result
            residual = np.linalg.norm(
                (matrix[:, :2] @ support.T).T + matrix[:, 2] - dst[keep], axis=1
            )
            error = float(np.median(residual))
            scale = math.hypot(matrix[0, 0], matrix[1, 0])
            turn = abs(math.atan2(matrix[1, 0], matrix[0, 0]))
            center = np.array([160.0, 90.0])
            shift = matrix[:, :2] @ center + matrix[:, 2] - center
            if (
                error > 0.5
                or not 0.96 <= scale <= 1.04
                or turn > 0.05
                or np.linalg.norm(shift) * 4 > 1200 * dt + 2
            ):
                result["reason"] = "nonrigid_or_large_motion"
                return result
            matrix[:, 2] *= 4
            result.update(
                valid=True,
                reason="",
                matrix=matrix.tolist(),
                confidence=min(1.0, fraction * (1 - error)),
                inliers=int(keep.sum()),
                cells=len(cells),
                residual_px=error * 4,
            )
            return result
        except cv2.error:
            result["reason"] = "opencv_rejected"
            return result
