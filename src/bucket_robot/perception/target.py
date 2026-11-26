"""Detection acquisition and single-object tracking on unannotated C++ frames."""

import math

import cv2
import numpy as np
from bucket_robot.perception.filtering import MotionFilter
from bucket_robot.perception.capture import NativeCapture as NativeCapture

cv2.setNumThreads(1)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def valid_box(box, shape=(720, 1280)):
    if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        return False
    x1, y1, x2, y2 = box
    h, w = shape[:2]
    return (
        8 <= x1 < x2 <= w - 8
        and 8 <= y1 < y2 <= h - 8
        and x2 - x1 >= 24
        and y2 - y1 >= 24
        and (x2 - x1) * (y2 - y1) <= 0.7 * w * h
    )


def visible_detection_box(box, shape=(720, 1280)):
    """An observed clipped ROI, never an estimate of the invisible full object.

    Keep the ordinary flow/search boundary unchanged. Only the detector-backed
    edge path can use these inset pixels; negative/out-of-frame boxes and tiny
    slivers are not evidence of a usable target.
    """
    if valid_box(box, shape):
        return list(box)
    if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        return None
    x1, y1, x2, y2 = box
    h, w = shape[:2]
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        return None
    if min(x2 - x1, y2 - y1) < 64 or (x2 - x1) * (y2 - y1) > 0.7 * w * h:
        return None
    inset = [max(8, x1), max(8, y1), min(w - 8, x2), min(h - 8, y2)]
    return inset if valid_box(inset, shape) else None


class Acquirer:
    """Accumulate associated detections in time; a missed frame is not a reset."""

    def __init__(self):
        self.candidate = None
        self.hits = []
        self.last = -1.0
        self.frame_stamp = -1.0

    def update(self, detections, stamp):
        if stamp <= self.frame_stamp:
            return None
        self.frame_stamp = stamp
        self.hits = [h for h in self.hits if stamp - h[0] <= 1.2]
        boxes = [
            (tuple(d[k] for k in ("left", "top", "right", "bottom")), d["score"])
            for d in detections
            if d["score"] >= 0.536 and valid_box([d[k] for k in ("left", "top", "right", "bottom")])
        ]
        if stamp - self.last > 0.6:
            self.candidate = None
            self.hits = []
        if not boxes:
            return None
        related = [
            (b, s) for b, s in boxes if self.candidate is not None and iou(b, self.candidate) >= 0.4
        ]
        match = max(related, key=lambda item: iou(item[0], self.candidate)) if related else None
        if match is None:
            match = max(boxes, key=lambda x: x[1])
            self.hits = []
        self.candidate, score = match
        self.last = stamp
        self.hits.append((stamp, score))
        strong = any(s >= 0.65 for _, s in self.hits)
        required = 3 if strong else 5
        # Three recent high-confidence observations can confirm a moving bucket;
        # neither its center nor its velocity needs to settle first.
        recent = [h for h in self.hits if stamp - h[0] <= 0.3]
        fast = (
            len(recent) >= 3 and all(s >= 0.8 for _, s in recent) and stamp - recent[0][0] >= 0.12
        )
        return (
            self.candidate
            if fast or len(self.hits) >= required and stamp - self.hits[0][0] >= 0.3
            else None
        )


def weighted_ncc(a, b, weights):
    # Identity scoring need not revisit every pixel of a large bucket seven
    # times per frame. Keep the same aligned sampling grid for all references.
    stride = max(1, math.ceil(max(a.shape) / 160))
    if stride > 1:
        a, b, weights = a[::stride, ::stride], b[::stride, ::stride], weights[::stride, ::stride]
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    w = weights.astype(np.float32)
    total = float(w.sum())
    if total < 1:
        return -1.0
    a = a - float((a * w).sum() / total)
    b = b - float((b * w).sum() / total)
    denominator = float(np.sqrt((a * a * w).sum() * (b * b * w).sum()))
    return float((a * b * w).sum() / denominator) if denominator > 1e-4 else -1.0


def coarse_ncc(a, b, weights):
    """Low-frequency foreground similarity, tolerant of brief motion blur."""

    def coarse(x):
        return cv2.GaussianBlur(cv2.resize(x, (48, 48), interpolation=cv2.INTER_AREA), (5, 5), 1.2)

    return weighted_ncc(coarse(a), coarse(b), cv2.resize(weights, (48, 48)))


def object_weights(image, box):
    """Foreground learned from this ROI, not a new detection model.

    GrabCut excludes the changing background around the rim/corners. A central
    spatial prior is retained if foreground segmentation is inconclusive.
    """
    x1, y1, x2, y2 = map(int, box)
    h, w = y2 - y1, x2 - x1
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    prior = np.exp(-(((xs / w - 0.5) / 0.34) ** 4 + ((ys / h - 0.57) / 0.40) ** 4)).astype(
        np.float32
    )
    prior[: max(2, int(h * 0.12))] *= 0.15
    patch = image[y1:y2, x1:x2]
    # This is a coarse foreground weight, not a pixel-accurate segmentation.
    # Bound initialization work so the next frame can still use optical flow.
    small = cv2.resize(patch, (min(w, 80), min(h, 80)))
    sh, sw = small.shape[:2]
    mask = np.full((sh, sw), cv2.GC_PR_BGD, np.uint8)
    mask[int(sh * 0.18) : int(sh * 0.91), int(sw * 0.14) : int(sw * 0.86)] = cv2.GC_PR_FGD
    mask[int(sh * 0.40) : int(sh * 0.73), int(sw * 0.36) : int(sw * 0.64)] = cv2.GC_FGD
    mask[:2] = mask[-2:] = cv2.GC_BGD
    mask[:, :2] = mask[:, -2:] = cv2.GC_BGD
    try:
        cv2.grabCut(
            small, mask, None, np.zeros((1, 65)), np.zeros((1, 65)), 1, cv2.GC_INIT_WITH_MASK
        )
        fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.float32)
        if 0.18 < float(fg.mean()) < 0.94:
            fg = cv2.erode(fg, np.ones((3, 3), np.uint8))
            fg = cv2.resize(fg, (w, h))
            prior *= 0.005 + 0.995 * fg
    except cv2.error:
        pass
    return np.maximum(prior, 0.001)


from bucket_robot.perception.matching import weighted_response, SpectrumCache


class TargetTracker:
    """Foreground-weighted identity plus flow/local search across detector gaps."""

    def __init__(self, image, box, stamp):
        if not valid_box(box, image.shape):
            raise RuntimeError("invalid_target_box")
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.gray_small = cv2.resize(self.gray, (640, 360), interpolation=cv2.INTER_AREA)
        self.initial = np.asarray(box, dtype=np.float64)
        self.box = self.initial.copy()
        x1, y1, x2, y2 = map(int, box)
        self.anchor = self.gray[y1:y2, x1:x2].copy()
        self.adaptive = self.anchor.astype(np.float32)
        self.references = [self.anchor.copy()]
        self.supported = []
        self.reference_at = stamp
        self.initial_quality = 1.0
        self.weights = object_weights(image, box)
        if float(self.anchor.std()) < 4:
            raise RuntimeError("target_has_too_little_texture")
        self.transform = np.eye(3)
        self.stamp = stamp
        self.quality = 1.0
        self.method = "initialized"
        self.velocity = np.zeros(2)
        self.recovery_candidate = None
        self.recovery_count = 0
        self.recovery_since = 0.0
        self.motion = MotionFilter(2, measurement_std=3.0, acceleration_std=300.0, gate_floor=35.0)
        self.motion.reset((self.box[:2] + self.box[2:]) / 2, stamp)
        self.previous_aligned = self.anchor.copy()
        self.detector_support = []
        self.last_strict_at = stamp
        self.last_flow_at = stamp
        self.last_verified_view = stamp
        self.last_rejections = []
        self.last_appearance = {}
        self.anchor_detail = float(
            cv2.Laplacian(cv2.resize(self.anchor, (96, 96)), cv2.CV_32F).var()
        )
        self.points = self.features(self.gray, self.box)
        self.detector_view = None

    @staticmethod
    def measured_view(image, box):
        """Detector coordinates avoid accumulated affine/old-view warp error."""
        x1, y1, x2, y2 = map(int, box)
        patch = cv2.resize(image[y1:y2, x1:x2], (96, 96), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        # Exclude the rim, hands and the box/background boundary.
        gray = gray[16:88, 12:84]
        return cv2.GaussianBlur(gray, (5, 5), 1.2)

    def measured_continuity(self, image, detection, stamp):
        warmup = stamp <= getattr(self, "measurement_warmup_until", -1.0)
        if (
            detection is None
            or detection["score"] < (0.65 if warmup else 0.8)
            or self.detector_view is None
        ):
            return False
        previous, oldbox, oldstamp = self.detector_view
        if not 0.0 < stamp - oldstamp <= (0.16 if warmup else 0.12) or not self.detection_confirmed(
            stamp
        ):
            return False
        box = [detection[k] for k in ("left", "top", "right", "bottom")]
        if iou(oldbox, box) < 0.55:
            return False
        view = self.measured_view(image, box)
        if min(float(view.std()), float(previous.std())) < 4:
            return False
        return weighted_ncc(view, previous, np.ones(view.shape, np.float32)) >= (
            0.60 if warmup else 0.72
        )

    def remember_detection(self, image, detection, stamp, result):
        # Store only a unique, strong, current detection after visual acceptance.
        # A rejected or missing observation never refreshes this reference.
        warmup = stamp <= getattr(self, "measurement_warmup_until", -1.0)
        if detection is not None and detection["score"] >= (0.65 if warmup else 0.8):
            box = [detection[k] for k in ("left", "top", "right", "bottom")]
            if iou(result["box"], box) >= 0.55:
                self.detector_view = (self.measured_view(image, box), box, stamp)
        return result

    def associated_detection(self, detections, box, minimum=0.536):
        """Reject ambiguous or abrupt associations before using model support."""
        box = np.asarray(box, float)
        size = box[2:] - box[:2]
        center = (box[:2] + box[2:]) / 2
        matches = []
        weak = []
        for d in detections:
            b = np.asarray([d[k] for k in ("left", "top", "right", "bottom")], float)
            ratio = (b[2:] - b[:2]) / size
            if (
                d["score"] >= minimum
                and valid_box(b)
                and iou(box, b) >= 0.35
                and np.all((ratio >= 0.7) & (ratio <= 1.4))
                and np.linalg.norm((b[:2] + b[2:]) / 2 - center) <= 120
            ):
                matches.append(d)
            elif (
                d.get("continuation_only", False)
                and 0.25 <= d["score"] < 0.536
                and valid_box(b)
                and iou(box, b) >= 0.55
                and np.all((ratio >= 0.8) & (ratio <= 1.25))
                and np.linalg.norm((b[:2] + b[2:]) / 2 - center) <= 50
            ):
                weak.append(d)
        if matches:
            return matches[0] if len(matches) == 1 else None
        return weak[0] if len(weak) == 1 else None

    def detection_confirmed(self, stamp):
        recent = [t for t in self.detector_support if 0 <= stamp - t <= 0.3]
        return len(recent) >= 2 and recent[-1] - recent[0] >= 0.025

    def features(self, gray, box):
        x1, y1, x2, y2 = np.rint(box).astype(int)
        fg = cv2.resize(self.weights, (x2 - x1, y2 - y1))
        mask = (fg > 0.25).astype(np.uint8) * 255
        points = cv2.goodFeaturesToTrack(gray[y1:y2, x1:x2], 100, 0.01, 5, mask=mask, blockSize=5)
        if points is not None:
            points += np.float32([x1, y1])
        return points

    def flow_transform(self, gray):
        if self.points is None or len(self.points) < 6:
            raise RuntimeError("target_lost_features")
        small = cv2.resize(gray, (640, 360), interpolation=cv2.INTER_AREA)
        points = self.points * 0.5
        settings = dict(
            winSize=(25, 25),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.02),
        )
        # Prediction is only an LK search hint. Forward/backward and RANSAC
        # checks still decide whether there is a current observed target.
        guess = (
            points
            + np.float32(np.clip(self.velocity * self.flow_dt, -60, 60) * 0.5).reshape(1, 1, 2)
            if hasattr(self, "flow_dt")
            else None
        )
        forward, ok, _ = cv2.calcOpticalFlowPyrLK(
            self.gray_small,
            small,
            points,
            guess,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW if guess is not None else 0,
            **settings,
        )
        if forward is None:
            raise RuntimeError("target_lost_flow")
        backward, back_ok, _ = cv2.calcOpticalFlowPyrLK(
            small, self.gray_small, forward, None, **settings
        )
        if backward is None:
            raise RuntimeError("target_lost_flow")
        forward *= 2.0
        backward *= 2.0
        good = (
            (ok.ravel() == 1)
            & (back_ok.ravel() == 1)
            & (np.linalg.norm(backward - self.points, axis=2).ravel() < 1.4)
        )
        src, dst = self.points[good].reshape(-1, 2), forward[good].reshape(-1, 2)
        if len(src) < 6:
            raise RuntimeError("target_lost_features")
        matrix, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.5, maxIters=1000, confidence=0.99
        )
        if matrix is None or inliers.sum() < 6 or inliers.mean() < 0.6:
            raise RuntimeError("target_lost_geometry")
        supported = src[inliers.ravel() == 1]
        if min(np.ptp(supported, axis=0)) < 15:
            raise RuntimeError("insufficient_feature_coverage")
        scale = math.hypot(matrix[0, 0], matrix[1, 0])
        if not 0.87 <= scale <= 1.15 or abs(math.atan2(matrix[1, 0], matrix[0, 0])) > 0.15:
            raise RuntimeError("target_geometry_jump")
        return np.vstack((matrix, [0.0, 0.0, 1.0])) @ self.transform

    def alignment(self, gray, total):
        x1, y1, x2, y2 = self.initial
        corners = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]])
        transformed = (total @ corners.T).T[:, :2]
        box = np.r_[transformed.min(axis=0), transformed.max(axis=0)]
        if not valid_box(box):
            raise RuntimeError("target_outside_tracking_window")
        h, w = self.anchor.shape
        warp = cv2.getAffineTransform(
            transformed[[0, 1, 3]].astype(np.float32), np.float32([[0, 0], [w, 0], [0, h]])
        )
        aligned = cv2.warpAffine(gray, warp, (w, h))
        fixed = max(weighted_ncc(aligned, reference, self.weights) for reference in self.references)
        recent = max(
            weighted_ncc(aligned, self.adaptive, self.weights),
            weighted_ncc(aligned, self.references[-1], self.weights),
        )
        return box, aligned, fixed, recent

    def box_transform(self, box):
        x1, y1, x2, y2 = self.initial
        a, b, c, d = box
        sx = (c - a) / (x2 - x1)
        sy = (d - b) / (y2 - y1)
        return np.array([[sx, 0.0, a - sx * x1], [0.0, sy, b - sy * y1], [0.0, 0.0, 1.0]])

    def local_search(self, gray, stamp, recovering=False):
        reference = self.references[-1] if recovering else self.adaptive
        if not hasattr(self, "search_cache"):
            self.search_cache = SpectrumCache()
        self.search_cache.bind(reference, self.weights)
        age = stamp - self.stamp
        center = (self.box[:2] + self.box[2:]) / 2 + np.clip(self.velocity * min(age, 0.4), -80, 80)
        bw, bh = self.box[2:] - self.box[:2]
        radius = min(260, 80 + 100 * age) if recovering else 90
        sx = max(0, int(center[0] - bw * 0.65 - radius))
        sy = max(0, int(center[1] - bh * 0.65 - radius))
        ex = min(1280, int(center[0] + bw * 0.65 + radius))
        ey = min(720, int(center[1] + bh * 0.65 + radius))
        # Wide recovery runs at a coarser pyramid level; the existing full-size
        # refinement and identity thresholds still decide acceptance.
        factor = 4 if recovering else 2
        search = cv2.resize(
            gray[sy:ey, sx:ex], (max(1, (ex - sx) // factor), max(1, (ey - sy) // factor))
        )
        choices = []
        for scale in (0.85, 0.93, 1.0, 1.08, 1.18) if recovering else (1.0,):
            w, h = round(bw * scale / factor), round(bh * scale / factor)
            if min(w, h) < 12 or w >= search.shape[1] or h >= search.shape[0]:
                continue
            template = cv2.resize(self.references[-1] if recovering else self.adaptive, (w, h))
            weights = cv2.resize(self.weights, (w, h))
            response = weighted_response(search, template, weights)
            _, score, _, where = cv2.minMaxLoc(response)
            px, py = where
            alternatives = response.copy()
            r = max(8, int(min(w, h) * 0.1))
            alternatives[max(0, py - r) : py + r + 1, max(0, px - r) : px + r + 1] = -1
            if score < 0.40 or score - float(alternatives.max()) < 0.01:
                continue
            box = [
                sx + px * factor,
                sy + py * factor,
                sx + (px + w) * factor,
                sy + (py + h) * factor,
            ]
            if not valid_box(box):
                continue
            # Refine the half-resolution peak at full resolution. A one-pixel
            # phase error must not make a textured identity fail reacquisition.
            bx, by, br, bb = box
            refine = factor * 2
            rx = max(0, bx - refine)
            ry = max(0, by - refine)
            full_template = cv2.resize(
                self.references[-1] if recovering else self.adaptive, (br - bx, bb - by)
            )
            full_weights = cv2.resize(self.weights, (br - bx, bb - by))
            fine = weighted_response(
                gray[ry : min(720, bb + refine), rx : min(1280, br + refine)],
                full_template,
                full_weights,
                cache=self.search_cache,
            )
            _, score, _, where = cv2.minMaxLoc(fine)
            if score < (0.85 if recovering else 0.78):
                continue
            nx, ny = rx + where[0], ry + where[1]
            box = [nx, ny, nx + br - bx, ny + bb - by]
            if not valid_box(box):
                continue
            total = self.box_transform(box)
            _, _, fixed, recent = self.alignment(gray, total)
            if fixed < (0.75 if recovering else 0.62) or recent < (0.85 if recovering else 0.78):
                continue
            choices.append((min(recent, score), total))
        if not choices:
            raise RuntimeError("target_appearance_lost")
        return max(choices, key=lambda item: item[0])[1]

    def accept(self, gray, total, stamp, method, detections=()):
        box, aligned, fixed, recent = self.alignment(gray, total)
        strict = fixed >= 0.62 and recent >= 0.76
        detection = self.associated_detection(detections, box)
        pair = weighted_ncc(aligned, self.previous_aligned, self.weights)
        # A similar background can match the old template after occlusion.
        # Without a current model box, only a very strong image match within
        # 150ms of verified flow may bridge the gap. Appearance cannot renew
        # that deadline itself; longer fallback needs independent detection.
        if (
            method == "appearance"
            and (detection is None or detection["score"] < 0.65)
            and not (0 <= stamp - self.last_flow_at <= 0.15 and min(fixed, recent, pair) >= 0.90)
        ):
            raise RuntimeError("appearance_requires_current_evidence")
        supported = detection is not None and self.detection_confirmed(stamp)
        weak = detection is not None and detection["score"] < 0.536
        coarse_fixed = coarse_pair = None
        self.last_appearance = {"fixed": fixed, "recent": recent, "pair": pair, "method": method}
        if (not strict or method == "detector_bridge") and method != "detector_continuity":
            coarse_fixed = max(coarse_ncc(aligned, r, self.weights) for r in self.references)
            coarse_pair = coarse_ncc(aligned, self.previous_aligned, self.weights)
            self.last_appearance.update(coarse_fixed=coarse_fixed, coarse_pair=coarse_pair)
            close_support = (
                supported
                and not weak
                and detection["score"] >= 0.65
                and iou(box, [detection[k] for k in ("left", "top", "right", "bottom")]) >= 0.6
            )
            if (
                method == "flow"
                and supported
                and coarse_pair >= (0.85 if weak else 0.72)
                and (coarse_fixed >= (0.70 if weak else 0.55) or not weak and pair >= 0.85)
            ):
                method = "flow_supported"
            elif (
                method == "flow"
                and stamp - self.last_strict_at <= 0.2
                and coarse_fixed >= 0.70
                and coarse_pair >= 0.78
            ):
                method = "flow_blur"
            elif (
                method == "flow"
                and pair >= 0.85
                and coarse_pair >= 0.90
                and (close_support or stamp - self.last_verified_view <= 0.20)
            ):
                # Current forward/backward flow + a matching recent view may
                # bridge an old anchor mismatch. Only independent strong model
                # support can refresh this deadline; the bridge cannot renew itself.
                method = "flow_recent"
            elif (
                method == "detector_bridge"
                and supported
                and coarse_fixed >= 0.55
                and coarse_pair >= 0.65
            ):
                pass
            else:
                raise RuntimeError("target_appearance_lost")
        delta = ((box[:2] + box[2:]) - (self.box[:2] + self.box[2:])) / 2
        if np.linalg.norm(delta) > 120:
            raise RuntimeError("target_geometry_jump")
        dt = stamp - self.stamp
        if 0 < dt <= 0.5:
            self.motion.update((box[:2] + box[2:]) / 2, stamp, min(fixed, recent))
            self.velocity = self.motion.reliable_velocity(400.0)
        associated = detection is not None and not weak
        if (
            method in ("flow", "flow_supported", "flow_recent")
            and supported
            and associated
            and detection["score"] >= 0.65
            and pair >= 0.85
            and iou(box, [detection[k] for k in ("left", "top", "right", "bottom")]) >= 0.6
        ):
            self.last_verified_view = stamp
        # Adapt only with independent detector support and a strong match to
        # a trusted identity reference. Keep the initial reference immutable.
        self.supported = [item for item in self.supported if stamp - item[0] <= 1.2]
        detail = float(cv2.Laplacian(cv2.resize(aligned, (96, 96)), cv2.CV_32F).var())
        verified_view = (
            method == "flow_supported"
            and detection is not None
            and detection["score"] >= 0.8
            and pair >= 0.8
            and coarse_fixed >= 0.55
            and detail >= 0.5 * self.anchor_detail
        )
        # Strong current detections select flow_supported above, before
        # flow_recent. Reference eligibility must include that verified path.
        recent_view = (
            method in ("flow_supported", "flow_recent")
            and supported
            and associated
            and detection["score"] >= 0.65
            and pair >= 0.90
            and coarse_pair >= 0.95
            and iou(box, [detection[k] for k in ("left", "top", "right", "bottom")]) >= 0.6
        )
        if (
            associated
            and detail >= 0.5 * self.anchor_detail
            and (
                method == "flow" and fixed > 0.70 and recent > 0.78 or verified_view or recent_view
            )
        ):
            self.supported.append((stamp, aligned.copy()))
            if (
                len(self.supported) >= 3
                and stamp - self.supported[0][0] >= 0.15
                and stamp - self.reference_at >= 0.5
            ):
                reference = np.mean([item[1] for item in self.supported[-3:]], axis=0).astype(
                    np.float32
                )
                self.references.append(reference)
                if len(self.references) > 4:
                    self.references.pop(1)  # Never discard the initial identity.
                self.adaptive = 0.6 * self.adaptive + 0.4 * reference
                self.reference_at = stamp
                self.supported = []
        self.initial_quality = weighted_ncc(aligned, self.anchor, self.weights)
        self.previous_aligned = aligned.copy()
        if strict and method != "detector_bridge":
            self.last_strict_at = stamp
        if method in ("flow", "flow_supported", "flow_blur", "flow_recent"):
            self.last_flow_at = stamp
        self.gray, self.box, self.transform, self.stamp = gray, box, total, stamp
        self.gray_small = cv2.resize(gray, (640, 360), interpolation=cv2.INTER_AREA)
        self.points = self.features(gray, box)
        self.quality = min(fixed, recent) if coarse_pair is None else min(coarse_fixed, coarse_pair)
        self.method = method
        self.recovery_count = 0
        self.recovery_candidate = None
        return self.result()

    def update(self, image, stamp, detections=(), flow_only=False):
        if not 0 < stamp - self.stamp <= 0.45:
            raise RuntimeError("tracking_frame_gap")
        self.flow_dt = stamp - self.stamp
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.last_rejections = []
        self.detector_support = [t for t in self.detector_support if stamp - t <= 0.3]
        detection = self.associated_detection(detections, self.box)
        if detection is not None:
            self.detector_support.append(stamp)

        def finish(result):
            return self.remember_detection(image, detection, stamp, result)

        try:
            return finish(self.accept(gray, self.flow_transform(gray), stamp, "flow", detections))
        except RuntimeError as exc:
            self.last_rejections.append({"stage": "flow", "reason": str(exc)})
            # Provisional identities require fresh geometric flow. Appearance
            # search cannot verify them, so do not compute a rejected fallback.
            if flow_only:
                raise
        # Only current model evidence + a fresh accepted detector view can use
        # this route. No prediction, weak boxes or stale coordinates certify it.
        if self.measured_continuity(image, detection, stamp):
            b = [detection[k] for k in ("left", "top", "right", "bottom")]
            return finish(
                self.accept(gray, self.box_transform(b), stamp, "detector_continuity", detections)
            )
        # Use a new measured box only, never extrapolate through occlusion.
        # A short bridge must regain geometric flow or use explicit reacquisition.
        if (
            detection is not None
            and detection["score"] >= 0.8
            and len(self.detector_support) >= 3
            and stamp - self.detector_support[0] >= 0.06
            and stamp - self.last_flow_at <= 0.35
        ):
            try:
                b = [detection[k] for k in ("left", "top", "right", "bottom")]
                return finish(
                    self.accept(gray, self.box_transform(b), stamp, "detector_bridge", detections)
                )
            except RuntimeError as exc:
                self.last_rejections.append({"stage": "detector_bridge", "reason": str(exc)})
        try:
            return finish(
                self.accept(gray, self.local_search(gray, stamp), stamp, "appearance", detections)
            )
        except RuntimeError as exc:
            self.last_rejections.append({"stage": "local_search", "reason": str(exc)})
            raise

    def recover(self, image, stamp, detections=()):
        """Return only after three mutually consistent fresh identity matches."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        candidates = []
        age = stamp - self.stamp
        predicted = (self.box[:2] + self.box[2:]) / 2 + np.clip(
            self.velocity * min(age, 0.4), -80, 80
        )
        for d in detections:
            b = np.array([d[k] for k in ("left", "top", "right", "bottom")], dtype=float)
            if not valid_box(b) or np.linalg.norm((b[:2] + b[2:]) / 2 - predicted) > min(
                300, 100 + 100 * age
            ):
                continue
            ratio = (b[2:] - b[:2]) / (self.box[2:] - self.box[:2])
            if np.any(ratio < 0.65) or np.any(ratio > 1.5):
                continue
            total = self.box_transform(b)
            _, _, fixed, recent = self.alignment(gray, total)
            if fixed >= 0.75 and recent >= 0.85:
                candidates.append((min(fixed, recent), total))
        try:
            total = self.local_search(gray, stamp, True)
            _, _, fixed, recent = self.alignment(gray, total)
            candidates.append((min(fixed, recent), total))
        except RuntimeError:
            pass
        if not candidates:
            self.recovery_count = 0
            self.recovery_candidate = None
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        total = candidates[0][1]
        box, _, _, _ = self.alignment(gray, total)
        for score, alternative in candidates[1:]:
            other, _, _, _ = self.alignment(gray, alternative)
            if candidates[0][0] - score < 0.04 and iou(box, other) < 0.35:
                self.recovery_count = 0
                return None
        if self.recovery_candidate is not None and iou(box, self.recovery_candidate) > 0.5:
            self.recovery_count += 1
        else:
            self.recovery_count = 1
            self.recovery_since = stamp
        self.recovery_candidate = box
        if self.recovery_count < 3 or stamp - self.recovery_since < 0.18:
            return None
        # Recovery may span a wider region than a normal adjacent frame.
        oldbox = self.box.copy()
        self.box = box.copy()
        try:
            return self.accept(gray, total, stamp, "reacquired", detections)
        except RuntimeError:
            self.box = oldbox
            return None

    def result(self):
        b = np.rint(self.box).astype(int).tolist()
        return {
            "box": b,
            "offset_px": [
                (self.box[0] + self.box[2]) / 2 - 640,
                (self.box[1] + self.box[3]) / 2 - 360,
            ],
            "quality": self.quality,
            "timestamp": self.stamp,
            "method": self.method,
            "reference_count": len(self.references),
            "initial_similarity": self.initial_quality,
        }


class EdgeTracker(TargetTracker):
    """Follow only a currently measured visible portion until the box is whole.

    No optical-flow/search fallback is allowed to keep a clipped target alive
    without a new detection. The visible center steers inward, and is not a
    reconstruction of the physical bucket center outside the image.
    """

    def __init__(self, image, box, stamp):
        super().__init__(image, box, stamp)
        self.full_votes = []
        self.ready_for_full = False

    def update(self, image, stamp, detections=(), flow_only=False):
        if flow_only:
            raise RuntimeError("edge_requires_current_detection")
        if not 0 < stamp - self.stamp <= 0.16:
            raise RuntimeError("edge_frame_gap")
        candidates = []
        for d in detections:
            if d["score"] < 0.536 or d.get("continuation_only", False):
                continue
            raw = [d[k] for k in ("left", "top", "right", "bottom")]
            b = visible_detection_box(raw, image.shape)
            if b is not None:
                candidates.append((d, raw, np.asarray(b, float)))
        if len(candidates) != 1 or candidates[0][0]["score"] < 0.65:
            raise RuntimeError("edge_requires_unique_detection")
        d, raw, b = candidates[0]
        ratio = (b[2:] - b[:2]) / (self.box[2:] - self.box[:2])
        if iou(b, self.box) < 0.4 or np.any((ratio < 0.65) | (ratio > 1.5)):
            raise RuntimeError("edge_geometry_jump")
        view = self.measured_view(image, b)
        previous = self.detector_view[0]
        pair = weighted_ncc(view, previous, np.ones(view.shape, np.float32))
        if min(float(view.std()), float(previous.std())) < 4 or pair < 0.60:
            raise RuntimeError("edge_appearance_lost")
        # Fresh strong complete boxes hand over to ordinary tracking, after
        # the previously clipped geometry has stabilized across several frames.
        if valid_box(raw, image.shape) and d["score"] >= 0.8:
            self.full_votes = [t for t in self.full_votes if stamp - t <= 0.3] + [stamp]
        else:
            self.full_votes = []
        self.ready_for_full = len(self.full_votes) >= 3 and stamp - self.full_votes[0] >= 0.12
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.box = b
        self.stamp = stamp
        self.quality = min(pair, d["score"])
        self.method = "edge_detector"
        self.detector_view = (view, b.tolist(), stamp)
        self.detector_support = [t for t in self.detector_support if stamp - t <= 0.3] + [stamp]
        return self.result()

    def recover(self, image, stamp, detections=()):
        return None  # Reacquisition must collect new independent model evidence.

    def result(self):
        result = super().result()
        result["partial_visible"] = True
        return result


class TargetLock:
    """Separate detector misses, short visual loss, and expired identity."""

    def __init__(self, mode="auto"):
        self.mode = mode
        self.tracker = None
        self.acquirer = Acquirer()
        self.state = "searching"
        self.loss_since = None
        self.reason = ""
        self.generation = 0
        self.reacquired = 0
        self.expired = False
        self.relock = Acquirer()
        self.recovery_attempt_at = -1.0
        self.provisional = None
        self.memory = None
        self.memory_until = 0.0
        self.wide_views = []
        self.wide_previous = None

    def select(self, image, box, stamp):
        self.install(TargetTracker(image, box, stamp))

    def install(self, tracker):
        self.tracker = tracker
        self.state = "tracking"
        self.generation += 1
        self.loss_since = None
        self.reason = ""
        self.expired = False
        self.relock = Acquirer()
        self.recovery_attempt_at = -1.0
        self.provisional = None
        self.memory = None
        self.memory_until = 0.0
        self.wide_views = []
        self.wide_previous = None

    def measured_reappearance(self, image, stamp, detections):
        """Confirm a new automatic acquisition without the old position gate.

        Independent current model votes and pairwise image continuity replace
        proximity to the pre-occlusion box. This is a NEW identity generation,
        never a claim that an unobserved object followed a predicted trajectory.
        """
        if self.wide_views and stamp <= self.wide_views[-1][0]:
            return None
        candidates = [
            d
            for d in detections
            if d["score"] >= 0.536
            and not d.get("continuation_only", False)
            and visible_detection_box(
                [d[k] for k in ("left", "top", "right", "bottom")], image.shape
            )
            is not None
        ]
        if len(candidates) != 1 or candidates[0]["score"] < 0.65:
            self.wide_views = []
            self.wide_previous = None
            return None
        d = candidates[0]
        raw = [d[k] for k in ("left", "top", "right", "bottom")]
        edge = not valid_box(raw, image.shape)
        box = np.array(visible_detection_box(raw, image.shape), float)
        if edge and d["score"] < 0.8:
            self.wide_views = []
            self.wide_previous = None
            return None
        view = TargetTracker.measured_view(image, box)
        if float(view.std()) < 4:
            self.wide_views = []
            self.wide_previous = None
            return None
        consistent = False
        pair = 1.0
        if self.wide_views:
            last, oldbox, _, _ = self.wide_views[-1]
            ratio = (box[2:] - box[:2]) / (oldbox[2:] - oldbox[:2])
            pair = weighted_ncc(view, self.wide_previous, np.ones(view.shape, np.float32))
            consistent = (
                0 < stamp - last <= 0.16
                and iou(oldbox, box) >= 0.4
                and np.all((ratio >= 0.65) & (ratio <= 1.5))
                and pair >= 0.30
            )
        if not consistent:
            self.wide_views = []
        self.wide_views = [r for r in self.wide_views if stamp - r[0] <= 0.4]
        self.wide_views.append((stamp, box, d["score"], pair if consistent else 1.0))
        self.wide_previous = view
        span = stamp - self.wide_views[0][0]
        strict = (
            len(self.wide_views) >= 3
            and span >= 0.12
            and all(p >= 0.72 for _, _, _, p in self.wide_views)
            and sum(score >= 0.8 for _, _, score, _ in self.wide_views) >= 2
        )
        # A changing viewpoint may keep geometry/class evidence while pixel
        # correlation is only moderate. Require a longer, all-strong sequence
        # for that route, rather than relaxing the ordinary appearance tracker.
        changed_view = (
            len(self.wide_views) >= 4
            and span >= 0.20
            and all(score >= 0.8 for _, _, score, _ in self.wide_views)
        )
        if d["score"] < 0.8 or not (strict or changed_view):
            return None
        delay = stamp - self.loss_since if self.loss_since is not None else None
        span = stamp - self.wide_views[0][0]
        votes = len(self.wide_views)
        if edge and not all(score >= 0.8 for _, _, score, _ in self.wide_views):
            return None
        try:
            candidate = (EdgeTracker if edge else TargetTracker)(
                image, box.astype(int).tolist(), stamp
            )
        except RuntimeError:
            return None
        if candidate.points is None or len(candidate.points) < 6:
            return None
        candidate.detector_support = [t for t, _, _, _ in self.wide_views if stamp - t <= 0.3]
        candidate.detector_view = (view, box.astype(int).tolist(), stamp)
        # Carry the observed detector evidence into the new track, rather than
        # immediately losing it while optical flow initializes under rotation.
        # This deadline never renews itself, and every step needs a current box.
        candidate.measurement_warmup_until = stamp + 0.35
        self.install(candidate)
        candidate.method = "detector_edge_relock" if edge else "detector_wide_relock"
        result = candidate.result()
        result.update(
            relock_delay_s=delay,
            candidate_verify_s=span,
            candidate_detection_votes=votes,
            candidate_evidence="appearance_and_detection"
            if strict
            else "strong_detection_sequence",
        )
        return result

    def verified_reappearance(self, image, stamp, detections, reference=None):
        """A known appearance seeds a candidate; fresh geometric frames verify it.

        No provisional box is exposed to motion. This can bridge detector misses
        without reusing an old coordinate or accepting a lone unverified hit.
        """
        if self.provisional is not None:
            pending = self.provisional
            candidate = pending["tracker"]
            if stamp <= candidate.stamp:
                return None
            try:
                if stamp - pending["started"] > 0.4:
                    raise RuntimeError("candidate_timeout")
                if stamp - candidate.stamp > 0.12:
                    raise RuntimeError("candidate_frame_gap")
                result = candidate.update(image, stamp, [], flow_only=True)
                if result["method"] != "flow" or result["quality"] < 0.8:
                    raise RuntimeError("candidate_not_geometric")
                pending["frames"] += 1
                associated = candidate.associated_detection(
                    detections, candidate.box, minimum=0.536
                )
                if associated is not None:
                    pending["votes"].append((stamp, associated["score"]))
                strong = any(score >= 0.65 for _, score in pending["votes"])
                detector_confirmed = len(pending["votes"]) >= (2 if strong else 3)
                required_span = 0.06 if pending["known"] and pending["score"] >= 0.8 else 0.12
                if (
                    pending["frames"] >= 3
                    and stamp - pending["started"] >= required_span
                    and (pending["known"] or detector_confirmed)
                ):
                    delay = stamp - self.loss_since if self.loss_since is not None else None
                    initial = self.generation == 0
                    self.install(candidate)
                    candidate.method = (
                        "detector_flow_lock"
                        if initial
                        else "appearance_relock"
                        if pending["known"]
                        else "detector_flow_relock"
                    )
                    result = candidate.result()
                    result["candidate_verify_s"] = stamp - pending["started"]
                    result["candidate_seed_score"] = pending["score"]
                    result["candidate_detection_votes"] = len(pending["votes"])
                    if delay is not None:
                        result["relock_delay_s"] = delay
                    return result
                return None
            except RuntimeError:
                self.provisional = None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        matches = []
        nearby = []
        old = reference.box if reference is not None else None
        size = old[2:] - old[:2] if old is not None else None
        for d in detections:
            b = np.asarray([d[k] for k in ("left", "top", "right", "bottom")], float)
            if d["score"] < 0.536 or not valid_box(b):
                continue
            if reference is None:
                matches.append((d, b, False))
                continue
            ratio = (b[2:] - b[:2]) / size
            # Strong appearance can identify a substantially nearer/farther
            # bucket. Unknown appearances still need independent confirmation.
            if np.any((ratio < 0.25) | (ratio > 4.0)):
                continue
            try:
                _, aligned, fixed, recent = reference.alignment(gray, reference.box_transform(b))
                known = fixed >= 0.70 and recent >= 0.70
                if not known and d["score"] >= 0.65:
                    coarse_fixed = max(
                        coarse_ncc(aligned, r, reference.weights) for r in reference.references
                    )
                    known = (
                        coarse_fixed >= 0.82
                        and coarse_ncc(aligned, reference.previous_aligned, reference.weights)
                        >= 0.82
                    )
                if d["score"] < 0.65:
                    known = known and fixed >= 0.82 and recent >= 0.82
                if known:
                    matches.append((d, b, True))
                if (
                    np.all((ratio >= 0.65) & (ratio <= 1.5))
                    and iou(old, b) >= 0.15
                    and np.linalg.norm((b[:2] + b[2:] - old[:2] - old[2:]) / 2)
                    <= min(240, 0.75 * np.linalg.norm(size))
                ):
                    nearby.append((d, b, False))
            except RuntimeError:
                continue
        # Changed views need independent detector votes plus fresh geometry.
        # This is a new automatic acquisition, not proof of old visual identity.
        if not matches and len(nearby) == 1:
            matches = nearby
        if len(matches) == 1:
            d, b, known = matches[0]
            try:
                candidate = TargetTracker(image, b.astype(int).tolist(), stamp)
                if candidate.points is not None and len(candidate.points) >= 6:
                    self.provisional = {
                        "tracker": candidate,
                        "started": stamp,
                        "frames": 1,
                        "score": d["score"],
                        "known": known,
                        "votes": [(stamp, d["score"])],
                    }
            except RuntimeError:
                pass
        return None

    def update(self, image, metadata):
        stamp = metadata["arrival"]
        detections = metadata.get("detections", [])
        self.expired = False
        if self.tracker is not None and self.state == "tracking":
            # ByteTrack-style second association: weak model boxes may support
            # an observed track, but never create/reacquire one on their own.
            detections = list(detections) + [
                {**d, "continuation_only": True}
                for d in metadata.get("tracking_candidates", [])
                if 0.25 <= d["score"] < 0.536
            ]
        if self.tracker is None:
            if self.mode == "auto":
                if self.memory is not None and stamp > self.memory_until:
                    self.memory = None
                    self.provisional = None
                result = self.verified_reappearance(
                    image,
                    stamp,
                    detections,
                    None if isinstance(self.memory, EdgeTracker) else self.memory,
                )
                if result is not None:
                    return result
                initial_edge = any(
                    not valid_box([d[k] for k in ("left", "top", "right", "bottom")], image.shape)
                    for d in detections
                )
                if self.generation > 0 or self.wide_views or initial_edge:
                    result = self.measured_reappearance(image, stamp, detections)
                    if result is not None:
                        return result
            box = self.acquirer.update(detections, stamp) if self.mode == "auto" else None
            if box is not None:
                try:
                    self.select(image, box, stamp)
                    return self.tracker.result()
                except RuntimeError:
                    return None
            return None
        if self.state == "tracking":
            if stamp == self.tracker.stamp:
                return self.tracker.result()
            try:
                result = self.tracker.update(image, stamp, detections)
                if isinstance(self.tracker, EdgeTracker) and self.tracker.ready_for_full:
                    previous = self.tracker
                    candidate = TargetTracker(image, previous.box.astype(int).tolist(), stamp)
                    candidate.detector_support = previous.detector_support.copy()
                    candidate.detector_view = previous.detector_view
                    candidate.measurement_warmup_until = stamp + 0.35
                    self.install(candidate)
                    candidate.method = "edge_full_lock"
                    result = candidate.result()
                return result
            except RuntimeError as exc:
                self.state = "reacquiring"
                self.loss_since = stamp
                self.reason = str(exc)
                self.relock = Acquirer()
                self.recovery_attempt_at = -1.0
                self.provisional = None
                self.wide_views = []
                self.wide_previous = None
                return None
        if self.mode == "auto":
            result = self.verified_reappearance(
                image,
                stamp,
                detections,
                None if isinstance(self.tracker, EdgeTracker) else self.tracker,
            )
            if result is not None:
                return result
            result = self.measured_reappearance(image, stamp, detections)
            if result is not None:
                return result
            # A blurred/changed view can fail old-template identity matching
            # while the detector repeatedly sees a nearby bucket. Confirm that
            # hypothesis in parallel; do not spend five seconds on the old view.
            nearby = []
            old = self.tracker.box
            size = old[2:] - old[:2]
            center = (old[:2] + old[2:]) / 2
            for d in detections:
                b = np.asarray([d[k] for k in ("left", "top", "right", "bottom")], float)
                ratio = (b[2:] - b[:2]) / size
                if (
                    d["score"] >= 0.65
                    and valid_box(b)
                    and np.all((ratio >= 0.65) & (ratio <= 1.5))
                    and iou(old, b) >= 0.15
                    and np.linalg.norm((b[:2] + b[2:]) / 2 - center)
                    <= min(240, 0.75 * np.linalg.norm(size))
                ):
                    nearby.append(d)
            if len(nearby) > 1:
                self.relock = Acquirer()  # No automatic choice among competing buckets.
            else:
                box = self.relock.update(nearby, stamp)
                votes = self.relock.hits
                confirmed = (
                    len(votes) >= 5
                    or len(votes) >= 3
                    and all(score >= 0.8 for _, score in votes[-3:])
                )
                if box is not None and confirmed and stamp - self.loss_since >= 0.18:
                    # This is renewed detector acquisition, not proof that the
                    # old appearance identity survived. Publish a new target ID.
                    delay = stamp - self.loss_since
                    try:
                        self.select(image, box, stamp)
                    except RuntimeError:
                        return None
                    self.tracker.method = "detector_relock"
                    result = self.tracker.result()
                    result["relock_delay_s"] = delay
                    return result
                if nearby and stamp - self.loss_since <= 5:
                    # Let fresh detector votes arrive at camera cadence instead
                    # of blocking them behind a costly multi-scale old-view search.
                    return None
        if stamp - self.loss_since > 5:
            if self.mode == "auto":
                self.memory = self.tracker
                self.memory_until = stamp + 10
            self.tracker = None
            self.acquirer = Acquirer()
            self.state = "searching"
            self.expired = True
            self.reason = "target_reacquire_timeout"
            return None
        if self.provisional is not None:
            return None
        if self.mode == "auto" and stamp - self.recovery_attempt_at < 0.15:
            return None
        self.recovery_attempt_at = stamp
        result = self.tracker.recover(image, stamp, detections)
        if result is not None:
            self.state = "tracking"
            self.loss_since = None
            self.reason = ""
            self.reacquired += 1
        return result
