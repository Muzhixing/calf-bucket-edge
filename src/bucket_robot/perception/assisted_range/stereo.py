"""Calibrated classical-first stereo with frozen Fast-ACVNet fallback.

No camera, serial, or motion I/O. Distances use the stereo optical midpoint.
The bucket ROI is a surface proxy, not a bucket-mouth or collision model.
"""

import hashlib
import ctypes
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

CAL_SHA = "5917ede19309a4f78d810b3c9c91d4e8033c23751307e6879f7e3f55f5c7031e"
MODEL_SHA = "6f3e86c00b6ba3dc63a9d6d7b070d6502b721d2e5ac1eb4f11087ced1b9405ac"


def verified(path, digest):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
        raise ValueError("asset_hash_mismatch: " + str(path))


class Geometry:
    def __init__(self, calibration):
        verified(calibration, CAL_SHA)
        c = json.loads(Path(calibration).read_text())
        self.c = {
            k: np.asarray(c[k], dtype=np.float64)
            for k in ("K1", "K2", "D1", "D2", "R", "T", "R1", "R2", "P1", "P2")
        }
        self.maps = [
            cv2.initUndistortRectifyMap(
                self.c["K" + n],
                self.c["D" + n],
                self.c["R" + n],
                self.c["P" + n],
                (1280, 720),
                cv2.CV_16SC2,
            )
            for n in ("1", "2")
        ]
        self.p = self.c["P1"].copy()
        self.p[:2] *= 0.5
        self.factor = -float(self.c["P2"][0, 3]) * 0.5 / 1000
        self.origin = self.c["R1"] @ (-self.c["R"].T @ self.c["T"].reshape(3)) / 2000

    def rectify(self, raw):
        if raw.shape != (720, 2560, 3) or raw.dtype != np.uint8:
            raise ValueError("invalid_stereo_pair")
        return [
            cv2.remap(im, *mp, cv2.INTER_LINEAR)
            for im, mp in zip((raw[:, :1280], raw[:, 1280:]), self.maps)
        ]

    def distance_map(self, disparity):
        yy, xx = np.indices(disparity.shape)
        z = np.divide(
            self.factor, disparity, out=np.full_like(disparity, np.nan), where=disparity > 0
        )
        xyz = np.stack(
            (
                (xx + 0.25 - self.p[0, 2]) * z / self.p[0, 0],
                (yy + 0.25 - self.p[1, 2]) * z / self.p[1, 1],
                z,
            ),
            -1,
        )
        return np.linalg.norm(xyz - self.origin, axis=-1)


def consistent(d, rd):
    yy, xx = np.indices(d.shape, dtype=np.float32)
    sampled = cv2.remap(
        rd, xx - d, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-999
    )
    return (
        np.isfinite(d)
        & (d > 2)
        & (d < 126)
        & (xx - d >= 0)
        & (xx - d < d.shape[1] - 1)
        & (sampled > 0)
        & (abs(d - sampled) <= 1)
    )


class FastACV:
    def __init__(self, model):
        verified(model, MODEL_SHA)
        bridge = os.environ.get("BUCKET_ACV_BRIDGE")
        if not bridge:
            raise RuntimeError("validated_ACV_bridge_required; stock runtime lacks CPU GridSample")
        self.runtime = ctypes.CDLL(bridge)
        self.runtime.acv_open.argtypes = [ctypes.c_char_p]
        self.runtime.acv_open.restype = ctypes.c_void_p
        pointer = ctypes.POINTER(ctypes.c_float)
        self.runtime.acv_infer.argtypes = [ctypes.c_void_p, pointer, pointer, pointer]
        self.runtime.acv_infer.restype = ctypes.c_int
        self.runtime.acv_close.argtypes = [ctypes.c_void_p]
        self.runtime.acv_close.restype = None
        self.handle = self.runtime.acv_open(os.fsencode(model))
        if not self.handle:
            raise RuntimeError("model_runtime_initialization_failed")

    @staticmethod
    def tensor(image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None]
        value = rgb.astype(np.float32) / 255
        mean = np.array([0.485, 0.456, 0.406], np.float32)[None, :, None, None]
        std = np.array([0.229, 0.224, 0.225], np.float32)[None, :, None, None]
        return np.ascontiguousarray(np.pad((value - mean) / std, ((0, 0), (0, 0), (24, 0), (0, 0))))

    def infer(self, left, right):
        a, b = self.tensor(left), self.tensor(right)
        output = np.empty((384, 640), np.float32)
        pointer = ctypes.POINTER(ctypes.c_float)
        rc = self.runtime.acv_infer(
            self.handle,
            a.ctypes.data_as(pointer),
            b.ctypes.data_as(pointer),
            output.ctypes.data_as(pointer),
        )
        if rc:
            raise RuntimeError("model_inference_failed: " + str(rc))
        d = output[24:].copy()
        if not np.isfinite(d).all():
            raise RuntimeError("nonfinite_model_output")
        return d

    def maps(self, left, right):
        d = self.infer(left, right)
        rd = np.flip(self.infer(np.flip(right, 1).copy(), np.flip(left, 1).copy()), 1).copy()
        return d, consistent(d, rd)

    def close(self):
        if self.handle:
            self.runtime.acv_close(self.handle)
            self.handle = None


class AssistedRange:
    def __init__(self, calibration, model):
        self.geometry = Geometry(calibration)
        self.model = FastACV(model)

        def matcher(minimum):
            return cv2.StereoSGBM_create(
                minimum,
                128,
                7,
                8 * 3 * 49,
                32 * 3 * 49,
                1,
                63,
                12,
                120,
                2,
                cv2.STEREO_SGBM_MODE_SGBM_3WAY,
            )

        self.left_matcher, self.right_matcher = matcher(0), matcher(-128)

    def summarize(self, d, valid, roi, source):
        x1, y1, x2, y2 = roi
        mask = valid[y1:y2, x1:x2]
        result = dict(
            valid=False,
            distance_m=None,
            source=source,
            roi=roi,
            distance_kind="stereo_optical_midpoint_to_bucket_surface",
            support_pixels=int(mask.sum()),
            support_fraction=float(mask.mean()),
        )
        if mask.sum() < 15 or mask.mean() < 0.25:
            return dict(result, reason="insufficient_stereo_support")
        values = self.geometry.distance_map(d)[y1:y2, x1:x2][mask]
        q10, median, q90 = np.quantile(values, [0.1, 0.5, 0.9])
        result.update(lower_m=float(q10), upper_m=float(q90))
        if not np.isfinite(values).all() or (q90 - q10) / median > 0.12:
            return dict(result, reason="ambiguous_surface")
        if not 0.17 <= median <= 1.2:
            return dict(result, reason="outside_working_range")
        return dict(result, valid=True, distance_m=float(median), reason="consistent_surface")

    def measure(self, rectified, box, point_roi=None):
        """point_roi is only for paired point replay; live uses an inner body ROI."""
        started = time.monotonic()
        left, right = [cv2.resize(im, (640, 360), interpolation=cv2.INTER_AREA) for im in rectified]
        x1, y1, x2, y2 = map(float, box)
        if (
            not all(np.isfinite(v) for v in box)
            or not 0 <= x1 < x2 <= 1280
            or not 0 <= y1 < y2 <= 720
        ):
            raise ValueError("invalid_bucket_box")
        w, h = x2 - x1, y2 - y1
        roi = point_roi or [
            int((x1 + 0.35 * w) / 2),
            int((y1 + 0.35 * h) / 2),
            int((x1 + 0.65 * w) / 2),
            int((y1 + 0.65 * h) / 2),
        ]
        a, b, c, e = roi
        if not (0 <= a < c <= 640 and 0 <= b < e <= 360):
            raise ValueError("invalid_surface_roi")
        # Preserve the classical branch's grayscale-before-resize ordering.
        grey = [
            cv2.resize(
                cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (640, 360), interpolation=cv2.INTER_AREA
            )
            for im in rectified
        ]
        d = self.left_matcher.compute(*grey).astype(np.float32) / 16
        rd = -self.right_matcher.compute(grey[1], grey[0]).astype(np.float32) / 16
        mask = consistent(d, rd)
        f = grey[0].astype(np.float32)
        mu = cv2.boxFilter(f, -1, (9, 9))
        mask &= np.maximum(0, cv2.boxFilter(f * f, -1, (9, 9)) - mu * mu) >= 36
        result = self.summarize(d, mask, roi, "classical")
        if not result["valid"]:
            rejected = result
            d, mask = self.model.maps(left, right)
            result = self.summarize(d, mask, roi, "fast_acv")
            result["classical_rejected"] = rejected
        result["elapsed_s"] = time.monotonic() - started
        return result

    def close(self):
        self.model.close()
