#!/usr/bin/env python3
"""Capture raw stereo checkerboard pairs. No serial access or actuator commands."""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Cannot save {path}")
    recovered = cv2.imread(str(path))
    if recovered is None or not np.array_equal(recovered, image):
        raise RuntimeError(f"PNG verification failed: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="/dev/v4l/by-id/usb-USB_Camera_USB_Camera_01.00.00-video-index0"
    )
    parser.add_argument("--out", required=True, help="New pose directory; never overwrites")
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--role", choices=["pilot", "fit", "holdout"], default="pilot")
    args = parser.parse_args()
    if not 1 <= args.pairs <= 20:
        parser.error("pairs must be 1..20")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        start_utc=datetime.now(timezone.utc).isoformat(),
        device=args.device,
        resolved_device=str(Path(args.device).resolve()),
        opencv=cv2.__version__,
        role=args.role,
        raw_images=True,
        rectification_applied=False,
        split_order="SBS left half / right half; physical identity must match camera setup",
        inner_corners=[9, 6],
        square_size_mm=30.0,
        scale_basis="User measured flat mounted board 300 x 210 mm on 2026-11-22",
        serial_access=False,
        frames=[],
    )
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    try:
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        manifest["negotiated"] = {
            name: cap.get(prop)
            for name, prop in [
                ("width", cv2.CAP_PROP_FRAME_WIDTH),
                ("height", cv2.CAP_PROP_FRAME_HEIGHT),
                ("fps", cv2.CAP_PROP_FPS),
                ("fourcc", cv2.CAP_PROP_FOURCC),
            ]
        }
        # Warm up exposure. All processing happens after capture to avoid queueing old frames.
        captured = []
        for index in range(30 + args.pairs * 6):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera read failed")
            if frame.shape != (720, 2560, 3):
                raise RuntimeError(f"Unexpected SBS shape: {frame.shape}")
            if index >= 30 and (index - 30) % 6 == 0:
                captured.append(
                    (datetime.now(timezone.utc).isoformat(), time.monotonic_ns(), frame.copy())
                )
        cap.release()
        for index, (utc, mono, frame) in enumerate(captured):
            record = dict(index=index, utc=utc, host_read_monotonic_ns=mono, eyes={})
            record["sbs_sha256"] = save_image(out / f"{index:03d}_sbs.png", frame)
            previews = []
            for eye, image in [("left", frame[:, :1280]), ("right", frame[:, 1280:])]:
                digest = save_image(out / f"{index:03d}_{eye}.png", image)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                ok, corners = cv2.findChessboardCornersSB(
                    gray, (9, 6), cv2.CALIB_CB_NORMALIZE_IMAGE
                )
                result = dict(found=bool(ok), sha256=digest, corners_px=None)
                overlay = image.copy()
                if ok:
                    pts = corners.reshape(-1, 2)
                    result.update(
                        corners_px=pts.tolist(),
                        inner_extent_px=[pts.min(0).tolist(), pts.max(0).tolist()],
                    )
                    cv2.drawChessboardCorners(overlay, (9, 6), corners, ok)
                cv2.putText(
                    overlay,
                    f"{eye.upper()} {54 if ok else 0}/54 | RAW",
                    (24, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0) if ok else (0, 0, 255),
                    2,
                )
                record["eyes"][eye] = result
                previews.append(overlay)
            save_image(out / f"{index:03d}_corners.png", np.hstack(previews))
            manifest["frames"].append(record)
            print(
                json.dumps(
                    {
                        "index": index,
                        "left": record["eyes"]["left"]["found"],
                        "right": record["eyes"]["right"]["found"],
                    }
                ),
                flush=True,
            )
    except BaseException as exc:
        manifest["error"] = str(exc)
        raise
    finally:
        cap.release()
        manifest["end_utc"] = datetime.now(timezone.utc).isoformat()
        (out / "capture.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
