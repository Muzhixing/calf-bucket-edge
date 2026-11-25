"""Fit a candidate from raw captures; compare held-out poses with old calibration.

Never changes the active runtime calibration. One image pair per physical pose.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

SIZE = (1280, 720)
OBJECT = np.zeros((54, 3), np.float32)
OBJECT[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 30


def array_calibration(data):
    return {
        k: np.array(data[k], np.float64)
        for k in ("K1", "D1", "K2", "D2", "R", "T", "R1", "R2", "P1", "P2", "Q")
    }


def load_poses(root):
    rows = []
    for path in sorted(root.rglob("capture.json")):
        d = json.loads(path.read_text())
        if d["role"] not in ("fit", "holdout") or d.get("error"):
            continue
        if d["square_size_mm"] != 30 or d["inner_corners"] != [9, 6] or not d["raw_images"]:
            raise ValueError(f"Incompatible geometry: {path}")
        f = d["frames"][0]
        if not all(f["eyes"][e]["found"] for e in ("left", "right")):
            continue
        points = []
        for eye in ("left", "right"):
            im = path.parent / f"{f['index']:03d}_{eye}.png"
            if hashlib.sha256(im.read_bytes()).hexdigest() != f["eyes"][eye]["sha256"]:
                raise ValueError(f"Image hash mismatch: {im}")
            pts = np.array(f["eyes"][eye]["corners_px"], np.float32).reshape(-1, 1, 2)
            if pts.shape != (54, 1, 2) or not np.isfinite(pts).all():
                raise ValueError(f"Invalid corners: {path}")
            points.append(pts)
        left, right = points
        reversed_right = bool(
            np.dot((left[-1] - left[0]).ravel(), (right[-1] - right[0]).ravel()) < 0
        )
        if reversed_right:
            right = right[::-1].copy()
        rows.append(
            dict(
                id=str(path.parent.relative_to(root)),
                role=d["role"],
                left=left,
                right=right,
                group=d.get("board_orientation_group", "printed_board_initial"),
                right_order_reversed=reversed_right,
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    return rows


def evaluate(c, row):
    pl = cv2.undistortPoints(row["left"], c["K1"], c["D1"], R=c["R1"], P=c["P1"]).reshape(-1, 2)
    pr = cv2.undistortPoints(row["right"], c["K2"], c["D2"], R=c["R2"], P=c["P2"]).reshape(-1, 2)
    h = cv2.triangulatePoints(c["P1"], c["P2"], pl.T, pr.T)
    xyz = (h[:3] / h[3]).T
    if not np.isfinite(xyz).all() or np.any(xyz[:, 2] <= 0):
        raise ValueError("Invalid stereo triangulation")
    grid = xyz.reshape(6, 9, 3)
    edges = np.concatenate(
        [
            np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel(),
            np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel(),
        ]
    )
    ok, rv, tv = cv2.solvePnP(OBJECT, row["left"], c["K1"], c["D1"])
    if not ok:
        raise ValueError("PnP failed")
    reproj = cv2.projectPoints(OBJECT, rv, tv, c["K1"], c["D1"])[0]
    normal = cv2.Rodrigues(rv)[0][:, 2]
    raw_xyz = xyz @ c["R1"]
    pnp_center = cv2.Rodrigues(rv)[0] @ OBJECT.mean(0) + tv.ravel()
    if normal[2] < 0:
        normal = -normal
    return dict(
        id=row["id"],
        role=row["role"],
        group=row["group"],
        vertical_abs_median_px=float(np.median(np.abs(pl[:, 1] - pr[:, 1]))),
        vertical_abs_p95_px=float(np.percentile(np.abs(pl[:, 1] - pr[:, 1]), 95)),
        left_reprojection_rms_px=float(
            np.sqrt(np.mean(np.sum((reproj - row["left"]) ** 2, axis=2)))
        ),
        grid_edge_median_mm=float(np.median(edges)),
        grid_edge_abs_error_p95_mm=float(np.percentile(np.abs(edges - 30), 95)),
        board_center_rectified_z_mm=float(xyz.mean(0)[2]),
        board_center_raw_left_z_mm=float(raw_xyz.mean(0)[2]),
        board_center_slant_mm=float(np.linalg.norm(raw_xyz.mean(0))),
        left_pnp_center_z_mm=float(pnp_center[2]),
        stereo_minus_pnp_raw_z_mm=float(raw_xyz.mean(0)[2] - pnp_center[2]),
        normal_pitch_deg=float(np.degrees(np.arctan2(normal[1], normal[2]))),
        normal_yaw_deg=float(np.degrees(np.arctan2(normal[0], normal[2]))),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--captures", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rows = load_poses(args.captures)
    fit = [r for r in rows if r["role"] == "fit"]
    holdout = [r for r in rows if r["role"] == "holdout"]
    if len(fit) < 10 or len(holdout) < 3:
        p.error("Need at least 10 fit poses and 3 separately positioned holdout poses")
    args.out.mkdir(parents=True, exist_ok=False)
    baseline_data = json.loads(args.baseline.read_text())
    baseline = array_calibration(baseline_data)
    objects = [OBJECT.copy() for _ in fit]
    # Keep the deployed radial2 lens family for a comparable calibration update.
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-9)
    eyes = []
    for index, eye in enumerate(("left", "right"), 1):
        result = cv2.calibrateCameraExtended(
            objects,
            [r[eye] for r in fit],
            SIZE,
            baseline[f"K{index}"].copy(),
            baseline[f"D{index}"].copy(),
            flags=flags,
            criteria=criteria,
        )
        eyes.append(result)
    stereo = cv2.stereoCalibrate(
        objects,
        [r["left"] for r in fit],
        [r["right"] for r in fit],
        eyes[0][1],
        eyes[0][2],
        eyes[1][1],
        eyes[1][2],
        SIZE,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=criteria,
    )
    rms, k1, d1, k2, d2, r, t, e, f = stereo
    rect = cv2.stereoRectify(k1, d1, k2, d2, SIZE, r, t)
    c = dict(
        K1=k1,
        D1=d1,
        K2=k2,
        D2=d2,
        R=r,
        T=t,
        R1=rect[0],
        R2=rect[1],
        P1=rect[2],
        P2=rect[3],
        Q=rect[4],
    )
    result = {k: v.tolist() for k, v in c.items()}
    result.update(
        units="mm",
        image_size=list(SIZE),
        square_size_mm=30.0,
        baseline_mm=float(np.linalg.norm(t)),
        fit_left_rms_px=eyes[0][0],
        fit_right_rms_px=eyes[1][0],
        fit_stereo_rms_px=rms,
        intrinsic_std_left=eyes[0][5].ravel().tolist(),
        intrinsic_std_right=eyes[1][5].ravel().tolist(),
        lens_model="double_radial_zero_tangential",
        deployment_status="CANDIDATE_ONLY_NOT_ACTIVATED",
        independent_scale_measurement=True,
        scale_basis="user measured flat printed board 300 x 210 mm",
        metric_range_validated=False,
        fit_pose_ids=[r["id"] for r in fit],
        validation_pose_ids=[r["id"] for r in holdout],
        source_pose_sha256={r["id"]: r["source_sha256"] for r in rows},
        baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        opencv=cv2.__version__,
    )
    (args.out / "candidate.json").write_text(json.dumps(result, indent=2) + "\n")
    scores = []
    for model, cal in [("existing", baseline), ("candidate", c)]:
        scores.extend(dict(model=model, **evaluate(cal, row)) for row in rows)
    with (args.out / "per_pose.csv").open("w") as out:
        writer = csv.DictWriter(out, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)
    summary = dict(
        fit_poses=len(fit),
        holdout_poses=len(holdout),
        orientation_groups=sorted(set(r["group"] for r in rows)),
        candidate_baseline_mm=result["baseline_mm"],
        candidate_rms_px=rms,
        active_calibration_changed=False,
        measurements={},
    )
    for model in ("existing", "candidate"):
        held = [s for s in scores if s["model"] == model and s["role"] == "holdout"]
        summary["measurements"][model] = {
            key: dict(
                median=float(np.median([s[key] for s in held])),
                max=float(max(s[key] for s in held)),
            )
            for key in (
                "vertical_abs_median_px",
                "vertical_abs_p95_px",
                "grid_edge_abs_error_p95_mm",
            )
        }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
