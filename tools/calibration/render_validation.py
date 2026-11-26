"""Render the same held-out raw pair with old and candidate rectification."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from solve import array_calibration, SIZE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--captures", type=Path, required=True)
    p.add_argument("--solution", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    args = p.parse_args()
    rows = list(csv.DictReader((args.solution / "per_pose.csv").open()))
    worst = max(
        (r for r in rows if r["role"] == "holdout" and r["model"] == "existing"),
        key=lambda r: float(r["vertical_abs_p95_px"]),
    )
    pose = args.captures / worst["id"]
    manifest = json.loads((pose / "capture.json").read_text())
    index = manifest["frames"][0]["index"]
    views = []
    for name, path in [
        ("existing", args.baseline),
        ("candidate", args.solution / "candidate.json"),
    ]:
        c = array_calibration(json.loads(path.read_text()))
        eyes = []
        for i, eye in enumerate(("left", "right"), 1):
            raw = cv2.imread(str(pose / f"{index:03d}_{eye}.png"))
            if raw is None or raw.shape[:2] != (720, 1280):
                raise ValueError("Missing raw image")
            maps = cv2.initUndistortRectifyMap(
                c[f"K{i}"], c[f"D{i}"], c[f"R{i}"], c[f"P{i}"], SIZE, cv2.CV_32FC1
            )
            im = cv2.remap(raw, *maps, cv2.INTER_LINEAR)
            points = cv2.undistortPoints(
                np.array(manifest["frames"][0]["eyes"][eye]["corners_px"], np.float32).reshape(
                    -1, 1, 2
                ),
                c[f"K{i}"],
                c[f"D{i}"],
                R=c[f"R{i}"],
                P=c[f"P{i}"],
            )
            cv2.drawChessboardCorners(im, (9, 6), points, True)
            for y in range(60, 720, 60):
                cv2.line(im, (0, y), (1279, y), (0, 220, 255), 1)
            eyes.append(im)
        line = next(r for r in rows if r["model"] == name and r["id"] == worst["id"])
        view = cv2.resize(np.hstack(eyes), (1536, 432))
        view[:42] = 0
        text = f"{name.upper()} | held-out pair | vertical abs p95: {float(line['vertical_abs_p95_px']):.3f} px | 30 mm edge error p95: {float(line['grid_edge_abs_error_p95_mm']):.3f} mm"
        cv2.putText(
            view, text, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA
        )
        views.append(view)
    output = args.solution / "rectification_comparison.jpg"
    if not cv2.imwrite(str(output), np.vstack(views)):
        raise RuntimeError("Cannot save comparison")
    (args.solution / "representative_pose.json").write_text(
        json.dumps(
            {
                "pose": worst["id"],
                "selection": "largest existing-calibration held-out vertical p95; same raw pair in both panels",
            },
            indent=2,
        )
        + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
