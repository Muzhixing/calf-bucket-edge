"""Export a candidate for an explicit C++ validation run; no default switch."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("candidate", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.candidate.read_text())
    if data["units"] != "mm" or data["image_size"] != [1280, 720]:
        raise ValueError("Unsupported units/resolution")
    if args.out.exists():
        raise FileExistsError(args.out)
    keys = ("K1", "D1", "K2", "D2", "R1", "R2", "P1", "P2", "T")
    values = {k: np.array(data[k], np.float64) for k in keys}
    if any(not np.isfinite(v).all() for v in values.values()):
        raise ValueError("Nonfinite matrix")
    f = cv2.FileStorage(str(args.out), cv2.FILE_STORAGE_WRITE)
    if not f.isOpened():
        raise RuntimeError("Cannot open YAML output")
    for key, mat in values.items():
        f.write(key, mat)
    f.release()
    f = cv2.FileStorage(str(args.out), cv2.FILE_STORAGE_READ)
    for key, mat in values.items():
        if not np.array_equal(f.getNode(key).mat(), mat):
            raise RuntimeError("YAML round-trip mismatch")
    f.release()
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    args.out.with_suffix(".sha256").write_text(digest + "\n")
    args.out.with_suffix(".binding.json").write_text(
        json.dumps(
            dict(
                source=str(args.candidate.resolve()),
                source_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
                yaml_sha256=digest,
                active_runtime_changed=False,
            ),
            indent=2,
        )
        + "\n"
    )
    print(digest)


if __name__ == "__main__":
    main()
