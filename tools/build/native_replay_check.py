"""Exercise the built native HTTP process with synthetic replay, without devices."""

import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

import cv2
import numpy as np
from bucket_robot.perception.capture import NativeCapture


def unused_port():
    for _ in range(30):
        with socket.socket() as first:
            first.bind(("127.0.0.1", 0))
            port = first.getsockname()[1]
            try:
                with socket.socket() as second:
                    second.bind(("127.0.0.1", port + 1))
                return port
            except OSError:
                pass
    raise RuntimeError("no adjacent local ports")


def check_application(binary, root, calibration, digest):
    port = unused_port()
    command = [
        sys.executable,
        "-m",
        "bucket_robot",
        "track",
        "--observe-only",
        "--host",
        "127.0.0.1",
        "--native",
        str(binary.resolve()),
        "--port",
        str(port),
        "--seconds",
        "4",
        "--serial-port",
        str(root / "absent-uart"),
        "--output",
        str(root / "application"),
        "--replay",
        str(root / "replay.txt"),
        "--model",
        str(root / "invalid.rknn"),
        "--calibration",
        str(calibration),
        "--calibration-sha",
        digest,
        "--assets",
        str(root),
    ]
    with (root / "application.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 6
            available = False
            while time.monotonic() < deadline and process.poll() is None:
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/status", timeout=0.5
                    ) as response:
                        status = json.load(response)
                    available = status["health"]["capabilities"]["preview"]["available"]
                    if available:
                        assert not status["health"]["capabilities"]["detection"]["available"]
                        assert not status["tracking"]["motion_available"]
                        assert (
                            status["health"]["components"]["arm"]["code"] == "DEVICE_STATE_UNKNOWN"
                        )
                        with urllib.request.urlopen(
                            f"http://127.0.0.1:{port}/frame.jpg?remote=1", timeout=0.5
                        ) as frame:
                            assert len(frame.read()) > 100
                        (root / "application_status.json").write_text(
                            json.dumps(status, ensure_ascii=False, indent=2) + "\n"
                        )
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(0.03)
            assert available, "real native process must supply video through the Python shell"
            assert list((root / "application").glob("native_*/binding.json"))
            assert process.wait(timeout=8) == 0
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=8)
    return dict(
        case="application_shell_model_failure",
        preview=True,
        detection=False,
        motion=False,
        controller_state="unknown",
        npu_inference=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    image = np.full((720, 2560, 3), 80, np.uint8)
    image[:, 1280:] = 120
    assert cv2.imwrite(str(root / "synthetic.png"), image)
    (root / "replay.txt").write_text(str(root / "synthetic.png") + "\n")
    calibration = root / "calibration.yml"
    storage = cv2.FileStorage(str(calibration), cv2.FILE_STORAGE_WRITE)
    try:
        k = np.array([[700.0, 0, 640], [0, 700, 360], [0, 0, 1.0]])
        for name in ("K1", "K2"):
            storage.write(name, k)
        for name in ("D1", "D2"):
            storage.write(name, np.zeros((1, 5)))
        for name in ("R1", "R2"):
            storage.write(name, np.eye(3))
        storage.write("P1", np.column_stack((k, np.zeros(3))))
        storage.write("P2", np.column_stack((k, [-70000.0, 0, 0])))
        storage.write("T", np.array([[100.0], [0], [0]]))
    finally:
        storage.release()
    (root / "invalid.rknn").write_bytes(b"not a model: hash must fail before RKNN init")
    digest = hashlib.sha256(calibration.read_bytes()).hexdigest()
    outcomes = []
    for mode in ("missing_calibration", "bad_calibration_hash", "bad_model"):
        port = unused_port()
        command = [
            str(args.native.resolve()),
            "--replay",
            str(root / "replay.txt"),
            "--output",
            str(root / mode),
            "--assets",
            str(root),
            "--port",
            str(port),
            "--seconds",
            "3",
            "--model",
            str(root / "invalid.rknn"),
            "--calibration",
            str(calibration if mode != "missing_calibration" else root / "missing.yml"),
            "--calibration-sha",
            digest if mode != "bad_calibration_hash" else "0" * 64,
        ]
        if mode != "bad_model":
            command.append("--disable-model")
        with (root / (mode + ".log")).open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                capture = NativeCapture(port)
                stereo = NativeCapture(port, stereo=True)
                frames = []
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        break
                    try:
                        pixels, meta = capture.read()
                        assert pixels.shape == (720, 1280, 3)
                        detector = meta["components"]["detector"]
                        calibrated = meta["components"]["calibration"]["state"] == "ready"
                        assert calibrated == (mode == "bad_model")
                        assert not meta["detections"]
                        if detector["state"] == "unavailable":
                            frames.append(meta["frame_id"])
                        if len(frames) >= 3:
                            pair, _ = stereo.read()
                            assert pair.shape == (720, 2560, 3)
                            assert np.array_equal(pair, image)
                            break
                    except (OSError, RuntimeError):
                        time.sleep(0.02)
                assert len(set(frames)) >= 3, (mode, frames)
                assert process.wait(timeout=6) == 0, mode
                outcomes.append(
                    dict(
                        case=mode,
                        fresh_frames=len(set(frames)),
                        raw_stereo_preserved=True,
                        npu_inference=False,
                    )
                )
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=2)
    outcomes.append(check_application(args.native, root, calibration, digest))
    (root / "result.json").write_text(json.dumps(outcomes, indent=2) + "\n")
    print(json.dumps(outcomes, indent=2))


if __name__ == "__main__":
    main()
