"""Latest-frame HTTP adapters with one set of frame/age/shape validation."""

import json
import time
import urllib.request

import numpy as np


class NativeCapture:
    """Read the newest rectified left frame, paired with its own metadata."""

    def __init__(self, port, *, stereo=False, max_age=0.4):
        self.stereo = stereo
        self.max_age = max_age
        endpoint = "capture.sbs.bgr" if stereo else "capture.bgr"
        self.url = f"http://127.0.0.1:{port}/{endpoint}"
        self.last_id = -1

    def read(self):
        width = 2560 if self.stereo else 1280
        expected = 720 * width * 3
        with urllib.request.urlopen(
            self.url + f"?after={self.last_id}", timeout=0.5 if self.stereo else 0.25
        ) as response:
            if response.status != 200:
                raise RuntimeError("frame_not_advanced")
            # The native backend writes UTF-8 JSON into this header. Python's
            # HTTP parser preserves header bytes as Latin-1 code points.
            metadata = json.loads(
                response.headers["X-Observation"].encode("latin-1").decode("utf-8")
            )
            frame_id = int(response.headers["X-Frame-Id"])
            encoded = response.read(expected + 1)
        if frame_id != metadata["frame_id"]:
            raise RuntimeError("invalid_capture")
        if not 0 <= time.monotonic() - float(metadata["arrival"]) <= self.max_age:
            raise RuntimeError("stale_camera")
        if frame_id <= self.last_id:
            raise RuntimeError("frame_not_advanced")
        if len(encoded) != expected:
            raise RuntimeError("invalid_camera_shape")
        self.last_id = frame_id
        return np.frombuffer(encoded, np.uint8).reshape(720, width, 3), metadata
