"""Isolated camera-only fallback for an unavailable native binary. No motor I/O."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import signal
import threading
import time


class RawFrames:
    def __init__(self):
        self.lock = threading.Lock()
        self.item = None

    def publish(self, raw):
        with self.lock:
            seq = (self.item[1]["frame_id"] + 1) if self.item else 1
            self.item = (
                raw,
                dict(
                    frame_id=seq,
                    arrival=time.monotonic(),
                    detections=[],
                    tracking_candidates=[],
                    components={
                        "detector": dict(
                            state="unavailable",
                            code="NATIVE_UNAVAILABLE",
                            message="原始画面预览，检测程序不可用",
                        ),
                        "calibration": dict(
                            state="unavailable",
                            code="RAW_PREVIEW",
                            message="未校正的原始画面，禁止定位与控制",
                        ),
                        "ranging": dict(
                            state="unavailable", code="RANGING_UNAVAILABLE", message="测距不可用"
                        ),
                    },
                ),
            )


def handler_for(frames):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            with frames.lock:
                item = frames.item
            if item is None or time.monotonic() - item[1]["arrival"] > 0.5:
                self.send_response(503)
                self.end_headers()
                return
            image, meta = item
            if self.path.split("?")[0] != "/capture.bgr":
                self.send_response(404)
                self.end_headers()
                return
            after = self.path.split("after=")[-1] if "after=" in self.path else "-1"
            try:
                advanced = meta["frame_id"] > int(after)
            except ValueError:
                advanced = False
            if not advanced:
                self.send_response(204)
                self.end_headers()
                return
            data = image[:, :1280].tobytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Observation", json.dumps(meta))
            self.send_header("X-Frame-Id", str(meta["frame_id"]))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    frames = RawFrames()
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *args: stop.set())
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(frames))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    import cv2

    camera = None
    try:
        while not stop.is_set():
            try:
                if camera is None:
                    camera = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
                    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                ok, image = camera.read()
                if not ok or image.shape != (720, 2560, 3):
                    raise RuntimeError("camera unavailable")
                frames.publish(image)
            except Exception:
                if camera is not None:
                    camera.release()
                    camera = None
                stop.wait(1)
    finally:
        if camera is not None:
            camera.release()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
