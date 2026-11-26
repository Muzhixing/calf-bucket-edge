"""Bounded MJPEG delivery; slow viewers consume the latest encoded frame."""

import socket
import threading
import time
from collections import deque
from urllib.parse import parse_qs, urlsplit


class VideoWindow:
    """At most ten unacknowledged frames per viewer, including proxy buffers."""

    def __init__(self):
        self.condition = threading.Condition()
        self.pending = deque()
        self.last_ack = time.monotonic()

    def wait(self, stop_event):
        with self.condition:
            while len(self.pending) >= 10 and not stop_event.is_set():
                if time.monotonic() - self.last_ack > 2:
                    raise TimeoutError("video acknowledgement timeout")
                self.condition.wait(0.1)

    def sent(self, seq):
        with self.condition:
            self.pending.append(seq)

    def acknowledge(self, seq):
        with self.condition:
            if self.pending and seq > self.pending[-1]:
                return
            advanced = False
            while self.pending and self.pending[0] <= seq:
                self.pending.popleft()
                advanced = True
            if advanced:
                self.last_ack = time.monotonic()
            self.condition.notify_all()


class VideoStreams:
    def __init__(self):
        self.lock = threading.Lock()
        self.viewers = {}

    def acknowledge(self, client, seq):
        with self.lock:
            window = self.viewers.get(client)
        if window:
            window.acknowledge(seq)


def stream_mjpeg(handler, session, streams):
    query = parse_qs(urlsplit(handler.path).query)
    client = query.get("client", [""])[0]
    window = VideoWindow() if client else None
    if window:
        with streams.lock:
            if len(streams.viewers) >= 16 or len(client) > 64:
                handler.respond(429, {"error": "too_many_viewers"})
                return
            streams.viewers[client] = window
    try:
        _stream_mjpeg(handler, session, query, window)
    finally:
        if window:
            with streams.lock:
                if streams.viewers.get(client) is window:
                    streams.viewers.pop(client)


def _stream_mjpeg(handler, session, query, window):
    compact = query.get("remote") == ["1"]
    smooth = query.get("profile", ["smooth"]) == ["smooth"]
    handler.connection.settimeout(1)
    handler.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16384)
    handler.send_response(200)
    handler.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("X-Accel-Buffering", "no")
    handler.send_header("Connection", "close")
    handler.end_headers()
    previous = -1
    sequence_field = "remote_video_id" if compact else "video_id"
    while not session.stop_event.is_set():
        with session.condition:
            session.condition.wait_for(
                lambda: session.stop_event.is_set() or getattr(session, sequence_field) != previous,
                timeout=0.5,
            )
            if session.stop_event.is_set():
                break
        if window:
            window.wait(session.stop_event)
            if session.stop_event.is_set():
                break
        with session.condition:
            seq = session.remote_video_id if compact else session.video_id
            stamp = session.remote_arrival if compact else session.video_arrival
            jpeg = (
                (getattr(session, "smooth_jpeg", None) if smooth else None) or session.remote_jpeg
                if compact
                else session.jpeg
            )
        age = max(0, time.monotonic() - stamp)
        if seq == previous:
            continue
        previous = seq
        if jpeg is None or age > 0.4:
            # Wait for another publication, rather than busy-looping on stale bytes.
            continue
        header = (
            f"--frame\r\nContent-Type: image/jpeg\r\nX-Frame-Id: {seq}\r\n"
            f"X-Frame-Age-Ms: {age * 1000:.1f}\r\nContent-Length: {len(jpeg)}\r\n\r\n"
        ).encode("ascii")
        if window:
            window.sent(seq)
        # One write avoids splitting small headers and images into delayed TCP packets.
        handler.wfile.write(header + jpeg + b"\r\n")
        handler.wfile.flush()
