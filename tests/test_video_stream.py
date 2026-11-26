"""Streaming never repeats a frame or publishes expired camera images."""

import http.client
from http.server import ThreadingHTTPServer
import threading
import time
import unittest
from types import SimpleNamespace

from bucket_robot.runtime.http import handler_for
from bucket_robot.runtime.video import VideoWindow


class VideoStreamTests(unittest.TestCase):
    def test_ack_window_bounds_proxy_backlog_and_releases_only_consumed_frames(self):
        window = VideoWindow()
        for seq in range(1, 11):
            window.sent(seq)
        done = threading.Event()
        stop = threading.Event()
        thread = threading.Thread(target=lambda: (window.wait(stop), done.set()))
        thread.start()
        try:
            self.assertFalse(done.wait(0.03))
            window.acknowledge(999)
            self.assertEqual(len(window.pending), 10)
            self.assertFalse(done.is_set())
            window.acknowledge(2)
            self.assertTrue(done.wait(0.5))
            self.assertEqual(list(window.pending), list(range(3, 11)))
        finally:
            stop.set()
            thread.join(0.5)

    def test_compact_stream_skips_stale_frames_and_selects_latest_profile(self):
        lock = threading.RLock()
        session = SimpleNamespace(
            lock=lock,
            condition=threading.Condition(lock),
            stop_event=threading.Event(),
            remote_jpeg=b"clear",
            smooth_jpeg=b"smooth",
            jpeg=b"full",
            remote_video_id=1,
            video_id=1,
            remote_arrival=time.monotonic() - 2,
            video_arrival=time.monotonic() - 2,
        )
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(session))
        server.daemon_threads = True
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()

        def publish():
            time.sleep(0.08)
            with session.condition:
                session.remote_video_id = session.video_id = 10
                session.remote_arrival = session.video_arrival = time.monotonic()
                session.condition.notify_all()

        publisher = threading.Thread(target=publish)
        publisher.start()
        client = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        try:
            client.request("GET", "/stream.mjpg?remote=1&profile=smooth")
            response = client.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(response.readline().strip(), b"--frame")
            headers = {}
            while True:
                line = response.readline().strip()
                if not line:
                    break
                key, value = line.decode().split(":", 1)
                headers[key] = value.strip()
            self.assertEqual(headers["X-Frame-Id"], "10")
            self.assertLess(float(headers["X-Frame-Age-Ms"]), 400)
            self.assertEqual(response.read(int(headers["Content-Length"])), b"smooth")
            with session.condition:
                session.stop_event.set()
                session.condition.notify_all()
            self.assertEqual(response.read(), b"\r\n")
        finally:
            session.stop_event.set()
            with session.condition:
                session.condition.notify_all()
            client.close()
            publisher.join(1)
            server.shutdown()
            server.server_close()
            worker.join(2)
