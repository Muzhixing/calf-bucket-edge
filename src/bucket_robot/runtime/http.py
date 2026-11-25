"""HTTP adapter; UI and health do not import camera or motion dependencies."""

from http.server import BaseHTTPRequestHandler
import json, secrets, time
from pathlib import Path
from urllib.parse import urlsplit


def handler_for(session):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def respond(
            self, code, body, content_type="application/json", extra=None, keep_alive=False
        ):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            keep_alive = keep_alive and self.headers.get("Connection", "").lower() != "close"
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive" if keep_alive else "close")
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, str(v))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
            self.close_connection = not keep_alive

        def do_GET(self):
            try:
                self.connection.settimeout(3)
                path = urlsplit(self.path).path
                if path == "/":
                    try:
                        html = (Path(__file__).parent / "index.html").read_text()
                    except OSError:
                        html = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>设备状态</title><h1>界面资源不可用</h1><p>请修复界面资源。状态服务仍在运行。</p><pre id="health"></pre><script>setInterval(async()=>{document.getElementById("health").textContent=JSON.stringify(await(await fetch("/health")).json(),null,2)},1000)</script></html>'
                    html = html.replace("__CONTROL_TOKEN__", session.token)
                    self.respond(200, html.encode(), "text/html; charset=utf-8")
                elif path in ("/health", "/ready"):
                    health = session.status().get("health", {})
                    available = (
                        health.get("capabilities", {}).get("preview", {}).get("available", False)
                    )
                    self.respond(200 if path == "/health" or available else 503, health)
                elif path == "/status":
                    self.respond(200, session.status(), keep_alive=True)
                elif path == "/snapshot.jpg":
                    with session.lock:
                        source = (
                            session.preview_image
                            if session.preview_image is not None
                            else session.image
                        )
                        meta = session.preview_metadata or session.metadata
                        if source is None or time.monotonic() - meta.get("arrival", 0) > 0.4:
                            self.respond(503, {"error": "画面尚未就绪"})
                            return
                        image = source.copy()
                        seq = meta["frame_id"]
                        session.snapshots.append((seq, meta["arrival"], image))
                    import cv2

                    _, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    self.respond(200, encoded.tobytes(), "image/jpeg", {"X-Frame-Id": seq})
                elif path == "/frame.jpg":
                    with session.lock:
                        compact = urlsplit(self.path).query == "remote=1"
                        jpeg, seq, stamp = (
                            (session.remote_jpeg, session.remote_video_id, session.remote_arrival)
                            if compact
                            else (session.jpeg, session.video_id, session.video_arrival)
                        )
                        age = max(0, time.monotonic() - stamp)
                    fresh = jpeg is not None and age <= 0.4
                    self.respond(
                        200 if fresh else 503,
                        jpeg if fresh else b"",
                        "image/jpeg",
                        {"X-Frame-Id": seq, "X-Frame-Age-Ms": round(age * 1000, 1)},
                        keep_alive=True,
                    )
                elif path == "/stream.mjpg":
                    self.connection.settimeout(2)
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    previous = -1
                    while not session.stop_event.is_set():
                        with session.condition:
                            session.condition.wait_for(
                                lambda: session.stop_event.is_set() or session.video_id != previous,
                                timeout=0.5,
                            )
                            jpeg, seq = session.jpeg, session.video_id
                        if jpeg is None or seq == previous:
                            continue
                        previous = seq
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\nX-Frame-Id: "
                            + str(seq).encode()
                            + b"\r\nContent-Length: "
                            + str(len(jpeg)).encode()
                            + b"\r\n\r\n"
                            + jpeg
                            + b"\r\n"
                        )
                        self.wfile.flush()
                else:
                    self.respond(404, {"error": "not_found"})
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
            except Exception as exc:
                session.health.report(
                    "application", "degraded", "REQUEST_FAILED", str(exc)
                ) if getattr(session, "health", None) else None
                self.respond(503, {"error": "请求暂不可用", "code": "REQUEST_FAILED"})

        def do_POST(self):
            self.connection.settimeout(2)
            try:
                if not secrets.compare_digest(
                    self.headers.get("X-Control-Token", ""), session.token
                ):
                    self.respond(403, {"error": "invalid_control_token"})
                    return
                if self.headers.get("Sec-Fetch-Site") == "cross-site":
                    self.respond(403, {"error": "cross_site_control_rejected"})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 4096:
                    raise ValueError("request_too_large")
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("request_body_must_be_object")
                path = urlsplit(self.path).path
                if not path.startswith("/api/"):
                    raise ValueError("unknown_action")
                session.command(path[5:], body)
                self.respond(200, {"accepted": True, "tracking": session.status()["tracking"]})
            except (ValueError, TypeError, KeyError) as exc:
                self.respond(400, {"error": str(exc)})
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
            except Exception as exc:
                session.health.report(
                    "application", "degraded", "REQUEST_FAILED", str(exc)
                ) if getattr(session, "health", None) else None
                self.respond(503, {"error": "请求暂不可用", "code": "REQUEST_FAILED"})

    return Handler
