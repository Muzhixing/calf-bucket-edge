#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import socket
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from app.application.app_controller import AppController
from app.application.board_config import BoardConfig, env_enabled, split_csv_env
from app.application.event_bus import AsyncEventBus
from app.application.state_store import RangingStateStore
from app.application.task_state import TaskStateStore
from app.infrastructure.ranging_service import RangingService
from app.infrastructure.bucket_tracking import BucketTrackingConfig
from app.infrastructure.serial_port import SerialPortGateway
from app.infrastructure.status_reporter import StatusReporter
from app.infrastructure.task_client import MockTaskFileClient, TaskPullClient


DEFAULT_WEBRTC_SIGNAL_CANDIDATES = ("ws://120.48.24.192:5173/ws/python",)
DEFAULT_WEBRTC_STUN_URLS = ("stun:stun.l.google.com:19302",)


def _normalize_signal_url(value):
    raw = (value or "").strip()
    if not raw:
        return None
    if "://" not in raw:
        raw = f"ws://{raw}"

    parsed = urlparse(raw)
    scheme = parsed.scheme
    if scheme == "http":
        scheme = "ws"
    elif scheme == "https":
        scheme = "wss"
    elif scheme not in {"ws", "wss"}:
        return None

    path = parsed.path
    if not path or path == "/":
        path = "/ws/python"

    return urlunparse(parsed._replace(scheme=scheme, path=path))


def _with_device_id(url, device_id):
    parsed = urlparse(url)
    query = parse_qsl(parsed.query, keep_blank_values=True)
    if not any(key in {"deviceId", "deviceID"} for key, _ in query):
        query.append(("deviceId", device_id))
    return urlunparse(parsed._replace(query=urlencode(query)))


def _tcp_reachable(url, timeout=1.5):
    parsed = urlparse(url)
    if not parsed.hostname:
        return False
    try:
        port = parsed.port or (443 if parsed.scheme == "wss" else 80)
    except ValueError:
        return False
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            return True
    except OSError:
        return False


def _signal_candidates():
    candidates = []
    candidates.extend(split_csv_env("WEBRTC_SIGNAL_CANDIDATES"))

    server_host = os.getenv("SERVER_HOST")
    if server_host:
        server_port = os.getenv("SERVER_PORT", "5173")
        if ":" in server_host:
            candidates.append(f"ws://{server_host}/ws/python")
        else:
            candidates.append(f"ws://{server_host}:{server_port}/ws/python")

    candidates.extend(DEFAULT_WEBRTC_SIGNAL_CANDIDATES)

    seen = set()
    for candidate in candidates:
        normalized = _normalize_signal_url(candidate)
        if normalized and normalized not in seen:
            seen.add(normalized)
            yield normalized


def resolve_webrtc_signal_url(device_id):
    configured_url = _normalize_signal_url(os.getenv("WEBRTC_SIGNAL_URL"))
    if configured_url:
        return _with_device_id(configured_url, device_id)

    for candidate in _signal_candidates():
        url = _with_device_id(candidate, device_id)
        if _tcp_reachable(url):
            print(f"WebRTC 自动发现服务器: {url}")
            return url
        print(f"WebRTC 候选服务器不可达: {url}")

    return None


def main():
    from pathlib import Path

    BASE = Path(__file__).resolve().parent
    os.chdir(BASE)
    print("CWD =", os.getcwd())

    board_config = BoardConfig.from_env()
    camera_device = os.getenv("CAMERA_DEVICE", "auto")
    camera_fourcc = os.getenv("CAMERA_FOURCC", "MJPG")
    camera_fps = float(os.getenv("CAMERA_FPS", "30"))
    push_fps = float(os.getenv("PUSH_FPS", "8"))

    enable_push = env_enabled("ENABLE_PUSH")
    video_push_url = os.getenv("VIDEO_PUSH_URL")
    meta_push_url = os.getenv("META_PUSH_URL")

    explicit_webrtc = os.getenv("ENABLE_WEBRTC")
    enable_webrtc = env_enabled("ENABLE_WEBRTC")
    auto_webrtc = explicit_webrtc is None and env_enabled("WEBRTC_AUTO_CONNECT", "1")
    webrtc_signal_url = (
        resolve_webrtc_signal_url(board_config.device_id)
        if (enable_webrtc or auto_webrtc)
        else None
    )
    if webrtc_signal_url:
        enable_webrtc = True
    elif auto_webrtc:
        print("WebRTC 自动连接未找到可用服务器，继续启动本地服务。")
    webrtc_peer_id = os.getenv("WEBRTC_PEER_ID", "browser")
    webrtc_fps = float(os.getenv("WEBRTC_FPS", os.getenv("PUSH_FPS", "15")))
    webrtc_stun_urls = split_csv_env("WEBRTC_STUN_URLS") or list(
        DEFAULT_WEBRTC_STUN_URLS
    )
    disparity_mode = os.getenv("DISPARITY_MODE", "full")
    disparity_interval = max(1, int(os.getenv("DISPARITY_INTERVAL", "8")))
    disparity_full_fallback = env_enabled("DISPARITY_FULL_FALLBACK", "1")
    disparity_roi_expand_ratio = float(os.getenv("DISPARITY_ROI_EXPAND_RATIO", "0.6"))
    disparity_scale = float(os.getenv("DISPARITY_SCALE", "1"))
    async_capture = env_enabled("ASYNC_CAPTURE", "0")
    async_distance = env_enabled("ASYNC_DISTANCE", "0")
    loop_sleep_seconds = float(os.getenv("LOOP_SLEEP_SECONDS", "0"))

    status_push_url = os.getenv("STATUS_PUSH_URL")
    status_push_interval = float(os.getenv("STATUS_PUSH_INTERVAL", "1"))
    task_pull_url = os.getenv("TASK_PULL_URL")
    mock_task_file = os.getenv("MOCK_TASK_FILE")
    task_poll_interval = float(os.getenv("TASK_POLL_INTERVAL", "2"))
    task_forward_mps = float(os.getenv("TASK_FORWARD_MPS", "0.2"))
    task_horizontal_mps = float(os.getenv("TASK_HORIZONTAL_MPS", "0.1"))
    task_default_step_duration = float(os.getenv("TASK_DEFAULT_STEP_DURATION", "2"))
    enable_local_web = env_enabled("ENABLE_LOCAL_WEB", "0")
    serial_port = os.getenv("SERIAL_PORT", "/dev/myserial")
    serial_baudrate = int(os.getenv("SERIAL_BAUDRATE", "115200"))
    serial_send_hz = float(os.getenv("SERIAL_SEND_HZ", "10"))
    serial_log_frames = env_enabled("SERIAL_LOG_FRAMES", "0")
    serial_command_repeats = int(os.getenv("SERIAL_COMMAND_REPEATS", "1"))
    serial_inter_frame_delay = float(os.getenv("SERIAL_INTER_FRAME_DELAY", "5.0"))
    serial_stop_repeats = int(os.getenv("SERIAL_STOP_REPEATS", "1"))
    bucket_tracking_config = BucketTrackingConfig(
        enabled=env_enabled("BUCKET_TRACKING_ENABLED", "1"),
        center_tolerance=float(os.getenv("BUCKET_CENTER_TOLERANCE", "0.08")),
        target_distance_m=float(os.getenv("BUCKET_TARGET_DISTANCE_M", "0.45")),
        distance_tolerance_m=float(os.getenv("BUCKET_DISTANCE_TOLERANCE_M", "0.05")),
        track_timeout_s=float(os.getenv("BUCKET_TRACK_TIMEOUT_S", "20")),
        lost_timeout_s=float(os.getenv("BUCKET_LOST_TIMEOUT_S", "1.0")),
        forward_speed=int(os.getenv("BUCKET_TRACK_FORWARD_SPEED", "30")),
        rudder_max_delta_deg=float(os.getenv("BUCKET_TRACK_RUDDER_MAX_DELTA", "45")),
        arm_extend_mm=int(os.getenv("BUCKET_ARM_EXTEND_MM", "180")),
        arm_hold_s=float(os.getenv("BUCKET_ARM_HOLD_S", "1.0")),
    )

    event_bus = AsyncEventBus(max_queue=8, worker_threads=4)
    state_store = RangingStateStore()
    task_state = TaskStateStore(device_id=board_config.device_id)
    serial_gateway = SerialPortGateway(
        port=serial_port,
        baudrate=serial_baudrate,
        enabled=env_enabled("ENABLE_SERIAL"),
        send_hz=serial_send_hz,
        log_frames=serial_log_frames,
        command_repeats=serial_command_repeats,
        inter_frame_delay_s=serial_inter_frame_delay,
        stop_repeats=serial_stop_repeats,
        bucket_tracking_config=bucket_tracking_config,
    )
    task_client = None
    if task_pull_url:
        task_client = TaskPullClient(
            task_pull_url=task_pull_url,
            task_state=task_state,
            serial_gateway=serial_gateway,
            device_id=board_config.device_id,
            poll_interval=task_poll_interval,
            forward_mps=task_forward_mps,
            horizontal_mps=task_horizontal_mps,
            default_step_duration=task_default_step_duration,
        )
    elif mock_task_file:
        task_client = MockTaskFileClient(
            mock_task_file=mock_task_file,
            task_state=task_state,
            serial_gateway=serial_gateway,
            device_id=board_config.device_id,
            poll_interval=task_poll_interval,
            forward_mps=task_forward_mps,
            horizontal_mps=task_horizontal_mps,
            default_step_duration=task_default_step_duration,
        )
    status_reporter = None
    if status_push_url:
        status_reporter = StatusReporter(
            status_url=status_push_url,
            board_config=board_config,
            state_store=state_store,
            task_state=task_state,
            interval=status_push_interval,
        )
    service = RangingService(
        camera_device=camera_device,
        camera_fourcc=camera_fourcc,
        camera_fps=camera_fps,
        enable_push=enable_push,
        video_push_url=video_push_url,
        meta_push_url=meta_push_url,
        push_fps=push_fps,
        render_on_device=False,
        cache_display_frame=enable_local_web,
        event_bus=event_bus,
        device_id=board_config.device_id,
        enable_webrtc=enable_webrtc,
        webrtc_signal_url=webrtc_signal_url,
        webrtc_peer_id=webrtc_peer_id,
        webrtc_fps=webrtc_fps,
        webrtc_stun_urls=webrtc_stun_urls,
        disparity_mode=disparity_mode,
        disparity_interval=disparity_interval,
        disparity_full_fallback=disparity_full_fallback,
        disparity_roi_expand_ratio=disparity_roi_expand_ratio,
        disparity_scale=disparity_scale,
        async_capture=async_capture,
        async_distance=async_distance,
        loop_sleep_seconds=loop_sleep_seconds,
    )
    controller = AppController(
        service,
        event_bus,
        state_store,
        serial_gateway=serial_gateway,
        task_client=task_client,
        status_reporter=status_reporter,
        web_enabled=enable_local_web,
    )
    controller.run()


if __name__ == "__main__":
    main()
