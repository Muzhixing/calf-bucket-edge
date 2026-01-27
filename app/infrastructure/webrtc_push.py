#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebRTC push client for sending video + metadata to a server.

Signaling protocol (WebSocket JSON):
  - offer: {"type":"offer","sdp":"..."}
  - answer: {"type":"answer","sdp":"..."}
  - candidate: {"type":"candidate","candidate":"...","sdpMid":"0","sdpMLineIndex":0}
"""

import asyncio
import json
import threading
import time
from fractions import Fraction

import numpy as np

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, RTCIceCandidate
    from aiortc import VideoStreamTrack
    from av import VideoFrame
except Exception as exc:  # pragma: no cover - optional dependency
    raise ImportError("aiortc/av not available, install aiortc and av to use WebRTC push") from exc

try:
    import websockets
except Exception as exc:  # pragma: no cover - optional dependency
    raise ImportError("websockets not available, install websockets to use WebRTC push") from exc


class LatestFrameTrack(VideoStreamTrack):
    def __init__(self, service, fps=8):
        super().__init__()
        self._service = service
        self._fps = max(1.0, float(fps))
        self._frame_interval = 1.0 / self._fps
        self._next_time = None
        self._timestamp = 0
        self._time_base = Fraction(1, 90000)
        self._fallback_shape = (int(service.frame_height), int(service.frame_width // 2), 3)

    async def recv(self):
        if self._next_time is None:
            self._next_time = time.monotonic()
        else:
            self._next_time += self._frame_interval
            delay = self._next_time - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)

        frame = self._service.get_latest_left_frame()
        if frame is None:
            frame = np.zeros(self._fallback_shape, dtype=np.uint8)
        else:
            self._fallback_shape = frame.shape

        frame = np.ascontiguousarray(frame)
        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        self._timestamp += int(self._frame_interval * 90000)
        video_frame.pts = self._timestamp
        video_frame.time_base = self._time_base
        return video_frame


class WebRTCPushClient:
    def __init__(self, service, signal_url, fps=8, stun_urls=None, meta_channel="meta", log_prefix="WebRTC"):
        self._service = service
        self._signal_url = signal_url
        self._fps = fps
        self._stun_urls = [u for u in (stun_urls or []) if u]
        self._meta_channel = meta_channel or "meta"
        self._log_prefix = log_prefix

        self._loop = None
        self._thread = None
        self._stop_event = threading.Event()
        self._meta_queue = None
        self._pc = None
        self._ws = None
        self._data_channel = None

    def start(self):
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._loop is not None:
            fut = asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
            try:
                fut.result(timeout=2.0)
            except Exception:
                pass

    def send_meta(self, payload):
        if self._loop is None or self._meta_queue is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._meta_queue.put(payload), self._loop)
        except Exception:
            pass

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._meta_queue = asyncio.Queue()
        self._loop.create_task(self._connect_loop())
        self._loop.run_forever()

    async def _shutdown(self):
        try:
            if self._ws is not None:
                await self._ws.close()
        finally:
            if self._pc is not None:
                await self._pc.close()
        if self._loop is not None:
            self._loop.stop()

    async def _connect_loop(self):
        backoff = 1.0
        while not self._stop_event.is_set():
            try:
                await self._connect_once()
                backoff = 1.0
            except Exception as exc:
                print(f"[{self._log_prefix}] signaling/connection error: {exc}")
            if self._stop_event.is_set():
                break
            await asyncio.sleep(backoff)
            backoff = min(10.0, backoff * 2.0)

    async def _connect_once(self):
        async with websockets.connect(self._signal_url, ping_interval=20, ping_timeout=20) as ws:
            self._ws = ws
            config = None
            if self._stun_urls:
                config = RTCConfiguration(iceServers=[RTCIceServer(urls=self._stun_urls)])
            self._pc = RTCPeerConnection(configuration=config)

            track = LatestFrameTrack(self._service, fps=self._fps)
            self._pc.addTrack(track)

            self._data_channel = self._pc.createDataChannel(self._meta_channel)
            self._data_channel.on("open", self._on_channel_open)

            @self._pc.on("icecandidate")
            async def on_icecandidate(candidate):
                if candidate is None:
                    return
                msg = {
                    "type": "candidate",
                    "candidate": candidate.to_sdp(),
                    "sdpMid": candidate.sdpMid,
                    "sdpMLineIndex": candidate.sdpMLineIndex,
                }
                await ws.send(json.dumps(msg, ensure_ascii=True))

            offer = await self._pc.createOffer()
            await self._pc.setLocalDescription(offer)
            await ws.send(json.dumps({
                "type": "offer",
                "sdp": self._pc.localDescription.sdp
            }, ensure_ascii=True))

            async for raw in ws:
                msg = json.loads(raw)
                msg_type = msg.get("type")
                if msg_type == "answer":
                    await self._pc.setRemoteDescription(
                        RTCSessionDescription(sdp=msg.get("sdp", ""), type="answer")
                    )
                elif msg_type == "candidate":
                    candidate = RTCIceCandidate(
                        sdpMid=msg.get("sdpMid"),
                        sdpMLineIndex=msg.get("sdpMLineIndex"),
                        candidate=msg.get("candidate")
                    )
                    await self._pc.addIceCandidate(candidate)
                elif msg_type == "ping":
                    await ws.send(json.dumps({"type": "pong"}, ensure_ascii=True))

    def _on_channel_open(self):
        if self._loop is None:
            return
        self._loop.create_task(self._meta_sender())

    async def _meta_sender(self):
        while not self._stop_event.is_set():
            payload = await self._meta_queue.get()
            if self._data_channel is None or self._data_channel.readyState != "open":
                continue
            try:
                self._data_channel.send(json.dumps(payload, ensure_ascii=True))
            except Exception:
                pass
