#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebRTC push client for sending video + metadata to a server.

Signaling protocol (WebSocket JSON):
  - offer: {"to":"browser","type":"offer","sdp":"..."}
  - answer: {"to":"python","type":"answer","sdp":"..."}
  - candidate: {"to":"browser","type":"candidate","candidate":"...","sdpMid":"0","sdpMLineIndex":0}
"""

import asyncio
import json
import threading
import time
from fractions import Fraction

import cv2
import numpy as np

try:
    from aiortc import (
        RTCPeerConnection,
        RTCSessionDescription,
        RTCConfiguration,
        RTCIceServer,
    )
    from aiortc import VideoStreamTrack
    from aiortc.sdp import candidate_from_sdp, candidate_to_sdp
    from av import VideoFrame
except Exception as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "aiortc/av not available, install aiortc and av to use WebRTC push"
    ) from exc

try:
    import websockets
except Exception as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "websockets not available, install websockets to use WebRTC push"
    ) from exc


class LatestFrameTrack(VideoStreamTrack):
    def __init__(self, service, fps=8):
        super().__init__()
        self._service = service
        self._fps = max(1.0, float(fps))
        self._frame_interval = 1.0 / self._fps
        self._next_time = None
        self._timestamp = 0
        self._time_base = Fraction(1, 90000)
        self._fallback_shape = (
            int(service.frame_height),
            int(service.frame_width // 2),
            3,
        )

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
    def __init__(
        self,
        service,
        signal_url,
        fps=8,
        stun_urls=None,
        meta_channel="meta",
        peer_id="browser",
        from_id=None,
        log_prefix="WebRTC",
        answer_timeout=8.0,
        disconnected_timeout=3.0,
    ):
        self._service = service
        self._signal_url = signal_url
        self._fps = fps
        self._stun_urls = [u for u in (stun_urls or []) if u]
        self._meta_channel = meta_channel or "meta"
        self._peer_id = peer_id
        self._from_id = from_id
        self._log_prefix = log_prefix
        self._answer_timeout = max(2.0, float(answer_timeout))
        self._disconnected_timeout = max(1.0, float(disconnected_timeout))

        self._loop = None
        self._thread = None
        self._stop_event = threading.Event()
        self._ws = None
        self._peers = {}
        self._frame_push_mode = "/sfu/push/" in (signal_url or "")

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
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._loop = None
        self._peers = {}

    def send_meta(self, payload):
        if self._loop is None:
            return
        try:
            if self._frame_push_mode:
                asyncio.run_coroutine_threadsafe(
                    self._send_frame_push_meta(payload), self._loop
                )
            else:
                asyncio.run_coroutine_threadsafe(
                    self._broadcast_meta(payload), self._loop
                )
        except Exception:
            pass

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.create_task(self._connect_loop())
        self._loop.run_forever()

    async def _shutdown(self):
        try:
            if self._ws is not None:
                await self._ws.close()
        finally:
            await self._close_all_peers()
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
        if self._frame_push_mode:
            await self._frame_push_once()
            return

        async with websockets.connect(
            self._signal_url, ping_interval=20, ping_timeout=20
        ) as ws:
            self._ws = ws
            restart_event = asyncio.Event()
            offer_task = None

            def request_restart(reason):
                if restart_event.is_set() or self._stop_event.is_set():
                    return
                print(f"[{self._log_prefix}] reconnecting signaling: {reason}")
                restart_event.set()
                asyncio.create_task(ws.close())

            print(f"[{self._log_prefix}] signaling connected, starting offer loop")
            if self._peer_id:
                offer_task = asyncio.create_task(
                    self._offer_retry_loop(self._peer_id, ws)
                )
            try:
                async for raw in ws:
                    msg = json.loads(raw)
                    msg_type = msg.get("type")
                    peer_id = msg.get("from") or msg.get("peerId") or self._peer_id
                    if msg_type in {"watch", "join", "connect"}:
                        await self._create_peer(peer_id, ws)
                    elif msg_type == "answer":
                        peer = self._peers.get(peer_id)
                        if peer is None:
                            print(
                                f"[{self._log_prefix}] answer ignored, unknown peer: {peer_id}"
                            )
                            continue
                        await peer["pc"].setRemoteDescription(
                            RTCSessionDescription(sdp=msg.get("sdp", ""), type="answer")
                        )
                        peer["answer_received"] = True
                        print(f"[{self._log_prefix}] answer received: {peer_id}")
                    elif msg_type == "candidate":
                        peer = self._peers.get(peer_id)
                        if peer is not None:
                            candidate = self._candidate_from_message(msg)
                            await peer["pc"].addIceCandidate(candidate)
                    elif msg_type in {"leave", "close"}:
                        await self._close_peer(peer_id)
                    elif msg_type == "ping":
                        await self._send_signal(ws, {"type": "pong"})
            finally:
                if offer_task is not None:
                    offer_task.cancel()
                await self._close_all_peers()
                if self._ws is ws:
                    self._ws = None

    async def _offer_retry_loop(self, peer_id, ws):
        """Keep sending offers until a browser answers.

        The 5173 browser page waits for board offers after the WebSocket opens.
        If the board connected before the browser, the first offer can be lost by
        the relay. Retrying while no peer is active keeps the workflow tolerant
        to either startup order.
        """
        retry_interval = max(2.0, self._answer_timeout + 1.0)
        while not self._stop_event.is_set():
            if getattr(ws, "closed", False):
                return
            peer = self._peers.get(peer_id)
            if peer is None:
                try:
                    await self._create_peer(peer_id, ws)
                except Exception as exc:
                    print(f"[{self._log_prefix}] offer retry failed: {exc}")
            await asyncio.sleep(retry_interval)

    async def _frame_push_once(self):
        interval = 1.0 / max(1.0, float(self._fps))
        async with websockets.connect(
            self._signal_url,
            ping_interval=20,
            ping_timeout=20,
            max_size=8 * 1024 * 1024,
        ) as ws:
            self._ws = ws
            print(f"[{self._log_prefix}] frame relay connected")
            try:
                while not self._stop_event.is_set():
                    started_at = time.monotonic()
                    frame = self._service.get_latest_left_frame()
                    if frame is not None:
                        frame = np.ascontiguousarray(frame)
                        ok, encoded = cv2.imencode(
                            ".jpg",
                            frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 80],
                        )
                        if ok:
                            await ws.send(encoded.tobytes())
                    elapsed = time.monotonic() - started_at
                    await asyncio.sleep(max(0.0, interval - elapsed))
            finally:
                if self._ws is ws:
                    self._ws = None

    async def _send_frame_push_meta(self, payload):
        ws = self._ws
        if ws is None:
            return
        try:
            await ws.send(json.dumps(payload, ensure_ascii=True))
        except Exception:
            pass

    async def _create_peer(self, peer_id, ws):
        if not peer_id:
            return
        await self._close_peer(peer_id)

        config = None
        if self._stun_urls:
            config = RTCConfiguration(iceServers=[RTCIceServer(urls=self._stun_urls)])
        pc = RTCPeerConnection(configuration=config)
        track = LatestFrameTrack(self._service, fps=self._fps)
        pc.addTrack(track)
        data_channel = pc.createDataChannel(self._meta_channel)
        peer = {
            "pc": pc,
            "data_channel": data_channel,
            "answer_received": False,
            "watchdog_task": None,
        }
        self._peers[peer_id] = peer

        data_channel.on(
            "open", lambda: print(f"[{self._log_prefix}] data channel open: {peer_id}")
        )
        data_channel.on("close", lambda: asyncio.create_task(self._close_peer(peer_id)))

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            state = pc.connectionState
            if state == "connected":
                print(f"[{self._log_prefix}] peer connected: {peer_id}")
            elif state in {"failed", "closed"}:
                await self._close_peer(peer_id)
            elif state == "disconnected":
                asyncio.create_task(
                    self._restart_if_still_disconnected(
                        peer_id, pc, f"peer disconnected: {peer_id}"
                    )
                )

        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            state = pc.iceConnectionState
            if state in {"failed", "closed"}:
                await self._close_peer(peer_id)
            elif state == "disconnected":
                asyncio.create_task(
                    self._restart_if_still_disconnected(
                        peer_id, pc, f"ice disconnected: {peer_id}"
                    )
                )

        @pc.on("icecandidate")
        async def on_icecandidate(candidate):
            if candidate is None:
                return
            candidate_sdp = candidate_to_sdp(candidate)
            if not candidate_sdp.startswith("candidate:"):
                candidate_sdp = "candidate:" + candidate_sdp
            await self._send_signal(
                ws,
                {
                    "type": "candidate",
                    "candidate": candidate_sdp,
                    "sdpMid": candidate.sdpMid,
                    "sdpMLineIndex": candidate.sdpMLineIndex,
                    "to": peer_id,
                },
            )

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        await self._send_signal(
            ws,
            {
                "type": "offer",
                "sdp": pc.localDescription.sdp,
                "to": peer_id,
            },
        )
        print(f"[{self._log_prefix}] offer sent: {peer_id}")
        peer["watchdog_task"] = asyncio.create_task(self._answer_watchdog(peer_id, pc))

    async def _answer_watchdog(self, peer_id, pc):
        started_at = time.monotonic()
        while not self._stop_event.is_set() and pc.connectionState not in {
            "closed",
            "failed",
        }:
            await asyncio.sleep(1.0)
            peer = self._peers.get(peer_id)
            if peer is None:
                return
            if peer.get("answer_received"):
                return
            if time.monotonic() - started_at >= self._answer_timeout:
                print(f"[{self._log_prefix}] answer timeout: {peer_id}")
                await self._close_peer(peer_id)
                return

    async def _restart_if_still_disconnected(self, peer_id, pc, reason):
        await asyncio.sleep(self._disconnected_timeout)
        if (
            pc.connectionState == "disconnected"
            or pc.iceConnectionState == "disconnected"
        ):
            print(f"[{self._log_prefix}] closing peer: {reason}")
            await self._close_peer(peer_id)

    async def _close_peer(self, peer_id):
        peer = self._peers.pop(peer_id, None)
        if peer is None:
            return
        task = peer.get("watchdog_task")
        if task is not None:
            task.cancel()
        pc = peer.get("pc")
        if pc is not None:
            await pc.close()

    async def _close_all_peers(self):
        for peer_id in list(self._peers.keys()):
            await self._close_peer(peer_id)

    async def _broadcast_meta(self, payload):
        if not self._peers:
            return
        data = json.dumps(payload, ensure_ascii=True)
        for peer in list(self._peers.values()):
            data_channel = peer.get("data_channel")
            if data_channel is not None and data_channel.readyState == "open":
                try:
                    data_channel.send(data)
                except Exception:
                    pass

    async def _send_signal(self, ws, payload):
        await ws.send(json.dumps(self._signal_payload(payload), ensure_ascii=True))

    def _signal_payload(self, payload):
        msg = dict(payload)
        if self._peer_id:
            msg.setdefault("to", self._peer_id)
        if self._from_id:
            msg.setdefault("from", self._from_id)
        return msg

    def _candidate_from_message(self, msg):
        raw_candidate = msg.get("candidate")
        if raw_candidate and raw_candidate.startswith("candidate:"):
            raw_candidate = raw_candidate[len("candidate:") :]
        candidate = candidate_from_sdp(raw_candidate)
        candidate.sdpMid = msg.get("sdpMid")
        candidate.sdpMLineIndex = msg.get("sdpMLineIndex")
        return candidate
