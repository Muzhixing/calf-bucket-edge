#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local WebRTC sender (simulates board):
  - captures webcam via OpenCV
  - sends video to WebRTC server with WebSocket signaling

Usage:
  python tools/webrtc_sender.py \
    --signal ws://172.16.8.103:8080/ws \
    --device-id robot-001
"""

import argparse
import asyncio
import json
import time

import cv2
import numpy as np

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc import VideoStreamTrack
from av import VideoFrame
import websockets


class CameraTrack(VideoStreamTrack):
    def __init__(self, device=0, width=640, height=480, fps=15):
        super().__init__()
        self._cap = cv2.VideoCapture(device)
        if width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._fps = max(1.0, float(fps))
        self._frame_interval = 1.0 / self._fps
        self._next_time = None
        self._pts = 0

    async def recv(self):
        if self._next_time is None:
            self._next_time = time.monotonic()
        else:
            self._next_time += self._frame_interval
            delay = self._next_time - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)

        ok, frame = self._cap.read()
        if not ok:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        frame = np.ascontiguousarray(frame)
        video = VideoFrame.from_ndarray(frame, format="bgr24")
        self._pts += int(self._frame_interval * 90000)
        video.pts = self._pts
        video.time_base = (1, 90000)
        return video


async def run(signal_url, device_id, fps, width, height, camera_index, ice_servers):
    ws_url = signal_url
    if device_id:
        ws_url = f"{signal_url}{'&' if '?' in signal_url else '?'}deviceId={device_id}"

    ice = None
    if ice_servers:
        ice = RTCConfiguration(iceServers=[RTCIceServer(urls=ice_servers)])

    async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as ws:
        pc = RTCPeerConnection(configuration=ice)
        pc.addTrack(CameraTrack(device=camera_index, width=width, height=height, fps=fps))

        @pc.on("icecandidate")
        async def on_icecandidate(candidate):
            if candidate is None:
                return
            await ws.send(json.dumps({
                "type": "candidate",
                "candidate": candidate.to_sdp(),
                "sdpMid": candidate.sdpMid,
                "sdpMLineIndex": candidate.sdpMLineIndex,
            }))

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        await ws.send(json.dumps({"type": "offer", "sdp": pc.localDescription.sdp}))
        print(f"Connected signaling: {ws_url}")

        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "answer":
                await pc.setRemoteDescription(
                    RTCSessionDescription(sdp=msg.get("sdp", ""), type="answer")
                )
            elif msg.get("type") == "candidate":
                await pc.addIceCandidate({
                    "candidate": msg.get("candidate"),
                    "sdpMid": msg.get("sdpMid"),
                    "sdpMLineIndex": msg.get("sdpMLineIndex"),
                })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal", required=True, help="ws://host:port/ws")
    parser.add_argument("--device-id", default="robot-001")
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--ice", default="", help="comma separated ICE urls")
    args = parser.parse_args()

    ice_servers = [s.strip() for s in args.ice.split(",") if s.strip()]
    asyncio.run(run(
        signal_url=args.signal,
        device_id=args.device_id,
        fps=args.fps,
        width=args.width,
        height=args.height,
        camera_index=args.camera,
        ice_servers=ice_servers
    ))


if __name__ == "__main__":
    main()
