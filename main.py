#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from app.application.app_controller import AppController
from app.application.event_bus import AsyncEventBus
from app.application.state_store import RangingStateStore
from app.infrastructure.ranging_service import RangingService
from app.infrastructure.serial_port import SerialPortGateway


def main():
    import os
    from pathlib import Path
    BASE = Path(__file__).resolve().parent
    os.chdir(BASE)
    print("CWD =", os.getcwd())

    enable_push = os.getenv("ENABLE_PUSH", "0") == "1"
    webrtc_signal_url = os.getenv("WEBRTC_SIGNAL_URL")
    webrtc_stun_urls = os.getenv("WEBRTC_STUN_URLS")
    push_fps = float(os.getenv("PUSH_FPS", "8"))
    event_bus = AsyncEventBus(max_queue=8, worker_threads=4)
    state_store = RangingStateStore()
    serial_gateway = SerialPortGateway(enabled=False)
    service = RangingService(
        enable_push=enable_push,
        webrtc_signal_url=webrtc_signal_url,
        webrtc_stun_urls=webrtc_stun_urls,
        push_fps=push_fps,
        render_on_device=False,
        event_bus=event_bus
    )
    controller = AppController(service, event_bus, state_store, serial_gateway=serial_gateway)
    controller.run()


if __name__ == '__main__':
    main()
