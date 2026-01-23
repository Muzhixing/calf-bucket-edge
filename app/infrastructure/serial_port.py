#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Serial port gateway placeholder."""

from app.domain.events import RangingUpdateEvent


class SerialPortGateway:
    """Placeholder for serial output layer.

    This class is intentionally minimal and can be expanded later to
    encode and transmit ranging results over a serial interface.
    """

    def __init__(self, port=None, baudrate=115200, enabled=False):
        self.port = port
        self.baudrate = baudrate
        self.enabled = enabled

    def start(self):
        # Placeholder: initialize serial resources here.
        pass

    def stop(self):
        # Placeholder: release serial resources here.
        pass

    def handle_ranging_update(self, event: RangingUpdateEvent):
        if not self.enabled:
            return
        # Placeholder: serialize and write event data to serial port.
        _ = event
