#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Asynchronous in-process event bus."""

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Type


_STOP = object()


class AsyncEventBus:
    """Dispatch events asynchronously to subscribed handlers."""

    def __init__(self, max_queue=64, worker_threads=4):
        self._queue = queue.Queue(maxsize=max_queue)
        self._subscribers: Dict[Type, List[Callable]] = {}
        self._stop = threading.Event()
        self._dispatcher = None
        self._executor = ThreadPoolExecutor(max_workers=worker_threads, thread_name_prefix="event-bus")
        self._running = False

    def start(self):
        if self._running:
            return
        self._stop.clear()
        self._dispatcher = threading.Thread(target=self._run, name="event-bus-dispatcher", daemon=True)
        self._dispatcher.start()
        self._running = True

    def stop(self, timeout=1.0):
        if not self._running:
            return
        self._running = False
        self._stop.set()
        try:
            self._queue.put_nowait(_STOP)
        except queue.Full:
            pass
        if self._dispatcher is not None:
            self._dispatcher.join(timeout=timeout)
        self._executor.shutdown(wait=False)

    def subscribe(self, event_type: Type, handler: Callable):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    def publish(self, event):
        if self._stop.is_set():
            return
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            try:
                _ = self._queue.get_nowait()
                self._queue.put_nowait(event)
            except queue.Empty:
                pass

    def _run(self):
        while not self._stop.is_set():
            try:
                event = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if event is _STOP:
                continue
            for event_type, handlers in self._subscribers.items():
                if isinstance(event, event_type):
                    for handler in handlers:
                        self._executor.submit(handler, event)
