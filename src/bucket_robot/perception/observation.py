"""Bounded, atomic display snapshots of the target actually owned by a task.

These files are telemetry only. Control never consumes them.
"""

import json
from pathlib import Path
import time


class TaskObservationPublisher:
    def __init__(self, folder, *, task_root=None, clock=time.monotonic):
        self.folder = Path(folder)
        self.task_id = str(Path(task_root or folder).resolve())
        self.clock = clock
        self.last_write = -float("inf")
        self.last_present = None

    def publish(self, observation):
        now = self.clock()
        present = observation.get("target") is not None
        if now - self.last_write < 0.1 and present == self.last_present:
            return
        meta = observation.get("metadata") or {}
        packet = dict(
            task_id=self.task_id,
            target_id=observation.get("identity"),
            frame_id=observation.get("frame_id"),
            timestamp=meta.get("arrival", now),
            published_at=now,
            target=observation.get("target"),
            tracking_state=observation.get(
                "tracking_state", "tracking" if present else "recovering"
            ),
            reason=observation.get("reason"),
            recovery_count=observation.get("recovery_count", 0),
        )
        target = observation.get("target")
        selected = meta.get("selected")
        if target and selected and meta.get("distance_source_frame") == observation.get("frame_id"):
            from bucket_robot.perception.target import iou

            box = [selected[k] for k in ("left", "top", "right", "bottom")]
            if iou(box, target.get("raw_box", target["box"])) >= 0.85:
                keys = (
                    "distance_m",
                    "distance_kind",
                    "distance_source_frame",
                    "distance_age_ms",
                    "measurement_status",
                    "detected",
                )
                packet["depth"] = {k: meta.get(k) for k in keys}
        temp = self.folder / "task_observation.json.new"
        temp.write_text(json.dumps(packet, allow_nan=False))
        temp.replace(self.folder / "task_observation.json")
        self.last_write, self.last_present = now, present


class TaskObservationReader:
    def __init__(self, folder, *, clock=time.monotonic):
        self.folder = Path(folder)
        self.clock = clock

    def read(self):
        empty = dict(
            task_id=str(self.folder.resolve()),
            target_id=None,
            frame_id=None,
            target=None,
            tracking_state="waiting",
            reason="waiting_task_observation",
        )
        try:
            parent = json.loads((self.folder / "task_status.json").read_text())
            if not isinstance(parent, dict):
                return empty
            stage = (
                self.folder / "approach" if parent.get("phase") == "left_approach" else self.folder
            )
            packet = json.loads((stage / "task_observation.json").read_text())
            if packet["task_id"] != empty["task_id"]:
                return dict(empty, reason="task_identity_mismatch")
            target = packet.get("target")
            if (
                not 0 <= self.clock() - packet["published_at"] <= 0.4
                or target
                and not 0 <= self.clock() - target["timestamp"] <= 0.35
            ):
                return dict(packet, target=None, depth=None, reason="task_observation_stale")
            return packet
        except (OSError, ValueError, KeyError, TypeError):
            return empty
