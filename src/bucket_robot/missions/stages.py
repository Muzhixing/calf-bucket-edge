"""Explicit orchestration transitions; tracking/recovery retain their own policies."""

from enum import Enum
import json
import time


class Stage(str, Enum):
    ACQUIRE = "acquire"
    TRACK = "track"
    POST_ALIGN = "post_align"
    DRAIN = "drain"
    APPROACH = "approach"
    POST_ACTION = "post_action"
    COMPLETE = "complete"
    FAULT = "fault"


class MissionStages:
    transitions = {
        Stage.ACQUIRE: {Stage.TRACK, Stage.DRAIN},
        Stage.TRACK: {Stage.POST_ALIGN, Stage.DRAIN},
        Stage.POST_ALIGN: {Stage.DRAIN},
        Stage.DRAIN: {Stage.APPROACH, Stage.POST_ACTION, Stage.COMPLETE, Stage.FAULT},
        Stage.APPROACH: {Stage.POST_ACTION, Stage.COMPLETE, Stage.FAULT},
        Stage.POST_ACTION: {Stage.COMPLETE, Stage.FAULT},
    }

    def __init__(self, folder):
        self.folder = folder
        self.error = None
        self.stage = Stage.ACQUIRE
        self._record()

    def _record(self):
        try:
            with (self.folder / "task_lifecycle.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(dict(stage=self.stage.value, monotonic=time.monotonic())) + "\n"
                )
        except OSError as exc:
            # Telemetry failure must never interrupt the disable/close finally.
            self.error = repr(exc)

    def check(self):
        if self.error:
            raise RuntimeError("stage_journal_failed:" + self.error)

    def transition(self, stage):
        if stage == self.stage:
            return
        if stage not in self.transitions.get(self.stage, set()):
            raise RuntimeError(f"invalid_mission_transition:{self.stage.value}:{stage.value}")
        self.stage = stage
        self._record()
