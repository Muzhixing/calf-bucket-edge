"""Read-only task presentation. Never sends commands or advances a mission."""

import json
from pathlib import Path
import threading


class TaskProgressReader:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.lock = threading.Lock()
        self.cache = {}

    def read(self, name, *, lines=False):
        path = self.folder / name
        try:
            stat = path.stat()
            key = (stat.st_mtime_ns, stat.st_size)
            if name in self.cache and self.cache[name][0] == key:
                return self.cache[name][1]
            # Presentation input is bounded, including a damaged/growing journal.
            with path.open("rb") as stream:
                stream.seek(max(0, stat.st_size - 65536))
                data = stream.read(65536)
            if lines:
                value = []
                for line in data.splitlines():
                    try:
                        row = json.loads(line)
                        if isinstance(row, dict) and isinstance(row.get("stage"), str):
                            value.append(row["stage"])
                    except (ValueError, TypeError):
                        continue
            else:
                value = json.loads(data)
                if not isinstance(value, dict):
                    return {}
            self.cache[name] = (key, value)
            return value
        except (OSError, ValueError, TypeError):
            # Do not present an old phase as a new successful observation.
            return [] if lines else {}

    def snapshot(self, status, *, running=False, requested=False, observe_only=False):
        with self.lock:
            parent = self.read("task_status.json")
            source = parent or (status if isinstance(status, dict) else {})
            if parent.get("phase") == "left_approach":
                source = self.read("approach/task_status.json") or source
            return dict(
                running=running,
                requested=requested,
                observe_only=observe_only,
                phase=source.get("phase"),
                reason=source.get("reason"),
                disabled_confirmed=source.get("disabled_confirmed") is True,
                stop_error=source.get("stop_error"),
                stages=self.read("task_lifecycle.jsonl", lines=True),
                profile=self.read("mission_profile.json"),
                summary=self.read("summary.json"),
                post=self.read("post_execution.json") or source.get("execution_progress", {}),
                timing=source.get("timing"),
            )
