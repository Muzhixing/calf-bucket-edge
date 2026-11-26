"""Read-only Linux thermal telemetry, independent of motion capabilities."""

from pathlib import Path
import threading
import time


class ThermalMonitor:
    labels = {
        "soc-thermal": "SoC",
        "bigcore0-thermal": "CPU 大核 0",
        "bigcore1-thermal": "CPU 大核 1",
        "littlecore-thermal": "CPU 小核",
        "center-thermal": "中心域",
        "gpu-thermal": "GPU",
        "npu-thermal": "NPU",
    }
    max_age = 3.0

    def __init__(self, root=Path("/sys/class/thermal"), clock=time.monotonic):
        self.root, self.clock = Path(root), clock
        self.lock = threading.Lock()
        self.sampled_at = None
        self.sensors = self.empty_sensors()

    def empty_sensors(self):
        return {
            kind: {"type": kind, "label": label, "celsius": None}
            for kind, label in self.labels.items()
        }

    def sample(self):
        sensors = self.empty_sensors()
        try:
            zones = sorted(self.root.glob("thermal_zone*"))
        except OSError:
            zones = []
        for zone in zones:
            try:
                kind = (zone / "type").read_text().strip()
            except (OSError, UnicodeError):
                continue
            label = self.labels.get(kind, "Wi-Fi" if kind.startswith("iwlwifi") else kind)
            try:
                celsius = int((zone / "temp").read_text().strip()) / 1000
            except (OSError, ValueError, OverflowError):
                celsius = None
            sensors[kind] = {"type": kind, "label": label, "celsius": celsius}
        with self.lock:
            self.sensors, self.sampled_at = sensors, self.clock()

    def snapshot(self):
        with self.lock:
            age = None if self.sampled_at is None else max(0, self.clock() - self.sampled_at)
            fresh = age is not None and age <= self.max_age
            sensors = [
                dict(item, celsius=item["celsius"] if fresh else None)
                for item in self.sensors.values()
            ]
        return {"age_s": age, "sensors": sensors}

    def run(self, stop_event):
        while not stop_event.is_set():
            self.sample()
            stop_event.wait(1)
