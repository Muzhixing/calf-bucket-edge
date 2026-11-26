"""Continuous ranging with capture-time prediction and static final verification."""

import threading
import time

from bucket_robot.missions.recovering_approach import static_surface_matches
from bucket_robot.missions.range_prediction import RangePrediction


class ForwardRange:
    def __init__(self, observer, calibration, model, surface_band=None, cruise=8):
        self.observer = observer
        self.calibration, self.model, self.band = calibration, model, surface_band
        self.lock = threading.Lock()
        self.end = threading.Event()
        self.answer = None
        self.error = None
        self.last_frame = -1
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.static_since = None
        self.near = None
        self.prediction = RangePrediction(cruise=cruise)
        self.verify_started = None
        self.last_invalid = None

    def run(self):
        delay = 0.5
        while not self.end.is_set():
            self._run_once()
            if self.end.wait(delay):
                break
            delay = min(8.0, delay * 2)

    def _run_once(self):
        from bucket_robot.perception.assisted_range.stereo import AssistedRange
        import cv2
        import numpy as np

        estimator = None
        try:
            estimator = AssistedRange(self.calibration, self.model)
            while not self.end.wait(0.01):
                obs, left, raw, error = self.observer.range_snapshot()
                if error or obs is None or obs.get("target") is None or raw is None:
                    continue
                if obs["frame_id"] <= self.last_frame:
                    continue
                self.last_frame = obs["frame_id"]
                target = obs["target"]
                if not 0 <= time.monotonic() - target["timestamp"] <= 0.4:
                    continue
                box = target["box"]
                x1, y1, x2, y2 = map(int, box)
                roi = None
                if self.band is not None:
                    u1, v1, u2, v2 = self.band
                    roi = [
                        int((x1 + u1 * (x2 - x1)) / 2),
                        int((y1 + v1 * (y2 - y1)) / 2),
                        int((x1 + u2 * (x2 - x1)) / 2),
                        int((y1 + v2 * (y2 - y1)) / 2),
                    ]
                right = cv2.remap(raw[:, 1280:], *estimator.geometry.maps[1], cv2.INTER_LINEAR)
                measurement = estimator.measure([left, right], box, point_roi=roi)
                measurement.update(frame_id=obs["frame_id"], capture_time=target["timestamp"])
                patch = cv2.resize(cv2.cvtColor(left[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY), (96, 96))
                answer = dict(
                    measurement=measurement,
                    identity=obs["identity"],
                    recovery_count=obs.get("recovery_count", 0),
                    box=box,
                    patch=patch,
                    weights=np.ones((96, 96), np.float32),
                )
                with self.lock:
                    if measurement.get("valid"):
                        self.answer = answer
                    else:
                        self.last_invalid = measurement
                    self.error = None
        except Exception as exc:
            with self.lock:
                self.error = repr(exc)
        finally:
            if estimator is not None:
                estimator.close()

    def evaluate(self, now, obs, image, stopped):
        """Compensate slow range; verify the final stop using a disabled snapshot."""
        if stopped:
            if self.static_since is None:
                self.static_since = now
        else:
            self.static_since = None
        with self.lock:
            answer, error = self.answer, self.error
        if error:
            return dict(permit=False, near=False, reason="range_worker_unavailable", error=error)
        if answer is None:
            return dict(permit=False, near=False, reason="waiting_range")
        m = answer["measurement"]
        target = (obs or {}).get("target")
        current = (
            target is not None
            and obs["identity"] == answer["identity"]
            and obs.get("recovery_count", 0) == answer.get("recovery_count", 0)
            and 0 <= now - target["timestamp"] <= 0.35
        )
        valid = (
            current
            and m.get("valid")
            and m.get("distance_kind") == "stereo_optical_midpoint_to_bucket_surface"
        )
        static = (
            valid
            and stopped
            and self.static_since is not None
            and self.static_since <= m["capture_time"]
            and 0 <= now - m["capture_time"] <= 5
            and static_surface_matches(
                image, target["box"], answer["box"], answer["patch"], answer["weights"]
            )
        )
        if not valid:
            return dict(
                permit=False,
                near=False,
                speed_cap=0,
                reason="waiting_valid_same_target_range",
                measurement=m,
            )
        self.prediction.add(answer["identity"], m)
        current_box = target["box"]
        width_ratio = (answer["box"][2] - answer["box"][0]) / (current_box[2] - current_box[0])
        plan = self.prediction.estimate(now, width_ratio)
        if static:
            # No approach occurred during this latency interval. Old pre-stop
            # closing rates must not force repeated predicted stops at rest.
            plan.update(
                permit=True,
                predicted_distance_m=m["distance_m"],
                predicted_lower_m=m["lower_m"],
                closing_mps=0.0,
                verify_stop=m["lower_m"] <= 0.205,
            )
        plan["measurement"] = m
        if self.verify_started is not None:
            # A predicted stop is provisional. Complete only on a newly captured
            # disabled, image-verified measurement, not on extrapolation alone.
            if static and m["capture_time"] >= self.verify_started:
                self.verify_started = None
            else:
                return dict(
                    plan, permit=False, near=False, speed_cap=0, reason="verifying_20cm_stop"
                )
        if static and (m["distance_m"] <= 0.20 or m["lower_m"] <= 0.21):
            return dict(
                plan, permit=False, near=True, speed_cap=0, reason="target_distance_reached"
            )
        if plan.get("verify_stop"):
            self.verify_started = now
            return dict(
                plan, permit=False, near=False, speed_cap=0, reason="predicted_stop_remeasure"
            )
        return plan

    def close(self):
        self.end.set()
        self.thread.join(timeout=1)
