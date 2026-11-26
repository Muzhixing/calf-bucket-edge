"""Shared target observation and lifecycle; tasks own no tracking implementation."""

from dataclasses import dataclass
from contextlib import nullcontext
import json
import threading
from bucket_robot.domain.lifecycle import CleanupResult, worker_threads
import time


@dataclass(frozen=True)
class FrameBundle:
    observation: dict | None
    left: object = None
    raw: object = None
    error: str | None = None


class _DiscardLog:
    def write(self, text):
        pass


class TargetObserver:
    def __init__(
        self,
        port,
        folder,
        seed_image,
        box,
        bearing_tracker=False,
        recover_loss=False,
        range_calibration=None,
        *,
        geometry=None,
        seed_timestamp=None,
        target_session=None,
        handoff=False,
        publisher=None,
    ):
        self.geometry = geometry
        self.seed_timestamp = seed_timestamp
        self.target_session = target_session
        self.handoff = handoff
        self.publisher = publisher
        self.port = port
        self.range_calibration = range_calibration
        self.latest_raw = None
        self.folder = folder
        self.seed_image = seed_image
        self.box = box
        self.bearing_tracker = bearing_tracker
        self.recover_loss = recover_loss
        self.latest = None
        self.latest_image = None
        self.error = None
        self.end = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        try:
            self._run()
        except BaseException as exc:
            with self.lock:
                self.error = repr(exc)

    def _run(self):
        import cv2, numpy as np
        from bucket_robot.perception.target import TargetLock, weighted_ncc
        from bucket_robot.perception.capture import NativeCapture
        from bucket_robot.perception.session import new_target_session

        cv2.setNumThreads(1)
        geometry = self.geometry
        if self.range_calibration:
            from bucket_robot.perception.assisted_range.stereo import Geometry

            geometry = Geometry(self.range_calibration)
        self.geometry = geometry
        capture_args = {"stereo": True} if geometry is not None else {}
        if self.handoff:
            capture_args["max_age"] = 0.5
        cap = NativeCapture(self.port, **capture_args)
        manager = self.target_session or new_target_session(
            same_target=self.recover_loss, bearing=self.bearing_tracker
        )
        first = self.target_session is None
        self.target_session = manager
        count = 0
        last_frame = -1
        failures = 0
        last_saved_loss = None
        automatic = TargetLock("auto") if self.seed_image is None else None
        try:
            with (
                (self.folder / "vision.jsonl").open("x", buffering=1)
                if self.folder
                else nullcontext(_DiscardLog())
            ) as log:
                while not self.end.is_set():
                    try:
                        image, meta = cap.read()
                        raw = image if geometry is not None else None
                        if geometry is not None:
                            image = cv2.remap(raw[:, :1280], *geometry.maps[0], cv2.INTER_LINEAR)
                        last_frame = meta["frame_id"]
                        if first:
                            if automatic is not None:
                                acquired = automatic.update(image, meta)
                                if acquired is None:
                                    obs = dict(
                                        target=None,
                                        identity=None,
                                        frame_id=last_frame,
                                        metadata=meta,
                                        tracking_state="searching",
                                        reason="automatic_acquisition",
                                        recovery_count=0,
                                    )
                                    with self.lock:
                                        self.latest = obs
                                        self.latest_image = image
                                    self._publish(obs)
                                    log.write(json.dumps(obs) + "\n")
                                    continue
                                self.box = acquired.get("raw_box", acquired["box"])
                                cv2.imwrite(str(self.folder / "automatic_target.png"), image)
                                (self.folder / "automatic_target.json").write_text(
                                    json.dumps(
                                        dict(
                                            box=self.box,
                                            frame_id=last_frame,
                                            source="original_TargetLock_auto",
                                        )
                                    )
                                )
                            else:
                                seed = cv2.imread(str(self.seed_image))
                                x1, y1, x2, y2 = self.box
                                if seed is None:
                                    raise RuntimeError("missing_seed_image")
                                a = cv2.cvtColor(seed[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                                b = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                                if (
                                    a.shape != b.shape
                                    or weighted_ncc(a, b, np.ones(a.shape, np.float32)) < 0.85
                                ):
                                    if (
                                        self.handoff
                                        and self.recover_loss
                                        and self.seed_timestamp is not None
                                    ):
                                        manager.load_verified_reference(
                                            seed, self.box, self.seed_timestamp
                                        )
                                    else:
                                        raise RuntimeError(
                                            "target_changed_during_handoff"
                                            if self.handoff
                                            else "scene_changed_since_seed"
                                        )
                                else:
                                    manager.select(image, self.box, meta["arrival"])
                            if automatic is not None:
                                manager.select(image, self.box, meta["arrival"])
                            first = False
                        target = manager.update(image, meta)
                        failures = 0
                    except Exception as exc:
                        if not self.recover_loss:
                            raise
                        manager.missing(repr(exc))
                        failures += 1
                        obs = {
                            "target": None,
                            "identity": 1,
                            "frame_id": last_frame,
                            "metadata": {},
                            "tracking_state": "recovering",
                            "reason": repr(exc),
                            "recovery_count": manager.reacquired,
                        }
                        with self.lock:
                            self.latest = obs
                        self._publish(obs)
                        log.write(json.dumps(obs) + "\n")
                        cap = NativeCapture(self.port, **capture_args)
                        self.end.wait(min(1.0, 0.05 * failures))
                        continue
                    if (
                        self.folder is not None
                        and self.recover_loss
                        and manager.state == "recovering"
                        and manager.reacquired != last_saved_loss
                        and manager.previous_image is not None
                        and manager.latest is not None
                    ):
                        suffix = str(manager.reacquired)
                        cv2.imwrite(
                            str(self.folder / ("recovery_reference_" + suffix + ".png")),
                            manager.previous_image,
                        )
                        cv2.imwrite(str(self.folder / ("recovery_frame_" + suffix + ".png")), image)
                        (self.folder / ("recovery_reference_" + suffix + ".json")).write_text(
                            json.dumps(dict(reference=manager.latest, current=meta), indent=2)
                        )
                        last_saved_loss = manager.reacquired
                    if manager.state != "tracking" or target is None:
                        if not self.recover_loss:
                            if self.folder is None:
                                raise RuntimeError("target_lost_" + manager.reason)
                            cv2.imwrite(str(self.folder / "failure.jpg"), image)
                            if self.bearing_tracker and manager.previous_image is not None:
                                cv2.imwrite(
                                    str(self.folder / "failure_previous.png"),
                                    manager.previous_image,
                                )
                            (self.folder / "failure.json").write_text(
                                json.dumps({"reason": manager.reason, "metadata": meta}, indent=2)
                            )
                            raise RuntimeError("target_lost_" + manager.reason)
                    if manager.generation != 1 or (manager.reacquired and not self.recover_loss):
                        raise RuntimeError("target_identity_changed")
                    obs = {
                        "target": target,
                        "identity": manager.generation,
                        "frame_id": meta["frame_id"],
                        "metadata": meta,
                        "tracking_state": manager.state,
                        "reason": manager.reason,
                        "recovery_count": manager.reacquired,
                    }
                    with self.lock:
                        self.latest = obs
                        self.latest_image = image
                        self.latest_raw = raw
                    self._publish(obs)
                    log.write(json.dumps(obs, allow_nan=False) + "\n")
                    if self.folder is not None and count % 15 == 0:
                        cv2.imwrite(str(self.folder / "latest.jpg"), image)
                    count += 1
        except BaseException as e:
            with self.lock:
                self.error = repr(e)
        finally:
            if hasattr(manager, "close"):
                cleanup = manager.close()
                if cleanup is not None and not cleanup.stopped:
                    with self.lock:
                        self.error = "nested_worker_cleanup_timeout"

    def cleanup_children(self):
        manager = self.target_session
        return (manager,) if manager is not None else ()

    def request_stop(self):
        self.end.set()
        for child in self.cleanup_children():
            if hasattr(child, "request_stop"):
                child.request_stop()

    def await_stopped(self, timeout=2):
        until = time.monotonic() + timeout
        for thread in worker_threads(self):
            thread.join(timeout=max(0, until - time.monotonic()))
        pending = tuple(t.name for t in worker_threads(self) if t.is_alive())
        return CleanupResult(not pending, pending)

    def _publish(self, observation):
        if self.publisher is not None:
            self.publisher.publish(observation)

    def frame(self):
        with self.lock:
            return FrameBundle(self.latest, self.latest_image, self.latest_raw, self.error)

    def get(self):
        """One observation contract for every task, including stereo handoff."""
        frame = self.frame()
        if frame.error:
            raise RuntimeError(frame.error)
        return (
            dict(frame.observation, left=frame.left, raw=frame.raw) if frame.observation else None
        )

    def snapshot(self):
        """Legacy view; data still originates from the common FrameBundle."""
        frame = self.frame()
        return frame.observation, frame.left, frame.error

    def range_snapshot(self):
        frame = self.frame()
        return frame.observation, frame.left, frame.raw, frame.error


class StereoObserver(TargetObserver):
    """Stopped stage handoff using the same capture/session worker."""

    def __init__(
        self,
        port,
        geometry,
        seed_image,
        box,
        bearing_tracker=False,
        recover_loss=False,
        seed_timestamp=None,
        target_session=None,
        *,
        publisher=None,
    ):
        super().__init__(
            port,
            None,
            seed_image,
            box,
            bearing_tracker,
            recover_loss,
            geometry=geometry,
            seed_timestamp=seed_timestamp,
            target_session=target_session,
            handoff=True,
            publisher=publisher,
        )
