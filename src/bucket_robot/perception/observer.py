"""Stereo target observation, independent of motion and task execution."""

import threading


class StereoObserver:
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
    ):
        self.port = port
        self.geometry, self.seed_image, self.box = geometry, seed_image, box
        self.bearing_tracker = bearing_tracker
        self.recover_loss = recover_loss
        self.seed_timestamp = seed_timestamp
        self.target_session = target_session
        self.latest = None
        self.error = None
        self.lock = threading.Lock()
        self.end = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        import cv2
        import numpy as np
        from bucket_robot.perception.target import weighted_ncc
        from bucket_robot.perception.session import new_target_session

        reuse_session = self.target_session is not None
        manager = self.target_session or new_target_session(
            same_target=self.recover_loss, bearing=self.bearing_tracker
        )
        self.target_session = manager
        from bucket_robot.perception.capture import NativeCapture

        capture = NativeCapture(self.port, stereo=True, max_age=0.5)
        last = -1
        first = not reuse_session
        failures = 0
        try:
            while not self.end.is_set():
                try:
                    raw, meta = capture.read()
                    fid = meta["frame_id"]
                    left = cv2.remap(raw[:, :1280], *self.geometry.maps[0], cv2.INTER_LINEAR)
                    if first:
                        seed = cv2.imread(str(self.seed_image))
                        x1, y1, x2, y2 = map(int, self.box)
                        if seed is None:
                            raise RuntimeError("missing_handoff_image")
                        a, b = [
                            cv2.cvtColor(im[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                            for im in (seed, left)
                        ]
                        if (
                            a.shape != b.shape
                            or weighted_ncc(a, b, np.ones(a.shape, np.float32)) < 0.85
                        ):
                            if self.recover_loss and self.seed_timestamp is not None:
                                manager.load_verified_reference(seed, self.box, self.seed_timestamp)
                            else:
                                raise RuntimeError("target_changed_during_handoff")
                        else:
                            manager.select(left, self.box, meta["arrival"])
                        first = False
                    target = manager.update(left, meta)
                    failures = 0
                except Exception as exc:
                    if not self.recover_loss:
                        raise
                    manager.missing(repr(exc))
                    failures += 1
                    with self.lock:
                        self.latest = dict(
                            target=None, identity=1, frame_id=last, metadata={}, reason=repr(exc)
                        )
                    last = -1
                    capture.last_id = -1
                    self.end.wait(min(1.0, 0.05 * failures))
                    continue
                if (
                    manager.state != "tracking"
                    or target is None
                    or manager.generation != 1
                    or manager.reacquired
                ):
                    if not self.recover_loss:
                        raise RuntimeError("target_lost_or_switched")
                with self.lock:
                    self.latest = dict(
                        target=target,
                        identity=manager.generation,
                        frame_id=fid,
                        metadata=meta,
                        raw=raw,
                        left=left,
                        reason=manager.reason,
                        recovery_count=manager.reacquired,
                    )
                last = fid
        except BaseException as exc:
            with self.lock:
                self.error = repr(exc)
        finally:
            if hasattr(manager, "close"):
                manager.close()

    def get(self):
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            return self.latest
