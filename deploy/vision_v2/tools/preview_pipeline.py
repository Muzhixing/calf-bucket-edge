"""Latest-pair capture, per-frame detection and single-flight stereo for live preview."""
import multiprocessing
import os
import signal
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import cv2
import numpy as np

from app.domain.events import RangingUpdateEvent
from app.infrastructure import camera_config as calib, detector
from diagnostics import range_audit
from guidance import same_target


class Latest:
    """One replaceable item per stage. Readers count gaps from the camera sequence."""
    def __init__(self):
        self.condition = threading.Condition()
        self.item = None
        self.error = None

    def put(self, item):
        with self.condition:
            self.item = item
            self.condition.notify_all()

    def fail(self, error):
        with self.condition:
            self.error = error
            self.condition.notify_all()

    def after(self, previous):
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.error or (self.item and self.item[0] > previous), 3)
            if not ready:
                raise TimeoutError("No new camera pair for three seconds")
            if self.error:
                raise self.error
            return self.item


def remap_rows(raw, out, begin, end):
    # Slicing the destination maps preserves their absolute source coordinates.
    cv2.remap(raw, calib.left_map1[begin:end], calib.left_map2[begin:end],
              cv2.INTER_LINEAR, dst=out[begin:end])


class CameraStages:
    def __init__(self, capture, left_roi, right_roi):
        self.right_roi = right_roi
        self.capture = capture
        self.left_roi = left_roi
        self.raw = Latest()
        self.rectified = Latest()
        self.stop = threading.Event()
        self.reader = threading.Thread(target=self.read, name="camera", daemon=True)
        self.rectifier = threading.Thread(target=self.rectify, name="rectify", daemon=True)

    def read(self):
        try:
            os.sched_setaffinity(0, {4})
            seq = 0
            while not self.stop.is_set():
                ok, image = self.capture.read()
                if not ok or image is None or image.shape != (720, 2560, 3):
                    raise RuntimeError("Missing or malformed stereo camera frame")
                seq += 1
                self.raw.put((seq, time.monotonic(), image))
        except Exception as exc:
            self.raw.fail(exc)

    def rectify(self):
        try:
            os.sched_setaffinity(0, {5})
            seq = 0
            with ThreadPoolExecutor(max_workers=1, initializer=os.sched_setaffinity,
                                    initargs=(0, {2}), thread_name_prefix="rectify-rows") as pool:
                while not self.stop.is_set():
                    seq, arrival, image = self.raw.after(seq)
                    raw_left = crop(image, self.left_roi)
                    left = np.empty(raw_left.shape, dtype=raw_left.dtype)
                    split = left.shape[0] * 2 // 3
                    tail = pool.submit(remap_rows, raw_left, left, split, left.shape[0])
                    remap_rows(raw_left, left, 0, split)
                    tail.result()
                    self.rectified.put((seq, arrival, crop(image, self.right_roi), left))
        except Exception as exc:
            self.rectified.fail(exc)


def detect(image, model):
    boxes, classes, scores = model.infer(image)
    if boxes is None:
        return []
    return [
        dict(left=int(x1), top=int(y1), right=int(x2), bottom=int(y2),
             class_id=int(cls), label=detector.CLASSES[int(cls)], score=float(score))
        for (x1, y1, x2, y2), cls, score in
        zip(detector.scale_boxes(image.shape, boxes), classes, scores)
    ]


def crop(frame, roi):
    x, y, w, h = roi
    return frame[y:y+h, x:x+w]


def stereo_images(buffer):
    return np.frombuffer(buffer, np.uint8).reshape(2, 720, 1280, 3)


class StereoWorker:
    def __init__(self, buffer, params, scale):
        from app.infrastructure.ranging_service import RangingService
        self.images = stereo_images(buffer)
        self.service = RangingService(**params)
        self.scale = scale
        self.stereo = self.service._create_stereo_sgbm(self.service.sgbm_num_disparities * scale)
        self.wls = self.right_matcher = None
        if self.service.use_wls and scale == 1 and hasattr(getattr(cv2, 'ximgproc', None), 'createRightMatcher'):
            self.right_matcher = cv2.ximgproc.createRightMatcher(self.stereo)
            self.wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.stereo)
            self.wls.setLambda(self.service.wls_lambda)
            self.wls.setSigmaColor(self.service.wls_sigma)

    def measure(self, box):
        left, raw_right = self.images
        right = cv2.remap(raw_right, calib.right_map1, calib.right_map2, cv2.INTER_LINEAR)
        disparity = self.service._compute_disparity(
            self.stereo, cv2.cvtColor(left, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), self.wls, self.right_matcher, scale=self.scale)
        z = self.service._calculate_distance_for_box(disparity, *box)
        audit = range_audit(self.service, calib, disparity, *box)
        assert audit['value_m'] == z
        return z, audit


def start_stereo(buffer, params, scale):
    global stereo_worker
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    for tid in os.listdir("/proc/self/task"):
        try:
            os.sched_setaffinity(int(tid), {4, 6, 7})
            os.setpriority(os.PRIO_PROCESS, int(tid), 5)
        except ProcessLookupError:
            pass  # An OpenCV startup thread already exited.
    cv2.setNumThreads(3)
    stereo_worker = StereoWorker(buffer, params, scale)


def measure_shared(box):
    return stereo_worker.measure(box)


def run(service, publish, params, scale):
    # Keep the full-resolution rectifier on its own Cortex-A76.
    assert service.disparity_mode == 'full'
    cv2.setNumThreads(1)
    os.sched_setaffinity(0, {6})
    model = detector.RknnDetector()
    try:
        _, cap, _ = service._open_camera_capture()
        service.cap = cap
        if not cap.set(cv2.CAP_PROP_BUFFERSIZE, 2):
            raise RuntimeError("Camera rejected the two-buffer capture queue")
        camera = CameraStages(cap, service.left_roi, service.right_roi)
        ctx = multiprocessing.get_context('spawn')
        buffer = ctx.RawArray('B', 2 * 720 * 1280 * 3)
        images = stereo_images(buffer)
        camera.reader.start()
        camera.rectifier.start()
        try:
            with ProcessPoolExecutor(max_workers=1, mp_context=ctx, initializer=start_stereo,
                                     initargs=(buffer, params, scale)) as pool:
                seq = frame_id = epoch = 0
                previous = accepted = value = job = None
                while service.camera_active:
                    seq, arrival, right, left = camera.rectified.after(seq)
                    frame_id += 1
                    detections = detect(left, model)
                    best = max(detections, key=lambda item: item['score'], default=None)
                    box = None
                    if best and best['score'] >= detector.DETECT_SCORE_MIN:
                        box = tuple(best[k] for k in ('left','top','right','bottom'))
                    if not same_target(previous, box):
                        epoch += 1
                        accepted = value = None
                    previous = box
                    now = time.monotonic()
                    if accepted and (now - accepted['time'] > service.hold_last_seconds
                                     or not same_target(accepted['box'], box)):
                        accepted = value = None
                    details = dict(camera_seq=seq, capture_monotonic_s=arrival)
                    if job and job.done():
                        raw, audit = job.result()
                        job = None
                        details['attempt_frame'] = source['id']
                        if (epoch == source['epoch'] and same_target(box, source['box'])
                                and now - source['time'] <= service.hold_last_seconds):
                            details.update(raw=raw, attempt=audit)
                            if raw is None and audit['attempts']:
                                details['attempt_reason'] = audit['attempts'][-1]['reason']
                            if raw is not None:
                                value = raw if value is None else service.smooth_alpha * raw + (1-service.smooth_alpha) * value
                                accepted = source
                        else:
                            details.update(discarded_attempt=True, attempt_reason='target_changed_or_expired')
                    if box and job is None:
                        source = dict(id=frame_id, camera_seq=seq, time=arrival, box=box, epoch=epoch)
                        # No overwrite until the previous future has returned.
                        images[0] = left
                        images[1] = right
                        job = pool.submit(measure_shared, box)
                    if accepted:
                        details.update(source_frame=accepted['id'], source_time=accepted['time'],
                                       source_camera_seq=accepted['camera_seq'])
                    service.latest_metadata = dict(distance_source=f'full@{scale:g}' if accepted else 'none',
                                                   distance_held=value is not None and details.get('raw') is None,
                                                   profile_ms={})
                    publish(RangingUpdateEvent(time.time() - (time.monotonic() - arrival), box is not None, value, detections,
                                               frame_id=frame_id, left_frame=left), details)
        finally:
            camera.stop.set()
            camera.raw.fail(RuntimeError("Camera stopped"))
            camera.reader.join()
            camera.rectifier.join()
    finally:
        model.release()
