import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def traced_detections(detector, decoded, anchors, image_shape):
    boxes, classes, scores = decoded
    if boxes is None:
        return []
    pixel_boxes = detector.scale_boxes(image_shape, boxes)
    result = []
    for b, s, c, pixels, anchor in zip(boxes, scores, classes, pixel_boxes, anchors):
        anchor = int(anchor)
        stride, offset, width = (8, 0, 80) if anchor < 6400 else (16, 6400, 40) if anchor < 8000 else (32, 8000, 20)
        cell = anchor - offset
        left, top, right, bottom = map(int, pixels)
        result.append(dict(left=left, top=top, right=right, bottom=bottom,
                           score=float(s), class_id=int(c), label='bucket', anchor=anchor,
                           stride=stride, grid_y=cell // width, grid_x=cell % width,
                           model_box=list(map(float, b)), center_x=(left + right) / 2,
                           center_error_from_image_px=(left + right) / 2 - image_shape[1] / 2))
    return result
