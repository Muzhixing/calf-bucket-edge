"""Decode only the anchors passing the frozen single-class RKNN score filter."""
import numpy as np


def decode(detector, outputs):
    boxes, scores, anchors = [], [], []
    offset = 0
    threshold = max(detector.OBJ_THRESH, detector.DETECT_SCORE_MIN)
    for branch in (0, 3, 6):
        regression, confidence = outputs[branch:branch + 2]
        height, width = confidence.shape[2:]
        values = confidence.reshape(height * width)
        indices = np.flatnonzero(values >= threshold)
        if indices.size:
            logits = regression.reshape(1, 64, -1)[:, :, indices]
            distances = detector.dfl(np.ascontiguousarray(logits[..., None]))[0, :, :, 0].T
            y, x = np.divmod(indices, width)
            centers = np.column_stack((x, y)) + 0.5
            stride = detector.MODEL_SIZE[0] // width
            boxes.append(np.column_stack((centers - distances[:, :2],
                                          centers + distances[:, 2:])) * stride)
            scores.append(values[indices])
            anchors.append(indices + offset)
        offset += values.size
    if not boxes:
        return (None, None, None), ()
    boxes = np.concatenate(boxes)
    scores = np.concatenate(scores)
    keep = detector.nms_boxes(boxes, scores)
    return (boxes[keep], np.zeros(len(keep), dtype=int), scores[keep]), np.concatenate(anchors)[keep]
