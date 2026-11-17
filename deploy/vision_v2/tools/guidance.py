"""Image-centred guidance for a fixed camera. All outputs are observations or dry-run events."""

def same_target(a, b):
    if a is None or b is None:
        return False
    x1, y1, x2, y2 = a
    u1, v1, u2, v2 = b
    overlap = max(0, min(x2, u2) - max(x1, u1)) * max(0, min(y2, v2) - max(y1, v1))
    union = (x2-x1)*(y2-y1) + (u2-u1)*(v2-v1) - overlap
    return union > 0 and overlap / union >= 0.5


class VisionGuidance:
    def __init__(self, run_id, tolerance_x, tolerance_y, fx, fy):
        self.run_id = run_id
        self.tolerance_x = tolerance_x
        self.tolerance_y = tolerance_y
        self.fx = float(fx)
        self.fy = float(fy)
        self.previous_box = None
        self.target_start = None
        self.aligned_since = None
        self.preview_id = None

    def update(self, row, width, height):
        frame = row['frame_id']
        selected = row['selected'] if row['detected'] else None
        box = tuple(selected[k] for k in ('left', 'top', 'right', 'bottom')) if selected else None
        if not same_target(self.previous_box, box):
            self.target_start = frame
            self.aligned_since = self.preview_id = None
        self.previous_box = box
        cx, cy = width / 2, height / 2
        tx, ty = cx * self.tolerance_x, cy * self.tolerance_y
        result = dict(image_center_px=[cx, cy], tolerance_px=[tx, ty], target_center_px=None,
                      offset_px=None, estimated_offset_mm=None, aligned_xy=False, state='no_target',
                      reference_depth_mm=None, cycle_id=None, feed_preview_started=False)
        if not box:
            return result
        x, y = (box[0]+box[2])/2, (box[1]+box[3])/2
        dx, dy = x-cx, y-cy
        aligned = abs(dx) <= tx and abs(dy) <= ty
        result.update(target_center_px=[x, y], offset_px=[dx, dy], aligned_xy=aligned)
        source = row['distance_source_frame']
        valid = (row['measurement_status'] in ('new', 'held') and row['distance_m'] is not None
                 and source is not None and self.target_start <= source <= frame)
        if valid:
            depth_mm = row['distance_m'] * 1000
            result['estimated_offset_mm'] = [round(dx*depth_mm/self.fx, 1), round(dy*depth_mm/self.fy, 1)]
        if not aligned:
            self.aligned_since = self.preview_id = None
            result['state'] = 'adjust'
            return result
        if self.aligned_since is None:
            self.aligned_since = frame
        result['state'] = 'aligned_wait_depth'
        if valid and source >= self.aligned_since:
            if self.preview_id is None and row['measurement_status'] == 'new':
                self.preview_id = f'{self.run_id}/{self.aligned_since}'
                result['feed_preview_started'] = True
            if self.preview_id is not None:
                result.update(state='ready_dry_run', cycle_id=self.preview_id,
                              reference_depth_mm=round(depth_mm))
        return result
