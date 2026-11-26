"""Control aim regions, independent of detection identity and ranging pixels."""

import math


def aimed_target(target, config=None):
    if target is None or config is None or "box" not in target:
        return target
    fraction = config["area_fraction"]
    if config["placement"] != "top_center" or not 0 < fraction < 1:
        raise ValueError("invalid_aim_area")
    x1, y1, x2, y2 = target.get("raw_box", target["box"])
    scale = math.sqrt(fraction)
    width, height = (x2 - x1) * scale, (y2 - y1) * scale
    center_x = (x1 + x2) / 2
    region = [center_x - width / 2, y1, center_x + width / 2, y1 + height]
    point = [center_x, y1 + height / 2]
    return dict(
        target,
        body_offset_px=target.get("body_offset_px", target["offset_px"]),
        offset_px=[point[0] - 640, point[1] - 360],
        aim_region_px=region,
        aim_point_px=point,
        aim_area_fraction=fraction,
        horizontal_aim_aligned=region[0] <= 640 <= region[2],
        vertical_aim_aligned=region[1] <= 360 <= region[3],
        aim_region_visible=8 <= y1 and region[3] <= 712,
    )


def horizontal_aligned(target, tolerance=20):
    return bool(target) and target.get(
        "horizontal_aim_aligned", abs(target["offset_px"][0]) <= tolerance
    )
