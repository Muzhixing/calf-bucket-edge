# camera_config.py
# -*- coding: utf-8 -*-

import cv2
import numpy as np


def _as_cv_calib_matrix(values, shape):
    """将标定参数规范为 OpenCV 兼容的 float64 连续数组。"""
    return np.ascontiguousarray(np.array(values, dtype=np.float64).reshape(shape))

# ===================== 左相机内参（CameraParameters1） =====================
# K = [698.2342, 0, 624.0385;
#      0, 704.4849, 390.3262;
#      0, 0, 1];

left_camera_matrix = _as_cv_calib_matrix([
    [698.2342,   0.0,      624.0385],
    [0.0,        704.4849, 390.3262],
    [0.0,        0.0,      1.0]
], (3, 3))

# RadialDistortion = [-0.0263, 0.0391]
# TangentialDistortion = [0, 0]
# OpenCV: [k1, k2, p1, p2, k3]
left_distortion = _as_cv_calib_matrix(
    [-0.0263, 0.0391, 0.0, 0.0, 0.0],
    (5, 1),
)

# ===================== 右相机内参（CameraParameters2） =====================
# K = [698.8948, 0, 622.6062;
#      0, 705.5602, 381.0941;
#      0, 0, 1];

right_camera_matrix = _as_cv_calib_matrix([
    [698.8948,   0.0,      622.6062],
    [0.0,        705.5602, 381.0941],
    [0.0,        0.0,      1.0]
], (3, 3))

# RadialDistortion = [-0.0416, 0.0839]
# TangentialDistortion = [0, 0]
right_distortion = _as_cv_calib_matrix(
    [-0.0416, 0.0839, 0.0, 0.0, 0.0],
    (5, 1),
)

# ===================== 双目外参（来自 PoseCamera2） =====================

R = _as_cv_calib_matrix([
    [1.0000,  -0.0002,   0.0013],
    [0.0002,   1.0000,  -0.0034],
    [-0.0013,  0.0034,   1.0000]
], (3, 3))

T = _as_cv_calib_matrix([
    [-60.2876],
    [0.0105],
    [0.6070]
], (3, 1))   # 单位：mm

# ===================== 立体校正 & 重映射 =====================

image_size = (1280, 720)

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    left_camera_matrix, left_distortion,
    right_camera_matrix, right_distortion,
    image_size, R, T
)

left_map1, left_map2 = cv2.initUndistortRectifyMap(
    left_camera_matrix, left_distortion, R1, P1, image_size, cv2.CV_16SC2
)

right_map1, right_map2 = cv2.initUndistortRectifyMap(
    right_camera_matrix, right_distortion, R2, P2, image_size, cv2.CV_16SC2
)
