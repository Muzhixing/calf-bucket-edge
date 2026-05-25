#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双目测距核心模块
- 负责摄像头采集、视差计算、距离估计
- 提供线程安全的数据访问接口
"""

import cv2
import json
import numpy as np
import threading
import time
import urllib.request

try:
    from cv2 import ximgproc
except Exception:
    ximgproc = None

from app.domain.events import RangingUpdateEvent
from app.infrastructure import camera_config, detector


class RangingService:
    """双目测距服务"""

    def __init__(self,
                 camera_device=21,
                 frame_width=2560,
                 frame_height=720,
                 enable_display=False,
                 render_on_device=False,
                 enable_push=False,
                 video_push_url=None,
                 meta_push_url=None,
                 push_fps=8,
                 push_timeout=1.0,
                 push_jpeg_quality=85,
                 event_bus=None,
                 use_wls=True,
                 wls_lambda=8000.0,
                 wls_sigma=1.5,
                 sgbm_num_disparities=160,
                 sgbm_block_size=7,
                 sgbm_uniqueness_ratio=12,
                 sgbm_speckle_window_size=120,
                 sgbm_speckle_range=2,
                 smooth_alpha=0.9,
                 min_valid_ratio=0.35,
                 max_std_ratio=0.55,
                 min_median_disp=1.2,
                 roi_expand_ratio=0.25,
                 hold_last_seconds=1.5,
                 hold_lost_seconds=0.4,
                 distance_min_m=0.05,
                 distance_max_m=1.2,
                 relax_valid_ratio=0.18,
                 relax_max_std_ratio=0.75,
                 relax_min_median_disp=0.8,
                 enable_debug=False):
        self.camera_device = camera_device
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.enable_display = enable_display
        self.render_on_device = render_on_device
        self.enable_push = enable_push
        self.video_push_url = video_push_url
        self.meta_push_url = meta_push_url
        self.push_fps = push_fps
        self.push_timeout = push_timeout
        self.push_jpeg_quality = push_jpeg_quality
        self.event_bus = event_bus
        self.smooth_alpha = smooth_alpha
        self.min_valid_ratio = min_valid_ratio
        self.max_std_ratio = max_std_ratio
        self.min_median_disp = min_median_disp
        self.roi_expand_ratio = roi_expand_ratio
        self.hold_last_seconds = hold_last_seconds
        self.hold_lost_seconds = hold_lost_seconds
        self.distance_min_m = distance_min_m
        self.distance_max_m = distance_max_m
        self.relax_valid_ratio = relax_valid_ratio
        self.relax_max_std_ratio = relax_max_std_ratio
        self.relax_min_median_disp = relax_min_median_disp
        self.enable_debug = enable_debug

        self.use_wls = use_wls
        self.wls_lambda = wls_lambda
        self.wls_sigma = wls_sigma
        self.sgbm_num_disparities = sgbm_num_disparities
        self.sgbm_block_size = sgbm_block_size
        self.sgbm_uniqueness_ratio = sgbm_uniqueness_ratio
        self.sgbm_speckle_window_size = sgbm_speckle_window_size
        self.sgbm_speckle_range = sgbm_speckle_range

        self.left_roi = (0, 0, 1280, 720)
        self.right_roi = (1280, 0, 1280, 720)

        self.cap = None
        self.camera_active = False

        self.latest_frame = None
        self.latest_left_frame = None
        self.latest_disparity = None
        self.latest_distance = None
        self.latest_display_frame = None
        self.latest_detected = False
        self.latest_detections = []
        self.latest_frame_ts = None
        self.data_lock = threading.Lock()

        self._camera_thread = None
        self._display_thread = None
        self._video_push_thread = None
        self._meta_push_thread = None

    def _create_stereo_sgbm(self):
        # numDisparities 必须是 16 的倍数
        num_disparities = int(max(16, round(self.sgbm_num_disparities / 16) * 16))
        block_size = int(max(3, self.sgbm_block_size))
        if block_size % 2 == 0:
            block_size += 1

        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * 3 * block_size ** 2,
            P2=32 * 3 * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=int(self.sgbm_uniqueness_ratio),
            speckleWindowSize=int(self.sgbm_speckle_window_size),
            speckleRange=int(self.sgbm_speckle_range),
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        return stereo

    def _calculate_distance_for_box(self, disparity_map, left, top, right, bottom):
        """根据检测框在视差图上估计距离。

        目标：在保证不乱跳的前提下，尽量减少 N/A。
        做法：先用严格阈值算一次；失败后再用更宽松阈值 + 更大 ROI/中心裁剪再算一次。
        """
        if disparity_map is None:
            return None

        h, w = disparity_map.shape

        def _calc_once(min_valid_ratio, max_std_ratio, min_median_disp, roi_expand_ratio, center_crop_ratio=None):
            # 适当扩大 ROI（桶表面纹理不足时，扩大一点范围更容易得到稳定视差）
            bw = max(1, right - left)
            bh = max(1, bottom - top)
            dx = int(bw * roi_expand_ratio)
            dy = int(bh * roi_expand_ratio)

            x1 = max(0, min(left - dx, w - 1))
            x2 = max(0, min(right + dx, w - 1))
            y1 = max(0, min(top - dy, h - 1))
            y2 = max(0, min(bottom + dy, h - 1))

            if x2 <= x1 or y2 <= y1:
                return None

            roi = disparity_map[y1:y2, x1:x2]

            # 可选：只取中心区域，减少背景干扰（尤其桶边缘/地面纹理影响）
            if center_crop_ratio is not None and 0.2 < center_crop_ratio < 1.0:
                rh, rw = roi.shape
                cx1 = int(rw * (0.5 - center_crop_ratio / 2.0))
                cx2 = int(rw * (0.5 + center_crop_ratio / 2.0))
                cy1 = int(rh * (0.5 - center_crop_ratio / 2.0))
                cy2 = int(rh * (0.5 + center_crop_ratio / 2.0))
                roi = roi[cy1:cy2, cx1:cx2]

            valid = roi[roi > 0]
            valid = valid[np.isfinite(valid)]
            if valid.size < roi.size * min_valid_ratio:
                return None

            # 1) 百分位裁剪，去掉极端离群（反光/遮挡/匹配失败常见）
            lo, hi = np.percentile(valid, [10, 90])
            valid = valid[(valid >= lo) & (valid <= hi)]
            if valid.size < roi.size * (min_valid_ratio * 0.6):
                return None

            # 2) MAD 离群剔除（比 std 更抗噪）
            median0 = float(np.median(valid))
            mad = float(np.median(np.abs(valid - median0))) + 1e-6
            sigma = 1.4826 * mad
            inliers = valid[np.abs(valid - median0) <= 3.0 * sigma]
            if inliers.size < roi.size * (min_valid_ratio * 0.6):
                return None

            median_disp = float(np.median(inliers))
            std_disp = float(np.std(inliers))

            # 阈值：有效视差点足够多 + 中位视差不能太小 + 波动不能太大
            if median_disp < min_median_disp:
                return None
            if std_disp / max(1e-6, median_disp) > max_std_ratio:
                return None

            focal_length = float(camera_config.P1[0, 0])
            baseline_m = float(np.linalg.norm(camera_config.T)) / 1000.0
            distance_m = (focal_length * baseline_m) / median_disp

            # 超出工作范围通常是视差不稳定导致，判为无效。
            if distance_m < self.distance_min_m or distance_m > self.distance_max_m:
                return None
            return distance_m

        # 先严格算一次
        d1 = _calc_once(
            min_valid_ratio=self.min_valid_ratio,
            max_std_ratio=self.max_std_ratio,
            min_median_disp=self.min_median_disp,
            roi_expand_ratio=self.roi_expand_ratio,
            center_crop_ratio=None
        )
        if d1 is not None:
            return d1

        # 再宽松算一次：更低的有效占比、更高的允许波动、更低的最小视差 + 更大 ROI + 只取中心区域
        roi_expand = max(self.roi_expand_ratio, 0.35)
        d2 = _calc_once(
            min_valid_ratio=self.relax_valid_ratio,
            max_std_ratio=self.relax_max_std_ratio,
            min_median_disp=self.relax_min_median_disp,
            roi_expand_ratio=roi_expand,
            center_crop_ratio=0.6
        )
        return d2

    def _publish_update(self, frame_ts, detected, distance_m, detections, display_frame, left_frame):
        if self.event_bus is None:
            return
        event = RangingUpdateEvent(
            frame_ts=frame_ts,
            detected=detected,
            distance_m=distance_m,
            detections=list(detections) if detections is not None else [],
            display_frame=display_frame,
            left_frame=left_frame
        )
        self.event_bus.publish(event)

    def _camera_loop(self):
        print("摄像头线程启动...")
        print(f"尝试打开摄像头设备 /dev/video{self.camera_device} ...")
        cap_local = cv2.VideoCapture(self.camera_device, cv2.CAP_V4L2)
        if not cap_local.isOpened():
            print(f"错误: 无法打开摄像头设备 /dev/video{self.camera_device}")
            self.camera_active = False
            return

        cap_local.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap_local.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        ret, test_frame = cap_local.read()
        if not ret or test_frame is None:
            print("错误: 摄像头打开但无法读取帧")
            cap_local.release()
            self.camera_active = False
            return

        h0, w0 = test_frame.shape[:2]
        print(f"摄像头打开成功，当前帧分辨率: {w0}x{h0}")
        if w0 != self.frame_width or h0 != self.frame_height:
            print("警告: 摄像头分辨率与预期不一致，但继续尝试运行。")

        self.cap = cap_local
        stereo = self._create_stereo_sgbm()

        right_matcher = None
        wls_filter = None
        if self.use_wls:
            if ximgproc is None:
                print("警告: OpenCV ximgproc 不可用，无法启用 WLS 视差滤波，将退回 medianBlur。")
            else:
                try:
                    right_matcher = ximgproc.createRightMatcher(stereo)
                    wls_filter = ximgproc.createDisparityWLSFilter(matcher_left=stereo)
                    wls_filter.setLambda(float(self.wls_lambda))
                    wls_filter.setSigmaColor(float(self.wls_sigma))
                    print(f"WLS 视差滤波已启用: lambda={self.wls_lambda}, sigma={self.wls_sigma}")
                except Exception as exc:
                    print(f"警告: 启用 WLS 失败，将退回 medianBlur: {exc}")
                    right_matcher = None
                    wls_filter = None

        print("摄像头初始化完成，开始采集...")

        frame_count = 0
        last_distance = None
        last_distance_ts = 0.0
        detector_instance = None
        try:
            detector_instance = detector.RknnDetector()
            print("RKNN 检测器初始化完成")
        except Exception as exc:
            print(f"RKNN 检测器初始化失败: {exc}")

        while self.camera_active:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("警告: 无法读取摄像头帧")
                time.sleep(0.05)
                continue
            frame_ts = time.time()

            left_frame = frame[self.left_roi[1]:self.left_roi[1] + self.left_roi[3],
                               self.left_roi[0]:self.left_roi[0] + self.left_roi[2]]
            right_frame = frame[self.right_roi[1]:self.right_roi[1] + self.right_roi[3],
                                self.right_roi[0]:self.right_roi[0] + self.right_roi[2]]

            left_rectified = cv2.remap(left_frame, camera_config.left_map1, camera_config.left_map2,
                                       cv2.INTER_LINEAR)
            right_rectified = cv2.remap(right_frame, camera_config.right_map1, camera_config.right_map2,
                                        cv2.INTER_LINEAR)

            img1_rectified = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
            img2_rectified = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

            if wls_filter is not None and right_matcher is not None:
                disp_left = stereo.compute(img1_rectified, img2_rectified)
                disp_right = right_matcher.compute(img2_rectified, img1_rectified)
                disp_wls = wls_filter.filter(disp_left, img1_rectified, None, disp_right)
                disparity_filtered = disp_wls.astype(np.float32) / 16.0
                disparity_filtered[disparity_filtered <= 0] = 0
                # 轻度去噪，避免把边缘抹得太干净
                disparity_filtered = cv2.medianBlur(disparity_filtered, 5)
            else:
                disparity_raw = stereo.compute(img1_rectified, img2_rectified).astype(np.float32) / 16.0
                disparity_filtered = cv2.medianBlur(disparity_raw, 5)

            detected = False
            distance = None
            display_frame = left_rectified.copy()
            detections = []

            if detector_instance is not None:
                boxes, classes, scores = detector_instance.infer(left_rectified)
                if boxes is not None and len(boxes) > 0:
                    pixel_boxes = detector.scale_boxes(left_rectified.shape, boxes)
                    detector.draw(display_frame, boxes, scores, classes)
                    best_idx = int(np.argmax(scores))
                    best_score = float(scores[best_idx])
                    if best_score >= detector.DETECT_SCORE_MIN:
                        detected = True
                        left, top, right, bottom = pixel_boxes[best_idx]
                        distance = self._calculate_distance_for_box(
                            disparity_filtered, left, top, right, bottom
                        )
                    for (left, top, right, bottom), score, class_id in zip(pixel_boxes, scores, classes):
                        detections.append({
                            "left": int(left),
                            "top": int(top),
                            "right": int(right),
                            "bottom": int(bottom),
                            "score": float(score),
                            "class_id": int(class_id),
                            "label": detector.CLASSES[int(class_id)]
                        })

            distance_held = False
            now_ts = time.time()

            if detected and distance is not None:
                if last_distance is None:
                    smooth_distance = distance
                else:
                    smooth_distance = self.smooth_alpha * distance + (1.0 - self.smooth_alpha) * last_distance
                last_distance = smooth_distance
                last_distance_ts = now_ts

            elif detected and last_distance is not None and (now_ts - last_distance_ts) <= self.hold_last_seconds:
                # 桶仍被检测到，但某一帧视差不稳定：短时间沿用上一帧有效距离，减少 N/A
                smooth_distance = last_distance
                distance_held = True

            elif (not detected) and last_distance is not None and (now_ts - last_distance_ts) <= self.hold_lost_seconds:
                # 检测短暂丢失（1~2 帧）也沿用上一帧距离，避免频繁 N/A/No bucket 抖动
                smooth_distance = last_distance
                distance_held = True

            else:
                smooth_distance = None
                last_distance = None

            if self.render_on_device:
                if smooth_distance is not None and (detected or distance_held):
                    text = f"Distance: {smooth_distance:.3f} m"
                    # 沿用上一帧距离时用黄色提示
                    color_fg = (0, 255, 255) if distance_held else (0, 255, 0)
                elif detected:
                    text = "Measuring..."  # 不显示 N/A，避免干扰
                    color_fg = (0, 255, 255)
                else:
                    text = "No bucket"
                    color_fg = (0, 0, 255)

                cv2.putText(display_frame, text, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(display_frame, text, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_fg, 2, cv2.LINE_AA)

            with self.data_lock:
                frame_copy = frame.copy()
                left_frame_copy = left_rectified.copy()
                disparity_copy = disparity_filtered.copy()
                display_frame_copy = display_frame.copy()
                self.latest_frame = frame_copy
                self.latest_left_frame = left_frame_copy
                self.latest_disparity = disparity_copy
                self.latest_distance = smooth_distance
                self.latest_display_frame = display_frame_copy
                self.latest_detected = detected
                self.latest_detections = detections
                self.latest_frame_ts = frame_ts
            self._publish_update(
                frame_ts=frame_ts,
                detected=detected,
                distance_m=smooth_distance,
                detections=detections,
                display_frame=display_frame_copy,
                left_frame=left_frame_copy
            )

            frame_count += 1
            if frame_count % 30 == 0:
                if smooth_distance is None:
                    dist_text = "N/A"
                else:
                    dist_text = f"{smooth_distance:.3f} m"
                print(f"[{frame_count} 帧] 距离 = {dist_text} (原始: {distance if distance is not None else 'None'})")

            time.sleep(1.0 / 30.0)

        if self.cap is not None:
            self.cap.release()
            print("摄像头已释放")

    def _display_loop(self):
        if not self.enable_display:
            print("显示功能已禁用（ENABLE_DISPLAY=False），仅运行 Flask 服务器。")
            try:
                while self.camera_active:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                self.camera_active = False
            return

        print("显示线程启动...")
        try:
            cv2.namedWindow('Left Camera', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('Right Camera', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('Disparity Map', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('Ranging Display', cv2.WINDOW_AUTOSIZE)

            while self.camera_active:
                with self.data_lock:
                    frame = self.latest_frame
                    disparity = self.latest_disparity
                    display_frame = self.latest_display_frame

                if frame is not None:
                    left_frame = frame[self.left_roi[1]:self.left_roi[1] + self.left_roi[3],
                                       self.left_roi[0]:self.left_roi[0] + self.left_roi[2]]
                    right_frame = frame[self.right_roi[1]:self.right_roi[1] + self.right_roi[3],
                                        self.right_roi[0]:self.right_roi[0] + self.right_roi[2]]
                    cv2.imshow('Left Camera', left_frame)
                    cv2.imshow('Right Camera', right_frame)

                if display_frame is not None:
                    cv2.imshow('Ranging Display', display_frame)

                if disparity is not None:
                    disp_norm = cv2.normalize(disparity, None, alpha=0, beta=255,
                                              norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
                    cv2.imshow('Disparity Map', disp_color)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.camera_active = False
                    break

                time.sleep(0.03)

            cv2.destroyAllWindows()
            print("显示窗口已关闭。")

        except Exception as exc:
            print(f"显示线程异常（可能无显示环境）：{exc}")
            while self.camera_active:
                time.sleep(0.5)

    def start(self):
        self.camera_active = True
        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._camera_thread.start()
        time.sleep(2.0)
        self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self._display_thread.start()
        if self.enable_push and self.video_push_url:
            self._video_push_thread = threading.Thread(target=self._video_push_loop, daemon=True)
            self._video_push_thread.start()
        if self.enable_push and self.meta_push_url:
            self._meta_push_thread = threading.Thread(target=self._meta_push_loop, daemon=True)
            self._meta_push_thread.start()

    def stop(self):
        self.camera_active = False

    def is_active(self):
        return self.camera_active

    def get_latest_display_frame(self):
        with self.data_lock:
            if self.latest_display_frame is None:
                return None
            return self.latest_display_frame.copy()

    def get_latest_left_frame(self):
        with self.data_lock:
            if self.latest_left_frame is None:
                return None
            return self.latest_left_frame.copy()

    def get_latest_distance(self):
        with self.data_lock:
            return self.latest_distance

    def get_latest_detected(self):
        with self.data_lock:
            return self.latest_detected

    def _post_json(self, url, payload):
        data = json.dumps(payload, ensure_ascii=True).encode('utf-8')
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.push_timeout) as resp:
            resp.read()

    def _post_jpeg(self, url, jpeg_bytes, frame_ts):
        headers = {
            "Content-Type": "image/jpeg",
            "X-Frame-Timestamp": f"{frame_ts:.6f}"
        }
        req = urllib.request.Request(url, data=jpeg_bytes, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.push_timeout) as resp:
            resp.read()

    def _video_push_loop(self):
        print("视频推送线程启动...")
        interval = 1.0 / max(1, float(self.push_fps))
        next_ts = time.monotonic()
        while self.camera_active:
            now = time.monotonic()
            if now < next_ts:
                time.sleep(max(0.0, next_ts - now))
                continue
            next_ts = time.monotonic() + interval

            frame = None
            frame_ts = None
            with self.data_lock:
                if self.latest_left_frame is not None:
                    frame = self.latest_left_frame.copy()
                    frame_ts = self.latest_frame_ts
            if frame is None or frame_ts is None:
                time.sleep(0.05)
                continue

            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, int(self.push_jpeg_quality)])
            if not ret:
                continue
            try:
                self._post_jpeg(self.video_push_url, jpeg.tobytes(), frame_ts)
            except Exception:
                time.sleep(0.2)

    def _meta_push_loop(self):
        print("检测结果推送线程启动...")
        interval = 1.0 / max(1, float(self.push_fps))
        next_ts = time.monotonic()
        while self.camera_active:
            now = time.monotonic()
            if now < next_ts:
                time.sleep(max(0.0, next_ts - now))
                continue
            next_ts = time.monotonic() + interval

            with self.data_lock:
                payload = {
                    "ts": self.latest_frame_ts,
                    "detected": self.latest_detected,
                    "distance_m": self.latest_distance,
                    "detections": list(self.latest_detections)
                }
            if payload["ts"] is None:
                time.sleep(0.05)
                continue
            try:
                self._post_json(self.meta_push_url, payload)
            except Exception:
                time.sleep(0.2)
