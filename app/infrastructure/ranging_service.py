#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双目测距核心模块
- 负责摄像头采集、视差计算、距离估计
- 提供线程安全的数据访问接口
"""

import cv2
import os
import json
import numpy as np
import re
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

    def __init__(
        self,
        camera_device="auto",
        frame_width=2560,
        frame_height=720,
        camera_fourcc="MJPG",
        camera_fps=30.0,
        enable_display=False,
        render_on_device=False,
        cache_display_frame=True,
        enable_push=False,
        video_push_url=None,
        meta_push_url=None,
        push_fps=8,
        push_timeout=1.0,
        push_jpeg_quality=85,
        event_bus=None,
        device_id="robot001",
        enable_webrtc=False,
        webrtc_signal_url=None,
        webrtc_peer_id="browser",
        webrtc_fps=15,
        webrtc_stun_urls=None,
        use_wls=True,
        wls_lambda=8000.0,
        wls_sigma=1.5,
        sgbm_num_disparities=160,
        sgbm_block_size=7,
        sgbm_uniqueness_ratio=12,
        sgbm_speckle_window_size=120,
        sgbm_speckle_range=2,
        disparity_mode="full",
        disparity_interval=8,
        disparity_full_fallback=True,
        disparity_roi_expand_ratio=0.6,
        disparity_scale=1.0,
        async_capture=False,
        async_distance=False,
        loop_sleep_seconds=0.0,
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
        enable_debug=False,
    ):
        self.camera_device = camera_device
        self.camera_device_index = None
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.camera_fourcc = (camera_fourcc or "").strip().upper()
        self.camera_fps = max(0.0, float(camera_fps))
        self.enable_display = enable_display
        self.render_on_device = render_on_device
        self.cache_display_frame = cache_display_frame
        self.enable_push = enable_push
        self.video_push_url = video_push_url
        self.meta_push_url = meta_push_url
        self.push_fps = push_fps
        self.push_timeout = push_timeout
        self.push_jpeg_quality = push_jpeg_quality
        self.event_bus = event_bus
        self.device_id = device_id
        self.enable_webrtc = enable_webrtc
        self.webrtc_signal_url = webrtc_signal_url
        self.webrtc_peer_id = webrtc_peer_id or "browser"
        self.webrtc_fps = webrtc_fps
        self.webrtc_stun_urls = [u for u in (webrtc_stun_urls or []) if u]
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
        self.disparity_mode = (disparity_mode or "roi").strip().lower()
        if self.disparity_mode not in {"roi", "full"}:
            self.disparity_mode = "roi"
        self.disparity_interval = max(1, int(disparity_interval))
        self.disparity_full_fallback = bool(disparity_full_fallback)
        self.disparity_roi_expand_ratio = max(0.0, float(disparity_roi_expand_ratio))
        self.disparity_scale = min(1.0, max(0.25, float(disparity_scale)))
        self.async_capture = bool(async_capture)
        self.async_distance = bool(async_distance)
        self.loop_sleep_seconds = max(0.0, float(loop_sleep_seconds))

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
        self.latest_frame_id = 0
        self.latest_loop_fps = None
        self.latest_profile_ms = None
        self.latest_metadata = None
        self.data_lock = threading.Lock()
        self._http_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        self._camera_thread = None
        self._display_thread = None
        self._video_push_thread = None
        self._meta_push_thread = None
        self._webrtc_client = None

    def get_camera_device_label(self):
        value = (
            self.camera_device_index
            if self.camera_device_index is not None
            else self.camera_device
        )
        if isinstance(value, int):
            return f"/dev/video{value}"
        raw = str(value).strip()
        if raw.isdigit():
            return f"/dev/video{raw}"
        return raw or "auto"

    def _parse_camera_device(self):
        raw = self.camera_device
        if raw is None:
            return None
        if isinstance(raw, int):
            return raw
        value = str(raw).strip()
        if value.lower() in {"", "auto", "-1", "none"}:
            return None
        match = re.search(r"/dev/video(\d+)$", value)
        if match:
            return int(match.group(1))
        try:
            return int(value)
        except ValueError:
            return None

    def _video_device_name(self, index):
        try:
            with open(
                f"/sys/class/video4linux/video{index}/name", "r", encoding="utf-8"
            ) as f:
                return f.read().strip()
        except OSError:
            return ""

    def _auto_camera_candidates(self):
        indices = []
        for entry in os.listdir("/dev"):
            match = re.fullmatch(r"video(\d+)", entry)
            if match:
                indices.append(int(match.group(1)))
        indices.sort()

        def priority(index):
            name = self._video_device_name(index).lower()
            if "usb" in name and "camera" in name:
                return 0
            if "camera" in name:
                return 1
            return 2

        return sorted(indices, key=lambda idx: (priority(idx), idx))

    def _open_camera_index(self, index, strict_size=False):
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            return None, None

        if len(self.camera_fourcc) >= 4:
            fourcc = cv2.VideoWriter_fourcc(*self.camera_fourcc[:4])
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        if self.camera_fps > 0:
            cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            cap.release()
            return None, None

        h0, w0 = test_frame.shape[:2]
        if strict_size and (w0 != self.frame_width or h0 != self.frame_height):
            print(
                f"跳过 /dev/video{index}: 分辨率 {w0}x{h0} 不匹配预期 {self.frame_width}x{self.frame_height}"
            )
            cap.release()
            return None, None

        return cap, test_frame

    def _open_camera_capture(self):
        configured_index = self._parse_camera_device()
        if configured_index is not None:
            cap, test_frame = self._open_camera_index(
                configured_index, strict_size=False
            )
            return configured_index, cap, test_frame

        for index in self._auto_camera_candidates():
            cap, test_frame = self._open_camera_index(index, strict_size=True)
            if cap is not None:
                print(
                    f"自动选择摄像头: /dev/video{index} ({self._video_device_name(index) or 'unknown'})"
                )
                return index, cap, test_frame
        return None, None, None

    def _create_stereo_sgbm(self, num_disparities=None):
        # numDisparities 必须是 16 的倍数
        if num_disparities is None:
            num_disparities = self.sgbm_num_disparities
        num_disparities = int(max(16, round(float(num_disparities) / 16) * 16))
        block_size = int(max(3, self.sgbm_block_size))
        if block_size % 2 == 0:
            block_size += 1

        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * 3 * block_size**2,
            P2=32 * 3 * block_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=int(self.sgbm_uniqueness_ratio),
            speckleWindowSize=int(self.sgbm_speckle_window_size),
            speckleRange=int(self.sgbm_speckle_range),
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        return stereo

    def _compute_disparity(
        self,
        stereo,
        left_gray,
        right_gray,
        wls_filter=None,
        right_matcher=None,
        scale=1.0,
    ):
        scale = min(1.0, max(0.25, float(scale)))
        orig_h, orig_w = left_gray.shape[:2]
        if scale < 0.999:
            scaled_w = max(1, int(round(orig_w * scale)))
            scaled_h = max(1, int(round(orig_h * scale)))
            left_gray = cv2.resize(
                left_gray, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA
            )
            right_gray = cv2.resize(
                right_gray, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA
            )
            wls_filter = None
            right_matcher = None

        if wls_filter is not None and right_matcher is not None:
            disp_left = stereo.compute(left_gray, right_gray)
            disp_right = right_matcher.compute(right_gray, left_gray)
            disp_wls = wls_filter.filter(disp_left, left_gray, None, disp_right)
            disparity = disp_wls.astype(np.float32) / 16.0
            disparity[disparity <= 0] = 0
        else:
            disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

        if scale < 0.999:
            disparity = disparity / scale
            disparity = cv2.resize(
                disparity, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
            )

        disparity[disparity <= 0] = 0
        return cv2.medianBlur(disparity, 5)

    def _disparity_roi(self, image_shape, box):
        h, w = image_shape[:2]
        left, top, right, bottom = box
        bw = max(1, right - left)
        bh = max(1, bottom - top)
        num_disparities = int(max(16, round(self.sgbm_num_disparities / 16) * 16))
        block_pad = int(max(16, self.sgbm_block_size * 4))
        x_margin = max(
            num_disparities + block_pad, int(bw * self.disparity_roi_expand_ratio)
        )
        y_margin = max(block_pad, int(bh * self.disparity_roi_expand_ratio))

        x1 = max(0, left - x_margin)
        x2 = min(w, right + x_margin)
        y1 = max(0, top - y_margin)
        y2 = min(h, bottom + y_margin)

        if x2 - x1 < num_disparities + block_pad or y2 - y1 < block_pad * 2:
            return None

        local_box = (left - x1, top - y1, right - x1, bottom - y1)
        return (x1, y1, x2, y2), local_box

    def _calculate_distance_for_box(self, disparity_map, left, top, right, bottom):
        """根据检测框在视差图上估计距离。

        目标：在保证不乱跳的前提下，尽量减少 N/A。
        做法：先用严格阈值算一次；失败后再用更宽松阈值 + 更大 ROI/中心裁剪再算一次。
        """
        if disparity_map is None:
            return None

        h, w = disparity_map.shape

        def _calc_once(
            min_valid_ratio,
            max_std_ratio,
            min_median_disp,
            roi_expand_ratio,
            center_crop_ratio=None,
        ):
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
            center_crop_ratio=None,
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
            center_crop_ratio=0.6,
        )
        return d2

    def _publish_update(
        self,
        frame_id,
        frame_ts,
        detected,
        distance_m,
        detections,
        display_frame,
        left_frame,
        image_width=None,
        image_height=None,
    ):
        if self.event_bus is None:
            return
        event = RangingUpdateEvent(
            frame_id=frame_id,
            frame_ts=frame_ts,
            detected=detected,
            distance_m=distance_m,
            detections=list(detections) if detections is not None else [],
            image_width=image_width,
            image_height=image_height,
            display_frame=display_frame,
            left_frame=left_frame,
        )
        self.event_bus.publish(event)

    def _format_detections_for_meta(self, detections):
        formatted = []
        for item in detections or []:
            formatted.append(
                {
                    "label": item.get("label"),
                    "score": item.get("score"),
                    "class_id": item.get("class_id"),
                    "bbox": {
                        "left": item.get("left"),
                        "top": item.get("top"),
                        "right": item.get("right"),
                        "bottom": item.get("bottom"),
                    },
                    "left": item.get("left"),
                    "top": item.get("top"),
                    "right": item.get("right"),
                    "bottom": item.get("bottom"),
                }
            )
        return formatted

    def _build_metadata_payload(
        self,
        frame_id,
        frame_ts,
        detected,
        distance_m,
        detections,
        image_shape,
        loop_fps=None,
        profile_ms=None,
        distance_source=None,
        distance_held=False,
    ):
        height, width = image_shape[:2]
        return {
            "deviceID": self.device_id,
            "frame_id": frame_id,
            "frame_ts": frame_ts,
            "loop_fps": loop_fps,
            "profile_ms": profile_ms or {},
            "image": {
                "width": int(width),
                "height": int(height),
                "source": "left_rectified",
            },
            "detected": bool(detected),
            "distance_m": distance_m,
            "distance_source": distance_source or "none",
            "distance_held": bool(distance_held),
            "detections": self._format_detections_for_meta(detections),
        }

    def _send_webrtc_meta(self, payload):
        client = self._webrtc_client
        if client is not None:
            client.send_meta(payload)

    def _camera_loop(self):
        print("摄像头线程启动...")
        print(f"尝试打开摄像头设备 {self.get_camera_device_label()} ...")
        camera_index, cap_local, test_frame = self._open_camera_capture()
        if cap_local is None or test_frame is None:
            print(f"错误: 无法打开摄像头设备 {self.get_camera_device_label()}")
            self.camera_active = False
            return
        self.camera_device_index = camera_index

        h0, w0 = test_frame.shape[:2]
        print(f"摄像头打开成功: /dev/video{camera_index}，当前帧分辨率: {w0}x{h0}")
        actual_fourcc = int(cap_local.get(cv2.CAP_PROP_FOURCC))
        actual_fourcc_text = "".join(
            chr((actual_fourcc >> (8 * i)) & 0xFF) for i in range(4)
        )
        actual_fps = cap_local.get(cv2.CAP_PROP_FPS)
        print(f"摄像头格式: {actual_fourcc_text or 'unknown'}, FPS: {actual_fps:.2f}")
        if w0 != self.frame_width or h0 != self.frame_height:
            print("警告: 摄像头分辨率与预期不一致，但继续尝试运行。")

        self.cap = cap_local
        capture_state = None
        capture_condition = None
        capture_thread = None
        if self.async_capture:
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            capture_state = {
                "frame": test_frame,
                "ts": time.time(),
                "seq": 0,
            }
            capture_condition = threading.Condition()

            def _capture_latest_loop():
                while self.camera_active:
                    ret_capture, frame_capture = self.cap.read()
                    if not ret_capture or frame_capture is None:
                        print("警告: 无法读取摄像头帧")
                        time.sleep(0.02)
                        continue
                    with capture_condition:
                        capture_state["frame"] = frame_capture
                        capture_state["ts"] = time.time()
                        capture_state["seq"] += 1
                        capture_condition.notify()

            capture_thread = threading.Thread(target=_capture_latest_loop, daemon=True)
            capture_thread.start()

        stereo = self._create_stereo_sgbm()
        disparity_stereo = stereo
        if self.disparity_scale < 0.999:
            disparity_stereo = self._create_stereo_sgbm(
                self.sgbm_num_disparities * self.disparity_scale
            )

        right_matcher = None
        wls_filter = None
        if self.use_wls:
            if ximgproc is None:
                print(
                    "警告: OpenCV ximgproc 不可用，无法启用 WLS 视差滤波，将退回 medianBlur。"
                )
            else:
                try:
                    right_matcher = ximgproc.createRightMatcher(stereo)
                    wls_filter = ximgproc.createDisparityWLSFilter(matcher_left=stereo)
                    wls_filter.setLambda(float(self.wls_lambda))
                    wls_filter.setSigmaColor(float(self.wls_sigma))
                    print(
                        f"WLS 视差滤波已启用: lambda={self.wls_lambda}, sigma={self.wls_sigma}"
                    )
                except Exception as exc:
                    print(f"警告: 启用 WLS 失败，将退回 medianBlur: {exc}")
                    right_matcher = None
                    wls_filter = None

        def _measure_distance_for_frame(
            left_rectified_arg,
            right_frame_arg,
            box,
            stereo_arg,
            wls_filter_arg=None,
            right_matcher_arg=None,
        ):
            distance_value = None
            disparity_value = None
            source_value = "none"

            if self.disparity_mode == "roi":
                roi = self._disparity_roi(left_rectified_arg.shape, box)
                if roi is not None:
                    (x1, y1, x2, y2), local_box = roi
                    left_roi_rectified = left_rectified_arg[y1:y2, x1:x2]
                    right_roi_rectified = cv2.remap(
                        right_frame_arg,
                        camera_config.right_map1[y1:y2, x1:x2],
                        camera_config.right_map2[y1:y2, x1:x2],
                        cv2.INTER_LINEAR,
                    )
                    left_gray = cv2.cvtColor(left_roi_rectified, cv2.COLOR_BGR2GRAY)
                    right_gray = cv2.cvtColor(right_roi_rectified, cv2.COLOR_BGR2GRAY)
                    disparity_value = self._compute_disparity(
                        stereo_arg,
                        left_gray,
                        right_gray,
                        wls_filter_arg,
                        right_matcher_arg,
                        scale=self.disparity_scale,
                    )
                    distance_value = self._calculate_distance_for_box(
                        disparity_value, *local_box
                    )
                    source_value = f"roi@{self.disparity_scale:g}"

            if distance_value is None and (
                self.disparity_mode == "full" or self.disparity_full_fallback
            ):
                right_rectified = cv2.remap(
                    right_frame_arg,
                    camera_config.right_map1,
                    camera_config.right_map2,
                    cv2.INTER_LINEAR,
                )
                img1_rectified = cv2.cvtColor(left_rectified_arg, cv2.COLOR_BGR2GRAY)
                img2_rectified = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
                disparity_value = self._compute_disparity(
                    stereo_arg,
                    img1_rectified,
                    img2_rectified,
                    wls_filter_arg,
                    right_matcher_arg,
                    scale=self.disparity_scale,
                )
                distance_value = self._calculate_distance_for_box(disparity_value, *box)
                source_name = (
                    "full" if self.disparity_mode == "full" else "full_fallback"
                )
                source_value = f"{source_name}@{self.disparity_scale:g}"

            return distance_value, disparity_value, source_value

        distance_condition = None
        distance_state = None
        distance_thread = None
        if self.async_distance:
            distance_condition = threading.Condition()
            distance_state = {
                "request": None,
                "result": None,
                "stop": False,
            }

            def _distance_worker_loop():
                worker_stereo = self._create_stereo_sgbm()
                worker_disparity_stereo = worker_stereo
                if self.disparity_scale < 0.999:
                    worker_disparity_stereo = self._create_stereo_sgbm(
                        self.sgbm_num_disparities * self.disparity_scale
                    )
                worker_right_matcher = None
                worker_wls_filter = None
                if self.use_wls and ximgproc is not None:
                    try:
                        worker_right_matcher = ximgproc.createRightMatcher(
                            worker_stereo
                        )
                        worker_wls_filter = ximgproc.createDisparityWLSFilter(
                            matcher_left=worker_stereo
                        )
                        worker_wls_filter.setLambda(float(self.wls_lambda))
                        worker_wls_filter.setSigmaColor(float(self.wls_sigma))
                    except Exception:
                        worker_right_matcher = None
                        worker_wls_filter = None

                while True:
                    with distance_condition:
                        while (
                            distance_state["request"] is None
                            and not distance_state["stop"]
                        ):
                            distance_condition.wait(timeout=0.2)
                        if distance_state["stop"]:
                            return
                        request = distance_state["request"]
                        distance_state["request"] = None

                    started_at = time.perf_counter()
                    try:
                        distance_value, disparity_value, source_value = (
                            _measure_distance_for_frame(
                                request["left_rectified"],
                                request["right_frame"],
                                request["box"],
                                worker_disparity_stereo,
                                worker_wls_filter,
                                worker_right_matcher,
                            )
                        )
                        error = None
                    except Exception as exc:
                        distance_value = None
                        disparity_value = None
                        source_value = "error"
                        error = str(exc)
                    compute_ms = (time.perf_counter() - started_at) * 1000.0

                    with distance_condition:
                        distance_state["result"] = {
                            "frame_id": request["frame_id"],
                            "frame_ts": request["frame_ts"],
                            "distance": distance_value,
                            "disparity": disparity_value,
                            "source": source_value,
                            "compute_ms": compute_ms,
                            "error": error,
                        }

            distance_thread = threading.Thread(
                target=_distance_worker_loop, daemon=True
            )
            distance_thread.start()
            print("异步测距线程已启用。")

        print("摄像头初始化完成，开始采集...")

        frame_count = 0
        loop_fps = None
        profile_ms = None
        profile_started_at = time.monotonic()
        profile_started_frame = 0
        profile_count = 0
        profile_totals = {
            "capture": 0.0,
            "rectify": 0.0,
            "disparity": 0.0,
            "distance_worker": 0.0,
            "detect": 0.0,
            "publish": 0.0,
            "loop": 0.0,
        }
        last_distance = None
        last_distance_ts = 0.0
        last_disparity = None
        last_distance_result_frame = -1
        last_capture_seq = -1
        detector_instance = None
        try:
            detector_instance = detector.RknnDetector()
            print(
                f"RKNN 检测器初始化完成: {detector_instance.model_path}, "
                f"core={getattr(detector_instance, 'core_mask_name', 'default')}"
            )
        except Exception as exc:
            print(f"RKNN 检测器初始化失败: {exc}")

        try:
            while self.camera_active:
                loop_started_at = time.perf_counter()
                frame = None
                frame_ts = None
                if self.async_capture:
                    with capture_condition:
                        if capture_state["seq"] == last_capture_seq:
                            capture_condition.wait(timeout=0.25)
                        if (
                            capture_state["seq"] != last_capture_seq
                            and capture_state["frame"] is not None
                        ):
                            frame = capture_state["frame"]
                            frame_ts = capture_state["ts"]
                            last_capture_seq = capture_state["seq"]
                else:
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        frame_ts = time.time()
                captured_at = time.perf_counter()
                if frame is None:
                    time.sleep(0.005)
                    continue
                frame_id = frame_count + 1

                left_frame = frame[
                    self.left_roi[1] : self.left_roi[1] + self.left_roi[3],
                    self.left_roi[0] : self.left_roi[0] + self.left_roi[2],
                ]
                right_frame = frame[
                    self.right_roi[1] : self.right_roi[1] + self.right_roi[3],
                    self.right_roi[0] : self.right_roi[0] + self.right_roi[2],
                ]

                left_rectified = cv2.remap(
                    left_frame,
                    camera_config.left_map1,
                    camera_config.left_map2,
                    cv2.INTER_LINEAR,
                )
                rectified_at = time.perf_counter()

                detected = False
                distance = None
                distance_source = "none"
                best_box = None
                disparity_filtered = None
                detections = []

                if detector_instance is not None:
                    boxes, classes, scores = detector_instance.infer(left_rectified)
                    if boxes is not None and len(boxes) > 0:
                        pixel_boxes = detector.scale_boxes(left_rectified.shape, boxes)
                        best_idx = int(np.argmax(scores))
                        best_score = float(scores[best_idx])
                        if best_score >= detector.DETECT_SCORE_MIN:
                            detected = True
                            best_box = pixel_boxes[best_idx]
                        for (left, top, right, bottom), score, class_id in zip(
                            pixel_boxes, scores, classes
                        ):
                            detections.append(
                                {
                                    "left": int(left),
                                    "top": int(top),
                                    "right": int(right),
                                    "bottom": int(bottom),
                                    "score": float(score),
                                    "class_id": int(class_id),
                                    "label": detector.CLASSES[int(class_id)],
                                }
                            )
                detected_at = time.perf_counter()

                should_measure_distance = (
                    (not self.async_distance)
                    and detected
                    and best_box is not None
                    and (
                        last_distance is None
                        or (frame_id - 1) % self.disparity_interval == 0
                    )
                )
                distance_worker_elapsed = 0.0
                if self.async_distance and detected and best_box is not None:
                    new_result = None
                    with distance_condition:
                        result = distance_state.get("result")
                        if (
                            result is not None
                            and result.get("frame_id", -1) > last_distance_result_frame
                        ):
                            new_result = result
                            last_distance_result_frame = int(result.get("frame_id", -1))

                    if new_result is not None:
                        distance = new_result.get("distance")
                        disparity_filtered = new_result.get("disparity")
                        distance_source = new_result.get("source") or "async"
                        distance_worker_elapsed = (
                            float(new_result.get("compute_ms") or 0.0) / 1000.0
                        )
                        if disparity_filtered is not None:
                            last_disparity = disparity_filtered
                        if new_result.get("error"):
                            print(f"异步测距失败: {new_result.get('error')}")

                    can_submit_distance = False
                    with distance_condition:
                        can_submit_distance = distance_state.get("request") is None
                    if can_submit_distance:
                        request = {
                            "frame_id": frame_id,
                            "frame_ts": frame_ts,
                            "left_rectified": left_rectified.copy(),
                            "right_frame": right_frame.copy(),
                            "box": tuple(best_box),
                        }
                        with distance_condition:
                            if distance_state.get(
                                "request"
                            ) is None and not distance_state.get("stop"):
                                distance_state["request"] = request
                                distance_condition.notify()

                if should_measure_distance:
                    distance, disparity_filtered, distance_source = (
                        _measure_distance_for_frame(
                            left_rectified,
                            right_frame,
                            best_box,
                            disparity_stereo,
                            wls_filter,
                            right_matcher,
                        )
                    )
                    if disparity_filtered is not None:
                        last_disparity = disparity_filtered
                elif detected and distance is None and distance_source == "none":
                    distance_source = (
                        "pending"
                        if self.async_distance and last_distance is None
                        else "held"
                    )
                elif not detected:
                    distance_source = "no_detection"
                disparity_at = time.perf_counter()

                display_frame = (
                    left_rectified.copy()
                    if (self.render_on_device or self.cache_display_frame)
                    else None
                )
                distance_held = False
                now_ts = time.time()

                if detected and distance is not None:
                    if last_distance is None:
                        smooth_distance = distance
                    else:
                        smooth_distance = (
                            self.smooth_alpha * distance
                            + (1.0 - self.smooth_alpha) * last_distance
                        )
                    last_distance = smooth_distance
                    last_distance_ts = now_ts

                elif (
                    detected
                    and last_distance is not None
                    and (now_ts - last_distance_ts) <= self.hold_last_seconds
                ):
                    # 桶仍被检测到，但某一帧视差不稳定：短时间沿用上一帧有效距离，减少 N/A
                    smooth_distance = last_distance
                    distance_held = True

                elif (
                    (not detected)
                    and last_distance is not None
                    and (now_ts - last_distance_ts) <= self.hold_lost_seconds
                ):
                    # 检测短暂丢失（1~2 帧）也沿用上一帧距离，避免频繁 N/A/No bucket 抖动
                    smooth_distance = last_distance
                    distance_held = True

                else:
                    smooth_distance = None
                    last_distance = None

                if self.render_on_device and display_frame is not None:
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

                    cv2.putText(
                        display_frame,
                        text,
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 0),
                        3,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        display_frame,
                        text,
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        color_fg,
                        2,
                        cv2.LINE_AA,
                    )

                with self.data_lock:
                    frame_copy = frame.copy() if self.enable_display else None
                    left_frame_copy = (
                        left_rectified.copy() if self.enable_display else left_rectified
                    )
                    disparity_copy = (
                        last_disparity.copy()
                        if (self.enable_display and last_disparity is not None)
                        else last_disparity
                    )
                    display_frame_copy = (
                        display_frame.copy() if display_frame is not None else None
                    )
                    metadata_payload = self._build_metadata_payload(
                        frame_id=frame_id,
                        frame_ts=frame_ts,
                        detected=detected,
                        distance_m=smooth_distance,
                        detections=detections,
                        image_shape=left_rectified.shape,
                        loop_fps=loop_fps,
                        profile_ms=profile_ms,
                        distance_source=distance_source,
                        distance_held=distance_held,
                    )
                    self.latest_frame = frame_copy
                    self.latest_left_frame = left_frame_copy
                    self.latest_disparity = disparity_copy
                    self.latest_distance = smooth_distance
                    self.latest_display_frame = display_frame_copy
                    self.latest_detected = detected
                    self.latest_detections = detections
                    self.latest_frame_ts = frame_ts
                    self.latest_frame_id = frame_id
                    self.latest_loop_fps = loop_fps
                    self.latest_profile_ms = profile_ms
                    self.latest_metadata = metadata_payload
                self._publish_update(
                    frame_id=frame_id,
                    frame_ts=frame_ts,
                    detected=detected,
                    distance_m=smooth_distance,
                    detections=detections,
                    display_frame=display_frame_copy,
                    left_frame=left_frame_copy,
                    image_width=left_rectified.shape[1],
                    image_height=left_rectified.shape[0],
                )
                self._send_webrtc_meta(metadata_payload)
                published_at = time.perf_counter()

                profile_totals["capture"] += captured_at - loop_started_at
                profile_totals["rectify"] += rectified_at - captured_at
                profile_totals["detect"] += detected_at - rectified_at
                profile_totals["disparity"] += disparity_at - detected_at
                profile_totals["distance_worker"] += distance_worker_elapsed
                profile_totals["publish"] += published_at - disparity_at
                profile_totals["loop"] += published_at - loop_started_at
                profile_count += 1

                profile_elapsed = time.monotonic() - profile_started_at
                if profile_elapsed >= 5.0 and profile_count > 0:
                    loop_fps = (frame_id - profile_started_frame) / profile_elapsed
                    profile_ms = {
                        key: round((value / profile_count) * 1000.0, 2)
                        for key, value in profile_totals.items()
                    }
                    profile_started_at = time.monotonic()
                    profile_started_frame = frame_id
                    profile_count = 0
                    for key in profile_totals:
                        profile_totals[key] = 0.0

                frame_count = frame_id
                if frame_count % 30 == 0:
                    if smooth_distance is None:
                        dist_text = "N/A"
                    else:
                        dist_text = f"{smooth_distance:.3f} m"
                    fps_text = f"{loop_fps:.2f}" if loop_fps is not None else "warming"
                    print(
                        f"[{frame_count} 帧] FPS = {fps_text}, 距离 = {dist_text} "
                        f"(原始: {distance if distance is not None else 'None'}, source: {distance_source}, "
                        f"profile_ms: {profile_ms or {}})"
                    )

                if self.loop_sleep_seconds > 0:
                    time.sleep(self.loop_sleep_seconds)
        finally:
            if detector_instance is not None:
                detector_instance.release()

        if distance_thread is not None:
            with distance_condition:
                distance_state["stop"] = True
                distance_condition.notify()
            distance_thread.join(timeout=1.0)

        self.camera_active = False
        if capture_thread is not None:
            capture_thread.join(timeout=1.0)
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
            cv2.namedWindow("Left Camera", cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow("Right Camera", cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow("Disparity Map", cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow("Ranging Display", cv2.WINDOW_AUTOSIZE)

            while self.camera_active:
                with self.data_lock:
                    frame = self.latest_frame
                    disparity = self.latest_disparity
                    display_frame = self.latest_display_frame

                if frame is not None:
                    left_frame = frame[
                        self.left_roi[1] : self.left_roi[1] + self.left_roi[3],
                        self.left_roi[0] : self.left_roi[0] + self.left_roi[2],
                    ]
                    right_frame = frame[
                        self.right_roi[1] : self.right_roi[1] + self.right_roi[3],
                        self.right_roi[0] : self.right_roi[0] + self.right_roi[2],
                    ]
                    cv2.imshow("Left Camera", left_frame)
                    cv2.imshow("Right Camera", right_frame)

                if display_frame is not None:
                    cv2.imshow("Ranging Display", display_frame)

                if disparity is not None:
                    disp_norm = cv2.normalize(
                        disparity,
                        None,
                        alpha=0,
                        beta=255,
                        norm_type=cv2.NORM_MINMAX,
                        dtype=cv2.CV_8U,
                    )
                    disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
                    cv2.imshow("Disparity Map", disp_color)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
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
        self._start_webrtc_client()
        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._camera_thread.start()
        time.sleep(2.0)
        if self.enable_display:
            self._display_thread = threading.Thread(
                target=self._display_loop, daemon=True
            )
            self._display_thread.start()
        else:
            print("OpenCV 本地显示已禁用。")
        if self.enable_push and self.video_push_url:
            self._video_push_thread = threading.Thread(
                target=self._video_push_loop, daemon=True
            )
            self._video_push_thread.start()
        if self.enable_push and self.meta_push_url:
            self._meta_push_thread = threading.Thread(
                target=self._meta_push_loop, daemon=True
            )
            self._meta_push_thread.start()

    def stop(self):
        self.camera_active = False
        if self._webrtc_client is not None:
            self._webrtc_client.stop()
            self._webrtc_client = None

    def _start_webrtc_client(self):
        if not self.enable_webrtc or not self.webrtc_signal_url:
            return
        try:
            from app.infrastructure.webrtc_push import WebRTCPushClient

            self._webrtc_client = WebRTCPushClient(
                service=self,
                signal_url=self.webrtc_signal_url,
                fps=self.webrtc_fps,
                stun_urls=self.webrtc_stun_urls,
                peer_id=self.webrtc_peer_id,
                from_id="python",
            )
            self._webrtc_client.start()
            print(
                f"WebRTC 推流已启用: {self.webrtc_signal_url} -> {self.webrtc_peer_id}"
            )
        except Exception as exc:
            self._webrtc_client = None
            print(f"WebRTC 推流启动失败: {exc}")

    def is_active(self):
        return self.camera_active

    def get_latest_display_frame(self):
        with self.data_lock:
            if self.latest_display_frame is None:
                return None
            return self.latest_display_frame.copy()

    def get_latest_left_frame(self):
        with self.data_lock:
            return self.latest_left_frame

    def get_latest_distance(self):
        with self.data_lock:
            return self.latest_distance

    def get_latest_detected(self):
        with self.data_lock:
            return self.latest_detected

    def get_latest_metadata(self):
        with self.data_lock:
            if self.latest_metadata is None:
                return None
            return dict(self.latest_metadata)

    def _post_json(self, url, payload):
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with self._http_opener.open(req, timeout=self.push_timeout) as resp:
            resp.read()

    def _post_jpeg(self, url, jpeg_bytes, frame_ts):
        headers = {"Content-Type": "image/jpeg", "X-Frame-Timestamp": f"{frame_ts:.6f}"}
        req = urllib.request.Request(
            url, data=jpeg_bytes, headers=headers, method="POST"
        )
        with self._http_opener.open(req, timeout=self.push_timeout) as resp:
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

            ret, jpeg = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(self.push_jpeg_quality)]
            )
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
                payload = (
                    dict(self.latest_metadata)
                    if self.latest_metadata is not None
                    else None
                )
            if payload is None or payload.get("frame_ts") is None:
                time.sleep(0.05)
                continue
            try:
                self._post_json(self.meta_push_url, payload)
            except Exception:
                time.sleep(0.2)
