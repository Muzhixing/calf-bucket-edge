#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RKNN 目标检测模块（bucket）

该模块实现了基于 RKNN（Rockchip Neural Network）的目标检测功能，
支持 YOLO 系列模型的推理和后处理，专门用于检测"bucket"（桶）目标。

主要功能：
- RKNN 模型加载和推理
- 图像预处理（resize、letterbox）
- 检测结果后处理（NMS、置信度过滤）
- 检测框绘制和可视化
"""

import os
import threading
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite


def _env_float(name, default):
    """读取浮点环境变量，非法值时回退默认值。"""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# ==================== 模型配置 ====================
DEFAULT_MODEL_ENV = "RKNN_MODEL_PATH"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
RKNN_MODEL = str(_PROJECT_ROOT / "model" / "bucket.rknn")  # 默认优先使用仓库内模型文件

# ==================== 类别配置 ====================
CLASSES = ['bucket']  # 类别名称列表，索引对应类别ID
TARGET_CLASS_IDS = [0]  # 仅保留指定类别ID（与 CLASSES 索引对应），空列表表示保留所有类别

# ==================== 检测阈值配置 ====================
# 依据训练评估曲线：F1 在 conf≈0.536 附近达到最佳，因此将阈值对齐到该位置
OBJ_THRESH = _env_float("RKNN_OBJ_THRESH", 0.536)  # 置信度阈值（用于初筛：obj_conf * class_conf）
NMS_THRESH = _env_float("RKNN_NMS_THRESH", 0.45)   # NMS（非极大值抑制）阈值
DETECT_SCORE_MIN = _env_float("RKNN_DETECT_SCORE_MIN", 0.536)  # 最终检测分数门槛

# ==================== 模型输入配置 ====================
MODEL_SIZE = (640, 640)  # 模型输入尺寸（宽，高），单位：像素
MODEL_LAYOUT = os.getenv("RKNN_MODEL_LAYOUT", "NCHW").upper()  # 当前 bucket.rknn 默认使用 NCHW

# ==================== 全局变量 ====================
color_palette = np.random.uniform(0, 255, size=(len(CLASSES), 3))  # 随机颜色表，用于绘制不同类别的检测框
_rknn_lock = threading.Lock()  # RKNN 推理锁，防止多线程并发访问导致错误


def infer(rknn, inp):
    """
    执行 RKNN 模型推理
    
    对输入数据进行预处理（确保连续内存布局），然后调用 RKNN 模型进行推理。
    使用线程锁确保推理过程的线程安全。
    
    Args:
        rknn: RKNNLite 实例，已加载并初始化模型
        inp: 输入数据，numpy 数组，形状为 (batch, height, width, channels)
            数据类型为 np.uint8 或 np.float32
    
    Returns:
        list: 模型输出列表，包含多个输出张量
    
    Raises:
        AssertionError: 如果输入维度不是4或数据类型不支持
    """
    # 关键：确保数组在内存中连续存储，RKNN 推理需要连续内存布局
    inp = np.ascontiguousarray(inp)
    
    # 输入验证：确保输入格式正确，避免底层崩溃
    assert inp.ndim == 4, f"输入维度应为4，实际为 {inp.ndim}"
    assert inp.dtype in (np.uint8, np.float32), f"输入数据类型应为 uint8 或 float32，实际为 {inp.dtype}"

    data_format = "nchw" if MODEL_LAYOUT == "NCHW" else "nhwc"

    # 关键：使用线程锁防止多线程并发访问导致推理错误
    with _rknn_lock:
        return rknn.inference(inputs=[inp], data_format=[data_format])


def filter_boxes(boxes, box_confidences, box_class_probs):
    """
    根据置信度阈值过滤检测框
    
    计算每个检测框的综合置信度（类别最大概率 × 目标置信度），
    仅保留置信度高于阈值的检测框。
    
    Args:
        boxes: 检测框数组，形状为 (N, 4)，格式为 (x1, y1, x2, y2)
        box_confidences: 目标置信度数组，形状为 (N,)
        box_class_probs: 类别概率数组，形状为 (N, num_classes)
    
    Returns:
        tuple: (过滤后的检测框, 类别ID数组, 置信度分数数组)
    """
    box_confidences = box_confidences.reshape(-1)

    # 获取每个检测框的最大类别概率和对应的类别ID
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    # 计算综合置信度：类别最大概率 × 目标置信度
    # 仅保留置信度高于阈值的检测框
    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]

    # 根据过滤条件提取对应的检测框和类别
    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    """
    非极大值抑制（Non-Maximum Suppression, NMS）
    
    去除重叠的检测框，保留置信度最高的检测框。
    当两个检测框的 IoU（交并比）超过阈值时，保留分数更高的检测框。
    
    Args:
        boxes: 检测框数组，形状为 (N, 4)，格式为 (x1, y1, x2, y2)
        scores: 置信度分数数组，形状为 (N,)
    
    Returns:
        np.ndarray: 保留的检测框索引数组
    """
    # 提取检测框的坐标和尺寸
    x = boxes[:, 0]  # 左上角 x 坐标
    y = boxes[:, 1]  # 左上角 y 坐标
    w = boxes[:, 2] - boxes[:, 0]  # 宽度
    h = boxes[:, 3] - boxes[:, 1]  # 高度

    # 计算每个检测框的面积
    areas = w * h
    # 按置信度分数降序排序，获取索引
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        # 选择置信度最高的检测框
        i = order[0]
        keep.append(i)

        # 计算当前检测框与其他检测框的交集区域
        xx1 = np.maximum(x[i], x[order[1:]])  # 交集左上角 x
        yy1 = np.maximum(y[i], y[order[1:]])  # 交集左上角 y
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])  # 交集右下角 x
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])  # 交集右下角 y

        # 计算交集区域的宽度和高度（避免除零，添加小值）
        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1  # 交集面积

        # 计算 IoU（交并比）：交集面积 / 并集面积
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留 IoU 小于阈值的检测框（重叠度低）
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]  # 更新待处理列表
    
    return np.array(keep)


def softmax(x, axis=None):
    """
    计算 Softmax 函数
    
    对输入数组应用 Softmax 归一化，将数值转换为概率分布。
    使用数值稳定的实现方式（减去最大值避免溢出）。
    
    Args:
        x: 输入数组
        axis: 计算 Softmax 的轴，None 表示对整个数组计算
    
    Returns:
        np.ndarray: Softmax 归一化后的数组，所有元素和为1
    """
    # 数值稳定：减去最大值，避免 exp 溢出
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def dfl(position):
    """
    Distribution Focal Loss (DFL) 解码函数
    
    将模型输出的分布表示转换为具体的坐标值。
    DFL 是 YOLOv8 等模型使用的边界框回归方法，通过预测分布而非直接预测坐标值。
    
    Args:
        position: 位置分布张量，形状为 (batch, channels, height, width)
                 channels 应为 4 * mc，其中 4 表示 4 个坐标（x1, y1, x2, y2），
                 mc 表示每个坐标的分布维度
    
    Returns:
        np.ndarray: 解码后的坐标张量，形状为 (batch, 4, height, width)
    """
    n, c, h, w = position.shape
    p_num = 4  # 4 个坐标值（x1, y1, x2, y2）
    mc = c // p_num  # 每个坐标的分布维度
    
    # 重塑为 (batch, 4, mc, height, width)
    y = position.reshape(n, p_num, mc, h, w)
    # 对分布维度应用 Softmax，转换为概率分布
    y = softmax(y, 2)
    # 创建累积矩阵 [0, 1, 2, ..., mc-1]，用于加权求和
    acc_metrix = np.array(range(mc), dtype=float).reshape(1, 1, mc, 1, 1)
    # 加权求和得到最终的坐标值
    y = (y * acc_metrix).sum(2)
    return y


def box_process(position):
    """
    处理边界框位置信息，将相对坐标转换为绝对坐标
    
    从模型输出中提取边界框信息，通过 DFL 解码和网格坐标转换，
    将相对位置转换为模型输入尺寸下的绝对坐标（xyxy 格式）。
    
    Args:
        position: 位置分布张量，形状为 (batch, channels, height, width)
                 channels 应为 4 * mc（mc 为分布维度）
    
    Returns:
        np.ndarray: 边界框坐标数组，形状为 (batch, 4, height, width)，
                   格式为 (x1, y1, x2, y2)，单位为像素
    """
    grid_h, grid_w = position.shape[2:4]  # 网格高度和宽度
    
    # 创建网格坐标：每个网格单元的中心点坐标
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)  # x 坐标网格
    row = row.reshape(1, 1, grid_h, grid_w)  # y 坐标网格
    grid = np.concatenate((col, row), axis=1)  # 合并为 (1, 2, h, w)
    
    # 计算每个网格单元对应的像素步长（stride）
    stride = np.array([MODEL_SIZE[1] // grid_h, MODEL_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)

    # 使用 DFL 解码位置分布，得到相对坐标偏移
    position = dfl(position)
    
    # 计算边界框的左上角和右下角坐标
    # 左上角：网格中心 - 相对偏移
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    # 右下角：网格中心 + 相对偏移
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    
    # 将网格坐标转换为像素坐标，并合并为 xyxy 格式
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

    return xyxy


def post_process(input_data):
    """
    后处理模型输出，提取检测结果
    
    处理模型的多尺度输出，进行边界框解码、置信度过滤、类别过滤和 NMS，
    最终返回检测到的目标框、类别和置信度。
    
    Args:
        input_data: 模型输出列表，包含多个尺度的输出张量
                   每个尺度包含位置信息和类别置信度信息
    
    Returns:
        tuple: (boxes, classes, scores) 或 (None, None, None)
            - boxes: 检测框数组，形状为 (N, 4)，格式为 (x1, y1, x2, y2)
            - classes: 类别ID数组，形状为 (N,)
            - scores: 置信度分数数组，形状为 (N,)
            如果没有检测到目标，返回 (None, None, None)
    """
    boxes, scores, classes_conf = [], [], []
    defualt_branch = 3  # 默认分支数（多尺度检测）
    pair_per_branch = len(input_data) // defualt_branch  # 每个分支的输出数量
    
    # 处理每个尺度的输出
    for i in range(defualt_branch):
        # 提取位置信息并解码为边界框坐标
        boxes.append(box_process(input_data[pair_per_branch * i]))
        # 提取类别置信度
        classes_conf.append(input_data[pair_per_branch * i + 1])
        # 创建分数数组（初始为全1，后续会与类别置信度结合）
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

    def sp_flatten(_in):
        """将张量从 (batch, channels, height, width) 展平为 (N, channels)"""
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)  # 转换为 (batch, height, width, channels)
        return _in.reshape(-1, ch)  # 展平为 (N, channels)

    # 展平所有尺度的输出
    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    # 合并所有尺度的输出
    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # 第一步过滤：根据置信度阈值过滤检测框
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # 第二步过滤：仅保留指定类别的检测框
    if TARGET_CLASS_IDS:
        keep = np.isin(classes, TARGET_CLASS_IDS)
        boxes = boxes[keep]
        classes = classes[keep]
        scores = scores[keep]
    
    # 第三步过滤：二次门槛过滤，再按最终分数过滤一次
    # 避免 OBJ_THRESH 调整后带来过多低分检测框
    if DETECT_SCORE_MIN is not None:
        keep = scores >= DETECT_SCORE_MIN
        boxes = boxes[keep]
        classes = classes[keep]
        scores = scores[keep]

    # 如果没有检测到目标，直接返回
    if boxes.size == 0:
        return None, None, None

    # 第四步：按类别分组进行 NMS
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        # 提取当前类别的所有检测框
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        # 对当前类别执行 NMS
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    # 如果 NMS 后没有剩余检测框，返回 None
    if not nclasses and not nscores:
        return None, None, None

    # 合并所有类别的检测结果
    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def resize_image(image, size, letterbox_image=True):
    """
    调整图像尺寸，支持 letterbox 模式
    
    将输入图像调整到指定尺寸。letterbox 模式会保持图像宽高比，
    在图像周围填充灰色区域（128），避免图像变形。
    
    Args:
        image: 输入图像，BGR 格式，形状为 (height, width, 3)
        size: 目标尺寸，元组 (width, height)
        letterbox_image: 是否使用 letterbox 模式，True 表示保持宽高比并填充，False 表示直接拉伸
    
    Returns:
        np.ndarray: 调整后的图像，形状为 (height, width, 3)，数据类型为 uint8
    """
    ih, iw, _ = image.shape  # 原始图像高度和宽度
    h, w = size  # 目标高度和宽度
    
    if letterbox_image:
        # 计算缩放比例，保持宽高比
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)  # 缩放后的宽度
        nh = int(ih * scale)  # 缩放后的高度
        
        # 缩放图像
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        
        # 创建灰色背景（128）并居中放置缩放后的图像
        image_back = np.ones((h, w, 3), dtype=np.uint8) * 128
        image_back[(h - nh) // 2: (h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw, :] = image
    else:
        # 直接拉伸到目标尺寸（不保持宽高比）
        image_back = image
    
    return image_back


def scale_boxes(image_shape, boxes):
    """
    将模型坐标映射回原图尺寸，返回像素坐标框列表
    
    由于模型输入使用了 letterbox 预处理（保持宽高比并填充），
    需要将模型输出的坐标（基于模型输入尺寸）转换回原始图像尺寸。
    同时处理坐标偏移（letterbox 填充导致的偏移）和边界裁剪。
    
    Args:
        image_shape: 原始图像形状，元组或数组，至少包含 (height, width)
        boxes: 检测框数组，形状为 (N, 4)，格式为 (x1, y1, x2, y2)，
               坐标基于模型输入尺寸（MODEL_SIZE）
    
    Returns:
        list: 像素坐标框列表，每个元素为 (left, top, right, bottom)，
              坐标已映射到原始图像尺寸，并进行了边界裁剪
    """
    img_h, img_w = image_shape[:2]  # 原始图像高度和宽度
    
    # 根据图像宽高比计算缩放因子和偏移量
    if img_w >= img_h:
        # 横向图像：宽度方向填满，高度方向有填充
        x_factor = img_w / MODEL_SIZE[0]  # x 方向缩放因子
        y_rsz = img_h * MODEL_SIZE[1] / img_w  # 缩放后的高度
        y_factor = img_h / y_rsz  # y 方向缩放因子
        offset = (MODEL_SIZE[1] - y_rsz) / 2  # y 方向偏移量（填充区域）
    else:
        # 纵向图像：高度方向填满，宽度方向有填充
        y_factor = img_h / MODEL_SIZE[1]  # y 方向缩放因子
        x_rsz = img_w * MODEL_SIZE[0] / img_h  # 缩放后的宽度
        x_factor = img_w / x_rsz  # x 方向缩放因子
        offset = (MODEL_SIZE[0] - x_rsz) / 2  # x 方向偏移量（填充区域）

    pixel_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = [int(_b) for _b in box]
        
        # 根据图像方向进行坐标转换
        if img_w >= img_h:
            # 横向图像：x 直接缩放，y 需要减去偏移后缩放
            left = int(x1 * x_factor)
            top = int((y1 - offset) * y_factor)
            right = int(x2 * x_factor)
            bottom = int((y2 - offset) * y_factor)
        else:
            # 纵向图像：x 需要减去偏移后缩放，y 直接缩放
            left = int((x1 - offset) * x_factor)
            top = int(y1 * y_factor)
            right = int((x2 - offset) * x_factor)
            bottom = int((y2 - offset) * y_factor)

        # 边界裁剪：确保坐标在图像范围内
        left = max(0, min(left, img_w - 1))
        right = max(0, min(right, img_w - 1))
        top = max(0, min(top, img_h - 1))
        bottom = max(0, min(bottom, img_h - 1))
        pixel_boxes.append((left, top, right, bottom))

    return pixel_boxes


def prepare_input(image_bgr):
    """
    根据模型布局准备输入张量。

    根据 RKNN_MODEL_LAYOUT 返回 NHWC 或 NCHW 的连续数组。
    """
    img = resize_image(image_bgr.copy(), MODEL_SIZE, True)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if MODEL_LAYOUT == "NCHW":
        img = np.transpose(img, (2, 0, 1))
    elif MODEL_LAYOUT != "NHWC":
        raise ValueError(f"不支持的模型输入布局: {MODEL_LAYOUT}")
    return np.expand_dims(np.ascontiguousarray(img), axis=0)


def draw_detections(img, left, top, right, bottom, score, class_id):
    """
    在图像上绘制单个检测结果
    
    绘制检测框、中心点、类别标签和置信度分数。
    
    Args:
        img: 输入图像（BGR 格式），会被直接修改
        left: 检测框左边界（像素）
        top: 检测框上边界（像素）
        right: 检测框右边界（像素）
        bottom: 检测框下边界（像素）
        score: 置信度分数（0-1）
        class_id: 类别ID，用于获取类别名称和颜色
    """
    # 获取该类别对应的颜色
    color = color_palette[class_id]
    
    # 绘制检测框（矩形）
    cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), color, 2)
    
    # 绘制中心点（红色实心圆）
    center_point = (int((left + right) / 2), int((top + bottom) / 2))
    cv2.circle(img, center_point, 5, (0, 0, 255), -1)
    
    # 准备标签文本：类别名称 + 置信度分数
    label = f"{CLASSES[class_id]}: {score:.2f}"
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    
    # 计算标签位置：优先放在检测框上方，如果空间不足则放在下方
    label_x = left
    label_y = top - 10 if top - 10 > label_height else top + 10
    
    # 绘制标签背景（填充矩形）
    cv2.rectangle(img, (label_x, label_y - label_height),
                  (label_x + label_width, label_y + label_height), color, cv2.FILLED)
    
    # 绘制标签文本（黑色文字）
    cv2.putText(img, label, (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw(image, boxes, scores, classes):
    """
    在图像上绘制所有检测结果
    
    将模型输出的检测框坐标映射到原始图像尺寸，然后在图像上绘制所有检测框。
    
    Args:
        image: 输入图像（BGR 格式），会被直接修改
        boxes: 检测框数组，形状为 (N, 4)，格式为 (x1, y1, x2, y2)，
               坐标基于模型输入尺寸
        scores: 置信度分数数组，形状为 (N,)
        classes: 类别ID数组，形状为 (N,)
    """
    # 将模型坐标映射回原图尺寸
    pixel_boxes = scale_boxes(image.shape, boxes)
    
    # 绘制每个检测结果
    for (left, top, right, bottom), score, cl in zip(pixel_boxes, scores, classes):
        draw_detections(image, left, top, right, bottom, score, cl)


class RknnDetector:
    """
    RKNN 目标检测器封装类
    
    封装了 RKNN 模型的加载、推理和资源释放功能，提供简洁的接口进行目标检测。
    
    Attributes:
        rknn: RKNNLite 实例，用于模型推理
    
    Example:
        >>> detector = RknnDetector()
        >>> boxes, classes, scores = detector.infer(image)
        >>> detector.release()
    """

    @staticmethod
    def resolve_model_path(model_path=None):
        """解析并校验 RKNN 模型路径。"""
        candidates = []
        if model_path:
            candidates.append(Path(model_path).expanduser())
        env_model_path = Path(os.environ[DEFAULT_MODEL_ENV]).expanduser() if DEFAULT_MODEL_ENV in os.environ else None
        if env_model_path is not None:
            candidates.append(env_model_path)
        candidates.extend([
            Path(RKNN_MODEL),
            _PROJECT_ROOT / "model" / "bucket.rknn",
            Path("/mnt/tfcard/work/calf/model/bucket.rknn"),
            Path("/userdata/project/calf-bucket-edge/model/bucket.rknn"),
        ])

        checked = []
        for candidate in candidates:
            resolved = candidate.resolve(strict=False)
            resolved_str = str(resolved)
            if resolved_str not in checked:
                checked.append(resolved_str)
            if resolved.is_file():
                return resolved_str

        checked_desc = ", ".join(checked)
        raise FileNotFoundError(
            f"未找到 RKNN 模型文件。请设置 {DEFAULT_MODEL_ENV} 或确认以下路径存在: {checked_desc}"
        )

    def __init__(self, model_path=RKNN_MODEL):
        """
        初始化检测器，加载并初始化 RKNN 模型
        
        Args:
            model_path: RKNN 模型文件路径，默认为全局配置的路径
        
        Raises:
            RuntimeError: 如果模型加载失败或运行时环境初始化失败
        """
        self.rknn = None
        resolved_model_path = self.resolve_model_path(model_path)
        self.model_path = resolved_model_path
        self.rknn = RKNNLite()
        
        # 加载 RKNN 模型文件
        ret = self.rknn.load_rknn(resolved_model_path)
        if ret != 0:
            self.release()
            raise RuntimeError(f"加载 RKNN 模型失败，错误代码: {ret}")
        
        # 初始化运行时环境
        ret = self.rknn.init_runtime()
        if ret != 0:
            self.release()
            raise RuntimeError(f"初始化运行时环境失败，错误代码: {ret}")

    def infer(self, image_bgr):
        """
        对输入图像进行目标检测
        
        对输入图像进行预处理（resize、letterbox），然后调用模型推理，
        最后进行后处理得到检测结果。
        
        Args:
            image_bgr: 输入图像，BGR 格式，形状为 (height, width, 3)
        
        Returns:
            tuple: (boxes, classes, scores) 或 (None, None, None)
                - boxes: 检测框数组，形状为 (N, 4)，格式为 (x1, y1, x2, y2)
                - classes: 类别ID数组，形状为 (N,)
                - scores: 置信度分数数组，形状为 (N,)
                如果没有检测到目标，返回 (None, None, None)
        """
        # 图像预处理：按模型要求组装输入张量
        input_data = prepare_input(image_bgr)
        # 模型推理
        outputs = infer(self.rknn, input_data)
        # 后处理：解码、过滤、NMS
        return post_process(outputs)

    def release(self):
        """
        释放 RKNN 资源
        
        释放模型占用的内存和资源，应在不再使用检测器时调用。
        """
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None

    def __del__(self):
        self.release()
