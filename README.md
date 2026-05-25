# Bucket 双目测距系统

Bucket 是一个面向嵌入式设备的双目视觉测距项目。系统从并排双目相机采集画面，使用 RKNN 模型检测 `bucket` 目标，并基于检测框区域内的视差估计目标距离；同时通过 Flask Web 页面展示实时画面、检测状态和最新测距结果。

当前 `main` 分支采用分层架构，并通过进程内异步事件总线把测距服务、状态缓存、Web 展示和串口预留层解耦。

## 功能概览

- 双目相机采集：默认读取 `/dev/video21`，期望输入分辨率为 `2560x720`。
- 左右图分割：默认将输入画面拆成左目 `1280x720` 和右目 `1280x720`。
- 双目标定与校正：使用 `app/infrastructure/camera_config.py` 中的内参、畸变参数和外参进行立体校正。
- 目标检测：使用 RKNNLite 加载 `bucket.rknn`，仅检测 `bucket` 类别。
- 距离估计：使用 StereoSGBM 计算视差，优先启用 OpenCV `ximgproc` WLS 滤波；不可用时退回 `medianBlur`。
- 稳定性处理：对检测框 ROI 进行有效视差筛选、百分位裁剪、MAD 离群剔除、严格/宽松两阶段估计、距离平滑和短时保持。
- Web 展示：Flask 提供 MJPEG 视频流、最新距离和检测状态接口。
- HTTP 推送：可选将左目 JPEG 帧与检测元数据 POST 到外部服务。
- GPS 工具：提供 NMEA GGA/RMC 解析、经纬度转局部 ENU 坐标等辅助能力。
- WebRTC 辅助文件：仓库包含 WebRTC 推送/观看相关代码，但主程序当前未直接接入 WebRTC 信令流程。

## 系统架构

```text
main.py
  └─ AppController
      ├─ AsyncEventBus
      ├─ RangingStateStore
      ├─ SerialPortGateway
      ├─ RangingService
      │   ├─ Camera capture
      │   ├─ Stereo rectification
      │   ├─ RKNN bucket detection
      │   ├─ Disparity and distance estimation
      │   └─ Optional HTTP push
      └─ Flask Web server
```

事件流：

```text
RangingService
  -> RangingUpdateEvent
  -> AsyncEventBus
  -> RangingStateStore
  -> Flask routes: /video, /distance, /status
```

分层职责：

| 层级 | 目录 | 职责 |
| --- | --- | --- |
| Domain | `app/domain/` | 定义领域事件，目前核心事件为 `RangingUpdateEvent`。 |
| Application | `app/application/` | 应用控制器、异步事件总线、线程安全状态缓存。 |
| Infrastructure | `app/infrastructure/` | 摄像头、双目测距、RKNN 推理、GPS、HTTP 推送、串口预留、WebRTC 辅助客户端。 |
| Presentation | `app/presentation/` | Flask Web 页面与 HTTP 接口。 |
| Tools / UI | `tools/`, `src/views/` | 本地 WebRTC 发送工具和 Vue WebRTC 观看页。 |

## 目录结构

```text
.
├── main.py                         # 主入口，装配事件总线、状态缓存、测距服务和 Web 服务
├── requirements.txt                # Python 依赖
├── model/
│   └── bucket.rknn                 # RKNN bucket 检测模型
├── app/
│   ├── domain/
│   │   └── events.py               # RangingUpdateEvent
│   ├── application/
│   │   ├── app_controller.py       # 应用生命周期与组件编排
│   │   ├── event_bus.py            # 异步事件总线
│   │   └── state_store.py          # 最新测距状态缓存
│   ├── infrastructure/
│   │   ├── camera_config.py        # 双目标定、校正和重映射参数
│   │   ├── detector.py             # RKNN 检测与 YOLO 后处理
│   │   ├── ranging_service.py      # 双目测距主服务
│   │   ├── gps.py                  # GPS NMEA 解析工具
│   │   ├── serial_port.py          # 串口输出占位层
│   │   └── webrtc_push.py          # WebRTC 推送客户端辅助实现
│   └── presentation/
│       └── web.py                  # Flask 页面和接口
├── tools/
│   └── webrtc_sender.py            # 本地摄像头 WebRTC 发送模拟工具
├── src/views/
│   └── WebRTCViewer.vue            # Vue WebRTC 观看端页面
└── api-rules.md                    # 外部 API 数据格式草案
```

## 运行环境

建议运行环境：

- Python 3。
- Rockchip/RKNNLite 可运行环境，用于加载 `model/bucket.rknn`。
- 支持 V4L2 的双目相机设备，默认设备号 `/dev/video21`。
- OpenCV，推荐包含 `ximgproc` 的 `opencv-contrib-python`，否则 WLS 视差滤波会自动降级。
- 可访问板卡 IP 的浏览器，用于查看 Flask 页面。

安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` 中没有默认启用 `rknnlite` 安装行。如果目标板卡不能通过 pip 直接安装，请使用芯片厂商提供的 RKNNLite wheel 包手动安装。

## 模型文件

默认模型文件位于：

```text
model/bucket.rknn
```

`RknnDetector.resolve_model_path()` 会检查传入路径、仓库内模型路径、`RKNN_MODEL_PATH` 环境变量以及常见部署路径：

```text
/mnt/tfcard/work/calf/model/bucket.rknn
/userdata/project/calf-bucket-edge/model/bucket.rknn
```

如果需要在部署环境中使用其他模型路径，可以设置：

```bash
export RKNN_MODEL_PATH=/path/to/bucket.rknn
```

注意：当前默认构造会优先使用仓库内 `model/bucket.rknn`。如果该文件存在，通常无需设置 `RKNN_MODEL_PATH`。

## 启动方式

```bash
python main.py
```

启动后程序会把工作目录切到项目根目录，并打印类似信息：

```text
CWD = /path/to/bucket
双目测距系统启动（分层架构 + 事件驱动）
摄像头设备: /dev/video21
分辨率: 2560x720
Flask: http://<板卡IP>:5050/
```

浏览器访问：

```text
http://<板卡IP>:5050/
```

默认 Web 服务监听 `0.0.0.0:5050`，可在 `AppController` 初始化参数中调整 `web_host` 和 `web_port`。

## Web 接口

| 路径 | 方法 | 返回 | 说明 |
| --- | --- | --- | --- |
| `/` | GET | HTML | 实时监控页面。 |
| `/video` | GET | `multipart/x-mixed-replace` | MJPEG 实时视频流，来自最新 `display_frame`。 |
| `/distance` | GET | 文本 | 最新距离，格式如 `0.532 m`；无有效距离时返回 `N/A`。 |
| `/status` | GET | 文本 | 检测到 bucket 时返回 `bucket`，否则返回 `未检测到 bucket`。 |

Web 页面每 500 ms 拉取一次 `/distance` 和 `/status`，视频通过 `<img src="/video">` 播放 MJPEG 流。

## 环境变量

主入口 `main.py` 当前读取以下环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ENABLE_PUSH` | `0` | 设为 `1` 时启动推送相关线程。 |
| `VIDEO_PUSH_URL` | 未设置 | JPEG 视频帧 HTTP POST 地址。 |
| `META_PUSH_URL` | 未设置 | 检测元数据 JSON HTTP POST 地址。 |
| `PUSH_FPS` | `8` | 推送帧率。 |
| `RKNN_MODEL_PATH` | 未设置 | RKNN 模型备用路径。 |

仅运行本地 Flask 页面时不需要设置推送变量。若已修复下方提到的推送日志字段问题，并确认需要启用 HTTP 推送，可按以下方式配置：

```bash
export ENABLE_PUSH=1
export VIDEO_PUSH_URL=http://server.example.com/api/videoupload
export META_PUSH_URL=http://server.example.com/api/basicdata
export PUSH_FPS=8
python main.py
```

当前 `main` 分支中，`RangingService` 的推送实现是 HTTP POST，而不是 WebRTC。旧文档里出现过的 `WEBRTC_SIGNAL_URL`、`WEBRTC_STUN_URLS` 没有被 `main.py` 读取。

另外，`AppController.run()` 中的推送启动日志仍引用了未接入的 WebRTC 字段名。启用 `ENABLE_PUSH=1` 前，需要先把这段日志与当前 `RangingService` 字段对齐，否则会访问不存在的属性。

## HTTP 推送格式

启用推送且配置 `VIDEO_PUSH_URL` 时，系统会按 `PUSH_FPS` 从最新左目校正图编码 JPEG 并发送：

```http
POST <VIDEO_PUSH_URL>
Content-Type: image/jpeg
X-Frame-Timestamp: <frame_ts>

<jpeg bytes>
```

启用推送且配置 `META_PUSH_URL` 时，系统会发送检测元数据：

```json
{
  "ts": 1710000000.123456,
  "detected": true,
  "distance_m": 0.532,
  "detections": [
    {
      "left": 120,
      "top": 80,
      "right": 360,
      "bottom": 420,
      "score": 0.91,
      "class_id": 0,
      "label": "bucket"
    }
  ]
}
```

`distance_m` 在以下场景可能为 `null`：

- 未检测到 bucket。
- 检测到了 bucket，但检测框 ROI 内有效视差不足。
- 视差波动过大或中位视差过小。
- 估计距离超出当前工作范围。

## 测距流程

1. 从 `/dev/video21` 读取一帧 `2560x720` 图像。
2. 按左右 ROI 拆分为左目和右目：
   - 左目：`(0, 0, 1280, 720)`
   - 右目：`(1280, 0, 1280, 720)`
3. 使用 `camera_config.py` 中的映射表对左右图做畸变校正和立体校正。
4. 将左右图转灰度，使用 StereoSGBM 计算视差。
5. 若 `cv2.ximgproc` 可用，启用 WLS 滤波；否则使用 `medianBlur` 做基础去噪。
6. 在左目校正图上运行 RKNN 检测，筛选 `bucket` 检测框。
7. 选取得分最高的检测框，在框内 ROI 统计有效视差。
8. 经过百分位裁剪、MAD 离群剔除和阈值检查后，用公式估计距离：

```text
distance_m = focal_length_px * baseline_m / median_disparity_px
```

9. 对有效距离做指数平滑；短时视差不稳定或检测短暂丢失时，保留上一帧有效距离，降低 `N/A` 抖动。
10. 发布 `RangingUpdateEvent`，由状态缓存和 Web 层读取最新结果。

## 关键默认参数

| 参数 | 默认值 | 位置 | 说明 |
| --- | --- | --- | --- |
| `camera_device` | `21` | `RangingService` | 对应 `/dev/video21`。 |
| `frame_width` | `2560` | `RangingService` | 输入总宽度。 |
| `frame_height` | `720` | `RangingService` | 输入总高度。 |
| `left_roi` | `(0, 0, 1280, 720)` | `RangingService` | 左目区域。 |
| `right_roi` | `(1280, 0, 1280, 720)` | `RangingService` | 右目区域。 |
| `sgbm_num_disparities` | `160` | `RangingService` | SGBM 视差搜索范围，会规整为 16 的倍数。 |
| `sgbm_block_size` | `7` | `RangingService` | SGBM block size，会保证为奇数且至少为 3。 |
| `use_wls` | `True` | `RangingService` | 是否优先使用 WLS 视差滤波。 |
| `distance_min_m` | `0.05` | `RangingService` | 有效距离下限。 |
| `distance_max_m` | `1.2` | `RangingService` | 有效距离上限。 |
| `smooth_alpha` | `0.9` | `RangingService` | 距离平滑系数。 |
| `hold_last_seconds` | `1.5` | `RangingService` | 检测仍存在但视差不稳定时保留上一距离的时间。 |
| `hold_lost_seconds` | `0.4` | `RangingService` | 检测短暂丢失时保留上一距离的时间。 |
| `MODEL_SIZE` | `(640, 640)` | `detector.py` | RKNN 模型输入尺寸。 |
| `OBJ_THRESH` | `0.536` | `detector.py` | 检测初筛阈值。 |
| `NMS_THRESH` | `0.45` | `detector.py` | NMS 阈值。 |
| `DETECT_SCORE_MIN` | `0.536` | `detector.py` | 最终检测分数门槛。 |

## GPS 工具

`app/infrastructure/gps.py` 可以独立运行，用于读取串口 NMEA 数据并输出定位状态。

```bash
python -m app.infrastructure.gps --port /dev/ttyS9 --baud 9600
```

常用参数：

```bash
# 指定原点，输出相对原点的 ENU 米制坐标
python -m app.infrastructure.gps --origin 31.123456 121.123456

# 在首次有效定位时自动设置原点
python -m app.infrastructure.gps --set-origin-on-fix

# 跳过 NMEA 校验和检查
python -m app.infrastructure.gps --no-checksum
```

## WebRTC 辅助能力

仓库中包含 WebRTC 相关文件：

- `app/infrastructure/webrtc_push.py`：WebRTC 推送客户端实现，可通过 WebSocket 信令发送视频轨和数据通道元数据。
- `tools/webrtc_sender.py`：本地摄像头 WebRTC 发送模拟工具。
- `src/views/WebRTCViewer.vue`：Vue WebRTC 观看端页面。

本地发送工具示例：

```bash
python tools/webrtc_sender.py \
  --signal ws://<server>:<port>/ws \
  --device-id robot-001 \
  --fps 15 \
  --width 640 \
  --height 480 \
  --camera 0 \
  --ice stun:stun.l.google.com:19302
```

注意：这些 WebRTC 文件当前没有被 `main.py` 自动启动。主程序 Web 页面使用 Flask MJPEG，主程序推送使用 HTTP POST。

## 串口输出占位

`app/infrastructure/serial_port.py` 中的 `SerialPortGateway` 已接入事件总线，但默认 `enabled=False`，当前不会实际写串口。

后续可以在 `SerialPortGateway.handle_ranging_update()` 中补充：

- 串口初始化与释放。
- 测距事件编码协议。
- `distance_m`、`detected`、`detections` 等字段发送逻辑。

## 常见问题

### 无法打开摄像头

默认设备是 `/dev/video21`。请确认设备存在、权限正确，并且输出分辨率支持 `2560x720`：

```bash
ls /dev/video*
```

如需调整设备号或分辨率，修改 `main.py` 中创建 `RangingService` 时传入的参数，或修改 `RangingService` 默认值。

### WLS 视差滤波不可用

如果日志提示 `OpenCV ximgproc 不可用`，程序会自动退回 `medianBlur`。要启用 WLS，请安装包含 contrib 模块的 OpenCV：

```bash
pip install "opencv-contrib-python>=4.8,<5"
```

### RKNN 模型加载失败

确认以下条件：

- `model/bucket.rknn` 存在。
- `rknnlite` 已在目标环境中正确安装。
- 当前设备支持 RKNNLite runtime。
- 如使用外部模型路径，确认 `RKNN_MODEL_PATH` 指向有效文件。

### 距离显示为 N/A

`N/A` 通常表示当前帧没有可靠距离结果。常见原因：

- 未检测到 bucket。
- bucket 表面纹理不足、反光严重或光照过暗。
- 检测框内有效视差比例不足。
- 目标距离超出默认有效范围 `0.05 m` 到 `1.2 m`。
- 双目标定参数与实际相机不匹配。

### Web 页面没有画面

确认主程序仍在运行，并检查 `/video` 是否有 MJPEG 数据输出。页面画面来自 `RangingStateStore` 中的最新 `display_frame`，如果摄像头线程未正常采集，Web 页面也不会更新。

## 许可证

本项目使用 MIT License，详见 `LICENSE`。
