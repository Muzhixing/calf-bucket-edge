# 双目测距系统（Bucket）

本项目用于在嵌入式设备上进行双目视觉测距，并通过 Web 页面展示实时画面与距离结果。
当前已采用**分层架构（Layered Architecture）**，并在层间通过**异步事件驱动（Event-Driven）**进行数据交互。

---

## 功能概览

- 双目相机采集、视差计算、距离估计
- RKNN 模型目标检测（bucket）
- Web 实时视频流与测距结果展示
- WebRTC 推流（视频 + 检测结果）
- 串口输出层预留（占位，待补充）

---

## 架构说明（分层 + 事件驱动）

- **Domain（领域层）**：定义领域事件（如测距更新事件）。
- **Application（应用层）**：事件总线、状态缓存、应用控制器。
- **Infrastructure（基础设施层）**：摄像头采集、RKNN 推理、测距逻辑、串口网关占位。
- **Presentation（表现层）**：Flask Web UI。

**事件流示意**：
`RangingService`（基础设施层）产生测距结果 → 发布 `RangingUpdateEvent` →
`RangingStateStore`（应用层）更新状态 → Web（表现层）读取展示。

串口层（`SerialPortGateway`）已预留，后续可订阅同一事件完成串口输出。

---

## 目录结构

```
app/
  domain/
    events.py
  application/
    app_controller.py
    event_bus.py
    state_store.py
  infrastructure/
    ranging_service.py
    detector.py
    camera_config.py
    gps.py
    serial_port.py
  presentation/
    web.py
main.py
model/
  bucket.rknn
```

---

## 运行方式

```bash
python main.py
```

启动后，终端会输出访问地址：
```
Flask: http://<板卡IP>:5050/
```

---

## 环境变量配置

在运行前可按需设置：

- `ENABLE_PUSH=1` 启用推送
- `WEBRTC_SIGNAL_URL` WebRTC 信令 WebSocket 地址
- `WEBRTC_STUN_URLS` STUN 地址列表（逗号分隔，可选）
- `PUSH_FPS` 推送帧率（默认 8）

示例：
```bash
export ENABLE_PUSH=1
export WEBRTC_SIGNAL_URL=ws://<host>:<port>/ws
export WEBRTC_STUN_URLS=stun:stun.l.google.com:19302
export PUSH_FPS=8
python main.py
```

---

## 依赖说明

常用依赖（按功能模块）：

- `opencv-python`（或带 `ximgproc` 的 OpenCV 版本）
- `numpy`
- `flask`
- `rknnlite`（RKNN 推理，运行在 RK 设备上）
- `pyserial`（GPS / 串口相关）
- `aiortc` / `av` / `websockets`（WebRTC 推送）

> RKNN 模型路径目前在 `app/infrastructure/detector.py` 中为绝对路径：
> `/mnt/tfcard/work/calf/model/bucket.rknn`，如需调整请修改该常量。

---

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

# 启用稳定滤波，降低静止时 GPS 坐标抖动
python -m app.infrastructure.gps --port /dev/ttyS9 --baud 9600 --filter --set-origin-on-fix

# 固定点场景：假定 GPS 模块静止，使用长期平均进一步稳定坐标
python -m app.infrastructure.gps --port /dev/ttyS9 --baud 9600 --stationary --set-origin-on-fix

# 跳过 NMEA 校验和检查
python -m app.infrastructure.gps --no-checksum
```

滤波模式会对经纬度做中值滤波、EMA 低通滤波、静止小范围保持和异常跳点抑制。常用调参：

```bash
python -m app.infrastructure.gps \
  --port /dev/ttyS9 \
  --baud 9600 \
  --filter \
  --filter-window 9 \
  --filter-alpha 0.2 \
  --static-hold-radius 1.5 \
  --origin-warmup-samples 8 \
  --show-raw
```

`--filter-alpha` 越小输出越稳定但响应越慢；`--static-hold-radius` 越大，静止时越不容易抖动，但低速移动时响应也会更慢。
如果 GPS 模块固定不动，优先使用 `--stationary`；该模式会把当前位置当作固定点持续平均，输出会随样本数增加越来越稳定，但不适合移动场景。

---

## 串口层占位说明

`app/infrastructure/serial_port.py` 中的 `SerialPortGateway` 已预留，
目前不做实际发送，后续可实现：

- 初始化串口
- 将测距事件编码为协议数据
- 发送至串口

启用方式（示例）：
在 `main.py` 中将 `SerialPortGateway(enabled=False)` 改为 `enabled=True` 并补充实现。

---

## 备注

- 摄像头设备号、分辨率、推送参数等均在 `RangingService` 中可配置。
- Web 展示默认读取事件缓存（`RangingStateStore`）。
- 如果环境缺少显示输出，`render_on_device` 可关闭显示以仅提供 Web 服务。
