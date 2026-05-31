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

记录纯 GPS 测试日志：

```bash
mkdir -p logs
python -m app.infrastructure.gps \
  --port /dev/ttyS9 \
  --baud 9600 \
  --filter \
  --set-origin-on-fix \
  --show-raw \
  --log-file logs/gps_$(date +%Y%m%d_%H%M%S).csv
```

日志为 CSV 格式，包含时间、经纬度、ENU 米制坐标、速度、航向、HDOP、卫星数、原始坐标和滤波状态。程序用 `Ctrl+C` 停止后会自动关闭日志文件。

## IMU 与 GPS+IMU 融合

MPU-6050 可通过 40Pin 的 I2C4 读取，默认连接参数：

- `VCC -> 3.3V`
- `GND -> GND`
- `SDA -> pin3 / I2C4_SDA`
- `SCL -> pin5 / I2C4_SCL`
- 默认 I2C 设备：`/dev/i2c-4`
- 默认 I2C 地址：`0x68`

单独读取 IMU：

```bash
sudo python -m app.infrastructure.imu \
  --bus 4 \
  --address 0x68 \
  --rate 50 \
  --calibrate 200
```

运行 GPS+IMU 融合：

```bash
sudo python -m app.infrastructure.gps_imu_fusion \
  --gps-port /dev/ttyS9 \
  --gps-baud 9600 \
  --imu-bus 4 \
  --imu-address 0x68 \
  --imu-rate 50 \
  --calibrate-samples 200
```

测试 30 秒：

```bash
sudo python -m app.infrastructure.gps_imu_fusion \
  --gps-port /dev/ttyS9 \
  --gps-baud 9600 \
  --duration 30
```

记录融合轨迹日志：

```bash
mkdir -p logs
sudo python -m app.infrastructure.gps_imu_fusion \
  --gps-port /dev/ttyS9 \
  --gps-baud 9600 \
  --imu-bus 4 \
  --imu-address 0x68 \
  --imu-rate 50 \
  --calibrate-samples 200 \
  --log-file logs/fusion_$(date +%Y%m%d_%H%M%S).csv \
  --log-rate 5
```

融合日志字段包含 `fused_lat/fused_lon`、`fused_east_m/fused_north_m`、GPS 原始坐标、速度、航向、`sigma`、HDOP、卫星数、IMU 前向/右向加速度和偏航角速度。`--log-rate` 控制 CSV 记录频率，0 表示跟随终端输出频率。

停止后分析日志：

```bash
python tools/analyze_gps_log.py logs/fusion_20260531_120000.csv

# 如果有已知参考点，输入参考经纬度可直接估计误差
python tools/analyze_gps_log.py logs/fusion_20260531_120000.csv \
  --reference 38.4947700 106.1086836
```

分析脚本会输出点数、时长、首尾坐标、轨迹长度、首尾位移、ENU 坐标范围，以及 RMS / P50 / P95 / 最大漂移或误差。

融合程序会先要求车辆静止，采集 IMU 零偏；随后使用 GPS 首个有效定位点作为本地 ENU 原点。输出字段中：

- `fused_lat/fused_lon` 是融合后的经纬度。
- `E/N` 是相对启动原点的东向/北向米制坐标。
- `speed` 是融合速度。
- `heading` 是航向角，0 度为正北，顺时针增加。
- `sigma` 是当前水平位置不确定度估计，越小越稳定。
- `gps_age` 是距离最近一次 GPS 修正的时间。

如果 IMU 安装方向与车体方向不一致，需要调整轴映射。例如模块的 `Y` 轴朝车头、`X` 轴朝车右：

```bash
sudo python -m app.infrastructure.gps_imu_fusion \
  --forward-axis y \
  --right-axis x \
  --yaw-gyro-axis z
```

如果转弯时航向变化方向反了，把 `--yaw-gyro-axis z` 改为 `--yaw-gyro-axis -z`。

注意：普通 GPS + MPU-6050 不能达到 RTK 的厘米级绝对精度。该融合主要用于降低抖动、让短时速度/航向更稳定、在短暂 GPS 弱信号时保持连续轨迹。想要车载绝对定位达到 0.1m 级甚至厘米级，需要 RTK GPS、双天线航向、轮速计/编码器和更完整的标定。

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
