# F407 首轮只收联调

**2026-11-21 工作范围更新：** B 顶部为0，当前下降上限为 **96 mm**（在136 mm处按用户要求上调40 mm）。右侧车体会阻碍低位A旋转，A的可用角度必须结合B高度判断。用户估计机械底部约166 mm，不作为运动目标。视觉框选测试阶段C固定10 mm；伸缩杆受摄像头重量挠曲，C不用于上下纠偏。详见 `arm_reference.json` 的 `operating_envelope`。该约束由上位机执行，原固件未修改。

独立于 `vision_cpp`。`receive.py` 仅接收、记录和离线解析新版 F407 的 20 字节状态帧，没有发送入口。另有明确授权后运行的 `disable_once.py`，只提供固定禁用帧单次发送；当前视觉追踪导入本目录的回传解析器与工作范围；真正发送仍复用原单帧发送器。板端位置：`/home/elf/deploy/serial_rx`。

**2026-11-21 现场确认基准：** 后续机械臂控制先读取 [arm_reference.json](arm_reference.json)。A 结构零位对应原协议 **75°**（增大向内、减小向外）；B 顶部为 0 mm（增大下降）；C 收回起点为 0 mm（增大伸出）。B 的 0→6→0、C 的 0→30→0 已由用户观察确认；航向传感器固定于底盘，实测左转角度增大、右转减小。继续使用板上原固件和原协议，独立新固件方案已停止。此 JSON 由新增主程序追踪控制器读取，不被旧发送器本身自动加载；原固件上电默认 A=90° 未改变，不能当作结构零位。以下保留 11 月 18 日的历史记录，不作为最新硬件状态。

## 单次禁用验证（2026-11-18 20:02～20:03）

用户明确要求发一次禁用指令后，使用 `disable_once.py` 先监听 3 秒、一次写入 `B3 00 00 00 5A 00 00 00 00 B4`，再监听 12 秒。实际发送前已有 14 个有效帧（命令计数0），发送后有48个（命令计数1），结果码0、状态禁用、使能false，系统/电机/UART故障均0。支持上下行通信及固件接受禁用指令；不是禁用指令唤醒了上报。零点均未确认、脉冲估计位置0、航向/超声无效。早期无回传的原因尚未确定。

`disable_once.py` 没有可配置动作帧、不重试、不发送退出帧；`test_disable_once.py` 用伪终端验证实际仅发固定10字节并正确读取响应。它会清非电机故障（固件原有语义），不使能、不确认零点；既有零点不会被清除。只有明确请求禁用验证时运行：

```sh
python3 disable_once.py --out runs/NEW_UNIQUE_ID --authorization '填写当次实际授权'
```

正常观察继续使用 `receive.py capture`，不重复发送禁用指令。所有状态都属于固件报告，轴位置是脉冲估计，舵机角度是软件值。

协议来源：[MCU-slave-device，9db673c](https://github.com/Muzhixing/MCU-slave-device/blob/9db673c9103d3da79434c08d7217c2c402871855/new_project_f407/F407_HAL_Template/docs/protocol.md)。115200、8N1，上行 `B5…B6` 共 20 字节，版本为 1，偏移 18 校验偏移 1～17 的异或。下行 10 字节没有校验字段，本工具不生成下行。

## 当前识别结果（2026-11-18）

- RK3588：`elf@192.168.31.175`，`elf2-desktop`，ELF 2 Board。
- USB 转串口：CH340，VID:PID `1a86:7523`，驱动 `ch341`，当前 `/dev/ttyUSB0`。
- 默认固定插口路径：`/dev/serial/by-path/platform-fc880000.usb-usb-0:1.2:1.0-port0`。
- 另有 `/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0`，但设备没有唯一序列号；多个同类模块时不能靠这个名字区分。换 USB 插口后必须重新识别 by-path。
- 相机为 `1bcf:0b15`，不属于这个 tty 设备。旧 `/dev/myserial` 当前恰好指向 ttyUSB0，本工具不依赖它。

## 首轮实测（2026-11-18 18:45）

用户确认伸缩臂允许动作，其他执行机构连接已断开后，已执行 `physical_rx_20261118_01`：30.006 秒收到 0 字节、0 有效帧，采集无异常，CH340 保持在线。无法据此判断使能、零点、位置或故障。首轮使用普通用户及 `--allow-partial-occupancy`，权限限制保存在 manifest。下一步核对 PC10→转换器 RX 与共地；协议周期上报不需要先发送指令。

## 打开物理端口前

现场确认并记录：

1. 底盘、升降、横杆、舵机和水泵的动力/使能已隔离或确认禁用，STM32 和串口模块仍有正常逻辑供电。静止本身不能证明已禁用。
2. F407 PC10（USART3 TX）→转换器 RX，共地。双向接线时转换器 TX→PC11；核对为兼容 3.3 V 的 UART 逻辑电平，不能接 RS232 或 RS485 信号。不要把接口的 +5 V 供电脚当成信号脚。
3. DTR/RTS 不连接 NRST、BOOT0 或执行机构使能；PC11 不并接其他主动发送器。Linux CH340 驱动可能在 `open()` 时改变 modem 电平，`O_RDONLY` 不能消除驱动自身的这一行为。

严格保证物理线上只收的推荐接法：在现有 USB 转串口方案内，仅连接转换器 RX 与 GND，暂时隔离转换器 TX、DTR、RTS，保持原有逻辑供电。现场接线变化由操作者核实后进行，不改用 RK3588 板载 UART。即使使用只读 FD，初次打开到关闭 ECHO 之间仍可能存在内核行规程窗口；TX 物理隔离能排除这个窗口产生下行字节。

## 采集与复现

以下命令在 RK3588 上运行。`SITE_CONFIRMATION` 必须填写实际完成的现场检查，不能用模板文字代替确认。下列为完整权限检查的采样方式；首轮实际使用了下述普通用户选项。

```sh
cd /home/elf/deploy/serial_rx
# 现场完成上述检查后填写；为空时工具拒绝打开端口。
SITE_CONFIRMATION=''
RUN_ID="rx_$(date -u +%Y%m%dT%H%M%SZ)"
sudo python3 receive.py capture --seconds 30 \
  --port /dev/serial/by-path/platform-fc880000.usb-usb-0:1.2:1.0-port0 \
  --site-confirmation "$SITE_CONFIRMATION" --out "runs/$RUN_ID"
```

使用 sudo 是为了完整检查 `/proc/*/fd` 占用；当前 SSH 用户没有免密 sudo。默认不能查看所有进程 FD 时拒绝采集。可显式添加 `--allow-partial-occupancy` 以普通用户观察：仍拒绝已知占用，将不可见 PID 和检查范围记录到 manifest；不退出其他进程。使用 `TIOCEXCL` 防止后续普通进程打开。板上 ModemManager 当前运行，但本次启动日志对 CH340 报告 `not supported by any plugin`；若重新插拔，重新检查日志和占用，不把这次阴性检查视为永久保证。

工具只读打开串口，关闭回显和软硬件流控，不设置 DTR/RTS、不发 break、不 flush。关闭时保留 raw/no-echo/HUPCL-off 设置，不恢复可能开启回显的旧配置。它不提供停止控制；遇到已使能或运动状态会结束采样，操作者应通过现场硬件处置，不能把程序退出或断线当成停车。

每次用新目录，避免覆盖原始证据：

- `manifest.json`：设备、参数、源码哈希、现场确认原文、固件身份依据。
- `raw.bin`：原始接收字节，未丢弃采样前内核已缓存的输入。
- `chunks.jsonl`：逐次 read 的十六进制、UTC 时间、monotonic_ns 和原始文件偏移。
- `frames.jsonl`：逐个有效帧、两轴零点、估计位置、航向、超声和故障。
- `summary.json`：有效帧、校验/重同步统计、主机接收周期、使能帧数、状态和取值范围。

```sh
python3 receive.py replay "runs/$RUN_ID/chunks.jsonl" \
  --out "runs/$RUN_ID/replay_summary.json"
```

正常周期预期约 250 ms；同一次 read 内的多个帧共享主机时间，不能据此恢复精确板端周期。命令计数不是状态帧序号，正常无命令时可以一直不变，不能据此计算丢帧。版本号只能确认协议兼容性，不能唯一证明烧录提交。

状态中的轴位置为脉冲累计估计；舵机为软件角度，并非实测反馈。此状态帧没有水泵实测状态，也不能证明物理驱动 EN 接法。超声距离无效或角度未就绪时解析为 null。状态不是逐包 ACK。

若 30 秒没有字节，保留零字节文件和 summary，依次核对枚举/占用、逻辑供电、共地、PC10→RX、电气电平及实际固件；有字节无有效帧则从 raw 检查协议/波特率/噪声。不能用旧停机帧、复位或重烧录代替排查。

## 离线检查

本机：

```sh
python3 deploy/serial_rx/verify_firmware.py /path/to/MCU-slave-device
python3 -m unittest discover -s deploy/serial_rx -p 'test_*.py' -v
```

`verify_firmware.py` 仅在临时目录以主机 mock 编译固定提交的真实固件状态机与测试，导出 `SystemControl_Status` 产生的状态帧。`firmware_fixture.json` 记录来源及哈希。Linux 的两个端到端测试只打开新建的 `/dev/pts/*`，验证只读打开、拆包/粘包、原始记录、无回显及遇已使能状态退出；不打开物理串口。
