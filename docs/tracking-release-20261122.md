# RK3588桶追踪实测版本：2026-11-22

本版完成自动识别、自动确认、A/B实时跟随、大幅换位置/距离重获、近距离边缘部分可见重获，并保留遮挡暂停和圈内停止微调。用户对最后一轮边缘实测反馈“完美”。本次发布冻结算法，不追加优化。

## 验证依据

- 同录像边缘重获：2.17秒→0.20秒；本地与RK3588板端均126项测试通过。
- 119.15秒实机记录：13次边缘重获，确认跨度132–268ms；最后圈内保持18.95秒，没有继续微调。
- 停机后独立回传14帧均禁用、故障0。历史位置快照A77°/B76mm/C10mm不能代替下一次启动时的实时状态。
- 细节及适用范围：[边缘重获](new-model-tracking-joint-20261122/edge-reacquire.md)、[位置与尺度重获](new-model-tracking-joint-20261122/wide-reacquire.md)、[线程及长帧修正](new-model-tracking-joint-20261122/stutter-fix.md)。

## 精确运行条件

| 项目 | 验收配置 |
|---|---|
| 模型 | B1_C0_REG_FP16，RGB，NPU 0/1/2 |
| 模型SHA256 | `4a59b48b634c25ca79d843cc746d8e062ca1161ec5ef7c0c0c96533c78d3fce7` |
| 原生程序SHA256 | `f73296b04b72769c61a0f8c0baa06d626feecf60be78cced2f78293d65eef526` |
| 标定YAML SHA256 | `345f68e9e4fbaf114c5ee982d11a75cfc0e0c23105df3de6ee6e4b51c3ee7b00` |
| 图像计算 | OpenCV 4.10.0，跟踪CPU 4/6、单线程；OpenCL校正 |
| 控制 | 原B3/B4下行、B5/B6回传；A60–100°、B0–96mm、C10mm |
| 模式 | 自动连续跟随，开启背景运动诊断，不启用CSRT，无标定响应profile |

精确源码及验证摘要见本目录`tracking-release-evidence/`。原生二进制、本轮B1 RKNN权重和Python依赖包不新增到Git，按上述哈希从现有板端保留副本取用。算法测试使用PTY和合成图像，不触碰真实串口。

## 当前板端启动

已验证且保留的独立入口：

```sh
sh /home/elf/deploy/vision_edge_stage_20261122/run.sh inspect_01 --observe-only --seconds 120
# 现场具备运动条件时才使用自动跟随入口，编号须唯一：
sh /home/elf/deploy/vision_edge_stage_20261122/run.sh follow_01 --seconds 600
```

该入口在8798端口提供控制页，8799为原生后端。本机既有SSH隧道将18798映射至8798。每次重启后刷新页面取得新控制令牌。原固件统一使能也可能驱动车轮；最后一轮测试时车身已垫高、车轮连接。B/C须以真实原点确认，不能在当前非零位置重新登记零位。

普通仓库部署入口为`deploy/vision_cpp/run.sh`，需先安装`tracking`中的源代码、`serial_rx/receive.py`与`arm_reference.json`，以及经核验的模型/原生依赖。本轮验收使用板端隔离OpenCV包：

```sh
export PYTHONPATH=/home/elf/deploy/vision_reference_stage_20261122/python_deps
./run.sh --motion-compensation --port 8798 --seconds 600
```

原协议发送器仍位于`/home/elf/v1/seed2_reg_fp16/app/infrastructure/serial_port.py`，该目录不能整体删除。`--csrt`仍是可选研究路径，本版未启用。背景运动在没有经过验证的响应profile时只作诊断，不宣称已使用实测动力学前馈。

本次GitHub提交不烧录STM32、不启动机构、不覆盖板端生产入口。退出运行时向`tracking/runtime.py`发送SIGINT并检查禁用回传，不能只凭进程消失判断停机。

## 回归命令

```sh
PYTHONPATH=deploy/vision_cpp/tracking python3 -m unittest discover -s deploy/vision_cpp/tracking/tests -v
```

Python环境须提供NumPy、OpenCV与pyserial。126项最终测试日志随验证摘要保留；完整原始录像、逐帧日志及旧候选放在冷资料归档，路径见[工作区导航](../WORKSPACE.md)。
