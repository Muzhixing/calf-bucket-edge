# 投喂车视觉识别与机械臂追踪

当前实时视觉后端为 **C++**，模型为 **B1_C0_REG_FP16**。原启动入口已接入目标追踪与 A/B 居中控制：连续识别后锁定同一个目标，点击“开始机构追踪”后随目标位置调整。C 保持 10 mm，B 工作范围为 0–96 mm；默认仅观察。当前视角未识别到目标时可“框选容器”；本次白色容器在调整后的视角已完成一次自动识别与居中，其他视角的识别率仍需验证。

```sh
cd deploy/vision_v2
./deploy.sh
./view_live.sh                     # 快速测距
./view_live.sh --depth-scale 1     # 全分辨率测距
./view_live.sh --observe-only     # 完全禁用机构控制，仅观察
```

本机通过 SSH 打开 `http://127.0.0.1:8788/`，Ctrl+C 停止并释放相机。板端在 `/home/elf/deploy/vision_v2` 运行 `./view_hdmi.sh` 可直接显示 HDMI 画面；桌面须先停止。

- `deploy/vision_cpp/`：当前原生计算与显示、构建及启动脚本。
- `deploy/vision_cpp/tracking/`：Python 目标追踪、控制页面及原协议串口控制；采集、检测和测距仍在 C++ 中。
- `deploy/vision_v2/`：当前冻结模型、标定、相机配置、采集与独立参照验收工具，以及本机启动入口。
- `docs/`：运行说明、视觉输出接口及保留的硬件参考/论文快照。
- `research/`：按要求保留的论文、封存资料及研究证据，不上传工程仓库。

2026-11-18 未接入追踪控制时，同源双目回放、相同 HDMI 显示负载的对照中，C++ 进程树 PSS 为 179.9 MiB，较原实现减少约 160.4 MiB；稳定画面约 30 FPS、新距离约 10 次/秒。该内存结果不包含新增 Python 追踪进程。

模型和标定在启动时核验 SHA-256。当前测的是视差采样区域对应的相机前向 Z；桶框中心用于对齐，桶口作业中心尚未独立定位。伸缩臂装配后须校准零位、方向、可用行程和投放点偏移。

[运行说明](docs/runtime.md) · [视觉输出接口](docs/interfaces/api-rules.md) · [采集与验收](deploy/vision_v2/README.md) · [C++ 构建](deploy/vision_cpp/README.md) · [当前实测版本](docs/tracking-release-20261122.md)
