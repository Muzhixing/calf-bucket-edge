# 投喂车固定相机视觉工程

当前实时后端为 **C++**，模型固定为 **seed2_REG_FP16**。提供桶检测、双目桶身前向测距、相对画面中心的左右/上下偏差及两轴对齐预演，不启动 STM32、串口或执行机构。

```sh
cd deploy/vision_v2
./deploy.sh
./view_live.sh                     # 快速测距
./view_live.sh --depth-scale 1     # 全分辨率测距
```

本机通过 SSH 打开 `http://127.0.0.1:8788/`，Ctrl+C 停止并释放相机。板端在 `/home/elf/deploy/vision_v2` 运行 `./view_hdmi.sh` 可直接显示 HDMI 画面；桌面须先停止。

- `deploy/vision_cpp/`：当前原生计算与显示、构建及启动脚本。
- `deploy/vision_v2/`：当前冻结模型、标定、相机配置、采集与独立参照验收工具，以及本机启动入口。目录名沿用现有部署路径；Python 只用于准备和采集，实时预览只使用 C++。
- `docs/`：运行说明、视觉输出接口及保留的硬件参考/论文快照。
- `research/`：按要求保留的论文、封存资料及研究证据，不上传工程仓库。

2026-11-18 同源双目回放、相同 HDMI 显示负载的对照中，C++ 进程树 PSS 为 179.9 MiB，较原实现减少约 160.4 MiB；稳定画面约 30 FPS、新距离约 10 次/秒。现场更新率随场景变化；此次 45 秒实机窗口稳定画面约 30 FPS、新距离约 7.4 次/秒。检测框和测距数值对照一致。

模型和标定在启动时核验 SHA-256。当前测的是视差采样区域对应的相机前向 Z；桶框中心用于对齐，桶口作业中心尚未独立定位。伸缩臂装配后须校准零位、方向、可用行程和投放点偏移。

[运行说明](docs/runtime.md) · [视觉输出接口](docs/interfaces/api-rules.md) · [采集与验收](deploy/vision_v2/README.md) · [C++ 构建](deploy/vision_cpp/README.md)
