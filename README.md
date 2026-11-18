# Bucket 投喂车视觉工程

当前固定相机联调使用冻结的 **seed2_REG_FP16** 模型；V2表示标定和运行版本。

本机启动实时画面：

```bash
cd deploy/vision_v2
./deploy.sh
./view_live.sh
```

打开本机 http://127.0.0.1:8788/ ，Ctrl+C停止。启动与采集入口保留在 `/home/elf/deploy/vision_v2`；实时预览默认使用 `/home/elf/deploy/vision_cpp` 的 C++ 后端，`--backend python` 可回退。部署保留旧 V1/V2 目录。

实时预览默认使用快速双目测距。原 Python 版固定机位单桶实测 45 秒，电脑收图 30.01 FPS，桶身新距离更新 9.92 次/秒。运行 `./view_live.sh --depth-scale 1` 可切换到全分辨率双目，实测收图 30.02 FPS、新距离更新 2.98 次/秒。

预览页以画面中心的白色十字为基准，显示桶框中心的左右、上下偏差。两轴对齐并取得新距离后，板端生成带周期编号的投喂预演结果，后续控制器可从 `/status` 的 `guidance` 字段读取。

2026-11-18 同源双目回放、相同 HDMI 显示负载的 30 秒对照：C++ 进程树 PSS 中位值 179.9 MiB，Python 340.3 MiB，减少约 160.4 MiB；两者约 30 FPS、10 次/秒新测距。此项是运行开销验证，现场更新率仍受场景影响。

板端外接 HDMI 也可直接显示同一视觉管线：在板端 `/home/elf/deploy/vision_v2` 运行 `./view_hdmi.sh`。桌面临时切换与恢复命令见下方部署说明。

- [视觉部署、测距及启动说明](deploy/vision_v2/README.md)
- [完整应用结构](docs/runtime.md)
- [应用接口](docs/interfaces/api-rules.md)

工程目录：

- `deploy/vision_cpp/`：默认原生实时后端，含独立双目进程、网页与 HDMI 显示。
- `deploy/vision_v2/`：独立视觉入口、采集/预览工具、冻结模型、标定及必要配置；不启动STM32和执行机构。
- `main.py`、`app/`：原完整应用、设备任务、串口及Web模块。
- `config/`：应用任务配置；`model/`为旧完整应用模型，V2显式加载自己的冻结模型。
- `src/views/`、`tools/webrtc_sender.py`：原WebRTC观看组件与发送工具。

当前先验收桶身相对距离，暂不要求精确桶口中心。相机旁的出水口由伸缩臂送到桶口上方；装配时再确认伸缩范围以确定停车区间。

仓库只保存工程运行文件和使用说明。测试文件、采集数据、日志、实验结果、缓存、历史研究和论文均不纳入本次提交。
